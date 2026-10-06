//! Turning image bytes into the inputs each model was trained on.

use image::imageops::FilterType;
use image::{AnimationDecoder, DynamicImage, Frames, RgbImage};

use crate::{Result, VisionError};

/// How one model wants its input.
#[derive(Clone, Copy)]
pub(crate) struct InputSpec {
    pub size: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub fit: Fit,
}

/// How an image of any shape becomes a square.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fit {
    /// Short side to size, then the middle: what timm's eval transform does.
    CropCentre,
    /// Both sides to size, the whole image squashed in: SigLIP's transform.
    Squash,
}

/// timm's eval transform for Marqo's ViT: 384 px, mean and std 0.5.
pub(crate) const MARQO: InputSpec = InputSpec {
    size: 384,
    mean: [0.5; 3],
    std: [0.5; 3],
    fit: Fit::CropCentre,
};

/// SigLIP 2's: 224 px, mean and std 0.5, nothing cropped. A tall photo of a
/// person keeps the whole person, which is what the filter is judging.
pub(crate) const SIGLIP: InputSpec = InputSpec {
    size: 224,
    mean: [0.5; 3],
    std: [0.5; 3],
    fit: Fit::Squash,
};

/// Larger than this on a side is decoded only to be thrown away at 384 px.
const MAX_SIDE: u32 = 12_000;

/// An animation is judged on its first frame, then one frame this often.
const FRAME_EVERY_MS: u32 = 1_000;
/// At most this many frames of one animation: its first eight seconds or so.
/// Decoding further costs more than the frames are worth.
const MAX_FRAMES: usize = 8;
/// Browsers play a frame delay under 20 ms as 100 ms, and so do we.
const MIN_DELAY_MS: u32 = 20;
const SHORT_DELAY_AS_MS: u32 = 100;

fn check_size(bytes: &[u8]) -> Result<()> {
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| VisionError::Image(e.to_string()))?;
    let (w, h) = reader
        .into_dimensions()
        .map_err(|e| VisionError::Image(e.to_string()))?;
    if w > MAX_SIDE || h > MAX_SIDE {
        return Err(VisionError::Image(format!("{w}×{h} is too large")));
    }
    Ok(())
}

/// The image, or an animation's first frame: what the page shows first.
pub(crate) fn decode(bytes: &[u8]) -> Result<RgbImage> {
    check_size(bytes)?;
    image::load_from_memory(bytes)
        .map(|img| img.to_rgb8())
        .map_err(|e| VisionError::Image(e.to_string()))
}

/// The frames worth judging: one for a still image; for an animated GIF or
/// WebP, the first frame and then one every [`FRAME_EVERY_MS`]. A GIF that
/// opens on a harmless frame need not stay harmless.
pub(crate) fn decode_frames(bytes: &[u8]) -> Result<Vec<RgbImage>> {
    check_size(bytes)?;
    let cursor = std::io::Cursor::new(bytes);
    let frames = match bytes {
        [b'G', b'I', b'F', b'8', ..] => image::codecs::gif::GifDecoder::new(cursor)
            .ok()
            .map(|d| sample(d.into_frames())),
        [
            b'R',
            b'I',
            b'F',
            b'F',
            _,
            _,
            _,
            _,
            b'W',
            b'E',
            b'B',
            b'P',
            ..,
        ] => image::codecs::webp::WebPDecoder::new(cursor)
            .ok()
            .filter(|d| d.has_animation())
            .map(|d| sample(d.into_frames())),
        _ => None,
    };
    match frames {
        Some(frames) if !frames.is_empty() => Ok(frames),
        _ => Ok(vec![decode(bytes)?]),
    }
}

/// Frames at 0, 1 s, 2 s… of play time, as the frame delays add up, and the
/// last frame decoded. A frame that fails to decode ends the animation there:
/// a truncated download still has its earlier frames. The last one matters
/// most for those: the extension sends a GIF's first half-megabyte for a
/// quick verdict, which may hold under a second of it, and the first and
/// last frames of it are two looks instead of one.
fn sample(frames: Frames) -> Vec<RgbImage> {
    let mut out = Vec::new();
    let (mut at, mut next) = (0u32, 0u32);
    // The latest frame not sampled, kept undecoded until it is known to be last.
    let mut last = None;
    for frame in frames {
        let Ok(frame) = frame else { break };
        let (numer, denom) = frame.delay().numer_denom_ms();
        let delay = numer.checked_div(denom).unwrap_or(0);
        let delay = if delay < MIN_DELAY_MS {
            SHORT_DELAY_AS_MS
        } else {
            delay
        };
        if at >= next {
            last = None;
            out.push(DynamicImage::ImageRgba8(frame.into_buffer()).to_rgb8());
            if out.len() == MAX_FRAMES {
                return out;
            }
            next = at + FRAME_EVERY_MS;
        } else {
            last = Some(frame);
        }
        at = at.saturating_add(delay);
    }
    if let Some(frame) = last {
        out.push(DynamicImage::ImageRgba8(frame.into_buffer()).to_rgb8());
    }
    out
}

/// Resize (bicubic) and fit to a `spec.size` square, normalise; CHW floats.
pub(crate) fn square(img: &RgbImage, spec: InputSpec) -> Vec<f32> {
    let size = spec.size;
    let (w, h) = img.dimensions();
    let (nw, nh) = match spec.fit {
        Fit::Squash => (size, size),
        Fit::CropCentre => {
            let scale = size as f32 / w.min(h).max(1) as f32;
            (
                ((w as f32 * scale).round() as u32).max(size),
                ((h as f32 * scale).round() as u32).max(size),
            )
        }
    };
    let resized = image::imageops::resize(img, nw, nh, FilterType::CatmullRom);
    let (x0, y0) = ((nw - size) / 2, (nh - size) / 2);

    let plane = (size * size) as usize;
    let mut out = vec![0f32; 3 * plane];
    for y in 0..size {
        for x in 0..size {
            let px = resized.get_pixel(x0 + x, y0 + y);
            let at = (y * size + x) as usize;
            for c in 0..3 {
                out[c * plane + at] = (f32::from(px[c]) / 255.0 - spec.mean[c]) / spec.std[c];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_crop_is_the_size_and_layout_the_model_wants() {
        // A wide image, left half black and right half white.
        let img = RgbImage::from_fn(400, 200, |x, _| {
            if x < 200 {
                image::Rgb([0, 0, 0])
            } else {
                image::Rgb([255, 255, 255])
            }
        });
        let spec = InputSpec {
            size: 4,
            mean: [0.0; 3],
            std: [1.0; 3],
            fit: Fit::CropCentre,
        };
        let out = square(&img, spec);
        assert_eq!(out.len(), 3 * 4 * 4);
        // The centre crop straddles the edge: dark on the left, light on the right.
        assert!(out[0] < 0.2);
        assert!(out[3] > 0.8);
    }

    #[test]
    fn a_squashed_image_keeps_both_ends() {
        // A tall image: a dark top quarter, the rest light. A centre crop
        // would lose the dark band entirely.
        let img = RgbImage::from_fn(100, 400, |_, y| {
            if y < 100 {
                image::Rgb([0, 0, 0])
            } else {
                image::Rgb([255, 255, 255])
            }
        });
        let spec = InputSpec {
            size: 8,
            mean: [0.0; 3],
            std: [1.0; 3],
            fit: Fit::Squash,
        };
        let out = square(&img, spec);
        assert_eq!(out.len(), 3 * 8 * 8);
        assert!(out[0] < 0.2, "the top row is the dark band");
        assert!(out[7 * 8] > 0.8, "the bottom row is light");
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(decode(b"not an image").is_err());
        assert!(decode_frames(b"not an image").is_err());
    }

    /// An animated GIF: `count` frames of `delay_ms` each, frame i filled
    /// with grey level i.
    fn gif(count: u8, delay_ms: u32) -> Vec<u8> {
        use image::codecs::gif::GifEncoder;
        use image::{Delay, Frame, RgbaImage};
        let mut out = Vec::new();
        {
            let mut encoder = GifEncoder::new(&mut out);
            let frames = (0..count).map(|i| {
                Frame::from_parts(
                    RgbaImage::from_pixel(8, 8, image::Rgba([i, i, i, 255])),
                    0,
                    0,
                    Delay::from_numer_denom_ms(delay_ms, 1),
                )
            });
            encoder.encode_frames(frames).expect("encodes");
        }
        out
    }

    #[test]
    fn an_animation_is_judged_a_frame_a_second_and_at_its_end() {
        // 30 frames of 200 ms: 6 s, so frames 0, 5, 10, 15, 20 and 25, and
        // the last, 29.
        let frames = decode_frames(&gif(30, 200)).expect("decodes");
        let greys: Vec<u8> = frames.iter().map(|f| f.get_pixel(0, 0)[0]).collect();
        assert_eq!(greys, [0, 5, 10, 15, 20, 25, 29]);
    }

    #[test]
    fn a_cut_off_animation_is_judged_at_its_first_and_last_frames() {
        // The start of a GIF, as the extension sends it: under a second.
        let whole = gif(30, 100);
        let start = &whole[..whole.len() / 4];
        let frames = decode_frames(start).expect("decodes");
        let greys: Vec<u8> = frames.iter().map(|f| f.get_pixel(0, 0)[0]).collect();
        assert_eq!(greys.len(), 2, "first and last frames: {greys:?}");
        assert_eq!(greys[0], 0);
        assert!(greys[1] > 0);
    }

    #[test]
    fn a_long_animation_is_cut_short() {
        assert_eq!(
            decode_frames(&gif(100, 500)).expect("decodes").len(),
            MAX_FRAMES
        );
    }

    #[test]
    fn a_zero_delay_plays_as_a_browser_plays_it() {
        // 0 ms frames run at 100 ms each: one sample every ten frames.
        let frames = decode_frames(&gif(25, 0)).expect("decodes");
        let greys: Vec<u8> = frames.iter().map(|f| f.get_pixel(0, 0)[0]).collect();
        assert_eq!(greys, [0, 10, 20, 24]);
    }

    #[test]
    fn a_still_image_is_one_frame() {
        let mut png = Vec::new();
        RgbImage::from_pixel(8, 8, image::Rgb([9, 9, 9]))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("encodes");
        assert_eq!(decode_frames(&png).expect("decodes").len(), 1);
        assert_eq!(decode_frames(&gif(1, 0)).expect("decodes").len(), 1);
    }
}
