//! Turning image bytes into the inputs each model was trained on.

use image::RgbImage;
use image::imageops::FilterType;

use crate::{Result, VisionError};

/// How one model wants its input.
#[derive(Clone, Copy)]
pub(crate) struct InputSpec {
    pub size: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// timm's eval transform for Marqo's ViT: 384 px, mean and std 0.5.
pub(crate) const MARQO: InputSpec = InputSpec {
    size: 384,
    mean: [0.5; 3],
    std: [0.5; 3],
};

/// CLIP's: 224 px, its own channel statistics.
pub(crate) const CLIP: InputSpec = InputSpec {
    size: 224,
    mean: [0.481_454_7, 0.457_827_5, 0.408_210_7],
    std: [0.268_629_5, 0.261_302_6, 0.275_777_1],
};

/// Larger than this on a side is decoded only to be thrown away at 384 px.
const MAX_SIDE: u32 = 12_000;

pub(crate) fn decode(bytes: &[u8]) -> Result<RgbImage> {
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| VisionError::Image(e.to_string()))?;
    let (w, h) = reader
        .into_dimensions()
        .map_err(|e| VisionError::Image(e.to_string()))?;
    if w > MAX_SIDE || h > MAX_SIDE {
        return Err(VisionError::Image(format!("{w}×{h} is too large")));
    }
    // An animated GIF decodes to its first frame, which is what the page
    // shows first too.
    image::load_from_memory(bytes)
        .map(|img| img.to_rgb8())
        .map_err(|e| VisionError::Image(e.to_string()))
}

/// Short side to `spec.size` (bicubic), centre crop, normalise; CHW floats.
pub(crate) fn square(img: &RgbImage, spec: InputSpec) -> Vec<f32> {
    let size = spec.size;
    let (w, h) = img.dimensions();
    let scale = size as f32 / w.min(h).max(1) as f32;
    let nw = ((w as f32 * scale).round() as u32).max(size);
    let nh = ((h as f32 * scale).round() as u32).max(size);
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
        };
        let out = square(&img, spec);
        assert_eq!(out.len(), 3 * 4 * 4);
        // The centre crop straddles the edge: dark on the left, light on the right.
        assert!(out[0] < 0.2);
        assert!(out[3] > 0.8);
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(decode(b"not an image").is_err());
    }
}
