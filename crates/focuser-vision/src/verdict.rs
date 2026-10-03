//! From model outputs to "hide this image" at each strictness level.

use focuser_common::types::ImageFilter;
use serde::{Deserialize, Serialize};

/// What a verdict rests on, each 0–1.
///
/// - `nsfw` is Marqo's ViT, trained on nudity against everything else. It is
///   reliable about nudity and knows nothing of "suggestive".
/// - `nudity` and `suggestive` are CLIP's: the share of an image's similarity
///   that goes to prompts describing each, against neutral prompts (sport,
///   portraits, landscapes, food, objects). This is what tells a posed
///   cleavage shot from a sprinter: an earlier classifier (NSFWJS) scored
///   athletes 0.98 "sexy"; CLIP scores them 0.01–0.05 and the posed shots
///   0.80–0.99.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    pub nsfw: f32,
    pub nudity: f32,
    pub suggestive: f32,
}

impl Scores {
    /// Compact form for the extension's `data-focuser-score` attribute.
    pub fn describe(&self) -> String {
        format!(
            "nsfw {:.2} · nudity {:.2} · suggestive {:.2}",
            self.nsfw, self.nudity, self.suggestive
        )
    }
}

struct Thresholds {
    /// Marqo alone.
    nsfw: f32,
    /// CLIP's nudity share, counted only when Marqo gives at least
    /// `nudity_floor`: CLIP's nudity prompts are the first place an image no
    /// prompt describes well lands, and Marqo is the nudity specialist.
    nudity: f32,
    nudity_floor: f32,
    /// CLIP's suggestive share. Infinity: not at this level.
    suggestive: f32,
}

/// Each level's numbers are at or below the one before it, so each level
/// hides everything the one before it does.
///
/// Calibrated on a small set: four suggestive photos that should hide
/// (CLIP suggestive 0.72–0.92) against 56 that should not, including athletes,
/// dancers, a festival crowd, abstract art (suggestive at most 0.43, Marqo at
/// most 0.17) and three ordinary fashion photos (suggestive up to
/// 0.66, nudity up to 0.47 with Marqo 0.24–0.48).
fn thresholds(level: ImageFilter) -> Option<Thresholds> {
    let t = |nsfw, nudity, nudity_floor, suggestive| Thresholds {
        nsfw,
        nudity,
        nudity_floor,
        suggestive,
    };
    match level {
        ImageFilter::Off => None,
        ImageFilter::Explicit => Some(t(0.7, 0.6, 0.4, f32::INFINITY)),
        ImageFilter::Balanced => Some(t(0.6, 0.55, 0.35, 0.85)),
        ImageFilter::Strict => Some(t(0.5, 0.5, 0.3, 0.7)),
    }
}

/// Whether an image should be hidden at `level`. `None` when the filter is off.
pub fn is_hidden(scores: &Scores, level: ImageFilter) -> Option<bool> {
    let t = thresholds(level)?;
    Some(
        scores.nsfw >= t.nsfw
            || (scores.nudity >= t.nudity && scores.nsfw >= t.nudity_floor)
            || scores.suggestive >= t.suggestive,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEVELS: [ImageFilter; 3] = [
        ImageFilter::Explicit,
        ImageFilter::Balanced,
        ImageFilter::Strict,
    ];

    fn hidden_at(nsfw: f32, nudity: f32, suggestive: f32) -> Vec<ImageFilter> {
        let s = Scores {
            nsfw,
            nudity,
            suggestive,
        };
        LEVELS
            .into_iter()
            .filter(|l| is_hidden(&s, *l) == Some(true))
            .collect()
    }

    #[test]
    fn off_judges_nothing() {
        let s = Scores {
            nsfw: 1.0,
            nudity: 1.0,
            suggestive: 1.0,
        };
        assert_eq!(is_hidden(&s, ImageFilter::Off), None);
    }

    #[test]
    fn nudity_is_hidden_at_every_level() {
        assert_eq!(hidden_at(0.9, 0.1, 0.0), LEVELS);
        assert_eq!(hidden_at(0.45, 0.8, 0.1), LEVELS);
    }

    #[test]
    fn clip_alone_cannot_call_nudity() {
        // Real scores: a wallpaper CLIP put nearest the nudity prompts.
        assert!(hidden_at(0.06, 0.48, 0.03).is_empty());
        assert!(hidden_at(0.17, 0.43, 0.19).is_empty());
    }

    #[test]
    fn posed_suggestive_pictures_are_hidden_from_strict_up() {
        // Real scores from photos that should hide.
        assert_eq!(hidden_at(0.07, 0.09, 0.73), [ImageFilter::Strict]);
        assert_eq!(
            hidden_at(0.06, 0.01, 0.92),
            [ImageFilter::Balanced, ImageFilter::Strict]
        );
    }

    #[test]
    fn ordinary_fashion_photos_stay_shown() {
        // Real scores, labelled by hand.
        assert!(hidden_at(0.48, 0.17, 0.66).is_empty());
        assert!(hidden_at(0.38, 0.44, 0.52).is_empty());
        assert!(hidden_at(0.24, 0.47, 0.34).is_empty());
    }

    #[test]
    fn ordinary_photos_show_at_every_level() {
        // Real scores: a festival crowd, an athlete, a dancer.
        assert!(hidden_at(0.11, 0.02, 0.43).is_empty());
        assert!(hidden_at(0.05, 0.0, 0.0).is_empty());
        assert!(hidden_at(0.04, 0.23, 0.31).is_empty());
    }

    #[test]
    fn each_level_hides_everything_the_one_before_does() {
        for a in 0..=10 {
            for b in 0..=10 {
                for c in 0..=10 {
                    let hidden = hidden_at(a as f32 / 10.0, b as f32 / 10.0, c as f32 / 10.0);
                    let first = hidden.first().map_or(LEVELS.len(), |l| {
                        LEVELS.iter().position(|x| x == l).unwrap_or(LEVELS.len())
                    });
                    assert_eq!(hidden, LEVELS[first..]);
                }
            }
        }
    }
}
