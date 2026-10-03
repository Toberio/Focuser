//! From model outputs to "hide this image" at each strictness level.

use focuser_common::types::ImageFilter;
use serde::{Deserialize, Serialize};

/// What a verdict rests on, each 0–1.
///
/// - `nsfw` is Marqo's ViT, trained on nudity against everything else. It is
///   reliable about real nudity, but close-ups of skin (tattoos) and painted
///   skin set it off too, so it never decides alone.
/// - `nudity` and `suggestive` are CLIP's: the share of an image's similarity
///   that goes to prompts describing each, against neutral prompts (sport,
///   portraits, art, tattoos, activewear and more).
/// - `personal` is the user's own trained filter ([`crate::probe::Probe`]),
///   when they have labelled enough images: how likely they would want this
///   one hidden.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    pub nsfw: f32,
    pub nudity: f32,
    pub suggestive: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub personal: Option<f32>,
}

impl Scores {
    /// Compact form for the extension's `data-focuser-score` attribute.
    pub fn describe(&self) -> String {
        let mut s = format!(
            "nsfw {:.2} · nudity {:.2} · suggestive {:.2}",
            self.nsfw, self.nudity, self.suggestive
        );
        if let Some(p) = self.personal {
            s.push_str(&format!(" · yours {p:.2}"));
        }
        s
    }
}

struct Thresholds {
    /// Nudity: Marqo at least `nsfw` *and* CLIP's nudity plus suggestive at
    /// least `clip`. Both, because each misfires where the other does not.
    nsfw: f32,
    clip: f32,
    /// The user's filter, where there is one. Infinity: not at this level.
    personal: f32,
    /// Without one: CLIP's suggestive share alone.
    suggestive: f32,
}

/// Each level's numbers are at or below the one before it, so each level
/// hides everything the one before it does.
fn thresholds(level: ImageFilter) -> Option<Thresholds> {
    let t = |nsfw, clip, personal, suggestive| Thresholds {
        nsfw,
        clip,
        personal,
        suggestive,
    };
    match level {
        ImageFilter::Off => None,
        ImageFilter::Explicit => Some(t(0.8, 0.7, f32::INFINITY, f32::INFINITY)),
        ImageFilter::Balanced => Some(t(0.7, 0.65, 0.5, 0.85)),
        ImageFilter::Strict => Some(t(0.6, 0.6, 0.3, 0.7)),
    }
}

/// Below this the user's filter is sure they would want the image shown, and
/// it may overrule the nudity check at the levels it applies to. Their labels
/// taught it that paintings and tattoos are fine; the nudity models never
/// learn that.
const PERSONAL_SURE_SHOW: f32 = 0.05;

/// Whether an image should be hidden at `level`. `None` when the filter is off.
pub fn is_hidden(scores: &Scores, level: ImageFilter) -> Option<bool> {
    let t = thresholds(level)?;
    let personal = scores.personal.filter(|_| t.personal.is_finite());
    let nudity = scores.nsfw >= t.nsfw && scores.nudity + scores.suggestive >= t.clip;
    if nudity && personal.is_none_or(|p| p >= PERSONAL_SURE_SHOW) {
        return Some(true);
    }
    Some(match personal {
        Some(p) => p >= t.personal,
        None => scores.suggestive >= t.suggestive,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEVELS: [ImageFilter; 3] = [
        ImageFilter::Explicit,
        ImageFilter::Balanced,
        ImageFilter::Strict,
    ];

    fn scores(nsfw: f32, nudity: f32, suggestive: f32, personal: Option<f32>) -> Scores {
        Scores {
            nsfw,
            nudity,
            suggestive,
            personal,
        }
    }

    fn hidden_at(s: Scores) -> Vec<ImageFilter> {
        LEVELS
            .into_iter()
            .filter(|l| is_hidden(&s, *l) == Some(true))
            .collect()
    }

    #[test]
    fn off_judges_nothing() {
        assert_eq!(
            is_hidden(&scores(1.0, 1.0, 1.0, Some(1.0)), ImageFilter::Off),
            None
        );
    }

    #[test]
    fn nudity_needs_both_models() {
        assert_eq!(hidden_at(scores(0.9, 0.8, 0.1, None)), LEVELS);
        // Real scores: tattoos and paintings Marqo took for nudity, CLIP not.
        assert!(hidden_at(scores(0.95, 0.01, 0.01, None)).is_empty());
        assert!(hidden_at(scores(0.85, 0.0, 0.0, None)).is_empty());
        // And a wallpaper CLIP put nearest its nudity prompts, Marqo not.
        assert!(hidden_at(scores(0.06, 0.48, 0.03, None)).is_empty());
    }

    #[test]
    fn without_labels_suggestive_prompts_decide() {
        assert_eq!(
            hidden_at(scores(0.07, 0.09, 0.73, None)),
            [ImageFilter::Strict]
        );
        assert_eq!(
            hidden_at(scores(0.06, 0.01, 0.92, None)),
            [ImageFilter::Balanced, ImageFilter::Strict]
        );
    }

    #[test]
    fn with_labels_the_users_filter_decides() {
        // CLIP calls it suggestive; the user taught their filter otherwise.
        assert!(hidden_at(scores(0.08, 0.09, 0.80, Some(0.1))).is_empty());
        // CLIP saw little; the user's filter knows what they meant.
        assert_eq!(
            hidden_at(scores(0.26, 0.02, 0.21, Some(0.4))),
            [ImageFilter::Strict]
        );
        assert_eq!(
            hidden_at(scores(0.26, 0.02, 0.21, Some(0.9))),
            [ImageFilter::Balanced, ImageFilter::Strict]
        );
    }

    #[test]
    fn a_sure_users_filter_overrules_the_nudity_check_but_not_at_explicit() {
        // Both nudity models agree; the user's labels say this kind is fine.
        assert_eq!(
            hidden_at(scores(0.9, 0.8, 0.1, Some(0.01))),
            [ImageFilter::Explicit]
        );
        // Unsure, the user's filter does not overrule them.
        assert_eq!(hidden_at(scores(0.9, 0.8, 0.1, Some(0.2))), LEVELS);
        // Real scores: a painting both nudity models half-believed. Below the
        // explicit bar, and the user's filter shows it.
        assert!(hidden_at(scores(0.87, 0.51, 0.12, Some(0.01))).is_empty());
    }

    #[test]
    fn each_level_hides_everything_the_one_before_does_without_labels() {
        for a in 0..=10 {
            for b in 0..=10 {
                for c in 0..=10 {
                    let hidden = hidden_at(scores(
                        a as f32 / 10.0,
                        b as f32 / 10.0,
                        c as f32 / 10.0,
                        None,
                    ));
                    let first = hidden.first().map_or(LEVELS.len(), |l| {
                        LEVELS.iter().position(|x| x == l).unwrap_or(LEVELS.len())
                    });
                    assert_eq!(hidden, LEVELS[first..]);
                }
            }
        }
    }
}
