//! SigLIP's half of the judgement: how an image compares with descriptions.
//!
//! The descriptions are embedded once, ahead of time, by
//! `examples/embed_prompts.rs`, so the app never needs SigLIP's text tower or
//! tokenizer. Change a prompt and that example must be run again; the test
//! below fails if the two drift apart.

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Group {
    Nudity,
    Suggestive,
    Neutral,
}

/// The prompts, in the order their embeddings are stored.
pub const PROMPTS: &[(Group, &str)] = &[
    (Group::Nudity, "a nude photo of a person"),
    (Group::Nudity, "a pornographic image"),
    (Group::Nudity, "a photo of exposed breasts or genitals"),
    (Group::Suggestive, "a sexually suggestive photo of a woman"),
    (Group::Suggestive, "a woman showing cleavage"),
    (
        Group::Suggestive,
        "a woman in lingerie or a bikini posing seductively",
    ),
    (
        Group::Suggestive,
        "a provocative selfie of a woman in a tight revealing outfit",
    ),
    (Group::Suggestive, "a thirst trap photo"),
    (Group::Neutral, "a photo of an athlete playing sport"),
    (Group::Neutral, "a photo of a person in everyday clothes"),
    (Group::Neutral, "a portrait photo of a person"),
    (Group::Neutral, "a photo of people at a public event"),
    (Group::Neutral, "a landscape photo"),
    (Group::Neutral, "abstract art"),
    (Group::Neutral, "a photo of food"),
    (Group::Neutral, "a photo of an object"),
    // Ordinary images need somewhere to land. Without these, a wallpaper or a
    // flower spread its similarity over the nearest prompts, nudity included.
    (Group::Neutral, "a photo of nature or flowers"),
    (Group::Neutral, "a photo of an animal"),
    (Group::Neutral, "a photo of a building or a city"),
    (Group::Neutral, "a screenshot or a picture of text"),
    (Group::Neutral, "a colourful abstract wallpaper"),
    // What these models took for nudity or suggestive in testing was mostly
    // art, tattoos on skin, and people in activewear.
    (Group::Neutral, "a painting or drawing of a person"),
    (Group::Neutral, "a digital illustration or anime artwork"),
    (Group::Neutral, "a close-up photo of a tattoo on skin"),
    (Group::Neutral, "a woman doing yoga or meditating"),
    (Group::Neutral, "a person in sportswear or a tank top"),
];

#[derive(Deserialize)]
pub(crate) struct Embedded {
    pub model: String,
    pub logit_scale: f32,
    pub prompts: Vec<EmbeddedPrompt>,
}

#[derive(Deserialize)]
pub(crate) struct EmbeddedPrompt {
    /// Compared with `PROMPTS` by the test below; the app goes by position.
    #[cfg_attr(not(test), allow(dead_code))]
    pub text: String,
    pub group: Group,
    pub embedding: Vec<f32>,
}

pub(crate) const EMBEDDED_JSON: &str = include_str!("prompts.json");

pub(crate) fn embedded() -> crate::Result<Embedded> {
    let e: Embedded = serde_json::from_str(EMBEDDED_JSON)
        .map_err(|e| crate::VisionError::Model(e.to_string()))?;
    // Embeddings from another model would compare as noise against this one's
    // images, and every verdict would be wrong without anything failing.
    if e.model != crate::models::SIGLIP_IMAGE.name || e.prompts.len() != PROMPTS.len() {
        return Err(crate::VisionError::Model(
            "prompts.json does not match the SigLIP model; run examples/embed_prompts.rs".into(),
        ));
    }
    Ok(e)
}

/// Shares of probability per group, from one image's similarities.
pub(crate) fn shares(logits: &[f32], groups: &[Group]) -> (f32, f32) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let total: f32 = exp.iter().sum();
    let share = |g: Group| {
        exp.iter()
            .zip(groups)
            .filter(|(_, x)| **x == g)
            .map(|(e, _)| e)
            .sum::<f32>()
            / total
    };
    (share(Group::Nudity), share(Group::Suggestive))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_prompts_match_the_list() {
        let e = embedded().expect("prompts.json parses");
        assert_eq!(e.model, crate::models::SIGLIP_IMAGE.name);
        let texts: Vec<_> = e
            .prompts
            .iter()
            .map(|p| (p.group, p.text.as_str()))
            .collect();
        assert_eq!(texts, PROMPTS, "run examples/embed_prompts.rs again");
        assert!(e.prompts.iter().all(|p| p.embedding.len() == 768));
    }

    #[test]
    fn shares_add_up_to_their_groups() {
        let groups = [
            Group::Nudity,
            Group::Suggestive,
            Group::Neutral,
            Group::Neutral,
        ];
        let (nudity, suggestive) = shares(&[0.0, 0.0, 0.0, 0.0], &groups);
        assert!((nudity - 0.25).abs() < 1e-6);
        assert!((suggestive - 0.25).abs() < 1e-6);
    }
}
