//! Regenerate `src/prompts.json` after changing `prompts::PROMPTS`.
//!
//! ```text
//! cargo run --release -p focuser-vision --example embed_prompts -- \
//!     open_clip_model.safetensors tokenizer.json
//! ```
//!
//! Both files are in https://huggingface.co/timm/ViT-B-16-SigLIP2, the same
//! source `models::SIGLIP_IMAGE` downloads the image tower from.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use focuser_vision::prompts::{Group, PROMPTS};
use focuser_vision::{models, vit, weights::Weights};

/// open_clip's cleaning for SigLIP: no punctuation, lower case, single spaces.
fn canonicalise(text: &str) -> String {
    text.replace('_', " ")
        .chars()
        .filter(|c| !c.is_ascii_punctuation())
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let (weights, tokenizer) = match args.as_slice() {
        [_, w, t] => (w, t),
        _ => {
            return Err(
                "usage: embed_prompts <open_clip_model.safetensors> <tokenizer.json>".into(),
            );
        }
    };
    let device = WgpuDevice::default();
    let text = vit::siglip_text::<Wgpu>(&Weights::read(std::path::Path::new(weights))?, &device)?;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer)?;
    let mut prompts = Vec::new();
    for (group, prompt) in PROMPTS {
        // The tokenizer ends each prompt with SigLIP's end token itself.
        let ids = tokenizer
            .encode(canonicalise(prompt), true)?
            .get_ids()
            .to_vec();
        if ids.len() > text.context() {
            return Err(format!("{prompt:?} is longer than SigLIP reads").into());
        }
        let embedding = text.embed(&ids);
        let group = match group {
            Group::Nudity => "nudity",
            Group::Suggestive => "suggestive",
            Group::Neutral => "neutral",
        };
        prompts.push(serde_json::json!({ "text": prompt, "group": group, "embedding": embedding }));
    }
    let out = serde_json::json!({
        "model": models::SIGLIP_IMAGE.name,
        // SigLIP 2 B/16's trained logit scale, exp(4.7245): 112.67. Its logit
        // bias is left out: the filter compares prompts with each other, and
        // a constant added to all of them cancels.
        "logit_scale": 112.668_9,
        "prompts": prompts,
    });
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/prompts.json");
    std::fs::write(path, serde_json::to_string(&out)? + "\n")?;
    println!("wrote {} prompts to {path}", PROMPTS.len());
    Ok(())
}
