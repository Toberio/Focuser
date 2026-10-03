//! Regenerate `src/prompts.json` after changing `prompts::PROMPTS`.
//!
//! ```text
//! cargo run --release -p focuser-vision --example embed_prompts -- \
//!     open_clip_model.safetensors tokenizer.json
//! ```
//!
//! Both files are in https://huggingface.co/timm/vit_base_patch32_clip_224.openai,
//! the same source `models::CLIP_IMAGE` downloads the image tower from.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use focuser_vision::prompts::{Group, PROMPTS};
use focuser_vision::{models, vit, weights::Weights};

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
    let text = vit::clip_text::<Wgpu>(&Weights::read(std::path::Path::new(weights))?, &device)?;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer)?;
    let mut prompts = Vec::new();
    for (group, prompt) in PROMPTS {
        let ids = tokenizer.encode(*prompt, true)?.get_ids().to_vec();
        let embedding = text.embed(&ids, &device);
        let group = match group {
            Group::Nudity => "nudity",
            Group::Suggestive => "suggestive",
            Group::Neutral => "neutral",
        };
        prompts.push(serde_json::json!({ "text": prompt, "group": group, "embedding": embedding }));
    }
    let out = serde_json::json!({
        "model": models::CLIP_IMAGE.name,
        // OpenAI CLIP's trained logit scale, exp(4.6052): 100.
        "logit_scale": 100.0,
        "prompts": prompts,
    });
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/prompts.json");
    std::fs::write(path, serde_json::to_string(&out)? + "\n")?;
    println!("wrote {} prompts to {path}", PROMPTS.len());
    Ok(())
}
