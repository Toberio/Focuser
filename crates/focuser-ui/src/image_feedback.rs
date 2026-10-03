//! Debug-only: images the user marked as wrongly hidden or wrongly shown.
//!
//! The extension's debug build puts a Show/Hide button on judged images. A
//! confirmed click sends the image here, and it is kept with its scores so the
//! filter's thresholds and prompts can be tuned against real pages:
//!
//! - `image-feedback/labels.jsonl`: one line per click;
//! - `image-feedback/<hash>.<ext>`: the image itself, to re-score later.
//!
//! Local to this fork. Nothing is sent anywhere; delete the folder to forget.

use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

use focuser_common::types::ImageFilter;

static DIR: OnceLock<PathBuf> = OnceLock::new();

pub fn init(dir: PathBuf) {
    let _ = DIR.set(dir);
}

fn extension(bytes: &[u8]) -> &'static str {
    match bytes {
        [0xFF, 0xD8, ..] => "jpg",
        [0x89, b'P', b'N', b'G', ..] => "png",
        [b'G', b'I', b'F', ..] => "gif",
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
        ] => "webp",
        _ => "bin",
    }
}

/// Keep one labelled image. `label` is what the user says should happen.
pub fn record(bytes: &[u8], label: &str, url: &str, level: ImageFilter) -> Result<String, String> {
    let dir = DIR.get().ok_or("feedback is not set up")?;
    if label != "show" && label != "hide" {
        return Err(format!("unknown label {label:?}"));
    }
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    let file = format!("{:016x}.{}", hasher.finish(), extension(bytes));
    std::fs::write(dir.join(&file), bytes).map_err(|e| e.to_string())?;

    let scores = crate::image_filter::classify(bytes).ok();
    let line = serde_json::json!({
        "time": chrono::Utc::now().to_rfc3339(),
        "label": label,
        "url": url,
        "level": level,
        "file": file,
        "scores": scores,
        "hidden_at_level": scores.as_ref().and_then(|s| focuser_vision::is_hidden(s, level)),
    });
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("labels.jsonl"))
        .map_err(|e| e.to_string())?;
    writeln!(log, "{line}").map_err(|e| e.to_string())?;
    Ok(file)
}
