//! Debug-only: images the user marked as wrongly hidden or wrongly shown, and
//! the personal filter trained on them.
//!
//! The extension's debug build puts a Show/Hide button on judged images. A
//! confirmed click sends the image here. It is kept with its scores and CLIP's
//! embedding, and the user's filter ([`focuser_vision::probe::Probe`]) is
//! retrained on every label so far:
//!
//! - `image-feedback/labels.jsonl`: one line per click;
//! - `image-feedback/<hash>.<ext>`: the image itself;
//! - `image-feedback/embeddings/<hash>.json`: CLIP's embedding, for labels
//!   saved before embeddings were kept;
//! - `image-feedback/probe.json`: the trained filter, for inspection.
//!
//! Local to this fork. Nothing is sent anywhere; delete the folder to forget.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, RwLock};

use focuser_common::types::ImageFilter;
use focuser_vision::probe::Probe;
use tracing::{info, warn};

static DIR: OnceLock<PathBuf> = OnceLock::new();
static PROBE: RwLock<Option<Probe>> = RwLock::new(None);
/// One retrain at a time; labels arriving meanwhile wait for it.
static TRAINING: Mutex<()> = Mutex::new(());

pub fn init(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// How likely the user would want this image hidden, once they have labelled
/// enough to say.
pub fn personal(embedding: &[f32]) -> Option<f32> {
    PROBE
        .read()
        .ok()?
        .as_ref()
        .map(|p| p.probability(embedding))
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

    let judged = crate::image_filter::judge(bytes).ok();
    let scores = judged.as_ref().map(|j| {
        let mut s = j.scores;
        s.personal = personal(&j.embedding);
        s
    });
    let line = serde_json::json!({
        "time": chrono::Utc::now().to_rfc3339(),
        "label": label,
        "url": url,
        "level": level,
        "file": file,
        "scores": scores,
        "hidden_at_level": scores.as_ref().and_then(|s| focuser_vision::is_hidden(s, level)),
        "embedding": judged.as_ref().map(|j| &j.embedding),
    });
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("labels.jsonl"))
        .map_err(|e| e.to_string())?;
    writeln!(log, "{line}").map_err(|e| e.to_string())?;

    std::thread::spawn(retrain);
    Ok(file)
}

/// Train the user's filter on every label so far. The last label for an
/// image wins: a mind changed is a mind changed.
pub fn retrain() {
    let Some(dir) = DIR.get() else { return };
    let Ok(_guard) = TRAINING.lock() else { return };
    let Ok(log) = std::fs::read_to_string(dir.join("labels.jsonl")) else {
        return;
    };
    let mut latest: HashMap<String, (bool, Option<Vec<f32>>)> = HashMap::new();
    for line in log.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let (Some(file), Some(label)) = (v["file"].as_str(), v["label"].as_str()) else {
            continue;
        };
        let embedding = serde_json::from_value::<Vec<f32>>(v["embedding"].clone()).ok();
        let previous = latest.remove(file).and_then(|(_, e)| e);
        latest.insert(file.to_string(), (label == "hide", embedding.or(previous)));
    }

    let mut examples = Vec::with_capacity(latest.len());
    for (file, (hide, embedding)) in latest {
        if let Some(e) = embedding.or_else(|| cached_embedding(dir, &file)) {
            examples.push((e, hide));
        }
    }
    let probe = Probe::train(&examples);
    match &probe {
        Some(p) => {
            info!(
                hide = p.hide_labels,
                show = p.show_labels,
                "trained the personal image filter"
            );
            let _ = std::fs::write(
                dir.join("probe.json"),
                serde_json::to_string(p).unwrap_or_default(),
            );
        }
        None => info!(
            labels = examples.len(),
            "not enough labels for a personal image filter yet"
        ),
    }
    if let Ok(mut slot) = PROBE.write() {
        *slot = probe;
    }
}

/// An older label's embedding: computed from its saved image once, then kept.
fn cached_embedding(dir: &Path, file: &str) -> Option<Vec<f32>> {
    let cache = dir.join("embeddings").join(format!("{file}.json"));
    if let Ok(text) = std::fs::read_to_string(&cache)
        && let Ok(e) = serde_json::from_str(&text)
    {
        return Some(e);
    }
    let bytes = std::fs::read(dir.join(file)).ok()?;
    let judged = match crate::image_filter::judge(&bytes) {
        Ok(j) => j,
        Err(_) => {
            warn!(file, "could not embed a labelled image");
            return None;
        }
    };
    let _ = std::fs::create_dir_all(dir.join("embeddings"));
    let _ = std::fs::write(
        &cache,
        serde_json::to_string(&judged.embedding).unwrap_or_default(),
    );
    Some(judged.embedding)
}
