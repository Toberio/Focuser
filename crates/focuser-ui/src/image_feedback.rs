//! Debug-only: images the user marked as wrongly hidden or wrongly shown, and
//! the personal filter trained on them.
//!
//! The extension's debug build puts a Show/Hide button on judged images. A
//! confirmed click sends the image here. It is kept with its scores and the
//! image model's embedding, and the user's filter
//! ([`focuser_vision::probe::Probe`]) is retrained on every label so far:
//!
//! - `image-feedback/labels.jsonl`: one line per click;
//! - `image-feedback/<hash>.<ext>`: the image itself;
//! - `image-feedback/embeddings/<model>/<hash>.frames.json`: the embedding of
//!   each judged frame, for labels saved before embeddings (or an
//!   animation's frames) were kept, or by an earlier model. A new model
//!   re-embeds every saved image once, so no label is lost to a switch;
//! - `image-feedback/probe.json`: the trained filter, for inspection.
//!
//! Local to this fork. Nothing is sent anywhere; delete the folder to forget.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, RwLock};

use focuser_common::types::ImageFilter;
use focuser_vision::Scores;
use focuser_vision::probe::Probe;
use tracing::{info, warn};

/// The model whose embeddings the user's filter is trained on.
const EMBEDDER: &str = focuser_vision::models::SIGLIP_IMAGE.name;

static DIR: OnceLock<PathBuf> = OnceLock::new();
static PROBE: RwLock<Option<Probe>> = RwLock::new(None);
/// One retrain at a time; labels arriving meanwhile wait for it.
static TRAINING: Mutex<()> = Mutex::new(());

pub fn init(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Each judged frame's scores, with the user's filter's say added.
pub fn scores_of(frames: &[focuser_vision::Judged]) -> Vec<Scores> {
    frames
        .iter()
        .map(|j| {
            let mut s = j.scores;
            s.personal = personal(&j.embedding);
            s
        })
        .collect()
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

    // Every frame the filter judges, as it judged them: an animation is
    // hidden by any one.
    let frames = crate::image_filter::judge_frames(bytes).unwrap_or_default();
    let scores = scores_of(&frames);
    let hidden = focuser_vision::first_hidden(&scores, level);
    let line = serde_json::json!({
        "time": chrono::Utc::now().to_rfc3339(),
        "label": label,
        "url": url,
        "level": level,
        "file": file,
        "scores": scores.get(hidden.unwrap_or(0)),
        "hidden_at_level": (!scores.is_empty() && level != ImageFilter::Off).then_some(hidden.is_some()),
        "embedding": frames.first().map(|j| &j.embedding),
        "frames": (frames.len() > 1).then(|| {
            frames
                .iter()
                .map(|j| serde_json::json!({ "scores": j.scores, "embedding": j.embedding }))
                .collect::<Vec<_>>()
        }),
        "model": EMBEDDER,
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
    // Each image's latest label, and the embeddings logged with any label.
    let mut latest: HashMap<String, (bool, Option<Vec<Logged>>)> = HashMap::new();
    for line in log.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let (Some(file), Some(label)) = (v["file"].as_str(), v["label"].as_str()) else {
            continue;
        };
        let logged = logged(&v);
        let previous = latest.remove(file).and_then(|(_, e)| e);
        latest.insert(file.to_string(), (label == "hide", logged.or(previous)));
    }

    let mut examples = Vec::with_capacity(latest.len());
    for (file, (hide, logged)) in latest {
        // An animation logged with one frame was labelled before frames were.
        let frames = match logged {
            Some(frames) if !(animated(&file) && frames.len() == 1) => Some(frames),
            _ => cached_frames(dir, &file),
        };
        examples.extend(frames.map(|f| examples_of(f, hide)).unwrap_or_default());
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

/// One judged frame of a labelled image, as the log keeps it.
#[derive(serde::Serialize, serde::Deserialize)]
struct Logged {
    embedding: Vec<f32>,
    /// How nude or suggestive the prompts found it.
    explicitness: f32,
}

/// The frames logged with a label, if this model logged them: another
/// model's numbers mean nothing to this one's filter.
fn logged(v: &serde_json::Value) -> Option<Vec<Logged>> {
    if v["model"].as_str() != Some(EMBEDDER) {
        return None;
    }
    let frame = |f: &serde_json::Value| {
        let scores = serde_json::from_value::<Scores>(f["scores"].clone()).ok();
        Some(Logged {
            embedding: serde_json::from_value(f["embedding"].clone()).ok()?,
            explicitness: scores.map_or(0.0, |s| s.nudity + s.suggestive),
        })
    };
    match v["frames"].as_array() {
        Some(frames) => {
            Some(frames.iter().filter_map(frame).collect::<Vec<_>>()).filter(|f| !f.is_empty())
        }
        None => frame(v).map(|f| vec![f]),
    }
}

/// What a label teaches. "Show" on an animation means every frame is fine.
/// "Hide" means some frame is not, and the harmless ones (often the first)
/// must not be learned as hide-worthy: only the frame the prompts find most
/// explicit is.
fn examples_of(frames: Vec<Logged>, hide: bool) -> Vec<(Vec<f32>, bool)> {
    if !hide {
        return frames.into_iter().map(|f| (f.embedding, false)).collect();
    }
    frames
        .into_iter()
        .max_by(|a, b| a.explicitness.total_cmp(&b.explicitness))
        .map(|f| vec![(f.embedding, true)])
        .unwrap_or_default()
}

/// Whether a saved image may be an animation, judged on several frames.
fn animated(file: &str) -> bool {
    file.ends_with(".gif") || file.ends_with(".webp")
}

/// A label's frames where its log line has none for this model: computed
/// from the saved image once, then kept. An animation labelled before its
/// frames were logged is judged again here, frame by frame.
fn cached_frames(dir: &Path, file: &str) -> Option<Vec<Logged>> {
    let cached = dir.join("embeddings").join(EMBEDDER);
    let cache = cached.join(format!("{file}.frames.json"));
    if let Ok(text) = std::fs::read_to_string(&cache)
        && let Ok(frames) = serde_json::from_str(&text)
    {
        return Some(frames);
    }
    // One embedding per still image, as earlier versions kept them.
    if !animated(file)
        && let Ok(text) = std::fs::read_to_string(cached.join(format!("{file}.json")))
        && let Ok(embedding) = serde_json::from_str(&text)
    {
        return Some(vec![Logged {
            embedding,
            explicitness: 0.0,
        }]);
    }
    let bytes = std::fs::read(dir.join(file)).ok()?;
    let frames: Vec<Logged> = match crate::image_filter::judge_frames(&bytes) {
        Ok(frames) => frames
            .into_iter()
            .map(|j| Logged {
                explicitness: j.scores.nudity + j.scores.suggestive,
                embedding: j.embedding,
            })
            .collect(),
        Err(_) => {
            warn!(file, "could not embed a labelled image");
            return None;
        }
    };
    let _ = std::fs::create_dir_all(&cached);
    let _ = std::fs::write(&cache, serde_json::to_string(&frames).unwrap_or_default());
    Some(frames)
}
