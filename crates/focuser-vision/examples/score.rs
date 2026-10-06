//! Score images from the command line, for tuning the thresholds.
//!
//! ```text
//! cargo run --release -p focuser-vision --example score -- <models dir> <image>...
//! ```
//!
//! Downloads the models into the directory first if they are not there.

use focuser_common::types::ImageFilter;
use focuser_vision::{Classifier, is_hidden, models};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let Some((dir, images)) = args.get(1).map(|d| (std::path::Path::new(d), &args[2..])) else {
        return Err("usage: score <models dir> <image>...".into());
    };
    models::ensure(dir, &mut |done, total| {
        eprint!("\rdownloading {done}/{total} bytes")
    })?;
    eprintln!();
    let t = Instant::now();
    let classifier = Classifier::load(dir)?;
    eprintln!("loaded in {:?}", t.elapsed());
    let t = Instant::now();
    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = images
            .iter()
            .map(|path| {
                let c = classifier.clone();
                s.spawn(move || {
                    (
                        path,
                        std::fs::read(path)
                            .map_err(Into::into)
                            .and_then(|b| c.judge_frames(&b)),
                    )
                })
            })
            .collect();
        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    let elapsed = t.elapsed();
    for (path, result) in results {
        let name = std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?");
        match result {
            Ok(frames) => {
                for (i, judged) in frames.iter().enumerate() {
                    let s = judged.scores;
                    // Frames after the first get their own names.
                    let name = match i {
                        0 => name.to_string(),
                        i => format!("{name}#{i}"),
                    };
                    if let Ok(dir) = std::env::var("FOCUSER_EMBEDDINGS_DIR") {
                        let file = std::path::Path::new(&dir).join(format!("{name}.json"));
                        let _ = std::fs::write(
                            file,
                            serde_json::to_string(&judged.embedding).unwrap_or_default(),
                        );
                    }
                    let levels: Vec<&str> = [
                        (ImageFilter::Explicit, "E"),
                        (ImageFilter::Balanced, "B"),
                        (ImageFilter::Strict, "S"),
                    ]
                    .iter()
                    .filter(|(l, _)| is_hidden(&s, *l) == Some(true))
                    .map(|(_, n)| *n)
                    .collect();
                    println!("{name:28} {} hidden at [{}]", s.describe(), levels.join(""));
                }
            }
            Err(e) => println!("{name:28} error: {e}"),
        }
    }
    eprintln!("{} images in {elapsed:?}", images.len());
    Ok(())
}
