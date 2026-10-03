//! Fetching the two models on first use.
//!
//! The weights are not shipped with the app: they are 190 MB, and most people
//! never turn the filter on. They come from Hugging Face the first time it is
//! turned on, and are checked twice:
//!
//! - the source file must be the one pinned here (Hugging Face reports each
//!   file's SHA-256 in `x-linked-etag`), and
//! - the converted file written to disk must hash to the value pinned here,
//!   so a truncated or tampered download is never loaded.
//!
//! Only the tensors needed are downloaded, by byte range: CLIP's image tower
//! is 350 MB of a 605 MB file whose text half the app never uses. Everything
//! is stored as float16, halving it again.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tracing::info;

use crate::{Result, VisionError};

pub struct ModelSpec {
    /// Also the file name on disk, with `.safetensors`.
    pub name: &'static str,
    /// Pinned to a repository commit, not `main`.
    pub url: &'static str,
    /// SHA-256 of the whole remote file, as Hugging Face reports it.
    pub source_sha256: &'static str,
    /// Tensors to keep, by name prefix. Empty keeps all.
    pub keep_prefix: &'static str,
    /// SHA-256 of the float16 file written to disk.
    pub output_sha256: &'static str,
    pub licence: &'static str,
}

/// Marqo/nsfw-image-detection-384 (Apache-2.0): nudity, 11 MB on disk.
pub const MARQO: ModelSpec = ModelSpec {
    name: "marqo-nsfw-384",
    url: "https://huggingface.co/Marqo/nsfw-image-detection-384/resolve/0c26ec22111b83f106d72a55f611ec35962bcb65/model.safetensors",
    source_sha256: "6bf2e0f64a1d20169736c2836e3a787b12379fdc08ba87f7d94a7a3d58eeefce",
    keep_prefix: "",
    output_sha256: "afea5a2d46e301d8b2fd07dea7835f9383b69ce32c2ae648eb67026f010305ee",
    licence: "Apache-2.0",
};

/// OpenAI CLIP ViT-B/32 (MIT), image tower only: 176 MB on disk. Taken from
/// timm's mirror, which has it as safetensors; OpenAI's own repository has
/// only a pickle.
pub const CLIP_IMAGE: ModelSpec = ModelSpec {
    name: "openai-clip-vit-b-32-image",
    url: "https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/a6f597a30f7b82c51704746581f9a4e41421e878/open_clip_model.safetensors",
    source_sha256: "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31",
    keep_prefix: "visual.",
    output_sha256: "775b9cbd3b597f54e6784a04448d96143f5507a8a4c11d376a85ec5c44709bc7",
    licence: "MIT",
};

pub const ALL: [&ModelSpec; 2] = [&MARQO, &CLIP_IMAGE];

pub fn path(dir: &Path, spec: &ModelSpec) -> PathBuf {
    dir.join(format!("{}.safetensors", spec.name))
}

/// Whether every model is on disk and intact.
pub fn present(dir: &Path) -> bool {
    ALL.iter()
        .all(|spec| verify(&path(dir, spec), spec).is_ok())
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn verify(file: &Path, spec: &ModelSpec) -> Result<()> {
    let bytes = std::fs::read(file)?;
    check_output(&bytes, spec)
}

fn check_output(bytes: &[u8], spec: &ModelSpec) -> Result<()> {
    if spec.output_sha256.is_empty() {
        // Not pinned yet: a development build, where the hash is being found.
        return Ok(());
    }
    let digest = sha256_hex(bytes);
    if digest != spec.output_sha256 {
        return Err(VisionError::Model(format!(
            "{}: converted weights hash {digest}, expected {}",
            spec.name, spec.output_sha256
        )));
    }
    Ok(())
}

/// Progress across all models: bytes received so far, and the total.
pub type Progress<'a> = &'a mut dyn FnMut(u64, u64);

/// Download whatever is missing into `dir`. Safe to call when all is present.
pub fn ensure(dir: &Path, progress: Progress) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    // reqwest is built without a default TLS crypto provider, as in the
    // updater; install the same one it does if nobody has yet.
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
    let client = reqwest::blocking::Client::builder()
        .user_agent("Focuser image filter")
        .build()
        .map_err(|e| VisionError::Download(e.to_string()))?;

    let missing: Vec<_> = ALL
        .iter()
        .filter(|s| verify(&path(dir, s), s).is_err())
        .collect();
    let mut plans = Vec::new();
    for spec in &missing {
        plans.push((*spec, plan(&client, spec)?));
    }
    let total: u64 = plans.iter().map(|(_, p)| p.bytes()).sum();
    let mut done = 0u64;
    progress(done, total);
    for (spec, plan) in plans {
        let out = fetch(&client, spec, &plan, &mut |n| {
            done += n;
            progress(done, total);
        })?;
        check_output(&out, spec)?;
        if spec.output_sha256.is_empty() {
            info!(model = spec.name, sha256 = %sha256_hex(&out), "converted weights (hash not pinned)");
        }
        // Written beside, then renamed: a half-written file is never loaded.
        let file = path(dir, spec);
        let tmp = file.with_extension("partial");
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&out)?;
        f.sync_all()?;
        std::fs::rename(&tmp, &file)?;
        info!(
            model = spec.name,
            licence = spec.licence,
            bytes = out.len(),
            "image filter model ready"
        );
    }
    Ok(())
}

#[derive(serde::Deserialize, Clone)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

struct Plan {
    /// Where the bytes are: Hugging Face redirects each file to its CDN.
    url: String,
    data_start: u64,
    tensors: BTreeMap<String, TensorInfo>,
    /// Byte ranges to request, relative to `data_start`, end exclusive.
    ranges: Vec<(u64, u64)>,
}

impl Plan {
    fn bytes(&self) -> u64 {
        self.ranges.iter().map(|(a, b)| b - a).sum()
    }
}

fn get_range(
    client: &reqwest::blocking::Client,
    url: &str,
    from: u64,
    to_inclusive: u64,
) -> Result<reqwest::blocking::Response> {
    let response = client
        .get(url)
        .header("Range", format!("bytes={from}-{to_inclusive}"))
        .send()
        .map_err(|e| VisionError::Download(e.to_string()))?;
    if response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(VisionError::Download(format!(
            "{url}: HTTP {}",
            response.status()
        )));
    }
    Ok(response)
}

/// Read the remote header and work out which byte ranges are needed.
fn plan(client: &reqwest::blocking::Client, spec: &ModelSpec) -> Result<Plan> {
    // Hugging Face answers with a redirect to its CDN, and names the file's
    // content by its SHA-256 on that redirect. A different file is refused
    // before anything is read from it.
    let no_redirect = reqwest::blocking::Client::builder()
        .user_agent("Focuser image filter")
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| VisionError::Download(e.to_string()))?;
    let head = no_redirect
        .head(spec.url)
        .send()
        .map_err(|e| VisionError::Download(e.to_string()))?;
    let header = |name: &str| {
        head.headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .trim_matches('"')
            .to_string()
    };
    let etag = header("x-linked-etag");
    if etag != spec.source_sha256 {
        return Err(VisionError::Download(format!(
            "{}: the source file is not the pinned one (etag {etag:?}); refusing it",
            spec.name
        )));
    }
    let url = match header("location") {
        l if l.starts_with("https://") => l,
        _ => spec.url.to_string(),
    };

    let first = get_range(client, &url, 0, 7)?;
    let len_bytes = first
        .bytes()
        .map_err(|e| VisionError::Download(e.to_string()))?;
    let header_len = u64::from_le_bytes(
        len_bytes
            .as_ref()
            .try_into()
            .map_err(|_| VisionError::Download("short safetensors header".into()))?,
    );
    if header_len > 16 * 1024 * 1024 {
        return Err(VisionError::Download(
            "implausible safetensors header".into(),
        ));
    }
    let header = get_range(client, &url, 8, 7 + header_len)?
        .bytes()
        .map_err(|e| VisionError::Download(e.to_string()))?;
    let mut all: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header).map_err(|e| VisionError::Download(e.to_string()))?;
    all.remove("__metadata__");
    let mut tensors = BTreeMap::new();
    for (name, value) in all {
        if !name.starts_with(spec.keep_prefix) || name.ends_with("position_ids") {
            continue;
        }
        let info: TensorInfo =
            serde_json::from_value(value).map_err(|e| VisionError::Download(e.to_string()))?;
        tensors.insert(name, info);
    }
    if tensors.is_empty() {
        return Err(VisionError::Download(format!(
            "{}: no tensors to keep",
            spec.name
        )));
    }

    // Neighbouring tensors are fetched together; a small gap costs less than
    // another round trip.
    let mut spans: Vec<(u64, u64)> = tensors
        .values()
        .map(|t| (t.data_offsets[0], t.data_offsets[1]))
        .collect();
    spans.sort_unstable();
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    for (a, b) in spans {
        match ranges.last_mut() {
            Some(last) if a.saturating_sub(last.1) < 256 * 1024 => last.1 = last.1.max(b),
            _ => ranges.push((a, b)),
        }
    }
    Ok(Plan {
        url,
        data_start: 8 + header_len,
        tensors,
        ranges,
    })
}

/// Download the planned ranges and write them out as a float16 safetensors file.
fn fetch(
    client: &reqwest::blocking::Client,
    spec: &ModelSpec,
    plan: &Plan,
    progress: &mut dyn FnMut(u64),
) -> Result<Vec<u8>> {
    use std::io::Read;

    let mut raw: Vec<(u64, Vec<u8>)> = Vec::new();
    for &(a, b) in &plan.ranges {
        let mut response = get_range(
            client,
            &plan.url,
            plan.data_start + a,
            plan.data_start + b - 1,
        )?;
        let mut buf = Vec::with_capacity((b - a) as usize);
        let mut chunk = vec![0u8; 1 << 20];
        loop {
            let n = response
                .read(&mut chunk)
                .map_err(|e| VisionError::Download(e.to_string()))?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
            progress(n as u64);
        }
        if buf.len() as u64 != b - a {
            return Err(VisionError::Download(format!("{}: short read", spec.name)));
        }
        raw.push((a, buf));
    }
    let slice = |from: u64, to: u64| -> Result<&[u8]> {
        raw.iter()
            .find(|(a, buf)| *a <= from && to <= a + buf.len() as u64)
            .map(|(a, buf)| &buf[(from - a) as usize..(to - a) as usize])
            .ok_or_else(|| VisionError::Download("tensor outside downloaded ranges".into()))
    };

    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, info) in &plan.tensors {
        let bytes = slice(info.data_offsets[0], info.data_offsets[1])?;
        let halves: Vec<u8> = match info.dtype.as_str() {
            "F32" => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|c| half::f16::from_f32(f32::from_le_bytes(*c)).to_le_bytes())
                .collect(),
            "F16" => bytes.to_vec(),
            other => {
                return Err(VisionError::Download(format!(
                    "{name}: unsupported dtype {other}"
                )));
            }
        };
        let start = data.len();
        data.extend_from_slice(&halves);
        header.insert(
            name.clone(),
            serde_json::json!({ "dtype": "F16", "shape": info.shape, "data_offsets": [start, data.len()] }),
        );
    }
    let mut header =
        serde_json::to_vec(&header).map_err(|e| VisionError::Download(e.to_string()))?;
    // safetensors pads the header to 8 bytes with spaces.
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut out = Vec::with_capacity(8 + header.len() + data.len());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    Ok(out)
}
