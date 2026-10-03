//! On-device image classification for the explicit-image filter.
//!
//! Two models judge every image, on the GPU through wgpu (Vulkan, Metal or
//! DirectX 12) so nothing vendor-specific needs installing:
//!
//! - Marqo's nsfw-image-detection-384, a small ViT, for nudity;
//! - OpenAI's CLIP ViT-B/32 image tower, compared with a handful of text
//!   prompts, for suggestive pictures, which nudity models do not see.
//!
//! Images never leave the machine. The models are downloaded once, from
//! Hugging Face, the first time the filter is turned on (`models`).

mod classifier;
pub mod models;
mod preprocess;
pub mod probe;
pub mod prompts;
mod verdict;
pub mod vit;
pub mod weights;

pub use classifier::{Classifier, Judged};
pub use verdict::{Scores, is_hidden};

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error("model: {0}")]
    Model(String),
    #[error("image: {0}")]
    Image(String),
    #[error("download: {0}")]
    Download(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, VisionError>;
