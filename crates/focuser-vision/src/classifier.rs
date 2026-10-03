//! The GPU worker: loads both models once and judges images in batches.

use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Tensor, TensorData, activation};
use tracing::{info, warn};

use crate::preprocess::{self, CLIP, MARQO};
use crate::prompts::{self, Group};
use crate::verdict::Scores;
use crate::vit::{self, Vit};
use crate::weights::Weights;
use crate::{Result, VisionError, models};

type B = Wgpu;

/// Images judged in one pass. A feed arrives a screenful at a time.
const MAX_BATCH: usize = 16;
/// How long a request may wait for its verdict, queue included.
const TIMEOUT: Duration = Duration::from_secs(20);

/// One image's inputs: Marqo's 384 px square, then CLIP's 224 px one.
type Inputs = (Vec<f32>, Vec<f32>);

struct Job {
    marqo: Vec<f32>,
    clip: Vec<f32>,
    reply: mpsc::Sender<Result<Judged>>,
}

/// A handle to the worker. Cheap to clone; every clone feeds the same queue.
#[derive(Clone)]
pub struct Classifier {
    jobs: mpsc::Sender<Job>,
}

impl Classifier {
    /// Load the models from `dir` onto the GPU. Blocks until they are ready,
    /// a few seconds, most of it compiling GPU kernels.
    pub fn load(dir: &Path) -> Result<Self> {
        let marqo = Weights::read(&models::path(dir, &models::MARQO))?;
        let clip = Weights::read(&models::path(dir, &models::CLIP_IMAGE))?;
        let embedded = prompts::embedded()?;

        let (jobs, queue) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
        std::thread::Builder::new()
            .name("image-filter-gpu".into())
            .spawn(move || {
                // wgpu picks the best adapter it finds: the discrete GPU where
                // there is one, else integrated graphics or a software device.
                let device = WgpuDevice::default();
                let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Models::load(&marqo, &clip, &embedded, &device)
                }));
                let models = match loaded {
                    Ok(Ok(m)) => m,
                    Ok(Err(e)) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                    Err(_) => {
                        let _ = ready_tx.send(Err(VisionError::Model(
                            "GPU initialisation panicked".into(),
                        )));
                        return;
                    }
                };
                drop((marqo, clip));
                let _ = ready_tx.send(Ok(()));
                worker(models, queue, &device);
            })?;
        ready_rx
            .recv()
            .map_err(|_| VisionError::Model("image filter worker exited while loading".into()))??;
        info!("image filter models loaded");
        Ok(Self { jobs })
    }

    /// Judge one image. Decoding runs here, on the caller's thread, so many
    /// requests decode in parallel; the GPU work is batched by the worker.
    pub fn classify(&self, bytes: &[u8]) -> Result<Scores> {
        self.judge(bytes).map(|j| j.scores)
    }

    /// The scores and CLIP's embedding of the image: a unit vector of 512
    /// numbers describing what is in it, for learning from labels.
    pub fn judge(&self, bytes: &[u8]) -> Result<Judged> {
        let img = preprocess::decode(bytes)?;
        let job_marqo = preprocess::square(&img, MARQO);
        let job_clip = preprocess::square(&img, CLIP);
        let (reply, result) = mpsc::channel();
        self.jobs
            .send(Job {
                marqo: job_marqo,
                clip: job_clip,
                reply,
            })
            .map_err(|_| VisionError::Model("image filter worker has stopped".into()))?;
        result
            .recv_timeout(TIMEOUT)
            .map_err(|_| VisionError::Model("image filter timed out".into()))?
    }
}

/// Everything the models say about one image.
#[derive(Debug, Clone)]
pub struct Judged {
    pub scores: Scores,
    pub embedding: Vec<f32>,
}

struct Models {
    marqo: Vit<B>,
    clip: Vit<B>,
    /// `[prompts, 512]`, unit vectors.
    prompts: Tensor<B, 2>,
    groups: Vec<Group>,
    logit_scale: f32,
}

impl Models {
    fn load(
        marqo: &Weights,
        clip: &Weights,
        embedded: &prompts::Embedded,
        device: &WgpuDevice,
    ) -> Result<Self> {
        let marqo = vit::marqo(marqo, device)?;
        let clip = vit::clip_image(clip, device)?;
        let dim = embedded.prompts.first().map_or(0, |p| p.embedding.len());
        let flat: Vec<f32> = embedded
            .prompts
            .iter()
            .flat_map(|p| p.embedding.iter().copied())
            .collect();
        let prompts = Tensor::<B, 1>::from_data(
            TensorData::new(flat, [embedded.prompts.len() * dim]),
            device,
        )
        .reshape([embedded.prompts.len(), dim]);
        let models = Self {
            marqo,
            clip,
            prompts,
            groups: embedded.prompts.iter().map(|p| p.group).collect(),
            logit_scale: embedded.logit_scale,
        };
        // Compile the kernels now rather than on the first real image.
        let warm = |size: usize| vec![0f32; 3 * size * size];
        models.run(&[(warm(384), warm(224))], device)?;
        Ok(models)
    }

    fn run(&self, batch: &[Inputs], device: &WgpuDevice) -> Result<Vec<Judged>> {
        let n = batch.len();
        let stack = |size: usize, pick: &dyn Fn(&Inputs) -> &Vec<f32>| {
            let flat: Vec<f32> = batch.iter().flat_map(|b| pick(b).iter().copied()).collect();
            Tensor::<B, 1>::from_data(TensorData::new(flat, [n * 3 * size * size]), device)
                .reshape([n, 3, size, size])
        };
        let nsfw = activation::softmax(self.marqo.forward(stack(384, &|b| &b.0)), 1);
        let embeds = self.clip.forward(stack(224, &|b| &b.1));
        let embeds = embeds.clone() / embeds.powi_scalar(2).sum_dim(1).sqrt();
        let logits = embeds.clone().matmul(self.prompts.clone().transpose()) * self.logit_scale;
        let dim = embeds.dims()[1];
        let embeds = embeds
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| VisionError::Model(format!("{e:?}")))?;

        let nsfw = nsfw
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| VisionError::Model(format!("{e:?}")))?;
        let logits = logits
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| VisionError::Model(format!("{e:?}")))?;
        let p = self.groups.len();
        Ok((0..n)
            .map(|i| {
                let (nudity, suggestive) =
                    prompts::shares(&logits[i * p..(i + 1) * p], &self.groups);
                // Marqo's class 0 is NSFW.
                Judged {
                    scores: Scores {
                        nsfw: nsfw[i * 2],
                        nudity,
                        suggestive,
                        personal: None,
                    },
                    embedding: embeds[i * dim..(i + 1) * dim].to_vec(),
                }
            })
            .collect())
    }
}

fn worker(models: Models, queue: mpsc::Receiver<Job>, device: &WgpuDevice) {
    while let Ok(first) = queue.recv() {
        let mut jobs = vec![first];
        while jobs.len() < MAX_BATCH {
            match queue.try_recv() {
                Ok(job) => jobs.push(job),
                Err(_) => break,
            }
        }
        let inputs: Vec<Inputs> = jobs
            .iter_mut()
            .map(|j| (std::mem::take(&mut j.marqo), std::mem::take(&mut j.clip)))
            .collect();
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| models.run(&inputs, device)));
        match result {
            Ok(Ok(scores)) => {
                for (job, s) in jobs.into_iter().zip(scores) {
                    let _ = job.reply.send(Ok(s));
                }
            }
            Ok(Err(e)) => {
                warn!(error = %e, "image filter batch failed");
                for job in jobs {
                    let _ = job.reply.send(Err(VisionError::Model(e.to_string())));
                }
            }
            Err(_) => {
                warn!("image filter batch panicked");
                for job in jobs {
                    let _ = job
                        .reply
                        .send(Err(VisionError::Model("inference panicked".into())));
                }
            }
        }
    }
}
