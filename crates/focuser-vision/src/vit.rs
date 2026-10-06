//! A Vision Transformer, and SigLIP's text transformer, written out in Burn.
//!
//! Both models the filter uses are ViTs: Marqo's NSFW classifier (timm's
//! `vit_tiny_patch16_384`) and SigLIP 2's image tower (`ViT-B/16`). They differ
//! only in how they sum an image up: Marqo reads a class token, SigLIP pools
//! every patch with a small attention head. One implementation serves both.
//!
//! Checked against independent implementations: SigLIP 2's image and text
//! towers agree with open_clip running the same weights; Marqo agrees with a
//! TF.js port to ±0.01.

use burn::tensor::backend::Backend;
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Tensor, activation};

use crate::weights::Weights;
use crate::{Result, VisionError};

/// `x @ w + b`, with `w` stored as `[in, out]`.
pub(crate) struct Linear<B: Backend> {
    w: Tensor<B, 2>,
    b: Option<Tensor<B, 1>>,
}

impl<B: Backend> Linear<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, tokens, dim] = x.dims();
        let out = self.w.dims()[1];
        let y = x.reshape([batch * tokens, dim]).matmul(self.w.clone());
        let y = match &self.b {
            Some(bias) => y + bias.clone().unsqueeze_dim(0),
            None => y,
        };
        y.reshape([batch, tokens, out])
    }
}

pub(crate) struct LayerNorm<B: Backend> {
    gamma: Tensor<B, 1>,
    beta: Tensor<B, 1>,
    eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = x.clone().mean_dim(2);
        let centred = x - mean;
        let variance = centred.clone().powi_scalar(2).mean_dim(2);
        let normed = centred / (variance + self.eps).sqrt();
        normed * self.gamma.clone().unsqueeze::<3>() + self.beta.clone().unsqueeze::<3>()
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Activation {
    /// PyTorch's default: exact, with erf.
    Gelu,
    /// The tanh approximation, which SigLIP's text tower was trained with.
    GeluTanh,
}

impl Activation {
    fn apply<B: Backend, const D: usize>(self, x: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Self::Gelu => activation::gelu(x),
            Self::GeluTanh => {
                let inner = (x.clone() + x.clone().powi_scalar(3) * 0.044_715)
                    * (2.0 / std::f64::consts::PI).sqrt();
                x * (inner.tanh() + 1.0) * 0.5
            }
        }
    }
}

/// A pre-norm transformer block: attention, then an MLP, each residual.
pub(crate) struct Block<B: Backend> {
    act: Activation,
    norm1: LayerNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    norm2: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

/// Run `blocks` over `[batch, tokens, dim]`. Every token sees every other:
/// neither model here masks attention.
fn transformer<B: Backend>(blocks: &[Block<B>], mut x: Tensor<B, 3>, heads: usize) -> Tensor<B, 3> {
    let [batch, tokens, dim] = x.dims();
    let head_dim = dim / heads;
    let scale = 1.0 / (head_dim as f64).sqrt();
    for block in blocks {
        let y = block.norm1.forward(x.clone());
        // [batch, tokens, dim] → [batch, heads, tokens, head_dim]
        let split = |l: &Linear<B>| {
            l.forward(y.clone())
                .reshape([batch, tokens, heads, head_dim])
                .swap_dims(1, 2)
        };
        let (q, k, v) = (split(&block.q), split(&block.k), split(&block.v));
        let scores = q.matmul(k.swap_dims(2, 3)) * scale;
        let mixed = activation::softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, tokens, dim]);
        x = x + block.out.forward(mixed);

        let h = block.fc1.forward(block.norm2.forward(x.clone()));
        x = x + block.fc2.forward(block.act.apply(h));
    }
    x
}

/// How an image transformer sums its tokens up into one vector.
enum Pool<B: Backend> {
    /// Read a class token, prepended to the patches, through a linear head.
    Class { cls: Tensor<B, 1>, head: Linear<B> },
    /// SigLIP's: one learned query attends over every patch, then an MLP.
    Attention(AttentionPool<B>),
}

/// timm's `AttentionPoolLatent`, with one query and no head after it.
struct AttentionPool<B: Backend> {
    latent: Tensor<B, 1>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    norm: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> AttentionPool<B> {
    fn forward(&self, x: Tensor<B, 3>, heads: usize) -> Tensor<B, 2> {
        let [batch, tokens, dim] = x.dims();
        let head_dim = dim / heads;
        let latent = self
            .latent
            .clone()
            .reshape([1, 1, dim])
            .expand([batch, 1, dim]);
        let q = self
            .q
            .forward(latent)
            .reshape([batch, 1, heads, head_dim])
            .swap_dims(1, 2);
        let split = |l: &Linear<B>| {
            l.forward(x.clone())
                .reshape([batch, tokens, heads, head_dim])
                .swap_dims(1, 2)
        };
        let (k, v) = (split(&self.k), split(&self.v));
        let scores = q.matmul(k.swap_dims(2, 3)) * (1.0 / (head_dim as f64).sqrt());
        let mixed = activation::softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, 1, dim]);
        let y = self.out.forward(mixed);
        let h = activation::gelu(self.fc1.forward(self.norm.forward(y.clone())));
        (y + self.fc2.forward(h)).reshape([batch, dim])
    }
}

/// An image transformer: patches in, one vector per image out.
pub(crate) struct Vit<B: Backend> {
    patch: usize,
    heads: usize,
    patch_w: Tensor<B, 4>,
    patch_b: Option<Tensor<B, 1>>,
    pos: Tensor<B, 2>,
    blocks: Vec<Block<B>>,
    final_norm: LayerNorm<B>,
    pool: Pool<B>,
}

impl<B: Backend> Vit<B> {
    /// `images` is `[batch, 3, size, size]`, already normalised.
    /// Returns `[batch, outputs]`: logits for Marqo, an embedding for SigLIP.
    pub(crate) fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch, ..] = images.dims();
        let patches = conv2d(
            images,
            self.patch_w.clone(),
            self.patch_b.clone(),
            ConvOptions::new([self.patch, self.patch], [0, 0], [1, 1], 1),
        );
        let [_, dim, h, w] = patches.dims();
        let patches = patches.reshape([batch, dim, h * w]).swap_dims(1, 2);
        let x = match &self.pool {
            Pool::Class { cls, .. } => {
                let cls = cls.clone().reshape([1, 1, dim]).expand([batch, 1, dim]);
                Tensor::cat(vec![cls, patches], 1)
            }
            Pool::Attention(_) => patches,
        } + self.pos.clone().unsqueeze::<3>();
        let x = transformer(&self.blocks, x, self.heads);
        match &self.pool {
            Pool::Class { head, .. } => {
                // LayerNorm works per token, so normalising only the class
                // token is the same as timm's "normalise all, then pool".
                let pooled = self.final_norm.forward(x.slice([0..batch, 0..1, 0..dim]));
                let out = head.forward(pooled);
                let [_, _, outputs] = out.dims();
                out.reshape([batch, outputs])
            }
            Pool::Attention(pool) => pool.forward(self.final_norm.forward(x), self.heads),
        }
    }
}

/// SigLIP's text tower. Used only to embed the filter's prompts ahead of time
/// (see `examples/embed_prompts.rs`); the app never runs it.
pub struct TextTower<B: Backend> {
    token: Tensor<B, 2>,
    pos: Tensor<B, 2>,
    blocks: Vec<Block<B>>,
    final_norm: LayerNorm<B>,
    proj: Linear<B>,
    heads: usize,
}

impl<B: Backend> TextTower<B> {
    /// Tokens SigLIP always reads: a prompt is padded out to this many.
    pub fn context(&self) -> usize {
        self.pos.dims()[0]
    }

    /// One tokenised prompt, end token included, to a unit vector. SigLIP
    /// pads to its full context with id 0, masks nothing, and pools at the
    /// last position, so the padding is part of the computation.
    pub fn embed(&self, ids: &[u32]) -> Vec<f32> {
        let n = self.context();
        let dim = self.token.dims()[1];
        let rows: Vec<Tensor<B, 2>> = (0..n)
            .map(|i| {
                let id = ids.get(i).copied().unwrap_or(0) as usize;
                self.token.clone().slice([id..id + 1, 0..dim])
            })
            .collect();
        let x = Tensor::cat(rows, 0) + self.pos.clone();
        let x = transformer(&self.blocks, x.unsqueeze::<3>(), self.heads);
        let pooled = self.final_norm.forward(x.slice([0..1, n - 1..n, 0..dim]));
        let e = self.proj.forward(pooled).reshape([1, dim]);
        let e = e.clone() / e.powi_scalar(2).sum_dim(1).sqrt();
        e.into_data().to_vec::<f32>().unwrap_or_default()
    }
}

// ─── Loading ────────────────────────────────────────────────────────

fn linear<B: Backend>(w: &Weights, prefix: &str, device: &Device<B>) -> Result<Linear<B>> {
    Ok(Linear {
        w: w.tensor::<B, 2>(&format!("{prefix}.weight"), device)?
            .transpose(),
        b: Some(w.tensor(&format!("{prefix}.bias"), device)?),
    })
}

fn layer_norm<B: Backend>(
    w: &Weights,
    prefix: &str,
    eps: f64,
    device: &Device<B>,
) -> Result<LayerNorm<B>> {
    Ok(LayerNorm {
        gamma: w.tensor(&format!("{prefix}.weight"), device)?,
        beta: w.tensor(&format!("{prefix}.bias"), device)?,
        eps,
    })
}

/// A fused `[n·dim, dim]` projection, such as q, k and v, split into its parts.
fn split<B: Backend>(
    w: &Weights,
    weight: &str,
    bias: &str,
    dim: usize,
    device: &Device<B>,
) -> Result<Vec<Linear<B>>> {
    let weight: Tensor<B, 2> = w.tensor(weight, device)?;
    let bias: Tensor<B, 1> = w.tensor(bias, device)?;
    let parts = weight.dims()[0] / dim;
    Ok((0..parts)
        .map(|j| Linear {
            w: weight
                .clone()
                .slice([j * dim..(j + 1) * dim, 0..dim])
                .transpose(),
            b: Some(bias.clone().slice(j * dim..(j + 1) * dim)),
        })
        .collect())
}

/// A timm ViT block, as Marqo and SigLIP's image tower store them.
fn timm_block<B: Backend>(
    w: &Weights,
    p: &str,
    dim: usize,
    device: &Device<B>,
) -> Result<Block<B>> {
    const EPS: f64 = 1e-6;
    let [q, k, v]: [Linear<B>; 3] = split(
        w,
        &format!("{p}.attn.qkv.weight"),
        &format!("{p}.attn.qkv.bias"),
        dim,
        device,
    )?
    .try_into()
    .map_err(|_| VisionError::Model(format!("{p}: qkv is not three projections")))?;
    Ok(Block {
        act: Activation::Gelu,
        norm1: layer_norm(w, &format!("{p}.norm1"), EPS, device)?,
        q,
        k,
        v,
        out: linear(w, &format!("{p}.attn.proj"), device)?,
        norm2: layer_norm(w, &format!("{p}.norm2"), EPS, device)?,
        fc1: linear(w, &format!("{p}.mlp.fc1"), device)?,
        fc2: linear(w, &format!("{p}.mlp.fc2"), device)?,
    })
}

/// Marqo/nsfw-image-detection-384: timm `vit_tiny_patch16_384`, 2 classes
/// (NSFW, SFW).
pub(crate) fn marqo<B: Backend>(w: &Weights, device: &Device<B>) -> Result<Vit<B>> {
    const DIM: usize = 192;
    let blocks = (0..12)
        .map(|i| timm_block(w, &format!("blocks.{i}"), DIM, device))
        .collect::<Result<Vec<_>>>()?;
    let cls: Tensor<B, 3> = w.tensor("cls_token", device)?;
    let pos: Tensor<B, 3> = w.tensor("pos_embed", device)?;
    let tokens = pos.dims()[1];
    Ok(Vit {
        patch: 16,
        heads: 3,
        patch_w: w.tensor("patch_embed.proj.weight", device)?,
        patch_b: Some(w.tensor("patch_embed.proj.bias", device)?),
        pos: pos.reshape([tokens, DIM]),
        blocks,
        final_norm: layer_norm(w, "norm", 1e-6, device)?,
        pool: Pool::Class {
            cls: cls.reshape([DIM]),
            head: linear(w, "head", device)?,
        },
    })
}

/// SigLIP 2 ViT-B/16's image tower, in open_clip naming: timm's
/// `vit_base_patch16_siglip_224` under `visual.trunk`.
pub(crate) fn siglip_image<B: Backend>(w: &Weights, device: &Device<B>) -> Result<Vit<B>> {
    const DIM: usize = 768;
    const EPS: f64 = 1e-6;
    let p = "visual.trunk";
    let blocks = (0..12)
        .map(|i| timm_block(w, &format!("{p}.blocks.{i}"), DIM, device))
        .collect::<Result<Vec<_>>>()?;
    let pos: Tensor<B, 3> = w.tensor(&format!("{p}.pos_embed"), device)?;
    let tokens = pos.dims()[1];
    let latent: Tensor<B, 3> = w.tensor(&format!("{p}.attn_pool.latent"), device)?;
    let [k, v]: [Linear<B>; 2] = split(
        w,
        &format!("{p}.attn_pool.kv.weight"),
        &format!("{p}.attn_pool.kv.bias"),
        DIM,
        device,
    )?
    .try_into()
    .map_err(|_| VisionError::Model("attn_pool.kv is not two projections".into()))?;
    Ok(Vit {
        patch: 16,
        heads: 12,
        patch_w: w.tensor(&format!("{p}.patch_embed.proj.weight"), device)?,
        patch_b: Some(w.tensor(&format!("{p}.patch_embed.proj.bias"), device)?),
        pos: pos.reshape([tokens, DIM]),
        blocks,
        final_norm: layer_norm(w, &format!("{p}.norm"), EPS, device)?,
        pool: Pool::Attention(AttentionPool {
            latent: latent.reshape([DIM]),
            q: linear(w, &format!("{p}.attn_pool.q"), device)?,
            k,
            v,
            out: linear(w, &format!("{p}.attn_pool.proj"), device)?,
            norm: layer_norm(w, &format!("{p}.attn_pool.norm"), EPS, device)?,
            fc1: linear(w, &format!("{p}.attn_pool.mlp.fc1"), device)?,
            fc2: linear(w, &format!("{p}.attn_pool.mlp.fc2"), device)?,
        }),
    })
}

/// SigLIP 2 ViT-B/16's text tower, in open_clip naming.
pub fn siglip_text<B: Backend>(w: &Weights, device: &Device<B>) -> Result<TextTower<B>> {
    const DIM: usize = 768;
    const EPS: f64 = 1e-6;
    let blocks = (0..12)
        .map(|i| {
            let p = format!("text.transformer.resblocks.{i}");
            let [q, k, v]: [Linear<B>; 3] = split(
                w,
                &format!("{p}.attn.in_proj_weight"),
                &format!("{p}.attn.in_proj_bias"),
                DIM,
                device,
            )?
            .try_into()
            .map_err(|_| VisionError::Model(format!("{p}: in_proj is not three projections")))?;
            Ok(Block {
                act: Activation::GeluTanh,
                norm1: layer_norm(w, &format!("{p}.ln_1"), EPS, device)?,
                q,
                k,
                v,
                out: linear(w, &format!("{p}.attn.out_proj"), device)?,
                norm2: layer_norm(w, &format!("{p}.ln_2"), EPS, device)?,
                fc1: linear(w, &format!("{p}.mlp.c_fc"), device)?,
                fc2: linear(w, &format!("{p}.mlp.c_proj"), device)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TextTower {
        token: w.tensor("text.token_embedding.weight", device)?,
        pos: w.tensor("text.positional_embedding", device)?,
        blocks,
        final_norm: layer_norm(w, "text.ln_final", EPS, device)?,
        proj: linear(w, "text.text_projection", device)?,
        heads: 12,
    })
}
