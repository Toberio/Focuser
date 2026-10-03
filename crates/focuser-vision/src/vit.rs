//! A Vision Transformer, and CLIP's text transformer, written out in Burn.
//!
//! Both models the filter uses are ViTs: Marqo's NSFW classifier (timm's
//! `vit_tiny_patch16_384`) and CLIP's image tower (`ViT-B/32`). They differ
//! only in configuration: CLIP normalises before the blocks and projects after,
//! and uses QuickGELU. One implementation serves both.
//!
//! Checked against independent implementations: CLIP's image and text towers
//! agree with ONNX Runtime running the Hugging Face export to cosine 1.00000
//! on identical inputs; Marqo agrees with a TF.js port to ±0.01.

use burn::tensor::backend::Backend;
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Tensor, TensorData, activation};

use crate::Result;
use crate::weights::Weights;

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
    /// OpenAI CLIP's `x · sigmoid(1.702 x)`.
    QuickGelu,
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

/// Run `blocks` over `[batch, tokens, dim]`. `mask` is added to the attention
/// scores before softmax: CLIP's text tower is causal, images are not.
fn transformer<B: Backend>(
    blocks: &[Block<B>],
    mut x: Tensor<B, 3>,
    heads: usize,
    mask: Option<Tensor<B, 2>>,
) -> Tensor<B, 3> {
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
        let mut scores = q.matmul(k.swap_dims(2, 3)) * scale;
        if let Some(mask) = &mask {
            scores = scores + mask.clone().unsqueeze::<4>();
        }
        let mixed = activation::softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, tokens, dim]);
        x = x + block.out.forward(mixed);

        let h = block.fc1.forward(block.norm2.forward(x.clone()));
        let h = match block.act {
            Activation::Gelu => activation::gelu(h),
            Activation::QuickGelu => h.clone() * activation::sigmoid(h * 1.702),
        };
        x = x + block.fc2.forward(h);
    }
    x
}

/// An image transformer: patches in, one vector per image out.
pub(crate) struct Vit<B: Backend> {
    patch: usize,
    heads: usize,
    patch_w: Tensor<B, 4>,
    patch_b: Option<Tensor<B, 1>>,
    cls: Tensor<B, 1>,
    pos: Tensor<B, 2>,
    pre_norm: Option<LayerNorm<B>>,
    blocks: Vec<Block<B>>,
    final_norm: LayerNorm<B>,
    head: Linear<B>,
}

impl<B: Backend> Vit<B> {
    /// `images` is `[batch, 3, size, size]`, already normalised.
    /// Returns `[batch, outputs]`: logits for Marqo, an embedding for CLIP.
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
        let cls = self
            .cls
            .clone()
            .reshape([1, 1, dim])
            .expand([batch, 1, dim]);
        let mut x = Tensor::cat(vec![cls, patches], 1) + self.pos.clone().unsqueeze::<3>();
        if let Some(norm) = &self.pre_norm {
            x = norm.forward(x);
        }
        let x = transformer(&self.blocks, x, self.heads, None);
        // LayerNorm works per token, so normalising only the class token is
        // the same as timm's "normalise all, then pool".
        let pooled = self.final_norm.forward(x.slice([0..batch, 0..1, 0..dim]));
        let out = self.head.forward(pooled);
        let [_, _, outputs] = out.dims();
        out.reshape([batch, outputs])
    }
}

/// CLIP's text tower. Used only to embed the filter's prompts ahead of time
/// (see `examples/embed_prompts.rs`); the app never runs it.
pub struct TextTower<B: Backend> {
    token: Tensor<B, 2>,
    pos: Tensor<B, 2>,
    blocks: Vec<Block<B>>,
    final_norm: LayerNorm<B>,
    proj: Linear<B>,
}

impl<B: Backend> TextTower<B> {
    /// One tokenised prompt, start and end tokens included, to a unit vector.
    /// Pooled at the end token, which has the highest id in CLIP's vocabulary.
    pub fn embed(&self, ids: &[u32], device: &Device<B>) -> Vec<f32> {
        let n = ids.len();
        let dim = self.token.dims()[1];
        let rows: Vec<Tensor<B, 2>> = ids
            .iter()
            .map(|&id| {
                self.token
                    .clone()
                    .slice([id as usize..id as usize + 1, 0..dim])
            })
            .collect();
        let x = Tensor::cat(rows, 0) + self.pos.clone().slice([0..n, 0..dim]);
        let mut mask = vec![0f32; n * n];
        for row in 0..n {
            for col in (row + 1)..n {
                mask[row * n + col] = -1e9;
            }
        }
        let mask =
            Tensor::<B, 1>::from_data(TensorData::new(mask, [n * n]), device).reshape([n, n]);
        let x = transformer(&self.blocks, x.unsqueeze::<3>(), 8, Some(mask));
        let end = ids
            .iter()
            .enumerate()
            .max_by_key(|&(_, id)| *id)
            .map_or(n - 1, |(i, _)| i);
        let pooled = self
            .final_norm
            .forward(x.slice([0..1, end..end + 1, 0..dim]));
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

/// A fused `[3·dim, dim]` projection split into q, k and v.
fn split_qkv<B: Backend>(
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
    dim: usize,
) -> (Linear<B>, Linear<B>, Linear<B>) {
    let part = |j: usize| Linear {
        w: weight
            .clone()
            .slice([j * dim..(j + 1) * dim, 0..dim])
            .transpose(),
        b: Some(bias.clone().slice(j * dim..(j + 1) * dim)),
    };
    (part(0), part(1), part(2))
}

/// Marqo/nsfw-image-detection-384: timm `vit_tiny_patch16_384`, 2 classes
/// (NSFW, SFW).
pub(crate) fn marqo<B: Backend>(w: &Weights, device: &Device<B>) -> Result<Vit<B>> {
    const DIM: usize = 192;
    const EPS: f64 = 1e-6;
    let mut blocks = Vec::with_capacity(12);
    for i in 0..12 {
        let p = format!("blocks.{i}");
        let (q, k, v) = split_qkv(
            w.tensor(&format!("{p}.attn.qkv.weight"), device)?,
            w.tensor(&format!("{p}.attn.qkv.bias"), device)?,
            DIM,
        );
        blocks.push(Block {
            act: Activation::Gelu,
            norm1: layer_norm(w, &format!("{p}.norm1"), EPS, device)?,
            q,
            k,
            v,
            out: linear(w, &format!("{p}.attn.proj"), device)?,
            norm2: layer_norm(w, &format!("{p}.norm2"), EPS, device)?,
            fc1: linear(w, &format!("{p}.mlp.fc1"), device)?,
            fc2: linear(w, &format!("{p}.mlp.fc2"), device)?,
        });
    }
    let cls: Tensor<B, 3> = w.tensor("cls_token", device)?;
    let pos: Tensor<B, 3> = w.tensor("pos_embed", device)?;
    let tokens = pos.dims()[1];
    Ok(Vit {
        patch: 16,
        heads: 3,
        patch_w: w.tensor("patch_embed.proj.weight", device)?,
        patch_b: Some(w.tensor("patch_embed.proj.bias", device)?),
        cls: cls.reshape([DIM]),
        pos: pos.reshape([tokens, DIM]),
        pre_norm: None,
        blocks,
        final_norm: layer_norm(w, "norm", EPS, device)?,
        head: linear(w, "head", device)?,
    })
}

/// An open_clip residual block, as OpenAI's CLIP stores them.
fn clip_block<B: Backend>(
    w: &Weights,
    p: &str,
    dim: usize,
    device: &Device<B>,
) -> Result<Block<B>> {
    const EPS: f64 = 1e-5;
    let (q, k, v) = split_qkv(
        w.tensor(&format!("{p}.attn.in_proj_weight"), device)?,
        w.tensor(&format!("{p}.attn.in_proj_bias"), device)?,
        dim,
    );
    Ok(Block {
        act: Activation::QuickGelu,
        norm1: layer_norm(w, &format!("{p}.ln_1"), EPS, device)?,
        q,
        k,
        v,
        out: linear(w, &format!("{p}.attn.out_proj"), device)?,
        norm2: layer_norm(w, &format!("{p}.ln_2"), EPS, device)?,
        fc1: linear(w, &format!("{p}.mlp.c_fc"), device)?,
        fc2: linear(w, &format!("{p}.mlp.c_proj"), device)?,
    })
}

/// OpenAI CLIP ViT-B/32's image tower, in open_clip naming.
pub(crate) fn clip_image<B: Backend>(w: &Weights, device: &Device<B>) -> Result<Vit<B>> {
    let blocks = (0..12)
        .map(|i| clip_block(w, &format!("visual.transformer.resblocks.{i}"), 768, device))
        .collect::<Result<Vec<_>>>()?;
    Ok(Vit {
        patch: 32,
        heads: 12,
        patch_w: w.tensor("visual.conv1.weight", device)?,
        patch_b: None,
        cls: w.tensor("visual.class_embedding", device)?,
        pos: w.tensor("visual.positional_embedding", device)?,
        pre_norm: Some(layer_norm(w, "visual.ln_pre", 1e-5, device)?),
        blocks,
        final_norm: layer_norm(w, "visual.ln_post", 1e-5, device)?,
        // open_clip computes `x @ proj`, so it is already `[in, out]`.
        head: Linear {
            w: w.tensor("visual.proj", device)?,
            b: None,
        },
    })
}

/// OpenAI CLIP ViT-B/32's text tower, in open_clip naming.
pub fn clip_text<B: Backend>(w: &Weights, device: &Device<B>) -> Result<TextTower<B>> {
    let blocks = (0..12)
        .map(|i| clip_block(w, &format!("transformer.resblocks.{i}"), 512, device))
        .collect::<Result<Vec<_>>>()?;
    Ok(TextTower {
        token: w.tensor("token_embedding.weight", device)?,
        pos: w.tensor("positional_embedding", device)?,
        blocks,
        final_norm: layer_norm(w, "ln_final", 1e-5, device)?,
        proj: Linear {
            w: w.tensor("text_projection", device)?,
            b: None,
        },
    })
}
