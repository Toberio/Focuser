import * as tf from "@tensorflow/tfjs";

/**
 * Marqo/nsfw-image-detection-384, a timm `vit_tiny_patch16_384`, written out
 * in TF.js ops.
 *
 * Written out rather than converted because the usual route (PyTorch → ONNX →
 * TensorFlow → TF.js) needs a stack of Python tools to reproduce, and the other
 * browser runtime, ONNX Runtime Web, would add 10–20 MB of WebAssembly on top
 * of the TF.js already here for NSFWJS. A ViT is small enough to write down:
 * patch embedding, twelve pre-norm transformer blocks, a classifier on the
 * class token. It mirrors timm's `VisionTransformer` line for line.
 *
 * Weights come from `scripts/convert-marqo-model.mjs`.
 */

export const VIT_SIZE = 384;
const PATCH = 16;
const DIM = 192;
const HEADS = 3;
const HEAD_DIM = DIM / HEADS;
const DEPTH = 12;
/** timm's ViTs build their LayerNorms with this epsilon, not PyTorch's 1e-5. */
const LN_EPS = 1e-6;

interface TensorEntry {
  name: string;
  shape: number[];
  offset: number;
  length: number;
}

export interface VitManifest {
  dtype: "float16";
  tensors: TensorEntry[];
}

/** float16 bits → float32, for the weights file. */
export function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >>> 10) & 0x1f;
  const mantissa = h & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
  if (exponent === 31) return mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

type Weights = Map<string, tf.Tensor>;

/** A linear layer's weight, transposed once to `[in, out]` for `matMul`. */
interface Linear {
  w: tf.Tensor2D;
  b: tf.Tensor1D;
}

interface Block {
  norm1: [tf.Tensor1D, tf.Tensor1D];
  qkv: Linear;
  proj: Linear;
  norm2: [tf.Tensor1D, tf.Tensor1D];
  fc1: Linear;
  fc2: Linear;
}

export class Vit {
  private constructor(
    private readonly patchW: tf.Tensor4D,
    private readonly patchB: tf.Tensor1D,
    private readonly cls: tf.Tensor2D,
    private readonly pos: tf.Tensor2D,
    private readonly blocks: Block[],
    private readonly norm: [tf.Tensor1D, tf.Tensor1D],
    private readonly head: Linear,
  ) {}

  static fromBuffers(manifest: VitManifest, weights: ArrayBuffer): Vit {
    const map: Weights = new Map();
    for (const t of manifest.tensors) {
      const halves = new Uint16Array(weights, t.offset, t.length);
      const floats = new Float32Array(t.length);
      for (let i = 0; i < t.length; i++) floats[i] = halfToFloat(halves[i] ?? 0);
      map.set(t.name, tf.tensor(floats, t.shape));
    }
    const get = (name: string) => {
      const t = map.get(name);
      if (!t) throw new Error(`model is missing ${name}`);
      return t;
    };
    const linear = (prefix: string): Linear => ({
      w: tf.transpose(get(`${prefix}.weight`)) as tf.Tensor2D,
      b: get(`${prefix}.bias`) as tf.Tensor1D,
    });
    const norm = (prefix: string): [tf.Tensor1D, tf.Tensor1D] => [
      get(`${prefix}.weight`) as tf.Tensor1D,
      get(`${prefix}.bias`) as tf.Tensor1D,
    ];

    const blocks: Block[] = [];
    for (let i = 0; i < DEPTH; i++) {
      const p = `blocks.${i}`;
      blocks.push({
        norm1: norm(`${p}.norm1`),
        qkv: linear(`${p}.attn.qkv`),
        proj: linear(`${p}.attn.proj`),
        norm2: norm(`${p}.norm2`),
        fc1: linear(`${p}.mlp.fc1`),
        fc2: linear(`${p}.mlp.fc2`),
      });
    }

    return new Vit(
      // PyTorch convolutions are [out, in, h, w]; TF.js wants [h, w, in, out].
      tf.transpose(get("patch_embed.proj.weight"), [2, 3, 1, 0]) as tf.Tensor4D,
      get("patch_embed.proj.bias") as tf.Tensor1D,
      tf.reshape(get("cls_token"), [1, DIM]) as tf.Tensor2D,
      tf.reshape(get("pos_embed"), [-1, DIM]) as tf.Tensor2D,
      blocks,
      norm("norm"),
      linear("head"),
    );
  }

  /**
   * The probability that an image is NSFW.
   *
   * `pixels` is `[384, 384, 3]` RGB in 0–255, already resized the way timm's
   * eval transform does it (bicubic, short side to 384, centre crop).
   */
  nsfwProbability(pixels: tf.Tensor3D): Promise<number> {
    const probabilities = tf.tidy(() => {
      // mean = std = 0.5 per channel: x / 255 - 0.5, over 0.5.
      const x = tf.sub(tf.div(tf.cast(pixels, "float32"), 127.5), 1).expandDims(0) as tf.Tensor4D;
      const patches = tf.add(
        tf.conv2d(x, this.patchW, PATCH, "valid"),
        this.patchB,
      ) as tf.Tensor4D;
      let tokens = tf.concat([this.cls, tf.reshape(patches, [-1, DIM])], 0) as tf.Tensor2D;
      tokens = tf.add(tokens, this.pos);

      for (const block of this.blocks) {
        tokens = tf.add(tokens, attention(layerNorm(tokens, block.norm1), block));
        tokens = tf.add(tokens, mlp(layerNorm(tokens, block.norm2), block));
      }

      const pooled = tf.slice(layerNorm(tokens, this.norm), [0, 0], [1, DIM]);
      // Class 0 is NSFW, class 1 SFW.
      return tf.softmax(dense(pooled, this.head));
    });
    return probabilities.data().then((p) => {
      probabilities.dispose();
      return p[0] ?? 0;
    });
  }
}

function dense(x: tf.Tensor2D, layer: Linear): tf.Tensor2D {
  return tf.add(tf.matMul(x, layer.w), layer.b);
}

function layerNorm(x: tf.Tensor2D, [gamma, beta]: [tf.Tensor1D, tf.Tensor1D]): tf.Tensor2D {
  const { mean, variance } = tf.moments(x, -1, true);
  const normed = tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, LN_EPS)));
  return tf.add(tf.mul(normed, gamma), beta);
}

/** PyTorch's default GELU is the exact one, with erf, not the tanh approximation. */
function gelu(x: tf.Tensor): tf.Tensor {
  return tf.mul(tf.mul(x, 0.5), tf.add(1, tf.erf(tf.div(x, Math.SQRT2))));
}

function mlp(x: tf.Tensor2D, block: Block): tf.Tensor2D {
  return dense(gelu(dense(x, block.fc1)) as tf.Tensor2D, block.fc2);
}

function attention(x: tf.Tensor2D, block: Block): tf.Tensor2D {
  const tokens = x.shape[0];
  // [tokens, 3·dim] → [3 (q, k, v), heads, tokens, head_dim], as timm splits it.
  const qkv = tf.transpose(tf.reshape(dense(x, block.qkv), [tokens, 3, HEADS, HEAD_DIM]), [
    1, 2, 0, 3,
  ]);
  const [q, k, v] = tf.unstack(qkv) as [tf.Tensor3D, tf.Tensor3D, tf.Tensor3D];
  const scores = tf.mul(tf.matMul(q, k, false, true), 1 / Math.sqrt(HEAD_DIM));
  const mixed = tf.matMul(tf.softmax(scores), v);
  // [heads, tokens, head_dim] → [tokens, dim]
  const merged = tf.reshape(tf.transpose(mixed, [1, 0, 2]), [tokens, DIM]) as tf.Tensor2D;
  return dense(merged, block.proj);
}
