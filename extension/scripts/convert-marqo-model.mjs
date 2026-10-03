/**
 * Converts Marqo/nsfw-image-detection-384 into the files the extension loads.
 *
 *   node scripts/convert-marqo-model.mjs [path/to/model.safetensors]
 *
 * Without a path it downloads the official weights from Hugging Face and
 * checks them against SOURCE_SHA256, so the committed output can be rebuilt
 * byte for byte by anyone, store reviewers included.
 *
 * Output, in `public/models/marqo-nsfw-384/`:
 * - `weights.bin`: every tensor as little-endian float16, back to back. Half
 *   the size of the float32 original, and the model's outputs move in the
 *   third decimal place at most.
 * - `manifest.json`: name, shape and offset of each tensor.
 *
 * Model: https://huggingface.co/Marqo/nsfw-image-detection-384 (Apache-2.0),
 * a fine-tune of timm's vit_tiny_patch16_384.
 */

import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const SOURCE_URL =
  "https://huggingface.co/Marqo/nsfw-image-detection-384/resolve/main/model.safetensors";
const SOURCE_SHA256 = "6bf2e0f64a1d20169736c2836e3a787b12379fdc08ba87f7d94a7a3d58eeefce";
const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), "../public/models/marqo-nsfw-384");

async function source() {
  const path = process.argv[2];
  const bytes = path
    ? readFileSync(path)
    : Buffer.from(await (await fetch(SOURCE_URL)).arrayBuffer());
  const digest = createHash("sha256").update(bytes).digest("hex");
  if (digest !== SOURCE_SHA256) {
    throw new Error(`unexpected weights: sha256 ${digest}, wanted ${SOURCE_SHA256}`);
  }
  return bytes;
}

/** Round-to-nearest-even float32 → float16. */
function toHalf(value) {
  const f32 = new Float32Array([value]);
  const bits = new Uint32Array(f32.buffer)[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    // Subnormal or zero in half precision.
    if (exponent < -10) return sign;
    mantissa |= 0x800000;
    const shift = 14 - exponent;
    const half = mantissa >>> shift;
    const rest = mantissa & ((1 << shift) - 1);
    const midpoint = 1 << (shift - 1);
    return sign | (rest > midpoint || (rest === midpoint && half & 1) ? half + 1 : half);
  }
  if (exponent >= 31) return sign | 0x7c00;
  const half = (exponent << 10) | (mantissa >>> 13);
  const rest = mantissa & 0x1fff;
  return sign | (rest > 0x1000 || (rest === 0x1000 && half & 1) ? half + 1 : half);
}

const bytes = await source();
const headerLength = Number(bytes.readBigUInt64LE(0));
const header = JSON.parse(bytes.subarray(8, 8 + headerLength).toString("utf8"));
const dataStart = 8 + headerLength;

const names = Object.keys(header)
  .filter((name) => name !== "__metadata__")
  .sort();
const tensors = [];
const chunks = [];
let offset = 0;
for (const name of names) {
  const { dtype, shape, data_offsets: [from, to] } = header[name];
  if (dtype !== "F32") throw new Error(`${name}: expected F32, got ${dtype}`);
  const floats = new Float32Array(
    bytes.buffer.slice(bytes.byteOffset + dataStart + from, bytes.byteOffset + dataStart + to),
  );
  const halves = new Uint16Array(floats.length);
  for (let i = 0; i < floats.length; i++) halves[i] = toHalf(floats[i]);
  chunks.push(Buffer.from(halves.buffer));
  tensors.push({ name, shape, offset, length: floats.length });
  offset += halves.byteLength;
}

mkdirSync(OUT_DIR, { recursive: true });
writeFileSync(join(OUT_DIR, "weights.bin"), Buffer.concat(chunks));
writeFileSync(
  join(OUT_DIR, "manifest.json"),
  `${JSON.stringify({ source: SOURCE_URL, sha256: SOURCE_SHA256, dtype: "float16", tensors }, null, 1)}\n`,
);
console.log(`wrote ${tensors.length} tensors, ${offset} bytes, to ${OUT_DIR}`);
