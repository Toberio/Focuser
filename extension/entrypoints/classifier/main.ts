import * as tf from "@tensorflow/tfjs";
import { load, type NSFWJS } from "nsfwjs/core";
import { MobileNetV2Model } from "nsfwjs/models/mobilenet_v2";
import {
  MAX_CACHE_KEY,
  NSFWJS_SIZE,
  ScoreCache,
  type Scores,
  scoresOf,
} from "@/lib/image-filter";
import { VIT_SIZE, Vit, type VitManifest } from "@/lib/vit";
import type { Message, MessageReply } from "@/lib/messages";

/**
 * Judges images for the explicit-image filter.
 *
 * A page rather than the background, because the model wants WebGL and a
 * Chrome service worker has neither a DOM nor a lifetime long enough to keep a
 * model warm. Chrome hosts this as an offscreen document; Firefox, whose
 * background is a page already, as a frame inside it.
 *
 * Two models judge each image; `Scores` in `lib/image-filter.ts` says why and
 * how they are combined. NSFWJS's 3.5 MB MobileNetV2 is the only one of its
 * models bundled (its default `load` would pull in two more), and Marqo's ViT
 * is 11 MB of float16 weights in `public/models/`.
 */

/** Downloads in flight at once. Inference itself runs one at a time. */
const FETCH_CONCURRENCY = 6;

interface Models {
  nsfwjs: NSFWJS;
  vit: Vit;
}

let models: Promise<Models> | null = null;
const cache = new ScoreCache();
const inFlight = new Map<string, Promise<Scores | undefined>>();

let fetching = 0;
const waiting: Array<() => void> = [];
/** Inference is serialised: parallel WebGL work only competes for the GPU. */
let inferenceChain: Promise<unknown> = Promise.resolve();

async function loadVit(): Promise<Vit> {
  const base = "/models/marqo-nsfw-384/";
  const url = (file: string) => browser.runtime.getURL(`${base}${file}` as never);
  const [manifest, weights] = await Promise.all([
    fetch(url("manifest.json")).then((r) => r.json() as Promise<VitManifest>),
    fetch(url("weights.bin")).then((r) => r.arrayBuffer()),
  ]);
  return Vit.fromBuffers(manifest, weights);
}

function getModels(): Promise<Models> {
  models ??= (async () => {
    await tf.ready();
    const [nsfwjs, vit] = await Promise.all([
      load("MobileNetV2", { modelDefinitions: [MobileNetV2Model] }),
      loadVit(),
    ]);
    return { nsfwjs, vit };
  })().catch((error) => {
    // Let the next request try again rather than failing for ever.
    models = null;
    throw error;
  });
  return models;
}

async function withFetchSlot<T>(task: () => Promise<T>): Promise<T> {
  if (fetching >= FETCH_CONCURRENCY) await new Promise<void>((r) => waiting.push(r));
  fetching++;
  try {
    return await task();
  } finally {
    fetching--;
    waiting.shift()?.();
  }
}

/** One image, decoded once and shaped the way each model was trained on. */
interface Inputs {
  /** Short side to 384, centre-cropped: timm's eval transform for the ViT. */
  vit: ImageBitmap;
  /** Squashed to 224, as NSFWJS resizes. */
  nsfwjs: ImageBitmap;
}

async function decode(src: string): Promise<Inputs> {
  return withFetchSlot(async () => {
    // `force-cache` reuses this profile's copy where there is one. The page's
    // own cache is partitioned away from extensions, so a cross-origin image
    // is usually downloaded a second time; that is the cost of not needing
    // CORS from every image host on the web.
    const response = await fetch(src, { cache: "force-cache", credentials: "include" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const full = await createImageBitmap(await response.blob());
    try {
      const side = Math.min(full.width, full.height);
      const [vit, nsfwjs] = await Promise.all([
        createImageBitmap(
          full,
          Math.floor((full.width - side) / 2),
          Math.floor((full.height - side) / 2),
          side,
          side,
          { resizeWidth: VIT_SIZE, resizeHeight: VIT_SIZE, resizeQuality: "high" },
        ),
        createImageBitmap(full, {
          resizeWidth: NSFWJS_SIZE,
          resizeHeight: NSFWJS_SIZE,
          resizeQuality: "medium",
        }),
      ]);
      return { vit, nsfwjs };
    } finally {
      full.close();
    }
  });
}

async function judge(src: string): Promise<Scores | undefined> {
  try {
    const loading = getModels();
    // Awaited below; this only keeps a failed download from also reporting
    // the model's failure as unhandled.
    loading.catch(() => undefined);
    const inputs = await decode(src);
    const run = inferenceChain.then(async (): Promise<Scores> => {
      try {
        const { nsfwjs, vit } = await loading;
        const small = tf.browser.fromPixels(inputs.nsfwjs);
        const large = tf.browser.fromPixels(inputs.vit);
        try {
          const predictions = await nsfwjs.classify(small, 5);
          return {
            nsfw: await vit.nsfwProbability(large),
            ...scoresOf(predictions),
            backend: tf.getBackend(),
          };
        } finally {
          small.dispose();
          large.dispose();
        }
      } finally {
        inputs.vit.close();
        inputs.nsfwjs.close();
      }
    });
    inferenceChain = run.catch(() => undefined);
    return await run;
  } catch {
    return undefined;
  }
}

function classify(src: string): Promise<Scores | undefined> {
  const cacheable = src.length <= MAX_CACHE_KEY;
  const known = cacheable ? cache.get(src) : undefined;
  if (known) return Promise.resolve(known);

  // The same image is often on a page many times, and in many tabs.
  const pending = inFlight.get(src);
  if (pending) return pending;

  const job = judge(src).then((scores) => {
    inFlight.delete(src);
    // A failure may be a network blip, so it is not remembered.
    if (cacheable && scores) cache.set(src, scores);
    return scores;
  });
  inFlight.set(src, job);
  return job;
}

browser.runtime.onMessage.addListener(
  (raw: unknown, _sender, sendResponse: (reply: MessageReply) => void) => {
    const message = raw as Message;
    // Content scripts' `classify-image` reaches this page too, since a
    // runtime message goes to every extension page. The background answers
    // those; this page only answers the background.
    if (message.type !== "classifier-run") return false;
    void classify(message.src).then((scores) => sendResponse({ type: "classifier-run", scores }));
    return true;
  },
);

// Load the models now, so the first image on the first page does not wait for them.
void getModels().catch(() => undefined);
