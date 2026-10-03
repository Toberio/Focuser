import * as tf from "@tensorflow/tfjs";
import { load, type NSFWJS } from "nsfwjs/core";
import { MobileNetV2Model } from "nsfwjs/models/mobilenet_v2";
import {
  isExplicit,
  MAX_CACHE_KEY,
  MODEL_SIZE,
  type Verdict,
  VerdictCache,
} from "@/lib/image-filter";
import type { Message, MessageReply } from "@/lib/messages";

/**
 * Judges images for the explicit-image filter.
 *
 * A page rather than the background, because the model wants WebGL and a
 * Chrome service worker has neither a DOM nor a lifetime long enough to keep a
 * model warm. Chrome hosts this as an offscreen document; Firefox, whose
 * background is a page already, as a frame inside it.
 *
 * Only the 3.5 MB MobileNetV2 model is bundled. NSFWJS ships two bigger ones
 * as well, which its default `load` would pull into the build.
 */

/** Downloads in flight at once. Inference itself runs one at a time. */
const FETCH_CONCURRENCY = 6;

let model: Promise<NSFWJS> | null = null;
const cache = new VerdictCache();
const inFlight = new Map<string, Promise<Verdict>>();

let fetching = 0;
const waiting: Array<() => void> = [];
/** Inference is serialised: parallel WebGL work only competes for the GPU. */
let inferenceChain: Promise<unknown> = Promise.resolve();

function getModel(): Promise<NSFWJS> {
  model ??= (async () => {
    await tf.ready();
    return load("MobileNetV2", { modelDefinitions: [MobileNetV2Model] });
  })().catch((error) => {
    // Let the next request try again rather than failing for ever.
    model = null;
    throw error;
  });
  return model;
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

/** Download and decode straight to the model's input size. */
async function decode(src: string): Promise<ImageBitmap> {
  return withFetchSlot(async () => {
    // `force-cache` reuses this profile's copy where there is one. The page's
    // own cache is partitioned away from extensions, so a cross-origin image
    // is usually downloaded a second time; that is the cost of not needing
    // CORS from every image host on the web.
    const response = await fetch(src, { cache: "force-cache", credentials: "include" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const blob = await response.blob();
    return createImageBitmap(blob, {
      resizeWidth: MODEL_SIZE,
      resizeHeight: MODEL_SIZE,
      resizeQuality: "medium",
    });
  });
}

async function judge(src: string): Promise<Verdict> {
  try {
    const loading = getModel();
    // Awaited below; this only keeps a failed download from also reporting
    // the model's failure as unhandled.
    loading.catch(() => undefined);
    const bitmap = await decode(src);
    const run = inferenceChain.then(async () => {
      try {
        const nsfw = await loading;
        const input = tf.browser.fromPixels(bitmap);
        try {
          return await nsfw.classify(input, 5);
        } finally {
          input.dispose();
        }
      } finally {
        bitmap.close();
      }
    });
    inferenceChain = run.catch(() => undefined);
    return isExplicit(await run) ? "hidden" : "clear";
  } catch {
    return "error";
  }
}

function classify(src: string): Promise<Verdict> {
  const cacheable = src.length <= MAX_CACHE_KEY;
  const known = cacheable ? cache.get(src) : undefined;
  if (known) return Promise.resolve(known);

  // The same image is often on a page many times, and in many tabs.
  const pending = inFlight.get(src);
  if (pending) return pending;

  const job = judge(src).then((verdict) => {
    inFlight.delete(src);
    if (cacheable) cache.set(src, verdict);
    return verdict;
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
    void classify(message.src).then((verdict) =>
      sendResponse({ type: "classifier-run", verdict }),
    );
    return true;
  },
);

// Load the model now, so the first image on the first page does not wait for it.
void getModel().catch(() => undefined);
