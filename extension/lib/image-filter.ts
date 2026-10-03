/**
 * The explicit-image filter. Pure functions only, so every decision can be
 * tested without a browser.
 *
 * How the parts fit together:
 *
 * - The desktop app sets `filter_explicit_images` in the rule set while an
 *   active list asks for it.
 * - The background then registers the `image-filter` content script, which
 *   blurs every image from `document_start` and lifts the blur one image at a
 *   time as verdicts come back.
 * - Verdicts come from `classifier.html`, an extension page running NSFWJS
 *   (MobileNetV2, MIT) on WebGL. Chrome hosts it as an offscreen document,
 *   Firefox as a frame in the background page. Images are decoded and judged
 *   there, on this machine, and nothing about them is sent anywhere.
 */

/** NSFWJS's five classes. */
export type NsfwClass = "Drawing" | "Hentai" | "Neutral" | "Porn" | "Sexy";

export interface Prediction {
  className: NsfwClass | string;
  probability: number;
}

/**
 * What the content script does with an image.
 *
 * `error` is shown, like `clear`. The filter fails open for the same reason the
 * block page does: a filter that leaves every image blurred when the model
 * cannot load would be switched off, and then it catches nothing.
 */
export type Verdict = "clear" | "hidden" | "error";

/**
 * Hide when explicit classes together pass this.
 *
 * Porn and Hentai are summed because the model splits a borderline image
 * between them; either alone can stay under a threshold the pair is well over.
 */
export const EXPLICIT_THRESHOLD = 0.4;

/**
 * Hide when explicit plus suggestive pass this. "Sexy" on its own covers
 * swimwear and gym photos, so it only tips the balance, never decides it.
 */
export const SUGGESTIVE_THRESHOLD = 0.7;

export function isExplicit(predictions: Prediction[]): boolean {
  const p = (name: NsfwClass) =>
    predictions.find((x) => x.className === name)?.probability ?? 0;
  const explicit = p("Porn") + p("Hentai");
  return explicit >= EXPLICIT_THRESHOLD || explicit + p("Sexy") >= SUGGESTIVE_THRESHOLD;
}

/**
 * Below this on either side an image is an icon, an avatar or a spacer. The
 * model sees too little to judge it, and there are dozens on every page.
 */
export const MIN_SIDE = 64;

/** Whether an image is big enough to be worth a verdict. */
export function worthChecking(
  natural: { width: number; height: number },
  rendered: { width: number; height: number },
): boolean {
  // The natural size says what the image *is*; the rendered size says what the
  // user *sees*. A large photo shown as a thumbnail still shows its content.
  const smallest = (s: { width: number; height: number }) => Math.min(s.width, s.height);
  if (smallest(natural) < MIN_SIDE) return false;
  // Not laid out yet (hidden, or in a closed menu). It may be shown full size
  // later, so no size says it is safe to skip.
  if (rendered.width === 0 || rendered.height === 0) return true;
  return smallest(rendered) >= MIN_SIDE / 2;
}

/**
 * How a source reaches the classifier.
 *
 * - `url`: the classifier fetches it. Extension pages may read any origin, so
 *   cross-origin images (most of them, on CDNs) work without CORS.
 * - `pixels`: only this page can read it — a `blob:` URL belongs to the page
 *   that made it — so the content script copies it out through a canvas.
 * - `skip`: nothing to judge. SVG is drawings and icons, and cannot be
 *   decoded to a bitmap off the page anyway.
 */
export function sourceKind(src: string): "url" | "pixels" | "skip" {
  if (!src) return "skip";
  const lower = src.slice(0, 32).toLowerCase();
  if (lower.startsWith("data:image/svg")) return "skip";
  if (lower.startsWith("data:") || lower.startsWith("blob:")) return "pixels";
  if (lower.startsWith("http:") || lower.startsWith("https:")) {
    try {
      if (new URL(src).pathname.toLowerCase().endsWith(".svg")) return "skip";
    } catch {
      return "skip";
    }
    return "url";
  }
  return "skip";
}

/** Side length NSFWJS's MobileNetV2 takes. Decoding straight to it saves memory. */
export const MODEL_SIZE = 224;

/** A small LRU. `Map` keeps insertion order, so the first key is the oldest. */
export class VerdictCache {
  private readonly entries = new Map<string, Verdict>();

  constructor(private readonly limit = 2_000) {}

  get(key: string): Verdict | undefined {
    const hit = this.entries.get(key);
    if (hit === undefined) return undefined;
    this.entries.delete(key);
    this.entries.set(key, hit);
    return hit;
  }

  set(key: string, verdict: Verdict): void {
    // A failure may be a network blip. Remembering it would show that image
    // unjudged for the rest of the session.
    if (verdict === "error") return;
    this.entries.delete(key);
    this.entries.set(key, verdict);
    while (this.entries.size > this.limit) {
      const oldest = this.entries.keys().next().value;
      if (oldest === undefined) break;
      this.entries.delete(oldest);
    }
  }

  get size(): number {
    return this.entries.size;
  }
}

/** Data URLs can be megabytes. Past this a key costs more than a re-check. */
export const MAX_CACHE_KEY = 2_048;
