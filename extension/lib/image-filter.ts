/**
 * The explicit-image filter. Pure functions only, so every decision can be
 * tested without a browser.
 *
 * How the parts fit together:
 *
 * - The desktop app sets `image_filter` in the rule set to the strictest
 *   level any active list asks for.
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

/** The two numbers a verdict rests on, each 0–1. */
export interface Scores {
  /** Porn plus Hentai. Summed because the model splits a borderline image between them. */
  explicit: number;
  /** Sexy: no nudity, but posed or dressed to arouse. */
  suggestive: number;
}

export function scoresOf(predictions: Prediction[]): Scores {
  const p = (name: NsfwClass) =>
    predictions.find((x) => x.className === name)?.probability ?? 0;
  return { explicit: p("Porn") + p("Hentai"), suggestive: p("Sexy") };
}

/** How strict the filter is, as the desktop app sends it. Off is never sent. */
export type FilterLevel = "explicit" | "balanced" | "strict";

/**
 * Per level: hide when the explicit score passes `explicit`, or explicit plus
 * suggestive passes `combined`.
 *
 * Each level hides everything the one before it does. "Strict" is low on
 * purpose: people turn it on to stop being drawn in, and what does that on a
 * feed is mostly suggestive, not nude. On a real Pinterest feed, a bar that let
 * "Sexy" alone through let most of it through.
 */
export const THRESHOLDS: Record<FilterLevel, { explicit: number; combined: number }> = {
  explicit: { explicit: 0.5, combined: Number.POSITIVE_INFINITY },
  balanced: { explicit: 0.35, combined: 0.6 },
  strict: { explicit: 0.25, combined: 0.4 },
};

export function isExplicit(scores: Scores, level: FilterLevel): boolean {
  const t = THRESHOLDS[level];
  return scores.explicit >= t.explicit || scores.explicit + scores.suggestive >= t.combined;
}

/** Compact form for the `data-focuser-score` attribute: `explicit/suggestive`. */
export function formatScores(scores: Scores): string {
  return `${scores.explicit.toFixed(2)}/${scores.suggestive.toFixed(2)}`;
}

/** A verdict, and the scores behind it when the model produced any. */
export interface Judgement {
  verdict: Verdict;
  score?: string;
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
export class ScoreCache {
  private readonly entries = new Map<string, Scores>();

  constructor(private readonly limit = 2_000) {}

  get(key: string): Scores | undefined {
    const hit = this.entries.get(key);
    if (hit === undefined) return undefined;
    this.entries.delete(key);
    this.entries.set(key, hit);
    return hit;
  }

  /** Scores, not verdicts, so a change of level needs no second look. */
  set(key: string, scores: Scores): void {
    this.entries.delete(key);
    this.entries.set(key, scores);
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
