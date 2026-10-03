/**
 * The explicit-image filter's browser half. Pure functions only, so every
 * decision can be tested without a browser.
 *
 * How the parts fit together:
 *
 * - The desktop app sets `image_filter` in the rule set to the strictest
 *   level any active list asks for.
 * - The background then registers the `image-filter` content script, which
 *   finds images and videos and asks the background about each.
 * - The background gets the image's bytes (on Firefox, copied from the page's
 *   own download; elsewhere fetched) and sends them to the desktop app, which
 *   judges them on the GPU and answers `clear` or `hidden`. The app holds the
 *   models and the thresholds; the extension holds neither.
 * - Nothing leaves the machine: the app answers on 127.0.0.1.
 */

/**
 * What the content script does with an image.
 *
 * `error` is shown, like `clear`. The filter fails open for the same reason the
 * block page does: a filter that leaves every image blurred when the app is
 * busy or closed would be switched off, and then it catches nothing.
 */
export type Verdict = "clear" | "hidden" | "error";

/** How strict the filter is, as the desktop app sends it. Off is never sent. */
export type FilterLevel = "explicit" | "balanced" | "strict";

/** A verdict, and the scores behind it when there were any. */
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
 * How a source reaches the app.
 *
 * - `url`: the background gets it, from Firefox's copy of the page's download
 *   or by fetching it. Extension contexts may read any origin, so cross-origin
 *   images (most of them, on CDNs) work without CORS.
 * - `pixels`: only this page can read it — a `blob:` URL belongs to the page
 *   that made it — so the content script copies it out through a canvas.
 * - `skip`: nothing to judge. SVG is drawings and icons.
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

/**
 * Longest side of the copy a content script makes of an image only the page
 * can read. The largest input either model takes, so neither gets less.
 */
export const COPY_SIZE = 384;

/** A small LRU. `Map` keeps insertion order, so the first key is the oldest. */
export class VerdictCache {
  private readonly entries = new Map<string, Judgement>();

  constructor(private readonly limit = 2_000) {}

  get(key: string): Judgement | undefined {
    const hit = this.entries.get(key);
    if (hit === undefined) return undefined;
    this.entries.delete(key);
    this.entries.set(key, hit);
    return hit;
  }

  set(key: string, judgement: Judgement): void {
    // A failure may be the app still loading. Remembering it would show that
    // image unjudged for the rest of the session.
    if (judgement.verdict === "error") return;
    this.entries.delete(key);
    this.entries.set(key, judgement);
    while (this.entries.size > this.limit) {
      const oldest = this.entries.keys().next().value;
      if (oldest === undefined) break;
      this.entries.delete(oldest);
    }
  }

  clear(): void {
    this.entries.clear();
  }

  get size(): number {
    return this.entries.size;
  }
}

/** Data URLs can be megabytes. Past this a key costs more than a re-check. */
export const MAX_CACHE_KEY = 2_048;
