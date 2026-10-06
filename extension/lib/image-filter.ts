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

import { canonicalHost } from "./rules";

/** How strict the filter is, as the desktop app sends it. Off is never sent. */
export type FilterLevel = "explicit" | "balanced" | "strict";

/** A verdict, and the scores behind it when there were any. */
export interface Judgement {
  verdict: Verdict;
  score?: string;
  /** Frames the app judged, each a look; absent from an older app, so one. */
  frames?: number;
  /** The whole file was judged, not just the start of it. */
  complete?: boolean;
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

/**
 * The images in a computed `background-image`: every `url(...)` layer that
 * can be fetched. Gradients and other generated layers are left out.
 */
export function backgroundUrls(css: string): string[] {
  if (!css.includes("url(")) return [];
  return [...css.matchAll(/url\(\s*(["']?)(.*?)\1\s*\)/g)]
    .map((m) => (m[2] ?? "").replace(/\\(.)/g, "$1"))
    .filter((u) => sourceKind(u) === "url");
}

/**
 * Whether a source is a GIF, by its URL. GIFs are treated like videos: an
 * animation may turn explicit after its first frame, and Reddit galleries
 * are full of multi-megabyte ones that took seconds to judge while they
 * played unblurred. So they stay blurred until two frames pass, and are
 * judged from the start of the file, not all of it.
 */
export function isGif(src: string): boolean {
  if (!/^https?:/i.test(src)) return false;
  try {
    return new URL(src).pathname.toLowerCase().endsWith(".gif");
  } catch {
    return false;
  }
}

/**
 * A GIF's first look: this much of its start, enough for a few frames. The
 * app judges the first frame and the last it could decode, two looks, so a
 * clean GIF is shown after one small download. A whole GIF could be twenty
 * megabytes, fetched again alongside the page's own download.
 */
export const GIF_FIRST_BYTES = 512 * 1024;

/**
 * A GIF's second look, taken after it is shown and at low priority: a frame
 * a second through its first seconds. An explicit one hides it again, as a
 * later frame does a video.
 */
export const GIF_LATER_BYTES = 6 * 1024 * 1024;

/**
 * Whether the app can decode these bytes itself: JPEG, PNG, GIF, WebP or BMP,
 * by their first bytes. Anything else (AVIF above all, which some image hosts
 * serve to every browser that accepts it) the browser decodes and sends on as
 * a JPEG copy. Sent as it was, the app could not read it, and the filter,
 * failing open, showed it.
 */
export function appDecodes(head: Uint8Array): boolean {
  const starts = (...bytes: number[]) => bytes.every((b, i) => head[i] === b);
  const ascii = (at: number, text: string) =>
    [...text].every((c, i) => head[at + i] === c.charCodeAt(0));
  return (
    starts(0xff, 0xd8, 0xff) ||
    starts(0x89, 0x50, 0x4e, 0x47) ||
    ascii(0, "GIF8") ||
    (ascii(0, "RIFF") && ascii(8, "WEBP")) ||
    ascii(0, "BM")
  );
}

/**
 * Where a video stands. Unlike an image, a video starts out hidden: it is
 * judged a frame at a time, and sites swap its source or element as it starts
 * playing, so "show until judged" let it play unfiltered each time.
 */
export interface VideoWatch {
  state: "pending" | "clear" | "hidden";
  /** Different frames (or the poster) judged clear, in a row. */
  clearLooks: number;
  /** Looks that could not be judged: the app was unreachable, or the frame unreadable. */
  failedLooks: number;
}

/** Clear looks in a row before a video is shown: one harmless frame proves little. */
export const VIDEO_CLEAR_LOOKS = 2;
/**
 * Unjudgeable looks before a pending video is shown anyway: fail open, as the
 * filter does everywhere (see `Verdict`). A frame from another site served
 * without CORS can never be read, and would otherwise stay hidden for good.
 */
export const VIDEO_FAILED_LOOKS = 2;

export function newVideoWatch(): VideoWatch {
  return { state: "pending", clearLooks: 0, failedLooks: 0 };
}

/**
 * What one answer from the app adds: a look for each frame it judged. A
 * file judged whole has had every look it will get, so it counts as enough.
 */
export function afterLooks(watch: VideoWatch, judgement: Judgement): VideoWatch {
  if (judgement.verdict !== "clear") return afterVideoLook(watch, judgement.verdict);
  const looks = judgement.complete
    ? Math.max(VIDEO_CLEAR_LOOKS, judgement.frames ?? 1)
    : Math.max(1, judgement.frames ?? 1);
  let next = watch;
  for (let i = 0; i < looks; i++) next = afterVideoLook(next, "clear");
  return next;
}

/** One more look at a video. Any explicit frame hides it for good. */
export function afterVideoLook(watch: VideoWatch, verdict: Verdict): VideoWatch {
  if (watch.state === "hidden" || verdict === "hidden") {
    return { ...watch, state: "hidden" };
  }
  if (verdict === "clear") {
    const clearLooks = watch.clearLooks + 1;
    const shown = watch.state === "clear" || clearLooks >= VIDEO_CLEAR_LOOKS;
    return { ...watch, clearLooks, state: shown ? "clear" : "pending" };
  }
  const failedLooks = watch.failedLooks + 1;
  const shown = watch.state === "clear" || failedLooks >= VIDEO_FAILED_LOOKS;
  return { ...watch, failedLooks, state: shown ? "clear" : "pending" };
}

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

/**
 * The `excludeMatches` that keep the content script off the sites the filter
 * skips. A match pattern's `*.` does not cover the bare domain, so each site
 * needs both forms.
 */
export function skipPatterns(sites: string[] | undefined): string[] {
  const hosts = new Set((sites ?? []).map(canonicalHost).filter((h) => h && !h.includes("*")));
  return [...hosts].sort().flatMap((h) => [`*://${h}/*`, `*://*.${h}/*`]);
}
