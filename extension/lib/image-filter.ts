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
 * - Verdicts come from `classifier.html`, an extension page running two models
 *   on WebGL: Marqo's nsfw-image-detection-384 (ViT-tiny, Apache-2.0) and
 *   NSFWJS (MobileNetV2, MIT). See `Scores` for how they are combined. Chrome hosts it as an offscreen document,
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
 * What a verdict rests on, each 0–1. Two models look at every image:
 *
 * - Marqo's ViT (`nsfw`) is the judge. It is far better at telling skin from
 *   things that merely look like it: NSFWJS scored a red paper-spiral
 *   wallpaper 0.74 porn.
 * - NSFWJS (`explicit`, `suggestive`) is the only one of the two that grades
 *   *suggestive*, which the ViT folds into plain safe or unsafe. Its view only
 *   counts at the stricter levels, and only when the ViT is not sure the image
 *   is safe, so the ViT can overrule its false alarms.
 */
export interface Scores {
  /** Marqo ViT: probability the image is NSFW. */
  nsfw: number;
  /** NSFWJS Porn plus Hentai. Summed because it splits a borderline image between them. */
  explicit: number;
  /** NSFWJS Sexy: no nudity, but posed or dressed to arouse. */
  suggestive: number;
  /** Where TF.js ran: `webgl` is the GPU, `cpu` and `wasm` are not. For the readout. */
  backend?: string;
}

/** NSFWJS's half of the scores. */
export function scoresOf(predictions: Prediction[]): Omit<Scores, "nsfw"> {
  const p = (name: NsfwClass) =>
    predictions.find((x) => x.className === name)?.probability ?? 0;
  return { explicit: p("Porn") + p("Hentai"), suggestive: p("Sexy") };
}

/** How strict the filter is, as the desktop app sends it. Off is never sent. */
export type FilterLevel = "explicit" | "balanced" | "strict";

/**
 * Per level, hide when any of these holds:
 *
 * - `sure`: the ViT alone is sure.
 * - `agree`: both lean that way. In the ViT's uncertain middle it hides only
 *   if NSFWJS also sees something. Each model has false alarms the other does
 *   not share: glossy abstract art got 0.48 from the ViT and 0.02 from NSFWJS;
 *   an abstract wallpaper got 0.16 from the ViT and 0.88 Porn from NSFWJS.
 * - `explicit`: NSFWJS's Porn plus Hentai is high and the ViT is not sure the
 *   image is safe.
 * - `suggestive`: NSFWJS's Sexy is high and the ViT gives it more than its
 *   floor. This is the only way a cleavage shot is caught: the ViT was trained
 *   on nudity against everything else, and scored them 0.10–0.13. NSFWJS's
 *   false alarms so far were all Porn, with Sexy near zero.
 *
 * Every threshold falls as the level rises, so each level hides everything
 * the one before it does.
 *
 * The ViT was trained with label smoothing, so it rarely says much below 0.05
 * or above 0.95: plain landscapes score about 0.06.
 */
interface Rule {
  vit: number;
  nsfwjs: number;
  /** NSFWJS's Sexy must also reach this. */
  suggestive?: number;
}

export const THRESHOLDS: Record<
  FilterLevel,
  { sure: number; agree: Rule; explicit: Rule; suggestive: Rule }
> = {
  explicit: {
    sure: 0.85,
    agree: { vit: 0.6, nsfwjs: 0.3 },
    explicit: { vit: 1, nsfwjs: Number.POSITIVE_INFINITY },
    suggestive: { vit: 1, nsfwjs: Number.POSITIVE_INFINITY },
  },
  balanced: {
    sure: 0.75,
    agree: { vit: 0.45, nsfwjs: 0.25 },
    explicit: { vit: 0.3, nsfwjs: 0.7 },
    suggestive: { vit: 0.08, nsfwjs: 0.75 },
  },
  // At strict, the ViT gets no say in NSFWJS's two rules. Pictures that should
  // be hidden scored as low on the ViT as ones that should not (0.10–0.13
  // against 0.05–0.16), so no ViT floor keeps one and loses the other. What
  // did separate them: a real photo NSFWJS calls Porn also carries some Sexy
  // (0.14), while its false alarms on abstract art and objects put nearly all
  // of it on Porn (Sexy 0.03 in all three). Fitted on four images, so expect
  // to move it.
  strict: {
    sure: 0.65,
    agree: { vit: 0.3, nsfwjs: 0.15 },
    explicit: { vit: 0, nsfwjs: 0.5, suggestive: 0.08 },
    suggestive: { vit: 0, nsfwjs: 0.35 },
  },
};

export function isExplicit(scores: Scores, level: FilterLevel): boolean {
  const t = THRESHOLDS[level];
  const meets = (rule: Rule, nsfwjs: number) =>
    scores.nsfw >= rule.vit &&
    nsfwjs >= rule.nsfwjs &&
    scores.suggestive >= (rule.suggestive ?? 0);
  return (
    scores.nsfw >= t.sure ||
    // At "explicit", NSFWJS's agreement means Porn or Hentai, never Sexy.
    meets(t.agree, scores.explicit + (level === "explicit" ? 0 : scores.suggestive)) ||
    meets(t.explicit, scores.explicit) ||
    meets(t.suggestive, scores.suggestive)
  );
}

/** Compact form for the `data-focuser-score` attribute. */
export function formatScores(scores: Scores): string {
  const parts = [
    `vit ${scores.nsfw.toFixed(2)}`,
    `explicit ${scores.explicit.toFixed(2)}`,
    `suggestive ${scores.suggestive.toFixed(2)}`,
  ];
  if (scores.backend) parts.push(scores.backend);
  return parts.join(" · ");
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

/** Side length NSFWJS's MobileNetV2 takes. */
export const NSFWJS_SIZE = 224;

/**
 * Longest side of the copy a content script makes of an image only the page
 * can read. The ViT's input size, so neither model gets less than it uses.
 */
export const COPY_SIZE = 384;

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
