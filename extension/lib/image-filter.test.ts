import { describe, expect, it } from "vitest";
import {
  type FilterLevel,
  formatScores,
  isExplicit,
  MIN_SIDE,
  type Prediction,
  ScoreCache,
  scoresOf,
  sourceKind,
  worthChecking,
} from "./image-filter";

const scores = (s: Partial<Record<string, number>>): Prediction[] =>
  Object.entries(s).map(([className, probability]) => ({ className, probability: probability ?? 0 }));

const LEVELS: FilterLevel[] = ["explicit", "balanced", "strict"];

describe("isExplicit", () => {
  const hiddenAt = (nsfw: number, s: Partial<Record<string, number>> = {}) =>
    LEVELS.filter((level) => isExplicit({ nsfw, ...scoresOf(scores(s)) }, level));

  it("hides what the ViT is sure of at every level", () => {
    expect(hiddenAt(0.9)).toEqual(LEVELS);
  });

  it("hides what the ViT is less sure of only at stricter levels", () => {
    expect(hiddenAt(0.8)).toEqual(["balanced", "strict"]);
    expect(hiddenAt(0.7)).toEqual(["strict"]);
  });

  it("shows what both models call safe", () => {
    expect(hiddenAt(0.06, { Neutral: 0.95 })).toEqual([]);
  });

  it("needs NSFWJS to agree when the ViT is unsure", () => {
    // Real scores: glossy abstract art the ViT half-suspected.
    expect(hiddenAt(0.48, { Porn: 0.02, Neutral: 0.98 })).toEqual([]);
    expect(hiddenAt(0.48, { Porn: 0.3, Neutral: 0.7 })).toEqual(["balanced", "strict"]);
  });

  it("lets the ViT overrule NSFWJS's false alarms", () => {
    // Real scores: abstract wallpapers NSFWJS called porn.
    expect(hiddenAt(0.16, { Porn: 0.85, Hentai: 0.03, Sexy: 0.03 })).toEqual([]);
    expect(hiddenAt(0.06, { Porn: 0.7, Hentai: 0.04, Sexy: 0.03 })).toEqual([]);
  });

  it("catches suggestive pictures, which the ViT is only unsure about", () => {
    expect(hiddenAt(0.22, { Sexy: 0.55, Neutral: 0.45 })).toEqual(["strict"]);
    expect(hiddenAt(0.35, { Sexy: 0.75, Neutral: 0.25 })).toEqual(["balanced", "strict"]);
  });

  it("does not count suggestive at the explicit-only level", () => {
    expect(hiddenAt(0.65, { Sexy: 0.9 })).not.toContain("explicit");
    expect(hiddenAt(0.65, { Porn: 0.4 })).toContain("explicit");
  });

  it("hides at a level everything the level below it hides", () => {
    for (let nsfw = 0; nsfw <= 1; nsfw += 0.05) {
      for (let p = 0; p <= 1; p += 0.1) {
        for (let s = 0; s <= 1 - p; s += 0.1) {
          const hidden = hiddenAt(nsfw, { Porn: p, Sexy: s });
          const first = hidden[0] ? LEVELS.indexOf(hidden[0]) : LEVELS.length;
          expect(hidden).toEqual(LEVELS.slice(first));
        }
      }
    }
  });
});

describe("formatScores", () => {
  it("shows every score and where it ran", () => {
    expect(
      formatScores({
        nsfw: 0.123,
        ...scoresOf(scores({ Porn: 0.1, Hentai: 0.05, Sexy: 0.333 })),
        backend: "webgl",
      }),
    ).toBe("vit 0.12 · explicit 0.15 · suggestive 0.33 · webgl");
  });
});

describe("worthChecking", () => {
  const big = { width: 800, height: 600 };

  it("skips icons by their real size", () => {
    expect(worthChecking({ width: 32, height: 32 }, { width: 300, height: 300 })).toBe(false);
    expect(worthChecking({ width: 800, height: MIN_SIDE - 1 }, big)).toBe(false);
  });

  it("checks a large image even when shown as a thumbnail", () => {
    expect(worthChecking(big, { width: 48, height: 36 })).toBe(true);
  });

  it("skips a large image squeezed to a dot", () => {
    expect(worthChecking(big, { width: 16, height: 12 })).toBe(false);
  });

  it("checks an image that is not laid out yet, since it may be shown later", () => {
    expect(worthChecking(big, { width: 0, height: 0 })).toBe(true);
  });
});

describe("sourceKind", () => {
  it("lets the classifier fetch web images", () => {
    expect(sourceKind("https://cdn.example.com/a.jpg")).toBe("url");
    expect(sourceKind("http://example.com/a.webp?x=1")).toBe("url");
  });

  it("copies out what only the page can read", () => {
    expect(sourceKind("data:image/jpeg;base64,AAAA")).toBe("pixels");
    expect(sourceKind("blob:https://example.com/1234")).toBe("pixels");
  });

  it("skips SVG and anything unfetchable", () => {
    expect(sourceKind("https://example.com/logo.svg")).toBe("skip");
    expect(sourceKind("https://example.com/logo.SVG?v=2")).toBe("skip");
    expect(sourceKind("data:image/svg+xml;utf8,<svg/>")).toBe("skip");
    expect(sourceKind("chrome-extension://abc/x.png")).toBe("skip");
    expect(sourceKind("")).toBe("skip");
  });
});

describe("ScoreCache", () => {
  it("drops the least recently used entry", () => {
    const cache = new ScoreCache(2);
    const s = { nsfw: 0, explicit: 0, suggestive: 0 };
    cache.set("a", s);
    cache.set("b", s);
    cache.get("a");
    cache.set("c", s);
    expect(cache.get("a")).toBe(s);
    expect(cache.get("b")).toBeUndefined();
    expect(cache.size).toBe(2);
  });
});
