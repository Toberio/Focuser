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
  const hiddenAt = (s: Partial<Record<string, number>>) =>
    LEVELS.filter((level) => isExplicit(scoresOf(scores(s)), level));

  it("hides porn and hentai at every level", () => {
    expect(hiddenAt({ Porn: 0.9, Neutral: 0.1 })).toEqual(LEVELS);
    expect(hiddenAt({ Hentai: 0.8, Drawing: 0.2 })).toEqual(LEVELS);
  });

  it("adds porn and hentai, since the model splits between them", () => {
    expect(hiddenAt({ Porn: 0.3, Hentai: 0.25, Neutral: 0.45 })).toEqual(LEVELS);
  });

  it("leaves suggestive pictures to the stricter levels", () => {
    // A cleavage selfie on a feed.
    expect(hiddenAt({ Sexy: 0.45, Neutral: 0.55 })).toEqual(["strict"]);
    expect(hiddenAt({ Sexy: 0.7, Neutral: 0.3 })).toEqual(["balanced", "strict"]);
  });

  it("shows neutral pictures and drawings at every level", () => {
    expect(hiddenAt({ Neutral: 0.9, Sexy: 0.1 })).toEqual([]);
    expect(hiddenAt({ Drawing: 0.85, Hentai: 0.1, Neutral: 0.05 })).toEqual([]);
    expect(hiddenAt({})).toEqual([]);
  });

  it("hides at a level everything the level below it hides", () => {
    for (let p = 0; p <= 1; p += 0.05) {
      for (let s = 0; s <= 1 - p; s += 0.05) {
        const hidden = hiddenAt({ Porn: p, Sexy: s });
        // Once a level hides it, every stricter one does too.
        const first = hidden[0] ? LEVELS.indexOf(hidden[0]) : LEVELS.length;
        expect(hidden).toEqual(LEVELS.slice(first));
      }
    }
  });
});

describe("formatScores", () => {
  it("is explicit/suggestive to two places", () => {
    expect(formatScores(scoresOf(scores({ Porn: 0.1, Hentai: 0.05, Sexy: 0.333 })))).toBe(
      "0.15/0.33",
    );
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
    const s = { explicit: 0, suggestive: 0 };
    cache.set("a", s);
    cache.set("b", s);
    cache.get("a");
    cache.set("c", s);
    expect(cache.get("a")).toBe(s);
    expect(cache.get("b")).toBeUndefined();
    expect(cache.size).toBe(2);
  });
});
