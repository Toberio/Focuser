import { describe, expect, it } from "vitest";
import {
  isExplicit,
  MIN_SIDE,
  type Prediction,
  sourceKind,
  VerdictCache,
  worthChecking,
} from "./image-filter";

const scores = (s: Partial<Record<string, number>>): Prediction[] =>
  Object.entries(s).map(([className, probability]) => ({ className, probability: probability ?? 0 }));

describe("isExplicit", () => {
  it("hides what the model calls porn or hentai", () => {
    expect(isExplicit(scores({ Porn: 0.9, Neutral: 0.1 }))).toBe(true);
    expect(isExplicit(scores({ Hentai: 0.8, Drawing: 0.2 }))).toBe(true);
  });

  it("adds porn and hentai, since the model splits between them", () => {
    expect(isExplicit(scores({ Porn: 0.25, Hentai: 0.2, Neutral: 0.55 }))).toBe(true);
  });

  it("leaves swimwear-level 'sexy' alone unless explicit scores join it", () => {
    expect(isExplicit(scores({ Sexy: 0.6, Neutral: 0.4 }))).toBe(false);
    expect(isExplicit(scores({ Sexy: 0.5, Porn: 0.25, Neutral: 0.25 }))).toBe(true);
  });

  it("shows neutral pictures and drawings", () => {
    expect(isExplicit(scores({ Neutral: 0.95, Sexy: 0.05 }))).toBe(false);
    expect(isExplicit(scores({ Drawing: 0.9, Hentai: 0.1 }))).toBe(false);
    expect(isExplicit([])).toBe(false);
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

describe("VerdictCache", () => {
  it("drops the least recently used entry", () => {
    const cache = new VerdictCache(2);
    cache.set("a", "clear");
    cache.set("b", "hidden");
    cache.get("a");
    cache.set("c", "clear");
    expect(cache.get("a")).toBe("clear");
    expect(cache.get("b")).toBeUndefined();
    expect(cache.size).toBe(2);
  });

  it("does not remember failures, so they are retried", () => {
    const cache = new VerdictCache();
    cache.set("a", "error");
    expect(cache.get("a")).toBeUndefined();
  });
});
