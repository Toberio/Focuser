import { describe, expect, it } from "vitest";
import { MIN_SIDE, sourceKind, VerdictCache, worthChecking } from "./image-filter";

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
  it("lets the background fetch web images", () => {
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
    cache.set("a", { verdict: "clear" });
    cache.set("b", { verdict: "hidden" });
    cache.get("a");
    cache.set("c", { verdict: "clear" });
    expect(cache.get("a")?.verdict).toBe("clear");
    expect(cache.get("b")).toBeUndefined();
    expect(cache.size).toBe(2);
  });

  it("does not remember failures, so they are retried", () => {
    const cache = new VerdictCache();
    cache.set("a", { verdict: "error" });
    expect(cache.get("a")).toBeUndefined();
  });
});
