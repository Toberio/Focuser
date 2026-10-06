import { describe, expect, it } from "vitest";
import {
  afterVideoLook,
  appDecodes,
  MIN_SIDE,
  newVideoWatch,
  skipPatterns,
  sourceKind,
  type Verdict,
  VerdictCache,
  worthChecking,
} from "./image-filter";

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

describe("skipPatterns", () => {
  it("covers each site and its subdomains, however it was typed", () => {
    expect(skipPatterns(["https://www.YouTube.com/feed", "netflix.com", "youtube.com"])).toEqual([
      "*://netflix.com/*",
      "*://*.netflix.com/*",
      "*://youtube.com/*",
      "*://*.youtube.com/*",
    ]);
  });

  it("is empty when nothing is skipped", () => {
    expect(skipPatterns(undefined)).toEqual([]);
    expect(skipPatterns(["", "  "])).toEqual([]);
  });
});

describe("afterVideoLook", () => {
  const looks = (...verdicts: Verdict[]) => verdicts.reduce(afterVideoLook, newVideoWatch()).state;

  it("starts hidden, and one clear frame is not enough", () => {
    expect(newVideoWatch().state).toBe("pending");
    expect(looks("clear")).toBe("pending");
  });

  it("shows a video after two clear looks", () => {
    expect(looks("clear", "clear")).toBe("clear");
  });

  it("hides it for good on any explicit frame, before or after it was shown", () => {
    expect(looks("clear", "hidden", "clear", "clear")).toBe("hidden");
    expect(looks("clear", "clear", "hidden", "clear")).toBe("hidden");
  });

  it("fails open when its frames cannot be judged, like the rest of the filter", () => {
    expect(looks("error")).toBe("pending");
    expect(looks("error", "error")).toBe("clear");
    expect(looks("error", "error", "hidden")).toBe("hidden");
  });

  it("does not let a failed look undo a shown video", () => {
    expect(looks("clear", "clear", "error")).toBe("clear");
  });
});

describe("appDecodes", () => {
  const bytes = (...b: (number | string)[]) =>
    new Uint8Array(b.flatMap((x) => (typeof x === "string" ? [...x].map((c) => c.charCodeAt(0)) : [x])));

  it("passes the formats the app reads straight through", () => {
    expect(appDecodes(bytes(0xff, 0xd8, 0xff, 0xe0))).toBe(true);
    expect(appDecodes(bytes(0x89, "PNG", 0x0d, 0x0a))).toBe(true);
    expect(appDecodes(bytes("GIF89a"))).toBe(true);
    expect(appDecodes(bytes("RIFF", 0, 0, 0, 0, "WEBPVP8 "))).toBe(true);
    expect(appDecodes(bytes("BM", 0, 0))).toBe(true);
  });

  it("sends AVIF and anything unknown through the browser first", () => {
    // An AVIF file as one image host served it.
    expect(appDecodes(bytes(0, 0, 0, 0x1c, "ftypavif", 0, 0, 0, 0))).toBe(false);
    expect(appDecodes(bytes("RIFF", 0, 0, 0, 0, "WAVE"))).toBe(false);
    expect(appDecodes(new Uint8Array())).toBe(false);
  });
});
