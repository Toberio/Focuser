import { describe, expect, it } from "vitest";
import { isTappable, MAX_CACHE_BYTES, ResponseTap } from "./response-tap";

const buf = (n: number) => new ArrayBuffer(n);

describe("ResponseTap", () => {
  it("hands over a finished download", async () => {
    const tap = new ResponseTap();
    tap.started("a");
    const bytes = buf(10);
    tap.finished("a", bytes);
    expect(await tap.take("a", 0)).toBe(bytes);
  });

  it("waits for a download still streaming", async () => {
    const tap = new ResponseTap();
    tap.started("a");
    const waiting = tap.take("a", 1_000);
    const bytes = buf(10);
    tap.finished("a", bytes);
    expect(await waiting).toBe(bytes);
  });

  it("says fetch it yourself for an image it never saw", async () => {
    expect(await new ResponseTap().take("a", 1_000)).toBeNull();
  });

  it("gives up waiting after the timeout", async () => {
    const tap = new ResponseTap();
    tap.started("a");
    expect(await tap.take("a", 5)).toBeNull();
  });

  it("drops the oldest copies past the byte budget", () => {
    const tap = new ResponseTap();
    const big = MAX_CACHE_BYTES / 2 + 1;
    for (const url of ["a", "b", "c"]) {
      tap.started(url);
      tap.finished(url, buf(big));
    }
    expect(tap.size).toBe(1);
    expect(tap.bytes).toBeLessThanOrEqual(MAX_CACHE_BYTES);
  });

  it("forgets copies nobody asked for in time", async () => {
    let now = 0;
    const tap = new ResponseTap(() => now);
    tap.started("a");
    tap.finished("a", buf(1));
    now = 61_000;
    expect(await tap.take("a", 0)).toBeNull();
  });
});

describe("isTappable", () => {
  it("copies ordinary images", () => {
    expect(isTappable("image/jpeg", "12345")).toBe(true);
    expect(isTappable("image/webp", undefined)).toBe(true);
  });

  it("skips SVG, non-images and huge files", () => {
    expect(isTappable("image/svg+xml", "100")).toBe(false);
    expect(isTappable("text/html", "100")).toBe(false);
    expect(isTappable("image/png", String(64 * 1024 * 1024))).toBe(false);
  });
});
