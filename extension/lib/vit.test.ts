import { describe, expect, it } from "vitest";
import { halfToFloat } from "./vit";

describe("halfToFloat", () => {
  it("decodes the float16 weights file", () => {
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x3555)).toBeCloseTo(1 / 3, 3);
    expect(halfToFloat(0x0000)).toBe(0);
    // Subnormal: the smallest positive half.
    expect(halfToFloat(0x0001)).toBeCloseTo(5.96e-8, 10);
    expect(halfToFloat(0x7c00)).toBe(Number.POSITIVE_INFINITY);
  });
});
