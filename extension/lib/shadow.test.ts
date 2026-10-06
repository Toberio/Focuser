import { afterEach, describe, expect, it, vi } from "vitest";
import { deepElementsFromPoint } from "./shadow";

/**
 * Nested components, as Reddit nests them. As a real browser does, each
 * shadow root's hit-test names the next component in, its own content, and
 * every host around it all the way out.
 */
function nested(depth: number) {
  let tests = 0;
  const hosts: { name: string; shadowRoot: unknown }[] = [];
  const roots: { elementsFromPoint: () => unknown[] }[] = [];
  for (let i = 0; i < depth; i++) {
    const root = {
      elementsFromPoint: () => {
        tests++;
        const inner = hosts[i + 1];
        const content = { name: `content ${i}`, shadowRoot: null };
        return [...(inner ? [inner] : []), content, ...hosts.slice(0, i + 1).reverse()];
      },
    };
    roots.push(root);
    hosts.push({ name: `host ${i}`, shadowRoot: root });
  }
  const document = {
    elementsFromPoint: () => {
      tests++;
      return [hosts[0]];
    },
  };
  return { document, tests: () => tests };
}

describe("deepElementsFromPoint", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("finds what is inside the innermost component, on top", () => {
    const page = nested(3);
    vi.stubGlobal("document", page.document);
    const names = deepElementsFromPoint(1, 1).map((e) => (e as unknown as { name: string }).name);
    expect(names[0]).toBe("content 2");
    expect(names.indexOf("content 2")).toBeLessThan(names.indexOf("host 2"));
    expect(names).toContain("host 0");
  });

  it("hit-tests each root once, however deep the nesting", () => {
    const page = nested(12);
    vi.stubGlobal("document", page.document);
    deepElementsFromPoint(1, 1);
    // The document and twelve roots: 13, not 5^12.
    expect(page.tests()).toBe(13);
  });
});
