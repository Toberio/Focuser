/**
 * Shadow DOM, for the image filter. Sites built from web components keep
 * their media inside shadow roots (Reddit's `shreddit-player` holds its
 * `<video>` in one), where the document's styles, its media events and its
 * hit-testing never reach.
 */

/**
 * Everything under a point, topmost first, looking inside open shadow roots:
 * `document.elementsFromPoint` stops at each root's host.
 *
 * Each root is hit-tested once. A shadow root's own `elementsFromPoint` also
 * returns the hosts around it, so following every host it names leads back
 * into roots already seen: with Reddit's six or more nested components that
 * was 5^depth hit-tests, and one click froze the tab.
 */
export function deepElementsFromPoint(x: number, y: number): Element[] {
  const out: Element[] = [];
  const seen = new Set<Document | ShadowRoot>();
  const visit = (root: Document | ShadowRoot) => {
    if (seen.has(root)) return;
    seen.add(root);
    for (const el of root.elementsFromPoint(x, y)) {
      if (el.shadowRoot) visit(el.shadowRoot);
      if (!out.includes(el)) out.push(el);
    }
  };
  visit(document);
  return out;
}
