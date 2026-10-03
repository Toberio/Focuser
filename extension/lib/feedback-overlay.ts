/**
 * Debug builds only: a Show/Hide button over judged images, for labelling
 * the filter's mistakes on real pages.
 *
 * Hovering an image the filter judged shows "Hide" (it was shown) or "Show"
 * (it was blurred). The first click asks for confirmation, the second sends
 * the label. The button lives in its own closed shadow root, so the page's
 * styles cannot reach it, and it finds the image under the pointer through
 * whatever overlays the page puts on top (Pinterest covers every pin).
 *
 * Local to this fork, and only built with WXT_IMAGE_FILTER_DEBUG=true.
 */

export type FeedbackLabel = "show" | "hide";
type Media = HTMLImageElement | HTMLVideoElement;

const CSS = `
button {
  position: fixed; display: none; pointer-events: auto; z-index: 2147483647;
  font: 600 12px/1 system-ui, sans-serif; padding: 6px 11px; border: 0;
  border-radius: 999px; background: rgba(17, 17, 17, 0.85); color: #fff;
  cursor: pointer; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
}
button.armed { background: #c2410c; }
button.done { background: #15803d; }
`;

export function startFeedbackOverlay(
  attr: string,
  send: (el: Media, label: FeedbackLabel) => Promise<boolean>,
): () => void {
  const host = document.createElement("div");
  host.style.cssText = "position:fixed;top:0;left:0;width:0;height:0;z-index:2147483647";
  const root = host.attachShadow({ mode: "closed" });
  const style = document.createElement("style");
  style.textContent = CSS;
  const button = document.createElement("button");
  root.append(style, button);

  let target: Media | null = null;
  let armed = false;
  let busy = false;
  let disarm: ReturnType<typeof setTimeout> | undefined;
  let frame = 0;
  let pointer = { x: -1, y: -1 };

  const labelFor = (el: Media): FeedbackLabel =>
    el.getAttribute(attr) === "hidden" ? "show" : "hide";

  function place() {
    if (!target || !target.isConnected) return hide();
    const rect = target.getBoundingClientRect();
    if (rect.width < 40 || rect.height < 40) return hide();
    button.style.left = `${Math.max(4, rect.left + 8)}px`;
    button.style.top = `${Math.max(4, rect.top + 8)}px`;
    button.style.display = "block";
  }

  function reset(el: Media) {
    target = el;
    armed = false;
    clearTimeout(disarm);
    button.className = "";
    button.textContent = labelFor(el) === "show" ? "Show" : "Hide";
    place();
  }

  function hide() {
    if (busy) return;
    target = null;
    armed = false;
    button.style.display = "none";
  }

  function mediaAt(x: number, y: number): Media | null {
    for (const el of document.elementsFromPoint(x, y)) {
      if (el === host) return target;
      if ((el instanceof HTMLImageElement || el instanceof HTMLVideoElement) && el.hasAttribute(attr)) return el;
    }
    return null;
  }

  const onMove = (e: MouseEvent) => {
    pointer = { x: e.clientX, y: e.clientY };
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      if (busy) return;
      const el = mediaAt(pointer.x, pointer.y);
      if (!el) return hide();
      if (el !== target) reset(el);
    });
  };
  const onScroll = () => place();

  // The page must not see these clicks: a pin opens on click.
  for (const type of ["pointerdown", "mousedown", "mouseup", "pointerup"]) {
    button.addEventListener(type, (e) => e.stopPropagation());
  }
  button.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!target || busy) return;
    const label = labelFor(target);
    if (!armed) {
      armed = true;
      button.className = "armed";
      button.textContent = label === "show" ? "Confirm: should show" : "Confirm: should hide";
      disarm = setTimeout(() => target && reset(target), 4_000);
      return;
    }
    clearTimeout(disarm);
    busy = true;
    button.textContent = "Saving…";
    const el = target;
    void send(el, label).then((ok) => {
      button.className = ok ? "done" : "";
      // "v2" tells a stale build apart in a report.
      button.textContent = ok ? "Logged ✓ v2" : "Not logged (v2)";
      setTimeout(() => {
        busy = false;
        hide();
      }, 1_000);
    });
  });

  const mount = () => (document.body ?? document.documentElement).append(host);
  if (document.body) mount();
  else document.addEventListener("DOMContentLoaded", mount, { once: true });
  document.addEventListener("mousemove", onMove, { passive: true, capture: true });
  window.addEventListener("scroll", onScroll, { passive: true, capture: true });

  return () => {
    document.removeEventListener("mousemove", onMove, { capture: true });
    window.removeEventListener("scroll", onScroll, { capture: true });
    host.remove();
  };
}
