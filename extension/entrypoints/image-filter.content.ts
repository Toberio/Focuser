import { MODEL_SIZE, sourceKind, type Verdict, worthChecking } from "@/lib/image-filter";
import type { Message, MessageReply } from "@/lib/messages";
import { send } from "@/lib/messages";

/**
 * Blurs every image, then shows each one the classifier clears.
 *
 * Registered by the background only while a list has the filter on, so a page
 * pays nothing for it otherwise. It runs at `document_start` so the blur is in
 * place before the first image can paint: hiding an image after it has been
 * seen would be no filter at all.
 *
 * Covers `<img>`, which includes `<picture>` and `srcset`. Not yet: CSS
 * background images, `<video>`, `<canvas>` and images inside shadow roots.
 */

const ATTR = "data-focuser-image";
const STYLE = `img:not([${ATTR}="clear"]){filter:blur(28px) grayscale(1)!important;clip-path:inset(0)!important}`;
/** Start judging an image this far before it scrolls into view. */
const LOOKAHEAD = "100% 0px";

type FilterWindow = Window & { __focuserImageFilter?: boolean };

export default defineContentScript({
  matches: ["<all_urls>"],
  runAt: "document_start",
  allFrames: true,
  registration: "runtime",

  main() {
    // Registered for new pages and injected into open ones, so a page can get
    // it twice.
    const self = window as FilterWindow;
    if (self.__focuserImageFilter) return;
    self.__focuserImageFilter = true;

    const style = document.createElement("style");
    style.textContent = STYLE;
    (document.head ?? document.documentElement).appendChild(style);

    /** Which source each image's verdict is for. A new source needs a new one. */
    const judged = new WeakMap<HTMLImageElement, string>();
    const verdicts = new Map<string, Promise<Verdict>>();

    function mark(img: HTMLImageElement, verdict: Verdict) {
      img.setAttribute(ATTR, verdict === "hidden" ? "hidden" : "clear");
    }

    /** Copy an image only this page can read into something the classifier can. */
    function toDataUrl(img: HTMLImageElement): string | null {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = MODEL_SIZE;
        canvas.height = MODEL_SIZE;
        canvas.getContext("2d")?.drawImage(img, 0, 0, MODEL_SIZE, MODEL_SIZE);
        return canvas.toDataURL("image/jpeg", 0.9);
      } catch {
        return null;
      }
    }

    function verdictFor(src: string, img: HTMLImageElement): Promise<Verdict> {
      const known = verdicts.get(src);
      if (known) return known;
      const payload = sourceKind(src) === "pixels" ? toDataUrl(img) : src;
      const job = payload
        ? send({ type: "classify-image", src: payload }).then((r) => r?.verdict ?? "error")
        : Promise.resolve<Verdict>("error");
      verdicts.set(src, job);
      return job;
    }

    async function judge(img: HTMLImageElement) {
      const src = img.currentSrc || img.src;
      if (judged.get(img) !== src) return;
      const verdict = await verdictFor(src, img);
      // The page may have swapped the source while we waited.
      if (judged.get(img) === src) mark(img, verdict);
    }

    const nearView = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          nearView.unobserve(entry.target);
          void judge(entry.target as HTMLImageElement);
        }
      },
      { rootMargin: LOOKAHEAD },
    );

    function consider(img: HTMLImageElement) {
      // Not decoded yet; its `load` event brings it back here.
      if (!img.complete || img.naturalWidth === 0) return;
      const src = img.currentSrc || img.src;
      if (judged.get(img) === src) return;
      judged.set(img, src);

      const rect = img.getBoundingClientRect();
      if (
        sourceKind(src) === "skip" ||
        !worthChecking(
          { width: img.naturalWidth, height: img.naturalHeight },
          { width: rect.width, height: rect.height },
        )
      ) {
        mark(img, "clear");
        return;
      }
      // Offscreen images wait until they are about to be seen. A long feed
      // can hold hundreds, and most are never scrolled to.
      nearView.observe(img);
    }

    /** A changed source is a new image, and it is blurred again until judged. */
    function forget(img: HTMLImageElement) {
      judged.delete(img);
      img.removeAttribute(ATTR);
      nearView.unobserve(img);
    }

    // `load` does not bubble, but it can be caught on the way down.
    const onLoad = (event: Event) => {
      if (event.target instanceof HTMLImageElement) consider(event.target);
    };
    document.addEventListener("load", onLoad, true);

    // Runs before the next paint, so a swapped-in image never shows unjudged.
    const sources = new MutationObserver((records) => {
      for (const record of records) {
        const target = record.target;
        if (target instanceof HTMLImageElement) forget(target);
        else if (target instanceof HTMLSourceElement) {
          const img = target.parentElement?.querySelector("img");
          if (img) forget(img);
        }
      }
    });
    sources.observe(document.documentElement, {
      subtree: true,
      attributes: true,
      attributeFilter: ["src", "srcset"],
    });

    // Injected into an open tab, the images are loaded already.
    for (const img of Array.from(document.images)) consider(img);

    const onMessage = (
      raw: unknown,
      _sender: unknown,
      sendResponse: (reply: MessageReply) => void,
    ) => {
      if ((raw as Message).type !== "image-filter-off") return false;
      nearView.disconnect();
      sources.disconnect();
      document.removeEventListener("load", onLoad, true);
      browser.runtime.onMessage.removeListener(onMessage);
      style.remove();
      for (const img of Array.from(document.querySelectorAll(`img[${ATTR}]`)))
        img.removeAttribute(ATTR);
      self.__focuserImageFilter = false;
      sendResponse({ type: "image-filter-off", ok: true });
      return false;
    };
    browser.runtime.onMessage.addListener(onMessage);
  },
});
