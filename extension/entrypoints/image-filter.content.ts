import { type Judgement, MODEL_SIZE, sourceKind, worthChecking } from "@/lib/image-filter";
import type { Message, MessageReply } from "@/lib/messages";
import { send } from "@/lib/messages";

/**
 * Blurs every image and video, then shows each one the classifier clears.
 *
 * Registered by the background only while a list has the filter on, so a page
 * pays nothing for it otherwise. It runs at `document_start` so the blur is in
 * place before the first image can paint: hiding an image after it has been
 * seen would be no filter at all.
 *
 * Images are judged once per source. Videos are judged on their poster and
 * then on a frame every few seconds while they play, since a clip that opens
 * on a harmless frame need not stay harmless. A video found explicit stays
 * hidden until its source changes.
 *
 * Covers `<img>` (including `<picture>` and `srcset`) and `<video>`. Not yet:
 * CSS background images, `<canvas>` and anything inside shadow roots.
 */

const ATTR = "data-focuser-image";
/** The scores behind a verdict, `explicit/suggestive`, for tuning the thresholds. */
const SCORE_ATTR = "data-focuser-score";
const STYLE = `:is(img,video):not([${ATTR}="clear"]){filter:blur(28px) grayscale(1)!important;clip-path:inset(0)!important}`;
/** Start judging an image this far before it scrolls into view. */
const LOOKAHEAD = "100% 0px";
/** How often a playing, visible video has a frame checked. */
const FRAME_INTERVAL_MS = 2_500;

type Media = HTMLImageElement | HTMLVideoElement;
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

    /** Which source each element's verdict is for. A new source needs a new one. */
    const judged = new WeakMap<Media, string>();
    const verdicts = new Map<string, Promise<Judgement>>();
    /** Videos near the viewport, the only ones worth sampling. */
    const visibleVideos = new Set<HTMLVideoElement>();

    function mark(el: Media, judgement: Judgement) {
      el.setAttribute(ATTR, judgement.verdict === "hidden" ? "hidden" : "clear");
      if (judgement.score) el.setAttribute(SCORE_ATTR, judgement.score);
    }

    /** Copy what only this page can read into something the classifier can. */
    function toDataUrl(el: Media): string | null {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = MODEL_SIZE;
        canvas.height = MODEL_SIZE;
        canvas.getContext("2d")?.drawImage(el, 0, 0, MODEL_SIZE, MODEL_SIZE);
        // Throws on a cross-origin image or video served without CORS.
        return canvas.toDataURL("image/jpeg", 0.9);
      } catch {
        return null;
      }
    }

    function classify(payload: string | null): Promise<Judgement> {
      if (!payload) return Promise.resolve({ verdict: "error" });
      return send({ type: "classify-image", src: payload }).then(
        (r): Judgement => (r ? { verdict: r.verdict, score: r.score } : { verdict: "error" }),
      );
    }

    function verdictFor(src: string, img: HTMLImageElement): Promise<Judgement> {
      const known = verdicts.get(src);
      if (known) return known;
      const job = classify(sourceKind(src) === "pixels" ? toDataUrl(img) : src);
      verdicts.set(src, job);
      return job;
    }

    function sourceOf(el: Media): string {
      return el instanceof HTMLVideoElement
        ? `${el.currentSrc || el.src}#${el.poster}`
        : el.currentSrc || el.src;
    }

    async function judgeImage(img: HTMLImageElement) {
      const src = sourceOf(img);
      if (judged.get(img) !== src) return;
      const judgement = await verdictFor(src, img);
      // The page may have swapped the source while we waited.
      if (judged.get(img) === src) mark(img, judgement);
    }

    /** One look at a video: its current frame, or its poster before it has one. */
    async function judgeVideo(video: HTMLVideoElement) {
      const src = sourceOf(video);
      if (judged.get(video) !== src || video.getAttribute(ATTR) === "hidden") return;

      const looks: Promise<Judgement>[] = [];
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
        looks.push(classify(toDataUrl(video)));
      }
      if (video.poster && !video.hasAttribute(ATTR)) {
        const poster = video.poster;
        looks.push(
          verdicts.get(poster) ??
            (() => {
              const job = classify(sourceKind(poster) === "url" ? poster : null);
              verdicts.set(poster, job);
              return job;
            })(),
        );
      }
      if (looks.length === 0) return;

      const results = await Promise.all(looks);
      if (judged.get(video) !== src) return;
      // A frame we could not read says nothing either way; a cleared one
      // shows the video. Every later frame is checked again regardless.
      const decisive =
        results.find((r) => r.verdict === "hidden") ?? results.find((r) => r.verdict === "clear");
      if (decisive) mark(video, decisive);
      else if (!video.hasAttribute(ATTR)) mark(video, { verdict: "error" });
    }

    const nearView = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const el = entry.target as Media;
          if (el instanceof HTMLVideoElement) {
            if (entry.isIntersecting) {
              visibleVideos.add(el);
              void judgeVideo(el);
            } else visibleVideos.delete(el);
            continue;
          }
          if (!entry.isIntersecting) continue;
          nearView.unobserve(el);
          void judgeImage(el as HTMLImageElement);
        }
      },
      { rootMargin: LOOKAHEAD },
    );

    function consider(el: Media) {
      const src = sourceOf(el);
      if (judged.get(el) === src) return;

      if (el instanceof HTMLVideoElement) {
        judged.set(el, src);
        // Videos stay observed: they are re-checked as they play.
        nearView.observe(el);
        return;
      }

      // Not decoded yet; its `load` event brings it back here.
      if (!el.complete || el.naturalWidth === 0) return;
      judged.set(el, src);
      const rect = el.getBoundingClientRect();
      if (
        sourceKind(el.currentSrc || el.src) === "skip" ||
        !worthChecking(
          { width: el.naturalWidth, height: el.naturalHeight },
          { width: rect.width, height: rect.height },
        )
      ) {
        mark(el, { verdict: "clear" });
        return;
      }
      // Offscreen images wait until they are about to be seen. A long feed
      // can hold hundreds, and most are never scrolled to.
      nearView.observe(el);
    }

    /** A changed source is a new image, and it is blurred again until judged. */
    function forget(el: Media) {
      judged.delete(el);
      el.removeAttribute(ATTR);
      el.removeAttribute(SCORE_ATTR);
      if (el instanceof HTMLImageElement) nearView.unobserve(el);
    }

    // `load` and the media events do not bubble, but they can be caught on
    // the way down.
    const onLoad = (event: Event) => {
      const target = event.target;
      if (target instanceof HTMLImageElement) consider(target);
      else if (target instanceof HTMLVideoElement) {
        consider(target);
        if (visibleVideos.has(target)) void judgeVideo(target);
      }
    };
    const onEmptied = (event: Event) => {
      if (event.target instanceof HTMLVideoElement) {
        forget(event.target);
        consider(event.target);
      }
    };
    document.addEventListener("load", onLoad, true);
    document.addEventListener("loadeddata", onLoad, true);
    document.addEventListener("playing", onLoad, true);
    document.addEventListener("emptied", onEmptied, true);

    // Runs before the next paint, so a swapped-in image never shows unjudged.
    const mutations = new MutationObserver((records) => {
      for (const record of records) {
        if (record.type === "childList") {
          for (const node of Array.from(record.addedNodes)) {
            if (node instanceof HTMLVideoElement) consider(node);
            else if (node instanceof Element)
              for (const video of Array.from(node.querySelectorAll("video"))) consider(video);
          }
          continue;
        }
        const target = record.target;
        if (target instanceof HTMLImageElement || target instanceof HTMLVideoElement) {
          forget(target);
          consider(target);
        } else if (target instanceof HTMLSourceElement) {
          const media = target.parentElement?.querySelector("img, video");
          if (media instanceof HTMLImageElement || media instanceof HTMLVideoElement) {
            forget(media);
            consider(media);
          }
        }
      }
    });
    mutations.observe(document.documentElement, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["src", "srcset", "poster"],
    });

    // A clip that opens on a harmless frame need not stay harmless.
    const sampler = setInterval(() => {
      for (const video of visibleVideos) {
        if (!video.isConnected) visibleVideos.delete(video);
        else if (!video.paused && video.getAttribute(ATTR) !== "hidden") void judgeVideo(video);
      }
    }, FRAME_INTERVAL_MS);

    // Injected into an open tab, the images are loaded already.
    for (const img of Array.from(document.images)) consider(img);
    for (const video of Array.from(document.querySelectorAll("video"))) consider(video);

    const onMessage = (
      raw: unknown,
      _sender: unknown,
      sendResponse: (reply: MessageReply) => void,
    ) => {
      const type = (raw as Message).type;
      if (type === "image-filter-rejudge") {
        // The classifier keeps its scores, so this is quick: each image is
        // blurred again for as long as a cached lookup takes.
        verdicts.clear();
        for (const el of Array.from(document.querySelectorAll(`[${ATTR}]`))) {
          if (el instanceof HTMLImageElement || el instanceof HTMLVideoElement) {
            forget(el);
            consider(el);
          }
        }
        sendResponse({ type: "image-filter-rejudge", ok: true });
        return false;
      }
      if (type !== "image-filter-off") return false;
      clearInterval(sampler);
      nearView.disconnect();
      mutations.disconnect();
      document.removeEventListener("load", onLoad, true);
      document.removeEventListener("loadeddata", onLoad, true);
      document.removeEventListener("playing", onLoad, true);
      document.removeEventListener("emptied", onEmptied, true);
      browser.runtime.onMessage.removeListener(onMessage);
      style.remove();
      for (const el of Array.from(document.querySelectorAll(`[${ATTR}]`))) {
        el.removeAttribute(ATTR);
        el.removeAttribute(SCORE_ATTR);
      }
      self.__focuserImageFilter = false;
      sendResponse({ type: "image-filter-off", ok: true });
      return false;
    };
    browser.runtime.onMessage.addListener(onMessage);
  },
});
