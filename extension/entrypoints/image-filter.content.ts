import {
  afterVideoLook,
  COPY_SIZE,
  type Judgement,
  newVideoWatch,
  sourceKind,
  type VideoWatch,
  worthChecking,
} from "@/lib/image-filter";
import type { Message, MessageReply } from "@/lib/messages";
import { send } from "@/lib/messages";
import { startFeedbackOverlay } from "@/lib/feedback-overlay";
import { deepElementsFromPoint } from "@/lib/shadow";

/** Debug builds put Show/Hide buttons on judged images, for labelling mistakes. */
const FEEDBACK = (import.meta.env as Record<string, unknown>).WXT_IMAGE_FILTER_DEBUG === "true";

/**
 * Blurs the images and videos the classifier finds explicit.
 *
 * Registered by the background only while a list has the filter on, so a page
 * pays nothing for it otherwise.
 *
 * Images show until judged, and are blurred if the verdict says so. Blurring
 * everything first and revealing what passes would never show an explicit
 * image even for a moment, and was the first design. In use it made ordinary
 * sites misbehave: a filter and clip on every image and video, the whole time
 * they waited. A verdict takes tens of milliseconds on a GPU, and a cached
 * one none, so the window it trades away is short.
 *
 * Images are judged once per source. Videos are judged on their poster and
 * then on a frame every few seconds while they play, since a clip that opens
 * on a harmless frame need not stay harmless. Videos, unlike images, start
 * hidden and are shown only after two clear looks (see `VideoWatch`): sites
 * swap a video's source or element as it starts playing, and showing it until
 * judged let it play unfiltered each time. A video found explicit stays
 * hidden until its source changes.
 *
 * A blurred image or video cannot be right-clicked or dragged, so it cannot
 * be saved, copied or opened in a tab to see it unblurred.
 *
 * Covers `<img>` (including `<picture>` and `srcset`) and `<video>`, in the
 * page and in its open shadow roots. Not yet: CSS background images,
 * `<canvas>` and closed shadow roots.
 */

const ATTR = "data-focuser-image";
/** The scores behind a verdict, readable in DevTools, for tuning the thresholds. */
const SCORE_ATTR = "data-focuser-score";
const BLUR = "filter:blur(28px) grayscale(1)!important;clip-path:inset(0)!important";
/** Hidden images, and videos still waiting for their second clear look. */
const STYLE = `:is(img,video)[${ATTR}="hidden"],video[${ATTR}="pending"]{${BLUR}}`;
/** Start judging an image this far before it scrolls into view. */
const LOOKAHEAD = "100% 0px";
/** How often a playing, visible video has a frame checked. */
const FRAME_INTERVAL_MS = 2_500;
/** A playing video one clear look short of shown is looked at again this soon. */
const SECOND_LOOK_MS = 700;
/** How often the page is swept for shadow roots attached after their host was added. */
const SHADOW_SWEEP_MS = 2_500;
/** How long a twin video may take to load or seek before the look counts as failed. */
const TWIN_WAIT_MS = 5_000;
/**
 * Twin videos kept loaded at once, the least recently used released first.
 * Each is a media player: one per video on a long feed doubles the page's
 * players, and Chromium caps how many a page may have.
 */
const MAX_TWINS = 4;

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
    /** Each video's looks so far, for its current source. */
    const watches = new WeakMap<HTMLVideoElement, VideoWatch>();
    /** The frame last judged, by time: events and the sampler can ask about the same one. */
    const lastFrame = new WeakMap<HTMLVideoElement, number>();
    const posterLooked = new WeakSet<HTMLVideoElement>();
    /**
     * Hidden copies of videos whose frames the page cannot read. A video from
     * another origin, played without a `crossorigin` attribute, taints any
     * canvas it is drawn on, even where its server allows this page to read
     * it (Reddit's GIFs-as-MP4 on preview.redd.it do). The twin asks for the
     * same file with CORS, so its frames can be copied; usually from cache.
     */
    const twins = new Map<HTMLVideoElement, HTMLVideoElement>();

    function mark(el: Media, judgement: Judgement) {
      el.setAttribute(ATTR, judgement.verdict === "hidden" ? "hidden" : "clear");
      if (judgement.score) el.setAttribute(SCORE_ATTR, judgement.score);
      restyle(el);
    }

    /** Copy what only this page can read into something the background can send. */
    function toDataUrl(el: Media): string | null {
      try {
        const width = el instanceof HTMLVideoElement ? el.videoWidth : el.naturalWidth;
        const height = el instanceof HTMLVideoElement ? el.videoHeight : el.naturalHeight;
        if (!width || !height) return null;
        // Shape kept: each model crops or squashes it its own way.
        const scale = Math.min(1, COPY_SIZE / Math.max(width, height));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(width * scale));
        canvas.height = Math.max(1, Math.round(height * scale));
        canvas.getContext("2d")?.drawImage(el, 0, 0, canvas.width, canvas.height);
        // Throws on a cross-origin image or video served without CORS.
        return canvas.toDataURL("image/jpeg", 0.9);
      } catch {
        return null;
      }
    }

    /** Resolves on `event`, or with false after `ms`. */
    function eventOrTimeout(el: HTMLVideoElement, event: string, ms: number): Promise<boolean> {
      return new Promise((resolve) => {
        const done = (ok: boolean) => {
          clearTimeout(timer);
          el.removeEventListener(event, onEvent);
          el.removeEventListener("error", onError);
          resolve(ok);
        };
        const onEvent = () => done(true);
        const onError = () => done(false);
        const timer = setTimeout(() => done(false), ms);
        el.addEventListener(event, onEvent, { once: true });
        el.addEventListener("error", onError, { once: true });
      });
    }

    /** The video's current frame, copied: from the video itself, else from its twin. */
    async function frameOf(video: HTMLVideoElement): Promise<string | null> {
      const direct = toDataUrl(video);
      if (direct) return direct;
      const src = video.currentSrc || video.src;
      // Only a plain URL can be fetched again; `blob:` sources (streamed
      // video) are readable already, so failing here means something else.
      if (!src.startsWith("http")) return null;
      let twin = twins.get(video);
      if (twin && twin.src !== src) {
        dropTwin(video);
        twin = undefined;
      }
      if (!twin) {
        twin = document.createElement("video");
        twin.crossOrigin = "anonymous";
        twin.muted = true;
        twin.preload = "auto";
        twin.src = src;
      }
      // Most recently used last, so the first is the one to release.
      twins.delete(video);
      twins.set(video, twin);
      for (const oldest of twins.keys()) {
        if (twins.size <= MAX_TWINS) break;
        dropTwin(oldest);
      }
      if (
        twin.readyState < HTMLMediaElement.HAVE_METADATA &&
        !(await eventOrTimeout(twin, "loadedmetadata", TWIN_WAIT_MS))
      ) {
        return null;
      }
      const at = Math.min(video.currentTime, Math.max(0, twin.duration - 0.05) || 0);
      if (Math.abs(twin.currentTime - at) > 0.01 || twin.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        const seeked = eventOrTimeout(twin, "seeked", TWIN_WAIT_MS);
        twin.currentTime = at;
        if (!(await seeked)) return null;
      }
      return toDataUrl(twin);
    }

    function dropTwin(video: HTMLVideoElement) {
      const twin = twins.get(video);
      if (!twin) return;
      twins.delete(video);
      twin.removeAttribute("src");
      twin.load();
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

    /** One look at a video: its current frame if not judged yet, and its poster once. */
    async function judgeVideo(video: HTMLVideoElement) {
      const src = sourceOf(video);
      if (judged.get(video) !== src || watches.get(video)?.state === "hidden") return;

      const looks: Promise<Judgement>[] = [];
      if (
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        video.videoWidth > 0 &&
        lastFrame.get(video) !== video.currentTime
      ) {
        lastFrame.set(video, video.currentTime);
        looks.push(frameOf(video).then(classify));
      }
      if (video.poster && !posterLooked.has(video)) {
        posterLooked.add(video);
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
      let watch = watches.get(video) ?? newVideoWatch();
      for (const r of results) watch = afterVideoLook(watch, r.verdict);
      watches.set(video, watch);
      const latest = results.find((r) => r.verdict === "hidden") ?? results.find((r) => r.score);
      if (latest?.score) video.setAttribute(SCORE_ATTR, latest.score);
      if (watch.state !== "pending") {
        mark(video, { verdict: watch.state === "hidden" ? "hidden" : "clear", score: latest?.score });
      } else if (watch.clearLooks > 0 && !video.paused) {
        // One clear look in: take the second soon, not at the next sample.
        setTimeout(() => void judgeVideo(video), SECOND_LOOK_MS);
      }
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
        // Hidden until shown to be clear. A verdict already in place (a
        // re-judge at a new level) stays until its replacement arrives.
        if (!watches.has(el)) watches.set(el, newVideoWatch());
        if (!el.hasAttribute(ATTR)) el.setAttribute(ATTR, "pending");
        restyle(el);
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

    /** A changed source is a new image, and the old verdict does not apply to it. */
    function forget(el: Media) {
      judged.delete(el);
      el.removeAttribute(ATTR);
      el.removeAttribute(SCORE_ATTR);
      if (el instanceof HTMLImageElement) nearView.unobserve(el);
      else {
        watches.delete(el);
        lastFrame.delete(el);
        posterLooked.delete(el);
        dropTwin(el);
      }
    }

    // `load` and the media events do not bubble, but they can be caught on
    // the way down.
    const OBSERVED = {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["src", "srcset", "poster"],
    };
    /** Open shadow roots found so far, each watched like the document. */
    const roots = new Set<ShadowRoot>();
    /**
     * The style, for shadow roots. Adopted, never added as a `<style>` child:
     * a component server-rendered with its shadow root (Reddit's are) checks
     * that root's children against its own template as it hydrates, and a
     * stray node can make it fail and reload, over and over.
     */
    const sheet = new CSSStyleSheet();
    sheet.replaceSync(STYLE);
    /** Elements styled inline because their root would not adopt the sheet. */
    const inlined = new Set<Media>();

    /**
     * Make sure an element's shadow root, if it is in one, has the style.
     * A component may replace its adopted sheets as it renders, so this runs
     * whenever an element is tagged. Where adopting fails (a browser that
     * keeps a content script's sheets apart from the page's), the element's
     * own inline style stands in.
     */
    function restyle(el: Media) {
      const root = el.getRootNode();
      if (!(root instanceof ShadowRoot)) return;
      try {
        if (!root.adoptedStyleSheets.includes(sheet)) {
          root.adoptedStyleSheets = [...root.adoptedStyleSheets, sheet];
        }
        return;
      } catch {
        const hide = ["hidden", "pending"].includes(el.getAttribute(ATTR) ?? "");
        if (hide) {
          el.style.setProperty("filter", "blur(28px) grayscale(1)", "important");
          el.style.setProperty("clip-path", "inset(0)", "important");
          inlined.add(el);
        } else unstyle(el);
      }
    }

    function unstyle(el: Media) {
      if (inlined.delete(el)) {
        el.style.removeProperty("filter");
        el.style.removeProperty("clip-path");
      }
    }
    const MEDIA_EVENTS = ["load", "loadeddata", "playing", "emptied"] as const;

    /** Every element the filter has tagged, in the page and its shadow roots. */
    function tagged(): Element[] {
      const all = Array.from(document.querySelectorAll(`[${ATTR}]`));
      for (const root of roots) all.push(...Array.from(root.querySelectorAll(`[${ATTR}]`)));
      return all;
    }

    /** Consider the media in a subtree, and take in any shadow roots it holds. */
    function watchTree(node: Element | Document | ShadowRoot) {
      const inside = Array.from(node.querySelectorAll("*"));
      for (const el of node instanceof Element ? [node, ...inside] : inside) {
        if (el instanceof HTMLImageElement || el instanceof HTMLVideoElement) consider(el);
        if (el.shadowRoot) adopt(el.shadowRoot);
      }
    }

    /**
     * Watch a shadow root as the document is watched. Neither the document's
     * style nor its media events (which do not cross a shadow boundary)
     * reach inside, so each root gets both.
     */
    function adopt(root: ShadowRoot) {
      // Not before its component is defined: until then a server-rendered
      // root is waiting to be hydrated, and is left exactly as it came. The
      // sweep comes back for it.
      if (roots.has(root) || !root.host.matches(":defined")) return;
      roots.add(root);
      mutations.observe(root, OBSERVED);
      for (const type of MEDIA_EVENTS) root.addEventListener(type, onMediaEvent, true);
      watchTree(root);
    }

    function onMediaEvent(event: Event) {
      if (event.type === "emptied") onEmptied(event);
      else onLoad(event);
    }

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

    // The browser's own menu would offer to save, copy or open the image,
    // and dragging it out saves it too. Sites often lay a transparent link or
    // overlay over their images, so look at everything under the pointer,
    // not just the event's target.
    const blurred = `:is(img,video)[${ATTR}="hidden"],video[${ATTR}="pending"]`;
    const onSaveAttempt = (event: MouseEvent) => {
      const target = event.target;
      const hit =
        (target instanceof Element && target.matches(blurred)) ||
        deepElementsFromPoint(event.clientX, event.clientY).some((el) => el.matches(blurred));
      if (hit) event.preventDefault();
    };
    document.addEventListener("contextmenu", onSaveAttempt, true);
    document.addEventListener("dragstart", onSaveAttempt, true);

    // Runs before the next paint, so an image swapped in for a hidden one is
    // not blurred for its predecessor's sake, nor shown with its blur kept.
    const mutations = new MutationObserver((records) => {
      for (const record of records) {
        if (record.type === "childList") {
          for (const node of Array.from(record.addedNodes)) {
            if (node instanceof HTMLVideoElement) consider(node);
            else if (node instanceof Element) {
              for (const video of Array.from(node.querySelectorAll("video"))) consider(video);
              // A component added with its shadow root already attached.
              for (const el of [node, ...Array.from(node.querySelectorAll("*"))]) {
                if (el.shadowRoot) adopt(el.shadowRoot);
              }
            }
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
    mutations.observe(document.documentElement, OBSERVED);

    // A component defined after it was added attaches its shadow root, or
    // hydrates the one it came with, later, and no mutation says so. Sweep
    // for those, and put back any style a component's own rendering dropped.
    const shadowSweep = setInterval(() => {
      for (const root of [document, ...roots]) {
        for (const el of Array.from(root.querySelectorAll("*"))) {
          if (el.shadowRoot) adopt(el.shadowRoot);
        }
      }
      for (const root of roots) {
        if (!root.host.isConnected) roots.delete(root);
        else {
          for (const el of Array.from(root.querySelectorAll(`[${ATTR}]`))) restyle(el as Media);
        }
      }
    }, SHADOW_SWEEP_MS);

    // A clip that opens on a harmless frame need not stay harmless.
    const sampler = setInterval(() => {
      for (const video of visibleVideos) {
        if (!video.isConnected) visibleVideos.delete(video);
        else if (!video.paused && video.getAttribute(ATTR) !== "hidden") void judgeVideo(video);
      }
    }, FRAME_INTERVAL_MS);

    const stopFeedback = FEEDBACK
      ? startFeedbackOverlay(ATTR, async (el, label) => {
          const src = el instanceof HTMLVideoElement ? "" : el.currentSrc || el.src;
          // What the user saw: the URL where the background can get it,
          // otherwise a copy through a canvas (blob: images, video frames).
          const payload = src && sourceKind(src) === "url" ? src : toDataUrl(el);
          if (!payload) return false;
          const reply = await send({ type: "image-feedback", src: payload, label });
          // Do what the user asked straight away, logged or not, to every
          // copy of the image: a pin is often several stacked <img>s.
          const verdict = label === "show" ? "clear" : "hidden";
          el.setAttribute(ATTR, verdict);
          if (src) {
            for (const other of tagged()) {
              const media = other as HTMLImageElement;
              if ((media.currentSrc || media.src) === src) other.setAttribute(ATTR, verdict);
            }
          }
          return reply?.ok === true;
        })
      : () => {};

    // Injected into an open tab, the images are loaded already.
    watchTree(document);

    const onMessage = (
      raw: unknown,
      _sender: unknown,
      sendResponse: (reply: MessageReply) => void,
    ) => {
      const type = (raw as Message).type;
      if (type === "image-filter-rejudge") {
        // Each verdict stays in place until its replacement arrives, so
        // nothing hidden is shown while the new level is applied.
        verdicts.clear();
        for (const el of tagged()) {
          if (el instanceof HTMLImageElement || el instanceof HTMLVideoElement) {
            judged.delete(el);
            consider(el);
          }
        }
        sendResponse({ type: "image-filter-rejudge", ok: true });
        return false;
      }
      if (type !== "image-filter-off") return false;
      clearInterval(sampler);
      clearInterval(shadowSweep);
      for (const video of [...twins.keys()]) dropTwin(video);
      stopFeedback();
      nearView.disconnect();
      mutations.disconnect();
      document.removeEventListener("load", onLoad, true);
      document.removeEventListener("loadeddata", onLoad, true);
      document.removeEventListener("playing", onLoad, true);
      document.removeEventListener("emptied", onEmptied, true);
      document.removeEventListener("contextmenu", onSaveAttempt, true);
      document.removeEventListener("dragstart", onSaveAttempt, true);
      browser.runtime.onMessage.removeListener(onMessage);
      style.remove();
      for (const el of tagged()) {
        el.removeAttribute(ATTR);
        el.removeAttribute(SCORE_ATTR);
      }
      for (const root of roots) {
        try {
          root.adoptedStyleSheets = root.adoptedStyleSheets.filter((s) => s !== sheet);
        } catch {
          // Inline styles stood in for it, removed below.
        }
        for (const type of MEDIA_EVENTS) root.removeEventListener(type, onMediaEvent, true);
      }
      roots.clear();
      for (const el of inlined) unstyle(el);
      self.__focuserImageFilter = false;
      sendResponse({ type: "image-filter-off", ok: true });
      return false;
    };
    browser.runtime.onMessage.addListener(onMessage);
  },
});
