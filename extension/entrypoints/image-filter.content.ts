import {
  afterVideoLook,
  COPY_SIZE,
  afterLooks,
  backgroundUrls,
  GIF_FIRST_BYTES,
  GIF_LATER_BYTES,
  isGif,
  type Judgement,
  MIN_SIDE,
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
 * hidden until its source changes. GIFs, judged as files, are treated like
 * videos in one way: they too stay blurred until their verdict is in.
 *
 * A blurred image or video cannot be right-clicked or dragged, so it cannot
 * be saved, copied or opened in a tab to see it unblurred.
 *
 * Covers `<img>` (including `<picture>` and `srcset`) and `<video>`, in the
 * page and in its open shadow roots. Not yet: CSS background images,
 * `<canvas>` and closed shadow roots.
 */

const ATTR = "data-focuser-image";
/**
 * The same verdicts for an element's CSS background image. Kept apart from
 * `ATTR`: a background is hidden by blacking out the image alone, so text
 * laid over it stays readable, not by blurring the whole element.
 */
const BG_ATTR = "data-focuser-bg";
/** The scores behind a verdict, readable in DevTools, for tuning the thresholds. */
const SCORE_ATTR = "data-focuser-score";
const BLUR = "filter:blur(28px) grayscale(1)!important;clip-path:inset(0)!important";
/** Hidden media, and media still waiting for a verdict that must come first: videos and GIFs. */
const STYLE =
  `:is(img,video):is([${ATTR}="hidden"],[${ATTR}="pending"]){${BLUR}}` +
  `:is([${BG_ATTR}="hidden"],[${BG_ATTR}="pending"]){background-image:none!important;background-color:#000!important}`;
/** Start judging an image this far before it scrolls into view. */
const LOOKAHEAD = "100% 0px";
/** How often a playing, visible video has a frame checked. */
const FRAME_INTERVAL_MS = 2_500;
/**
 * How soon a playing video is looked at again while its first frames are
 * being checked: one clear look short of shown, or shown on its poster's
 * word for its first `EARLY_FRAME_LOOKS` frames.
 */
const SECOND_LOOK_MS = 700;
const EARLY_FRAME_LOOKS = 3;
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
    /** Frames of each video judged so far, for its quick early looks. */
    const frameLooks = new WeakMap<HTMLVideoElement, number>();
    /**
     * Videos that have played through and started over. A GIF-as-video
     * loops every few seconds; once a whole pass was clear, every frame
     * after it is one already judged, and copying them (a canvas read-back
     * on the page's own thread) buys nothing.
     */
    const looped = new WeakSet<HTMLVideoElement>();
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

    /** Whether any of an element is in the viewport now, not just near it. */
    function onScreen(el: Element): boolean {
      const r = el.getBoundingClientRect();
      return r.width > 0 && r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth;
    }

    function classify(payload: string | null, urgent = true, prefix?: number): Promise<Judgement> {
      if (!payload) return Promise.resolve({ verdict: "error" });
      return send({ type: "classify-image", src: payload, urgent, prefix }).then(
        (r): Judgement =>
          r
            ? { verdict: r.verdict, score: r.score, frames: r.frames, complete: r.complete }
            : { verdict: "error" },
      );
    }

    function verdictFor(src: string, img: HTMLImageElement): Promise<Judgement> {
      const known = verdicts.get(src);
      if (known) return known;
      const job = classify(sourceKind(src) === "pixels" ? toDataUrl(img) : src, onScreen(img));
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
      if (isGif(src)) {
        return judgeGif(
          img,
          src,
          () => judged.get(img) === src,
          (j) => mark(img, j),
        );
      }
      const judgement = await verdictFor(src, img);
      // The page may have swapped the source while we waited.
      if (judged.get(img) === src) mark(img, judgement);
    }

    /** One look at the start of a GIF, shared by every copy of it on the page. */
    function gifLook(src: string, prefix: number, urgent: boolean): Promise<Judgement> {
      const key = `${prefix}|${src}`;
      const known = verdicts.get(key);
      if (known) return known;
      const job = classify(src, urgent, prefix);
      verdicts.set(key, job);
      return job;
    }

    /**
     * A GIF is judged like a video: blurred until two frames pass, hidden by
     * any that does not. The first look is a small download of its start,
     * whose first and last frames are two looks, so a clean GIF shows after
     * one round trip. The second, once it shows and at low priority, judges
     * a frame a second through its first seconds and can hide it again.
     */
    async function judgeGif(
      el: HTMLElement,
      src: string,
      current: () => boolean,
      apply: (judgement: Judgement) => void,
    ) {
      let watch = newVideoWatch();
      for (const prefix of [GIF_FIRST_BYTES, GIF_LATER_BYTES]) {
        // Urgent while the user is looking at a blur; not once it shows.
        const urgent = watch.state === "pending" && onScreen(el);
        const look = await gifLook(src, prefix, urgent);
        // The page may have swapped the source while we waited.
        if (!current()) return;
        watch = afterLooks(watch, look);
        if (look.score) el.setAttribute(SCORE_ATTR, look.score);
        if (watch.state !== "pending") {
          apply({ verdict: watch.state === "hidden" ? "hidden" : "clear", score: look.score });
        }
        if (watch.state === "hidden" || look.complete) return;
      }
      // One clear look and one that failed: fail open, as everywhere.
      if (watch.state === "pending") apply({ verdict: "error" });
    }

    /** The background images each element was last judged on, space-separated. */
    const backgrounds = new WeakMap<HTMLElement, string>();

    /**
     * Take in an element's CSS background image, if it has one: judged like
     * an image (a GIF blacked out until two frames pass), and hidden by
     * blacking out the image alone. Read from the computed style, so it
     * finds backgrounds set by class, by a custom property, or inline.
     */
    /**
     * Elements whose background is yet to be read. Reading a computed style
     * straight after the page changed something makes the browser restyle
     * the whole page early, and again for its next change: on a 30,000-node
     * page that was 26 ms per batch of posts added. Read at an idle moment,
     * once the page has settled, it costs next to nothing.
     */
    const backgroundQueue = new Set<HTMLElement>();
    let backgroundFlush = 0;
    /** Elements read per idle callback when the deadline has already passed. */
    const BACKGROUND_CHUNK = 500;

    function considerBackground(el: Element) {
      if (!(el instanceof HTMLElement) || el instanceof HTMLImageElement || el instanceof HTMLVideoElement) return;
      backgroundQueue.add(el);
      if (!backgroundFlush) backgroundFlush = requestIdleCallback(flushBackgrounds, { timeout: 500 });
    }

    function flushBackgrounds(deadline: IdleDeadline) {
      backgroundFlush = 0;
      let done = 0;
      for (const el of backgroundQueue) {
        backgroundQueue.delete(el);
        if (el.isConnected) checkBackground(el);
        done++;
        const out = deadline.didTimeout ? done >= BACKGROUND_CHUNK : deadline.timeRemaining() < 1;
        if (out) break;
      }
      if (backgroundQueue.size > 0) backgroundFlush = requestIdleCallback(flushBackgrounds, { timeout: 500 });
    }

    function checkBackground(el: HTMLElement) {
      // A background this filter blacked out reads as none; look past the
      // rule to what the page set. Restored before the next paint.
      const state = el.getAttribute(BG_ATTR);
      if (state) el.removeAttribute(BG_ATTR);
      const urls = backgroundUrls(getComputedStyle(el).backgroundImage);
      const key = urls.join(" ");
      if (urls.length > 0 && backgrounds.get(el) === key) {
        if (state) el.setAttribute(BG_ATTR, state);
        return;
      }
      el.removeAttribute(SCORE_ATTR);
      if (urls.length === 0) {
        if (backgrounds.delete(el)) nearView.unobserve(el);
        return;
      }
      backgrounds.set(el, key);
      if (urls.some(isGif)) {
        el.setAttribute(BG_ATTR, "pending");
        restyle(el);
      }
      nearView.observe(el);
    }

    function markBackground(el: HTMLElement, judgement: Judgement) {
      el.setAttribute(BG_ATTR, judgement.verdict === "hidden" ? "hidden" : "clear");
      if (judgement.score) el.setAttribute(SCORE_ATTR, judgement.score);
      restyle(el);
    }

    async function judgeBackground(el: HTMLElement) {
      const key = backgrounds.get(el);
      if (!key) return;
      const current = () => backgrounds.get(el) === key;
      // An icon or a texture, as small images are skipped.
      const box = el.getBoundingClientRect();
      if (box.width > 0 && box.height > 0 && Math.min(box.width, box.height) < MIN_SIDE / 2) {
        markBackground(el, { verdict: "clear" });
        return;
      }
      const urls = key.split(" ");
      const gif = urls.find(isGif);
      if (gif) return judgeGif(el, gif, current, (j) => markBackground(el, j));
      const looks = await Promise.all(
        urls.map((url) => {
          const known = verdicts.get(url);
          if (known) return known;
          const job = classify(url, onScreen(el));
          verdicts.set(url, job);
          return job;
        }),
      );
      if (!current()) return;
      markBackground(
        el,
        looks.find((l) => l.verdict === "hidden") ?? looks.find((l) => l.score) ?? { verdict: "error" },
      );
    }

    /** One look at a video: its current frame if not judged yet, and its poster once. */
    async function judgeVideo(video: HTMLVideoElement) {
      const src = sourceOf(video);
      if (judged.get(video) !== src || watches.get(video)?.state === "hidden") return;
      const previous = lastFrame.get(video);
      if (
        previous !== undefined &&
        video.currentTime < previous &&
        (video.loop || previous >= video.duration - FRAME_INTERVAL_MS / 1000)
      ) {
        looped.add(video);
      }
      if (looped.has(video) && watches.get(video)?.state === "clear") return;

      let frameLook: Promise<Judgement> | null = null;
      if (
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        video.videoWidth > 0 &&
        lastFrame.get(video) !== video.currentTime
      ) {
        lastFrame.set(video, video.currentTime);
        frameLook = frameOf(video).then(classify);
      }
      let posterLook: Promise<Judgement> | null = null;
      if (video.poster && !posterLooked.has(video)) {
        posterLooked.add(video);
        const poster = video.poster;
        posterLook =
          verdicts.get(poster) ??
          (() => {
            const job = classify(sourceKind(poster) === "url" ? poster : null);
            verdicts.set(poster, job);
            return job;
          })();
      }
      if (!frameLook && !posterLook) return;

      const [frame, poster] = await Promise.all([frameLook, posterLook]);
      if (judged.get(video) !== src) return;
      let watch = watches.get(video) ?? newVideoWatch();
      // A clear poster vouches for the video: it is shown, as the thumbnail
      // the page shows first, while its frames are checked. Any explicit
      // frame still hides it.
      if (poster) {
        watch =
          poster.verdict === "clear"
            ? afterLooks(watch, { verdict: "clear", complete: true })
            : afterVideoLook(watch, poster.verdict);
      }
      if (frame) {
        watch = afterVideoLook(watch, frame.verdict);
        frameLooks.set(video, (frameLooks.get(video) ?? 0) + 1);
      }
      watches.set(video, watch);
      const latest = [frame, poster].find((r) => r?.verdict === "hidden") ?? [frame, poster].find((r) => r?.score);
      if (latest?.score) video.setAttribute(SCORE_ATTR, latest.score);
      if (watch.state !== "pending") {
        mark(video, { verdict: watch.state === "hidden" ? "hidden" : "clear", score: latest?.score });
      }
      // The first frames come quickly, not at the next sample: one clear
      // look short of shown, or shown on its poster's word.
      const early =
        watch.state === "pending" ? watch.clearLooks > 0 : (frameLooks.get(video) ?? 0) < EARLY_FRAME_LOOKS;
      if (watch.state !== "hidden" && early && !video.paused && onScreen(video)) {
        setTimeout(() => void judgeVideo(video), SECOND_LOOK_MS);
      }
    }

    const nearView = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const el = entry.target;
          if (el instanceof HTMLVideoElement) {
            if (entry.isIntersecting) {
              visibleVideos.add(el);
              void judgeVideo(el);
            } else visibleVideos.delete(el);
            continue;
          }
          if (!entry.isIntersecting) continue;
          nearView.unobserve(el);
          if (el instanceof HTMLImageElement) void judgeImage(el);
          else if (el instanceof HTMLElement) void judgeBackground(el);
        }
      },
      { rootMargin: LOOKAHEAD },
    );

    function consider(el: Media) {
      const src = sourceOf(el);
      if (judged.get(el) === src) return;

      if (el instanceof HTMLVideoElement) {
        judged.set(el, src);
        // Hidden until shown to be clear, unless it has a poster: that is
        // judged like any image, shown until then, and vouches for the
        // video if it passes. A verdict already in place (a re-judge at a
        // new level) stays until its replacement arrives.
        if (!watches.has(el)) watches.set(el, newVideoWatch());
        if (!el.hasAttribute(ATTR) && sourceKind(el.poster) !== "url") el.setAttribute(ATTR, "pending");
        restyle(el);
        // Videos stay observed: they are re-checked as they play.
        nearView.observe(el);
        return;
      }

      // A GIF stays blurred until judged, from before it has loaded: a GIF
      // shows its first frames while it is still downloading.
      if (isGif(src) && !el.hasAttribute(ATTR)) {
        el.setAttribute(ATTR, "pending");
        restyle(el);
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
        frameLooks.delete(el);
        looped.delete(el);
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
      // `style` for backgrounds set inline, as lazy-loaders and cards do.
      attributeFilter: ["src", "srcset", "poster", "style"],
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
    const inlined = new Set<HTMLElement>();

    /**
     * Make sure an element's shadow root, if it is in one, has the style.
     * A component may replace its adopted sheets as it renders, so this runs
     * whenever an element is tagged. Where adopting fails (a browser that
     * keeps a content script's sheets apart from the page's), the element's
     * own inline style stands in.
     */
    function restyle(el: HTMLElement) {
      const root = el.getRootNode();
      if (!(root instanceof ShadowRoot)) return;
      try {
        if (!root.adoptedStyleSheets.includes(sheet)) {
          root.adoptedStyleSheets = [...root.adoptedStyleSheets, sheet];
        }
        return;
      } catch {
        const hiding = (attr: string) => ["hidden", "pending"].includes(el.getAttribute(attr) ?? "");
        if (hiding(BG_ATTR)) {
          if (el.style.getPropertyValue("background-image") !== "none") {
            el.style.setProperty("background-image", "none", "important");
            el.style.setProperty("background-color", "#000", "important");
          }
          inlined.add(el);
        } else if (hiding(ATTR)) {
          el.style.setProperty("filter", "blur(28px) grayscale(1)", "important");
          el.style.setProperty("clip-path", "inset(0)", "important");
          inlined.add(el);
        } else unstyle(el);
      }
    }

    function unstyle(el: HTMLElement) {
      if (inlined.delete(el)) {
        for (const property of ["filter", "clip-path", "background-image", "background-color"]) {
          el.style.removeProperty(property);
        }
      }
    }
    const MEDIA_EVENTS = ["load", "loadeddata", "playing", "emptied"] as const;

    /** Every element the filter has tagged, in the page and its shadow roots. */
    function tagged(attr = ATTR): Element[] {
      const all = Array.from(document.querySelectorAll(`[${attr}]`));
      for (const root of roots) all.push(...Array.from(root.querySelectorAll(`[${attr}]`)));
      return all;
    }

    /** Consider the media in a subtree, and take in any shadow roots it holds. */
    function watchTree(node: Element | Document | ShadowRoot) {
      const inside = Array.from(node.querySelectorAll("*"));
      for (const el of node instanceof Element ? [node, ...inside] : inside) {
        if (el instanceof HTMLImageElement || el instanceof HTMLVideoElement) consider(el);
        else considerBackground(el);
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
    const blurred = `:is(img,video):is([${ATTR}="hidden"],[${ATTR}="pending"])`;
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
              // A component added with its shadow root already attached, and
              // any background images in what was added.
              for (const el of [node, ...Array.from(node.querySelectorAll("*"))]) {
                considerBackground(el);
                if (el.shadowRoot) adopt(el.shadowRoot);
              }
            }
          }
          continue;
        }
        const target = record.target;
        if (record.attributeName === "style") {
          // Only a background can change here; an image's own source cannot.
          if (target instanceof Element) considerBackground(target);
          continue;
        }
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
      if (document.hidden) return;
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
    // Not in a tab nobody can see, and only for videos actually on screen.
    const sampler = setInterval(() => {
      if (document.hidden) return;
      for (const video of visibleVideos) {
        if (!video.isConnected) visibleVideos.delete(video);
        else if (!video.paused && video.getAttribute(ATTR) !== "hidden" && onScreen(video)) {
          void judgeVideo(video);
        }
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
        for (const el of tagged(BG_ATTR)) {
          if (el instanceof HTMLElement) {
            // Judged again from scratch; the old verdict holds meanwhile.
            backgrounds.delete(el);
            const state = el.getAttribute(BG_ATTR);
            checkBackground(el);
            if (state && !el.hasAttribute(BG_ATTR)) el.setAttribute(BG_ATTR, state);
          }
        }
        sendResponse({ type: "image-filter-rejudge", ok: true });
        return false;
      }
      if (type !== "image-filter-off") return false;
      clearInterval(sampler);
      clearInterval(shadowSweep);
      if (backgroundFlush) cancelIdleCallback(backgroundFlush);
      backgroundQueue.clear();
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
      for (const el of [...tagged(), ...tagged(BG_ATTR)]) {
        el.removeAttribute(ATTR);
        el.removeAttribute(BG_ATTR);
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
