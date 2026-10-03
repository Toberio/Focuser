import {
  type BrowserName,
  clampIncrement,
  detectBrowser,
  fetchRules,
  imageVerdict,
  isIncognitoAllowed,
  sendImageFeedback,
  POLL_INTERVAL_MS,
  reportBlocked,
  sendAllowanceTick,
  sendHeartbeat,
  showApp,
} from "@/lib/api";
import {
  buildIndex,
  categoryForHost,
  categoryForKeyword,
  type CategoryIndex,
  EMPTY_INDEX,
} from "@/lib/categories";
import {
  type BlockMatch,
  type CompiledRules,
  compile,
  EMPTY_RULES,
  isInternalUrl,
  match,
  ruleCount,
  type RuleSet,
  trackingKey,
} from "@/lib/rules";
import { type FilterLevel, type Judgement, MAX_CACHE_KEY, VerdictCache } from "@/lib/image-filter";
import type { BlockContext, Message, MessageReply } from "@/lib/messages";
import { isTappable, MAX_IMAGE_BYTES, ResponseTap } from "@/lib/response-tap";
import { SharedActivity } from "@/lib/shared-activity";

/**
 * Blocking works by *replacing* the page, not redirecting it.
 *
 * A redirect to an extension page loses the URL the user typed, pollutes
 * history, and makes "go back" land on the blocked site again. Injecting into
 * the tab keeps the address bar honest about where they tried to go.
 */

const ALLOWANCE_TICK_MS = 30_000;
const INJECTION_DEDUP_MS = 1_500;
const REPORT_DEDUP_MS = 5_000;
const IMAGE_FILTER_ID = "focuser-image-filter";
const IMAGE_FILTER_SCRIPT = "/content-scripts/image-filter.js";

export default defineBackground(() => {
  const browserName: BrowserName = detectBrowser();

  let rules: CompiledRules = EMPTY_RULES;
  let rawRules: RuleSet | null = null;
  let connected = false;
  let index: CategoryIndex = EMPTY_INDEX;

  // Keyed by `tabId:target` — a single navigation fires both onCommitted and
  // onCompleted, and without this the page is built twice.
  const recentInjections = new Map<string, number>();
  const recentReports = new Map<string, number>();
  // Tabs whose current page is the block page. The site's own scripts keep
  // running underneath it, and an SPA rewriting its URL on load must not set
  // off the in-page reload below — that reload brings the block straight
  // back, and the site rewrites again, forever.
  const showingBlock = new Set<number>();
  let lastTickAt = 0;
  const sharedActivity = new SharedActivity();
  let samplingShared = false;

  async function tickShared() {
    // Only an app that sends scopes has a shared allowance running, and only
    // that app knows these ticks. An older one would take each as a normal
    // tick and charge the site's own allowance a second time.
    if (!rules.scopes?.length) {
      sharedActivity.sample(null, Date.now());
      return;
    }
    if (samplingShared) return;
    samplingShared = true;
    try {
      const window = await browser.windows.getLastFocused();
      const idle = await browser.idle.queryState(60);
      const [tab] = await browser.tabs.query({ active: true, lastFocusedWindow: true });
      let url: string | null = null;
      // A video plays with no hand on the mouse, so "no input" still counts.
      // A locked screen does not.
      if (connected && window.focused && idle !== "locked" && tab?.url) {
        const parsed = new URL(tab.url);
        if (
          (parsed.protocol === "https:" || parsed.protocol === "http:") &&
          !match(rules, parsed.hostname, tab.url)
        )
          url = tab.url;
      }
      const report = sharedActivity.sample(url, Date.now());
      if (report)
        await sendAllowanceTick(
          new URL(report.url).hostname,
          report.seconds,
          "shared-activity",
          report.url,
          true,
        );
    } catch {
      sharedActivity.sample(null, Date.now());
    } finally {
      samplingShared = false;
    }
  }

  // ─── Rules ────────────────────────────────────────────────────────

  async function loadIndex() {
    try {
      const url = browser.runtime.getURL("/premade-lists.json" as never);
      index = buildIndex(await (await fetch(url)).json());
    } catch {
      // Categories are decoration; blocking still works without them.
      index = EMPTY_INDEX;
    }
  }

  async function refreshRules() {
    const next = await fetchRules(browserName);
    const wasConnected = connected;
    connected = next !== null;

    if (next && JSON.stringify(next) !== JSON.stringify(rawRules)) {
      rawRules = next;
      rules = compile(next);
      await enforceOnOpenTabs();
      await syncImageFilter(next.image_filter ?? null);
    }
    if (connected !== wasConnected) updateBadge();
    else if (connected) updateBadge();
  }

  /**
   * Tell the app we're alive, and whether we can actually see incognito
   * windows. Checked fresh each time rather than cached once — toggling
   * "Allow in Incognito" reloads the extension anyway, but the check is a
   * cheap local call, not worth caching around a reload edge case.
   */
  async function heartbeat() {
    const incognitoAllowed = await isIncognitoAllowed();
    await sendHeartbeat(browserName, incognitoAllowed);
  }

  function updateBadge() {
    if (!connected) {
      browser.action.setBadgeText({ text: "!" });
      browser.action.setBadgeBackgroundColor({ color: "#f87171" });
      browser.action.setTitle({ title: "Focuser — desktop app not running" });
      return;
    }
    const count = rules.blockEverything ? "∞" : String(ruleCount(rules));
    browser.action.setBadgeText({ text: count === "0" ? "" : count });
    browser.action.setBadgeBackgroundColor({ color: "#8b5cf6" });
    browser.action.setTitle({
      title: rules.blockEverything
        ? "Focuser — blocking the entire internet"
        : `Focuser — ${count} rules active`,
    });
  }

  // ─── Enforcement ──────────────────────────────────────────────────

  /** Everything the block page needs, resolved once in the background. */
  async function buildContext(hit: BlockMatch, hostname: string): Promise<BlockContext> {
    const category =
      hit.reason === "keyword" || hit.reason === "wildcard" || hit.reason === "url-path"
        ? categoryForKeyword(index, hit.target)
        : categoryForHost(index, hostname);

    const key = trackingKey(hit);
    const count = await reportBlocked(hostname, key);
    return { hostname, target: hit.target, reason: hit.reason, category, count };
  }

  function shouldInject(tabId: number, key: string): boolean {
    const now = Date.now();
    const entry = `${tabId}:${key}`;
    const last = recentInjections.get(entry);
    if (last && now - last < INJECTION_DEDUP_MS) return false;

    recentInjections.set(entry, now);
    if (recentInjections.size > 200) {
      for (const [k, at] of recentInjections) {
        if (now - at > INJECTION_DEDUP_MS * 4) recentInjections.delete(k);
      }
    }
    return true;
  }

  async function blockTab(tabId: number, hostname: string, url: string) {
    const hit = match(rules, hostname, url);
    if (!hit) return;
    if (!shouldInject(tabId, trackingKey(hit))) return;
    // Marked before anything is awaited: the site runs while the app is asked
    // for the count, and a URL rewrite in that gap would reload the tab.
    showingBlock.add(tabId);

    const context = await buildContext(hit, hostname);
    try {
      await browser.scripting.executeScript({
        target: { tabId },
        func: (payload: string) => {
          // Handed over on `window` because an injected file cannot take
          // arguments. The block script reads and deletes it immediately.
          (window as unknown as Record<string, unknown>).__focuser = payload;
        },
        args: [JSON.stringify(context)],
      });
      await browser.scripting.executeScript({
        target: { tabId },
        files: ["/block-page.js"],
      });
    } catch {
      // Chrome refuses injection on its own pages and the web store. Nothing
      // to do but leave the tab alone.
      showingBlock.delete(tabId);
    }
  }

  async function enforceOnOpenTabs() {
    const tabs = await browser.tabs.query({});
    for (const tab of tabs) {
      if (!tab.id || !tab.url) continue;
      try {
        const parsed = new URL(tab.url);
        if (isInternalUrl(parsed.protocol)) continue;
        await blockTab(tab.id, parsed.hostname, tab.url);
      } catch {
        /* unparseable tab URL */
      }
    }
  }

  function noteReport(hostname: string): boolean {
    const now = Date.now();
    const last = recentReports.get(hostname);
    if (last && now - last < REPORT_DEDUP_MS) return false;
    recentReports.set(hostname, now);
    return true;
  }

  // ─── Image filter ─────────────────────────────────────────────────

  let imageFilter: FilterLevel | null = null;
  let imageFilterSync: Promise<void> = Promise.resolve();

  /**
   * Match the registered content script to what the app asks for.
   *
   * Checked against the browser rather than a flag of ours: a Chrome service
   * worker restarts with its variables reset, but the registration survives.
   */
  function syncImageFilter(level: FilterLevel | null): Promise<void> {
    const previous = imageFilter;
    imageFilter = level;
    const on = level !== null;
    setTapping(on);
    imageFilterSync = imageFilterSync.then(async () => {
      try {
        const registered = await browser.scripting.getRegisteredContentScripts({
          ids: [IMAGE_FILTER_ID],
        });
        // Load the model before the first page asks, not when it does.
        const tabs = await browser.tabs.query({});
        if (on === registered.length > 0) {
          // Still on, at another level: what was cleared may not be now.
          if (on && previous !== null && previous !== level) {
            verdicts.clear();
            for (const tab of tabs) {
              if (tab.id === undefined) continue;
              browser.tabs
                .sendMessage(tab.id, { type: "image-filter-rejudge" })
                .catch(() => undefined);
            }
          }
          return;
        }

        if (on) {
          await browser.scripting.registerContentScripts([
            {
              id: IMAGE_FILTER_ID,
              js: [IMAGE_FILTER_SCRIPT],
              matches: ["<all_urls>"],
              runAt: "document_start",
              allFrames: true,
              // Off until the app says otherwise after a restart, like every
              // other rule: nothing is enforced on rules from a past session.
              persistAcrossSessions: false,
            },
          ]);
          // Pages already open get it too, or switching it on would do nothing
          // until every tab was reloaded.
          for (const tab of tabs) {
            if (tab.id === undefined) continue;
            browser.scripting
              .executeScript({ target: { tabId: tab.id, allFrames: true }, files: [IMAGE_FILTER_SCRIPT] })
              .catch(() => undefined);
          }
        } else {
          await browser.scripting.unregisterContentScripts({ ids: [IMAGE_FILTER_ID] });
          for (const tab of tabs) {
            if (tab.id === undefined) continue;
            browser.tabs.sendMessage(tab.id, { type: "image-filter-off" }).catch(() => undefined);
          }
        }
      } catch {
        // Retried on the next rules change; the poll keeps coming.
      }
    });
    return imageFilterSync;
  }

  // ─── Firefox: copy images from the page's own downloads ──────────

  interface StreamFilter {
    ondata: ((event: { data: ArrayBuffer }) => void) | null;
    onstop: (() => void) | null;
    onerror: (() => void) | null;
    write(data: ArrayBuffer): void;
    close(): void;
    disconnect(): void;
  }
  type HeadersDetails = {
    requestId: string;
    url: string;
    type: string;
    responseHeaders?: Array<{ name: string; value?: string }>;
  };
  const webRequest = (
    browser as unknown as {
      webRequest?: {
        filterResponseData?: (requestId: string) => StreamFilter;
        onHeadersReceived: {
          addListener(cb: (d: HeadersDetails) => void, filter: object, extra: string[]): void;
          removeListener(cb: (d: HeadersDetails) => void): void;
        };
      };
    }
  ).webRequest;
  const tap = webRequest?.filterResponseData ? new ResponseTap() : null;

  function tapResponse(details: HeadersDetails) {
    if (!tap || !webRequest?.filterResponseData) return;
    const header = (name: string) =>
      details.responseHeaders?.find((h) => h.name.toLowerCase() === name)?.value;
    if (!isTappable(header("content-type"), header("content-length"))) return;

    const filter = webRequest.filterResponseData(details.requestId);
    const chunks: ArrayBuffer[] = [];
    let size = 0;
    tap.started(details.url);
    filter.ondata = ({ data }) => {
      // Straight on to the page first: it must not wait for us.
      filter.write(data);
      size += data.byteLength;
      if (size <= MAX_IMAGE_BYTES) chunks.push(data);
    };
    filter.onstop = () => {
      filter.close();
      if (size > MAX_IMAGE_BYTES) return tap.finished(details.url, null);
      const all = new Uint8Array(size);
      let at = 0;
      for (const chunk of chunks) {
        all.set(new Uint8Array(chunk), at);
        at += chunk.byteLength;
      }
      tap.finished(details.url, all.buffer);
    };
    filter.onerror = () => tap.finished(details.url, null);
  }

  /** Copy image downloads only while the filter is on: otherwise it is all cost. */
  function setTapping(on: boolean) {
    if (!tap || !webRequest) return;
    webRequest.onHeadersReceived.removeListener(tapResponse);
    if (on)
      webRequest.onHeadersReceived.addListener(
        tapResponse,
        { urls: ["<all_urls>"], types: ["image", "imageset"] },
        ["blocking", "responseHeaders"],
      );
  }

  /** Verdicts by level and URL, so a picture seen in many tabs is judged once. */
  const verdicts = new VerdictCache();

  /** The image itself: the page's own download where Firefox shared it, else fetched. */
  async function imageBytes(src: string): Promise<{ bytes: ArrayBuffer; via: string } | null> {
    if (tap && src.startsWith("http")) {
      // The page asks once the image has loaded, so it has nearly always
      // finished streaming.
      const tapped = await tap.take(src, 1_500);
      if (tapped) return { bytes: tapped, via: "tap" };
    }
    try {
      // `force-cache` reuses this profile's copy where there is one. A
      // service worker may fetch any origin it has host permission for.
      const response = await fetch(src, { cache: "force-cache", credentials: "include" });
      if (!response.ok) return null;
      return { bytes: await response.arrayBuffer(), via: src.startsWith("data:") ? "copy" : "fetch" };
    } catch {
      return null;
    }
  }

  async function classifyImage(src: string): Promise<Judgement> {
    // A tab that has not heard the filter went off yet: nothing to hide.
    const level = imageFilter;
    if (level === null) return { verdict: "clear" };
    const key = `${level}|${src}`;
    const cacheable = src.length <= MAX_CACHE_KEY;
    const known = cacheable ? verdicts.get(key) : undefined;
    if (known) return known;

    const started = performance.now();
    const image = await imageBytes(src);
    if (!image) return { verdict: "error" };
    const judgement = await imageVerdict(image.bytes);
    if (judgement.score) {
      const ms = Math.round(performance.now() - started);
      judgement.score = `${judgement.score} · ${ms} ms total · ${image.via}`;
    }
    if (cacheable) verdicts.set(key, judgement);
    return judgement;
  }

  // ─── Allowances ───────────────────────────────────────────────────

  async function tickAllowance(source: string) {
    const now = Date.now();
    if (now - lastTickAt < 3_000) return;
    const elapsed = lastTickAt === 0 ? ALLOWANCE_TICK_MS : now - lastTickAt;
    lastTickAt = now;

    const [active] = await browser.tabs.query({ active: true, lastFocusedWindow: true });
    if (!active?.url) return;

    try {
      const parsed = new URL(active.url);
      if (isInternalUrl(parsed.protocol)) return;
      const hostname = parsed.hostname.toLowerCase().replace(/^www\./, "");
      await sendAllowanceTick(hostname, clampIncrement(elapsed), source, active.url);
    } catch {
      /* unparseable */
    }
  }

  // ─── Wiring ───────────────────────────────────────────────────────

  /** Where a main-frame navigation is going, if it is ours to judge. */
  function destination(details: { frameId: number; url: string }) {
    if (details.frameId !== 0) return null;
    try {
      const parsed = new URL(details.url);
      if (isInternalUrl(parsed.protocol)) return null;
      return { hostname: parsed.hostname, url: details.url };
    } catch {
      return null;
    }
  }

  /** Drop what was injected into a tab's earlier pages. */
  function forgetTab(tabId: number) {
    for (const key of recentInjections.keys()) {
      if (key.startsWith(`${tabId}:`)) recentInjections.delete(key);
    }
    showingBlock.delete(tabId);
  }

  browser.webNavigation.onCommitted.addListener(async (details) => {
    const to = destination(details);
    if (!to) return;
    // This is a new page, and nothing has gone into it yet. Without this, a
    // second visit within the dedup window got no block page: the site stayed
    // hidden behind the early style, but loaded and able to play sound.
    forgetTab(details.tabId);
    await blockTab(details.tabId, to.hostname, to.url);
  });

  // Sites like YouTube move between pages without loading one, so an allowed
  // page would be a door to the rest of the site. Reloading makes it a real
  // navigation: the block then replaces a page that has not started playing.
  browser.webNavigation.onHistoryStateUpdated.addListener((details) => {
    if (showingBlock.has(details.tabId)) return;
    const to = destination(details);
    if (to && match(rules, to.hostname, to.url)) void browser.tabs.reload(details.tabId);
  });

  browser.tabs.onRemoved.addListener(forgetTab);

  browser.runtime.onMessage.addListener(
    (raw: unknown, sender, sendResponse: (reply: MessageReply) => void) => {
      const message = raw as Message;

      switch (message.type) {
        case "close-tab": {
          // The block page cannot close itself: `window.close()` only works on
          // tabs that script opened. The tab id has to come from the sender.
          const tabId = sender.tab?.id;
          if (tabId === undefined) {
            sendResponse({ type: "close-tab", ok: false });
            return false;
          }
          void browser.tabs
            .remove(tabId)
            .then(() => sendResponse({ type: "close-tab", ok: true }))
            .catch(() => sendResponse({ type: "close-tab", ok: false }));
          return true;
        }
        case "open-app": {
          void showApp().then((ok) => sendResponse({ type: "open-app", ok }));
          return true;
        }
        case "check-url": {
          const hit = match(rules, message.hostname, message.url);
          if (hit && noteReport(message.hostname)) {
            void reportBlocked(message.hostname, trackingKey(hit));
          }
          sendResponse({ type: "check-url", blocked: hit !== null });
          return false;
        }
        case "status": {
          sendResponse({
            type: "status",
            connected,
            rules: rawRules,
            ruleCount: ruleCount(rules),
            blockEverything: rules.blockEverything,
          });
          return false;
        }
        case "classify-image": {
          void classifyImage(message.src).then((judgement) =>
            sendResponse({ type: "classify-image", ...judgement }),
          );
          return true;
        }
        case "image-feedback": {
          void imageBytes(message.src)
            .then((image) =>
              image ? sendImageFeedback(image.bytes, message.label, message.src) : false,
            )
            .then((ok) => sendResponse({ type: "image-feedback", ok }));
          return true;
        }
        // Handled by the content scripts.
        case "image-filter-off":
        case "image-filter-rejudge":
          return false;
        case "refresh": {
          void refreshRules().then(() => sendResponse({ type: "refresh", ok: true }));
          return true;
        }
      }
      return false;
    },
  );

  // Open the welcome page on install, and on a feature update. Patch releases
  // stay quiet: a new tab for every bug fix is a nuisance, not news.
  browser.runtime.onInstalled.addListener(({ reason, previousVersion }) => {
    if (reason !== "install" && reason !== "update") return;

    const version = browser.runtime.getManifest().version;
    const minor = (v: string | undefined) => (v ?? "").split(".").slice(0, 2).join(".");
    if (reason === "update" && minor(previousVersion) === minor(version)) return;

    void browser.tabs.create({
      url: browser.runtime.getURL(`/welcome.html?reason=${reason}` as never),
    });
  });

  // The service worker sleeps; an alarm is what wakes it. The interval covers
  // the window before the first alarm fires.
  browser.alarms.create("focuser-tick", { periodInMinutes: 0.5 });
  browser.alarms.onAlarm.addListener((alarm) => {
    if (alarm.name !== "focuser-tick") return;
    void heartbeat();
    void refreshRules();
    void tickAllowance("extension-alarm");
    void tickShared();
  });

  browser.tabs.onActivated.addListener(() => void tickAllowance("tab-switch"));
  browser.tabs.onActivated.addListener(() => void tickShared());
  browser.idle.onStateChanged.addListener(() => void tickShared());
  browser.windows.onFocusChanged.addListener((windowId) => {
    void tickShared();
    if (windowId !== browser.windows.WINDOW_ID_NONE) void tickAllowance("window-focus");
  });

  void (async () => {
    await loadIndex();
    await heartbeat();
    await refreshRules();
    setInterval(() => void refreshRules(), POLL_INTERVAL_MS);
    void tickShared();
    setInterval(() => void tickShared(), POLL_INTERVAL_MS);
  })();
});
