import {
  type BrowserName,
  clampIncrement,
  detectBrowser,
  fetchRules,
  isIncognitoAllowed,
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
import type { Verdict } from "@/lib/image-filter";
import type { BlockContext, Message, MessageReply } from "@/lib/messages";
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
const CLASSIFIER_PAGE = "/classifier.html";

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
      await syncImageFilter(next.filter_explicit_images === true);
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

  let imageFilterOn = false;
  let imageFilterSync: Promise<void> = Promise.resolve();

  /**
   * Match the registered content script to what the app asks for.
   *
   * Checked against the browser rather than a flag of ours: a Chrome service
   * worker restarts with its variables reset, but the registration survives.
   */
  function syncImageFilter(on: boolean): Promise<void> {
    imageFilterOn = on;
    imageFilterSync = imageFilterSync.then(async () => {
      try {
        const registered = await browser.scripting.getRegisteredContentScripts({
          ids: [IMAGE_FILTER_ID],
        });
        // Load the model before the first page asks, not when it does.
        if (on) void ensureClassifier().catch(() => undefined);
        if (on === registered.length > 0) return;

        const tabs = await browser.tabs.query({});
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
          await closeClassifier();
        }
      } catch {
        // Retried on the next rules change; the poll keeps coming.
      }
    });
    return imageFilterSync;
  }

  let classifierFrame: HTMLIFrameElement | null = null;
  let openingClassifier: Promise<void> | null = null;

  /**
   * Start the classifier page if it is not running.
   *
   * Chrome's service worker has no DOM, so the page is an offscreen document.
   * Firefox has no offscreen API, but its background is a page and can hold
   * the classifier in a frame.
   */
  function ensureClassifier(): Promise<void> {
    openingClassifier ??= (async () => {
      const url = browser.runtime.getURL(CLASSIFIER_PAGE as never);
      if (browser.offscreen) {
        const open = await browser.runtime.getContexts({
          contextTypes: ["OFFSCREEN_DOCUMENT" as never],
          documentUrls: [url],
        });
        if (open.length > 0) return;
        await browser.offscreen.createDocument({
          url,
          reasons: ["BLOBS" as never],
          justification:
            "Decodes images and runs an on-device classifier to hide explicit images.",
        });
      } else if (!classifierFrame?.isConnected) {
        const frame = document.createElement("iframe");
        frame.src = url;
        const loaded = new Promise((resolve) => frame.addEventListener("load", resolve, { once: true }));
        document.body.appendChild(frame);
        classifierFrame = frame;
        await loaded;
      }
    })().finally(() => {
      openingClassifier = null;
    });
    return openingClassifier;
  }

  async function closeClassifier() {
    classifierFrame?.remove();
    classifierFrame = null;
    if (browser.offscreen) await browser.offscreen.closeDocument().catch(() => undefined);
  }

  async function classifyImage(src: string): Promise<Verdict> {
    // A tab that has not heard the filter went off yet: nothing to hide.
    if (!imageFilterOn) return "clear";
    // The page's listener may not be up the moment the page is, so a refused
    // first message is retried rather than taken as an answer.
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        await ensureClassifier();
        const reply = (await browser.runtime.sendMessage({ type: "classifier-run", src })) as
          | Extract<MessageReply, { type: "classifier-run" }>
          | undefined;
        if (reply?.verdict) return reply.verdict;
      } catch {
        /* not listening yet */
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    return "error";
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
          void classifyImage(message.src).then((verdict) =>
            sendResponse({ type: "classify-image", verdict }),
          );
          return true;
        }
        // Handled by the classifier page and the content scripts.
        case "classifier-run":
        case "image-filter-off":
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
