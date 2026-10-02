import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fakeBrowser } from "wxt/testing/fake-browser";
import background from "@/entrypoints/background";
import type { RuleSet } from "./rules";

const RULES: RuleSet = {
  blocked_domains: ["youtube.com"],
  blocked_keywords: [],
  blocked_wildcards: [],
  blocked_url_paths: [],
  block_entire_internet: false,
  allowed_domains: [],
  allowed_wildcards: [],
  allowed_url_paths: ["youtube.com/@YouTube"],
};

/** Start the background worker with `rules` already loaded. */
async function start(rules: RuleSet = RULES) {
  const fetched = vi.fn(async (url: RequestInfo | URL, _init?: RequestInit) =>
    String(url).includes("/api/rules")
      ? new Response(JSON.stringify(rules))
      : new Response("{}"),
  );
  vi.stubGlobal("fetch", fetched);

  // fake-browser has no toolbar button, script injection or idle detection.
  // "idle" is someone watching a video: the screen is on, the hands are not.
  const noop = vi.fn(async () => undefined);
  const badge = vi.fn(async () => undefined);
  Object.assign(fakeBrowser, {
    action: { setBadgeText: badge, setBadgeBackgroundColor: noop, setTitle: noop },
    scripting: { executeScript: noop },
    idle: { queryState: vi.fn(async () => "idle"), onStateChanged: { addListener: noop } },
  });

  background.main();
  // The badge is set once the first rules fetch has been applied.
  await vi.waitFor(() => expect(badge).toHaveBeenCalled());
  return fetched;
}

beforeEach(() => fakeBrowser.reset());
afterEach(() => vi.restoreAllMocks());

describe("moving between pages without a page load", () => {
  it("reloads the tab when an allowed page leads to a blocked one", async () => {
    const reload = vi.spyOn(fakeBrowser.tabs, "reload").mockResolvedValue();
    await start();

    const move = (url: string, frameId = 0) =>
      fakeBrowser.webNavigation.onHistoryStateUpdated.trigger({
        tabId: 7,
        frameId,
        url,
      } as never);

    // #21: YouTube changes page in place, so the allowed channel page was a
    // door to every video on the site.
    await move("https://www.youtube.com/@YouTube/videos");
    await move("https://www.youtube.com/watch?v=abc", 3);
    expect(reload).not.toHaveBeenCalled();

    await move("https://www.youtube.com/watch?v=abc");
    expect(reload).toHaveBeenCalledWith(7);
  });
});

describe("time on a page under a shared allowance", () => {
  const VIDEO = "https://www.youtube.com/watch?v=abc";
  /** The app sends scopes only while a shared allowance is running. */
  const SHARED: RuleSet = {
    ...RULES,
    scopes: [{ rules: RULES, shared_permits: true, scheduled: true }],
  };

  /** Five seconds on the video, in the front tab of the focused window. */
  async function watch(rules: RuleSet) {
    const clock = vi.spyOn(Date, "now").mockReturnValue(1_000_000);
    vi.spyOn(fakeBrowser.windows, "getLastFocused").mockResolvedValue({ focused: true } as never);
    vi.spyOn(fakeBrowser.tabs, "query").mockResolvedValue([{ id: 7, url: VIDEO }] as never);
    const fetched = await start(rules);

    const ticks = () =>
      fetched.mock.calls
        .filter(([url]) => String(url).includes("/api/allowance-tick"))
        .map(([, init]) => JSON.parse(String(init?.body)));

    clock.mockReturnValue(1_005_000);
    await fakeBrowser.windows.onFocusChanged.trigger(1);
    // The normal allowance tick goes out on the same event, scopes or not.
    await vi.waitFor(() => expect(ticks().some((t) => !t.shared_only)).toBe(true));
    return ticks;
  }

  it("is counted while a video plays with no input", async () => {
    const ticks = await watch(SHARED);

    await vi.waitFor(() =>
      expect(ticks().filter((t) => t.shared_only)).toEqual([
        expect.objectContaining({ url: VIDEO, increment_secs: 5, shared_active: true }),
      ]),
    );
  });

  it("is not reported to an app that sent no scopes", async () => {
    // An app from before shared allowances reads such a tick as a normal one
    // and takes the time off the site's own allowance a second time.
    const ticks = await watch({ ...RULES, blocked_domains: ["reddit.com"] });

    expect(ticks().filter((t) => t.shared_only)).toEqual([]);
  });
});
