import { beforeEach, describe, expect, it, vi } from "vitest";
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

/** Start the background worker with `RULES` already loaded. */
async function start() {
  const fetched = vi.fn(async (url: RequestInfo | URL) =>
    String(url).includes("/api/rules")
      ? new Response(JSON.stringify(RULES))
      : new Response("{}"),
  );
  vi.stubGlobal("fetch", fetched);

  // fake-browser has no toolbar button and no script injection.
  const noop = vi.fn(async () => undefined);
  Object.assign(fakeBrowser, {
    action: { setBadgeText: noop, setBadgeBackgroundColor: noop, setTitle: noop },
    scripting: { executeScript: noop },
  });

  background.main();
  await vi.waitFor(() => expect(noop).toHaveBeenCalled());
}

describe("moving between pages without a page load", () => {
  beforeEach(() => fakeBrowser.reset());

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
