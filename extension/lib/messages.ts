/**
 * The message contract between the extension's parts.
 *
 * Typed both ways so a reply that does not match its request is a compile
 * error rather than an `undefined` at runtime.
 */

import type { Verdict } from "./image-filter";
import type { BlockMatch, RuleSet } from "./rules";

/** Everything the block page renders, resolved in the background. */
export interface BlockContext {
  hostname: string;
  /** What actually matched — a host, or the keyword that caught the URL. */
  target: string;
  reason: BlockMatch["reason"];
  category: string;
  /** How many times this target has been blocked. Drives the message tier. */
  count: number;
}

export type Message =
  | { type: "check-url"; hostname: string; url: string }
  | { type: "status" }
  | { type: "refresh" }
  // Both are sent by the block page, which is a content script and so cannot
  // close its own tab or reach the desktop app directly.
  | { type: "close-tab" }
  | { type: "open-app" }
  // From the image filter to the background. `src` is a URL the classifier
  // can fetch, or a data URL the content script copied out of the page.
  | { type: "classify-image"; src: string }
  // From the background to the classifier page. A separate type so the
  // classifier never answers a content script's request a second time.
  | { type: "classifier-run"; src: string }
  // From the background to open tabs, when the last list using it goes off.
  | { type: "image-filter-off" };

export type MessageReply =
  | { type: "check-url"; blocked: boolean }
  | {
      type: "status";
      connected: boolean;
      rules: RuleSet | null;
      ruleCount: number;
      blockEverything: boolean;
    }
  | { type: "refresh"; ok: boolean }
  | { type: "close-tab"; ok: boolean }
  | { type: "open-app"; ok: boolean }
  | { type: "classify-image"; verdict: Verdict }
  | { type: "classifier-run"; verdict: Verdict }
  | { type: "image-filter-off"; ok: boolean };

/** Send a message and get the reply narrowed to its request type. */
export async function send<T extends Message["type"]>(
  message: Extract<Message, { type: T }>,
): Promise<Extract<MessageReply, { type: T }> | null> {
  try {
    const reply = await browser.runtime.sendMessage(message);
    return (reply ?? null) as Extract<MessageReply, { type: T }> | null;
  } catch {
    // The background worker can be mid-restart; callers treat null as
    // "unknown" rather than "not blocked".
    return null;
  }
}
