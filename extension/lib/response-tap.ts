/**
 * Image bytes copied from the browser's own downloads, so the classifier does
 * not download each image a second time.
 *
 * Firefox only: `webRequest.filterResponseData` lets an extension see a
 * response as it streams to the page. Every chunk is passed straight on, so
 * the page gets its image exactly as fast as without us; we keep a copy.
 * Chrome has no equivalent, and fetches again.
 *
 * Kept small and short-lived. An image is wanted moments after it arrives,
 * when the page shows it, and then never again: the classifier caches its
 * verdict by URL.
 */

/** Larger than this is a photo at print size; it is fetched like on Chrome. */
export const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
/** Total bytes held. Past this the oldest entries go. */
export const MAX_CACHE_BYTES = 96 * 1024 * 1024;
/** How long a copy is kept waiting for the page to ask about it. */
export const ENTRY_TTL_MS = 60_000;

interface Entry {
  bytes: ArrayBuffer;
  at: number;
}

export class ResponseTap {
  private readonly entries = new Map<string, Entry>();
  private readonly waiters = new Map<string, Array<(bytes: ArrayBuffer | null) => void>>();
  private readonly inFlight = new Set<string>();
  private total = 0;

  constructor(private readonly now: () => number = Date.now) {}

  /** A download for `url` has started streaming. */
  started(url: string): void {
    this.inFlight.add(url);
  }

  /** A download finished; `bytes` is null when it failed or was too large. */
  finished(url: string, bytes: ArrayBuffer | null): void {
    this.inFlight.delete(url);
    if (bytes) {
      const old = this.entries.get(url);
      if (old) this.total -= old.bytes.byteLength;
      this.entries.delete(url);
      this.entries.set(url, { bytes, at: this.now() });
      this.total += bytes.byteLength;
      this.evict();
    }
    for (const resolve of this.waiters.get(url) ?? []) resolve(bytes);
    this.waiters.delete(url);
  }

  /**
   * The bytes for `url`, waiting up to `waitMs` if its download is still
   * streaming. Null means fetch it yourself.
   */
  take(url: string, waitMs: number): Promise<ArrayBuffer | null> {
    const entry = this.entries.get(url);
    if (entry && this.now() - entry.at <= ENTRY_TTL_MS) return Promise.resolve(entry.bytes);
    if (!this.inFlight.has(url)) return Promise.resolve(null);
    return new Promise((resolve) => {
      const list = this.waiters.get(url) ?? [];
      list.push(resolve);
      this.waiters.set(url, list);
      setTimeout(() => {
        const pending = this.waiters.get(url);
        if (!pending?.includes(resolve)) return;
        pending.splice(pending.indexOf(resolve), 1);
        resolve(null);
      }, waitMs);
    });
  }

  get size(): number {
    return this.entries.size;
  }

  get bytes(): number {
    return this.total;
  }

  private evict(): void {
    const cutoff = this.now() - ENTRY_TTL_MS;
    for (const [url, entry] of this.entries) {
      if (this.total <= MAX_CACHE_BYTES && entry.at >= cutoff) break;
      this.entries.delete(url);
      this.total -= entry.bytes.byteLength;
    }
  }
}

/** Whether a response is an image worth copying, from its headers. */
export function isTappable(contentType: string | undefined, contentLength: string | undefined): boolean {
  const type = (contentType ?? "").toLowerCase();
  if (!type.startsWith("image/") || type.startsWith("image/svg")) return false;
  const length = Number(contentLength);
  return !(Number.isFinite(length) && length > MAX_IMAGE_BYTES);
}
