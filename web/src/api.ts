// Requests to the api on the same machine (web.md sections 4.1 and 4.2).
//
// analyze() sends POST /analyze once and sorts the answer into the cases of the table
// in web.md section 4.2. analyzeWithRetry() adds the 413 case: shrink the images and
// send once more. A request can be cancelled with an AbortSignal and times out after
// ANALYZE_TIMEOUT_MS. Nothing here keeps the images or the result.
import { ANALYZE_PATH, ANALYZE_TIMEOUT_MS, HEALTH_PATH, HEALTH_TIMEOUT_MS } from "./constants.ts";
import type { AnalyzeResult, Health, Meta } from "./types.ts";

export type FetchFn = (input: string, init?: RequestInit) => Promise<Response>;

export interface AnalyzeImages {
  frontal: Blob;
  profile: Blob | null;
}

export type AnalyzeOutcome =
  | { kind: "ok"; result: AnalyzeResult }
  | { kind: "bad_image" } // 400 or 415: the file does not read as an image
  | { kind: "too_large" } // 413
  | { kind: "invalid" } // 422: the web sent something incomplete
  | { kind: "server" } // 500, another status, a broken answer, or no connection
  | { kind: "timeout" }
  | { kind: "cancelled" };

export interface RequestOptions {
  fetch?: FetchFn;
  signal?: AbortSignal;
  timeoutMs?: number;
}

const TIMEOUT = "timeout";

function defaultFetch(): FetchFn {
  return (input, init) => globalThis.fetch(input, init);
}

function fileName(name: string, blob: Blob): string {
  return `${name}.${blob.type === "image/png" ? "png" : "jpg"}`;
}

/** The multipart body of api.md section 2.1. */
export function buildForm(images: AnalyzeImages, meta: Meta): FormData {
  const form = new FormData();
  form.append("frontal", images.frontal, fileName("frontal", images.frontal));
  if (images.profile) form.append("profile", images.profile, fileName("profile", images.profile));
  form.append("meta", JSON.stringify(meta));
  return form;
}

/** A signal that aborts on the caller's signal or after timeoutMs, and tells which one it was. */
function linkedSignal(outer: AbortSignal | undefined, timeoutMs: number) {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort(TIMEOUT);
  }, timeoutMs);
  const onAbort = () => controller.abort(outer?.reason);
  if (outer?.aborted) onAbort();
  else outer?.addEventListener("abort", onAbort, { once: true });
  return {
    signal: controller.signal,
    timedOut: () => timedOut,
    done: () => {
      clearTimeout(timer);
      outer?.removeEventListener("abort", onAbort);
    },
  };
}

export function outcomeOfStatus(status: number): AnalyzeOutcome["kind"] {
  if (status === 200) return "ok";
  if (status === 400 || status === 415) return "bad_image";
  if (status === 413) return "too_large";
  if (status === 422) return "invalid";
  return "server";
}

function looksLikeResult(data: unknown): data is AnalyzeResult {
  const d = data as Partial<AnalyzeResult> | null;
  return !!d && typeof d === "object" && Array.isArray(d.regions) && Array.isArray(d.retake) && typeof d.disclaimer === "string";
}

/** POST /analyze once. */
export async function analyze(images: AnalyzeImages, meta: Meta, options: RequestOptions = {}): Promise<AnalyzeOutcome> {
  const fetchFn = options.fetch ?? defaultFetch();
  const link = linkedSignal(options.signal, options.timeoutMs ?? ANALYZE_TIMEOUT_MS);
  try {
    const response = await fetchFn(ANALYZE_PATH, { method: "POST", body: buildForm(images, meta), signal: link.signal });
    const kind = outcomeOfStatus(response.status);
    if (kind !== "ok") return { kind } as AnalyzeOutcome;
    const data: unknown = await response.json();
    return looksLikeResult(data) ? { kind: "ok", result: data } : { kind: "server" };
  } catch {
    if (link.timedOut()) return { kind: "timeout" };
    if (link.signal.aborted) return { kind: "cancelled" };
    return { kind: "server" };
  } finally {
    link.done();
  }
}

/**
 * POST /analyze; on 413, shrink() the images and send once more (web.md section 4.2).
 * shrink returns null when nothing can be made smaller, which leaves the 413.
 */
export async function analyzeWithRetry(
  images: AnalyzeImages,
  meta: Meta,
  shrink: () => Promise<AnalyzeImages | null>,
  options: RequestOptions & { onShrink?: () => void } = {},
): Promise<AnalyzeOutcome> {
  const first = await analyze(images, meta, options);
  if (first.kind !== "too_large") return first;
  options.onShrink?.();
  const smaller = await shrink();
  if (!smaller || options.signal?.aborted) return smaller ? { kind: "cancelled" } : first;
  return analyze(smaller, meta, options);
}

export type HealthOutcome = { ok: true; health: Health } | { ok: false };

/** GET /health; any failure means the api is not running (web.md section 4.1). */
export async function checkHealth(options: RequestOptions = {}): Promise<HealthOutcome> {
  const fetchFn = options.fetch ?? defaultFetch();
  const link = linkedSignal(options.signal, options.timeoutMs ?? HEALTH_TIMEOUT_MS);
  try {
    const response = await fetchFn(HEALTH_PATH, { signal: link.signal });
    if (!response.ok) return { ok: false };
    const data = (await response.json()) as Health;
    return data && data.status === "ok" ? { ok: true, health: data } : { ok: false };
  } catch {
    return { ok: false };
  } finally {
    link.done();
  }
}
