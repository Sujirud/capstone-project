// api.ts with a fake fetch: the table of web.md section 4.2, the timeout, 413 and cancel.
import { afterEach, describe, expect, it, vi } from "vitest";
import { analyze, analyzeWithRetry, buildForm, checkHealth, outcomeOfStatus } from "../src/api.ts";
import { ANALYZE_TIMEOUT_MS } from "../src/constants.ts";
import type { Meta } from "../src/types.ts";
import { JPEG_HEAD, PNG_HEAD, fakeFile, fixture, hangingFetch, jsonResponse, scriptedFetch } from "./helpers.ts";

const META: Meta = { age: 30, sex: "female", frontal_source: "upload", lang: "th" };
const IMAGES = { frontal: fakeFile(JPEG_HEAD), profile: fakeFile(PNG_HEAD, 512, "image/png") };
const RESULT = fixture("female_clear").result;

afterEach(() => {
  vi.useRealTimers();
});

describe("the request", () => {
  it("is multipart with frontal, profile and the meta as JSON (api.md section 2.1)", async () => {
    const form = buildForm(IMAGES, { ...META, lab: true });
    expect([...form.keys()]).toEqual(["frontal", "profile", "meta"]);
    expect(JSON.parse(form.get("meta") as string)).toEqual({ ...META, lab: true });
    expect((form.get("frontal") as File).name).toBe("frontal.jpg");
    expect((form.get("profile") as File).name).toBe("profile.png");
  });

  it("leaves the profile out when there is none", () => {
    const form = buildForm({ frontal: IMAGES.frontal, profile: null }, META);
    expect(form.has("profile")).toBe(false);
  });

  it("is a POST to /analyze", async () => {
    const fetch = scriptedFetch(jsonResponse(200, RESULT));
    await analyze(IMAGES, META, { fetch });
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe("/analyze");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBeInstanceOf(FormData);
  });
});

describe("the answers of web.md section 4.2", () => {
  it.each([
    [200, "ok"],
    [400, "bad_image"],
    [415, "bad_image"],
    [413, "too_large"],
    [422, "invalid"],
    [500, "server"],
    [503, "server"],
    [404, "server"],
  ])("HTTP %i is %s", async (status, kind) => {
    expect(outcomeOfStatus(status)).toBe(kind);
    const body = status === 200 ? RESULT : { error: "x", request_id: RESULT.request_id };
    const outcome = await analyze(IMAGES, META, { fetch: scriptedFetch(jsonResponse(status, body)) });
    expect(outcome.kind).toBe(kind);
    if (outcome.kind === "ok") expect(outcome.result).toEqual(RESULT);
  });

  it("no connection is a server failure", async () => {
    const outcome = await analyze(IMAGES, META, { fetch: scriptedFetch(new TypeError("Failed to fetch")) });
    expect(outcome.kind).toBe("server");
  });

  it("a 200 that is not a result is a server failure", async () => {
    const broken = new Response("<html>", { status: 200 });
    expect((await analyze(IMAGES, META, { fetch: scriptedFetch(broken) })).kind).toBe("server");
    expect((await analyze(IMAGES, META, { fetch: scriptedFetch(jsonResponse(200, { ok: 1 })) })).kind).toBe("server");
  });
});

describe("timeout and cancel", () => {
  it("gives up after 60 seconds", async () => {
    vi.useFakeTimers();
    expect(ANALYZE_TIMEOUT_MS).toBe(60_000);
    const fetch = hangingFetch();
    const pending = analyze(IMAGES, META, { fetch });
    await vi.advanceTimersByTimeAsync(ANALYZE_TIMEOUT_MS - 1);
    let settled = false;
    void pending.then(() => (settled = true));
    await vi.advanceTimersByTimeAsync(0);
    expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect((await pending).kind).toBe("timeout");
    expect(fetch.mock.calls[0][1]?.signal?.aborted).toBe(true);
  });

  it("can be cancelled", async () => {
    const controller = new AbortController();
    const pending = analyze(IMAGES, META, { fetch: hangingFetch(), signal: controller.signal });
    controller.abort();
    expect((await pending).kind).toBe("cancelled");
  });

  it("clears its timer when the answer comes", async () => {
    vi.useFakeTimers();
    await analyze(IMAGES, META, { fetch: scriptedFetch(jsonResponse(200, RESULT)) });
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe("413: shrink and send once more", () => {
  it("sends the smaller images once", async () => {
    const fetch = scriptedFetch(jsonResponse(413, {}), jsonResponse(200, RESULT));
    const smaller = { frontal: fakeFile(JPEG_HEAD, 100), profile: null };
    const shrink = vi.fn(async () => smaller);
    const onShrink = vi.fn();
    const outcome = await analyzeWithRetry(IMAGES, META, shrink, { fetch, onShrink });
    expect(outcome.kind).toBe("ok");
    expect(shrink).toHaveBeenCalledTimes(1);
    expect(onShrink).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(((fetch.mock.calls[1][1]?.body as FormData).get("frontal") as File).size).toBe(smaller.frontal.size);
  });

  it("stops after the second 413", async () => {
    const fetch = scriptedFetch(jsonResponse(413, {}));
    const outcome = await analyzeWithRetry(IMAGES, META, async () => IMAGES, { fetch });
    expect(outcome.kind).toBe("too_large");
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("does not send again when nothing can be made smaller", async () => {
    const fetch = scriptedFetch(jsonResponse(413, {}));
    const outcome = await analyzeWithRetry(IMAGES, META, async () => null, { fetch });
    expect(outcome.kind).toBe("too_large");
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("does not shrink on other answers", async () => {
    const shrink = vi.fn(async () => IMAGES);
    await analyzeWithRetry(IMAGES, META, shrink, { fetch: scriptedFetch(jsonResponse(500, {})) });
    expect(shrink).not.toHaveBeenCalled();
  });
});

describe("GET /health", () => {
  const health = { status: "ok", versions: {}, stub_heads: [], device: "cpu", languages: ["th"] };

  it("is ok when the api answers", async () => {
    const fetch = scriptedFetch(jsonResponse(200, health));
    expect(await checkHealth({ fetch })).toEqual({ ok: true, health });
    expect(fetch.mock.calls[0][0]).toBe("/health");
  });

  it.each([
    ["no connection", new TypeError("Failed to fetch")],
    ["an error status", jsonResponse(500, {})],
    ["a page that is not the api", new Response("<html>", { status: 200 })],
  ])("is not ok with %s", async (_name, answer) => {
    expect(await checkHealth({ fetch: scriptedFetch(answer) })).toEqual({ ok: false });
  });

  it("is not ok when it does not answer in time", async () => {
    vi.useFakeTimers();
    const pending = checkHealth({ fetch: hangingFetch(), timeoutMs: 5000 });
    await vi.advanceTimersByTimeAsync(5000);
    expect(await pending).toEqual({ ok: false });
  });
});
