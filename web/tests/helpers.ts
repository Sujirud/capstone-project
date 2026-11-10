// Shared pieces of the web tests: the language file, the example results written by
// model/scripts/make_web_fixtures.py, and fakes of what lies outside the page.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { vi } from "vitest";
import { I18n, type Texts } from "../src/i18n.ts";
import type { ImageTools } from "../src/images.ts";
import type { LandmarkPoint } from "../src/metrics.ts";
import { resultBody } from "../src/results/view.ts";
import type { AnalyzeResult, ImageName, ImageSource, Meta } from "../src/types.ts";

export const WEB = join(import.meta.dirname, "..");
export const ROOT = join(WEB, "..");
export const SHARED_I18N = join(ROOT, "shared", "i18n");

export function readText(path: string): string {
  return readFileSync(path, "utf-8");
}

export function readJson<T>(path: string): T {
  return JSON.parse(readText(path)) as T;
}

export const TH: Texts = readJson(join(SHARED_I18N, "th.json"));

/** Every language file of shared/i18n, by code. */
export function languageFiles(): Record<string, Texts> {
  const out: Record<string, Texts> = {};
  for (const name of readdirSync(SHARED_I18N)) {
    if (name.endsWith(".json")) out[name.replace(/\.json$/, "")] = readJson(join(SHARED_I18N, name));
  }
  return out;
}

/** Files under a folder whose name ends with one of the extensions. */
export function filesUnder(folder: string, extensions: string[]): string[] {
  const out: string[] = [];
  for (const name of readdirSync(folder)) {
    const path = join(folder, name);
    if (statSync(path).isDirectory()) out.push(...filesUnder(path, extensions));
    else if (extensions.some((e) => name.endsWith(e))) out.push(path);
  }
  return out;
}

/** The Thai text of key with {name} filled, as the page shows it. */
export function th(key: string, params: Record<string, string | number> = {}): string {
  const text = TH[key];
  if (text === undefined) throw new Error(`th.json has no ${key}`);
  return text.replace(/\{(\w+)\}/g, (_, name: string) => String(params[name]));
}

export function thaiI18n(): I18n {
  return new I18n({ th: TH });
}

// ---------------------------------------------------------------- fixtures

export interface Fixture {
  name: string;
  meta: Required<Meta>;
  images: Record<ImageName, { width: number; height: number; source: ImageSource } | null>;
  result: AnalyzeResult;
}

const FIXTURE_FILES = import.meta.glob<Fixture>("./fixtures/*.json", { eager: true, import: "default" });

export const FIXTURES: Record<string, Fixture> = Object.fromEntries(
  Object.values(FIXTURE_FILES)
    .filter((f) => "result" in f)
    .map((f) => [f.name, f]),
);

export interface CodesFile {
  regions: string[];
  frontal_regions: string[];
  profile_regions: string[];
  images: string[];
  sexes: string[];
  frontal_sources: string[];
  measures: string[];
  units: Record<string, string>;
  status_codes: string[];
  frontal_codes: string[];
  profile_codes: string[];
  procedures: string[];
  notes: string[];
  notes_with_procedures: string[];
  withheld: string[];
  grades: string[];
  chin_grades: string[];
}

export const CODES: CodesFile = readJson(join(WEB, "tests", "fixtures", "codes.json"));

export function fixture(name: string): Fixture {
  const f = FIXTURES[name];
  if (!f) throw new Error(`no fixture ${name}`);
  return structuredClone(f);
}

/** The results page body of a fixture, in the document. */
export function renderResult(f: Fixture, sources?: Partial<Record<ImageName, ImageSource>>, i18n: I18n = thaiI18n()) {
  const copy = vi.fn(async () => {});
  const images = {
    frontal: f.images.frontal ? { url: "data:image/png;base64,", ...f.images.frontal, source: sources?.frontal ?? f.images.frontal.source } : null,
    profile: f.images.profile ? { url: "data:image/png;base64,", ...f.images.profile, source: sources?.profile ?? f.images.profile.source } : null,
  };
  const root = document.createElement("div");
  root.append(...resultBody(f.result, { i18n, images, copy }));
  document.body.replaceChildren(root);
  return { root, copy };
}

// ---------------------------------------------------------------- fakes

/** A Response of status with a JSON body. */
export function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

/** A fetch that never answers, but rejects as a browser does when its signal aborts. */
export function hangingFetch() {
  return vi.fn(
    (_input: string, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }),
  );
}

/** A fetch that gives the answers in order, then repeats the last. */
export function scriptedFetch(...answers: (Response | (() => Response) | Error)[]) {
  let i = 0;
  return vi.fn(async (_input: string, _init?: RequestInit) => {
    const answer = answers[Math.min(i, answers.length - 1)];
    i += 1;
    if (answer instanceof Error) throw answer;
    return typeof answer === "function" ? answer() : answer.clone();
  });
}

export const JPEG_HEAD = [0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46];
export const PNG_HEAD = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/** A file that starts with head and is size bytes long. */
export function fakeFile(head: number[], size = 1024, type = "image/jpeg"): File {
  const bytes = new Uint8Array(size);
  bytes.set(head.slice(0, size));
  return new File([bytes], "photo", { type });
}

/** Image tools that need no decoder: every image is 1280 x 1600, and a JPEG of it is 2 kB. */
export function fakeImageTools(size = { width: 1280, height: 1600 }) {
  const tools: ImageTools = {
    size: vi.fn(async () => ({ ...size })),
    encodeJpeg: vi.fn(async () => fakeFile(JPEG_HEAD, 2048)),
  };
  return tools;
}

export function fakeUrlApi() {
  let n = 0;
  return {
    createObjectURL: vi.fn((_blob: Blob) => `blob:test/${++n}`),
    revokeObjectURL: vi.fn((_url: string) => {}),
  };
}

/** Wait for queued promises and microtasks. */
export async function settle(times = 5): Promise<void> {
  for (let i = 0; i < times; i++) await new Promise((r) => setTimeout(r, 0));
}

/** 478 MediaPipe points at the middle of the frame, then the given ones (x, y in [0, 1]). */
export function syntheticLandmarks(points: Record<number, [number, number]>): LandmarkPoint[] {
  const out = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  for (const [n, [x, y]] of Object.entries(points)) out[Number(n)] = { x, y, z: 0 };
  return out;
}

// ---------------------------------------------------------------- camera fakes

export interface FakeTrack {
  kind: "video";
  stop: ReturnType<typeof vi.fn>;
  getSettings: () => MediaTrackSettings;
}

/** A MediaStream with one video track whose settings are given. */
export function fakeStream(settings: MediaTrackSettings = { width: 1920, height: 1080, deviceId: "cam-1" }) {
  const track: FakeTrack = { kind: "video", stop: vi.fn(), getSettings: () => ({ ...settings }) };
  // a MediaStream of the test window, so <video>.srcObject takes it, with the fake track
  const stream = new MediaStream();
  Object.assign(stream, { getVideoTracks: () => [track], getTracks: () => [track] });
  return { stream, track };
}

/** navigator.mediaDevices that gives a new stream with settings on each call and keeps them. */
export function fakeMedia(settings?: MediaTrackSettings, devices: Partial<MediaDeviceInfo>[] = [{ kind: "videoinput", deviceId: "cam-1", label: "Camera" }]) {
  const streams: ReturnType<typeof fakeStream>[] = [];
  const getUserMedia = vi.fn(async (constraints: MediaStreamConstraints) => {
    const video = constraints.video as MediaTrackConstraints;
    const asked = (video.deviceId as ConstrainDOMStringParameters | undefined)?.exact as string | undefined;
    const s = fakeStream({ ...(settings ?? { width: 1920, height: 1080, deviceId: "cam-1" }), ...(asked ? { deviceId: asked } : {}) });
    streams.push(s);
    return s.stream;
  });
  const enumerateDevices = vi.fn(async () => devices as MediaDeviceInfo[]);
  return { getUserMedia, enumerateDevices, streams };
}

/** A canvas that records its size when drawn on, every call on its 2d context, and toBlob. */
export function recordingCanvas() {
  const calls: unknown[][] = [];
  const canvas = {
    width: 300,
    height: 150,
    calls,
    getContext(type: "2d") {
      calls.push(["size", canvas.width, canvas.height]);
      if (type !== "2d") return null;
      return new Proxy({} as CanvasRenderingContext2D, {
        get: (_target, name) => (...args: unknown[]) => {
          calls.push([String(name), ...args]);
        },
        set: (_target, name, value) => {
          calls.push(["set", String(name), value]);
          return true;
        },
      });
    },
    toBlob(callback: (blob: Blob | null) => void, type?: string, quality?: number) {
      calls.push(["toBlob", type, quality]);
      callback(new Blob([new Uint8Array([0xff, 0xd8, 0xff, 0xe0])], { type: type ?? "image/png" }));
    },
  };
  return canvas;
}

/** requestAnimationFrame by hand: tick(t) runs the callbacks waiting at that time. */
export function manualClock() {
  let next = 1;
  const waiting = new Map<number, (t: number) => void>();
  return {
    request: vi.fn((callback: (t: number) => void) => {
      waiting.set(next, callback);
      return next++;
    }),
    cancel: vi.fn((id: number) => {
      waiting.delete(id);
    }),
    tick(t: number) {
      const now = [...waiting.values()];
      waiting.clear();
      for (const callback of now) callback(t);
    },
    get pending() {
      return waiting.size;
    },
  };
}
