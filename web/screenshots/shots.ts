// What the screenshot tool takes (npm run screenshots in web/): every screen of the page in
// Thai and never in lab mode, with no face in any picture.
// The results come from the demo mode (src/demo.ts), which draws the lines of a fixture on a
// plain picture; the fake camera plays a plain frame; the files chosen from the computer
// are plain pictures. The files go to outputs/ui_screenshots/ of config.toml, outside git.
import { resolve } from "node:path";
import { projectPath } from "../e2e/paths.ts";

export const LANG = "th";
export const VIEWPORT = { width: 1440, height: 900 };
export const SCALE = 2; // device pixels per CSS pixel, sharp enough to print
export const FOLDER = "ui_screenshots"; // in outputs
export const CAMERA_FILE = "plain_camera.y4m";
export const CAMERA_SIZE = { width: 1920, height: 1080 }; // a 1080p webcam, so no low-resolution notice
export const HEALTH = {
  status: "ok",
  versions: { api: "0.1.0", model_bundle: "0.2.0", facemeasure: "0.1.0", mediapipe: "1.2.0", torch: "2.15.0" },
  stub_heads: [],
  device: "cpu",
  languages: ["th", "en"],
};

type Rgb = readonly [number, number, number];
/** The colours of the plain pictures of the demo mode (src/demo.ts), so every picture matches. */
export const PLAIN: Record<"frontal" | "profile", Rgb> = { frontal: [0xd8, 0xc3, 0xb0], profile: [0xcd, 0xb8, 0xa6] };

export function screenshotFolder(): string {
  return resolve(projectPath("outputs"), FOLDER);
}

/** A URL of the page in Thai; lab mode is refused. */
export function pageUrl(params: Record<string, string> = {}): string {
  if ("lab" in params) throw new Error("the screenshots never use lab mode");
  return `/?${new URLSearchParams({ ...params, lang: LANG })}`;
}

/** The screens before the results, in the order of a round. */
export const SCREENS = [
  { file: "01_home.png", what: "home" },
  { file: "02_home_api_down.png", what: "home, the api not running" },
  { file: "03_frontal.png", what: "frontal photo, camera off" },
  { file: "04_frontal_camera.png", what: "frontal photo, camera on with the guide" },
  { file: "05_profile.png", what: "profile photo, camera off" },
  { file: "06_profile_camera.png", what: "profile photo, camera on with the guide" },
  { file: "07_review.png", what: "review before sending" },
] as const;

export type ScreenFile = (typeof SCREENS)[number]["file"];

export interface FixtureFile {
  name: string;
  result: { lab: boolean; regions: { status_reasons: string[]; warnings: string[]; procedures: unknown[] }[] };
}

export interface ResultCase {
  name: string;
  procedures: boolean; // a card has the "show procedures" button, so the page is taken opened too
}

/**
 * The fixtures whose results page goes in a report: not lab mode, and no head that is
 * still a stub, since a released bundle has none.
 */
export function resultCases(fixtures: FixtureFile[]): ResultCase[] {
  const stub = (r: FixtureFile["result"]["regions"][number]) => [...r.status_reasons, ...r.warnings].includes("model_not_ready");
  return fixtures
    .filter((f) => !f.result.lab && !f.result.regions.some(stub))
    .map((f) => ({ name: f.name, procedures: f.result.regions.some((r) => r.procedures.length > 0) }))
    .sort((a, b) => (a.name < b.name ? -1 : 1));
}

export function resultFile(name: string, opened = false): string {
  return `08_results_${name}${opened ? "_procedures" : ""}.png`;
}

/** BT.601 full range, the colour space of the C420jpeg tag of the camera file. */
export function rgbToYuv([r, g, b]: Rgb): Rgb {
  const clamp = (v: number) => Math.min(255, Math.max(0, Math.round(v)));
  return [
    clamp(0.299 * r + 0.587 * g + 0.114 * b),
    clamp(128 - 0.168736 * r - 0.331264 * g + 0.5 * b),
    clamp(128 + 0.5 * r - 0.418688 * g - 0.081312 * b),
  ];
}

/** One plain frame as a YUV4MPEG2 file, which Chromium's fake camera plays in a loop. */
export function plainCameraY4m(rgb: Rgb = PLAIN.frontal, size = CAMERA_SIZE): Uint8Array {
  const { width, height } = size;
  const header = new TextEncoder().encode(`YUV4MPEG2 W${width} H${height} F30:1 Ip A1:1 C420jpeg\nFRAME\n`);
  const luma = width * height;
  const chroma = (width / 2) * (height / 2);
  const [y, u, v] = rgbToYuv(rgb);
  const out = new Uint8Array(header.length + luma + 2 * chroma);
  out.set(header);
  out.fill(y, header.length, header.length + luma);
  out.fill(u, header.length + luma, header.length + luma + chroma);
  out.fill(v, header.length + luma + chroma);
  return out;
}
