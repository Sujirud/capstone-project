// Paths of the end-to-end test, from config.toml of the project root like every other
// path, and the files that model/scripts/make_e2e_video.py writes.
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
export const E2E_FOLDER = "e2e"; // in outputs, as in make_e2e_video.py
export const E2E_CONFIG = "e2e.json";
export const API_TMP_FOLDER = "api_tmp"; // the temporary folder of the api the test starts

/** The [paths] table of config.toml: only `name = "value"` lines, which is all it holds. */
export function configPaths(root = ROOT): Record<string, string> {
  const text = readFileSync(resolve(root, "config.toml"), "utf-8");
  const out: Record<string, string> = {};
  let inPaths = false;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (line.startsWith("[")) inPaths = line === "[paths]";
    const m = inPaths ? /^([A-Za-z0-9_]+)\s*=\s*"([^"]*)"/.exec(line) : null;
    if (m) out[m[1]] = m[2];
  }
  return out;
}

export function projectPath(name: string, root = ROOT): string {
  const value = configPaths(root)[name];
  if (value === undefined) throw new Error(`config.toml has no path ${name}`);
  return resolve(root, value);
}

export interface E2EFiles {
  video: string;
  exif_image: string;
  profile: string;
  frame: { width: number; height: number; iod_px: number };
  frontal_image: string;
  profile_image: string;
}

export function e2eFolder(): string {
  return resolve(projectPath("outputs"), E2E_FOLDER);
}

/** The files of make_e2e_video.py, or null when it has not run or a file is gone. */
export function e2eFiles(): E2EFiles | null {
  const path = resolve(e2eFolder(), E2E_CONFIG);
  if (!existsSync(path)) return null;
  const files = JSON.parse(readFileSync(path, "utf-8")) as E2EFiles;
  return [files.video, files.profile].every((p) => existsSync(p)) ? files : null;
}

/** An empty-at-first folder for the temporary files of the api; the test checks it stays empty. */
export function apiTmpFolder(): string {
  const folder = resolve(e2eFolder(), API_TMP_FOLDER);
  mkdirSync(folder, { recursive: true });
  return folder;
}
