// The end-to-end test through Chromium with a fake camera (web.md
// section 8.1). Optional: npm run e2e in web/, after
//
//   npx playwright install chromium
//   npm run build
//   uv run python model/scripts/make_e2e_video.py      (at the project root)
//
// Playwright starts the api itself with `uv run python -m api` on 127.0.0.1:8000, with its
// temporary folder in data/outputs/e2e/api_tmp/, so the test can check that the api
// leaves no file there. Close any api that is already running first.
import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";
import { ROOT, apiTmpFolder, e2eFiles, e2eFolder } from "./e2e/paths.ts";

export const BASE_URL = "http://127.0.0.1:8000";
const tmp = apiTmpFolder();
// Chromium's own camera that plays the video in a loop, without asking for permission
const video = e2eFiles()?.video;
const fakeCamera = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"];
if (video) fakeCamera.push(`--use-file-for-fake-video-capture=${video}`);

export default defineConfig({
  testDir: "e2e",
  // results stay with the other files that show faces, outside git; no screenshot, video or trace
  outputDir: resolve(e2eFolder(), "test-results"),
  timeout: 240_000,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: BASE_URL,
    browserName: "chromium",
    launchOptions: { args: fakeCamera },
    viewport: { width: 1280, height: 900 },
    screenshot: "off",
    video: "off",
    trace: "off",
  },
  webServer: {
    command: "uv run python -m api",
    cwd: ROOT,
    url: `${BASE_URL}/health`,
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe",
    env: { TMPDIR: tmp, TEMP: tmp, TMP: tmp },
  },
});
