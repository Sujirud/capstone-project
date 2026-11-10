// Screenshots of the pages: npm run screenshots in web/, after
//
//   npx playwright install chromium
//
// Playwright starts the Vite dev server (npm run dev) itself, or uses the one that is
// running, because the results come from the demo mode, which only the dev server has.
// The page's /health is answered inside the browser, so no api or model file is needed.
// The pictures go to outputs/ui_screenshots/ of config.toml (screenshots/shots.ts).
import { defineConfig } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { CAMERA_FILE, SCALE, VIEWPORT, plainCameraY4m, screenshotFolder } from "./screenshots/shots.ts";

export const DEV_URL = "http://127.0.0.1:5173";
const folder = screenshotFolder();
mkdirSync(folder, { recursive: true });
const camera = resolve(folder, CAMERA_FILE);
writeFileSync(camera, plainCameraY4m());

export default defineConfig({
  testDir: "screenshots",
  outputDir: resolve(folder, "test-results"),
  timeout: 180_000, // the first camera screen loads MediaPipe from the internet
  workers: 1,
  reporter: "list",
  use: {
    baseURL: DEV_URL,
    browserName: "chromium",
    viewport: VIEWPORT,
    deviceScaleFactor: SCALE,
    // Chromium's own camera playing one plain frame, without asking for permission
    launchOptions: { args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-video-capture=${camera}`] },
    screenshot: "off",
    video: "off",
    trace: "off",
  },
  webServer: {
    command: "npm run dev",
    url: DEV_URL,
    reuseExistingServer: true,
    timeout: 60_000,
    stdout: "ignore",
    stderr: "pipe",
  },
});
