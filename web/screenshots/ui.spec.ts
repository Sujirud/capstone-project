// Takes the screenshots of screenshots/shots.ts. Each test opens the page afresh, so a
// failed screen does not stop the others. Skips without the Chromium of Playwright.
import { existsSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import { type Locator, type Page, chromium, expect, test } from "@playwright/test";
import { projectPath } from "../e2e/paths.ts";
import { type FixtureFile, HEALTH, PLAIN, type ScreenFile, CAMERA_SIZE, SCREENS, pageUrl, resultCases, resultFile, screenshotFolder } from "./shots.ts";

const folder = screenshotFolder();
const TH: Record<string, string> = JSON.parse(readFileSync(resolve(projectPath("shared"), "i18n", "th.json"), "utf-8"));
const fixtures: FixtureFile[] = readdirSync(projectPath("web_fixtures"))
  .filter((f) => f.endsWith(".json"))
  .map((f) => JSON.parse(readFileSync(resolve(projectPath("web_fixtures"), f), "utf-8")))
  .filter((f) => f && typeof f === "object" && "result" in f);
const CAMERA_TIMEOUT_MS = 150_000; // MediaPipe comes from the internet the first time

test.skip(!existsSync(chromium.executablePath()), "no Chromium of Playwright: run npx playwright install chromium in web/");

test.beforeAll(() => {
  // the pictures of an earlier run, so none is left from a screen that is gone
  for (const f of readdirSync(folder)) if (f.endsWith(".png")) rmSync(resolve(folder, f));
});

/** Open the page in Thai, with /health answered by the browser and nothing sent to /analyze. */
async function open(page: Page, params: Record<string, string> = {}, health: "ok" | "down" = "ok"): Promise<void> {
  await page.route("**/health", (route) => (health === "ok" ? route.fulfill({ json: HEALTH }) : route.abort()));
  await page.route("**/analyze", (route) => route.abort());
  await page.goto(pageUrl(params));
  await expect(page.locator("html")).toHaveAttribute("lang", "th");
}

async function shoot(page: Page, file: string): Promise<void> {
  await page.evaluate(() => document.fonts.ready.then(() => undefined));
  await page.waitForFunction(() => [...document.images].every((i) => i.complete));
  await page.screenshot({ path: resolve(folder, file), fullPage: true, animations: "disabled", caret: "hide" });
}

/** A plain JPEG made in the page, standing in for a photo chosen from the computer. */
async function plainPhoto(page: Page, name: "frontal" | "profile") {
  const dataUrl = await page.evaluate(
    ({ rgb, size }) => {
      const canvas = document.createElement("canvas");
      canvas.width = size.width;
      canvas.height = size.height;
      const ctx = canvas.getContext("2d")!;
      ctx.fillStyle = `rgb(${rgb.join(",")})`;
      ctx.fillRect(0, 0, size.width, size.height);
      return canvas.toDataURL("image/jpeg", 0.92);
    },
    { rgb: [...PLAIN[name]], size: CAMERA_SIZE },
  );
  return { name: `${name}.jpg`, mimeType: "image/jpeg", buffer: Buffer.from(dataUrl.split(",")[1], "base64") };
}

async function toFrontal(page: Page) {
  await open(page);
  const home = page.locator('[data-screen="home"]');
  await home.locator('input[name="age"]').fill("30");
  await home.locator('input[name="sex"][value="female"]').check();
  await home.locator('button[type="submit"]').click();
  const frontal = page.locator('[data-screen="frontal"]');
  await expect(frontal).toBeVisible();
  return frontal;
}

async function toProfile(page: Page) {
  const frontal = await toFrontal(page);
  await frontal.locator('input[type="file"]').setInputFiles(await plainPhoto(page, "frontal"));
  const profile = page.locator('[data-screen="profile"]');
  await expect(profile).toBeVisible();
  return profile;
}

async function openCamera(screen: Locator) {
  await screen.locator('[data-action="camera-open"]').click();
  await expect(screen.locator("[data-camera-stage][data-live]")).toBeVisible({ timeout: CAMERA_TIMEOUT_MS });
  await expect(screen.locator("[data-guide]")).toBeVisible();
}

/** How to reach each screen of SCREENS; the type makes every screen have its steps. */
const STEPS: Record<ScreenFile, (page: Page) => Promise<void>> = {
  "01_home.png": async (page) => {
    await open(page);
    await expect(page.locator('[data-screen="home"]')).toBeVisible();
  },
  "02_home_api_down.png": async (page) => {
    await open(page, {}, "down");
    await expect(page.getByText(TH["ui.health.down"])).toBeVisible();
  },
  "03_frontal.png": async (page) => {
    await toFrontal(page);
  },
  "04_frontal_camera.png": async (page) => {
    const frontal = await toFrontal(page);
    await openCamera(frontal);
    // MediaPipe is ready once the guide asks for the face it cannot find in the plain frame
    await expect(frontal.locator("[data-guide-message]")).toHaveText(TH["quality.no_face"], { timeout: CAMERA_TIMEOUT_MS });
  },
  "05_profile.png": async (page) => {
    await toProfile(page);
  },
  "06_profile_camera.png": async (page) => {
    await openCamera(await toProfile(page));
  },
  "07_review.png": async (page) => {
    const profile = await toProfile(page);
    await profile.locator('input[type="file"]').setInputFiles(await plainPhoto(page, "profile"));
    await expect(page.locator('[data-screen="review"]')).toBeVisible();
  },
};

for (const screen of SCREENS) {
  test(screen.what, async ({ page }) => {
    await STEPS[screen.file](page);
    await shoot(page, screen.file);
  });
}

for (const c of resultCases(fixtures)) {
  test(`results: ${c.name}`, async ({ page }) => {
    await open(page, { demo: c.name });
    await expect(page.locator('[data-screen="results"]')).toBeVisible();
    await expect(page.locator("[data-lab]")).toHaveCount(0);
    await shoot(page, resultFile(c.name));
    if (!c.procedures) return;
    for (const button of await page.locator('[data-action="procedures"]').all()) await button.click();
    await shoot(page, resultFile(c.name, true));
  });
}
