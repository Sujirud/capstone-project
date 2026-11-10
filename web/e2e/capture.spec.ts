// One round from the home page to the results through Chromium with a fake camera
// (web.md section 8.1). The camera plays a SCUT-FBP5500 test image on a 1920 x 1080 frame;
// the profile is a FEI full profile chosen from the computer. Skips without the files of
// make_e2e_video.py or without the Chromium of Playwright.
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { relative, resolve } from "node:path";
import { type Page, chromium, expect, test } from "@playwright/test";
import { type Finding, audit } from "./audit.ts";
import { apiTmpFolder, e2eFiles, projectPath } from "./paths.ts";

const files = e2eFiles();
const CARDS = ["jaw", "chin", "undereye", "nose_base", "nose_dorsum"]; // the order of model.md (web.md section 5.2)
const CAPTURE_TIMEOUT_MS = 150_000; // the first round loads MediaPipe from the internet
const RESULT_TIMEOUT_MS = 60_000; // ANALYZE_TIMEOUT_MS of the page
// The wasm of MediaPipe writes its information lines to stderr, which the browser shows
// as console errors, e.g. "INFO: Created TensorFlow Lite XNNPACK delegate for CPU."
const MEDIAPIPE_INFO = /^INFO: /;

test.skip(files === null, "run uv run python model/scripts/make_e2e_video.py at the project root first");
test.skip(!existsSync(chromium.executablePath()), "no Chromium of Playwright: run npx playwright install chromium in web/");

/** Every file under folder, as paths relative to it. */
function listFiles(folder: string): string[] {
  return readdirSync(folder, { recursive: true, withFileTypes: true })
    .filter((d) => d.isFile())
    .map((d) => relative(folder, resolve(d.parentPath, d.name)))
    .sort();
}

/** The origins the Content-Security-Policy of the built page allows, besides the api itself. */
function allowedOrigins(): Set<string> {
  const html = readFileSync(resolve(projectPath("web_dist"), "index.html"), "utf-8");
  const policy = /http-equiv="Content-Security-Policy" content="([^"]+)"/.exec(html)?.[1];
  if (!policy) throw new Error("web/dist/index.html has no Content-Security-Policy; build the page again");
  return new Set(policy.match(/https:\/\/[^\s;]+/g) ?? []);
}

/**
 * What the reader of lang would trip on: in English, any Thai outside the language picker
 * (shared/i18n/en.json); in every language, a page wider than the window or text cut off
 * by its box.
 */
async function languageFindings(page: Page, lang: string): Promise<string[]> {
  return page.evaluate((lang) => {
    const out: string[] = [];
    if (lang !== "th") {
      const thai = /[\u0E00-\u0E7F]/;
      const body = document.body.cloneNode(true) as HTMLElement;
      body.querySelector(".lang-picker")?.remove();
      const walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT);
      for (let n = walker.nextNode(); n; n = walker.nextNode()) if (thai.test(n.textContent ?? "")) out.push(`thai: "${(n.textContent ?? "").trim().slice(0, 40)}"`);
      for (const e of body.querySelectorAll("*")) {
        for (const a of ["aria-label", "alt", "title"]) if (thai.test(e.getAttribute(a) ?? "")) out.push(`thai ${a}: "${e.getAttribute(a)}"`);
      }
    }
    const page = document.documentElement;
    if (page.scrollWidth > page.clientWidth + 1) out.push(`page: ${page.scrollWidth}px wide in a ${page.clientWidth}px window`);
    for (const e of document.body.querySelectorAll<HTMLElement>("*")) {
      const s = getComputedStyle(e);
      if (s.display === "inline" || s.display === "none" || s.overflowX === "visible" || e.matches("pre, select, video, canvas, img, svg *")) continue;
      if (e.getBoundingClientRect().width > 1 && e.scrollWidth > e.clientWidth + 1) out.push(`cut: ${e.tagName.toLowerCase()} "${(e.textContent ?? "").trim().slice(0, 40)}"`);
    }
    return out;
  }, lang);
}

for (const lang of ["th", "en"]) {
  test(`a lab round in ${lang}: auto capture, a profile from the computer, then every card`, async ({ page, baseURL }) => {
    const before = listFiles(apiTmpFolder());
    const requests: string[] = [];
    const errors: string[] = [];
    page.on("request", (r) => requests.push(r.url()));
    page.on("console", (m) => {
      if (m.type() === "error" && !MEDIAPIPE_INFO.test(m.text())) errors.push(m.text());
    });
    page.on("pageerror", (e) => errors.push(String(e)));
    // keep every stream the page opens, to see that each track ends
    await page.addInitScript(() => {
      const w = window as unknown as { __e2eStreams: MediaStream[] };
      w.__e2eStreams = [];
      const media = navigator.mediaDevices;
      const open = media.getUserMedia.bind(media);
      media.getUserMedia = async (constraints) => {
        const stream = await open(constraints);
        w.__e2eStreams.push(stream);
        return stream;
      };
    });
    const liveTracks = () =>
      page.evaluate(() => {
        const w = window as unknown as { __e2eStreams: MediaStream[] };
        return w.__e2eStreams.flatMap((s) => s.getTracks()).filter((t) => t.readyState !== "ended").length;
      });

    const findings: Record<string, (Finding | string)[]> = {};
    const check = async (screen: string) => {
      const found: (Finding | string)[] = [...(await audit(page)), ...(await languageFindings(page, lang))];
      if (found.length) findings[screen] = found;
    };

    await page.goto(`/?lab=1&lang=${lang}`);
    await expect(page.locator("html")).toHaveAttribute("lang", lang);
    const home = page.locator('[data-screen="home"]');
    await expect(home).toBeVisible();
    await check("home");
    await home.locator('input[name="age"]').fill("30");
    await home.locator('input[name="sex"][value="female"]').check();
    await home.locator('button[type="submit"]').click();

    const frontal = page.locator('[data-screen="frontal"]');
    await expect(frontal).toBeVisible();
    await check("frontal");
    await frontal.locator('[data-action="camera-open"]').click();
    await expect(frontal.locator("[data-guide-message]")).toBeVisible();
    await check("frontal camera");
    // the page takes the photo by itself once every check passes, then shows the profile
    const profile = page.locator('[data-screen="profile"]');
    await expect(profile).toBeVisible({ timeout: CAPTURE_TIMEOUT_MS });
    expect(await page.evaluate(() => (window as unknown as { __e2eStreams: MediaStream[] }).__e2eStreams.length)).toBeGreaterThan(0);
    expect(await liveTracks()).toBe(0);

    await check("profile");
    await profile.locator('input[type="file"]').setInputFiles(files!.profile);
    const review = page.locator('[data-screen="review"]');
    await expect(review).toBeVisible();
    await check("review");
    await review.locator('[data-action="send"]').click();

    const results = page.locator('[data-screen="results"]');
    await expect(results).toBeVisible({ timeout: RESULT_TIMEOUT_MS });
    expect(await results.locator("[data-card]").evaluateAll((cards) => cards.map((c) => c.getAttribute("data-card")))).toEqual(CARDS);
    // the photo passed the checks of the page, so the api measured every frontal region
    for (const region of ["jaw", "undereye_right", "undereye_left", "nose_base"]) {
      await expect(results.locator(`.card__row[data-region="${region}"]`)).toHaveAttribute("data-status", "assessed");
    }
    await expect(results.locator('[data-lab="compare"]')).toBeVisible();
    await check("results");
    for (const button of await results.locator('[data-action="procedures"]').all()) await button.click();
    await check("results with procedures");
    // The accessibility audit: contrast, text size, button size and names on every screen, and
    // in English no Thai and no text cut off
    expect(findings).toEqual({});

    // web.md section 9: nothing kept in the browser, and requests only to the allowed hosts
    const storage = await page.evaluate(async () => ({
      local: localStorage.length,
      session: sessionStorage.length,
      cookie: document.cookie,
      indexedDB: (await indexedDB.databases()).length,
      caches: (await caches.keys()).length,
      serviceWorkers: (await navigator.serviceWorker.getRegistrations()).length,
    }));
    expect(storage).toEqual({ local: 0, session: 0, cookie: "", indexedDB: 0, caches: 0, serviceWorkers: 0 });
    const allowed = new Set([new URL(baseURL!).origin, ...allowedOrigins()]);
    const outside = requests.filter((u) => /^https?:/.test(u) && !allowed.has(new URL(u).origin));
    expect(outside).toEqual([]);
    expect(errors).toEqual([]);
    expect(await liveTracks()).toBe(0);
    // api.md section 6: the api left no file in its temporary folder
    expect(listFiles(apiTmpFolder()).filter((f) => !before.includes(f))).toEqual([]);
  });
}

// A JPEG stored on its side with EXIF orientation 6 shows upright, and
// the lines are drawn on the upright image the api measured (api.md section 2.1)
test("a frontal photo turned by EXIF shows upright, with the lines on it", async ({ page }) => {
  await page.goto("/");
  const home = page.locator('[data-screen="home"]');
  await home.locator('input[name="age"]').fill("30");
  await home.locator('input[name="sex"][value="female"]').check();
  await home.locator('button[type="submit"]').click();
  await page.locator('[data-screen="frontal"] input[type="file"]').setInputFiles(files!.exif_image);
  const profile = page.locator('[data-screen="profile"]');
  await expect(profile).toBeVisible();
  await profile.locator('[data-action="skip-profile"]').click();
  await page.locator('[data-screen="review"] [data-action="send"]').click();

  const results = page.locator('[data-screen="results"]');
  await expect(results).toBeVisible({ timeout: RESULT_TIMEOUT_MS });
  const { width, height } = files!.frame; // the upright frame, wider than tall
  const layer = results.locator('svg[data-image="frontal"]');
  await expect(layer).toHaveAttribute("viewBox", `0 0 ${width} ${height}`);
  const shown = await layer.evaluate((svg) => {
    const img = svg.parentElement!.querySelector("img")!;
    return { width: img.naturalWidth, height: img.naturalHeight };
  });
  expect(shown).toEqual({ width, height });
  await expect(results.locator('.card__row[data-region="jaw"]')).toHaveAttribute("data-status", "assessed");
  // the jaw line joins the two jaw angles of an upright face: nearly level, near the middle
  const ends = await results.locator('svg[data-image="frontal"] g[data-region="jaw"] circle').evaluateAll((dots) =>
    dots.map((d) => [Number(d.getAttribute("cx")), Number(d.getAttribute("cy"))]),
  );
  expect(ends.length).toBeGreaterThanOrEqual(2);
  for (const [x, y] of ends) {
    expect(Math.abs(x - width / 2)).toBeLessThan(width / 4);
    expect(Math.abs(y - height / 2)).toBeLessThan(height / 3);
  }
});
