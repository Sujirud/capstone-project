// The audit of audit.ts finds what it should on a page made to fail, and nothing on one made to pass.
import { existsSync } from "node:fs";
import { chromium, expect, test } from "@playwright/test";
import { audit } from "./audit.ts";

test.skip(!existsSync(chromium.executablePath()), "no Chromium of Playwright: run npx playwright install chromium in web/");

test("the audit finds low contrast, small text, short buttons and missing names", async ({ page }) => {
  await page.setContent(`
    <style>button { font: inherit; }</style>
    <body style="background:#FAF7F2;color:#4B3A31;font-size:16px">
      <p id="ok">readable</p>
      <p style="color:#BFD4E7">pale on marble</p>
      <div style="background:#BFD4E7"><span style="color:#6E5A4F">soft on sky</span></div>
      <p style="font-size:12px">tiny</p>
      <h1 style="font-size:28px;color:#8f7a6c">large text needs 3:1 only</h1>
      <button class="btn" style="height:30px">short</button>
      <button class="btn" style="height:44px">tall</button>
      <button aria-label="icon only"></button>
      <button></button>
      <input type="text">
      <label>age <input type="number"></label>
      <img src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" width="10" height="10">
      <div style="position:relative;width:200px;height:60px">
        <img alt="photo" src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" width="200" height="60">
        <span style="position:absolute;left:0;top:0;color:#FAF7F2">over the photo, not judged</span>
      </div>
      <p style="position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);font-size:8px">visually hidden</p>
    </body>`);
  const found = (await audit(page)).map((f) => `${f.rule}: ${f.where}`);
  expect(found).toEqual([
    'contrast: p "pale on marble"',
    'contrast: span "soft on sky"',
    'font-size: p "tiny"',
    'button-height: button.btn "short"',
    'name: button ""',
    'name: input ""',
    'name: img ""',
  ]);
});
