// The screenshot tool (screenshots/shots.ts): Thai only, never lab mode, no stub results,
// and a fake camera file that is one plain frame.
import { relative } from "node:path";
import { describe, expect, it } from "vitest";
import { plainPicture } from "../src/demo.ts";
import { projectPath } from "../e2e/paths.ts";
import {
  CAMERA_SIZE,
  type FixtureFile,
  PLAIN,
  SCREENS,
  pageUrl,
  plainCameraY4m,
  resultCases,
  resultFile,
  rgbToYuv,
  screenshotFolder,
} from "../screenshots/shots.ts";
import { FIXTURES } from "./helpers.ts";

const cases = resultCases(Object.values(FIXTURES) as FixtureFile[]);

describe("the screenshots", () => {
  it("open the page in Thai and refuse lab mode", () => {
    expect(new URLSearchParams(pageUrl().slice(2)).get("lang")).toBe("th");
    const demo = new URLSearchParams(pageUrl({ demo: "female_clear", lang: "en" }).slice(2));
    expect([demo.get("demo"), demo.get("lang"), demo.has("lab")]).toEqual(["female_clear", "th", false]);
    expect(() => pageUrl({ lab: "1" })).toThrow(/lab/);
  });

  it("show the results of every fixture but lab mode and stub heads", () => {
    const names = cases.map((c) => c.name);
    expect(names).not.toContain("lab");
    expect(names).not.toContain("stub_heads");
    expect(names).toEqual(expect.arrayContaining(["female_clear", "borderline", "male", "upload_frontal", "no_face_no_profile"]));
    for (const c of cases) expect(FIXTURES[c.name].result.lab).toBe(false);
    expect(cases.find((c) => c.name === "female_clear")!.procedures).toBe(true);
    expect(cases.find((c) => c.name === "male")!.procedures).toBe(false);
  });

  it("have one file name each, in the order of a round", () => {
    const files = [...SCREENS.map((s) => s.file), ...cases.flatMap((c) => [resultFile(c.name), ...(c.procedures ? [resultFile(c.name, true)] : [])])];
    expect(new Set(files).size).toBe(files.length);
    expect(files.every((f) => /^\d\d_[a-z0-9_]+\.png$/.test(f))).toBe(true);
    expect(files).toEqual([...files].sort()); // a folder listed by name shows them in this order
  });

  it("go to outputs of config.toml, outside git", () => {
    expect(relative(projectPath("outputs"), screenshotFolder())).toBe("ui_screenshots");
  });

  it("use the plain colours of the demo pictures", () => {
    const hex = (rgb: readonly number[]) => `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`.toUpperCase();
    for (const name of ["frontal", "profile"] as const) {
      expect(decodeURIComponent(plainPicture(name, 1, 1))).toContain(`fill="${hex(PLAIN[name])}"`);
    }
  });
});

describe("the fake camera file", () => {
  it("converts RGB to full-range YUV", () => {
    expect(rgbToYuv([0, 0, 0])).toEqual([0, 128, 128]);
    expect(rgbToYuv([255, 255, 255])).toEqual([255, 128, 128]);
    expect(rgbToYuv([255, 0, 0])).toEqual([76, 85, 255]);
  });

  it("is one plain 4:2:0 frame of the camera size", () => {
    const file = plainCameraY4m();
    const text = new TextDecoder().decode(file.subarray(0, 80));
    const header = `YUV4MPEG2 W${CAMERA_SIZE.width} H${CAMERA_SIZE.height} F30:1 Ip A1:1 C420jpeg\nFRAME\n`;
    expect(text.startsWith(header)).toBe(true);
    const luma = CAMERA_SIZE.width * CAMERA_SIZE.height;
    expect(file.length).toBe(header.length + luma * 1.5);
    const [y, u, v] = rgbToYuv(PLAIN.frontal);
    const body = file.subarray(header.length);
    expect(new Set(body.subarray(0, luma))).toEqual(new Set([y]));
    expect(new Set(body.subarray(luma, luma + luma / 4))).toEqual(new Set([u]));
    expect(new Set(body.subarray(luma + luma / 4))).toEqual(new Set([v]));
  });
});
