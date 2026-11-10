// images.ts: the type from the first bytes, and when to re-encode (web.md section 3.4).
import { describe, expect, it } from "vitest";
import { IMAGE_BUDGET_BYTES, MAX_REQUEST_BYTES, SHRINK_MAX_SIDE_PX } from "../src/constants.ts";
import { chooseToShrink, fitWithin, loadUpload, shrinkImage, sniff } from "../src/images.ts";
import { JPEG_HEAD, PNG_HEAD, fakeFile, fakeImageTools } from "./helpers.ts";

const MB = 1024 * 1024;

describe("sniff", () => {
  it("tells JPEG and PNG from their first bytes", () => {
    expect(sniff(new Uint8Array(JPEG_HEAD))).toBe("image/jpeg");
    expect(sniff(new Uint8Array(PNG_HEAD))).toBe("image/png");
  });

  it.each([
    ["HEIC", [0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70, 0x68, 0x65, 0x69, 0x63]],
    ["GIF", [0x47, 0x49, 0x46, 0x38, 0x39, 0x61]],
    ["WebP", [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50]],
    ["a cut PNG signature", PNG_HEAD.slice(0, 4)],
    ["an empty file", []],
  ])("refuses %s", (_name, bytes) => {
    expect(sniff(new Uint8Array(bytes))).toBeNull();
  });
});

describe("loadUpload", () => {
  it("keeps a JPEG and reads its rotated size", async () => {
    const tools = fakeImageTools({ width: 3024, height: 4032 });
    const file = fakeFile(JPEG_HEAD);
    const out = await loadUpload(file, tools);
    expect(out).toEqual({ ok: true, blob: file, type: "image/jpeg", width: 3024, height: 4032 });
  });

  it("goes by the bytes, not by the type the browser reports", async () => {
    const out = await loadUpload(fakeFile(PNG_HEAD, 64, "image/jpeg"), fakeImageTools());
    expect(out.ok && out.type).toBe("image/png");
    expect(out.ok && out.blob.type).toBe("image/png");
    const heic = await loadUpload(fakeFile([0, 0, 0, 0x18, 0x66, 0x74, 0x79, 0x70], 64, "image/jpeg"), fakeImageTools());
    expect(heic).toEqual({ ok: false, reason: "not_jpeg_png" });
  });

  it("reports a file that does not decode", async () => {
    const tools = fakeImageTools();
    tools.size = async () => {
      throw new Error("decode");
    };
    expect(await loadUpload(fakeFile(JPEG_HEAD), tools)).toEqual({ ok: false, reason: "unreadable" });
  });
});

describe("shrinking", () => {
  it("leaves room for the request headers", () => {
    expect(IMAGE_BUDGET_BYTES).toBeLessThan(MAX_REQUEST_BYTES);
    expect(MAX_REQUEST_BYTES).toBe(26_214_400); // shared/quality.json
  });

  it("shrinks nothing when the images fit", () => {
    expect(chooseToShrink([{ name: "frontal", bytes: 8 * MB, shrunk: false }, { name: "profile", bytes: 6 * MB, shrunk: false }])).toEqual([]);
    expect(chooseToShrink([{ name: "frontal", bytes: IMAGE_BUDGET_BYTES, shrunk: false }])).toEqual([]);
  });

  it("shrinks the largest image first, and only as many as needed", () => {
    const big = [
      { name: "frontal" as const, bytes: 9 * MB, shrunk: false },
      { name: "profile" as const, bytes: 20 * MB, shrunk: false },
    ];
    expect(chooseToShrink(big)).toEqual(["profile"]);
    const both = [
      { name: "frontal" as const, bytes: 24 * MB, shrunk: false },
      { name: "profile" as const, bytes: 23 * MB, shrunk: false },
    ];
    expect(chooseToShrink(both)).toEqual(["frontal"]);
    // asked again with the real size after re-encoding, the other one goes too
    expect(chooseToShrink([{ name: "frontal", bytes: 4 * MB, shrunk: true }, both[1]])).toEqual(["profile"]);
  });

  it("does not shrink an image twice", () => {
    const images = [
      { name: "frontal" as const, bytes: 20 * MB, shrunk: true },
      { name: "profile" as const, bytes: 10 * MB, shrunk: false },
    ];
    expect(chooseToShrink(images)).toEqual(["profile"]);
    expect(chooseToShrink([{ name: "frontal", bytes: 30 * MB, shrunk: true }])).toEqual([]);
  });

  it("keeps the long side at most 4032 px and never enlarges", () => {
    expect(SHRINK_MAX_SIDE_PX).toBe(4032);
    expect(fitWithin({ width: 6048, height: 8064 })).toEqual({ width: 3024, height: 4032 });
    expect(fitWithin({ width: 8000, height: 2000 })).toEqual({ width: 4032, height: 1008 });
    expect(fitWithin({ width: 1280, height: 1600 })).toEqual({ width: 1280, height: 1600 });
  });

  it("re-encodes from the rotated size", async () => {
    const tools = fakeImageTools({ width: 4000, height: 6000 });
    const out = await shrinkImage(fakeFile(PNG_HEAD, 4096, "image/png"), tools);
    expect(tools.encodeJpeg).toHaveBeenCalledWith(expect.any(Blob), { width: 2688, height: 4032 });
    expect(out.width).toBe(2688);
    expect(out.height).toBe(4032);
  });
});
