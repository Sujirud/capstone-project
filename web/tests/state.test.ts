// state.ts: memory only, and reset() drops everything (web.md section 1.2).
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { Session } from "../src/state.ts";
import { JPEG_HEAD, WEB, fakeFile, fakeUrlApi, filesUnder, fixture, readText } from "./helpers.ts";

const INFO = { width: 1280, height: 1600, source: "upload" as const, shrunk: false };

describe("Session", () => {
  it("reset() revokes every object URL and clears every value of the round", () => {
    const urls = fakeUrlApi();
    const s = new Session({ lang: "th", lab: true }, urls);
    s.age = 30;
    s.sex = "female";
    s.setImage("frontal", fakeFile(JPEG_HEAD), INFO);
    s.setImage("profile", fakeFile(JPEG_HEAD), INFO);
    s.profileSkipped = true;
    s.result = fixture("female_clear").result;
    s.reset();
    expect(urls.revokeObjectURL.mock.calls.map((c) => c[0]).sort()).toEqual(["blob:test/1", "blob:test/2"]);
    expect(s.liveUrls).toEqual([]);
    expect({
      age: s.age,
      sex: s.sex,
      images: s.images,
      profileSkipped: s.profileSkipped,
      result: s.result,
    }).toEqual({ age: null, sex: null, images: { frontal: null, profile: null }, profileSkipped: false, result: null });
    // settings of the page, not of the round
    expect([s.lang, s.lab]).toEqual(["th", true]);
  });

  it("revokes the URL of an image when another replaces it or it is cleared", () => {
    const urls = fakeUrlApi();
    const s = new Session({ lang: "th", lab: false }, urls);
    s.setImage("frontal", fakeFile(JPEG_HEAD), INFO);
    s.setImage("frontal", fakeFile(JPEG_HEAD), INFO);
    expect(urls.revokeObjectURL).toHaveBeenCalledWith("blob:test/1");
    expect(s.liveUrls).toEqual(["blob:test/2"]);
    s.clearImage("frontal");
    expect(urls.revokeObjectURL).toHaveBeenCalledWith("blob:test/2");
    expect(s.images.frontal).toBeNull();
  });

  it("does not revoke URLs it did not make", () => {
    const urls = fakeUrlApi();
    const s = new Session({ lang: "th", lab: false }, urls);
    s.setShownImage("frontal", "data:image/svg+xml,", INFO);
    s.reset();
    expect(urls.revokeObjectURL).not.toHaveBeenCalled();
  });
});

describe("the code", () => {
  it("uses no browser storage (web.md section 1.2)", () => {
    const banned = /\b(localStorage|sessionStorage|indexedDB|caches)\b|document\.cookie/;
    const found = filesUnder(join(WEB, "src"), [".ts"]).filter((p) => banned.test(readText(p)));
    expect(found).toEqual([]);
  });
});
