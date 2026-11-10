// Language files and i18n.ts (web.md sections 8.4 and 9).
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BUNDLED_LANGUAGES, DEFAULT_LANG, I18n, applyDocumentLang, languagesFromFiles, placeholders } from "../src/i18n.ts";
import * as types from "../src/types.ts";
import { CODES, TH, WEB, filesUnder, languageFiles, readText } from "./helpers.ts";

const THAI = /[\u0E00-\u0E7F]/;
// "abnormal" and "not beautiful" in Thai, written as code points so this file has no Thai letters
const RULED_OUT = ["\u0E1C\u0E34\u0E14\u0E1B\u0E01\u0E15\u0E34", "\u0E44\u0E21\u0E48\u0E2A\u0E27\u0E22"];

afterEach(() => {
  vi.restoreAllMocks();
});

/** Every ui.* key written in the code as a string. */
function uiKeysInCode(): Set<string> {
  const keys = new Set<string>();
  for (const path of filesUnder(join(WEB, "src"), [".ts"])) {
    for (const m of readText(path).matchAll(/["'`](ui\.[a-z0-9_.]+)["'`]/g)) keys.add(m[1]);
  }
  return keys;
}

describe("the texts in th.json", () => {
  it("has a text for every code of codes.py", () => {
    const keys = [
      ...CODES.status_codes.map((c) => `quality.${c}`),
      ...CODES.withheld.map((c) => `withheld.${c}`),
      ...CODES.notes.map((c) => `note.${c}`),
      ...CODES.procedures.flatMap((c) => [`procedure.${c}.name`, `procedure.${c}.when`]),
      ...[...CODES.regions, "undereye"].flatMap((r) => [`region.${r}.name`, `region.${r}.subtitle`]),
      ...["jaw", "chin", "undereye", "nose_base", "nose_dorsum"].flatMap((r) => [`region.${r}.what`, `region.${r}.how`]),
      ...CODES.measures.flatMap((m) => [`measure.${m}.name`, `measure.${m}.subtitle`]),
      ...CODES.grades.map((g) => `grade.default.${g}`),
      ...CODES.chin_grades.map((g) => `grade.chin.${g}`),
      ...CODES.sexes.map((s) => `sex.${s}`),
      ...["clear", "borderline", "values_only", "not_assessable", "upload"].map((l) => `level.${l}`),
      "grade.between",
      "grade.between_wide",
      "measure.method.model",
      "disclaimer",
    ];
    expect(keys.filter((k) => !TH[k])).toEqual([]);
  });

  it("has every ui.* key the code uses, and no ui.* key the code does not use", () => {
    const used = uiKeysInCode();
    const inFile = new Set(Object.keys(TH).filter((k) => k.startsWith("ui.")));
    expect([...used].filter((k) => !inFile.has(k)).sort()).toEqual([]);
    expect([...inFile].filter((k) => !used.has(k)).sort()).toEqual([]);
  });

  it("uses no word the documents rule out (model.md section 1.3)", () => {
    for (const [key, text] of Object.entries(TH)) {
      for (const word of RULED_OUT) expect(text.includes(word), key).toBe(false);
    }
  });

  it("keeps the code lists of types.ts the same as codes.py", () => {
    expect([...types.REGIONS]).toEqual(CODES.regions);
    expect([...types.FRONTAL_REGIONS]).toEqual(CODES.frontal_regions);
    expect([...types.PROFILE_REGIONS]).toEqual(CODES.profile_regions);
    expect([...types.IMAGES]).toEqual(CODES.images);
    expect([...types.SEXES]).toEqual(CODES.sexes);
    expect([...types.FRONTAL_SOURCES]).toEqual(CODES.frontal_sources);
    expect([...types.MEASURES]).toEqual(CODES.measures);
    expect([...types.FRONTAL_CODES]).toEqual(CODES.frontal_codes);
    expect([...types.PROFILE_CODES]).toEqual(CODES.profile_codes);
    expect([...types.STATUS_CODES]).toEqual(CODES.status_codes);
    expect([...types.PROCEDURES]).toEqual(CODES.procedures);
    expect([...types.NOTES]).toEqual(CODES.notes);
    expect([...types.WITHHELD]).toEqual(CODES.withheld);
    expect([...types.GRADES]).toEqual(CODES.grades);
    expect([...types.CHIN_GRADES]).toEqual(CODES.chin_grades);
  });
});

describe("other language files", () => {
  it("have no key that Thai lacks and the same placeholders", () => {
    for (const [lang, texts] of Object.entries(languageFiles())) {
      if (lang === DEFAULT_LANG) continue;
      for (const [key, text] of Object.entries(texts)) {
        expect(key in TH, `${lang}.json: ${key}`).toBe(true);
        expect(placeholders(text), `${lang}.json: ${key}`).toEqual(placeholders(TH[key]));
      }
    }
  });

  it("include English with every key of Thai, written in English, and empty subtitles", () => {
    const en = languageFiles().en;
    expect(Object.keys(en)).toEqual(Object.keys(TH));
    // The label of the picker keeps the Thai word, so a Thai reader finds the way back
    expect(Object.entries(en).filter(([k, t]) => k !== "ui.lang.label" && THAI.test(t)).map(([k]) => k)).toEqual([]);
    const subtitles = Object.keys(en).filter((k) => k.endsWith(".subtitle"));
    expect(subtitles.length).toBeGreaterThan(0);
    for (const k of subtitles) expect(en[k], k).toBe(""); // the names are English already (cards.ts named())
    for (const [k, t] of Object.entries(en)) if (!k.endsWith(".subtitle")) expect(t.trim(), k).not.toBe("");
  });

  it("are all bundled, so a new file needs only a new build", () => {
    expect(Object.keys(BUNDLED_LANGUAGES).sort()).toEqual(Object.keys(languageFiles()).sort());
    expect(BUNDLED_LANGUAGES.th).toEqual(TH);
  });
});

describe("i18n", () => {
  const languages = { th: { a: "th-a {x}", b: "th-b", "language.name": "Thai" }, en: { a: "A {x}", "language.name": "English" } };

  it("takes the language of ?lang= and falls back to Thai", () => {
    expect(new I18n(languages, "en").lang).toBe("en");
    expect(new I18n(languages, "fr").lang).toBe("th");
    expect(new I18n(languages, null).lang).toBe("th");
    expect(new I18n(BUNDLED_LANGUAGES, "xx").lang).toBe("th");
  });

  it("uses Thai for a key the language lacks", () => {
    const i18n = new I18n(languages, "en");
    expect(i18n.t("a", { x: 1 })).toBe("A 1");
    expect(i18n.t("b")).toBe(languages.th.b);
    expect(i18n.codes).toEqual(["th", "en"]);
    expect(i18n.nameOf("en")).toBe("English");
  });

  it("shows the key and warns when even Thai lacks it", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const i18n = new I18n(languages);
    expect(i18n.t("nope")).toBe("nope");
    expect(warn).toHaveBeenCalledTimes(1);
    i18n.t("a"); // a missing value warns too, and keeps the placeholder
    expect(warn).toHaveBeenCalledTimes(2);
  });

  it("formats numbers for the language", () => {
    const i18n = new I18n(languages);
    expect(i18n.number(83.6, 1)).toBe("83.6");
    expect(i18n.number(1.4, 2)).toBe("1.40");
    expect(i18n.number(27, 1)).toBe("27.0");
  });

  it("sets <html lang>", () => {
    const i18n = new I18n(languages, "en");
    applyDocumentLang(i18n);
    expect(document.documentElement.lang).toBe("en");
    i18n.setLang("th");
    applyDocumentLang(i18n);
    expect(document.documentElement.lang).toBe("th");
  });

  it("names a language by its file", () => {
    expect(Object.keys(languagesFromFiles({ "../../shared/i18n/th.json": {}, "../../shared/i18n/pt-BR.json": {}, "x/README.json": {} }))).toEqual(["th", "pt-BR"]);
  });
});

describe("the code", () => {
  it("has no Thai letters in .ts files outside i18n.ts", () => {
    const files = [...filesUnder(join(WEB, "src"), [".ts"]), ...filesUnder(join(WEB, "tests"), [".ts"]), ...filesUnder(join(WEB, "build"), [".ts"]), join(WEB, "vite.config.ts")];
    const found = files.filter((p) => !p.endsWith("i18n.ts") && THAI.test(readText(p)));
    expect(found).toEqual([]);
  });

  it("never puts data in through innerHTML", () => {
    const found = filesUnder(join(WEB, "src"), [".ts"]).filter((p) => /\.(innerHTML|outerHTML)\b|insertAdjacentHTML/.test(readText(p)));
    expect(found).toEqual([]);
  });
});
