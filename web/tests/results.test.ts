// The results page against every example result (web.md section 5).
import { describe, expect, it } from "vitest";
import { BUNDLED_LANGUAGES, I18n } from "../src/i18n.ts";
import { CARD_GROUPS, named } from "../src/results/cards.ts";
import { LEGEND } from "../src/results/overlay.ts";
import type { Grade, Region, RegionResult } from "../src/types.ts";
import { FIXTURES, fixture, renderResult, th } from "./helpers.ts";

const NAMES = Object.keys(FIXTURES).sort();
const FRONTAL: Region[] = ["jaw", "undereye_left", "undereye_right", "nose_base"];

function row(root: HTMLElement, region: Region): HTMLElement {
  const found = root.querySelector<HTMLElement>(`.card__row[data-region="${region}"]`);
  if (!found) throw new Error(`no row for ${region}`);
  return found;
}

function cardOf(root: HTMLElement, region: Region): HTMLElement {
  return row(root, region).closest<HTMLElement>(".card")!;
}

/** Text of the page without the raw JSON of lab mode, which holds every text of the result. */
function pageText(root: HTMLElement): string {
  const copy = root.cloneNode(true) as HTMLElement;
  copy.querySelector('[data-lab="json"]')?.remove();
  return copy.textContent ?? "";
}

function gradeKey(region: Region, grade: Grade): string {
  return region === "chin" ? `grade.chin.${grade}` : `grade.default.${grade}`;
}

function expectedBadge(r: RegionResult): string {
  if (r.status === "not_assessable") return "not_assessable";
  return r.decision === null ? "values_only" : r.decision.level;
}

const BADGE_KEY: Record<string, string> = {
  clear: "level.clear",
  borderline: "level.borderline",
  values_only: "level.values_only",
  not_assessable: "level.not_assessable",
};

describe.each(NAMES)("fixture %s", (name) => {
  const f = fixture(name);
  const regions = f.result.regions;

  it("has every card, in the order of model.md", () => {
    const { root } = renderResult(f);
    const expected = CARD_GROUPS.filter((g) => g.regions.some((r) => regions.some((x) => x.region === r))).map((g) => g.key);
    expect([...root.querySelectorAll<HTMLElement>(".card")].map((c) => c.dataset.card)).toEqual(expected);
    // the under-eye card: the person's right eye (image left) first
    const eyes = [...root.querySelectorAll<HTMLElement>('[data-card="undereye"] .card__row')].map((r) => r.dataset.region);
    if (eyes.length) expect(eyes).toEqual(["undereye_right", "undereye_left"]);
  });

  it("gives each region its badge and the meaning of its grade", () => {
    const { root } = renderResult(f);
    for (const r of regions) {
      const el = row(root, r.region);
      const badge = expectedBadge(r);
      const shown = el.querySelector<HTMLElement>("[data-badge]")!;
      expect(shown.dataset.badge).toBe(badge);
      expect(shown.textContent).toBe(th(BADGE_KEY[badge]));
      const level = el.querySelector(".level");
      const d = r.decision;
      if (r.status !== "assessed" || d === null) {
        expect(level).toBeNull();
      } else if (d.level === "clear") {
        expect(level!.textContent).toBe(th("ui.grade.line", { grade: d.grade!, text: th(gradeKey(r.region, d.grade!)) }));
      } else {
        const [low, high] = d.grade_range!;
        const details = [...el.querySelectorAll(".level__details li")].map((li) => li.textContent);
        const wide = details.length > 2;
        expect(level!.textContent).toBe(th(wide ? "grade.between_wide" : "grade.between", { low, high }));
        expect(details[0]).toBe(th("ui.grade.line", { grade: low, text: th(gradeKey(r.region, low)) }));
        expect(details.at(-1)).toBe(th("ui.grade.line", { grade: high, text: th(gradeKey(r.region, high)) }));
      }
    }
  });

  it("shows why procedures are withheld on the procedures line, instead of the button", () => {
    const { root } = renderResult(f);
    for (const card of root.querySelectorAll<HTMLElement>(".card")) {
      // in the order of the card: the person's right eye first
      const rows = [...card.querySelectorAll<HTMLElement>(".card__row")].map((el) => regions.find((r) => r.region === el.dataset.region)!);
      const button = card.querySelector('[data-action="procedures"]');
      expect(button !== null).toBe(rows.some((r) => r.procedures.length > 0));
      // every reason of every assessed region without procedures, once per reason
      const expected = new Map<string, string[]>();
      for (const r of rows.filter((x) => x.status === "assessed" && x.procedures.length === 0)) {
        for (const code of r.procedures_withheld) expected.set(code, [...(expected.get(code) ?? []), r.region]);
      }
      const shown = [...card.querySelectorAll<HTMLElement>("[data-withheld]")];
      expect(shown.map((w) => [w.dataset.withheld, w.dataset.regions!.split(" ")])).toEqual([...expected]);
      for (const w of shown) {
        expect(w.closest(".card__procedures")).not.toBeNull();
        const text = th(`withheld.${w.dataset.withheld}`);
        const named = w.dataset.regions!.split(" ");
        const partial = rows.length > 1 && named.length < rows.length;
        expect(w.textContent).toBe(partial ? th("ui.card.for_region", { region: th(`region.${named[0]}.name`), text }) : text);
      }
      if (shown.length || button) expect(card.querySelector(".card__procedures .card__label")!.textContent).toBe(th("ui.card.procedures"));
    }
  });

  it("lists every region in the summary with its badge, linked to its card", () => {
    const { root } = renderResult(f);
    const items = [...root.querySelectorAll<HTMLElement>(".summary [data-summary]")];
    const order = CARD_GROUPS.flatMap((g) => g.regions).filter((r) => regions.some((x) => x.region === r));
    expect(items.map((i) => i.dataset.summary)).toEqual(order);
    for (const item of items) {
      const r = regions.find((x) => x.region === item.dataset.summary)!;
      expect(item.querySelector<HTMLElement>("[data-badge]")!.dataset.badge).toBe(expectedBadge(r));
      const target = item.querySelector("a")!.getAttribute("href")!.slice(1);
      expect(document.getElementById(target)!.contains(row(root, r.region))).toBe(true);
      const level = item.querySelector(".summary__level")!.textContent;
      const d = r.decision;
      if (r.status !== "assessed") expect(level).toBe("");
      else if (d === null) expect(level).toBe(row(root, r.region).querySelector(".measure__value")!.textContent);
      else if (d.level === "clear") expect(level).toBe(th("ui.grade.short", { grade: d.grade! }));
      else expect(level).toBe(th("ui.grade.range_short", { low: d.grade_range![0], high: d.grade_range![1] }));
    }
  });

  it("hides procedures until the button is pressed, and their notes stay with them", () => {
    const { root } = renderResult(f);
    const procedureNotes = regions.flatMap((r) => r.notes.filter((n) => n.with_procedures));
    const names = regions.flatMap((r) => r.procedures.map((p) => p.name));
    for (const section of root.querySelectorAll<HTMLElement>("[data-procedures]")) {
      expect(section.hidden).toBe(true);
      expect(section.textContent).toBe("");
    }
    for (const n of procedureNotes) expect(pageText(root)).not.toContain(n.text);
    expect(root.querySelector("[data-procedure]")).toBeNull();

    for (const button of root.querySelectorAll<HTMLButtonElement>('[data-action="procedures"]')) {
      expect(button.getAttribute("aria-expanded")).toBe("false");
      button.click();
      expect(button.getAttribute("aria-expanded")).toBe("true");
      const section = document.getElementById(button.getAttribute("aria-controls")!)!;
      expect(section.hidden).toBe(false);
      expect(section.textContent).toContain(th("ui.procedures.doctor"));
    }
    const sections = [...root.querySelectorAll<HTMLElement>("[data-procedures]")].map((s) => s.textContent).join("\n");
    for (const p of names) expect(sections).toContain(p);
    for (const n of procedureNotes) {
      expect(sections).toContain(n.text);
      const outside = [...root.querySelectorAll<HTMLElement>(`[data-note="${n.code}"]`)].filter((e) => !e.closest("[data-procedures]"));
      expect(outside).toEqual([]);
    }
    // in the order received, least invasive first
    for (const r of regions.filter((x) => x.procedures.length)) {
      const shown = [...cardOf(root, r.region).querySelectorAll<HTMLElement>("[data-procedure]")].map((p) => p.dataset.procedure);
      const codes = r.procedures.map((p) => p.code);
      expect(shown.filter((c) => codes.includes(c as never)).slice(0, codes.length)).toEqual(codes);
    }
  });

  it("shows the notes that do not belong to procedures on the card", () => {
    const { root } = renderResult(f);
    for (const r of regions) {
      for (const n of r.notes.filter((x) => !x.with_procedures)) {
        const note = cardOf(root, r.region).querySelector<HTMLElement>(`.note[data-note="${n.code}"]`);
        expect(note?.textContent).toBe(n.text);
      }
    }
  });

  it("has the disclaimer at the bottom (web.md section 5.2)", () => {
    const { root } = renderResult(f);
    const shown = [...root.querySelectorAll<HTMLElement>(".disclaimer")];
    expect(shown.length).toBe(1);
    expect(shown[0].textContent).toBe(f.result.disclaimer);
    expect(root.lastElementChild).toBe(shown[0]);
  });

  it("explains each region and each measurement", () => {
    const { root } = renderResult(f);
    for (const card of root.querySelectorAll<HTMLElement>(".card")) {
      const key = card.dataset.card!;
      expect(card.querySelector(".card__name")!.textContent).toBe(`${th(`region.${key}.name`)} ${th(`region.${key}.subtitle`)}`);
      expect(card.querySelector(".card__about")!.textContent).toBe(th("ui.card.about", { what: th(`region.${key}.what`), how: th(`region.${key}.how`) }));
    }
    for (const r of regions) {
      for (const m of r.measurements) {
        const block = row(root, r.region).querySelector<HTMLElement>(`[data-measure="${m.name}"]`)!;
        expect(block.querySelector(".measure__name")!.textContent).toBe(`${th(`measure.${m.name}.name`)} ${th(`measure.${m.name}.subtitle`)}`);
        expect(block.querySelector(".measure__criterion")!.textContent).toBe(th("ui.measure.criterion", { reference: m.reference, source: m.source }));
        expect(block.querySelector('[data-method="model"]') !== null).toBe(m.method === "model");
        if (m.method === "model") expect(block.textContent).toContain(th("measure.method.model"));
        // the strip of the value against the cutoffs, hidden from screen readers since the text says the same
        const strip = block.querySelector<SVGSVGElement>("svg.scale");
        expect(strip !== null).toBe(m.cutoffs.length > 0);
        if (strip) {
          expect(strip.getAttribute("aria-hidden")).toBe("true");
          expect(strip.querySelectorAll(".scale__tick").length).toBe(m.cutoffs.length);
          expect(strip.querySelectorAll(".scale__band").length).toBe(m.cutoffs.length);
          expect(strip.querySelectorAll(".scale__value").length).toBe(1);
        }
      }
    }
  });

  it("says what to do for regions that cannot be assessed, with no button to take an image again", () => {
    const { root } = renderResult(f);
    for (const r of regions) {
      const reasons = [...row(root, r.region).querySelectorAll<HTMLElement>("[data-reason]")].map((x) => x.textContent);
      expect(reasons).toEqual(r.status === "assessed" ? [] : r.status_reasons.map((c) => th(`quality.${c}`)));
    }
    // the user goes back to change an image (web.md section 4.3)
    expect(root.querySelectorAll("[data-retake]").length).toBe(0);
    const procedureButtons = root.querySelectorAll('[data-action="procedures"]').length;
    const copyButtons = root.querySelectorAll('[data-action="copy-json"]').length;
    expect(root.querySelectorAll("button").length).toBe(procedureButtons + copyButtons);
  });

  it("draws every overlay on an SVG the size of its image, with a legend for every region", () => {
    const { root } = renderResult(f);
    for (const name of ["frontal", "profile"] as const) {
      const size = f.images[name];
      const layer = root.querySelector<SVGSVGElement>(`svg.overlay-layer[data-image="${name}"]`);
      if (!size) {
        expect(layer).toBeNull();
        continue;
      }
      expect(layer!.getAttribute("viewBox")).toBe(`0 0 ${size.width} ${size.height}`);
      expect(layer!.getAttribute("role")).toBe("img");
      expect(layer!.getAttribute("aria-label")).toBeTruthy();
      // image and SVG share one container
      expect(layer!.parentElement!.querySelector("img")).not.toBeNull();
      const drawn = regions.filter((r) => r.overlay?.image === name);
      for (const r of drawn) {
        const g = layer!.querySelector(`g[data-region="${r.region}"]`)!;
        const lines = g.querySelectorAll(".overlay__line");
        const box = r.region.startsWith("undereye");
        expect(lines.length).toBe(box ? 1 : r.overlay!.lines.length);
        expect(g.querySelectorAll(".overlay__under").length).toBe(lines.length); // white under every line
        if (box) expect(lines[0].tagName.toLowerCase()).toBe("polygon");
        else expect(g.querySelectorAll(".overlay__dot").length).toBe(new Set(r.overlay!.lines.flat()).size);
      }
      expect(layer!.querySelectorAll("g[data-region]").length).toBe(drawn.length);
      const legend = [...root.querySelectorAll<HTMLElement>(`[data-legend="${name}"] li`)].map((li) => li.dataset.legendRegions!.split(" "));
      const expected = LEGEND[name].filter((e) => e.regions.some((x) => drawn.some((r) => r.region === x))).map((e) => e.regions);
      expect(legend).toEqual(expected);
    }
  });

  it("keeps probabilities and landmark values for lab mode", () => {
    const { root } = renderResult(f);
    const text = root.textContent ?? "";
    const lab = f.result.lab;
    expect(root.querySelector('[data-lab="json"]') !== null).toBe(lab);
    expect(root.querySelectorAll("[data-lab]").length > 0).toBe(lab);
    expect(text.includes("P(")).toBe(lab);
    // the words between the model value and the landmark value of ui.lab.model_vs_landmark
    const between = th("ui.lab.model_vs_landmark", { model: "<m>", landmark: "<l>" }).split("<m>")[1].split("<l>")[0].trim();
    expect(text.includes(between)).toBe(lab && regions.some((r) => r.measurements.some((m) => m.landmark_value !== null)));
    if (!lab) {
      for (const m of regions.flatMap((r) => r.measurements)) {
        if (m.landmark_value === null || m.landmark_value === m.value) continue;
        const digits = m.unit === "ratio" ? 2 : 1;
        expect(text).not.toContain(m.landmark_value.toFixed(digits));
      }
    }
    for (const r of regions.filter((x) => x.status === "assessed")) {
      const el = row(root, r.region);
      const landmarks = r.measurements.filter((x) => x.landmark_value !== null).length;
      expect(el.querySelectorAll('[data-lab="landmark"]').length).toBe(lab ? landmarks : 0);
      if (r.probabilities) {
        const shown = el.querySelector('[data-lab="probabilities"]');
        expect(shown !== null).toBe(lab);
        if (lab) {
          for (const [k, p] of Object.entries(r.probabilities.at_least)) {
            expect(shown!.textContent).toContain(th("ui.lab.at_least", { k: k.replace("+", ""), p: p!.toFixed(2) }));
          }
        }
      }
      const warnings = el.querySelector('[data-lab="warnings"]');
      expect(warnings !== null).toBe(lab && r.warnings.length > 0);
      if (warnings) expect([...warnings.querySelectorAll("code")].map((c) => c.textContent)).toEqual(r.warnings);
    }
  });
});

describe("details of single cases", () => {
  it("labels the cards measured on an uploaded image", () => {
    const { root } = renderResult(fixture("upload_frontal"));
    for (const card of root.querySelectorAll<HTMLElement>(".card")) {
      const frontal = ["jaw", "undereye", "nose_base"].includes(card.dataset.card!);
      const chip = card.querySelector("[data-upload]");
      expect(chip !== null).toBe(frontal);
      if (chip) expect(chip.textContent).toBe(th("level.upload"));
    }
    const both = renderResult(fixture("upload_frontal"), { profile: "upload" }).root;
    expect(both.querySelectorAll(".card [data-upload]").length).toBe(5);
    const camera = renderResult(fixture("female_clear")).root;
    expect(camera.querySelectorAll("[data-upload]").length).toBe(0);
  });

  it("shows the alar value without a grade for an uploaded image", () => {
    const f = fixture("upload_frontal");
    const { root } = renderResult(f);
    const nose = row(root, "nose_base");
    expect(nose.querySelector<HTMLElement>("[data-badge]")!.dataset.badge).toBe("values_only");
    expect(cardOf(root, "nose_base").querySelector("[data-withheld]")!.textContent).toBe(th("withheld.distance_unknown"));
    expect(nose.querySelector(".measure__value")!.textContent).toBe("1.40"); // ratios with two decimals
  });

  it("formats percent and degrees with one decimal", () => {
    const { root } = renderResult(fixture("api_example"));
    expect(row(root, "jaw").querySelector(".measure__value")!.textContent).toBe("83.6%");
    expect(row(root, "jaw").querySelector(".measure__criterion")!.textContent).toBe(th("reference.facial_taperness.female", { cut: 83, band: 1 }) + " [3]");
    expect(row(root, "nose_dorsum").querySelector(".measure__value")!.textContent).toBe("28.5°");
    const chin = renderResult(fixture("chin3_nose_base2")).root;
    expect(row(chin, "chin").querySelector(".measure__value")!.textContent).toBe("27.0°");
    // cutoffs on the strip without trailing zeros
    expect([...row(root, "jaw").querySelectorAll(".scale__label")].map((l) => l.textContent)).toEqual(["83%", "87%"]);
  });

  it("says 'between 0 and 2+' for an under-eye borderline at both cutpoints", () => {
    const { root } = renderResult(fixture("borderline"));
    const right = row(root, "undereye_right");
    expect(right.querySelector(".level")!.textContent).toBe(th("grade.between_wide", { low: "0", high: "2+" }));
    expect(right.querySelectorAll(".level__details li").length).toBe(3);
    const left = row(root, "undereye_left");
    expect(left.querySelector(".level")!.textContent).toBe(th("grade.between", { low: "1", high: "2+" }));
  });

  it("gives each eye its own badge when hair covers one", () => {
    const { root } = renderResult(fixture("hair_jaw_one_eye"));
    expect(row(root, "undereye_left").querySelector<HTMLElement>("[data-badge]")!.dataset.badge).toBe("not_assessable");
    expect(row(root, "undereye_right").querySelector<HTMLElement>("[data-badge]")!.dataset.badge).toBe("clear");
    expect(row(root, "undereye_left").querySelector("[data-reason]")!.textContent).toBe(th("quality.hair_occlusion"));
    expect(row(root, "jaw").querySelector("[data-reason]")!.textContent).toBe(th("quality.hair_occlusion"));
    // the undereye_scope note of the right eye is shown once
    expect(cardOf(root, "undereye_right").querySelectorAll('[data-note="undereye_scope"]').length).toBe(1);
    // a reason that holds for one eye names it
    const withheld = cardOf(root, "undereye_right").querySelector<HTMLElement>("[data-withheld]")!;
    expect(withheld.textContent).toBe(th("ui.card.for_region", { region: th("region.undereye_right.name"), text: th("withheld.no_procedure_at_grade") }));
  });

  it("says when there is no face or no profile image", () => {
    const { root } = renderResult(fixture("no_face_no_profile"));
    for (const region of ["chin", "nose_dorsum"] as const) {
      expect(row(root, region).querySelector("[data-reason]")!.textContent).toBe(th("quality.missing_input"));
    }
    for (const region of FRONTAL) {
      expect(row(root, region).querySelector("[data-reason]")!.textContent).toBe(th("quality.no_face"));
    }
    expect(root.querySelector('svg[data-image="profile"]')).toBeNull();
    expect(root.querySelector('[data-legend="frontal"]')).toBeNull(); // nothing drawn
  });

  it("says a stub head is not ready", () => {
    const { root } = renderResult(fixture("stub_heads"));
    for (const region of FRONTAL) {
      expect(row(root, region).querySelector("[data-reason]")!.textContent).toBe(th("quality.model_not_ready"));
    }
    expect(row(root, "chin").dataset.status).toBe("assessed");
  });

  it("shows the jaw value of men without a grade", () => {
    const { root } = renderResult(fixture("male"));
    const jaw = row(root, "jaw");
    expect(jaw.querySelector<HTMLElement>("[data-badge]")!.dataset.badge).toBe("values_only");
    expect(cardOf(root, "jaw").querySelector("[data-withheld]")!.textContent).toBe(th("withheld.male_jaw_no_criterion"));
    expect(jaw.querySelector("svg.scale")).toBeNull(); // no cutoffs for men
  });

  it("withholds every procedure under 18 and the under-eye ones up to 21", () => {
    const young = renderResult(fixture("age_17")).root;
    expect(young.querySelectorAll('[data-action="procedures"]').length).toBe(0);
    expect(young.querySelectorAll('[data-withheld="age_under_18"]').length).toBe(5); // once per card
    const twenty = renderResult(fixture("age_20")).root;
    const eyes = twenty.querySelectorAll<HTMLElement>('[data-withheld="age_21_or_under"]');
    expect(eyes.length).toBe(1);
    expect(eyes[0].dataset.regions).toBe("undereye_right undereye_left");
    expect(eyes[0].textContent).toBe(th("withheld.age_21_or_under"));
    expect(cardOf(twenty, "undereye_left").querySelector('[data-action="procedures"]')).toBeNull();
    expect(cardOf(twenty, "jaw").querySelector('[data-action="procedures"]')).not.toBeNull();
  });

  it("copies the whole JSON in lab mode, which has no image", async () => {
    const f = fixture("lab");
    const { root, copy } = renderResult(f);
    const pre = root.querySelector('[data-lab="json"] pre')!;
    expect(JSON.parse(pre.textContent!)).toEqual(f.result);
    root.querySelector<HTMLButtonElement>('[data-action="copy-json"]')!.click();
    await Promise.resolve();
    expect(copy).toHaveBeenCalledWith(JSON.stringify(f.result, null, 2));
    expect(root.querySelector('[data-lab="landmark"]')!.textContent).toBe(th("ui.lab.model_vs_landmark", { model: "86.1%", landmark: "87.2%" }));
  });

  it("puts api texts in as text, never as markup", () => {
    const f = fixture("female_clear");
    const jaw = f.result.regions.find((r) => r.region === "jaw")!;
    jaw.procedures[0].name = '<img src=x onerror="alert(1)">';
    f.result.disclaimer = "<b>bold</b>";
    const { root } = renderResult(f);
    cardOf(root, "jaw").querySelector<HTMLButtonElement>('[data-action="procedures"]')!.click();
    expect(root.querySelector("img[src=x]")).toBeNull();
    expect(root.querySelector("b")).toBeNull();
    expect(root.textContent).toContain('<img src=x onerror="alert(1)">');
  });
});

describe("in English (shared/i18n/en.json)", () => {
  // Written as code points so this file has no Thai letters
  const THAI = /[\u0E00-\u0E7F]/;
  const ATTRIBUTES = ["aria-label", "alt", "title", "placeholder"];

  /** Every text in a result: the texts the api wrote, in the language of the request. */
  function apiTexts(value: unknown, out: string[] = []): string[] {
    if (typeof value === "string") out.push(value);
    else if (Array.isArray(value)) for (const v of value) apiTexts(v, out);
    else if (value && typeof value === "object") for (const v of Object.values(value)) apiTexts(v, out);
    return out;
  }

  it("shows a name alone when its subtitle is empty", () => {
    const box = document.createElement("p");
    box.append(...named("Jaw", ""));
    expect(box.innerHTML).toBe("Jaw");
    box.replaceChildren(...named("Jaw", "Mandible"));
    expect(box.querySelector(".subtitle")?.textContent).toBe("Mandible");
  });

  it.each(NAMES)("%s: Thai on the page comes only from the api, never from the texts of the page", (name) => {
    const f = fixture(name); // written in Thai by the api, so its texts are Thai here
    const fromApi = apiTexts(f.result).filter((t) => THAI.test(t)).sort((a, b) => b.length - a.length);
    const { root } = renderResult(f, undefined, new I18n(BUNDLED_LANGUAGES, "en"));
    for (const button of root.querySelectorAll<HTMLButtonElement>('[data-action="procedures"]')) button.click();
    const shown: string[] = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) shown.push(n.textContent ?? "");
    for (const e of root.querySelectorAll("*")) for (const a of ATTRIBUTES) shown.push(e.getAttribute(a) ?? "");
    const left = shown.map((t) => fromApi.reduce((rest, api) => rest.split(api).join(""), t)).filter((t) => THAI.test(t));
    expect(left).toEqual([]);
    expect(root.querySelector(".subtitle")).toBeNull(); // English names need no English subtitle
  });
});
