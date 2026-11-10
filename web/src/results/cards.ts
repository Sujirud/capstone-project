// The cards of the results page (web.md sections 5.2-5.4), in the order of model.md:
// jaw, chin, under-eye, nasal base, nasal dorsum. The under-eye card holds both eyes,
// the person's right eye first, which is on the image left; each eye has its own
// badge and text.
//
// A card reads top to bottom: what the region is; the badge and what the grade means;
// the measured value with its criterion; and a "procedures" line at the bottom with
// either the button that shows them or why they are not shown. Procedures are built
// on the first press of the button, and the notes with with_procedures true go with
// them only. Outside lab mode no probability and no landmark_value reach the page.
import { button, el, uid } from "../dom.ts";
import type { I18n } from "../i18n.ts";
import {
  FRONTAL_REGIONS,
  type AnalyzeResult,
  type ImageName,
  type ImageSource,
  type Note,
  type Region,
  type RegionResult,
  type WithheldCode,
} from "../types.ts";
import { measureScale } from "./scale.ts";
import { BADGE_TEXT, atLeastLabel, badgeOf, criterionLine, formatProbability, formatValue, levelText } from "./text.ts";

export interface CardGroup {
  /** Key of the card: the region, or "undereye" for both eyes. */
  key: string;
  regions: Region[];
}

export const CARD_GROUPS: CardGroup[] = [
  { key: "jaw", regions: ["jaw"] },
  { key: "chin", regions: ["chin"] },
  { key: "undereye", regions: ["undereye_right", "undereye_left"] }, // right eye first: image left
  { key: "nose_base", regions: ["nose_base"] },
  { key: "nose_dorsum", regions: ["nose_dorsum"] },
];

export interface CardContext {
  i18n: I18n;
  /** Where each image came from; null when it was not sent. */
  sources: Record<ImageName, ImageSource | null>;
}

export function imageOf(region: Region): ImageName {
  return (FRONTAL_REGIONS as readonly Region[]).includes(region) ? "frontal" : "profile";
}

/** The id of a card, for the links of the summary. */
export function cardId(key: string): string {
  return `card-${key}`;
}

/**
 * A name with its English subtitle, a space between them so screen readers keep them apart.
 * A language whose names are English already leaves the subtitles empty (en.json), and
 * then the name stands alone.
 */
export function named(name: string, subtitle: string, subtitleClass = "subtitle"): Node[] {
  if (subtitle === "") return [document.createTextNode(name)];
  return [document.createTextNode(`${name} `), el("span", { class: subtitleClass, text: subtitle, lang: "en" })];
}

/** The "image from the computer" label (web.md section 3.4). */
export function uploadFlag(i18n: I18n): HTMLElement {
  return el("span", { class: "flag", text: i18n.t("level.upload"), attrs: { "data-upload": "" } });
}

export function badge(r: RegionResult, i18n: I18n): HTMLElement {
  const b = badgeOf(r);
  return el("span", { class: `badge badge--${b}`, text: i18n.t(BADGE_TEXT[b]), attrs: { "data-badge": b } });
}

/** The value large on the left; the strip and the criterion on the right. */
function measurementBlock(r: RegionResult, lab: boolean, i18n: I18n): HTMLElement[] {
  return r.measurements.map((m) =>
    el("div", { class: "measure", attrs: { "data-measure": m.name } }, [
      el("div", { class: "measure__main" }, [
        el("p", { class: "measure__name" }, named(i18n.t(`measure.${m.name}.name`), i18n.t(`measure.${m.name}.subtitle`))),
        el("p", { class: "measure__figure" }, [
          el("span", { class: "measure__value", text: formatValue(m.value, m.unit, i18n) }),
          m.method === "model" ? el("span", { class: "measure__method", text: i18n.t("measure.method.model"), attrs: { "data-method": "model" } }) : null,
        ]),
        lab && m.landmark_value !== null
          ? el("p", {
              class: "lab-note",
              text: i18n.t("ui.lab.model_vs_landmark", {
                model: formatValue(m.value, m.unit, i18n),
                landmark: formatValue(m.landmark_value, m.unit, i18n),
              }),
              attrs: { "data-lab": "landmark" },
            })
          : null,
      ]),
      el("div", { class: "measure__against" }, [measureScale(m, i18n), el("p", { class: "measure__criterion", text: criterionLine(m, i18n) })]),
    ]),
  );
}

function labBlock(r: RegionResult, i18n: I18n): HTMLElement[] {
  const out: HTMLElement[] = [];
  if (r.probabilities) {
    const items = Object.entries(r.probabilities.at_least).map(([k, p]) =>
      el("span", { text: i18n.t("ui.lab.at_least", { k: atLeastLabel(k), p: formatProbability(p ?? 0, i18n) }) }),
    );
    out.push(el("p", { class: "lab-note", attrs: { "data-lab": "probabilities" } }, items));
  }
  if (r.warnings.length) {
    out.push(
      el("p", { class: "lab-note", attrs: { "data-lab": "warnings" } }, [
        el("span", { text: i18n.t("ui.lab.warnings") }),
        ...r.warnings.map((code) => el("code", { class: "code", text: code })),
      ]),
    );
  }
  return out;
}

/** The badge and the meaning of the grade, or why the region cannot be assessed. */
function verdict(r: RegionResult, result: AnalyzeResult, ctx: CardContext, title: boolean): HTMLElement {
  const i18n = ctx.i18n;
  const box = el("div", { class: "verdict" });
  if (title) box.append(el("h4", { class: "verdict__title" }, named(i18n.t(`region.${r.region}.name`), i18n.t(`region.${r.region}.subtitle`))));
  const level = r.status === "assessed" ? levelText(r.region, r.decision, i18n) : null;
  box.append(
    el("p", { class: "verdict__line" }, [
      badge(r, i18n),
      level ? el("span", { class: "level", text: level.summary, attrs: { "data-level": r.decision!.level } }) : null,
    ]),
  );
  if (level?.details.length) box.append(el("ul", { class: "level__details" }, level.details.map((d) => el("li", { text: d }))));
  if (r.status === "not_assessable") {
    box.append(
      el(
        "ul",
        { class: "reasons" },
        r.status_reasons.map((code) => el("li", { text: i18n.t(`quality.${code}`), attrs: { "data-reason": code } })),
      ),
    );
  }
  if (result.lab && r.status === "assessed") box.append(...labBlock(r, i18n));
  return box;
}

/** One region of a card: the verdict, then the measurement when there is one. */
function regionRow(r: RegionResult, result: AnalyzeResult, ctx: CardContext, titled: boolean): HTMLElement {
  const measures = r.status === "assessed" ? measurementBlock(r, result.lab, ctx.i18n) : [];
  return el("div", { class: "card__row", attrs: { "data-region": r.region, "data-status": r.status } }, [
    verdict(r, result, ctx, titled),
    ...measures,
  ]);
}

function uniqueNotes(rows: RegionResult[], withProcedures: boolean): Note[] {
  const out: Note[] = [];
  for (const r of rows) {
    for (const n of r.notes) if (n.with_procedures === withProcedures && !out.some((o) => o.code === n.code)) out.push(n);
  }
  return out;
}

/** The "show procedures" button and its section, or null when no region of the card has procedures. */
function proceduresToggle(rows: RegionResult[], ctx: CardContext, titled: boolean): HTMLElement[] {
  const i18n = ctx.i18n;
  const withProcedures = rows.filter((r) => r.procedures.length > 0);
  if (withProcedures.length === 0) return [];
  const id = uid("procedures");
  const section = el("div", { class: "procedures", attrs: { id, hidden: true, "data-procedures": "" } });
  const fill = () => {
    for (const r of withProcedures) {
      const list = el(
        "ol",
        { class: "procedures__list" },
        r.procedures.map((p) =>
          el("li", { class: "procedure", attrs: { "data-procedure": p.code } }, [
            el("p", { class: "procedure__name", text: p.name }),
            el("p", { class: "procedure__detail" }, [el("span", { class: "procedure__label", text: i18n.t("ui.procedures.when") }), document.createTextNode(p.when)]),
            p.checks
              ? el("p", { class: "procedure__detail" }, [el("span", { class: "procedure__label", text: i18n.t("ui.procedures.checks") }), document.createTextNode(p.checks)])
              : null,
          ]),
        ),
      );
      section.append(titled ? el("div", { class: "procedures__group" }, [el("h4", { class: "procedures__title", text: i18n.t(`region.${r.region}.name`) }), list]) : list);
    }
    for (const n of uniqueNotes(withProcedures, true)) {
      section.append(el("p", { class: "note", text: n.text, attrs: { "data-note": n.code } }));
    }
    section.append(el("p", { class: "procedures__doctor", text: i18n.t("ui.procedures.doctor") }));
  };
  const toggle = button(
    i18n.t("ui.procedures.show"),
    () => {
      const open = toggle.getAttribute("aria-expanded") !== "true";
      if (open && section.childElementCount === 0) fill();
      toggle.setAttribute("aria-expanded", String(open));
      toggle.textContent = i18n.t(open ? "ui.procedures.hide" : "ui.procedures.show");
      section.hidden = !open;
    },
    { class: "btn btn--secondary", attrs: { "aria-expanded": "false", "aria-controls": id, "data-action": "procedures" } },
  );
  return [toggle, section];
}

/**
 * Why procedures are not shown, once per reason. On the under-eye card a reason that
 * holds for one eye only names that eye.
 */
function withheldLines(rows: RegionResult[], ctx: CardContext, titled: boolean): HTMLElement[] {
  const i18n = ctx.i18n;
  const without = rows.filter((r) => r.status === "assessed" && r.procedures.length === 0);
  const byCode = new Map<WithheldCode, Region[]>();
  for (const r of without) for (const code of r.procedures_withheld) byCode.set(code, [...(byCode.get(code) ?? []), r.region]);
  return [...byCode].map(([code, regions]) => {
    const text = i18n.t(`withheld.${code}`);
    const partial = titled && regions.length < rows.length;
    return el("p", {
      class: "withheld",
      text: partial ? i18n.t("ui.card.for_region", { region: i18n.t(`region.${regions[0]}.name`), text }) : text,
      attrs: { "data-withheld": code, "data-regions": regions.join(" ") },
    });
  });
}

/** The bottom line of a card: the procedures button, or why procedures are not shown. */
function proceduresLine(rows: RegionResult[], ctx: CardContext, titled: boolean): HTMLElement | null {
  const content = [...withheldLines(rows, ctx, titled), ...proceduresToggle(rows, ctx, titled)];
  if (content.length === 0) return null;
  return el("div", { class: "card__procedures" }, [
    el("p", { class: "card__label", text: ctx.i18n.t("ui.card.procedures") }),
    el("div", { class: "card__procedures-body" }, content),
  ]);
}

/** One card: a region, or both eyes for the under-eye. Null when the result has none of its regions. */
export function card(group: CardGroup, result: AnalyzeResult, ctx: CardContext): HTMLElement | null {
  const i18n = ctx.i18n;
  const rows = group.regions
    .map((name) => result.regions.find((r) => r.region === name))
    .filter((r): r is RegionResult => r !== undefined);
  if (rows.length === 0) return null;
  const titled = group.regions.length > 1;
  const headingId = uid(`card-${group.key}-title`);
  const fromUpload = rows.some((r) => r.status === "assessed" && ctx.sources[imageOf(r.region)] === "upload");

  const header = el("header", { class: "card__header" }, [
    el("div", { class: "card__heading" }, [
      el("h3", { class: "card__name", attrs: { id: headingId } }, named(i18n.t(`region.${group.key}.name`), i18n.t(`region.${group.key}.subtitle`))),
      fromUpload ? uploadFlag(i18n) : null,
    ]),
    el("p", { class: "card__about", text: i18n.t("ui.card.about", { what: i18n.t(`region.${group.key}.what`), how: i18n.t(`region.${group.key}.how`) }) }),
  ]);

  const body = el("div", { class: `card__body${titled ? " card__body--pair" : ""}` }, rows.map((r) => regionRow(r, result, ctx, titled)));
  const notes = uniqueNotes(rows, false).map((n) => el("p", { class: "note", text: n.text, attrs: { "data-note": n.code } }));
  return el("article", { class: "card", attrs: { id: cardId(group.key), "data-card": group.key, "aria-labelledby": headingId } }, [
    header,
    body,
    ...notes,
    proceduresLine(rows, ctx, titled),
  ]);
}

export function cards(result: AnalyzeResult, ctx: CardContext): HTMLElement[] {
  return CARD_GROUPS.map((g) => card(g, result, ctx)).filter((c): c is HTMLElement => c !== null);
}
