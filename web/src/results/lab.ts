// Lab mode on the results page (web.md section 7): the whole JSON of the api, which
// holds no image, with a button that copies it; and what the page measured when it took
// the frontal photo beside quality.frontal of the api.
import { button, el } from "../dom.ts";
import type { I18n } from "../i18n.ts";
import type { FrontalValues } from "../metrics.ts";
import type { AnalyzeResult, FrontalQuality } from "../types.ts";

export function labJson(result: AnalyzeResult, i18n: I18n, copy: (text: string) => Promise<void>): HTMLElement {
  const text = JSON.stringify(result, null, 2);
  const done = el("span", { class: "lab__copied", attrs: { "aria-live": "polite" } });
  const copyButton = button(
    i18n.t("ui.lab.copy_json"),
    async () => {
      try {
        await copy(text);
        done.textContent = i18n.t("ui.lab.copied");
      } catch {
        done.textContent = i18n.t("ui.lab.copy_failed");
      }
    },
    { class: "btn btn--secondary", attrs: { "data-action": "copy-json" } },
  );
  return el("section", { class: "lab", attrs: { "data-lab": "json" } }, [
    el("h2", { text: i18n.t("ui.lab.json_title") }),
    el("div", { class: "actions" }, [copyButton, done]),
    el("pre", { class: "lab__json", text, attrs: { tabindex: 0 } }),
  ]);
}

const COMPARED = [
  ["iod_px", 1],
  ["yaw_deg", 1],
  ["pitch_deg", 1],
  ["roll_deg", 1],
  ["smile", 2],
  ["jaw_open", 2],
  ["blink_left", 2],
  ["blink_right", 2],
] as const satisfies readonly (readonly [keyof FrontalValues & keyof FrontalQuality, number])[];

/** What the page measured when it took the frontal photo, beside quality.frontal of the api. */
export function labCompare(result: AnalyzeResult, web: FrontalValues | null, i18n: I18n): HTMLElement {
  const none = i18n.t("ui.lab.no_value");
  const fmt = (v: number | null | undefined, digits: number) => (v === null || v === undefined ? none : i18n.number(v, digits));
  const api = result.quality.frontal;
  const table = el("table", { class: "lab-compare" }, [
    el("thead", {}, [
      el("tr", {}, [
        el("th", { text: i18n.t("ui.lab.compare_value"), attrs: { scope: "col" } }),
        el("th", { text: i18n.t("ui.lab.compare_web"), attrs: { scope: "col" } }),
        el("th", { text: i18n.t("ui.lab.compare_api"), attrs: { scope: "col" } }),
      ]),
    ]),
    el(
      "tbody",
      {},
      COMPARED.map(([name, digits]) =>
        el("tr", { attrs: { "data-value": name } }, [
          el("th", { class: "code", text: name, attrs: { scope: "row" } }),
          el("td", { text: fmt(web?.[name], digits) }),
          el("td", { text: fmt(api[name], digits) }),
        ]),
      ),
    ),
  ]);
  return el("section", { class: "lab", attrs: { "data-lab": "compare" } }, [
    el("h2", { text: i18n.t("ui.lab.compare_title") }),
    web ? null : el("p", { class: "hint", text: i18n.t("ui.lab.compare_none") }),
    table,
  ]);
}
