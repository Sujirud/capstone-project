// The summary above the cards: every region on one line with its badge and grade, so
// the whole result reads at a glance; each name links to its card. Procedures are not
// mentioned here, as they show only when asked for (model.md section 1.3).
import { el, uid } from "../dom.ts";
import type { I18n } from "../i18n.ts";
import type { AnalyzeResult } from "../types.ts";
import { CARD_GROUPS, badge, cardId } from "./cards.ts";
import { shortLevel } from "./text.ts";

export function summary(result: AnalyzeResult, i18n: I18n): HTMLElement {
  const headingId = uid("summary");
  const items = CARD_GROUPS.flatMap((group) =>
    group.regions.flatMap((name) => {
      const r = result.regions.find((x) => x.region === name);
      if (!r) return [];
      return [
        el("li", { class: "summary__item", attrs: { "data-summary": name } }, [
          el("a", { class: "summary__name", text: i18n.t(`region.${name}.name`), attrs: { href: `#${cardId(group.key)}` } }),
          badge(r, i18n),
          el("span", { class: "summary__level", text: shortLevel(r, i18n) }),
        ]),
      ];
    }),
  );
  return el("section", { class: "summary", attrs: { "aria-labelledby": headingId } }, [
    el("h2", { class: "summary__title", text: i18n.t("ui.results.summary"), attrs: { id: headingId } }),
    el("ul", { class: "summary__list" }, items),
  ]);
}
