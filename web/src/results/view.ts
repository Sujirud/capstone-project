// The results page (web.md section 5): the disclaimer at the top, the images with their
// lines, the cards, lab details when the result is from lab mode, and the disclaimer
// again at the bottom.
import { el, uid } from "../dom.ts";
import type { I18n } from "../i18n.ts";
import { IMAGES, type AnalyzeResult, type ImageName, type ImageSource } from "../types.ts";
import { cards } from "./cards.ts";
import type { FrontalValues } from "../metrics.ts";
import { labCompare, labJson } from "./lab.ts";
import { overlayFigure } from "./figure.ts";
import type { ShownImage } from "./overlay.ts";
import { summary } from "./summary.ts";

export interface ResultImage extends ShownImage {
  source: ImageSource;
}

export interface ResultContext {
  i18n: I18n;
  images: Record<ImageName, ResultImage | null>;
  copy: (text: string) => Promise<void>;
  /** Lab mode: what the page measured on the frontal photo. */
  lab?: { frontalValues: FrontalValues | null };
}

/** Everything of the results page below its heading. */
export function resultBody(result: AnalyzeResult, ctx: ResultContext): HTMLElement[] {
  const { i18n } = ctx;
  const figures = IMAGES.flatMap((name) => {
    const image = ctx.images[name];
    return image ? [overlayFigure(name, image, result.regions, i18n)] : [];
  });
  const sources = { frontal: ctx.images.frontal?.source ?? null, profile: ctx.images.profile?.source ?? null };
  const cardsId = uid("cards");
  const out: HTMLElement[] = [
    el("div", { class: "overview" }, [el("div", { class: "figures" }, figures), summary(result, i18n)]),
    el("section", { class: "cards", attrs: { "aria-labelledby": cardsId } }, [
      el("h2", { class: "cards__title", text: i18n.t("ui.results.cards"), attrs: { id: cardsId } }),
      ...cards(result, { i18n, sources }),
    ]),
  ];
  if (result.lab) {
    if (ctx.lab) out.push(labCompare(result, ctx.lab.frontalValues, i18n));
    out.push(labJson(result, i18n, ctx.copy));
  }
  out.push(el("p", { class: "disclaimer", text: result.disclaimer, attrs: { role: "note" } }));
  return out;
}
