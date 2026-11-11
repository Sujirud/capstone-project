// One figure of the results page (web.md section 5.1): an image in the arch with its
// measurement lines, its name, a button that opens it full screen (viewer.ts), and the
// legend whose buttons hide or show the lines of each region (overlay.ts).
import { el, uid } from "../dom.ts";
import type { I18n } from "../i18n.ts";
import type { ImageName, RegionResult } from "../types.ts";
import { FIGURE_PX, IMAGE_TEXT, LEGEND, LineState, type ShownImage, archImage, legendList, lineWidth, overlayLayer } from "./overlay.ts";
import { openViewer } from "./viewer.ts";

/**
 * One image with the lines of every region measured on it. regions are all the regions
 * of the result; those whose overlay is on another image, or that have none, are left out.
 */
export function overlayFigure(name: ImageName, image: ShownImage, regions: RegionResult[], i18n: I18n): HTMLElement {
  const drawn = regions.filter((r) => r.overlay !== null && r.overlay.image === name);
  const groups = LEGEND[name].filter((entry) => drawn.some((r) => entry.regions.includes(r.region)));
  const state = new LineState(name, groups, i18n);
  const layer = overlayLayer(name, image, drawn, lineWidth(image.width, FIGURE_PX));
  const legend = groups.length ? legendList(state) : null;
  state.attach(layer, legend);

  const nameId = uid("figure-name");
  const open = el("button", {
    class: "result-figure__open",
    text: i18n.t("ui.overlay.open"),
    attrs: { type: "button", "aria-describedby": nameId, "data-action": "viewer-open" },
  });
  const picture = el("div", { class: "result-figure__picture", attrs: { "data-viewer-open": name } }, [archImage(image.url, "", layer)]);
  const figure = el("figure", { class: "result-figure", attrs: { "data-figure": name } }, [
    picture,
    el("figcaption", { class: "result-figure__caption" }, [
      el("div", { class: "result-figure__head" }, [el("span", { class: "result-figure__name", text: i18n.t(IMAGE_TEXT[name]), attrs: { id: nameId } }), open]),
      legend,
    ]),
  ]);
  const show = () => openViewer({ image, drawn, state, host: figure, opener: open });
  open.addEventListener("click", show);
  picture.addEventListener("click", show); // a shortcut for the pointer; the button is the way in for the keyboard
  return figure;
}
