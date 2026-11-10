// The images with their measurement lines (web.md section 5.1).
//
// The image and the SVG sit in one container and scale together. The SVG's viewBox is
// the pixel size of the image as sent, so the api's coordinates go in unchanged. The
// container clips the arch. Each region on one image has its own colour and its own
// dash pattern, so they can be told apart without colour: a white line under each
// line, and brown dots with a white edge, show on every skin tone.
import { el, svg } from "../dom.ts";
import type { I18n } from "../i18n.ts";
import type { ImageName, Overlay, Region, RegionResult } from "../types.ts";

export interface LineStyle {
  ink: "blue" | "rose" | "cocoa";
  /** Dash pattern in multiples of the line width. */
  dash: number[];
  /** The under-eye is a crop frame, not lines between points. */
  box: boolean;
}

export const LINE_STYLE: Record<Region, LineStyle> = {
  jaw: { ink: "blue", dash: [5, 2.5], box: false }, // long dashes
  nose_base: { ink: "rose", dash: [1.6, 1.6], box: false }, // short dashes
  undereye_left: { ink: "cocoa", dash: [0, 2.2], box: true }, // dots
  undereye_right: { ink: "cocoa", dash: [0, 2.2], box: true },
  chin: { ink: "blue", dash: [5, 2.5], box: false }, // long dashes
  nose_dorsum: { ink: "rose", dash: [5, 1.8, 0.8, 1.8], box: false }, // dash and dot
};

/** Regions drawn on each image, in legend order; both under-eye sides share one entry. */
export const LEGEND: Record<ImageName, { regions: Region[]; text: string }[]> = {
  frontal: [
    { regions: ["jaw"], text: "region.jaw.name" },
    { regions: ["undereye_right", "undereye_left"], text: "region.undereye.name" },
    { regions: ["nose_base"], text: "region.nose_base.name" },
  ],
  profile: [
    { regions: ["chin"], text: "region.chin.name" },
    { regions: ["nose_dorsum"], text: "region.nose_dorsum.name" },
  ],
};

const IMAGE_TEXT: Record<ImageName, string> = { frontal: "ui.image.frontal", profile: "ui.image.profile" };

export interface ShownImage {
  url: string;
  width: number;
  height: number;
}

/** Line width in image pixels, about 1.5 px on the page whatever the image size. */
export function lineWidth(width: number, height: number): number {
  return Math.max(width, height) / 220;
}

function dashOf(style: LineStyle, w: number): string {
  return style.dash.map((d) => +(d * w).toFixed(2)).join(" ");
}

function drawRegion(region: Region, overlay: Overlay, w: number): SVGGElement {
  const style = LINE_STYLE[region];
  const g = svg("g", { class: `overlay overlay--${style.ink}`, "data-region": region });
  const lineAttrs = (cls: string, width: number) => ({
    class: cls,
    "stroke-width": +width.toFixed(2),
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    fill: "none",
  });
  if (style.box) {
    const corners = ["tl", "tr", "br", "bl"].map((k) => overlay.points[k]).filter(Boolean);
    const points = corners.map(([x, y]) => `${x},${y}`).join(" ");
    g.append(
      svg("polygon", { ...lineAttrs("overlay__under", w * 2.4), points }),
      svg("polygon", { ...lineAttrs("overlay__line", w), points, "stroke-dasharray": dashOf(style, w) }),
    );
    return g;
  }
  const used: string[] = [];
  for (const [a, b] of overlay.lines) {
    const p = overlay.points[a];
    const q = overlay.points[b];
    if (!p || !q) continue;
    const ends = { x1: p[0], y1: p[1], x2: q[0], y2: q[1] };
    g.append(
      svg("line", { ...lineAttrs("overlay__under", w * 2.4), ...ends }),
      svg("line", { ...lineAttrs("overlay__line", w), ...ends, "stroke-dasharray": dashOf(style, w) }),
    );
    for (const name of [a, b]) if (!used.includes(name)) used.push(name);
  }
  for (const name of used) {
    const [cx, cy] = overlay.points[name];
    g.append(svg("circle", { class: "overlay__dot", cx, cy, r: +(w * 1.7).toFixed(2), "stroke-width": +(w * 0.7).toFixed(2) }));
  }
  return g;
}

/** The image inside the arch, with an optional SVG over it. */
export function archImage(url: string, alt: string, layer?: SVGSVGElement): HTMLElement {
  const img = el("img", { class: "arch__image", attrs: { src: url, alt, decoding: "async" } });
  return el("div", { class: "arch-wrap" }, [el("div", { class: "arch" }, [img, layer])]);
}

/** A small sample of a line style for the legend. */
function legendSample(style: LineStyle): SVGSVGElement {
  const w = 2;
  const sample = svg("svg", { class: `legend__sample overlay--${style.ink}`, viewBox: "0 0 44 14", "aria-hidden": "true" });
  if (style.box) {
    const pts = "4,2 40,2 40,12 4,12";
    sample.append(
      svg("polygon", { class: "overlay__under", points: pts, "stroke-width": w * 2.4, fill: "none" }),
      svg("polygon", { class: "overlay__line", points: pts, "stroke-width": w, fill: "none", "stroke-linecap": "round", "stroke-dasharray": dashOf(style, w) }),
    );
  } else {
    sample.append(
      svg("line", { class: "overlay__under", x1: 3, y1: 7, x2: 41, y2: 7, "stroke-width": w * 2.4, "stroke-linecap": "round" }),
      svg("line", { class: "overlay__line", x1: 3, y1: 7, x2: 41, y2: 7, "stroke-width": w, "stroke-linecap": "round", "stroke-dasharray": dashOf(style, w) }),
    );
  }
  return sample;
}

/**
 * One image with the lines of every region measured on it, and the legend under it.
 * regions are all the regions of the result; those whose overlay is on another image,
 * or that have none, are left out.
 */
export function overlayFigure(name: ImageName, image: ShownImage, regions: RegionResult[], i18n: I18n): HTMLElement {
  const drawn = regions.filter((r) => r.overlay !== null && r.overlay.image === name);
  const w = lineWidth(image.width, image.height);
  const groups = LEGEND[name].filter((entry) => drawn.some((r) => entry.regions.includes(r.region)));
  const imageName = i18n.t(IMAGE_TEXT[name]);
  const label = groups.length
    ? i18n.t("ui.overlay.label", { image: imageName, regions: i18n.list(groups.map((g) => i18n.t(g.text))) })
    : imageName;
  const layer = svg("svg", {
    class: "overlay-layer",
    viewBox: `0 0 ${image.width} ${image.height}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
    "aria-label": label,
    "data-image": name,
  });
  for (const r of drawn) layer.append(drawRegion(r.region, r.overlay!, w));

  const legend = el(
    "ul",
    { class: "legend", attrs: { "data-legend": name } },
    groups.map((g) =>
      el("li", { class: "legend__item", attrs: { "data-legend-regions": g.regions.join(" ") } }, [
        legendSample(LINE_STYLE[g.regions[0]]),
        el("span", { text: i18n.t(g.text) }),
      ]),
    ),
  );
  return el("figure", { class: "result-figure", attrs: { "data-figure": name } }, [
    el("div", { class: "result-figure__picture" }, [archImage(image.url, "", layer)]),
    el("figcaption", { class: "result-figure__caption" }, [el("span", { class: "result-figure__name", text: imageName }), groups.length ? legend : null]),
  ]);
}
