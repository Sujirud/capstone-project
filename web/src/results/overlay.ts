// The images with their measurement lines (web.md section 5.1).
//
// The image and the SVG sit in one container and scale together. The SVG's viewBox is
// the pixel size of the image as sent, so the api's coordinates go in unchanged. The
// container clips the arch. Each region on one image has its own colour and its own
// dash pattern, so they can be told apart without colour: a white line under each
// line, and dots of the line's colour with a white edge, show on every skin tone.
//
// Line widths are in pixels of the image, so they scale with the picture as the window
// changes. Each picture gets the width that is LINE_PX on the screen at the size it is
// shown: the figure of the page at FIGURE_PX wide, the full-screen view at the size it
// opens at (viewer.ts).
//
// The legend under the picture is a row of buttons: pressing one hides or shows the
// lines of its region, and pointing at one (or reaching it with the keyboard) fades
// the lines of the other regions. The figure and its full-screen view share this state
// (figure.ts and viewer.ts).
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

export type LegendGroup = (typeof LEGEND)[ImageName][number];

export const IMAGE_TEXT: Record<ImageName, string> = { frontal: "ui.image.frontal", profile: "ui.image.profile" };

/** Width of a measurement line on the screen, in CSS pixels. */
export const LINE_PX = 2;
/** Width of the image in a figure of the results page on a screen 1440 px wide, in CSS pixels. */
export const FIGURE_PX = 244;
/** In line widths: the white under a line (half a line width of white on each side), and the dots. */
export const UNDER_WIDTH = 2;
export const DOT_RADIUS = 1.7;
export const DOT_EDGE = 0.7;

export interface ShownImage {
  url: string;
  width: number;
  height: number;
}

/** Line width in pixels of an image imageWidth wide, for LINE_PX on the screen when it is shown shownWidth wide. */
export function lineWidth(imageWidth: number, shownWidth: number): number {
  return (LINE_PX * imageWidth) / shownWidth;
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
      svg("polygon", { ...lineAttrs("overlay__under", w * UNDER_WIDTH), points }),
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
      svg("line", { ...lineAttrs("overlay__under", w * UNDER_WIDTH), ...ends }),
      svg("line", { ...lineAttrs("overlay__line", w), ...ends, "stroke-dasharray": dashOf(style, w) }),
    );
    for (const name of [a, b]) if (!used.includes(name)) used.push(name);
  }
  for (const name of used) {
    const [cx, cy] = overlay.points[name];
    g.append(svg("circle", { class: "overlay__dot", cx, cy, r: +(w * DOT_RADIUS).toFixed(2), "stroke-width": +(w * DOT_EDGE).toFixed(2) }));
  }
  return g;
}

/** The SVG over one image, with the lines of the regions drawn on it, w wide in pixels of the image. */
export function overlayLayer(name: ImageName, image: ShownImage, drawn: RegionResult[], w: number): SVGSVGElement {
  const layer = svg("svg", {
    class: "overlay-layer",
    viewBox: `0 0 ${image.width} ${image.height}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
    "data-image": name,
  });
  for (const r of drawn) layer.append(drawRegion(r.region, r.overlay!, w));
  return layer;
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
      svg("polygon", { class: "overlay__under", points: pts, "stroke-width": w * UNDER_WIDTH, fill: "none" }),
      svg("polygon", { class: "overlay__line", points: pts, "stroke-width": w, fill: "none", "stroke-linecap": "round", "stroke-dasharray": dashOf(style, w) }),
    );
  } else {
    sample.append(
      svg("line", { class: "overlay__under", x1: 3, y1: 7, x2: 41, y2: 7, "stroke-width": w * UNDER_WIDTH, "stroke-linecap": "round" }),
      svg("line", { class: "overlay__line", x1: 3, y1: 7, x2: 41, y2: 7, "stroke-width": w, "stroke-linecap": "round", "stroke-dasharray": dashOf(style, w) }),
    );
  }
  return sample;
}

/**
 * Which lines of one image are hidden, and which are pointed at, for every SVG and
 * legend that shows that image: the figure, and the full-screen view while it is open.
 */
export class LineState {
  readonly name: ImageName;
  /** The legend entries of the regions drawn on the image. */
  readonly groups: LegendGroup[];
  readonly i18n: I18n;
  readonly hidden = new Set<Region>();
  private pointed: Region[] | null = null;
  private readonly layers = new Set<SVGSVGElement>();
  private readonly legends = new Set<HTMLElement>();

  constructor(name: ImageName, groups: LegendGroup[], i18n: I18n) {
    this.name = name;
    this.groups = groups;
    this.i18n = i18n;
  }

  attach(layer: SVGSVGElement, legend: HTMLElement | null): void {
    this.layers.add(layer);
    if (legend) this.legends.add(legend);
    this.render();
  }

  detach(layer: SVGSVGElement, legend: HTMLElement | null): void {
    this.layers.delete(layer);
    if (legend) this.legends.delete(legend);
  }

  isShown(group: LegendGroup): boolean {
    return group.regions.some((r) => !this.hidden.has(r));
  }

  toggle(group: LegendGroup): void {
    const show = !this.isShown(group);
    for (const r of group.regions) {
      if (show) this.hidden.delete(r);
      else this.hidden.add(r);
    }
    this.render();
  }

  /** Fade every other region while one is pointed at; null when the pointer or the focus leaves. */
  point(group: LegendGroup | null): void {
    this.pointed = group ? group.regions : null;
    this.render();
  }

  /** The regions shown at full strength while others fade: none when nothing is pointed at or it is hidden. */
  focused(): Region[] | null {
    const p = this.pointed;
    return p && p.some((r) => !this.hidden.has(r)) ? p : null;
  }

  /** The text of the SVG: the image and the regions whose lines are shown. */
  label(): string {
    const image = this.i18n.t(IMAGE_TEXT[this.name]);
    const shown = this.groups.filter((g) => this.isShown(g));
    return shown.length ? this.i18n.t("ui.overlay.label", { image, regions: this.i18n.list(shown.map((g) => this.i18n.t(g.text))) }) : image;
  }

  private render(): void {
    const focused = this.focused();
    const label = this.label();
    for (const layer of this.layers) {
      layer.setAttribute("aria-label", label);
      layer.toggleAttribute("data-focus", focused !== null);
      for (const g of layer.querySelectorAll<SVGGElement>("g[data-region]")) {
        const region = g.dataset.region as Region;
        g.toggleAttribute("data-hidden", this.hidden.has(region));
        g.toggleAttribute("data-focus", focused?.includes(region) ?? false);
      }
    }
    for (const legend of this.legends) {
      for (const button of legend.querySelectorAll<HTMLButtonElement>("button[data-legend-toggle]")) {
        const group = this.groups.find((g) => g.regions.join(" ") === button.dataset.legendToggle)!;
        button.setAttribute("aria-pressed", String(this.isShown(group)));
      }
    }
  }
}

/** Whether the focus of target came from the keyboard; a click focuses buttons in some browsers too. */
function focusFromKeyboard(target: Element): boolean {
  try {
    return target.matches(":focus-visible");
  } catch {
    return true;
  }
}

/** The legend of one image: a button for each region drawn on it, to hide or show its lines. */
export function legendList(state: LineState): HTMLUListElement {
  const { i18n } = state;
  return el(
    "ul",
    { class: "legend", attrs: { "data-legend": state.name, "aria-label": i18n.t("ui.overlay.lines") } },
    state.groups.map((g) => {
      const toggle = el("button", { class: "legend__toggle", attrs: { type: "button", "aria-pressed": "true", "data-legend-toggle": g.regions.join(" ") } }, [
        legendSample(LINE_STYLE[g.regions[0]]),
        el("span", { class: "legend__name", text: i18n.t(g.text) }),
      ]);
      toggle.addEventListener("click", () => state.toggle(g));
      toggle.addEventListener("pointerenter", (e) => {
        if (e.pointerType === "mouse") state.point(g);
      });
      toggle.addEventListener("pointerleave", () => state.point(null));
      toggle.addEventListener("focus", () => {
        if (focusFromKeyboard(toggle)) state.point(g);
      });
      toggle.addEventListener("blur", () => state.point(null));
      return el("li", { class: "legend__item", attrs: { "data-legend-regions": g.regions.join(" ") } }, [toggle]);
    }),
  );
}
