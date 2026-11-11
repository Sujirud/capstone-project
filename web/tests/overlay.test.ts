// The measurement lines of the results page, their legend and the full-screen view (web.md section 5.1).
import { afterEach, describe, expect, it, vi } from "vitest";
import { DOT_EDGE, DOT_RADIUS, FIGURE_PX, LEGEND, LINE_PX, UNDER_WIDTH, lineWidth } from "../src/results/overlay.ts";
import { shownWidth } from "../src/results/viewer.ts";
import type { ImageName, Region } from "../src/types.ts";
import { WEB, fixture, readText, renderResult, th } from "./helpers.ts";
import { join } from "node:path";

afterEach(() => {
  vi.restoreAllMocks();
});

function figure(root: HTMLElement, name: ImageName): HTMLElement {
  return root.querySelector<HTMLElement>(`[data-figure="${name}"]`)!;
}

function layerOf(scope: Element, name: ImageName): SVGSVGElement {
  return scope.querySelector<SVGSVGElement>(`svg.overlay-layer[data-image="${name}"]`)!;
}

function group(layer: SVGSVGElement, region: Region): SVGGElement {
  return layer.querySelector<SVGGElement>(`g[data-region="${region}"]`)!;
}

function toggleOf(scope: Element, regions: Region[]): HTMLButtonElement {
  return scope.querySelector<HTMLButtonElement>(`button[data-legend-toggle="${regions.join(" ")}"]`)!;
}

function hiddenRegions(layer: SVGSVGElement): string[] {
  return [...layer.querySelectorAll<SVGGElement>("g[data-region][data-hidden]")].map((g) => g.dataset.region!);
}

function pointer(type: string, pointerType: string): PointerEvent {
  return new PointerEvent(type, { pointerType, bubbles: false });
}

const UNDEREYE: Region[] = ["undereye_right", "undereye_left"];

describe("the line width", () => {
  it("is LINE_PX on the screen whatever the size of the image", () => {
    for (const width of [640, 1280, 1920, 3840]) {
      expect(lineWidth(width, FIGURE_PX) * (FIGURE_PX / width)).toBeCloseTo(LINE_PX, 10);
      expect(lineWidth(width, 1376) * (1376 / width)).toBeCloseTo(LINE_PX, 10);
    }
  });

  it("is set by the width of the image, which the arch holds, so portrait and landscape images match", () => {
    const portrait = fixture("female_clear");
    const landscape = fixture("webcam_1080p");
    expect(portrait.images.frontal!.height).toBeGreaterThan(portrait.images.frontal!.width);
    expect(landscape.images.frontal).toMatchObject({ width: 1920, height: 1080 });
    for (const f of [portrait, landscape]) {
      const { root } = renderResult(f);
      const w = lineWidth(f.images.frontal!.width, FIGURE_PX);
      const jaw = group(layerOf(figure(root, "frontal"), "frontal"), "jaw");
      expect(Number(jaw.querySelector(".overlay__line")!.getAttribute("stroke-width"))).toBeCloseTo(w, 2);
    }
  });

  it("puts a white line twice as wide under each line, and dots of the line's colour with a white edge", () => {
    const f = fixture("female_clear");
    const { root } = renderResult(f);
    const w = lineWidth(f.images.frontal!.width, FIGURE_PX);
    const layer = layerOf(figure(root, "frontal"), "frontal");
    for (const g of layer.querySelectorAll<SVGGElement>("g[data-region]")) {
      for (const under of g.querySelectorAll(".overlay__under")) expect(Number(under.getAttribute("stroke-width"))).toBeCloseTo(w * UNDER_WIDTH, 2);
      for (const dot of g.querySelectorAll(".overlay__dot")) {
        expect(Number(dot.getAttribute("r"))).toBeCloseTo(w * DOT_RADIUS, 2);
        expect(Number(dot.getAttribute("stroke-width"))).toBeCloseTo(w * DOT_EDGE, 2);
      }
    }
    expect(UNDER_WIDTH).toBe(2); // half a line width of white on each side
    // the colour of a region is one variable, for its line and its dots
    const css = readText(join(WEB, "src", "styles", "components.css"));
    expect(css).toMatch(/\.overlay__line \{[^}]*stroke: var\(--ink\);/);
    expect(css).toMatch(/\.overlay__dot \{[^}]*fill: var\(--ink\);[^}]*stroke: var\(--white\);/);
  });
});

describe("the legend", () => {
  it("has a pressed button for each region drawn, named by the region", () => {
    const f = fixture("female_clear");
    const { root } = renderResult(f);
    for (const name of ["frontal", "profile"] as const) {
      const legend = figure(root, name).querySelector<HTMLElement>(`[data-legend="${name}"]`)!;
      expect(legend.getAttribute("aria-label")).toBe(th("ui.overlay.lines"));
      const buttons = [...legend.querySelectorAll<HTMLButtonElement>("button")];
      expect(buttons.map((b) => b.dataset.legendToggle)).toEqual(LEGEND[name].map((e) => e.regions.join(" ")));
      for (const [i, b] of buttons.entries()) {
        expect(b.getAttribute("type")).toBe("button");
        expect(b.getAttribute("aria-pressed")).toBe("true");
        expect(b.textContent).toBe(th(LEGEND[name][i].text));
      }
    }
  });

  it("hides and shows the lines of a region, both under-eye sides together", () => {
    const { root } = renderResult(fixture("female_clear"));
    const fig = figure(root, "frontal");
    const layer = layerOf(fig, "frontal");
    const all = th("ui.overlay.label", { image: th("ui.image.frontal"), regions: "" });
    expect(layer.getAttribute("aria-label")!.startsWith(all.trim())).toBe(true);
    expect(layer.getAttribute("aria-label")).toContain(th("region.undereye.name"));

    const undereye = toggleOf(fig, UNDEREYE);
    undereye.click();
    expect(undereye.getAttribute("aria-pressed")).toBe("false");
    expect(hiddenRegions(layer).sort()).toEqual([...UNDEREYE].sort());
    expect(layer.getAttribute("aria-label")).not.toContain(th("region.undereye.name"));
    expect(layer.getAttribute("aria-label")).toContain(th("region.jaw.name"));

    undereye.click();
    expect(undereye.getAttribute("aria-pressed")).toBe("true");
    expect(hiddenRegions(layer)).toEqual([]);
  });

  it("names the image alone when every line is hidden", () => {
    const { root } = renderResult(fixture("female_clear"));
    const fig = figure(root, "profile");
    for (const b of fig.querySelectorAll<HTMLButtonElement>("[data-legend-toggle]")) b.click();
    const layer = layerOf(fig, "profile");
    expect(hiddenRegions(layer).sort()).toEqual(["chin", "nose_dorsum"]);
    expect(layer.getAttribute("aria-label")).toBe(th("ui.image.profile"));
  });

  it("fades the other regions while the mouse points at one, but not for a touch or a hidden region", () => {
    const { root } = renderResult(fixture("female_clear"));
    const fig = figure(root, "frontal");
    const layer = layerOf(fig, "frontal");
    const jaw = toggleOf(fig, ["jaw"]);
    const focused = () => [...layer.querySelectorAll<SVGGElement>("g[data-focus]")].map((g) => g.dataset.region);

    jaw.dispatchEvent(pointer("pointerenter", "mouse"));
    expect(layer.hasAttribute("data-focus")).toBe(true);
    expect(focused()).toEqual(["jaw"]);
    jaw.dispatchEvent(pointer("pointerleave", "mouse"));
    expect(layer.hasAttribute("data-focus")).toBe(false);
    expect(focused()).toEqual([]);

    jaw.dispatchEvent(pointer("pointerenter", "touch"));
    expect(layer.hasAttribute("data-focus")).toBe(false);

    // pressed while pointed at: once hidden, nothing fades; shown again, the others fade again
    jaw.dispatchEvent(pointer("pointerenter", "mouse"));
    jaw.click();
    expect(layer.hasAttribute("data-focus")).toBe(false);
    jaw.click();
    expect(focused()).toEqual(["jaw"]);

    const undereye = toggleOf(fig, UNDEREYE);
    undereye.dispatchEvent(pointer("pointerenter", "mouse"));
    expect(focused().sort()).toEqual([...UNDEREYE].sort());
  });
});

describe("the full-screen view", () => {
  it("opens the whole image with its lines and legend, from the button or the picture", () => {
    const f = fixture("female_clear");
    const { root } = renderResult(f);
    for (const name of ["frontal", "profile"] as const) {
      const fig = figure(root, name);
      const open = fig.querySelector<HTMLButtonElement>('[data-action="viewer-open"]')!;
      expect(open.textContent).toBe(th("ui.overlay.open"));
      expect(document.getElementById(open.getAttribute("aria-describedby")!)!.textContent).toBe(th(`ui.image.${name}`));
      open.click();

      const dialog = fig.querySelector<HTMLDialogElement>(`dialog[data-viewer="${name}"]`)!;
      expect(dialog.open).toBe(true);
      expect(document.getElementById(dialog.getAttribute("aria-labelledby")!)!.textContent).toBe(th(`ui.image.${name}`));
      expect(dialog.querySelector(".arch")).toBeNull(); // the whole image, not cut by the arch
      expect(dialog.querySelector("img")!.getAttribute("src")).toBe(fig.querySelector(".arch img")!.getAttribute("src"));
      const size = f.images[name]!;
      const layer = layerOf(dialog, name);
      expect(layer.getAttribute("viewBox")).toBe(`0 0 ${size.width} ${size.height}`);
      const regions = (l: SVGSVGElement) => [...l.querySelectorAll<SVGGElement>("g[data-region]")].map((g) => g.dataset.region);
      expect(regions(layer)).toEqual(regions(layerOf(fig.querySelector(".arch")!, name)));
      expect(dialog.querySelectorAll("[data-legend-toggle]").length).toBe(fig.querySelectorAll(".result-figure__caption [data-legend-toggle]").length);

      dialog.querySelector<HTMLButtonElement>('[data-action="viewer-close"]')!.click();
      expect(fig.querySelector("dialog")).toBeNull();
      expect(document.activeElement).toBe(open);

      fig.querySelector<HTMLElement>(`[data-viewer-open="${name}"]`)!.click();
      expect(fig.querySelector<HTMLDialogElement>("dialog")!.open).toBe(true);
      fig.querySelector<HTMLDialogElement>("dialog")!.close();
      expect(fig.querySelector("dialog")).toBeNull();
    }
  });

  it("draws lines LINE_PX wide at the size the image opens at", () => {
    const f = fixture("webcam_1080p");
    const { root } = renderResult(f);
    const shown = 1376; // a 1440 px screen
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({ width: shown, height: shown * (1080 / 1920) } as DOMRect);
    const fig = figure(root, "frontal");
    fig.querySelector<HTMLButtonElement>('[data-action="viewer-open"]')!.click();
    const line = group(layerOf(fig.querySelector("dialog")!, "frontal"), "jaw").querySelector(".overlay__line")!;
    const w = Number(line.getAttribute("stroke-width"));
    expect(w).toBeCloseTo(lineWidth(1920, shown), 2);
    expect(w * (shown / 1920)).toBeCloseTo(LINE_PX, 1);
    // on the page the same image has lines LINE_PX wide at FIGURE_PX
    const onPage = group(layerOf(fig.querySelector(".arch")!, "frontal"), "jaw").querySelector(".overlay__line")!;
    expect(Number(onPage.getAttribute("stroke-width"))).toBeCloseTo(lineWidth(1920, FIGURE_PX), 2);
  });

  it("measures the width of the image once shown, and falls back before there is a layout", () => {
    const box = document.createElement("div");
    expect(shownWidth(box, 1920)).toBe(1920); // happy-dom has no layout
    vi.spyOn(box, "getBoundingClientRect").mockReturnValue({ width: 900 } as DOMRect);
    expect(shownWidth(box, 1920)).toBe(900);
  });

  it("shares the hidden lines with the figure, also after it closes", () => {
    const { root } = renderResult(fixture("female_clear"));
    const fig = figure(root, "frontal");
    const pageLayer = layerOf(fig.querySelector(".arch")!, "frontal");
    toggleOf(fig.querySelector(".result-figure__caption")!, ["nose_base"]).click();
    fig.querySelector<HTMLButtonElement>('[data-action="viewer-open"]')!.click();
    const dialog = fig.querySelector<HTMLDialogElement>("dialog")!;
    const viewLayer = layerOf(dialog, "frontal");
    expect(hiddenRegions(viewLayer)).toEqual(["nose_base"]); // opens as the figure was
    expect(toggleOf(dialog, ["nose_base"]).getAttribute("aria-pressed")).toBe("false");

    toggleOf(dialog, ["jaw"]).click();
    expect(hiddenRegions(pageLayer).sort()).toEqual(["jaw", "nose_base"]);
    expect(toggleOf(fig.querySelector(".result-figure__caption")!, ["jaw"]).getAttribute("aria-pressed")).toBe("false");

    toggleOf(dialog, ["jaw"]).dispatchEvent(pointer("pointerenter", "mouse"));
    dialog.close();
    expect(pageLayer.hasAttribute("data-focus")).toBe(false); // nothing stays faded
    toggleOf(fig, ["jaw"]).click();
    expect(hiddenRegions(pageLayer)).toEqual(["nose_base"]);
  });

  it("opens an image without lines too, with no legend", () => {
    const { root } = renderResult(fixture("no_face_no_profile"));
    const fig = figure(root, "frontal");
    expect(fig.querySelector("[data-legend]")).toBeNull();
    fig.querySelector<HTMLButtonElement>('[data-action="viewer-open"]')!.click();
    const dialog = fig.querySelector<HTMLDialogElement>("dialog")!;
    expect(dialog.open).toBe(true);
    expect(dialog.querySelector("[data-legend]")).toBeNull();
    expect(layerOf(dialog, "frontal").getAttribute("aria-label")).toBe(th("ui.image.frontal"));
  });
});
