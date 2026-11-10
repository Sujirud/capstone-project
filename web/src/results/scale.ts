// A small strip under a measurement that shows where the value sits against the
// cutoffs of the result: a tick and its value at each cutoff, the borderline band
// around it shaded, and a dot at the value. It draws only what the api sent (value and
// cutoffs), so no grading rule lives in the web. It repeats the text next to it, so
// screen readers skip it.
import { svg } from "../dom.ts";
import type { I18n } from "../i18n.ts";
import type { Measurement } from "../types.ts";
import { formatCutoff } from "./text.ts";

const WIDTH = 300;
const HEIGHT = 42;
const LEFT = 18;
const RIGHT = WIDTH - 18;
const AXIS_Y = 14;

export interface Domain {
  low: number;
  high: number;
}

/** The value range of the strip: the cutoffs with room on both sides, and the value. */
export function scaleDomain(m: Measurement): Domain | null {
  if (m.cutoffs.length === 0) return null;
  const values = m.cutoffs.map((c) => c.value);
  const first = Math.min(...values);
  const last = Math.max(...values);
  const widest = Math.max(...m.cutoffs.map((c) => c.band));
  const pad = Math.max((last - first) * 0.4, widest * 3, Math.abs(last) * 0.02, 1e-6);
  return {
    low: Math.min(first - pad, m.value - pad * 0.3),
    high: Math.max(last + pad, m.value + pad * 0.3),
  };
}

export function measureScale(m: Measurement, i18n: I18n): SVGSVGElement | null {
  const domain = scaleDomain(m);
  if (!domain) return null;
  const x = (v: number) => +(LEFT + ((v - domain.low) / (domain.high - domain.low)) * (RIGHT - LEFT)).toFixed(2);
  const strip = svg("svg", {
    class: "scale",
    viewBox: `0 0 ${WIDTH} ${HEIGHT}`,
    "aria-hidden": "true",
    focusable: "false",
    "data-scale": m.name,
  });
  strip.append(svg("line", { class: "scale__axis", x1: LEFT, y1: AXIS_Y, x2: RIGHT, y2: AXIS_Y }));
  for (const c of m.cutoffs) {
    const a = x(c.value - c.band);
    const b = x(c.value + c.band);
    strip.append(
      svg("rect", { class: "scale__band", x: a, y: AXIS_Y - 5, width: +(b - a).toFixed(2), height: 10, rx: 2 }),
      svg("line", { class: "scale__tick", x1: x(c.value), y1: AXIS_Y - 7, x2: x(c.value), y2: AXIS_Y + 7 }),
    );
    const label = svg("text", { class: "scale__label", x: x(c.value), y: HEIGHT - 4, "text-anchor": "middle" });
    label.textContent = formatCutoff(c.value, m.unit, i18n);
    strip.append(label);
  }
  strip.append(svg("circle", { class: "scale__value", cx: x(m.value), cy: AXIS_Y, r: 5.5, "data-value": m.value }));
  return strip;
}
