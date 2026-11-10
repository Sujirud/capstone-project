// The guides over the live camera (web.md sections 3.2, 3.3 and 6.3): the outline of a
// head with its neck and shoulders, from the front for the frontal photo and from the
// side for the profile, drawn by hand for this project and not traced from a photo of
// anyone. Both have one style: a solid --marble line, with the --cocoa veil outside.
//
// The outlines are in units of the guide width W (guideFrameWidthPx), u to the right and
// v down from the middle of the head. The frontal head is W wide at the roots of the
// ears, the width the "too close" check uses for the face between points 234 and 454.
// Both heads are 1.5 W from the top to the chin, centred across the frame and at
// GUIDE_FRAME_CENTER_Y of its height, so the profile is taken from the same distance as
// the frontal photo. The profile looks to the left of the screen; either side will do.
// BOTTOM is below the frame, where the body goes on.
import { GUIDE_FRAME_CENTER_Y } from "../constants.ts";
import { svg } from "../dom.ts";
import { guideFrameWidthPx } from "../guidance.ts";
import type { Size } from "../images.ts";

type Pt = readonly [number, number];
/** A cubic Bézier when it has control points, else a straight line. */
interface Seg {
  readonly to: Pt;
  readonly c?: readonly [Pt, Pt];
}
interface Run {
  readonly start: Pt;
  readonly segs: readonly Seg[];
}

const BOTTOM = Number.POSITIVE_INFINITY;

const curve = (c1: Pt, c2: Pt, to: Pt): Seg => ({ to, c: [c1, c2] });
const line = (to: Pt): Seg => ({ to });

/** The top of both heads, and the chin: 1.5 W apart. */
export const HEAD_TOP_V = -0.75;
export const CHIN_V = 0.755;
/** Where the right ear meets the frontal head: W/2 from the middle, so the head is W wide there. */
export const FRONT_EAR_ROOT: Pt = [0.5, -0.08];

/** The right half of the frontal head, from the top down the side to the shoulder. */
const FRONT_HALF: Run = {
  start: [0, HEAD_TOP_V],
  segs: [
    curve([0.29, -0.75], [0.52, -0.55], [0.52, -0.25]), // the crown
    curve([0.52, -0.17], [0.51, -0.12], FRONT_EAR_ROOT), // the temple, to the top of the ear
    curve([0.56, -0.16], [0.62, -0.08], [0.6, 0.06]), // the ear
    curve([0.59, 0.18], [0.56, 0.3], [0.49, 0.3]),
    curve([0.48, 0.4], [0.44, 0.47], [0.39, 0.52]), // the cheek, to the angle of the jaw
    curve([0.37, 0.55], [0.345, 0.575], [0.32, 0.6]), // the jaw, to the neck
    curve([0.31, 0.75], [0.31, 0.88], [0.33, 0.98]), // the neck
    curve([0.45, 1.08], [0.9, 1.12], [1.15, 1.2]), // the slope of the shoulder
    curve([1.4, 1.27], [1.5, 1.45], [1.52, 1.7]),
    line([1.55, BOTTOM]),
  ],
};

/** The right half of the jaw in front of the neck, from the neck to the middle of the chin. */
const FRONT_CHIN_HALF: Run = {
  start: [0.32, 0.6],
  segs: [curve([0.26, 0.67], [0.18, 0.73], [0.1, 0.746]), curve([0.05, 0.754], [0.02, CHIN_V], [0, CHIN_V])],
};

/** The profile from the top down the face, the throat and the chest. */
const SIDE_FRONT: Run = {
  start: [0.02, HEAD_TOP_V],
  segs: [
    curve([-0.3, -0.75], [-0.52, -0.6], [-0.56, -0.35]), // the forehead
    curve([-0.58, -0.25], [-0.6, -0.17], [-0.59, -0.12]), // the brow
    curve([-0.585, -0.095], [-0.56, -0.08], [-0.56, -0.04]), // the root of the nose
    curve([-0.56, 0.0], [-0.68, 0.12], [-0.74, 0.2]), // the dorsum, to the tip
    curve([-0.765, 0.25], [-0.7, 0.3], [-0.61, 0.31]), // under the nose
    curve([-0.62, 0.35], [-0.64, 0.39], [-0.625, 0.43]), // the lips, kept smooth: they are small on screen
    curve([-0.615, 0.46], [-0.63, 0.5], [-0.615, 0.53]),
    curve([-0.6, 0.57], [-0.565, 0.58], [-0.565, 0.62]), // the fold under the lip, then the chin
    curve([-0.565, 0.67], [-0.6, 0.72], [-0.54, 0.76]),
    curve([-0.49, 0.79], [-0.35, 0.79], [-0.26, 0.83]), // under the chin, to the throat
    curve([-0.21, 0.86], [-0.22, 1.0], [-0.27, 1.12]), // the neck
    curve([-0.33, 1.25], [-0.46, 1.33], [-0.5, 1.55]), // the chest
    line([-0.56, BOTTOM]),
  ],
};

/** The profile from the top down the back of the head, the neck and the back. */
const SIDE_BACK: Run = {
  start: [0.02, HEAD_TOP_V],
  segs: [
    curve([0.4, -0.75], [0.64, -0.48], [0.66, -0.15]), // the back of the head
    curve([0.68, 0.1], [0.6, 0.3], [0.48, 0.42]),
    curve([0.45, 0.46], [0.44, 0.51], [0.44, 0.56]), // the nape
    curve([0.44, 0.75], [0.47, 0.92], [0.56, 1.02]), // the neck
    curve([0.68, 1.14], [0.78, 1.3], [0.8, 1.5]), // the back
    line([0.84, BOTTOM]),
  ],
};

const SIDE_EAR: Run = {
  start: [0.06, -0.06],
  segs: [curve([0.12, -0.14], [0.26, -0.12], [0.27, 0.04]), curve([0.28, 0.17], [0.2, 0.26], [0.14, 0.3]), curve([0.1, 0.33], [0.06, 0.3], [0.07, 0.26])],
};

const mirror = (p: Pt): Pt => [-p[0], p[1]];

function mirrored(run: Run): Run {
  return { start: mirror(run.start), segs: run.segs.map((s) => (s.c ? curve(mirror(s.c[0]), mirror(s.c[1]), mirror(s.to)) : line(mirror(s.to)))) };
}

/** The same run drawn from its end back to its start. */
function reversed(run: Run): Run {
  const points = [run.start, ...run.segs.map((s) => s.to)];
  const segs = run.segs.map((s, i) => (s.c ? curve(s.c[1], s.c[0], points[i]) : line(points[i]))).reverse();
  return { start: points[points.length - 1], segs };
}

/** Runs joined end to start into one path, in pixels of the frame. */
function pathOf(size: Size, runs: Run[], close: boolean): string {
  const w = guideFrameWidthPx(size);
  const cx = size.width / 2;
  const cy = GUIDE_FRAME_CENTER_Y * size.height;
  const f = (n: number) => Number(n.toFixed(2));
  const at = ([u, v]: Pt) => `${f(cx + u * w)} ${f(v === BOTTOM ? size.height + w : cy + v * w)}`; // BOTTOM: past the frame
  const parts = [`M${at(runs[0].start)}`];
  for (const run of runs) for (const s of run.segs) parts.push(s.c ? `C${at(s.c[0])} ${at(s.c[1])} ${at(s.to)}` : `L${at(s.to)}`);
  if (close) parts.push("Z");
  return parts.join(" ");
}

export interface GuideOutline {
  /** The closed outline: inside it is clear, outside it is veiled. */
  outline: string;
  /** Lines inside the outline: the chin in front of the neck, the ear of the profile. */
  details: string[];
}

export type GuideKind = "frontal" | "profile";

/** The outline of web.md section 6.3 in pixels of the frame. */
export function guideOutline(kind: GuideKind, size: Size): GuideOutline {
  if (kind === "frontal") {
    return {
      outline: pathOf(size, [reversed(mirrored(FRONT_HALF)), FRONT_HALF], true),
      details: [pathOf(size, [mirrored(FRONT_CHIN_HALF), reversed(FRONT_CHIN_HALF)], false)],
    };
  }
  return { outline: pathOf(size, [reversed(SIDE_FRONT), SIDE_BACK], true), details: [pathOf(size, [SIDE_EAR], false)] };
}

/** The guide over the video: the veil outside the outline, then the lines. */
export function guideLayer(kind: GuideKind, size: Size): SVGSVGElement {
  const { outline, details } = guideOutline(kind, size);
  return svg("svg", { class: "camera__overlay", viewBox: `0 0 ${size.width} ${size.height}`, preserveAspectRatio: "none", "aria-hidden": "true", "data-guide": kind }, [
    svg("path", { class: "camera__veil", d: `M0 0 H${size.width} V${size.height} H0 Z ${outline}`, "fill-rule": "evenodd" }),
    svg("path", { class: "camera__outline", d: outline, "data-outline": "" }),
    ...details.map((d) => svg("path", { class: "camera__outline", d })),
  ]);
}
