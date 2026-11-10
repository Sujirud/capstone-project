// The outlines of the camera guides (web.md sections 3.2, 3.3 and 6.3): where they are
// in the frame and how wide the frontal head is, which the "too close" check relies on.
import { describe, expect, it } from "vitest";
import { GUIDE_FRAME_CENTER_Y } from "../src/constants.ts";
import { guideFrameWidthPx } from "../src/guidance.ts";
import { CHIN_V, FRONT_EAR_ROOT, HEAD_TOP_V, guideLayer, guideOutline, type GuideKind } from "../src/screens/guide_layer.ts";

const FRAMES = [
  { width: 1920, height: 1080 },
  { width: 1280, height: 720 },
  { width: 3840, height: 2160 },
];

/** The (x, y) end and control points of a path of M, C, L and Z commands. */
function points(d: string): [number, number][] {
  const n = [...d.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  return Array.from({ length: n.length / 2 }, (_, i) => [n[2 * i], n[2 * i + 1]]);
}

// the paths are written to 0.01 px, so a point and its mirror may round apart by that much
const near = (a: [number, number], b: [number, number]) => Math.abs(a[0] - b[0]) <= 0.011 && Math.abs(a[1] - b[1]) <= 0.011;

describe("the guide outlines", () => {
  for (const size of FRAMES) {
    const w = guideFrameWidthPx(size);
    const cx = size.width / 2;
    const cy = GUIDE_FRAME_CENTER_Y * size.height;

    it(`makes the frontal head as wide at the ears as the check, on ${size.width} x ${size.height}`, () => {
      const all = points(guideOutline("frontal", size).outline);
      expect(FRONT_EAR_ROOT[0]).toBe(0.5); // W/2 each side: W between the ears
      for (const x of [cx - w / 2, cx + w / 2]) expect(all.some((p) => near(p, [x, cy + FRONT_EAR_ROOT[1] * w]))).toBe(true);
    });

    it(`is left-right symmetric from the front, on ${size.width} x ${size.height}`, () => {
      const { outline, details } = guideOutline("frontal", size);
      for (const d of [outline, ...details]) {
        const all = points(d);
        for (const p of all) expect(all.some((q) => near(q, [2 * cx - p[0], p[1]]))).toBe(true);
      }
    });

    for (const kind of ["frontal", "profile"] as GuideKind[]) {
      it(`puts the ${kind} head 1.5 W high around GUIDE_FRAME_CENTER_Y, inside the frame, on ${size.width} x ${size.height}`, () => {
        const { outline, details } = guideOutline(kind, size);
        const head = points(outline);
        const top = Math.min(...head.map((p) => p[1]));
        expect(top).toBeCloseTo(cy + HEAD_TOP_V * w, 1);
        expect(top).toBeGreaterThan(0);
        if (kind === "frontal") expect(Math.max(...points(details[0]).map((p) => p[1]))).toBeCloseTo(cy + CHIN_V * w, 1);
        // the head itself, above the neck, is across the middle of the frame
        const face = head.filter((p) => p[1] < cy + 0.6 * w);
        expect(Math.min(...face.map((p) => p[0]))).toBeGreaterThan(0);
        expect(Math.max(...face.map((p) => p[0]))).toBeLessThan(size.width);
        expect(Math.abs((Math.min(...face.map((p) => p[0])) + Math.max(...face.map((p) => p[0]))) / 2 - cx)).toBeLessThan(0.1 * w);
      });

      it(`closes the ${kind} outline below the frame, where the body goes on, on ${size.width} x ${size.height}`, () => {
        const { outline } = guideOutline(kind, size);
        expect(outline).toMatch(/^M.* Z$/);
        const head = points(outline);
        expect(head[0][1]).toBeGreaterThan(size.height);
        expect(head.at(-1)![1]).toBeGreaterThan(size.height);
      });
    }

    it(`turns the profile to the left of the screen, nose first, on ${size.width} x ${size.height}`, () => {
      const head = points(guideOutline("profile", size).outline).filter((p) => p[1] < cy + 0.6 * w);
      const front = head.reduce((a, b) => (b[0] < a[0] ? b : a));
      expect(front[1]).toBeGreaterThan(cy); // the tip of the nose, below the middle of the head
      expect(front[1]).toBeLessThan(cy + 0.3 * w);
    });
  }

  it("draws both kinds with the same parts: the veil with the outline cut out, then solid lines", () => {
    const size = { width: 1920, height: 1080 };
    for (const kind of ["frontal", "profile"] as GuideKind[]) {
      const layer = guideLayer(kind, size);
      const { outline } = guideOutline(kind, size);
      expect(layer.getAttribute("data-guide")).toBe(kind);
      expect(layer.getAttribute("viewBox")).toBe("0 0 1920 1080");
      expect(layer.getAttribute("aria-hidden")).toBe("true");
      const parts = [...layer.children];
      expect(parts.map((c) => c.getAttribute("class"))).toEqual(["camera__veil", "camera__outline", "camera__outline"]);
      expect(parts[0].getAttribute("d")).toBe(`M0 0 H1920 V1080 H0 Z ${outline}`);
      expect(parts[0].getAttribute("fill-rule")).toBe("evenodd");
      expect(parts[1].getAttribute("d")).toBe(outline);
      for (const part of parts.slice(1)) expect(part.getAttribute("stroke-dasharray")).toBeNull();
    }
  });
});
