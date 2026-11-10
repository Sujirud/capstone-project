// metrics.ts: the same IOD, angles and blendshapes as the api (web.md section 4.1).
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  blendshapeValues,
  faceValues,
  faceWidthPx,
  iodPx,
  matrixLayout,
  matrixRows,
  poseAngles,
  poseFromMatrix,
  rotation,
  svd3,
  toTasksMatrix,
} from "../src/metrics.ts";
import { ROOT, readJson, syntheticLandmarks as landmarks } from "./helpers.ts";

interface PoseCase {
  name: string;
  matrix: number[][];
  yaw_deg: number;
  pitch_deg: number;
  roll_deg: number;
}

const POSE = readJson<{ layout: string; tolerance_deg: number; cases: PoseCase[] }>(join(ROOT, "shared", "testdata", "pose_cases.json"));

/** The same rows flattened row by row, the order tasks-vision does not use. */
function rowOrder(rows: number[][]) {
  return { rows: rows.length, columns: rows[0].length, data: rows.flat() };
}

describe("yaw pitch and roll", () => {
  it("are written in rows in pose_cases.json", () => {
    expect(POSE.layout).toBe("rows");
    expect(POSE.cases.length).toBeGreaterThanOrEqual(10);
  });

  it.each(POSE.cases)("match the api within the tolerance after the change of order: $name", (c) => {
    const pose = poseFromMatrix(toTasksMatrix(c.matrix));
    expect(Math.abs(pose.yaw_deg - c.yaw_deg)).toBeLessThanOrEqual(POSE.tolerance_deg);
    expect(Math.abs(pose.pitch_deg - c.pitch_deg)).toBeLessThanOrEqual(POSE.tolerance_deg);
    expect(Math.abs(pose.roll_deg - c.roll_deg)).toBeLessThanOrEqual(POSE.tolerance_deg);
  });

  it("read tasks-vision's data in column order", () => {
    const c = POSE.cases.find((k) => k.name === "yaw_pitch_roll_scaled_moved")!;
    const m = toTasksMatrix(c.matrix);
    expect(m.data[14]).toBe(c.matrix[2][3]); // tz
    expect(matrixRows(m)).toEqual(c.matrix);
    // read in the wrong order, the same matrix gives other angles
    const wrong = poseAngles(matrixRows(rowOrder(c.matrix)));
    expect(Math.abs(wrong.yaw_deg - c.yaw_deg)).toBeGreaterThan(1);
  });

  it("tell the two orders apart by where the large negative z translation is", () => {
    const moved = POSE.cases.filter((c) => c.matrix[2][3] < -1);
    expect(moved.length).toBeGreaterThan(0);
    for (const c of moved) {
      expect(matrixLayout(toTasksMatrix(c.matrix)), c.name).toBe("columns");
      expect(matrixLayout(rowOrder(c.matrix)), c.name).toBe("rows");
    }
    const still = POSE.cases.find((c) => c.name === "identity")!;
    expect(matrixLayout(toTasksMatrix(still.matrix))).toBeNull(); // no translation: cannot tell
    expect(matrixLayout({ rows: 3, columns: 3, data: [1, 0, 0, 0, 1, 0, 0, 0, 1] })).toBeNull();
  });

  it("follow the signs of api.md section 4", () => {
    const by = (name: string) => poseFromMatrix(toTasksMatrix(POSE.cases.find((c) => c.name === name)!.matrix));
    expect(by("yaw_person_left").yaw_deg).toBeGreaterThan(0);
    expect(by("yaw_person_right").yaw_deg).toBeLessThan(0);
    expect(by("pitch_up").pitch_deg).toBeGreaterThan(0);
    expect(by("pitch_down").pitch_deg).toBeLessThan(0);
    expect(by("roll_left_eye_higher").roll_deg).toBeGreaterThan(0);
    expect(by("roll_right_eye_higher").roll_deg).toBeLessThan(0);
  });

  it("refuse a matrix that is not finite or not 4x4 or 3x3", () => {
    expect(() => poseAngles([[1, 0], [0, 1]])).toThrow();
    expect(() => poseAngles([[NaN, 0, 0], [0, 1, 0], [0, 0, 1]])).toThrow();
    expect(() => matrixRows({ rows: 4, columns: 4, data: [1, 2, 3] })).toThrow();
  });
});

describe("making the matrix orthonormal with SVD", () => {
  // Expected values from facemeasure.pose.rotation and pose.angles (numpy) on the same
  // matrices, which are sheared, so the SVD is not a no-op as it is for a pure rotation.
  const NUMPY = [
    {
      matrix: [[1.2, 0.3, -0.1], [0.05, 0.9, 0.2], [0.1, -0.25, 1.1]],
      rotation: [[0.988375309794762, 0.12499676561378151, -0.08654510716499651], [-0.10214330443398585, 0.96758507551781, 0.23096724225417714], [0.11260991229644715, -0.21944231638698838, 0.969104781451056]],
      angles: [-5.103214850787107, 13.354024020136835, -5.900266820699548],
    },
    {
      matrix: [[0.8, -0.4, 0.3], [0.2, 1.3, -0.1], [-0.3, 0.15, 0.7]],
      rotation: [[0.8982514528170659, -0.24394596297948265, 0.3655607947497149], [0.27958384530681313, 0.9589676736945583, -0.04705182517504267], [-0.3390828821328, 0.1444692630107166, 0.9295974564776164]],
      angles: [21.467051517835934, -2.69686671310777, 17.28893271839972],
    },
    {
      // det < 0: the smallest singular direction is flipped so the result is a rotation
      matrix: [[-1.0, 0.2, 0.1], [0.1, 1.0, 0.3], [0.2, -0.1, 0.9]],
      rotation: [[-0.6472969029421446, 0.6019121176167794, -0.4676628295124515], [-0.22219080771564187, 0.4378956484091235, 0.8711364107136289], [0.729135079671921, 0.6677942825230222, -0.1497091574399632]],
      angles: [-107.75101187001178, 60.590967483276444, -161.0547299426072],
    },
    {
      matrix: [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-14]], // rank 2
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      angles: [0, 0, 0],
    },
  ];

  it.each(NUMPY.map((c, i) => ({ ...c, i })))("gives the rotation of numpy, case $i", (c) => {
    const r = rotation(c.matrix);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(r[i][j]).toBeCloseTo(c.rotation[i][j], 9);
    const p = poseAngles(c.matrix);
    expect(Math.abs(p.yaw_deg - c.angles[0])).toBeLessThan(1e-6);
    expect(Math.abs(p.pitch_deg - c.angles[1])).toBeLessThan(1e-6);
    expect(Math.abs(p.roll_deg - c.angles[2])).toBeLessThan(1e-6);
  });

  it("finds singular values from the largest, with a = u s v^T", () => {
    const a = NUMPY[1].matrix;
    const { u, s, v } = svd3(a);
    expect(s[0]).toBeGreaterThanOrEqual(s[1]);
    expect(s[1]).toBeGreaterThanOrEqual(s[2]);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const back = u[i][0] * s[0] * v[j][0] + u[i][1] * s[1] * v[j][1] + u[i][2] * s[2] * v[j][2];
        expect(back).toBeCloseTo(a[i][j], 12);
      }
    }
  });
});

describe("the IOD", () => {
  const eyes = landmarks({ 33: [0.4, 0.45], 133: [0.46, 0.46], 362: [0.54, 0.44], 263: [0.6, 0.43] });

  it("is the distance between the midpoints of 33-133 and 362-263, in pixels of the frame", () => {
    // centers (0.43, 0.455) and (0.57, 0.435) of a 1920 x 1080 frame
    expect(iodPx(eyes, { width: 1920, height: 1080 })).toBeCloseTo(Math.hypot(0.14 * 1920, 0.02 * 1080), 9);
  });

  it("scales x by the width and y by the height of the image that is sent", () => {
    const full = iodPx(eyes, { width: 3840, height: 2160 });
    const preview = iodPx(eyes, { width: 960, height: 540 });
    expect(full).toBeCloseTo(4 * preview, 9);
    expect(iodPx(eyes, { width: 1920, height: 1920 })).toBeCloseTo(Math.hypot(0.14 * 1920, 0.02 * 1920), 9);
  });

  it("is not changed by the other points", () => {
    const more = landmarks({ 33: [0.4, 0.45], 133: [0.46, 0.46], 362: [0.54, 0.44], 263: [0.6, 0.43], 468: [0.1, 0.1], 473: [0.9, 0.9] });
    expect(iodPx(more, { width: 1920, height: 1080 })).toBe(iodPx(eyes, { width: 1920, height: 1080 }));
  });
});

describe("the face width", () => {
  it("is the distance between points 234 and 454 in pixels", () => {
    const face = landmarks({ 234: [0.44, 0.5], 454: [0.56, 0.52] });
    expect(faceWidthPx(face, { width: 1920, height: 1080 })).toBeCloseTo(Math.hypot(0.12 * 1920, 0.02 * 1080), 9);
  });
});

describe("blendshapes", () => {
  const cat = (scores: Record<string, number>) => Object.entries(scores).map(([categoryName, score]) => ({ categoryName, score }));

  it("take the larger smile and keep both blinks, as quality.blendshape_values", () => {
    const values = blendshapeValues(cat({ mouthSmileLeft: 0.1, mouthSmileRight: 0.35, jawOpen: 0.05, eyeBlinkLeft: 0.2, eyeBlinkRight: 0.6, browInnerUp: 0.9 }));
    expect(values).toEqual({ smile: 0.35, jaw_open: 0.05, blink_left: 0.2, blink_right: 0.6 });
  });

  it("are null when one is missing", () => {
    expect(blendshapeValues(cat({ mouthSmileLeft: 0.1, jawOpen: 0, eyeBlinkLeft: 0, eyeBlinkRight: 0 }))).toBeNull();
  });

  it("go with the other values of the face", () => {
    const face = landmarks({ 33: [0.4, 0.45], 133: [0.46, 0.46], 362: [0.54, 0.44], 263: [0.6, 0.43] });
    const c = POSE.cases.find((k) => k.name === "pitch_up")!;
    const values = faceValues(face, toTasksMatrix(c.matrix), cat({ mouthSmileLeft: 0, mouthSmileRight: 0, jawOpen: 0, eyeBlinkLeft: 0, eyeBlinkRight: 0 }), { width: 1920, height: 1080 });
    expect(values?.pitch_deg).toBeCloseTo(12, 6);
    expect(values?.iod_px).toBeCloseTo(iodPx(face, { width: 1920, height: 1080 }), 9);
    expect(faceValues(face, undefined, [], { width: 1, height: 1 })).toBeNull();
    expect(faceValues(face.slice(0, 10), toTasksMatrix(c.matrix), [], { width: 1, height: 1 })).toBeNull();
  });
});
