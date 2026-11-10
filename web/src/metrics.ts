// What the page measures on a frame with the same formulas as the api (web.md section
// 4.1): the IOD, the width of the face, yaw pitch and roll from the facial
// transformation matrix, and the blendshapes. Each step follows the api's code in
// src/facemeasure/: mp_points.py (points and IOD), pose.py (angles) and quality.py
// (blendshapes). tests/metrics.test.ts runs every case of shared/testdata/pose_cases.json.
//
// Pixels are pixels of the image that is sent: a landmark at (x, y) in [0, 1] is at
// (x * width, y * height) of the frame of the stream, which is also the size of the
// captured JPEG (camera.ts), not the size of the preview on the screen.
import type { Matrix } from "@mediapipe/tasks-vision";
import type { Size } from "./images.ts";

export const N_POINTS = 478;
// Eye corners, as in mp_points.py: the eye center is the midpoint of the two corners
export const EYE_PERSON_RIGHT = [33, 133] as const; // outer and inner corner; image left
export const EYE_PERSON_LEFT = [362, 263] as const; // inner and outer corner; image right
// The face edge near eye level, image left and image right: the width of the face
export const CHEEKS = [234, 454] as const;

export interface LandmarkPoint {
  x: number;
  y: number;
}

export interface Pose {
  yaw_deg: number;
  pitch_deg: number;
  roll_deg: number;
}

export interface Blendshapes {
  smile: number;
  jaw_open: number;
  blink_left: number;
  blink_right: number;
}

/** One face, named as quality.frontal of the api (api.md section 2.2), plus the face width. */
export interface FrontalValues extends Pose, Blendshapes {
  iod_px: number;
  face_width_px: number;
}

export const FRONTAL_VALUE_NAMES = [
  "iod_px",
  "face_width_px",
  "yaw_deg",
  "pitch_deg",
  "roll_deg",
  "smile",
  "jaw_open",
  "blink_left",
  "blink_right",
] as const satisfies readonly (keyof FrontalValues)[];

// ---------------------------------------------------------------- points

function pixel(landmarks: readonly LandmarkPoint[], n: number, size: Size): [number, number] {
  const p = landmarks[n];
  if (!p) throw new Error(`no landmark ${n}`);
  return [p.x * size.width, p.y * size.height];
}

function midpoint(landmarks: readonly LandmarkPoint[], pair: readonly [number, number], size: Size): [number, number] {
  const [ax, ay] = pixel(landmarks, pair[0], size);
  const [bx, by] = pixel(landmarks, pair[1], size);
  return [(ax + bx) / 2, (ay + by) / 2];
}

/** Eye centers (image-left eye, image-right eye) = (person's right, person's left), in pixels. */
export function eyeCenters(landmarks: readonly LandmarkPoint[], size: Size): [[number, number], [number, number]] {
  return [midpoint(landmarks, EYE_PERSON_RIGHT, size), midpoint(landmarks, EYE_PERSON_LEFT, size)];
}

/** Distance in pixels between the two eye centers (mp_points.iod). */
export function iodPx(landmarks: readonly LandmarkPoint[], size: Size): number {
  const [[lx, ly], [rx, ry]] = eyeCenters(landmarks, size);
  return Math.hypot(rx - lx, ry - ly);
}

/** Distance in pixels between points 234 and 454; wider than the arch means too close. */
export function faceWidthPx(landmarks: readonly LandmarkPoint[], size: Size): number {
  const [ax, ay] = pixel(landmarks, CHEEKS[0], size);
  const [bx, by] = pixel(landmarks, CHEEKS[1], size);
  return Math.hypot(bx - ax, by - ay);
}

// ---------------------------------------------------------------- the order of Matrix.data
//
// tasks-vision gives the facial transformation matrix as { rows, columns, data }, where
// data is one flat array that does not say its order. It is in COLUMN order
// (column-major): element (row r, column c) is data[c * rows + r], and the translation
// (tx, ty, tz) is data[12], data[13], data[14]. How this was found:
//
// - The translation along z is a large negative number when the face is in front of the
//   camera, since the face sits tens of centimetres away along -z. In column order it is
//   data[14], and data[11] is 0 as part of the bottom row (0, 0, 0, 1). Read in row order
//   it would be the other way round. A frontal face of SCUT-FBP5500 drawn into a
//   1920 x 1080 frame and seen by tasks-vision 1.2.0 in Chrome gave
//   data[14] = -57.67, with data[3], data[7] and data[11] = 0 and data[15] = 1.
// - tasks-vision 1.2.0 copies packed_data of MediaPipe's MatrixData as it is and never
//   reads its layout field, whose default is COLUMN_MAJOR; the C++ graph fills it from
//   an Eigen matrix, which keeps its values in column order.
// - The Python package says the same: mediapipe/tasks/python/components/containers/
//   matrix_c.py reads the data as (cols, rows) and transposes it, which is why the api
//   gets rows and shared/testdata/pose_cases.json is written in rows.
//
// matrixLayout() tells the two orders apart by where tz is; guidance.ts uses it to
// warn once if a frame ever disagrees with MATRIX_LAYOUT.

export type MatrixLayout = "columns" | "rows";
export const MATRIX_LAYOUT: MatrixLayout = "columns";
const ZERO = 1e-4;
const MIN_DEPTH = 1; // centimetres in front of the camera

/** The order of a 4x4 transformation matrix from where its z translation is; null when unclear. */
export function matrixLayout(m: Matrix): MatrixLayout | null {
  if (m.rows !== 4 || m.columns !== 4 || m.data.length !== 16) return null;
  const d = m.data;
  const zero = (...i: number[]) => i.every((k) => Math.abs(d[k]) < ZERO);
  if (d[14] < -MIN_DEPTH && zero(3, 7, 11)) return "columns";
  if (d[11] < -MIN_DEPTH && zero(12, 13, 14)) return "rows";
  return null;
}

/** The matrix as a list of rows, as numpy has it in the api. */
export function matrixRows(m: Matrix, layout: MatrixLayout = MATRIX_LAYOUT): number[][] {
  const { rows, columns, data } = m;
  if (!(rows > 0 && columns > 0 && data.length === rows * columns)) {
    throw new Error(`matrix data has ${data.length} values for ${rows} x ${columns}`);
  }
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: columns }, (_, c) => (layout === "columns" ? data[c * rows + r] : data[r * columns + c])),
  );
}

/** A matrix given as rows, flattened the way tasks-vision does (for tests and checks). */
export function toTasksMatrix(rows: readonly (readonly number[])[]): Matrix {
  const n = rows.length;
  const m = rows[0]?.length ?? 0;
  const data: number[] = [];
  for (let c = 0; c < m; c++) for (let r = 0; r < n; r++) data.push(rows[r][c]);
  return { rows: n, columns: m, data };
}

// ---------------------------------------------------------------- SVD of a 3x3 matrix

type M3 = number[][];

function transpose(a: M3): M3 {
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => a[j][i]));
}

function multiply(a: M3, b: M3): M3 {
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]));
}

function det(a: M3): number {
  return (
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
    a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
    a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
  );
}

/** Eigenvalues and eigenvectors (as columns) of a symmetric 3x3 matrix, by cyclic Jacobi rotations. */
function symmetricEigen(s: M3): { values: number[]; vectors: M3 } {
  const a = s.map((row) => [...row]);
  const v: M3 = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const scale = Math.max(...a.flat().map(Math.abs)) || 1;
  for (let sweep = 0; sweep < 64; sweep++) {
    const off = Math.hypot(a[0][1], a[0][2], a[1][2]);
    if (off <= 1e-18 * scale) break;
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ]) {
      if (a[p][q] === 0) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const sn = t * c;
      for (let k = 0; k < 3; k++) {
        const kp = a[k][p];
        const kq = a[k][q];
        a[k][p] = c * kp - sn * kq;
        a[k][q] = sn * kp + c * kq;
      }
      for (let k = 0; k < 3; k++) {
        const pk = a[p][k];
        const qk = a[q][k];
        a[p][k] = c * pk - sn * qk;
        a[q][k] = sn * pk + c * qk;
      }
      for (let k = 0; k < 3; k++) {
        const kp = v[k][p];
        const kq = v[k][q];
        v[k][p] = c * kp - sn * kq;
        v[k][q] = sn * kp + c * kq;
      }
    }
  }
  return { values: [a[0][0], a[1][1], a[2][2]], vectors: v };
}

export interface Svd {
  u: M3; // left singular vectors as columns
  s: number[]; // singular values, largest first, as numpy gives them
  v: M3; // right singular vectors as columns (numpy's vt transposed)
}

/** a = u diag(s) v^T, with s from the largest to the smallest as in numpy.linalg.svd. */
export function svd3(a: M3): Svd {
  const { values, vectors } = symmetricEigen(multiply(transpose(a), a));
  const order = [0, 1, 2].sort((i, j) => values[j] - values[i]);
  const s = order.map((i) => Math.sqrt(Math.max(values[i], 0)));
  if (!(s[0] > 0)) throw new Error("the matrix is zero");
  const vcols = order.map((i) => [vectors[0][i], vectors[1][i], vectors[2][i]]);
  const ucols = vcols.map((vc, k) => {
    if (s[k] <= 1e-12 * s[0]) return null;
    return [0, 1, 2].map((i) => (a[i][0] * vc[0] + a[i][1] * vc[1] + a[i][2] * vc[2]) / s[k]);
  });
  if (!ucols[1]) throw new Error("the matrix has rank below 2");
  if (!ucols[2]) {
    const [x, y] = [ucols[0]!, ucols[1]];
    ucols[2] = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
  }
  const cols = (c: number[][]): M3 => [0, 1, 2].map((i) => [c[0][i], c[1][i], c[2][i]]);
  return { u: cols(ucols as number[][]), s, v: cols(vcols) };
}

/** The rotation nearest to the 3x3 part of a 4x4 (or 3x3) matrix given as rows (pose.rotation). */
export function rotation(rows: readonly (readonly number[])[]): M3 {
  const shapeOk = (rows.length === 4 && rows.every((r) => r.length === 4)) || (rows.length === 3 && rows.every((r) => r.length === 3));
  if (!shapeOk || !rows.flat().every(Number.isFinite)) throw new Error("matrix must be a finite 4x4 or 3x3 array");
  const a = [0, 1, 2].map((i) => [0, 1, 2].map((j) => rows[i][j]));
  const { u, v } = svd3(a);
  // u @ diag([1, 1, d]) @ vt with d = sign(det(u @ vt)) or 1
  const d = Math.sign(det(u) * det(v)) || 1;
  const ud = u.map((row) => [row[0], row[1], row[2] * d]);
  return multiply(ud, transpose(v));
}

const DEG = 180 / Math.PI;

/**
 * Yaw, pitch and roll in degrees (pose.angles), from a matrix given as rows.
 *
 * f = R (0, 0, 1) is where the face points and r = R (1, 0, 0) runs from the person's
 * right side to their left. yaw = atan2(fx, fz) is positive when the person turns to
 * their own left, pitch = atan2(fy, hypot(fx, fz)) when they look up, and
 * roll = atan2(ry, rx) when their left eye is higher than their right eye in the image
 * (api.md section 4).
 */
export function poseAngles(rows: readonly (readonly number[])[]): Pose {
  const r = rotation(rows);
  const [fx, fy, fz] = [r[0][2], r[1][2], r[2][2]];
  const [rx, ry] = [r[0][0], r[1][0]];
  return {
    yaw_deg: Math.atan2(fx, fz) * DEG,
    pitch_deg: Math.atan2(fy, Math.hypot(fx, fz)) * DEG,
    roll_deg: Math.atan2(ry, rx) * DEG,
  };
}

/** Yaw, pitch and roll from a Matrix of tasks-vision. */
export function poseFromMatrix(m: Matrix): Pose {
  return poseAngles(matrixRows(m));
}

// ---------------------------------------------------------------- blendshapes

export interface Category {
  categoryName: string;
  score: number;
}

/**
 * smile, jaw_open, blink_left and blink_right as quality.blendshape_values: smile is the
 * larger of mouthSmileLeft and mouthSmileRight; eyes_closed later uses the larger blink.
 * null when a blendshape is missing.
 */
export function blendshapeValues(categories: readonly Category[]): Blendshapes | null {
  const score = new Map(categories.map((c) => [c.categoryName, c.score]));
  const get = (name: string) => score.get(name);
  const [smileL, smileR, jaw, blinkL, blinkR] = ["mouthSmileLeft", "mouthSmileRight", "jawOpen", "eyeBlinkLeft", "eyeBlinkRight"].map(get);
  if ([smileL, smileR, jaw, blinkL, blinkR].some((v) => v === undefined || !Number.isFinite(v))) return null;
  return { smile: Math.max(smileL!, smileR!), jaw_open: jaw!, blink_left: blinkL!, blink_right: blinkR! };
}

/** Every value of one face, or null when a part of the result is missing. */
export function faceValues(landmarks: readonly LandmarkPoint[], matrix: Matrix | undefined, categories: readonly Category[] | undefined, size: Size): FrontalValues | null {
  if (landmarks.length < N_POINTS || !matrix || !categories) return null;
  const blend = blendshapeValues(categories);
  if (!blend) return null;
  let pose: Pose;
  try {
    pose = poseFromMatrix(matrix);
  } catch {
    return null;
  }
  return { iod_px: iodPx(landmarks, size), face_width_px: faceWidthPx(landmarks, size), ...pose, ...blend };
}
