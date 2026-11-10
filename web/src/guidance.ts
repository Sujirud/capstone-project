// Live pose guidance for the frontal photo (web.md sections 3.2 and 4.1).
//
// The Face Landmarker of MediaPipe for the web runs in VIDEO mode on the camera's video,
// at most GUIDANCE_MAX_FPS frames a second, looking for up to two faces with blendshapes
// and the transformation matrix. Each frame gives the values of the largest face (the
// api measures the face with the largest IOD too); the values are averaged over the last
// AVERAGE_WINDOW_MS, because with numFaces above 1 MediaPipe does not smooth them across
// frames. The checks go in the order face, distance, pose, then expression, and the page
// shows the first one that fails. When every check passes for AUTO_CAPTURE_HOLD_MS
// without a break, one frame is taken.
//
// The limits are those of the api, from shared/quality.json; a value equal to its limit
// passes, as in quality.evaluate. The web advises and the api decides (web.md section 1.2).
import type { Matrix } from "@mediapipe/tasks-vision";
import {
  AUTO_CAPTURE_HOLD_MS,
  AVERAGE_WINDOW_MS,
  FACE_LANDMARKER_MODEL_URL,
  GUIDANCE_MAX_FPS,
  GUIDE_FACE_WIDTH_PER_IOD,
  GUIDE_FRAME_WIDTH,
  MAX_FACES,
  MEDIAPIPE_WASM_URL,
  SHARED_QUALITY,
} from "./constants.ts";
import type { Params } from "./i18n.ts";
import type { Size } from "./images.ts";
import { FRONTAL_VALUE_NAMES, MATRIX_LAYOUT, faceValues, iodPx, matrixLayout, type Category, type FrontalValues, type LandmarkPoint } from "./metrics.ts";

export type FrontalCriteria = typeof SHARED_QUALITY.frontal;

// ---------------------------------------------------------------- the landmarker

/** The part of a FaceLandmarkerResult the page reads. */
export interface LandmarkerResultLike {
  faceLandmarks: LandmarkPoint[][];
  faceBlendshapes?: { categories: Category[] }[];
  facialTransformationMatrixes?: Matrix[];
}

/** The part of a FaceLandmarker the page uses; tests pass a fake. */
export interface LandmarkerLike {
  detectForVideo(frame: HTMLVideoElement, timestampMs: number): LandmarkerResultLike;
}

type VisionModule = Pick<typeof import("@mediapipe/tasks-vision"), "FaceLandmarker" | "FilesetResolver">;

// tasks-vision 1.2.0 sends usage statistics of each task (the task, its mode and its
// timings; no image or value) to this address every minute, and has no option to turn
// it off. The page talks only to the api, the CDN and the model host (web.md section 9),
// so such a request is refused here before it leaves the browser. The CSP does not allow
// the host either, but a blocked request would show as a CSP error in DevTools, and the
// dev server has no CSP.
export const USAGE_LOG_ORIGINS = ["https://odml.pa.googleapis.com"] as const;

export interface FetchHolder {
  fetch: typeof fetch;
}

const guarded = new WeakSet<FetchHolder>();

function requestUrl(input: RequestInfo | URL): string {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.href;
  return input.url;
}

/** True for a request to a host of USAGE_LOG_ORIGINS. */
export function isUsageLog(input: RequestInfo | URL): boolean {
  let origin: string;
  try {
    origin = new URL(requestUrl(input), globalThis.location?.href).origin;
  } catch {
    return false;
  }
  return (USAGE_LOG_ORIGINS as readonly string[]).includes(origin);
}

/**
 * Make target.fetch refuse requests to USAGE_LOG_ORIGINS, as a network error, without
 * sending them; every other request goes on unchanged. tasks-vision then stops its log.
 */
export function blockUsageLogs(target: FetchHolder = globalThis): void {
  if (guarded.has(target)) return;
  const original = target.fetch;
  target.fetch = function (this: unknown, input: RequestInfo | URL, init?: RequestInit) {
    if (isUsageLog(input)) return Promise.reject(new TypeError("usage log refused by the page"));
    return original.call(this === undefined || this === target ? globalThis : this, input, init);
  } as typeof fetch;
  guarded.add(target);
}

/**
 * A Face Landmarker for the video: GPU first, the CPU when the GPU cannot be used. The
 * wasm files come from the CDN at the version of the package, the model from the link
 * of api/assets.lock.json (web.md section 8.1). Its usage log is refused first (blockUsageLogs).
 */
export async function createFaceLandmarker(
  vision: () => Promise<VisionModule> = () => import("@mediapipe/tasks-vision"),
  fetchHolder: FetchHolder = globalThis,
): Promise<LandmarkerLike> {
  blockUsageLogs(fetchHolder);
  const { FaceLandmarker, FilesetResolver } = await vision();
  const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
  const options = (delegate: "GPU" | "CPU") => ({
    baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL_URL, delegate },
    runningMode: "VIDEO" as const,
    numFaces: MAX_FACES,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });
  try {
    return await FaceLandmarker.createFromOptions(fileset, options("GPU"));
  } catch {
    return await FaceLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

let shared: Promise<LandmarkerLike> | null = null;

/** One landmarker for the whole page, loaded the first time it is asked for; a failed load can be tried again. */
export function sharedLandmarker(load: () => Promise<LandmarkerLike> = createFaceLandmarker): Promise<LandmarkerLike> {
  shared ??= load().catch((error: unknown) => {
    shared = null;
    throw error;
  });
  return shared;
}

// ---------------------------------------------------------------- one frame

export interface FrameReading {
  t: number; // ms
  faceCount: number;
  face: FrontalValues | null; // the face with the largest IOD; null without a face or a part of its result
  size: Size; // the frame of the stream, which is the size of the photo
}

let layoutWarned = false;

/** Warn once, without any value, if a frame ever says the matrix is not in MATRIX_LAYOUT. */
function checkLayout(m: Matrix | undefined): void {
  if (layoutWarned || !m) return;
  const found = matrixLayout(m);
  if (found && found !== MATRIX_LAYOUT) {
    layoutWarned = true;
    console.warn(`guidance: the transformation matrix looks ${found}-major, not ${MATRIX_LAYOUT}-major; see metrics.ts`);
  }
}

/** The values of a landmarker result on a frame of size. */
export function readResult(result: LandmarkerResultLike, size: Size, t: number): FrameReading {
  const faces = result.faceLandmarks ?? [];
  let best = -1;
  let bestIod = -Infinity;
  faces.forEach((points, k) => {
    if (points.length < 478) return;
    const iod = iodPx(points, size);
    if (iod > bestIod) {
      best = k;
      bestIod = iod;
    }
  });
  const matrix = best >= 0 ? result.facialTransformationMatrixes?.[best] : undefined;
  checkLayout(matrix);
  const face = best >= 0 ? faceValues(faces[best], matrix, result.faceBlendshapes?.[best]?.categories, size) : null;
  return { t, faceCount: faces.length, face, size };
}

// ---------------------------------------------------------------- the checks

export const CHECK_GROUPS = ["face", "distance", "pose", "expression"] as const;
export type CheckGroup = (typeof CHECK_GROUPS)[number];
export type CheckState = "pass" | "fail" | "unknown";

export type PoseDirection = "yaw_positive" | "yaw_negative" | "pitch_positive" | "pitch_negative" | "roll_positive" | "roll_negative";
export type IssueCode = "no_face" | "multiple_faces" | "too_close" | "face_too_small" | "frontal_pose" | "smiling" | "eyes_closed";
export type Issue = { code: Exclude<IssueCode, "frontal_pose"> } | { code: "frontal_pose"; direction: PoseDirection };

/**
 * Which way the pose is off: yaw first, then pitch, then roll, so the message does not
 * jump between axes. Signs of api.md section 4: yaw is positive when the person turns to
 * their own left, pitch when they look up, roll when their left eye is higher.
 */
export function poseDirection(v: FrontalValues, c: FrontalCriteria): PoseDirection | null {
  if (v.yaw_deg > c.max_abs_yaw_deg) return "yaw_positive";
  if (v.yaw_deg < -c.max_abs_yaw_deg) return "yaw_negative";
  if (v.pitch_deg > c.max_abs_pitch_deg) return "pitch_positive";
  if (v.pitch_deg < -c.max_abs_pitch_deg) return "pitch_negative";
  if (v.roll_deg > c.max_abs_roll_deg) return "roll_positive";
  if (v.roll_deg < -c.max_abs_roll_deg) return "roll_negative";
  return null;
}

/**
 * The width of the guide at the roots of the ears, in pixels of the frame: GUIDE_FRAME_WIDTH
 * of the frame's width, and never narrower than a face whose IOD is min_iod_px
 * (GUIDE_FACE_WIDTH_PER_IOD). screens/guide_layer.ts draws both outlines to this unit.
 */
export function guideFrameWidthPx(frame: Size, c: FrontalCriteria = SHARED_QUALITY.frontal, widthShare = GUIDE_FRAME_WIDTH): number {
  return Math.max(widthShare * frame.width, GUIDE_FACE_WIDTH_PER_IOD * c.min_iod_px);
}

/**
 * Too close: the face is wider than the guide, and stepping back until it fits would
 * still leave the IOD at min_iod_px or more. Without that second part a face that is
 * wide for its IOD would be told to step back and then to come closer, over and over;
 * the api checks the IOD, so the IOD wins (web.md sections 1.2 and 3.2).
 */
export function isTooClose(values: Pick<FrontalValues, "face_width_px" | "iod_px">, guidePx: number, c: FrontalCriteria = SHARED_QUALITY.frontal): boolean {
  return values.face_width_px > guidePx && (values.iod_px * guidePx) / values.face_width_px >= c.min_iod_px;
}

export interface Evaluation {
  issue: Issue | null; // the first check that fails; null when every check passes
  checks: Record<CheckGroup, CheckState>;
}

/** The checks of web.md section 3.2 on the values of one face (averaged) and the face count of the frame. */
export function evaluate(values: FrontalValues | null, faceCount: number, frame: Size, c: FrontalCriteria = SHARED_QUALITY.frontal, frameWidth = GUIDE_FRAME_WIDTH): Evaluation {
  if (faceCount < 1 || values === null) {
    return { issue: { code: "no_face" }, checks: { face: "fail", distance: "unknown", pose: "unknown", expression: "unknown" } };
  }
  const tooClose = isTooClose(values, guideFrameWidthPx(frame, c, frameWidth), c);
  const tooSmall = values.iod_px < c.min_iod_px;
  const direction = poseDirection(values, c);
  const smiling = values.smile > c.max_smile || values.jaw_open > c.max_jaw_open;
  const eyesClosed = Math.max(values.blink_left, values.blink_right) > c.max_eye_blink;
  const ordered: [boolean, Issue][] = [
    [faceCount > 1, { code: "multiple_faces" }],
    [tooClose, { code: "too_close" }],
    [tooSmall, { code: "face_too_small" }],
    [direction !== null, { code: "frontal_pose", direction: direction ?? "yaw_positive" }],
    [smiling, { code: "smiling" }],
    [eyesClosed, { code: "eyes_closed" }],
  ];
  const state = (fails: boolean): CheckState => (fails ? "fail" : "pass");
  return {
    issue: ordered.find(([fails]) => fails)?.[1] ?? null,
    checks: { face: state(faceCount > 1), distance: state(tooClose || tooSmall), pose: state(direction !== null), expression: state(smiling || eyesClosed) },
  };
}

/** The mean of each value. */
export function average(values: readonly FrontalValues[]): FrontalValues {
  const out = {} as FrontalValues;
  for (const name of FRONTAL_VALUE_NAMES) out[name] = values.reduce((sum, v) => sum + v[name], 0) / values.length;
  return out;
}

export interface GuideState extends Evaluation {
  t: number;
  values: FrontalValues | null; // averaged over the window
  frames: number; // frames in the average
  holdMs: number; // how long every check has passed; 0 when one fails
  capture: boolean; // true on the one update where holdMs reaches the hold time
}

export interface GuideOptions {
  windowMs: number;
  holdMs: number;
  frameWidth: number;
}

/** The checks over time: averaging, the hold before the automatic photo, one photo only. */
export class Guide {
  private window: { t: number; values: FrontalValues }[] = [];
  private passSince: number | null = null;
  private done = false;
  private criteria: FrontalCriteria;
  private options: GuideOptions;

  constructor(criteria: FrontalCriteria = SHARED_QUALITY.frontal, options: Partial<GuideOptions> = {}) {
    this.criteria = criteria;
    this.options = { windowMs: AVERAGE_WINDOW_MS, holdMs: AUTO_CAPTURE_HOLD_MS, frameWidth: GUIDE_FRAME_WIDTH, ...options };
  }

  update(r: FrameReading): GuideState {
    if (r.face) this.window.push({ t: r.t, values: r.face });
    // the frames of the last windowMs, this one included
    this.window = this.window.filter((w) => w.t <= r.t && r.t - w.t < this.options.windowMs);
    const values = r.face && this.window.length ? average(this.window.map((w) => w.values)) : null;
    const { issue, checks } = evaluate(values, r.faceCount, r.size, this.criteria, this.options.frameWidth);
    if (issue) this.passSince = null;
    else this.passSince ??= r.t;
    const holdMs = this.passSince === null ? 0 : r.t - this.passSince;
    const capture = !this.done && issue === null && holdMs >= this.options.holdMs;
    if (capture) this.done = true;
    return { t: r.t, issue, checks, values, frames: values ? this.window.length : 0, holdMs, capture };
  }

  /** Start again: no frame kept, no hold, a photo can be taken again. */
  reset(): void {
    this.window = [];
    this.passSince = null;
    this.done = false;
  }
}

// ---------------------------------------------------------------- texts

/** The texts of web.md section 5.4; too_close is the web's own. */
export const ISSUE_TEXT: Record<IssueCode, string> = {
  no_face: "quality.no_face",
  multiple_faces: "quality.multiple_faces",
  too_close: "quality.too_close",
  face_too_small: "quality.face_too_small",
  frontal_pose: "quality.frontal_pose",
  smiling: "quality.smiling",
  eyes_closed: "quality.eyes_closed",
};

/** Which way to move, from the user's side. */
export const DIRECTION_TEXT: Record<PoseDirection, string> = {
  yaw_positive: "ui.guide.pose.yaw_positive",
  yaw_negative: "ui.guide.pose.yaw_negative",
  pitch_positive: "ui.guide.pose.pitch_positive",
  pitch_negative: "ui.guide.pose.pitch_negative",
  roll_positive: "ui.guide.pose.roll_positive",
  roll_negative: "ui.guide.pose.roll_negative",
};

export function issueText(issue: Issue, t: (key: string, params?: Params) => string): string {
  if (issue.code === "frontal_pose") {
    return t("ui.guide.pose_hint", { text: t(ISSUE_TEXT.frontal_pose), direction: t(DIRECTION_TEXT[issue.direction]) });
  }
  return t(ISSUE_TEXT[issue.code]);
}

// ---------------------------------------------------------------- the loop

/** requestAnimationFrame, or a fake in tests. */
export interface FrameClock {
  request(callback: (t: number) => void): number;
  cancel(id: number): void;
}

export const browserClock: FrameClock = {
  request: (callback) => requestAnimationFrame(callback),
  cancel: (id) => cancelAnimationFrame(id),
};

export interface FrameSource {
  readonly ready: boolean;
  readonly size: Size;
  readonly video: HTMLVideoElement;
}

export interface LiveState extends GuideState {
  fps: number; // frames processed in the last second
}

export interface LoopHandlers {
  onState: (state: LiveState) => void;
  onError: (error: unknown) => void;
}

// VIDEO mode needs timestamps that only go up, for each landmarker
const lastTimestamp = new WeakMap<LandmarkerLike, number>();

/** Runs the landmarker on the video at most maxFps times a second and feeds the Guide. */
export class GuidanceLoop {
  private id: number | null = null;
  private last = -Infinity;
  private times: number[] = [];
  private landmarker: LandmarkerLike;
  private source: FrameSource;
  readonly guide: Guide;
  private clock: FrameClock;
  private handlers: LoopHandlers;
  private interval: number;

  constructor(landmarker: LandmarkerLike, source: FrameSource, guide: Guide, clock: FrameClock, handlers: LoopHandlers, maxFps = GUIDANCE_MAX_FPS) {
    this.landmarker = landmarker;
    this.source = source;
    this.guide = guide;
    this.clock = clock;
    this.handlers = handlers;
    // a frame of the display may come a little early; never more than about maxFps
    this.interval = (1000 / maxFps) * 0.9;
  }

  get running(): boolean {
    return this.id !== null;
  }

  start(): void {
    if (this.id !== null) return;
    this.guide.reset();
    this.times = [];
    this.last = -Infinity;
    this.id = this.clock.request(this.tick);
  }

  stop(): void {
    if (this.id !== null) this.clock.cancel(this.id);
    this.id = null;
  }

  private tick = (t: number): void => {
    if (this.id === null) return;
    this.id = this.clock.request(this.tick);
    if (t - this.last < this.interval || !this.source.ready) return;
    this.last = t;
    const timestamp = Math.max(Math.round(t), (lastTimestamp.get(this.landmarker) ?? 0) + 1);
    lastTimestamp.set(this.landmarker, timestamp);
    let result: LandmarkerResultLike;
    try {
      result = this.landmarker.detectForVideo(this.source.video, timestamp);
    } catch (error) {
      this.stop();
      this.handlers.onError(error);
      return;
    }
    this.times = this.times.filter((x) => t - x < 1000);
    this.times.push(t);
    const state = this.guide.update(readResult(result, this.source.size, t));
    this.handlers.onState({ ...state, fps: this.times.length });
  };
}
