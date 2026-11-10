// guidance.ts: the order of the checks, the directions, the two distance checks that
// never ask for both ways, averaging over half a second, the automatic photo after one
// second, the loop at about 15 frames a second, the landmarker's options, and the usage
// log of tasks-vision that never leaves the page (web.md sections 3.2 and 9).
import { describe, expect, it, vi } from "vitest";
import {
  AUTO_CAPTURE_HOLD_MS,
  AVERAGE_WINDOW_MS,
  FACE_LANDMARKER_MODEL_URL,
  GUIDANCE_MAX_FPS,
  GUIDE_FACE_WIDTH_PER_IOD,
  GUIDE_FRAME_WIDTH,
  MEDIAPIPE_WASM_URL,
  SHARED_QUALITY,
} from "../src/constants.ts";
import {
  Guide,
  GuidanceLoop,
  blockUsageLogs,
  createFaceLandmarker,
  evaluate,
  guideFrameWidthPx,
  isTooClose,
  isUsageLog,
  issueText,
  readResult,
  sharedLandmarker,
  type FrameReading,
  type LandmarkerResultLike,
} from "../src/guidance.ts";
import type { FrontalValues } from "../src/metrics.ts";
import { toTasksMatrix } from "../src/metrics.ts";
import { manualClock, syntheticLandmarks, th } from "./helpers.ts";

const C = SHARED_QUALITY.frontal;
const FRAME = { width: 1920, height: 1080 };
const t = (key: string, params?: Record<string, string | number>) => th(key, params);

/** A face that passes every check on a 1920 x 1080 frame. */
function good(over: Partial<FrontalValues> = {}): FrontalValues {
  return { iod_px: 110, face_width_px: 200, yaw_deg: 1, pitch_deg: -1, roll_deg: 0.5, smile: 0.05, jaw_open: 0.02, blink_left: 0.1, blink_right: 0.1, ...over };
}

function frame(time: number, face: FrontalValues | null, faceCount = face ? 1 : 0): FrameReading {
  return { t: time, faceCount, face, size: FRAME };
}

const message = (values: FrontalValues | null, faceCount = 1) => {
  const { issue } = evaluate(values, faceCount, FRAME);
  return issue ? issueText(issue, t) : null;
};

describe("the checks", () => {
  it("go face, distance, pose, then expression, one message at a time", () => {
    // wider than the arch (249.6 px), with an IOD that still passes after stepping back
    const all = good({ face_width_px: 300, iod_px: 120, yaw_deg: 20, smile: 0.9, blink_left: 0.9 });
    const steps: [FrontalValues | null, number, string | null][] = [
      [null, 0, th("quality.no_face")],
      [all, 2, th("quality.multiple_faces")],
      [all, 1, th("quality.too_close")],
      [{ ...all, face_width_px: 170, iod_px: C.min_iod_px - 1 }, 1, th("quality.face_too_small")],
      [{ ...all, face_width_px: 200, iod_px: 110 }, 1, th("ui.guide.pose_hint", { text: th("quality.frontal_pose"), direction: th("ui.guide.pose.yaw_positive") })],
      [{ ...all, face_width_px: 200, iod_px: 110, yaw_deg: 0 }, 1, th("quality.smiling")],
      [{ ...all, face_width_px: 200, iod_px: 110, yaw_deg: 0, smile: 0 }, 1, th("quality.eyes_closed")],
      [good(), 1, null],
    ];
    for (const [values, count, text] of steps) expect(message(values, count)).toBe(text);
  });

  it("give each group a state for the dots under the picture", () => {
    expect(evaluate(null, 0, FRAME).checks).toEqual({ face: "fail", distance: "unknown", pose: "unknown", expression: "unknown" });
    expect(evaluate(good({ pitch_deg: -15 }), 1, FRAME).checks).toEqual({ face: "pass", distance: "pass", pose: "fail", expression: "pass" });
    expect(evaluate(good({ jaw_open: 0.5 }), 2, FRAME).checks).toEqual({ face: "fail", distance: "pass", pose: "pass", expression: "fail" });
  });

  it("count a value equal to its limit as passing, as the api does", () => {
    const edge = good({ iod_px: C.min_iod_px, yaw_deg: C.max_abs_yaw_deg, pitch_deg: -C.max_abs_pitch_deg, roll_deg: C.max_abs_roll_deg, smile: C.max_smile, jaw_open: C.max_jaw_open, blink_left: C.max_eye_blink, face_width_px: GUIDE_FRAME_WIDTH * FRAME.width });
    expect(message(edge)).toBeNull();
  });

  it("say the way to move from the user's side for both signs of each angle", () => {
    const pose = (over: Partial<FrontalValues>, direction: string) =>
      expect(message(good(over))).toBe(th("ui.guide.pose_hint", { text: th("quality.frontal_pose"), direction: th(`ui.guide.pose.${direction}`) }));
    // yaw > 0: turned to their own left, so back to their right
    pose({ yaw_deg: C.max_abs_yaw_deg + 0.1 }, "yaw_positive");
    pose({ yaw_deg: -C.max_abs_yaw_deg - 0.1 }, "yaw_negative");
    // pitch > 0: looking up, so down a little; pitch < 0: up a little
    pose({ pitch_deg: C.max_abs_pitch_deg + 0.1 }, "pitch_positive");
    pose({ pitch_deg: -C.max_abs_pitch_deg - 0.1 }, "pitch_negative");
    // roll > 0: left eye higher, so tilt toward the left shoulder
    pose({ roll_deg: C.max_abs_roll_deg + 0.1 }, "roll_positive");
    pose({ roll_deg: -C.max_abs_roll_deg - 0.1 }, "roll_negative");
    // yaw first when more than one angle is off
    pose({ yaw_deg: -30, pitch_deg: 30, roll_deg: 30 }, "yaw_negative");
  });

  it("word the directions as web.md gives them", () => {
    expect(th("ui.guide.pose.pitch_negative")).toBe("\u0E40\u0E07\u0E22\u0E02\u0E36\u0E49\u0E19\u0E40\u0E25\u0E47\u0E01\u0E19\u0E49\u0E2D\u0E22"); // look up a little
    expect(th("ui.guide.pose.yaw_positive")).toContain("\u0E02\u0E27\u0E32"); // right
    expect(th("ui.guide.pose.yaw_negative")).toContain("\u0E0B\u0E49\u0E32\u0E22"); // left
    expect(th("ui.guide.pose.roll_positive")).toContain("\u0E0B\u0E49\u0E32\u0E22");
    expect(th("ui.guide.pose.roll_negative")).toContain("\u0E02\u0E27\u0E32");
  });
});

describe("the two distance checks", () => {
  type Distance = "small" | "pass" | "close";

  /** The distance check for a face of ratio width / IOD at each IOD, from far to near. */
  function walkCloser(frame: { width: number; height: number }, ratio: number): Distance[] {
    const out: Distance[] = [];
    for (let iod = 30; iod <= 400; iod += 0.5) {
      const { issue } = evaluate(good({ iod_px: iod, face_width_px: ratio * iod }), 1, frame);
      out.push(issue?.code === "too_close" ? "close" : issue?.code === "face_too_small" ? "small" : "pass");
    }
    return out;
  }

  it.each([
    { frame: { width: 1280, height: 720 }, ratio: 2.4 },
    { frame: { width: 1280, height: 720 }, ratio: 3.5 },
    { frame: { width: 1920, height: 1080 }, ratio: 2.2 },
    { frame: { width: 1920, height: 1080 }, ratio: 2.4 },
    { frame: { width: 1920, height: 1080 }, ratio: 3.5 },
    { frame: { width: 3840, height: 2160 }, ratio: 2.4 },
  ])("never ask to step back and come closer in turn: $frame.width px wide, face $ratio IOD wide", ({ frame, ratio }) => {
    // walking toward the camera: too small, then passing, then too close; never back
    const runs = walkCloser(frame, ratio).filter((d, i, all) => i === 0 || d !== all[i - 1]);
    expect(runs.join(" ")).toMatch(/^(small )?pass( close)?$/);
  });

  it("make the arch at least as wide as a face with the minimum IOD", () => {
    expect(guideFrameWidthPx({ width: 1920, height: 1080 })).toBeCloseTo(GUIDE_FRAME_WIDTH * 1920, 9);
    expect(guideFrameWidthPx({ width: 1280, height: 720 })).toBeCloseTo(GUIDE_FACE_WIDTH_PER_IOD * C.min_iod_px, 9);
    expect(GUIDE_FACE_WIDTH_PER_IOD).toBeGreaterThan(2.41); // a SCUT-FBP5500 face measured 2.41 IOD wide
  });

  it("ask to step back only when the IOD would still pass once the face fits the arch", () => {
    const arch = 249.6;
    expect(isTooClose({ face_width_px: 300, iod_px: 120 }, arch)).toBe(true); // 120 * 249.6 / 300 = 99.8
    expect(isTooClose({ face_width_px: 260, iod_px: 76 }, arch)).toBe(false); // would end at 73
    expect(isTooClose({ face_width_px: 249.6, iod_px: 100 }, arch)).toBe(false); // fits
  });
});

describe("Guide over time", () => {
  const STEP = 1000 / 15;

  it("takes the photo once every check has passed for one second without a break", () => {
    const guide = new Guide();
    const captured: number[] = [];
    for (let i = 0; i * STEP <= 3000; i++) {
      const s = guide.update(frame(i * STEP, good()));
      if (s.capture) captured.push(s.t);
    }
    expect(captured).toHaveLength(1); // one photo only
    expect(captured[0]).toBeGreaterThanOrEqual(AUTO_CAPTURE_HOLD_MS);
    expect(captured[0] - STEP).toBeLessThan(AUTO_CAPTURE_HOLD_MS);
  });

  it("starts counting again when a check fails", () => {
    const guide = new Guide();
    const captured: number[] = [];
    let failedAt = -1;
    for (let i = 0; i * STEP <= 3000; i++) {
      const time = i * STEP;
      // a face is lost for one frame at 0.8 s
      const lost = failedAt < 0 && time >= 800;
      if (lost) failedAt = time;
      const s = guide.update(frame(time, lost ? null : good()));
      if (lost) expect(s.holdMs).toBe(0);
      if (s.capture) captured.push(time);
    }
    expect(captured).toHaveLength(1);
    expect(captured[0]).toBeGreaterThanOrEqual(failedAt + STEP + AUTO_CAPTURE_HOLD_MS);
    expect(captured[0] - STEP).toBeLessThan(failedAt + STEP + AUTO_CAPTURE_HOLD_MS);
  });

  it("averages the values of the last half second", () => {
    const guide = new Guide();
    // a turn of 40 degrees at 0 ms, then straight
    let s = guide.update(frame(0, good({ yaw_deg: 40 })));
    for (const time of [100, 200, 300, 400]) s = guide.update(frame(time, good({ yaw_deg: 0 })));
    expect(s.frames).toBe(5);
    expect(s.values!.yaw_deg).toBeCloseTo(8, 9); // (40 + 0 * 4) / 5
    expect(s.issue).toBeNull();
    s = guide.update(frame(AVERAGE_WINDOW_MS, good({ yaw_deg: 0 }))); // 500 ms after the turn: it has left the window
    expect(s.frames).toBe(5);
    expect(s.values!.yaw_deg).toBe(0);
    expect(AVERAGE_WINDOW_MS).toBe(500);
  });

  it("is not thrown off by one jittery frame", () => {
    const guide = new Guide();
    let s = guide.update(frame(0, good({ yaw_deg: 0 })));
    for (let i = 1; i <= 7; i++) s = guide.update(frame(i * STEP, good({ yaw_deg: i === 7 ? 40 : 0 })));
    expect(s.values!.yaw_deg).toBeCloseTo(40 / 8, 9);
    expect(s.issue).toBeNull();
    expect(s.holdMs).toBeCloseTo(7 * STEP, 9);
  });

  it("can take a photo again after reset()", () => {
    const guide = new Guide();
    let count = 0;
    for (let i = 0; i <= 20; i++) if (guide.update(frame(i * 100, good())).capture) count++;
    guide.reset();
    for (let i = 30; i <= 50; i++) if (guide.update(frame(i * 100, good())).capture) count++;
    expect(count).toBe(2);
  });
});

describe("reading a result", () => {
  const cat = (smile: number) => ({
    categories: [
      { categoryName: "mouthSmileLeft", score: smile },
      { categoryName: "mouthSmileRight", score: 0 },
      { categoryName: "jawOpen", score: 0 },
      { categoryName: "eyeBlinkLeft", score: 0 },
      { categoryName: "eyeBlinkRight", score: 0 },
    ],
  });
  const identity = toTasksMatrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -50], [0, 0, 0, 1]]);
  const small = syntheticLandmarks({ 33: [0.1, 0.5], 133: [0.12, 0.5], 362: [0.14, 0.5], 263: [0.16, 0.5] });
  const large = syntheticLandmarks({ 33: [0.4, 0.5], 133: [0.46, 0.5], 362: [0.54, 0.5], 263: [0.6, 0.5] });

  it("measures the face with the largest IOD, as the api does, in pixels of the frame of the stream", () => {
    const result: LandmarkerResultLike = { faceLandmarks: [small, large], faceBlendshapes: [cat(0.9), cat(0.1)], facialTransformationMatrixes: [identity, identity] };
    const r = readResult(result, { width: 3840, height: 2160 }, 5);
    expect(r.faceCount).toBe(2);
    expect(r.face!.smile).toBe(0.1);
    expect(r.face!.iod_px).toBeCloseTo(0.14 * 3840, 9);
    expect(r.size).toEqual({ width: 3840, height: 2160 });
  });

  it("has no face values without a face or without its matrix", () => {
    expect(readResult({ faceLandmarks: [] }, FRAME, 0)).toMatchObject({ faceCount: 0, face: null });
    expect(readResult({ faceLandmarks: [large], faceBlendshapes: [cat(0)] }, FRAME, 0)).toMatchObject({ faceCount: 1, face: null });
  });
});

describe("GuidanceLoop", () => {
  const RESULT: LandmarkerResultLike = { faceLandmarks: [] };

  function setup(ready = true) {
    const clock = manualClock();
    const landmarker = { detectForVideo: vi.fn((_video: HTMLVideoElement, _ts: number) => RESULT) };
    const source = { ready, size: FRAME, video: document.createElement("video") };
    const states: number[] = [];
    const onError = vi.fn();
    const loop = new GuidanceLoop(landmarker, source, new Guide(), clock, { onState: (s) => states.push(s.fps), onError });
    return { clock, landmarker, source, states, onError, loop };
  }

  it("processes at most about 15 frames a second of a 60 Hz display", () => {
    const { clock, landmarker, states, loop } = setup();
    loop.start();
    for (let i = 1; i <= 120; i++) clock.tick(1000 + (i * 1000) / 60); // two seconds
    const calls = landmarker.detectForVideo.mock.calls.length;
    expect(calls).toBeGreaterThanOrEqual(28);
    expect(calls).toBeLessThanOrEqual(2 * GUIDANCE_MAX_FPS + 1);
    expect(Math.max(...states)).toBeLessThanOrEqual(GUIDANCE_MAX_FPS + 1);
    // timestamps only go up
    const stamps = landmarker.detectForVideo.mock.calls.map((c) => c[1]);
    expect(stamps.every((s, i) => i === 0 || s > stamps[i - 1])).toBe(true);
  });

  it("waits until the video has a frame", () => {
    const { clock, landmarker, loop } = setup(false);
    loop.start();
    for (let i = 1; i <= 30; i++) clock.tick(i * 100);
    expect(landmarker.detectForVideo).not.toHaveBeenCalled();
  });

  it("stops asking for frames when stopped, and on an error of the landmarker", () => {
    const { clock, landmarker, onError, loop } = setup();
    loop.start();
    clock.tick(100);
    loop.stop();
    expect(clock.pending).toBe(0);
    landmarker.detectForVideo.mockImplementation(() => {
      throw new Error("lost the GPU");
    });
    loop.start();
    clock.tick(300);
    expect(onError).toHaveBeenCalledTimes(1);
    expect(loop.running).toBe(false);
    expect(clock.pending).toBe(0);
  });
});

describe("the usage log of tasks-vision", () => {
  it("is told apart from the page's own requests", () => {
    expect(isUsageLog("https://odml.pa.googleapis.com/v1/log")).toBe(true);
    expect(isUsageLog(new URL("https://odml.pa.googleapis.com/v1/log?x=1"))).toBe(true);
    expect(isUsageLog(new Request("https://odml.pa.googleapis.com/v1/log"))).toBe(true);
    expect(isUsageLog("/analyze")).toBe(false);
    expect(isUsageLog(FACE_LANDMARKER_MODEL_URL)).toBe(false);
    expect(isUsageLog(`${MEDIAPIPE_WASM_URL}/vision_wasm_internal.wasm`)).toBe(false);
  });

  it("is refused before it is sent; every other request goes on", async () => {
    const original = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response("ok"));
    const holder = { fetch: original as unknown as typeof fetch };
    blockUsageLogs(holder);
    blockUsageLogs(holder); // once is enough
    await expect(holder.fetch("https://odml.pa.googleapis.com/v1/log", { method: "POST", body: "x" })).rejects.toBeInstanceOf(TypeError);
    expect(original).not.toHaveBeenCalled();
    const init = { method: "POST" };
    await holder.fetch("/analyze", init);
    await holder.fetch(FACE_LANDMARKER_MODEL_URL);
    expect(original.mock.calls).toEqual([
      ["/analyze", init],
      [FACE_LANDMARKER_MODEL_URL, undefined],
    ]);
  });
});

describe("the landmarker", () => {
  function fakeVision(gpuFails: boolean) {
    const created: { delegate?: string }[] = [];
    const options: unknown[] = [];
    const vision = {
      FilesetResolver: { forVisionTasks: vi.fn(async (path: string) => ({ wasmLoaderPath: path, wasmBinaryPath: path })) },
      FaceLandmarker: {
        createFromOptions: vi.fn(async (_fileset: unknown, o: { baseOptions: { delegate?: string } }) => {
          options.push(o);
          if (gpuFails && o.baseOptions.delegate === "GPU") throw new Error("no WebGL");
          created.push({ delegate: o.baseOptions.delegate });
          return { detectForVideo: () => ({ faceLandmarks: [] }) };
        }),
      },
    };
    return { vision, created, options };
  }

  it("runs in VIDEO mode with two faces, blendshapes and matrices, on the GPU first, with its log refused", async () => {
    const { vision, created, options } = fakeVision(false);
    const original = vi.fn(async () => new Response("ok"));
    const holder = { fetch: original as unknown as typeof fetch };
    await createFaceLandmarker(async () => vision as never, holder);
    await expect(holder.fetch("https://odml.pa.googleapis.com/v1/log")).rejects.toThrow();
    expect(original).not.toHaveBeenCalled();
    expect(vision.FilesetResolver.forVisionTasks).toHaveBeenCalledWith(MEDIAPIPE_WASM_URL);
    expect(options[0]).toEqual({
      baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 2,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    expect(created).toEqual([{ delegate: "GPU" }]);
  });

  it("uses the CPU when the GPU cannot be used", async () => {
    const { vision, created } = fakeVision(true);
    await createFaceLandmarker(async () => vision as never, { fetch: vi.fn() as unknown as typeof fetch });
    expect(created).toEqual([{ delegate: "CPU" }]);
  });

  it("is loaded once for the page, and again after a failed load", async () => {
    const load = vi.fn().mockRejectedValueOnce(new Error("offline")).mockResolvedValue({ detectForVideo: () => ({ faceLandmarks: [] }) });
    await expect(sharedLandmarker(load)).rejects.toThrow("offline");
    const a = await sharedLandmarker(load);
    const b = await sharedLandmarker(load);
    expect(a).toBe(b);
    expect(load).toHaveBeenCalledTimes(2);
  });
});
