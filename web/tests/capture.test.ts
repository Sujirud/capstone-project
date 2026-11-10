// The capture screens with a fake camera, a fake landmarker, a canvas that records its
// calls, fake timers and a hand-driven frame clock (web.md sections 3, 7 and 9).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App, type AppDeps } from "../src/app.ts";
import {
  AUTO_CAPTURE_HOLD_MS,
  COUNTDOWN_SECONDS,
  FACE_LANDMARKER_MODEL_URL,
  FLASH_MS,
  GUIDE_FACE_WIDTH_PER_IOD,
  GUIDE_FRAME_WIDTH,
  MEDIAPIPE_VERSION,
  MEDIAPIPE_WASM_URL,
  SHARED_QUALITY,
} from "../src/constants.ts";
import { guideFrameWidthPx, type LandmarkerResultLike } from "../src/guidance.ts";
import { guideOutline } from "../src/screens/guide_layer.ts";
import { toTasksMatrix } from "../src/metrics.ts";
import { ROOT, WEB, fakeMedia, fakeUrlApi, fixture, jsonResponse, manualClock, readJson, recordingCanvas, scriptedFetch, settle, syntheticLandmarks, th } from "./helpers.ts";

const HEALTH = { status: "ok", versions: {}, stub_heads: [], device: "cpu", languages: ["th"] };
const STEP = 1000 / 15;

/** A face that passes every check on a 1920 x 1080 frame: IOD 115 px, 230 px wide, straight, still. */
function face(over: { yaw?: number; smile?: number; eyes?: number } = {}): LandmarkerResultLike {
  const points = syntheticLandmarks({ 33: [0.46, 0.45], 133: [0.48, 0.45], 362: [0.52, 0.45], 263: [0.54, 0.45], 234: [0.44, 0.47], 454: [0.56, 0.47] });
  const a = ((over.yaw ?? 0) * Math.PI) / 180;
  const matrix = toTasksMatrix([
    [Math.cos(a), 0, Math.sin(a), 0],
    [0, 1, 0, 0],
    [-Math.sin(a), 0, Math.cos(a), -50],
    [0, 0, 0, 1],
  ]);
  const score = (categoryName: string, value: number) => ({ categoryName, score: value });
  return {
    faceLandmarks: [points],
    facialTransformationMatrixes: [matrix],
    faceBlendshapes: [
      {
        categories: [
          score("mouthSmileLeft", over.smile ?? 0.02),
          score("mouthSmileRight", 0.01),
          score("jawOpen", 0.01),
          score("eyeBlinkLeft", over.eyes ?? 0.05),
          score("eyeBlinkRight", over.eyes ?? 0.05),
        ],
      },
    ],
  };
}

function setup(options: { search?: string; media?: ReturnType<typeof fakeMedia> | null; landmarker?: "ready" | "never" | "fails"; deps?: Partial<AppDeps> } = {}) {
  document.body.innerHTML = '<div id="app"></div>';
  const root = document.getElementById("app")!;
  const media = options.media === undefined ? fakeMedia() : options.media;
  const clock = manualClock();
  let result: LandmarkerResultLike = { faceLandmarks: [] };
  const landmarker = { detectForVideo: vi.fn(() => result) };
  const loadLandmarker = vi.fn(() =>
    options.landmarker === "never" ? new Promise<never>(() => {}) : options.landmarker === "fails" ? Promise.reject(new Error("offline")) : Promise.resolve(landmarker),
  );
  const canvases: ReturnType<typeof recordingCanvas>[] = [];
  const beeper = { prepare: vi.fn(), tick: vi.fn(), shot: vi.fn() };
  const app = new App(root, options.search ?? "", {
    fetch: scriptedFetch(jsonResponse(200, HEALTH)),
    urlApi: fakeUrlApi(),
    copy: vi.fn(async () => {}),
    media,
    loadLandmarker,
    frames: clock,
    makeCanvas: () => {
      const c = recordingCanvas();
      canvases.push(c);
      return c;
    },
    beeper,
    ...options.deps,
  });
  return {
    app,
    root,
    media,
    clock,
    landmarker,
    loadLandmarker,
    canvases,
    beeper,
    setResult: (r: LandmarkerResultLike) => (result = r),
  };
}

function $(root: ParentNode, selector: string): HTMLElement {
  const found = root.querySelector<HTMLElement>(selector);
  if (!found) throw new Error(`nothing matches ${selector}`);
  return found;
}

/** Let promises and zero-delay timers run, with real or fake timers. */
async function flush(): Promise<void> {
  if (vi.isFakeTimers()) for (let i = 0; i < 10; i++) await vi.advanceTimersByTimeAsync(0);
  else await settle();
}

function screenName(root: HTMLElement): string | undefined {
  return root.querySelector<HTMLElement>("[data-screen]")?.dataset.screen;
}

/** The video of the screen has a frame, as a playing video does. */
function videoPlays(root: HTMLElement): void {
  Object.defineProperty($(root, "video"), "readyState", { value: 4, configurable: true });
}

/** Press the button that opens the camera (it never opens by itself). */
async function openCamera(s: ReturnType<typeof setup>) {
  $(s.root, '[data-action="camera-open"]').click();
  await flush();
}

/** Home, then the frontal screen; the camera is opened unless open is false. */
async function toFrontal(s: ReturnType<typeof setup>, age = "30", sex = "female", open = true) {
  await s.app.start();
  ($(s.root, 'input[name="age"]') as HTMLInputElement).value = age;
  ($(s.root, `input[name="sex"][value="${sex}"]`) as HTMLInputElement).checked = true;
  $(s.root, "form").dispatchEvent(new Event("submit", { cancelable: true }));
  await flush();
  if (open) await openCamera(s);
}

/** Frames of the display every 1/60 s from t0 for ms. */
function runFrames(s: ReturnType<typeof setup>, t0: number, ms: number): number {
  let t = t0;
  for (; t <= t0 + ms; t += 1000 / 60) s.clock.tick(t);
  return t;
}

function setHidden(hidden: boolean): void {
  Object.defineProperty(document, "visibilityState", { value: hidden ? "hidden" : "visible", configurable: true });
  document.dispatchEvent(new Event("visibilitychange"));
}

beforeEach(() => setHidden(false));
afterEach(() => {
  vi.useRealTimers();
  setHidden(false);
});

describe("opening the camera", () => {
  it("does not open the camera or load MediaPipe until the button is pressed", async () => {
    const s = setup();
    await toFrontal(s, "30", "female", false);
    expect(s.media!.getUserMedia).not.toHaveBeenCalled();
    expect(s.loadLandmarker).not.toHaveBeenCalled();
    expect(s.root.querySelector("video")).toBeNull();
    expect($(s.root, '[data-action="camera-open"]').textContent).toBe(th("ui.camera.start"));
    expect($(s.root, "[data-capture-main]").children).toHaveLength(0); // no image yet: the middle is empty
    await openCamera(s);
    expect(s.media!.getUserMedia).toHaveBeenCalledTimes(1);
    expect(s.loadLandmarker).toHaveBeenCalledTimes(1);
    expect(s.root.querySelector("[data-capture-main] video")).not.toBeNull();
  });

  it("opens and closes the camera with one button at the bottom, which keeps the focus", async () => {
    const s = setup();
    await toFrontal(s, "30", "female", false);
    const toggle = $(s.root, '[data-action="camera-open"]');
    expect(toggle.classList.contains("btn--primary")).toBe(true);
    toggle.focus();
    await openCamera(s);
    expect(toggle.dataset.action).toBe("camera-close");
    expect(toggle.textContent).toBe(th("ui.camera.stop"));
    expect(toggle.classList.contains("btn--secondary")).toBe(true); // the capture button is the main one now
    expect(document.activeElement).toBe(toggle);
    videoPlays(s.root);
    s.setResult(face({ yaw: -20 }));
    runFrames(s, 0, 300);
    toggle.click();
    expect(s.media!.streams[0].track.stop).toHaveBeenCalledTimes(1);
    expect(s.clock.pending).toBe(0); // the guide stops too
    expect(s.root.querySelector("video")).toBeNull();
    expect(toggle.dataset.action).toBe("camera-open");
    expect(toggle.textContent).toBe(th("ui.camera.start"));
    expect(document.activeElement).toBe(toggle);
    await openCamera(s);
    expect(s.media!.streams).toHaveLength(2);
    expect(s.loadLandmarker).toHaveBeenCalledTimes(1); // MediaPipe is loaded once
  });

  it("has the heading and the tips at the top, the middle, then the actions in this order", async () => {
    const s = setup();
    await toFrontal(s, "30", "female", false);
    const page = $(s.root, '[data-screen="frontal"]');
    const parts = [...page.children].map((c) => c.tagName.toLowerCase() + (c.className ? `.${c.className.split(" ").at(-1)}` : ""));
    expect(parts).toEqual(["header.page__head", "section.capture__tips", "div.capture__main", "div.capture__actions"]);
    expect($(page, ".capture__tips h2").textContent).toBe(th("ui.capture.tips_title"));
    expect($(page, ".capture__tips").getAttribute("aria-labelledby")).toBe($(page, ".capture__tips h2").id);
    const actions = () => [...page.querySelectorAll<HTMLElement>(".capture__actions button")].map((b) => b.dataset.action);
    expect(actions()).toEqual(["camera-open", "choose-file"]);
    s.app.show("profile");
    await flush();
    expect([...s.root.querySelectorAll<HTMLElement>(".capture__actions button")].map((b) => b.dataset.action)).toEqual(["camera-open", "choose-file", "skip-profile"]);
  });

  it("does not open the camera of the profile screen by itself either, even after the frontal photo", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const s = setup();
    await toFrontal(s);
    videoPlays(s.root);
    s.setResult(face());
    runFrames(s, 0, AUTO_CAPTURE_HOLD_MS + 400);
    await vi.advanceTimersByTimeAsync(FLASH_MS);
    await flush();
    expect(screenName(s.root)).toBe("profile");
    expect(s.media!.getUserMedia).toHaveBeenCalledTimes(1);
    expect(s.root.querySelector("video")).toBeNull();
    await openCamera(s);
    expect(s.media!.getUserMedia).toHaveBeenCalledTimes(2);
  });
});

describe("the frontal camera", () => {
  it("says it is preparing the camera while MediaPipe loads", async () => {
    const s = setup({ landmarker: "never" });
    await toFrontal(s);
    expect($(s.root, "[data-guide-message]").textContent).toBe(th("ui.camera.preparing"));
    expect($(s.root, "[data-guide-message]").getAttribute("aria-live")).toBe("polite");
    // the camera works and the photo can be taken by hand meanwhile
    expect(($(s.root, '[data-action="capture"]') as HTMLButtonElement).disabled).toBe(false);
  });

  it("draws the head of the guide in pixels of the stream, GUIDE_FRAME_WIDTH wide at the ears", async () => {
    const s = setup();
    await toFrontal(s);
    const layer = $(s.root, '[data-camera-stage] [data-guide="frontal"]');
    expect(layer.getAttribute("viewBox")).toBe("0 0 1920 1080");
    const outline = $(layer, "[data-outline]").getAttribute("d")!;
    expect(outline).toBe(guideOutline("frontal", { width: 1920, height: 1080 }).outline); // its geometry: guide_layer.test.ts
    expect(guideFrameWidthPx({ width: 1920, height: 1080 })).toBeCloseTo(GUIDE_FRAME_WIDTH * 1920, 6);
    expect($(layer, ".camera__veil").getAttribute("d")).toBe(`M0 0 H1920 V1080 H0 Z ${outline}`);
  });

  it("never draws the head narrower than a face with the minimum IOD, as on a 1280 x 720 stream", async () => {
    const s = setup({ media: fakeMedia({ width: 1280, height: 720, deviceId: "cam-1" }) });
    await toFrontal(s);
    const outline = $(s.root, '[data-guide="frontal"] [data-outline]').getAttribute("d")!;
    expect(outline).toBe(guideOutline("frontal", { width: 1280, height: 720 }).outline);
    expect(guideFrameWidthPx({ width: 1280, height: 720 })).toBeCloseTo(GUIDE_FACE_WIDTH_PER_IOD * SHARED_QUALITY.frontal.min_iod_px, 6);
    expect(GUIDE_FACE_WIDTH_PER_IOD * SHARED_QUALITY.frontal.min_iod_px).toBeGreaterThan(GUIDE_FRAME_WIDTH * 1280);
  });

  it("shows the guide only once the video shows a frame, never over 'preparing the camera'", async () => {
    const s = setup();
    await toFrontal(s);
    const stage = $(s.root, "[data-camera-stage]");
    const wait = $(stage, ".camera__wait");
    expect(stage.querySelector('[data-guide="frontal"]')).not.toBeNull(); // drawn, but hidden by the style until live
    expect(stage.dataset.live).toBeUndefined();
    expect(wait.hidden).toBe(false);
    expect(wait.textContent).toBe(th("ui.camera.preparing"));
    const video = $(stage, "video");
    video.dispatchEvent(new Event("playing")); // the event alone, with no frame yet
    expect(stage.dataset.live).toBeUndefined();
    videoPlays(s.root);
    video.dispatchEvent(new Event("playing"));
    expect(stage.dataset.live).toBe("");
    expect(wait.hidden).toBe(true);
    $(s.root, '[data-action="camera-close"]').click();
    expect(stage.dataset.live).toBeUndefined();
    Object.defineProperty(video, "readyState", { value: 0, configurable: true }); // as a video with no stream
    await openCamera(s);
    expect(stage.dataset.live).toBeUndefined(); // not until the new stream shows a frame
    expect(stage.querySelectorAll("[data-guide]")).toHaveLength(1);
    const css = readFileSync(join(WEB, "src", "styles", "components.css"), "utf-8");
    expect(css).toMatch(/\.camera__stage:not\(\[data-live\]\) \.camera__overlay \{\s*visibility: hidden;/);
  });

  it("shows the message of the first check that fails and a dot for each group", async () => {
    const s = setup();
    await toFrontal(s);
    videoPlays(s.root);
    s.setResult({ faceLandmarks: [] });
    let t = runFrames(s, 0, 200);
    expect($(s.root, "[data-guide-message]").textContent).toBe(th("quality.no_face"));
    expect($(s.root, '[data-check="face"]').dataset.state).toBe("fail");
    s.setResult(face({ yaw: -20 }));
    t = runFrames(s, t + 600, 600);
    expect($(s.root, "[data-guide-message]").textContent).toBe(th("ui.guide.pose_hint", { text: th("quality.frontal_pose"), direction: th("ui.guide.pose.yaw_negative") }));
    expect($(s.root, '[data-check="pose"]').dataset.state).toBe("fail");
    expect($(s.root, '[data-check="face"]').dataset.state).toBe("pass");
    s.setResult(face({ smile: 0.8 }));
    runFrames(s, t + 600, 600);
    expect($(s.root, "[data-guide-message]").textContent).toBe(th("quality.smiling"));
    expect([...s.root.querySelectorAll<HTMLElement>("[data-check]")].map((d) => d.dataset.state)).toEqual(["pass", "pass", "pass", "fail"]);
  });

  it("takes one frame at the size of the stream, not mirrored, after one second of passing, then goes on", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const s = setup();
    await toFrontal(s);
    videoPlays(s.root);
    s.setResult(face());
    runFrames(s, 0, AUTO_CAPTURE_HOLD_MS - 2 * STEP);
    expect(s.canvases).toHaveLength(0);
    expect($(s.root, "[data-guide-message]").textContent).toBe(th("ui.guide.hold"));
    runFrames(s, AUTO_CAPTURE_HOLD_MS, 400);
    await vi.advanceTimersByTimeAsync(0);
    expect(s.canvases).toHaveLength(1); // one photo
    const calls = s.canvases[0].calls;
    expect(calls[0]).toEqual(["size", 1920, 1080]);
    expect(calls[1]).toEqual(["drawImage", $(s.root, "video"), 0, 0, 1920, 1080]);
    expect(calls.filter((c) => ["scale", "translate", "setTransform", "transform", "rotate"].includes(String(c[0])))).toEqual([]);
    expect(s.root.querySelector(".camera__flash--on")).not.toBeNull();
    expect(s.app.session.images.frontal).toMatchObject({ width: 1920, height: 1080, source: "camera", shrunk: false });
    expect(s.app.session.frontalValues!.iod_px).toBeCloseTo(0.06 * 1920, 6);
    expect(s.media!.streams[0].track.stop).toHaveBeenCalled(); // the light goes off
    await vi.advanceTimersByTimeAsync(FLASH_MS);
    expect(screenName(s.root)).toBe("profile");
  });

  it("starts counting again when a check fails before the second is up", async () => {
    const s = setup();
    await toFrontal(s);
    videoPlays(s.root);
    s.setResult(face());
    let t = runFrames(s, 0, 700);
    s.setResult(face({ eyes: 0.9 }));
    t = runFrames(s, t, 500); // closed for half a second: the average fails
    expect($(s.root, "[data-guide-message]").textContent).toBe(th("quality.eyes_closed"));
    expect(s.canvases).toHaveLength(0);
    s.setResult(face());
    // the closed eyes leave the average within half a second, then one whole second again
    t = runFrames(s, t, 900);
    await flush();
    expect(s.canvases).toHaveLength(0);
    runFrames(s, t, 900);
    await flush();
    expect(s.canvases).toHaveLength(1);
  });

  it("takes the photo by hand at any time, even with nothing detected", async () => {
    const s = setup({ landmarker: "never" });
    await toFrontal(s);
    $(s.root, '[data-action="capture"]').click();
    await flush();
    expect(s.canvases).toHaveLength(1);
    expect(s.app.session.images.frontal).toMatchObject({ source: "camera" });
    expect(s.app.session.frontalValues).toBeNull();
  });

  it("says the guide is not available when MediaPipe cannot be loaded, and keeps the button", async () => {
    const s = setup({ landmarker: "fails" });
    await toFrontal(s);
    await flush();
    expect($(s.root, "[data-guide-message]").textContent).toBe(th("ui.guide.unavailable"));
    expect(($(s.root, '[data-action="capture"]') as HTMLButtonElement).disabled).toBe(false);
  });

  it("says why where there is no camera, with no button and without loading MediaPipe", async () => {
    const s = setup({ media: null });
    await toFrontal(s, "30", "female", false);
    expect(s.root.querySelector('[data-action="camera-open"]')).toBeNull();
    expect(s.root.querySelector("[data-camera-part]")).toBeNull();
    expect($(s.root, '.capture__actions [data-camera-notice="unsupported"]').textContent).toBe(th("ui.camera.unsupported"));
    expect(s.root.querySelector('[data-action="choose-file"]')).not.toBeNull();
    expect(s.loadLandmarker).not.toHaveBeenCalled();
  });

  it("warns when the stream is under 1080 px high", async () => {
    const s = setup({ media: fakeMedia({ width: 1280, height: 720, deviceId: "cam-1" }) });
    await toFrontal(s);
    expect($(s.root, '[data-camera-notice="low_resolution"]').textContent).toBe(th("ui.camera.low_resolution", { width: 1280, height: 720, min: 1080 }));
    const full = setup();
    await toFrontal(full);
    expect(full.root.querySelector('[data-camera-notice="low_resolution"]')).toBeNull();
  });

  it("says what to do when the camera is not allowed, and tries again", async () => {
    const media = fakeMedia();
    media.getUserMedia.mockRejectedValueOnce(Object.assign(new Error("no"), { name: "NotAllowedError" }));
    const s = setup({ media });
    await toFrontal(s);
    expect($(s.root, '[data-camera-notice="denied"]').textContent).toContain(th("ui.camera.denied"));
    $(s.root, '[data-action="camera-retry"]').click();
    await flush();
    expect(s.root.querySelector("[data-camera-notice]")).toBeNull();
    expect(media.streams).toHaveLength(1);
  });

  it("lists the cameras when there is more than one and keeps the choice in memory", async () => {
    const media = fakeMedia(undefined, [
      { kind: "videoinput", deviceId: "cam-1", label: "Built-in" },
      { kind: "videoinput", deviceId: "cam-2", label: "" },
    ]);
    const s = setup({ media });
    await toFrontal(s);
    await flush();
    const select = $(s.root, "[data-camera-picker]") as HTMLSelectElement;
    expect([...select.options].map((o) => o.textContent)).toEqual(["Built-in", th("ui.camera.unnamed", { n: 2 })]);
    select.value = "cam-2";
    select.dispatchEvent(new Event("change"));
    await flush();
    expect(media.streams[0].track.stop).toHaveBeenCalled();
    expect(media.getUserMedia.mock.calls.at(-1)![0].video).toMatchObject({ deviceId: { exact: "cam-2" } });
    expect(s.app.session.cameraId).toBe("cam-2");
    s.app.session.reset(); // a setting of the page, like the language
    expect(s.app.session.cameraId).toBe("cam-2");
  });

  it("shows the image there is with 'use the same image', and the camera in its place when opened", async () => {
    const s = setup();
    await toFrontal(s);
    $(s.root, '[data-action="capture"]').click();
    await flush();
    const taken = s.app.session.images.frontal!;
    s.app.show("frontal");
    await flush();
    expect(s.media!.getUserMedia).toHaveBeenCalledTimes(1); // the camera stays off
    expect(s.root.querySelector("video")).toBeNull();
    const current = $(s.root, "[data-capture-main] [data-current-image]");
    expect($(current, "img").getAttribute("src")).toBe(taken.url);
    expect($(current, '[data-action="keep"]').textContent).toBe(th("ui.capture.keep"));
    expect($(current, '[data-action="keep"]').classList.contains("btn--secondary")).toBe(true);
    expect(s.root.querySelectorAll('[data-action="camera-open"]')).toHaveLength(1); // no second "take a new one" button

    await openCamera(s);
    expect(s.media!.getUserMedia).toHaveBeenCalledTimes(2);
    expect(s.root.querySelector("[data-current-image]")).toBeNull();
    expect(s.root.querySelector('[data-action="keep"]')).toBeNull();
    $(s.root, '[data-action="camera-close"]').click(); // closed before a new photo: the image is still the one there was
    expect(s.app.session.images.frontal).toBe(taken);
    expect($(s.root, "[data-current-image] img").getAttribute("src")).toBe(taken.url);
    $(s.root, '[data-action="keep"]').click();
    expect(screenName(s.root)).toBe("profile");
  });

  it("puts a new photo in the place of the image there is", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const s = setup();
    await toFrontal(s);
    $(s.root, '[data-action="capture"]').click();
    await flush();
    const first = s.app.session.images.frontal!;
    s.app.show("frontal");
    await openCamera(s);
    $(s.root, '[data-action="capture"]').click();
    await flush();
    await vi.advanceTimersByTimeAsync(FLASH_MS);
    expect(screenName(s.root)).toBe("profile");
    expect(s.app.session.images.frontal).not.toBe(first);
  });

  it("turns the camera off before the file dialog opens, so no photo is taken meanwhile", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const s = setup();
    await toFrontal(s);
    videoPlays(s.root);
    const input = $(s.root, 'input[type="file"]') as HTMLInputElement;
    const dialog = vi.spyOn(input, "click").mockImplementation(() => {});
    $(s.root, '[data-action="choose-file"]').click();
    expect(dialog).toHaveBeenCalledTimes(1);
    expect(s.media!.streams[0].track.stop).toHaveBeenCalledTimes(1);
    expect(s.root.querySelector('[data-action="camera-open"]')).not.toBeNull();
    s.setResult(face());
    runFrames(s, 0, AUTO_CAPTURE_HOLD_MS + 400);
    await vi.advanceTimersByTimeAsync(FLASH_MS);
    expect(s.canvases).toHaveLength(0);
    expect(screenName(s.root)).toBe("frontal");
  });
});

describe("stopping the camera", () => {
  it("stops the track when the screen is left", async () => {
    const s = setup();
    await toFrontal(s);
    const track = s.media!.streams[0].track;
    expect(track.stop).not.toHaveBeenCalled();
    $(s.root, '[data-action="back"]').click();
    expect(screenName(s.root)).toBe("home");
    expect(track.stop).toHaveBeenCalledTimes(1);
  });

  it("stops the track when the tab is hidden, and waits for the button when it is shown", async () => {
    const s = setup();
    await toFrontal(s);
    setHidden(true);
    expect(s.media!.streams[0].track.stop).toHaveBeenCalledTimes(1);
    expect(s.clock.pending).toBe(0); // the guide stops too
    setHidden(false);
    await flush();
    expect(s.media!.streams).toHaveLength(1); // not opened again by itself
    expect(s.root.querySelector("video")).toBeNull();
    expect($(s.root, ".capture__actions .btn--primary").dataset.action).toBe("camera-open");
    await openCamera(s);
    expect(s.media!.streams).toHaveLength(2);
    expect(s.media!.streams[1].track.stop).not.toHaveBeenCalled();
  });

  it("stops the track of the profile screen when the tab is hidden", async () => {
    const s = setup();
    await toFrontal(s);
    $(s.root, '[data-action="capture"]').click();
    await flush();
    s.app.show("profile");
    await openCamera(s);
    const track = s.media!.streams.at(-1)!.track;
    setHidden(true);
    expect(track.stop).toHaveBeenCalledTimes(1);
    setHidden(false);
    await flush();
    expect(s.root.querySelector('[data-action="camera-open"]')).not.toBeNull();
  });

  it("stops the track on start over", async () => {
    const s = setup();
    await toFrontal(s);
    s.app.restart();
    expect(s.media!.streams[0].track.stop).toHaveBeenCalledTimes(1);
    expect(screenName(s.root)).toBe("home");
  });

  it("stops a stream that arrives after the screen was left", async () => {
    let give: (stream: MediaStream) => void = () => {};
    const media = fakeMedia();
    const late = media.getUserMedia.getMockImplementation()!;
    media.getUserMedia.mockImplementationOnce((c) => new Promise((resolve) => (give = (st) => resolve(st))).then(() => late(c)));
    const s = setup({ media });
    await toFrontal(s);
    $(s.root, '[data-action="back"]').click();
    give(new MediaStream());
    await flush();
    expect(media.streams).toHaveLength(1);
    expect(media.streams[0].track.stop).toHaveBeenCalled();
  });

  it("stops the track on the profile screen too when it is left", async () => {
    const s = setup();
    await toFrontal(s);
    $(s.root, '[data-action="capture"]').click();
    await flush();
    s.app.show("profile");
    await openCamera(s);
    const track = s.media!.streams.at(-1)!.track;
    expect(screenName(s.root)).toBe("profile");
    $(s.root, '[data-action="skip-profile"]').click();
    expect(track.stop).toHaveBeenCalled();
  });
});

describe("the profile camera", () => {
  async function toProfile(s: ReturnType<typeof setup>) {
    await toFrontal(s);
    $(s.root, '[data-action="capture"]').click();
    await flush();
    s.app.show("profile");
    await flush();
    await openCamera(s);
  }

  it("draws the head from the side in the style of the frontal guide, not a photo", async () => {
    const s = setup();
    await toProfile(s);
    const stage = $(s.root, "[data-camera-stage]");
    const guide = $(stage, '[data-guide="profile"]');
    expect(guide.tagName.toLowerCase()).toBe("svg");
    expect(stage.querySelector("img")).toBeNull(); // no picture
    expect($(guide, "[data-outline]").getAttribute("d")).toBe(guideOutline("profile", { width: 1920, height: 1080 }).outline);
    // the same parts and classes as the frontal guide: the veil, then solid lines
    expect([...guide.children].map((c) => c.getAttribute("class"))).toEqual(["camera__veil", "camera__outline", "camera__outline"]);
    expect(stage.dataset.live).toBeUndefined();
    videoPlays(s.root);
    $(stage, "video").dispatchEvent(new Event("loadeddata"));
    expect(stage.dataset.live).toBe("");
    expect(s.root.textContent).toContain(th("ui.profile.guide_note"));
  });

  it("counts down 3 seconds with a beep each second, then takes one frame", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const s = setup();
    await toProfile(s);
    const before = s.canvases.length;
    $(s.root, '[data-action="countdown"]').click();
    expect(s.beeper.prepare).toHaveBeenCalledTimes(1);
    expect($(s.root, ".camera__count").textContent).toBe("3");
    await vi.advanceTimersByTimeAsync(1000);
    expect($(s.root, ".camera__count").textContent).toBe("2");
    await vi.advanceTimersByTimeAsync(1999);
    expect(s.canvases.length).toBe(before); // not yet
    expect(s.beeper.tick).toHaveBeenCalledTimes(COUNTDOWN_SECONDS);
    await vi.advanceTimersByTimeAsync(1);
    expect(s.beeper.shot).toHaveBeenCalledTimes(1);
    expect(s.canvases.length).toBe(before + 1);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(s.canvases.length).toBe(before + 1); // only one
    expect(s.canvases.at(-1)!.calls[1]).toEqual(["drawImage", expect.anything(), 0, 0, 1920, 1080]);
    // the photo to keep or take again, with the camera off
    expect(s.root.querySelector("[data-camera-shot]")).not.toBeNull();
    expect(s.media!.streams.at(-1)!.track.stop).toHaveBeenCalled();
  });

  it("can cancel the countdown", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const s = setup();
    await toProfile(s);
    const before = s.canvases.length;
    $(s.root, '[data-action="countdown"]').click();
    expect($(s.root, '[data-action="countdown"]').textContent).toBe(th("ui.profile.countdown_cancel"));
    $(s.root, '[data-action="countdown"]').click();
    await vi.advanceTimersByTimeAsync(5000);
    expect(s.canvases.length).toBe(before);
  });

  it("drops the photo not kept when the camera is closed", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const s = setup();
    await toProfile(s);
    $(s.root, '[data-action="countdown"]').click();
    await vi.advanceTimersByTimeAsync(COUNTDOWN_SECONDS * 1000);
    const shot = $(s.root, "[data-camera-shot] img").getAttribute("src")!;
    expect(s.root.querySelector('.capture__actions [data-action="camera-close"]')).not.toBeNull(); // the photo is part of the camera
    $(s.root, '[data-action="camera-close"]').click();
    expect(s.root.querySelector("[data-camera-shot]")).toBeNull();
    expect(s.app.deps.urlApi.revokeObjectURL).toHaveBeenCalledWith(shot);
    expect(s.app.session.images.profile).toBeNull();
    expect($(s.root, "[data-capture-main]").children).toHaveLength(0);
    await openCamera(s);
    expect(s.root.querySelector("[data-capture-main] video")).not.toBeNull();
    expect(s.root.querySelector("[data-camera-shot]")).toBeNull();
  });

  it("takes the photo again or keeps it", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const s = setup();
    await toProfile(s);
    $(s.root, '[data-action="countdown"]').click();
    await vi.advanceTimersByTimeAsync(COUNTDOWN_SECONDS * 1000);
    $(s.root, '[data-action="retake"]').click();
    await vi.advanceTimersByTimeAsync(0);
    expect(s.root.querySelector("video")).not.toBeNull();
    $(s.root, '[data-action="countdown"]').click();
    await vi.advanceTimersByTimeAsync(COUNTDOWN_SECONDS * 1000);
    $(s.root, '[data-action="use-photo"]').click();
    await vi.advanceTimersByTimeAsync(0);
    expect(screenName(s.root)).toBe("review");
    expect(s.app.session.images.profile).toMatchObject({ width: 1920, height: 1080, source: "camera" });
  });
});

describe("lab mode", () => {
  it("shows the live values under the picture, only in lab mode", async () => {
    const s = setup({ search: "?lab=1" });
    await toFrontal(s);
    videoPlays(s.root);
    s.setResult(face({ yaw: 4 }));
    runFrames(s, 0, 300);
    const live = $(s.root, '[data-lab="live"]');
    const value = (name: string) => [...live.querySelectorAll("dt")].find((d) => d.textContent === name)!.nextElementSibling!.textContent;
    expect(value("yaw_deg")).toBe("4.0");
    expect(value("iod_px")).toBe("115.2");
    expect(value("stream")).toBe("1920×1080");
    expect(Number(value("fps"))).toBeGreaterThan(0);
    for (const name of ["pitch_deg", "roll_deg", "smile", "jaw_open", "blink_left", "blink_right"]) expect(value(name)).not.toBe(th("ui.lab.no_value"));

    const plain = setup();
    await toFrontal(plain);
    expect(plain.root.querySelector('[data-lab="live"]')).toBeNull();
  });

  it("shows the web's values beside quality.frontal", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const result = fixture("lab").result;
    const s = setup({ search: "?lab=1" });
    s.app.deps.fetch = scriptedFetch(jsonResponse(200, HEALTH), jsonResponse(200, result));
    await toFrontal(s, "37", "female");
    videoPlays(s.root);
    s.setResult(face({ yaw: 3 }));
    runFrames(s, 0, AUTO_CAPTURE_HOLD_MS + 400);
    await vi.advanceTimersByTimeAsync(FLASH_MS);
    expect(screenName(s.root)).toBe("profile");
    await openCamera(s);
    $(s.root, '[data-action="countdown"]').click();
    await vi.advanceTimersByTimeAsync(COUNTDOWN_SECONDS * 1000);
    $(s.root, '[data-action="use-photo"]').click();
    await vi.advanceTimersByTimeAsync(0);
    $(s.root, '[data-action="send"]').click();
    await vi.advanceTimersByTimeAsync(0);
    expect(screenName(s.root)).toBe("results");

    const compare = $(s.root, '[data-lab="compare"]');
    const row = (name: string) => [...compare.querySelectorAll<HTMLElement>(`[data-value="${name}"] td`)].map((c) => c.textContent);
    expect(row("yaw_deg")).toEqual(["3.0", "1.2"]);
    expect(row("iod_px")).toEqual(["115.2", "62.4"]);
    expect(row("blink_left")).toEqual(["0.05", "0.71"]);
  });
});

describe("the MediaPipe files", () => {
  it("come from the CDN at the version of the package, and the model is the one of api/assets.lock.json", () => {
    const pkg = readJson<{ dependencies: Record<string, string> }>(join(WEB, "package.json"));
    expect(pkg.dependencies["@mediapipe/tasks-vision"]).toBe(MEDIAPIPE_VERSION);
    const installed = readJson<{ version: string }>(join(WEB, "node_modules", "@mediapipe", "tasks-vision", "package.json"));
    expect(installed.version).toBe(MEDIAPIPE_VERSION);
    expect(MEDIAPIPE_WASM_URL).toBe(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`);
    const lock = readJson<{ files: { name: string; url: string }[] }>(join(ROOT, "api", "assets.lock.json"));
    expect(FACE_LANDMARKER_MODEL_URL).toBe(lock.files.find((f) => f.name === "face_landmarker.task")!.url);
  });
});
