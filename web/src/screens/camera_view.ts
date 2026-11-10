// The live camera of the capture screens (web.md sections 3.1 and 6.3): the video in a
// stage of the stream's shape with the guide of the screen over it (guide_layer.ts), the
// camera picker when there is more than one camera, the warning of a stream under
// 1080 px, and a message that says what to do when the camera cannot be opened.
//
// "Preparing the camera" stays on the stage until the video shows a frame; only then is
// the stage marked live (data-live) and the guide shown, so the two never overlap.
//
// The camera opens only when the user presses the button at the bottom of the screen
// (capture.ts): never by itself when a screen is shown or when the tab comes back. It
// is stopped when the user presses it again, when the screen is left (dispose), when
// the tab is hidden (the screen then shows the button again), and when the owner no
// longer needs it. The chosen camera is kept in the session, in memory only.
import type { App } from "../app.ts";
import { CAMERA_PROBLEM_TEXT, Camera, CameraError, cameraProblem, lowResolution, type CameraChoice } from "../camera.ts";
import { CAMERA_MIN_HEIGHT, FLASH_MS } from "../constants.ts";
import { button, el, uid } from "../dom.ts";
import type { Size } from "../images.ts";
import { guideLayer, type GuideKind } from "./guide_layer.ts";

/** Whether the page can ask for a camera at all; where it cannot, the screen says why instead. */
export function cameraSupported(app: App): boolean {
  return Boolean(app.deps.media && typeof app.deps.media.getUserMedia === "function");
}

/** The camera block in the middle of a capture screen while the camera is on. */
export interface CameraPart {
  element: HTMLElement;
  /** Show the live camera and open it: the user pressed the button. */
  open: () => void;
  /** Turn the camera off, and drop a photo that was not kept. */
  close: () => void;
  dispose: () => void;
}

export interface CameraPartHandlers {
  /** The photo is now the image of this screen: go on. */
  onDone: () => void;
  /** The camera turned itself off because the tab was hidden: show the screen without it. */
  onClosed: () => void;
}

export interface LiveCameraHandlers {
  /** The video shows the stream; size is the frame of the stream and of every photo. */
  onOpen: (size: Size) => void;
  /** The stream has ended: stopped, hidden or failed. */
  onClose: () => void;
  /** The tab was hidden and the camera stopped; it stays off until the user opens it again. */
  onHidden: () => void;
}

/** Read each time: the tab may be hidden while the camera opens. */
const tabHidden = (): boolean => document.visibilityState === "hidden";

export class LiveCamera {
  readonly camera: Camera;
  readonly element: HTMLElement;
  /** Where the owner puts its overlays, above the video. */
  readonly stage: HTMLElement;
  private app: App;
  private guide: GuideKind;
  private handlers: LiveCameraHandlers;
  private video: HTMLVideoElement;
  private wait: HTMLElement;
  private flashBox: HTMLElement;
  private picker: HTMLElement;
  private notices: HTMLElement;
  private wanted = false; // the owner wants the camera on
  private opening = false;
  private disposed = false;
  private flashTimer: ReturnType<typeof setTimeout> | null = null;
  private onVisibility = () => this.visibilityChanged();

  constructor(app: App, guide: GuideKind, handlers: LiveCameraHandlers) {
    this.app = app;
    this.guide = guide;
    this.handlers = handlers;
    const t = app.t;
    const video = el("video", { class: "camera__video", attrs: { "aria-label": t("ui.camera.video_label") } });
    this.video = video;
    this.camera = new Camera(app.deps.media, video);
    this.wait = el("p", { class: "camera__wait", text: t("ui.camera.preparing"), attrs: { hidden: true } });
    this.flashBox = el("div", { class: "camera__flash", attrs: { "aria-hidden": "true" } });
    this.stage = el("div", { class: "camera__stage", attrs: { "data-camera-stage": "" } }, [video, this.wait, this.flashBox]);
    for (const name of ["loadeddata", "playing"]) video.addEventListener(name, () => this.showLive());
    this.picker = el("div", { class: "camera__picker", attrs: { hidden: true } });
    this.notices = el("div", { class: "camera__notices", attrs: { "aria-live": "polite" } });
    this.element = el("div", { class: "camera" }, [this.picker, this.stage, this.notices]);
    document.addEventListener("visibilitychange", this.onVisibility);
  }

  get supported(): boolean {
    return this.camera.supported;
  }

  get active(): boolean {
    return this.camera.active;
  }

  /** Open the camera; false when it cannot be opened, with a message saying what to do. */
  async start(): Promise<boolean> {
    if (this.disposed) return false;
    this.wanted = true;
    this.clearProblem();
    if (!this.camera.supported) {
      this.showProblem("unsupported", false);
      return false;
    }
    if (tabHidden()) return false;
    this.opening = true;
    this.wait.hidden = false;
    try {
      const opened = await this.camera.open(this.app.session.cameraId);
      this.opening = false;
      if (this.disposed || !this.wanted) {
        this.camera.stop();
        return false;
      }
      if (opened.deviceId && this.app.session.cameraId && opened.deviceId !== this.app.session.cameraId) this.app.session.cameraId = null; // the chosen one is gone
      const size = this.camera.size.width > 0 ? this.camera.size : opened.size;
      // the shape of the stream, for the stage and for the width of the camera block
      this.element.style.setProperty("--camera-aspect", `${size.width} / ${size.height}`);
      this.element.style.setProperty("--camera-ratio", String(size.width / size.height));
      this.stage.querySelector("[data-guide]")?.remove();
      this.stage.insertBefore(guideLayer(this.guide, size), this.wait); // hidden until the stage is live
      this.showLive(); // when a frame is there already
      this.showResolution(opened.size);
      void this.refreshPicker(opened.deviceId);
      this.handlers.onOpen(size);
      return true;
    } catch (error) {
      this.opening = false;
      this.wait.hidden = true;
      if (this.disposed || !this.wanted || tabHidden()) return false;
      this.showProblem(error instanceof CameraError ? error.problem : cameraProblem(error), true);
      this.handlers.onClose();
      return false;
    }
  }

  /** End the stream; the owner no longer needs the camera. */
  stop(): void {
    this.wanted = false;
    this.end();
  }

  /** A short white flash over the picture when a photo is taken. */
  flash(): void {
    this.flashBox.classList.remove("camera__flash--on");
    void this.flashBox.offsetWidth; // start the animation again
    this.flashBox.classList.add("camera__flash--on");
    if (this.flashTimer) clearTimeout(this.flashTimer);
    this.flashTimer = setTimeout(() => this.flashBox.classList.remove("camera__flash--on"), FLASH_MS);
  }

  dispose(): void {
    this.disposed = true;
    this.stop();
    if (this.flashTimer) clearTimeout(this.flashTimer);
    document.removeEventListener("visibilitychange", this.onVisibility);
  }

  private end(): void {
    const was = this.camera.active || this.opening;
    this.opening = false;
    this.camera.stop();
    this.wait.hidden = true;
    delete this.stage.dataset.live;
    if (was) this.handlers.onClose();
  }

  /** The video shows the camera: take away "preparing" and show the guide. */
  private showLive(): void {
    // a frame to show (HAVE_CURRENT_DATA), not only the event: some fire "playing" on play()
    if (this.disposed || !this.wanted || !this.camera.active || this.video.readyState < 2) return;
    this.wait.hidden = true;
    this.stage.dataset.live = "";
  }

  private visibilityChanged(): void {
    if (this.disposed || !tabHidden() || !(this.wanted || this.camera.active || this.opening)) return;
    this.stop();
    this.handlers.onHidden();
  }

  private showResolution(size: Size): void {
    if (!lowResolution(size)) return;
    this.notices.append(
      el("p", {
        class: "notice notice--camera",
        text: this.app.t("ui.camera.low_resolution", { width: size.width, height: size.height, min: CAMERA_MIN_HEIGHT }),
        attrs: { "data-camera-notice": "low_resolution" },
      }),
    );
  }

  private showProblem(problem: keyof typeof CAMERA_PROBLEM_TEXT, retry: boolean): void {
    const children: Node[] = [el("p", { text: this.app.t(CAMERA_PROBLEM_TEXT[problem]) })];
    if (retry) children.push(button(this.app.t("ui.camera.retry"), () => void this.start(), { class: "btn btn--secondary", attrs: { "data-action": "camera-retry" } }));
    this.notices.replaceChildren(el("div", { class: "notice notice--camera", attrs: { "data-camera-notice": problem } }, children));
    this.stage.hidden = true;
  }

  private clearProblem(): void {
    this.notices.replaceChildren();
    this.stage.hidden = false;
  }

  /** The camera picker, only when there is more than one camera. */
  private async refreshPicker(current: string | null): Promise<void> {
    const cameras: CameraChoice[] = await this.camera.cameras();
    if (this.disposed) return;
    if (cameras.length < 2) {
      this.picker.hidden = true;
      this.picker.replaceChildren();
      return;
    }
    const id = uid("camera");
    const select = el("select", { class: "select", attrs: { id, "data-camera-picker": "" } });
    cameras.forEach((c, i) => {
      const option = el("option", { text: c.label || this.app.t("ui.camera.unnamed", { n: i + 1 }), attrs: { value: c.deviceId } });
      option.selected = c.deviceId === current;
      select.append(option);
    });
    select.addEventListener("change", () => {
      this.app.session.cameraId = select.value;
      this.end();
      void this.start();
    });
    this.picker.replaceChildren(el("label", { text: this.app.t("ui.camera.choose"), attrs: { for: id } }), select);
    this.picker.hidden = false;
  }
}
