// The camera of this computer (web.md section 3.1).
//
// - Ask for the largest size with a large `ideal`, then read the real size from
//   track.getSettings(); a height under CAMERA_MIN_HEIGHT gets a warning on the screen.
// - A photo is one frame of the video drawn on a canvas of the size of the stream and
//   encoded as JPEG CAPTURE_JPEG_QUALITY: not scaled and not mirrored, because the api
//   tells the person's left eye from the right on an image that is not mirrored
//   (api.md section 2.3). Only the preview is mirrored, with CSS.
// - stop() ends every track at once; a stream that arrives after stop() is ended as
//   soon as it comes. The screens stop the camera when they are left, when the tab is
//   hidden and on "start over".
// - Nothing is kept: no frame, no photo and no setting outside memory.
import { CAMERA_IDEAL_HEIGHT, CAMERA_IDEAL_WIDTH, CAMERA_MIN_HEIGHT, CAPTURE_JPEG_QUALITY } from "./constants.ts";
import type { Size } from "./images.ts";

/** The part of navigator.mediaDevices the page uses; tests pass a fake. */
export interface MediaDevicesLike {
  getUserMedia(constraints: MediaStreamConstraints): Promise<MediaStream>;
  enumerateDevices(): Promise<MediaDeviceInfo[]>;
}

export type CameraProblem = "unsupported" | "denied" | "not_found" | "overconstrained" | "in_use" | "failed";

export class CameraError extends Error {
  readonly problem: CameraProblem;
  constructor(problem: CameraProblem, cause?: unknown) {
    super(`camera: ${problem}`, { cause });
    this.problem = problem;
  }
}

/** Which problem a getUserMedia error is, from its name. */
export function cameraProblem(error: unknown): CameraProblem {
  if (error instanceof CameraError) return error.problem;
  const name = (error as { name?: unknown } | null)?.name;
  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
    case "SecurityError":
      return "denied";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "not_found";
    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
      return "overconstrained";
    case "NotReadableError":
    case "TrackStartError":
    case "AbortError":
      return "in_use";
    default:
      return "failed";
  }
}

/** The text of each problem; every one says what to do (web.md section 5.4). */
export const CAMERA_PROBLEM_TEXT: Record<CameraProblem, string> = {
  unsupported: "ui.camera.unsupported",
  denied: "ui.camera.denied",
  not_found: "ui.camera.not_found",
  overconstrained: "ui.camera.overconstrained",
  in_use: "ui.camera.in_use",
  failed: "ui.camera.failed",
};

/** The request: the largest size the camera gives, from the chosen camera if there is one. */
export function cameraConstraints(deviceId: string | null): MediaStreamConstraints {
  const video: MediaTrackConstraints = { width: { ideal: CAMERA_IDEAL_WIDTH }, height: { ideal: CAMERA_IDEAL_HEIGHT } };
  if (deviceId) video.deviceId = { exact: deviceId };
  return { audio: false, video };
}

/** The stream is too small for an IOD of min_iod_px at 1 m (web.md section 3.1). */
export function lowResolution(size: Size, minHeight = CAMERA_MIN_HEIGHT): boolean {
  return size.height < minHeight;
}

export interface CameraChoice {
  deviceId: string;
  label: string; // may be empty before the camera is allowed
}

export interface OpenedCamera {
  size: Size; // from track.getSettings()
  deviceId: string | null;
}

/** The video element and its stream; one per capture screen. */
export class Camera {
  readonly video: HTMLVideoElement;
  private media: MediaDevicesLike | null;
  private stream: MediaStream | null = null;
  private settings: MediaTrackSettings | null = null;
  private generation = 0;

  constructor(media: MediaDevicesLike | null, video: HTMLVideoElement) {
    this.media = media;
    this.video = video;
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
    video.setAttribute("playsinline", "");
  }

  get supported(): boolean {
    return this.media !== null && typeof this.media.getUserMedia === "function";
  }

  get active(): boolean {
    return this.stream !== null;
  }

  /** The size of a frame: the video's own size once it plays, else the track's settings. */
  get size(): Size {
    const { videoWidth, videoHeight } = this.video;
    if (videoWidth > 0 && videoHeight > 0) return { width: videoWidth, height: videoHeight };
    return { width: this.settings?.width ?? 0, height: this.settings?.height ?? 0 };
  }

  /** The video has a frame to show and to draw. */
  get ready(): boolean {
    return this.active && this.video.readyState >= 2 && this.size.width > 0 && this.size.height > 0;
  }

  /**
   * Open the camera deviceId (or the default one) and show it in the video.
   * A camera that has gone away is replaced by the default one. Rejects with CameraError.
   */
  async open(deviceId: string | null): Promise<OpenedCamera> {
    if (!this.supported) throw new CameraError("unsupported");
    this.stop();
    const generation = ++this.generation;
    let stream: MediaStream;
    let used = deviceId;
    try {
      stream = await this.media!.getUserMedia(cameraConstraints(deviceId));
    } catch (error) {
      const problem = cameraProblem(error);
      if (!deviceId || (problem !== "overconstrained" && problem !== "not_found")) throw new CameraError(problem, error);
      if (generation !== this.generation) throw new CameraError("failed", error);
      used = null;
      try {
        stream = await this.media!.getUserMedia(cameraConstraints(null));
      } catch (again) {
        throw new CameraError(cameraProblem(again), again);
      }
    }
    if (generation !== this.generation) {
      // stopped or reopened while asking: end this stream at once
      endStream(stream);
      throw new CameraError("failed");
    }
    const track = stream.getVideoTracks()[0];
    if (!track) {
      endStream(stream);
      throw new CameraError("not_found");
    }
    this.stream = stream;
    this.settings = track.getSettings();
    this.video.srcObject = stream;
    try {
      await this.video.play();
    } catch {
      // autoplay of a muted video is allowed; a refusal still leaves the stream shown
    }
    if (generation !== this.generation) throw new CameraError("failed");
    return { size: { width: this.settings.width ?? this.size.width, height: this.settings.height ?? this.size.height }, deviceId: this.settings.deviceId ?? used };
  }

  /** End every track now and let go of the stream. */
  stop(): void {
    this.generation += 1;
    if (this.stream) endStream(this.stream);
    this.stream = null;
    this.settings = null;
    if (this.video.srcObject) this.video.srcObject = null;
  }

  /** The cameras of this computer. */
  async cameras(): Promise<CameraChoice[]> {
    if (!this.media) return [];
    try {
      const devices = await this.media.enumerateDevices();
      return devices.filter((d) => d.kind === "videoinput" && d.deviceId).map((d) => ({ deviceId: d.deviceId, label: d.label }));
    } catch {
      return [];
    }
  }

  /** One frame as JPEG, at the size of the stream and not mirrored. */
  capture(makeCanvas: () => CanvasLike = browserCanvas): Promise<{ blob: Blob; size: Size }> {
    const size = this.size;
    return grabFrame(this.video, size, makeCanvas).then((blob) => ({ blob, size }));
  }
}

function endStream(stream: MediaStream): void {
  for (const track of stream.getTracks()) track.stop();
}

// ---------------------------------------------------------------- frames

/** What grabFrame needs of a canvas; tests pass one that records the calls. */
export interface CanvasLike {
  width: number;
  height: number;
  getContext(type: "2d"): Pick<CanvasRenderingContext2D, "drawImage"> | null;
  toBlob(callback: (blob: Blob | null) => void, type?: string, quality?: number): void;
}

export const browserCanvas = (): CanvasLike => document.createElement("canvas");

/**
 * Draw source on a canvas of exactly size and encode it as JPEG CAPTURE_JPEG_QUALITY.
 * No scale, no mirror and no other transform: the pixels are those of the stream.
 */
export async function grabFrame(source: CanvasImageSource, size: Size, makeCanvas: () => CanvasLike = browserCanvas): Promise<Blob> {
  if (!(size.width > 0 && size.height > 0)) throw new Error("the camera has no frame yet");
  const canvas = makeCanvas();
  canvas.width = size.width;
  canvas.height = size.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no 2d canvas");
  ctx.drawImage(source, 0, 0, size.width, size.height);
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", CAPTURE_JPEG_QUALITY));
  canvas.width = canvas.height = 0; // let the pixels go at once
  if (!blob) throw new Error("the canvas gave no JPEG");
  return blob;
}
