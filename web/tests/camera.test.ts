// camera.ts: the request, the real size, the errors, stopping the tracks, and the frame
// that is sent: the size of the stream, not mirrored (web.md section 3.1).
import { describe, expect, it, vi } from "vitest";
import { Camera, CameraError, cameraConstraints, cameraProblem, grabFrame, lowResolution } from "../src/camera.ts";
import { CAMERA_IDEAL_HEIGHT, CAMERA_IDEAL_WIDTH, CAPTURE_JPEG_QUALITY } from "../src/constants.ts";
import { fakeMedia, fakeStream, recordingCanvas } from "./helpers.ts";

describe("the request", () => {
  it("asks for the largest size with ideal values, no sound, and the chosen camera exactly", () => {
    expect(cameraConstraints(null)).toEqual({ audio: false, video: { width: { ideal: CAMERA_IDEAL_WIDTH }, height: { ideal: CAMERA_IDEAL_HEIGHT } } });
    expect(cameraConstraints("cam-2").video).toMatchObject({ deviceId: { exact: "cam-2" } });
    expect(CAMERA_IDEAL_HEIGHT).toBeGreaterThan(1080);
  });

  it("warns under 1080 px of height", () => {
    expect(lowResolution({ width: 1280, height: 720 })).toBe(true);
    expect(lowResolution({ width: 1920, height: 1080 })).toBe(false);
  });
});

describe("Camera", () => {
  it("reads the real size from the track's settings", async () => {
    const media = fakeMedia({ width: 1920, height: 1080, deviceId: "cam-1" });
    const camera = new Camera(media, document.createElement("video"));
    const opened = await camera.open(null);
    expect(opened).toEqual({ size: { width: 1920, height: 1080 }, deviceId: "cam-1" });
    expect(camera.active).toBe(true);
    expect(camera.size).toEqual({ width: 1920, height: 1080 });
    expect(camera.video.srcObject).toBe(media.streams[0].stream);
    expect(camera.video.muted).toBe(true);
  });

  it("stops every track at once", async () => {
    const media = fakeMedia();
    const camera = new Camera(media, document.createElement("video"));
    await camera.open(null);
    camera.stop();
    expect(media.streams[0].track.stop).toHaveBeenCalledTimes(1);
    expect(camera.active).toBe(false);
    expect(camera.video.srcObject).toBeNull();
  });

  it("stops a stream that arrives after stop()", async () => {
    let give: (s: MediaStream) => void = () => {};
    const late = fakeStream();
    const media = { getUserMedia: vi.fn(() => new Promise<MediaStream>((resolve) => (give = resolve))), enumerateDevices: vi.fn(async () => []) };
    const camera = new Camera(media, document.createElement("video"));
    const opening = camera.open(null);
    camera.stop();
    give(late.stream);
    await expect(opening).rejects.toBeInstanceOf(CameraError);
    expect(late.track.stop).toHaveBeenCalledTimes(1);
    expect(camera.active).toBe(false);
  });

  it("opening again stops the stream before", async () => {
    const media = fakeMedia();
    const camera = new Camera(media, document.createElement("video"));
    await camera.open(null);
    await camera.open("cam-2");
    expect(media.streams[0].track.stop).toHaveBeenCalled();
    expect(media.streams[1].track.stop).not.toHaveBeenCalled();
    expect(media.getUserMedia.mock.calls[1][0].video).toMatchObject({ deviceId: { exact: "cam-2" } });
  });

  it("falls back to the default camera when the chosen one is gone", async () => {
    const media = fakeMedia();
    media.getUserMedia.mockRejectedValueOnce(Object.assign(new Error("gone"), { name: "OverconstrainedError" }));
    const camera = new Camera(media, document.createElement("video"));
    const opened = await camera.open("unplugged");
    expect(media.getUserMedia).toHaveBeenCalledTimes(2);
    expect(media.getUserMedia.mock.calls[1][0].video).not.toHaveProperty("deviceId");
    expect(opened.deviceId).toBe("cam-1");
  });

  it("names each problem so the screen can say what to do", async () => {
    const named = (name: string) => Object.assign(new Error(name), { name });
    expect(cameraProblem(named("NotAllowedError"))).toBe("denied");
    expect(cameraProblem(named("SecurityError"))).toBe("denied");
    expect(cameraProblem(named("NotFoundError"))).toBe("not_found");
    expect(cameraProblem(named("OverconstrainedError"))).toBe("overconstrained");
    expect(cameraProblem(named("NotReadableError"))).toBe("in_use");
    expect(cameraProblem(new Error("?"))).toBe("failed");
    const media = fakeMedia();
    media.getUserMedia.mockRejectedValueOnce(named("NotAllowedError"));
    await expect(new Camera(media, document.createElement("video")).open(null)).rejects.toMatchObject({ problem: "denied" });
    await expect(new Camera(null, document.createElement("video")).open(null)).rejects.toMatchObject({ problem: "unsupported" });
  });

  it("lists the video inputs only", async () => {
    const media = fakeMedia();
    media.enumerateDevices.mockResolvedValue([
      { kind: "videoinput", deviceId: "a", label: "Front", groupId: "", toJSON: () => ({}) },
      { kind: "audioinput", deviceId: "m", label: "Mic", groupId: "", toJSON: () => ({}) },
      { kind: "videoinput", deviceId: "b", label: "", groupId: "", toJSON: () => ({}) },
    ]);
    expect(await new Camera(media, document.createElement("video")).cameras()).toEqual([
      { deviceId: "a", label: "Front" },
      { deviceId: "b", label: "" },
    ]);
  });
});

describe("the frame that is sent", () => {
  it("is drawn at the size of the stream, without a scale or a mirror, and encoded as JPEG 0.92", async () => {
    const video = document.createElement("video");
    const canvas = recordingCanvas();
    const blob = await grabFrame(video, { width: 1920, height: 1080 }, () => canvas);
    expect(blob.type).toBe("image/jpeg");
    expect(canvas.calls).toEqual([
      ["size", 1920, 1080],
      ["drawImage", video, 0, 0, 1920, 1080],
      ["toBlob", "image/jpeg", CAPTURE_JPEG_QUALITY],
    ]);
    expect(CAPTURE_JPEG_QUALITY).toBe(0.92);
  });

  it("is taken by the camera at the size of its stream", async () => {
    const media = fakeMedia({ width: 3840, height: 2160, deviceId: "cam-4k" });
    const camera = new Camera(media, document.createElement("video"));
    await camera.open(null);
    const canvas = recordingCanvas();
    const shot = await camera.capture(() => canvas);
    expect(shot.size).toEqual({ width: 3840, height: 2160 });
    expect(canvas.calls[0]).toEqual(["size", 3840, 2160]);
    expect(canvas.calls[1]).toEqual(["drawImage", camera.video, 0, 0, 3840, 2160]);
  });

  it("is refused before the camera has a frame", async () => {
    await expect(grabFrame(document.createElement("video"), { width: 0, height: 0 }, recordingCanvas)).rejects.toThrow();
  });
});
