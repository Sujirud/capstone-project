// Values used only by the web. The criteria shared with the api come from
// shared/quality.json, the same file the api reads (web.md section 4.1).
import quality from "../../shared/quality.json" with { type: "json" };

export const SHARED_QUALITY = quality;

// api (web.md sections 4.1 and 4.2); the dev server proxies both paths to the api
export const HEALTH_PATH = "/health";
export const ANALYZE_PATH = "/analyze";
export const ANALYZE_TIMEOUT_MS = 60_000;
export const HEALTH_TIMEOUT_MS = 5_000;

// Size of a request (web.md section 3.4): the images must fit in request.max_bytes,
// less room for the multipart boundaries, the part headers and meta
export const MAX_REQUEST_BYTES = quality.request.max_bytes;
export const REQUEST_OVERHEAD_BYTES = 64 * 1024;
export const IMAGE_BUDGET_BYTES = MAX_REQUEST_BYTES - REQUEST_OVERHEAD_BYTES;

// Re-encoding an image that is too large (web.md section 3.4)
export const SHRINK_JPEG_QUALITY = 0.92;
export const SHRINK_MAX_SIDE_PX = 4032;

// The file chooser: JPEG first, so Safari turns iPhone HEIC photos into JPEG; never
// image/heic, which makes Safari 17+ convert to HEIC instead (web.md section 3.4)
export const FILE_ACCEPT = "image/jpeg,image/png";
export const SNIFF_BYTES = 8;

// Digits of the numbers on the results page
export const DIGITS = {
  percent: 1,
  deg: 1,
  ratio: 2,
  probability: 2, // lab mode only
} as const;

// URL parameters
export const LAB_PARAM = "lab"; // ?lab=1 (web.md section 7)
export const LANG_PARAM = "lang"; // ?lang=th (web.md section 8.4)
export const DEMO_PARAM = "demo"; // ?demo=<fixture> while developing only

// ---------------------------------------------------------------- camera (web.md section 3.1)

// Ask for the most the camera gives with a large ideal; the real size comes from the track
export const CAMERA_IDEAL_WIDTH = 3840;
export const CAMERA_IDEAL_HEIGHT = 2160;
// Below this height the IOD at 1 m falls under min_iod_px of shared/quality.json (web.md section 3.1)
export const CAMERA_MIN_HEIGHT = 1080;
// A frame of the stream, at its full size and not mirrored, as JPEG (web.md section 3.1)
export const CAPTURE_JPEG_QUALITY = 0.92;

// ---------------------------------------------------------------- Face Landmarker (web.md sections 3.2 and 8.1)

// The wasm files come from the CDN at the version of the package in package.json
// (tests/capture.test.ts checks that they match); the model is the file the api uses
// (api/assets.lock.json).
export const MEDIAPIPE_VERSION = "1.2.0";
export const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
export const FACE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
export const MAX_FACES = 2; // to know that another face is in the picture, as the api does
export const GUIDANCE_MAX_FPS = 15;
// numFaces above 1 turns off the smoothing across frames, so the values are averaged
export const AVERAGE_WINDOW_MS = 500;
// Every check passes this long without a break, then one frame is taken
export const AUTO_CAPTURE_HOLD_MS = 1000;

// ---------------------------------------------------------------- the pose guide (web.md sections 3.2 and 9)

// The head of the guide is this share of the image width at the roots of the ears: the
// face between points 234 and 454 at 1 m with a 1080p camera of about 70 degrees across
// is about 0.11 of the image, plus room around the face. A face wider than the guide is
// too close. Both guides are drawn to this width (screens/guide_layer.ts), so the
// profile is taken from the same distance. Measured at 1 m with the team's
// Mac M2 camera, a 1080 x 1920 stream: the face was 184.9 px wide with an IOD of 78.4 px,
// so the guide is the 217.5 px floor below and the face fills 85% of it; 0.13 stays
// (README, "the guide frame at 1 m").
export const GUIDE_FRAME_WIDTH = 0.13;
// The guide is never narrower than this many times min_iod_px, so the face can fit the
// guide with an IOD that passes: the face between 234 and 454 is about 2.4 IOD wide
// (a SCUT-FBP5500 face measured 2.41 in the browser), plus room. On a stream narrower
// than about 1700 px, such as 1280 x 720, 0.13 of the width is less than that, and
// "too close" and "too small" could never pass together.
export const GUIDE_FACE_WIDTH_PER_IOD = 2.9;
export const GUIDE_FRAME_CENTER_Y = 0.46; // the middle of the head of the guide, as a share of the image height
export const FLASH_MS = 250;

// ---------------------------------------------------------------- the profile countdown (web.md section 3.3)

export const COUNTDOWN_SECONDS = 3;
// Beeps from Web Audio, no sound files: one each second, a higher one with the photo
export const BEEP_TICK = { hz: 660, ms: 120 } as const;
export const BEEP_SHOT = { hz: 990, ms: 320 } as const;
export const BEEP_GAIN = 0.2;
