"""The fake camera and the files of the end-to-end test (web.md section 8.1).

Run from the project root (the same on macOS and Windows), after frontal_check.py and profile_check.py:

    uv run python model/scripts/make_e2e_video.py

- Picks a SCUT-FBP5500 test image that passed every check in data/labels/frontal_mediapipe.csv
  (frontal_check.py), with the smallest pose and expression and a face large enough to be only
  shrunk, then scales it so the IOD that facemeasure measures on the frame is about
  TARGET_IOD_PX and puts it in the middle of a plain FRAME_SIZE frame. The frame is
  measured again with analyze_frontal, as the api does, and must pass every check;
  otherwise the next image is tried.
- Writes to e2e/ in outputs (data/outputs/e2e/, never committed):
  frontal.mjpeg  FRAMES copies of the frame as JPEG, for Chromium's
                 --use-file-for-fake-video-capture (Chromium plays an .mjpeg at 30 fps, in a loop)
  exif_orientation_6.jpg  the same frame stored turned on its side, with EXIF
                 orientation 6: it must show upright, with the lines in place, when it is
                 chosen from the computer
  e2e.json       what the test reads: these files, a FEI full profile that passed in
                 data/labels/fei_profile.csv (profile_check.py), the IOD measured on the frame
                 and the names of the images
- Nothing random: running it again gives the same files.
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from facemeasure import fei, image_io, paths
from facemeasure.frontal import analyze_frontal
from facemeasure.landmarks import FaceModels

FRAME_SIZE = (1920, 1080)  # a 1080p stream (web.md section 3.1)
TARGET_IOD_PX = 90.0  # about the IOD of a face 1 m from a 1080p camera, above min_iod_px
IOD_TOLERANCE_PX = 3.0
BACKGROUND = (233, 228, 222)  # a plain wall
FRAMES = 30  # one second at the 30 fps Chromium plays an .mjpeg at
JPEG_QUALITY = 95
CANDIDATES = 10  # images tried, best first
OUT_FOLDER = "e2e"
VIDEO_NAME = "frontal.mjpeg"
EXIF_NAME = "exif_orientation_6.jpg"
CONFIG_NAME = "e2e.json"
EXIF_ORIENTATION_TAG = 0x0112
MAX_JPEG_DIFF = 4.0  # mean absolute difference of 0-255 values left by one JPEG save
SCUT_IMAGES = "Images"


def read_rows(name: str) -> list[dict]:
    path = paths.path("labels") / name
    if not path.is_file():
        raise SystemExit(f"{path} is missing; run frontal_check.py or profile_check.py first")
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def score(row: dict) -> float:
    """Smaller is better: a straight face with a neutral expression and open eyes."""
    pose = sum(abs(float(row[k])) for k in ("yaw_deg", "pitch_deg", "roll_deg"))
    face = float(row["smile"]) + float(row["jaw_open"]) + max(float(row["blink_left"]), float(row["blink_right"]))
    return pose + 20.0 * face


def frontal_candidates(rows: list[dict]) -> list[dict]:
    """SCUT test images that passed every check, large enough to be shrunk to TARGET_IOD_PX, best first."""
    ok = [
        r
        for r in rows
        if r["source"] == "scut" and r["split"] == "test" and r["face_count"] == "1" and r["failed"] == ""
        and float(r["iod_px"]) >= TARGET_IOD_PX
    ]
    return sorted(ok, key=lambda r: (score(r), r["image"]))


def profile_choice(rows: list[dict]) -> Path:
    """The first FEI full profile, by name, that was measured without a failed code."""
    for r in sorted(rows, key=lambda r: r["image"]):
        if r["pose"] == "full" and r["failed"] == "" and r["read_error"] == "":
            path = fei.images_dir() / r["image"]
            if path.is_file():
                return path
    raise SystemExit("no FEI full profile without a failed code in fei_profile.csv")


def compose(face: np.ndarray, scale: float) -> np.ndarray:
    """The face image scaled by scale, in the middle of a plain frame."""
    h, w = face.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = Image.fromarray(face).resize(size, Image.Resampling.LANCZOS)
    frame = Image.new("RGB", FRAME_SIZE, BACKGROUND)
    frame.paste(small, ((FRAME_SIZE[0] - size[0]) // 2, (FRAME_SIZE[1] - size[1]) // 2))
    return np.asarray(frame)


def place(face: np.ndarray, iod_px: float, models: FaceModels) -> tuple[np.ndarray, float, tuple[str, ...]]:
    """A frame with the IOD near TARGET_IOD_PX: scale by the IOD of the label, then correct once."""
    scale = TARGET_IOD_PX / iod_px
    for _ in range(2):
        frame = compose(face, scale)
        result = analyze_frontal(frame, models)
        if result.quality is None or result.quality.iod_px is None:
            return frame, 0.0, result.failed
        measured = float(result.quality.iod_px)
        if abs(measured - TARGET_IOD_PX) <= IOD_TOLERANCE_PX:
            break
        scale *= TARGET_IOD_PX / measured
    return frame, measured, result.failed


def jpeg(frame: np.ndarray, **save) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=JPEG_QUALITY, **save)
    return buffer.getvalue()


def exif_rotated(frame: np.ndarray) -> bytes:
    """The frame stored turned 90 degrees, with the EXIF orientation that turns it back (6)."""
    # orientation 6 shows the stored pixels turned 90 degrees clockwise, so they are stored
    # turned 90 degrees counter-clockwise (np.rot90 with k=1)
    stored = np.ascontiguousarray(np.rot90(frame, k=1))
    exif = Image.Exif()
    exif[EXIF_ORIENTATION_TAG] = 6
    data = jpeg(stored, exif=exif.tobytes())
    upright = image_io.read_image(data)
    if upright.shape != frame.shape:
        raise RuntimeError(f"the EXIF image reads back as {upright.shape}, expected {frame.shape}")
    diff = float(np.abs(upright.astype(np.int16) - frame.astype(np.int16)).mean())
    if diff > MAX_JPEG_DIFF:
        raise RuntimeError(f"the EXIF image reads back turned the wrong way (mean difference {diff:.1f})")
    return data


def write(folder: Path, frame: np.ndarray, chosen: dict, iod_px: float, profile: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    frame_jpeg = jpeg(frame)
    (folder / VIDEO_NAME).write_bytes(frame_jpeg * FRAMES)
    (folder / EXIF_NAME).write_bytes(exif_rotated(frame))
    config = {
        "video": str((folder / VIDEO_NAME).resolve()),
        "exif_image": str((folder / EXIF_NAME).resolve()),
        "profile": str(profile.resolve()),
        "frame": {"width": FRAME_SIZE[0], "height": FRAME_SIZE[1], "iod_px": round(iod_px, 1)},
        "frontal_image": chosen["image"],
        "profile_image": profile.name,
    }
    path = folder / CONFIG_NAME
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Make the fake camera video of the end-to-end test").parse_args(argv)
    candidates = frontal_candidates(read_rows("frontal_mediapipe.csv"))[:CANDIDATES]
    profile = profile_choice(read_rows("fei_profile.csv"))
    models = FaceModels()
    try:
        for n, row in enumerate(candidates, 1):
            face = image_io.read_image(paths.path("scut") / SCUT_IMAGES / row["image"])
            frame, iod_px, failed = place(face, float(row["iod_px"]), models)
            print(f"[{n}/{len(candidates)}] IOD {iod_px:.1f} px, failed: {', '.join(failed) or '-'}", flush=True)
            if not failed and abs(iod_px - TARGET_IOD_PX) <= IOD_TOLERANCE_PX:
                path = write(paths.ensure_dir("outputs", OUT_FOLDER), frame, row, iod_px, profile)
                print(f"wrote {path.parent} (never commit it: it holds faces from the datasets)")
                return 0
    finally:
        models.close()
    print(f"none of the {len(candidates)} best SCUT test images passed on the frame", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
