"""Square crops for DINOv2: the whole face and each under-eye (model.md sections 2.3, 3.3 and 6).

- Rotate about the midpoint of the eye centers so both eyes are level, then place the
  boxes in that level frame.
- Whole face: a square with the center of the box around all 478 points and a side of
  FACE_SCALE times the longer side of that box, so the jaw and cheekbones are inside.
- Under-eye: a square of UNDEREYE_SIDE_IOD x IOD, centered on that eye horizontally,
  with its top edge at eye level: the lower half of the eye, the lower lid and the
  upper cheek.
- Cut each box at the resolution of the image it is given (black outside the image),
  then resize to CROP_SIZE with Pillow's Image.resize, which filters before it shrinks.
  A single affine step that shrinks would alias.
- The person's left under-eye (image right) is mirrored so the nose side is on the
  right of the crop, as in the person's right under-eye crop.

Coordinates are continuous pixels: pixel i covers [i, i + 1) and its center is i + 0.5,
as in Pillow's transforms. Corners come back in image coordinates, in the order
top-left, top-right, bottom-right, bottom-left of the box in the level frame (before
any mirroring).

The numbers here are a starting point. Change them only in this file and raise
CROP_VERSION, because features made with other crops do not match.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from facemeasure import mp_points
from facemeasure.face_geometry import LevelFrame

CROP_VERSION = 1
CROP_SIZE = 224
FACE_SCALE = 1.25
UNDEREYE_SIDE_IOD = 0.6
CUT_RESAMPLE = Image.Resampling.BILINEAR  # at scale 1, only to rotate
RESIZE_RESAMPLE = Image.Resampling.BICUBIC
FILL = (0, 0, 0)


@dataclass(frozen=True)
class Crop:
    image: np.ndarray  # (CROP_SIZE, CROP_SIZE, 3) uint8
    corners: np.ndarray  # (4, 2) image coordinates: top-left, top-right, bottom-right, bottom-left
    mirrored: bool


@dataclass(frozen=True)
class Crops:
    face: Crop
    undereye_left: Crop  # person's left eye, image right, mirrored
    undereye_right: Crop  # person's right eye, image left
    frame: LevelFrame
    iod_px: float
    version: int = CROP_VERSION


def level_frame(eye_l, eye_r) -> LevelFrame:
    """Rotation that puts the image-left eye center left of the image-right one, on one level.

    Unlike LevelFrame.from_eyes it accepts any angle, so an upside-down face still gives
    upright crops; its pose fails the quality check anyway.
    """
    (x1, y1), (x2, y2) = np.asarray(eye_l, dtype=np.float64), np.asarray(eye_r, dtype=np.float64)
    if x1 == x2 and y1 == y2:
        raise ValueError("the two eye centers are the same point")
    angle = np.arctan2(y2 - y1, x2 - x1)
    return LevelFrame(center=np.array([(x1 + x2) / 2, (y1 + y2) / 2]), cos=float(np.cos(angle)), sin=float(np.sin(angle)))


def square(cx: float, cy: float, side: float) -> np.ndarray:
    """Corners of an axis-aligned square: top-left, top-right, bottom-right, bottom-left."""
    h = side / 2
    return np.array([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])


def level_boxes(landmarks: np.ndarray) -> tuple[LevelFrame, float, dict[str, np.ndarray]]:
    """The level frame, the IOD and the three boxes as corners in level coordinates."""
    eye_l, eye_r = mp_points.eye_centers(landmarks)
    frame = level_frame(eye_l, eye_r)
    iod = float(np.hypot(*(eye_r - eye_l)))
    pts = frame.to_level(np.asarray(landmarks, dtype=np.float64)[:, :2])
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    boxes = {"face": square((x0 + x1) / 2, (y0 + y1) / 2, FACE_SCALE * max(x1 - x0, y1 - y0))}
    side = UNDEREYE_SIDE_IOD * iod
    for name, eye in (("undereye_right", eye_l), ("undereye_left", eye_r)):
        ex, ey = frame.to_level(eye)
        boxes[name] = square(ex, ey + side / 2, side)
    return frame, iod, boxes


def cut(image: np.ndarray, corners: np.ndarray, size: int = CROP_SIZE, mirror: bool = False) -> np.ndarray:
    """Cut the box with these image-coordinate corners at about scale 1, then resize to size x size."""
    c = np.asarray(corners, dtype=np.float64)
    ex, ey = c[1] - c[0], c[3] - c[0]
    side = float(np.hypot(*ex))
    if side <= 0:
        raise ValueError("box has zero size")
    n = max(1, round(side))
    ex, ey = ex / n, ey / n
    src = Image.fromarray(np.ascontiguousarray(image))
    data = (ex[0], ey[0], c[0][0], ex[1], ey[1], c[0][1])
    patch = src.transform((n, n), Image.Transform.AFFINE, data, resample=CUT_RESAMPLE, fillcolor=FILL)
    if n != size:
        patch = patch.resize((size, size), RESIZE_RESAMPLE)
    out = np.array(patch, dtype=np.uint8)
    return out[:, ::-1].copy() if mirror else out


def make_crops(image: np.ndarray, landmarks: np.ndarray, size: int = CROP_SIZE) -> Crops:
    """The whole-face crop and the two under-eye crops of one face."""
    frame, iod, boxes = level_boxes(landmarks)
    out = {}
    for name, box in boxes.items():
        corners = frame.to_image(box)
        mirror = name == "undereye_left"
        out[name] = Crop(image=cut(image, corners, size, mirror), corners=corners, mirrored=mirror)
    return Crops(face=out["face"], undereye_left=out["undereye_left"], undereye_right=out["undereye_right"], frame=frame, iod_px=iod)
