"""Frontal image quality: values, hair occlusion and failed codes (api.md sections 2.2, 2.3 and 4).

Every limit comes from shared/quality.json, which the web reads too (web.md section 4.1).
A value equal to its limit passes.

Hair is the hair class of the selfie multiclass segmenter, run on an axis-aligned square
crop around the face that is neither rotated nor resized: HAIR_CROP_SCALE times the
longer side of the box around all 478 points, black outside the image. Fractions count
only pixels inside the image.

- hair_jaw: a band on the face side of the smoothed face edge (face_geometry),
  hair_band_width_iod x IOD wide, from halfway between eye level and subnasale level
  down to halfway between mouth level and the chin; levels are taken in the frame that
  puts the eyes level. Each side is counted alone and the larger fraction is kept.
  Hair outside the edge does not hide it, and at eye level the edge runs next to the
  sideburns and the hair at the temples, so neither counts.
- hair_undereye_left and hair_undereye_right: inside each under-eye crop box. left is
  the person's left eye, on the image right.
"""

import itertools
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from facemeasure import codes, mp_points, paths
from facemeasure.face_geometry import LevelFrame, catmull_rom
from facemeasure.landmarks import HAIR

QUALITY_FILE = "quality.json"
QUALITY_VERSION = 1
HAIR_CROP_SCALE = 1.3


class QualityConfigError(ValueError):
    """shared/quality.json is missing a key, has an extra one, or holds a bad value."""


@dataclass(frozen=True)
class FrontalCriteria:
    min_iod_px: float
    max_abs_yaw_deg: float
    max_abs_pitch_deg: float
    max_abs_roll_deg: float
    max_smile: float
    max_jaw_open: float
    max_eye_blink: float
    hair_band_width_iod: float
    max_hair_fraction: float


@dataclass(frozen=True)
class ProfileCriteria:
    min_abs_yaw_deg: float


@dataclass(frozen=True)
class RequestLimits:
    max_bytes: int


@dataclass(frozen=True)
class QualityConfig:
    version: int
    frontal: FrontalCriteria
    profile: ProfileCriteria
    request: RequestLimits


_SECTIONS = {"frontal": FrontalCriteria, "profile": ProfileCriteria, "request": RequestLimits}


def _section(name: str, raw, cls):
    if not isinstance(raw, dict):
        raise QualityConfigError(f"{name} must be an object")
    want = [f.name for f in fields(cls)]
    missing, extra = sorted(set(want) - set(raw)), sorted(set(raw) - set(want))
    if missing or extra:
        raise QualityConfigError(f"{name}: missing keys {missing}, extra keys {extra}")
    values = {}
    for f in fields(cls):
        v = raw[f.name]
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v < 0:
            raise QualityConfigError(f"{name}.{f.name} must be a finite number >= 0, got {v!r}")
        if f.type is int and not isinstance(v, int):
            raise QualityConfigError(f"{name}.{f.name} must be an integer, got {v!r}")
        values[f.name] = v
    return cls(**values)


def parse_config(data) -> QualityConfig:
    """QualityConfig from the parsed JSON; every key must be present and no other key."""
    if not isinstance(data, dict):
        raise QualityConfigError("quality.json must hold an object")
    want = {"version", *_SECTIONS}
    missing, extra = sorted(want - set(data)), sorted(set(data) - want)
    if missing or extra:
        raise QualityConfigError(f"quality.json: missing keys {missing}, extra keys {extra}")
    if data["version"] != QUALITY_VERSION:
        raise QualityConfigError(f"quality.json version {data['version']!r} is not {QUALITY_VERSION}")
    return QualityConfig(version=QUALITY_VERSION, **{k: _section(k, data[k], cls) for k, cls in _SECTIONS.items()})


def load_config(path: Path | None = None) -> QualityConfig:
    """Read shared/quality.json (or path)."""
    path = Path(path) if path is not None else paths.path("shared") / QUALITY_FILE
    with open(path, encoding="utf-8") as f:
        return parse_config(json.load(f))


@dataclass(frozen=True)
class FrontalQuality:
    """quality.frontal of api.md section 2.2."""

    iod_px: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    smile: float
    jaw_open: float
    blink_left: float
    blink_right: float
    hair_jaw: float
    hair_undereye_left: float
    hair_undereye_right: float
    failed: tuple[str, ...]

    def as_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["failed"] = list(self.failed)
        return out


def blendshape_values(blendshapes: dict[str, float]) -> dict[str, float]:
    """smile, jaw_open, blink_left and blink_right from MediaPipe blendshapes.

    blink_left and blink_right keep MediaPipe's names; whether they are image or person
    sides is not known, so eyes_closed uses the larger of the two.
    """
    return {
        "smile": max(blendshapes["mouthSmileLeft"], blendshapes["mouthSmileRight"]),
        "jaw_open": blendshapes["jawOpen"],
        "blink_left": blendshapes["eyeBlinkLeft"],
        "blink_right": blendshapes["eyeBlinkRight"],
    }


def evaluate(values: dict[str, float], face_count: int, c: FrontalCriteria) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """(failed codes, failed codes per frontal region) for the values of one face."""
    if face_count < 1:
        return (codes.NO_FACE,), codes.by_region((codes.NO_FACE,))
    hair_over = tuple(name for name in codes.HAIR_REGIONS if values[name] > c.max_hair_fraction)
    checks = {
        codes.MULTIPLE_FACES: face_count > 1,
        codes.FACE_TOO_SMALL: values["iod_px"] < c.min_iod_px,
        codes.FRONTAL_POSE: abs(values["yaw_deg"]) > c.max_abs_yaw_deg
        or abs(values["pitch_deg"]) > c.max_abs_pitch_deg
        or abs(values["roll_deg"]) > c.max_abs_roll_deg,
        codes.SMILING: values["smile"] > c.max_smile or values["jaw_open"] > c.max_jaw_open,
        codes.EYES_CLOSED: max(values["blink_left"], values["blink_right"]) > c.max_eye_blink,
        codes.HAIR_OCCLUSION: bool(hair_over),
    }
    failed = tuple(code for code in codes.FRONTAL_CODES if checks.get(code, False))
    return failed, codes.by_region(failed, hair_over)


# ------------------------------------------------------------------ hair


@dataclass(frozen=True)
class HairCrop:
    """Axis-aligned square crop: its top-left pixel (x0, y0) in the image and its side."""

    x0: int
    y0: int
    side: int

    def pixel_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Image coordinates (x, y) of the centers of the crop's pixels, each (side, side)."""
        r = np.arange(self.side) + 0.5
        return np.meshgrid(self.x0 + r, self.y0 + r)

    def inside_image(self, width: int, height: int) -> np.ndarray:
        """(side, side) bool: the crop pixel is a pixel of the image."""
        r = np.arange(self.side)
        xs, ys = self.x0 + r, self.y0 + r
        return ((ys >= 0) & (ys < height))[:, None] & ((xs >= 0) & (xs < width))[None, :]

    def cut(self, image: np.ndarray) -> np.ndarray:
        """The crop, black outside the image."""
        h, w = image.shape[:2]
        out = np.zeros((self.side, self.side, 3), dtype=np.uint8)
        x1, y1 = self.x0 + self.side, self.y0 + self.side
        sx0, sy0, sx1, sy1 = max(self.x0, 0), max(self.y0, 0), min(x1, w), min(y1, h)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - self.y0 : sy1 - self.y0, sx0 - self.x0 : sx1 - self.x0] = image[sy0:sy1, sx0:sx1]
        return out


def hair_crop(landmarks: np.ndarray) -> HairCrop:
    pts = np.asarray(landmarks, dtype=np.float64)[:, :2]
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    side = max(1, math.ceil(HAIR_CROP_SCALE * max(x1 - x0, y1 - y0)))
    return HairCrop(x0=round((x0 + x1) / 2 - side / 2), y0=round((y0 + y1) / 2 - side / 2), side=side)


def face_edges_level(landmarks: np.ndarray, frame: LevelFrame) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed face edges (image left, image right) from forehead to chin, in the level frame."""
    lm = np.asarray(landmarks, dtype=np.float64)
    return tuple(
        catmull_rom(frame.to_level(mp_points.xy(lm, chain)))
        for chain in (mp_points.CONTOUR_IMAGE_LEFT, mp_points.CONTOUR_IMAGE_RIGHT)
    )


def jaw_band_levels(landmarks: np.ndarray, frame: LevelFrame) -> tuple[float, float]:
    """Top and bottom of the jaw band in the level frame.

    Top: halfway between eye level and subnasale level. Bottom: halfway between mouth
    level and the chin.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    eye_y = float(frame.to_level(np.mean(mp_points.eye_centers(lm), axis=0))[1])
    sub_y = float(frame.to_level(mp_points.xy(lm, mp_points.SUBNASALE))[1])
    mouth_y = float(frame.to_level(mp_points.xy(lm, mp_points.MOUTH_CORNERS))[:, 1].mean())
    chin_y = float(frame.to_level(mp_points.xy(lm, mp_points.CHIN))[1])
    top, bottom = sorted(((eye_y + sub_y) / 2, (mouth_y + chin_y) / 2))
    return top, bottom


def band_mask(crop: HairCrop, curve: np.ndarray, distance: float) -> np.ndarray:
    """(side, side) bool: pixel centers within distance of the polyline curve (image coordinates)."""
    mask = np.zeros((crop.side, crop.side), dtype=bool)
    c = np.asarray(curve, dtype=np.float64)
    if len(c) == 0 or distance <= 0:
        return mask
    # Work only on the crop pixels near the curve
    (cx0, cy0), (cx1, cy1) = c.min(axis=0) - distance, c.max(axis=0) + distance
    i0, i1 = max(0, math.floor(cy0 - crop.y0)), min(crop.side, math.ceil(cy1 - crop.y0) + 1)
    j0, j1 = max(0, math.floor(cx0 - crop.x0)), min(crop.side, math.ceil(cx1 - crop.x0) + 1)
    if i1 <= i0 or j1 <= j0:
        return mask
    px, py = np.meshgrid(crop.x0 + np.arange(j0, j1) + 0.5, crop.y0 + np.arange(i0, i1) + 0.5)
    best = np.full(px.shape, np.inf)
    if len(c) == 1:
        best = np.hypot(px - c[0, 0], py - c[0, 1])
    for (ax, ay), (bx, by) in itertools.pairwise(c):
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        t = np.zeros(px.shape) if length2 == 0 else np.clip(((px - ax) * dx + (py - ay) * dy) / length2, 0.0, 1.0)
        np.minimum(best, np.hypot(px - (ax + t * dx), py - (ay + t * dy)), out=best)
    mask[i0:i1, j0:j1] = best <= distance
    return mask


def jaw_band_masks(crop: HairCrop, landmarks: np.ndarray, frame: LevelFrame, width_px: float) -> tuple[np.ndarray, np.ndarray]:
    """(side, side) bool masks of the image-left and image-right jaw bands.

    A pixel is in a band when, in the level frame, it lies between the band's top and
    bottom levels, on the face side of that edge, and within width_px of the edge.
    """
    top, bottom = jaw_band_levels(landmarks, frame)
    px, py = crop.pixel_centers()
    level = frame.to_level(np.stack([px, py], axis=-1))
    lx, ly = level[..., 0], level[..., 1]
    rows = (ly >= top) & (ly <= bottom)
    out = []
    for edge, inward in zip(face_edges_level(landmarks, frame), (1.0, -1.0), strict=True):
        # Keep the edge a little past both levels, so pixels near the ends measure to the edge, not to a cut end
        part = edge[(edge[:, 1] >= top - width_px) & (edge[:, 1] <= bottom + width_px)]
        if len(part) < 2:
            out.append(np.zeros((crop.side, crop.side), dtype=bool))
            continue
        order = np.argsort(part[:, 1], kind="stable")
        edge_x = np.interp(ly, part[order, 1], part[order, 0])
        inner = (lx - edge_x) * inward > 0
        out.append(rows & inner & band_mask(crop, frame.to_image(part), width_px))
    return out[0], out[1]


def quad_mask(crop: HairCrop, corners: np.ndarray) -> np.ndarray:
    """(side, side) bool: pixel centers inside the convex quadrilateral corners (either winding)."""
    px, py = crop.pixel_centers()
    c = np.asarray(corners, dtype=np.float64)
    sides = []
    for k in range(len(c)):
        (ax, ay), (bx, by) = c[k], c[(k + 1) % len(c)]
        sides.append((bx - ax) * (py - ay) - (by - ay) * (px - ax))
    s = np.stack(sides)
    return (s >= 0).all(axis=0) | (s <= 0).all(axis=0)


def fraction(category: np.ndarray, region: np.ndarray, valid: np.ndarray) -> float:
    """Share of hair pixels among the region's pixels inside the image; 0 for an empty region."""
    use = region & valid
    n = int(use.sum())
    return float((category[use] == HAIR).sum() / n) if n else 0.0


def hair_fractions(
    category: np.ndarray,
    crop: HairCrop,
    image_size: tuple[int, int],
    landmarks: np.ndarray,
    frame: LevelFrame,
    iod_px: float,
    undereye_left: np.ndarray,
    undereye_right: np.ndarray,
    band_width_iod: float,
) -> dict[str, float]:
    """hair_jaw, hair_undereye_left and hair_undereye_right from the crop's category mask.

    image_size is (width, height); undereye_left and undereye_right are the corners of
    the person's left and right under-eye boxes in image coordinates.
    """
    if category.shape != (crop.side, crop.side):
        raise ValueError(f"category mask is {category.shape}, the crop is {crop.side} x {crop.side}")
    valid = crop.inside_image(*image_size)
    bands = jaw_band_masks(crop, landmarks, frame, band_width_iod * iod_px)
    sides = [fraction(category, band, valid) for band in bands]
    return {
        "hair_jaw": max(sides),
        "hair_undereye_left": fraction(category, quad_mask(crop, undereye_left), valid),
        "hair_undereye_right": fraction(category, quad_mask(crop, undereye_right), valid),
    }
