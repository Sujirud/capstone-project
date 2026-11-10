"""Jaw and alar-base ratios from frontal face points (model.md sections 3.1, 3.4 and 5.2).

The rules work on generic points, so SCUT-FBP5500 labels (facemeasure.scut) and the
MediaPipe baseline go through the same code:

- Rotate about the midpoint of the eye centers so the eyes are level.
- Draw each face edge as a Catmull-Rom curve through its points, top to bottom.
- Zy is the widest curve sample between eye level and subnasale level, per side.
- Go is the first point, from the top, where each curve crosses mouth level
  (the mean level of the two mouth corners).
- Jaw ratio = Go-Go / Zy-Zy. Nose ratio = alar base width / inner canthal distance.
- Go and Zy go back to the coordinates of the original image.

Suffixes _l and _r are the left and right of the image (image_side), as in the
overlay names of api.md section 2.2. Ratios are proportions (0.83), not percent.
"""

from dataclasses import dataclass, fields

import numpy as np

SAMPLES_PER_SEGMENT = 24

Point = tuple[float, float]


class GeometryError(ValueError):
    """The points cannot give the measurement, for example a face edge that misses mouth level."""


@dataclass(frozen=True)
class FrontalPoints:
    """Points of one frontal face in image coordinates (x right, y down).

    contour_l and contour_r are the face edges on the image left and image right,
    each ordered from top to bottom. eye_l and eye_r are the eye centers.
    """

    contour_l: np.ndarray
    contour_r: np.ndarray
    eye_l: np.ndarray
    eye_r: np.ndarray
    subnasale: np.ndarray
    mouth_l: np.ndarray
    mouth_r: np.ndarray
    alar_l: np.ndarray
    alar_r: np.ndarray
    canthus_l: np.ndarray
    canthus_r: np.ndarray

    def __post_init__(self):
        for f in fields(self):
            value = np.asarray(getattr(self, f.name), dtype=np.float64)
            if f.name.startswith("contour"):
                if value.ndim != 2 or value.shape[1] != 2 or len(value) < 2:
                    raise ValueError(f"{f.name} must be an (n, 2) array with n >= 2, got shape {value.shape}")
            elif value.shape != (2,):
                raise ValueError(f"{f.name} must be one (x, y) point, got shape {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{f.name} has a value that is not finite")
            object.__setattr__(self, f.name, value)

    def map_points(self, func) -> "FrontalPoints":
        """Return a copy with func applied to every (n, 2) or (2,) array."""
        return FrontalPoints(**{f.name: func(getattr(self, f.name)) for f in fields(self)})


@dataclass(frozen=True)
class JawNose:
    """Ratios and the Go and Zy points in image coordinates."""

    jaw_ratio: float
    nose_ratio: float
    iod_px: float
    go_l: Point
    go_r: Point
    zy_l: Point
    zy_r: Point


@dataclass(frozen=True)
class LevelFrame:
    """Rotation about the midpoint of the eye centers that puts the eyes on one level."""

    center: np.ndarray
    cos: float
    sin: float

    @classmethod
    def from_eyes(cls, eye_l, eye_r) -> "LevelFrame":
        """Frame from the image-left and image-right eye centers.

        Raises GeometryError when eye_l is not to the left of eye_r, because the
        rotation would then turn the face upside down.
        """
        (x1, y1), (x2, y2) = np.asarray(eye_l, dtype=np.float64), np.asarray(eye_r, dtype=np.float64)
        if x2 <= x1:
            raise GeometryError("eye centers are not ordered from image left to image right")
        angle = np.arctan2(y2 - y1, x2 - x1)
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        return cls(center=center, cos=float(np.cos(angle)), sin=float(np.sin(angle)))

    def to_level(self, points) -> np.ndarray:
        """Image coordinates to level coordinates, with the eye midpoint at (0, 0)."""
        p = np.asarray(points, dtype=np.float64)
        dx, dy = p[..., 0] - self.center[0], p[..., 1] - self.center[1]
        return np.stack([dx * self.cos + dy * self.sin, -dx * self.sin + dy * self.cos], axis=-1)

    def to_image(self, points) -> np.ndarray:
        """Level coordinates back to image coordinates."""
        p = np.asarray(points, dtype=np.float64)
        x, y = p[..., 0], p[..., 1]
        return np.stack(
            [self.center[0] + x * self.cos - y * self.sin, self.center[1] + x * self.sin + y * self.cos], axis=-1
        )


def catmull_rom(chain, samples_per_segment: int = SAMPLES_PER_SEGMENT) -> np.ndarray:
    """Samples of a uniform Catmull-Rom curve through every point of chain, in order.

    The end points are repeated as the outer control points, so the curve starts at
    chain[0] and ends at chain[-1]. Each segment gives samples_per_segment samples at
    t = 0, 1/n, ..., (n-1)/n, then the last point of chain closes the curve.
    """
    c = np.asarray(chain, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2 or len(c) < 2:
        raise ValueError(f"chain must be an (n, 2) array with n >= 2, got shape {c.shape}")
    last = len(c) - 1
    seg = np.arange(last)
    p0 = c[np.maximum(seg - 1, 0)][:, None, :]
    p1 = c[seg][:, None, :]
    p2 = c[seg + 1][:, None, :]
    p3 = c[np.minimum(seg + 2, last)][:, None, :]
    t = (np.arange(samples_per_segment) / samples_per_segment)[None, :, None]
    curve = 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
    )
    return np.concatenate([curve.reshape(-1, 2), c[-1:]])


def first_crossing(curve, y: float) -> Point | None:
    """First point along curve (from its start) that crosses level y, or None if it never does.

    The point is interpolated linearly between the two samples on either side of y.
    """
    c = np.asarray(curve, dtype=np.float64)
    a, b = c[:-1], c[1:]
    hits = np.flatnonzero(((a[:, 1] - y) * (b[:, 1] - y) <= 0) & (a[:, 1] != b[:, 1]))
    if len(hits) == 0:
        return None
    (px, py), (qx, qy) = a[hits[0]], b[hits[0]]
    t = (y - py) / (qy - py)
    return (float(px + t * (qx - px)), y)


def _distance(p, q) -> float:
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def _ratio(top: float, bottom: float, what: str) -> float:
    if bottom <= 0:
        raise GeometryError(f"{what} has zero length")
    return top / bottom


def nose_ratio(alar_l, alar_r, canthus_l, canthus_r) -> float:
    """Alar base width / inner canthal distance (raises GeometryError when the canthi coincide)."""
    return _ratio(_distance(alar_l, alar_r), _distance(canthus_l, canthus_r), "inner canthal distance")


def measure_jaw_nose(points: FrontalPoints, samples_per_segment: int = SAMPLES_PER_SEGMENT) -> JawNose:
    """Jaw ratio, nose ratio, IOD and the Go and Zy points of one face.

    Raises GeometryError when a rule cannot place a point: eye centers out of order,
    no part of a face edge between eye level and subnasale level, or a face edge
    that never reaches mouth level.
    """
    frame = LevelFrame.from_eyes(points.eye_l, points.eye_r)
    level = points.map_points(frame.to_level)
    # Eye level is y = 0 in the level frame
    y_sub = float(level.subnasale[1])
    lo, hi = min(0.0, y_sub), max(0.0, y_sub)
    y_mouth = float((level.mouth_l[1] + level.mouth_r[1]) / 2)

    found = {}
    sides = (("l", "image-left", level.contour_l, np.argmin), ("r", "image-right", level.contour_r, np.argmax))
    for side, name, contour, widest in sides:
        curve = catmull_rom(contour, samples_per_segment)
        band = curve[(curve[:, 1] >= lo) & (curve[:, 1] <= hi)]
        if len(band) == 0:
            raise GeometryError(f"{name} face edge has no point between eye level and subnasale level")
        found[f"zy_{side}"] = band[widest(band[:, 0])]
        go = first_crossing(curve, y_mouth)
        if go is None:
            raise GeometryError(f"{name} face edge does not reach mouth level")
        found[f"go_{side}"] = go

    image = {k: tuple(float(v) for v in frame.to_image(p)) for k, p in found.items()}
    return JawNose(
        jaw_ratio=_ratio(_distance(image["go_l"], image["go_r"]), _distance(image["zy_l"], image["zy_r"]), "Zy-Zy"),
        nose_ratio=nose_ratio(points.alar_l, points.alar_r, points.canthus_l, points.canthus_r),
        iod_px=_distance(points.eye_l, points.eye_r),
        **image,
    )
