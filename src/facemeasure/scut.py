"""SCUT-FBP5500 landmark files, point numbers and file names (model.md section 5.2).

Point numbers follow the figure in the dataset repo and run 1-86, as in the docs.
Arrays from the read functions hold point n in row n - 1, and take() is the only
place that turns point numbers into row indices.

In the landmark files, point 59 is the eye center, 43 the inner canthus, 65 the
alar base and 80 the mouth corner on the image left (the person's right).
"""

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from facemeasure import draw
from facemeasure.face_geometry import FrontalPoints, JawNose, measure_jaw_nose

N_POINTS = 86
PTS_FORMAT = "<i172f"  # int32 point count, then x, y of points 1-86 as float32
PTS_SIZE = struct.calcsize(PTS_FORMAT)  # 692 bytes

# Point numbers (docs and dataset figure, 1-86)
CONTOUR = tuple(range(1, 23))  # face edge
CONTOUR_IMAGE_LEFT = tuple(range(2, 12))  # 2 -> 11, top to bottom
CONTOUR_IMAGE_RIGHT = tuple(range(22, 12, -1))  # 22 -> 13, top to bottom
INNER_CANTHI = (43, 55)  # image left, image right
EYE_CENTERS = (59, 60)  # image left, image right
ALAR_BASE = (65, 69)  # image left, image right
SUBNASALE = 67
MOUTH_CORNERS = (80, 74)  # image left, image right

GROUPS = ("AF", "AM", "CF", "CM")
SEX = {"F": "female", "M": "male"}
ETHNICITY = {"A": "asian", "C": "caucasian"}
NAME = re.compile(r"^(?P<eth>[AC])(?P<sex>[FM])(?P<number>[1-9]\d*)(?:\.(?:jpg|pts|txt))?$")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


class PtsError(ValueError):
    """A landmark file that does not hold 86 points."""


@dataclass(frozen=True)
class Subject:
    group: str  # AF, AM, CF or CM
    sex: str  # female or male
    ethnicity: str  # asian or caucasian
    number: int


def parse_name(name: str) -> Subject:
    """Group, sex and ethnicity from a file name such as AF1.jpg (AF = Asian female)."""
    m = NAME.match(name)
    if not m:
        raise ValueError(f"{name!r} is not a SCUT-FBP5500 name such as AF1.jpg or CM290.pts")
    return Subject(
        group=m["eth"] + m["sex"], sex=SEX[m["sex"]], ethnicity=ETHNICITY[m["eth"]], number=int(m["number"])
    )


def _checked(values, source: str) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    if len(points) != N_POINTS:
        raise PtsError(f"{source}: {len(points)} points, expected {N_POINTS}")
    if not np.isfinite(points).all():
        raise PtsError(f"{source}: a coordinate is not finite")
    return points


def read_pts(path: Path) -> np.ndarray:
    """86 x 2 array (x, y) from a binary SCUT-FBP5500 .pts file."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) != PTS_SIZE:
        raise PtsError(f"{path.name}: {len(data)} bytes, expected {PTS_SIZE} (int32 count + 172 float32)")
    count, *values = struct.unpack(PTS_FORMAT, data)
    if count != N_POINTS:
        raise PtsError(f"{path.name}: header says {count} points, expected {N_POINTS}")
    return _checked(values, path.name)


def read_points_txt(path: Path) -> np.ndarray:
    """86 x 2 array from a text file with one "x y" point per line, as made by pts2txt.py.

    Lines with two or three numbers count as points (the last two are x and y). If
    that does not give 86 points, the last 172 numbers in the file are used.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    points = []
    for line in text.splitlines():
        nums = _NUMBER.findall(line)
        if len(nums) in (2, 3):
            points.append((float(nums[-2]), float(nums[-1])))
    if len(points) != N_POINTS:
        nums = [float(n) for n in _NUMBER.findall(text)]
        if len(nums) >= 2 * N_POINTS:
            points = nums[-2 * N_POINTS :]
    return _checked(points, path.name)


def read_points(path: Path) -> np.ndarray:
    """86 x 2 array from a .pts (binary) or .txt landmark file."""
    path = Path(path)
    if path.suffix.lower() == ".pts":
        return read_pts(path)
    if path.suffix.lower() == ".txt":
        return read_points_txt(path)
    raise PtsError(f"{path.name}: expected a .pts or .txt file")


def take(points: np.ndarray, numbers) -> np.ndarray:
    """Rows of points for SCUT point numbers (1-86): one number gives (2,), a sequence (n, 2)."""
    idx = np.asarray(numbers)
    if idx.size == 0 or not np.issubdtype(idx.dtype, np.integer) or idx.min() < 1 or idx.max() > N_POINTS:
        raise ValueError(f"point numbers must be integers from 1 to {N_POINTS}, got {numbers!r}")
    return np.asarray(points)[idx - 1]


def frontal_points(points: np.ndarray) -> FrontalPoints:
    """The generic input of face_geometry from the 86 SCUT points."""
    return FrontalPoints(
        contour_l=take(points, CONTOUR_IMAGE_LEFT),
        contour_r=take(points, CONTOUR_IMAGE_RIGHT),
        eye_l=take(points, EYE_CENTERS[0]),
        eye_r=take(points, EYE_CENTERS[1]),
        subnasale=take(points, SUBNASALE),
        mouth_l=take(points, MOUTH_CORNERS[0]),
        mouth_r=take(points, MOUTH_CORNERS[1]),
        alar_l=take(points, ALAR_BASE[0]),
        alar_r=take(points, ALAR_BASE[1]),
        canthus_l=take(points, INNER_CANTHI[0]),
        canthus_r=take(points, INNER_CANTHI[1]),
    )


def measure(points: np.ndarray) -> JawNose:
    """Jaw ratio, nose ratio, IOD, Go and Zy from the 86 SCUT points (raises GeometryError)."""
    return measure_jaw_nose(frontal_points(points))


def point_colors() -> list[draw.Color]:
    """Dot colors of the 86 points: red canthi and alar base, blue face edge, white for the rest."""
    red = set(INNER_CANTHI + ALAR_BASE)
    return [draw.RED if n in red else draw.BLUE if n in CONTOUR else draw.WHITE for n in range(1, N_POINTS + 1)]


def draw_points(
    image: Image.Image,
    points: np.ndarray,
    result: JawNose,
    *,
    numbered: bool = False,
    label: str | None = None,
    scale: int = 1,
) -> Image.Image:
    """Image with all 86 points (numbered when asked), the measurement lines, Go and Zy."""
    labels = [str(n) for n in range(1, N_POINTS + 1)] if numbered else None
    return draw.draw_jaw_nose(
        image,
        frontal_points(points),
        result,
        points=points,
        point_colors=point_colors(),
        point_labels=labels,
        label=label,
        scale=scale,
    )
