"""The one entry point for a frontal image (model.md section 1.4, api.md section 3).

analyze_frontal() finds the face, its pose and quality, the three DINOv2 crops and the
jaw and alar base ratios from the MediaPipe points. The data scripts (extract_features.py) and the
api call only this function, so training and use go through the same steps.

With two faces the image gets multiple_faces, and everything is still measured on the
face with the larger IOD so lab mode can go on. A face that fails a quality check is
still cropped and measured for the same reason. Without a face only no_face is set.
"""

from dataclasses import dataclass

import numpy as np

from facemeasure import crops, mp_points, pose, quality
from facemeasure.face_geometry import GeometryError, measure_jaw_nose, nose_ratio
from facemeasure.landmarks import FaceModels, check_rgb

Point = tuple[float, float]


@dataclass(frozen=True)
class FrontalRatios:
    """Jaw and alar base ratios from the MediaPipe points, not yet calibrated.

    nose_ratios has every candidate pair of mp_points.ALAR_BASE_PAIRS by pair_key.
    nose_ratio and the al_l and al_r points use mp_points.ALAR_BASE and are None (absent)
    until that pair is chosen. points uses the overlay names of api.md section 2.2,
    where _l and _r are image sides.
    """

    jaw_ratio: float
    nose_ratio: float | None
    nose_ratios: dict[str, float]
    points: dict[str, Point]


@dataclass(frozen=True)
class FrontalResult:
    width: int
    height: int
    face_count: int
    landmarks: np.ndarray | None  # (478, 3) pixels of the upright image
    blendshapes: dict[str, float] | None
    matrix: np.ndarray | None  # (4, 4)
    quality: quality.FrontalQuality | None
    failed: tuple[str, ...]  # codes of the image, also in quality.failed
    region_failed: dict[str, tuple[str, ...]]  # frontal region -> codes that affect it
    crops: crops.Crops | None
    ratios: FrontalRatios | None
    ratios_error: str | None  # why ratios is None when a face was found


def _point(p) -> Point:
    return (float(p[0]), float(p[1]))


def measure_ratios(landmarks: np.ndarray, alar_base: tuple[int, int] | None = None) -> FrontalRatios:
    """Ratios and overlay points from the 478 points (raises GeometryError)."""
    if alar_base is None:
        alar_base = mp_points.ALAR_BASE
    fp = mp_points.frontal_points(landmarks, alar_base or mp_points.ALAR_BASE_PAIRS[0])
    jaw = measure_jaw_nose(fp)
    canthi = mp_points.xy(landmarks, mp_points.INNER_CANTHI)
    by_pair = {}
    for pair in mp_points.ALAR_BASE_PAIRS:
        alar = mp_points.xy(landmarks, pair)
        by_pair[mp_points.pair_key(pair)] = nose_ratio(alar[0], alar[1], canthi[0], canthi[1])
    points = {
        "go_l": jaw.go_l,
        "go_r": jaw.go_r,
        "zy_l": jaw.zy_l,
        "zy_r": jaw.zy_r,
        "en_l": _point(canthi[0]),
        "en_r": _point(canthi[1]),
    }
    chosen = None
    if alar_base is not None:
        alar = mp_points.xy(landmarks, alar_base)
        points["al_l"], points["al_r"] = _point(alar[0]), _point(alar[1])
        chosen = nose_ratio(alar[0], alar[1], canthi[0], canthi[1])
    return FrontalRatios(jaw_ratio=jaw.jaw_ratio, nose_ratio=chosen, nose_ratios=by_pair, points=points)


def analyze_frontal(
    image: np.ndarray, models: FaceModels, criteria: quality.FrontalCriteria | None = None
) -> FrontalResult:
    """Everything the data scripts and the api need from one upright RGB uint8 frontal image.

    criteria defaults to the frontal section of shared/quality.json.
    """
    image = check_rgb(image)
    if criteria is None:
        criteria = quality.load_config().frontal
    height, width = image.shape[:2]
    faces = models.detect(image)
    if not faces:
        failed, region_failed = quality.evaluate({}, 0, criteria)
        return FrontalResult(width, height, 0, None, None, None, None, failed, region_failed, None, None, None)

    face = faces[0]
    lm = face.landmarks
    head = pose.angles(face.matrix)
    face_crops = crops.make_crops(image, lm)

    box = quality.hair_crop(lm)
    category = models.segment(box.cut(image)).category
    hair = quality.hair_fractions(
        category,
        box,
        (width, height),
        lm,
        face_crops.frame,
        face.iod_px,
        face_crops.undereye_left.corners,
        face_crops.undereye_right.corners,
        criteria.hair_band_width_iod,
    )
    values = {
        "iod_px": face.iod_px,
        "yaw_deg": head.yaw_deg,
        "pitch_deg": head.pitch_deg,
        "roll_deg": head.roll_deg,
        **quality.blendshape_values(face.blendshapes),
        **hair,
    }
    failed, region_failed = quality.evaluate(values, len(faces), criteria)

    ratios, ratios_error = None, None
    try:
        ratios = measure_ratios(lm)
    except GeometryError as e:
        ratios_error = str(e)

    return FrontalResult(
        width=width,
        height=height,
        face_count=len(faces),
        landmarks=lm,
        blendshapes=face.blendshapes,
        matrix=face.matrix,
        quality=quality.FrontalQuality(**values, failed=failed),
        failed=failed,
        region_failed=region_failed,
        crops=face_crops,
        ratios=ratios,
        ratios_error=ratios_error,
    )

