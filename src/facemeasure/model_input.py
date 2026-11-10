"""The api's way in: a full-resolution frontal image -> what the model takes (api.md section 3 step 5).

analyze_for_model() runs analyze_frontal on the full image and keeps that run's quality
values, codes, blendshapes and matrix. When its IOD is above max_iod_px, the top of the
bundle's train_face_iod_px, the whole image is shrunk with degrade.shrink until the IOD
is max_iod_px and analyze_frontal runs again; the landmarks, crops and ratios come from
that second run, so the model never sees a face in more detail than in training
(model.md section 3.3). Their coordinates are turned back into coordinates of the full
image, so the result is a FrontalResult like analyze_frontal's and response.py takes it
unchanged; the crop images themselves stay as cut from the smaller image. If the second
run finds no face, the first run's result is returned.

extract_features.py calls this on every degraded copy, so training and use take the
same steps. Without max_iod_px (a bundle with no train_face_iod_px) nothing is shrunk.
"""

from dataclasses import replace

import numpy as np

from facemeasure import crops, degrade, mp_points, quality
from facemeasure.frontal import FrontalRatios, FrontalResult, analyze_frontal
from facemeasure.landmarks import FaceModels, check_rgb


def _scale_points(points: np.ndarray, sx: float, sy: float) -> np.ndarray:
    p = np.array(points, dtype=np.float64, copy=True)
    p[..., 0] *= sx
    p[..., 1] *= sy
    return p


def to_full_image(small: FrontalResult, first: FrontalResult) -> FrontalResult:
    """The second run's points, crops and ratios in coordinates of the first run's image, with the first run's quality."""
    sx, sy = first.width / small.width, first.height / small.height
    lm = _scale_points(small.landmarks, sx, sy)
    lm[:, 2] *= sx  # z is in units of the image width
    c = small.crops
    face_crops = crops.Crops(
        face=replace(c.face, corners=_scale_points(c.face.corners, sx, sy)),
        undereye_left=replace(c.undereye_left, corners=_scale_points(c.undereye_left.corners, sx, sy)),
        undereye_right=replace(c.undereye_right, corners=_scale_points(c.undereye_right.corners, sx, sy)),
        frame=crops.level_frame(*mp_points.eye_centers(lm)),
        iod_px=mp_points.iod(lm),
        version=c.version,
    )
    ratios = None
    if small.ratios is not None:
        r = small.ratios
        points = {k: (float(x * sx), float(y * sy)) for k, (x, y) in r.points.items()}
        ratios = FrontalRatios(jaw_ratio=r.jaw_ratio, nose_ratio=r.nose_ratio, nose_ratios=dict(r.nose_ratios), points=points)
    return replace(first, landmarks=lm, crops=face_crops, ratios=ratios, ratios_error=small.ratios_error)


def analyze_for_model(
    image: np.ndarray,
    models: FaceModels,
    max_iod_px: float | None,
    criteria: quality.FrontalCriteria | None = None,
) -> FrontalResult:
    """analyze_frontal for the model: crops and ratios at no more than max_iod_px between the eyes."""
    image = check_rgb(image)
    first = analyze_frontal(image, models, criteria)
    if max_iod_px is None or first.face_count == 0 or first.quality.iod_px <= max_iod_px:
        return first
    small = analyze_frontal(degrade.shrink(image, max_iod_px / first.quality.iod_px), models, criteria)
    if small.face_count == 0:
        return first
    return to_full_image(small, first)
