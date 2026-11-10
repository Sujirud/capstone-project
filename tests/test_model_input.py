"""model_input.analyze_for_model with fake models whose points follow the image size."""

import numpy as np
import pytest

from facemeasure import mp_points, quality
from facemeasure.frontal import analyze_frontal
from facemeasure.landmarks import SEGMENT_CLASSES, Face, Segmentation, largest_first
from facemeasure.model_input import analyze_for_model

CRITERIA = quality.load_config().frontal
NEUTRAL = {"mouthSmileLeft": 0.02, "mouthSmileRight": 0.03, "jawOpen": 0.01, "eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.04}
W, H = 1000, 1250


class ScalingModels:
    """Finds one face whose points scale with the image, as a real landmarker would."""

    def __init__(self, base: np.ndarray, min_width: int | None = None):
        self.base, self.min_width, self.detected = base, min_width, []

    def detect(self, image):
        h, w = image.shape[:2]
        self.detected.append((w, h))
        if self.min_width is not None and w < self.min_width:
            return []
        lm = self.base.copy()
        lm[:, 0] *= w / W
        lm[:, 1] *= h / H
        lm[:, 2] *= w / W
        return largest_first([Face(landmarks=lm, blendshapes=dict(NEUTRAL), matrix=np.eye(4), iod_px=mp_points.iod(lm))])

    def segment(self, image):
        h, w = image.shape[:2]
        return Segmentation(category=np.zeros((h, w), np.uint8), confidence=np.zeros((len(SEGMENT_CLASSES), h, w), np.float32))


@pytest.fixture
def base(mp_face):
    lm = mp_face(cx=500, iod=250, eye_y=500, sub_y=650, mouth_y=760, chin_y=900, top_y=220, half_width=270)
    lm[:, 2] = 3.0
    return lm


def picture() -> np.ndarray:
    y, x = np.mgrid[0:H, 0:W]
    return np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8)


def test_large_face_is_shrunk_and_coordinates_come_back(base):
    models = ScalingModels(base)
    image = picture()
    full = analyze_frontal(image, ScalingModels(base), CRITERIA)
    r = analyze_for_model(image, models, 110.0, CRITERIA)
    assert models.detected == [(W, H), (440, 550)]  # IOD 250 -> 110
    assert (r.width, r.height) == (W, H) and r.face_count == 1
    assert r.quality == full.quality and r.failed == full.failed and r.region_failed == full.region_failed
    assert r.quality.iod_px == pytest.approx(250)
    # Points, boxes and ratios are in coordinates of the full image
    assert np.allclose(r.landmarks, base, atol=1e-9)
    assert r.crops.iod_px == pytest.approx(250)
    for name in ("face", "undereye_left", "undereye_right"):
        assert np.allclose(getattr(r.crops, name).corners, getattr(full.crops, name).corners, atol=1e-6)
        assert getattr(r.crops, name).image.shape == (224, 224, 3)
    assert np.allclose(r.crops.frame.center, full.crops.frame.center)
    for k, p in full.ratios.points.items():
        if k.startswith("zy"):
            # The synthetic edges are straight, so every level of the Zy band is as wide; only x is defined
            assert r.ratios.points[k][0] == pytest.approx(p[0], abs=1e-6) and 500 <= r.ratios.points[k][1] <= 650
        else:
            assert r.ratios.points[k] == pytest.approx(p, abs=1e-6)
    # Ratios are measured on the smaller image; the curve sampling is in pixels, so they differ a little
    assert r.ratios.jaw_ratio == pytest.approx(full.ratios.jaw_ratio, rel=0.01)
    # The crops were cut from the smaller image
    assert not np.array_equal(r.crops.face.image, full.crops.face.image)


def test_face_in_range_is_not_changed(base):
    models = ScalingModels(base)
    image = picture()
    full = analyze_frontal(image, ScalingModels(base), CRITERIA)
    for limit in (250.0, 300.0, None):
        models.detected.clear()
        r = analyze_for_model(image, models, limit, CRITERIA)
        assert models.detected == [(W, H)]
        assert np.array_equal(r.landmarks, full.landmarks) and r.quality == full.quality
        assert np.array_equal(r.crops.face.image, full.crops.face.image)
        assert np.array_equal(r.crops.undereye_left.corners, full.crops.undereye_left.corners)
        assert r.ratios == full.ratios


def test_no_face_on_the_smaller_image_keeps_the_first_run(base):
    models = ScalingModels(base, min_width=W)  # finds the face only at full size
    image = picture()
    r = analyze_for_model(image, models, 110.0, CRITERIA)
    assert models.detected == [(W, H), (440, 550)]
    full = analyze_frontal(image, ScalingModels(base), CRITERIA)
    assert np.array_equal(r.crops.face.image, full.crops.face.image) and r.face_count == 1


def test_no_face_at_all(base):
    class Empty(ScalingModels):
        def detect(self, image):
            self.detected.append(image.shape[1::-1])
            return []

    models = Empty(base)
    r = analyze_for_model(picture(), models, 110.0, CRITERIA)
    assert r.face_count == 0 and r.failed == ("no_face",) and len(models.detected) == 1
