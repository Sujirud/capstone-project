"""analyze_frontal with fake models, and on real SCUT-FBP5500 images when they are present."""

import numpy as np
import pytest

from facemeasure import assets, image_io, mp_points, paths, quality
from facemeasure.frontal import analyze_frontal, measure_ratios
from facemeasure.landmarks import (
    HAIR,
    SEGMENT_CLASSES,
    Face,
    Segmentation,
    largest_first,
)

CRITERIA = quality.load_config().frontal
NEUTRAL = {"mouthSmileLeft": 0.02, "mouthSmileRight": 0.03, "jawOpen": 0.01, "eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.04}


def face(lm, blendshapes=None) -> Face:
    return Face(landmarks=lm, blendshapes=dict(blendshapes or NEUTRAL), matrix=np.eye(4), iod_px=mp_points.iod(lm))


class FakeModels:
    def __init__(self, faces, hair=False):
        self.faces, self.hair, self.segmented = faces, hair, []

    def detect(self, image):
        return largest_first(self.faces)

    def segment(self, image):
        h, w = image.shape[:2]
        self.segmented.append(image.shape)
        category = np.full((h, w), HAIR if self.hair else 0, dtype=np.uint8)
        return Segmentation(category=category, confidence=np.zeros((len(SEGMENT_CLASSES), h, w), dtype=np.float32))


IMAGE = np.full((500, 400, 3), 120, dtype=np.uint8)


def test_no_face_gives_only_no_face():
    r = analyze_frontal(IMAGE, FakeModels([]), CRITERIA)
    assert (r.width, r.height, r.face_count) == (400, 500, 0)
    assert r.failed == ("no_face",) and set(r.region_failed.values()) == {("no_face",)}
    assert r.landmarks is None and r.quality is None and r.crops is None and r.ratios is None
    assert r.blendshapes is None and r.matrix is None and r.ratios_error is None


def test_one_good_face(mp_face):
    models = FakeModels([face(mp_face())])
    r = analyze_frontal(IMAGE, models, CRITERIA)
    assert r.face_count == 1 and r.failed == () and all(v == () for v in r.region_failed.values())
    assert r.quality.iod_px == pytest.approx(100) and r.quality.yaw_deg == 0 and r.quality.hair_jaw == 0
    assert r.crops.face.image.shape == (224, 224, 3)
    assert len(models.segmented) == 1 and models.segmented[0][0] == models.segmented[0][1]
    assert set(r.ratios.nose_ratios) == {mp_points.pair_key(p) for p in mp_points.ALAR_BASE_PAIRS}
    assert {"go_l", "go_r", "zy_l", "zy_r", "en_l", "en_r"} <= set(r.ratios.points)
    assert r.ratios.jaw_ratio == pytest.approx(1.0)  # both edges are straight lines
    if mp_points.ALAR_BASE is None:
        assert r.ratios.nose_ratio is None and "al_l" not in r.ratios.points
    else:
        assert r.ratios.nose_ratio == pytest.approx(r.ratios.nose_ratios[mp_points.pair_key(mp_points.ALAR_BASE)])
        assert {"al_l", "al_r"} <= set(r.ratios.points)


def test_two_faces_are_flagged_but_still_measured_on_the_larger_one(mp_face):
    small, large = face(mp_face(iod=80)), face(mp_face(iod=100))
    r = analyze_frontal(IMAGE, FakeModels([small, large]), CRITERIA)
    assert r.face_count == 2 and r.failed == ("multiple_faces",)
    assert r.region_failed["jaw"] == ("multiple_faces",)
    assert r.quality.iod_px == pytest.approx(100) and r.crops is not None and r.ratios is not None


def test_failing_face_is_still_cropped_and_measured(mp_face):
    smiling = dict(NEUTRAL, mouthSmileRight=0.9, eyeBlinkLeft=0.8)
    r = analyze_frontal(IMAGE, FakeModels([face(mp_face(), smiling)], hair=True), CRITERIA)
    assert r.failed == ("smiling", "eyes_closed", "hair_occlusion")
    assert r.region_failed["nose_base"] == ("smiling",)
    assert r.region_failed["undereye_left"] == ("smiling", "eyes_closed", "hair_occlusion")
    assert r.quality.hair_jaw == 1 and r.crops is not None and r.ratios is not None


def test_chosen_alar_pair_adds_overlay_points(mp_face):
    lm = mp_face()
    r = measure_ratios(lm, (98, 327))
    assert r.nose_ratio == pytest.approx(r.nose_ratios["98_327"])
    assert r.points["al_l"] == pytest.approx(tuple(lm[98, :2]))
    assert r.points["en_r"] == pytest.approx(tuple(lm[362, :2]))


def test_bad_image():
    with pytest.raises(ValueError):
        analyze_frontal(np.zeros((10, 10), dtype=np.uint8), FakeModels([]), CRITERIA)


# ------------------------------------------------------------------ real data

# Real images: one face, crops and the same points on a second run
REAL_IMAGES = ("AF1075.jpg", "AM1382.jpg", "CM290.jpg")
# AF1075 and CM290 really smile, so these three check passing instead. They pass every
# check in data/labels/frontal_mediapipe.csv with room to spare (IOD >= 90 px, pose
# within 5 degrees, smile <= 0.1, hair_jaw <= 0.03), picked by a fixed sha256 ranking.
PASSING_IMAGES = ("AF1034.jpg", "AM624.jpg", "CF535.jpg")


def real_models():
    try:
        assets.asset_path("face_landmarker.task")
        assets.asset_path("selfie_multiclass_256x256.tflite")
    except assets.AssetError as e:
        pytest.skip(f"model files missing: {e}")
    from facemeasure.landmarks import FaceModels

    return FaceModels()


@pytest.fixture(scope="module")
def models():
    m = real_models()
    yield m
    m.close()


def real_image(name):
    path = paths.path("scut") / "Images" / name
    if not path.is_file():
        pytest.skip(f"{path} missing")
    return image_io.read_image(path)


@pytest.mark.parametrize("name", REAL_IMAGES)
def test_real_scut_image(models, name):
    image = real_image(name)
    first = analyze_frontal(image, models)
    assert first.face_count == 1
    for crop in (first.crops.face, first.crops.undereye_left, first.crops.undereye_right):
        assert crop.image.shape == (224, 224, 3) and crop.image.dtype == np.uint8
    again = analyze_frontal(image, models)
    assert np.array_equal(first.landmarks, again.landmarks)
    assert np.array_equal(first.matrix, again.matrix)
    assert first.quality == again.quality


@pytest.mark.parametrize("name", PASSING_IMAGES)
def test_real_scut_image_passes(models, name):
    r = analyze_frontal(real_image(name), models)
    assert r.face_count == 1
    assert r.failed == (), f"{name} fails {r.failed}: {r.quality}"
    assert all(codes == () for codes in r.region_failed.values())
    assert r.crops.face.image.shape == (224, 224, 3) and r.ratios is not None and r.ratios.nose_ratio is not None
