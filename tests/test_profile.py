"""analyze_profile and measure_silhouette on synthetic profiles, and on FEI when it is present."""

import math

import numpy as np
import pytest

from facemeasure import assets, codes, fei, image_io, quality
from facemeasure import profile as P
from facemeasure.landmarks import HAIR, SEGMENT_CLASSES, Face, Segmentation

FACE_SKIN = SEGMENT_CLASSES.index("face_skin")
BODY = SEGMENT_CLASSES.index("body_skin")
CRITERIA = quality.load_config().profile


def yaw_face(yaw_deg: float) -> Face:
    a = math.radians(yaw_deg)
    m = np.eye(4)
    m[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    return Face(landmarks=np.zeros((478, 3)), blendshapes={}, matrix=m, iod_px=50.0)


class ColorModels:
    """Fake models: the segmenter reads hair from red, face skin from green and body skin from blue."""

    def __init__(self, faces=()):
        self.faces = list(faces)
        self.segmented = []

    def detect(self, image):
        return list(self.faces)

    def segment(self, image):
        h, w = image.shape[:2]
        self.segmented.append((h, w))
        rgb = image.astype(np.float32) / 255
        conf = np.zeros((len(SEGMENT_CLASSES), h, w), dtype=np.float32)
        conf[HAIR], conf[FACE_SKIN], conf[BODY] = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        conf[0] = np.clip(1 - rgb.sum(axis=-1), 0, 1)
        return Segmentation(category=conf.argmax(axis=0).astype(np.uint8), confidence=conf)


def measure(s) -> P.Silhouette:
    return P.measure_silhouette(s.head, P.face_rows(s.face), None, s.face)


# ---------------------------------------------------------------- the edge alone


def test_synthetic_angles_and_points(profile_face):
    s = profile_face()
    m = measure(s)
    assert m.reason is None
    assert m.legan_angle == pytest.approx(s.legan, abs=1.0)
    assert m.nasofacial_angle == pytest.approx(s.nasofacial, abs=1.0)
    for name in P.POINT_NAMES:
        # G and Pog may slide along the facial line, which barely moves the angles
        along = 6.0 if name in ("g", "pog") else 2.0
        assert np.abs(m.points[name] - s.points[name]).max() <= along, name


@pytest.mark.parametrize("angle", [-10.0, 10.0])
def test_turning_the_head_keeps_the_angles(profile_face, angle):
    base = measure(profile_face())
    turned = measure(profile_face(angle_deg=angle))
    assert turned.reason is None
    assert turned.legan_angle == pytest.approx(base.legan_angle, abs=0.5)
    assert turned.nasofacial_angle == pytest.approx(base.nasofacial_angle, abs=0.5)


def test_half_size_keeps_the_angles(profile_face):
    base = measure(profile_face())
    half = measure(profile_face(scale=0.5))
    assert half.reason is None
    assert half.legan_angle == pytest.approx(base.legan_angle, abs=0.5)
    assert half.nasofacial_angle == pytest.approx(base.nasofacial_angle, abs=0.5)


def test_legan_sign(profile_face):
    back = profile_face()  # Sn in front of G-Pog: the chin is behind the extension of G-Sn
    ahead = profile_face(sn_x=-15.0)  # Sn behind G-Pog: the chin is in front of it
    assert back.legan > 0 > ahead.legan
    assert measure(back).legan_angle > 0
    m = measure(ahead)
    assert m.reason is None and m.legan_angle < 0
    assert m.legan_angle == pytest.approx(ahead.legan, abs=1.0)


def test_hair_below_the_chin_is_not_chin(profile_face):
    base = measure(profile_face())
    s = profile_face(hair_below_chin=True)
    rows, x, _ = P.front_edge(s.head)
    assert x[rows > s.points["pog"][1] + 100].max() > s.points["pog"][0]  # the hair reaches in front of Pog
    m = measure(s)
    assert m.reason is None
    assert np.abs(m.points["pog"] - s.points["pog"]).max() <= 6.0
    assert m.legan_angle == pytest.approx(base.legan_angle, abs=0.5)


def test_hair_over_the_forehead_is_invalid(profile_face):
    s = profile_face()
    face = s.face.copy()
    face[: round(s.points["n"][1]) - 5] = 0  # the segmenter calls everything above the nasion hair
    head = s.head
    m = P.measure_silhouette(head, P.face_rows(s.face), None, face)
    assert m.reason == P.NO_FOREHEAD and m.legan_angle is None


def test_angle_functions():
    g, pog = (0.0, 0.0), (0.0, 400.0)
    assert P.legan_angle(g, (20.0, 200.0), pog) > 0
    assert P.legan_angle(g, (-20.0, 200.0), pog) < 0
    assert P.legan_angle(g, (0.0, 200.0), pog) == pytest.approx(0.0)
    assert P.nasofacial_angle(g, pog, (0.0, 60.0), (60.0, 120.0)) == pytest.approx(45.0)


def test_chin_cut_at_the_array_bottom(profile_face):
    s = profile_face()
    cut = round(s.points["pog"][1])  # the image ends at Pog
    rows = P.face_rows(s.face)
    m = P.measure_silhouette(s.head[:cut], (rows[0], min(rows[1], cut - 1)), None, s.face[:cut])
    assert m.reason in (P.HEAD_CUT, P.CHIN_CUT) and m.legan_angle is None


def test_empty_edge():
    m = P.measure_silhouette(np.zeros((50, 50)), (10, 40))
    assert m.reason == P.EDGE_SHORT and m.legan_angle is None


# ---------------------------------------------------------------- the whole pipeline with fake models


def test_pipeline_points_are_in_image_coordinates(profile_face):
    s = profile_face()
    r = P.analyze_profile(s.image, ColorModels(), CRITERIA)
    assert r.failed == () and r.reasons == () and r.measured
    assert r.quality.as_dict() == {
        "face_found": False, "face_yaw_deg": None, "silhouette_found": True, "flipped": False, "failed": [],
    }
    assert r.legan_angle == pytest.approx(s.legan, abs=1.0)
    for name in ("n", "prn", "sn"):
        assert np.abs(np.subtract(r.points[name], s.points[name])).max() <= 2.0
    assert r.edge.shape[1] == 2 and r.region_failed == {"chin": (), "nose_dorsum": ()}


def test_flip_gives_the_same_angles(profile_face):
    s = profile_face()
    plain = P.analyze_profile(s.image, ColorModels(), CRITERIA)
    mirrored = np.ascontiguousarray(s.image[:, ::-1])
    flipped = P.analyze_profile(mirrored, ColorModels(), CRITERIA)
    assert not plain.quality.flipped and flipped.quality.flipped
    assert flipped.measured
    # the mirror image is segmented turned back, so it is measured on the same pixels
    assert flipped.legan_angle == plain.legan_angle and flipped.nasofacial_angle == plain.nasofacial_angle
    width = s.image.shape[1]
    for name, (x, y) in plain.points.items():
        assert flipped.points[name] == pytest.approx((width - x, y), abs=1e-9), name
    assert np.allclose(flipped.edge[:, 0], width - plain.edge[:, 0]) and np.allclose(flipped.edge[:, 1], plain.edge[:, 1])
    assert flipped.overlays["nose_dorsum"].points["prn"] == flipped.points["prn"]
    assert flipped.crop == P.mirror_crop(plain.crop, width)  # crop is in coordinates of the received image


class PixelModels(ColorModels):
    """ColorModels that also keep a digest of every image it segments."""

    def __init__(self, faces=()):
        super().__init__(faces)
        self.digests = []

    def segment(self, image):
        import hashlib

        self.digests.append(hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())
        return super().segment(image)


def test_a_face_to_the_left_is_segmented_on_the_mirrored_image(profile_face):
    s = profile_face()
    plain_models, mirror_models = PixelModels(), PixelModels()
    P.analyze_profile(s.image, plain_models, CRITERIA)
    P.analyze_profile(np.ascontiguousarray(s.image[:, ::-1]), mirror_models, CRITERIA)
    assert len(plain_models.digests) == 2  # a face to the right: the whole image and the head crop, once
    assert len(mirror_models.digests) == 4  # to the left: once to find the direction, then on the mirrored image
    assert mirror_models.digests[2:] == plain_models.digests


def test_direction_from_yaw_segments_the_mirrored_image_too(profile_face):
    s = profile_face()
    plain = P.analyze_profile(s.image, ColorModels([yaw_face(75.0)]), CRITERIA)
    mirrored = P.analyze_profile(np.ascontiguousarray(s.image[:, ::-1]), ColorModels([yaw_face(-75.0)]), CRITERIA)
    assert mirrored.quality.flipped and not plain.quality.flipped
    assert mirrored.legan_angle == plain.legan_angle and mirrored.nasofacial_angle == plain.nasofacial_angle


def test_profile_version_marks_the_change():
    assert P.PROFILE_VERSION == 2


def test_direction_from_yaw(profile_face):
    s = profile_face()
    mirrored = np.ascontiguousarray(s.image[:, ::-1])
    r = P.analyze_profile(mirrored, ColorModels([yaw_face(-75.0)]), CRITERIA)
    assert r.quality.face_found and r.quality.face_yaw_deg == pytest.approx(-75.0)
    assert r.quality.flipped and r.failed == () and r.measured


def test_small_yaw_is_invalid_but_still_measured(profile_face):
    s = profile_face()
    yaw = CRITERIA.min_abs_yaw_deg - 1
    r = P.analyze_profile(s.image, ColorModels([yaw_face(yaw)]), CRITERIA)
    assert r.failed == (codes.PROFILE_INVALID,) and r.reasons == (P.FRONTAL_YAW,)
    assert r.region_failed == {"chin": ("profile_invalid",), "nose_dorsum": ("profile_invalid",)}
    assert r.measured  # lab mode can still show the angles


def test_yaw_just_over_the_limit_passes(profile_face):
    s = profile_face()
    r = P.analyze_profile(s.image, ColorModels([yaw_face(CRITERIA.min_abs_yaw_deg + 0.01)]), CRITERIA)
    assert r.failed == ()


def test_no_head():
    r = P.analyze_profile(np.zeros((300, 300, 3), dtype=np.uint8), ColorModels(), CRITERIA)
    assert r.quality.silhouette_found is False and r.failed == (codes.PROFILE_INVALID,)
    assert r.reasons == (P.NO_SILHOUETTE,) and not r.measured and r.overlays == {}


def test_chin_cut_by_the_image(profile_face):
    s = profile_face()
    cut = np.ascontiguousarray(s.image[: round(s.points["pog"][1])])
    r = P.analyze_profile(cut, ColorModels(), CRITERIA)
    assert r.failed == (codes.PROFILE_INVALID,) and not r.measured
    assert set(r.reasons) & {P.HEAD_CUT, P.CHIN_CUT}


def test_missing_profile():
    r = P.analyze_profile(None, ColorModels(), CRITERIA)
    assert r.quality is None and r.failed == (codes.MISSING_INPUT,)
    assert r.region_failed == {"chin": ("missing_input",), "nose_dorsum": ("missing_input",)}
    assert not r.measured


def test_overlays(profile_face):
    r = P.analyze_profile(profile_face().image, ColorModels(), CRITERIA)
    chin, dorsum = r.overlays[codes.CHIN].as_dict(), r.overlays[codes.NOSE_DORSUM].as_dict()
    assert chin["image"] == dorsum["image"] == "profile"
    assert list(chin["points"]) == ["g", "sn", "pog"] and chin["lines"] == [["g", "sn"], ["sn", "pog"]]
    assert list(dorsum["points"]) == ["g", "n", "prn", "pog"] and dorsum["lines"] == [["g", "pog"], ["n", "prn"]]
    assert dorsum["points"]["prn"] == list(r.points["prn"])


def test_crop_is_square_around_the_head(profile_face):
    models = ColorModels()
    r = P.analyze_profile(profile_face().image, models, CRITERIA)
    assert len(models.segmented) == 2 and models.segmented[1][0] == models.segmented[1][1] == r.crop.side


# ---------------------------------------------------------------- FEI


def _fei_ready() -> bool:
    try:
        assets.asset_path("face_landmarker.task")
        assets.asset_path("selfie_multiclass_256x256.tflite")
    except assets.AssetError:
        return False
    return all(fei.image_path(1, n).is_file() for n in range(1, fei.POSES + 1))


@pytest.fixture(scope="module")
def fei_models():
    if not _fei_ready():
        pytest.skip("FEI images or the MediaPipe model files are not in data/ and api/assets/")
    from facemeasure.landmarks import FaceModels

    with FaceModels() as models:
        yield models


@pytest.mark.parametrize("side", [0, 1])
def test_fei_first_person_full_profiles_are_measured(fei_models, side):
    r = P.analyze_profile(image_io.read_image(fei.image_path(1, fei.PROFILE_FULL[side])), fei_models)
    assert r.failed == () and r.measured, r.reasons


@pytest.mark.parametrize("which", ["neutral", "smile"])
def test_fei_frontal_is_invalid(fei_models, which):
    number = fei.FRONTAL_NEUTRAL if which == "neutral" else fei.FRONTAL_SMILE
    r = P.analyze_profile(image_io.read_image(fei.image_path(1, number)), fei_models)
    assert r.failed == (codes.PROFILE_INVALID,)
