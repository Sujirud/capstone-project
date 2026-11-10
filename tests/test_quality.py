"""Quality limits from shared/quality.json, failed codes and hair fractions."""

import json

import numpy as np
import pytest

from facemeasure import crops, quality
from facemeasure.landmarks import HAIR

SPEC = {
    "version": 1,
    "frontal": {
        "min_iod_px": 75,
        "max_abs_yaw_deg": 10,
        "max_abs_pitch_deg": 10,
        "max_abs_roll_deg": 8,
        "max_smile": 0.3,
        "max_jaw_open": 0.2,
        "max_eye_blink": 0.5,
        "hair_band_width_iod": 0.05,
        "max_hair_fraction": 0.1,
    },
    "profile": {"min_abs_yaw_deg": 60},
    "request": {"max_bytes": 26214400},
}
EPS = 1e-6


def write(tmp_path, data):
    path = tmp_path / "quality.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def test_shared_file_has_the_agreed_values():
    c = quality.load_config()
    assert c.frontal == quality.FrontalCriteria(**SPEC["frontal"])
    assert c.profile.min_abs_yaw_deg == 40 and c.request.max_bytes == 26214400


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["frontal"].pop("max_smile"),
        lambda d: d["frontal"].update(extra=1),
        lambda d: d.pop("request"),
        lambda d: d.update(more={}),
        lambda d: d["profile"].update(min_abs_yaw_deg="60"),
        lambda d: d["frontal"].update(max_smile=True),
        lambda d: d["request"].update(max_bytes=1.5),
        lambda d: d.update(version=2),
    ],
)
def test_bad_config_is_an_error(tmp_path, change):
    data = json.loads(json.dumps(SPEC))
    change(data)
    with pytest.raises(quality.QualityConfigError):
        quality.load_config(write(tmp_path, data))


def good_values() -> dict:
    return {
        "iod_px": 100.0, "yaw_deg": 1.0, "pitch_deg": -1.0, "roll_deg": 0.5,
        "smile": 0.05, "jaw_open": 0.02, "blink_left": 0.05, "blink_right": 0.05,
        "hair_jaw": 0.0, "hair_undereye_left": 0.0, "hair_undereye_right": 0.0,
    }


CRITERIA = quality.FrontalCriteria(**SPEC["frontal"])
# value name, value at the limit, value just past it, code
LIMITS = [
    ("iod_px", 75.0, 75.0 - EPS, "face_too_small"),
    ("yaw_deg", 10.0, 10.0 + EPS, "frontal_pose"),
    ("yaw_deg", -10.0, -10.0 - EPS, "frontal_pose"),
    ("pitch_deg", 10.0, 10.0 + EPS, "frontal_pose"),
    ("pitch_deg", -10.0, -10.0 - EPS, "frontal_pose"),
    ("roll_deg", 8.0, 8.0 + EPS, "frontal_pose"),
    ("roll_deg", -8.0, -8.0 - EPS, "frontal_pose"),
    ("smile", 0.3, 0.3 + EPS, "smiling"),
    ("jaw_open", 0.2, 0.2 + EPS, "smiling"),
    ("blink_left", 0.5, 0.5 + EPS, "eyes_closed"),
    ("blink_right", 0.5, 0.5 + EPS, "eyes_closed"),
    ("hair_jaw", 0.1, 0.1 + EPS, "hair_occlusion"),
    ("hair_undereye_left", 0.1, 0.1 + EPS, "hair_occlusion"),
    ("hair_undereye_right", 0.1, 0.1 + EPS, "hair_occlusion"),
]


def test_good_values_pass():
    failed, by_region = quality.evaluate(good_values(), 1, CRITERIA)
    assert failed == () and all(v == () for v in by_region.values())


@pytest.mark.parametrize("name, at, past, code", LIMITS)
def test_limit_passes_and_just_past_fails(name, at, past, code):
    v = good_values()
    v[name] = at
    assert quality.evaluate(v, 1, CRITERIA)[0] == ()
    v[name] = past
    assert quality.evaluate(v, 1, CRITERIA)[0] == (code,)


def test_hair_codes_go_to_covered_regions():
    v = good_values()
    v["hair_undereye_left"] = 0.5
    _, by_region = quality.evaluate(v, 1, CRITERIA)
    assert by_region == {"jaw": (), "undereye_left": ("hair_occlusion",), "undereye_right": (), "nose_base": ()}


def test_two_faces_and_no_face():
    failed, by_region = quality.evaluate(good_values(), 2, CRITERIA)
    assert failed == ("multiple_faces",) and by_region["nose_base"] == ("multiple_faces",)
    failed, by_region = quality.evaluate({}, 0, CRITERIA)
    assert failed == ("no_face",) and all(v == ("no_face",) for v in by_region.values())


def test_as_dict_matches_api_fields():
    q = quality.FrontalQuality(**good_values(), failed=("smiling",))
    assert list(q.as_dict()) == [*good_values(), "failed"] and q.as_dict()["failed"] == ["smiling"]


def test_blendshape_values():
    bs = {"mouthSmileLeft": 0.2, "mouthSmileRight": 0.4, "jawOpen": 0.1, "eyeBlinkLeft": 0.3, "eyeBlinkRight": 0.6}
    assert quality.blendshape_values(bs) == {"smile": 0.4, "jaw_open": 0.1, "blink_left": 0.3, "blink_right": 0.6}


# ------------------------------------------------------------------ hair on synthetic masks

# A face 4 times the size of the default one, so the 0.05 x IOD band is 20 pixels wide
K = 4.0
CX, IOD, EYE_Y, SUB_Y, MOUTH_Y, CHIN_Y, HALF_W, TOP_Y = (K * v for v in (200, 100, 200, 260, 300, 360, 110, 80))
SIZE = (int(K * 400), int(K * 500))
BAND_W = 0.05 * IOD  # 20 px on the face side of each edge
BAND_TOP = (EYE_Y + SUB_Y) / 2  # 920
BAND_BOTTOM = (MOUTH_Y + CHIN_Y) / 2  # 1320
LEFT_EDGE, RIGHT_EDGE = CX - HALF_W, CX + HALF_W  # 360 and 1240


def setup_face(mp_face):
    lm = mp_face(cx=CX, iod=IOD, eye_y=EYE_Y, sub_y=SUB_Y, mouth_y=MOUTH_Y, chin_y=CHIN_Y, top_y=TOP_Y, half_width=HALF_W)
    face_crops = crops.make_crops(np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8), lm)
    box = quality.hair_crop(lm)
    xs, ys = box.pixel_centers()
    return lm, face_crops, box, xs, ys


def run(lm, face_crops, box, category, size=SIZE):
    return quality.hair_fractions(
        category, box, size, lm, face_crops.frame, IOD,
        face_crops.undereye_left.corners, face_crops.undereye_right.corners, BAND_W / IOD,
    )


def test_band_is_inside_the_edge_between_its_levels(mp_face):
    lm, face_crops, box, xs, ys = setup_face(mp_face)
    # Levels are in the level frame, where eye level is y = 0
    assert quality.jaw_band_levels(lm, face_crops.frame) == pytest.approx((BAND_TOP - EYE_Y, BAND_BOTTOM - EYE_Y))
    left, right = quality.jaw_band_masks(box, lm, face_crops.frame, BAND_W)
    for band, lo, hi in ((left, LEFT_EDGE, LEFT_EDGE + BAND_W), (right, RIGHT_EDGE - BAND_W, RIGHT_EDGE)):
        assert band.any()
        assert xs[band].min() > lo and xs[band].max() <= hi
        assert ys[band].min() == pytest.approx(BAND_TOP, abs=1) and ys[band].max() == pytest.approx(BAND_BOTTOM, abs=1)
    want = BAND_W * (BAND_BOTTOM - BAND_TOP)  # 5 x 100 px
    assert left.sum() == pytest.approx(want, rel=0.03) and right.sum() == pytest.approx(want, rel=0.03)


def test_hair_beside_the_face_does_not_count(mp_face):
    lm, face_crops, box, xs, _ = setup_face(mp_face)
    category = np.zeros((box.side, box.side), dtype=np.uint8)
    category[(xs < LEFT_EDGE) | (xs > RIGHT_EDGE)] = HAIR  # long hair outside both edges
    assert run(lm, face_crops, box, category)["hair_jaw"] == 0


def test_hair_over_half_of_one_band(mp_face):
    lm, face_crops, box, xs, _ = setup_face(mp_face)
    category = np.zeros((box.side, box.side), dtype=np.uint8)
    category[xs < LEFT_EDGE + BAND_W / 2] = HAIR  # outer half of the image-left band, and beyond
    out = run(lm, face_crops, box, category)
    assert out["hair_jaw"] == pytest.approx(0.5, abs=0.02)
    assert out["hair_undereye_left"] == 0 and out["hair_undereye_right"] == 0


def test_hair_over_the_top_quarter_of_one_band(mp_face):
    lm, face_crops, box, xs, ys = setup_face(mp_face)
    category = np.zeros((box.side, box.side), dtype=np.uint8)
    split = BAND_TOP + 0.25 * (BAND_BOTTOM - BAND_TOP)
    category[(ys < split) & (xs > CX)] = HAIR  # top quarter of the image-right band, and above it
    assert run(lm, face_crops, box, category)["hair_jaw"] == pytest.approx(0.25, abs=0.02)


def test_band_follows_a_tilted_face(mp_face):
    lm, _, _, _, _ = setup_face(mp_face)
    t = np.radians(12)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    tilted = lm.copy()
    tilted[:, :2] = (lm[:, :2] - (CX, K * 250)) @ r.T + (CX, K * 250)
    face_crops = crops.make_crops(np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8), tilted)
    box = quality.hair_crop(tilted)
    left, right = quality.jaw_band_masks(box, tilted, face_crops.frame, BAND_W)
    want = BAND_W * (BAND_BOTTOM - BAND_TOP)
    assert left.sum() == pytest.approx(want, rel=0.05) and right.sum() == pytest.approx(want, rel=0.05)


def test_hair_in_the_under_eye_boxes(mp_face):
    lm, face_crops, box, xs, ys = setup_face(mp_face)
    # Person's right eye is on the image left at x = 600; its box spans x 480-720, y 800-1040
    assert np.allclose(face_crops.undereye_right.corners, [(480, 800), (720, 800), (720, 1040), (480, 1040)])
    category = np.zeros((box.side, box.side), dtype=np.uint8)
    category[xs < 540] = HAIR  # left quarter of the person's right box, nothing in the other box
    category[(xs > CX) & (ys > 980)] = HAIR  # bottom quarter of the person's left box (y 800-1040)
    out = run(lm, face_crops, box, category)
    assert out["hair_undereye_right"] == pytest.approx(0.25, abs=0.02)
    assert out["hair_undereye_left"] == pytest.approx(0.25, abs=0.02)


def test_pixels_outside_the_image_do_not_count(mp_face):
    lm, face_crops, box, xs, _ = setup_face(mp_face)
    category = np.zeros((box.side, box.side), dtype=np.uint8)
    category[xs < CX] = HAIR
    # An image 368 px wide keeps only the 8 outer pixel columns of the image-left band, all
    # hair, and the image-right band is outside the image, so it counts as 0
    out = run(lm, face_crops, box, category, size=(int(LEFT_EDGE) + 8, SIZE[1]))
    assert out["hair_jaw"] == pytest.approx(1.0)


def test_hair_crop_is_square_and_padded(mp_face):
    lm = mp_face()
    box = quality.hair_crop(lm)
    pts = lm[:, :2]
    longer = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    assert box.side == pytest.approx(1.3 * longer, abs=1)
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    cut = quality.HairCrop(x0=-10, y0=90, side=30).cut(image)
    assert cut.shape == (30, 30, 3)
    assert cut[0, 0].max() == 0 and cut[5, 15].min() == 255 and cut[15, 15].max() == 0
