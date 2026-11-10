"""response.build: normal and lab mode, not assessable regions, rounding and texts (api.md sections 2.2 and 2.4)."""

import json

import pytest

from facemeasure import bundle, codes, response
from facemeasure.i18n import Texts
from facemeasure.response import Meta, Predictions, ResponseError
from make_stub_bundle import stub_bundle, stub_parts

REGION_KEYS = [
    "region", "status", "status_reasons", "warnings", "measurements", "overlay", "probabilities",
    "decision", "procedures", "procedures_withheld", "notes",
]
ALL = Predictions(jaw=0.80, nose_base=1.05, undereye_left=(0.1, 0.05), undereye_right=(0.95, 0.8))


@pytest.fixture(scope="module")
def texts():
    return Texts()


@pytest.fixture(scope="module")
def stub():
    return stub_bundle()


@pytest.fixture
def run(frontal_result, profile_result, texts, ready_bundle):
    def go(frontal=None, profile="default", predictions=ALL, meta=None, lab=False, b=None, t=None):
        frontal = frontal_result() if frontal is None else frontal
        profile = profile_result() if profile == "default" else profile
        meta = Meta(age=30, sex="female") if meta is None else meta
        out = response.build(frontal, profile, predictions, meta, b or ready_bundle, t or texts, lab=lab)
        json.dumps(out, allow_nan=False)  # plain JSON, no numpy numbers or NaN
        return out, {r["region"]: r for r in out["regions"]}

    return go


def test_every_region_in_order_with_every_field(run):
    out, _ = run()
    assert [r["region"] for r in out["regions"]] == list(codes.REGIONS)
    assert all(list(r) == REGION_KEYS for r in out["regions"])
    assert all(r["status"] == "assessed" and r["warnings"] == [] for r in out["regions"])
    assert out["retake"] == []


def test_quality_codes_make_regions_not_assessable_and_ask_for_retake(run, frontal_result):
    frontal = frontal_result(failed=("smiling", "hair_occlusion"), hair_over=("hair_jaw",))
    out, regions = run(frontal=frontal)
    assert regions["jaw"]["status_reasons"] == ["smiling", "hair_occlusion"]
    assert regions["undereye_left"]["status_reasons"] == ["smiling"]
    for name in codes.FRONTAL_REGIONS:
        r = regions[name]
        assert r["status"] == "not_assessable"
        assert r["measurements"] == [] and r["overlay"] is None and r["decision"] is None and r["probabilities"] is None
        assert r["procedures"] == [] and r["procedures_withheld"] == [] and r["notes"] == []
    assert regions["chin"]["status"] == "assessed"
    assert out["retake"] == [
        {"image": "frontal", "code": "smiling", "regions": ["jaw", "undereye_left", "undereye_right", "nose_base"], "text": "ทำหน้าปกติ ปิดปาก ไม่ยิ้ม"},
        {"image": "frontal", "code": "hair_occlusion", "regions": ["jaw"], "text": "รวบผมให้เห็นกรามและใต้ตา"},
    ]


@pytest.mark.parametrize("profile", [None, "missing"])
def test_no_profile_image(run, profile):
    from facemeasure.profile import missing_profile

    out, regions = run(profile=missing_profile() if profile == "missing" else None)
    for name in codes.PROFILE_REGIONS:
        assert regions[name]["status"] == "not_assessable" and regions[name]["status_reasons"] == ["missing_input"]
    assert out["retake"] == [
        {"image": "profile", "code": "missing_input", "regions": ["chin", "nose_dorsum"], "text": "ยังไม่มีภาพด้านข้าง"}
    ]
    assert out["quality"]["profile"] == {
        "face_found": None, "face_yaw_deg": None, "silhouette_found": None, "flipped": None, "failed": ["missing_input"],
    }


def test_no_face(run, frontal_result):
    out, regions = run(frontal=frontal_result(face=False), predictions=Predictions())
    assert all(regions[r]["status_reasons"] == ["no_face"] for r in codes.FRONTAL_REGIONS)
    assert out["quality"]["frontal"]["failed"] == ["no_face"]
    assert {v for k, v in out["quality"]["frontal"].items() if k != "failed"} == {None}
    assert out["retake"][0]["code"] == "no_face"


def test_lab_mode_measures_what_it_can(run, frontal_result, profile_result):
    frontal = frontal_result(failed=("face_too_small", "eyes_closed"))
    profile = profile_result(failed=("profile_invalid",), measured=True, face_found=True, yaw=35.0)
    out, regions = run(frontal=frontal, profile=profile, lab=True)
    assert out["retake"] == []
    assert regions["jaw"]["status"] == "assessed" and regions["jaw"]["warnings"] == ["face_too_small"]
    assert regions["undereye_left"]["warnings"] == ["face_too_small", "eyes_closed"]
    assert regions["chin"]["status"] == "assessed" and regions["chin"]["warnings"] == ["profile_invalid"]
    assert all(r["status_reasons"] == [] for r in out["regions"])
    assert regions["undereye_right"]["decision"]["grade"] == "2+"


def test_lab_mode_cannot_measure_without_a_face_or_an_edge(run, frontal_result, profile_result):
    profile = profile_result(failed=("profile_invalid",), measured=False)
    out, regions = run(frontal=frontal_result(face=False), profile=profile, predictions=Predictions(), lab=True)
    assert out["retake"] == []
    for name in codes.REGIONS:
        r = regions[name]
        assert r["status"] == "not_assessable" and r["warnings"] == []
    assert regions["jaw"]["status_reasons"] == ["no_face"] and regions["chin"]["status_reasons"] == ["profile_invalid"]


def test_men_get_the_jaw_ratio_without_a_grade(run):
    _, regions = run(predictions=Predictions(0.95, 1.05, (0.1, 0.0), (0.1, 0.0)), meta=Meta(age=30, sex="male"))
    jaw = regions["jaw"]
    m = jaw["measurements"][0]
    assert jaw["status"] == "assessed" and jaw["decision"] is None
    assert (m["value"], m["cutoffs"], m["reference"]) == (95.0, [], "ยังไม่มีเกณฑ์ของผู้ชาย จึงแสดงเฉพาะค่า")
    assert jaw["procedures_withheld"] == ["male_jaw_no_criterion"] and jaw["procedures"] == []
    nose = regions["nose_base"]["measurements"][0]
    assert nose["reference"] == "ชาย เกรด 0 อยู่ที่ 0.99–1.21 เกรด ≥2 เมื่อมากกว่า 1.32"


def test_uploaded_frontal_image_shows_only_the_alar_value(run):
    p = Predictions(0.80, 1.40, (0.1, 0.0), (0.1, 0.0))
    _, regions = run(predictions=p, meta=Meta(age=30, sex="female", frontal_source="upload"))
    nose = regions["nose_base"]
    assert nose["status"] == "assessed" and nose["decision"] is None
    assert nose["procedures_withheld"] == ["distance_unknown"] and nose["notes"] == []
    m = nose["measurements"][0]
    assert (m["value"], m["unit"], m["method"]) == (1.4, "ratio", "model")
    assert m["cutoffs"] == [{"value": 0.95, "band": 0.04}, {"value": 1.15, "band": 0.04}, {"value": 1.25, "band": 0.04}]
    assert regions["jaw"]["decision"] is not None  # only the alar base loses its grade


def test_alar_base_with_procedures(run):
    _, regions = run(predictions=Predictions(0.80, 1.30, (0.1, 0.0), (0.1, 0.0)))
    nose = regions["nose_base"]
    assert nose["decision"] == {"grade": "2+", "grade_range": None, "level": "clear"}
    assert [p["code"] for p in nose["procedures"]] == ["alar_reduction"]
    assert nose["procedures"][0] == {
        "code": "alar_reduction", "name": "ตัดปีกจมูก", "when": "เมื่อฐานจมูกกว้างกว่าช่วงที่คนไทยชอบ", "checks": None,
    }
    assert [(n["code"], n["with_procedures"]) for n in nose["notes"]] == [("nose_base_indications", True)]
    assert nose["overlay"] == {
        "image": "frontal",
        "points": {"al_l": [585.0, 800.0], "al_r": [695.0, 801.0], "en_l": [590.0, 560.0], "en_r": [690.0, 561.0]},
        "lines": [["al_l", "al_r"], ["en_l", "en_r"]],
    }
    assert nose["measurements"][0]["landmark_value"] == 1.08  # stub baseline: slope 1, intercept 0


def test_undereye(run):
    _, regions = run()
    left, right = regions["undereye_left"], regions["undereye_right"]
    assert left["measurements"] == []
    assert left["probabilities"] == {"grades": {"0": 0.9, "1": 0.05, "2+": 0.05}, "at_least": {"1": 0.1, "2+": 0.05}}
    assert left["decision"]["grade"] == "0" and left["procedures_withheld"] == ["no_procedure_at_grade"]
    assert left["notes"] == [{"code": "undereye_scope", "text": "ระบบประเมินเฉพาะร่องลึกใต้ตา ไม่ได้ประเมินถุงใต้ตาหรือรอยคล้ำ", "with_procedures": False}]
    assert [p["code"] for p in right["procedures"]] == ["undereye_filler"]
    assert left["overlay"]["image"] == "frontal" and list(left["overlay"]["points"]) == ["tl", "tr", "br", "bl"]
    assert left["overlay"]["points"]["tl"] == [683.2, 560.5]  # pixels to one decimal
    assert left["overlay"]["lines"] == [["tl", "tr"], ["tr", "br"], ["br", "bl"], ["bl", "tl"]]
    assert right["overlay"]["points"]["br"] == [618.6, 617.6]


@pytest.mark.parametrize("age, reasons", [(17, ["age_under_18"]), (21, ["age_21_or_under"]), (22, [])])
def test_undereye_ages(run, age, reasons):
    _, regions = run(meta=Meta(age=age, sex="female"))
    r = regions["undereye_right"]
    assert r["procedures_withheld"] == reasons
    assert bool(r["procedures"]) == (not reasons)
    assert [n["code"] for n in r["notes"]] == ["undereye_scope"]


def test_bands_come_from_the_bundle_and_keep_three_digits(run, profile_result, trained):
    manifest, arrays = stub_parts()
    manifest = trained(manifest)
    manifest["heads"]["jaw"]["band"] = 0.0262  # capped at 2 points by the cutoffs 83 and 87
    manifest["files"] = {"heads.npz": "0" * 64}
    b = bundle.parse(manifest, arrays)
    _, regions = run(predictions=Predictions(0.8455, 1.05, (0.1, 0.0), (0.1, 0.0)), b=b, profile=profile_result(nasofacial=33.0))
    m = regions["jaw"]["measurements"][0]
    assert m["value"] == 84.5 and m["cutoffs"] == [{"value": 83.0, "band": 2.0}, {"value": 87.0, "band": 2.0}]
    assert m["reference"] == "จุดตัด 83% ช่วงก้ำกึ่ง ±2"
    assert regions["jaw"]["decision"]["grade_range"] == ["0", "1"]
    dorsum = regions["nose_dorsum"]["measurements"][0]
    assert [c["band"] for c in dorsum["cutoffs"]] == [1.35, 1.35, 1.5]
    assert regions["nose_dorsum"]["decision"]["grade_range"] == ["0", "1"]


def test_chin(run, profile_result):
    _, regions = run(profile=profile_result(legan=21.0), meta=Meta(age=40, sex="male"))
    chin = regions["chin"]
    m = chin["measurements"][0]
    assert (m["name"], m["value"], m["unit"], m["method"], m["landmark_value"], m["source"]) == (
        "legan_angle", 21.0, "deg", "measured", None, "[5][26]"
    )
    assert m["reference"] == "ค่าปกติ 12° ช่วงที่ยอมรับได้ 8–16° แบ่งระดับทีละ 4°"
    assert chin["decision"] == {"grade": None, "grade_range": ["1", "2"], "level": "borderline"}
    assert [p["code"] for p in chin["procedures"]] == ["chin_filler", "chin_implant", "genioplasty"]
    assert [n["code"] for n in chin["notes"]] == ["chin_check_occlusion"]
    assert chin["overlay"]["lines"] == [["g", "sn"], ["sn", "pog"]]


def test_rounding_of_quality(run, frontal_result):
    values = {
        "iod_px": 96.04, "yaw_deg": -0.04, "pitch_deg": 1.25, "roll_deg": 0.0, "smile": 0.12345, "jaw_open": 0.0001,
        "blink_left": 0.5, "blink_right": 0.4999, "hair_jaw": 0.0, "hair_undereye_left": 0.1234, "hair_undereye_right": 0.0,
    }
    out, _ = run(frontal=frontal_result(values=values))
    q = out["quality"]["frontal"]
    assert (q["iod_px"], q["yaw_deg"], q["smile"], q["jaw_open"], q["blink_right"], q["hair_undereye_left"]) == (
        96.0, 0.0, 0.123, 0.0, 0.5, 0.123
    )
    assert json.dumps(q["yaw_deg"]) == "0.0"  # never -0.0


@pytest.mark.parametrize(
    "value, digits, text",
    [(83.0, 3, "83"), (1.35, 3, "1.35"), (1.3499999999999979, 3, "1.35"), (30.1, 1, "30.1"), (-0.0001, 3, "0"), (12, 3, "12")],
)
def test_short_numbers(value, digits, text):
    assert response.short(value, digits) == text


def test_texts_follow_the_language_and_fall_back_to_thai(run, tmp_path):
    with open(tmp_path / "th.json", "w", encoding="utf-8") as f, open(i18n_th(), encoding="utf-8") as src:
        f.write(src.read())
    with open(tmp_path / "en.json", "w", encoding="utf-8") as f:
        json.dump({"disclaimer": "Not a diagnosis.", "procedure.nose_filler.name": "Nose filler"}, f)
    t = Texts(tmp_path)
    out, regions = run(meta=Meta(age=30, sex="female", lang="en"), t=t, predictions=ALL)
    assert out["disclaimer"] == "Not a diagnosis."
    assert regions["undereye_left"]["notes"][0]["text"].startswith("ระบบ")  # no English text yet
    out, _ = run(meta=Meta(age=30, sex="female", lang="xx"))
    assert out["disclaimer"].startswith("ผลนี้")


def i18n_th():
    from facemeasure import i18n

    return i18n.folder() / "th.json"


def test_a_found_face_needs_its_predictions(run):
    with pytest.raises(ResponseError, match="prediction"):
        run(predictions=Predictions(jaw=0.8))


def test_unmeasurable_region_without_a_code_is_an_error(run, frontal_result):
    with pytest.raises(ResponseError, match="no quality code"):
        run(frontal=frontal_result(ratios=False))


def test_regions_with_codes_need_no_predictions(run, frontal_result):
    _, regions = run(frontal=frontal_result(failed=("frontal_pose",)), predictions=Predictions())
    assert all(regions[r]["status"] == "not_assessable" for r in codes.FRONTAL_REGIONS)


@pytest.mark.parametrize("kw", [{"age": -1}, {"age": 20.5}, {"sex": "other"}, {"frontal_source": "file"}])
def test_bad_meta(kw):
    with pytest.raises(ValueError):
        Meta(**{"age": 30, "sex": "female", **kw})


def test_stub_heads_give_model_not_ready_and_no_retake(run, stub, frontal_result, profile_result):
    frontal = frontal_result(failed=("smiling",))
    out, regions = run(frontal=frontal, b=stub)
    for name in codes.FRONTAL_REGIONS:
        r = regions[name]
        assert r["status"] == "not_assessable" and r["status_reasons"] == ["smiling", "model_not_ready"]
    for name in codes.PROFILE_REGIONS:  # measured on the edge, no head
        assert regions[name]["status"] == "assessed" and regions[name]["status_reasons"] == []
    assert [(i["image"], i["code"]) for i in out["retake"]] == [("frontal", "smiling")]
    assert all("model_not_ready" not in i["regions"] and i["code"] != "model_not_ready" for i in out["retake"])

    out, regions = run(b=stub)
    assert out["retake"] == []
    assert all(regions[n]["status_reasons"] == ["model_not_ready"] for n in codes.FRONTAL_REGIONS)

    out, regions = run(frontal=frontal, b=stub, lab=True)
    assert out["retake"] == []
    for name in codes.FRONTAL_REGIONS:
        r = regions[name]
        assert r["status"] == "assessed" and r["warnings"] == ["smiling", "model_not_ready"] and r["status_reasons"] == []
        assert r["decision"] is not None


def test_only_the_stub_heads_are_not_ready(run, trained):
    manifest, arrays = stub_parts()
    manifest = trained(manifest)
    manifest["heads"]["undereye"]["stub"] = True
    manifest["stub"] = True
    manifest["files"] = {"heads.npz": "0" * 64}
    _, regions = run(b=bundle.parse(manifest, arrays))
    assert [n for n in codes.REGIONS if regions[n]["status"] == "not_assessable"] == ["undereye_left", "undereye_right"]
    assert regions["undereye_left"]["status_reasons"] == ["model_not_ready"]


def test_lab_mode_without_a_face_and_a_stub_head_lists_both(run, stub, frontal_result):
    _, regions = run(frontal=frontal_result(face=False), predictions=Predictions(), b=stub, lab=True)
    assert regions["jaw"]["status"] == "not_assessable"
    assert regions["jaw"]["status_reasons"] == ["no_face", "model_not_ready"]
