"""model/scripts/make_web_fixtures.py: the example results of the web tests."""

import json

import pytest

from facemeasure import codes, paths
from make_web_fixtures import (
    CODES_NAME,
    FRONTAL_POINTS,
    PORTRAIT,
    PROFILE_POINTS,
    WEBCAM,
    WEBCAM_IOD_PX,
    cases,
    main,
    write_all,
)


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    out = tmp_path_factory.mktemp("fixtures")
    write_all(out)
    return out


def load(folder, name):
    with open(folder / name, encoding="utf-8") as f:
        return json.load(f)


def test_every_case_and_the_codes_are_written(written):
    names = sorted(p.name for p in written.glob("*.json"))
    assert names == sorted([f"{n}.json" for n in cases()] + [CODES_NAME])


def test_the_committed_files_are_up_to_date(written):
    # A change of the texts, the criteria or response.py needs the files written again (README)
    committed = paths.path("web_fixtures")
    for path in sorted(written.glob("*.json")):
        mine = committed / path.name
        assert mine.is_file(), f"{path.name} is missing; run uv run python model/scripts/make_web_fixtures.py"
        assert mine.read_bytes() == path.read_bytes(), f"{path.name} is out of date; run make_web_fixtures.py"


def test_files_are_written_the_same_on_every_machine(written, tmp_path):
    main(["--out", str(tmp_path)])
    for path in written.glob("*.json"):
        raw = path.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")
        assert (tmp_path / path.name).read_bytes() == raw


def test_api_example_is_the_one_of_api_md(written, api_example):
    data = load(written, "api_example.json")
    assert data["result"] == api_example
    assert data["meta"] == {"age": 22, "sex": "female", "frontal_source": "camera", "lang": "th", "lab": False}


def test_each_file_holds_meta_images_and_a_full_result(written):
    for name in cases():
        data = load(written, f"{name}.json")
        assert list(data) == ["name", "meta", "images", "result"] and data["name"] == name
        assert data["result"]["lab"] is data["meta"]["lab"]
        frontal = data["images"]["frontal"]
        assert frontal["source"] == data["meta"]["frontal_source"] and frontal["width"] > 0 and frontal["height"] > 0
        if name != "api_example":
            assert [r["region"] for r in data["result"]["regions"]] == list(codes.REGIONS)
        for region in data["result"]["regions"]:
            overlay = region["overlay"]
            if overlay is None:
                continue
            size = data["images"][overlay["image"]]
            for x, y in overlay["points"].values():
                assert 0 <= x <= size["width"] and 0 <= y <= size["height"]


def test_the_default_frame_leaves_the_points_where_they_are():
    assert PORTRAIT.points(FRONTAL_POINTS) == FRONTAL_POINTS
    assert PORTRAIT.points(PROFILE_POINTS) == PROFILE_POINTS


def test_webcam_case_is_a_small_face_in_a_1080p_frame(written):
    # the size of real photos (web.md section 3.1), next to the portrait images of the other cases
    data = load(written, "webcam_1080p.json")
    for name in ("frontal", "profile"):
        assert {k: data["images"][name][k] for k in ("width", "height")} == {"width": 1920, "height": 1080}
    assert data["result"]["quality"]["frontal"]["iod_px"] == WEBCAM_IOD_PX
    clear = load(written, "female_clear.json")
    assert [r["region"] for r in data["result"]["regions"]] == [r["region"] for r in clear["result"]["regions"]]
    assert [r["decision"] for r in data["result"]["regions"]] == [r["decision"] for r in clear["result"]["regions"]]
    jaw = next(r for r in data["result"]["regions"] if r["region"] == codes.JAW)["overlay"]["points"]
    width = jaw["go_r"][0] - jaw["go_l"][0]
    assert width == pytest.approx((FRONTAL_POINTS["go_r"][0] - FRONTAL_POINTS["go_l"][0]) * WEBCAM.scale, abs=0.2)
    assert width < 0.15 * WEBCAM.width
    for name in cases():
        if name != "webcam_1080p":
            assert load(written, f"{name}.json")["images"]["frontal"]["width"] == PORTRAIT.width


def test_no_profile_image_has_no_profile_size(written):
    data = load(written, "no_face_no_profile.json")
    assert data["images"]["profile"] is None
    assert [(r["image"], r["code"]) for r in data["result"]["retake"]] == [("frontal", "no_face"), ("profile", "missing_input")]


def test_the_cases_cover_what_the_web_shows(written):
    seen = {"withheld": set(), "levels": set(), "status_codes": set(), "ranges": set()}
    for name in cases():
        result = load(written, f"{name}.json")["result"]
        for r in result["regions"]:
            seen["withheld"] |= set(r["procedures_withheld"])
            seen["status_codes"] |= set(r["status_reasons"]) | set(r["warnings"])
            d = r["decision"]
            seen["levels"].add(None if d is None else d["level"])
            if d is not None and d["grade_range"] is not None:
                seen["ranges"].add((r["region"].split("_")[0], *d["grade_range"]))
    assert seen["withheld"] == set(codes.WITHHELD)
    assert seen["levels"] == {"clear", "borderline", None}
    assert ("undereye", "0", "2+") in seen["ranges"]
    assert {"no_face", "missing_input", "hair_occlusion", "model_not_ready", "face_too_small"} <= seen["status_codes"]
    lab = load(written, "lab.json")["result"]
    assert lab["lab"] is True and lab["retake"] == [] and all(r["warnings"] for r in lab["regions"])


def test_codes_file_has_every_code(written):
    data = load(written, CODES_NAME)
    assert data["status_codes"] == list(codes.STATUS_CODES)
    assert data["withheld"] == list(codes.WITHHELD)
    assert data["procedures"] == list(codes.PROCEDURES)
    assert data["notes"] == list(codes.NOTES)
    assert data["regions"] == list(codes.REGIONS)
    assert set(data["notes_with_procedures"]) == set(codes.NOTES_WITH_PROCEDURES)
