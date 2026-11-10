"""Which quality code affects which region (api.md section 2.3)."""

import pytest

from facemeasure import codes

ALL_FRONTAL = {"jaw", "undereye_left", "undereye_right", "nose_base"}
# The table of api.md section 2.3, without hair_occlusion, which depends on where the hair is
EXPECTED = {
    "no_face": ALL_FRONTAL,
    "multiple_faces": ALL_FRONTAL,
    "face_too_small": ALL_FRONTAL,
    "frontal_pose": ALL_FRONTAL,
    "smiling": ALL_FRONTAL,
    "eyes_closed": {"undereye_left", "undereye_right"},
}


def test_region_codes():
    assert codes.REGIONS == ("jaw", "chin", "undereye_left", "undereye_right", "nose_base", "nose_dorsum")
    assert set(codes.FRONTAL_REGIONS) == ALL_FRONTAL


@pytest.mark.parametrize("code", sorted(EXPECTED))
def test_code_affects_regions(code):
    out = codes.by_region((code,))
    assert {r for r, c in out.items() if code in c} == EXPECTED[code]
    assert set(out) == ALL_FRONTAL


@pytest.mark.parametrize(
    "hair_over, regions",
    [
        (("hair_jaw",), {"jaw"}),
        (("hair_undereye_left",), {"undereye_left"}),
        (("hair_undereye_right",), {"undereye_right"}),
        (("hair_jaw", "hair_undereye_right"), {"jaw", "undereye_right"}),
    ],
)
def test_hair_affects_only_covered_parts(hair_over, regions):
    out = codes.by_region(("hair_occlusion",), hair_over)
    assert {r for r, c in out.items() if "hair_occlusion" in c} == regions
    assert out["nose_base"] == ()


def test_codes_keep_their_order():
    out = codes.by_region(("smiling", "multiple_faces", "eyes_closed"))
    assert out["undereye_left"] == ("multiple_faces", "smiling", "eyes_closed")
    assert out["jaw"] == ("multiple_faces", "smiling")


def test_bad_input():
    with pytest.raises(ValueError):
        codes.by_region(("not_a_code",))
    with pytest.raises(ValueError):
        codes.by_region(("hair_occlusion",))


def test_profile_codes():
    assert codes.PROFILE_REGIONS == ("chin", "nose_dorsum")
    assert codes.QUALITY_CODES == codes.FRONTAL_CODES + ("profile_invalid", "missing_input")
    for code in ("profile_invalid", "missing_input"):
        assert codes.profile_by_region((code,)) == {"chin": (code,), "nose_dorsum": (code,)}
        assert set(codes.AFFECTS[code]) == {"chin", "nose_dorsum"}
    assert codes.profile_by_region(()) == {"chin": (), "nose_dorsum": ()}
    with pytest.raises(ValueError):
        codes.profile_by_region(("no_face",))
    with pytest.raises(ValueError):
        codes.by_region(("profile_invalid",))  # the frontal regions do not take profile codes


def test_measurements_and_methods_of_api_md():
    assert codes.REGION_MEASURE == {
        "jaw": "facial_taperness",
        "chin": "legan_angle",
        "nose_base": "alar_intercanthal_ratio",
        "nose_dorsum": "nasofacial_angle",
    }
    assert codes.MEASURE_METHOD == {
        "facial_taperness": "model",
        "legan_angle": "measured",
        "alar_intercanthal_ratio": "model",
        "nasofacial_angle": "measured",
    }


def test_rule_codes():
    assert codes.WITHHELD == (
        "male_jaw_no_criterion", "distance_unknown", "age_under_18", "age_21_or_under", "no_procedure_at_grade"
    )
    assert len(codes.PROCEDURES) == 9 and len(set(codes.PROCEDURES)) == 9
    assert codes.NOTES_WITH_PROCEDURES == set(codes.NOTES) - {"undereye_scope"}
    assert "model_not_ready" in codes.STATUS_CODES and "model_not_ready" not in codes.QUALITY_CODES
    every = [*codes.REGIONS, *codes.QUALITY_CODES, codes.MODEL_NOT_READY, *codes.PROCEDURES, *codes.NOTES, *codes.WITHHELD]
    assert len(every) == len(set(every))
