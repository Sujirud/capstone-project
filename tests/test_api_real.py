"""The api with the real engine on dataset images (api.md section 7).

The checks are those of model/scripts/api_check.py, with fewer images. Every test
skips when the model files, the bundle or the images it needs are not on this machine.
"""

import pytest

import api_check


@pytest.fixture(scope="module")
def client():
    api_check.project_on_path()
    from api.engine import StartupError

    try:
        c = api_check.real_client()
        c.__enter__()
    except StartupError as e:
        pytest.skip(f"the real engine does not start here: {e}")
    yield c
    c.__exit__(None, None, None)


@pytest.fixture(scope="module")
def cannot(client):
    """check name -> why it cannot run here."""
    return {**api_check.missing_inputs(), **api_check.stub_reasons(client)}


def run(client, cannot, name, **kw):
    if name in cannot:
        pytest.skip(cannot[name])
    return api_check.CHECKS[name](client, **kw)


def test_values_match_the_data_preparation(client, cannot):
    ratio = run(client, cannot, "ratio_fidelity", n=5)
    assert ratio["images"] == 5 and ratio["same_codes"], ratio
    assert ratio["passed"], ratio


def test_undereye_probabilities_match_the_data_preparation(client, cannot):
    result = run(client, cannot, "undereye_fidelity", n=5)
    assert result["passed"], result


def test_the_same_request_gives_the_same_result(client, cannot):
    assert run(client, cannot, "repeat")["passed"]


def test_exif_orientation(client, cannot):
    result = run(client, cannot, "exif", pairs=2)
    assert result["png_same_result"] and result["upright_shape_kept"], result
    assert result["png_max_jaw_pp"] <= api_check.EXIF_JAW_PP and result["png_max_angle_deg"] <= api_check.ANGLE_DEG
    assert result["passed"], result


def test_a_mirrored_profile_keeps_its_angles(client, cannot):
    result = run(client, cannot, "profile_flip", people=5)
    assert result["direction_flips"], result
    assert result["passed"], result


def test_quality_codes(client, cannot):
    result = run(client, cannot, "quality_codes", images=3, smile_people=10, invalid_people=5)
    for code in ("face_too_small", "frontal_pose", "smiling", "profile_invalid", "missing_input"):
        assert result[code]["passed"], (code, result[code])


def test_lab_mode_measures_a_photo_that_fails(client, cannot):
    result = run(client, cannot, "lab", people=10)
    assert result["found"] and result["lab"] is True and result["retake_in_lab"] == 0, result
    assert result["passed"], result
