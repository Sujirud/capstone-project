"""The example result of docs/api.md section 2.2, rebuilt through response.py.

The JSON is read from the document itself (conftest.load_api_example), so a change of
wording in th.json, of the criteria or of the rounding that breaks the example fails
here. The inputs are those of the example: a woman of 22, a camera frontal image, the
jaw head at 0.836 and the MediaPipe jaw ratio 0.842 through the stub baseline, and a
nasofacial angle of 28.5. The stub bundle is marked as trained (ready_bundle), because a
stub head gives model_not_ready.
"""

import pytest

from facemeasure import response
from facemeasure.i18n import Texts


@pytest.fixture(scope="module")
def example(api_example):
    return api_example


@pytest.fixture
def built(example, frontal_result, profile_result, ready_bundle):
    regions = {r["region"]: r for r in example["regions"]}
    jaw_points = {k: tuple(v) for k, v in regions["jaw"]["overlay"]["points"].items()}
    profile_points = {k: tuple(v) for k, v in regions["nose_dorsum"]["overlay"]["points"].items()}
    frontal = frontal_result(
        values={k: v for k, v in example["quality"]["frontal"].items() if k != "failed"},
        jaw_ratio=0.842,
        points={**jaw_points, "en_l": (590.0, 560.0), "en_r": (690.0, 561.0), "al_l": (585.0, 800.0), "al_r": (695.0, 801.0)},
    )
    q = example["quality"]["profile"]
    profile = profile_result(
        nasofacial=28.5, legan=14.0, face_found=q["face_found"], yaw=q["face_yaw_deg"],
        points={**profile_points, "sn": (700.0, 800.0)},
    )
    predictions = response.Predictions(jaw=0.836, nose_base=1.08, undereye_left=(0.1, 0.05), undereye_right=(0.1, 0.05))
    meta = response.Meta(age=22, sex="female", frontal_source="camera")
    return response.build(frontal, profile, predictions, meta, ready_bundle, Texts(), lab=example["lab"])


def test_example_regions_match(example, built):
    mine = {r["region"]: r for r in built["regions"]}
    for region in example["regions"]:
        assert mine[region["region"]] == region


def test_example_disclaimer_quality_and_retake_match(example, built):
    assert built["disclaimer"] == example["disclaimer"]
    assert built["quality"] == example["quality"]
    assert built["retake"] == example["retake"]


def test_example_keys_are_the_top_level_of_the_document(example, built):
    # request_id, versions and lab are added by the api
    assert list(built) == [k for k in example if k not in ("request_id", "versions", "lab")]
