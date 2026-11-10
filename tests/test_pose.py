"""facemeasure.pose against shared/testdata/pose_cases.json, which the web tests too."""

import json

import numpy as np
import pytest

from facemeasure import paths, pose

TOLERANCE_DEG = 0.01


def load_cases() -> dict:
    with open(paths.path("shared") / "testdata" / "pose_cases.json", encoding="utf-8") as f:
        return json.load(f)


CASES = load_cases()


def test_file_layout():
    assert CASES["layout"] == "rows"
    assert 8 <= len(CASES["cases"]) <= 10
    names = [c["name"] for c in CASES["cases"]]
    assert len(set(names)) == len(names)
    for c in CASES["cases"]:
        m = np.array(c["matrix"])
        assert m.shape == (4, 4) and np.allclose(m[3], [0, 0, 0, 1])


def test_cases_cover_both_directions_of_each_axis_and_scaled_moved_matrices():
    built = [c["built_from"] for c in CASES["cases"]]
    for key in ("yaw_deg", "pitch_deg", "roll_z_deg"):
        others = [k for k in ("yaw_deg", "pitch_deg", "roll_z_deg") if k != key]
        single = [b[key] for b in built if b[key] != 0 and all(b[o] == 0 for o in others)]
        assert any(v > 0 for v in single) and any(v < 0 for v in single), key
    assert any(all(b[k] != 0 for k in ("yaw_deg", "pitch_deg", "roll_z_deg")) for b in built)
    assert any(b["scale"] != 1 and any(b["translation"]) for b in built)


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda c: c["name"])
def test_case(case):
    p = pose.angles(case["matrix"])
    assert p.yaw_deg == pytest.approx(case["yaw_deg"], abs=TOLERANCE_DEG)
    assert p.pitch_deg == pytest.approx(case["pitch_deg"], abs=TOLERANCE_DEG)
    assert p.roll_deg == pytest.approx(case["roll_deg"], abs=TOLERANCE_DEG)


def test_rotation_is_orthonormal_for_a_scaled_matrix():
    m = np.array(CASES["cases"][-1]["matrix"])
    r = pose.rotation(m)
    assert np.allclose(r @ r.T, np.eye(3)) and np.linalg.det(r) == pytest.approx(1)


def test_bad_matrix():
    with pytest.raises(ValueError):
        pose.angles(np.eye(2))
    with pytest.raises(ValueError):
        pose.angles(np.full((4, 4), np.nan))
