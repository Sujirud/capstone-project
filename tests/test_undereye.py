"""Under-eye grade from P(grade >= k) (model.md section 4.2)."""

import math

import pytest

from facemeasure import undereye


def test_example_of_model_md():
    # P(0) = 0.40, P(1) = 0.35, P(>=2) = 0.25: the most likely grade is 0, but P(>=1) = 0.60
    r = undereye.decide(1 - 0.40, 0.25)
    assert r.decision.grade is None and r.decision.grade_range == ("0", "1")
    assert r.decision.level == "borderline"
    assert r.grades == pytest.approx({"0": 0.40, "1": 0.35, "2+": 0.25})
    assert r.at_least == pytest.approx({"1": 0.60, "2+": 0.25})


@pytest.mark.parametrize(
    "p1, p2, grade, grade_range",
    [
        (0.10, 0.05, "0", None),  # not reached at 1
        (0.30, 0.30, "0", None),  # 0.3 itself does not reach
        (0.85, 0.10, "1", None),  # reached 1, not 2
        (0.70, 0.30, "1", None),  # 0.7 itself reaches
        (0.95, 0.80, "2+", None),
        (0.70, 0.70, "2+", None),
        (0.50, 0.20, None, ("0", "1")),  # borderline at 1, not reached at 2
        (0.90, 0.50, None, ("1", "2+")),  # reached 1, borderline at 2
        (0.60, 0.40, None, ("0", "2+")),  # borderline at both
    ],
)
def test_every_combination(p1, p2, grade, grade_range):
    d = undereye.decide(p1, p2).decision
    assert (d.grade, d.grade_range) == (grade, grade_range)


def test_p2_above_p1_is_lowered():
    r = undereye.decide(0.5, 0.6)
    assert r.at_least_2 == 0.5 and r.grades["1"] == 0.0
    assert r.decision.grade_range == ("0", "2+")


def test_thresholds_from_the_bundle():
    assert undereye.decide(0.6, 0.1, low=0.2, high=0.55).decision.grade == "1"
    assert undereye.decide(0.6, 0.1, low=0.2, high=0.65).decision.grade_range == ("0", "1")


def test_grades_add_up_to_one():
    for p1, p2 in [(0.0, 0.0), (1.0, 1.0), (0.37, 0.11), (0.8, 0.79)]:
        r = undereye.decide(p1, p2)
        assert math.fsum(r.grades.values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in r.grades.values())


@pytest.mark.parametrize("p1, p2", [(-0.1, 0.0), (1.1, 0.5), (math.nan, 0.1), (0.5, math.inf), (True, 0.1)])
def test_bad_probabilities_raise(p1, p2):
    with pytest.raises(ValueError, match="probability"):
        undereye.decide(p1, p2)


@pytest.mark.parametrize("low, high", [(0.7, 0.3), (0.5, 0.5), (-0.1, 0.7)])
def test_bad_thresholds_raise(low, high):
    with pytest.raises(ValueError):
        undereye.decide(0.5, 0.2, low, high)
