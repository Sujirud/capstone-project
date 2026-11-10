"""Measures of the under-eye head on small examples worked by hand."""

import numpy as np
import pandas as pd
import pytest

from facemeasure.training import undereye_metrics as um


def test_single_grade_counts_transitions_at_or_above_half():
    p = np.array([[0.2, 0.1], [0.5, 0.2], [0.9, 0.5], [0.49, 0.6], [0.7, 0.49]])
    assert um.single_grade(p).tolist() == [0, 1, 2, 0, 1]  # P(>= 2) is lowered to P(>= 1) first


def test_ece_by_hand():
    p = np.array([0.05, 0.15, 0.15, 0.95])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    # bins: [0, 0.1) one row |0.05 - 0|, [0.1, 0.2) two rows |0.15 - 0.5|, [0.9, 1] one row |0.95 - 1|
    want = 0.25 * 0.05 + 0.5 * 0.35 + 0.25 * 0.05
    assert um.ece(p, y) == pytest.approx(want, abs=1e-12)
    bins = um.reliability(p, y)
    assert [b["n"] for b in bins] == [1, 2, 0, 0, 0, 0, 0, 0, 0, 1]
    assert bins[1]["mean_p"] == pytest.approx(0.15) and bins[1]["mean_y"] == pytest.approx(0.5) and bins[2]["mean_p"] is None
    assert um.bin_index([0.0, 0.1, 0.999, 1.0]).tolist() == [0, 1, 9, 9]
    assert um.ece([], []) is None


def test_soft_labels_in_ece_and_calibration_of_both_transitions():
    p = np.array([[0.75, 0.25], [0.75, 0.25]])
    y = np.array([[1.0, 0.5], [0.5, 0.0]])
    c = um.calibration(p, y)
    assert c["k1"]["ece"] == pytest.approx(0.0) and c["k2"]["ece"] == pytest.approx(0.0)


def test_versus_rater_skips_rows_without_a_grade():
    model = [0, 1, 2, 2, 0]
    rater = um.rater_grades(pd.array([0, 1, 3, pd.NA, 1], dtype="Int64"))
    assert rater.tolist() == [0, 1, 2, -1, 1]
    r = um.versus_rater(model, rater)
    assert r["n"] == 4 and r["accuracy"] == 0.75
    assert r["confusion"] == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert r["kappa"] is not None and 0 < r["kappa"] < 1
    assert um.versus_rater([1, 1], [1, 1])["kappa"] is None


def test_decision_outcomes_with_the_default_thresholds():
    p = np.array([[0.1, 0.05], [0.9, 0.1], [0.5, 0.1], [0.9, 0.5], [0.9, 0.9]])
    decided = um.decisions(p, 0.3, 0.7)
    assert [d.grade for d in decided] == ["0", "1", None, None, "2+"]
    assert decided[2].grade_range == ("0", "1") and decided[3].grade_range == ("1", "2+")
    rater = np.array([0, 2, 1, 0, 2])
    o = um.decision_outcomes(decided, rater)
    assert o["n"] == 5 and o["borderline"] == pytest.approx(0.4)
    assert o["clear_match"] == pytest.approx(2 / 3) and o["borderline_cover"] == pytest.approx(0.5)
    assert um.decision_outcomes(decided, [-1] * 5)["n"] == 0
    edge = np.array([[0.25, 0.1], [0.75, 0.1]])
    assert um.borderline_share(um.decisions(edge, 0.2, 0.8)) == 1.0
    assert um.borderline_share(um.decisions(edge, 0.3, 0.7)) == 0.0
    assert um.borderline_share([]) is None


def test_between_raters():
    r = um.between_raters([0, 1, 2, -1], [0, 1, 1, 2])
    assert r["n"] == 3 and r["accuracy"] == pytest.approx(2 / 3)
