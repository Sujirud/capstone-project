"""facemeasure.training.metrics on small examples worked out by hand."""

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from facemeasure import codes, criteria
from facemeasure.training import metrics


def test_summary_and_slope():
    s = metrics.summary([0.82, 0.86, 0.90], [0.80, 0.86, 0.94])
    assert s["n"] == 3 and s["mae"] == pytest.approx(0.02) and s["bias"] == pytest.approx(-0.02 / 3)
    assert s["rmse"] == pytest.approx(np.sqrt((0.0004 + 0 + 0.0016) / 3))
    # predictions pulled halfway to the mean: slope 0.5
    y = np.array([0.7, 0.8, 0.9])
    assert metrics.slope(0.8 + 0.5 * (y - 0.8), y) == pytest.approx(0.5)
    assert metrics.line_fit([1.0, 2.0, 3.0], [3.0, 5.0, 7.0]) == pytest.approx((2.0, 1.0))


def test_jaw_grades_only_women_and_nose_grades_by_sex():
    assert metrics.grades(codes.JAW, [0.80, 0.85, 0.90], ["female", "female", "male"]) == ["0", "1", None]
    np.testing.assert_array_equal(metrics.graded(codes.JAW, ["female", "male", "female"]), [True, False, True])
    assert metrics.graded(codes.NOSE_BASE, ["female", "male"]).all()
    # 1.18: wide side grade 1 for women (above 1.15), grade 0 for men (up to 1.21)
    assert metrics.grades(codes.NOSE_BASE, [1.18, 1.18, 0.90, 1.30], ["female", "male", "female", "female"]) == ["1", "0", "1", "2+"]


def test_kappa_and_confusion_by_hand():
    labels = ["0", "0", "1", "1", "2+", "2+"]
    preds = ["0", "1", "1", "1", "1", "2+"]
    m = metrics.confusion(labels, preds)
    assert m.tolist() == [[1, 1, 0], [0, 2, 0], [0, 1, 1]]
    # weights 1, 0.5, 0: observed (1 + 2 + 1 + 0.5 + 0.5) / 6 = 5/6; rows 2, 2, 2 and columns 1, 4, 1:
    # expected sum(w_ij * r_i * c_j) / 36 = (2*(1 + 2 + 0) + 2*(0.5 + 4 + 0.5) + 2*(0 + 2 + 1)) / 36 = 22/36
    kappa = (5 / 6 - 22 / 36) / (1 - 22 / 36)
    assert metrics.weighted_kappa(labels, preds) == pytest.approx(kappa)
    assert metrics.weighted_kappa(labels, preds) == pytest.approx(cohen_kappa_score(labels, preds, weights="linear"))
    assert metrics.weighted_kappa(["1", "1"], ["1", "1"]) is None


def test_kappa_of_the_jaw_leaves_out_men():
    values = np.array([0.80, 0.85, 0.90, 0.90])
    labels = np.array([0.80, 0.85, 0.90, 0.70])
    sexes = np.array(["female", "female", "female", "male"])
    g = metrics.graded(codes.JAW, sexes)
    truth, pred = metrics.grades(codes.JAW, labels[g], sexes[g]), metrics.grades(codes.JAW, values[g], sexes[g])
    assert len(truth) == 3 and metrics.weighted_kappa(truth, pred) == pytest.approx(1.0)


def test_band_from_rmse_and_the_cap_of_criteria():
    band = metrics.band_from_rmse(0.05)
    assert round(band, 4) == 0.0262
    assert criteria.capped_bands(criteria.JAW_FEMALE_CUTOFFS, band) == pytest.approx((0.02, 0.02))
    caps = metrics.band_caps(codes.JAW, band)
    assert list(caps) == ["female"] and caps["female"]["capped"] == [True, True]
    small = metrics.band_caps(codes.JAW, 0.01)
    assert small["female"]["bands"] == pytest.approx([0.01, 0.01]) and small["female"]["capped"] == [False, False]


def test_band_outcomes_by_hand():
    # band 0.01 around 0.83 and 0.87, women only
    values = [0.800, 0.835, 0.850, 0.865, 0.900, 0.800]
    labels = [0.800, 0.820, 0.860, 0.880, 0.860, 0.900]
    sexes = ["female"] * 5 + ["male"]
    o = metrics.band_outcomes(codes.JAW, values, labels, sexes, 0.01)
    # 0.800 clear 0 (label 0, hit); 0.835 0-1 (label 0, covered); 0.850 clear 1 (label 1, hit);
    # 0.865 1-2+ (label 2+, covered); 0.900 clear 2+ (label 1, miss); the man is left out
    assert o["n"] == 5 and o["clear_n"] == 3 and o["borderline_n"] == 2
    assert o["borderline"] == pytest.approx(0.4) and o["clear_match"] == pytest.approx(2 / 3) and o["borderline_cover"] == 1.0


def test_bootstrap_is_the_same_with_the_same_seed():
    rng = np.random.default_rng(0)
    images = np.repeat([f"im{i}" for i in range(50)], 2)
    a, b = rng.normal(0, 0.05, 100), rng.normal(0, 0.04, 100)
    one = metrics.bootstrap_rmse_diff(images, a, b, rounds=200, seed=7)
    two = metrics.bootstrap_rmse_diff(images, a, b, rounds=200, seed=7)
    other = metrics.bootstrap_rmse_diff(images, a, b, rounds=200, seed=8)
    assert one == two and one != other
    assert one["diff"] == pytest.approx(np.sqrt(np.mean(a**2)) - np.sqrt(np.mean(b**2)))
    assert one["low"] < one["diff"] < one["high"] and one["images"] == 50


def test_bootstrap_draws_whole_images():
    # one image has both copies way off; drawing whole images makes the interval wider than drawing rows
    images = np.repeat(["a", "b", "c", "d"], 2)
    err = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0])
    r = metrics.bootstrap_rmse_diff(images, err, np.zeros(8), rounds=500, seed=1)
    assert r["low"] == 0.0 and r["high"] > 0.7
