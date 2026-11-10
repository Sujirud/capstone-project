"""The ordinal head and its temperature on synthetic data."""

import numpy as np
import pytest

from facemeasure import bundle
from facemeasure.training import coral


def ordinal_data(n=1500, d=12, seed=0, shift=(1.0, -1.5)):
    """Features with an offset and scale, a known direction, and grades drawn from the ordinal model."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) * rng.uniform(0.5, 3.0, d) + rng.normal(0, 5, d)
    w = np.zeros(d)
    w[:3] = (1.5, -1.0, 0.8)
    s = ((x - x.mean(axis=0)) / x.std(axis=0)) @ w
    p1, p2 = 1 / (1 + np.exp(-(s + shift[0]))), 1 / (1 + np.exp(-(s + shift[1])))
    u = rng.uniform(size=n)
    grade = (u < p1).astype(int) + (u < p2).astype(int)
    y = np.column_stack([grade >= 1, grade >= 2]).astype(float)
    return x, y, w


def test_known_direction_and_ordered_biases():
    x, y, w = ordinal_data()
    m = coral.fit(x, y, 1e-4)
    assert m.converged and m.iterations > 0
    assert np.corrcoef(m.coef, w)[0, 1] > 0.95
    assert np.sign(m.coef[:3]).tolist() == [1.0, -1.0, 1.0]
    assert m.intercept[0] > m.intercept[1]
    assert m.intercept == pytest.approx([1.0, -1.5], abs=0.3)


def test_refit_gives_the_same_head():
    x, y, _ = ordinal_data(seed=1)
    a, b = coral.fit(x, y, 1e-2), coral.Problem(x, y).fit(1e-2)
    np.testing.assert_array_equal(a.coef, b.coef)
    np.testing.assert_array_equal(a.intercept, b.intercept)


def test_folded_weights_predict_as_the_standardized_head():
    x, y, _ = ordinal_data(seed=2)
    m = coral.fit(x, y, 1e-3)
    w, b = m.folded()
    assert b.shape == (2,)
    np.testing.assert_allclose(coral.folded_logits(x, w, b), m.logits(x), atol=1e-9, rtol=0)
    for t in (0.7, 1.0, 1.8):
        np.testing.assert_allclose(bundle.ordinal(x, w, b, t), m.predict(x, t), atol=1e-9, rtol=0)
    assert bundle.ordinal(x[0], w, b, 1.0).shape == (2,)


def test_penalty_shrinks_the_weights_but_not_the_biases():
    x, y, _ = ordinal_data(seed=3)
    small, big = coral.fit(x, y, 1e-4), coral.fit(x, y, 10.0)
    assert np.linalg.norm(big.coef) < 0.2 * np.linalg.norm(small.coef)
    # with weights near 0 the biases still match the share of each transition
    p = big.predict(x).mean(axis=0)
    assert p == pytest.approx(y.mean(axis=0), abs=0.03)


def test_cross_entropy_of_soft_labels():
    z = np.array([[0.0, 0.0], [2.0, -1.0]])
    y = np.array([[0.5, 0.5], [1.0, 0.0]])
    want = (2 * np.log(2) + (np.log1p(np.exp(-2.0)) + np.log1p(np.exp(-1.0)))) / 2
    assert coral.cross_entropy(z, y) == pytest.approx(want, abs=1e-12)
    with pytest.raises(ValueError):
        coral.cross_entropy(z, np.array([[1.5, 0.0], [0.0, 0.0]]))


def test_temperature_of_overconfident_logits_is_above_one():
    rng = np.random.default_rng(4)
    true = rng.normal(0, 1.5, size=(2000, 2))
    true[:, 1] -= 2.0
    y = (rng.uniform(size=true.shape) < 1 / (1 + np.exp(-true))).astype(float)
    t = coral.fit_temperature(3.0 * true, y)
    assert t.temperature > 1 and t.temperature == pytest.approx(3.0, rel=0.15)
    assert t.ce_after < t.ce_before and not t.at_bound
    under = coral.fit_temperature(0.5 * true, y)
    assert under.temperature < 1


def test_temperature_stops_at_the_bounds():
    z = np.array([[20.0, -20.0]] * 50 + [[-20.0, 20.0]] * 50)
    y = np.array([[0.5, 0.5]] * 100)
    t = coral.fit_temperature(z, y)
    assert t.temperature == pytest.approx(coral.T_BOUNDS[1]) and t.at_bound


def test_bad_inputs():
    with pytest.raises(ValueError):
        coral.fit(np.zeros((3, 2)), np.zeros((3, 2)), 0.0)
    with pytest.raises(ValueError):
        coral.Problem(np.zeros((3, 2)), np.zeros((2, 2)))
