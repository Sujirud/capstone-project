"""facemeasure.training.ridge on random and synthetic data."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge as SkRidge

from facemeasure.bundle import ridge as bundle_ridge
from facemeasure.training import ridge

ALL_ALPHAS = (*ridge.ALPHAS, 1e-4, 1e6, 1e7)


def random_data(n, d, k=None, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) * rng.uniform(0.1, 5.0, d) + rng.normal(0, 3, d)
    w = rng.normal(size=(d,) if k is None else (d, k))
    y = x @ w + rng.normal(0, 0.5, n if k is None else (n, k))
    return x, y


@pytest.mark.parametrize("n,d", [(200, 30), (40, 120)])  # primal and dual
def test_fit_and_path_equal_sklearn_at_every_alpha(n, d):
    x, y = random_data(n, d)
    z = ridge.Standardizer.fit(x).transform(x)
    path = ridge.RidgePath(x, y)
    assert path.dual == (n < d)
    for alpha in ALL_ALPHAS:
        sk = SkRidge(alpha=alpha).fit(z, y)
        for model in (ridge.fit(x, y, alpha), path.model(alpha)):
            np.testing.assert_allclose(model.coef, sk.coef_, rtol=0, atol=1e-6)
            assert model.intercept == pytest.approx(sk.intercept_, abs=1e-6)
            np.testing.assert_allclose(model.predict(x), sk.predict(z), rtol=0, atol=1e-6)


def test_two_targets_at_once_equal_one_at_a_time():
    x, y = random_data(150, 20, k=2, seed=1)
    path = ridge.RidgePath(x, y)
    for alpha in (0.1, 10.0, 1e4):
        both = ridge.fit(x, y, alpha)
        for k in range(2):
            one = ridge.fit(x, y[:, k], alpha)
            np.testing.assert_allclose(both.column(k).coef, one.coef, atol=1e-9)
            np.testing.assert_allclose(path.model(alpha).column(k).predict(x), one.predict(x), atol=1e-9)


def test_standardizer_uses_population_sd_with_a_floor():
    x = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    s = ridge.Standardizer.fit(x)
    np.testing.assert_allclose(s.mean, [3.0, 5.0])
    np.testing.assert_allclose(s.scale, [np.sqrt(8 / 3), ridge.STD_FLOOR])
    assert np.isfinite(ridge.fit(x, [1.0, 2.0, 3.0], 1.0).predict(x)).all()


@pytest.mark.parametrize("n,d", [(300, 40), (50, 200)])
def test_folded_weights_predict_the_same_as_standardizing_first(n, d):
    x, y = random_data(n, d, seed=2)
    x = x.astype(np.float32)  # stored features are float32
    model = ridge.fit(x, y, 3.0)
    w, b = model.folded()
    x_new = random_data(25, d, seed=3)[0].astype(np.float32)
    np.testing.assert_allclose(bundle_ridge(x_new, w, b), model.predict(x_new), rtol=0, atol=1e-9)
    assert bundle_ridge(x_new[0], w, b) == pytest.approx(float(model.predict(x_new[:1])[0]), abs=1e-9)


def test_alpha_must_be_positive():
    x, y = random_data(20, 3)
    for bad in (0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            ridge.fit(x, y, bad)


def test_grid_grows_past_the_end_where_the_best_alpha_is():
    assert ridge.past_edge([1, 10, 100], 100) == (1000.0, 10000.0)
    assert ridge.past_edge([1, 10, 100], 1) == (0.1, 0.01)
    assert ridge.past_edge([1, 10, 100], 10) == ()
    calls = []

    def score(alphas):
        calls.append(list(alphas))
        return [abs(np.log10(a) - 7) for a in alphas]  # best at 1e7, beyond the grid

    g = ridge.search("f", score, alphas=(1.0, 1e2, 1e4, 1e5))
    assert calls == [[1.0, 1e2, 1e4, 1e5], [1e6, 1e7]]  # grows once, two decades
    assert g.expanded == (1e6, 1e7) and g.best.alpha == 1e7 and g.at_edge
    inner = ridge.search("f", lambda a: [abs(np.log10(v) - 2) for v in a], alphas=(1.0, 1e2, 1e4))
    assert inner.expanded == () and inner.best.alpha == 1e2 and not inner.at_edge


def test_within_one_percent_the_smaller_feature_and_larger_alpha_win():
    grids = [
        ridge.Grid("big", (1.0, 10.0), (0.1000, 0.1050)),
        ridge.Grid("small", (1.0, 10.0, 100.0), (0.1009, 0.10095, 0.1020)),
    ]
    c = ridge.choose(grids, {"big": 100, "small": 10})
    assert (c.feature, c.alpha) == ("small", 10.0)  # 0.10095 is within 1% of 0.1, 0.1020 is not
    far = [ridge.Grid("big", (1.0,), (0.1,)), ridge.Grid("small", (1.0,), (0.102,))]
    assert ridge.choose(far, {"big": 100, "small": 10}).feature == "big"


def test_search_finds_the_feature_set_that_carries_the_signal():
    rng = np.random.default_rng(4)
    n_train, n_val, d = 400, 200, 30
    signal = rng.normal(size=(n_train + n_val, d))
    noise = rng.normal(size=(n_train + n_val, d // 2))  # fewer dimensions, so only accuracy can make it lose
    y = signal @ rng.normal(0, 0.05, d) + 0.85 + rng.normal(0, 0.01, n_train + n_val)
    sets = {"signal": signal, "noise": noise}
    grids = []
    for name, x in sets.items():
        path = ridge.RidgePath(x[:n_train], y[:n_train])

        def score(alphas, path=path, x=x):
            return [ridge.rmse(path.predict(x[n_train:], a), y[n_train:]) for a in alphas]

        grids.append(ridge.search(name, score))
    choice = ridge.choose(grids, {"signal": d, "noise": d // 2})
    assert choice.feature == "signal"
    assert choice.rmse < 0.5 * min(g.best.rmse for g in grids if g.feature == "noise")
