"""Ridge regression for the jaw and alar base heads, and the choice of feature and alpha.

Each feature is standardized with the mean and standard deviation of the train rows
(population SD, at least STD_FLOOR), then ridge fits with an intercept:
minimize |y - Z w - b|^2 + alpha |w|^2, the same as sklearn.linear_model.Ridge on Z.

- fit() solves one alpha directly (Cholesky), in the primal (Z'Z, d x d) when there
  are at least as many rows as dimensions and in the dual (ZZ', n x n) otherwise,
  so grid4 (12,288 dimensions) on 11,212 train rows needs an 11,212 x 11,212 matrix.
- RidgePath does one eigendecomposition of the same matrix and then gives every alpha
  and every target column at almost no cost, for the search of select.
- fold() puts the standardization into the weights in float64, so the bundle and the
  api use x . w + b on raw features (bundle.ridge).

Choice (search, choose): the RMSE on val of every feature type and alpha; when the
best alpha of a feature is at an end of the grid, the grid grows EXPAND_DECADES
decades past that end once. Then every pair within TOLERANCE (1%) of the lowest RMSE
counts as equal, and the smallest feature type wins, with its largest alpha.
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.linalg

STD_FLOOR = 1e-6
ALPHAS = tuple(10.0**k for k in range(-2, 6))
EXPAND_DECADES = 2
TOLERANCE = 0.01


def _matrix(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError(f"features must be (n, d), got shape {a.shape}")
    return a


def _targets(y) -> tuple[np.ndarray, bool]:
    a = np.asarray(y, dtype=np.float64)
    if a.ndim == 1:
        return a[:, None], True
    if a.ndim != 2:
        raise ValueError(f"targets must be (n,) or (n, k), got shape {a.shape}")
    return a, False


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray  # (d,) float64
    scale: np.ndarray  # (d,) float64, >= STD_FLOOR

    @classmethod
    def fit(cls, x) -> "Standardizer":
        a = _matrix(x)
        mean = a.mean(axis=0, dtype=np.float64)
        var = np.zeros(a.shape[1])
        for start in range(0, len(a), 4096):  # in parts, so a float32 matrix is not copied whole to float64
            part = a[start : start + 4096].astype(np.float64) - mean
            var += (part * part).sum(axis=0)
        std = np.sqrt(var / len(a))
        return cls(mean=mean, scale=np.maximum(std, STD_FLOOR))

    def transform(self, x) -> np.ndarray:
        z = np.array(_matrix(x), dtype=np.float64, copy=True)
        z -= self.mean
        z /= self.scale
        return z


def fold(standardizer: Standardizer, coef: np.ndarray, intercept) -> tuple[np.ndarray, np.ndarray | float]:
    """Weights and bias on raw features: ((x - m) / s) . c + b = x . (c / s) + (b - m . (c / s))."""
    c = np.asarray(coef, dtype=np.float64)
    w = c / (standardizer.scale if c.ndim == 1 else standardizer.scale[:, None])
    b = np.asarray(intercept, dtype=np.float64) - standardizer.mean @ w
    return w, (float(b) if np.ndim(b) == 0 else b)


@dataclass(frozen=True)
class Ridge:
    """A fitted ridge: coefficients on standardized features, intercept, alpha."""

    standardizer: Standardizer
    coef: np.ndarray  # (d,) or (d, k)
    intercept: float | np.ndarray
    alpha: float

    def predict(self, x) -> np.ndarray:
        return self.standardizer.transform(x) @ self.coef + self.intercept

    def folded(self) -> tuple[np.ndarray, np.ndarray | float]:
        return fold(self.standardizer, self.coef, self.intercept)

    def column(self, k: int) -> "Ridge":
        """The ridge of one target column of a multi-target fit."""
        if self.coef.ndim == 1:
            raise ValueError("this ridge has one target")
        return Ridge(self.standardizer, self.coef[:, k].copy(), float(np.asarray(self.intercept)[k]), self.alpha)


def _check_alpha(alpha: float) -> float:
    if not (isinstance(alpha, int | float) and math.isfinite(alpha) and alpha > 0):
        raise ValueError(f"alpha must be a finite number > 0, got {alpha!r}")
    return float(alpha)


def _centered(x, y):
    std = Standardizer.fit(x)
    z = std.transform(x)
    z_mean = z.mean(axis=0)
    z -= z_mean
    y2, one = _targets(y)
    if len(y2) != len(z):
        raise ValueError(f"{len(z)} feature rows but {len(y2)} targets")
    y_mean = y2.mean(axis=0)
    return std, z, z_mean, y2 - y_mean, y_mean, one


def _finish(std, coef, z_mean, y_mean, one, alpha) -> Ridge:
    intercept = y_mean - z_mean @ coef
    if one:
        return Ridge(std, coef[:, 0], float(intercept[0]), alpha)
    return Ridge(std, coef, intercept, alpha)


def fit(x, y, alpha: float) -> Ridge:
    """Ridge with an intercept on standardized x for one alpha; y is (n,) or (n, k)."""
    alpha = _check_alpha(alpha)
    std, z, z_mean, yc, y_mean, one = _centered(x, y)
    n, d = z.shape
    if n >= d:
        gram = z.T @ z
        gram[np.diag_indices_from(gram)] += alpha
        coef = scipy.linalg.solve(gram, z.T @ yc, assume_a="pos", overwrite_a=True)
    else:
        gram = z @ z.T
        gram[np.diag_indices_from(gram)] += alpha
        coef = z.T @ scipy.linalg.solve(gram, yc, assume_a="pos", overwrite_a=True)
    return _finish(std, coef, z_mean, y_mean, one, alpha)


class RidgePath:
    """One eigendecomposition of the standardized train rows, then ridge for any alpha and target column."""

    def __init__(self, x, y):
        started = time.perf_counter()
        self.standardizer, z, self._z_mean, yc, self._y_mean, self._one = _centered(x, y)
        n, d = z.shape
        self.dual = n < d
        self.shape = (n, d)
        if self.dual:
            gram = z @ z.T
            self._values, vectors = scipy.linalg.eigh(gram, overwrite_a=True, check_finite=False)
            del gram
            self._z = z  # coef = Z' U diag(1 / (l + alpha)) U' y
            self._vectors = vectors
            self._projected = vectors.T @ yc
        else:
            gram = z.T @ z
            zy = z.T @ yc
            del z
            self._values, vectors = scipy.linalg.eigh(gram, overwrite_a=True, check_finite=False)
            del gram
            self._z = None  # coef = V diag(1 / (l + alpha)) V' Z' y
            self._vectors = vectors
            self._projected = vectors.T @ zy
        self._values = np.maximum(self._values, 0.0)  # the matrix is positive semidefinite; drop rounding below 0
        self.seconds = time.perf_counter() - started

    @property
    def memory_bytes(self) -> int:
        """Bytes of the arrays the path keeps (the standardized rows in the dual, the eigenvectors)."""
        kept = self._vectors.nbytes + self._projected.nbytes + self._values.nbytes
        return kept + (self._z.nbytes if self._z is not None else 0)

    @property
    def peak_matrix_bytes(self) -> int:
        """Bytes of the largest float64 arrays alive at once while building: the rows, the matrix and its eigenvectors."""
        n, d = self.shape
        m = min(n, d)
        return 8 * (n * d + 2 * m * m)

    def coef(self, alpha: float) -> np.ndarray:
        alpha = _check_alpha(alpha)
        scaled = self._projected / (self._values + alpha)[:, None]
        if self.dual:
            return self._z.T @ (self._vectors @ scaled)
        return self._vectors @ scaled

    def model(self, alpha: float) -> Ridge:
        return _finish(self.standardizer, self.coef(alpha), self._z_mean, self._y_mean, self._one, float(alpha))

    def predict(self, x, alpha: float) -> np.ndarray:
        return self.model(alpha).predict(x)


# ---------------------------------------------------------------- choice


def rmse(pred, y) -> float:
    e = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean(e * e)))


@dataclass(frozen=True)
class Trial:
    feature: str
    alpha: float
    rmse: float


@dataclass(frozen=True)
class Grid:
    """RMSE of one feature type at every alpha tried, in increasing alpha."""

    feature: str
    alphas: tuple[float, ...]
    rmse: tuple[float, ...]
    expanded: tuple[float, ...] = ()  # alphas added past an end of the first grid
    at_edge: bool = False  # the best alpha is still at an end after growing the grid

    @property
    def best(self) -> Trial:
        k = int(np.argmin(self.rmse))
        return Trial(self.feature, self.alphas[k], self.rmse[k])

    def trials(self) -> list[Trial]:
        return [Trial(self.feature, a, r) for a, r in zip(self.alphas, self.rmse, strict=True)]


def past_edge(alphas: Sequence[float], best: float, decades: int = EXPAND_DECADES) -> tuple[float, ...]:
    """Alphas to add when best is the smallest or largest alpha: decades more decades past that end."""
    low, high = min(alphas), max(alphas)
    if best == high:
        return tuple(high * 10.0**k for k in range(1, decades + 1))
    if best == low:
        return tuple(low / 10.0**k for k in range(1, decades + 1))
    return ()


def search(feature: str, score: Callable[[Sequence[float]], Sequence[float]], alphas: Sequence[float] = ALPHAS,
           decades: int = EXPAND_DECADES) -> Grid:
    """RMSE of every alpha; the grid grows once past an end when the best alpha sits there."""
    tried = {float(a): float(r) for a, r in zip(alphas, score(list(alphas)), strict=True)}
    best = min(tried, key=tried.get)
    extra = past_edge(list(tried), best, decades)
    if extra:
        tried.update({float(a): float(r) for a, r in zip(extra, score(list(extra)), strict=True)})
        best = min(tried, key=tried.get)
    order = sorted(tried)
    at_edge = best in (order[0], order[-1])
    return Grid(feature, tuple(order), tuple(tried[a] for a in order), tuple(sorted(extra)), at_edge)


def choose(grids: Sequence[Grid], dims: dict[str, int], tolerance: float = TOLERANCE) -> Trial:
    """The lowest RMSE, but within tolerance of it the feature with fewer dimensions, then the larger alpha."""
    trials = [t for g in grids for t in g.trials()]
    if not trials:
        raise ValueError("nothing to choose from")
    lowest = min(t.rmse for t in trials)
    near = [t for t in trials if t.rmse == lowest or t.rmse < lowest * (1 + tolerance)]
    return min(near, key=lambda t: (dims[t.feature], -t.alpha))
