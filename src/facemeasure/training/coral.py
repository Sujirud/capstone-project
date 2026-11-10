"""Ordinal head of the under-eye grade in the form of CORAL [28], and its temperature.

model.md sections 4.2 and 6. One score s = z . w on the standardized features z and one
bias per transition: P(grade >= k) = sigmoid(s + b_k) for k = 1 and 2.

- Loss: the mean over rows of BCE(k = 1) + BCE(k = 2) against the soft labels, plus
  lam |w|^2; the biases are not penalized.
- Standardization: mean and population SD of the train rows (ridge.Standardizer, SD at
  least ridge.STD_FLOOR), then fold() puts it into w and b in float64, so the bundle and
  the api use sigmoid((x . w + b_k) / T) on raw features (bundle.ordinal).
- Fit: PyTorch in float64 on the CPU, full-batch L-BFGS with a strong Wolfe line search,
  from all zeros, so the same rows give the same head every time.
- Temperature: one T for both transitions that minimizes the same cross-entropy on the
  validation rows, searched on log T within T_BOUNDS (model.md section 4.2, [10]).
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import scipy.optimize
import torch
import torch.nn.functional as F

from facemeasure.training.ridge import Standardizer, fold

TRANSITIONS = (1, 2)
MAX_ITER = 3000
HISTORY = 20
TOLERANCE_GRAD = 1e-7  # largest absolute gradient at which L-BFGS stops
TOLERANCE_CHANGE = 1e-12
T_BOUNDS = (0.25, 4.0)
T_XATOL = 1e-9  # on log T


def _targets(y) -> np.ndarray:
    a = np.asarray(y, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != len(TRANSITIONS):
        raise ValueError(f"targets must be (n, 2): P(grade >= 1) and P(grade >= 2), got shape {a.shape}")
    if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError("targets must be probabilities between 0 and 1")
    return a


def cross_entropy(logits, y) -> float:
    """Mean over rows of the binary cross-entropy summed over both transitions."""
    z, t = np.asarray(logits, dtype=np.float64), _targets(y)
    if z.shape != t.shape:
        raise ValueError(f"logits {z.shape} and targets {t.shape} differ")
    if not len(z):
        return math.nan
    return float(np.mean(np.sum(np.logaddexp(0.0, z) - t * z, axis=1)))


def sigmoid(z) -> np.ndarray:
    from scipy.special import expit

    return expit(np.asarray(z, dtype=np.float64))


@dataclass(frozen=True)
class Ordinal:
    """A fitted head: coefficients on standardized features, two biases, lam and how L-BFGS ended."""

    standardizer: Standardizer
    coef: np.ndarray  # (d,)
    intercept: np.ndarray  # (2,): b_1 and b_2
    lam: float
    iterations: int
    evaluations: int
    converged: bool  # stopped before MAX_ITER
    grad_max: float  # largest absolute gradient at the end
    loss: float  # training objective at the end (with the penalty)
    seconds: float

    def logits(self, x) -> np.ndarray:
        """(n, 2) logits s + b_k of raw features."""
        s = self.standardizer.transform(np.atleast_2d(x)) @ self.coef
        return s[:, None] + self.intercept[None, :]

    def predict(self, x, temperature: float = 1.0) -> np.ndarray:
        return sigmoid(self.logits(x) / float(temperature))

    def folded(self) -> tuple[np.ndarray, np.ndarray]:
        """Weights and biases on raw features: x . w + b_k equals the standardized logits."""
        w, b = fold(self.standardizer, self.coef, self.intercept)
        return w, np.asarray(b, dtype=np.float64)

    def info(self) -> dict:
        return {"lam": self.lam, "iterations": self.iterations, "evaluations": self.evaluations, "converged": self.converged,
                "grad_max": self.grad_max, "loss": self.loss, "seconds": round(self.seconds, 3)}


def folded_logits(x, weights, biases) -> np.ndarray:
    """(n, 2) x . w + b_k on raw features, as bundle.ordinal before the temperature."""
    s = np.asarray(x, dtype=np.float64) @ np.asarray(weights, dtype=np.float64)
    return np.atleast_1d(s)[:, None] + np.asarray(biases, dtype=np.float64)[None, :]


def _check_lam(lam: float) -> float:
    if not (isinstance(lam, int | float) and math.isfinite(lam) and lam > 0):
        raise ValueError(f"lam must be a finite number > 0, got {lam!r}")
    return float(lam)


class Problem:
    """The standardized train rows and targets once, then a fit for any lam (each from zeros)."""

    def __init__(self, x, y):
        a = np.asarray(x)
        if a.ndim != 2:
            raise ValueError(f"features must be (n, d), got shape {a.shape}")
        t = _targets(y)
        if len(t) != len(a):
            raise ValueError(f"{len(a)} feature rows but {len(t)} targets")
        if not len(a):
            raise ValueError("no train rows")
        self.standardizer = Standardizer.fit(a)
        self._z = torch.from_numpy(self.standardizer.transform(a))
        self._y = torch.from_numpy(np.ascontiguousarray(t))
        self.shape = a.shape

    def fit(self, lam: float, max_iter: int = MAX_ITER) -> Ordinal:
        lam = _check_lam(lam)
        started = time.perf_counter()
        d = self.shape[1]
        theta = torch.zeros(d + 2, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, max_eval=max_iter * 2, tolerance_grad=TOLERANCE_GRAD,
                                tolerance_change=TOLERANCE_CHANGE, history_size=HISTORY, line_search_fn="strong_wolfe")
        z, y = self._z, self._y

        def objective():
            w, b = theta[:d], theta[d:]
            logits = (z @ w)[:, None] + b[None, :]
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(dim=1).mean()
            return bce + lam * (w * w).sum()

        def closure():
            opt.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        opt.step(closure)
        state = opt.state[theta]
        loss = objective()
        grad = torch.autograd.grad(loss, theta)[0]
        iterations = int(state.get("n_iter", 0))
        coef = theta[:d].detach().numpy().copy()
        intercept = theta[d:].detach().numpy().copy()
        return Ordinal(
            standardizer=self.standardizer,
            coef=coef,
            intercept=intercept,
            lam=lam,
            iterations=iterations,
            evaluations=int(state.get("func_evals", 0)),
            converged=iterations < max_iter,
            grad_max=float(grad.abs().max()),
            loss=float(loss.detach()),
            seconds=time.perf_counter() - started,
        )


def fit(x, y, lam: float, max_iter: int = MAX_ITER) -> Ordinal:
    """The ordinal head for one lam."""
    return Problem(x, y).fit(lam, max_iter)


# ---------------------------------------------------------------- temperature


@dataclass(frozen=True)
class Temperature:
    temperature: float
    ce_before: float  # at T = 1
    ce_after: float
    at_bound: bool  # T sits at an end of T_BOUNDS

    def as_dict(self) -> dict:
        return {"temperature": self.temperature, "ce_before": self.ce_before, "ce_after": self.ce_after, "at_bound": self.at_bound,
                "bounds": list(T_BOUNDS)}


def fit_temperature(logits, y, bounds: tuple[float, float] = T_BOUNDS) -> Temperature:
    """One T for both transitions that minimizes cross_entropy(logits / T, y) within bounds."""
    z, t = np.asarray(logits, dtype=np.float64), _targets(y)
    if not len(z):
        raise ValueError("no rows to fit the temperature on")
    low, high = math.log(bounds[0]), math.log(bounds[1])

    def f(u: float) -> float:
        return cross_entropy(z / math.exp(u), t)

    r = scipy.optimize.minimize_scalar(f, bounds=(low, high), method="bounded", options={"xatol": T_XATOL})
    candidates = [(float(r.fun), float(r.x)), (f(low), low), (f(high), high)]
    best_f, best_u = min(candidates, key=lambda c: (c[0], abs(c[1])))
    temperature = math.exp(best_u)
    at_bound = math.isclose(best_u, low, abs_tol=1e-6) or math.isclose(best_u, high, abs_tol=1e-6)
    return Temperature(temperature, cross_entropy(z, t), best_f, at_bound)
