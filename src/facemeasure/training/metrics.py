"""Measures of the ratio heads on val and test (model.md sections 4.1 and 7.1).

- summary: MAE, RMSE and bias (mean of prediction minus label).
- slope: least-squares slope of the predictions on the labels; 1 means no pull toward
  the mean, ridge gives less.
- grades and weighted_kappa: grades without a borderline band (criteria.jaw_grade and
  criteria.nose_grade), linear (Cicchetti-Allison) weighted kappa and the confusion
  matrix. The jaw grades only women, as the system does; the alar base uses the cutoffs
  of each person's sex.
- bootstrap_rmse_diff: RMSE of one method minus another, with a percentile interval
  from resampling images (every copy of a drawn image comes along).
- band_from_rmse and band_outcomes: the borderline band RMSE_BAND_FACTOR x RMSE, and how
  it behaves on a set: share borderline, share of clear results with the label's grade,
  and share of borderline results whose range holds the label's grade.
"""

import hashlib
from collections.abc import Sequence

import numpy as np

from facemeasure import codes, criteria
from facemeasure.training.ratio_data import MEASURE

GRADES = criteria.GRADES
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = int.from_bytes(hashlib.sha256(b"ratio-heads-bootstrap-v1").digest()[:8], "big") >> 1


def _floats(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def summary(pred, y) -> dict:
    e = _floats(pred) - _floats(y)
    if not len(e):
        return {"n": 0, "mae": None, "rmse": None, "bias": None}
    return {"n": len(e), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e))), "bias": float(np.mean(e))}


def slope(pred, y) -> float:
    """Least-squares slope of pred on y."""
    p, t = _floats(pred), _floats(y)
    t0 = t - t.mean()
    return float((t0 * (p - p.mean())).sum() / (t0 * t0).sum())


def line_fit(x, y) -> tuple[float, float]:
    """Least-squares y = slope * x + intercept, the baseline that turns a MediaPipe ratio into landmark_value."""
    a, b = _floats(x), _floats(y)
    a0 = a - a.mean()
    s = float((a0 * (b - b.mean())).sum() / (a0 * a0).sum())
    return s, float(b.mean() - s * a.mean())


# ---------------------------------------------------------------- grades


def grade(target: str, value: float, sex: str) -> str | None:
    """Grade without a band; None for the jaw of men."""
    if target == codes.JAW:
        return criteria.jaw_grade(float(value), sex)
    if target == codes.NOSE_BASE:
        return criteria.nose_grade(float(value), sex)
    raise ValueError(f"unknown target {target!r}")


def graded(target: str, sexes: Sequence[str]) -> np.ndarray:
    """Which rows get a grade: women only for the jaw, everyone for the alar base."""
    s = np.asarray(sexes)
    return s == codes.FEMALE if target == codes.JAW else np.ones(len(s), dtype=bool)


def grades(target: str, values, sexes) -> list[str | None]:
    return [grade(target, v, s) for v, s in zip(_floats(values), sexes, strict=True)]


def confusion(labels: Sequence[str], preds: Sequence[str], categories: Sequence[str] = GRADES) -> np.ndarray:
    """counts[i, j]: label grade categories[i], predicted grade categories[j]."""
    k = {c: i for i, c in enumerate(categories)}
    out = np.zeros((len(categories), len(categories)), dtype=np.int64)
    for a, b in zip(labels, preds, strict=True):
        out[k[a], k[b]] += 1
    return out


def weighted_kappa(labels: Sequence[str], preds: Sequence[str], categories: Sequence[str] = GRADES) -> float | None:
    """Linear weighted kappa; None when it is undefined (one category only)."""
    m = confusion(labels, preds, categories).astype(np.float64)
    n = m.sum()
    if n == 0:
        return None
    k = len(categories)
    i, j = np.indices((k, k))
    w = 1.0 - np.abs(i - j) / (k - 1)  # agreement weight
    observed = (w * m).sum() / n
    expected = (w * np.outer(m.sum(axis=1), m.sum(axis=0))).sum() / (n * n)
    if expected == 1.0:
        return None
    return float((observed - expected) / (1.0 - expected))


# ---------------------------------------------------------------- bootstrap


def bootstrap_rmse_diff(images: Sequence[str], err_a, err_b, rounds: int = BOOTSTRAP_ROUNDS, seed: int = BOOTSTRAP_SEED,
                        level: float = 0.95) -> dict:
    """RMSE(err_a) - RMSE(err_b) and its percentile interval, resampling images with every copy of each."""
    a, b = _floats(err_a), _floats(err_b)
    keys, inverse = np.unique(np.asarray(images), return_inverse=True)
    sse_a = np.bincount(inverse, weights=a * a, minlength=len(keys))
    sse_b = np.bincount(inverse, weights=b * b, minlength=len(keys))
    count = np.bincount(inverse, minlength=len(keys)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(rounds)
    for r in range(rounds):
        pick = rng.integers(0, len(keys), len(keys))
        n = count[pick].sum()
        draws[r] = np.sqrt(sse_a[pick].sum() / n) - np.sqrt(sse_b[pick].sum() / n)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    diff = float(np.sqrt(np.mean(a * a)) - np.sqrt(np.mean(b * b)))
    return {"diff": diff, "low": float(low), "high": float(high), "rounds": rounds, "seed": seed, "images": len(keys)}


# ---------------------------------------------------------------- band


def band_from_rmse(rmse: float) -> float:
    """Borderline band before the cap: RMSE_BAND_FACTOR x RMSE (model.md section 4.1)."""
    return criteria.RMSE_BAND_FACTOR * float(rmse)


def band_caps(target: str, band: float) -> dict[str, dict]:
    """The band at each cutoff after criteria.capped_bands, per sex that has a scale."""
    out = {}
    for sex in codes.SEXES:
        s = criteria.scale(MEASURE[target], sex)
        if s is None:
            continue
        capped = criteria.capped_bands(s.cutoffs, band)
        out[sex] = {"cutoffs": list(s.cutoffs), "bands": list(capped), "capped": [c < band - criteria.TOLERANCE for c in capped]}
    return out


def band_outcomes(target: str, values, labels, sexes, band: float) -> dict:
    """How the band does on rows that get a grade: borderline share, clear hits and borderline cover."""
    v, y = _floats(values), _floats(labels)
    clear = hit = border = cover = 0
    for value, label, sex in zip(v, y, sexes, strict=True):
        d = criteria.decide(MEASURE[target], float(value), sex, band)
        if d is None:
            continue
        truth = grade(target, label, sex)
        if d.grade is not None:
            clear += 1
            hit += d.grade == truth
        else:
            border += 1
            cover += truth in d.grades()
    n = clear + border
    return {
        "n": n,
        "borderline": border / n if n else None,
        "clear_match": hit / clear if clear else None,
        "borderline_cover": cover / border if border else None,
        "clear_n": clear,
        "borderline_n": border,
    }
