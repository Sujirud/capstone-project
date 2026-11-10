"""Measures of the under-eye head on val and test (model.md sections 4.2 and 7.1 item 3).

- single_grade: the model's one grade, the number of transitions with P(grade >= k) at
  least 0.5 (P(grade >= 2) is first lowered to P(grade >= 1), as undereye.decide does).
  The temperature does not move a probability across 0.5, so it is the same before and
  after calibration.
- versus_rater: accuracy and linear weighted kappa of grades 0/1/>=2 against one
  rater's grades (sides the rater could not assess left out).
- ece and reliability: expected calibration error of one transition in ECE_BINS bins of
  equal width over [0, 1], comparing the mean probability with the mean soft label.
- decision_outcomes: the decisions of undereye.decide at given thresholds: share
  borderline, share of clear results that match the rater, and share of borderline
  results whose range holds the rater's grade.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from facemeasure import criteria, undereye
from facemeasure.training.undereye_labels import GRADES3, collapse, linear_kappa

ECE_BINS = 10
GRADE_TEXT = dict(zip(GRADES3, criteria.GRADES, strict=True))  # 0 -> "0", 1 -> "1", 2 -> "2+"
CANDIDATE_THRESHOLDS = ((0.2, 0.8), (0.25, 0.75), (0.3, 0.7), (0.35, 0.65), (0.4, 0.6))


def _probs(p) -> np.ndarray:
    a = np.asarray(p, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"probabilities must be (n, 2), got shape {a.shape}")
    return a


def single_grade(p) -> np.ndarray:
    """0, 1 or 2 (>= 2) per row from (n, 2) probabilities: transitions with P >= 0.5."""
    a = _probs(p)
    p1, p2 = a[:, 0], np.minimum(a[:, 1], a[:, 0])
    return (p1 >= 0.5).astype(int) + (p2 >= 0.5).astype(int)


def rater_grades(values) -> np.ndarray:
    """A rater's grade column (0-3, missing where not graded) as 0, 1 or 2, with -1 where the rater gave no grade."""
    s = pd.Series(values, dtype="object")
    missing = s.isna().to_numpy()
    out = np.full(len(s), -1, dtype=int)
    out[~missing] = [collapse(int(v)) for v in s[~missing]]
    return out


def versus_rater(model: Sequence[int], rater: Sequence[int]) -> dict:
    """Accuracy and linear weighted kappa of 0/1/>=2 on rows the rater graded (rater -1 left out)."""
    m, r = np.asarray(model, dtype=int), np.asarray(rater, dtype=int)
    keep = r >= 0
    m, r = m[keep], r[keep]
    n = len(m)
    confusion = np.zeros((3, 3), dtype=int)
    for a, b in zip(r, m, strict=True):
        confusion[a, b] += 1
    kappa = linear_kappa(r.tolist(), m.tolist(), GRADES3) if n else math.nan
    return {
        "n": n,
        "accuracy": float(np.mean(m == r)) if n else None,
        "kappa": None if not math.isfinite(kappa) else kappa,
        "confusion": confusion.tolist(),  # rows: rater grade, columns: model grade
    }


def between_raters(a: Sequence[int], b: Sequence[int]) -> dict:
    """Agreement of two raters (0/1/>=2) on rows both graded."""
    x, y = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    keep = (x >= 0) & (y >= 0)
    x, y = x[keep], y[keep]
    k = linear_kappa(x.tolist(), y.tolist(), GRADES3) if len(x) else math.nan
    return {"n": len(x), "accuracy": float(np.mean(x == y)) if len(x) else None, "kappa": k if math.isfinite(k) else None}


# ---------------------------------------------------------------- calibration


def bin_index(p, bins: int = ECE_BINS) -> np.ndarray:
    """Bin of each probability: [0, 1/bins), ..., [1 - 1/bins, 1]."""
    a = np.asarray(p, dtype=np.float64)
    return np.minimum(np.floor(a * bins).astype(int), bins - 1)


def reliability(p, y, bins: int = ECE_BINS) -> list[dict]:
    """Per bin: edges, rows, mean probability and mean soft label (None when empty)."""
    a, t = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    k = bin_index(a, bins)
    out = []
    for b in range(bins):
        m = k == b
        out.append({"low": b / bins, "high": (b + 1) / bins, "n": int(m.sum()),
                    "mean_p": float(a[m].mean()) if m.any() else None, "mean_y": float(t[m].mean()) if m.any() else None})
    return out


def ece(p, y, bins: int = ECE_BINS) -> float | None:
    """Expected calibration error: sum over bins of (rows / n) |mean p - mean y|; None without rows."""
    a, t = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if a.shape != t.shape:
        raise ValueError(f"probabilities {a.shape} and labels {t.shape} differ")
    n = len(a)
    if n == 0:
        return None
    k = bin_index(a, bins)
    total = 0.0
    for b in range(bins):
        m = k == b
        if m.any():
            total += m.sum() / n * abs(a[m].mean() - t[m].mean())
    return float(total)


def calibration(p, y, bins: int = ECE_BINS) -> dict:
    """ECE and reliability bins of both transitions; p and y are (n, 2)."""
    a, t = _probs(p), _probs(y)
    return {f"k{k}": {"ece": ece(a[:, i], t[:, i], bins), "bins": reliability(a[:, i], t[:, i], bins)}
            for i, k in enumerate((1, 2))}


# ---------------------------------------------------------------- decisions


def decisions(p, low: float, high: float) -> list[criteria.Decision]:
    a = _probs(p)
    return [undereye.decide(float(p1), float(p2), low, high).decision for p1, p2 in a]


def decision_outcomes(decided: Sequence[criteria.Decision], rater: Sequence[int]) -> dict:
    """Borderline share, clear results that match the rater and borderline ranges that hold the rater's grade."""
    clear = hit = border = cover = 0
    for d, g in zip(decided, rater, strict=True):
        if g < 0:
            continue
        truth = GRADE_TEXT[int(g)]
        if d.grade is not None:
            clear += 1
            hit += d.grade == truth
        else:
            border += 1
            cover += truth in d.grades()
    n = clear + border
    return {"n": n, "borderline": border / n if n else None, "clear_match": hit / clear if clear else None,
            "borderline_cover": cover / border if border else None, "clear_n": clear, "borderline_n": border}


def borderline_share(decided: Sequence[criteria.Decision]) -> float | None:
    return sum(d.grade is None for d in decided) / len(decided) if len(decided) else None
