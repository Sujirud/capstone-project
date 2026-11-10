"""Under-eye grade from the head's cumulative probabilities (model.md section 4.2).

The head gives P(grade >= 1) and P(grade >= 2) after temperature scaling. Each
transition k is decided on its own with the bundle's thresholds (0.3 and 0.7 by
default, criteria.UNDEREYE_THRESHOLDS): at or above high the grade reaches k, at or
below low it does not, in between it is borderline. The two transitions then give one
grade or a range:

| grade >= 1 | grade >= 2 | result |
| --- | --- | --- |
| not reached | not reached | "0" |
| reached | not reached | "1" |
| reached | reached | "2+" |
| borderline | not reached | "0" to "1" |
| reached | borderline | "1" to "2+" |
| borderline | borderline | "0" to "2+" |

P(grade >= 2) above P(grade >= 1) can only come from numerical error, so it is
lowered to P(grade >= 1) first; the other combinations then cannot occur.
"""

import math
import numbers
from dataclasses import dataclass

from facemeasure import criteria

NOT_REACHED = "not_reached"
BORDERLINE = "borderline"
REACHED = "reached"

_COMBINED = {
    (NOT_REACHED, NOT_REACHED): ("0", None),
    (REACHED, NOT_REACHED): ("1", None),
    (REACHED, REACHED): ("2+", None),
    (BORDERLINE, NOT_REACHED): (None, ("0", "1")),
    (REACHED, BORDERLINE): (None, ("1", "2+")),
    (BORDERLINE, BORDERLINE): (None, ("0", "2+")),
}


@dataclass(frozen=True)
class UndereyeResult:
    at_least_1: float  # P(grade >= 1)
    at_least_2: float  # P(grade >= 2), at most at_least_1
    decision: criteria.Decision

    @property
    def grades(self) -> dict[str, float]:
        """P(grade 0), P(grade 1) and P(grade >= 2)."""
        return {"0": 1.0 - self.at_least_1, "1": self.at_least_1 - self.at_least_2, "2+": self.at_least_2}

    @property
    def at_least(self) -> dict[str, float]:
        return {"1": self.at_least_1, "2+": self.at_least_2}


def _probability(p: float, name: str) -> float:
    if isinstance(p, bool) or not isinstance(p, numbers.Real) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be a probability between 0 and 1, got {p!r}")
    return float(p)


def check_thresholds(low: float, high: float) -> None:
    low, high = _probability(low, "low"), _probability(high, "high")
    if not low < high:
        raise ValueError(f"low threshold {low} must be below high threshold {high}")


def step(p: float, low: float, high: float) -> str:
    """Decision at one transition."""
    if p >= high:
        return REACHED
    if p <= low:
        return NOT_REACHED
    return BORDERLINE


def decide(at_least_1: float, at_least_2: float, low: float | None = None, high: float | None = None) -> UndereyeResult:
    """Grade or grade range from P(grade >= 1) and P(grade >= 2); thresholds default to criteria."""
    default_low, default_high = criteria.UNDEREYE_THRESHOLDS
    low = default_low if low is None else low
    high = default_high if high is None else high
    check_thresholds(low, high)
    p1 = _probability(at_least_1, "P(grade >= 1)")
    p2 = min(_probability(at_least_2, "P(grade >= 2)"), p1)
    grade, grade_range = _COMBINED[(step(p1, low, high), step(p2, low, high))]
    return UndereyeResult(p1, p2, criteria.Decision(grade, grade_range, criteria.GRADES))
