"""Grade cutoffs, borderline bands and the other criteria of model.md sections 1.3, 3 and 4, kept in one place.

Ratios are proportions, not percent, and angles are degrees; the jaw ratio becomes
percent only in the JSON (response.py). Grades are the strings "0", "1" and "2+" of
api.md section 2.2, and "0" to "3" for the chin.

Two kinds of grading:

- jaw_grade, nose_grade, chin_grade and dorsum_grade give one grade with no
  borderline band, for training reports and for checks.
- decide() gives the result of model.md section 4.1: a value within the band of a
  cutoff gets the range of the grades on both sides of it, and a value near two
  cutoffs gets every grade involved. Each cutoff's band is at most half the distance
  to the next cutoff on either side, so bands of neighbours never overlap.

The bands of the jaw and alar base ratios come from the bundle once train_ratio_heads.py has set
them from the RMSE of the heads (RMSE_BAND_FACTOR x RMSE); until then, and for the
angles, DEFAULT_BANDS applies.
"""

import math
import numbers
from dataclasses import dataclass

from facemeasure import codes

GRADES = ("0", "1", "2+")
CHIN_GRADES = ("0", "1", "2", "3")
SEXES = codes.SEXES

# Where each criterion comes from (reference numbers of docs/)
SOURCES = {
    codes.FACIAL_TAPERNESS: "[3]",
    codes.LEGAN_ANGLE: "[5][26]",
    codes.ALAR_INTERCANTHAL_RATIO: "[39]",
    codes.NASOFACIAL_ANGLE: "[39]",
}

# facial_taperness = Go-Go / Zy-Zy, women only [3]: v <= 0.83 -> 0, 0.83 < v <= 0.87 -> 1, v > 0.87 -> 2+
JAW_FEMALE_CUTOFFS = (0.83, 0.87)


@dataclass(frozen=True)
class NoseCutoffs:
    """alar_intercanthal_ratio cutoffs: grade 0 on [low, high], 2+ above wide, 1 otherwise.

    The narrow side has no procedure, so values below low get at most grade 1.
    """

    low: float  # mean - 1 SD
    high: float  # mean + 1 SD
    wide: float  # mean + 2 SD


# Preferred alar base / intercanthal ratio of the general group in [39], mean and SD
NOSE_PREFERENCE = {"female": (1.05, 0.10), "male": (1.10, 0.11)}
# The same values as mean ± SD and mean + 2 SD, written out as in the table of section 3.4
NOSE_CUTOFFS = {
    "female": NoseCutoffs(low=0.95, high=1.15, wide=1.25),
    "male": NoseCutoffs(low=0.99, high=1.21, wide=1.32),
}

# legan_angle, chin [5][26]: ideal 12 degrees, acceptable 8-16, grades in steps of 4 (model.md section 3.2)
LEGAN_IDEAL = 12.0
LEGAN_ACCEPTABLE = (8.0, 16.0)
LEGAN_STEP = 4.0
# v <= 16 -> 0, 16 < v <= 20 -> 1, 20 < v <= 24 -> 2, v > 24 -> 3
LEGAN_CUTOFFS = (16.0, 20.0, 24.0)


@dataclass(frozen=True)
class DorsumCutoffs:
    """nasofacial_angle cutoffs: grade 0 on [low, high], 2+ below small, 1 otherwise.

    A nose that sticks out a lot (above high) has no procedure, so it gets at most grade 1.
    """

    small: float  # mean - 2 SD
    low: float  # mean - 1 SD
    high: float  # mean + 1 SD


# Preferred nasofacial angle of the general group in [39], mean and SD in degrees
DORSUM_PREFERENCE = {"female": (35.60, 2.76), "male": (37.18, 2.99)}
# The same values rounded as in the table of model.md section 3.4
DORSUM_CUTOFFS = {
    "female": DorsumCutoffs(small=30.1, low=32.8, high=38.4),
    "male": DorsumCutoffs(small=31.2, low=34.2, high=40.2),
}

# Borderline band before training (model.md section 4.1), in the unit of the stored value
DEFAULT_BANDS = {
    codes.FACIAL_TAPERNESS: 0.01,
    codes.LEGAN_ANGLE: 1.5,
    codes.ALAR_INTERCANTHAL_RATIO: 0.04,
    codes.NASOFACIAL_ANGLE: 1.5,
}
# A normal error is beyond a cutoff with probability 0.3 to 0.7 within 0.5244 SD of it
RMSE_BAND_FACTOR = 0.5244
# "Within the band" allows this much floating-point error, so 28.75 is within 1.35 of 30.1
TOLERANCE = 1e-9

# Under-eye: P(grade >= k) at or above high reaches grade k, at or below low does not,
# in between is borderline (model.md section 4.2). The bundle keeps the values in use.
UNDEREYE_THRESHOLDS = (0.3, 0.7)

# Ages (model.md section 1.3): no procedures under MIN_PROCEDURE_AGE, and no under-eye
# procedures up to UNDEREYE_WITHHELD_MAX_AGE
MIN_PROCEDURE_AGE = 18
UNDEREYE_WITHHELD_MAX_AGE = 21


@dataclass(frozen=True)
class Scale:
    """Grades along the value axis.

    grades[i] holds below cutoffs[i] and grades[-1] above the last cutoff. lower[i] is
    True when a value equal to cutoffs[i] takes the grade below it. order is the grade
    order of the region, which sorts the grades of a range.
    """

    cutoffs: tuple[float, ...]
    grades: tuple[str, ...]
    lower: tuple[bool, ...]
    order: tuple[str, ...] = GRADES

    def __post_init__(self):
        n = len(self.cutoffs)
        if n == 0 or len(self.grades) != n + 1 or len(self.lower) != n:
            raise ValueError("a scale needs n cutoffs, n + 1 grades and n lower flags")
        if any(b <= a for a, b in zip(self.cutoffs, self.cutoffs[1:], strict=False)):
            raise ValueError("cutoffs must increase")
        if not set(self.grades) <= set(self.order):
            raise ValueError(f"grades {self.grades} are not all in {self.order}")
        if any(a == b for a, b in zip(self.grades, self.grades[1:], strict=False)):
            raise ValueError("neighbouring grades must differ, or the cutoff between them does nothing")

    def grade(self, value: float) -> str:
        _finite(value)
        for c, g, low in zip(self.cutoffs, self.grades, self.lower, strict=False):
            if value < c or (value == c and low):
                return g
        return self.grades[-1]


JAW_FEMALE_SCALE = Scale(JAW_FEMALE_CUTOFFS, ("0", "1", "2+"), (True, True))
LEGAN_SCALE = Scale(LEGAN_CUTOFFS, CHIN_GRADES, (True, True, True), CHIN_GRADES)
# below low -> 1, low..high -> 0, up to wide -> 1, above wide -> 2+
NOSE_SCALES = {s: Scale((c.low, c.high, c.wide), ("1", "0", "1", "2+"), (False, True, True)) for s, c in NOSE_CUTOFFS.items()}
# below small -> 2+, small up to low -> 1, low..high -> 0, above high -> 1
DORSUM_SCALES = {s: Scale((c.small, c.low, c.high), ("2+", "1", "0", "1"), (False, False, True)) for s, c in DORSUM_CUTOFFS.items()}


def _finite(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"value must be a finite number, got {value!r}")


def _check_sex(sex: str) -> None:
    if sex not in SEXES:
        raise ValueError(f"sex must be one of {', '.join(SEXES)}, got {sex!r}")


def _check(value: float, sex: str) -> None:
    _check_sex(sex)
    _finite(value)


def scale(measure: str, sex: str) -> Scale | None:
    """The scale of a measurement for one sex; None for the jaw of men, who have no criterion."""
    _check_sex(sex)
    if measure == codes.FACIAL_TAPERNESS:
        return JAW_FEMALE_SCALE if sex == codes.FEMALE else None
    if measure == codes.LEGAN_ANGLE:
        return LEGAN_SCALE
    if measure == codes.ALAR_INTERCANTHAL_RATIO:
        return NOSE_SCALES[sex]
    if measure == codes.NASOFACIAL_ANGLE:
        return DORSUM_SCALES[sex]
    raise ValueError(f"unknown measurement {measure!r}")


def capped_bands(cutoffs: tuple[float, ...], band: float) -> tuple[float, ...]:
    """The band of each cutoff: band, but at most half the distance to each neighbouring cutoff."""
    _finite(band)
    if band < 0:
        raise ValueError(f"band must be >= 0, got {band!r}")
    out = []
    for i, c in enumerate(cutoffs):
        b = band
        if i > 0:
            b = min(b, (c - cutoffs[i - 1]) / 2)
        if i + 1 < len(cutoffs):
            b = min(b, (cutoffs[i + 1] - c) / 2)
        out.append(b)
    return tuple(out)


@dataclass(frozen=True)
class Decision:
    """decision of api.md section 2.2: one grade (clear) or the lowest and highest grade of a range (borderline)."""

    grade: str | None
    grade_range: tuple[str, str] | None
    order: tuple[str, ...] = GRADES

    def __post_init__(self):
        if (self.grade is None) == (self.grade_range is None):
            raise ValueError("a decision has either a grade or a grade range")
        for g in (self.grade,) if self.grade is not None else tuple(self.grade_range):
            if g not in self.order:
                raise ValueError(f"grade {g!r} is not in {self.order}")
        if self.grade_range is not None and self.order.index(self.grade_range[0]) >= self.order.index(self.grade_range[1]):
            raise ValueError(f"grade range {self.grade_range} must go from a lower to a higher grade")

    @property
    def level(self) -> str:
        return codes.CLEAR if self.grade is not None else codes.BORDERLINE

    def grades(self) -> tuple[str, ...]:
        """Every grade of the decision, lowest first."""
        if self.grade is not None:
            return (self.grade,)
        low, high = (self.order.index(g) for g in self.grade_range)
        return self.order[low : high + 1]

    def as_dict(self) -> dict:
        return {
            "grade": self.grade,
            "grade_range": list(self.grade_range) if self.grade_range is not None else None,
            "level": self.level,
        }


def decide_on(s: Scale, value: float, band: float) -> Decision:
    """Grade or grade range of a value on a scale, with band around every cutoff (capped)."""
    found = {s.grade(value)}
    for i, (c, b) in enumerate(zip(s.cutoffs, capped_bands(s.cutoffs, band), strict=True)):
        if abs(value - c) <= b + TOLERANCE:
            found.update(s.grades[i : i + 2])
    if len(found) == 1:
        return Decision(found.pop(), None, s.order)
    ranks = sorted(s.order.index(g) for g in found)
    return Decision(None, (s.order[ranks[0]], s.order[ranks[-1]]), s.order)


def decide(measure: str, value: float, sex: str, band: float | None = None) -> Decision | None:
    """model.md section 4.1 for one measurement; None for the jaw of men.

    band is the bundle's band of the jaw or alar base head; None uses DEFAULT_BANDS.
    """
    _check(value, sex)
    s = scale(measure, sex)
    if s is None:
        return None
    return decide_on(s, value, DEFAULT_BANDS[measure] if band is None else band)


def jaw_grade(value: float, sex: str) -> str | None:
    """Jaw grade of a facial taperness value; None for men, who get the ratio but no grade."""
    _check(value, sex)
    s = scale(codes.FACIAL_TAPERNESS, sex)
    return None if s is None else s.grade(value)


def nose_grade(value: float, sex: str) -> str:
    """Alar base grade of an alar base / intercanthal ratio, with the cutoffs of that sex."""
    _check(value, sex)
    return NOSE_SCALES[sex].grade(value)


def chin_grade(value: float) -> str:
    """Chin grade "0" to "3" of a Legan angle in degrees."""
    _finite(value)
    return LEGAN_SCALE.grade(value)


def dorsum_grade(value: float, sex: str) -> str:
    """Nose dorsum grade of a nasofacial angle in degrees, with the cutoffs of that sex."""
    _check(value, sex)
    return DORSUM_SCALES[sex].grade(value)
