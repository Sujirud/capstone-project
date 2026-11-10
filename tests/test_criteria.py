"""Grade cutoffs of the jaw, chin and nose (model.md sections 3.1, 3.2 and 3.4)."""

import math

import pytest

from facemeasure import criteria


@pytest.mark.parametrize(
    "value, grade",
    [(0.60, "0"), (0.83, "0"), (0.8301, "1"), (0.85, "1"), (0.87, "1"), (0.8701, "2+"), (1.2, "2+")],
)
def test_jaw_grade_women(value, grade):
    assert criteria.jaw_grade(value, "female") == grade


def test_jaw_grade_men_is_none():
    assert criteria.jaw_grade(0.95, "male") is None


@pytest.mark.parametrize(
    "sex, value, grade",
    [
        ("female", 0.50, "1"),  # narrow side tops out at grade 1
        ("female", 0.9499, "1"),
        ("female", 0.95, "0"),
        ("female", 1.15, "0"),
        ("female", 1.1501, "1"),
        ("female", 1.25, "1"),
        ("female", 1.2501, "2+"),
        ("male", 0.9899, "1"),
        ("male", 0.99, "0"),
        ("male", 1.21, "0"),
        ("male", 1.2101, "1"),
        ("male", 1.32, "1"),
        ("male", 1.3201, "2+"),
    ],
)
def test_nose_grade(sex, value, grade):
    assert criteria.nose_grade(value, sex) == grade


@pytest.mark.parametrize("sex", criteria.SEXES)
def test_nose_cutoffs_are_mean_and_sd_of_the_preference(sex):
    mean, sd = criteria.NOSE_PREFERENCE[sex]
    c = criteria.NOSE_CUTOFFS[sex]
    assert (c.low, c.high, c.wide) == pytest.approx((mean - sd, mean + sd, mean + 2 * sd), abs=1e-9)


def test_values_are_proportions():
    assert all(0 < c < 1 for c in criteria.JAW_FEMALE_CUTOFFS)
    assert criteria.GRADES == ("0", "1", "2+")


@pytest.mark.parametrize("call", [criteria.jaw_grade, criteria.nose_grade])
def test_bad_input_raises(call):
    with pytest.raises(ValueError, match="sex"):
        call(1.0, "F")
    with pytest.raises(ValueError, match="finite"):
        call(math.nan, "female")


def test_legan_cutoffs():
    # 12 +/- 4 degrees, then steps of 4 (model.md section 3.2)
    assert criteria.LEGAN_CUTOFFS == (16.0, 20.0, 24.0)


@pytest.mark.parametrize("sex", ["female", "male"])
def test_dorsum_cutoffs_follow_the_preference(sex):
    mean, sd = criteria.DORSUM_PREFERENCE[sex]
    c = criteria.DORSUM_CUTOFFS[sex]
    assert c.small == pytest.approx(round(mean - 2 * sd, 1))
    assert c.low == pytest.approx(round(mean - sd, 1))
    assert c.high == pytest.approx(round(mean + sd, 1))
    assert c.small < c.low < c.high


# ---------------------------------------------------------------- one grade per value


@pytest.mark.parametrize(
    "value, grade",
    [(-5.0, "0"), (16.0, "0"), (16.01, "1"), (20.0, "1"), (20.01, "2"), (24.0, "2"), (24.01, "3"), (40.0, "3")],
)
def test_chin_grade(value, grade):
    assert criteria.chin_grade(value) == grade


@pytest.mark.parametrize(
    "sex, value, grade",
    [
        ("female", 30.09, "2+"),
        ("female", 30.1, "1"),  # 30.1 <= v < 32.8 -> 1
        ("female", 32.79, "1"),
        ("female", 32.8, "0"),
        ("female", 38.4, "0"),
        ("female", 38.41, "1"),  # a nose that sticks out a lot tops out at 1
        ("female", 60.0, "1"),
        ("male", 31.19, "2+"),
        ("male", 31.2, "1"),
        ("male", 34.2, "0"),
        ("male", 40.2, "0"),
        ("male", 40.21, "1"),
    ],
)
def test_dorsum_grade(sex, value, grade):
    assert criteria.dorsum_grade(value, sex) == grade


@pytest.mark.parametrize(
    "measure, sex, cutoffs",
    [
        ("facial_taperness", "female", (0.83, 0.87)),
        ("legan_angle", "female", (16.0, 20.0, 24.0)),
        ("legan_angle", "male", (16.0, 20.0, 24.0)),
        ("alar_intercanthal_ratio", "female", (0.95, 1.15, 1.25)),
        ("alar_intercanthal_ratio", "male", (0.99, 1.21, 1.32)),
        ("nasofacial_angle", "female", (30.1, 32.8, 38.4)),
        ("nasofacial_angle", "male", (31.2, 34.2, 40.2)),
    ],
)
def test_scales_hold_the_table(measure, sex, cutoffs):
    assert criteria.scale(measure, sex).cutoffs == cutoffs


def test_legan_constants_agree():
    low, high = criteria.LEGAN_ACCEPTABLE
    assert (low, high) == (criteria.LEGAN_IDEAL - criteria.LEGAN_STEP, criteria.LEGAN_IDEAL + criteria.LEGAN_STEP)
    assert criteria.LEGAN_CUTOFFS == tuple(high + k * criteria.LEGAN_STEP for k in range(3))


def test_men_have_no_jaw_scale_or_decision():
    assert criteria.scale("facial_taperness", "male") is None
    assert criteria.decide("facial_taperness", 0.95, "male") is None


# ---------------------------------------------------------------- borderline bands


def test_default_bands_of_model_md():
    assert criteria.DEFAULT_BANDS == {
        "facial_taperness": 0.01,
        "legan_angle": 1.5,
        "alar_intercanthal_ratio": 0.04,
        "nasofacial_angle": 1.5,
    }


@pytest.mark.parametrize(
    "sex, band, expected",
    [("female", 1.5, (1.35, 1.35, 1.5)), ("male", 1.5, (1.5, 1.5, 1.5))],
)
def test_nasofacial_bands_are_capped(sex, band, expected):
    got = criteria.capped_bands(criteria.scale("nasofacial_angle", sex).cutoffs, band)
    assert got == pytest.approx(expected)


def test_alar_bands_from_a_large_rmse_are_capped():
    got = criteria.capped_bands(criteria.scale("alar_intercanthal_ratio", "female").cutoffs, 0.12)
    assert got == pytest.approx((0.10, 0.05, 0.05))


def test_rmse_band_factor_gives_probability_0_3_to_0_7():
    from scipy.stats import norm

    assert norm.cdf(criteria.RMSE_BAND_FACTOR) == pytest.approx(0.7, abs=1e-4)


def d(grade=None, grade_range=None, order=criteria.GRADES):
    return criteria.Decision(grade, grade_range, order)


@pytest.mark.parametrize(
    "measure, sex, value, band, expected",
    [
        # model.md section 4.1: 83.6% is within 1 point of 83%, 85.0% is not near either cutoff
        ("facial_taperness", "female", 0.836, None, d(grade_range=("0", "1"))),
        ("facial_taperness", "female", 0.850, None, d("1")),
        ("facial_taperness", "female", 0.82, None, d(grade_range=("0", "1"))),  # exactly one band away
        ("facial_taperness", "female", 0.8199, None, d("0")),
        ("facial_taperness", "female", 0.88, None, d(grade_range=("1", "2+"))),
        ("facial_taperness", "female", 0.95, None, d("2+")),
        ("facial_taperness", "female", 0.836, 0.005, d("1")),  # the bundle's band replaces the default
        ("legan_angle", "female", 12.0, None, d("0", order=criteria.CHIN_GRADES)),
        ("legan_angle", "male", 17.0, None, d(grade_range=("0", "1"), order=criteria.CHIN_GRADES)),
        ("legan_angle", "female", 21.4, None, d(grade_range=("1", "2"), order=criteria.CHIN_GRADES)),
        ("legan_angle", "female", 22.0, None, d("2", order=criteria.CHIN_GRADES)),
        ("legan_angle", "female", 24.0, None, d(grade_range=("2", "3"), order=criteria.CHIN_GRADES)),
        ("legan_angle", "female", 30.0, None, d("3", order=criteria.CHIN_GRADES)),
        ("alar_intercanthal_ratio", "female", 0.93, None, d(grade_range=("0", "1"))),  # narrow side
        ("alar_intercanthal_ratio", "female", 1.05, None, d("0")),
        ("alar_intercanthal_ratio", "female", 1.20, None, d("1")),
        ("alar_intercanthal_ratio", "female", 1.20, 0.12, d(grade_range=("0", "2+"))),  # within 0.05 of 1.15 and 1.25
        ("alar_intercanthal_ratio", "female", 1.29, None, d(grade_range=("1", "2+"))),
        ("alar_intercanthal_ratio", "female", 1.30, None, d("2+")),
        ("alar_intercanthal_ratio", "male", 1.30, None, d(grade_range=("1", "2+"))),
        ("nasofacial_angle", "female", 28.5, None, d("2+")),  # the example of api.md section 2.2
        ("nasofacial_angle", "female", 28.75, None, d(grade_range=("1", "2+"))),  # 30.1 - 1.35
        ("nasofacial_angle", "female", 31.45, None, d(grade_range=("0", "2+"))),  # midway, within both bands
        ("nasofacial_angle", "female", 35.6, None, d("0")),
        ("nasofacial_angle", "female", 39.0, None, d(grade_range=("0", "1"))),
        ("nasofacial_angle", "male", 29.7, None, d(grade_range=("1", "2+"))),  # men keep the full 1.5
        ("nasofacial_angle", "male", 29.69, None, d("2+")),
    ],
)
def test_decide(measure, sex, value, band, expected):
    got = criteria.decide(measure, value, sex, band)
    assert got == expected
    assert got.level == ("clear" if expected.grade is not None else "borderline")


def test_decision_lists_every_grade_in_the_range():
    assert d(grade_range=("0", "2+")).grades() == ("0", "1", "2+")
    assert d(grade_range=("1", "3"), order=criteria.CHIN_GRADES).grades() == ("1", "2", "3")
    assert d("1").grades() == ("1",)
    assert d(grade_range=("0", "1")).as_dict() == {"grade": None, "grade_range": ["0", "1"], "level": "borderline"}
    assert d("2+").as_dict() == {"grade": "2+", "grade_range": None, "level": "clear"}


@pytest.mark.parametrize(
    "grade, grade_range",
    [(None, None), ("0", ("0", "1")), ("3", None), (None, ("1", "0")), (None, ("1", "1"))],
)
def test_bad_decisions_raise(grade, grade_range):
    with pytest.raises(ValueError):
        criteria.Decision(grade, grade_range)


def test_ages_of_model_md():
    assert (criteria.MIN_PROCEDURE_AGE, criteria.UNDEREYE_WITHHELD_MAX_AGE) == (18, 21)
    assert criteria.UNDEREYE_THRESHOLDS == (0.3, 0.7)
