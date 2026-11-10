"""Recommendation rules (model.md sections 3, 4.3 and 1.3) over every age group, sex, source, grade and range."""

import itertools

import pytest

from facemeasure import codes, rules
from facemeasure.criteria import CHIN_GRADES, GRADES, Decision

AGES = (17, 18, 21, 22)
SEXES = ("female", "male")
SOURCES = ("camera", "upload")


def decisions(order):
    """Every grade, and every range from a lower to a higher grade."""
    out = [Decision(g, None, order) for g in order]
    out += [Decision(None, (a, b), order) for a, b in itertools.combinations(order, 2)]
    return out


# What an adult woman with a camera image gets for each grade (the tables of model.md section 3)
PROCEDURES_AT = {
    "jaw": {"2+": ("botox_masseter", "jaw_reduction")},
    "chin": {"1": ("chin_filler",), "2": ("chin_filler", "chin_implant", "genioplasty"), "3": ("genioplasty",)},
    "undereye_left": {"2+": ("undereye_filler",)},
    "undereye_right": {"2+": ("undereye_filler",)},
    "nose_base": {"2+": ("alar_reduction",)},
    "nose_dorsum": {"2+": ("nose_filler", "rhinoplasty_implant")},
}
NOTES_AT = {
    "jaw": {"2+": ("jaw_cause_unknown",)},
    "chin": {"2": ("chin_check_occlusion",), "3": ("chin_check_mandible",)},
    "nose_base": {"2+": ("nose_base_indications",)},
    "nose_dorsum": {"2+": ("nose_filler_vision_risk",)},
}
ORDER = list(codes.PROCEDURES)


def expected_reasons(region, age, sex, source):
    out = []
    if region == "jaw" and sex == "male":
        out.append("male_jaw_no_criterion")
    if region == "nose_base" and source == "upload":
        out.append("distance_unknown")
    if age < 18:
        out.append("age_under_18")
    elif region.startswith("undereye") and age <= 21:
        out.append("age_21_or_under")
    return out


CASES = [
    (region, decision, age, sex, source)
    for region in codes.REGIONS
    for decision in decisions(CHIN_GRADES if region == "chin" else GRADES)
    for age in AGES
    for sex in SEXES
    for source in SOURCES
]


@pytest.mark.parametrize("region, decision, age, sex, source", CASES)
def test_rules(region, decision, age, sex, source):
    if region == "jaw" and sex == "male" or region == "nose_base" and source == "upload":
        decision = None  # these show the value without a grade
    a = rules.advise(region, decision, age, sex, source)
    grades = decision.grades() if decision is not None else ()
    procedures = sorted({p for g in grades for p in PROCEDURES_AT[region].get(g, ())}, key=ORDER.index)
    reasons = expected_reasons(region, age, sex, source)
    if not reasons and not procedures:
        reasons = ["no_procedure_at_grade"]
    assert list(a.withheld) == reasons
    shown = [] if reasons else procedures
    assert list(a.procedures) == shown
    region_notes = ["undereye_scope"] if region.startswith("undereye") else []
    tied = [n for g in grades for n in NOTES_AT.get(region, {}).get(g, ())] if shown else []
    assert list(a.notes) == region_notes + tied


def test_procedures_follow_the_table_order():
    a = rules.advise("chin", Decision(None, ("1", "3"), CHIN_GRADES), 30, "female", "camera")
    assert a.procedures == ("chin_filler", "chin_implant", "genioplasty")
    assert a.notes == ("chin_check_occlusion", "chin_check_mandible")


def test_adult_examples():
    assert rules.advise("nose_dorsum", Decision("2+", None), 22, "female", "camera") == rules.Advice(
        ("nose_filler", "rhinoplasty_implant"), (), ("nose_filler_vision_risk",)
    )
    assert rules.advise("jaw", Decision(None, ("0", "1")), 22, "female", "camera") == rules.Advice(
        (), ("no_procedure_at_grade",), ()
    )
    assert rules.advise("undereye_left", Decision("2+", None), 21, "female", "camera") == rules.Advice(
        (), ("age_21_or_under",), ("undereye_scope",)
    )
    assert rules.advise("jaw", None, 17, "male", "upload") == rules.Advice(
        (), ("male_jaw_no_criterion", "age_under_18"), ()
    )


def test_every_code_of_the_rules_is_known():
    assert set(itertools.chain.from_iterable(rules.PROCEDURES.values())) == set(codes.PROCEDURES)
    notes = set(itertools.chain.from_iterable(rules.PROCEDURE_NOTES.values()))
    assert notes == set(codes.NOTES_WITH_PROCEDURES)
    assert set(itertools.chain.from_iterable(rules.REGION_NOTES.values())) == {"undereye_scope"}


@pytest.mark.parametrize(
    "args",
    [
        ("forehead", Decision("0", None), 30, "female", "camera"),
        ("jaw", Decision("0", None), -1, "female", "camera"),
        ("jaw", Decision("0", None), 30.5, "female", "camera"),
        ("jaw", Decision("0", None), 30, "other", "camera"),
        ("jaw", Decision("0", None), 30, "female", "web"),
        ("chin", None, 30, "female", "camera"),  # only the male jaw and the uploaded alar base lack a grade
    ],
)
def test_bad_input_raises(args):
    with pytest.raises(ValueError):
        rules.advise(*args)
