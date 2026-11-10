"""Recommendation rules (model.md sections 3, 4.3 and 1.3): codes only, the texts are in shared/i18n/.

For an assessed region the rules give the procedures, why procedures are not shown,
and the notes. A grade range uses the procedures of every grade in it, once each, in
the order of the tables below. Procedure notes go with the procedures and are left out
when no procedure is shown; undereye_scope comes with every assessed under-eye side.

Why procedures are not shown, every reason that applies, in codes.WITHHELD order:
male_jaw_no_criterion for the jaw of men, distance_unknown for the alar base of an
uploaded frontal image, age_under_18 for every region under criteria.MIN_PROCEDURE_AGE,
age_21_or_under for the under-eye from that age up to criteria.UNDEREYE_WITHHELD_MAX_AGE,
and no_procedure_at_grade when no other reason applies and the grades have no procedure.
"""

from dataclasses import dataclass

from facemeasure import codes, criteria
from facemeasure.criteria import Decision

# (region, grade) -> procedures, least invasive first (model.md sections 3.1-3.4)
PROCEDURES: dict[tuple[str, str], tuple[str, ...]] = {
    (codes.JAW, "2+"): (codes.BOTOX_MASSETER, codes.JAW_REDUCTION),
    (codes.CHIN, "1"): (codes.CHIN_FILLER,),
    (codes.CHIN, "2"): (codes.CHIN_FILLER, codes.CHIN_IMPLANT, codes.GENIOPLASTY),
    (codes.CHIN, "3"): (codes.GENIOPLASTY,),
    (codes.UNDEREYE_LEFT, "2+"): (codes.UNDEREYE_FILLER,),
    (codes.UNDEREYE_RIGHT, "2+"): (codes.UNDEREYE_FILLER,),
    (codes.NOSE_BASE, "2+"): (codes.ALAR_REDUCTION,),
    (codes.NOSE_DORSUM, "2+"): (codes.NOSE_FILLER, codes.RHINOPLASTY_IMPLANT),
}
# (region, grade) -> notes shown only with the procedures
PROCEDURE_NOTES: dict[tuple[str, str], tuple[str, ...]] = {
    (codes.JAW, "2+"): (codes.JAW_CAUSE_UNKNOWN,),
    (codes.CHIN, "2"): (codes.CHIN_CHECK_OCCLUSION,),
    (codes.CHIN, "3"): (codes.CHIN_CHECK_MANDIBLE,),
    (codes.NOSE_BASE, "2+"): (codes.NOSE_BASE_INDICATIONS,),
    (codes.NOSE_DORSUM, "2+"): (codes.NOSE_FILLER_VISION_RISK,),
}
# region -> notes of every assessed result
REGION_NOTES: dict[str, tuple[str, ...]] = {
    codes.UNDEREYE_LEFT: (codes.UNDEREYE_SCOPE,),
    codes.UNDEREYE_RIGHT: (codes.UNDEREYE_SCOPE,),
}


@dataclass(frozen=True)
class Advice:
    procedures: tuple[str, ...]
    withheld: tuple[str, ...]
    notes: tuple[str, ...]


def _unique(groups) -> tuple[str, ...]:
    out: list[str] = []
    for group in groups:
        out.extend(code for code in group if code not in out)
    return tuple(out)


def withheld_reasons(region: str, age: int, sex: str, frontal_source: str) -> tuple[str, ...]:
    """Reasons other than the grade, in codes.WITHHELD order."""
    reasons = []
    if region == codes.JAW and sex == codes.MALE:
        reasons.append(codes.MALE_JAW_NO_CRITERION)
    if region == codes.NOSE_BASE and frontal_source == codes.UPLOAD:
        reasons.append(codes.DISTANCE_UNKNOWN)
    if age < criteria.MIN_PROCEDURE_AGE:
        reasons.append(codes.AGE_UNDER_18)
    elif region in codes.UNDEREYE and age <= criteria.UNDEREYE_WITHHELD_MAX_AGE:
        reasons.append(codes.AGE_21_OR_UNDER)
    return tuple(reasons)


def advise(region: str, decision: Decision | None, age: int, sex: str, frontal_source: str) -> Advice:
    """Procedures, reasons they are not shown, and notes of one assessed region.

    decision is None only for the jaw of men and the alar base of an uploaded image,
    which show their value without a grade.
    """
    if region not in codes.REGIONS:
        raise ValueError(f"unknown region {region!r}")
    if isinstance(age, bool) or not isinstance(age, int) or age < 0:
        raise ValueError(f"age must be a whole number of years >= 0, got {age!r}")
    if sex not in codes.SEXES:
        raise ValueError(f"sex must be one of {', '.join(codes.SEXES)}, got {sex!r}")
    if frontal_source not in codes.FRONTAL_SOURCES:
        raise ValueError(f"frontal_source must be one of {', '.join(codes.FRONTAL_SOURCES)}, got {frontal_source!r}")

    reasons = withheld_reasons(region, age, sex, frontal_source)
    if decision is None and not reasons:
        raise ValueError(f"{region} needs a decision: only the jaw of men and the alar base of an upload have none")
    grades = decision.grades() if decision is not None else ()
    procedures = _unique(PROCEDURES.get((region, g), ()) for g in grades)
    if not reasons and not procedures:
        reasons = (codes.NO_PROCEDURE_AT_GRADE,)
    if reasons:
        procedures = ()
    notes = REGION_NOTES.get(region, ())
    if procedures:
        notes = _unique([notes, *(PROCEDURE_NOTES.get((region, g), ()) for g in grades)])
    return Advice(procedures=procedures, withheld=reasons, notes=notes)
