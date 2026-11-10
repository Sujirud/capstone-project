"""Every code of the results (api.md section 2.3) and which code affects which region.

This is the one place for these codes. The texts of the codes are in shared/i18n/.
"""

# Regions, in the order of the results. undereye_left is the person's left eye, on the image right.
JAW = "jaw"
CHIN = "chin"
UNDEREYE_LEFT = "undereye_left"
UNDEREYE_RIGHT = "undereye_right"
NOSE_BASE = "nose_base"
NOSE_DORSUM = "nose_dorsum"
REGIONS = (JAW, CHIN, UNDEREYE_LEFT, UNDEREYE_RIGHT, NOSE_BASE, NOSE_DORSUM)
FRONTAL_REGIONS = (JAW, UNDEREYE_LEFT, UNDEREYE_RIGHT, NOSE_BASE)
PROFILE_REGIONS = (CHIN, NOSE_DORSUM)
UNDEREYE = (UNDEREYE_LEFT, UNDEREYE_RIGHT)

# Images of a request, also the overlay image names
FRONTAL = "frontal"
PROFILE = "profile"
IMAGES = (FRONTAL, PROFILE)

# meta of the request (api.md section 2.1)
FEMALE = "female"
MALE = "male"
SEXES = (FEMALE, MALE)
CAMERA = "camera"
UPLOAD = "upload"
FRONTAL_SOURCES = (CAMERA, UPLOAD)

# Measurements and how each region gets its value
FACIAL_TAPERNESS = "facial_taperness"
LEGAN_ANGLE = "legan_angle"
ALAR_INTERCANTHAL_RATIO = "alar_intercanthal_ratio"
NASOFACIAL_ANGLE = "nasofacial_angle"
MEASURES = (FACIAL_TAPERNESS, LEGAN_ANGLE, ALAR_INTERCANTHAL_RATIO, NASOFACIAL_ANGLE)
MODEL = "model"
MEASURED = "measured"
REGION_MEASURE = {JAW: FACIAL_TAPERNESS, CHIN: LEGAN_ANGLE, NOSE_BASE: ALAR_INTERCANTHAL_RATIO, NOSE_DORSUM: NASOFACIAL_ANGLE}
MEASURE_METHOD = {FACIAL_TAPERNESS: MODEL, LEGAN_ANGLE: MEASURED, ALAR_INTERCANTHAL_RATIO: MODEL, NASOFACIAL_ANGLE: MEASURED}

# Region status and decision level
ASSESSED = "assessed"
NOT_ASSESSABLE = "not_assessable"
CLEAR = "clear"
BORDERLINE = "borderline"

# Quality codes of the frontal image, in the order they are listed in results
NO_FACE = "no_face"
MULTIPLE_FACES = "multiple_faces"
FACE_TOO_SMALL = "face_too_small"
FRONTAL_POSE = "frontal_pose"
SMILING = "smiling"
EYES_CLOSED = "eyes_closed"
HAIR_OCCLUSION = "hair_occlusion"
FRONTAL_CODES = (NO_FACE, MULTIPLE_FACES, FACE_TOO_SMALL, FRONTAL_POSE, SMILING, EYES_CLOSED, HAIR_OCCLUSION)

# Codes of the profile image: not turned far enough or no full face edge, and no profile sent
PROFILE_INVALID = "profile_invalid"
MISSING_INPUT = "missing_input"
PROFILE_CODES = (PROFILE_INVALID, MISSING_INPUT)

QUALITY_CODES = FRONTAL_CODES + PROFILE_CODES

# A head of the bundle is still a stub. Not about the image, so never in retake;
# response.py sets it from the bundle.
MODEL_NOT_READY = "model_not_ready"
STATUS_CODES = QUALITY_CODES + (MODEL_NOT_READY,)

# Procedures, least invasive first within each region (api.md section 2.3)
BOTOX_MASSETER = "botox_masseter"
JAW_REDUCTION = "jaw_reduction"
CHIN_FILLER = "chin_filler"
CHIN_IMPLANT = "chin_implant"
GENIOPLASTY = "genioplasty"
UNDEREYE_FILLER = "undereye_filler"
ALAR_REDUCTION = "alar_reduction"
NOSE_FILLER = "nose_filler"
RHINOPLASTY_IMPLANT = "rhinoplasty_implant"
PROCEDURES = (
    BOTOX_MASSETER, JAW_REDUCTION, CHIN_FILLER, CHIN_IMPLANT, GENIOPLASTY,
    UNDEREYE_FILLER, ALAR_REDUCTION, NOSE_FILLER, RHINOPLASTY_IMPLANT,
)

# Notes. All but undereye_scope belong to procedures and are shown only with them.
JAW_CAUSE_UNKNOWN = "jaw_cause_unknown"
CHIN_CHECK_OCCLUSION = "chin_check_occlusion"
CHIN_CHECK_MANDIBLE = "chin_check_mandible"
UNDEREYE_SCOPE = "undereye_scope"
NOSE_BASE_INDICATIONS = "nose_base_indications"
NOSE_FILLER_VISION_RISK = "nose_filler_vision_risk"
NOTES = (JAW_CAUSE_UNKNOWN, CHIN_CHECK_OCCLUSION, CHIN_CHECK_MANDIBLE, UNDEREYE_SCOPE, NOSE_BASE_INDICATIONS, NOSE_FILLER_VISION_RISK)
NOTES_WITH_PROCEDURES = frozenset(NOTES) - {UNDEREYE_SCOPE}

# Why procedures are not shown, in the order they are listed in results
MALE_JAW_NO_CRITERION = "male_jaw_no_criterion"
DISTANCE_UNKNOWN = "distance_unknown"
AGE_UNDER_18 = "age_under_18"
AGE_21_OR_UNDER = "age_21_or_under"
NO_PROCEDURE_AT_GRADE = "no_procedure_at_grade"
WITHHELD = (MALE_JAW_NO_CRITERION, DISTANCE_UNKNOWN, AGE_UNDER_18, AGE_21_OR_UNDER, NO_PROCEDURE_AT_GRADE)

# Regions each code makes not assessable. hair_occlusion affects only the parts the hair
# covers, given by HAIR_REGIONS: the jaw when hair_jaw is over the limit, and each
# under-eye side whose own value is over the limit.
AFFECTS: dict[str, tuple[str, ...]] = {
    NO_FACE: FRONTAL_REGIONS,
    MULTIPLE_FACES: FRONTAL_REGIONS,
    FACE_TOO_SMALL: FRONTAL_REGIONS,
    FRONTAL_POSE: FRONTAL_REGIONS,
    SMILING: FRONTAL_REGIONS,
    EYES_CLOSED: UNDEREYE,
    HAIR_OCCLUSION: (JAW, UNDEREYE_LEFT, UNDEREYE_RIGHT),
    PROFILE_INVALID: PROFILE_REGIONS,
    MISSING_INPUT: PROFILE_REGIONS,
}
HAIR_REGIONS: dict[str, str] = {
    "hair_jaw": JAW,
    "hair_undereye_left": UNDEREYE_LEFT,
    "hair_undereye_right": UNDEREYE_RIGHT,
}


def by_region(failed: tuple[str, ...], hair_over: tuple[str, ...] = ()) -> dict[str, tuple[str, ...]]:
    """Codes per frontal region, in FRONTAL_CODES order.

    failed are the codes of the image; hair_over the names in HAIR_REGIONS whose value is
    over the limit, which say where hair_occlusion applies.
    """
    unknown = set(failed) - set(FRONTAL_CODES)
    if unknown:
        raise ValueError(f"unknown frontal codes: {', '.join(sorted(unknown))}")
    if HAIR_OCCLUSION in failed and not hair_over:
        raise ValueError("hair_occlusion needs the hair values that are over the limit")
    hair_regions = {HAIR_REGIONS[name] for name in hair_over}
    out: dict[str, list[str]] = {r: [] for r in FRONTAL_REGIONS}
    for code in FRONTAL_CODES:
        if code not in failed:
            continue
        for region in AFFECTS[code]:
            if code != HAIR_OCCLUSION or region in hair_regions:
                out[region].append(code)
    return {r: tuple(c) for r, c in out.items()}


def profile_by_region(failed: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Codes per profile region (chin and nose_dorsum), in PROFILE_CODES order."""
    unknown = set(failed) - set(PROFILE_CODES)
    if unknown:
        raise ValueError(f"unknown profile codes: {', '.join(sorted(unknown))}")
    out: dict[str, list[str]] = {r: [] for r in PROFILE_REGIONS}
    for code in PROFILE_CODES:
        if code in failed:
            for region in AFFECTS[code]:
                out[region].append(code)
    return {r: tuple(c) for r, c in out.items()}
