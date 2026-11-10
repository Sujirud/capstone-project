"""Pydantic models of the requests and answers (api.md sections 2.1-2.3).

- Meta: the meta part of POST /analyze. age must be a JSON integer (22.0, "22" and
  true are refused), lab a JSON boolean, lang a language code; a code without a
  language file still gives Thai texts (web.md section 8.4).
- AnalyzeResult: the result of api.md section 2.2. The api checks every result
  against it before answering.
- Health: GET /health. ErrorBody: every error, only the code and request_id.

Every model forbids unknown keys, so a field response.py adds without a change here
fails the check (a 500) instead of reaching the web unnoticed. The codes come from
facemeasure.codes; only the error codes, which are about HTTP, live here.
"""

from typing import Annotated, Literal

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    StringConstraints,
)

from facemeasure import bundle, codes, criteria, response
from facemeasure.i18n import DEFAULT_LANG, LANG_CODE

MAX_AGE = 120  # api.md section 2.1: age is 0-120

# Error codes of api.md section 2.3 and the HTTP status of each
INVALID_REQUEST = "invalid_request"
UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
DECODE_FAILED = "decode_failed"
TOO_LARGE = "too_large"
INTERNAL_ERROR = "internal_error"
ERROR_STATUS = {
    INVALID_REQUEST: 422,
    UNSUPPORTED_MEDIA_TYPE: 415,
    DECODE_FAILED: 400,
    TOO_LARGE: 413,
    INTERNAL_ERROR: 500,
}
ERROR_CODES = tuple(ERROR_STATUS)

Region = Literal[*codes.REGIONS]
Image = Literal[*codes.IMAGES]
StatusCode = Literal[*codes.STATUS_CODES]
Grade = Literal[*dict.fromkeys(criteria.GRADES + criteria.CHIN_GRADES)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Point = tuple[float, float]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- request


class Meta(_Model):
    """meta of POST /analyze, sent as JSON text (api.md section 2.1)."""

    age: Annotated[StrictInt, Field(ge=0, le=MAX_AGE, description="age in years")]
    sex: Literal[*codes.SEXES]
    frontal_source: Literal[*codes.FRONTAL_SOURCES] = codes.CAMERA
    lab: StrictBool = False
    lang: Annotated[StrictStr, StringConstraints(pattern=LANG_CODE.pattern)] = DEFAULT_LANG


# ---------------------------------------------------------------- result


class Versions(_Model):
    api: str
    model_bundle: str


class RetakeItem(_Model):
    image: Image
    code: Literal[*codes.QUALITY_CODES]
    regions: list[Region]
    text: str


class FrontalQuality(_Model):
    """quality.frontal; the keys of response.FRONTAL_QUALITY_DIGITS, null without a face."""

    iod_px: float | None
    yaw_deg: float | None
    pitch_deg: float | None
    roll_deg: float | None
    smile: float | None
    jaw_open: float | None
    blink_left: float | None
    blink_right: float | None
    hair_jaw: float | None
    hair_undereye_left: float | None
    hair_undereye_right: float | None
    failed: list[Literal[*codes.FRONTAL_CODES]]


class ProfileQuality(_Model):
    """quality.profile; the keys of response.PROFILE_QUALITY_FIELDS, null without a profile image."""

    face_found: bool | None
    face_yaw_deg: float | None
    silhouette_found: bool | None
    flipped: bool | None
    failed: list[Literal[*codes.PROFILE_CODES]]


class Quality(_Model):
    frontal: FrontalQuality
    profile: ProfileQuality


class Cutoff(_Model):
    value: float
    band: float


class Measurement(_Model):
    name: Literal[*codes.MEASURES]
    value: float
    unit: Literal[*dict.fromkeys(response.UNITS.values())]
    method: Literal[codes.MODEL, codes.MEASURED]
    landmark_value: float | None
    cutoffs: list[Cutoff]
    reference: str
    source: str


class Overlay(_Model):
    image: Image
    points: dict[str, Point]
    lines: list[tuple[str, str]]


class Probabilities(_Model):
    grades: dict[Literal[*criteria.GRADES], Probability]
    at_least: dict[Literal[*criteria.GRADES[1:]], Probability]


class Decision(_Model):
    grade: Grade | None
    grade_range: tuple[Grade, Grade] | None
    level: Literal[codes.CLEAR, codes.BORDERLINE]


class Procedure(_Model):
    code: Literal[*codes.PROCEDURES]
    name: str
    when: str
    checks: str | None


class Note(_Model):
    code: Literal[*codes.NOTES]
    text: str
    with_procedures: bool


class RegionResult(_Model):
    region: Region
    status: Literal[codes.ASSESSED, codes.NOT_ASSESSABLE]
    status_reasons: list[StatusCode]
    warnings: list[StatusCode]
    measurements: list[Measurement]
    overlay: Overlay | None
    probabilities: Probabilities | None
    decision: Decision | None
    procedures: list[Procedure]
    procedures_withheld: list[Literal[*codes.WITHHELD]]
    notes: list[Note]


class AnalyzeResult(_Model):
    """The result of POST /analyze (api.md section 2.2)."""

    request_id: UUID4
    versions: Versions
    lab: bool
    retake: list[RetakeItem]
    quality: Quality
    regions: list[RegionResult]  # all six from response.build; the example of api.md shows two
    disclaimer: str


# ---------------------------------------------------------------- health and errors


class HealthVersions(_Model):
    api: str
    model_bundle: str
    facemeasure: str
    mediapipe: str
    torch: str


class Health(_Model):
    """GET /health."""

    status: Literal["ok"]
    versions: HealthVersions
    stub_heads: list[Literal[*bundle.HEADS]]
    device: str
    languages: list[str]


class ErrorBody(_Model):
    """Body of every error: nothing from the request, the images or the measurements."""

    error: Literal[*ERROR_CODES]
    request_id: UUID4
