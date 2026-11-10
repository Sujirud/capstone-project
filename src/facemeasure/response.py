"""The result of POST /analyze (api.md section 2.2), without request_id, versions and lab.

build() takes the result of analyze_frontal (or of the api's entry in model_input,
which has the same type), the result of analyze_profile or None, what the heads
predicted (the api runs them), the request's meta, the bundle and the texts, and
returns quality, retake, regions and disclaimer; the api adds the rest.

Status of a region:

- Normal: a region with a quality code is not_assessable with the codes in
  status_reasons, and each code of an image is one retake item: the image, the
  code, the regions it affects and its text.
- Lab mode (api.md section 2.4): every region that can be measured is assessed with
  its codes in warnings, retake is empty, and a region is not_assessable only when it
  cannot be measured, such as without a face.
- A region whose head in the bundle is a stub (HEAD_OF) gets model_not_ready after its
  quality codes, in status_reasons or in lab mode in warnings, the same way. It is not
  about the image, so it is never a retake item (api.md section 2.3). The chin and the
  nasal dorsum are measured on the edge and have no head.

A not_assessable region has no measurement, overlay, probabilities, decision,
procedures, reasons or notes. The jaw of men and the alar base of an uploaded frontal
image have their measurement and a null decision.

Numbers are rounded only here, and grades are decided before rounding: percent,
degrees and pixels to one decimal; ratios, probabilities, blendshapes and every value
in cutoffs to three, so a band of 1.35 stays 1.35. The jaw ratio becomes percent here.
Numbers inside texts take their shortest form, 83 rather than 83.0.
"""

import math
from dataclasses import dataclass

from facemeasure import codes, criteria, rules, undereye
from facemeasure.bundle import UNDEREYE_HEAD, Bundle
from facemeasure.frontal import FrontalResult
from facemeasure.i18n import DEFAULT_LANG, Texts
from facemeasure.profile import Overlay, ProfileResult, missing_profile

ONE_DIGIT = 1  # percent, degrees and pixels
THREE_DIGITS = 3  # ratios, probabilities, blendshapes and every value in cutoffs

PERCENT = "%"
DEGREES = "deg"
RATIO = "ratio"
UNITS = {
    codes.FACIAL_TAPERNESS: PERCENT,
    codes.LEGAN_ANGLE: DEGREES,
    codes.ALAR_INTERCANTHAL_RATIO: RATIO,
    codes.NASOFACIAL_ANGLE: DEGREES,
}

# quality.frontal and quality.profile of api.md section 2.2, with the digits of each value
FRONTAL_QUALITY_DIGITS = {
    "iod_px": ONE_DIGIT,
    "yaw_deg": ONE_DIGIT,
    "pitch_deg": ONE_DIGIT,
    "roll_deg": ONE_DIGIT,
    "smile": THREE_DIGITS,
    "jaw_open": THREE_DIGITS,
    "blink_left": THREE_DIGITS,
    "blink_right": THREE_DIGITS,
    "hair_jaw": THREE_DIGITS,
    "hair_undereye_left": THREE_DIGITS,
    "hair_undereye_right": THREE_DIGITS,
}
PROFILE_QUALITY_FIELDS = ("face_found", "face_yaw_deg", "silhouette_found", "flipped")

# Overlays of the frontal regions; _l and _r are image sides
FRONTAL_OVERLAY_POINTS = {
    codes.JAW: ("go_l", "go_r", "zy_l", "zy_r"),
    codes.NOSE_BASE: ("al_l", "al_r", "en_l", "en_r"),
}
FRONTAL_OVERLAY_LINES = {
    codes.JAW: (("go_l", "go_r"), ("zy_l", "zy_r")),
    codes.NOSE_BASE: (("al_l", "al_r"), ("en_l", "en_r")),
}
BOX_CORNERS = ("tl", "tr", "br", "bl")  # the order of crops.Crop.corners
BOX_LINES = (("tl", "tr"), ("tr", "br"), ("br", "bl"), ("bl", "tl"))

# The head of the bundle that gives each region its value
HEAD_OF = {
    codes.JAW: codes.JAW,
    codes.NOSE_BASE: codes.NOSE_BASE,
    codes.UNDEREYE_LEFT: UNDEREYE_HEAD,
    codes.UNDEREYE_RIGHT: UNDEREYE_HEAD,
}


class ResponseError(ValueError):
    """The inputs contradict each other, such as a face found with no prediction for it."""


@dataclass(frozen=True)
class Meta:
    """meta of the request (api.md section 2.1), already checked by the api."""

    age: int
    sex: str
    frontal_source: str = codes.CAMERA
    lang: str = DEFAULT_LANG

    def __post_init__(self):
        if isinstance(self.age, bool) or not isinstance(self.age, int) or self.age < 0:
            raise ValueError(f"age must be a whole number of years >= 0, got {self.age!r}")
        if self.sex not in codes.SEXES:
            raise ValueError(f"sex must be one of {', '.join(codes.SEXES)}, got {self.sex!r}")
        if self.frontal_source not in codes.FRONTAL_SOURCES:
            raise ValueError(f"frontal_source must be one of {', '.join(codes.FRONTAL_SOURCES)}, got {self.frontal_source!r}")


@dataclass(frozen=True)
class Predictions:
    """What the heads gave: ratios as proportions, and (P(grade >= 1), P(grade >= 2)) after temperature.

    None where the head was not run, such as without a face.
    """

    jaw: float | None = None
    nose_base: float | None = None
    undereye_left: tuple[float, float] | None = None
    undereye_right: tuple[float, float] | None = None


def number(value: float | None, digits: int) -> float | None:
    """value rounded for the JSON; never -0.0."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ResponseError(f"cannot put {value} in the result")
    return round(value, digits) + 0.0


def short(value: float, digits: int = THREE_DIGITS) -> str:
    """Shortest form of value rounded to digits, for texts: 83, 1.35, 30.1."""
    text = f"{number(value, digits):.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _scale_factor(measure: str) -> float:
    return 100.0 if UNITS[measure] == PERCENT else 1.0


def _digits(measure: str) -> int:
    return THREE_DIGITS if UNITS[measure] == RATIO else ONE_DIGIT


def reference(measure: str, sex: str, band: float, texts: Texts, lang: str) -> str:
    """The text that explains the criterion of a measurement (reference of api.md section 2.2)."""
    if measure == codes.FACIAL_TAPERNESS:
        if criteria.scale(measure, sex) is None:
            return texts.text("reference.facial_taperness.male", lang)
        cuts = criteria.JAW_FEMALE_CUTOFFS
        f = _scale_factor(measure)
        return texts.text(
            "reference.facial_taperness.female", lang,
            cut=short(cuts[0] * f), band=short(criteria.capped_bands(cuts, band)[0] * f),
        )
    if measure == codes.LEGAN_ANGLE:
        low, high = criteria.LEGAN_ACCEPTABLE
        return texts.text(
            "reference.legan_angle", lang,
            ideal=short(criteria.LEGAN_IDEAL), low=short(low), high=short(high), step=short(criteria.LEGAN_STEP),
        )
    sex_text = texts.text(f"sex.{sex}", lang)
    if measure == codes.ALAR_INTERCANTHAL_RATIO:
        c = criteria.NOSE_CUTOFFS[sex]
        return texts.text(
            "reference.alar_intercanthal_ratio", lang, sex=sex_text, low=short(c.low), high=short(c.high), g2=short(c.wide)
        )
    if measure == codes.NASOFACIAL_ANGLE:
        d = criteria.DORSUM_CUTOFFS[sex]
        return texts.text(
            "reference.nasofacial_angle", lang, sex=sex_text, low=short(d.low), high=short(d.high), g2=short(d.small)
        )
    raise ValueError(f"unknown measurement {measure!r}")


def measurement(
    measure: str, value: float, sex: str, band: float, landmark_value: float | None, texts: Texts, lang: str
) -> dict:
    """One item of measurements. value and landmark_value are proportions or degrees, band in the same unit."""
    f, digits = _scale_factor(measure), _digits(measure)
    s = criteria.scale(measure, sex)
    cutoffs = []
    if s is not None:
        for c, b in zip(s.cutoffs, criteria.capped_bands(s.cutoffs, band), strict=True):
            cutoffs.append({"value": number(c * f, THREE_DIGITS), "band": number(b * f, THREE_DIGITS)})
    return {
        "name": measure,
        "value": number(value * f, digits),
        "unit": UNITS[measure],
        "method": codes.MEASURE_METHOD[measure],
        "landmark_value": None if landmark_value is None else number(landmark_value * f, digits),
        "cutoffs": cutoffs,
        "reference": reference(measure, sex, band, texts, lang),
        "source": criteria.SOURCES[measure],
    }


def overlay_dict(overlay: Overlay) -> dict:
    return {
        "image": overlay.image,
        "points": {k: [number(x, ONE_DIGIT), number(y, ONE_DIGIT)] for k, (x, y) in overlay.points.items()},
        "lines": [list(line) for line in overlay.lines],
    }


def _blank_region(region: str, status: str, reasons=(), warnings=()) -> dict:
    return {
        "region": region,
        "status": status,
        "status_reasons": list(reasons),
        "warnings": list(warnings),
        "measurements": [],
        "overlay": None,
        "probabilities": None,
        "decision": None,
        "procedures": [],
        "procedures_withheld": [],
        "notes": [],
    }


def quality_dict(frontal: FrontalResult, profile: ProfileResult) -> dict:
    """quality of api.md section 2.2; values are null when there is no face or no profile image."""
    fq = frontal.quality
    front = {name: None if fq is None else number(getattr(fq, name), d) for name, d in FRONTAL_QUALITY_DIGITS.items()}
    front["failed"] = list(frontal.failed)
    pq = profile.quality
    side = {name: None if pq is None else getattr(pq, name) for name in PROFILE_QUALITY_FIELDS}
    side["face_yaw_deg"] = number(side["face_yaw_deg"], ONE_DIGIT)
    side["failed"] = list(profile.failed)
    return {"frontal": front, "profile": side}


def retake_items(frontal: FrontalResult, profile: ProfileResult, texts: Texts, lang: str) -> list[dict]:
    """One item per quality code of each image, with the regions that code affects."""
    items = []
    for image, result in ((codes.FRONTAL, frontal), (codes.PROFILE, profile)):
        for code in result.failed:
            regions = [r for r in codes.REGIONS if code in result.region_failed.get(r, ())]
            items.append({"image": image, "code": code, "regions": regions, "text": texts.text(f"quality.{code}", lang)})
    return items


@dataclass(frozen=True)
class _Request:
    frontal: FrontalResult
    profile: ProfileResult
    predictions: Predictions
    meta: Meta
    bundle: Bundle
    texts: Texts
    lab: bool

    @property
    def lang(self) -> str:
        return self.texts.resolve(self.meta.lang)

    def quality_codes_of(self, region: str) -> tuple[str, ...]:
        result = self.profile if region in codes.PROFILE_REGIONS else self.frontal
        return tuple(result.region_failed.get(region, ()))

    def codes_of(self, region: str) -> tuple[str, ...]:
        """Quality codes of the region, then model_not_ready when its head is a stub."""
        found = self.quality_codes_of(region)
        head = HEAD_OF.get(region)
        if head is not None and self.bundle.heads[head].stub:
            found = (*found, codes.MODEL_NOT_READY)
        return found

    def measurable(self, region: str) -> bool:
        if region in codes.PROFILE_REGIONS:
            return self.profile.measured and region in self.profile.overlays
        if self.frontal.crops is None:
            return False
        ratios = self.frontal.ratios
        if region == codes.JAW:
            return ratios is not None
        if region == codes.NOSE_BASE:
            return ratios is not None and ratios.nose_ratio is not None
        return True

    def region(self, region: str) -> dict:
        found = self.codes_of(region)
        if not self.lab and found:
            return _blank_region(region, codes.NOT_ASSESSABLE, reasons=found)
        if not self.measurable(region):
            if not self.quality_codes_of(region):
                raise ResponseError(f"{region} cannot be measured, but no quality code says why")
            return _blank_region(region, codes.NOT_ASSESSABLE, reasons=found)
        out = _blank_region(region, codes.ASSESSED, warnings=found if self.lab else ())
        if region in codes.UNDEREYE:
            decision = self._undereye(region, out)
        else:
            decision = self._measured(region, out)
        advice = rules.advise(region, decision, self.meta.age, self.meta.sex, self.meta.frontal_source)
        out["decision"] = None if decision is None else decision.as_dict()
        out["procedures"] = [
            {
                "code": code,
                "name": self.texts.text(f"procedure.{code}.name", self.lang),
                "when": self.texts.text(f"procedure.{code}.when", self.lang),
                "checks": None,
            }
            for code in advice.procedures
        ]
        out["procedures_withheld"] = list(advice.withheld)
        out["notes"] = [
            {"code": code, "text": self.texts.text(f"note.{code}", self.lang), "with_procedures": code in codes.NOTES_WITH_PROCEDURES}
            for code in advice.notes
        ]
        return out

    def _prediction(self, region: str):
        value = getattr(self.predictions, region)
        if value is None:
            raise ResponseError(f"{region} has a face to measure but no prediction from its head")
        return value

    def _measured(self, region: str, out: dict) -> criteria.Decision | None:
        measure = codes.REGION_MEASURE[region]
        sex = self.meta.sex
        if region in codes.PROFILE_REGIONS:
            value = self.profile.legan_angle if region == codes.CHIN else self.profile.nasofacial_angle
            band, landmark, overlay = criteria.DEFAULT_BANDS[measure], None, self.profile.overlays[region]
        else:
            ratios = self.frontal.ratios
            value = float(self._prediction(region))
            band = self.bundle.heads[region].band
            mp_ratio = ratios.jaw_ratio if region == codes.JAW else ratios.nose_ratio
            landmark = self.bundle.baseline[region].apply(mp_ratio)
            names = FRONTAL_OVERLAY_POINTS[region]
            overlay = Overlay(codes.FRONTAL, {k: ratios.points[k] for k in names}, FRONTAL_OVERLAY_LINES[region])
        out["measurements"] = [measurement(measure, value, sex, band, landmark, self.texts, self.lang)]
        out["overlay"] = overlay_dict(overlay)
        if region == codes.NOSE_BASE and self.meta.frontal_source == codes.UPLOAD:
            return None
        return criteria.decide(measure, value, sex, band)

    def _undereye(self, region: str, out: dict) -> criteria.Decision:
        p1, p2 = self._prediction(region)
        head = self.bundle.heads[UNDEREYE_HEAD]
        result = undereye.decide(p1, p2, head.low, head.high)
        crop = self.frontal.crops.undereye_left if region == codes.UNDEREYE_LEFT else self.frontal.crops.undereye_right
        points = {name: (float(x), float(y)) for name, (x, y) in zip(BOX_CORNERS, crop.corners, strict=True)}
        out["overlay"] = overlay_dict(Overlay(codes.FRONTAL, points, BOX_LINES))
        out["probabilities"] = {
            "grades": {g: number(p, THREE_DIGITS) for g, p in result.grades.items()},
            "at_least": {g: number(p, THREE_DIGITS) for g, p in result.at_least.items()},
        }
        return result.decision


def build(
    frontal: FrontalResult,
    profile: ProfileResult | None,
    predictions: Predictions,
    meta: Meta,
    bundle: Bundle,
    texts: Texts,
    lab: bool = False,
) -> dict:
    """quality, retake, regions (in codes.REGIONS order) and disclaimer of api.md section 2.2."""
    profile = profile if profile is not None else missing_profile()
    req = _Request(frontal, profile, predictions, meta, bundle, texts, bool(lab))
    return {
        "retake": [] if req.lab else retake_items(frontal, profile, texts, req.lang),
        "quality": quality_dict(frontal, profile),
        "regions": [req.region(r) for r in codes.REGIONS],
        "disclaimer": texts.text("disclaimer", req.lang),
    }
