"""Example results for the web tests and the demo mode (web.md section 9).

Run from the project root (the same on macOS and Windows):

    uv run python model/scripts/make_web_fixtures.py

Writes one JSON file per case to web_fixtures of config.toml (web/tests/fixtures/),
and codes.json with every code of codes.py. Each case is built by response.build from
synthetic measurement results, as the api would build it, then checked against
api.schemas.AnalyzeResult. No image is read or written: each file holds the meta of
the request and the size and source of each image, so the web draws a plain picture
of that size. Running it again gives the same bytes.

A case file holds:

- name: the file name without .json
- meta: the meta part of the request (api.md section 2.1)
- images: width, height and source ("camera" or "upload") of the frontal and the
  profile image; profile is null when none was sent. The profile source is known only
  to the web, which labels the cards of uploaded images.
- result: the answer of POST /analyze (api.md section 2.2)

The case api_example is the example of api.md section 2.2 with its two regions and its
request_id and versions; tests/test_make_web_fixtures.py checks it against the document.
"""

import argparse
import hashlib
import json
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from facemeasure import bundle, codes, criteria, crops, paths, quality, response
from facemeasure import profile as P
from facemeasure.face_geometry import LevelFrame
from facemeasure.frontal import FrontalRatios, FrontalResult
from facemeasure.i18n import Texts
from make_stub_bundle import stub_bundle, stub_parts

CODES_NAME = "codes.json"
WIDTH, HEIGHT = 1280, 1600  # pixels of every synthetic image, portrait like a phone photo

# A face centred in a 1280 x 1600 image, eyes 160 px apart; _l and _r are image sides
FRONTAL_POINTS = {
    "go_l": (440.0, 1010.0), "go_r": (840.0, 1012.0), "zy_l": (398.0, 760.0), "zy_r": (882.0, 761.0),
    "en_l": (600.0, 702.0), "en_r": (680.0, 702.0), "al_l": (590.0, 902.0), "al_r": (690.0, 903.0),
}
UNDEREYE_BOX_SIDE = 72.0
UNDEREYE_RIGHT_BOX = (524.0, 712.0)  # person's right eye, image left: top-left corner
UNDEREYE_LEFT_BOX = (684.0, 712.0)  # person's left eye, image right
FRONTAL_QUALITY = {
    "iod_px": 160.0, "yaw_deg": 1.2, "pitch_deg": -2.1, "roll_deg": 0.4, "smile": 0.04, "jaw_open": 0.01,
    "blink_left": 0.05, "blink_right": 0.06, "hair_jaw": 0.01, "hair_undereye_left": 0.0, "hair_undereye_right": 0.0,
}
# A profile facing the image right
PROFILE_POINTS = {"g": (720.0, 520.0), "n": (706.0, 610.0), "prn": (820.0, 830.0), "sn": (736.0, 880.0), "pog": (706.0, 1190.0)}

# The example of api.md section 2.2: its inputs (as in tests/test_api_example.py) and its fields
EXAMPLE_REGIONS = (codes.JAW, codes.NOSE_DORSUM)
EXAMPLE_REQUEST_ID = "5f0c2a9e-0d1b-4c8e-9a51-2f3e7b6d8c10"
EXAMPLE_VERSIONS = {"api": "1.0.0", "model_bundle": "1.0.0"}
EXAMPLE_FRONTAL_QUALITY = {
    "iod_px": 96.0, "yaw_deg": 0.9, "pitch_deg": -0.8, "roll_deg": 0.5, "smile": 0.05, "jaw_open": 0.02,
    "blink_left": 0.03, "blink_right": 0.04, "hair_jaw": 0.02, "hair_undereye_left": 0.0, "hair_undereye_right": 0.0,
}
EXAMPLE_FRONTAL_POINTS = {
    "go_l": (405.0, 980.0), "go_r": (877.0, 978.0), "zy_l": (356.0, 640.0), "zy_r": (920.0, 642.0),
    "en_l": (590.0, 560.0), "en_r": (690.0, 561.0), "al_l": (585.0, 800.0), "al_r": (695.0, 801.0),
}
EXAMPLE_PROFILE_POINTS = {"g": (640.0, 410.0), "n": (655.0, 520.0), "prn": (774.0, 760.0), "sn": (700.0, 800.0), "pog": (610.0, 1210.0)}


def _api_schemas():
    """api.schemas; the api package sits at the project root, which model/scripts/ does not have on sys.path."""
    root = str(paths.project_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from api import API_VERSION, schemas

    return API_VERSION, schemas


# ---------------------------------------------------------------- synthetic measurement results


def _box(x0: float, y0: float, side: float) -> np.ndarray:
    """Corners top-left, top-right, bottom-right, bottom-left of an axis-aligned square."""
    return np.array([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)], dtype=np.float64)


def frontal_result(
    failed: tuple[str, ...] = (),
    hair_over: tuple[str, ...] = (),
    face: bool = True,
    jaw_ratio: float = 0.85,
    nose_ratio: float = 1.08,
    values: dict | None = None,
    points: dict | None = None,
    boxes: tuple[tuple[float, float], tuple[float, float], float] | None = None,
) -> FrontalResult:
    """A FrontalResult as analyze_frontal returns it, without any model.

    jaw_ratio and nose_ratio are the MediaPipe ratios, which become landmark_value.
    boxes is (top-left of the left under-eye box, of the right one, side).
    """
    if not face:
        f = (codes.NO_FACE,)
        return FrontalResult(WIDTH, HEIGHT, 0, None, None, None, None, f, codes.by_region(f), None, None, None)
    values = dict(FRONTAL_QUALITY if values is None else values)
    left, right, side = boxes or (UNDEREYE_LEFT_BOX, UNDEREYE_RIGHT_BOX, UNDEREYE_BOX_SIDE)
    blank = np.zeros((crops.CROP_SIZE, crops.CROP_SIZE, 3), dtype=np.uint8)
    face_crops = crops.Crops(
        face=crops.Crop(blank, _box(300.0, 400.0, 680.0), False),
        undereye_left=crops.Crop(blank, _box(*left, side), True),
        undereye_right=crops.Crop(blank, _box(*right, side), False),
        frame=LevelFrame(center=np.array([WIDTH / 2, 700.0]), cos=1.0, sin=0.0),
        iod_px=values["iod_px"],
    )
    ratios = FrontalRatios(
        jaw_ratio=jaw_ratio,
        nose_ratio=nose_ratio,
        nose_ratios={"98_327": nose_ratio},
        points=dict(FRONTAL_POINTS if points is None else points),
    )
    failed = tuple(failed)
    return FrontalResult(
        width=WIDTH,
        height=HEIGHT,
        face_count=2 if codes.MULTIPLE_FACES in failed else 1,
        landmarks=np.zeros((478, 3)),
        blendshapes={},
        matrix=np.eye(4),
        quality=quality.FrontalQuality(**values, failed=failed),
        failed=failed,
        region_failed=codes.by_region(failed, tuple(hair_over)),
        crops=face_crops,
        ratios=ratios,
        ratios_error=None,
    )


def profile_result(
    legan: float = 13.0,
    nasofacial: float = 35.5,
    failed: tuple[str, ...] = (),
    face_found: bool = False,
    yaw: float | None = None,
    points: dict | None = None,
) -> P.ProfileResult:
    """A ProfileResult whose edge was measured, as analyze_profile returns it."""
    failed = tuple(failed)
    pts = dict(PROFILE_POINTS if points is None else points)
    q = P.ProfileQuality(face_found=face_found, face_yaw_deg=yaw, silhouette_found=True, flipped=False, failed=failed)
    overlays = {
        region: P.Overlay(codes.PROFILE, {k: pts[k] for k in names}, P.OVERLAY_LINES[region])
        for region, names in P.OVERLAY_POINTS.items()
    }
    reasons = (P.FRONTAL_YAW,) if failed else ()
    return P.ProfileResult(
        WIDTH, HEIGHT, q, failed, codes.profile_by_region(failed), reasons,
        legan_angle=legan, nasofacial_angle=nasofacial, nose_depth=0.2, face_height=900.0, points=pts, overlays=overlays,
    )


def ready_bundle() -> bundle.Bundle:
    """The stub bundle with every head marked as trained: fixed answers and assessed regions."""
    manifest, arrays = stub_parts()
    for head in manifest["heads"].values():
        head["stub"] = False
    manifest["stub"] = False
    manifest["files"] = {bundle.HEADS_NAME: "0" * 64}
    return bundle.parse(manifest, arrays)


# ---------------------------------------------------------------- the cases


@dataclass(frozen=True)
class Case:
    """One example result: what the web sent and what the measurement gave."""

    meta: response.Meta
    frontal: Callable[[], FrontalResult]
    profile: Callable[[], P.ProfileResult | None]
    predictions: response.Predictions
    lab: bool = False
    stub: bool = False  # the heads of the bundle are stubs (model_not_ready)
    profile_source: str = codes.CAMERA
    regions: tuple[str, ...] = codes.REGIONS  # regions kept in the file
    request_id: str | None = None  # default: derived from the name
    versions: dict | None = None  # default: this api and the bundle


def _female(age: int = 30, source: str = codes.CAMERA) -> response.Meta:
    return response.Meta(age=age, sex=codes.FEMALE, frontal_source=source)


def _pred(jaw: float, nose: float, left: tuple[float, float], right: tuple[float, float]) -> response.Predictions:
    return response.Predictions(jaw=jaw, nose_base=nose, undereye_left=left, undereye_right=right)


def cases() -> dict[str, Case]:
    """Every case of the web tests, by file name."""
    return {
        # api.md section 2.2
        "api_example": Case(
            meta=_female(22),
            frontal=lambda: frontal_result(
                jaw_ratio=0.842, values=EXAMPLE_FRONTAL_QUALITY, points=EXAMPLE_FRONTAL_POINTS,
                boxes=((683.25, 560.5), (561.0, 560.0), 57.6),
            ),
            profile=lambda: profile_result(legan=14.0, nasofacial=28.5, points=EXAMPLE_PROFILE_POINTS),
            predictions=_pred(0.836, 1.08, (0.1, 0.05), (0.1, 0.05)),
            regions=EXAMPLE_REGIONS,
            request_id=EXAMPLE_REQUEST_ID,
            versions=EXAMPLE_VERSIONS,
        ),
        # A woman with one clear grade in every region: jaw 2+, chin 1, under-eye 0 and 2+, nasal base 0, dorsum 0
        "female_clear": Case(
            meta=_female(30),
            frontal=lambda: frontal_result(jaw_ratio=0.893, nose_ratio=1.07),
            profile=lambda: profile_result(legan=18.0, nasofacial=35.5),
            predictions=_pred(0.90, 1.05, (0.1, 0.05), (0.95, 0.82)),
        ),
        # Borderline in many ways: jaw 0-1, chin 1-2, under-eye 1-2+ and 0-2+, nasal base 1-2+, dorsum 1-2+
        "borderline": Case(
            meta=_female(30),
            frontal=lambda: frontal_result(jaw_ratio=0.829, nose_ratio=1.29),
            profile=lambda: profile_result(legan=20.5, nasofacial=30.5),
            predictions=_pred(0.836, 1.27, (0.8, 0.5), (0.6, 0.45)),
        ),
        # A man: the jaw value has no grade; the nose uses the criteria of men
        "male": Case(
            meta=response.Meta(age=35, sex=codes.MALE),
            frontal=lambda: frontal_result(jaw_ratio=0.94, nose_ratio=1.06),
            profile=lambda: profile_result(legan=14.0, nasofacial=36.5),
            predictions=_pred(0.95, 1.05, (0.1, 0.05), (0.2, 0.05)),
        ),
        # The frontal image chosen from the computer: the nasal base shows only its value
        "upload_frontal": Case(
            meta=_female(28, codes.UPLOAD),
            frontal=lambda: frontal_result(jaw_ratio=0.81, nose_ratio=1.38),
            profile=lambda: profile_result(legan=12.5, nasofacial=35.5),
            predictions=_pred(0.80, 1.40, (0.15, 0.05), (0.1, 0.05)),
        ),
        # Under 18: grades that would have procedures, none shown
        "age_17": Case(
            meta=_female(17),
            frontal=lambda: frontal_result(jaw_ratio=0.90, nose_ratio=1.31),
            profile=lambda: profile_result(legan=22.0, nasofacial=28.0),
            predictions=_pred(0.90, 1.32, (0.1, 0.05), (0.95, 0.85)),
        ),
        # 20 years old: procedures everywhere but the under-eye
        "age_20": Case(
            meta=_female(20),
            frontal=lambda: frontal_result(jaw_ratio=0.90, nose_ratio=1.31),
            profile=lambda: profile_result(legan=22.0, nasofacial=28.0),
            predictions=_pred(0.90, 1.32, (0.92, 0.8), (0.95, 0.85)),
        ),
        # No face in the frontal image and no profile image
        "no_face_no_profile": Case(
            meta=_female(30),
            frontal=lambda: frontal_result(face=False),
            profile=lambda: None,
            predictions=response.Predictions(),
        ),
        # Hair over the jaw edge and over the person's left under-eye
        "hair_jaw_one_eye": Case(
            meta=_female(30),
            frontal=lambda: frontal_result(
                failed=(codes.HAIR_OCCLUSION,), hair_over=("hair_jaw", "hair_undereye_left"),
                values={**FRONTAL_QUALITY, "hair_jaw": 0.32, "hair_undereye_left": 0.24},
            ),
            profile=lambda: profile_result(legan=13.5, nasofacial=35.0),
            predictions=_pred(0.85, 1.05, (0.1, 0.05), (0.2, 0.05)),
        ),
        # Chin grade 3 and nasal base grade 2+
        "chin3_nose_base2": Case(
            meta=_female(40),
            frontal=lambda: frontal_result(jaw_ratio=0.81, nose_ratio=1.33),
            profile=lambda: profile_result(legan=27.0, nasofacial=36.0),
            predictions=_pred(0.80, 1.32, (0.1, 0.05), (0.1, 0.05)),
        ),
        # Lab mode: images that fail some checks are measured anyway, with the codes in warnings
        "lab": Case(
            meta=_female(30),
            frontal=lambda: frontal_result(
                failed=(codes.FACE_TOO_SMALL, codes.EYES_CLOSED), jaw_ratio=0.872, nose_ratio=1.12,
                values={**FRONTAL_QUALITY, "iod_px": 62.4, "blink_left": 0.71, "blink_right": 0.64},
            ),
            profile=lambda: profile_result(legan=16.8, nasofacial=37.9, failed=(codes.PROFILE_INVALID,), face_found=True, yaw=34.6),
            predictions=_pred(0.861, 1.14, (0.62, 0.21), (0.41, 0.33)),
            lab=True,
        ),
        # The heads of the bundle are still stubs: only the profile regions are assessed
        "stub_heads": Case(
            meta=_female(30),
            frontal=lambda: frontal_result(),
            profile=lambda: profile_result(legan=17.0, nasofacial=33.0),
            predictions=_pred(0.85, 1.09, (0.5, 0.2), (0.5, 0.2)),
            stub=True,
        ),
    }


def request_id(name: str) -> str:
    """A fixed uuid4 for a case, so the files are the same on every run."""
    return str(uuid.UUID(bytes=hashlib.sha256(name.encode("utf-8")).digest()[:16], version=4))


def build_case(name: str, case: Case, texts: Texts, bundles: dict[bool, bundle.Bundle]) -> dict:
    """The file of one case, its result checked against the api's schema."""
    api_version, schemas = _api_schemas()
    b = bundles[case.stub]
    built = response.build(case.frontal(), case.profile(), case.predictions, case.meta, b, texts, lab=case.lab)
    built["regions"] = [r for r in built["regions"] if r["region"] in case.regions]
    result = {
        "request_id": case.request_id or request_id(name),
        "versions": case.versions or {"api": api_version, "model_bundle": b.version},
        "lab": case.lab,
        **built,
    }
    result = schemas.AnalyzeResult.model_validate(result).model_dump(mode="json")
    meta = {
        "age": case.meta.age,
        "sex": case.meta.sex,
        "frontal_source": case.meta.frontal_source,
        "lang": case.meta.lang,
        "lab": case.lab,
    }
    schemas.Meta.model_validate(meta)
    has_profile = result["quality"]["profile"]["silhouette_found"] is not None
    images = {
        codes.FRONTAL: {"width": WIDTH, "height": HEIGHT, "source": case.meta.frontal_source},
        codes.PROFILE: {"width": WIDTH, "height": HEIGHT, "source": case.profile_source} if has_profile else None,
    }
    return {"name": name, "meta": meta, "images": images, "result": result}


def codes_file() -> dict:
    """Every code of codes.py and the grades of criteria.py, for the web tests of the texts."""
    return {
        "regions": list(codes.REGIONS),
        "frontal_regions": list(codes.FRONTAL_REGIONS),
        "profile_regions": list(codes.PROFILE_REGIONS),
        "images": list(codes.IMAGES),
        "sexes": list(codes.SEXES),
        "frontal_sources": list(codes.FRONTAL_SOURCES),
        "measures": list(codes.MEASURES),
        "units": dict(response.UNITS),
        "methods": [codes.MODEL, codes.MEASURED],
        "statuses": [codes.ASSESSED, codes.NOT_ASSESSABLE],
        "levels": [codes.CLEAR, codes.BORDERLINE],
        "frontal_codes": list(codes.FRONTAL_CODES),
        "profile_codes": list(codes.PROFILE_CODES),
        "status_codes": list(codes.STATUS_CODES),
        "procedures": list(codes.PROCEDURES),
        "notes": list(codes.NOTES),
        "notes_with_procedures": [n for n in codes.NOTES if n in codes.NOTES_WITH_PROCEDURES],
        "withheld": list(codes.WITHHELD),
        "grades": list(criteria.GRADES),
        "chin_grades": list(criteria.CHIN_GRADES),
    }


def dump(data: dict) -> str:
    """JSON text written the same on every machine."""
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def write_all(out: Path) -> list[Path]:
    """Write every case and codes.json to out; returns the files written."""
    out.mkdir(parents=True, exist_ok=True)
    texts = Texts()
    bundles = {False: ready_bundle(), True: stub_bundle()}
    written = []
    files = {f"{name}.json": build_case(name, case, texts, bundles) for name, case in cases().items()}
    files[CODES_NAME] = codes_file()
    for file_name, data in files.items():
        path = out / file_name
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(dump(data))
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write example results for the web tests and the demo mode")
    parser.add_argument("--out", type=Path, default=None, help="folder (default: web_fixtures in config.toml)")
    args = parser.parse_args(argv)
    out = args.out if args.out is not None else paths.path("web_fixtures")
    written = write_all(out)
    print(f"เขียนผลตัวอย่าง {len(written) - 1} ไฟล์และ {CODES_NAME} ที่ {out} แล้ว")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
