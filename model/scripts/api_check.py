"""The api with the real engine on dataset images (api.md section 7).

Run from the project root (the same on macOS and Windows):

    uv run python model/scripts/api_check.py

The api runs in this process through FastAPI's TestClient with the models of
api/assets/ and api/bundle/, so the steps are those of POST /analyze. It writes
model/reports/api_check.md, numbers only: no image, no file name, no measurement of a
single person. tests/test_api_real.py runs the same checks with fewer images.
Only SCUT-FBP5500, the graded images and FEI are used (model.md section 5.1).

- fidelity: the fixed copy 1 of SCUT test images (train_ratio_heads.py) and of graded test
  images (train_undereye_head.py) made again from their seeds and sent as PNG in lab mode; the
  jaw and alar base values, their landmark_value and the under-eye probabilities must
  equal test_predictions.csv to within the rounding of the JSON (half its last digit,
  plus FIDELITY_SLACK for the float16 features of training). Their warnings must equal
  the stored codes.
- repeat: the same request twice gives the same result but for request_id.
- exif: pixels turned and EXIF orientation 3, 6 or 8 set, against the upright image,
  both JPEG; the jaw within EXIF_JAW_PP percentage points and the angles within
  ANGLE_DEG degrees.
- profile_flip: FEI full profiles mirrored give the same angles within ANGLE_DEG.
- quality codes: SCUT shrunk to SMALL_IOD_FRACTION of min_iod_px gives face_too_small,
  turned by POSE_ROLL_DEG gives frontal_pose; most FEI smiling photos give smiling;
  FEI frontal photos sent as the profile give profile_invalid; no profile gives
  missing_input.
- lab: a photo that fails the checks, in lab mode, is assessed with warnings,
  retake is empty and lab is true.
- timing: TIMING_REQUESTS requests of FEI frontal and profile photos placed on a
  1920 x 1080 camera-sized frame, JPEG; p50 and p90 must be well under the 60 s the
  web waits (web.md section 4.2).
"""

import argparse
import hashlib
import io
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from facemeasure import degrade, fei, image_io, paths

FIDELITY_IMAGES = 5
FIDELITY_SLACK = 1e-5  # proportions and probabilities; training found at most 9e-6
JSON_HALF_DIGIT = {"%": 0.05, "ratio": 0.0005, "probability": 0.0005}
EXIF_JAW_PP = 0.3
ANGLE_DEG = 0.5
EXIF_ORIENTATIONS = (3, 6, 8)
# The turn that stores an upright image so that EXIF orientation o turns it back
STORE_TURN = {3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_90, 8: Image.Transpose.ROTATE_270}
EXIF_JPEG_QUALITY = 95
EXIF_PAIRS = 3
FLIP_PEOPLE = 10
SMALL_IOD_FRACTION = 0.8
POSE_ROLL_DEG = 15.0
QUALITY_IMAGES = 5
SMILE_PEOPLE = 20
SMILE_MAJORITY = 0.5  # "most": more than half
PROFILE_INVALID_PEOPLE = 10
TIMING_REQUESTS = 20
TIMING_FRAME = (1920, 1080)
TIMING_JPEG_QUALITY = 92
TIMING_FILL = (128, 128, 128)
WEB_TIMEOUT_S = 60
SELECT_SALT = "api-check-v1"
SEARCH_FACTOR = 20  # candidates per image wanted, when only images that pass the checks will do
REPORT_NAME = "api_check.md"
META = {"age": 30, "sex": "female"}
BLANK_FRONTAL_SIZE = (64, 64)  # a frontal image without a face, for checks of the profile only


class CheckError(RuntimeError):
    """The api answered something other than 200."""


# ---------------------------------------------------------------- the client


def project_on_path() -> None:
    """The api package sits at the project root, which a script in model/scripts/ does not have on sys.path."""
    root = str(paths.project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def real_client():
    """A TestClient of the real api; enter it (with) to load the engine."""
    project_on_path()
    from fastapi.testclient import TestClient

    from api.main import create_app

    return TestClient(create_app())


def png(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, "PNG")
    return buf.getvalue()


def jpeg(image: np.ndarray | Image.Image, quality: int, exif: bytes | None = None) -> bytes:
    im = image if isinstance(image, Image.Image) else Image.fromarray(image)
    buf = io.BytesIO()
    kw = {"exif": exif} if exif is not None else {}
    im.save(buf, "JPEG", quality=quality, subsampling=0, **kw)
    return buf.getvalue()


def blank_frontal() -> bytes:
    w, h = BLANK_FRONTAL_SIZE
    return png(np.full((h, w, 3), 128, dtype=np.uint8))


def send(client, frontal: bytes, profile: bytes | None = None, lab: bool = True, **meta) -> dict:
    files = {"frontal": ("frontal", frontal, "application/octet-stream")}
    if profile is not None:
        files["profile"] = ("profile", profile, "application/octet-stream")
    res = client.post("/analyze", files=files, data={"meta": json.dumps({**META, **meta, "lab": lab})})
    if res.status_code != 200:
        raise CheckError(f"POST /analyze answered {res.status_code}: {res.text}")
    return res.json()


def region(result: dict, name: str) -> dict:
    return next(r for r in result["regions"] if r["region"] == name)


def value(result: dict, name: str, key: str = "value") -> float | None:
    r = region(result, name)
    return r["measurements"][0][key] if r["measurements"] else None


def ranked(names, salt: str = SELECT_SALT) -> list[str]:
    return sorted(names, key=lambda s: hashlib.sha256(f"{salt}:{s}".encode()).hexdigest())


def engine_of(client):
    return client.app.state.engine


# ---------------------------------------------------------------- sources


def scut_test_images(n: int) -> list[Path]:
    """n SCUT-FBP5500 test images of labels/scut_labels.csv, in a fixed order."""
    import extract_features as ef

    sources = {s.image: s.path for s in ef.face_sources() if s.split == "test"}
    return [sources[name] for name in ranked(sources)[:n]]


def fixed_copy(task: str, name: str, path: Path, copy: int, client) -> np.ndarray:
    """The fixed copy of extract_features.py, made again from its seed: analyze_frontal on the source, then degrade."""
    from facemeasure.frontal import analyze_frontal

    engine = engine_of(client)
    image = image_io.read_image(path)
    first = analyze_frontal(image, engine.models, engine.config.frontal)
    source_iod = first.quality.iod_px if first.face_count and first.quality is not None else None
    params = degrade.copy_params(task, name, degrade.FIXED, copy, engine.config)
    return degrade.degrade(image, source_iod, params).image


def _codes(text) -> list[str]:
    return [c for c in str(text or "").split(";") if c]


# ---------------------------------------------------------------- checks


def check_ratio_fidelity(client, n: int = FIDELITY_IMAGES) -> dict:
    import extract_features as ef
    import train_ratio_heads as trh

    rows = trh.pick_fidelity(trh.read_predictions(trh.out_dir()), n)
    sources = {s.image: s.path for s in ef.face_sources()}
    tol = {"%": JSON_HALF_DIGIT["%"] + 100 * FIDELITY_SLACK, "ratio": JSON_HALF_DIGIT["ratio"] + FIDELITY_SLACK}
    worst = {"jaw_value": 0.0, "jaw_landmark": 0.0, "nose_value": 0.0, "nose_landmark": 0.0}
    same_codes = True
    for _, r in rows.iterrows():
        res = send(client, png(fixed_copy(ef.FACE, r["image"], sources[r["image"]], trh.LOW_COPY, client)))
        pairs = {
            "jaw_value": (value(res, "jaw"), 100 * r["pred_jaw"]),
            "jaw_landmark": (value(res, "jaw", "landmark_value"), 100 * r["landmark_jaw"]),
            "nose_value": (value(res, "nose_base"), r["pred_nose_base"]),
            "nose_landmark": (value(res, "nose_base", "landmark_value"), r["landmark_nose_base"]),
        }
        for k, (got, stored) in pairs.items():
            worst[k] = max(worst[k], float("inf") if got is None else abs(got - float(stored)))
        for name, column in (("jaw", "reasons_jaw"), ("nose_base", "reasons_nose_base")):
            same_codes &= [c for c in region(res, name)["warnings"] if c != "model_not_ready"] == _codes(r[column])
    passed = (len(rows) == n and worst["jaw_value"] <= tol["%"] and worst["jaw_landmark"] <= tol["%"]
              and worst["nose_value"] <= tol["ratio"] and worst["nose_landmark"] <= tol["ratio"] and same_codes)
    return {"images": len(rows), "max_diff": worst, "tolerance": tol, "same_codes": same_codes, "passed": passed}


def check_undereye_fidelity(client, n: int = FIDELITY_IMAGES) -> dict:
    import extract_features as ef
    import train_undereye_head as tuh

    pred = tuh.read_predictions(tuh.out_dir())
    order = tuh.pick_fidelity(pred, len(pred))
    images = list(dict.fromkeys(order["image"]))[:n]
    sources = {s.image: s.path for s in ef.undereye_sources()}
    tol = JSON_HALF_DIGIT["probability"] + FIDELITY_SLACK
    worst, sides, same_codes = 0.0, 0, True
    for image in images:
        res = send(client, png(fixed_copy(ef.UNDEREYE, image, sources[image], tuh.LOW_COPY, client)))
        rows = pred[(pred["image"] == image) & (pred["copy"] == tuh.LOW_COPY)]
        for _, r in rows.iterrows():
            reg = region(res, ef.SIDE_REGION[r["side"]])
            at_least = (reg["probabilities"] or {}).get("at_least")
            if at_least is None:
                worst = float("inf")
                continue
            worst = max(worst, abs(at_least["1"] - float(r["p1"])), abs(at_least["2+"] - float(r["p2"])))
            same_codes &= [c for c in reg["warnings"] if c != "model_not_ready"] == _codes(r["reasons"])
            sides += 1
    passed = len(images) == n and sides >= n and worst <= tol and same_codes
    return {"images": len(images), "sides": sides, "max_diff": worst, "tolerance": tol, "same_codes": same_codes, "passed": passed}


def check_repeat(client) -> dict:
    frontal = scut_test_images(1)[0].read_bytes()
    profile = fei.image_path(1, fei.PROFILE_FULL[1]).read_bytes()
    a, b = (send(client, frontal, profile, lab=False) for _ in range(2))
    differ = a["request_id"] != b["request_id"]
    a.pop("request_id"), b.pop("request_id")
    return {"requests": 2, "same": a == b, "request_ids_differ": differ, "passed": a == b and differ}


def turned(image: np.ndarray, orientation: int) -> Image.Image:
    """The pixels as a camera stores them when EXIF orientation must turn them upright."""
    return Image.fromarray(image).transpose(STORE_TURN[orientation])


def png_exif(im: Image.Image, orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    im.save(buf, "PNG", exif=exif.tobytes())
    return buf.getvalue()


def jpeg_exif(im: Image.Image, orientation: int, quality: int = EXIF_JPEG_QUALITY) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    return jpeg(im, quality, exif.tobytes())


def passing_frontals(client, n: int) -> list[np.ndarray]:
    """SCUT test images that pass every check of the frontal image, so normal mode assesses them."""
    out = []
    for path in scut_test_images(SEARCH_FACTOR * n):
        image = image_io.read_image(path)
        if not send(client, png(image), lab=False)["quality"]["frontal"]["failed"]:
            out.append(image)
            if len(out) == n:
                break
    return out


def passing_profiles(client, n: int, pose: int = fei.PROFILE_FULL[1]) -> list[np.ndarray]:
    """FEI full profiles that pass the profile check and get both angles."""
    out = []
    for person in range(1, fei.PEOPLE + 1):
        image = image_io.read_image(fei.image_path(person, pose))
        res = send(client, blank_frontal(), png(image), lab=False)
        if not res["quality"]["profile"]["failed"] and value(res, "chin") is not None:
            out.append(image)
            if len(out) == n:
                break
    return out


def _numbers(res: dict) -> dict[str, float | None]:
    return {"jaw": value(res, "jaw"), "chin": value(res, "chin"), "nose_dorsum": value(res, "nose_dorsum")}


def _diffs(a: dict, b: dict) -> tuple[float | None, float | None]:
    """Largest jaw difference (percentage points) and angle difference (degrees); None when one is missing."""
    x, y = _numbers(a), _numbers(b)
    if any(v is None for v in (*x.values(), *y.values())):
        return None, None
    return abs(x["jaw"] - y["jaw"]), max(abs(x["chin"] - y["chin"]), abs(x["nose_dorsum"] - y["nose_dorsum"]))


def _without_id(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "request_id"}


def check_exif(client, pairs: int = EXIF_PAIRS) -> dict:
    """EXIF orientation 3, 6 and 8 on SCUT frontal and FEI profile images that pass the checks.

    Passed when the PNG copies with EXIF orientation give the very result of the upright
    PNG; that is the api's EXIF step alone. The JPEG copies, as a camera sends them, are
    reported against the upright JPEG next to the change that saving the upright image
    once more (quality 94 instead of 95) makes, which is JPEG noise and not EXIF.
    """
    frontals, profiles = passing_frontals(client, pairs), passing_profiles(client, pairs)
    same, shape_kept, requests = True, True, 0
    lossless, lossy, resaved = [], [], []
    for f, p in zip(frontals, profiles, strict=False):
        base_png = send(client, png(f), png(p))
        base_jpeg = send(client, jpeg(f, EXIF_JPEG_QUALITY), jpeg(p, EXIF_JPEG_QUALITY))
        resaved.append(_diffs(base_jpeg, send(client, jpeg(f, EXIF_JPEG_QUALITY - 1), jpeg(p, EXIF_JPEG_QUALITY - 1))))
        requests += 3
        for o in EXIF_ORIENTATIONS:
            fb, pb = png_exif(turned(f, o), o), png_exif(turned(p, o), o)
            shape_kept &= image_io.read_image(fb).shape == f.shape and image_io.read_image(pb).shape == p.shape
            res = send(client, fb, pb)
            same &= _without_id(res) == _without_id(base_png)
            lossless.append(_diffs(res, base_png))
            lossy.append(_diffs(send(client, jpeg_exif(turned(f, o), o), jpeg_exif(turned(p, o), o)), base_jpeg))
            requests += 2

    def worst(rows, k):
        return None if any(r[k] is None for r in rows) or not rows else max(r[k] for r in rows)

    out = {
        "pairs": min(len(frontals), len(profiles)), "requests": requests, "upright_shape_kept": shape_kept,
        "png_same_result": same, "png_max_jaw_pp": worst(lossless, 0), "png_max_angle_deg": worst(lossless, 1),
        "jpeg_max_jaw_pp": worst(lossy, 0), "jpeg_max_angle_deg": worst(lossy, 1),
        "resave_max_jaw_pp": worst(resaved, 0), "resave_max_angle_deg": worst(resaved, 1),
    }
    out["jpeg_within_tolerance"] = (out["jpeg_max_jaw_pp"] is not None and out["jpeg_max_jaw_pp"] <= EXIF_JAW_PP
                                    and out["jpeg_max_angle_deg"] is not None and out["jpeg_max_angle_deg"] <= ANGLE_DEG)
    out["passed"] = out["pairs"] == pairs and shape_kept and same
    return out


def check_profile_flip(client, people: int = FLIP_PEOPLE) -> dict:
    """FEI full profiles and their mirror images, sent as PNG; both angles within ANGLE_DEG."""
    both, only_one, neither, legan, naso, direction_ok = 0, 0, 0, [], [], True
    for person in range(1, people + 1):
        for pose in fei.PROFILE_FULL:
            image = image_io.read_image(fei.image_path(person, pose))
            a = send(client, blank_frontal(), png(image))
            b = send(client, blank_frontal(), png(np.ascontiguousarray(image[:, ::-1])))
            got = [(value(x, "chin"), value(x, "nose_dorsum")) for x in (a, b)]
            measured = [g[0] is not None for g in got]
            if all(measured):
                both += 1
                legan.append(abs(got[0][0] - got[1][0]))
                naso.append(abs(got[0][1] - got[1][1]))
                direction_ok &= a["quality"]["profile"]["flipped"] != b["quality"]["profile"]["flipped"]
            elif any(measured):
                only_one += 1
            else:
                neither += 1
    diffs = np.array(legan + naso) if both else np.zeros(0)

    def spread(values):
        v = np.array(values)
        return None if not len(v) else {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max())}

    worst = float(diffs.max()) if len(diffs) else None
    return {
        "images": 2 * people, "measured_both": both, "measured_one_only": only_one, "measured_neither": neither,
        "direction_flips": direction_ok, "legan": spread(legan), "nasofacial": spread(naso),
        "share_within": float((diffs <= ANGLE_DEG).mean()) if len(diffs) else None, "max_angle_deg": worst,
        "passed": both > 0 and direction_ok and worst is not None and worst <= ANGLE_DEG,
    }


def check_quality_codes(client, images: int = QUALITY_IMAGES, smile_people: int = SMILE_PEOPLE,
                        invalid_people: int = PROFILE_INVALID_PEOPLE) -> dict:
    min_iod = engine_of(client).config.frontal.min_iod_px
    small, pose, base_ok = 0, 0, 0
    sources = scut_test_images(images)
    for path in sources:
        image = image_io.read_image(path)
        iod = send(client, png(image))["quality"]["frontal"]["iod_px"]
        if iod is None:
            continue
        base_ok += 1
        shrunk = degrade.shrink(image, SMALL_IOD_FRACTION * min_iod / iod) if iod > SMALL_IOD_FRACTION * min_iod else image
        small += "face_too_small" in send(client, png(shrunk))["quality"]["frontal"]["failed"]
        turned = np.array(Image.fromarray(image).rotate(POSE_ROLL_DEG, Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255)))
        pose += "frontal_pose" in send(client, png(turned))["quality"]["frontal"]["failed"]
    smiling = sum(
        "smiling" in send(client, fei.image_path(p, fei.FRONTAL_SMILE).read_bytes(), lab=False)["quality"]["frontal"]["failed"]
        for p in range(1, smile_people + 1)
    )
    invalid = sum(
        "profile_invalid" in send(client, blank_frontal(), fei.image_path(p, fei.FRONTAL_NEUTRAL).read_bytes())["quality"]["profile"]["failed"]
        for p in range(1, invalid_people + 1)
    )
    res = send(client, scut_test_images(1)[0].read_bytes(), None, lab=False)
    missing = (
        res["quality"]["profile"]["failed"] == ["missing_input"]
        and {"image": "profile", "code": "missing_input"} == {k: res["retake"][-1][k] for k in ("image", "code")}
        and all(region(res, r)["status_reasons"] == ["missing_input"] for r in ("chin", "nose_dorsum"))
    )
    out = {
        "face_too_small": {"images": len(sources), "found": small, "passed": base_ok == len(sources) and small == base_ok},
        "frontal_pose": {"images": len(sources), "found": pose, "roll_deg": POSE_ROLL_DEG, "passed": base_ok == len(sources) and pose == base_ok},
        "smiling": {"images": smile_people, "found": smiling, "share": smiling / smile_people, "passed": smiling / smile_people > SMILE_MAJORITY},
        "profile_invalid": {"images": invalid_people, "found": invalid, "passed": invalid == invalid_people},
        "missing_input": {"images": 1, "found": int(missing), "passed": missing},
    }
    out["passed"] = all(v["passed"] for v in out.values())
    return out


def check_lab(client, people: int = SMILE_PEOPLE) -> dict:
    """The first FEI smiling photo that fails the checks, in normal and in lab mode."""
    from facemeasure import codes

    profile = fei.image_path(1, fei.PROFILE_FULL[1]).read_bytes()
    for person in range(1, people + 1):
        frontal = fei.image_path(person, fei.FRONTAL_SMILE).read_bytes()
        normal = send(client, frontal, profile, lab=False)
        if normal["quality"]["frontal"]["failed"]:
            break
    else:
        return {"found": False, "passed": False}
    lab = send(client, frontal, profile, lab=True)
    failing = [r for r in codes.FRONTAL_REGIONS if region(normal, r)["status"] == "not_assessable"]
    ok = (lab["lab"] is True and lab["retake"] == [] and normal["retake"] != [] and bool(failing)
          and all(region(lab, r)["status"] == "assessed" and region(lab, r)["warnings"] for r in failing))
    return {"found": True, "regions_failing": len(failing), "assessed_in_lab": sum(region(lab, r)["status"] == "assessed" for r in failing),
            "retake_in_lab": len(lab["retake"]), "lab": lab["lab"], "passed": ok}


def timing_frame(image: np.ndarray) -> bytes:
    """The photo in the middle of a camera-sized frame, as JPEG."""
    w, h = TIMING_FRAME
    frame = Image.new("RGB", (w, h), TIMING_FILL)
    im = Image.fromarray(image)
    frame.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return jpeg(frame, TIMING_JPEG_QUALITY)


def check_timing(client, n: int = TIMING_REQUESTS) -> dict:
    pairs = [
        (timing_frame(image_io.read_image(fei.image_path(p, fei.FRONTAL_NEUTRAL))),
         timing_frame(image_io.read_image(fei.image_path(p, fei.PROFILE_FULL[1]))))
        for p in range(1, n + 1)
    ]
    seconds = []
    for frontal, profile in pairs:
        start = time.perf_counter()
        send(client, frontal, profile, lab=False)
        seconds.append(time.perf_counter() - start)
    s = np.array(seconds)
    p50, p90 = float(np.percentile(s, 50)), float(np.percentile(s, 90))
    return {"requests": n, "frame": list(TIMING_FRAME), "p50_s": p50, "p90_s": p90, "mean_s": float(s.mean()), "max_s": float(s.max()),
            "device": engine_of(client).device, "passed": p90 < WEB_TIMEOUT_S / 10}


CHECKS = {
    "ratio_fidelity": check_ratio_fidelity,
    "undereye_fidelity": check_undereye_fidelity,
    "repeat": check_repeat,
    "exif": check_exif,
    "profile_flip": check_profile_flip,
    "quality_codes": check_quality_codes,
    "lab": check_lab,
    "timing": check_timing,
}


# ---------------------------------------------------------------- what is there


def missing_inputs() -> dict[str, str]:
    """check name -> why it cannot run here; checks not listed can run."""
    import extract_features as ef
    import train_ratio_heads as trh
    import train_undereye_head as tuh

    out = {}
    scut = paths.path("labels") / ef.LABELS_NAME
    have_scut = scut.is_file() and (paths.path("scut") / "Images").is_dir()
    have_fei = fei.image_path(1, 1).is_file()
    for name in ("repeat", "exif", "quality_codes"):
        if not have_scut:
            out[name] = "no SCUT-FBP5500 images or labels/scut_labels.csv"
        elif not have_fei:
            out[name] = "no FEI images in data/raw/fei/images/"
    for name in ("profile_flip", "lab", "timing"):
        if not have_fei:
            out[name] = "no FEI images in data/raw/fei/images/"
    if not (trh.out_dir() / trh.PREDICTIONS_NAME).is_file() or not have_scut:
        out["ratio_fidelity"] = "no outputs/ratio_heads/test_predictions.csv of train_ratio_heads.py or no SCUT images"
    if not (tuh.out_dir() / tuh.PREDICTIONS_NAME).is_file() or not (paths.path("grading") / ef.MANIFEST_NAME).is_file():
        out["undereye_fidelity"] = "no outputs/undereye_head/test_predictions.csv of train_undereye_head.py or no grading/manifest.csv"
    return out


def stub_reasons(client) -> dict[str, str]:
    """Fidelity needs the trained heads that made test_predictions.csv."""
    stubs = set(engine_of(client).stub_heads)
    out = {}
    if stubs & {"jaw", "nose_base"}:
        out["ratio_fidelity"] = "the jaw and alar base heads of api/bundle/ are stubs"
    if "undereye" in stubs:
        out["undereye_fidelity"] = "the under-eye head of api/bundle/ is a stub"
    return out


# ---------------------------------------------------------------- report


def f(x, digits: int = 4) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "ผ่าน" if x else "ไม่ผ่าน"
    if x == float("inf"):
        return "∞"
    return f"{x:.{digits}f}"


def para(*parts: str) -> str:
    """One paragraph of the report from several pieces."""
    return "".join(parts)


TITLES = {
    "ratio_fidelity": "ความตรงกับตอนเทรน (กราม ฐานจมูก)",
    "undereye_fidelity": "ความตรงกับตอนเทรน (ใต้ตา)",
    "repeat": "ส่งซ้ำได้ผลเดิม",
    "exif": "หมุนด้วย EXIF",
    "profile_flip": "กลับภาพด้านข้าง",
    "quality_codes": "รหัสคุณภาพภาพ",
    "lab": "โหมดทดลอง",
    "timing": "เวลาต่อคำขอ",
}


def _fidelity_lines(r: dict | None, u: dict | None) -> list[str]:
    if not r and not u:
        return []
    lines = ["", "## ความตรงกับตอนเตรียมข้อมูล", ""]
    if r:
        tol = r["tolerance"]
        d = r["max_diff"]
        lines += [
            para(
                f"สำเนา fixed ชุดที่ 1 ของภาพ test ของ SCUT {r['images']} ภาพ สร้างใหม่จาก seed แล้วส่งเป็น PNG ในโหมดทดลอง ",
                "เทียบกับ `test_predictions.csv` ของ `train_ratio_heads.py` เกณฑ์คือครึ่งของหลักสุดท้ายใน JSON ",
                "บวกความคลาดจาก feature แบบ float16 ตอนเทรน",
            ),
            "",
            "| ค่า | คลาดสูงสุด | เกณฑ์ |",
            "| --- | --- | --- |",
            f"| กราม value (จุดเปอร์เซ็นต์) | {f(d['jaw_value'])} | {f(tol['%'])} |",
            f"| กราม landmark_value (จุดเปอร์เซ็นต์) | {f(d['jaw_landmark'])} | {f(tol['%'])} |",
            f"| ฐานจมูก value | {f(d['nose_value'], 5)} | {f(tol['ratio'], 5)} |",
            f"| ฐานจมูก landmark_value | {f(d['nose_landmark'], 5)} | {f(tol['ratio'], 5)} |",
            "",
            f"รหัสใน `warnings` ตรงกับรหัสที่เก็บไว้: {f(r['same_codes'])}",
        ]
    if u:
        lines += [
            "",
            para(
                f"ใต้ตา: สำเนา fixed ชุดที่ 1 ของภาพ test ที่เกรดแล้ว {u['images']} ภาพ ({u['sides']} ข้าง) ",
                "เทียบ P(เกรด ≥ 1) และ P(เกรด ≥ 2) กับ `test_predictions.csv` ของ `train_undereye_head.py` ",
                f"คลาดสูงสุด {f(u['max_diff'], 5)} เกณฑ์ {f(u['tolerance'], 5)} รหัสใน `warnings` ตรงกัน: {f(u['same_codes'])}",
            ),
        ]
    return lines


def _turn_lines(e: dict | None, pf: dict | None) -> list[str]:
    if not e and not pf:
        return []
    lines = ["", "## การหมุนและการกลับภาพ", ""]
    if e:
        lines += [
            para(
                f"ภาพหน้าตรง test ของ SCUT คู่กับภาพด้านข้างเต็มของ FEI {e['pairs']} คู่ ที่ผ่านเกณฑ์คุณภาพทั้งคู่ ",
                "หมุนพิกเซลแล้วใส่ EXIF orientation 3, 6 และ 8 เทียบกับภาพตั้งตรง ",
                f"รวม {e['requests']} คำขอ ภาพที่ถอดรหัสแล้วตั้งตรงขนาดเดิม: {f(e['upright_shape_kept'])}",
            ),
            "",
            "| ภาพที่ส่ง | กรามคลาดสูงสุด (จุดเปอร์เซ็นต์) | มุมคลาดสูงสุด (องศา) |",
            "| --- | --- | --- |",
            f"| PNG ใส่ EXIF เทียบ PNG ตั้งตรง | {f(e['png_max_jaw_pp'], 2)} | {f(e['png_max_angle_deg'], 2)} |",
            para(
                f"| JPEG {EXIF_JPEG_QUALITY} ใส่ EXIF เทียบ JPEG {EXIF_JPEG_QUALITY} ตั้งตรง ",
                f"| {f(e['jpeg_max_jaw_pp'], 2)} | {f(e['jpeg_max_angle_deg'], 2)} |",
            ),
            para(
                f"| JPEG {EXIF_JPEG_QUALITY - 1} ตั้งตรงเทียบ JPEG {EXIF_JPEG_QUALITY} ตั้งตรง (ไม่หมุน) ",
                f"| {f(e['resave_max_jaw_pp'], 2)} | {f(e['resave_max_angle_deg'], 2)} |",
            ),
            f"| เกณฑ์ | {f(EXIF_JAW_PP, 2)} | {f(ANGLE_DEG, 2)} |",
            "",
            para(
                f"PNG ใส่ EXIF ได้ผลเหมือนภาพตั้งตรงทุกค่ายกเว้น `request_id`: {f(e['png_same_result'])} ",
                "ซึ่งเป็นการตรวจขั้นหมุนตาม EXIF ของ api โดยตรง ส่วน JPEG อยู่ในเกณฑ์: ",
                f"{f(e['jpeg_within_tolerance'])} ความต่างของ JPEG มาจากการบีบอัดซ้ำ ไม่ใช่การหมุน ",
                "แถวที่บันทึกภาพตั้งตรงซ้ำโดยไม่หมุนจึงใช้เทียบขนาดของผลนั้น",
            ),
        ]
    if pf:
        lg, nf = pf["legan"] or {}, pf["nasofacial"] or {}
        lines += [
            "",
            para(
                f"ภาพด้านข้างเต็มของ FEI {pf['images']} ภาพ กลับซ้ายขวาแล้วส่งเป็น PNG วัดได้ทั้งสองแบบ {pf['measured_both']} ภาพ ",
                f"วัดได้แบบเดียว {pf['measured_one_only']} ภาพ วัดไม่ได้ทั้งคู่ {pf['measured_neither']} ภาพ ",
                f"ทิศที่หาได้กลับตามภาพ: {f(pf['direction_flips'])} เกณฑ์คือมุมคลาดไม่เกิน {f(ANGLE_DEG, 1)}° ",
                f"ซึ่งผ่าน {f(pf['share_within'] * 100 if pf['share_within'] is not None else None, 0)}% ของค่าที่วัดได้ทั้งสองแบบ",
            ),
            "",
            "| มุม | มัธยฐาน (องศา) | p90 | สูงสุด |",
            "| --- | --- | --- | --- |",
            f"| Legan | {f(lg.get('median'), 2)} | {f(lg.get('p90'), 2)} | {f(lg.get('max'), 2)} |",
            f"| nasofacial | {f(nf.get('median'), 2)} | {f(nf.get('p90'), 2)} | {f(nf.get('max'), 2)} |",
        ]
    return lines


def _quality_lines(q: dict | None) -> list[str]:
    if not q:
        return []
    rows = (
        ("face_too_small", f"SCUT test ย่อจน IOD เหลือ {SMALL_IOD_FRACTION:g} เท่าของเกณฑ์"),
        ("frontal_pose", f"SCUT test หมุน {POSE_ROLL_DEG:g}°"),
        ("smiling", "FEI ภาพยิ้ม ต้องได้เกินครึ่ง"),
        ("profile_invalid", "FEI ภาพหน้าตรงที่ส่งเป็นภาพด้านข้าง"),
        ("missing_input", "คำขอที่ไม่มีภาพด้านข้าง"),
    )
    lines = ["", "## รหัสคุณภาพภาพ", "", "| รหัส | ภาพที่ใช้ | ภาพ | ได้รหัส | ผล |", "| --- | --- | --- | --- | --- |"]
    for code, used in rows:
        c = q[code]
        lines.append(f"| {code} | {used} | {c['images']} | {c['found']} | {f(c['passed'])} |")
    return lines


def _lab_lines(lab: dict | None) -> list[str]:
    if not lab or not lab.get("found"):
        return []
    return [
        "",
        "## โหมดทดลอง",
        "",
        para(
            f"ภาพยิ้มของ FEI ที่ไม่ผ่านเกณฑ์ บริเวณที่ประเมินไม่ได้ในโหมดปกติ {lab['regions_failing']} บริเวณ ",
            f"ในโหมดทดลองได้ `assessed` พร้อม `warnings` {lab['assessed_in_lab']} บริเวณ ",
            f"`retake` มี {lab['retake_in_lab']} รายการ และ `lab` เป็น {str(lab['lab']).lower()}",
        ),
    ]


def _timing_lines(t: dict | None) -> list[str]:
    if not t:
        return []
    return [
        "",
        "## เวลาต่อคำขอ",
        "",
        para(
            f"{t['requests']} คำขอบนเครื่องนี้ ({t['device']}) แต่ละคำขอมีภาพหน้าตรงและภาพด้านข้างของ FEI คนละคน ",
            f"วางกลางภาพ {t['frame'][0]}×{t['frame'][1]} เป็น JPEG คุณภาพ {TIMING_JPEG_QUALITY} ให้ใกล้ภาพจากกล้อง ",
            "วัดหลังอุ่นเครื่อง ตั้งแต่ส่งจนได้ผล รวมการถอดรหัสภาพ MediaPipe DINOv2 กฎ และการตรวจผล",
        ),
        "",
        "| p50 | p90 | เฉลี่ย | สูงสุด | หน้าเว็บรอ |",
        "| --- | --- | --- | --- | --- |",
        para(
            f"| {f(t['p50_s'], 2)} วินาที | {f(t['p90_s'], 2)} วินาที | {f(t['mean_s'], 2)} วินาที ",
            f"| {f(t['max_s'], 2)} วินาที | {WEB_TIMEOUT_S} วินาที |",
        ),
    ]


def report(results: dict, skipped: dict, info: dict) -> str:
    lines = [
        "# ตรวจ api กับเครื่องจริง",
        "",
        para(
            "สร้างด้วย `uv run python model/scripts/api_check.py` ซึ่งเรียก api ผ่าน `TestClient` ด้วย engine จริง ",
            "(โมเดลใน `api/assets/` และ `api/bundle/`) รายงานนี้มีเฉพาะตัวเลข ไม่มีภาพหรือชื่อไฟล์ ",
            "ใช้ภาพจาก SCUT-FBP5500 ภาพที่เกรดแล้ว และ FEI เท่านั้น (model.md หัวข้อ 5.1)",
        ),
        "",
        f"- api {info['api']} bundle {info['bundle']} หัวที่เป็น stub: {info['stubs'] or 'ไม่มี'}",
        f"- อุปกรณ์ {info['device']} บน {info['machine']} Python {info['python']} mediapipe {info['mediapipe']} torch {info['torch']}",
        f"- สร้างเมื่อ {info['created']}",
        "",
        "| การตรวจ | ผล |",
        "| --- | --- |",
    ]
    for key, title in TITLES.items():
        state = f"ข้าม: {skipped[key]}" if key in skipped else f(results[key]["passed"])
        lines.append(f"| {title} | {state} |")
    lines += _fidelity_lines(results.get("ratio_fidelity"), results.get("undereye_fidelity"))
    rep = results.get("repeat")
    if rep:
        lines += ["", "## ส่งคำขอเดิมซ้ำ", "", f"ส่งคำขอเดียวกัน {rep['requests']} ครั้ง ผลเหมือนกันทุกค่ายกเว้น `request_id`: {f(rep['same'])}"]
    lines += _turn_lines(results.get("exif"), results.get("profile_flip"))
    lines += _quality_lines(results.get("quality_codes"))
    lines += _lab_lines(results.get("lab"))
    lines += _timing_lines(results.get("timing"))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the api with the real engine and write model/reports/api_check.md")
    parser.add_argument("--only", nargs="+", choices=list(CHECKS), help="run only these checks (the report then lists the rest as skipped)")
    args = parser.parse_args(argv)
    from facemeasure import bundle

    wanted = args.only or list(CHECKS)
    skipped = {k: v for k, v in missing_inputs().items() if k in wanted}
    skipped.update({k: "not asked for (--only)" for k in CHECKS if k not in wanted})
    results = {}
    with real_client() as client:
        skipped.update({k: v for k, v in stub_reasons(client).items() if k not in skipped})
        engine = engine_of(client)
        for name, check in CHECKS.items():
            if name in skipped:
                print(f"{name}: skipped ({skipped[name]})")
                continue
            start = time.perf_counter()
            results[name] = check(client)
            print(f"{name}: {'passed' if results[name]['passed'] else 'FAILED'} in {time.perf_counter() - start:.0f} s")
        runtime = bundle.runtime_versions()
        info = {
            "api": client.app.version, "bundle": engine.bundle.version, "stubs": ", ".join(engine.stub_heads),
            "device": engine.device, "machine": f"{platform.system()} {platform.machine()}", "python": runtime["python"],
            "mediapipe": runtime["mediapipe"], "torch": runtime["torch"],
            "created": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        }
    out = paths.ensure_dir("reports") / REPORT_NAME
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report(results, skipped, info))
    print(f"wrote {out}")
    return 0 if all(r["passed"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
