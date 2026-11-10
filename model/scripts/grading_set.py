"""Build the under-eye grading set (model.md sections 5.2 and 5.3).

Subcommands, run from the project root:

    uv run python model/scripts/grading_set.py select
    uv run python model/scripts/grading_set.py download
    uv run python model/scripts/grading_set.py check
    uv run python model/scripts/grading_set.py replace 00123.png AF45.jpg --reason "hand covers eye"
    uv run python model/scripts/grading_set.py package

select picks 450 FFHQ and 150 SCUT-FBP5500 images to grade, 24 practice images and
ordered reserves, then writes data/grading/manifest.csv. download fetches the FFHQ
images from Google Drive. check verifies every file, swaps grayscale images for
reserves and draws contact sheets. replace swaps images the user rejects. package
builds the folders and zips sent to the graders.

The selection logic below works on tables only, so tests run without images.
Every random choice is a sha256 ranking of a fixed seed and the file name, which
gives the same result on every machine and library version.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
import tomllib
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
import requests
from gdown.exceptions import DownloadCancelled, DownloadError, FileURLRetrievalError
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from facemeasure import paths

SEED = "grading-set-v1"

# FFHQ-Aging labels and FFHQ metadata
LABELS_URL = "https://raw.githubusercontent.com/royorel/FFHQ-Aging-Dataset/master/ffhq_aging_labels.csv"
LABELS_NAME = "ffhq_aging_labels.csv"
LABEL_COLUMNS = [
    "image_number", "age_group", "age_group_confidence", "gender", "gender_confidence",
    "head_pitch", "head_roll", "head_yaw", "left_eye_occluded", "right_eye_occluded", "glasses",
]
META_ID = "16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA"
META_MD5 = "425ae20f06a4da1d4dc0f46d40ba5fd6"
META_NAME = "ffhq-dataset-v2.json"
META_BROWSER_URL = f"https://drive.google.com/open?id={META_ID}"
FFHQ_SIZE = (1024, 1024)

# Selection criteria (model.md section 5.2; pose limit from api.md section 4)
AGE_BANDS = {"20-29": "20-29", "30-39": "30-39", "40-49": "40-49", "50-69": "50+", "70-120": "50+"}
MIN_AGE_CONFIDENCE = 0.6
MAX_EYE_OCCLUSION = 1.0  # left/right_eye_occluded run 0-100; keep values below this
MAX_POSE_DEG = 10.0
SCUT_NAME = re.compile(r"^(AF|AM)(\d+)\.jpg$")
SCUT_MAX_NUMBER = 2000
SCUT_SEX = {"AF": "female", "AM": "male"}


@dataclass(frozen=True)
class Quota:
    graded: int
    practice: int
    reserve: int
    pilot: int


# Subgroups are (source, age_band, sex). FFHQ age bands take 113/113/112/112 images; 113
# cannot split evenly, so 20-29 and 30-39 give the extra image to opposite sexes and the
# 450 FFHQ images stay 225/225. Reserves are 15% rounded up; the pilot share is one third
# of each subgroup, with the leftover images going to the oldest bands.
QUOTAS: dict[tuple[str, str, str], Quota] = {
    ("ffhq", "20-29", "female"): Quota(graded=57, practice=3, reserve=9, pilot=19),
    ("ffhq", "20-29", "male"): Quota(graded=56, practice=2, reserve=9, pilot=18),
    ("ffhq", "30-39", "female"): Quota(graded=56, practice=2, reserve=9, pilot=18),
    ("ffhq", "30-39", "male"): Quota(graded=57, practice=3, reserve=9, pilot=19),
    ("ffhq", "40-49", "female"): Quota(graded=56, practice=2, reserve=9, pilot=19),
    ("ffhq", "40-49", "male"): Quota(graded=56, practice=2, reserve=9, pilot=19),
    ("ffhq", "50+", "female"): Quota(graded=56, practice=2, reserve=9, pilot=19),
    ("ffhq", "50+", "male"): Quota(graded=56, practice=2, reserve=9, pilot=19),
    ("scut", "", "female"): Quota(graded=75, practice=3, reserve=12, pilot=25),
    ("scut", "", "male"): Quota(graded=75, practice=3, reserve=12, pilot=25),
}
# train/val/test 70/15/15 in every subgroup, 420/90/90 overall (model.md section 5.3)
SPLIT_PERCENT = {"train": 70, "val": 15, "test": 15}

MANIFEST_NAME = "manifest.csv"
MANIFEST_COLUMNS = [
    "image", "source", "source_id", "sex", "age_group", "age_band", "role", "pilot",
    "split", "reason", "file_md5", "license", "author", "photo_url",
]
ROLES = ("graded", "practice", "reserve", "dropped")
SPLITS = ("train", "val", "test")

# check: mean HSV saturation of the central half of the image. Whole-image means mix in
# white backgrounds; on SCUT the centre separates grayscale (0.00-0.03) from colour (>= 0.07).
CENTER_FRACTION = 0.5
MIN_SATURATION = 0.05
CONTACT_COLUMNS = 8
CONTACT_ROWS = 6
CONTACT_THUMB = 200
CONTACT_LABEL = 24

# download
DOWNLOAD_WORKERS = 4
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_BASE_DELAY = 5.0  # seconds; doubles after each failed attempt
DOWNLOAD_TIMEOUT = 60.0
QUOTA_HINTS = ("too many users", "many accesses", "quota")

# package
SHARE_SETS = {"practice": 24, "pilot_200": 200, "full_600": 600}
GRADER_NAME = "undereye_grader.html"
SHARE_README = Path(__file__).with_name("grading_share_readme.toml")
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------- selection


def rank(salt: str, name: str) -> str:
    """Seeded sort key: the same name always lands in the same place."""
    return hashlib.sha256(f"{SEED}|{salt}|{name}".encode()).hexdigest()


def read_labels(path: Path) -> pd.DataFrame:
    """Read the FFHQ-Aging labels; keep_default_na=False keeps glasses == "None" as text."""
    labels = pd.read_csv(path, keep_default_na=False, dtype={"age_group": str, "gender": str, "glasses": str})
    missing = [c for c in LABEL_COLUMNS if c not in labels.columns]
    if missing:
        raise ValueError(f"{path} has no column {', '.join(missing)}")
    return labels


def eligible_ffhq(labels: pd.DataFrame) -> pd.DataFrame:
    """Adults with confident age labels, no glasses, open eyes and a near-frontal pose."""
    mask = (
        labels["age_group"].isin(list(AGE_BANDS))
        & (labels["age_group_confidence"].astype(float) >= MIN_AGE_CONFIDENCE)
        & (labels["glasses"] == "None")
        & (labels["left_eye_occluded"].astype(float) < MAX_EYE_OCCLUSION)
        & (labels["right_eye_occluded"].astype(float) < MAX_EYE_OCCLUSION)
        & (labels["head_yaw"].astype(float).abs() <= MAX_POSE_DEG)
        & (labels["head_pitch"].astype(float).abs() <= MAX_POSE_DEG)
        & labels["gender"].isin(["female", "male"])
    )
    return labels[mask]


def ffhq_image_name(number: int) -> str:
    return f"{number:05d}.png"


def ffhq_file_path(number: int) -> str:
    """Relative path of an FFHQ 1024x1024 image, as image.file_path in ffhq-dataset-v2.json."""
    return f"images1024x1024/{number // 1000 * 1000:05d}/{number:05d}.png"


def scut_asian_names(names: Iterable[str]) -> list[str]:
    """Keep AF1-AF2000.jpg and AM1-AM2000.jpg."""
    keep = []
    for name in names:
        m = SCUT_NAME.match(name)
        if m and 1 <= int(m.group(2)) <= SCUT_MAX_NUMBER:
            keep.append(name)
    return keep


def candidate_pools(labels: pd.DataFrame, scut_names: Iterable[str]) -> dict[tuple[str, str, str], list[dict]]:
    pools: dict[tuple[str, str, str], list[dict]] = {key: [] for key in QUOTAS}
    for row in eligible_ffhq(labels).itertuples(index=False):
        number = int(row.image_number)
        band = AGE_BANDS[row.age_group]
        pools[("ffhq", band, row.gender)].append({
            "image": ffhq_image_name(number), "source": "ffhq", "source_id": str(number),
            "sex": row.gender, "age_group": row.age_group, "age_band": band,
        })
    for name in scut_asian_names(scut_names):
        stem = name.removesuffix(".jpg")
        sex = SCUT_SEX[stem[:2]]
        pools[("scut", "", sex)].append({
            "image": name, "source": "scut", "source_id": stem,
            "sex": sex, "age_group": "", "age_band": "",
        })
    return pools


def allocate_splits(sizes: dict, percent: dict[str, int] = SPLIT_PERCENT) -> dict:
    """Split each stratum 70/15/15 so the overall totals are exact.

    val and test get floor shares, then the leftover images go to the largest remainders.
    On ties a stratum that already got an extra test image is skipped for val, so train
    stays as even as it can.
    """
    total = sum(sizes.values())
    order = {key: i for i, key in enumerate(sizes)}
    counts = {key: {} for key in sizes}
    bumped: set = set()
    for split in ("test", "val"):
        target = total * percent[split] // 100
        base = {key: n * percent[split] // 100 for key, n in sizes.items()}
        remainder = {key: n * percent[split] % 100 for key, n in sizes.items()}
        extra = target - sum(base.values())
        for key in sorted(sizes, key=lambda k: (-remainder[k], k in bumped, order[k]))[:extra]:
            base[key] += 1
            bumped.add(key)
        for key in sizes:
            counts[key][split] = base[key]
    for key, n in sizes.items():
        counts[key]["train"] = n - counts[key]["val"] - counts[key]["test"]
    return counts


def subgroup(row) -> tuple[str, str, str]:
    return (row["source"], row["age_band"], row["sex"])


def select_images(labels: pd.DataFrame, scut_names: Iterable[str]) -> pd.DataFrame:
    """Pick graded, practice and reserve images and assign pilot and split.

    Within a subgroup the candidates are ranked once; graded images come first, then
    practice, then reserves in the order replace uses them. Metadata columns stay empty.
    """
    pools = candidate_pools(labels, scut_names)
    split_counts = allocate_splits({key: q.graded for key, q in QUOTAS.items()})
    rows = []
    for key, quota in QUOTAS.items():
        pool = sorted(pools[key], key=lambda r: rank("pick", r["image"]))
        need = quota.graded + quota.practice + quota.reserve
        if len(pool) < need:
            raise ValueError(f"subgroup {key} has {len(pool)} candidates, needs {need}")
        graded = pool[:quota.graded]
        practice = pool[quota.graded:quota.graded + quota.practice]
        reserve = pool[quota.graded + quota.practice:need]
        pilot = {r["image"] for r in sorted(graded, key=lambda r: rank("pilot", r["image"]))[:quota.pilot]}
        by_split = sorted(graded, key=lambda r: rank("split", r["image"]))
        split_of = {}
        start = 0
        for split in ("test", "val", "train"):
            for r in by_split[start:start + split_counts[key][split]]:
                split_of[r["image"]] = split
            start += split_counts[key][split]
        for role, members in (("graded", graded), ("practice", practice), ("reserve", reserve)):
            for r in members:
                rows.append({
                    **r, "role": role,
                    "pilot": "1" if r["image"] in pilot else "0",
                    "split": split_of.get(r["image"], ""),
                })
    manifest = pd.DataFrame(rows)
    order = {img: rank("pick", img) for img in manifest["image"]}
    key_order = {key: i for i, key in enumerate(QUOTAS)}
    manifest["_key"] = [key_order[subgroup(r)] for _, r in manifest.iterrows()]
    manifest["_rank"] = manifest["image"].map(order)
    manifest = manifest.sort_values(["_key", "_rank"]).drop(columns=["_key", "_rank"])
    for column in MANIFEST_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = ""
    return manifest[MANIFEST_COLUMNS].reset_index(drop=True)


def fill_metadata(manifest: pd.DataFrame, ffhq_meta: dict[str, dict], scut_md5: dict[str, str]) -> pd.DataFrame:
    """Add file_md5 and the FFHQ owner columns; SCUT md5 comes from the local file."""
    out = manifest.copy()
    for i, row in out.iterrows():
        if row["source"] == "ffhq":
            meta = ffhq_meta[row["source_id"]]
            for column in ("file_md5", "license", "author", "photo_url"):
                out.at[i, column] = meta[column]
        else:
            out.at[i, "file_md5"] = scut_md5[row["image"]]
    return out


def replace_images(manifest: pd.DataFrame, images: Iterable[str], reason: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Swap each image for the next reserve of its subgroup.

    The reserve takes over the role, split and pilot flag of the image it replaces; the
    replaced image becomes dropped. Returns the new manifest and {old: new}.
    """
    out = manifest.copy()
    swaps = {}
    for image in images:
        hits = out.index[out["image"] == image]
        if len(hits) != 1:
            raise ValueError(f"{image} is not in the manifest")
        old = hits[0]
        role = out.at[old, "role"]
        if role not in ("graded", "practice"):
            raise ValueError(f"{image} is {role}; only graded or practice images can be replaced")
        key = subgroup(out.loc[old])
        reserves = [
            i for i in out.index
            if out.at[i, "role"] == "reserve" and subgroup(out.loc[i]) == key
        ]
        if not reserves:
            raise ValueError(f"no reserve left for {image} in subgroup {key}")
        new = min(reserves, key=lambda i: rank("pick", out.at[i, "image"]))
        new_image = out.at[new, "image"]
        for column in ("role", "pilot", "split"):
            out.at[new, column] = out.at[old, column]
        out.at[new, "reason"] = f"replaces {image} ({reason})"
        history = out.at[old, "reason"]
        dropped_reason = f"{reason}; replaced by {new_image}"
        out.at[old, "reason"] = f"{history}; {dropped_reason}" if history else dropped_reason
        out.at[old, "role"] = "dropped"
        out.at[old, "pilot"] = "0"
        out.at[old, "split"] = ""
        swaps[image] = new_image
    return out, swaps


def summary_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Counts by source, age band and sex: roles, pilot images and splits of graded images."""
    rows = []
    for key in QUOTAS:
        source, band, sex = key
        part = manifest[(manifest["source"] == source) & (manifest["age_band"] == band) & (manifest["sex"] == sex)]
        graded = part[part["role"] == "graded"]
        rows.append({
            "source": source, "age_band": band or "-", "sex": sex,
            "graded": len(graded), "pilot": int((graded["pilot"] == "1").sum()),
            **{s: int((graded["split"] == s).sum()) for s in SPLITS},
            **{r: int((part["role"] == r).sum()) for r in ROLES if r != "graded"},
        })
    table = pd.DataFrame(rows)
    total = {"source": "total", "age_band": "", "sex": ""}
    total.update({c: int(table[c].sum()) for c in table.columns if c not in total})
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


# ---------------------------------------------------------------- manifest files


def manifest_path() -> Path:
    return paths.path("grading") / MANIFEST_NAME


def manifest_text(manifest: pd.DataFrame) -> str:
    return manifest[MANIFEST_COLUMNS].to_csv(index=False, lineterminator="\n")


def write_manifest(manifest: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(manifest_text(manifest))
    os.replace(tmp, path)


def read_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    return manifest[MANIFEST_COLUMNS]


def image_path(row) -> Path:
    if row["source"] == "ffhq":
        return paths.path("ffhq").joinpath(*ffhq_file_path(int(row["source_id"])).split("/"))
    return paths.path("scut") / "Images" / row["image"]


def file_md5(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "md5").hexdigest()


# ---------------------------------------------------------------- FFHQ metadata


class MetadataMissing(Exception):
    """The labels CSV or ffhq-dataset-v2.json is not usable; the message tells the user what to do."""


def staging_dir() -> Path:
    """Partial downloads live here until their md5 is right, so data/raw/ never holds half a file."""
    return paths.ensure_dir("grading", "download_tmp")


def remove_staging_if_empty() -> None:
    folder = paths.path("grading") / "download_tmp"
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()


def ensure_labels() -> Path:
    dest = paths.ensure_dir("ffhq") / LABELS_NAME
    if dest.is_file():
        return dest
    print(f"กำลังโหลดป้าย FFHQ-Aging จาก {LABELS_URL}")
    try:
        res = requests.get(LABELS_URL, timeout=DOWNLOAD_TIMEOUT)
        res.raise_for_status()
    except requests.RequestException as e:
        raise MetadataMissing(
            f"โหลดป้าย FFHQ-Aging ไม่ได้ ({e})\n"
            f"เปิด {LABELS_URL} ใน browser แล้วบันทึกเป็น {dest}"
        ) from e
    tmp = staging_dir() / LABELS_NAME
    tmp.write_bytes(res.content)
    os.replace(tmp, dest)
    return dest


def ensure_meta() -> Path:
    dest = paths.ensure_dir("ffhq") / META_NAME
    manual = (
        f"เปิด {META_BROWSER_URL} ใน browser ที่ล็อกอิน Google แล้ว\n"
        f"โหลดไฟล์ {META_NAME} (255 MB) มาวางที่ {dest} แล้วรันคำสั่งเดิมอีกครั้ง"
    )
    if dest.is_file():
        if file_md5(dest) != META_MD5:
            raise MetadataMissing(
                f"{dest} มี md5 ไม่ตรงกับ {META_MD5} ไฟล์อาจโหลดไม่ครบ\n"
                f"ลบไฟล์นี้เอง (สคริปต์ไม่ลบไฟล์ใน data/raw/) แล้ว{manual}"
            )
        return dest
    tmp = staging_dir() / META_NAME
    remove_partial(tmp)
    print(f"กำลังโหลด {META_NAME} (255 MB) จาก Google Drive")
    try:
        gdown.download(id=META_ID, output=str(tmp), quiet=False, timeout=DOWNLOAD_TIMEOUT)
    except Exception as e:
        remove_partial(tmp)
        raise MetadataMissing(f"โหลด {META_NAME} ไม่ได้ ({type(e).__name__})\n{manual}") from e
    if not tmp.is_file() or file_md5(tmp) != META_MD5:
        remove_partial(tmp)
        raise MetadataMissing(f"ไฟล์ {META_NAME} ที่โหลดได้มี md5 ไม่ตรง\n{manual}")
    os.replace(tmp, dest)
    return dest


def load_ffhq_meta(path: Path) -> dict[str, dict[str, str]]:
    """Keep only the fields the grading set needs from ffhq-dataset-v2.json."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        key: {
            "file_url": item["image"]["file_url"],
            "file_path": item["image"]["file_path"],
            "file_md5": item["image"]["file_md5"],
            "license": item["metadata"]["license"],
            "author": item["metadata"]["author"],
            "photo_url": item["metadata"]["photo_url"],
        }
        for key, item in data.items()
    }


# ---------------------------------------------------------------- download


# network trouble, Drive pages gdown cannot parse and short files: try again after a wait
RETRYABLE = (requests.RequestException, OSError, DownloadError, FileURLRetrievalError, RuntimeError)


class QuotaExceeded(Exception):
    """Google Drive refuses the file for now; every worker stops."""


class Cancelled(Exception):
    pass


@dataclass
class DownloadJob:
    image: str
    url: str
    dest: Path
    md5: str


@dataclass
class DownloadReport:
    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    not_started: list[str] = field(default_factory=list)
    quota_message: str = ""


Fetch = Callable[[str, Path, threading.Event], None]


def remove_partial(tmp: Path) -> None:
    """Remove a staged download and the <name>XXXXXXXX.part files gdown leaves next to it."""
    tmp.unlink(missing_ok=True)
    for part in tmp.parent.glob(f"{tmp.name}*.part"):
        part.unlink(missing_ok=True)


def gdown_fetch(url: str, out: Path, stop: threading.Event) -> None:
    try:
        result = gdown.download(url=url, output=str(out), quiet=True, timeout=DOWNLOAD_TIMEOUT, cancel=stop)
    except FileURLRetrievalError as e:
        if any(hint in str(e).lower() for hint in QUOTA_HINTS):
            raise QuotaExceeded(str(e)) from e
        raise
    except DownloadCancelled as e:
        raise Cancelled from e
    if result is None:
        raise RuntimeError("gdown returned no file")


def download_all(
    jobs: list[DownloadJob],
    fetch: Fetch,
    staging: Path,
    workers: int = DOWNLOAD_WORKERS,
    attempts: int = DOWNLOAD_ATTEMPTS,
    base_delay: float = DOWNLOAD_BASE_DELAY,
) -> DownloadReport:
    """Download missing files, at most `workers` at a time, with growing waits between attempts.

    A file already in place with the right md5 is skipped; one with the wrong md5 is
    reported and left alone. A file moves into place only after its md5 matches. A quota
    error stops every worker and keeps what is already done.
    """
    report = DownloadReport()
    todo = []
    for job in jobs:
        if job.dest.is_file():
            (report.skipped if file_md5(job.dest) == job.md5 else report.conflicts).append(job.image)
        else:
            todo.append(job)
    staging.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def one(job: DownloadJob) -> tuple[str, str]:
        tmp = staging / job.dest.name
        last = ""
        for attempt in range(attempts):
            if stop.is_set():
                return "not_started", ""
            remove_partial(tmp)
            try:
                fetch(job.url, tmp, stop)
            except QuotaExceeded as e:
                stop.set()
                remove_partial(tmp)
                return "quota", str(e)
            except Cancelled:
                remove_partial(tmp)
                return "not_started", ""
            except RETRYABLE as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if tmp.is_file() and file_md5(tmp) == job.md5:
                    job.dest.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(tmp, job.dest)
                    return "done", ""
                last = "md5 mismatch"
            remove_partial(tmp)
            if attempt + 1 < attempts and stop.wait(base_delay * 2 ** attempt):
                return "not_started", ""
        return "failed", last

    with ThreadPoolExecutor(max_workers=workers) as pool, tqdm(total=len(todo), unit="file") as bar:
        futures = {pool.submit(one, job): job for job in todo}
        for future in as_completed(futures):
            job = futures[future]
            status, message = future.result()
            if status == "done":
                report.done.append(job.image)
            elif status == "failed":
                report.failed[job.image] = message
            elif status == "quota":
                report.quota_message = report.quota_message or message
                report.not_started.append(job.image)
            else:
                report.not_started.append(job.image)
            bar.update()
    for attr in ("done", "skipped", "conflicts", "not_started"):
        getattr(report, attr).sort()
    return report


# ---------------------------------------------------------------- check


def center_saturation(image: Image.Image) -> float:
    """Mean HSV saturation (0-1) of the central CENTER_FRACTION of the image."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    mx, my = round(w * (1 - CENTER_FRACTION) / 2), round(h * (1 - CENTER_FRACTION) / 2)
    center = rgb.crop((mx, my, w - mx, h - my))
    center.thumbnail((256, 256))
    return float(np.asarray(center.convert("HSV"))[..., 1].mean()) / 255


@dataclass
class ImageCheck:
    image: str
    problem: str = ""
    saturation: float = math.nan


def check_image(image: str, path: Path, md5: str, size: tuple[int, int] | None) -> ImageCheck:
    if not path.is_file():
        return ImageCheck(image, "missing")
    if file_md5(path) != md5:
        return ImageCheck(image, "md5 mismatch")
    try:
        with Image.open(path) as im:
            im.load()
            if size and im.size != size:
                return ImageCheck(image, f"size {im.size[0]}x{im.size[1]}")
            return ImageCheck(image, "", center_saturation(im))
    except OSError as e:
        return ImageCheck(image, f"unreadable ({e})")


def check_rows(rows: pd.DataFrame) -> list[ImageCheck]:
    results = []
    for _, row in tqdm(list(rows.iterrows()), unit="image", leave=False):
        size = FFHQ_SIZE if row["source"] == "ffhq" else None
        results.append(check_image(row["image"], image_path(row), row["file_md5"], size))
    return results


def contact_sheets(rows: pd.DataFrame, out_dir: Path, prefix: str, problems: dict[str, str] | None = None) -> list[Path]:
    """Pages of CONTACT_COLUMNS x CONTACT_ROWS thumbnails with the file name under each.

    Images listed in problems get a gray box with the problem instead, so review can
    start before every download is done.
    """
    problems = problems or {}
    per_page = CONTACT_COLUMNS * CONTACT_ROWS
    cell_w, cell_h = CONTACT_THUMB, CONTACT_THUMB + CONTACT_LABEL
    font = ImageFont.load_default(size=15)
    rows = rows.sort_values(["source", "age_band", "sex", "image"])
    pages = []
    for page_no, start in enumerate(range(0, len(rows), per_page), start=1):
        sheet = Image.new("RGB", (CONTACT_COLUMNS * cell_w, CONTACT_ROWS * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (_, row) in enumerate(rows.iloc[start:start + per_page].iterrows()):
            x, y = (i % CONTACT_COLUMNS) * cell_w, (i // CONTACT_COLUMNS) * cell_h
            if row["image"] in problems:
                draw.rectangle((x + 2, y + 2, x + cell_w - 3, y + CONTACT_THUMB - 3), fill="lightgray")
                draw.text((x + 10, y + CONTACT_THUMB // 2), problems[row["image"]][:24], fill="black", font=font)
            else:
                with Image.open(image_path(row)) as im:
                    thumb = im.convert("RGB")
                thumb.thumbnail((CONTACT_THUMB - 4, CONTACT_THUMB - 4))
                sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 2))
            tag = row["image"]
            if row["source"] == "ffhq":
                tag += f"  {row['sex'][0].upper()} {row['age_band']}"
            draw.text((x + 4, y + CONTACT_THUMB + 3), tag, fill="black", font=font)
        page = out_dir / f"{prefix}_{page_no:02d}.jpg"
        sheet.save(page, quality=90)
        pages.append(page)
    return pages


# ---------------------------------------------------------------- package


def share_sets(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    graded = manifest[manifest["role"] == "graded"]
    return {
        "practice": manifest[manifest["role"] == "practice"],
        "pilot_200": graded[graded["pilot"] == "1"],
        "full_600": graded,
    }


def share_readmes(counts: dict[str, int]) -> dict[str, str]:
    with open(SHARE_README, "rb") as f:
        template = tomllib.load(f)
    return {
        name: template["body"].format(folder=name, count=count, intro=template["intro"][name].strip())
        for name, count in counts.items()
    }


def write_zip(folder: Path, out: Path) -> None:
    """Zip a folder under its own name, with fixed dates so the same files give the same zip."""
    tmp = out.with_name(out.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        for f in sorted(folder.iterdir()):
            info = zipfile.ZipInfo(f"{folder.name}/{f.name}", date_time=ZIP_DATE)
            info.external_attr = 0o644 << 16
            # images are already compressed
            info.compress_type = zipfile.ZIP_STORED if f.suffix in (".png", ".jpg") else zipfile.ZIP_DEFLATED
            with open(f, "rb") as src, zf.open(info, "w") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    os.replace(tmp, out)


def build_share(share: Path, sets: dict[str, list[Path]], grader: Path, readmes: dict[str, str]) -> dict[str, Path]:
    """Copy each set into share/<name>/ with the grader and README.txt, then zip it."""
    share.mkdir(parents=True, exist_ok=True)
    zips = {}
    for name, files in sets.items():
        folder = share / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        for src in files:
            shutil.copyfile(src, folder / src.name)
        shutil.copyfile(grader, folder / GRADER_NAME)
        # utf-8-sig so Notepad and TextEdit both read the Thai text correctly
        with open(folder / "README.txt", "w", encoding="utf-8-sig", newline="\r\n") as f:
            f.write(readmes[name])
        zips[name] = share / f"{name}.zip"
        write_zip(folder, zips[name])
    return zips


# ---------------------------------------------------------------- commands


def cmd_select(args) -> int:
    labels = read_labels(ensure_labels())
    meta_path = ensure_meta()
    scut_images = paths.path("scut") / "Images"
    manifest = select_images(labels, (p.name for p in scut_images.iterdir()))
    print(f"อ่าน {META_NAME}")
    meta = load_ffhq_meta(meta_path)
    scut_md5 = {img: file_md5(scut_images / img) for img in manifest.loc[manifest["source"] == "scut", "image"]}
    manifest = fill_metadata(manifest, meta, scut_md5)
    out = manifest_path()
    if out.is_file():
        with open(out, encoding="utf-8", newline="") as f:
            same = f.read() == manifest_text(manifest)
        if same:
            print(f"{out} เหมือนเดิม ไม่ได้เขียนใหม่")
        elif not args.force:
            print(
                f"{out} ต่างจากผลคัดใหม่ เช่นเคยสั่ง replace หรือ check เปลี่ยนภาพไปแล้ว จึงไม่เขียนทับ\n"
                "ถ้าต้องการเริ่มใหม่จริง ๆ ให้รันด้วย --force ซึ่งจะลบประวัติการเปลี่ยนภาพ"
            )
            return 1
        else:
            write_manifest(manifest, out)
            print(f"เขียนทับ {out}")
    else:
        write_manifest(manifest, out)
        print(f"เขียน {out}")
    print_summary(read_manifest(out))
    return 0


def print_summary(manifest: pd.DataFrame) -> None:
    print(summary_table(manifest).to_string(index=False))
    graded = manifest[manifest["role"] == "graded"]
    pilot = graded[graded["pilot"] == "1"]
    print(
        f"graded: FFHQ {int((graded['source'] == 'ffhq').sum())}, SCUT {int((graded['source'] == 'scut').sum())}"
        f" | practice {int((manifest['role'] == 'practice').sum())}"
        f" | pilot: FFHQ {int((pilot['source'] == 'ffhq').sum())}, SCUT {int((pilot['source'] == 'scut').sum())}"
        f" | split {'/'.join(str(int((graded['split'] == s).sum())) for s in SPLITS)} (train/val/test)"
    )


def cmd_download(args) -> int:
    manifest = read_manifest(manifest_path())
    print(f"อ่าน {META_NAME}")
    meta = load_ffhq_meta(ensure_meta())
    rows = manifest[(manifest["source"] == "ffhq") & manifest["role"].isin(["graded", "practice", "reserve"])]
    # images to grade first, so a quota stop leaves reserves missing rather than graded images
    rows = rows.sort_values("role", key=lambda r: r == "reserve", kind="stable")
    jobs = []
    for _, row in rows.iterrows():
        item = meta[row["source_id"]]
        expected = ffhq_file_path(int(row["source_id"]))
        if item["file_path"] != expected:
            raise ValueError(f"{row['image']}: file_path {item['file_path']} differs from {expected}")
        jobs.append(DownloadJob(row["image"], item["file_url"], image_path(row), item["file_md5"]))
    report = download_all(jobs, gdown_fetch, staging_dir())
    remove_staging_if_empty()
    print(
        f"FFHQ {len(jobs)} ไฟล์: โหลดใหม่ {len(report.done)}, มีอยู่แล้ว {len(report.skipped)}, "
        f"ล้มเหลว {len(report.failed)}, ยังไม่ได้โหลด {len(report.not_started)}, md5 ไม่ตรง {len(report.conflicts)}"
    )
    for image, message in sorted(report.failed.items()):
        print(f"  ล้มเหลว {image}: {message}")
    if report.conflicts:
        print("ไฟล์เหล่านี้มีอยู่แล้วแต่ md5 ไม่ตรง สคริปต์ไม่เขียนทับไฟล์ใน data/raw/ ให้ลบเองแล้วรัน download ใหม่:")
        for image in report.conflicts:
            print(f"  {image_path(rows[rows['image'] == image].iloc[0])}")
    if report.quota_message:
        drive_says = " ".join(report.quota_message.split())[:300]
        print(
            f"\nข้อความจาก Google Drive/gdown: {drive_says}\n"
            "Google Drive แจ้งว่าโควต้าการโหลดเต็ม จึงหยุดไว้ก่อน ไฟล์ที่โหลดเสร็จแล้วเก็บไว้ครบ\n"
            "ลองรัน download ใหม่หลังราว 24 ชั่วโมง สคริปต์จะโหลดต่อจากที่ค้าง\n"
            "หรือใช้วิธี \"Downloading with PyDrive\" ใน README ของ FFHQ-Aging "
            "(https://github.com/royorel/FFHQ-Aging-Dataset) ซึ่งใช้บัญชี Google ของผู้ใช้เอง "
            "แล้ววางภาพตาม image.file_path ใน data/raw/ffhq/ ภาพต้องมี md5 ตรงกับ JSON"
        )
        return 2
    if report.failed:
        print("รัน download ซ้ำเพื่อลองใหม่ ถ้าไฟล์ใดยังโหลดไม่ได้ ให้สั่ง replace ไฟล์นั้น เช่น --reason \"download failed\"")
        return 1
    return 0


def cmd_check(args) -> int:
    path = manifest_path()
    manifest = read_manifest(path)
    in_use = ["graded", "practice"]
    results: dict[str, ImageCheck] = {}
    swapped = []
    pending = list(manifest.loc[manifest["role"].isin(in_use), "image"])
    # a reserve that comes in is checked too, and swapped again if it is also grayscale
    while pending:
        checked = check_rows(manifest[manifest["image"].isin(pending)])
        results.update({r.image: r for r in checked})
        pending = []
        for r in checked:
            if r.problem or r.saturation >= args.min_saturation:
                continue
            try:
                manifest, swap = replace_images(manifest, [r.image], f"grayscale (saturation {r.saturation:.3f})")
            except ValueError as e:
                print(f"เปลี่ยนภาพขาวดำไม่ได้: {e}")
                continue
            swapped.extend(swap.items())
            pending.extend(swap.values())
    if swapped:
        write_manifest(manifest, path)
        print(f"เปลี่ยนภาพขาวดำเป็นภาพสำรอง {len(swapped)} ภาพ แล้วบันทึก {path}")
        for old, new in swapped:
            print(f"  {old} -> {new}")
    current = manifest[manifest["role"].isin(in_use)]
    checks = [results[img] for img in current["image"]]
    for r in checks:
        if not r.problem and r.saturation < args.min_saturation:
            r.problem = f"grayscale (saturation {r.saturation:.3f}), no reserve left"
    problems = [r for r in checks if r.problem]
    measured = sorted((r for r in checks if not r.problem), key=lambda r: r.saturation)
    print(f"ตรวจ {len(checks)} ภาพ (graded {int((current['role'] == 'graded').sum())}, practice {int((current['role'] == 'practice').sum())}): มีปัญหา {len(problems)} ภาพ")
    for r in problems:
        print(f"  {r.image}: {r.problem}")
    print(f"ความอิ่มสีกลางภาพต่ำสุด 10 ภาพ (เกณฑ์ขาวดำ < {args.min_saturation}):")
    for r in measured[:10]:
        print(f"  {r.image}: {r.saturation:.3f}")
    out_dir = paths.ensure_dir("outputs", "grading_contact")
    for old in out_dir.glob("*.jpg"):
        if re.fullmatch(r"(graded|practice)_\d{2}\.jpg", old.name):
            old.unlink()
    problem_of = {r.image: r.problem for r in problems}
    pages = contact_sheets(current[current["role"] == "graded"], out_dir, "graded", problem_of)
    pages += contact_sheets(current[current["role"] == "practice"], out_dir, "practice", problem_of)
    print(f"ทำภาพย่อ {len(pages)} หน้าที่ {out_dir}")
    if problems:
        print(
            f"ยังมี {len(problems)} ภาพที่ใช้ไม่ได้ (กรอบเทาในภาพย่อ) เช่นรัน download ต่อ หรือสั่ง replace "
            "แล้วรัน check ใหม่ ภาพที่ยังไม่ได้โหลดยังไม่ได้ตรวจขาวดำ"
        )
        return 1
    return 0


def cmd_replace(args) -> int:
    path = manifest_path()
    manifest = read_manifest(path)
    try:
        manifest, swaps = replace_images(manifest, args.images, args.reason)
    except ValueError as e:
        print(f"เปลี่ยนภาพไม่ได้: {e} (ยังไม่ได้แก้ manifest)")
        return 1
    write_manifest(manifest, path)
    for old, new in swaps.items():
        row = manifest[manifest["image"] == new].iloc[0]
        print(f"{old} -> {new} (role {row['role']}, split {row['split'] or '-'}, pilot {row['pilot']})")
    print("รัน check เพื่อตรวจภาพใหม่และทำภาพย่อ แล้วรัน package ใหม่")
    return 0


def cmd_package(args) -> int:
    manifest = read_manifest(manifest_path())
    sets = share_sets(manifest)
    wrong = {name: len(rows) for name, rows in sets.items() if len(rows) != SHARE_SETS[name]}
    if wrong:
        print(f"จำนวนภาพไม่ตรง {wrong} ต้องเป็น {SHARE_SETS}")
        return 1
    print("ตรวจภาพก่อนทำโฟลเดอร์")
    rows = pd.concat([sets["practice"], sets["full_600"]])
    bad = [r for r in check_rows(rows) if r.problem or r.saturation < args.min_saturation]
    if bad:
        for r in bad:
            print(f"  {r.image}: {r.problem or f'saturation {r.saturation:.3f}'}")
        print("รัน check ก่อน แล้วจึงรัน package")
        return 1
    share = paths.path("grading") / "share"
    files = {name: [image_path(row) for _, row in part.iterrows()] for name, part in sets.items()}
    zips = build_share(share, files, paths.path("tools") / GRADER_NAME, share_readmes(SHARE_SETS))
    for name, zip_path in zips.items():
        print(f"{share / name}: {SHARE_SETS[name]} ภาพ, {zip_path.name} {zip_path.stat().st_size / 1e6:.0f} MB")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the under-eye grading set")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select", help="pick images and write data/grading/manifest.csv")
    p.add_argument("--force", action="store_true", help="overwrite a manifest that differs, losing replacements")
    sub.add_parser("download", help="download the FFHQ images in the manifest")
    p = sub.add_parser("check", help="verify images, swap grayscale ones, draw contact sheets")
    p.add_argument("--min-saturation", type=float, default=MIN_SATURATION)
    p = sub.add_parser("replace", help="swap images for the next reserve of their subgroup")
    p.add_argument("images", nargs="+", help="file names as in the manifest, e.g. 00123.png AF45.jpg")
    p.add_argument("--reason", default="rejected on review")
    p = sub.add_parser("package", help="build data/grading/share/ for the graders")
    p.add_argument("--min-saturation", type=float, default=MIN_SATURATION)
    args = parser.parse_args(argv)
    command = {
        "select": cmd_select, "download": cmd_download, "check": cmd_check,
        "replace": cmd_replace, "package": cmd_package,
    }[args.command]
    try:
        return command(args)
    except MetadataMissing as e:
        print(e)
        return 1
    finally:
        remove_staging_if_empty()


if __name__ == "__main__":
    sys.exit(main())
