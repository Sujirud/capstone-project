"""Download FEI and check analyze_profile on it (model.md sections 6 and 7.1).

Subcommands, run from the project root:

    uv run python model/scripts/profile_check.py download
    uv run python model/scripts/profile_check.py run

download fetches the four FEI zips into data/raw/fei/, resuming a download that
stopped, records their sha256 in data/raw/fei/checksums.txt and unpacks the 2,800
photos into data/raw/fei/images/. It then runs Face Landmarker on the 14 photos of
the first five people to tell which pose number is which, and draws the 14 photos
of person 1 in data/outputs/fei_numbering.jpg.

run sends the full profiles, the next angles and the frontal photos (neutral and
smile) of all 200 people through analyze_profile, and the frontal photos through
analyze_frontal too. It writes

- labels/fei_profile.csv         one row per photo sent as a profile
- labels/fei_frontal_smile.csv   one row per frontal photo: quality values and every blendshape
- reports/fei_profiles.md        numbers and charts only (figures/fei_*.png), so it can be committed
- outputs/fei_profile_check/     both full profiles of 20 people picked by a fixed seed, and
                                 every full profile that could not be measured; FEI photos, never share

Work is saved PART_PEOPLE people at a time under features/fei_profile_parts/, so a run
that stops goes on from the last finished part. The parts record the pose numbers,
the quality limits, the profile constants and the model files; if any of them
changes the script stops and asks for --restart.

FEI photos must not be shared or copied: everything that shows them stays in data/.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

import mediapipe
import numpy as np
import pandas as pd
import requests
from matplotlib.figure import Figure
from PIL import Image
from tqdm import tqdm

from facemeasure import assets, criteria, draw, fei, image_io, paths, pose, quality
from facemeasure import profile as P
from facemeasure.frontal import analyze_frontal
from facemeasure.landmarks import LANDMARKER_FILE, SEGMENTER_FILE, FaceModels

# ---------------------------------------------------------------- download

DOWNLOAD_TIMEOUT = 60  # seconds without data before requests gives up
DOWNLOAD_ATTEMPTS = 5
CHUNK_BYTES = 1 << 20
PART_SUFFIX = ".part"
STAGING = ("download_tmp", "fei")  # under data/, so data/raw/ never holds half a file

Fetch = Callable[[str, Path], None]


class FeiError(RuntimeError):
    """Something the user has to fix; the message says what."""


def staging_dir() -> Path:
    return paths.ensure_dir("data", *STAGING)


def remove_staging_if_empty(staging: Path) -> None:
    """Remove data/download_tmp/fei/ and data/download_tmp/ when nothing is left in them."""
    for folder in (staging, staging.parent):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def fetch_resumable(url: str, part: Path) -> None:
    """Append the rest of url to part, asking the server for the bytes part does not have yet."""
    have = part.stat().st_size if part.is_file() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT) as res:
        if res.status_code == 416:  # nothing left to send: part is already whole
            return
        res.raise_for_status()
        if have and res.status_code != 206:
            have = 0  # the server ignored the range and sends the whole file
        left = int(res.headers.get("Content-Length", 0)) or None
        total = have + left if left is not None else None
        with open(part, "ab" if have else "wb") as f, tqdm(
            total=total, initial=have, unit="B", unit_scale=True, desc=part.name.removesuffix(PART_SUFFIX)
        ) as bar:
            for chunk in res.iter_content(CHUNK_BYTES):
                f.write(chunk)
                bar.update(len(chunk))


def zip_ok(path: Path) -> bool:
    """The file is a zip whose every member reads back with the right CRC."""
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except (zipfile.BadZipFile, OSError, EOFError):
        return False


def ensure_zip(part: int, raw: Path, staging: Path, fetch: Fetch = fetch_resumable, attempts: int = DOWNLOAD_ATTEMPTS) -> bool:
    """Make data/raw/fei/<zip> present and readable. Return True when it was downloaded now.

    A zip already in raw/ that reads is kept; one that does not read is left for the
    user to delete, because the script never deletes files in data/raw/. A download
    goes to staging/ first, resumes after a network error, and moves into raw/ only
    after every member reads back.
    """
    name = fei.zip_name(part)
    dest = raw / name
    if dest.is_file():
        if zip_ok(dest):
            return False
        raise FeiError(f"{dest} เปิดไม่ได้หรือไม่ครบ ให้ลบไฟล์นี้เองแล้วรันคำสั่งเดิมอีกครั้ง")
    tmp = staging / (name + PART_SUFFIX)
    last = ""
    for attempt in range(attempts):
        try:
            fetch(fei.zip_url(part), tmp)
        except (requests.RequestException, OSError) as e:
            last = f"{type(e).__name__}: {e}"
            print(f"{name}: โหลดไม่สำเร็จครั้งที่ {attempt + 1} ({last}) จะโหลดต่อจากที่ค้าง")
            continue
        if zip_ok(tmp):
            os.replace(tmp, dest)
            return True
        # Whole but unreadable: the bytes on disk are wrong, so start this file again
        tmp.unlink(missing_ok=True)
        last = "ไฟล์ที่โหลดมาเปิดไม่ได้"
        print(f"{name}: {last} จะโหลดใหม่ทั้งไฟล์")
    raise FeiError(
        f"โหลด {name} ไม่สำเร็จ {attempts} ครั้ง ({last}) ไฟล์ที่ค้างอยู่ใน {staging} "
        "ให้รันคำสั่งเดิมอีกครั้งเพื่อโหลดต่อ ถ้าเว็บของ FEI ยังโหลดไม่ได้ ให้แจ้งก่อนใช้แหล่งอื่น"
    )


def file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sha, name = line.split(maxsplit=1)
                out[name.strip()] = sha
    return out


def ensure_checksums(raw: Path) -> Path:
    """Write the sha256 of the four zips once (FEI publishes none); later runs compare against it."""
    path = raw / fei.CHECKSUMS_NAME
    have = {fei.zip_name(p): file_sha256(raw / fei.zip_name(p)) for p in fei.ZIP_PARTS}
    if path.is_file():
        recorded = read_checksums(path)
        differ = sorted(n for n in have if recorded.get(n) != have[n])
        if differ:
            raise FeiError(
                f"sha256 ของ {', '.join(differ)} ไม่ตรงกับ {path} "
                "ถ้าเพิ่งโหลด zip ใหม่ ให้ตรวจว่าไฟล์ถูกต้องแล้วลบ checksums.txt เองเพื่อบันทึกใหม่"
            )
        return path
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{sha}  {name}\n" for name, sha in have.items())
    return path


@dataclass
class Unpacked:
    written: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)  # a file in images/ differs from the zip
    ignored: list[str] = field(default_factory=list)  # members that are not FEI photos


def file_crc32(path: Path) -> int:
    crc = 0
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            crc = zlib.crc32(chunk, crc)
    return crc


def unpack(zip_path: Path, images: Path, staging: Path, seen: dict[str, str], out: Unpacked) -> None:
    """Copy the FEI photos in zip_path into images/, by file name only.

    Members are matched by their last path part against <person>-<number>.jpg, so no
    member can write outside images/. A photo already there with the same size and CRC
    is kept; one that differs is reported and left alone. seen maps names to the zip
    they came from, to catch a name that appears in two zips.
    """
    images.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename.replace("\\", "/")).name
            if fei.parse_name(name) is None:
                out.ignored.append(f"{zip_path.name}:{info.filename}")
                continue
            if name in seen:
                out.conflicts.append(f"{name} อยู่ทั้งใน {seen[name]} และ {zip_path.name}")
                continue
            seen[name] = zip_path.name
            target = images / name
            if target.is_file():
                same = target.stat().st_size == info.file_size and file_crc32(target) == info.CRC
                (out.kept if same else out.conflicts).append(name)
                continue
            tmp = staging / (name + PART_SUFFIX)
            with z.open(info) as src, open(tmp, "wb") as dst:
                while chunk := src.read(CHUNK_BYTES):
                    dst.write(chunk)
            os.replace(tmp, target)
            out.written.append(name)


def missing_images(images: Path) -> list[str]:
    return [
        fei.image_name(p, n)
        for p in range(1, fei.PEOPLE + 1)
        for n in range(1, fei.POSES + 1)
        if not (images / fei.image_name(p, n)).is_file()
    ]


# ---------------------------------------------------------------- pose numbers

NUMBERING_PEOPLE = 5
NUMBERING_SHEET = "fei_numbering.jpg"
NUMBERING_TILE = 320  # width of each photo on the sheet
NOT_FOUND_YAW = 90.0  # a full profile often has no face at all; rank it as fully turned
STRAIGHT_DEG = 5.0  # median |yaw| of a straight-on pose number; the nearest turned ones are about 7


@dataclass(frozen=True)
class PoseStat:
    number: int
    people: int
    found: int  # how many of the people had a face found
    yaws: tuple[float, ...]  # yaw of each face found, degrees
    smiles: tuple[float, ...]  # max(mouthSmileLeft, mouthSmileRight) of each face found
    brightness: float  # mean gray level 0-255 over the people's photos

    @property
    def median_yaw(self) -> float:
        return float(np.median(self.yaws)) if self.yaws else float("nan")

    @property
    def turn(self) -> float:
        """Median |yaw|, counting a photo without a face as fully turned."""
        values = [abs(y) for y in self.yaws] + [NOT_FOUND_YAW] * (self.people - self.found)
        return float(np.median(values))

    @property
    def mean_smile(self) -> float:
        return float(np.mean(self.smiles)) if self.smiles else float("nan")


def pose_stats(models: FaceModels, people: range) -> tuple[list[PoseStat], dict[tuple[int, int], tuple[int, float | None]]]:
    """Faces and yaw of every pose number over people; also (faces, yaw) per photo."""
    per_photo = {}
    stats = []
    for number in range(1, fei.POSES + 1):
        yaws, smiles, levels, found = [], [], [], 0
        for person in people:
            image = image_io.read_image(fei.image_path(person, number))
            levels.append(float(image.mean()))
            faces = models.detect(image)
            yaw = pose.angles(faces[0].matrix).yaw_deg if faces else None
            per_photo[(person, number)] = (len(faces), yaw)
            if faces:
                found += 1
                yaws.append(yaw)
                bs = faces[0].blendshapes
                smiles.append(max(bs["mouthSmileLeft"], bs["mouthSmileRight"]))
        stats.append(PoseStat(number, len(people), found, tuple(yaws), tuple(smiles), float(np.mean(levels))))
    return stats, per_photo


@dataclass(frozen=True)
class Guess:
    full: tuple[int, int]  # face to the image left, face to the image right
    second: tuple[int, int]
    neutral: int
    smile: int


def guess_numbers(stats: list[PoseStat]) -> Guess:
    """Guess the pose numbers from how far each number turns and how much it smiles.

    The full profiles turn the most on each side and the next angles come after them.
    Of the photos that look straight on (median |yaw| under STRAIGHT_DEG), the smile is
    the one with the highest smile score. FEI also has straight photos under other
    lighting, so the neutral one is the remaining straight photo whose brightness is
    closest to that of the turned photos, which share one lighting. The direction of a
    number whose faces are never found is unknown, so it is taken from the number next
    to it.
    """
    by_number = {s.number: s for s in stats}

    def side(s: PoseStat) -> int:
        if s.yaws:
            return 1 if s.median_yaw > 0 else -1
        for k in (s.number - 1, s.number + 1):  # a neighbour turned the same way
            if k in by_number and by_number[k].yaws:
                return 1 if by_number[k].median_yaw > 0 else -1
        return 0

    turned = sorted(stats, key=lambda s: -s.turn)
    left = [s.number for s in turned if side(s) < 0]
    right = [s.number for s in turned if side(s) > 0]
    if len(left) < 2 or len(right) < 2:
        raise FeiError("หาเลขท่าด้านข้างไม่ได้ครบทั้งสองข้าง ให้ดูแผ่นรวมภาพแล้วกำหนดเอง")
    straight = [s for s in stats if s.yaws and s.turn < STRAIGHT_DEG]
    if len(straight) < 2:
        raise FeiError("หาเลขท่าหน้าตรงไม่ได้ ให้ดูแผ่นรวมภาพแล้วกำหนดเอง")
    smile = max(straight, key=lambda s: s.mean_smile).number
    turned_level = float(np.median([s.brightness for s in stats if s.turn >= STRAIGHT_DEG]))
    neutral = min((s for s in straight if s.number != smile), key=lambda s: abs(s.brightness - turned_level)).number
    return Guess(full=(left[0], right[0]), second=(left[1], right[1]), neutral=neutral, smile=smile)


def numbering_sheet(person: int, per_photo: dict, guess: Guess) -> Image.Image:
    role = {
        guess.full[0]: "full profile, faces image left",
        guess.full[1]: "full profile, faces image right",
        guess.second[0]: "second angle, image left",
        guess.second[1]: "second angle, image right",
        guess.neutral: "frontal neutral",
        guess.smile: "frontal smile",
    }
    tiles, captions = [], []
    for number in range(1, fei.POSES + 1):
        im = Image.open(fei.image_path(person, number)).convert("RGB")
        im = im.resize((NUMBERING_TILE, round(im.height * NUMBERING_TILE / im.width)), Image.Resampling.LANCZOS)
        faces, yaw = per_photo[(person, number)]
        found = f"yaw {yaw:+.1f}" if yaw is not None else "no face"
        if faces > 1:
            found += f" ({faces} faces)"
        tiles.append(im)
        captions.append(f"{fei.image_name(person, number)}  {found}\n{role.get(number, '')}")
    return draw.contact_sheet(tiles, captions, columns=7, font_size=14)


def print_numbering(stats: list[PoseStat], guess: Guess) -> None:
    print(f"\nFace Landmarker กับ {NUMBERING_PEOPLE} คนแรก (yaw เป็นบวกเมื่อหน้าหันไปทางขวาของภาพ)")
    print("เลข  พบใบหน้า  yaw มัธยฐาน  |yaw| ที่นับภาพไม่พบเป็น 90  smile เฉลี่ย  ความสว่าง")
    for s in stats:
        median = f"{s.median_yaw:+7.1f}" if s.yaws else "      -"
        smile = f"{s.mean_smile:5.2f}" if s.smiles else "    -"
        print(f"{s.number:>3}  {s.found:>3}/{s.people}      {median}      {s.turn:7.1f}            {smile}      {s.brightness:5.1f}")
    print("\nเลขท่าที่เดา")
    print(f"  ด้านข้างเต็ม: {guess.full[0]} (หน้าหันไปทางซ้ายของภาพ), {guess.full[1]} (หันไปทางขวาของภาพ)")
    print(f"  มุมรองลงมา: {guess.second[0]} (ซ้ายของภาพ), {guess.second[1]} (ขวาของภาพ)")
    print(f"  หน้าตรงปกติ: {guess.neutral}  หน้าตรงยิ้ม: {guess.smile}")


def cmd_download(args) -> int:
    raw = paths.ensure_dir("fei")
    staging = staging_dir()
    try:
        for part in fei.ZIP_PARTS:
            if ensure_zip(part, raw, staging):
                print(f"โหลด {fei.zip_name(part)} แล้ว")
            else:
                print(f"มี {fei.zip_name(part)} อยู่แล้ว ไม่โหลดซ้ำ")
        checksums = ensure_checksums(raw)
        print(f"sha256 ของ zip อยู่ใน {checksums}")
        result, seen = Unpacked(), {}
        for part in fei.ZIP_PARTS:
            unpack(raw / fei.zip_name(part), fei.images_dir(), staging, seen, result)
    except FeiError as e:
        print(e, file=sys.stderr)
        return 1
    finally:
        remove_staging_if_empty(staging)
    print(f"แตกไฟล์ใหม่ {len(result.written)} ภาพ มีอยู่แล้ว {len(result.kept)} ภาพ ไฟล์อื่นใน zip {len(result.ignored)} ไฟล์")
    if result.ignored:
        print("  ไฟล์ที่ไม่ใช่ภาพ FEI: " + ", ".join(result.ignored[:10]) + (" ..." if len(result.ignored) > 10 else ""))
    if result.conflicts:
        print(f"ไฟล์ที่ไม่ตรงกับ zip {len(result.conflicts)} ไฟล์ ให้ลบเองแล้วรันใหม่: " + ", ".join(result.conflicts[:10]), file=sys.stderr)
        return 1
    missing = missing_images(fei.images_dir())
    if missing:
        print(f"ภาพไม่ครบ ขาด {len(missing)} ภาพ เช่น {', '.join(missing[:5])}", file=sys.stderr)
        return 1
    print(f"{fei.images_dir()} มีภาพครบ {fei.PEOPLE * fei.POSES} ภาพ")
    if args.skip_numbering:
        return 0

    with FaceModels() as models:
        stats, per_photo = pose_stats(models, range(1, NUMBERING_PEOPLE + 1))
    try:
        guess = guess_numbers(stats)
    except FeiError as e:
        print(e, file=sys.stderr)
        return 1
    print_numbering(stats, guess)
    sheet = numbering_sheet(1, per_photo, guess)
    out = paths.ensure_dir("outputs") / NUMBERING_SHEET
    sheet.save(out, quality=90)
    print(f"\nแผ่นรวม 14 ภาพของคนแรก: {out} (ภาพ FEI ห้ามส่งให้ใคร)")
    print("ยืนยันเลขท่าจากแผ่นรวมก่อน แล้วจึงเขียนลง src/facemeasure/fei.py")
    return 0


# ---------------------------------------------------------------- run

PARTS_VERSION = 1
PART_PEOPLE = 20
PARTS_FOLDER = "fei_profile_parts"
SETTINGS_NAME = "settings.json"
PROFILE_CSV = "fei_profile.csv"
FRONTAL_CSV = "fei_frontal_smile.csv"
REPORT_NAME = "fei_profiles.md"
FIGURES_FOLDER = "figures"
CHECK_FOLDER = "fei_profile_check"
CHECK_PEOPLE = 20
SEED = "fei-profile-check-v1"
YAW_STEPS = (30, 35, 40, 45, 50, 55, 60)  # yaw limits compared in the report
SMILE_STEPS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
POINTS = (*P.POINT_NAMES, "ls")
PROFILE_COLUMNS = [
    "image", "person", "number", "pose", "facing", "read_error", "width", "height",
    "face_found", "face_yaw_deg", "silhouette_found", "flipped", "failed", "reasons",
    "legan_angle", "nasofacial_angle", "nose_depth", "face_height",
    *[f"{p}_{axis}" for p in POINTS for axis in ("x", "y")],
]
FRONTAL_QUALITY = [
    "iod_px", "yaw_deg", "pitch_deg", "roll_deg", "smile", "jaw_open", "blink_left", "blink_right",
]
FRONTAL_COLUMNS = ["image", "person", "number", "pose", "read_error", "face_count", *FRONTAL_QUALITY, "failed"]
STR_COLUMNS = {"read_error": str, "failed": str, "reasons": str, "pose": str, "facing": str, "image": str}


# What each reason of facemeasure.profile means, for the report
REASON_TEXT = {
    P.FRONTAL_YAW: "พบใบหน้าที่ |yaw| ต่ำกว่าเกณฑ์",
    P.NO_SILHOUETTE: "ไม่พบเงาศีรษะ",
    P.NO_FACE_SKIN: "ไม่พบคลาสผิวหน้าในเงาศีรษะ",
    P.NO_DIRECTION: "บอกทิศของหน้าไม่ได้",
    P.HEAD_CUT: "ใบหน้าหรือเส้นขอบด้านหน้าชนขอบภาพ",
    P.CHIN_CUT: "ภาพจบใต้ Pog เร็วเกินไป คางอาจถูกตัด",
    P.EDGE_SHORT: "เส้นขอบสั้นเกินกว่าจะหาจุด",
    P.NO_FOREHEAD: "ขอบหน้าผากไม่ใช่ผิว หรือ G ไม่อยู่หน้าร่องโคนจมูก เช่น แว่นหรือผมบังหน้าผาก หรือหันไม่ถึง",
    P.NO_SUBNASALE: "หา Sn หรือคางใต้จมูกไม่ได้",
    P.ORDER: "จุดไม่เรียงตามลำดับ G N Prn Sn Pog",
    P.FACIAL_LINE_SHORT: "เส้น G–Pog สั้นผิดปกติ",
    P.NOSE_SHALLOW: "ปลายจมูกยื่นจากเส้น G–Pog น้อยกว่าเกณฑ์",
    P.ANGLE_RANGE: "มุมอยู่นอกช่วงที่เป็นไปได้",
}


def poses() -> dict[int, tuple[str, str]]:
    """Pose number -> (pose, facing): facing is the image side the face points to."""
    left, right = fei.PROFILE_FULL
    second_left, second_right = fei.PROFILE_SECOND
    return {
        left: ("full", "image_left"),
        right: ("full", "image_right"),
        second_left: ("second", "image_left"),
        second_right: ("second", "image_right"),
        fei.FRONTAL_NEUTRAL: ("frontal_neutral", ""),
        fei.FRONTAL_SMILE: ("frontal_smile", ""),
    }


def profile_constants() -> dict:
    """Every number and tuple constant of facemeasure.profile, so a change asks for --restart."""
    out = {}
    for name in sorted(dir(P)):
        value = getattr(P, name)
        if name.isupper() and isinstance(value, int | float | str | tuple):
            out[name] = value
    return out


def run_settings(config: quality.QualityConfig) -> dict:
    want = {
        "parts_version": PARTS_VERSION,
        "part_people": PART_PEOPLE,
        "poses": {str(k): list(v) for k, v in sorted(poses().items())},
        "quality": asdict(config),
        "profile": profile_constants(),
        "models": assets.fingerprint([LANDMARKER_FILE, SEGMENTER_FILE]),
        "mediapipe": mediapipe.__version__,
    }
    return json.loads(json.dumps(want, ensure_ascii=False))  # tuples become lists, as after reading


def prepare_parts(folder: Path, want: dict, restart: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / SETTINGS_NAME
    if restart:
        for p in folder.glob("part_*"):
            p.unlink()
        path.unlink(missing_ok=True)
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            have = json.load(f)
        if have != want:
            changed = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
            raise FeiError(f"ผลที่ค้างไว้ใน {folder} ทำด้วยค่าที่ต่างจากตอนนี้ ({', '.join(changed)}) ให้รันใหม่ด้วย --restart")
    else:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(want, ensure_ascii=False, indent=2) + "\n")


def part_paths(folder: Path, k: int) -> tuple[Path, Path]:
    return folder / f"part_{k:02d}_profile.csv", folder / f"part_{k:02d}_frontal.csv"


def profile_row(person: int, number: int, r: P.ProfileResult | None, read_error: str = "") -> dict:
    pose_name, facing = poses()[number]
    row = {c: None for c in PROFILE_COLUMNS}
    row.update(image=fei.image_name(person, number), person=person, number=number, pose=pose_name, facing=facing,
               read_error=read_error)
    if r is None:
        return row
    q = r.quality
    row.update(
        width=r.width, height=r.height, face_found=q.face_found, face_yaw_deg=q.face_yaw_deg,
        silhouette_found=q.silhouette_found, flipped=q.flipped, failed=";".join(r.failed), reasons=";".join(r.reasons),
        legan_angle=r.legan_angle, nasofacial_angle=r.nasofacial_angle, nose_depth=r.nose_depth,
        face_height=r.face_height,
    )
    for name, (x, y) in r.points.items():
        row[f"{name}_x"], row[f"{name}_y"] = x, y
    return row


def frontal_row(person: int, number: int, r, read_error: str = "") -> dict:
    row = {c: None for c in FRONTAL_COLUMNS}
    row.update(image=fei.image_name(person, number), person=person, number=number, pose=poses()[number][0],
               read_error=read_error)
    if r is None:
        return row
    row.update(face_count=r.face_count, failed=";".join(r.failed))
    if r.quality is not None:
        row.update({c: getattr(r.quality, c) for c in FRONTAL_QUALITY})
    for name, score in (r.blendshapes or {}).items():
        row[f"bs_{name}"] = score
    return row


def run_part(people: range, models: FaceModels, config: quality.QualityConfig, bar) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_rows, frontal_rows = [], []
    frontal_numbers = (fei.FRONTAL_NEUTRAL, fei.FRONTAL_SMILE)
    for person in people:
        for number in sorted(poses()):
            try:
                image = image_io.read_image(fei.image_path(person, number))
            except (image_io.ImageReadError, OSError) as e:
                profile_rows.append(profile_row(person, number, None, type(e).__name__))
                if number in frontal_numbers:
                    frontal_rows.append(frontal_row(person, number, None, type(e).__name__))
                bar.update(1)
                continue
            profile_rows.append(profile_row(person, number, P.analyze_profile(image, models, config.profile)))
            if number in frontal_numbers:
                frontal_rows.append(frontal_row(person, number, analyze_frontal(image, models, config.frontal)))
            bar.update(1)
    frontal = pd.DataFrame(frontal_rows)
    frontal = frontal[[*FRONTAL_COLUMNS, *sorted(c for c in frontal.columns if c.startswith("bs_"))]]
    return pd.DataFrame(profile_rows, columns=PROFILE_COLUMNS), frontal


def write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    tmp.replace(path)


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=STR_COLUMNS, keep_default_na=False, na_values=[""], encoding="utf-8")
    for c in STR_COLUMNS:
        if c in df:
            df[c] = df[c].fillna("")
    return df


def run_all(folder: Path, config: quality.QualityConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = [range(k, min(k + PART_PEOPLE, fei.PEOPLE + 1)) for k in range(1, fei.PEOPLE + 1, PART_PEOPLE)]
    todo = [k for k in range(len(groups)) if not part_paths(folder, k)[0].is_file()]
    per_person = len(poses())
    if todo:
        done = (len(groups) - len(todo)) * PART_PEOPLE * per_person
        with FaceModels() as models, tqdm(total=fei.PEOPLE * per_person, initial=done, unit="image", desc="FEI") as bar:
            for k in todo:
                prof, front = run_part(groups[k], models, config, bar)
                profile_path, frontal_path = part_paths(folder, k)
                write_csv(front, frontal_path)
                write_csv(prof, profile_path)  # the profile CSV marks the part as done, so it goes last
    profile = pd.concat([read_csv(part_paths(folder, k)[0]) for k in range(len(groups))], ignore_index=True)
    frontal = pd.concat([read_csv(part_paths(folder, k)[1]) for k in range(len(groups))], ignore_index=True)
    return profile, frontal


# ---------------------------------------------------------------- report numbers


def measured(df: pd.DataFrame) -> pd.Series:
    """The photo passed every check: no failed code."""
    return (df["failed"] == "") & (df["read_error"] == "")


def has_angles(df: pd.DataFrame) -> pd.Series:
    """The edge passed every check; only the yaw limit may have failed."""
    return df["legan_angle"].notna()


def passes_yaw(df: pd.DataFrame, limit: float) -> pd.Series:
    found = df["face_found"].astype(str) == "True"
    return ~found | (df["face_yaw_deg"].abs() >= limit)


def quantiles(v, qs=(10, 50, 90)) -> list[float]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, q)) if len(v) else math.nan for q in qs]


def fmt(v, digits=1) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return f"{v:.{digits}f}"


def pct(n: int, total: int) -> str:
    return f"{n} ({100 * n / total:.1f}%)" if total else "0 (–)"


def cell(value) -> str:
    """A table cell; | inside it, as in |yaw|, would split the cell, so it is escaped."""
    return str(value).replace("|", "\\|")


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(cell(h) for h in header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def legan_bands(v: pd.Series) -> list[tuple[str, int]]:
    """Counts in the grade bands of model.md section 3.2, from criteria.LEGAN_CUTOFFS."""
    a, b, c = criteria.LEGAN_CUTOFFS
    return [
        (f"0 (≤ {a:g}°)", int((v <= a).sum())),
        (f"1 ({a:g}–{b:g}°)", int(((v > a) & (v <= b)).sum())),
        (f"2 ({b:g}–{c:g}°)", int(((v > b) & (v <= c)).sum())),
        (f"3 (> {c:g}°)", int((v > c).sum())),
    ]


def dorsum_bands(v: pd.Series, sex: str) -> list[tuple[str, int]]:
    """(interval, count) for grade 0, grade 1 below, grade 1 above and grade 2+, with one sex's
    cutoffs of model.md section 3.4 (criteria.DORSUM_CUTOFFS)."""
    c = criteria.DORSUM_CUTOFFS[sex]
    return [
        (f"{c.low:g}–{c.high:g}°", int(((v >= c.low) & (v <= c.high)).sum())),
        (f"{c.small:g}–{c.low:g}°", int(((v >= c.small) & (v < c.low)).sum())),
        (f"> {c.high:g}°", int((v > c.high).sum())),
        (f"< {c.small:g}°", int((v < c.small).sum())),
    ]


DORSUM_GRADES = ("0", "1 (จมูกยื่นน้อย)", "1 (จมูกยื่นมาก)", "2+")


def left_right(df: pd.DataFrame, pose_name: str) -> pd.DataFrame:
    """Per person, the angles of the photo facing the image left and the one facing the image right."""
    part = df[(df["pose"] == pose_name) & has_angles(df)]
    wide = part.pivot(index="person", columns="facing", values=["legan_angle", "nasofacial_angle"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.dropna()


def against_full(df: pd.DataFrame) -> pd.DataFrame:
    """Next-angle photos against the full profile of the same person facing the same way."""
    keys = ["person", "facing"]
    cols = ["legan_angle", "nasofacial_angle"]
    full = df[(df["pose"] == "full") & has_angles(df)][keys + cols]
    second = df[(df["pose"] == "second") & has_angles(df)][keys + cols]
    return second.merge(full, on=keys, suffixes=("_second", "_full"))


def facial_ratio(df: pd.DataFrame) -> pd.Series:
    return np.hypot(df["pog_x"] - df["g_x"], df["pog_y"] - df["g_y"]) / df["face_height"]


# ---------------------------------------------------------------- figures


def save_figure(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=100, metadata={"Software": None})


def write_figures(df: pd.DataFrame, frontal: pd.DataFrame, config: quality.QualityConfig, folder: Path) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    full = df[(df["pose"] == "full") & measured(df)]

    fig = Figure(figsize=(12, 4.2), layout="constrained")
    ax1, ax2 = fig.subplots(1, 2)
    ax1.hist(full["legan_angle"], bins=np.arange(-20, 41, 2), color="#4a7ab5")
    for c in criteria.LEGAN_CUTOFFS:
        ax1.axvline(c, color="#c0392b", linestyle="--", linewidth=1)
    ax1.set_xlabel("Legan angle (deg)")
    ax1.set_ylabel("photos")
    ax2.hist(full["nasofacial_angle"], bins=np.arange(10, 61, 1.5), color="#4a7ab5")
    for sex, color in (("female", "#c0392b"), ("male", "#2471a3")):
        c = criteria.DORSUM_CUTOFFS[sex]
        for v in (c.small, c.low, c.high):
            ax2.axvline(v, color=color, linestyle="--", linewidth=1)
    ax2.set_xlabel("nasofacial angle (deg); cutoffs red female, blue male")
    fig.suptitle(f"FEI full profiles that pass every check (n = {len(full)})")
    out["angles"] = "fei_angles_hist.png"
    save_figure(fig, folder / out["angles"])

    lr = left_right(df, "full")
    sa = against_full(df)
    fig = Figure(figsize=(12, 8.4), layout="constrained")
    axes = fig.subplots(2, 2)
    for col, (name, lo, hi) in enumerate((("legan_angle", -20, 40), ("nasofacial_angle", 10, 60))):
        ax = axes[0, col]
        ax.scatter(lr[f"{name}_image_left"], lr[f"{name}_image_right"], s=8, alpha=0.6, color="#4a7ab5")
        ax.plot([lo, hi], [lo, hi], color="#999999", linewidth=0.8)
        ax.set_xlabel(f"{name}, face to the image left")
        ax.set_ylabel(f"{name}, face to the image right")
        ax.set_title(f"full profiles, same person (n = {len(lr)})", fontsize=10)
        ax = axes[1, col]
        ax.scatter(sa[f"{name}_full"], sa[f"{name}_second"], s=8, alpha=0.6, color="#d35400")
        ax.plot([lo, hi], [lo, hi], color="#999999", linewidth=0.8)
        ax.set_xlabel(f"{name}, full profile")
        ax.set_ylabel(f"{name}, next angle")
        ax.set_title(f"next angle against full profile, same side (n = {len(sa)})", fontsize=10)
    fig.suptitle("Angles of the same person across photos (edge checks passed, yaw not checked)")
    out["pairs"] = "fei_angle_pairs.png"
    save_figure(fig, folder / out["pairs"])

    fig = Figure(figsize=(12, 4.2), layout="constrained")
    ax = fig.subplots()
    bins = np.arange(0, 91, 2.5)
    for name, color in (("full", "#4a7ab5"), ("second", "#d35400"), ("frontal_neutral", "#27ae60")):
        part = df[(df["pose"] == name) & (df["face_found"].astype(str) == "True")]
        ax.hist(part["face_yaw_deg"].abs(), bins=bins, alpha=0.6, color=color,
                label=f"{name} (face found {len(part)} of {int((df['pose'] == name).sum())})")
    ax.axvline(config.profile.min_abs_yaw_deg, color="#c0392b", linestyle="--", label="min_abs_yaw_deg")
    ax.set_xlabel("|yaw| from Face Landmarker (deg)")
    ax.set_ylabel("photos")
    ax.legend()
    fig.suptitle("Face Landmarker yaw on FEI photos sent as profiles")
    out["yaw"] = "fei_yaw_hist.png"
    save_figure(fig, folder / out["yaw"])

    fig = Figure(figsize=(12, 4.2), layout="constrained")
    ax = fig.subplots()
    bins = np.linspace(0, 1, 41)
    for name, color in (("frontal_neutral", "#4a7ab5"), ("frontal_smile", "#d35400")):
        ax.hist(frontal.loc[frontal["pose"] == name, "smile"].dropna(), bins=bins, alpha=0.6, color=color, label=name)
    ax.axvline(config.frontal.max_smile, color="#c0392b", linestyle="--", label="max_smile")
    ax.set_xlabel("smile = max(mouthSmileLeft, mouthSmileRight)")
    ax.set_ylabel("photos")
    ax.legend()
    fig.suptitle("Smile blendshape on FEI frontal photos")
    out["smile"] = "fei_smile_hist.png"
    save_figure(fig, folder / out["smile"])
    return out


# ---------------------------------------------------------------- report


def report_text(df: pd.DataFrame, frontal: pd.DataFrame, config: quality.QualityConfig, figures: dict[str, str]) -> str:
    limit = config.profile.min_abs_yaw_deg
    full = df[df["pose"] == "full"]
    second = df[df["pose"] == "second"]
    front = df[df["pose"].str.startswith("frontal")]
    ok_full = full[measured(full)]
    lines = [
        "# ตรวจการหาเส้นขอบภาพด้านข้างกับ FEI",
        "",
        ("สร้างด้วย `uv run python model/scripts/profile_check.py run` ทุกภาพผ่าน "
        "`facemeasure.profile.analyze_profile` ซึ่งเป็นฟังก์ชันเดียวกับที่ api เรียก "
        f"(mediapipe {mediapipe.__version__}) เกณฑ์มาจาก `shared/quality.json` (`min_abs_yaw_deg` = {limit:g}) "
        "รายงานนี้มีเฉพาะตัวเลขและกราฟ ภาพ FEI ห้ามแจกจ่าย ภาพตรวจจึงอยู่ใน `fei_profile_check/` "
        "ของโฟลเดอร์ outputs เท่านั้น"),
        "",
        (f"- เลขท่าที่ใช้: ด้านข้างเต็ม {fei.PROFILE_FULL[0]} (หน้าหันไปทางซ้ายของภาพ) กับ {fei.PROFILE_FULL[1]} "
        f"(ขวาของภาพ), มุมรองลงมา {fei.PROFILE_SECOND[0]} กับ {fei.PROFILE_SECOND[1]}, "
        f"หน้าตรงปกติ {fei.FRONTAL_NEUTRAL}, หน้าตรงยิ้ม {fei.FRONTAL_SMILE}"),
        ("- \"วัดได้\" คือไม่มีรหัสที่ไม่ผ่าน ส่วน \"เส้นขอบผ่าน\" คือหาจุดและผ่านการตรวจความสมเหตุสมผลของเส้นขอบ "
        "โดยไม่นับเกณฑ์ yaw ภาพที่เส้นขอบผ่านมีมุมเสมอ ซึ่งโหมดทดลองแสดงได้"),
        "- FEI ไม่มีป้ายเพศรายภาพในไฟล์ที่โหลด จึงรายงานเส้นแบ่งของสันจมูกทั้งของหญิงและของชาย",
        "",
        "## ภาพด้านข้างเต็ม",
        "",
        table(
            ["", "ภาพ", "วัดได้", "เส้นขอบผ่าน", "ไม่พบใบหน้า", f"พบใบหน้าที่ |yaw| < {limit:g}°"],
            [[
                name, len(part), pct(int(measured(part).sum()), len(part)), pct(int(has_angles(part).sum()), len(part)),
                pct(int((part["face_found"].astype(str) != "True").sum()), len(part)),
                pct(int((~passes_yaw(part, limit)).sum()), len(part)),
            ] for name, part in (("ด้านข้างเต็ม", full), ("มุมรองลงมา", second), ("หน้าตรงทั้งสองแบบ", front))],
        ),
        "",
        "เหตุที่เส้นขอบไม่ผ่าน (ไม่นับ `frontal_yaw`) ของภาพด้านข้างเต็ม",
        "",
    ]
    reasons = full["reasons"].str.split(";").explode()
    reasons = reasons[(reasons != "") & (reasons != P.FRONTAL_YAW)].value_counts()
    lines += [
        table(["เหตุ", "ความหมาย", "ภาพ"], [[f"`{k}`", REASON_TEXT.get(k, ""), v] for k, v in reasons.items()])
        if len(reasons) else "ไม่มี",
        "",
    ]

    lines += [
        "### เกณฑ์ yaw ที่ระดับต่าง ๆ",
        "",
        "สัดส่วนที่ผ่านเกณฑ์ yaw (ไม่พบใบหน้า หรือพบที่ |yaw| ถึงเกณฑ์) และสัดส่วนที่วัดได้เมื่อรวมการตรวจเส้นขอบ",
        "",
        table(
            ["เกณฑ์ |yaw|", "ด้านข้างเต็ม ผ่าน yaw", "ด้านข้างเต็ม วัดได้", "มุมรองลงมา ผ่าน yaw", "หน้าตรง ผ่าน yaw"],
            [[
                f"{t}°" + (" (ปัจจุบัน)" if t == limit else ""),
                pct(int(passes_yaw(full, t).sum()), len(full)),
                pct(int((passes_yaw(full, t) & has_angles(full)).sum()), len(full)),
                pct(int(passes_yaw(second, t).sum()), len(second)),
                pct(int(passes_yaw(front, t).sum()), len(front)),
            ] for t in sorted({*YAW_STEPS, limit})],
        ),
        "",
        f"![yaw]({FIGURES_FOLDER}/{figures['yaw']})",
        "",
        "## มุมของภาพด้านข้างเต็มที่วัดได้",
        "",
    ]
    rows = []
    for name, label in (("legan_angle", "มุม Legan"), ("nasofacial_angle", "มุม nasofacial")):
        for subset, part in (("วัดได้", ok_full), ("เส้นขอบผ่าน", full[has_angles(full)])):
            p10, p50, p90 = quantiles(part[name])
            rows.append([label, subset, len(part), fmt(p10), fmt(p50), fmt(p90)])
    lines += [table(["มุม", "ชุด", "n", "p10", "มัธยฐาน", "p90"], rows), ""]
    lines += [
        "สัดส่วนต่อเกรดตามเส้นแบ่งของ model.md (ภาพที่วัดได้) ยังไม่มีช่วงก้ำกึ่ง ซึ่งกำหนดใน `facemeasure.criteria`",
        "",
        table(["มุม Legan", "ภาพ"], [[k, pct(v, len(ok_full))] for k, v in legan_bands(ok_full["legan_angle"])]),
        "",
    ]
    female = dorsum_bands(ok_full["nasofacial_angle"], "female")
    male = dorsum_bands(ok_full["nasofacial_angle"], "male")
    lines += [
        table(
            ["เกรดสันจมูก", "ช่วงของหญิง", "ภาพ", "ช่วงของชาย", "ภาพ"],
            [[grade, fk, pct(fv, len(ok_full)), mk, pct(mv, len(ok_full))]
             for grade, (fk, fv), (mk, mv) in zip(DORSUM_GRADES, female, male, strict=True)],
        ),
        "",
        f"![angles]({FIGURES_FOLDER}/{figures['angles']})",
        "",
    ]

    lr = left_right(df, "full")
    sa = against_full(df)
    lines += [
        "## ความคลาดเคลื่อนจากการถ่ายและการหาจุด",
        "",
        "ค่าต่างสัมบูรณ์ของมุมระหว่างภาพสองภาพของคนเดียวกัน เฉพาะภาพที่เส้นขอบผ่าน เพราะเกณฑ์ yaw ไม่เปลี่ยนตำแหน่งจุด",
        "",
    ]
    rows = []
    for title, data, a, b in (
        ("ด้านข้างเต็ม ซ้าย กับ ขวาของภาพ", lr, "_image_left", "_image_right"),
        ("มุมรองลงมา กับ ด้านข้างเต็ม ข้างเดียวกัน", sa, "_second", "_full"),
    ):
        for name, label in (("legan_angle", "Legan"), ("nasofacial_angle", "nasofacial")):
            d = (data[name + a] - data[name + b]).abs()
            p50, p90 = quantiles(d, (50, 90))
            rows.append([title, label, len(d), fmt(float(d.mean()) if len(d) else math.nan), fmt(p50), fmt(p90),
                         fmt(float(d.max()) if len(d) else math.nan)])
    lines += [table(["คู่ภาพ", "มุม", "คู่", "เฉลี่ย", "มัธยฐาน", "p90", "สูงสุด"], rows), ""]
    if len(sa):
        bias = [fmt(float((sa[f"{n}_second"] - sa[f"{n}_full"]).mean())) for n in ("legan_angle", "nasofacial_angle")]
        lines += [f"ค่าเฉลี่ยของ (มุมรองลงมา − ด้านข้างเต็ม): Legan {bias[0]}° nasofacial {bias[1]}°", ""]
    lines += [f"![pairs]({FIGURES_FOLDER}/{figures['pairs']})", ""]

    lines += [
        "## ภาพมุมรองลงมาและภาพหน้าตรงที่ส่งเป็นภาพด้านข้าง",
        "",
        (f"- มุมรองลงมาผ่านเกณฑ์ {limit:g}°: {pct(int(passes_yaw(second, limit).sum()), len(second))} "
        f"และวัดได้ {pct(int(measured(second).sum()), len(second))}"),
        f"- ภาพหน้าตรงได้ `profile_invalid`: {pct(int((front['failed'] == 'profile_invalid').sum()), len(front))}",
        "",
    ]
    reasons = front["reasons"].str.split(";").explode()
    reasons = reasons[reasons != ""].value_counts()
    lines += ["เหตุของภาพหน้าตรง (ภาพหนึ่งมีได้หลายเหตุ)", "",
              table(["เหตุ", "ภาพ"], [[f"`{k}`", v] for k, v in reasons.items()]), ""]

    lines += ["## การตรวจความสมเหตุสมผลของเส้นขอบ", "",
              "ค่าคงที่ใน `facemeasure/profile.py` ตั้งจากการกระจายนี้ ค่าของภาพที่เส้นขอบผ่าน", ""]
    rows = []
    for name, label in (("nose_depth", "Prn หน้าเส้น G–Pog ÷ |G–Pog|"), ("facial", "|G–Pog| ÷ ความสูงผิวหน้า"),
                        ("legan_angle", "มุม Legan"), ("nasofacial_angle", "มุม nasofacial")):
        for title, part in (("ด้านข้างเต็ม", full), ("มุมรองลงมา", second)):
            part = part[has_angles(part)]
            v = facial_ratio(part) if name == "facial" else part[name]
            q = quantiles(v, (0, 1, 50, 99, 100))
            rows.append([label, title, len(part), *[fmt(x, 3 if name in ("nose_depth", "facial") else 1) for x in q]])
    lines += [table(["ค่า", "ภาพ", "n", "ต่ำสุด", "p1", "มัธยฐาน", "p99", "สูงสุด"], rows), ""]
    lines += [
        table(["ค่าคงที่", "ค่า"], [
            ["`MIN_NOSE_DEPTH`", P.MIN_NOSE_DEPTH], ["`MIN_FACIAL_LINE`", P.MIN_FACIAL_LINE],
            ["`LEGAN_RANGE`", P.LEGAN_RANGE], ["`NASOFACIAL_RANGE`", P.NASOFACIAL_RANGE],
        ]),
        "",
    ]

    neutral = frontal[frontal["pose"] == "frontal_neutral"]
    smile = frontal[frontal["pose"] == "frontal_smile"]
    lines += [
        "## ค่า smile ของภาพหน้าตรง",
        "",
        ("ใช้ประกอบการปรับเกณฑ์ `max_smile` ใน `shared/quality.json` ซึ่งสคริปต์นี้ไม่แก้เอง "
        "`smiling` ไม่ผ่านเมื่อ smile หรือ jawOpen เกินเกณฑ์"),
        "",
    ]
    rows = []
    for title, part in (("หน้าตรงปกติ", neutral), ("หน้าตรงยิ้ม", smile)):
        p10, p50, p90 = quantiles(part["smile"])
        fails = part["failed"].str.split(";").apply(lambda c: "smiling" in c)
        rows.append([title, int(part["face_count"].ge(1).sum()), fmt(p10, 2), fmt(p50, 2), fmt(p90, 2),
                     pct(int(fails.sum()), len(part))])
    lines += [table(["ภาพ", "พบใบหน้า", "smile p10", "มัธยฐาน", "p90", "ได้ `smiling` ที่เกณฑ์ปัจจุบัน"], rows), ""]
    lines += [
        table(
            ["smile เกิน", "หน้าตรงปกติ", "หน้าตรงยิ้ม"],
            [[f"{t:g}" + (" (ปัจจุบัน)" if t == config.frontal.max_smile else ""),
              pct(int((neutral["smile"] > t).sum()), len(neutral)), pct(int((smile["smile"] > t).sum()), len(smile))]
             for t in sorted({*SMILE_STEPS, config.frontal.max_smile})],
        ),
        "",
        f"![smile]({FIGURES_FOLDER}/{figures['smile']})",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- check images


def rank(person: int) -> str:
    return hashlib.sha256(f"{SEED}|{person}".encode()).hexdigest()


def check_people() -> list[int]:
    return sorted(sorted(range(1, fei.PEOPLE + 1), key=rank)[:CHECK_PEOPLE])


def drawn(models: FaceModels, config: quality.QualityConfig, person: int, number: int) -> Image.Image:
    path = fei.image_path(person, number)
    r = P.analyze_profile(image_io.read_image(path), models, config.profile)
    return draw.draw_profile(Image.open(path), r, label=path.name)


def write_check_images(df: pd.DataFrame, config: quality.QualityConfig) -> tuple[Path, int, int]:
    folder = paths.ensure_dir("outputs", CHECK_FOLDER)
    for old in [*folder.glob("person_*.jpg"), *folder.glob("failed_*.jpg")]:
        old.unlink()  # check images this script made before; data/raw is never touched
    full = df[df["pose"] == "full"]
    failed = full[~measured(full)]
    with FaceModels() as models:
        for person in tqdm(check_people(), desc="check images", unit="person"):
            tiles = [drawn(models, config, person, n) for n in fei.PROFILE_FULL]
            sheet = Image.new("RGB", (sum(t.width for t in tiles), max(t.height for t in tiles)))
            x = 0
            for t in tiles:
                sheet.paste(t, (x, 0))
                x += t.width
            sheet.save(folder / f"person_{person:03d}.jpg", quality=90)
        for _, row in tqdm(failed.iterrows(), total=len(failed), desc="failed profiles", unit="image"):
            drawn(models, config, int(row["person"]), int(row["number"])).save(
                folder / f"failed_{Path(row['image']).stem}.jpg", quality=90
            )
    return folder, CHECK_PEOPLE, len(failed)


def cmd_run(args) -> int:
    missing = missing_images(fei.images_dir())
    if missing:
        print(f"ยังไม่มีภาพ FEI ครบ ({len(missing)} ภาพขาด) ให้รัน download ก่อน", file=sys.stderr)
        return 1
    config = quality.load_config()
    folder = paths.ensure_dir("features", PARTS_FOLDER)
    try:
        prepare_parts(folder, run_settings(config), args.restart)
    except FeiError as e:
        print(e, file=sys.stderr)
        return 1
    profile, frontal = run_all(folder, config)
    profile_path = paths.ensure_dir("labels") / PROFILE_CSV
    frontal_path = paths.ensure_dir("labels") / FRONTAL_CSV
    write_csv(profile, profile_path)
    write_csv(frontal, frontal_path)
    reports = paths.ensure_dir("reports")
    figures = write_figures(profile, frontal, config, reports / FIGURES_FOLDER)
    report_path = reports / REPORT_NAME
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(report_text(profile, frontal, config, figures) + "\n")
    full = profile[profile["pose"] == "full"]
    front = profile[profile["pose"].str.startswith("frontal")]
    print(f"\n{profile_path}\n{frontal_path}\n{report_path}")
    print(f"ด้านข้างเต็มวัดได้ {pct(int(measured(full).sum()), len(full))} "
          f"เส้นขอบผ่าน {pct(int(has_angles(full).sum()), len(full))}")
    print(f"ภาพหน้าตรงได้ profile_invalid {pct(int((front['failed'] == 'profile_invalid').sum()), len(front))}")
    ok = full[measured(full)]
    print(f"มัธยฐาน Legan {fmt(ok['legan_angle'].median())}° nasofacial {fmt(ok['nasofacial_angle'].median())}°")
    if not args.no_check_images:
        folder, people, failed = write_check_images(profile, config)
        print(f"ภาพตรวจ {people} คนและภาพที่วัดไม่ได้ {failed} ภาพใน {folder} (ภาพ FEI ห้ามส่งให้ใคร)")
    return 0


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download FEI and check analyze_profile on it")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="download and unpack FEI, then guess the pose numbers")
    p.add_argument("--skip-numbering", action="store_true", help="only download and unpack")
    p = sub.add_parser("run", help="run analyze_profile on FEI and write the CSVs, report and check images")
    p.add_argument("--restart", action="store_true", help="drop the saved parts and analyze every photo again")
    p.add_argument("--no-check-images", action="store_true", help="skip the check images in outputs/fei_profile_check/")
    args = parser.parse_args(argv)
    return {"download": cmd_download, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
