"""Jaw and alar-base labels from the SCUT-FBP5500 landmarks (model.md sections 3.1, 3.4, 5.2 and 5.3).

Run from the project root:

    uv run python model/scripts/make_scut_labels.py

Writes
- labels/scut_labels.csv     one row per image: group, split, beauty score, ratios, IOD, Go and Zy
- labels/scut_skipped.csv    landmark files that give no label, with the reason
- reports/scut_labels.md     counts and distributions, numbers and charts only, so it can be committed
- reports/figures/scut_*.png histograms with the grade cutoffs
- outputs/scut_label_check/  contact sheets with the points drawn; faces, so never commit them

The rules live in facemeasure (face_geometry, scut and criteria); this script only runs
them over every file. Split: the SCUT 60/40 test set is the test set, and 15% of the SCUT
train set goes to val, stratified by AF, AM, CF and CM. Every random choice is a sha256
ranking of a fixed seed and the file name, which gives the same result on every machine.
The whole run takes well under a minute, so it simply starts over each time.
"""

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from tqdm import tqdm

from facemeasure import criteria, paths, scut
from facemeasure.draw import contact_sheet
from facemeasure.face_geometry import GeometryError, JawNose, LevelFrame

SEED = "scut-labels-v1"
VAL_SHARE = 0.15
SPLIT_FOLDER = ("train_test_files", "split_of_60%training and 40%testing")
LANDMARK_FOLDER = "facial landmark"
IMAGE_FOLDER = "Images"
LABELS_NAME = "scut_labels.csv"
SKIPPED_NAME = "scut_skipped.csv"
REPORT_NAME = "scut_labels.md"
FIGURES_FOLDER = "figures"
JAW_FIGURE = "scut_jaw_hist.png"
NOSE_FIGURE = "scut_nose_hist.png"
CHECK_FOLDER = "scut_label_check"
CHECK_PER_GROUP = 12  # 4 groups x 12 = 48 random images
SHEET_TILES = 48
SHEET_COLUMNS = 8
OUTLIER_PERCENTILES = (1, 99)
AF_STOP_SHARE = 0.5  # more than half of AF above 0.83 means the points do not match [3] (model.md 3.1)
ZY_EDGE_TOLERANCE = 0.02  # Zy within 2% of the band height from an end counts as on that end
ZY_POSITIONS = ("eye", "inside", "subnasale")

COLUMNS = [
    "image", "group", "sex", "ethnicity", "split", "scut_split", "beauty_score",
    "jaw_ratio", "nose_ratio", "eye_dist_px",
    "go_l_x", "go_l_y", "go_r_x", "go_r_y", "zy_l_x", "zy_l_y", "zy_r_x", "zy_r_y",
]
RATIO_DECIMALS = 6
PIXEL_DECIMALS = 3
SPLITS = ("train", "val", "test")
MEASURES = {"jaw_ratio": "กราม", "nose_ratio": "ฐานจมูก"}
NOSE_KEYS = ("0", "1 narrow", "1 wide", "2+")


@dataclass(frozen=True)
class Measured:
    points: np.ndarray
    result: JawNose


def rank(salt: str, name: str) -> str:
    """Stable pseudo-random key of a file name."""
    return hashlib.sha256(f"{SEED}:{salt}:{name}".encode()).hexdigest()


def natural_key(image: str) -> tuple[int, int]:
    s = scut.parse_name(image)
    return (scut.GROUPS.index(s.group), s.number)


def read_split_file(path: Path) -> dict[str, float]:
    """Image name -> beauty score from a SCUT split file ("AF1.jpg 3.5" per line, no header)."""
    scores: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path.name} line {n}: expected '<image> <score>', got {line.strip()!r}")
            image, score = parts
            scut.parse_name(image)
            if image in scores:
                raise ValueError(f"{path.name} line {n}: {image} appears twice")
            scores[image] = float(score)
    return scores


def val_counts(sizes: dict[str, int], share: float = VAL_SHARE) -> dict[str, int]:
    """Val images per group: share of each group, rounded so the total is share of all images."""
    total = math.floor(share * sum(sizes.values()) + 0.5)
    exact = {g: share * n for g, n in sizes.items()}
    counts = {g: math.floor(x) for g, x in exact.items()}
    by_remainder = sorted(sizes, key=lambda g: (-(exact[g] - counts[g]), g))
    for g in by_remainder[: total - sum(counts.values())]:
        counts[g] += 1
    return counts


def assign_splits(scut_train: list[str], scut_test: list[str], share: float = VAL_SHARE) -> dict[str, tuple[str, str]]:
    """Image -> (split, scut_split). Val is drawn from the SCUT train set within each group."""
    both = set(scut_train) & set(scut_test)
    if both:
        raise ValueError(f"{len(both)} images are in both SCUT train and test, e.g. {min(both)}")
    by_group: dict[str, list[str]] = {g: [] for g in scut.GROUPS}
    for image in scut_train:
        by_group[scut.parse_name(image).group].append(image)
    counts = val_counts({g: len(v) for g, v in by_group.items()}, share)
    out = {image: ("test", "test") for image in scut_test}
    for g, images in by_group.items():
        ordered = sorted(images, key=lambda name: rank("val", name))
        for k, image in enumerate(ordered):
            out[image] = ("val" if k < counts[g] else "train", "train")
    return out


def measure_all(landmark_dir: Path, images: list[str]) -> tuple[dict[str, Measured], list[tuple[str, str]]]:
    """Measure every image; images without a label go to the skipped list with the reason."""
    measured: dict[str, Measured] = {}
    skipped: list[tuple[str, str]] = []
    for image in tqdm(images, unit="file", desc="measure"):
        pts_path = landmark_dir / f"{Path(image).stem}.pts"
        if not pts_path.is_file():
            skipped.append((image, "no landmark file"))
            continue
        try:
            points = scut.read_pts(pts_path)
            measured[image] = Measured(points, scut.measure(points))
        except (scut.PtsError, GeometryError) as err:
            skipped.append((image, str(err)))
    return measured, skipped


def label_table(scores: dict[str, float], splits: dict[str, tuple[str, str]], measured: dict[str, Measured]) -> pd.DataFrame:
    rows = []
    for image in sorted(measured, key=natural_key):
        s = scut.parse_name(image)
        r = measured[image].result
        split, scut_split = splits[image]
        rows.append({
            "image": image, "group": s.group, "sex": s.sex, "ethnicity": s.ethnicity,
            "split": split, "scut_split": scut_split, "beauty_score": round(scores[image], RATIO_DECIMALS),
            "jaw_ratio": round(r.jaw_ratio, RATIO_DECIMALS), "nose_ratio": round(r.nose_ratio, RATIO_DECIMALS),
            "eye_dist_px": round(r.iod_px, PIXEL_DECIMALS),
            **{f"{name}_{axis}": round(getattr(r, name)[i], PIXEL_DECIMALS)
               for name in ("go_l", "go_r", "zy_l", "zy_r") for i, axis in enumerate("xy")},
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Values below p1 or above p99 of their group, one row per image and ratio."""
    rows = []
    lo_q, hi_q = OUTLIER_PERCENTILES
    for group in scut.GROUPS:
        g = df[df["group"] == group]
        if g.empty:
            continue
        for measure in MEASURES:
            lo, hi = np.percentile(g[measure], [lo_q, hi_q])
            for _, r in g[(g[measure] < lo) | (g[measure] > hi)].sort_values(measure).iterrows():
                rows.append({
                    "image": r["image"], "group": group, "measure": measure, "value": r[measure],
                    "side": "low" if r[measure] < lo else "high", "p_low": lo, "p_high": hi,
                })
    return pd.DataFrame(rows, columns=["image", "group", "measure", "value", "side", "p_low", "p_high"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


# ---------------------------------------------------------------- summaries


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def group_rows(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [(g, df[df["group"] == g]) for g in scut.GROUPS]


def count_table(df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for name, g in [*group_rows(df), ("รวม", df)]:
        counts = [int((g["split"] == s).sum()) for s in SPLITS]
        rows.append([name, *map(str, counts), str(len(g))])
    return rows


def ratio_stats(df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for name, g in group_rows(df):
        cells = [name, str(len(g))]
        for measure in MEASURES:
            p10, p50, p90 = np.percentile(g[measure], [10, 50, 90]) if len(g) else (math.nan,) * 3
            cells += [f"{p50:.3f}", f"{p10:.3f}", f"{p90:.3f}"]
        rows.append(cells)
    return rows


def jaw_shares(df: pd.DataFrame) -> dict[str, tuple[int, dict[str, float]]]:
    """Share of each jaw grade for AF, CF and all women."""
    women = df[df["sex"] == "female"]
    out = {}
    for name, g in [("AF", women[women["group"] == "AF"]), ("CF", women[women["group"] == "CF"]), ("หญิงทั้งหมด", women)]:
        grades = [criteria.jaw_grade(v, "female") for v in g["jaw_ratio"]]
        out[name] = (len(g), {k: (grades.count(k) / len(g) if len(g) else math.nan) for k in criteria.GRADES})
    return out


def nose_shares(df: pd.DataFrame) -> dict[str, tuple[int, dict[str, float]]]:
    """Share of grade 0, grade 1 on the narrow and wide sides, and 2+, per group and sex."""
    sets = [*group_rows(df), ("หญิงทั้งหมด", df[df["sex"] == "female"]), ("ชายทั้งหมด", df[df["sex"] == "male"])]
    out = {}
    for name, g in sets:
        keys = []
        for v, sex in zip(g["nose_ratio"], g["sex"], strict=True):
            grade = criteria.nose_grade(v, sex)
            if grade == "1":
                grade = "1 narrow" if v < criteria.NOSE_CUTOFFS[sex].low else "1 wide"
            keys.append(grade)
        n = len(g)
        out[name] = (n, {k: (keys.count(k) / n if n else math.nan) for k in NOSE_KEYS})
    return out


def zy_position(m: Measured) -> tuple[str, str]:
    """Where Zy sits in its search band on each side: at eye level, inside, or at subnasale level."""
    fp = scut.frontal_points(m.points)
    frame = LevelFrame.from_eyes(fp.eye_l, fp.eye_r)
    y_sub = float(frame.to_level(fp.subnasale)[1])
    top, bottom = min(0.0, y_sub), max(0.0, y_sub)
    tol = ZY_EDGE_TOLERANCE * (bottom - top)
    out = []
    for zy in (m.result.zy_l, m.result.zy_r):
        y = float(frame.to_level(np.asarray(zy))[1])
        out.append("eye" if y - top <= tol else "subnasale" if bottom - y <= tol else "inside")
    return out[0], out[1]


def zy_table(df: pd.DataFrame, measured: dict[str, Measured]) -> list[list[str]]:
    """Share of sides with Zy at each band position, and of images whose two sides differ."""
    rows = []
    for group, g in group_rows(df):
        pos = [zy_position(measured[image]) for image in g["image"]]
        sides = [p for pair in pos for p in pair]
        n = len(sides)
        shares = [pct(sides.count(k) / n) if n else "-" for k in ZY_POSITIONS]
        differ = pct(sum(a != b for a, b in pos) / len(pos)) if pos else "-"
        rows.append([group, str(n), *shares, differ])
    return rows


def af_above_share(df: pd.DataFrame) -> float:
    af = df.loc[df["group"] == "AF", "jaw_ratio"]
    return float((af > criteria.JAW_FEMALE_CUTOFFS[0]).mean()) if len(af) else math.nan


# ---------------------------------------------------------------- figures


def _range(values: pd.Series, marks: list[float], step: float) -> tuple[float, float]:
    """Histogram range from p0.5 to p99.5 of all images, widened to show every cutoff."""
    lo = min(float(np.percentile(values, 0.5)), *marks)
    hi = max(float(np.percentile(values, 99.5)), *marks)
    return math.floor(lo / step) * step - step, math.ceil(hi / step) * step + step


def histogram_figure(df: pd.DataFrame, measure: str, title: str, cutoffs: dict[str, list[float]], step: float) -> Figure:
    """One panel per group with the cutoffs of its sex as dashed lines."""
    marks = [c for cs in cutoffs.values() for c in cs]
    lo, hi = _range(df[measure], marks, step)
    bins = np.arange(lo, hi + step / 2, step)
    fig = Figure(figsize=(11, 7), layout="constrained")
    axes = fig.subplots(2, 2, sharex=True).ravel()
    for ax, (group, g) in zip(axes, group_rows(df), strict=True):
        values = g[measure].to_numpy()
        inside = (values >= lo) & (values <= hi)
        outside = int((~inside).sum())
        ax.hist(values[inside], bins=bins, color="#4a7ab5", edgecolor="white", linewidth=0.4)
        sex = scut.parse_name(f"{group}1").sex
        for c in cutoffs.get(sex, []):
            ax.axvline(c, color="#c0392b", linestyle="--", linewidth=1.2)
        median = float(np.median(values)) if len(values) else math.nan
        ax.axvline(median, color="#222222", linewidth=1.2)
        note = "" if cutoffs.get(sex) else " (no grade)"
        off = f", {outside} off chart" if outside else ""
        ax.set_title(f"{group}{note}: n={len(values)}, median {median:.3f}{off}", fontsize=10)
        ax.set_ylabel("images")
    for ax in axes[2:]:
        ax.set_xlabel(f"{measure} (dashed = grade cutoffs, solid = median)")
    fig.suptitle(title)
    return fig


def save_figure(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=100, metadata={"Software": None})


def write_figures(df: pd.DataFrame, folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    jaw = histogram_figure(
        df, "jaw_ratio", "SCUT-FBP5500 jaw ratio Go-Go / Zy-Zy", {"female": list(criteria.JAW_FEMALE_CUTOFFS)}, 0.01
    )
    nose = histogram_figure(
        df, "nose_ratio", "SCUT-FBP5500 alar base / intercanthal ratio",
        {sex: [c.low, c.high, c.wide] for sex, c in criteria.NOSE_CUTOFFS.items()}, 0.02,
    )
    out = {"jaw": folder / JAW_FIGURE, "nose": folder / NOSE_FIGURE}
    save_figure(jaw, out["jaw"])
    save_figure(nose, out["nose"])
    return out


# ---------------------------------------------------------------- report


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def report_text(
    df: pd.DataFrame, skipped: list[tuple[str, str]], odd: pd.DataFrame, figures: dict[str, Path], zy: list[list[str]]
) -> str:
    af_share = af_above_share(df)
    jaw = jaw_shares(df)
    nose = nose_shares(df)
    f, m = criteria.NOSE_CUTOFFS["female"], criteria.NOSE_CUTOFFS["male"]
    j0, j1 = criteria.JAW_FEMALE_CUTOFFS
    status = (
        "**เกินครึ่ง ต้องหยุดตรวจการเลือกจุดก่อนเทรน** (model.md หัวข้อ 3.1)" if af_share > AF_STOP_SHARE
        else "ไม่ถึงครึ่ง จึงไม่เข้าเงื่อนไขหยุดของ model.md หัวข้อ 3.1"
    )
    parts = [
        "# label ของกรามและฐานจมูกจาก SCUT-FBP5500",
        "",
        (
            "สร้างด้วย `uv run python model/scripts/make_scut_labels.py` จากจุด 86 จุดของ SCUT-FBP5500 "
            "ด้วยกฎใน model.md หัวข้อ 5.2 ซึ่งอยู่ใน `facemeasure.face_geometry` และ `facemeasure.scut` "
            "รายงานนี้มีเฉพาะตัวเลขและกราฟ ภาพตรวจที่มีใบหน้าอยู่ใน `scut_label_check/` ของโฟลเดอร์ outputs ซึ่งห้าม commit"
        ),
        "",
        "- อัตราส่วนกราม = ระยะ Go–Go ÷ ระยะ Zy–Zy และอัตราส่วนฐานจมูก = ระยะจุด 65–69 ÷ ระยะจุด 43–55 เก็บเป็นสัดส่วน",
        (
            f"- ชุด test คือชุด test ของ SCUT (60/40) ชุด val คือ {VAL_SHARE:.0%} ของชุด train ของ SCUT "
            f"แบ่งแยก AF, AM, CF และ CM ด้วย seed `{SEED}` ก่อนตัดไฟล์ที่ไม่มี label"
        ),
        (
            "- ไฟล์ที่ไม่มี label คือไฟล์จุดที่อ่านไม่ได้ และภาพที่กฎหาจุดไม่ได้ เช่นกลางตา (จุด 59, 60) "
            "ไม่เรียงจากซ้ายไปขวาของภาพ หรือเส้นขอบหน้าไม่ถึงระดับมุมปาก (model.md หัวข้อ 5.2)"
        ),
        f"- ค่าผิดปกติคือค่าที่ต่ำกว่า p{OUTLIER_PERCENTILES[0]} หรือสูงกว่า p{OUTLIER_PERCENTILES[1]} ของกลุ่ม",
        "",
        "## จำนวนภาพ",
        "",
        md_table(["กลุ่ม", "train", "val", "test", "รวม"], count_table(df)),
        "",
        "### ไฟล์ที่ไม่มี label",
        "",
        md_table(["ภาพ", "เหตุ"], [[image, reason] for image, reason in skipped]) if skipped else "ไม่มี",
        "",
        "## การกระจายของอัตราส่วน",
        "",
        md_table(
            ["กลุ่ม", "n", "กราม มัธยฐาน", "กราม p10", "กราม p90", "ฐานจมูก มัธยฐาน", "ฐานจมูก p10", "ฐานจมูก p90"],
            ratio_stats(df),
        ),
        "",
        "## กรามของผู้หญิงเทียบเส้นแบ่ง (model.md หัวข้อ 3.1)",
        "",
        md_table(
            ["กลุ่ม", "n", f"≤ {j0} (เกรด 0)", f"{j0}–{j1} (เกรด 1)", f"> {j1} (เกรด 2+)"],
            [[name, str(n), *(pct(s[k]) for k in criteria.GRADES)] for name, (n, s) in jaw.items()],
        ),
        "",
        f"ผู้หญิงเอเชียที่อัตราส่วนเกิน {j0} มี {pct(af_share)} {status} ผู้ชายมีเฉพาะค่า ไม่มีเกณฑ์กราม",
        "",
        "## ฐานจมูกเทียบเส้นแบ่งตามเพศ (model.md หัวข้อ 3.4)",
        "",
        (
            f"หญิง: เกรด 0 ที่ {f.low}–{f.high}, 2+ เมื่อมากกว่า {f.wide} · "
            f"ชาย: เกรด 0 ที่ {m.low}–{m.high}, 2+ เมื่อมากกว่า {m.wide} · ฝั่งแคบได้สูงสุดเกรด 1"
        ),
        "",
        md_table(
            ["กลุ่ม", "n", "เกรด 0", "เกรด 1 ฝั่งแคบ", "เกรด 1 ฝั่งกว้าง", "เกรด 2+"],
            [[name, str(n), *(pct(s[k]) for k in NOSE_KEYS)] for name, (n, s) in nose.items()],
        ),
        "",
        "## ตำแหน่งของ Zy ในแถบที่ค้นหา",
        "",
        (
            "Zy คือจุดที่หน้ากว้างที่สุดระหว่างระดับกลางตากับระดับใต้จมูก ถ้าเส้นขอบกว้างขึ้นหรือแคบลงตลอดช่วงนี้ "
            "Zy จะอยู่ที่ปลายช่วง และถ้าสองข้างอยู่คนละตำแหน่ง เส้นเหลืองในภาพตรวจจะเอียงเทียบกับเส้นตา "
            f"ปลายช่วงนับเมื่อห่างจากปลายไม่เกิน {ZY_EDGE_TOLERANCE:.0%} ของความสูงของช่วง"
        ),
        "",
        md_table(["กลุ่ม", "จำนวนข้าง", "ที่ระดับกลางตา", "ในช่วง", "ที่ระดับใต้จมูก", "ภาพที่สองข้างต่างตำแหน่ง"], zy),
        "",
        "## Histogram",
        "",
        "เส้นประคือเส้นแบ่งเกรดของเพศนั้น เส้นทึบคือมัธยฐาน ค่าที่อยู่นอกช่วงของกราฟนับไว้ในชื่อกราฟ",
        "",
        f"![jaw ratio]({FIGURES_FOLDER}/{figures['jaw'].name})",
        "",
        f"![nose ratio]({FIGURES_FOLDER}/{figures['nose'].name})",
        "",
        "## ภาพที่ค่าผิดปกติ",
        "",
        f"{len(odd)} รายการจาก {odd['image'].nunique() if len(odd) else 0} ภาพ ภาพเหล่านี้อยู่ในแผ่น `outliers_*.png` ของภาพตรวจ",
        "",
        md_table(
            ["กลุ่ม", "อัตราส่วน", "ภาพ", "ค่า", f"p{OUTLIER_PERCENTILES[0]}", f"p{OUTLIER_PERCENTILES[1]}"],
            [[r.group, MEASURES[r.measure], r.image, f"{r.value:.4f}", f"{r.p_low:.4f}", f"{r.p_high:.4f}"]
             for r in odd.itertuples()],
        ) if len(odd) else "ไม่มี",
        "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------- check images


def random_check_images(df: pd.DataFrame) -> list[str]:
    picks = []
    for group in scut.GROUPS:
        names = sorted(df.loc[df["group"] == group, "image"], key=lambda n: rank("check", n))
        picks += names[:CHECK_PER_GROUP]
    return picks


def tile(images_dir: Path, image: str, m: Measured) -> Image.Image:
    path = images_dir / image
    if not path.is_file():
        return Image.new("RGB", (350, 350), (90, 90, 90))
    with Image.open(path) as im:
        return scut.draw_points(im, m.points, m.result)


def caption(row, note: str = "") -> str:
    text = f"{row.image} ({row.split})\njaw {row.jaw_ratio:.3f}  nose {row.nose_ratio:.3f}"
    return f"{text}\n{note}" if note else text


def write_sheets(
    images_dir: Path, out_dir: Path, prefix: str, items: list[tuple[str, str]], measured: dict[str, Measured]
) -> list[Path]:
    """Contact sheets of SHEET_TILES images each; items are (image, caption)."""
    written = []
    chunks = [items[i : i + SHEET_TILES] for i in range(0, len(items), SHEET_TILES)]
    for k, chunk in enumerate(tqdm(chunks, unit="sheet", desc=prefix), start=1):
        tiles = [tile(images_dir, image, measured[image]) for image, _ in chunk]
        path = out_dir / f"{prefix}_{k:02d}.png"
        contact_sheet(tiles, [c for _, c in chunk], columns=SHEET_COLUMNS).save(path)
        written.append(path)
    return written


def write_check_images(
    df: pd.DataFrame, odd: pd.DataFrame, measured: dict[str, Measured], images_dir: Path, out_dir: Path
) -> list[Path]:
    """random_01.png with 48 random images, then outliers_NN.png with every outlier."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in [*out_dir.glob("random_*.png"), *out_dir.glob("outliers_*.png")]:
        old.unlink()
    rows = {r.image: r for r in df.itertuples()}
    notes: dict[str, list[str]] = {}
    for r in odd.itertuples():
        mark, p = ("<", OUTLIER_PERCENTILES[0]) if r.side == "low" else (">", OUTLIER_PERCENTILES[1])
        notes.setdefault(r.image, []).append(f"{r.measure.removesuffix('_ratio')} {mark} p{p}")
    written = write_sheets(
        images_dir, out_dir, "random", [(n, caption(rows[n])) for n in random_check_images(df)], measured
    )
    if notes:
        items = [(n, caption(rows[n], ", ".join(v))) for n, v in notes.items()]
        written += write_sheets(images_dir, out_dir, "outliers", items, measured)
    return written


# ---------------------------------------------------------------- main


def run(scut_dir: Path, labels_dir: Path, report_dir: Path, check_dir: Path | None) -> pd.DataFrame:
    split_dir = scut_dir.joinpath(*SPLIT_FOLDER)
    train = read_split_file(split_dir / "train.txt")
    test = read_split_file(split_dir / "test.txt")
    splits = assign_splits(list(train), list(test))
    scores = {**train, **test}
    images = sorted(scores, key=natural_key)

    landmark_dir = scut_dir / LANDMARK_FOLDER
    measured, skipped = measure_all(landmark_dir, images)
    listed = {Path(i).stem for i in images}
    for extra in sorted(p for p in landmark_dir.glob("*.pts") if p.stem not in listed):
        skipped.append((f"{extra.stem}.jpg", "not in the SCUT train/test split files"))

    df = label_table(scores, splits, measured)
    labels_dir.mkdir(parents=True, exist_ok=True)
    write_csv(df, labels_dir / LABELS_NAME)
    write_csv(pd.DataFrame(skipped, columns=["image", "reason"]), labels_dir / SKIPPED_NAME)

    odd = outliers(df)
    figures = write_figures(df, report_dir / FIGURES_FOLDER)
    report = report_text(df, skipped, odd, figures, zy_table(df, measured))
    (report_dir / REPORT_NAME).write_text(report, encoding="utf-8", newline="\n")

    if check_dir is not None:
        write_check_images(df, odd, measured, scut_dir / IMAGE_FOLDER, check_dir)
    return df


def print_summary(df: pd.DataFrame, labels_dir: Path, report_dir: Path) -> None:
    print(f"เขียน {len(df)} แถวไว้ที่ {labels_dir / LABELS_NAME}")
    skipped = pd.read_csv(labels_dir / SKIPPED_NAME)
    print(f"ไม่มี label {len(skipped)} ไฟล์: " + ", ".join(f"{r.image} ({r.reason})" for r in skipped.itertuples()))
    print("ชุด: " + " / ".join(f"{s} {int((df['split'] == s).sum())}" for s in SPLITS))
    for group, g in group_rows(df):
        print(f"{group} {len(g)} ภาพ · กรามมัธยฐาน {g['jaw_ratio'].median():.3f} · ฐานจมูกมัธยฐาน {g['nose_ratio'].median():.3f}")
    share = af_above_share(df)
    print(f"ผู้หญิงเอเชียที่กรามเกิน {criteria.JAW_FEMALE_CUTOFFS[0]}: {pct(share)}")
    if share > AF_STOP_SHARE:
        print("เกินครึ่ง: หยุดตรวจการเลือกจุดก่อนเทรน (model.md หัวข้อ 3.1)")
    print(f"รายงานอยู่ที่ {report_dir / REPORT_NAME}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Jaw and alar-base labels from the SCUT-FBP5500 landmarks")
    ap.add_argument("--no-check-images", action="store_true", help="skip the contact sheets with points drawn")
    args = ap.parse_args(argv)
    labels_dir = paths.ensure_dir("labels")
    report_dir = paths.ensure_dir("reports")
    check_dir = None if args.no_check_images else paths.ensure_dir("outputs", CHECK_FOLDER)
    df = run(paths.path("scut"), labels_dir, report_dir, check_dir)
    print_summary(df, labels_dir, report_dir)
    if check_dir is not None:
        print(f"ภาพตรวจอยู่ที่ {check_dir} (มีใบหน้า ห้าม commit)")


if __name__ == "__main__":
    main()
