"""Check of the extracted features: model/reports/features.md and outputs/features_check/.

Run from the project root after both tasks of extract_features.py:

    uv run python model/scripts/extract_features.py report

The report holds numbers and charts only, so it can be committed:
- crops by split, copy and source; copies without a face by target IOD; quality codes
- IOD found on the copies against the target (sources.csv has the IOD of each source)
- time per crop on the device in use and on the CPU, and of one api request with three
  crops (api.md section 7)
- cosine similarity of CPU and device features on the same CHECK_CROPS crops, and of
  copies made again from their seeds through the api's way in against the stored ones
- file sizes
The check images in outputs/features_check/ show faces: never commit them.
"""

import hashlib
import itertools
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from PIL import Image, ImageDraw

import extract_features as ef
import grading_set
from facemeasure import codes, degrade, draw, image_io, paths, quality
from facemeasure.backbone import FEATURE_TYPES, Backbone
from facemeasure.frontal import analyze_frontal
from facemeasure.landmarks import FaceModels
from facemeasure.model_input import analyze_for_model

REPORT_NAME = "features.md"
FIGURES_FOLDER = "figures"
CHECK_FOLDER = "features_check"
CHECK_CROPS = 64  # half whole-face, half under-eye
CHECK_PER_GROUP = 6  # SCUT: 4 groups, FFHQ: 4 age bands -> 24 each
TIMING_IMAGES = 10
TIMING_WARMUP = 2
TIMING_FRAME = (1920, 1080)
TIMING_FRAME_IOD = 100.0
MIN_FOUND = 0.99  # share of the fixed copies at the lowest IOD with a face (done check of the extraction)
CELL = 464
LABEL_HEIGHT = 40
CROP_GAP = 8
SOURCE_COLORS = {"scut": "#4a7ab5", "ffhq": "#d35400"}


def pick(values, n: int, salt: str) -> list:
    """n values chosen the same way every run: smallest sha256 of salt/value."""
    return sorted(values, key=lambda v: hashlib.sha256(f"{salt}/{v}".encode()).hexdigest())[:n]


def pct(part: int, whole: int) -> str:
    return f"{part} ({100 * part / whole:.1f}%)" if whole else "0"


def fmt(x, digits: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{digits}f}"


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def source_of(task: str, df: pd.DataFrame) -> pd.Series:
    return pd.Series("scut", index=df.index) if task == ef.FACE else df["source"]


def fixed_names(fixed: tuple[float, float]) -> tuple[str, str]:
    return f"fixed {fixed[0]:g}", f"fixed {fixed[1]:g}"


def copy_group(df: pd.DataFrame, fixed: tuple[float, float]) -> pd.Series:
    """The fixed copy at the lowest IOD, the one at the highest, or train, per row."""
    out = pd.Series("train", index=df.index)
    for copy, name in enumerate(fixed_names(fixed), start=1):
        out[(df["kind"] == ef.FIXED) & (df["copy"] == copy)] = name
    return out


# ---------------------------------------------------------------- counts


def counts_section(indexes: dict[str, pd.DataFrame]) -> str:
    out = ["## จำนวนครอป", "", "หนึ่งแถวต่อครอป ใต้ตานับสองข้างแยกกัน ชุดของ face มาจาก `scut_labels.csv` และของ undereye มาจาก `manifest.csv`", ""]
    for task, df in indexes.items():
        src = source_of(task, df)
        g = df.assign(source=src).groupby(["source", "split", "kind", "copy"], sort=False)
        rows = []
        for (s, split, kind, copy), part in g:
            rows.append([s, split, kind, int(copy), len(part), int(part["row"].notna().sum())])
        rows.sort(key=lambda r: (r[0], ef.SPLITS.index(r[1]), r[2] != ef.TRAIN, r[3]))
        rows.append(["**รวม**", "", "", "", len(df), int(df["row"].notna().sum())])
        out += [f"### {task}", "", table(["แหล่ง", "ชุด", "ชนิด", "สำเนา", "ครอป", "มี feature"], rows), ""]
        if task == ef.FACE:
            by = df.groupby(["group", "split"]).size().unstack(fill_value=0).reindex(columns=list(ef.SPLITS), fill_value=0)
            out += [table(["กลุ่ม", *ef.SPLITS], [[k, *map(int, v)] for k, v in by.iterrows()]), ""]
    return "\n".join(out)


def noface_section(indexes: dict[str, pd.DataFrame], low: float, high: float, fixed: tuple[float, float]) -> tuple[str, dict]:
    out = ["## สำเนาที่ไม่พบใบหน้าแยกตาม IOD เป้าหมาย", "",
           "นับสำเนา (ไม่นับข้างของใต้ตาซ้ำ) สำเนาแบบ train แบ่งช่วงเป้าหมายละ 5 px", ""]
    checks = {}
    rows = []
    edges = list(np.arange(low, high, 5.0)) + [high + 1e-9]
    for task, df in indexes.items():
        copies = df.drop_duplicates(["image", "kind", "copy"]).copy()
        copies["source"] = source_of(task, copies)
        for s, part in copies.groupby("source"):
            groups = [(name, part[(part["kind"] == ef.FIXED) & (part["copy"] == copy)])
                      for copy, name in enumerate(fixed_names(fixed), start=1)]
            train = part[part["kind"] == ef.TRAIN]
            for a, b in itertools.pairwise(edges):
                groups.append((f"train {a:g}–{min(b, high):g}", train[(train["target_iod"] >= a) & (train["target_iod"] < b)]))
            for name, g in groups:
                if len(g) == 0:
                    continue
                lost = int((g["n_faces"] == 0).sum())
                rows.append([task, s, name, len(g), pct(lost, len(g))])
            mins = groups[0][1]
            if len(mins):
                checks[f"{task} {s}"] = float((mins["n_faces"] > 0).mean())
    out += [table(["งาน", "แหล่ง", "สำเนา", "n", "ไม่พบใบหน้า"], rows), ""]
    ok = all(v >= MIN_FOUND for v in checks.values())
    text = ", ".join(f"{k} {100 * v:.2f}%" for k, v in checks.items())
    out += [f"สำเนา fixed ที่ IOD ต่ำสุด ({fixed[0]:g} px) พบใบหน้า: {text} เกณฑ์คือไม่ต่ำกว่า {100 * MIN_FOUND:.0f}% ผล: **{'ผ่าน' if ok else 'ไม่ผ่าน ให้ดูภาพตรวจก่อน'}**", ""]
    return "\n".join(out), checks


def codes_section(indexes: dict[str, pd.DataFrame], low: float, fixed: tuple[float, float]) -> str:
    out = ["## รหัสคุณภาพของสำเนา", "",
           "รหัสของบริเวณนั้นจากภาพที่ย่อแล้ว ซึ่งเป็นค่าคุณภาพจากรอบแรกของ `analyze_for_model` เหมือนที่ api รายงาน ครอปหนึ่งมีได้หลายรหัส",
           (f"สำเนา fixed ชุดแรกตั้งไว้ที่ {fixed[0]:g} px สูงกว่าเกณฑ์ `min_iod_px` ({low:g} px) {degrade.FIXED_IOD_MARGIN_PX:g} px "
            "เพราะ IOD ที่วัดบนสำเนาคลาดได้ราว ±1.5% การย่อภาพรุ่น 1 ตั้งไว้เท่าเกณฑ์พอดี สำเนาชุดนี้จึงได้ `face_too_small` ราวครึ่งหนึ่ง "
            "`face_too_small` ที่ยังเหลือส่วนใหญ่มาจากภาพเดิมที่ใบหน้าเล็กกว่าเกณฑ์อยู่แล้ว ซึ่งไม่ถูกขยาย (ดูหัวข้อ IOD)"), ""]
    for task, df in indexes.items():
        df = df.assign(source=source_of(task, df), group_name=copy_group(df, fixed))
        cols = ["reasons_jaw", "reasons_nose_base"] if task == ef.FACE else ["reasons"]
        header, columns = ["รหัส"], []
        for s in sorted(df["source"].unique()):
            for g in (*fixed_names(fixed), "train"):
                part = df[(df["source"] == s) & (df["group_name"] == g)]
                for c in cols:
                    name = f"{s} {g}" + (f" {c.removeprefix('reasons_')}" if len(cols) > 1 else "")
                    header.append(name)
                    columns.append(part[c])
        rows = []
        for code in (*codes.FRONTAL_CODES, "ผ่านทุกข้อ"):
            row = [code]
            for col in columns:
                n = int((col == "").sum()) if code == "ผ่านทุกข้อ" else int(col.str.split(";").map(lambda v, c=code: c in v).sum())
                row.append(pct(n, len(col)))
            rows.append(row)
        out += [f"### {task}", "", table(header, rows), ""]
    return "\n".join(out)


# ---------------------------------------------------------------- IOD


def iod_frame(task: str, df: pd.DataFrame) -> pd.DataFrame:
    copies = df.drop_duplicates(["image", "kind", "copy"]).copy()
    copies["source"] = source_of(task, copies)
    src = pd.read_csv(ef.task_folder(task) / ef.SOURCES_NAME, encoding="utf-8").rename(columns={"iod_px": "source_iod", "n_faces": "source_faces"})
    m = copies.merge(src, on="image", how="left")
    m["task"] = task
    m["shrunk"] = m["source_iod"] > m["target_iod"]
    m["expected"] = np.where(m["shrunk"], m["target_iod"], m["source_iod"])
    m["error"] = m["iod_px"] / m["expected"] - 1
    return m


def iod_section(frames: list[pd.DataFrame], low: float, high: float, fixed: tuple[float, float], figure: str) -> str:
    out = ["## IOD ที่ได้จริงเทียบกับเป้าหมาย", "",
           ("IOD ที่ได้จริงคือค่าที่ `analyze_frontal` วัดบนสำเนา ภาพที่ใบหน้าเดิมเล็กกว่าเป้าหมายไม่ถูกขยาย จึงเทียบกับ IOD ของภาพเดิมแทน "
            f"สำเนาที่วัดได้เกิน {high:g} px ถูก `analyze_for_model` ย่ออีกรอบก่อนครอป เหมือนที่ api ทำ"), ""]
    rows = []
    for m in frames:
        for (task, s), part in m[m["iod_px"].notna()].groupby(["task", "source"]):
            for shrunk, p in part.groupby("shrunk"):
                e = p["error"].to_numpy()
                rows.append([task, s, "ย่อ (เทียบเป้าหมาย)" if shrunk else "ไม่ย่อ (เทียบภาพเดิม)", len(p),
                             f"{100 * np.median(e):+.2f}%", f"{100 * np.percentile(e, 5):+.2f}%", f"{100 * np.percentile(e, 95):+.2f}%",
                             f"{100 * np.mean(np.abs(e) <= 0.01):.1f}%", f"{100 * np.mean(np.abs(e) <= 0.02):.1f}%",
                             int((p["iod_px"] > high).sum())])
    out += [table(["งาน", "แหล่ง", "การย่อ", "n", "มัธยฐาน", "p5", "p95", "คลาด ≤ 1%", "คลาด ≤ 2%", f"เกิน {high:g} px"], rows), ""]
    rows = []
    for m in frames:
        first = m[(m["kind"] == ef.FIXED) & (m["copy"] == 1) & m["iod_px"].notna()]
        for (task, s), part in first.groupby(["task", "source"]):
            shrunk, kept = part[part["shrunk"]], part[~part["shrunk"]]
            rows.append([task, s, len(shrunk), pct(int((shrunk["iod_px"] < low).sum()), len(shrunk)),
                         len(kept), pct(int((kept["iod_px"] < low).sum()), len(kept))])
    out += [f"สำเนา fixed ชุดแรก (เป้าหมาย {fixed[0]:g} px) ที่วัดได้ต่ำกว่าเกณฑ์ `min_iod_px` {low:g} px", "",
            table(["งาน", "แหล่ง", "ย่อ n", "ย่อแล้วต่ำกว่าเกณฑ์", "ไม่ย่อ n", "ไม่ย่อและต่ำกว่าเกณฑ์"], rows), "",
            f"![IOD]({FIGURES_FOLDER}/{figure})", ""]
    return "\n".join(out)


def iod_figure(frames: list[pd.DataFrame], low: float, high: float, path: Path) -> None:
    m = pd.concat(frames, ignore_index=True)
    m = m[m["iod_px"].notna()]
    groups = list(m.groupby(["task", "source"]))
    fig = Figure(figsize=(5 * len(groups), 4.6), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(groups)))
    for ax, ((task, s), part) in zip(axes, groups, strict=True):
        for shrunk, color, label in ((True, SOURCE_COLORS[s], "shrunk"), (False, "#888888", "kept size")):
            p = part[part["shrunk"] == shrunk]
            ax.scatter(p["target_iod"], p["iod_px"], s=4, alpha=0.4, c=color, linewidths=0, label=f"{label} ({len(p)})")
        ax.plot([low, high], [low, high], color="#c0392b", linewidth=0.8)
        ax.axhline(low, color="#c0392b", linestyle="--", linewidth=0.6)
        ax.set_xlabel("target IOD (px)")
        ax.set_ylabel("IOD found on the copy (px)")
        ax.set_title(f"{task} / {s}", fontsize=10)
        ax.legend(fontsize=8, markerscale=3)
    fig.suptitle("IOD found by analyze_frontal on the copies against the target (red: target, dashed: min_iod_px)")
    fig.savefig(path, dpi=100, metadata={"Software": None})


def quality_figure(indexes: dict[str, pd.DataFrame], low: float, high: float, path: Path) -> None:
    """Share of copies with face_too_small by target IOD: train copies in 2.5 px bins, fixed copies as squares."""
    fig = Figure(figsize=(11, 4), layout="constrained")
    ax = fig.subplots()
    bins = np.arange(low, high + 2.5, 2.5)
    for task, df in indexes.items():
        col = "reasons_jaw" if task == ef.FACE else "reasons"
        copies = df.assign(source=source_of(task, df)).drop_duplicates(["image", "kind", "copy"])
        for s, part in copies.groupby("source"):
            train = part[part["kind"] == ef.TRAIN]
            k = np.clip(np.digitize(train["target_iod"], bins) - 1, 0, len(bins) - 2)
            mids = (bins[:-1] + bins[1:]) / 2
            small = [train.loc[k == i, col].str.contains(codes.FACE_TOO_SMALL).mean() if (k == i).any() else np.nan for i in range(len(mids))]
            (line,) = ax.plot(mids, small, marker="o", markersize=3, label=f"{task} {s}: train")
            fixed = part[part["kind"] == ef.FIXED].groupby("target_iod")[col].agg(lambda c: c.str.contains(codes.FACE_TOO_SMALL).mean())
            ax.plot(fixed.index, fixed.to_numpy(), linestyle="none", marker="s", markersize=7, color=line.get_color(),
                    label=f"{task} {s}: fixed")
    ax.set_xlabel("target IOD (px)")
    ax.set_ylabel("share of copies")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.suptitle("face_too_small on the copies by target IOD")
    fig.savefig(path, dpi=100, metadata={"Software": None})


# ---------------------------------------------------------------- regenerated crops, timing


def workers(models: FaceModels, config: quality.QualityConfig, embed=None, device="") -> ef.Workers:
    return ef.Workers(
        analyze=lambda im: analyze_frontal(im, models, config.frontal),
        model_input=lambda im: analyze_for_model(im, models, degrade.TRAIN_IOD_MAX, config.frontal),
        embed=embed,
        device=device,
    )


def source_paths() -> dict[tuple[str, str], Path]:
    out = {}
    for task in ef.TASKS:
        try:
            for s in ef.sources_for(task):
                out[(task, s.image)] = s.path
        except FileNotFoundError:
            pass
    return out


def check_rows(indexes: dict[str, pd.DataFrame]) -> list[tuple[str, pd.Series]]:
    """CHECK_CROPS rows with features: half whole-face, half under-eye (both sides of each copy)."""
    out = []
    half = CHECK_CROPS // 2
    if ef.FACE in indexes:
        df = indexes[ef.FACE][indexes[ef.FACE]["row"].notna()]
        keys = pick(df.index.tolist(), half, "face-check")
        out += [(ef.FACE, df.loc[k]) for k in sorted(keys)]
    if ef.UNDEREYE in indexes:
        df = indexes[ef.UNDEREYE][indexes[ef.UNDEREYE]["row"].notna()]
        copies = df.drop_duplicates(["image", "kind", "copy"])
        keys = pick([f"{r.image}|{r.kind}|{r.copy}" for r in copies.itertuples()], half // 2, "undereye-check")
        for key in sorted(keys):
            image, kind, copy = key.split("|")
            both = df[(df["image"] == image) & (df["kind"] == kind) & (df["copy"] == int(copy))]
            out += [(ef.UNDEREYE, r) for _, r in both.iterrows()]
    return out


def stored_features(task: str, rows: list[pd.Series]) -> dict[str, np.ndarray]:
    folder = ef.task_folder(task)
    out = {f: [] for f in FEATURE_TYPES}
    for r in rows:
        with np.load(ef.shard_path(folder, int(r["shard"])), allow_pickle=False) as z:
            for f in FEATURE_TYPES:
                out[f].append(z[f][int(r["row"])].astype(np.float32))
    return {f: np.stack(v) for f, v in out.items()}


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def regenerated_crops(rows, paths_by, w, config) -> list[np.ndarray]:
    crops, cache = [], {}
    for task, r in rows:
        key = (task, r["image"], r["kind"], int(r["copy"]))
        if key not in cache:
            cache[key] = ef.regenerate(task, r["image"], paths_by[(task, r["image"])], r["kind"], int(r["copy"]), w, config)[1]
        crops.append(ef.crop_of(task, cache[key], r.get("side")))
    return crops


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def time_crops(bb: Backbone, crops: np.ndarray, repeats: int = 3) -> float:
    """Seconds per crop at the device's batch size (median of repeats, after a warm-up)."""
    bb.features(crops[: bb.batch_size])
    sync(bb.device)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        bb.features(crops)
        sync(bb.device)
        times.append((time.perf_counter() - t) / len(crops))
    return statistics.median(times)


def webcam_frame(image: np.ndarray, iod: float) -> np.ndarray:
    """A TIMING_FRAME-sized frame with the face shrunk to TIMING_FRAME_IOD px, on the image's edge colour."""
    small = degrade.shrink(image, TIMING_FRAME_IOD / iod) if iod > TIMING_FRAME_IOD else image
    w, h = TIMING_FRAME
    frame = np.empty((h, w, 3), dtype=np.uint8)
    frame[:] = np.median(small[0], axis=0).astype(np.uint8)
    sh, sw = small.shape[:2]
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    frame[y0 : y0 + sh, x0 : x0 + sw] = small
    return frame


def time_requests(images: list[np.ndarray], models, config, backbones: dict[str, Backbone]) -> dict[str, dict[str, float]]:
    """Median seconds of the api's work for one request: analyze_for_model, then DINOv2 on three crops at once."""
    out = {}
    for name, bb in backbones.items():
        mp_t, dino_t = [], []
        for k, image in enumerate([*images[:TIMING_WARMUP], *images]):
            t = time.perf_counter()
            r = analyze_for_model(image, models, degrade.TRAIN_IOD_MAX, config.frontal)
            t1 = time.perf_counter()
            batch = np.stack([r.crops.face.image, r.crops.undereye_left.image, r.crops.undereye_right.image])
            bb.features(batch, batch=len(batch))
            sync(bb.device)
            t2 = time.perf_counter()
            if k >= TIMING_WARMUP:
                mp_t.append(t1 - t)
                dino_t.append(t2 - t1)
        total = [a + b for a, b in zip(mp_t, dino_t, strict=True)]
        out[name] = {"mediapipe": statistics.median(mp_t), "dinov2": statistics.median(dino_t),
                     "total": statistics.median(total), "max": max(total)}
    return out


def device_section(indexes, config, timing_images) -> str:
    out = ["## เวลาและความตรงของ feature", ""]
    paths_by = source_paths()
    rows = check_rows(indexes)
    with FaceModels() as models:
        device_bb = Backbone()
        cpu_bb = device_bb if device_bb.device.type == "cpu" else Backbone("cpu")
        w = workers(models, config)
        by_task = {}
        for task, r in rows:
            by_task.setdefault(task, []).append(r)
        crops = np.stack(regenerated_crops(rows, paths_by, w, config)) if rows else np.zeros((0, 224, 224, 3), np.uint8)
        dev_f = device_bb.features(crops)
        cpu_f = cpu_bb.features(crops)
        stored = {t: stored_features(t, rs) for t, rs in by_task.items()}
        stored_all = {f: np.concatenate([stored[t][f] for t in by_task]) for f in FEATURE_TYPES} if rows else {}

        t_dev = time_crops(device_bb, crops) if len(crops) else math.nan
        t_cpu = time_crops(cpu_bb, crops) if len(crops) else math.nan
        frames = [webcam_frame(im, iod) for im, iod in timing_images]
        full = time_requests([im for im, _ in timing_images], models, config, {str(device_bb.device): device_bb, "cpu": cpu_bb})
        framed = time_requests(frames, models, config, {str(device_bb.device): device_bb, "cpu": cpu_bb})

    dev = str(device_bb.device)
    out += ["### เวลาต่อครอป", "",
            (f"DINOv2 ViT-B/14 บน {dev} ชุดละ {device_bb.batch_size} และบน cpu ชุดละ {cpu_bb.batch_size} ครอปจริง {len(crops)} ครอป "
             f"มัธยฐานของ 3 รอบหลังอุ่นเครื่อง ใช้ {torch.get_num_threads()} thread บน CPU"), "",
            ("เวลาขึ้นกับความร้อนของเครื่อง เครื่องที่ไม่มีพัดลม เช่น MacBook Air ช้าลงได้หลายเท่าเมื่อรันต่อจากงานหนัก ถ้าจะใช้ตัวเลขในหัวข้อนี้ตัดสินใจเรื่องขนาดของ DINOv2 (api.md หัวข้อ 7) ให้รัน report ตอนเครื่องเย็น"), "",
            table(["อุปกรณ์", "ms ต่อครอป"], [[dev, fmt(1000 * t_dev, 1)], ["cpu", fmt(1000 * t_cpu, 1)]]), "",
            "### หนึ่งคำขอของ api", "",
            (f"`analyze_for_model` (MediaPipe) แล้ว DINOv2 กับสามครอปในชุดเดียว มัธยฐานของ {TIMING_IMAGES} ภาพหลังอุ่นเครื่อง {TIMING_WARMUP} ภาพ "
             "ยังไม่รวมการอ่านภาพ การหาเส้นขอบภาพด้านข้าง และการรันหัว (หัวเป็นชั้นเชิงเส้นซึ่งเร็วมาก)"), ""]
    req_rows = []
    for label, res in ((f"FFHQ 1024×1024 IOD ราว 245 px (ย่อเหลือ {degrade.TRAIN_IOD_MAX:g} px แล้วหาจุดสองรอบ)", full),
                       (f"เฟรม {TIMING_FRAME[0]}×{TIMING_FRAME[1]} IOD {TIMING_FRAME_IOD:g} px (หาจุดรอบเดียว)", framed)):
        for name, v in res.items():
            req_rows.append([label, name, fmt(1000 * v["mediapipe"], 0), fmt(1000 * v["dinov2"], 0), fmt(1000 * v["total"], 0), fmt(1000 * v["max"], 0)])
    out += [table(["ภาพ", "DINOv2 บน", "MediaPipe (ms)", "DINOv2 3 ครอป (ms)", "รวม (ms)", "รวมสูงสุด (ms)"], req_rows), "",
            "MediaPipe รันบน CPU ทุกกรณี คอลัมน์ \"DINOv2 บน\" บอกอุปกรณ์ของ DINOv2 เท่านั้น", ""]

    out += ["### feature จาก CPU เทียบกับอุปกรณ์ที่ใช้", "",
            (f"ครอปเดียวกัน {len(crops)} ครอป ({len(by_task.get(ef.FACE, []))} ทั้งใบหน้า และ {len(by_task.get(ef.UNDEREYE, []))} ใต้ตา) "
             "ที่สร้างขึ้นใหม่จาก seed ผ่าน `analyze_for_model` สามแถวล่างเทียบ feature จากอุปกรณ์ที่ใช้กับที่เก็บไว้ใน shard "
             "ซึ่งเป็นการตรวจ \"ความตรงกับตอนเตรียมข้อมูล\" ขั้นแรก (api.md หัวข้อ 7) grid4 เก็บเป็น float16"), ""]
    rows_out = []
    for f in FEATURE_TYPES:
        c = cosine(cpu_f[f], dev_f[f]) if len(crops) else np.array([math.nan])
        rows_out.append([f"cpu กับ {dev}", f, fmt(float(c.min()), 7), fmt(float(np.median(c)), 7)])
    for f in FEATURE_TYPES:
        c = cosine(dev_f[f], stored_all[f]) if rows else np.array([math.nan])
        rows_out.append([f"{dev} สร้างใหม่กับที่เก็บไว้", f, fmt(float(c.min()), 7), fmt(float(np.median(c)), 7)])
    out += [table(["เทียบ", "feature", "cosine ต่ำสุด", "มัธยฐาน"], rows_out), ""]
    return "\n".join(out)


def timing_images(n: int = TIMING_IMAGES + TIMING_WARMUP) -> list[tuple[np.ndarray, float]]:
    """Full-resolution FFHQ graded images of the test split and their IOD from sources.csv."""
    folder = ef.task_folder(ef.UNDEREYE)
    if not (folder / ef.SOURCES_NAME).is_file():
        return []
    src = pd.read_csv(folder / ef.SOURCES_NAME, encoding="utf-8").set_index("image")["iod_px"]
    manifest = grading_set.read_manifest(paths.path("grading") / ef.MANIFEST_NAME)
    ff = manifest[(manifest["role"] == "graded") & (manifest["source"] == "ffhq") & (manifest["split"] == "test")]
    chosen = pick(ff["image"].tolist(), n, "timing")
    out = []
    for image in chosen:
        row = ff[ff["image"] == image].iloc[0]
        if math.isfinite(src.get(image, math.nan)):
            out.append((image_io.read_image(grading_set.image_path(row)), float(src[image])))
    return out


# ---------------------------------------------------------------- sizes


def size_section() -> str:
    root = paths.path("features")
    rows, total = [], 0
    for task in ef.TASKS:
        folder = root / task
        if not folder.is_dir():
            continue
        shards = sorted(folder.glob("shard_*.npz"))
        shard_bytes = sum(p.stat().st_size for p in shards)
        other = sum(p.stat().st_size for p in folder.iterdir() if p.is_file() and p not in shards)
        rows.append([task, len(shards), f"{shard_bytes / 1e6:.1f}", f"{other / 1e6:.2f}"])
        total += shard_bytes + other
    meta = root / ef.META_NAME
    total += meta.stat().st_size if meta.is_file() else 0
    return "\n".join(["## ขนาดไฟล์", "", table(["งาน", "shard", "shard (MB)", "index และไฟล์อื่น (MB)"], rows), "",
                      f"รวม `data/features/{{face,undereye}}/` และ `meta.json` {total / 1e6:.1f} MB ไม่บีบอัด เพื่อให้อ่านเร็ว", ""])


# ---------------------------------------------------------------- check images


def fit(image: np.ndarray, side: int) -> Image.Image:
    """The image at 1:1 when it fits in side x side, else shrunk to fit."""
    im = Image.fromarray(image)
    if max(im.size) > side:
        s = side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.LANCZOS)
    return im


def check_sheet(original: np.ndarray, source_iod, copies: list[tuple[str, np.ndarray, object]]) -> Image.Image:
    font = draw.load_font(13)
    width = 3 * CELL
    crop_w = (width - 5 * CROP_GAP) // 6
    sheet = Image.new("RGB", (width, LABEL_HEIGHT + CELL + LABEL_HEIGHT + crop_w), draw.SHEET_BACKGROUND)
    d = ImageDraw.Draw(sheet)
    panels = [(f"original {original.shape[1]}x{original.shape[0]}\nIOD {fmt(source_iod, 1)} px", original)]
    for label, image, r in copies:
        iod = r.quality.iod_px if r.quality is not None else None
        failed = ",".join(r.failed) if r.failed else "pass"
        panels.append((f"{label}: {image.shape[1]}x{image.shape[0]} at 1:1\nIOD {fmt(iod, 1)} px  {failed}", image))
    for k, (label, image) in enumerate(panels):
        x = k * CELL
        d.multiline_text((x + 4, 4), label, fill=draw.WHITE, font=font, spacing=3)
        tile = fit(image, CELL)
        sheet.paste(tile, (x + (CELL - tile.width) // 2, LABEL_HEIGHT + (CELL - tile.height) // 2))
    y = LABEL_HEIGHT + CELL
    for c, (label, _, r) in enumerate(copies):
        x0 = 3 * c * (crop_w + CROP_GAP)
        d.text((x0 + 4, y + 4), f"{label}: face, under-eye R, under-eye L (224 px)", fill=draw.WHITE, font=font)
        if r.crops is None:
            continue
        for j, crop in enumerate((r.crops.face, r.crops.undereye_right, r.crops.undereye_left)):
            tile = Image.fromarray(crop.image).resize((crop_w, crop_w), Image.Resampling.NEAREST)
            sheet.paste(tile, (x0 + j * (crop_w + CROP_GAP), y + LABEL_HEIGHT))
    return sheet


def write_check_images(config: quality.QualityConfig) -> int:
    folder = paths.ensure_dir("outputs", CHECK_FOLDER)
    for p in folder.glob("*.png"):
        p.unlink()
    chosen = []
    face = [s for s in ef.face_sources() if s.split != "train"]
    for g in sorted({s.extra["group"] for s in face}):
        names = pick([s.image for s in face if s.extra["group"] == g], CHECK_PER_GROUP, "check-scut")
        chosen += [(ef.FACE, s) for s in face if s.image in names]
    manifest = grading_set.read_manifest(paths.path("grading") / ef.MANIFEST_NAME).set_index("image")
    eye = [s for s in ef.undereye_sources() if s.split != "train" and s.extra["source"] == "ffhq"]
    for band in sorted({manifest.loc[s.image, "age_band"] for s in eye}):
        names = pick([s.image for s in eye if manifest.loc[s.image, "age_band"] == band], CHECK_PER_GROUP, "check-ffhq")
        chosen += [(ef.UNDEREYE, s) for s in eye if s.image in names]
    fixed = degrade.fixed_iod(config)
    with FaceModels() as models:
        w = workers(models, config)
        for task, s in chosen:
            image = image_io.read_image(s.path)
            first = analyze_frontal(image, models, config.frontal)
            iod = first.quality.iod_px if first.face_count else None
            copies = []
            for copy, target in enumerate(fixed, start=1):
                params = degrade.copy_params(task, s.image, ef.FIXED, copy, config)
                copy_image = degrade.degrade(image, iod, params).image
                copies.append((f"fixed {target:g} px", copy_image, w.model_input(copy_image)))
            source = "scut" if task == ef.FACE else s.extra["source"]
            check_sheet(image, iod, copies).save(folder / f"{source}_{Path(s.image).stem}.png", optimize=True)
    return len(chosen)


# ---------------------------------------------------------------- report


def header(meta: dict) -> str:
    bb = meta["backbone"]
    rt = meta["runtime"]
    tasks = meta["tasks"]
    devices = sorted({d for t in tasks.values() for d in t["devices"]})
    low, high = meta["train_face_iod_px"]["min"], meta["train_face_iod_px"]["max"]
    lines = [
        "# ตรวจ feature ของ DINOv2",
        "",
        ("สร้างด้วย `uv run python model/scripts/extract_features.py report` จาก `data/features/` รายงานนี้มีเฉพาะตัวเลขและกราฟ "
         f"ภาพตรวจที่มีใบหน้าอยู่ใน `{CHECK_FOLDER}/` ของโฟลเดอร์ outputs ซึ่งห้าม commit"),
        "",
        f"- backbone `{bb['name']}` แบบแช่แข็ง ภาพ {bb['image_size']}×{bb['image_size']} โค้ด commit `{bb['commit'][:12]}` น้ำหนัก sha256 `{bb['weights_sha256'][:12]}…`",
        "- feature: " + ", ".join(f"`{k}` {v:,} มิติ" for k, v in bb["features"].items()),
        (f"- ครอปรุ่น {meta['crop_version']} การย่อภาพรุ่น {meta['degrade']['version']} ช่วง IOD ที่เทรน {low:g}–{high:g} px "
         "สำเนา fixed ที่ IOD " + " และ ".join(f"{v:g}" for v in meta["degrade"]["fixed"]["iod_px"]) + " px"),
        f"- Python {rt['python']}, torch {rt['torch']}, mediapipe {rt['mediapipe']}, facemeasure {rt['facemeasure']} อุปกรณ์ที่ดึง feature: {', '.join(devices)}",
        "- ทุกสำเนาผ่าน `degrade.degrade` แล้ว `model_input.analyze_for_model` ซึ่งเป็นทางเข้าเดียวกับ api",
        "",
    ]
    return "\n".join(lines)


def run(check_images: bool = True) -> Path:
    config = quality.load_config()
    low, high = degrade.iod_range(config)
    fixed = degrade.fixed_iod(config)
    meta_path = paths.path("features") / ef.META_NAME
    if not meta_path.is_file():
        raise SystemExit("ยังไม่มี data/features/meta.json ให้รัน extract_features.py face และ undereye ก่อน")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    indexes = {t: ef.read_index(ef.task_folder(t)) for t in ef.TASKS if t in meta["tasks"] and (ef.task_folder(t) / ef.INDEX_NAME).is_file()}
    report_dir = paths.ensure_dir("reports")
    fig_dir = report_dir / FIGURES_FOLDER
    fig_dir.mkdir(parents=True, exist_ok=True)

    frames = [iod_frame(t, df) for t, df in indexes.items()]
    iod_figure(frames, low, high, fig_dir / "features_iod.png")
    quality_figure(indexes, low, high, fig_dir / "features_too_small.png")
    noface, checks = noface_section(indexes, low, high, fixed)
    print("timing and device check ...")
    parts = [
        header(meta),
        counts_section(indexes),
        noface,
        codes_section(indexes, low, fixed),
        "![face_too_small](figures/features_too_small.png)\n",
        iod_section(frames, low, high, fixed, "features_iod.png"),
        device_section(indexes, config, timing_images()),
        size_section(),
    ]
    path = report_dir / REPORT_NAME
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(parts).rstrip() + "\n")
    print(f"report: {path}")
    for k, v in checks.items():
        print(f"{k}: face found on {100 * v:.2f}% of the fixed copies at IOD {fixed[0]:g} px")
    if check_images:
        n = write_check_images(config)
        print(f"{n} check images in {paths.path('outputs') / CHECK_FOLDER} (faces: never commit)")
    return path
