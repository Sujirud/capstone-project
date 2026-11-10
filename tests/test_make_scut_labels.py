"""make_scut_labels on a synthetic SCUT-FBP5500 folder in tmp_path."""

import random
import struct

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import make_scut_labels as ml


def write_pts(path, points):
    path.write_bytes(struct.pack("<i172f", 86, *np.asarray(points).ravel()))


def names(group, numbers):
    return [f"{group}{n}.jpg" for n in numbers]


def test_read_split_file(tmp_path):
    path = tmp_path / "train.txt"
    path.write_text("AF1.jpg 3.5\nCM290.jpg 2.25\n\n", encoding="utf-8")
    assert ml.read_split_file(path) == {"AF1.jpg": 3.5, "CM290.jpg": 2.25}
    for text, message in (("AF1.jpg 3.5\nAF1.jpg 2\n", "twice"), ("AF1.jpg\n", "expected"), ("XX1.jpg 2\n", "name")):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            ml.read_split_file(path)


def test_val_counts_keep_group_shares_and_total():
    sizes = {"AF": 1200, "AM": 1200, "CF": 450, "CM": 450}
    counts = ml.val_counts(sizes)
    assert sum(counts.values()) == 495
    for g, n in sizes.items():
        assert abs(counts[g] - 0.15 * n) < 1
    assert ml.val_counts({"AF": 7, "AM": 7, "CF": 6, "CM": 0}) == {"AF": 1, "AM": 1, "CF": 1, "CM": 0}


def test_assign_splits_is_stratified_and_stable():
    train = names("AF", range(1, 121)) + names("AM", range(1, 121)) + names("CF", range(1, 41)) + names("CM", range(1, 41))
    test = names("AF", range(200, 260)) + names("CM", range(200, 230))
    splits = ml.assign_splits(train, test)
    shuffled = train[:]
    random.Random(3).shuffle(shuffled)
    assert ml.assign_splits(shuffled, test[::-1]) == splits
    assert all(splits[n] == ("test", "test") for n in test)
    val = [n for n in train if splits[n][0] == "val"]
    assert all(splits[n][1] == "train" for n in train)
    assert sum(n.startswith("AF") for n in val) == 18
    assert sum(n.startswith("CF") for n in val) == 6
    assert len(val) == 48
    with pytest.raises(ValueError, match="both"):
        ml.assign_splits(train, ["AF1.jpg"])


def test_outliers_flag_values_beyond_p1_and_p99():
    values = np.linspace(0.7, 0.9, 101)
    df = pd.DataFrame({"image": [f"AF{i + 1}.jpg" for i in range(101)], "group": "AF",
                       "jaw_ratio": values, "nose_ratio": 1.0})
    odd = ml.outliers(df)
    assert sorted(odd["image"]) == ["AF1.jpg", "AF101.jpg"]
    assert set(odd["side"]) == {"low", "high"}
    assert set(odd["measure"]) == {"jaw_ratio"}


def make_dataset(root, scut_face):
    """10 synthetic faces per group, plus one short file, one with swapped eyes and one without points."""
    rng = np.random.default_rng(0)
    landmarks = root / "facial landmark"
    images = root / "Images"
    split_dir = root.joinpath(*ml.SPLIT_FOLDER)
    for folder in (landmarks, images, split_dir):
        folder.mkdir(parents=True)
    train, test = [], []
    for group in ("AF", "AM", "CF", "CM"):
        for n in range(1, 11):
            image = f"{group}{n}.jpg"
            points = scut_face(a=110 * rng.uniform(0.9, 1.1), mouth=rng.uniform(0.4, 0.6))
            points[[64, 68], 0] += rng.normal(0, 2, 2)
            write_pts(landmarks / f"{group}{n}.pts", points)
            Image.new("RGB", (350, 350), (120, 110, 100)).save(images / image)
            (test if n > 6 else train).append(f"{image} {rng.uniform(1, 5):.6f}")
    (landmarks / "CM11.pts").write_bytes(struct.pack("<i", 86))
    swapped = scut_face()
    swapped[[58, 59]] = swapped[[59, 58]]
    write_pts(landmarks / "AF11.pts", swapped)
    test += ["CM11.jpg 3.0", "AF11.jpg 3.0", "AM12.jpg 3.0"]
    (split_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")


def run_once(tmp_path, scut_face):
    root = tmp_path / "scut"
    if not root.exists():
        make_dataset(root, scut_face)
    out = tmp_path / "out"
    df = ml.run(root, out / "labels", out / "reports", out / "check")
    files = [
        out / "labels" / ml.LABELS_NAME, out / "labels" / ml.SKIPPED_NAME, out / "reports" / ml.REPORT_NAME,
        out / "reports" / ml.FIGURES_FOLDER / ml.JAW_FIGURE, out / "reports" / ml.FIGURES_FOLDER / ml.NOSE_FIGURE,
    ]
    return df, out, {p: p.read_bytes() for p in files}


def test_run_end_to_end(tmp_path, scut_face):
    _, out, files = run_once(tmp_path, scut_face)
    labels = pd.read_csv(out / "labels" / ml.LABELS_NAME)
    assert list(labels.columns) == ml.COLUMNS
    assert len(labels) == 40
    assert labels["image"].iloc[0] == "AF1.jpg"
    assert set(labels["split"]) == {"train", "val", "test"}
    assert (labels.loc[labels["scut_split"] == "test", "split"] == "test").all()
    assert labels.loc[labels["image"] == "CM3.jpg", ["group", "sex", "ethnicity"]].iloc[0].tolist() == ["CM", "male", "caucasian"]
    assert labels["jaw_ratio"].between(0.7, 1.0).all()

    skipped = pd.read_csv(out / "labels" / ml.SKIPPED_NAME).set_index("image")["reason"]
    assert sorted(skipped.index) == ["AF11.jpg", "AM12.jpg", "CM11.jpg"]
    assert "eye centers" in skipped["AF11.jpg"]
    assert skipped["AM12.jpg"] == "no landmark file"
    assert "4 bytes" in skipped["CM11.jpg"]

    report = files[out / "reports" / ml.REPORT_NAME].decode("utf-8")
    for heading in ("## จำนวนภาพ", "## การกระจายของอัตราส่วน", "## กรามของผู้หญิงเทียบเส้นแบ่ง",
                    "## ฐานจมูกเทียบเส้นแบ่งตามเพศ", "## ตำแหน่งของ Zy", "## Histogram", "## ภาพที่ค่าผิดปกติ"):
        assert heading in report
    assert "AF11.jpg" in report
    assert "\r" not in report

    sheets = sorted(p.name for p in (out / "check").iterdir())
    assert sheets[0] == "outliers_01.png"
    assert "random_01.png" in sheets
    with Image.open(out / "check" / "random_01.png") as im:
        assert im.width == ml.SHEET_COLUMNS * 350


def test_rerun_gives_the_same_files_and_clears_old_sheets(tmp_path, scut_face):
    _, out, first = run_once(tmp_path, scut_face)
    stale = out / "check" / "outliers_09.png"
    Image.new("RGB", (4, 4)).save(stale)
    _, _, second = run_once(tmp_path, scut_face)
    assert first == second
    assert not stale.exists()
