"""Grades of the raters as under-eye labels: reading, merging, soft labels and agreement."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from facemeasure.training import undereye_labels as ul

HEADER = ",".join(ul.CSV_COLUMNS)


def write_csv(path, rows):
    """A grader CSV as the tool writes it: UTF-8 with a BOM, CRLF line ends."""
    lines = [HEADER]
    for r in rows:
        r = {"set_id": "ABC123", "image_side": ul.IMAGE_SIDE.get(r["person_side"], ""), "unassessable": 0, "eye_bags": 0,
             "dark_circles": 0, "note": "", "seconds": 3.0, "seq": 1, "graded_at": "2026-11-01T10:00:00.000Z", **r}
        lines.append(",".join(str(r[c]) for c in ul.CSV_COLUMNS))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("﻿" + "\r\n".join(lines) + "\r\n")
    return path


def full_set(rater, images, grade=lambda image, side: 0, **extra):
    return [{"rater": rater, "image": i, "person_side": s, "grade": grade(i, s), **extra} for i in images for s in ul.SIDES]


def manifest(images, pilot=(), splits=None):
    rows = []
    for k, i in enumerate(images):
        rows.append({"image": i, "source": "ffhq" if k % 2 else "scut", "sex": "female" if k % 3 else "male",
                     "age_group": "30-39" if k % 2 else "", "role": "graded", "pilot": "1" if i in pilot else "0",
                     "split": (splits or {}).get(i, "train")})
    rows.append({"image": "practice1.png", "source": "ffhq", "sex": "male", "age_group": "20-29", "role": "practice",
                 "pilot": "0", "split": ""})
    return pd.DataFrame(rows)


IMAGES = [f"{k:05d}.png" for k in range(6)]


def test_reads_bom_and_crlf_and_merges_names_that_differ_in_spacing_or_case(tmp_path):
    write_csv(tmp_path / "a.csv", full_set("  Somchai   Jaidee ", IMAGES, grade=lambda i, s: 1))
    newer = [{"rater": "somchai jaidee", "image": IMAGES[0], "person_side": "left", "grade": 3,
              "graded_at": "2026-11-02T10:00:00.000Z"},
             {"rater": "SOMCHAI JAIDEE", "image": IMAGES[1], "person_side": "right", "grade": 2,
              "graded_at": "2026-10-01T10:00:00.000Z"}]  # older than the first file: loses
    write_csv(tmp_path / "b.csv", newer)
    df, skipped = ul.read_grader_csv(tmp_path / "a.csv")
    assert skipped == 0 and len(df) == 12 and df["rater"].iloc[0] == "Somchai Jaidee"
    assert df["graded_at"].iloc[0] == "2026-11-01T10:00:00.000Z"  # no \r left at the line end
    g = ul.collect([tmp_path / "a.csv", tmp_path / "b.csv"], IMAGES)
    assert g.raters == ["somchai jaidee"] and g.codes == {"somchai jaidee": "R1"}
    assert any("b.csv" in n and "does not cover" in n for n in g.notes)  # b.csv alone is not complete
    sides = ul.sides_table(g, manifest(IMAGES))
    # b.csv is skipped, so neither of its rows counts
    assert sides.loc[(sides["image"] == IMAGES[0]) & (sides["person_side"] == "left"), "grade_R1"].item() == 1


def test_latest_row_wins_within_the_usable_files(tmp_path):
    rows = full_set("Nok", IMAGES, grade=lambda i, s: 1)
    rows.append({"rater": "nok ", "image": IMAGES[2], "person_side": "right", "grade": 3, "graded_at": "2026-11-03T00:00:00.000Z"})
    rows.append({"rater": "NOK", "image": IMAGES[3], "person_side": "left", "grade": 0, "graded_at": "2026-10-03T00:00:00.000Z"})
    write_csv(tmp_path / "nok.csv", rows)
    g = ul.collect([tmp_path / "nok.csv"], IMAGES)
    sides = ul.sides_table(g, manifest(IMAGES)).set_index(["image", "person_side"])
    assert sides.loc[(IMAGES[2], "right"), "grade_R1"] == 3
    assert sides.loc[(IMAGES[3], "left"), "grade_R1"] == 1


def test_single_rater_labels_exclusion_and_flags(tmp_path):
    def grade(i, s):
        return {IMAGES[0]: 0, IMAGES[1]: 1, IMAGES[2]: 2, IMAGES[3]: 3}.get(i, 1)

    rows = full_set("Nok", IMAGES, grade=grade)
    for r in rows:
        if r["image"] == IMAGES[4] and r["person_side"] == "left":
            r.update(grade="", unassessable=1)
        if r["image"] == IMAGES[5] and r["person_side"] == "right":
            r.update(eye_bags=1)
        if r["image"] == IMAGES[5] and r["person_side"] == "left":
            r.update(dark_circles=1)
    write_csv(tmp_path / "nok.csv", rows)
    g = ul.collect([tmp_path / "nok.csv"], IMAGES)
    s = ul.sides_table(g, manifest(IMAGES)).set_index(["image", "person_side"])
    for image, (t1, t2) in zip(IMAGES[:4], ((0, 0), (1, 0), (1, 1), (1, 1)), strict=True):
        assert (s.loc[(image, "right"), "p_ge1"], s.loc[(image, "right"), "p_ge2"]) == (t1, t2)
        assert (s.loc[(image, "right"), "target_ge1"], s.loc[(image, "right"), "target_ge2"]) == (t1, t2)
    na = s.loc[(IMAGES[4], "left")]
    assert na["excluded"] == ul.UNASSESSABLE and na["n_unassessable"] == 1 and math.isnan(na["target_ge1"])
    assert pd.isna(na["grade_R1"])
    assert s.loc[(IMAGES[5], "right"), "flagged"] == 1 and s.loc[(IMAGES[5], "left"), "flagged"] == 1
    assert s["flagged"].sum() == 2 and (s["excluded"] != "").sum() == 1
    assert ul.agreement(g, IMAGES) == {}  # one rater: no agreement to measure
    assert s.loc[(IMAGES[0], "right"), "image_side"] == "left"


def test_soft_labels_half_unassessable_and_consensus(tmp_path):
    a = full_set("A", IMAGES, grade=lambda i, s: 1)
    b = full_set("B", IMAGES, grade=lambda i, s: 2 if i == IMAGES[0] else 1)
    for r in b:
        if r["image"] == IMAGES[1] and r["person_side"] == "right":
            r.update(grade="", unassessable=1)
        if r["image"] == IMAGES[2] and r["person_side"] == "right":
            r.update(eye_bags=1)
    write_csv(tmp_path / "a.csv", a)
    write_csv(tmp_path / "b.csv", b)
    write_csv(tmp_path / "c.csv", [{"rater": "Consensus", "image": IMAGES[0], "person_side": "left", "grade": 2}])
    g = ul.collect([tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"], IMAGES)
    assert g.raters == ["a", "b"] and g.has_consensus
    s = ul.sides_table(g, manifest(IMAGES)).set_index(["image", "person_side"])
    one_two = s.loc[(IMAGES[0], "right")]
    assert (one_two["p_ge1"], one_two["p_ge2"]) == (1.0, 0.5)
    assert (one_two["target_ge1"], one_two["target_ge2"]) == (1.0, 0.5) and pd.isna(one_two["consensus_grade"])
    agreed = s.loc[(IMAGES[0], "left")]
    assert agreed["p_ge2"] == 0.5 and agreed["consensus_grade"] == 2 and agreed["target_ge2"] == 1.0
    assert s.loc[(IMAGES[1], "right"), "excluded"] == ul.UNASSESSABLE  # one of two could not assess it
    assert s.loc[(IMAGES[2], "right"), "flagged"] == 1  # one of two marked it
    assert s.loc[(IMAGES[3], "right"), "n_raters"] == 2 and s.loc[(IMAGES[3], "right"), "excluded"] == ""
    # the consensus is not a rater and does not enter the agreement
    ag = ul.agreement(g, IMAGES)
    assert [(p["a"], p["b"]) for p in ag["pairs"]] == [("R1", "R2")]
    assert ag["pairs"][0]["n"] == 11 and ag["icc_sides"] == 11


def test_files_of_other_sets_are_skipped_and_nothing_usable_stops(tmp_path):
    write_csv(tmp_path / "practice.csv", full_set("A", ["practice1.png"]))
    write_csv(tmp_path / "pilot.csv", full_set("A", IMAGES[:3]))
    with pytest.raises(ul.LabelError, match="no grader file covers every graded image"):
        ul.collect([tmp_path / "practice.csv", tmp_path / "pilot.csv"], IMAGES)
    write_csv(tmp_path / "full.csv", full_set("A", [*IMAGES, "practice1.png"]))
    g = ul.collect(sorted(tmp_path.glob("*.csv")), IMAGES)
    used = {c.file: c.used for c in g.files}
    assert used == {"full.csv": True, "pilot.csv": False, "practice.csv": False}
    assert any("full.csv: 2 rows of images outside" in n for n in g.notes)
    assert len(g.rows) == 2 * len(IMAGES)


def test_rows_the_tool_rejects_are_skipped(tmp_path):
    rows = full_set("A", IMAGES)
    rows.append({"rater": "A", "image": IMAGES[0], "person_side": "up", "grade": 1})
    rows.append({"rater": "A", "image": IMAGES[1], "person_side": "left", "grade": 5})
    rows.append({"rater": "A", "image": IMAGES[1], "person_side": "left", "grade": ""})
    write_csv(tmp_path / "a.csv", rows)
    df, skipped = ul.read_grader_csv(tmp_path / "a.csv")
    assert skipped == 3 and len(df) == 12


def test_linear_kappa_matches_sklearn_for_both_scales():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, 200)
    b = np.clip(a + rng.integers(-1, 2, 200), 0, 3)
    want = cohen_kappa_score(a, b, weights="linear", labels=[0, 1, 2, 3])
    assert ul.linear_kappa(a.tolist(), b.tolist(), ul.GRADES4) == pytest.approx(want, abs=1e-12)
    a3, b3 = np.minimum(a, 2), np.minimum(b, 2)
    want3 = cohen_kappa_score(a3, b3, weights="linear", labels=[0, 1, 2])
    assert ul.linear_kappa(a3.tolist(), b3.tolist(), ul.GRADES3) == pytest.approx(want3, abs=1e-12)
    # categories that never occur still count, as labels= does in sklearn
    few = [0, 1, 1, 0, 1]
    other = [0, 1, 0, 0, 1]
    assert ul.linear_kappa(few, other, ul.GRADES4) == pytest.approx(
        cohen_kappa_score(few, other, weights="linear", labels=[0, 1, 2, 3]), abs=1e-12)
    assert math.isnan(ul.linear_kappa([1, 1], [1, 1], ul.GRADES4))


def test_icc21_of_shrout_and_fleiss_table_2():
    y = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
    assert ul.icc21(y) == pytest.approx(0.29, abs=0.005)
    assert math.isnan(ul.icc21([[1, 2]]))


def test_pair_stats_counts_only_sides_both_graded():
    a = {("x", "right"): {"grade": 1, "na": False}, ("x", "left"): {"grade": 3, "na": False}, ("y", "right"): {"grade": None, "na": True}}
    b = {("x", "right"): {"grade": 2, "na": False}, ("x", "left"): {"grade": 1, "na": False}, ("y", "right"): {"grade": 0, "na": False}}
    s = ul.pair_stats(a, b)
    assert s["n"] == 2 and s["exact"] == 0.0 and s["within1"] == 0.5


def test_distribution_and_low_share_at_two(tmp_path):
    write_csv(tmp_path / "a.csv", full_set("A", IMAGES, grade=lambda i, s: 2 if i == IMAGES[0] and s == "right" else 0))
    g = ul.collect([tmp_path / "a.csv"], IMAGES)
    sides = ul.sides_table(g, manifest(IMAGES))
    d = ul.distribution(sides, {"all": pd.Series(True, index=sides.index), "scut": sides["source"] == "scut"})
    assert d[0]["sides"] == 12 and d[0]["raters"]["R1"] == {"0": 11, "1": 0, "2": 1, "3": 0, "na": 0}
    assert d[0]["ge2_share"] == pytest.approx(1 / 12) and d[0]["ge2_share"] < ul.LOW_GE2_SHARE
    assert d[1]["sides"] == 6
    assert ul.median_seconds(g) == {"R1": 3.0}


def test_labels_file_round_trip(tmp_path):
    write_csv(tmp_path / "a.csv", full_set("A", IMAGES, grade=lambda i, s: 1))
    sides = ul.sides_table(ul.collect([tmp_path / "a.csv"], IMAGES), manifest(IMAGES, pilot={IMAGES[0]}))
    ul.write_labels(sides, tmp_path / "labels.csv")
    back = ul.read_labels(tmp_path / "labels.csv")
    assert back["pilot"].sum() == 2 and back["grade_R1"].tolist() == [1] * 12
    assert back["age_group"].iloc[0] == "" and back["excluded"].tolist() == [""] * 12
    assert b"\r" not in (tmp_path / "labels.csv").read_bytes()
