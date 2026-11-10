"""frontal_check helpers on synthetic points and parts in tmp_path."""

import json
import math

import numpy as np
import pandas as pd
import pytest

import frontal_check as fc
from facemeasure import mp_points


def turn(lm, deg, center=(200.0, 220.0)):
    t = math.radians(deg)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    out = lm.copy()
    out[:, :2] = (lm[:, :2] - center) @ r.T + center
    return out


def test_pose_geometry_signs(mp_face):
    base = mp_face()
    # Eye line: rotating the image by -10 degrees (counter-clockwise on screen) raises the image-right eye
    geo = fc.pose_geometry(np.stack([base, turn(base, -10), turn(base, 10)]))
    assert geo["eye_line_deg"].tolist() == pytest.approx([0, 10, -10])
    # Nose moved to the image right: the person turned to their own left
    turned = base.copy()
    turned[mp_points.NOSE_TIP, 0] += 20
    raised = base.copy()
    raised[mp_points.NOSE_TIP, 1] -= 20  # nose tip closer to eye level, as when looking up
    geo = fc.pose_geometry(np.stack([base, turned, raised, turn(turned, 25)]))
    assert geo["nose_offset_iod"].iloc[0] == pytest.approx(0)
    assert geo["nose_offset_iod"].iloc[1] == pytest.approx(0.2)
    assert geo["nose_offset_iod"].iloc[3] == pytest.approx(0.2)  # same in the level frame
    assert geo["nose_drop"].iloc[2] < geo["nose_drop"].iloc[0]


def test_pearson_needs_spread():
    assert fc.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == (4, pytest.approx(1))
    n, r = fc.pearson([1, 1, 1], [1, 2, 3])
    assert n == 3 and math.isnan(r)
    n, r = fc.pearson([1, np.nan, 3, 4, 5], [1, 2, np.nan, 4, 5])
    assert n == 3 and r == pytest.approx(1)


def item(name, source="scut", group="AF"):
    return fc.Item(name, source, group, "", "train", None)


def fake_part(items, with_face=True):
    rows = [fc.result_row(i, None) for i in items]
    for row in rows:
        row.update(width=10, height=10, face_count=1 if with_face else 0, failed="" if with_face else "no_face")
    n = len(items)
    names = np.array(["_neutral", "jawOpen"]) if with_face else np.array([], dtype=str)
    arrays = {
        "image": np.array([i.image for i in items]),
        "source": np.array([i.source for i in items]),
        "landmarks": np.zeros((n, mp_points.N_POINTS, 3)) if with_face else np.full((n, mp_points.N_POINTS, 3), np.nan),
        "blendshapes": np.ones((n, 2)) if with_face else np.empty((n, 0)),
        "blendshape_names": names,
        "matrix": np.zeros((n, 4, 4)),
    }
    return pd.DataFrame(rows, columns=fc.COLUMNS), arrays


def test_parts_round_trip_and_fill_missing_blendshapes(tmp_path):
    first, second = [item("AF1.jpg"), item("AF2.jpg")], [item("AF3.jpg")]
    fc.write_part(tmp_path, 0, *fake_part(first))
    fc.write_part(tmp_path, 1, *fake_part(second, with_face=False))
    assert not list(tmp_path.glob("*.tmp"))
    df, arrays = fc.merge_parts(tmp_path, 2)
    assert df["image"].tolist() == ["AF1.jpg", "AF2.jpg", "AF3.jpg"]
    assert df["failed"].tolist() == ["", "", "no_face"]
    assert arrays["landmarks"].shape == (3, mp_points.N_POINTS, 3)
    assert arrays["blendshapes"].shape == (3, 2) and np.isnan(arrays["blendshapes"][2]).all()
    out = fc.finish_table(df)
    assert out["face_count"].tolist() == [1, 1, 0]


def test_changed_settings_need_restart(tmp_path):
    fc.prepare_parts(tmp_path, {"crop_version": 1}, restart=False)
    (tmp_path / "part_0000.csv").write_text("x\n", encoding="utf-8")
    fc.prepare_parts(tmp_path, {"crop_version": 1}, restart=False)  # same settings: keep going
    assert (tmp_path / "part_0000.csv").is_file()
    with pytest.raises(SystemExit, match="crop_version"):
        fc.prepare_parts(tmp_path, {"crop_version": 2}, restart=False)
    fc.prepare_parts(tmp_path, {"crop_version": 2}, restart=True)
    assert not (tmp_path / "part_0000.csv").exists()
    with open(tmp_path / fc.SETTINGS_NAME, encoding="utf-8") as f:
        assert json.load(f) == {"crop_version": 2}


def test_check_selection_is_stable_and_per_group():
    items = [item(f"{g}{n}.jpg", group=g) for g in ("AF", "AM", "CF", "CM") for n in range(1, 31)]
    items += [item(f"{n:05d}.png", "ffhq", band) for band in ("20-29", "50+") for n in range(100 * len(band), 100 * len(band) + 20)]
    picked = fc.check_selection(items)
    assert picked == fc.check_selection(list(reversed(items)))
    counts = pd.Series([(i.source, i.group) for i in picked]).value_counts()
    assert (counts == fc.CHECK_PER_GROUP).all() and len(counts) == 6
