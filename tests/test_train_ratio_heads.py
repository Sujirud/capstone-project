"""model/scripts/train_ratio_heads.py on synthetic data: evaluation, flags, select and the bundle."""

import json
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import train_ratio_heads as trh
from facemeasure import bundle, codes, paths
from facemeasure.bundle import BundleWarning
from facemeasure.training import ratio_data, ridge

DIMS = {"cls": 768, "cls_mean": 1536, "grid4": 12288}
META = {
    "backbone": {"name": "dinov2_vitb14", "image_size": 224, "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                 "features": DIMS, "commit": "abc", "weights_sha256": "0" * 64},
    "train_face_iod_px": {"min": 75.0, "max": 110.0},
    "degrade": {"version": 2},
}


def eval_rows(n_images=60, seed=0):
    """Test rows of the evaluation: two copies per image, half women, all four groups."""
    rng = np.random.default_rng(seed)
    groups = np.array(["AF", "AM", "CF", "CM"])[np.arange(n_images) % 4]
    rows = []
    for i, g in enumerate(groups):
        jaw, nose = 0.84 + rng.normal(0, 0.05), 1.10 + rng.normal(0, 0.08)
        for c in (1, 2):
            rows.append({"image": f"{g}{i}.jpg", "group": g, "sex": "female" if g.endswith("F") else "male", "copy": c,
                         "jaw_ratio": jaw, "nose_ratio": nose})
    return pd.DataFrame(rows)


def test_evaluation_of_a_perfect_and_a_shrunk_head():
    rows = eval_rows()
    y = rows["jaw_ratio"].to_numpy()
    r = trh.evaluate_target(codes.JAW, rows, y, y)
    assert r["table"]["all"]["model"]["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert r["grading"]["kappa"]["model"] == pytest.approx(1.0)
    assert r["grading"]["rows"] == (rows["sex"] == "female").sum()  # the jaw grades only women
    shrunk = y.mean() + 0.5 * (y - y.mean())
    r = trh.evaluate_target(codes.JAW, rows, shrunk, y)
    assert r["slope"]["all"]["model"] == pytest.approx(0.5) and r["slope"]["all"]["baseline"] == pytest.approx(1.0)
    f = rows["sex"].to_numpy() == "female"
    assert r["band"]["rows"] == "female"
    assert r["band"]["rmse"] == pytest.approx(np.sqrt(np.mean((shrunk - y)[f] ** 2)))
    assert r["band"]["band"] == pytest.approx(0.5244 * r["band"]["rmse"])
    assert set(r["bootstrap"]) == {"all", "female"} and r["bootstrap"]["all"]["diff"] > 0
    assert r["copies"]["images"] == len(rows) // 2 and r["copies"]["model"]["mean_abs"] == pytest.approx(0.0)
    flags = trh.flags_of({codes.JAW: r, codes.NOSE_BASE: trh.evaluate_target(codes.NOSE_BASE, rows, rows["nose_ratio"], rows["nose_ratio"])})
    # pulling every value toward the overall mean also biases groups whose mean is elsewhere
    assert flags[0] == "jaw: slope of the predictions on the labels is 0.500, below 0.8"
    assert all(f.startswith("jaw:") for f in flags)


def test_alar_base_band_uses_every_image_and_group_bias_flags():
    rows = eval_rows(seed=1)
    y = rows["nose_ratio"].to_numpy()
    r = trh.evaluate_target(codes.NOSE_BASE, rows, y + 0.03, y)
    assert r["band"]["rows"] == "all" and r["band"]["rmse"] == pytest.approx(0.03)
    assert set(r["bootstrap"]) == {"all"} and r["grading"]["rows"] == len(rows)
    jaw = rows["jaw_ratio"].to_numpy()
    biased = jaw + np.where(rows["group"] == "CF", 0.015, 0.0)
    flags = trh.flags_of({codes.JAW: trh.evaluate_target(codes.JAW, rows, biased, jaw), codes.NOSE_BASE: r})
    assert flags == ["jaw: bias of group CF is +1.50 percentage points, beyond 1"]


def joined_frame(n_train=300, n_val=80, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            g = "AF" if i % 2 else "AM"
            rows.append({"image": f"{split}{i}", "group": g, "sex": "female" if g == "AF" else "male", "split": split, "copy": 1,
                         "reasons_jaw": "", "reasons_nose_base": ""})
    df = pd.DataFrame(rows)
    x = rng.normal(size=(len(df), 20))
    df["jaw_ratio"] = 0.84 + x[:, :5] @ rng.normal(0, 0.01, 5) + rng.normal(0, 0.005, len(df))
    df["nose_ratio"] = 1.1 + x[:, 5:10] @ rng.normal(0, 0.02, 5) + rng.normal(0, 0.01, len(df))
    return df, x


def test_select_scores_both_targets_on_val_rows_only():
    joined, x = joined_frame()
    joined.loc[joined.index[-5:], "reasons_jaw"] = "smiling"  # left out of the jaw val rows only
    parts = ratio_data.subsets(joined, ("train", "val"))
    info, grids = trh.select_feature("cls", x, parts)
    assert info["dim"] == 20 and info["train_rows"] == 300 and info["solver"].startswith("primal")
    for t in ratio_data.TARGETS:
        g = grids[t]
        assert g.feature == "cls" and len(g.alphas) >= len(ridge.ALPHAS)
        val = parts[(t, "val")]
        expect = ridge.rmse(ridge.fit(x[parts[(t, "train")].positions], parts[(t, "train")].y, g.alphas[0]).predict(x[val.positions]), val.y)
        assert g.rmse[0] == pytest.approx(expect, rel=1e-9)
    assert len(parts[("jaw", "val")].rows) == 75 and len(parts[("nose_base", "val")].rows) == 80


def fake_evaluate(seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(120, DIMS["cls"])).astype(np.float32)
    arrays, ev, models = {}, {"heads": {}, "baseline": {}}, {}
    for k, t in enumerate(ratio_data.TARGETS):
        y = (0.84, 1.1)[k] + x[:, :10] @ rng.normal(0, 0.01, 10)
        model = ridge.fit(x, y, 10.0)
        w, b = model.folded()
        arrays[f"{t}_weights"], arrays[f"{t}_bias"] = w, np.array(b)
        ev["heads"][t] = {"feature": "cls", "alpha": 10.0, "rmse": 0.04, "rmse_rows": "female" if t == codes.JAW else "all",
                          "rmse_all": 0.041, "band": 0.5244 * 0.04, "train_rows": 120, "train_images": 30}
        ev["baseline"][t] = {"slope": 0.9, "intercept": 0.08}
        models[t] = model
    return ev, arrays, models, x


def test_bundle_written_and_read_back_predicts_as_in_training(tmp_path):
    ev, arrays, models, x = fake_evaluate()
    manifest, out = trh.bundle_parts(ev, arrays, META)
    bundle.write_bundle(manifest, out, tmp_path / "bundle")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BundleWarning)
        b = bundle.load_bundle(tmp_path / "bundle")
    assert b.version == "0.1.0" and b.stub is True and b.stub_heads == ("undereye",)
    assert b.train_face_iod_px == {"min": 75.0, "max": 110.0} and b.manifest["degrade"] == {"version": 2}
    assert b.manifest["backbone"]["features"] == DIMS
    for t in ratio_data.TARGETS:
        head = b.heads[t]
        assert not head.stub and head.feature == "cls" and head.rmse == 0.04 and head.band == pytest.approx(0.020976)
        assert b.manifest["heads"][t]["alpha"] == 10.0
        np.testing.assert_allclose(head.predict(x[:7]), models[t].predict(x[:7]), atol=1e-9)
        assert head.predict(x[0]) == pytest.approx(float(models[t].predict(x[:1])[0]), abs=1e-9)
        assert b.baseline[t].apply(0.8) == pytest.approx(0.9 * 0.8 + 0.08)
    with open(tmp_path / "bundle" / "manifest.json", encoding="utf-8") as f:
        assert json.load(f)["heads"]["jaw"]["stub"] is False


def test_head_values_follow_the_bundle_formulas():
    ev, arrays, models, x = fake_evaluate()
    heads, base = trh.evaluate_heads(ev, arrays)
    result = SimpleNamespace(ratios=SimpleNamespace(jaw_ratio=0.82, nose_ratio=1.05))
    v = trh.head_values(heads, base, result, {"cls": x[3]})
    assert v["pred_jaw"] == pytest.approx(float(models["jaw"].predict(x[3:4])[0]), abs=1e-9)
    assert v["landmark_jaw"] == pytest.approx(0.9 * 0.82 + 0.08) and v["landmark_nose_base"] == pytest.approx(0.9 * 1.05 + 0.08)
    assert trh.head_values(heads, base, SimpleNamespace(ratios=None), {"cls": x[3]})["landmark_jaw"] is None


def test_fidelity_images_are_lowest_iod_copies_in_both_evaluations_and_fixed():
    rows = []
    for i in range(30):
        for c in (1, 2):
            rows.append({"image": f"AF{i}.jpg", "copy": c, "used_jaw": i % 3 != 0, "used_nose_base": i % 5 != 0})
    pred = pd.DataFrame(rows)
    one, two = trh.pick_fidelity(pred, 5), trh.pick_fidelity(pred.sample(frac=1, random_state=0), 5)
    assert one["image"].tolist() == two["image"].tolist() and len(one) == 5
    assert (one["copy"] == 1).all() and one["used_jaw"].all() and one["used_nose_base"].all()


def test_rescale_makes_small_images_bigger():
    image = np.zeros((40, 30, 3), dtype=np.uint8)
    assert trh.rescale(image, 2.5).shape == (100, 75, 3) and trh.rescale(image, 0.5).shape == (20, 15, 3)


def test_log_appends_what_each_command_read(tmp_path):
    trh.log("select", "reads train and val", tmp_path)
    trh.log("evaluate", "reads train and test", tmp_path)
    lines = (tmp_path / trh.LOG_NAME).read_text(encoding="utf-8").splitlines()
    assert [ln.split(" ", 1)[1] for ln in lines] == ["select: reads train and val", "evaluate: reads train and test"]


def fake_root(tmp_path, monkeypatch):
    """A project root in tmp_path with the real config.toml, so paths point inside it."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.toml").write_bytes((paths.project_root() / "config.toml").read_bytes())
    monkeypatch.setenv(paths.ROOT_ENV, str(root))
    return root


def test_report_has_every_section_from_synthetic_results(tmp_path, monkeypatch):
    fake_root(tmp_path, monkeypatch)
    folder = trh.out_dir()
    grids = {f: {t: ridge.Grid(f, (1.0, 10.0, 100.0), (0.03, 0.025 + k * 0.001, 0.027)) for t in ratio_data.TARGETS}
             for k, f in enumerate(("cls", "cls_mean"))}
    sel = {"fingerprint": {"alphas": [1.0, 10.0, 100.0]}, "counts": [{"target": "jaw", "split": "train", "group": "AF", "rows": 8, "images": 2}],
           "features": {f: {"dim": DIMS[f], "train_rows": 8, "solver": "primal (X^T X)", "eigh_seconds": 0.1, "matrix_mb": 1.0,
                            "peak_rss_mb": None, "targets": {t: trh.grid_dict(g) for t, g in grids[f].items()}} for f in grids}}
    sel["choice"] = trh.choose_all(sel, DIMS)
    rows = eval_rows()
    results, heads, table = {}, {}, rows.assign(target_iod=77.0, iod_px=80.0, reasons_jaw="", reasons_nose_base="",
                                                 jaw_ratio_mp=rows["jaw_ratio"], nose_ratio_mp=rows["nose_ratio"])
    for t in ratio_data.TARGETS:
        y = rows[ratio_data.LABEL_COLUMN[t]].to_numpy()
        results[t] = trh.evaluate_target(t, rows, y + 0.01, y + 0.02)
        heads[t] = {"feature": "cls", "alpha": 10.0, "train_rows": 8, "train_images": 2}
        table[f"pred_{t}"], table[f"landmark_{t}"], table[f"used_{t}"] = y + 0.01, y + 0.02, True
    ev = {"copy_iod": {"1": 77.0, "2": 110.0}, "counts": [], "heads": heads, "results": results, "flags": trh.flags_of(results),
          "baseline": {t: {"slope": 1.0, "intercept": 0.0} for t in ratio_data.TARGETS}}
    trh.write_json(folder / trh.SELECT_NAME, sel)
    trh.write_json(folder / trh.EVALUATE_NAME, ev)
    table.to_csv(folder / trh.PREDICTIONS_NAME, index=False, encoding="utf-8")
    wu = [{"image": k, "paper": v, "faces_scaled": 1, "model": v + 0.01, "baseline": v, "iod_source": 50.0, "iod_px": 92.5,
           "reasons_jaw": ""} for k, v in trh.WU_RATIOS.items()]
    trh.write_json(folder / trh.WU_NAME, {"target_iod": 92.5, "rows": wu, "summary": trh.wu_summary(wu)})
    trh.write_json(folder / trh.FIDELITY_NAME, {"bundle_version": "0.1.0", "images": 20, "tolerance": 0.0005, "max_diff": 1e-6,
                                                "max_by_value": {"pred_jaw": 1e-6}, "passed": True})
    trh.log("evaluate", "reads train and test", folder)
    text = trh.write_report().read_text(encoding="utf-8")
    for title in ("จำนวนข้อมูล", "การเลือกชนิด feature และ alpha", "ผลบนชุด test", "เทียบกับ baseline", "ช่วงก้ำกึ่ง",
                  "ภาพจาก \\[3\\]", "ความตรงกับตอนเตรียมข้อมูล", "ข้อจำกัด", "ชุดข้อมูลที่แต่ละคำสั่งอ่าน"):
        assert f"## {title}\n" in text
    assert "ยังไม่ได้รัน" not in text and "**0.02500**" in text
    assert "Spearman ρ 1.000" in text  # the model keeps the order of [3]
    for t in ratio_data.TARGETS:
        assert (paths.path("reports") / "figures" / f"ratio_heads_{t}_scatter.png").is_file()


def test_report_without_results_says_what_to_run(tmp_path, monkeypatch):
    fake_root(tmp_path, monkeypatch)
    text = trh.write_report().read_text(encoding="utf-8")
    assert "ยังไม่ได้รัน `select`" in text and "ยังไม่ได้รัน `bundle`" in text
