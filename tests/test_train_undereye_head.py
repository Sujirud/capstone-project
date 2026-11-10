"""model/scripts/train_undereye_head.py on synthetic data: labels, select, evaluation, cross flags, bundle and report."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from test_train_ratio_heads import DIMS, META, fake_evaluate, fake_root
from test_undereye_labels import full_set, write_csv

import grading_set
import train_ratio_heads as trh
import train_undereye_head as tuh
from facemeasure import bundle, criteria, paths
from facemeasure.bundle import BundleWarning
from facemeasure.training import coral, ridge
from facemeasure.training import undereye_data as ud
from facemeasure.training import undereye_labels as ul

IMAGES = [f"{k:05d}.png" for k in range(12)]


def write_manifest(path, images=IMAGES):
    rows = []
    for k, i in enumerate(images):
        rows.append({c: "" for c in grading_set.MANIFEST_COLUMNS} | {
            "image": i, "source": "ffhq" if k % 3 else "scut", "source_id": str(k), "sex": "female" if k % 2 else "male",
            "age_group": ("20-29", "50-69")[k % 2] if k % 3 else "", "role": "graded", "pilot": "1" if k < 4 else "0",
            "split": ("train", "val", "test")[k % 3]})
    rows.append({c: "" for c in grading_set.MANIFEST_COLUMNS} | {"image": "p.png", "source": "ffhq", "role": "practice", "pilot": "0"})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=grading_set.MANIFEST_COLUMNS).to_csv(path, index=False, encoding="utf-8")


def grading_root(tmp_path, monkeypatch, raters=("Real Name",)):
    fake_root(tmp_path, monkeypatch)
    write_manifest(paths.path("grading") / tuh.MANIFEST_NAME)
    results = paths.path("grading") / tuh.RESULTS_FOLDER
    results.mkdir(parents=True)
    for k, name in enumerate(raters):
        write_csv(results / f"undereye_{k}.csv", full_set(name, IMAGES, grade=lambda i, s, k=k: (int(i[:5]) + k) % 4))
    write_csv(results / "practice.csv", full_set(raters[0], ["p.png"]))


def test_labels_with_one_rater_skips_agreement_and_keeps_names_out_of_the_report(tmp_path, monkeypatch):
    grading_root(tmp_path, monkeypatch)
    assert tuh.main(["labels"]) == 0
    labels = ul.read_labels(tuh.labels_path())
    assert len(labels) == 2 * len(IMAGES) and list(ul.rater_columns(labels)) == ["grade_R1"]
    assert set(labels["target_ge1"]) <= {0.0, 1.0}
    lab = trh.read_json(tuh.out_dir() / tuh.LABELS_JSON)
    assert lab["agreement"] is None and lab["raters"] == ["R1"]
    assert [f["used"] for f in lab["files"]] == [False, True]  # practice.csv sorts first and is skipped
    assert trh.read_json(tuh.out_dir() / tuh.RATERS_JSON)["raters"] == {"R1": "Real Name"}
    text = (paths.path("reports") / tuh.REPORT_NAME).read_text(encoding="utf-8")
    assert "วัดความสอดคล้องระหว่างผู้ให้เกรดไม่ได้ เพราะมีผู้ให้เกรดคนเดียว" in text
    assert "Real Name" not in text and "undereye_0" not in text and "practice.csv" not in text
    assert "ยังไม่ได้รัน `select`" in text
    assert (paths.path("reports") / "figures" / "undereye_grades.png").is_file()
    log = (tuh.out_dir() / tuh.LOG_NAME).read_text(encoding="utf-8")
    assert "labels: reads grading/results" in log


def test_labels_with_two_raters_measure_agreement(tmp_path, monkeypatch):
    grading_root(tmp_path, monkeypatch, raters=("A", "B"))
    tuh.main(["labels"])
    lab = trh.read_json(tuh.out_dir() / tuh.LABELS_JSON)
    pair = lab["agreement"]["all"]["pairs"][0]
    assert (pair["a"], pair["b"], pair["n"]) == ("R1", "R2", 2 * len(IMAGES))
    assert lab["agreement"]["pilot"]["pairs"][0]["n"] == 8
    text = (paths.path("reports") / tuh.REPORT_NAME).read_text(encoding="utf-8")
    assert "R1 กับ R2" in text and "ICC(2,1)" in text


def test_labels_without_usable_files_stop(tmp_path, monkeypatch):
    fake_root(tmp_path, monkeypatch)
    write_manifest(paths.path("grading") / tuh.MANIFEST_NAME)
    results = paths.path("grading") / tuh.RESULTS_FOLDER
    results.mkdir(parents=True)
    write_csv(results / "pilot.csv", full_set("A", IMAGES[:4]))
    with pytest.raises(SystemExit, match="no grader file covers every graded image"):
        tuh.cmd_labels()


# ---------------------------------------------------------------- synthetic rows


def joined_rows(n_images=90, seed=0):
    """A joined table and features: train copies of train images, fixed copies of val and test, both sides."""
    rng = np.random.default_rng(seed)
    rows, feats = [], []
    w = rng.normal(size=16)
    for k in range(n_images):
        split = ("train", "train", "val", "test")[k % 4]
        source = "scut" if k % 5 == 0 else "ffhq"
        for side in ("left", "right"):
            latent = rng.normal()
            grade = int(latent > 0.2) + int(latent > 1.2) + int(latent > 2.0)
            copies = [("train", c) for c in (1, 2, 3)] if split == "train" else [("fixed", 1), ("fixed", 2)]
            for kind, c in copies:
                x = latent * w / np.linalg.norm(w) * 3 + rng.normal(0, 1, 16)
                rows.append({"image": f"{k:05d}.png", "source": source, "side": side, "split": split, "kind": kind, "copy": c,
                             "target_iod": 77.0 if c == 1 else 110.0, "iod_px": 80.0, "reasons": "smiling" if (k % 7 == 0 and c == 2) else "",
                             "sex": "female" if k % 2 else "male", "age_group": "" if source == "scut" else ("20-29", "50-69")[k % 2],
                             "flagged": int(k % 11 == 0), "excluded": "", "p_ge1": float(grade >= 1), "p_ge2": float(grade >= 2),
                             "target_ge1": float(grade >= 1), "target_ge2": float(grade >= 2), "grade_R1": grade})
                feats.append(x)
    df = pd.DataFrame(rows)
    df["grade_R1"] = df["grade_R1"].astype("Int64")
    return df, np.asarray(feats)


def test_select_feature_scores_every_lambda_on_val():
    joined, x = joined_rows()
    parts = ud.subsets(joined, ("train", "val"))
    out = tuh.select_feature("cls", x, parts["train"], parts["val"], lambdas=(1e-3, 1e-1, 10.0))
    g = out["grid"]
    assert out["dim"] == 16 and len(g["lambdas"]) >= 3 and len(out["fits"]) == len(g["lambdas"])
    m = coral.fit(x[parts["train"].positions], parts["train"].y, g["lambdas"][0])
    want = coral.cross_entropy(m.logits(x[parts["val"].positions]), parts["val"].y)
    assert g["ce"][0] == pytest.approx(want, rel=1e-9)
    choice = tuh.choose({"cls": out}, {"cls": 16})
    assert choice["feature"] == "cls" and choice["val_ce"] == pytest.approx(min(g["ce"]), rel=0.011)


def predictions(seed=1):
    joined, x = joined_rows(seed=seed)
    parts = ud.subsets(joined, ("train", "val", "test", "test_all"))
    m = coral.fit(x[parts["train"].positions], parts["train"].y, 0.01)
    w, b = m.folded()
    table = tuh.prediction_table(joined, coral.folded_logits(x, w, b), 1.3)
    table["used"] = False
    table.loc[parts["test"].positions, "used"] = True
    table["all_copies"] = False
    table.loc[parts["test_all"].positions, "all_copies"] = True
    return table[table["all_copies"]].reset_index(drop=True), m, x, w, b, parts


def test_prediction_table_and_evaluation_follow_the_formulas():
    table, m, x, _, _, parts = predictions()
    rows = table[table["used"]]
    p = m.predict(x[parts["test"].positions], 1.3)
    np.testing.assert_allclose(rows[["p1", "p2"]].to_numpy(), p, atol=1e-9)
    np.testing.assert_allclose(rows[["p1_raw", "p2_raw"]].to_numpy(), m.predict(x[parts["test"].positions]), atol=1e-9)
    masks = tuh.breakdown_masks(table)
    assert (table.loc[masks["all"], "flagged"] == 0).all() and (table.loc[masks["flagged"], "flagged"] == 1).all()
    assert masks["all_copies"].sum() > masks["all"].sum() and not (table.loc[masks["all_copies"], "flagged"] == 1).any()
    r = tuh.evaluate_rows(table[masks["all"]], *criteria.UNDEREYE_THRESHOLDS)
    v = r["versus"]["R1"]
    assert v["n"] == r["rows"] and sum(r["model_grades"]) == r["rows"] and v["kappa"] > 0.3
    assert r["between"] == [] and 0 <= r["borderline"] <= 1
    assert set(r["calibration"]) == {"before", "after"} and r["calibration"]["after"]["k1"]["ece"] is not None


def test_cross_flags():
    good = {"after": {"k1": {"ece": 0.05}, "k2": {"ece": 0.04}}}
    assert tuh.cross_flags(1.2, False, good, {"after": {"k1": {"ece": 0.08}, "k2": {"ece": 0.5}}}, 0) == []
    flags = tuh.cross_flags(2.5, True, good, {"after": {"k1": {"ece": 0.12}, "k2": {"ece": 0.5}}}, 3)
    assert flags[0] == "T 2.500 is outside 0.5-2.0" and "bound" in flags[1]
    assert any(f.startswith("k1:") for f in flags) and any(f.startswith("k2:") for f in flags)
    # 0.09 is not twice 0.05: no note even though it is higher
    assert tuh.cross_flags(1.0, False, good, {"after": {"k1": {"ece": 0.09}, "k2": {"ece": 0.0}}}, 1) == []


def test_fidelity_sides_are_lowest_iod_copies_of_the_main_evaluation():
    table, *_ = predictions()
    one = tuh.pick_fidelity(table, 6)
    two = tuh.pick_fidelity(table.sample(frac=1, random_state=0), 6)
    assert one[["image", "side"]].values.tolist() == two[["image", "side"]].values.tolist() and len(one) == 6
    assert (one["copy"] == 1).all() and one["used"].all() and (one["flagged"] == 0).all()


def old_bundle(tmp_path):
    ev, arrays, _, _ = fake_evaluate()
    manifest, out = trh.bundle_parts(ev, arrays, META)
    bundle.write_bundle(manifest, out, tmp_path / "old")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BundleWarning)
        return bundle.load_bundle(tmp_path / "old")


def fake_calibrate(m: coral.Ordinal, temperature: float) -> dict:
    return {"choice": {"feature": "cls", "lam": m.lam, "val_ce": 0.5},
            "head": {"train_rows": 100, "train_sides": 40, "train_images": 20, "val_rows": 30, "converged": True},
            "temperature": {"temperature": temperature, "ce_before": 0.6, "ce_after": 0.55, "at_bound": False, "bounds": [0.25, 4.0]}}


def test_bundle_written_and_read_back_gives_the_trained_probabilities(tmp_path):
    old = old_bundle(tmp_path)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, DIMS["cls"])) * 0.5 + 0.1
    s = x[:, :5].sum(axis=1)
    y = np.column_stack([s > -0.5, s > 1.0]).astype(float)
    m = coral.fit(x, y, 0.05)
    w, b = m.folded()
    manifest, arrays = tuh.bundle_parts(old, fake_calibrate(m, 1.7), {"weights": w, "bias": b}, META)
    bundle.write_bundle(manifest, arrays, tmp_path / "new")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BundleWarning)
        nb = bundle.load_bundle(tmp_path / "new")
    assert nb.version == "0.2.0" and nb.stub is False and nb.stub_heads == ()
    u = nb.heads["undereye"]
    assert u.temperature == 1.7 and (u.low, u.high) == criteria.UNDEREYE_THRESHOLDS and u.feature == "cls"
    np.testing.assert_allclose(u.predict(x[:9]), m.predict(x[:9], 1.7), atol=1e-9, rtol=0)
    for t in bundle.RATIO_HEADS:
        np.testing.assert_array_equal(nb.heads[t].weights, old.heads[t].weights)
        assert nb.heads[t].bias == old.heads[t].bias and nb.baseline[t] == old.baseline[t]
        assert nb.manifest["heads"][t]["from_bundle"] == "0.1.0"
    with open(tmp_path / "new" / "manifest.json", encoding="utf-8") as f:
        assert json.load(f)["heads"]["undereye"]["stub"] is False


def test_bundle_refuses_stub_ratio_heads_and_other_features(tmp_path):
    import make_stub_bundle

    rng = np.random.default_rng(6)
    m = coral.fit(rng.normal(size=(50, DIMS["cls"])), np.tile([[1.0, 0.0], [0.0, 0.0]], (25, 1)), 1.0)
    w, b = m.folded()
    stub = make_stub_bundle.stub_bundle()
    with pytest.raises(SystemExit):
        tuh.bundle_parts(stub, fake_calibrate(m, 1.0), {"weights": w, "bias": b}, {**META, "backbone": stub.manifest["backbone"],
                         "train_face_iod_px": None, "degrade": None})
    old = old_bundle(tmp_path)
    with pytest.raises(SystemExit, match="degrade"):
        tuh.bundle_parts(old, fake_calibrate(m, 1.0), {"weights": w, "bias": b}, {**META, "degrade": {"version": 3}})


def test_report_has_every_section_from_synthetic_results(tmp_path, monkeypatch):
    grading_root(tmp_path, monkeypatch)
    tuh.main(["labels"])
    folder = tuh.out_dir()
    joined, x = joined_rows(seed=2)
    parts = ud.subsets(joined, ud.SUBSETS)
    sel = {"fingerprint": {}, "counts": ud.counts({k: parts[k] for k in ("train", "val")}), "features": {}}
    sel["features"]["cls"] = tuh.select_feature("cls", x, parts["train"], parts["val"], lambdas=(1e-2, 1.0))
    sel["choice"] = tuh.choose(sel["features"], {"cls": 16})
    m = coral.fit(x[parts["train"].positions], parts["train"].y, sel["choice"]["lam"])
    w, b = m.folded()
    lv = coral.folded_logits(x[parts["val"].positions], w, b)
    temp = coral.fit_temperature(lv, parts["val"].y)
    p = coral.sigmoid(lv / temp.temperature)
    cal = {
        "choice": sel["choice"], "head": {**tuh.head_dict(m, parts["train"], "cls"), "standardize_gap": 0.0, "val_rows": 10, "val_sides": 5},
        "temperature": temp.as_dict(), "val": {"rows": len(lv), "sides": parts["val"].sides,
                                               "calibration": tuh.calibration_block(lv, parts["val"].y, temp.temperature)},
        "thresholds": tuh.threshold_table(p, parts["val"].rows), "bundle_thresholds": list(criteria.UNDEREYE_THRESHOLDS),
        "supporting": {"rows": 20, "sides": 10, "temperature": temp.as_dict(),
                       "calibration_with_val_t": tuh.calibration_block(lv, parts["val"].y, temp.temperature)},
    }
    table, *_ = predictions(seed=2)
    results = {k: tuh.evaluate_rows(table[mk], 0.3, 0.7) for k, mk in tuh.breakdown_masks(table).items()}
    ev = {"choice": sel["choice"], "temperature": temp.temperature, "thresholds": {"low": 0.3, "high": 0.7},
          "copy_iod": {"1": 77.0, "2": 110.0}, "counts": ud.counts({"test": parts["test"]}), "results": results}
    scut = tuh.evaluate_rows(table[table["source"] == "scut"], 0.3, 0.7)
    cross = {"select": {"choice": sel["choice"]}, "temperature": {**temp.as_dict(), "temperature": 2.6},
             "head": {"train_rows": 50}, "ffhq_val": {"rows": 10, "calibration": cal["val"]["calibration"]}, "scut": scut,
             "scut_flagged_rows": 0, "scut_ge1_sides": 3, "scut_ge2_sides": 0, "counts": ud.counts({"cross_eval": parts["cross_eval"]}),
             "flags": tuh.cross_flags(2.6, False, cal["val"]["calibration"], scut["calibration"], 0)}
    fid = {"bundle_version": "0.2.0", "ratio_heads_from": "0.1.0", "sides": 20, "images": 15, "tolerance": 1e-4, "max_diff": 3e-7,
           "passed": True, "same_reasons": True, "same_decisions": True}
    for name, data in ((tuh.SELECT_NAME, sel), (tuh.CALIBRATE_NAME, cal), (tuh.EVALUATE_NAME, ev), (tuh.CROSS_NAME, cross),
                       (tuh.FIDELITY_NAME, fid)):
        trh.write_json(folder / name, data)
    tuh.log("evaluate", "reads the test split once", folder)
    text = tuh.write_report().read_text(encoding="utf-8")
    for title in ("ผู้ให้เกรดและ label", "การกระจายของเกรด", "ความสอดคล้องระหว่างผู้ให้เกรด", "จำนวนแถว", "การเลือกชนิด feature และ λ",
                  "การปรับเทียบบนชุด val", "ผลบนชุด test", "ตรวจข้ามชุดข้อมูล", "bundle และความตรงกับตอนเตรียมข้อมูล",
                  "ข้อสังเกตและข้อจำกัด", "ชุดข้อมูลที่แต่ละคำสั่งอ่าน"):
        assert f"## {title}\n" in text
    assert "ยังไม่ได้รัน" not in text
    assert "เป็นการเทียบคนละแบบ" in text and "0.55" in text and "0.65" in text
    assert "ตรวจจุดเปลี่ยนที่ 2 ข้ามชุดข้อมูลไม่ได้" in text and "T อยู่นอกช่วง 0.5–2.0" in text
    assert "0.3 / 0.7 (ใช้อยู่)" in text and "**ผ่าน**" in text
    for name in ("undereye_grades.png", "undereye_reliability_test.png", "undereye_reliability_cross.png"):
        assert (paths.path("reports") / "figures" / name).is_file()


def test_report_without_results_says_what_to_run(tmp_path, monkeypatch):
    fake_root(tmp_path, monkeypatch)
    text = tuh.write_report().read_text(encoding="utf-8")
    assert "ยังไม่ได้รัน `labels`" in text and "ยังไม่ได้รัน `bundle`" in text and "ยังไม่ได้รัน `cross`" in text


def test_steps_need_their_inputs(tmp_path, monkeypatch):
    fake_root(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="labels"):
        tuh.fingerprint()


def test_grid_round_trip():
    g = ridge.Grid("cls", (0.1, 1.0), (0.6, 0.5), (), False)
    assert tuh.grid_of("cls", tuh.grid_dict(g)) == g
