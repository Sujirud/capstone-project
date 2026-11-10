"""ตรวจจุด 86 จุดของ SCUT-FBP5500 ก่อนใช้สร้าง label ของกรามและฐานจมูก (model.md หัวข้อ 3.1 และ 5.2)

ตัวอย่าง (รันจาก root ของโปรเจกต์)
  uv run python tools/scut_landmark_preview.py --images "data/raw/SCUT-FBP5500_v2/Images" --pts "data/raw/SCUT-FBP5500_v2/facial landmark" --out data/outputs/scut_preview

ทำสองอย่าง
1. สุ่มภาพมา --n ภาพ วาดจุดพร้อมเลขทับภาพไว้ใน preview/ เพื่อดูด้วยตาว่าจุดที่ใช้อยู่ถูกที่
   แดง = หัวตา 43, 55 และฐานปีกจมูก 65, 69 · ฟ้า = เส้นขอบหน้า 1–22
   เหลือง = โหนกแก้ม (Zy) · เขียว = มุมกราม (Go) ซึ่งคำนวณจากเส้นขอบด้วยกฎใน model.md หัวข้อ 5.2
   เส้นเหลืองกับเส้นเขียวควรขนานกับเส้นตา เส้นเขียวอยู่ระดับมุมปาก และวงกลมควรอยู่บนขอบหน้าพอดี
2. คำนวณอัตราส่วนกราม (Go–Go/Zy–Zy) และฐานจมูก (65–69 / 43–55) ของทุกไฟล์ลง preview/ratios.csv
   แล้วสรุปว่าแต่ละกลุ่มตกเกรดไหนกี่เปอร์เซ็นต์ เทียบกับเส้นแบ่งใน model.md หัวข้อ 3.1 และ 3.4

กฎ เส้นแบ่ง การอ่านไฟล์ และการวาดมาจากแพ็กเกจ facemeasure (scut, face_geometry, criteria, draw)
ตัวเดียวกับที่ model/scripts/make_scut_labels.py ใช้ จึงต้องรันผ่าน uv
ไฟล์ .pts ของ SCUT-FBP5500 เป็นไบนารี คือ int32 หนึ่งค่า ตามด้วย float32 172 ค่า (x, y ของ 86 จุด)
แบบเดียวกับ pts2txt.py ใน repo ของชุดข้อมูล ไฟล์ .txt แบบ "x y" บรรทัดละจุดก็อ่านได้
"""

import argparse
import csv
import random
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from facemeasure import criteria, scut
from facemeasure.face_geometry import GeometryError

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
SEX = {"F": "female", "M": "male"}
GRADE_LABEL = {"0": "0", "1": "1", "2+": "≥2"}


def nearest_point(p, pts, numbers):
    return min(numbers, key=lambda n: float(np.hypot(*(np.asarray(p) - scut.take(pts, n)))))


def measure(pts):
    r = scut.measure(pts)
    return {
        "nose": r.nose_ratio,
        "jaw": r.jaw_ratio,
        "go_near": f"{nearest_point(r.go_l, pts, scut.CONTOUR_IMAGE_LEFT)}/{nearest_point(r.go_r, pts, scut.CONTOUR_IMAGE_RIGHT)}",
        "zy_near": f"{nearest_point(r.zy_l, pts, scut.CONTOUR_IMAGE_LEFT)}/{nearest_point(r.zy_r, pts, scut.CONTOUR_IMAGE_RIGHT)}",
        "result": r,
    }


def draw(img_path, pts, m, out_path, scale=2):
    with Image.open(img_path) as im:
        if pts[:, 0].max() > im.width * 1.05 or pts[:, 1].max() > im.height * 1.05:
            print(f"เตือน {img_path.name}: พิกัดเกินขนาดภาพ ไฟล์จุดอาจใช้สเกลอื่น")
        label = f"jaw {m['jaw']:.3f}  nose {m['nose']:.3f}"
        out = scut.draw_points(im, pts, m["result"], numbered=True, label=label, scale=scale)
    out.save(out_path)


def summarize(rows):
    print("\nสรุปจากไฟล์จุดทั้งหมด")
    for group in scut.GROUPS:
        g = [r for r in rows if r["group"] == group]
        if not g:
            continue
        sex = SEX[group[1]]
        jaws = [r["jaw"] for r in g]
        noses = [r["nose"] for r in g]
        line = f"{group} {len(g)} ภาพ · กรามมัธยฐาน {statistics.median(jaws) * 100:.1f}%"
        if sex == "female":
            grades = [criteria.jaw_grade(x, sex) for x in jaws]
            share = {GRADE_LABEL[k]: grades.count(k) / len(g) * 100 for k in criteria.GRADES}
            line += f" (≤83%: {share['0']:.0f}% · 83–87%: {share['1']:.0f}% · >87%: {share['≥2']:.0f}%)"
        grades = [criteria.nose_grade(x, sex) for x in noses]
        share = {GRADE_LABEL[k]: grades.count(k) / len(g) * 100 for k in criteria.GRADES}
        line += (f" · ฐานจมูกมัธยฐาน {statistics.median(noses):.3f}"
                 f" (เกรด 0: {share['0']:.0f}% · 1: {share['1']:.0f}% · ≥2: {share['≥2']:.0f}%)")
        print(line)
    go = {}
    for r in rows:
        for n in r["go_near"].split("/"):
            go[n] = go.get(n, 0) + 1
    total = sum(go.values())
    print("Go อยู่ใกล้จุดเลข: " + " · ".join(f"{k}: {v / total * 100:.0f}%" for k, v in sorted(go.items(), key=lambda kv: int(kv[0]))))
    print("ถ้าผู้หญิงเกิน 83% เป็นส่วนใหญ่ ให้ดูภาพใน preview/ ก่อนสร้าง label (model.md หัวข้อ 3.1)")


def main():
    ap = argparse.ArgumentParser(description="ตรวจจุดของ SCUT-FBP5500")
    ap.add_argument("--images", help="โฟลเดอร์ภาพ (Images) ถ้าไม่ใส่จะคำนวณสถิติอย่างเดียว")
    ap.add_argument("--pts", required=True, help='โฟลเดอร์ไฟล์จุด ("facial landmark")')
    ap.add_argument("--n", type=int, default=12, help="จำนวนภาพที่สุ่มมาวาด")
    ap.add_argument("--out", default="preview", help="โฟลเดอร์ผล")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pts_files = sorted(p for p in Path(args.pts).iterdir() if p.suffix.lower() in (".pts", ".txt"))
    if not pts_files:
        raise SystemExit("ไม่พบไฟล์ .pts หรือ .txt ใน --pts")

    rows, bad, measured, readable = [], [], {}, []
    for p in pts_files:
        try:
            pts = scut.read_points(p)
        except scut.PtsError as err:
            bad.append(f"{p.name} ({err})")
            continue
        readable.append(p.stem)
        try:
            m = measure(pts)
        except GeometryError as err:
            bad.append(f"{p.name} (คำนวณไม่ได้: {err})")
            continue
        measured[p.stem] = (pts, m)
        rows.append({"image": p.stem, "group": p.stem[:2], "jaw": m["jaw"], "nose": m["nose"],
                     "go_near": m["go_near"], "zy_near": m["zy_near"]})
    if bad:
        print(f"อ่านไม่ได้ {len(bad)} ไฟล์ เช่น {', '.join(bad[:5])}")
    with open(out / "ratios.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "group", "jaw_ratio", "alar_intercanthal_ratio", "go_near_points", "zy_near_points"])
        for r in rows:
            w.writerow([r["image"], r["group"], round(r["jaw"], 4), round(r["nose"], 4), r["go_near"], r["zy_near"]])
    print(f"คำนวณได้ {len(rows)} ไฟล์ บันทึกไว้ที่ {out / 'ratios.csv'}")
    if rows:
        summarize(rows)

    if args.images:
        images = {p.stem: p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTS}
        # Shuffle every readable file, then drop the ones the rules could not measure,
        # so the same seed picks the same images as before those files were dropped
        names = [k for k in readable if k in images]
        random.Random(args.seed).shuffle(names)
        names = [k for k in names if k in measured]
        if not names:
            raise SystemExit("ไม่พบภาพที่ชื่อตรงกับไฟล์จุด ตรวจ --images")
        for name in names[: args.n]:
            pts, m = measured[name]
            draw(images[name], pts, m, out / f"{name}_pts.png")
        print(f"วาด {min(args.n, len(names))} ภาพไว้ที่ {out}/")


if __name__ == "__main__":
    main()
