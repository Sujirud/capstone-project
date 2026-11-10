"""model/scripts/api_check.py without the models: the report from synthetic results, and the picture helpers."""

import io

import numpy as np
from PIL import Image

import api_check

INFO = {
    "api": "0.1.0", "bundle": "0.2.0", "stubs": "", "device": "cpu", "machine": "Windows AMD64", "python": "3.12.16",
    "mediapipe": "1.2.0", "torch": "2.15.0", "created": "2026-11-07T16:00:00+07:00",
}
SPREAD = {"median": 0.5, "p90": 1.4, "max": 2.7}
RESULTS = {
    "ratio_fidelity": {"images": 5, "max_diff": {"jaw_value": 0.05, "jaw_landmark": 0.03, "nose_value": 0.0003, "nose_landmark": 0.0004},
                       "tolerance": {"%": 0.051, "ratio": 0.00051}, "same_codes": True, "passed": True},
    "undereye_fidelity": {"images": 5, "sides": 10, "max_diff": 0.0004, "tolerance": 0.00051, "same_codes": True, "passed": True},
    "repeat": {"requests": 2, "same": True, "request_ids_differ": True, "passed": True},
    "exif": {"pairs": 3, "requests": 27, "upright_shape_kept": True, "png_same_result": True, "png_max_jaw_pp": 0.0,
             "png_max_angle_deg": 0.0, "jpeg_max_jaw_pp": 0.3, "jpeg_max_angle_deg": 0.5, "resave_max_jaw_pp": 0.1,
             "resave_max_angle_deg": 0.5, "jpeg_within_tolerance": True, "passed": True},
    "profile_flip": {"images": 20, "measured_both": 17, "measured_one_only": 2, "measured_neither": 1, "direction_flips": True,
                     "legan": SPREAD, "nasofacial": SPREAD, "share_within": 0.53, "max_angle_deg": 2.7, "passed": False},
    "quality_codes": {
        "face_too_small": {"images": 5, "found": 5, "passed": True},
        "frontal_pose": {"images": 5, "found": 5, "roll_deg": 15.0, "passed": True},
        "smiling": {"images": 20, "found": 20, "share": 1.0, "passed": True},
        "profile_invalid": {"images": 10, "found": 10, "passed": True},
        "missing_input": {"images": 1, "found": 1, "passed": True},
        "passed": True,
    },
    "lab": {"found": True, "regions_failing": 4, "assessed_in_lab": 4, "retake_in_lab": 0, "lab": True, "passed": True},
    "timing": {"requests": 20, "frame": [1920, 1080], "p50_s": 0.59, "p90_s": 0.62, "mean_s": 0.6, "max_s": 0.75, "device": "cpu",
               "passed": True},
}


def test_report_has_every_check_and_only_numbers():
    text = api_check.report(RESULTS, {}, INFO)
    for title in api_check.TITLES.values():
        assert f"| {title} |" in text
    for heading in ("## ความตรงกับตอนเตรียมข้อมูล", "## ส่งคำขอเดิมซ้ำ", "## การหมุนและการกลับภาพ", "## รหัสคุณภาพภาพ",
                    "## โหมดทดลอง", "## เวลาต่อคำขอ"):
        assert heading in text
    assert "| 0.59 วินาที | 0.62 วินาที |" in text and "| Legan | 0.50 | 1.40 | 2.70 |" in text
    assert "| กลับภาพด้านข้าง | ไม่ผ่าน |" in text and "| หมุนด้วย EXIF | ผ่าน |" in text
    assert ".jpg" not in text and ".png" not in text and "![" not in text  # no file names, no pictures
    assert text.endswith("\n") and "\r" not in text


def test_report_lists_what_was_skipped():
    skipped = {"undereye_fidelity": "no outputs/undereye_head/test_predictions.csv", "timing": "no FEI images"}
    results = {k: v for k, v in RESULTS.items() if k not in skipped}
    text = api_check.report(results, skipped, INFO)
    assert "| ความตรงกับตอนเทรน (ใต้ตา) | ข้าม: no outputs/undereye_head/test_predictions.csv |" in text
    assert "## เวลาต่อคำขอ" not in text and "ใต้ตา: สำเนา" not in text


def test_stored_turns_come_back_upright_through_exif():
    from facemeasure import image_io

    rgb = np.random.default_rng(0).integers(0, 256, size=(30, 50, 3), dtype=np.uint8)
    for o in api_check.EXIF_ORIENTATIONS:
        stored = api_check.turned(rgb, o)
        if o != 3:
            assert stored.size == (30, 50)  # stored on its side
        assert np.array_equal(image_io.read_image(api_check.png_exif(stored, o)), rgb)


def test_timing_frame_is_camera_sized():
    from facemeasure import image_io

    face = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = image_io.read_image(api_check.timing_frame(face))
    assert frame.shape == (1080, 1920, 3)
    assert Image.open(io.BytesIO(api_check.timing_frame(face))).format == "JPEG"
