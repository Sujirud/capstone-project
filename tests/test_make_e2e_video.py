"""model/scripts/make_e2e_video.py on synthetic rows and frames."""

import io
import json

import numpy as np
import pytest
from PIL import Image

import make_e2e_video as mev
from facemeasure import image_io, paths


def row(image, split="test", failed="", iod=120.0, yaw=0.0, smile=0.0, source="scut", faces="1"):
    return {
        "image": image, "source": source, "split": split, "face_count": faces, "failed": failed, "iod_px": str(iod),
        "yaw_deg": str(yaw), "pitch_deg": "0", "roll_deg": "0", "smile": str(smile), "jaw_open": "0",
        "blink_left": "0.1", "blink_right": "0.2",
    }


def test_candidates_pass_every_check_are_large_enough_and_best_first():
    rows = [
        row("AF2.jpg", yaw=4.0),
        row("AF1.jpg", yaw=1.0),
        row("AF3.jpg", yaw=0.5, smile=0.2),  # 0.5 + 20 * (0.2 + 0.2) = 8.5
        row("AF4.jpg", split="train"),
        row("AF5.jpg", failed="smiling"),
        row("AF6.jpg", iod=mev.TARGET_IOD_PX - 1),  # would have to be enlarged
        row("AF7.jpg", faces="2"),
        row("x.png", source="ffhq"),
    ]
    assert [r["image"] for r in mev.frontal_candidates(rows)] == ["AF1.jpg", "AF2.jpg", "AF3.jpg"]
    assert mev.score(row("a", yaw=-2.0)) == pytest.approx(2.0 + 20 * 0.2)


def test_compose_puts_the_scaled_face_in_the_middle_of_a_plain_frame():
    face = np.full((100, 80, 3), 10, dtype=np.uint8)
    frame = mev.compose(face, 0.5)
    w, h = mev.FRAME_SIZE
    assert frame.shape == (h, w, 3) and frame.dtype == np.uint8
    ys, xs = np.nonzero(frame[:, :, 0] == 10)
    assert (ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) == (50, 40)
    assert abs((ys.min() + ys.max()) / 2 - h / 2) <= 1 and abs((xs.min() + xs.max()) / 2 - w / 2) <= 1
    assert tuple(frame[0, 0]) == mev.BACKGROUND


def test_the_exif_image_reads_back_upright():
    w, h = mev.FRAME_SIZE
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[: h // 2, : w // 4] = (220, 40, 40)  # a mark at the top left
    data = mev.exif_rotated(frame)
    assert image_io.read_image(data)[h // 8, w // 16].tolist() == pytest.approx([220, 40, 40], abs=12)
    stored = Image.open(io.BytesIO(data))
    assert stored.size == (h, w) and stored.getexif()[mev.EXIF_ORIENTATION_TAG] == 6


def make_project(tmp_path, monkeypatch, fei_rows):
    (tmp_path / "config.toml").write_text(
        '[paths]\nfei = "data/raw/fei"\nlabels = "data/labels"\noutputs = "data/outputs"\n', encoding="utf-8"
    )
    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path))
    images = tmp_path / "data" / "raw" / "fei" / "images"
    images.mkdir(parents=True)
    for r in fei_rows:
        if r.pop("exists", True):
            (images / r["image"]).write_bytes(b"jpeg")
    return images


def test_the_profile_is_the_first_full_profile_that_passed(tmp_path, monkeypatch):
    rows = [
        {"image": "2-10.jpg", "pose": "full", "failed": "", "read_error": ""},
        {"image": "1-10.jpg", "pose": "full", "failed": "profile_invalid", "read_error": ""},
        {"image": "1-09.jpg", "pose": "second", "failed": "", "read_error": ""},
        {"image": "1-01.jpg", "pose": "full", "failed": "", "read_error": "", "exists": False},
    ]
    images = make_project(tmp_path, monkeypatch, rows)
    assert mev.profile_choice(rows) == images / "2-10.jpg"
    with pytest.raises(SystemExit):
        mev.profile_choice(rows[1:])


def test_write_makes_the_video_the_exif_image_and_the_config(tmp_path, monkeypatch):
    images = make_project(tmp_path, monkeypatch, [{"image": "1-01.jpg"}])
    w, h = mev.FRAME_SIZE
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    folder = tmp_path / "out"
    path = mev.write(folder, frame, row("AF1.jpg"), 90.44, images / "1-01.jpg")
    video = (folder / mev.VIDEO_NAME).read_bytes()
    assert video.count(b"\xff\xd8\xff") == mev.FRAMES and video.startswith(b"\xff\xd8")
    assert image_io.read_image(folder / mev.EXIF_NAME).shape == (h, w, 3)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n") and "\r" not in text
    config = json.loads(text)
    assert config["frame"] == {"width": w, "height": h, "iod_px": 90.4}
    assert config["frontal_image"] == "AF1.jpg" and config["profile_image"] == "1-01.jpg"
    assert config["video"] == str((folder / mev.VIDEO_NAME).resolve())


def test_missing_label_files_stop_with_a_message(tmp_path, monkeypatch):
    make_project(tmp_path, monkeypatch, [])
    with pytest.raises(SystemExit, match="frontal_check.py or profile_check.py"):
        mev.read_rows("frontal_mediapipe.csv")
