"""facemeasure.scut: landmark files, point numbers, names, and the real-data check."""

import math
import struct

import numpy as np
import pytest
from PIL import Image

from facemeasure import draw, paths, scut
from facemeasure.face_geometry import GeometryError

# Values from tools/scut_landmark_preview.py on the real files
PREVIEW = {"AF1075": (0.7751, 1.0405), "AM1382": (0.8681, 1.1070), "CM290": (0.8868, 1.1951)}
BAD_ANNOTATIONS = {"AF1183": "eye centers", "CF434": "eye centers", "AM1320": "mouth level"}


def write_pts(path, points, count=scut.N_POINTS, extra=b""):
    path.write_bytes(struct.pack("<i", count) + struct.pack("<172f", *np.asarray(points).ravel()) + extra)
    return path


def sample_points():
    return np.arange(172, dtype=np.float64).reshape(86, 2) * 0.5  # exact in float32


def test_read_pts_round_trip(tmp_path):
    points = scut.read_pts(write_pts(tmp_path / "AF1.pts", sample_points()))
    assert points.shape == (86, 2)
    np.testing.assert_array_equal(points, sample_points())


def test_short_file_raises(tmp_path):
    path = tmp_path / "CM152.pts"
    path.write_bytes(struct.pack("<i", 86))
    with pytest.raises(scut.PtsError, match="4 bytes, expected 692"):
        scut.read_pts(path)


def test_long_file_raises(tmp_path):
    with pytest.raises(scut.PtsError, match="693 bytes"):
        scut.read_pts(write_pts(tmp_path / "AF2.pts", sample_points(), extra=b"\0"))


def test_wrong_point_count_raises(tmp_path):
    with pytest.raises(scut.PtsError, match="header says 68 points"):
        scut.read_pts(write_pts(tmp_path / "AF3.pts", sample_points(), count=68))


def test_non_finite_point_raises(tmp_path):
    points = sample_points()
    points[10, 1] = math.nan
    with pytest.raises(scut.PtsError, match="not finite"):
        scut.read_pts(write_pts(tmp_path / "AF4.pts", points))


def test_read_points_txt(tmp_path):
    points = sample_points()
    path = tmp_path / "AF5.txt"
    path.write_text("\n".join(f"{i + 1} {x} {y}" for i, (x, y) in enumerate(points)) + "\n", encoding="utf-8")
    np.testing.assert_array_equal(scut.read_points(path), points)
    path.write_text("\n".join(f"{x} {y}" for x, y in points[:85]) + "\n", encoding="utf-8")
    with pytest.raises(scut.PtsError, match="85 points"):
        scut.read_points(path)


def test_read_points_rejects_other_files(tmp_path):
    path = tmp_path / "AF6.png"
    path.write_bytes(b"x")
    with pytest.raises(scut.PtsError, match=r"\.pts or \.txt"):
        scut.read_points(path)


def test_take_converts_point_numbers():
    points = sample_points()
    np.testing.assert_array_equal(scut.take(points, 1), points[0])
    np.testing.assert_array_equal(scut.take(points, (2, 86)), points[[1, 85]])
    for bad in (0, 87, (1, 0), 1.5, ()):
        with pytest.raises(ValueError, match="1 to 86"):
            scut.take(points, bad)


@pytest.mark.parametrize(
    "name, group, sex, ethnicity, number",
    [
        ("AF1.jpg", "AF", "female", "asian", 1),
        ("AM1382.jpg", "AM", "male", "asian", 1382),
        ("CF434.pts", "CF", "female", "caucasian", 434),
        ("CM290", "CM", "male", "caucasian", 290),
    ],
)
def test_parse_name(name, group, sex, ethnicity, number):
    assert scut.parse_name(name) == scut.Subject(group, sex, ethnicity, number)


@pytest.mark.parametrize("name", ["XF1.jpg", "AF.jpg", "AF0.jpg", "AF01.jpg", "AF1.png", "af1.jpg", "AF1.jpg.bak"])
def test_parse_name_rejects(name):
    with pytest.raises(ValueError, match="SCUT-FBP5500 name"):
        scut.parse_name(name)


def test_frontal_points_use_the_documented_numbers():
    points = np.array([(n, -n) for n in range(1, 87)], dtype=np.float64)
    fp = scut.frontal_points(points)
    assert fp.contour_l[:, 0].tolist() == list(range(2, 12))
    assert fp.contour_r[:, 0].tolist() == list(range(22, 12, -1))
    picks = {"eye_l": 59, "eye_r": 60, "canthus_l": 43, "canthus_r": 55, "alar_l": 65, "alar_r": 69,
             "subnasale": 67, "mouth_l": 80, "mouth_r": 74}
    for field, number in picks.items():
        assert getattr(fp, field).tolist() == [number, -number]


def test_measure_synthetic_face(scut_face):
    r = scut.measure(scut_face())
    assert r.jaw_ratio == pytest.approx(math.sqrt(1 - 0.5**2), abs=0.01)
    assert r.nose_ratio == pytest.approx(0.36 / 0.30)


def test_swapped_eye_centers_raise(scut_face):
    points = scut_face()
    points[[58, 59]] = points[[59, 58]]
    with pytest.raises(GeometryError, match="eye centers"):
        scut.measure(points)


def test_point_colors_and_drawing(scut_face):
    colors = scut.point_colors()
    assert colors[42] == colors[54] == colors[64] == colors[68] == draw.RED
    assert colors[0] == colors[21] == draw.BLUE
    assert colors[22] == colors[85] == draw.WHITE
    points = scut_face()
    out = scut.draw_points(Image.new("RGB", (350, 350)), points, scut.measure(points), numbered=True, scale=2)
    assert out.size == (700, 700)


def landmark_dir():
    try:
        folder = paths.path("scut") / "facial landmark"
    except FileNotFoundError:
        pytest.skip("no config.toml")
    if not folder.is_dir():
        pytest.skip("SCUT-FBP5500 landmarks are not in data/raw")
    return folder


@pytest.mark.parametrize("name", sorted(PREVIEW))
def test_real_files_match_the_preview_script(name):
    path = landmark_dir() / f"{name}.pts"
    if not path.is_file():
        pytest.skip(f"{path.name} missing")
    r = scut.measure(scut.read_pts(path))
    jaw, nose = PREVIEW[name]
    assert r.jaw_ratio == pytest.approx(jaw, abs=0.0005)
    assert r.nose_ratio == pytest.approx(nose, abs=0.0005)


def test_real_cm152_is_unreadable():
    path = landmark_dir() / "CM152.pts"
    if not path.is_file():
        pytest.skip("CM152.pts missing")
    with pytest.raises(scut.PtsError):
        scut.read_pts(path)


@pytest.mark.parametrize("name", sorted(BAD_ANNOTATIONS))
def test_real_bad_annotations_are_rejected(name):
    path = landmark_dir() / f"{name}.pts"
    if not path.is_file():
        pytest.skip(f"{path.name} missing")
    with pytest.raises(GeometryError, match=BAD_ANNOTATIONS[name]):
        scut.measure(scut.read_pts(path))
