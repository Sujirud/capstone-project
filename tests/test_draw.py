"""facemeasure.draw on blank images."""

import numpy as np
import pytest
from PIL import Image

from facemeasure import draw
from facemeasure.face_geometry import FrontalPoints, JawNose


def simple_face():
    fp = FrontalPoints(
        contour_l=[(10.0, 20.0), (12.0, 60.0), (20.0, 90.0)],
        contour_r=[(90.0, 20.0), (88.0, 60.0), (80.0, 90.0)],
        eye_l=(30.0, 30.0), eye_r=(70.0, 30.0), subnasale=(50.0, 55.0),
        mouth_l=(38.0, 70.0), mouth_r=(62.0, 70.0),
        alar_l=(40.0, 52.0), alar_r=(60.0, 52.0), canthus_l=(42.0, 30.0), canthus_r=(58.0, 30.0),
    )
    result = JawNose(
        jaw_ratio=0.8, nose_ratio=1.25, iod_px=40.0,
        go_l=(14.0, 70.0), go_r=(86.0, 70.0), zy_l=(11.0, 45.0), zy_r=(89.0, 45.0),
    )
    return fp, result


def pixel(im, p, scale):
    return im.getpixel((round(p[0] * scale), round(p[1] * scale)))


def test_draw_jaw_nose_colors_and_scale():
    fp, result = simple_face()
    im = draw.draw_jaw_nose(Image.new("RGB", (100, 100)), fp, result, scale=2, label="jaw 0.800")
    assert im.size == (200, 200)
    assert pixel(im, (50.0, 30.0), 2) == draw.RED  # middle of the canthus line
    assert pixel(im, (50.0, 52.0), 2) == draw.RED  # middle of the alar line
    assert pixel(im, (50.0, 45.0), 2) == draw.YELLOW  # middle of the Zy line
    assert pixel(im, (50.0, 70.0), 2) == draw.GREEN  # middle of the Go line
    assert pixel(im, (12.0, 60.0), 2) == draw.BLUE  # a face edge point


def test_draw_numbered_points():
    fp, result = simple_face()
    points = np.array([[5.0, 5.0], [95.0, 95.0]])
    im = draw.draw_jaw_nose(
        Image.new("RGB", (100, 100)), fp, result,
        points=points, point_colors=[draw.WHITE, draw.BLUE], point_labels=["1", "2"],
    )
    assert im.getpixel((5, 5)) == draw.WHITE
    assert im.getpixel((95, 95)) == draw.BLUE


def test_contact_sheet_layout():
    tiles = [Image.new("RGB", (10, 10), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    sheet = draw.contact_sheet(tiles, ["a\nb", "c", "d"], columns=2)
    assert sheet.width == 20
    assert sheet.height > 20
    assert sheet.getpixel((0, 0)) == (255, 0, 0)
    assert sheet.getpixel((10, 0)) == (0, 255, 0)
    assert sheet.getpixel((0, sheet.height // 2 + 1)) == (0, 0, 255)


def test_alar_line_can_be_left_out():
    fp, result = simple_face()
    im = draw.draw_jaw_nose(Image.new("RGB", (100, 100)), fp, result, points=np.empty((0, 2)), alar_line=False)
    assert im.getpixel((50, 52)) == (0, 0, 0)
    assert im.getpixel((50, 30)) == draw.RED  # the canthus line stays


def test_draw_polylines_closed_and_open():
    box = np.array([(10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0)])
    im = draw.draw_polylines(Image.new("RGB", (100, 100)), [box], [draw.GREEN], width=1, scale=2)
    assert im.getpixel((50, 20)) == draw.GREEN and im.getpixel((20, 50)) == draw.GREEN  # closing side drawn
    im = draw.draw_polylines(Image.new("RGB", (100, 100)), [box], [draw.GREEN], width=1, scale=2, closed=False)
    assert im.getpixel((50, 20)) == draw.GREEN and im.getpixel((20, 50)) == (0, 0, 0)
    with pytest.raises(ValueError):
        draw.draw_polylines(Image.new("RGB", (10, 10)), [box], [])


def test_draw_profile(profile_face):
    from test_profile import CRITERIA, ColorModels

    from facemeasure import profile as P

    s = profile_face(scale=0.5)
    image = Image.fromarray(s.image)
    measured = P.analyze_profile(s.image, ColorModels(), CRITERIA)
    out = draw.draw_profile(image, measured, label="synthetic", scale=2)
    assert out.size == (image.width * 2, image.height * 2)
    plain = np.asarray(image.resize(out.size))
    assert (np.asarray(out) != plain).any()
    failed = P.analyze_profile(np.zeros((100, 100, 3), dtype=np.uint8), ColorModels(), CRITERIA)
    blank = draw.draw_profile(Image.new("RGB", (100, 100)), failed)
    assert blank.size == (100, 100) and np.asarray(blank).any()  # the reason is written
