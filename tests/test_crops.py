"""DINOv2 crops on a synthetic face with colored dots at known places."""

import math

import numpy as np
import pytest
from PIL import Image

from facemeasure import crops

W, H = 400, 500
RED_AT = (172.0, 230.0)  # nose side of the person's right under-eye box (x 120-180, y 200-260)
GREEN_AT = (228.0, 230.0)  # nose side of the person's left under-eye box (x 220-280)
BLUE_AT = (200.0, 330.0)  # chin area, inside the whole-face box only
DOT_RADIUS = 4.0


def face_image() -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W] + 0.5
    a = np.empty((H, W, 3), dtype=np.uint8)
    a[:] = (80 + ys / 10)[..., None].astype(np.uint8)
    for (x, y), color in ((RED_AT, (255, 0, 0)), (GREEN_AT, (0, 255, 0)), (BLUE_AT, (0, 0, 255))):
        a[np.hypot(xs - x, ys - y) <= DOT_RADIUS] = color
    return a


def dot_center(a: np.ndarray, channel: int) -> tuple[float, float]:
    others = [c for c in range(3) if c != channel]
    mask = (a[:, :, channel] > 150) & (a[:, :, others[0]] < 90) & (a[:, :, others[1]] < 90)
    ys, xs = np.nonzero(mask)
    assert len(xs), "dot not found"
    weights = a[ys, xs, channel].astype(float)
    return float(np.average(xs, weights=weights) + 0.5), float(np.average(ys, weights=weights) + 0.5)


def to_crop(corners: np.ndarray, p, size: int = crops.CROP_SIZE) -> tuple[float, float]:
    """Where image point p lands in a crop with these corners (before any mirroring)."""
    c = np.asarray(corners)
    ex, ey = c[1] - c[0], c[3] - c[0]
    d = np.asarray(p) - c[0]
    return float(d @ ex / (ex @ ex) * size), float(d @ ey / (ey @ ey) * size)


def rotation(deg: float, center=(W / 2, H / 2), new_center=(300.0, 300.0)):
    """Forward map of a rotation by deg about center, moved to new_center, and its Pillow data."""
    t = math.radians(deg)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    c, n = np.asarray(center), np.asarray(new_center)

    def forward(p):
        return (np.asarray(p, dtype=float)[..., :2] - c) @ r.T + n

    inv = r.T  # input = inv (output - n) + c
    off = c - inv @ n
    return forward, (inv[0, 0], inv[0, 1], off[0], inv[1, 0], inv[1, 1], off[1])


def test_boxes_in_image_coordinates(mp_face):
    lm = mp_face()  # eyes at (150, 200) and (250, 200), points from x 90 to 310 and y 80 to 360
    out = crops.make_crops(face_image(), lm)
    assert crops.CROP_VERSION == 1 and out.version == 1
    assert out.iod_px == pytest.approx(100)
    # Whole face: center of the point box (200, 220), side 1.25 x 280
    assert np.allclose(out.face.corners, [(25, 45), (375, 45), (375, 395), (25, 395)])
    assert np.allclose(out.undereye_right.corners, [(120, 200), (180, 200), (180, 260), (120, 260)])
    assert np.allclose(out.undereye_left.corners, [(220, 200), (280, 200), (280, 260), (220, 260)])
    for crop in (out.face, out.undereye_left, out.undereye_right):
        assert crop.image.shape == (224, 224, 3) and crop.image.dtype == np.uint8
    assert out.undereye_left.mirrored and not out.undereye_right.mirrored and not out.face.mirrored


def test_dots_land_where_the_corners_say(mp_face):
    out = crops.make_crops(face_image(), mp_face())
    want = to_crop(out.face.corners, BLUE_AT)
    assert dot_center(out.face.image, 2) == pytest.approx(want, abs=1.0)
    want = to_crop(out.undereye_right.corners, RED_AT)
    assert dot_center(out.undereye_right.image, 0) == pytest.approx(want, abs=1.0)


def test_person_left_under_eye_is_mirrored_so_the_nose_is_on_the_right(mp_face):
    out = crops.make_crops(face_image(), mp_face())
    red_x, _ = dot_center(out.undereye_right.image, 0)
    green_x, green_y = dot_center(out.undereye_left.image, 1)
    assert red_x > 160 and green_x > 160  # both nose-side dots on the right of the crop
    x, y = to_crop(out.undereye_left.corners, GREEN_AT)
    assert (green_x, green_y) == pytest.approx((224 - x, y), abs=1.0)
    assert green_x == pytest.approx(red_x, abs=1.0)


def test_crops_do_not_change_when_image_and_points_turn_15_degrees(mp_face):
    image, lm = face_image(), mp_face()
    forward, data = rotation(15)
    turned = np.array(Image.fromarray(image).transform((600, 600), Image.Transform.AFFINE, data, Image.Resampling.BICUBIC))
    lm_turned = lm.copy()
    lm_turned[:, :2] = forward(lm)
    a, b = crops.make_crops(image, lm), crops.make_crops(turned, lm_turned)
    for name in ("face", "undereye_left", "undereye_right"):
        ca, cb = getattr(a, name), getattr(b, name)
        assert np.allclose(forward(ca.corners), cb.corners, atol=1e-6)
        diff = np.abs(ca.image.astype(int) - cb.image.astype(int))
        assert diff.mean() < 3, name
    assert dot_center(a.face.image, 2) == pytest.approx(dot_center(b.face.image, 2), abs=1.0)
    assert dot_center(a.undereye_right.image, 0) == pytest.approx(dot_center(b.undereye_right.image, 0), abs=1.0)
    assert dot_center(a.undereye_left.image, 1) == pytest.approx(dot_center(b.undereye_left.image, 1), abs=1.0)


def test_corners_map_back_for_a_tilted_face(mp_face):
    forward, _ = rotation(-20)
    lm = mp_face()
    lm[:, :2] = forward(lm)
    out = crops.make_crops(np.zeros((600, 600, 3), dtype=np.uint8), lm)
    straight = crops.make_crops(np.zeros((500, 400, 3), dtype=np.uint8), mp_face())
    for name in ("face", "undereye_left", "undereye_right"):
        assert np.allclose(getattr(out, name).corners, forward(getattr(straight, name).corners), atol=1e-6)


def test_outside_the_image_is_black():
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    corners = crops.square(100, 100, 100)  # bottom-right quarter is outside
    out = crops.cut(image, corners, size=50)
    assert out[5, 5].min() == 200 and out[45, 45].max() == 0


def test_shrinking_filters_instead_of_aliasing():
    ys, xs = np.mgrid[0:672, 0:672]
    checker = (((xs + ys) % 2) * 255).astype(np.uint8)
    image = np.repeat(checker[..., None], 3, axis=2)
    out = crops.cut(image, crops.square(336, 336, 672))
    assert abs(out.mean() - 127.5) < 3 and out.std() < 10
