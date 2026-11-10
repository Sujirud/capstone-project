"""face_geometry on synthetic faces with known answers."""

import math

import numpy as np
import pytest

from facemeasure.face_geometry import (
    FrontalPoints,
    GeometryError,
    LevelFrame,
    catmull_rom,
    first_crossing,
    measure_jaw_nose,
)

CX, CY, A, B = 200.0, 240.0, 100.0, 130.0
EYE, SUB, MOUTH = -0.3, 0.15, 0.5  # levels as fractions of B below the ellipse center


def ellipse_face(n: int = 12, mouth: float = MOUTH) -> FrontalPoints:
    """Face edge on the ellipse ((x - CX) / A)^2 + ((y - CY) / B)^2 = 1, widest at y = CY."""
    phi = np.linspace(0.1 * np.pi, 0.95 * np.pi, n)
    left = np.column_stack([CX - A * np.sin(phi), CY - B * np.cos(phi)])
    right = np.column_stack([CX + A * np.sin(phi), CY - B * np.cos(phi)])
    return FrontalPoints(
        contour_l=left, contour_r=right,
        eye_l=(CX - 0.4 * A, CY + EYE * B), eye_r=(CX + 0.4 * A, CY + EYE * B),
        subnasale=(CX, CY + SUB * B),
        mouth_l=(CX - 0.3 * A, CY + mouth * B), mouth_r=(CX + 0.3 * A, CY + mouth * B),
        alar_l=(CX - 20.0, CY + 0.1 * B), alar_r=(CX + 22.0, CY + 0.1 * B),
        canthus_l=(CX - 18.0, CY + EYE * B), canthus_r=(CX + 18.0, CY + EYE * B),
    )


def rotate(p, degrees: float, center=(123.0, 45.0)) -> np.ndarray:
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    q = np.asarray(p, dtype=np.float64) - center
    return np.stack([q[..., 0] * c - q[..., 1] * s, q[..., 0] * s + q[..., 1] * c], axis=-1) + center


def mirrored(fp: FrontalPoints, width: float = 1000.0) -> FrontalPoints:
    """Flip x, then swap every left input with its right counterpart."""
    def flip(p):
        q = np.array(p, dtype=np.float64)
        q[..., 0] = width - q[..., 0]
        return q

    f = fp.map_points(flip)
    return FrontalPoints(
        contour_l=f.contour_r, contour_r=f.contour_l, eye_l=f.eye_r, eye_r=f.eye_l, subnasale=f.subnasale,
        mouth_l=f.mouth_r, mouth_r=f.mouth_l, alar_l=f.alar_r, alar_r=f.alar_l,
        canthus_l=f.canthus_r, canthus_r=f.canthus_l,
    )


def test_catmull_rom_passes_through_every_point():
    chain = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 3.0], [4.0, 6.0]])
    curve = catmull_rom(chain, samples_per_segment=8)
    assert len(curve) == 8 * 3 + 1
    np.testing.assert_allclose(curve[::8], chain)


def test_catmull_rom_of_points_on_a_line_stays_on_it():
    chain = np.column_stack([np.arange(5.0), 2 * np.arange(5.0) + 1])
    curve = catmull_rom(chain)
    np.testing.assert_allclose(curve[:, 1], 2 * curve[:, 0] + 1, atol=1e-12)


def test_first_crossing_is_the_first_from_the_top():
    curve = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 0.0], [3.0, 10.0]])
    assert first_crossing(curve, 5.0) == pytest.approx((0.5, 5.0))
    assert first_crossing(curve, 11.0) is None


def test_level_frame_round_trip_and_level_eyes():
    frame = LevelFrame.from_eyes((10.0, 20.0), (50.0, 35.0))
    p = np.array([[3.0, 4.0], [100.0, -7.0]])
    np.testing.assert_allclose(frame.to_image(frame.to_level(p)), p)
    eyes = frame.to_level(np.array([[10.0, 20.0], [50.0, 35.0]]))
    assert eyes[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert eyes[1, 1] == pytest.approx(0.0, abs=1e-12)


def test_ellipse_zy_is_at_the_widest_level():
    r = measure_jaw_nose(ellipse_face())
    for zy in (r.zy_l, r.zy_r):
        assert abs(zy[1] - CY) < 0.03 * B
    assert r.zy_l[0] == pytest.approx(CX - A, abs=0.5)
    assert r.zy_r[0] == pytest.approx(CX + A, abs=0.5)


@pytest.mark.parametrize("mouth", [0.3, 0.5, 0.7])
def test_ellipse_jaw_ratio_matches_the_formula(mouth):
    r = measure_jaw_nose(ellipse_face(mouth=mouth))
    assert r.jaw_ratio == pytest.approx(math.sqrt(1 - mouth**2), abs=0.01)
    for go in (r.go_l, r.go_r):
        assert go[1] == pytest.approx(CY + mouth * B, abs=1e-9)


def test_nose_ratio_and_iod():
    r = measure_jaw_nose(ellipse_face())
    assert r.nose_ratio == pytest.approx(42.0 / 36.0)
    assert r.iod_px == pytest.approx(0.8 * A)


def test_rotation_by_7_degrees_keeps_the_ratios():
    fp = ellipse_face()
    base = measure_jaw_nose(fp)
    turned = measure_jaw_nose(fp.map_points(lambda p: rotate(p, 7.0)))
    assert turned.jaw_ratio == pytest.approx(base.jaw_ratio, abs=1e-9)
    assert turned.nose_ratio == pytest.approx(base.nose_ratio, abs=1e-9)
    for name in ("go_l", "go_r", "zy_l", "zy_r"):
        np.testing.assert_allclose(getattr(turned, name), rotate(getattr(base, name), 7.0), atol=1e-6)


def test_mirroring_and_swapping_sides_keeps_the_ratios():
    fp = ellipse_face(mouth=0.45).map_points(lambda p: rotate(p, -4.0))  # break the symmetry a little
    fp = FrontalPoints(**{**fp.__dict__, "alar_r": fp.alar_r + (3.0, 1.0)})
    base = measure_jaw_nose(fp)
    flipped = measure_jaw_nose(mirrored(fp))
    assert flipped.jaw_ratio == pytest.approx(base.jaw_ratio, abs=1e-9)
    assert flipped.nose_ratio == pytest.approx(base.nose_ratio, abs=1e-9)
    np.testing.assert_allclose(flipped.go_l, (1000.0 - base.go_r[0], base.go_r[1]), atol=1e-6)
    np.testing.assert_allclose(flipped.zy_r, (1000.0 - base.zy_l[0], base.zy_l[1]), atol=1e-6)


def test_eye_centers_out_of_order_raise():
    fp = ellipse_face()
    swapped = FrontalPoints(**{**fp.__dict__, "eye_l": fp.eye_r, "eye_r": fp.eye_l})
    with pytest.raises(GeometryError, match="eye centers"):
        measure_jaw_nose(swapped)


def test_face_edge_above_mouth_level_raises():
    fp = ellipse_face()
    short = FrontalPoints(**{**fp.__dict__, "contour_l": fp.contour_l[:7]})  # ends between subnasale and mouth
    with pytest.raises(GeometryError, match="image-left face edge does not reach mouth level"):
        measure_jaw_nose(short)


def test_face_edge_outside_the_zy_band_raises():
    fp = ellipse_face()
    low = FrontalPoints(**{**fp.__dict__, "contour_r": fp.contour_r[-3:]})
    with pytest.raises(GeometryError, match="image-right face edge has no point"):
        measure_jaw_nose(low)


def test_zero_intercanthal_distance_raises():
    fp = ellipse_face()
    same = FrontalPoints(**{**fp.__dict__, "canthus_r": fp.canthus_l})
    with pytest.raises(GeometryError, match="inner canthal"):
        measure_jaw_nose(same)


@pytest.mark.parametrize(
    "field, value",
    [("contour_l", np.zeros((1, 2))), ("contour_r", np.zeros((5, 3))), ("eye_l", (1.0, 2.0, 3.0)), ("subnasale", (np.nan, 1.0))],
)
def test_bad_inputs_raise(field, value):
    fp = ellipse_face()
    with pytest.raises(ValueError, match=field):
        FrontalPoints(**{**fp.__dict__, field: value})
