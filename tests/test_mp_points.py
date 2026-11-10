"""MediaPipe point numbers and face edges."""

import numpy as np
import pytest

from facemeasure import mp_points


@pytest.mark.parametrize("chain", [mp_points.CONTOUR_IMAGE_LEFT, mp_points.CONTOUR_IMAGE_RIGHT])
def test_face_edges_run_from_forehead_to_chin(chain):
    assert chain[0] == 10 and chain[-1] == 152
    assert len(chain) == 19 and len(set(chain)) == 19


def test_edge_through_234_is_the_image_left_one():
    assert 234 in mp_points.CONTOUR_IMAGE_LEFT and 454 not in mp_points.CONTOUR_IMAGE_LEFT
    assert 454 in mp_points.CONTOUR_IMAGE_RIGHT and 234 not in mp_points.CONTOUR_IMAGE_RIGHT
    # Together the two edges are the 36-point FACE_OVAL loop
    assert len(set(mp_points.CONTOUR_IMAGE_LEFT) | set(mp_points.CONTOUR_IMAGE_RIGHT)) == 36


def test_eye_centers_and_iod(mp_face):
    lm = mp_face(cx=200, iod=100, eye_y=180)
    left, right = mp_points.eye_centers(lm)
    assert np.allclose(left, (150, 180)) and np.allclose(right, (250, 180))
    assert mp_points.iod(lm) == pytest.approx(100)


def test_frontal_points_use_image_sides(mp_face):
    lm = mp_face()
    fp = mp_points.frontal_points(lm, mp_points.ALAR_BASE_PAIRS[0])
    assert fp.eye_l[0] < fp.eye_r[0]
    assert fp.mouth_l[0] < fp.mouth_r[0] and fp.canthus_l[0] < fp.canthus_r[0] and fp.alar_l[0] < fp.alar_r[0]
    assert np.allclose(fp.canthus_l, lm[133, :2]) and np.allclose(fp.canthus_r, lm[362, :2])
    assert len(fp.contour_l) == 19 and fp.contour_l[:, 0].mean() < fp.contour_r[:, 0].mean()


def test_pairs_and_keys():
    assert len(mp_points.ALAR_BASE_PAIRS) == 7
    assert mp_points.pair_key((98, 327)) == "98_327"
    assert mp_points.ALAR_BASE is None or mp_points.ALAR_BASE in mp_points.ALAR_BASE_PAIRS


def test_bad_landmarks():
    with pytest.raises(ValueError):
        mp_points.xy(np.zeros((10, 3)), 1)
