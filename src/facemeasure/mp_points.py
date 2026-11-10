"""MediaPipe Face Landmarker point numbers, kept in this one file.

MediaPipe numbers its 478 points 0-477 and the code uses those numbers as they are.
Sides: in an image that is not mirrored the person's right eye (points 33 and 133)
is on the image left. Names ending in _l and _r here are image sides, as in the
overlay names of api.md section 2.2.
"""

import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarksConnections

from facemeasure.face_geometry import FrontalPoints

N_POINTS = 478

# Eye centers are the midpoint of the two corners of each eye, as in the MediaPipe
# model card. They give the IOD, the rotation that levels the eyes, and the jaw rule.
EYE_PERSON_RIGHT = (33, 133)  # outer and inner corner; image left
EYE_PERSON_LEFT = (362, 263)  # inner and outer corner; image right

NOSE_TIP = 1
SUBNASALE = 2
MOUTH_CORNERS = (61, 291)  # image left, image right
INNER_CANTHI = (133, 362)  # image left, image right
FOREHEAD = 10  # top of the face edge, middle of the forehead
CHIN = 152  # bottom of the face edge
CHEEKS = (234, 454)  # face edge near eye level: image left, image right

# Alar base candidates (image left, image right), compared by frontal_check.py. On the SCUT-FBP5500
# train split all seven pairs correlate alike with the alar base label (r 0.318-0.334;
# model/reports/frontal_mediapipe.md), so the pair is the one at the definition of
# model.md section 3.4: the outermost point of the alar base where it meets the lip.
ALAR_BASE_PAIRS = ((98, 327), (64, 294), (129, 358), (102, 331), (48, 278), (240, 460), (235, 455))
ALAR_BASE: tuple[int, int] | None = (98, 327)


def pair_key(pair: tuple[int, int]) -> str:
    """Column-friendly name of a point pair, such as 98_327."""
    return f"{pair[0]}_{pair[1]}"


def _face_oval_chains() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split the FACE_OVAL loop into two chains from FOREHEAD to CHIN: (via CHEEKS[0], via CHEEKS[1])."""
    edges = [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL]
    following = dict(edges)
    if len(following) != len(edges) or FOREHEAD not in following:
        raise RuntimeError("FACE_LANDMARKS_FACE_OVAL is not one loop through point 10; mediapipe has changed")
    loop = [FOREHEAD]
    while (nxt := following[loop[-1]]) != FOREHEAD:
        loop.append(nxt)
        if len(loop) > len(edges):
            raise RuntimeError("FACE_LANDMARKS_FACE_OVAL does not close; mediapipe has changed")
    if len(loop) != len(edges) or CHIN not in loop:
        raise RuntimeError("FACE_LANDMARKS_FACE_OVAL does not pass through point 152 once; mediapipe has changed")
    k = loop.index(CHIN)
    one = tuple(loop[: k + 1])
    other = tuple([FOREHEAD] + loop[k:][::-1])
    if CHEEKS[0] in one and CHEEKS[1] in other:
        return one, other
    if CHEEKS[0] in other and CHEEKS[1] in one:
        return other, one
    raise RuntimeError("FACE_LANDMARKS_FACE_OVAL halves do not pass through 234 and 454; mediapipe has changed")


# Face edges from 10 (forehead) to 152 (chin), 19 points each, on the image left and right
CONTOUR_IMAGE_LEFT, CONTOUR_IMAGE_RIGHT = _face_oval_chains()


def xy(landmarks: np.ndarray, numbers) -> np.ndarray:
    """(x, y) of point numbers: one number gives (2,), a sequence (n, 2)."""
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.ndim != 2 or lm.shape[0] != N_POINTS or lm.shape[1] < 2:
        raise ValueError(f"landmarks must be a ({N_POINTS}, 2) or ({N_POINTS}, 3) array, got shape {lm.shape}")
    return lm[np.asarray(numbers), :2]


def eye_centers(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eye centers (image-left eye, image-right eye) = (person's right, person's left)."""
    return xy(landmarks, EYE_PERSON_RIGHT).mean(axis=0), xy(landmarks, EYE_PERSON_LEFT).mean(axis=0)


def iod(landmarks: np.ndarray) -> float:
    """Distance in pixels between the two eye centers."""
    left, right = eye_centers(landmarks)
    return float(np.hypot(*(right - left)))


def frontal_points(landmarks: np.ndarray, alar_pair: tuple[int, int]) -> FrontalPoints:
    """The generic input of face_geometry from the 478 MediaPipe points, with one alar base pair."""
    eye_l, eye_r = eye_centers(landmarks)
    return FrontalPoints(
        contour_l=xy(landmarks, CONTOUR_IMAGE_LEFT),
        contour_r=xy(landmarks, CONTOUR_IMAGE_RIGHT),
        eye_l=eye_l,
        eye_r=eye_r,
        subnasale=xy(landmarks, SUBNASALE),
        mouth_l=xy(landmarks, MOUTH_CORNERS[0]),
        mouth_r=xy(landmarks, MOUTH_CORNERS[1]),
        alar_l=xy(landmarks, alar_pair[0]),
        alar_r=xy(landmarks, alar_pair[1]),
        canthus_l=xy(landmarks, INNER_CANTHI[0]),
        canthus_r=xy(landmarks, INNER_CANTHI[1]),
    )
