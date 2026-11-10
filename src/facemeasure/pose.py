"""Head pose from the MediaPipe facial transformation matrix (api.md section 4).

The web computes the same angles with the same formula; shared/testdata/pose_cases.json
holds the cases both sides test against.

MediaPipe axes: x points to the image right, y up, and a face that looks at the camera
faces +z. With R the 3x3 part of the matrix made orthonormal by SVD:

- f = R (0, 0, 1) is the direction the face points to,
- r = R (1, 0, 0) is the direction from the person's right side to their left side,
- yaw = atan2(fx, fz), positive when the person turns to their own left,
- pitch = atan2(fy, sqrt(fx^2 + fz^2)), positive when they look up,
- roll = atan2(ry, rx), positive when their left eye is higher than their right eye in the image.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


def rotation(matrix) -> np.ndarray:
    """Nearest rotation to the 3x3 part of a 4x4 (or 3x3) matrix, which may carry scale."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape not in ((4, 4), (3, 3)) or not np.isfinite(m).all():
        raise ValueError(f"matrix must be a finite 4x4 or 3x3 array, got shape {m.shape}")
    u, _, vt = np.linalg.svd(m[:3, :3])
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt


def angles(matrix) -> Pose:
    """Yaw, pitch and roll in degrees from a facial transformation matrix (rows as in numpy)."""
    r = rotation(matrix)
    fx, fy, fz = r[:, 2]
    rx, ry, _ = r[:, 0]
    return Pose(
        yaw_deg=math.degrees(math.atan2(fx, fz)),
        pitch_deg=math.degrees(math.atan2(fy, math.hypot(fx, fz))),
        roll_deg=math.degrees(math.atan2(ry, rx)),
    )
