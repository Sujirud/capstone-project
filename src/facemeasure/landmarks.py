"""MediaPipe Face Landmarker and selfie multiclass Image Segmenter (api.md sections 3 and 4).

Both run in IMAGE mode. The landmarker looks for up to two faces and returns blendshapes
and the facial transformation matrix with the default confidences; the segmenter returns
both the category mask and the confidence masks, because analyze_profile uses the confidences.

The models are read with pathlib and passed as model_asset_buffer, never as
model_asset_path, because a path with Thai letters (a Windows user name, say) may not
open. Build FaceModels once and reuse it; close it, or use it as a context manager.
"""

from dataclasses import dataclass
from typing import Self

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from facemeasure import assets, mp_points

LANDMARKER_FILE = "face_landmarker.task"
SEGMENTER_FILE = "selfie_multiclass_256x256.tflite"
MAX_FACES = 2

# Classes of the selfie multiclass mask
SEGMENT_CLASSES = ("background", "hair", "body_skin", "face_skin", "clothes", "others")
HAIR = SEGMENT_CLASSES.index("hair")


@dataclass(frozen=True)
class Face:
    """One face in pixels of the upright image: x * width, y * height and z * width."""

    landmarks: np.ndarray  # (478, 3) float64
    blendshapes: dict[str, float]  # MediaPipe category name -> score in [0, 1]
    matrix: np.ndarray  # (4, 4) float64 facial transformation matrix, rows as returned
    iod_px: float


@dataclass(frozen=True)
class Segmentation:
    category: np.ndarray  # (h, w) uint8 class index
    confidence: np.ndarray  # (6, h, w) float32, one mask per class


def check_rgb(image: np.ndarray) -> np.ndarray:
    """The image as a C-contiguous (h, w, 3) uint8 array, or ValueError."""
    a = np.asarray(image)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"image must be an (h, w, 3) uint8 RGB array, got {a.dtype} {a.shape}")
    return np.ascontiguousarray(a)


def largest_first(faces: list[Face]) -> list[Face]:
    """Faces from the largest IOD to the smallest; the first one is measured."""
    return sorted(faces, key=lambda f: -f.iod_px)


def _mp_image(image: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=check_rgb(image))


class FaceModels:
    """The landmarker and the segmenter, created once.

    Without arguments the model files come from api/assets/ after their sha256 is checked
    against api/assets.lock.json; pass the bytes yourself to use other copies.
    """

    def __init__(self, landmarker_model: bytes | None = None, segmenter_model: bytes | None = None):
        if landmarker_model is None:
            landmarker_model = assets.read_verified(LANDMARKER_FILE)
        if segmenter_model is None:
            segmenter_model = assets.read_verified(SEGMENTER_FILE)
        self._landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=landmarker_model),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=MAX_FACES,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        try:
            self._segmenter = vision.ImageSegmenter.create_from_options(
                vision.ImageSegmenterOptions(
                    base_options=BaseOptions(model_asset_buffer=segmenter_model),
                    running_mode=vision.RunningMode.IMAGE,
                    output_category_mask=True,
                    output_confidence_masks=True,
                )
            )
        except Exception:
            self._landmarker.close()
            raise

    def detect(self, image: np.ndarray) -> list[Face]:
        """Faces in the image, from the largest IOD to the smallest."""
        h, w = image.shape[:2]
        result = self._landmarker.detect(_mp_image(image))
        faces = []
        for k, points in enumerate(result.face_landmarks):
            lm = np.array([(p.x * w, p.y * h, p.z * w) for p in points], dtype=np.float64)
            if lm.shape != (mp_points.N_POINTS, 3):
                raise RuntimeError(f"Face Landmarker returned {len(lm)} points, expected {mp_points.N_POINTS}")
            faces.append(
                Face(
                    landmarks=lm,
                    blendshapes={c.category_name: float(c.score) for c in result.face_blendshapes[k]},
                    matrix=np.array(result.facial_transformation_matrixes[k], dtype=np.float64).reshape(4, 4),
                    iod_px=mp_points.iod(lm),
                )
            )
        return largest_first(faces)

    def segment(self, image: np.ndarray) -> Segmentation:
        """Category and confidence masks at the size of the image."""
        h, w = image.shape[:2]
        result = self._segmenter.segment(_mp_image(image))
        category = np.array(result.category_mask.numpy_view(), dtype=np.uint8).reshape(h, w)
        confidence = np.stack([np.array(m.numpy_view(), dtype=np.float32).reshape(h, w) for m in result.confidence_masks])
        if confidence.shape[0] != len(SEGMENT_CLASSES):
            raise RuntimeError(f"Image Segmenter returned {confidence.shape[0]} classes, expected {len(SEGMENT_CLASSES)}")
        return Segmentation(category=category, confidence=confidence)

    def close(self) -> None:
        self._landmarker.close()
        self._segmenter.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
