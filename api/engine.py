"""Everything the api loads once, and the steps of api.md section 3.

Engine.load() reads shared/quality.json and the language files, checks every file of
api/assets/ against api/assets.lock.json without downloading anything, loads the
bundle with load_bundle (which checks the sha256 of its files and crop_version), builds
the MediaPipe landmarker and segmenter, loads DINOv2, and runs one synthetic request so
the first real one is not slower than the rest. Anything missing stops it with a
StartupError whose message names the command to run.

Engine.analyze() takes the decoded images and the checked meta:

1. the frontal image through model_input.analyze_for_model, which shrinks a face with a
   larger IOD than the top of the bundle's train_face_iod_px (nothing is shrunk when the
   bundle has no range);
2. the profile image through profile.analyze_profile; without one, the chin and the
   nasal dorsum get missing_input;
3. with a face, DINOv2 on the whole-face crop and both under-eye crops in one batch;
   each head takes its own feature type and the formulas of bundle.py;
4. response.build in the language asked for.

Nothing is kept between requests: the images, crops and features are local to
analyze and go when it returns. The api calls analyze under one lock.
"""

import warnings
from dataclasses import dataclass

import numpy as np

import facemeasure
from facemeasure import assets, bundle, codes, crops, quality, response
from facemeasure.bundle import UNDEREYE_HEAD, Bundle, BundleError, BundleWarning
from facemeasure.frontal import FrontalResult
from facemeasure.i18n import TextError, Texts
from facemeasure.model_input import analyze_for_model
from facemeasure.profile import ProfileResult, analyze_profile
from facemeasure.quality import QualityConfig, QualityConfigError
from facemeasure.response import Meta, Predictions

ASSETS_COMMAND = "uv run python -m facemeasure.assets"
# Rows of the one DINOv2 batch
FACE_ROW, UNDEREYE_LEFT_ROW, UNDEREYE_RIGHT_ROW = 0, 1, 2
WARM_UP_SIZE = (640, 480)  # width, height of the synthetic warm-up image
WARM_UP_SEED = 0


class StartupError(RuntimeError):
    """Something the api needs is missing or does not match; the message says what to run or copy."""


@dataclass(frozen=True)
class Measured:
    """What the images gave, before the rules: the input of response.build."""

    frontal: FrontalResult
    profile: ProfileResult
    predictions: Predictions


@dataclass(frozen=True)
class Parts:
    """The files the api reads at start, checked, before any model is built."""

    config: QualityConfig
    texts: Texts
    bundle: Bundle
    warnings: tuple[str, ...]  # differences between the bundle and this machine (BundleWarning)


def load_parts() -> Parts:
    """shared/quality.json, the language files, api/assets/ against the lock, and api/bundle/."""
    try:
        config = quality.load_config()
    except (OSError, ValueError, QualityConfigError) as e:
        raise StartupError(f"cannot read shared/{quality.QUALITY_FILE}: {e}") from None
    try:
        texts = Texts()
    except (OSError, ValueError, TextError) as e:
        raise StartupError(f"cannot read the language files in shared/i18n/: {e}") from None
    try:
        assets.verify()
    except assets.AssetMismatch as e:
        raise StartupError(
            f"{e}; the files in api/assets/ must be the ones of api/assets.lock.json: copy the whole folder "
            "api/assets/ from the machine that made the lock"
        ) from None
    except (OSError, ValueError, assets.AssetError) as e:
        raise StartupError(f"{e}" if ASSETS_COMMAND in str(e) else f"{e}; run: {ASSETS_COMMAND}") from None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", BundleWarning)
        try:
            b = bundle.load_bundle()
        except BundleError as e:
            raise StartupError(f"cannot load api/bundle/: {e}") from None
    found = tuple(str(w.message) for w in caught if issubclass(w.category, BundleWarning))
    return Parts(config=config, texts=texts, bundle=b, warnings=found)


def check_features(b: Bundle, available: dict[str, int]) -> None:
    """Every head's feature type must be one the backbone gives, with the same number of dimensions."""
    for name in bundle.HEADS:
        feature = b.heads[name].feature
        dims = b.manifest["backbone"]["features"][feature]
        if available.get(feature) != dims:
            raise StartupError(
                f"the {name} head of bundle {b.version} takes {feature} with {dims} dimensions, but DINOv2 gives "
                f"{available.get(feature)}; write the bundle again with this code"
            )


class Engine:
    """The quality limits, texts, bundle and models of one api process."""

    def __init__(self, parts: Parts, models, backbone, device: str):
        self.config = parts.config
        self.texts = parts.texts
        self.bundle = parts.bundle
        self.warnings = parts.warnings
        self.models = models
        self.backbone = backbone
        self.device = device
        iod = self.bundle.train_face_iod_px
        self.max_iod_px = None if iod is None else float(iod["max"])

    @classmethod
    def load(cls, warm_up: bool = True) -> "Engine":
        """Load everything once (api.md section 5); raises StartupError."""
        parts = load_parts()
        # Imported here so that importing the api, and the tests with a fake engine, load no model
        from facemeasure.backbone import FEATURES, Backbone, BackboneError
        from facemeasure.landmarks import FaceModels

        check_features(parts.bundle, FEATURES)
        try:
            models = FaceModels()
        except (OSError, RuntimeError, ValueError, assets.AssetError) as e:
            raise StartupError(f"cannot build the MediaPipe models from api/assets/: {e}") from None
        try:
            backbone = Backbone()
        except (OSError, RuntimeError, ValueError, assets.AssetError, BackboneError) as e:
            models.close()
            raise StartupError(f"cannot load DINOv2 from api/assets/: {e}; run: {ASSETS_COMMAND}") from None
        engine = cls(parts, models, backbone, str(backbone.device))
        if warm_up:
            try:
                engine.warm_up()
            except Exception as e:
                engine.close()
                raise StartupError(f"the warm-up request failed: {type(e).__name__}") from e
        return engine

    # ------------------------------------------------------------ requests

    def measure(self, frontal: np.ndarray, profile: np.ndarray | None) -> Measured:
        """Steps 1-3: points, quality, edge and head predictions of one request."""
        f = analyze_for_model(frontal, self.models, self.max_iod_px, self.config.frontal)
        p = analyze_profile(profile, self.models, self.config.profile)
        return Measured(frontal=f, profile=p, predictions=self.predict(f))

    def predict(self, frontal: FrontalResult) -> Predictions:
        """DINOv2 on the three crops in one batch, then each head; nothing without a face."""
        if frontal.crops is None:
            return Predictions()
        c = frontal.crops
        feats = self.backbone.features(np.stack([c.face.image, c.undereye_left.image, c.undereye_right.image]))
        heads = self.bundle.heads
        jaw, nose, eye = heads[codes.JAW], heads[codes.NOSE_BASE], heads[UNDEREYE_HEAD]
        p = eye.predict(feats[eye.feature][[UNDEREYE_LEFT_ROW, UNDEREYE_RIGHT_ROW]])
        return Predictions(
            jaw=float(jaw.predict(feats[jaw.feature][FACE_ROW])),
            nose_base=float(nose.predict(feats[nose.feature][FACE_ROW])),
            undereye_left=(float(p[0, 0]), float(p[0, 1])),
            undereye_right=(float(p[1, 0]), float(p[1, 1])),
        )

    def analyze(self, frontal: np.ndarray, profile: np.ndarray | None, meta: Meta, lab: bool) -> dict:
        """quality, retake, regions and disclaimer of api.md section 2.2 (the api adds request_id, versions and lab)."""
        m = self.measure(frontal, profile)
        return response.build(m.frontal, m.profile, m.predictions, meta, self.bundle, self.texts, lab=lab)

    def warm_up(self) -> None:
        """One synthetic request, then DINOv2 on three blank crops, since noise has no face."""
        w, h = WARM_UP_SIZE
        image = np.random.default_rng(WARM_UP_SEED).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        self.analyze(image, image, Meta(age=30, sex=codes.FEMALE), lab=True)
        self.backbone.features(np.zeros((3, crops.CROP_SIZE, crops.CROP_SIZE, 3), dtype=np.uint8))

    # ------------------------------------------------------------ health

    @property
    def stub_heads(self) -> list[str]:
        return list(self.bundle.stub_heads)

    @property
    def languages(self) -> list[str]:
        return list(self.texts.codes)

    def versions(self) -> dict[str, str]:
        """Versions of /health except the api's own."""
        runtime = bundle.runtime_versions()
        return {
            "model_bundle": self.bundle.version,
            "facemeasure": facemeasure.__version__,
            "mediapipe": runtime["mediapipe"],
            "torch": runtime["torch"],
        }

    def close(self) -> None:
        if self.models is not None:
            self.models.close()
            self.models = None
        self.backbone = None
