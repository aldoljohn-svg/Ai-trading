"""Machine-learning layer.

The model is a *contributor to confidence*, never the decision maker.  It
estimates ``P_LONG`` / ``P_SHORT`` / ``P_NO_TRADE``, those probabilities are
calibrated so that "70%" means something close to 70% empirically, and the
signal engine blends them with the rule-based score at a bounded weight.

With no trained model present the predictor reports ``(0, 0, 1)`` - an explicit
"I do not know" - rather than a fabricated 50/50.

scikit-learn / LightGBM are used when installed; otherwise an equivalent
pure-Python softmax regression is trained.  Both paths produce the same
artefact format and the same calibrated interface.
"""

from app.ml.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
)
from app.ml.dataset import LabelledSample, build_dataset, triple_barrier_labels
from app.ml.features import FeatureSpec, vectorise
from app.ml.model_registry import ModelArtifact, ModelRegistry
from app.ml.predict import Predictor
from app.ml.train import TrainingResult, train_model

__all__ = [
    "FeatureSpec",
    "vectorise",
    "LabelledSample",
    "build_dataset",
    "triple_barrier_labels",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "brier_score",
    "expected_calibration_error",
    "ModelRegistry",
    "ModelArtifact",
    "Predictor",
    "train_model",
    "TrainingResult",
]
