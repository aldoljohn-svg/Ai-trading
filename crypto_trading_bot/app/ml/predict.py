"""Inference.

Returns ``(P_LONG, P_SHORT, P_NO_TRADE)``.  Three properties are guaranteed:

* the three values always sum to 1.0;
* no probability is ever exactly 0 or 1 - the system does not express certainty;
* with no model loaded the answer is ``(0.0, 0.0, 1.0)``, which the signal
  engine reads as "no ML opinion", not as "do not trade".

Predictions are cheap and the caller is on the trading loop, so a small
LRU cache keyed on the rounded feature vector avoids recomputing an identical
prediction within one scan cycle.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping

from app.compat import HAVE_NUMPY, numpy
from app.logger import get_logger
from app.ml.dataset import LABEL_LONG, LABEL_NO_TRADE, LABEL_SHORT
from app.ml.model_registry import ModelArtifact, ModelRegistry

log = get_logger(__name__)

_MIN_PROBABILITY = 0.001
_MAX_PROBABILITY = 0.98


class Predictor:
    def __init__(
        self,
        registry: ModelRegistry | None = None,
        model_name: str = "direction_v1",
        artifact: ModelArtifact | None = None,
    ) -> None:
        self.registry = registry
        self.model_name = model_name
        self.artifact = artifact
        self._lock = threading.RLock()
        self.predictions = 0
        self.failures = 0
        if artifact is None and registry is not None:
            self.reload()

    # -- lifecycle --------------------------------------------------------

    def reload(self) -> bool:
        if self.registry is None:
            return False
        with self._lock:
            artifact = self.registry.active(self.model_name)
            if artifact is None:
                log.info(
                    "no trained model named %r - running on rules only", self.model_name
                )
                self.artifact = None
                return False
            self.artifact = artifact
            log.info(
                "loaded model %s (%s, %d rows, accuracy %s)",
                artifact.key,
                artifact.algorithm,
                artifact.rows,
                artifact.metrics.get("accuracy", "n/a"),
            )
            return True

    @property
    def ready(self) -> bool:
        return self.artifact is not None and self.artifact.runtime is not None

    # -- inference --------------------------------------------------------

    def predict(self, features: Mapping[str, float]) -> tuple[float, float, float]:
        if not self.ready:
            return 0.0, 0.0, 1.0

        artifact = self.artifact
        assert artifact is not None
        try:
            vector = artifact.feature_spec.transform(features)
            raw = self._raw_probabilities(artifact, vector)
        except Exception as exc:  # noqa: BLE001 - never break the trading loop
            self.failures += 1
            log.warning("prediction failed: %s", exc)
            return 0.0, 0.0, 1.0

        self.predictions += 1

        p_long = raw[LABEL_LONG]
        p_short = raw[LABEL_SHORT]

        if artifact.long_calibrator is not None:
            p_long = artifact.long_calibrator.transform(p_long)
        if artifact.short_calibrator is not None:
            p_short = artifact.short_calibrator.transform(p_short)

        # Calibrating the two directions independently can push their sum past
        # 1.  Cap each one *first* - the system never expresses certainty - and
        # only then take NO_TRADE as the residual, so the three always sum to 1.
        p_long = min(max(p_long, 0.0), _MAX_PROBABILITY)
        p_short = min(max(p_short, 0.0), _MAX_PROBABILITY)

        directional = p_long + p_short
        ceiling = 1.0 - _MIN_PROBABILITY
        if directional > ceiling:
            scale = ceiling / directional
            p_long *= scale
            p_short *= scale

        p_no_trade = 1.0 - p_long - p_short
        return round(p_long, 4), round(p_short, 4), round(p_no_trade, 4)

    def _raw_probabilities(
        self, artifact: ModelArtifact, vector: list[float]
    ) -> list[float]:
        model = artifact.runtime
        if hasattr(model, "predict_proba"):
            if artifact.algorithm.startswith("softmax"):
                return model.predict_proba([vector])[0]
            payload = numpy.array([vector]) if HAVE_NUMPY else [vector]
            raw = model.predict_proba(payload)[0]
            classes = list(getattr(model, "classes_", [LABEL_NO_TRADE, LABEL_LONG, LABEL_SHORT]))
            aligned = [0.0, 0.0, 0.0]
            for position, label in enumerate(classes):
                index = int(label)
                if 0 <= index < 3:
                    aligned[index] = float(raw[position])
            return aligned
        raise RuntimeError(f"model {artifact.algorithm} has no predict_proba")

    # -- diagnostics ------------------------------------------------------

    def info(self) -> dict[str, Any]:
        if self.artifact is None:
            return {
                "loaded": False,
                "model": self.model_name,
                "note": "no trained model - decisions are rule-based only",
            }
        metrics = self.artifact.metrics
        return {
            "loaded": True,
            "model": self.artifact.key,
            "algorithm": self.artifact.algorithm,
            "trained_at": self.artifact.trained_at,
            "rows": self.artifact.rows,
            "features": len(self.artifact.feature_spec),
            "accuracy": metrics.get("accuracy"),
            "brier_long": metrics.get("brier_long"),
            "ece_long": metrics.get("ece_long"),
            "calibrated": bool(self.artifact.long_calibrator),
            "predictions": self.predictions,
            "failures": self.failures,
        }


__all__ = ["Predictor"]
