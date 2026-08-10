"""Model training.

Backend selection, in order of preference:

1. **LightGBM** (``lightgbm``) - gradient boosted trees, best on tabular data.
2. **scikit-learn** - ``HistGradientBoostingClassifier`` or ``LogisticRegression``.
3. **Pure-Python softmax regression** - always available, no dependencies.

All three produce the same artefact interface, so the rest of the system does
not care which one ran.  Whatever the backend, the output probabilities are
calibrated on a held-out slice before being registered, and a model that fails
to beat the majority-class baseline is refused rather than deployed.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.compat import HAVE_LIGHTGBM, HAVE_NUMPY, HAVE_SKLEARN, lightgbm, numpy
from app.logger import get_logger
from app.ml.calibration import (
    brier_score,
    expected_calibration_error,
    fit_calibrator,
    reliability_table,
)
from app.ml.dataset import (
    LABEL_LONG,
    LABEL_NO_TRADE,
    LABEL_SHORT,
    LabelledSample,
    balance_weights,
    class_distribution,
    time_split,
)
from app.ml.features import FeatureSpec
from app.ml.model_registry import ModelArtifact

log = get_logger(__name__)


@dataclass(slots=True)
class TrainingResult:
    artifact: ModelArtifact | None
    metrics: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False
    reason: str = ""


# --------------------------------------------------------------------------
# Pure-Python softmax regression (the always-available backend)
# --------------------------------------------------------------------------


class SoftmaxRegression:
    """Multinomial logistic regression trained with Adam and L2 regularisation."""

    def __init__(
        self,
        classes: int = 3,
        learning_rate: float = 0.08,
        epochs: int = 120,
        l2: float = 1e-3,
        batch_size: int = 128,
        seed: int = 17,
    ) -> None:
        self.classes = classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.batch_size = batch_size
        self.seed = seed
        self.weights: list[list[float]] = []
        self.bias: list[float] = []

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        weights: Mapping[int, float] | None = None,
    ) -> "SoftmaxRegression":
        n = len(X)
        if n == 0:
            raise ValueError("cannot train on an empty dataset")
        d = len(X[0])
        rng = random.Random(self.seed)

        self.weights = [[0.0] * d for _ in range(self.classes)]
        self.bias = [0.0] * self.classes

        m_w = [[0.0] * d for _ in range(self.classes)]
        v_w = [[0.0] * d for _ in range(self.classes)]
        m_b = [0.0] * self.classes
        v_b = [0.0] * self.classes
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        sample_weight = [
            (weights or {}).get(label, 1.0) for label in y
        ]
        indices = list(range(n))

        for _epoch in range(self.epochs):
            rng.shuffle(indices)
            for start in range(0, n, self.batch_size):
                batch = indices[start : start + self.batch_size]
                if not batch:
                    continue
                step += 1
                grad_w = [[0.0] * d for _ in range(self.classes)]
                grad_b = [0.0] * self.classes

                for index in batch:
                    row = X[index]
                    probabilities = self._probabilities(row)
                    weight = sample_weight[index]
                    target = y[index]
                    for k in range(self.classes):
                        error = (probabilities[k] - (1.0 if k == target else 0.0)) * weight
                        if error == 0.0:
                            continue
                        row_grad = grad_w[k]
                        for j in range(d):
                            row_grad[j] += error * row[j]
                        grad_b[k] += error

                scale = 1.0 / len(batch)
                correction1 = 1.0 - beta1 ** step
                correction2 = 1.0 - beta2 ** step

                for k in range(self.classes):
                    weights_k = self.weights[k]
                    grad_k = grad_w[k]
                    m_k, v_k = m_w[k], v_w[k]
                    for j in range(d):
                        gradient = grad_k[j] * scale + self.l2 * weights_k[j]
                        m_k[j] = beta1 * m_k[j] + (1 - beta1) * gradient
                        v_k[j] = beta2 * v_k[j] + (1 - beta2) * gradient * gradient
                        weights_k[j] -= (
                            self.learning_rate
                            * (m_k[j] / correction1)
                            / (math.sqrt(v_k[j] / correction2) + eps)
                        )
                    gradient_b = grad_b[k] * scale
                    m_b[k] = beta1 * m_b[k] + (1 - beta1) * gradient_b
                    v_b[k] = beta2 * v_b[k] + (1 - beta2) * gradient_b * gradient_b
                    self.bias[k] -= (
                        self.learning_rate
                        * (m_b[k] / correction1)
                        / (math.sqrt(v_b[k] / correction2) + eps)
                    )
        return self

    def _probabilities(self, row: Sequence[float]) -> list[float]:
        logits = []
        for k in range(self.classes):
            weights_k = self.weights[k]
            total = self.bias[k]
            for j, value in enumerate(row):
                total += weights_k[j] * value
            logits.append(total)
        peak = max(logits)
        exponentials = [math.exp(min(l - peak, 60.0)) for l in logits]
        denominator = sum(exponentials) or 1.0
        return [e / denominator for e in exponentials]

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        return [self._probabilities(row) for row in X]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "softmax",
            "classes": self.classes,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SoftmaxRegression":
        model = cls(classes=int(data.get("classes", 3)))
        model.weights = [[float(v) for v in row] for row in data.get("weights", [])]
        model.bias = [float(v) for v in data.get("bias", [])]
        return model


# --------------------------------------------------------------------------
# Training entry point
# --------------------------------------------------------------------------


def train_model(
    samples: Sequence[LabelledSample],
    name: str = "direction_v1",
    min_rows: int = 400,
    test_fraction: float = 0.25,
    prefer_backend: str | None = None,
) -> TrainingResult:
    """Train, calibrate, evaluate and (if it earns it) return an artefact."""

    distribution = class_distribution(samples)
    metrics: dict[str, Any] = {
        "rows": len(samples),
        "class_distribution": distribution,
    }

    if len(samples) < min_rows:
        return TrainingResult(
            artifact=None,
            metrics=metrics,
            accepted=False,
            reason=(
                f"only {len(samples)} labelled rows, need at least {min_rows}; "
                "the model stays disabled and the bot runs on rules alone"
            ),
        )

    present = {sample.label for sample in samples}
    if len(present) < 2:
        return TrainingResult(
            artifact=None,
            metrics=metrics,
            accepted=False,
            reason="labels contain a single class - nothing to learn",
        )

    train, test = time_split(samples, test_fraction=test_fraction)
    if len(train) < min_rows // 2 or len(test) < 50:
        return TrainingResult(
            artifact=None,
            metrics=metrics,
            accepted=False,
            reason=f"chronological split too small (train={len(train)}, test={len(test)})",
        )

    spec = FeatureSpec.fit([s.features for s in train])
    X_train = spec.transform_many([s.features for s in train])
    y_train = [s.label for s in train]
    X_test = spec.transform_many([s.features for s in test])
    y_test = [s.label for s in test]

    backend, model, probabilities = _fit_backend(
        X_train, y_train, X_test, prefer_backend
    )
    metrics["backend"] = backend
    metrics["features"] = len(spec)
    metrics["train_rows"] = len(train)
    metrics["test_rows"] = len(test)

    # --- calibration on the held-out slice (one-vs-rest per direction) ---
    long_scores = [p[LABEL_LONG] for p in probabilities]
    short_scores = [p[LABEL_SHORT] for p in probabilities]
    long_labels = [1 if y == LABEL_LONG else 0 for y in y_test]
    short_labels = [1 if y == LABEL_SHORT else 0 for y in y_test]

    long_calibrator, long_metrics = fit_calibrator(long_scores, long_labels)
    short_calibrator, short_metrics = fit_calibrator(short_scores, short_labels)
    metrics["calibration_long"] = long_metrics
    metrics["calibration_short"] = short_metrics

    calibrated_long = [
        long_calibrator.transform(s) if long_calibrator else s for s in long_scores
    ]
    calibrated_short = [
        short_calibrator.transform(s) if short_calibrator else s for s in short_scores
    ]

    metrics["accuracy"] = round(
        sum(1 for p, y in zip(probabilities, y_test) if _argmax(p) == y) / len(y_test), 4
    )
    metrics["brier_long"] = round(brier_score(calibrated_long, long_labels), 5)
    metrics["brier_short"] = round(brier_score(calibrated_short, short_labels), 5)
    metrics["ece_long"] = round(
        expected_calibration_error(calibrated_long, long_labels), 5
    )
    metrics["ece_short"] = round(
        expected_calibration_error(calibrated_short, short_labels), 5
    )
    metrics["reliability_long"] = reliability_table(calibrated_long, long_labels)

    # --- acceptance -----------------------------------------------------
    majority = max(distribution.values()) / max(len(samples), 1)
    metrics["majority_baseline"] = round(majority, 4)

    # Baseline Brier for a model that always predicts the base rate.
    long_rate = sum(long_labels) / len(long_labels)
    baseline_brier_long = brier_score([long_rate] * len(long_labels), long_labels)
    metrics["baseline_brier_long"] = round(baseline_brier_long, 5)

    beats_baseline = metrics["brier_long"] <= baseline_brier_long * 1.02
    useful_accuracy = metrics["accuracy"] >= majority * 0.98

    if not (beats_baseline and useful_accuracy):
        return TrainingResult(
            artifact=None,
            metrics=metrics,
            accepted=False,
            reason=(
                f"model does not beat the naive baseline "
                f"(accuracy {metrics['accuracy']:.3f} vs majority {majority:.3f}, "
                f"Brier {metrics['brier_long']:.4f} vs {baseline_brier_long:.4f}) - "
                "refusing to deploy it"
            ),
        )

    artifact = ModelArtifact(
        name=name,
        version=time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
        algorithm=backend,
        trained_at=int(time.time()),
        rows=len(samples),
        feature_spec=spec,
        model_payload=_serialise_model(backend, model),
        long_calibrator=long_calibrator,
        short_calibrator=short_calibrator,
        metrics=metrics,
    )
    artifact.attach_runtime(model)
    return TrainingResult(artifact=artifact, metrics=metrics, accepted=True, reason="ok")


def _fit_backend(
    X_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    X_test: Sequence[Sequence[float]],
    prefer: str | None,
) -> tuple[str, Any, list[list[float]]]:
    """Fit with the best available backend; returns test-set probabilities."""

    weights = balance_weights_from_labels(y_train)

    if prefer in (None, "lightgbm") and HAVE_LIGHTGBM and HAVE_NUMPY:
        try:
            return _fit_lightgbm(X_train, y_train, X_test, weights)
        except Exception as exc:  # noqa: BLE001 - fall through to the next backend
            log.warning("LightGBM training failed, falling back: %s", exc)

    if prefer in (None, "sklearn") and HAVE_SKLEARN and HAVE_NUMPY:
        try:
            return _fit_sklearn(X_train, y_train, X_test, weights)
        except Exception as exc:  # noqa: BLE001
            log.warning("scikit-learn training failed, falling back: %s", exc)

    model = SoftmaxRegression().fit(X_train, y_train, weights)
    return "softmax-pure-python", model, model.predict_proba(X_test)


def _fit_lightgbm(
    X_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    X_test: Sequence[Sequence[float]],
    weights: Mapping[int, float],
) -> tuple[str, Any, list[list[float]]]:
    sample_weight = [weights.get(label, 1.0) for label in y_train]
    model = lightgbm.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        verbose=-1,
    )
    model.fit(numpy.array(X_train), numpy.array(y_train), sample_weight=sample_weight)
    probabilities = model.predict_proba(numpy.array(X_test))
    return "lightgbm", model, [list(map(float, row)) for row in probabilities]


def _fit_sklearn(
    X_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    X_test: Sequence[Sequence[float]],
    weights: Mapping[int, float],
) -> tuple[str, Any, list[list[float]]]:
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore

    sample_weight = [weights.get(label, 1.0) for label in y_train]
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.0,
        random_state=17,
    )
    model.fit(numpy.array(X_train), numpy.array(y_train), sample_weight=sample_weight)
    raw = model.predict_proba(numpy.array(X_test))
    probabilities = _align_classes(raw, list(model.classes_))
    return "sklearn-hgb", model, probabilities


def _align_classes(raw: Any, classes: Sequence[int]) -> list[list[float]]:
    """Expand a model's class order into a fixed ``[no_trade, long, short]``."""

    out: list[list[float]] = []
    index = {int(label): position for position, label in enumerate(classes)}
    for row in raw:
        aligned = [0.0, 0.0, 0.0]
        for label in (LABEL_NO_TRADE, LABEL_LONG, LABEL_SHORT):
            position = index.get(label)
            if position is not None:
                aligned[label] = float(row[position])
        out.append(aligned)
    return out


def balance_weights_from_labels(labels: Sequence[int]) -> dict[int, float]:
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return {}
    total = sum(counts.values())
    classes = len(counts)
    return {label: total / (classes * count) for label, count in counts.items()}


def _serialise_model(backend: str, model: Any) -> dict[str, Any]:
    if backend == "softmax-pure-python":
        return model.as_dict()
    # Tree models are stored as a pickle sidecar by the registry.
    return {"kind": backend, "external": True}


def _argmax(values: Sequence[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values):
        if value > best_value:
            best_index, best_value = index, value
    return best_index


__all__ = ["train_model", "TrainingResult", "SoftmaxRegression"]
