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
from app.ml.meta import LABEL_WIN
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
    kind: str = "direction",
    reward: float = 2.0,
) -> TrainingResult:
    """Train, calibrate, evaluate and (if it earns it) return an artefact.

    ``kind`` selects the problem being solved.  ``direction`` is the three-class
    LONG/SHORT/NO_TRADE model; ``meta`` is the binary "does the rules' chosen
    side reach its target first" model, which is scored on whether it can *rank*
    trades rather than on argmax accuracy.

    ``reward`` is the payoff ratio the labels were built with (profit barrier
    divided by loss barrier).  A meta model is judged partly on whether its
    best-ranked trades clear break-even at that payoff, so getting this wrong
    would grade the model against a trade it was never taught.
    """

    if kind == "meta":
        from app.ml.meta import meta_class_distribution

        distribution = meta_class_distribution(samples)
    else:
        distribution = class_distribution(samples)
    metrics: dict[str, Any] = {
        "rows": len(samples),
        "kind": kind,
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

    if kind == "meta":
        return _finish_meta(
            samples=samples,
            spec=spec,
            model=model,
            backend=backend,
            probabilities=probabilities,
            y_test=y_test,
            metrics=metrics,
            distribution=distribution,
            name=name,
            reward=reward,
        )

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


#: Minimum relative edge between the best and worst slice.  Kept as a floor so
#: a trivially small but statistically detectable difference -- which large
#: samples will always find -- cannot pass on significance alone.
_MIN_META_LIFT = 1.10

#: The separation must also be unlikely to be luck.  A fixed lift threshold
#: ignores sample size, which is wrong in both directions: it waves through
#: noise on a few hundred rows and rejects a genuine edge on tens of thousands.
_MIN_META_Z = 1.96          # ~95% two-sided

#: Size of the top/bottom slices used for that comparison.
_META_TAIL = 0.2

#: Round-trip cost of a trade, expressed in R against a 1R stop: taker fees
#: both ways, the spread, and slippage.  Deliberately not optimistic -- a model
#: that only looks profitable when trading is free is not profitable.
_ASSUMED_COST_R = 0.09


def _decile_lift(scores: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    """Win rate in the highest-scoring slice against the lowest-scoring slice.

    This is the question a meta model actually has to answer: when it is more
    confident, does the trade win more often?  Accuracy cannot see that on an
    imbalanced binary problem, where predicting the majority class everywhere
    scores well while being useless for choosing between trades.
    """

    paired = sorted(zip(scores, labels), key=lambda pair: pair[0])
    size = max(int(len(paired) * _META_TAIL), 1)
    bottom = paired[:size]
    top = paired[-size:]

    bottom_rate = sum(label for _, label in bottom) / len(bottom)
    top_rate = sum(label for _, label in top) / len(top)
    # Guard the ratio: a bottom slice that never wins would divide by zero.
    lift = top_rate / bottom_rate if bottom_rate > 0 else (top_rate / 1e-6 if top_rate else 0.0)

    # Is the gap bigger than sampling noise?  A fixed lift threshold cannot
    # answer that, because the same lift means very different things on 200
    # rows and on 20,000.
    #
    # The rates are nudged off 0 and 1 (Haldane-Anscombe) before the variance is
    # taken.  Without it a perfectly separating model has zero variance in both
    # slices, the z-score divides by zero, and the best possible result would be
    # rejected as indistinguishable from noise.
    top_adj = (sum(label for _, label in top) + 0.5) / (size + 1)
    bottom_adj = (sum(label for _, label in bottom) + 0.5) / (size + 1)
    variance = (
        top_adj * (1 - top_adj) / size + bottom_adj * (1 - bottom_adj) / size
    )
    z = (top_adj - bottom_adj) / math.sqrt(variance) if variance > 0 else 0.0

    return {
        "top_win_rate": round(top_rate, 4),
        "bottom_win_rate": round(bottom_rate, 4),
        "lift": round(min(lift, 99.0), 3),
        "separation_z": round(z, 2),
        "slice_size": size,
    }


def expectancy_r(win_rate: float, reward: float, risk: float = 1.0) -> float:
    """Expected R per trade at this win rate, before costs."""

    return win_rate * reward - (1.0 - win_rate) * risk


def _finish_meta(
    samples: Sequence[LabelledSample],
    spec: FeatureSpec,
    model: Any,
    backend: str,
    probabilities: Sequence[Sequence[float]],
    y_test: Sequence[int],
    metrics: dict[str, Any],
    distribution: dict[str, int],
    name: str,
    reward: float = 2.0,
) -> TrainingResult:
    """Calibrate, score and judge a binary meta model."""

    win_scores = [p[LABEL_WIN] if len(p) > LABEL_WIN else 0.0 for p in probabilities]
    win_labels = [1 if y == LABEL_WIN else 0 for y in y_test]

    calibrator, calibration_metrics = fit_calibrator(win_scores, win_labels)
    metrics["calibration_win"] = calibration_metrics

    calibrated = [
        calibrator.transform(s) if calibrator else s for s in win_scores
    ]

    base_rate = sum(win_labels) / len(win_labels)
    metrics["base_win_rate"] = round(base_rate, 4)
    metrics["accuracy"] = round(
        sum(1 for p, y in zip(probabilities, y_test) if _argmax(p) == y) / len(y_test),
        4,
    )
    metrics["brier_win"] = round(brier_score(calibrated, win_labels), 5)
    metrics["ece_win"] = round(expected_calibration_error(calibrated, win_labels), 5)
    metrics["reliability_win"] = reliability_table(calibrated, win_labels)

    baseline_brier = brier_score([base_rate] * len(win_labels), win_labels)
    metrics["baseline_brier_win"] = round(baseline_brier, 5)

    lift_metrics = _decile_lift(calibrated, win_labels)
    metrics.update(lift_metrics)

    # --- would trading the model's best slice actually make money? --------
    #
    # Statistical separation is necessary but nowhere near sufficient.  A model
    # can rank trades detectably better than chance and still pick only losers,
    # which is exactly what happens when the underlying setup sits near
    # break-even: filtering a negative edge harder produces a smaller negative
    # edge, not a positive one.
    top_expectancy = expectancy_r(lift_metrics["top_win_rate"], reward)
    net_expectancy = top_expectancy - _ASSUMED_COST_R
    metrics["reward_r"] = round(reward, 3)
    metrics["base_expectancy_r"] = round(expectancy_r(base_rate, reward), 4)
    metrics["top_expectancy_r"] = round(top_expectancy, 4)
    metrics["net_expectancy_r"] = round(net_expectancy, 4)
    metrics["assumed_cost_r"] = _ASSUMED_COST_R
    metrics["break_even_win_rate"] = round(1.0 / (1.0 + reward), 4)

    beats_baseline = metrics["brier_win"] <= baseline_brier * 1.01
    separates = (
        lift_metrics["lift"] >= _MIN_META_LIFT
        and lift_metrics["separation_z"] >= _MIN_META_Z
    )
    profitable = net_expectancy > 0

    if not (beats_baseline and separates and profitable):
        reasons: list[str] = []
        if not separates:
            reasons.append(
                f"cannot rank trades (top {lift_metrics['top_win_rate']:.1%} vs "
                f"bottom {lift_metrics['bottom_win_rate']:.1%}, lift "
                f"{lift_metrics['lift']:.2f}, z {lift_metrics['separation_z']:.2f})"
            )
        elif not profitable:
            reasons.append(
                f"ranks trades genuinely (top {lift_metrics['top_win_rate']:.1%} vs "
                f"bottom {lift_metrics['bottom_win_rate']:.1%}, z "
                f"{lift_metrics['separation_z']:.2f}) but even its best slice "
                f"loses money: {top_expectancy:+.4f}R gross, "
                f"{net_expectancy:+.4f}R after {_ASSUMED_COST_R}R costs. "
                f"Break-even needs {metrics['break_even_win_rate']:.1%} at this "
                f"{reward:.1f}:1 payoff; the setup itself is the problem, not "
                "the model"
            )
        if not beats_baseline:
            reasons.append(
                f"Brier {metrics['brier_win']:.4f} vs baseline {baseline_brier:.4f}"
            )
        return TrainingResult(
            artifact=None,
            metrics=metrics,
            accepted=False,
            reason="meta model refused: " + "; ".join(reasons),
        )

    artifact = ModelArtifact(
        name=name,
        version=time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
        algorithm=backend,
        trained_at=int(time.time()),
        rows=len(samples),
        feature_spec=spec,
        model_payload=_serialise_model(backend, model),
        # A meta model has one head; the win calibrator lives in the long slot
        # and the short slot stays empty.
        long_calibrator=calibrator,
        short_calibrator=None,
        metrics=metrics,
        kind="meta",
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
