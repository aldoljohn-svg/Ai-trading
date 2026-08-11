"""Overfitting detection.

A model that scores brilliantly in training and mediocrely out of sample has not
learned the market; it has memorised the sample. This checks the signatures that
distinguish the two:

1. **train/test gap** - the classic tell
2. **out-of-sample collapse** - performance falls off a cliff on unseen data
3. **parameter sensitivity** - small parameter changes swing results wildly
4. **performance instability** - results vary hugely across folds or periods
5. **regime dependence** - all the edge comes from one regime
6. **implausible results** - a Sharpe of 9 or a 95% win rate is a bug, not an edge
7. **feature dependency** - one feature carries almost all the importance

The verdict is ``REJECT`` or ``REDUCE_WEIGHT``, and an implausible result is
always ``REJECT`` - being suspicious of one's own good news is the whole job.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class OverfittingVerdict(str, Enum):
    CLEAN = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    REDUCE_WEIGHT = "REDUCE_WEIGHT"
    REJECT = "REJECT"


@dataclass(slots=True)
class OverfittingReport:
    verdict: OverfittingVerdict = OverfittingVerdict.CLEAN
    severity: float = 0.0
    train_test_gap: float = 0.0
    oos_ratio: float = 1.0
    instability: float = 0.0
    parameter_sensitivity: float = 0.0
    regime_concentration: float = 0.0
    feature_concentration: float = 0.0
    signals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def overfitted(self) -> bool:
        return self.verdict in (
            OverfittingVerdict.REDUCE_WEIGHT,
            OverfittingVerdict.REJECT,
        )

    @property
    def weight_multiplier(self) -> float:
        return {
            OverfittingVerdict.CLEAN: 1.0,
            OverfittingVerdict.SUSPICIOUS: 0.8,
            OverfittingVerdict.REDUCE_WEIGHT: 0.4,
            OverfittingVerdict.REJECT: 0.0,
        }[self.verdict]

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "overfitted": self.overfitted,
            "severity": round(self.severity, 4),
            "weight_multiplier": self.weight_multiplier,
            "train_test_gap": round(self.train_test_gap, 4),
            "oos_ratio": round(self.oos_ratio, 4),
            "instability": round(self.instability, 4),
            "parameter_sensitivity": round(self.parameter_sensitivity, 4),
            "regime_concentration": round(self.regime_concentration, 4),
            "feature_concentration": round(self.feature_concentration, 4),
            "signals": self.signals,
            "notes": self.notes,
        }

    def summary(self) -> str:
        lines = [f"Overfitting check: {self.verdict.value} (severity {self.severity:.2f})"]
        for signal in self.signals:
            lines.append(f"  ⚠️ {signal}")
        for note in self.notes:
            lines.append(f"  · {note}")
        return "\n".join(lines)


def detect_overfitting(
    train_score: float | None = None,
    test_score: float | None = None,
    oos_score: float | None = None,
    fold_scores: Sequence[float] | None = None,
    parameter_scores: Sequence[float] | None = None,
    regime_scores: Mapping[str, float] | None = None,
    feature_importances: Mapping[str, float] | None = None,
    win_rate: float | None = None,
    sharpe: float | None = None,
    trades: int = 0,
    min_trades: int = 30,
) -> OverfittingReport:
    """Run every check for which data was supplied."""

    report = OverfittingReport()
    penalties: list[float] = []

    # --- sample size ------------------------------------------------------
    if trades and trades < min_trades:
        penalties.append(0.35)
        report.notes.append(
            f"only {trades} trades - not enough to distinguish edge from luck"
        )

    # --- 1. train/test gap -------------------------------------------------
    if train_score is not None and test_score is not None and train_score > 0:
        report.train_test_gap = (train_score - test_score) / abs(train_score)
        if report.train_test_gap > 0.5:
            penalties.append(min(0.85, report.train_test_gap))
            report.signals.append(
                f"train score {train_score:.3f} vs test {test_score:.3f} - "
                f"a {report.train_test_gap:.0%} gap"
            )
        elif report.train_test_gap > 0.25:
            penalties.append(0.25)
            report.notes.append(f"train/test gap {report.train_test_gap:.0%}")

    # --- 2. out-of-sample collapse -------------------------------------------
    reference = test_score if test_score is not None else train_score
    if oos_score is not None and reference is not None and reference > 0:
        report.oos_ratio = oos_score / reference
        if report.oos_ratio < 0.3:
            penalties.append(0.8)
            report.signals.append(
                f"out-of-sample performance is {report.oos_ratio:.0%} of in-sample"
            )
        elif report.oos_ratio < 0.6:
            penalties.append(0.35)
            report.notes.append(
                f"out-of-sample retains {report.oos_ratio:.0%} of in-sample"
            )

    # --- 3. parameter sensitivity ---------------------------------------------
    if parameter_scores and len(parameter_scores) >= 3:
        mean = statistics.fmean(parameter_scores)
        if abs(mean) > 1e-9:
            report.parameter_sensitivity = statistics.pstdev(parameter_scores) / abs(mean)
            if report.parameter_sensitivity > 0.6:
                penalties.append(min(0.75, report.parameter_sensitivity))
                report.signals.append(
                    f"results swing {report.parameter_sensitivity:.0%} across nearby "
                    "parameter values - the peak is a spike, not a plateau"
                )

    # --- 4. fold instability ----------------------------------------------------
    if fold_scores and len(fold_scores) >= 3:
        mean = statistics.fmean(fold_scores)
        stdev = statistics.pstdev(fold_scores)
        if abs(mean) > 1e-9:
            report.instability = stdev / abs(mean)
            if report.instability > 1.0:
                penalties.append(0.6)
                report.signals.append(
                    f"fold results vary by {report.instability:.0%} of their mean"
                )
        negative = sum(1 for s in fold_scores if s <= 0)
        if negative >= len(fold_scores) / 2:
            penalties.append(0.55)
            report.signals.append(
                f"{negative} of {len(fold_scores)} folds are unprofitable"
            )

    # --- 5. regime concentration -------------------------------------------------
    if regime_scores and len(regime_scores) >= 2:
        positives = {k: v for k, v in regime_scores.items() if v > 0}
        total_positive = sum(positives.values())
        if total_positive > 0:
            largest = max(positives.values())
            report.regime_concentration = largest / total_positive
            if report.regime_concentration > 0.85 and len(regime_scores) >= 3:
                penalties.append(0.5)
                report.signals.append(
                    f"{report.regime_concentration:.0%} of the edge comes from a "
                    "single regime"
                )

    # --- 6. implausible results -----------------------------------------------
    if win_rate is not None and win_rate > 0.85 and trades >= 20:
        penalties.append(0.9)
        report.signals.append(
            f"{win_rate:.0%} win rate is implausible - suspect lookahead or a bug"
        )
    if sharpe is not None and sharpe > 5 and trades >= 20:
        penalties.append(0.85)
        report.signals.append(
            f"Sharpe {sharpe:.1f} is implausible for this strategy class"
        )

    # --- 7. feature concentration ------------------------------------------------
    if feature_importances:
        total = sum(abs(v) for v in feature_importances.values())
        if total > 0:
            largest = max(abs(v) for v in feature_importances.values())
            report.feature_concentration = largest / total
            if report.feature_concentration > 0.6 and len(feature_importances) >= 5:
                penalties.append(0.4)
                report.signals.append(
                    f"one feature carries {report.feature_concentration:.0%} of the "
                    "model's importance"
                )

    report.severity = _noisy_or(penalties)

    # Implausible results are always a rejection, whatever the arithmetic says.
    implausible = any("implausible" in s for s in report.signals)
    if implausible or report.severity >= 0.7:
        report.verdict = OverfittingVerdict.REJECT
    elif report.severity >= 0.45:
        report.verdict = OverfittingVerdict.REDUCE_WEIGHT
    elif report.severity >= 0.2:
        report.verdict = OverfittingVerdict.SUSPICIOUS
    else:
        report.verdict = OverfittingVerdict.CLEAN

    if not report.signals and not report.notes:
        report.notes.append("no overfitting signature detected")
    return report


def _noisy_or(penalties: Sequence[float]) -> float:
    survival = 1.0
    for penalty in penalties:
        survival *= 1.0 - max(0.0, min(penalty, 1.0))
    return 1.0 - survival


__all__ = ["OverfittingReport", "OverfittingVerdict", "detect_overfitting"]
