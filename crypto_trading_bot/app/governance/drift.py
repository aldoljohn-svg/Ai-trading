"""Model drift detection.

Drift is decay of *recent* performance against the model's own long-run record.
It is distinct from a losing streak: five losses in a row is noise, while a
sustained fall in hit rate accompanied by a widening calibration gap is the
model no longer describing the market.

Three signals, combined into a severity in ``[0, 1]``:

1. **performance decay** - recent expectancy versus baseline
2. **accuracy decay** - recent hit rate versus baseline, with a two-proportion
   z-test so small samples do not raise false alarms
3. **calibration decay** - stated confidence drifting above realised hit rate

Actions escalate: reduce weight -> shadow -> retrain -> disable.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


class DriftAction(str, Enum):
    NONE = "NONE"
    REDUCE_WEIGHT = "REDUCE_WEIGHT"
    SHADOW = "SHADOW"
    RETRAIN = "RETRAIN"
    DISABLE = "DISABLE"


@dataclass(slots=True)
class DriftReport:
    model: str = ""
    drifting: bool = False
    severity: float = 0.0
    action: DriftAction = DriftAction.NONE
    baseline_hit_rate: float = 0.0
    recent_hit_rate: float = 0.0
    baseline_expectancy: float = 0.0
    recent_expectancy: float = 0.0
    calibration_gap: float = 0.0
    z_score: float = 0.0
    sample_size: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def weight_multiplier(self) -> float:
        """How much of its normal weight the model should keep."""

        return {
            DriftAction.NONE: 1.0,
            DriftAction.REDUCE_WEIGHT: 0.5,
            DriftAction.SHADOW: 0.0,
            DriftAction.RETRAIN: 0.25,
            DriftAction.DISABLE: 0.0,
        }[self.action]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "drifting": self.drifting,
            "severity": round(self.severity, 4),
            "action": self.action.value,
            "weight_multiplier": self.weight_multiplier,
            "baseline_hit_rate": round(self.baseline_hit_rate, 4),
            "recent_hit_rate": round(self.recent_hit_rate, 4),
            "baseline_expectancy": round(self.baseline_expectancy, 4),
            "recent_expectancy": round(self.recent_expectancy, 4),
            "calibration_gap": round(self.calibration_gap, 4),
            "z_score": round(self.z_score, 3),
            "sample_size": self.sample_size,
            "reasons": self.reasons,
        }

    def summary(self) -> str:
        if not self.drifting:
            return f"{self.model}: no drift detected"
        return (
            f"{self.model}: drift severity {self.severity:.2f} -> "
            f"{self.action.value}; " + "; ".join(self.reasons[:2])
        )


def two_proportion_z(
    successes_a: int, total_a: int, successes_b: int, total_b: int
) -> float:
    """Z statistic for two proportions. Guards small samples from false alarms."""

    if total_a < 5 or total_b < 5:
        return 0.0
    p1 = successes_a / total_a
    p2 = successes_b / total_b
    pooled = (successes_a + successes_b) / (total_a + total_b)
    denominator = math.sqrt(pooled * (1 - pooled) * (1 / total_a + 1 / total_b))
    if denominator <= 0:
        return 0.0
    return (p1 - p2) / denominator


def detect_drift(
    model: str,
    recent_outcomes: Sequence[float],
    baseline_outcomes: Sequence[float],
    recent_confidences: Sequence[float] | None = None,
    min_recent: int = 15,
    min_baseline: int = 30,
    z_threshold: float = 1.96,        # ~95% confidence
) -> DriftReport:
    """Compare a recent window against the long-run baseline.

    ``recent_outcomes`` and ``baseline_outcomes`` are R multiples.
    """

    report = DriftReport(model=model, sample_size=len(recent_outcomes))

    if len(recent_outcomes) < min_recent or len(baseline_outcomes) < min_baseline:
        report.reasons.append(
            f"not enough evidence to judge drift "
            f"(recent {len(recent_outcomes)}/{min_recent}, "
            f"baseline {len(baseline_outcomes)}/{min_baseline})"
        )
        return report

    recent_wins = sum(1 for r in recent_outcomes if r > 0)
    baseline_wins = sum(1 for r in baseline_outcomes if r > 0)

    report.recent_hit_rate = recent_wins / len(recent_outcomes)
    report.baseline_hit_rate = baseline_wins / len(baseline_outcomes)
    report.recent_expectancy = statistics.fmean(recent_outcomes)
    report.baseline_expectancy = statistics.fmean(baseline_outcomes)

    penalties: list[float] = []

    # --- accuracy decay, tested for significance -------------------------
    report.z_score = two_proportion_z(
        recent_wins, len(recent_outcomes), baseline_wins, len(baseline_outcomes)
    )
    if report.z_score <= -z_threshold:
        drop = report.baseline_hit_rate - report.recent_hit_rate
        penalties.append(min(0.8, 0.4 + drop))
        report.reasons.append(
            f"hit rate fell from {report.baseline_hit_rate:.0%} to "
            f"{report.recent_hit_rate:.0%} (z={report.z_score:.2f}, significant)"
        )

    # --- expectancy decay --------------------------------------------------
    if report.baseline_expectancy > 0:
        ratio = report.recent_expectancy / report.baseline_expectancy
        if ratio < 0.4:
            penalties.append(min(0.8, 0.5 + (0.4 - ratio)))
            report.reasons.append(
                f"expectancy fell from {report.baseline_expectancy:+.3f}R to "
                f"{report.recent_expectancy:+.3f}R"
            )
    if report.recent_expectancy < 0 <= report.baseline_expectancy:
        penalties.append(0.6)
        report.reasons.append(
            f"expectancy turned negative ({report.recent_expectancy:+.3f}R)"
        )

    # --- calibration decay ---------------------------------------------------
    if recent_confidences and len(recent_confidences) >= min_recent:
        mean_confidence = statistics.fmean(recent_confidences)
        report.calibration_gap = mean_confidence - report.recent_hit_rate
        if report.calibration_gap > 0.2:
            penalties.append(min(0.7, report.calibration_gap * 1.5))
            report.reasons.append(
                f"claims {mean_confidence:.0%} confidence but hits "
                f"{report.recent_hit_rate:.0%}"
            )

    report.severity = _noisy_or(penalties)
    report.drifting = report.severity >= 0.3

    if report.severity >= 0.8:
        report.action = DriftAction.DISABLE
    elif report.severity >= 0.6:
        report.action = DriftAction.RETRAIN
    elif report.severity >= 0.45:
        report.action = DriftAction.SHADOW
    elif report.severity >= 0.3:
        report.action = DriftAction.REDUCE_WEIGHT
    else:
        report.action = DriftAction.NONE

    if not report.reasons:
        report.reasons.append("recent performance is consistent with the baseline")
    return report


def _noisy_or(penalties: Sequence[float]) -> float:
    survival = 1.0
    for penalty in penalties:
        survival *= 1.0 - max(0.0, min(penalty, 1.0))
    return 1.0 - survival


__all__ = ["DriftReport", "DriftAction", "detect_drift", "two_proportion_z"]
