"""Expected-value engine.

Turns the whole picture into one number with an explicit unit: **expected R per
trade, net of costs.**

    EV = P(win) x avg_win_R - P(loss) x avg_loss_R - cost_R

Where P(win) is *not* the ensemble's raw confidence. Raw confidence is a
conviction score, not a frequency. It is mapped through the historically
calibrated relationship between stated confidence and realised hit rate when
one exists, and shrunk toward the base rate when it does not.

A trade is only permitted when EV clears a configured threshold, and the
threshold is in R so it is comparable across symbols and account sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

#: With no calibration history, confidence is shrunk this far toward the base
#: rate. An unproven "85% confident" is treated as far closer to a coin flip.
UNPROVEN_SHRINKAGE = 0.55
BASE_RATE = 0.5


@dataclass(slots=True)
class ExpectedValue:
    expected_r: float = 0.0
    win_probability: float = 0.0
    raw_confidence: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 1.0
    cost_r: float = 0.0
    threshold_r: float = 0.05
    calibrated: bool = False
    acceptable: bool = False
    reasoning: list[str] = field(default_factory=list)

    @property
    def edge_per_trade_pct(self) -> float:
        """EV expressed as a fraction of the risked amount."""

        return self.expected_r

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_r": round(self.expected_r, 4),
            "win_probability": round(self.win_probability, 4),
            "raw_confidence": round(self.raw_confidence, 4),
            "avg_win_r": round(self.avg_win_r, 4),
            "avg_loss_r": round(self.avg_loss_r, 4),
            "cost_r": round(self.cost_r, 4),
            "threshold_r": self.threshold_r,
            "calibrated": self.calibrated,
            "acceptable": self.acceptable,
            "reasoning": self.reasoning,
        }

    def summary(self) -> str:
        verdict = "acceptable" if self.acceptable else "below threshold"
        return (
            f"EV {self.expected_r:+.3f}R ({verdict}); "
            f"P(win) {self.win_probability:.0%}, "
            f"win {self.avg_win_r:.2f}R / loss {self.avg_loss_r:.2f}R, "
            f"cost {self.cost_r:.3f}R"
        )


def calibrate_probability(
    confidence: float,
    calibration: Any = None,
    historical_hit_rate: float | None = None,
    sample_size: int = 0,
) -> tuple[float, bool]:
    """Map conviction onto a probability with a frequentist meaning.

    Priority:
    1. an explicit calibrator (Platt/isotonic) fitted on realised outcomes
    2. the historical hit rate at this confidence level, shrunk by sample size
    3. heavy shrinkage toward the base rate
    """

    confidence = max(0.0, min(confidence, 1.0))

    if calibration is not None and getattr(calibration, "fitted", False):
        return max(0.01, min(calibration.transform(confidence), 0.95)), True

    if historical_hit_rate is not None and sample_size >= 20:
        # Shrink toward the stated confidence as evidence accumulates.
        weight = min(sample_size / 100.0, 1.0)
        blended = weight * historical_hit_rate + (1 - weight) * (
            BASE_RATE + (confidence - BASE_RATE) * (1 - UNPROVEN_SHRINKAGE)
        )
        return max(0.01, min(blended, 0.95)), True

    shrunk = BASE_RATE + (confidence - BASE_RATE) * (1 - UNPROVEN_SHRINKAGE)
    return max(0.01, min(shrunk, 0.95)), False


def compute_expected_value(
    confidence: float,
    rr: float,
    cost_pct: float = 0.0,
    stop_distance_pct: float = 0.0,
    partial_ladder: Sequence[tuple[float, float]] | None = None,
    calibration: Any = None,
    historical_hit_rate: float | None = None,
    sample_size: int = 0,
    threshold_r: float = 0.05,
) -> ExpectedValue:
    """Expected R per trade, net of costs.

    ``partial_ladder`` is ``[(fraction_closed, r_multiple), ...]``. When given,
    the average win is the ladder's weighted R rather than the headline target -
    scaling out lowers the average win, and pretending otherwise inflates EV.
    """

    result = ExpectedValue(raw_confidence=confidence, threshold_r=threshold_r)
    reasons: list[str] = []

    probability, calibrated = calibrate_probability(
        confidence, calibration, historical_hit_rate, sample_size
    )
    result.win_probability = probability
    result.calibrated = calibrated
    if calibrated:
        reasons.append(
            f"P(win) {probability:.0%} from calibrated history "
            f"(stated confidence {confidence:.0%})"
        )
    else:
        reasons.append(
            f"P(win) {probability:.0%} - stated confidence {confidence:.0%} shrunk "
            "toward the base rate because there is no calibration history yet"
        )

    # --- average win ------------------------------------------------------
    if partial_ladder:
        total_fraction = sum(f for f, _r in partial_ladder)
        if total_fraction > 0:
            result.avg_win_r = sum(f * r for f, r in partial_ladder) / total_fraction
            reasons.append(
                f"scaled exit averages {result.avg_win_r:.2f}R rather than the "
                f"{rr:.2f}R headline target"
            )
        else:
            result.avg_win_r = rr
    else:
        result.avg_win_r = rr

    # A loss is one R by construction, plus the cost of being wrong.
    result.avg_loss_r = 1.0

    # --- costs ---------------------------------------------------------------
    if stop_distance_pct > 0 and cost_pct > 0:
        # Convert a percentage cost into R by dividing by the stop distance.
        result.cost_r = cost_pct / stop_distance_pct
        reasons.append(
            f"round-trip cost {cost_pct:.3%} against a {stop_distance_pct:.2%} stop "
            f"= {result.cost_r:.3f}R"
        )

    result.expected_r = (
        probability * result.avg_win_r
        - (1 - probability) * result.avg_loss_r
        - result.cost_r
    )
    result.acceptable = result.expected_r >= threshold_r

    if not result.acceptable:
        reasons.append(
            f"EV {result.expected_r:+.3f}R is below the {threshold_r:+.3f}R threshold"
        )

    result.reasoning = reasons
    return result


def break_even_probability(rr: float, cost_r: float = 0.0) -> float:
    """The hit rate this reward:risk needs just to break even."""

    if rr <= 0:
        return 1.0
    return (1.0 + cost_r) / (1.0 + rr)


__all__ = [
    "ExpectedValue",
    "compute_expected_value",
    "calibrate_probability",
    "break_even_probability",
]
