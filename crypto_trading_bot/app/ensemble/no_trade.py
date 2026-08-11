"""The NO-TRADE model.

A first-class decision with its own model, not the absence of a decision.

It looks for the conditions that historically precede bad trades regardless of
how attractive the setup looks: model disagreement, thin participation, poor
data quality, hostile execution conditions, macro uncertainty, extreme
volatility, weak reward:risk, and recent clustered losses under similar
conditions.

Each condition contributes evidence toward ``NO_TRADE``. The verdict carries the
full list, so "why didn't it trade?" is always answerable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain import Regime
from app.ensemble.base import ModelContext, ModelSignal
from app.ensemble.engine import EnsembleResult


@dataclass(slots=True)
class NoTradeReason:
    code: str
    weight: float            # 0..1 evidence contribution
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "weight": round(self.weight, 3), "detail": self.detail}


@dataclass(slots=True)
class NoTradeVerdict:
    no_trade: bool
    score: float                                  # 0 = trade freely, 1 = definitely not
    reasons: list[NoTradeReason] = field(default_factory=list)
    threshold: float = 0.5

    @property
    def blocking_reasons(self) -> list[str]:
        return [f"{r.code}: {r.detail}" for r in sorted(
            self.reasons, key=lambda r: r.weight, reverse=True
        )]

    def as_dict(self) -> dict[str, Any]:
        return {
            "no_trade": self.no_trade,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "reasons": [r.as_dict() for r in self.reasons],
        }

    def summary(self) -> str:
        if not self.no_trade:
            return f"trade permitted (no-trade score {self.score:.2f})"
        return "NO TRADE: " + "; ".join(self.blocking_reasons[:3])


class NoTradeModel:
    """Scores the case *against* taking a trade."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_agreement: float = 0.60,
        min_participation: float = 0.40,
        min_data_quality: float = 0.35,
        max_aggregate_risk: float = 0.75,
        min_conviction: float = 0.35,
    ) -> None:
        self.threshold = threshold
        self.min_agreement = min_agreement
        self.min_participation = min_participation
        self.min_data_quality = min_data_quality
        self.max_aggregate_risk = max_aggregate_risk
        self.min_conviction = min_conviction

    def evaluate(
        self,
        ensemble: EnsembleResult,
        context: ModelContext,
        rr: float = 0.0,
        min_rr: float = 2.0,
        loss_cluster: Any = None,
        execution_risk: float | None = None,
        data_risk: float | None = None,
    ) -> NoTradeVerdict:
        reasons: list[NoTradeReason] = []

        # --- 1. no direction at all -------------------------------------
        if not ensemble.signal.is_directional:
            reasons.append(
                NoTradeReason("no_direction", 1.0, "the ensemble has no directional view")
            )

        # --- 1b. weak conviction ----------------------------------------
        # A direction the ensemble barely believes in is exactly the "low
        # quality trade" this model exists to prevent.
        if ensemble.signal.is_directional and ensemble.confidence < self.min_conviction:
            shortfall = (
                self.min_conviction - ensemble.confidence
            ) / max(self.min_conviction, 1e-9)
            reasons.append(
                NoTradeReason(
                    "weak_conviction",
                    min(0.9, 0.3 + 0.6 * shortfall),
                    f"ensemble conviction only {ensemble.confidence:.0%}",
                )
            )

        # --- 2. model disagreement --------------------------------------
        if ensemble.signal.is_directional and ensemble.agreement < self.min_agreement:
            shortfall = (self.min_agreement - ensemble.agreement) / self.min_agreement
            reasons.append(
                NoTradeReason(
                    "model_disagreement",
                    min(0.9, 0.4 + 0.6 * shortfall),
                    f"only {ensemble.agreement:.0%} of the directional vote agrees "
                    f"({ensemble.model_agreement_label})",
                )
            )

        # --- 3. thin participation --------------------------------------
        if ensemble.participation < self.min_participation:
            shortfall = (
                self.min_participation - ensemble.participation
            ) / max(self.min_participation, 1e-9)
            reasons.append(
                NoTradeReason(
                    "thin_participation",
                    min(0.9, 0.35 + 0.6 * shortfall),
                    f"only {ensemble.participation:.0%} of model weight formed an opinion",
                )
            )

        # --- 4. data quality --------------------------------------------
        if ensemble.data_quality < self.min_data_quality:
            reasons.append(
                NoTradeReason(
                    "poor_data_quality",
                    0.8,
                    f"aggregate data quality {ensemble.data_quality:.0%}",
                )
            )
        if data_risk is not None and data_risk > 0.7:
            reasons.append(
                NoTradeReason("data_risk", 0.9, f"data risk score {data_risk:.2f}")
            )

        # --- 5. aggregate model risk ------------------------------------
        if ensemble.aggregate_risk > self.max_aggregate_risk:
            reasons.append(
                NoTradeReason(
                    "hostile_conditions",
                    0.7,
                    f"models report aggregate risk {ensemble.aggregate_risk:.2f}",
                )
            )

        # --- 6. execution conditions ------------------------------------
        if execution_risk is not None and execution_risk > 0.7:
            reasons.append(
                NoTradeReason(
                    "execution_risk", 0.85, f"execution risk score {execution_risk:.2f}"
                )
            )

        # --- 7. regime ----------------------------------------------------
        if context.regime in (Regime.HIGH_VOLATILITY, Regime.UNKNOWN):
            reasons.append(
                NoTradeReason(
                    "regime_block", 1.0, f"{context.regime.value} regime forbids entries"
                )
            )
        elif context.regime is Regime.TRANSITION:
            reasons.append(
                NoTradeReason(
                    "regime_caution", 0.25, "structure is mid-transition"
                )
            )

        # --- 8. reward:risk ------------------------------------------------
        if rr > 0 and rr < min_rr:
            reasons.append(
                NoTradeReason(
                    "poor_reward_risk",
                    min(0.9, 0.5 + 0.5 * (min_rr - rr) / max(min_rr, 1e-9)),
                    f"reward:risk {rr:.2f} below the {min_rr:.2f} minimum",
                )
            )

        # --- 9. clustered losses in similar conditions --------------------
        if loss_cluster is not None and getattr(loss_cluster, "detected", False):
            reasons.append(
                NoTradeReason(
                    "loss_cluster",
                    0.6,
                    getattr(loss_cluster, "description", "recent losses in similar conditions"),
                )
            )

        score = _combine(reasons)
        return NoTradeVerdict(
            no_trade=score >= self.threshold,
            score=score,
            reasons=reasons,
            threshold=self.threshold,
        )


def _combine(reasons: list[NoTradeReason]) -> float:
    """Noisy-OR: independent pieces of evidence accumulate without ever exceeding 1.

    Using noisy-OR rather than a sum means five weak concerns can add up to a
    block, while one weak concern on its own cannot.
    """

    if not reasons:
        return 0.0
    survival = 1.0
    for reason in reasons:
        survival *= 1.0 - max(0.0, min(reason.weight, 1.0))
    return 1.0 - survival


__all__ = ["NoTradeModel", "NoTradeVerdict", "NoTradeReason"]
