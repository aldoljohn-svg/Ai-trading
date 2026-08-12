"""TRADE_QUALITY_SCORE - one 0-100 figure for how good this opportunity is.

Combines signal quality, regime fit, order flow, liquidity, execution
conditions, portfolio headroom, model risk and data quality.

Two properties that matter:

* **Risk components can only subtract.** A pristine execution environment does
  not make a weak signal into a strong trade, but a hostile one degrades a
  strong signal. Risk is a penalty, never a bonus.
* **It is explicitly not a probability of profit.** It is a relative ranking of
  opportunity quality. The probability lives in
  :mod:`app.quality.expected_value`, where it is calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain import Regime

#: Weights of the positive components. Risk components are applied as
#: multiplicative penalties afterwards.
COMPONENT_WEIGHTS: dict[str, float] = {
    "signal": 0.34,
    "regime_fit": 0.16,
    "order_flow": 0.16,
    "liquidity": 0.10,
    "reward_risk": 0.14,
    "portfolio_headroom": 0.10,
}


@dataclass(slots=True)
class TradeQuality:
    score: float = 0.0                    # 0..100
    components: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    grade: str = "F"
    acceptable: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "grade": self.grade,
            "acceptable": self.acceptable,
            "components": {k: round(v, 2) for k, v in self.components.items()},
            "penalties": {k: round(v, 4) for k, v in self.penalties.items()},
            "notes": self.notes,
        }

    def summary(self) -> str:
        return f"TRADE QUALITY: {self.score:.0f}/100 ({self.grade})"


def _grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 72:
        return "B"
    if score >= 60:
        return "C"
    if score >= 45:
        return "D"
    return "F"


def compute_trade_quality(
    ensemble: Any = None,
    regime: Regime = Regime.UNKNOWN,
    order_flow: Any = None,
    liquidity: Any = None,
    rr: float = 0.0,
    min_rr: float = 2.0,
    data_risk: Any = None,
    model_risk: Any = None,
    execution_risk: Any = None,
    portfolio_headroom: float = 1.0,
    minimum: float = 60.0,
) -> TradeQuality:
    """Blend everything into a single comparable quality figure."""

    quality = TradeQuality()
    components: dict[str, float] = {}
    notes: list[str] = []

    direction = 0
    if ensemble is not None and getattr(ensemble, "signal", None) is not None:
        direction = ensemble.signal.direction

    # --- signal ----------------------------------------------------------
    if ensemble is not None:
        # Ensemble confidence already *is* margin scaled by participation, so
        # re-adding agreement and participation here applied the same two
        # quantities a second time and dragged every score down.  Data quality
        # is a genuinely separate axis: a strong vote taken on thin data is
        # worth less than the same vote on good data, and that is not
        # represented anywhere else in this component.
        quality_factor = 0.6 + 0.4 * _clip(
            float(getattr(ensemble, "data_quality", 1.0) or 0.0), 0.0, 1.0
        )
        components["signal"] = 100.0 * ensemble.confidence * quality_factor
    else:
        components["signal"] = 0.0

    # --- regime fit --------------------------------------------------------
    regime_scores = {
        Regime.TREND_UP: 90.0,
        Regime.TREND_DOWN: 90.0,
        Regime.BREAKOUT: 75.0,
        Regime.RANGE: 65.0,
        Regime.LOW_VOLATILITY: 50.0,
        Regime.TRANSITION: 40.0,
        Regime.HIGH_VOLATILITY: 10.0,
        Regime.UNKNOWN: 15.0,
    }
    components["regime_fit"] = regime_scores.get(regime, 40.0)

    # --- order flow ---------------------------------------------------------
    if order_flow is not None and getattr(order_flow, "data_quality", 0) > 0.2:
        flow_direction = order_flow.direction
        if direction and flow_direction == direction:
            components["order_flow"] = 60.0 + 40.0 * order_flow.confidence
            notes.append("order flow confirms the direction")
        elif direction and flow_direction == -direction:
            components["order_flow"] = 30.0 - 25.0 * order_flow.confidence
            notes.append("order flow opposes the direction")
        else:
            components["order_flow"] = 50.0
        components["order_flow"] *= 0.5 + 0.5 * order_flow.data_quality
    else:
        components["order_flow"] = 45.0
        notes.append("no usable order-flow read")

    # --- liquidity target ----------------------------------------------------
    if liquidity is not None and direction:
        pull_with = liquidity.pull(direction)
        pull_against = liquidity.pull(-direction)
        total = pull_with + pull_against
        if total > 0:
            share = pull_with / total
            components["liquidity"] = 100.0 * share
            if share > 0.6:
                notes.append("liquidity sits in the direction of the trade")
            elif share < 0.4:
                notes.append("most nearby liquidity sits against the trade")
        else:
            components["liquidity"] = 50.0
    else:
        components["liquidity"] = 50.0

    # --- reward:risk ----------------------------------------------------------
    if rr > 0 and min_rr > 0:
        ratio = rr / min_rr
        # Saturating: 2x the minimum is good, 4x is not twice as good.
        components["reward_risk"] = 100.0 * min(1.0, 0.5 + 0.5 * (1 - 1 / max(ratio, 0.1)))
    else:
        # No reward:risk means there is no proposal to measure -- the signal
        # engine declined before geometry was computed.  Scoring that as zero
        # treated "not measured" as "measured and terrible", cost 14 points of
        # quality, and made the breakdown blame the wrong thing.  Unknown is
        # neutral.
        components["reward_risk"] = 50.0
        notes.append("no reward:risk to score - no entry was proposed")

    # --- portfolio headroom ----------------------------------------------------
    components["portfolio_headroom"] = 100.0 * max(0.0, min(portfolio_headroom, 1.0))

    # --- weighted base ---------------------------------------------------------
    total_weight = sum(COMPONENT_WEIGHTS.values())
    base = sum(
        components.get(name, 50.0) * weight
        for name, weight in COMPONENT_WEIGHTS.items()
    ) / total_weight

    # --- risk penalties (multiplicative, never additive bonuses) ---------------
    penalties: dict[str, float] = {}
    multiplier = 1.0
    for label, report in (
        ("data_risk", data_risk),
        ("model_risk", model_risk),
        ("execution_risk", execution_risk),
    ):
        score = float(getattr(report, "score", 0.0) or 0.0) if report is not None else 0.0
        penalties[label] = score
        # A 0.5 risk score costs 25% of the quality; 1.0 costs 50%.
        multiplier *= 1.0 - 0.5 * score
        if score >= 0.4:
            notes.append(f"{label.replace('_', ' ')} is elevated ({score:.2f})")

    quality.components = {k: _clip(v, 0.0, 100.0) for k, v in components.items()}
    quality.penalties = penalties
    # Rounded once, here, so every consumer formats the same number.  The
    # rejection text renders `score` directly while the journal stores
    # `as_dict()`'s `round(score, 2)`; with a raw score near a .x95 boundary
    # those two went through Python's round-half-to-even at different
    # precisions and disagreed by one point in either direction -- the journal
    # showed "quality 42" above a reason saying "trade quality 41 below the 55
    # minimum".  Cosmetic, but an audit trail that contradicts itself is not
    # much of an audit trail.
    quality.score = round(_clip(base * multiplier, 0.0, 100.0), 2)
    quality.grade = _grade(quality.score)
    quality.acceptable = quality.score >= minimum
    quality.notes = notes
    return quality


def _clip(value: float, low: float, high: float) -> float:
    if value != value:
        return low
    return low if value < low else high if value > high else value


__all__ = ["TradeQuality", "compute_trade_quality", "COMPONENT_WEIGHTS"]
