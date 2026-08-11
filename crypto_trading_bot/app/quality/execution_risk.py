"""EXECUTION_RISK_SCORE - can this actually be traded at an acceptable cost?

A signal that is right about direction still loses money if the spread, the
slippage and the fees consume the edge, or if the venue is unhealthy at the
moment of execution.

Explicitly compares expected round-trip cost against the trade's own expected
reward: a 0.15% cost is trivial on a 3R trade and fatal on a scalp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

HARD_LIMIT = 0.70


@dataclass(slots=True)
class ExecutionRiskReport:
    score: float = 0.0
    spread_pct: float = 0.0
    slippage_pct: float = 0.0
    fee_pct: float = 0.0
    total_cost_pct: float = 0.0
    cost_to_reward: float = 0.0        # round-trip cost / expected reward
    latency_ms: float = 0.0
    fillable: bool = True
    recommended_order_type: str = "market"
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.score >= HARD_LIMIT

    @property
    def level(self) -> str:
        if self.score >= HARD_LIMIT:
            return "HIGH"
        if self.score >= 0.4:
            return "MEDIUM"
        return "LOW"

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "level": self.level,
            "blocked": self.blocked,
            "spread_pct": round(self.spread_pct, 6),
            "slippage_pct": round(self.slippage_pct, 6),
            "fee_pct": round(self.fee_pct, 6),
            "total_cost_pct": round(self.total_cost_pct, 6),
            "cost_to_reward": round(self.cost_to_reward, 4),
            "latency_ms": round(self.latency_ms, 1),
            "fillable": self.fillable,
            "recommended_order_type": self.recommended_order_type,
            "problems": self.problems,
            "notes": self.notes,
        }


def assess_execution_risk(
    microstructure: Any,
    taker_fee: float = 0.0006,
    maker_fee: float = 0.0002,
    expected_reward_pct: float = 0.0,
    latency_ms: float = 0.0,
    max_latency_ms: float = 2000.0,
    exchange_health: Any = None,
    max_cost_to_reward: float = 0.25,
) -> ExecutionRiskReport:
    """Score how hostile execution conditions are right now."""

    report = ExecutionRiskReport(latency_ms=latency_ms)
    penalties: list[float] = []

    if microstructure is None:
        report.score = 0.8
        report.problems.append("no microstructure read available")
        return report

    report.spread_pct = getattr(microstructure, "spread_pct", 0.0)
    report.slippage_pct = getattr(microstructure, "expected_slippage_pct", 0.0)
    report.fillable = getattr(microstructure, "fillable", True)
    report.fee_pct = taker_fee * 2                  # round trip

    # Half the spread is paid on entry and again on exit.
    report.total_cost_pct = report.fee_pct + report.spread_pct + report.slippage_pct * 2

    for problem in getattr(microstructure, "problems", []) or []:
        report.problems.append(problem)
        penalties.append(0.55)

    if not report.fillable:
        penalties.append(0.9)

    if getattr(microstructure, "thin_liquidity", False):
        penalties.append(0.5)
    if getattr(microstructure, "spread_expanded", False):
        penalties.append(0.35)
        report.notes.append("spread is unusually wide versus its own norm")

    # --- cost against reward ---------------------------------------------
    if expected_reward_pct > 0:
        report.cost_to_reward = report.total_cost_pct / expected_reward_pct
        if report.cost_to_reward > max_cost_to_reward:
            penalties.append(
                min(0.85, 0.4 + (report.cost_to_reward - max_cost_to_reward))
            )
            report.problems.append(
                f"round-trip cost {report.total_cost_pct:.3%} is "
                f"{report.cost_to_reward:.0%} of the expected {expected_reward_pct:.2%} move"
            )
        elif report.cost_to_reward > max_cost_to_reward * 0.6:
            penalties.append(0.15)
            report.notes.append(
                f"costs consume {report.cost_to_reward:.0%} of expected reward"
            )

    # --- latency -------------------------------------------------------------
    if latency_ms > max_latency_ms:
        penalties.append(min(0.7, latency_ms / (max_latency_ms * 3)))
        report.problems.append(f"execution latency {latency_ms:.0f}ms is abnormal")
    elif latency_ms > max_latency_ms * 0.5:
        penalties.append(0.15)
        report.notes.append(f"latency {latency_ms:.0f}ms is elevated")

    # --- venue health ---------------------------------------------------------
    if exchange_health is not None:
        healthy = bool(
            exchange_health.get("ok", True)
            if isinstance(exchange_health, dict)
            else getattr(exchange_health, "ok", True)
        )
        if not healthy:
            penalties.append(0.85)
            report.problems.append("exchange connectivity is unhealthy")

    # --- order type recommendation ---------------------------------------------
    # Where the spread dominates the cost, resting passively is worth more than
    # immediacy - provided the setup is not a breakout that will leave without us.
    if report.spread_pct > taker_fee * 2 and report.fillable:
        report.recommended_order_type = "limit"
        report.notes.append(
            "spread exceeds the fee - a passive limit order would save "
            f"~{(taker_fee - maker_fee) + report.spread_pct / 2:.4%}"
        )

    report.score = _noisy_or(penalties)
    return report


def _noisy_or(penalties: list[float]) -> float:
    survival = 1.0
    for penalty in penalties:
        survival *= 1.0 - max(0.0, min(penalty, 1.0))
    return 1.0 - survival


__all__ = ["ExecutionRiskReport", "assess_execution_risk", "HARD_LIMIT"]
