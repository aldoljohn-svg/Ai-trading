"""The trade proposal - the contract between analysis and execution.

A proposal is produced for *every* evaluated symbol, including the ones that
are rejected.  That is deliberate: the rejections are as valuable as the
entries, they are persisted to the ``signals`` table, and they are what the
``/signals`` and ``/ai`` Telegram commands read back.

Nothing here decides position size or touches the exchange.  A proposal is a
priced idea; the risk engine decides whether it becomes a trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.domain import Bias, Regime, Side
from app.signals.scoring import ScoreBreakdown


class Decision(str, Enum):
    ENTER = "ENTER"
    NO_TRADE = "NO_TRADE"

    @property
    def emoji(self) -> str:
        return "✅" if self is Decision.ENTER else "⛔"


@dataclass(slots=True)
class TradeProposal:
    symbol: str
    side: Side | None
    decision: Decision
    entry: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    rr: float = 0.0                     # reward:risk to TP2 - the gated figure
    rr_weighted: float = 0.0            # expected R including the partial ladder
    confidence: float = 0.0             # 0..1, never 1.0
    atr: float = 0.0
    stop_distance: float = 0.0
    suggested_leverage: float = 1.0
    regime: Regime = Regime.UNKNOWN
    htf_bias: Bias = Bias.NEUTRAL
    alignment: float = 0.0
    p_long: float = 0.0
    p_short: float = 0.0
    p_no_trade: float = 1.0
    scores: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    reasons: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    stop_rationale: str = ""
    target_rationale: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    ts: int = 0

    # -- helpers ----------------------------------------------------------

    @property
    def is_entry(self) -> bool:
        return self.decision is Decision.ENTER and self.side is not None

    @property
    def opportunity_score(self) -> float:
        return self.scores.blended()

    @property
    def risk_per_unit(self) -> float:
        """Absolute price distance between entry and stop."""

        return abs(self.entry - self.stop_loss)

    def r_multiple_at(self, price: float) -> float:
        risk = self.risk_per_unit
        if risk <= 0 or self.side is None:
            return 0.0
        return (price - self.entry) * self.side.sign / risk

    def reject(self, reason: str) -> "TradeProposal":
        """Record a rejection and force the decision to NO_TRADE."""

        if reason not in self.rejections:
            self.rejections.append(reason)
        self.decision = Decision.NO_TRADE
        return self

    def reason_summary(self, limit: int = 6) -> str:
        return "\n+\n".join(self.reasons[:limit]) if self.reasons else "-"

    def as_record(self) -> dict[str, Any]:
        """Row shape for the ``signals`` table."""

        return {
            "ts": self.ts,
            "symbol": self.symbol,
            "side": self.side.value if self.side else None,
            "decision": self.decision.value,
            "confidence": self.confidence,
            "rr": self.rr,
            "entry": self.entry or None,
            "stop_loss": self.stop_loss or None,
            "tp1": self.tp1 or None,
            "tp2": self.tp2 or None,
            "tp3": self.tp3 or None,
            "regime": self.regime.value,
            "htf_bias": self.htf_bias.value,
            "p_long": self.p_long,
            "p_short": self.p_short,
            "p_no_trade": self.p_no_trade,
            "reasons": self.reasons,
            "rejections": self.rejections,
            "scores": self.scores.as_dict(),
            "features": self.features,
        }

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view for the dashboard and Telegram."""

        record = self.as_record()
        record.update(
            {
                "rr_weighted": self.rr_weighted,
                "atr": self.atr,
                "stop_distance": self.stop_distance,
                "suggested_leverage": self.suggested_leverage,
                "alignment": self.alignment,
                "opportunity_score": self.opportunity_score,
                "stop_rationale": self.stop_rationale,
                "target_rationale": self.target_rationale,
            }
        )
        return record


__all__ = ["TradeProposal", "Decision"]
