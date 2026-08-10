"""Opportunity ranking.

Turns the component scores into a single ``OPPORTUNITY_SCORE`` per symbol and
orders the shortlist.  The ranking answers "where should attention go?" - it
does **not** grant permission to trade.  A symbol can rank first and still be
rejected, and that is the normal case: the ``status`` field records which.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

from app.domain import Bias, Regime, Side
from app.signals.trade_proposal import Decision, TradeProposal

if TYPE_CHECKING:
    from app.scanner.scanner import SymbolAnalysis
    from app.signals.scoring import ScoreBreakdown


@dataclass(slots=True)
class OpportunityScores:
    technical: float = 50.0
    structure: float = 50.0
    ict: float = 50.0
    rtm: float = 50.0
    momentum: float = 50.0
    volume: float = 50.0
    volatility: float = 50.0
    htf_alignment: float = 50.0
    fundamental: float = 50.0
    ml_probability: float = 0.0
    risk_reward: float = 0.0

    @classmethod
    def from_breakdown(cls, breakdown: ScoreBreakdown) -> "OpportunityScores":
        return cls(
            technical=breakdown.technical,
            structure=breakdown.structure,
            ict=breakdown.ict,
            rtm=breakdown.rtm,
            momentum=breakdown.momentum,
            volume=breakdown.volume,
            volatility=breakdown.volatility,
            htf_alignment=breakdown.htf_alignment,
            fundamental=breakdown.fundamental,
            ml_probability=breakdown.ml_probability,
            risk_reward=breakdown.risk_reward,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "technical_score": round(self.technical, 2),
            "structure_score": round(self.structure, 2),
            "ict_score": round(self.ict, 2),
            "rtm_score": round(self.rtm, 2),
            "momentum_score": round(self.momentum, 2),
            "volume_score": round(self.volume, 2),
            "volatility_score": round(self.volatility, 2),
            "htf_alignment_score": round(self.htf_alignment, 2),
            "fundamental_score": round(self.fundamental, 2),
            "ml_probability": round(self.ml_probability, 4),
            "rr_score": round(self.risk_reward, 2),
        }


@dataclass(slots=True)
class Opportunity:
    symbol: str
    opportunity_score: float
    scores: OpportunityScores
    proposal: TradeProposal
    analysis: SymbolAnalysis
    status: str
    trend: Bias
    regime: Regime

    @property
    def side(self) -> Side | None:
        return self.proposal.side

    @property
    def confidence(self) -> float:
        return self.proposal.confidence

    @property
    def rr(self) -> float:
        return self.proposal.rr

    @property
    def tradable(self) -> bool:
        return self.proposal.is_entry

    def as_row(self) -> dict[str, object]:
        """Row for the ``opportunities`` table and the scanner dashboard."""

        row: dict[str, object] = {
            "ts": self.proposal.ts,
            "symbol": self.symbol,
            "opportunity_score": round(self.opportunity_score, 2),
            "trend": self.trend.value,
            "regime": self.regime.value,
            "status": self.status,
        }
        row.update(self.scores.as_dict())
        return row

    def as_dict(self) -> dict[str, object]:
        row = self.as_row()
        row.update(
            {
                "side": self.side.value if self.side else None,
                "confidence": round(self.confidence, 4),
                "rr": round(self.rr, 2),
                "entry": self.proposal.entry,
                "stop_loss": self.proposal.stop_loss,
                "tp1": self.proposal.tp1,
                "tp2": self.proposal.tp2,
                "tp3": self.proposal.tp3,
                "reasons": self.proposal.reasons,
                "rejections": self.proposal.rejections,
            }
        )
        return row


def status_for(proposal: TradeProposal, rank: int) -> str:
    """Short label shown in the scanner dashboard."""

    if proposal.decision is Decision.ENTER:
        return "BEST SETUP" if rank == 0 else "TRADABLE"
    if not proposal.rejections:
        return "WATCH"
    first = proposal.rejections[0]
    if "confidence" in first:
        return "LOW CONFIDENCE"
    if "reward:risk" in first:
        return "POOR R:R"
    if "regime" in first:
        return "REGIME BLOCK"
    if "spread" in first or "liquidity" in first or "depth" in first:
        return "ILLIQUID"
    if "volatility" in first:
        return "VOLATILITY"
    if "fundamental" in first or "anomaly" in first:
        return "RISK BLOCK"
    if "context" in first or "thesis" in first or "CONFLICT" in first:
        return "NO SETUP"
    return "REJECTED"


def build_opportunity(
    analysis: SymbolAnalysis, proposal: TradeProposal, rank: int = 0
) -> Opportunity:
    scores = OpportunityScores.from_breakdown(proposal.scores)
    return Opportunity(
        symbol=analysis.symbol,
        opportunity_score=proposal.opportunity_score,
        scores=scores,
        proposal=proposal,
        analysis=analysis,
        status=status_for(proposal, rank),
        trend=analysis.htf_bias,
        regime=analysis.regime.regime,
    )


def rank_opportunities(
    pairs: Iterable[tuple[SymbolAnalysis, TradeProposal]],
) -> list[Opportunity]:
    """Score, sort and label a batch of evaluated symbols.

    Tradable proposals always sort above non-tradable ones regardless of raw
    score, because a 95-scoring symbol that failed a hard gate is not an
    opportunity - it is a rejection with a nice number attached.
    """

    opportunities = [
        build_opportunity(analysis, proposal) for analysis, proposal in pairs
    ]
    opportunities.sort(
        key=lambda o: (o.tradable, o.opportunity_score, o.confidence), reverse=True
    )
    for rank, opportunity in enumerate(opportunities):
        opportunity.status = status_for(opportunity.proposal, rank)
    return opportunities


__all__ = [
    "Opportunity",
    "OpportunityScores",
    "rank_opportunities",
    "build_opportunity",
    "status_for",
]
