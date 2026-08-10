"""Signal generation: from analysis to an approved, priced trade proposal."""

from app.signals.scoring import ScoreBreakdown, score_direction
from app.signals.signal_engine import SignalEngine
from app.signals.trade_proposal import Decision, TradeProposal

__all__ = [
    "SignalEngine",
    "TradeProposal",
    "Decision",
    "ScoreBreakdown",
    "score_direction",
]
