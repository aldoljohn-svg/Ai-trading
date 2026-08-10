"""Take-profit management.

The ladder is TP1 / TP2 / TP3 with partial closes.  Order matters: TP1 is
banked and the stop goes to break-even, TP2 banks more and tightens the trail,
TP3 closes the runner.

Targets are evaluated against the **candle high/low**, not just the last price,
so a target that was touched between polls is not missed.  The same candle is
checked against the stop first - if a bar traded through both the stop and the
target, the pessimistic assumption (stop first) is used, because from OHLC
alone the order is genuinely unknown and assuming the favourable one is how
backtests stop matching reality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.domain import Candle, Side
from app.portfolio.portfolio_manager import ManagedPosition


@dataclass(frozen=True, slots=True)
class TargetHit:
    level: int                  # 1, 2 or 3
    price: float
    close_fraction: float       # fraction of the *current* quantity to close
    reason: str
    full_close: bool = False


class TargetManager:
    def __init__(
        self,
        tp1_close_pct: float = 0.4,
        tp2_close_pct: float = 0.35,
    ) -> None:
        self.tp1_close_pct = tp1_close_pct
        self.tp2_close_pct = tp2_close_pct

    # -- detection --------------------------------------------------------

    @staticmethod
    def reached(position: ManagedPosition, target: float, high: float, low: float) -> bool:
        if target <= 0:
            return False
        return high >= target if position.side is Side.LONG else low <= target

    def check(
        self,
        position: ManagedPosition,
        price: float,
        high: float | None = None,
        low: float | None = None,
    ) -> TargetHit | None:
        """The next target this position has reached, if any."""

        high = high if high is not None else price
        low = low if low is not None else price

        if not position.tp1_done and self.reached(position, position.tp1, high, low):
            return TargetHit(
                level=1,
                price=position.tp1,
                close_fraction=self.tp1_close_pct,
                reason=f"TP1 hit at {position.tp1:.6g}",
            )

        if (
            position.tp1_done
            and not position.tp2_done
            and self.reached(position, position.tp2, high, low)
        ):
            # The fraction is relative to what is *left*, so the overall plan
            # still leaves a runner.
            remaining = 1.0 - self.tp1_close_pct
            fraction = self.tp2_close_pct / remaining if remaining > 0 else 1.0
            return TargetHit(
                level=2,
                price=position.tp2,
                close_fraction=min(fraction, 0.9),
                reason=f"TP2 hit at {position.tp2:.6g}",
            )

        if (
            position.tp2_done
            and self.reached(position, position.tp3, high, low)
        ):
            return TargetHit(
                level=3,
                price=position.tp3,
                close_fraction=1.0,
                reason=f"TP3 hit at {position.tp3:.6g}",
                full_close=True,
            )

        return None

    @staticmethod
    def mark(position: ManagedPosition, hit: TargetHit) -> None:
        if hit.level == 1:
            position.tp1_done = True
        elif hit.level == 2:
            position.tp2_done = True

    # -- bar replay -------------------------------------------------------

    @staticmethod
    def stop_hit_first(
        position: ManagedPosition, candle: Candle
    ) -> bool:
        """Did this bar trade through the stop?

        When a single bar spans both the stop and a target, we assume the stop
        went first.  It is the conservative reading and the only one that keeps
        simulated results honest.
        """

        if position.stop_loss <= 0:
            return False
        if position.side is Side.LONG:
            return candle.low <= position.stop_loss
        return candle.high >= position.stop_loss


__all__ = ["TargetManager", "TargetHit"]
