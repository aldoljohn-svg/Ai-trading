"""Trailing stop logic.

Trailing only activates once the trade has banked a meaningful move
(``activate_r``).  Trailing from the first tick converts every winner into a
scratch: normal noise takes the stop out before the move develops.

Two trailing modes:

``atr``
    Follow price at a fixed ATR distance.  Volatility-aware and smooth.
``chandelier``
    Follow the highest high (long) or lowest low (short) since entry, less an
    ATR multiple.  Gives back less at the top of a strong move.

Both are one-directional by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.domain import Candle, Side
from app.portfolio.portfolio_manager import ManagedPosition
from app.position_manager.stop_manager import StopUpdate


def trailing_stop_price(
    side: Side, reference: float, atr: float, multiplier: float
) -> float:
    """Stop placed ``multiplier`` ATR behind ``reference``."""

    distance = multiplier * atr
    return reference - distance if side is Side.LONG else reference + distance


@dataclass(slots=True)
class TrailingStop:
    activate_r: float = 1.8
    atr_multiplier: float = 1.6
    mode: str = "chandelier"        # "chandelier" | "atr"
    tighten_after_tp2: float = 1.1  # ATR multiplier once TP2 is banked

    def active_multiplier(self, position: ManagedPosition) -> float:
        """Tighten the trail as more of the position is realised."""

        if position.tp2_done:
            return self.tighten_after_tp2
        if position.tp1_done:
            return self.atr_multiplier * 0.85
        return self.atr_multiplier

    def should_activate(self, position: ManagedPosition, price: float) -> bool:
        if position.trailing_active:
            return True
        if position.risk_per_unit <= 0:
            return False
        return position.r_multiple(price) >= self.activate_r

    def compute(
        self,
        position: ManagedPosition,
        price: float,
        atr: float,
        candles: Sequence[Candle] | None = None,
    ) -> StopUpdate | None:
        """Return a trailing stop update, or ``None`` if it should not move."""

        if atr <= 0 or price <= 0:
            return None
        if not self.should_activate(position, price):
            return None

        multiplier = self.active_multiplier(position)

        if self.mode == "chandelier" and candles:
            window = [c for c in candles if c.ts >= position.opened_at] or list(candles[-40:])
            reference = (
                max(c.high for c in window)
                if position.side is Side.LONG
                else min(c.low for c in window)
            )
        else:
            reference = price

        new_stop = trailing_stop_price(position.side, reference, atr, multiplier)

        # Never trail past the current price - that would close the position at
        # market rather than protect it.
        if position.side is Side.LONG and new_stop >= price:
            new_stop = price - 0.25 * atr
        elif position.side is Side.SHORT and new_stop <= price:
            new_stop = price + 0.25 * atr

        reason = (
            f"trailing {multiplier:g}x ATR behind "
            f"{'the high' if position.side is Side.LONG else 'the low'} "
            f"({reference:.6g})"
        )
        return StopUpdate(new_stop=new_stop, reason=reason, kind="trailing")


__all__ = ["TrailingStop", "trailing_stop_price"]
