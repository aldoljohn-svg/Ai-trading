"""Stop management.

One invariant governs everything here: **a stop only ever moves in the
direction that reduces risk.**  A long's stop can rise and never fall; a
short's can fall and never rise.  Widening a stop to "give the trade room" is
how a 0.5% risk becomes a 5% loss, so the code makes it impossible rather than
discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain import Side
from app.portfolio.portfolio_manager import ManagedPosition


@dataclass(frozen=True, slots=True)
class StopUpdate:
    new_stop: float
    reason: str
    kind: str            # "breakeven" | "trailing" | "structure" | "protective"

    @property
    def moved(self) -> bool:
        return self.new_stop > 0


class StopManager:
    def __init__(
        self,
        breakeven_at_r: float = 1.0,
        breakeven_offset_r: float = 0.1,
        protective_atr_mult: float = 2.0,
    ) -> None:
        self.breakeven_at_r = breakeven_at_r
        self.breakeven_offset_r = breakeven_offset_r
        self.protective_atr_mult = protective_atr_mult

    # -- invariant --------------------------------------------------------

    @staticmethod
    def is_improvement(position: ManagedPosition, new_stop: float) -> bool:
        """Does ``new_stop`` reduce risk relative to the current stop?"""

        if new_stop <= 0:
            return False
        if position.stop_loss <= 0:
            return True
        if position.side is Side.LONG:
            return new_stop > position.stop_loss
        return new_stop < position.stop_loss

    @staticmethod
    def is_valid(position: ManagedPosition, new_stop: float, price: float) -> bool:
        """A stop must not be placed on the wrong side of the current price."""

        if new_stop <= 0 or price <= 0:
            return False
        if position.side is Side.LONG:
            return new_stop < price
        return new_stop > price

    def apply(
        self, position: ManagedPosition, update: StopUpdate, price: float
    ) -> bool:
        """Move the stop if - and only if - it reduces risk and is placeable."""

        if not self.is_improvement(position, update.new_stop):
            return False
        if not self.is_valid(position, update.new_stop, price):
            return False
        position.stop_loss = update.new_stop
        if update.kind == "breakeven":
            position.breakeven_done = True
        elif update.kind == "trailing":
            position.trailing_active = True
        return True

    # -- candidate stops --------------------------------------------------

    def breakeven(self, position: ManagedPosition, price: float) -> StopUpdate | None:
        """Move to entry (plus a small buffer) once the trade has paid for itself."""

        if position.breakeven_done:
            return None
        risk = position.risk_per_unit
        if risk <= 0:
            return None
        if position.r_multiple(price) < self.breakeven_at_r:
            return None
        offset = self.breakeven_offset_r * risk * position.side.sign
        return StopUpdate(
            new_stop=position.entry_price + offset,
            reason=(
                f"reached {self.breakeven_at_r:g}R - stop moved to break-even "
                f"+{self.breakeven_offset_r:g}R"
            ),
            kind="breakeven",
        )

    def structure_stop(
        self, position: ManagedPosition, swing_price: float | None, atr: float
    ) -> StopUpdate | None:
        """Trail behind the most recent protective swing."""

        if swing_price is None or swing_price <= 0 or atr <= 0:
            return None
        buffer = 0.25 * atr
        new_stop = (
            swing_price - buffer if position.side is Side.LONG else swing_price + buffer
        )
        if not self.is_improvement(position, new_stop):
            return None
        return StopUpdate(
            new_stop=new_stop,
            reason=f"structure stop behind the swing at {swing_price:.6g}",
            kind="structure",
        )

    def protective(
        self, position: ManagedPosition, price: float, atr: float
    ) -> StopUpdate | None:
        """Emergency stop for an adopted position that has none.

        Placed at a wide ATR distance: the goal is to bound the loss, not to
        express a view we do not have.
        """

        if position.stop_loss > 0 or atr <= 0 or price <= 0:
            return None
        distance = self.protective_atr_mult * atr
        new_stop = (
            price - distance if position.side is Side.LONG else price + distance
        )
        return StopUpdate(
            new_stop=new_stop,
            reason=(
                f"position had no stop - placing a protective "
                f"{self.protective_atr_mult:g}x ATR stop"
            ),
            kind="protective",
        )

    def best(
        self,
        position: ManagedPosition,
        price: float,
        atr: float,
        swing_price: float | None = None,
        trailing: StopUpdate | None = None,
    ) -> StopUpdate | None:
        """Pick the tightest risk-reducing stop among all candidates."""

        candidates = [
            self.protective(position, price, atr),
            self.breakeven(position, price),
            self.structure_stop(position, swing_price, atr),
            trailing,
        ]
        valid = [
            c
            for c in candidates
            if c is not None
            and self.is_improvement(position, c.new_stop)
            and self.is_valid(position, c.new_stop, price)
        ]
        if not valid:
            return None
        if position.side is Side.LONG:
            return max(valid, key=lambda c: c.new_stop)
        return min(valid, key=lambda c: c.new_stop)


__all__ = ["StopManager", "StopUpdate"]
