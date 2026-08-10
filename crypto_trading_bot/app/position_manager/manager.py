"""Position manager - the loop that watches every open trade.

Runs far more often than the scanner (seconds, not minutes) and is responsible
for everything that happens *after* entry:

* stop hit -> close
* target hit -> partial or full close
* break-even and trailing stop maintenance
* structure invalidation -> close
* regime turning dangerous -> de-risk
* fundamental danger -> de-risk or close
* portfolio breach -> reduce
* adopted positions with no stop -> protective stop

Every decision produces a :class:`PositionAction` with a human-readable reason,
which is what gets sent to Telegram and stored as ``why_exited``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from app.config import Settings
from app.domain import Bias, Candle, Regime, Side, Timeframe
from app.logger import get_logger
from app.portfolio.portfolio_manager import ManagedPosition, PortfolioManager
from app.position_manager.stop_manager import StopManager, StopUpdate
from app.position_manager.target_manager import TargetHit, TargetManager
from app.position_manager.trailing import TrailingStop

log = get_logger(__name__)


class ActionKind(str, Enum):
    NONE = "NONE"
    STOP_HIT = "STOP_HIT"
    TARGET_HIT = "TARGET_HIT"
    MOVE_STOP = "MOVE_STOP"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    CLOSE = "CLOSE"
    REDUCE_RISK = "REDUCE_RISK"


@dataclass(slots=True)
class PositionAction:
    kind: ActionKind
    position: ManagedPosition
    reason: str
    close_fraction: float = 0.0
    new_stop: float = 0.0
    target_level: int = 0
    price: float = 0.0
    notify: bool = True

    @property
    def closes_anything(self) -> bool:
        return self.close_fraction > 0


@dataclass(slots=True)
class ManagementContext:
    """Fresh market context for one symbol, supplied by the orchestrator."""

    price: float
    atr: float = 0.0
    high: float | None = None
    low: float | None = None
    candles: Sequence[Candle] = field(default_factory=list)
    structure_bias: Bias = Bias.NEUTRAL
    regime: Regime = Regime.UNKNOWN
    protective_swing: float | None = None
    fundamental_danger: bool = False
    fundamental_reason: str = ""
    anomaly: str = ""
    volatility_ratio: float = 1.0


class PositionManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stops = StopManager(
            breakeven_at_r=settings.breakeven_at_r,
            breakeven_offset_r=settings.breakeven_offset_r,
        )
        self.targets = TargetManager(
            tp1_close_pct=settings.tp1_close_pct,
            tp2_close_pct=settings.tp2_close_pct,
        )
        self.trailing = TrailingStop(
            activate_r=settings.trailing_activate_r,
            atr_multiplier=settings.trailing_atr_mult,
        )

    # -- single position --------------------------------------------------

    def evaluate(
        self,
        position: ManagedPosition,
        context: ManagementContext,
        risk_reduction: float = 0.0,
    ) -> list[PositionAction]:
        """Decide what to do with one position, in priority order."""

        actions: list[PositionAction] = []
        price = context.price
        if price <= 0:
            return actions

        # --- 1. stop loss (always first) ---------------------------------
        low = context.low if context.low is not None else price
        high = context.high if context.high is not None else price
        if position.stop_loss > 0:
            breached = (
                low <= position.stop_loss
                if position.side is Side.LONG
                else high >= position.stop_loss
            )
            if breached:
                actions.append(
                    PositionAction(
                        kind=ActionKind.STOP_HIT,
                        position=position,
                        reason=f"stop loss hit at {position.stop_loss:.6g}",
                        close_fraction=1.0,
                        price=position.stop_loss,
                    )
                )
                return actions

        # --- 2. emergency de-risk ----------------------------------------
        if risk_reduction >= 1.0:
            actions.append(
                PositionAction(
                    kind=ActionKind.CLOSE,
                    position=position,
                    reason="emergency stop - closing all positions",
                    close_fraction=1.0,
                    price=price,
                )
            )
            return actions

        # --- 3. hard exits from context ----------------------------------
        exit_reason = self._invalidation_reason(position, context)
        if exit_reason:
            actions.append(
                PositionAction(
                    kind=ActionKind.CLOSE,
                    position=position,
                    reason=exit_reason,
                    close_fraction=1.0,
                    price=price,
                )
            )
            return actions

        # --- 4. targets ---------------------------------------------------
        hit = self.targets.check(position, price, high=high, low=low)
        if hit is not None:
            actions.append(
                PositionAction(
                    kind=ActionKind.TARGET_HIT,
                    position=position,
                    reason=hit.reason,
                    close_fraction=hit.close_fraction,
                    target_level=hit.level,
                    price=hit.price,
                )
            )
            # TP1 additionally forces the stop to break-even, handled below.

        # --- 5. partial de-risk -------------------------------------------
        if 0 < risk_reduction < 1.0 and not actions:
            actions.append(
                PositionAction(
                    kind=ActionKind.REDUCE_RISK,
                    position=position,
                    reason=(
                        f"portfolio de-risking: closing {risk_reduction:.0%} of "
                        "each open position"
                    ),
                    close_fraction=risk_reduction,
                    price=price,
                )
            )

        # --- 6. stop maintenance ------------------------------------------
        update = self._best_stop(position, context, forced_breakeven=hit is not None and hit.level >= 1)
        if update is not None:
            actions.append(
                PositionAction(
                    kind=ActionKind.MOVE_STOP,
                    position=position,
                    reason=update.reason,
                    new_stop=update.new_stop,
                    price=price,
                    notify=update.kind in {"breakeven", "protective"}
                    or position.trailing_active is False,
                )
            )
        return actions

    def _best_stop(
        self,
        position: ManagedPosition,
        context: ManagementContext,
        forced_breakeven: bool = False,
    ) -> StopUpdate | None:
        trailing_update = self.trailing.compute(
            position, context.price, context.atr, context.candles
        )
        update = self.stops.best(
            position,
            price=context.price,
            atr=context.atr,
            swing_price=context.protective_swing,
            trailing=trailing_update,
        )
        if update is None and forced_breakeven and not position.breakeven_done:
            risk = position.risk_per_unit
            if risk > 0:
                offset = self.settings.breakeven_offset_r * risk * position.side.sign
                candidate = StopUpdate(
                    new_stop=position.entry_price + offset,
                    reason="TP1 banked - stop moved to break-even",
                    kind="breakeven",
                )
                if self.stops.is_improvement(
                    position, candidate.new_stop
                ) and self.stops.is_valid(position, candidate.new_stop, context.price):
                    return candidate
        return update

    def _invalidation_reason(
        self, position: ManagedPosition, context: ManagementContext
    ) -> str:
        """Reasons to exit that have nothing to do with price hitting a level."""

        # Structure has flipped against the position.
        opposite = Bias.BEARISH if position.side is Side.LONG else Bias.BULLISH
        if context.structure_bias is opposite and position.r_multiple(context.price) < 0.5:
            return (
                f"market structure flipped to {context.structure_bias.value} - "
                "the entry thesis is invalidated"
            )

        if context.fundamental_danger:
            return f"fundamental danger: {context.fundamental_reason or 'market-wide risk'}"

        if context.anomaly:
            return f"price data anomaly: {context.anomaly}"

        if (
            context.regime is Regime.HIGH_VOLATILITY
            and context.volatility_ratio >= 2.5
            and position.r_multiple(context.price) < 0
        ):
            return (
                f"volatility spiked to {context.volatility_ratio:.1f}x normal while "
                "the position is underwater - stops cannot be trusted here"
            )

        return ""

    # -- application ------------------------------------------------------

    def apply_stop(self, action: PositionAction) -> bool:
        """Mutate the position for a MOVE_STOP action."""

        if action.kind is not ActionKind.MOVE_STOP:
            return False
        position = action.position
        update = StopUpdate(
            new_stop=action.new_stop,
            reason=action.reason,
            kind="trailing" if "trailing" in action.reason else (
                "breakeven" if "break-even" in action.reason else "structure"
            ),
        )
        return self.stops.apply(position, update, action.price)

    def mark_target(self, action: PositionAction) -> None:
        if action.kind is ActionKind.TARGET_HIT:
            self.targets.mark(
                action.position,
                TargetHit(
                    level=action.target_level,
                    price=action.price,
                    close_fraction=action.close_fraction,
                    reason=action.reason,
                ),
            )

    # -- portfolio sweep --------------------------------------------------

    def evaluate_all(
        self,
        portfolio: PortfolioManager,
        contexts: Mapping[str, ManagementContext],
        risk_reduction: float = 0.0,
    ) -> list[PositionAction]:
        actions: list[PositionAction] = []
        for position in portfolio.all():
            context = contexts.get(position.symbol)
            if context is None:
                price = portfolio.price(position.symbol)
                if price <= 0:
                    continue
                context = ManagementContext(price=price)
            actions.extend(self.evaluate(position, context, risk_reduction))
        return actions


__all__ = [
    "PositionManager",
    "PositionAction",
    "ActionKind",
    "ManagementContext",
]
