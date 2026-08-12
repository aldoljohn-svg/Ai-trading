"""Execution engine.

Turns an approved :class:`~app.risk.risk_engine.RiskDecision` into a live
position, and turns exit instructions into reduce-only orders.

The **mode gate** lives here and is checked on every single order:

.. code-block:: text

    TRADING_MODE=live      -> orders go to MEXC
    TRADING_MODE=paper     -> orders go to the paper engine, never the venue
    TRADING_MODE=backtest  -> the execution engine refuses to run at all

The gate is enforced twice - once here, and again inside
:class:`~app.exchange.mexc.MexcFuturesExchange` via ``allow_trading``.  Two
independent checks, because the cost of getting this wrong is real money.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, TradingMode
from app.domain import ContractSpec, OrderIntent, OrderType, Side
from app.exchange.base import ExchangeError
from app.execution.order_manager import Broker, OrderManager, TrackedOrder
from app.logger import get_logger
from app.portfolio.portfolio_manager import ManagedPosition, PortfolioManager
from app.risk.risk_engine import RiskDecision
from app.signals.trade_proposal import TradeProposal

log = get_logger(__name__)


class TradingModeViolation(RuntimeError):
    """Raised if anything attempts to send a real order outside LIVE mode."""


@dataclass(slots=True)
class ExecutionResult:
    ok: bool
    position: ManagedPosition | None = None
    order: TrackedOrder | None = None
    filled_quantity: float = 0.0
    average_price: float = 0.0
    error: str = ""
    notes: list[str] = field(default_factory=list)


class ExecutionEngine:
    def __init__(
        self,
        settings: Settings,
        broker: Broker,
        portfolio: PortfolioManager,
        repositories: Any = None,
        notifier: Any = None,
        live_enabled: bool = False,
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.portfolio = portfolio
        self.repositories = repositories
        self.notifier = notifier
        #: Set by the orchestrator only after every pre-flight check passed.
        self.live_enabled = live_enabled
        self.orders = OrderManager(
            broker=broker,
            repositories=repositories,
            mode=settings.trading_mode.value,
            poll_interval=settings.order_poll_seconds,
            instance=settings.instance_name,
        )

    # -- gate -------------------------------------------------------------

    def _assert_can_trade(self) -> None:
        mode = self.settings.trading_mode
        if mode is TradingMode.BACKTEST:
            raise TradingModeViolation(
                "the execution engine cannot run in BACKTEST mode; the backtest "
                "engine simulates its own fills"
            )
        if mode is TradingMode.LIVE and not self.live_enabled:
            raise TradingModeViolation(
                "LIVE mode is configured but live trading has not been enabled - "
                "pre-flight checks must pass first"
            )

    @property
    def is_live(self) -> bool:
        return self.settings.trading_mode is TradingMode.LIVE and self.live_enabled

    # -- entry ------------------------------------------------------------

    async def open_position(
        self,
        decision: RiskDecision,
        spec: ContractSpec,
        atr: float = 0.0,
    ) -> ExecutionResult:
        self._assert_can_trade()

        proposal = decision.proposal
        size = decision.size
        if proposal is None or size is None or not decision.approved:
            return ExecutionResult(ok=False, error="risk engine did not approve this trade")
        if proposal.side is None:
            return ExecutionResult(ok=False, error="proposal has no side")

        if self.portfolio.get(proposal.symbol) is not None:
            return ExecutionResult(
                ok=False, error=f"already holding {proposal.symbol}"
            )

        # Leverage must be set before the position exists, not after.
        try:
            await self.broker.set_leverage(proposal.symbol, decision.leverage)
        except ExchangeError as exc:
            log.warning("could not set leverage for %s: %s", proposal.symbol, exc)

        trade_id = self._record_trade_open(proposal, decision, spec)

        try:
            tracked = await self.orders.place(
                symbol=proposal.symbol,
                side=proposal.side,
                intent=OrderIntent.OPEN,
                quantity=size.contracts,
                order_type=OrderType.MARKET,
                leverage=decision.leverage,
                trade_id=trade_id,
            )
        except ExchangeError as exc:
            return ExecutionResult(ok=False, error=str(exc))

        tracked = await self.orders.wait_for_fill(tracked)

        if tracked.filled_quantity <= 0:
            return ExecutionResult(
                ok=False,
                order=tracked,
                error=(
                    f"entry order did not fill (status {tracked.status.value}); "
                    "no position was opened"
                ),
            )

        fill_price = tracked.average_price or proposal.entry
        # Risk scales with what actually filled.  Recording the planned risk on a
        # partial fill overstates open exposure, and that inflated figure is what
        # the portfolio limits and the correlation cluster cap are measured
        # against -- so a half-filled entry would quietly crowd out the next
        # trade by risk the account is not carrying.
        fill_ratio = (
            tracked.filled_quantity / size.contracts if size.contracts > 0 else 1.0
        )
        position = ManagedPosition(
            symbol=proposal.symbol,
            side=proposal.side,
            quantity=tracked.filled_quantity,
            entry_price=fill_price,
            leverage=decision.leverage,
            contract_size=spec.contract_size,
            stop_loss=proposal.stop_loss,
            initial_stop=proposal.stop_loss,
            tp1=proposal.tp1,
            tp2=proposal.tp2,
            tp3=proposal.tp3,
            initial_quantity=tracked.filled_quantity,
            risk_amount=size.actual_risk * fill_ratio,
            opened_at=int(time.time()),
            mode=self.settings.trading_mode.value,
            fees=tracked.fees,
            trade_id=trade_id,
            meta={
                "confidence": proposal.confidence,
                "regime": proposal.regime.value,
                "htf_bias": proposal.htf_bias.value,
                "rr": proposal.rr,
                "atr": atr or proposal.atr,
                "reasons": proposal.reasons,
                "stop_rationale": proposal.stop_rationale,
                "target_rationale": proposal.target_rationale,
                "scores": proposal.scores.as_dict(),
            },
        )
        self.portfolio.add(position)

        notes: list[str] = []
        if tracked.filled_quantity < tracked.quantity:
            notes.append(
                f"partial fill: {tracked.filled_quantity:g}/{tracked.quantity:g} contracts"
            )
            log.warning("partial entry fill on %s: %s", proposal.symbol, notes[-1])

        slippage = abs(fill_price - proposal.entry) / proposal.entry if proposal.entry else 0.0
        if slippage > self.settings.slippage_pct * 3:
            notes.append(f"entry slipped {slippage:.3%} from the quoted price")

        log.info(
            "opened %s %s %.4f contracts @ %.6g (risk $%.2f, %.2fx)",
            proposal.side.value,
            proposal.symbol,
            tracked.filled_quantity,
            fill_price,
            size.actual_risk,
            decision.leverage,
        )
        self._record_event(
            "execution",
            f"opened {proposal.side.value} {proposal.symbol}",
            detail={"decision": decision.as_dict(), "fill": fill_price},
        )
        return ExecutionResult(
            ok=True,
            position=position,
            order=tracked,
            filled_quantity=tracked.filled_quantity,
            average_price=fill_price,
            notes=notes,
        )

    # -- exit -------------------------------------------------------------

    async def close_position(
        self,
        position: ManagedPosition,
        fraction: float = 1.0,
        reason: str = "manual",
        spec: ContractSpec | None = None,
    ) -> ExecutionResult:
        self._assert_can_trade()

        fraction = max(0.0, min(fraction, 1.0))
        if fraction <= 0 or position.quantity <= 0:
            return ExecutionResult(ok=False, error="nothing to close")

        quantity = position.quantity * fraction
        if spec is not None:
            quantity = spec.round_volume(quantity)
            # Closing would leave an untradeable remainder - close it all.
            remainder = position.quantity - quantity
            if 0 < remainder < spec.min_volume:
                quantity = position.quantity
            if quantity < spec.min_volume:
                quantity = position.quantity
        if quantity <= 0:
            return ExecutionResult(ok=False, error="closing size rounds to zero")

        try:
            tracked = await self.orders.place(
                symbol=position.symbol,
                side=position.side,
                intent=OrderIntent.CLOSE if quantity >= position.quantity else OrderIntent.REDUCE,
                quantity=quantity,
                order_type=OrderType.MARKET,
                reduce_only=True,
                trade_id=position.trade_id,
            )
        except ExchangeError as exc:
            return ExecutionResult(ok=False, error=str(exc))

        tracked = await self.orders.wait_for_fill(tracked)
        if tracked.filled_quantity <= 0:
            return ExecutionResult(
                ok=False,
                order=tracked,
                error=f"close order did not fill (status {tracked.status.value})",
            )

        exit_price = tracked.average_price or self.portfolio.price(position.symbol)
        closed_base = tracked.filled_quantity * position.contract_size
        pnl = (exit_price - position.entry_price) * position.side.sign * closed_base
        fees = tracked.fees or (
            exit_price * closed_base * self.settings.taker_fee
        )

        position.quantity = max(0.0, position.quantity - tracked.filled_quantity)
        position.realized_pnl += pnl
        position.fees += fees
        self.portfolio.apply_realized(pnl, fees=fees)

        fully_closed = position.quantity <= 1e-12
        if fully_closed:
            position.state = "closed"
            self.portfolio.remove(position.symbol)
            self._record_trade_close(position, exit_price, reason)
        else:
            self.portfolio.persist(position)

        log.info(
            "closed %.4f contracts of %s @ %.6g (%s) pnl %+.2f",
            tracked.filled_quantity,
            position.symbol,
            exit_price,
            reason,
            pnl,
        )
        self._record_event(
            "execution",
            f"closed {tracked.filled_quantity:g} {position.symbol} ({reason})",
            detail={"pnl": pnl, "exit": exit_price, "fully_closed": fully_closed},
        )
        return ExecutionResult(
            ok=True,
            position=position,
            order=tracked,
            filled_quantity=tracked.filled_quantity,
            average_price=exit_price,
            notes=[f"realised {pnl:+.2f}"],
        )

    async def close_all(self, reason: str = "emergency") -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for position in list(self.portfolio.all()):
            try:
                results.append(await self.close_position(position, 1.0, reason=reason))
            except Exception as exc:  # noqa: BLE001 - keep closing the rest
                log.error("could not close %s: %s", position.symbol, exc)
                results.append(ExecutionResult(ok=False, error=str(exc)))
        return results

    async def cancel_all_orders(self) -> int:
        return await self.orders.cancel_all_open()

    # -- persistence ------------------------------------------------------

    def _record_trade_open(
        self, proposal: TradeProposal, decision: RiskDecision, spec: ContractSpec
    ) -> int | None:
        if self.repositories is None or proposal.side is None:
            return None
        try:
            return self.repositories.trades.open_trade(
                {
                    "symbol": proposal.symbol,
                    "side": proposal.side.value,
                    "mode": self.settings.trading_mode.value,
                    "entry_price": proposal.entry,
                    "quantity": decision.contracts,
                    "leverage": decision.leverage,
                    "stop_loss": proposal.stop_loss,
                    "tp1": proposal.tp1,
                    "tp2": proposal.tp2,
                    "tp3": proposal.tp3,
                    "risk_amount": decision.risk_amount,
                    "planned_rr": proposal.rr,
                    "confidence": proposal.confidence,
                    "regime": proposal.regime.value,
                    "why_entered": proposal.reasons,
                    "features": proposal.features,
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("could not record trade open: %s", exc)
            return None

    def _record_trade_close(
        self, position: ManagedPosition, exit_price: float, reason: str
    ) -> None:
        if self.repositories is None or position.trade_id is None:
            return
        risk = position.risk_amount or 1e-12
        try:
            self.repositories.trades.close_trade(
                trade_id=position.trade_id,
                exit_price=exit_price,
                realized_pnl=position.realized_pnl,
                fees=position.fees,
                funding=position.funding,
                r_multiple=position.realized_pnl / risk,
                why_exited=[reason],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("could not record trade close: %s", exc)

    def _record_event(self, component: str, message: str, detail: Any = None) -> None:
        if self.repositories is None:
            return
        try:
            self.repositories.events.system_event(component, message, detail=detail)
        except Exception as exc:  # noqa: BLE001
            log.debug("could not record system event: %s", exc)

    def health(self) -> dict[str, Any]:
        return {
            "mode": self.settings.trading_mode.value,
            "live_enabled": self.live_enabled,
            "orders": self.orders.health(),
        }


__all__ = ["ExecutionEngine", "ExecutionResult", "TradingModeViolation"]
