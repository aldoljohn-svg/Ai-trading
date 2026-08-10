"""Paper trading engine.

Implements the same :class:`~app.execution.order_manager.Broker` interface the
live exchange does, so the *entire* execution path - order manager, execution
engine, position manager, reconciliation - is exercised identically in PAPER
mode.  Only the fill comes from simulation instead of the venue.

**It has no exchange client and no credentials.** It is structurally incapable
of sending an order anywhere; that is the strongest possible guarantee that
PAPER mode cannot trade real money.

What is simulated
-----------------
* market orders: filled at the touch plus adverse slippage
* limit orders: rest until price trades through them
* latency: fills are dated forward by the configured latency
* fees: taker for market, maker for limit
* slippage: scaled by order size relative to available depth
* funding: charged on open positions every funding interval
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from app.domain import (
    ContractSpec,
    ExchangeOrder,
    Fill,
    OrderIntent,
    OrderStatus,
    OrderType,
    Side,
)
from app.exchange.base import ExchangeError
from app.logger import get_logger

log = get_logger(__name__)

PriceProvider = Callable[[str], float]


@dataclass(slots=True)
class PaperFill:
    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    fee: float
    ts: int
    is_maker: bool = False


@dataclass(slots=True)
class PaperOrder:
    order_id: str
    client_order_id: str
    symbol: str
    side: Side
    intent: OrderIntent
    order_type: OrderType
    quantity: float
    price: float | None
    reduce_only: bool
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    average_price: float = 0.0
    fee: float = 0.0
    created_at: int = 0
    filled_at: int = 0
    fills: list[PaperFill] = field(default_factory=list)


class PaperEngine:
    """Simulated broker.  Deterministic given the same price sequence."""

    name = "paper"

    def __init__(
        self,
        price_provider: PriceProvider,
        contracts: Mapping[str, ContractSpec] | None = None,
        taker_fee: float = 0.0006,
        maker_fee: float = 0.0002,
        slippage_pct: float = 0.0005,
        latency_ms: int = 250,
        funding_interval_hours: int = 8,
        depth_provider: Callable[[str], float] | None = None,
        simulate_latency: bool = False,
    ) -> None:
        self.price_provider = price_provider
        self.contracts = dict(contracts or {})
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.slippage_pct = slippage_pct
        self.latency_ms = latency_ms
        self.funding_interval = funding_interval_hours * 3600
        self.depth_provider = depth_provider
        #: Actually sleeping for the latency is useful in a live paper run and
        #: pointless (and slow) in tests.
        self.simulate_latency = simulate_latency

        self.orders: dict[str, PaperOrder] = {}
        self.leverage: dict[str, float] = {}
        self.funding_paid: dict[str, float] = {}
        self._last_funding_ts = int(time.time())
        self.rejected = 0

    # -- helpers ----------------------------------------------------------

    def set_contracts(self, contracts: Mapping[str, ContractSpec]) -> None:
        self.contracts = dict(contracts)

    def _spec(self, symbol: str) -> ContractSpec:
        spec = self.contracts.get(symbol.upper())
        if spec is None:
            # A permissive default keeps simulation working for symbols whose
            # metadata has not been loaded; live trading has no such fallback.
            return ContractSpec(
                symbol=symbol,
                exchange_symbol=symbol,
                base=symbol.replace("USDT", ""),
                quote="USDT",
                contract_size=1.0,
                min_volume=1.0,
                volume_scale=0,
                price_unit=0.0001,
            )
        return spec

    def _price(self, symbol: str) -> float:
        price = self.price_provider(symbol)
        if price is None or price <= 0:
            raise ExchangeError(f"no simulated price available for {symbol}")
        return float(price)

    def _slippage(self, symbol: str, notional: float) -> float:
        """Adverse slippage as a fraction, scaled by size versus depth."""

        base = self.slippage_pct
        if self.depth_provider is None:
            return base
        try:
            depth = self.depth_provider(symbol)
        except Exception:  # noqa: BLE001
            return base
        if depth <= 0:
            return base * 3
        # Consuming a large share of visible depth costs more.
        impact = min(notional / depth, 5.0)
        return base * (1.0 + impact)

    def _fill_price(
        self, symbol: str, side: Side, intent: OrderIntent, quantity: float
    ) -> float:
        """Market fill price including the spread crossing and slippage."""

        mid = self._price(symbol)
        spec = self._spec(symbol)
        notional = quantity * spec.contract_size * mid
        slippage = self._slippage(symbol, notional)

        # Opening a long buys (pays up); closing a long sells (gets hit down).
        buying = (side is Side.LONG) == (intent is OrderIntent.OPEN)
        direction = 1 if buying else -1
        return mid * (1 + direction * slippage)

    # -- Broker interface -------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: Side,
        intent: OrderIntent,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        leverage: float | None = None,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> ExchangeOrder:
        spec = self._spec(symbol)
        quantity = spec.round_volume(quantity)
        if quantity < spec.min_volume:
            self.rejected += 1
            raise ExchangeError(
                f"paper order below the minimum size for {symbol} "
                f"({quantity} < {spec.min_volume})"
            )

        order_id = f"paper-{uuid.uuid4().hex[:12]}"
        order = PaperOrder(
            order_id=order_id,
            client_order_id=client_order_id or order_id,
            symbol=symbol,
            side=side,
            intent=intent,
            order_type=order_type,
            quantity=quantity,
            price=price,
            reduce_only=reduce_only,
            created_at=int(time.time()),
        )
        self.orders[order_id] = order

        if self.simulate_latency and self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

        if order_type is OrderType.MARKET:
            self._execute(order, self._fill_price(symbol, side, intent, quantity), maker=False)
        else:
            # A resting limit order fills only if the market is already through it.
            current = self._price(symbol)
            buying = (side is Side.LONG) == (intent is OrderIntent.OPEN)
            crossed = (
                price is not None
                and ((buying and current <= price) or (not buying and current >= price))
            )
            if crossed:
                self._execute(order, price or current, maker=True)

        return self._as_exchange_order(order)

    def _execute(self, order: PaperOrder, price: float, maker: bool) -> None:
        spec = self._spec(order.symbol)
        notional = order.quantity * spec.contract_size * price
        fee = notional * (self.maker_fee if maker else self.taker_fee)

        # Latency means the fill is stamped slightly after submission.
        ts = order.created_at + int(self.latency_ms / 1000)
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_price = price
        order.fee = fee
        order.filled_at = ts
        order.fills.append(
            PaperFill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=price,
                fee=fee,
                ts=ts,
                is_maker=maker,
            )
        )
        log.debug(
            "paper fill %s %s %g @ %.6g (fee %.4f)",
            order.intent.value,
            order.symbol,
            order.quantity,
            price,
            fee,
        )

    def _as_exchange_order(self, order: PaperOrder) -> ExchangeOrder:
        return ExchangeOrder(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            intent=order.intent,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            status=order.status,
            filled_quantity=order.filled_quantity,
            average_price=order.average_price,
            client_order_id=order.client_order_id,
            reduce_only=order.reduce_only,
            ts=order.filled_at or order.created_at,
        )

    async def order_status(self, order_id: str, symbol: str | None = None) -> ExchangeOrder:
        order = self.orders.get(order_id)
        if order is None:
            raise ExchangeError(f"unknown paper order {order_id}")
        # A resting limit order may have been crossed since it was placed.
        if order.status is OrderStatus.NEW and order.order_type is OrderType.LIMIT:
            try:
                current = self._price(order.symbol)
            except ExchangeError:
                current = 0.0
            buying = (order.side is Side.LONG) == (order.intent is OrderIntent.OPEN)
            if order.price and current > 0:
                if (buying and current <= order.price) or (
                    not buying and current >= order.price
                ):
                    self._execute(order, order.price, maker=True)
        return self._as_exchange_order(order)

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        order = self.orders.get(order_id)
        if order is None or order.status.is_terminal:
            return False
        order.status = OrderStatus.CANCELED
        return True

    async def cancel_all(self, symbol: str | None = None) -> int:
        count = 0
        for order in self.orders.values():
            if not order.status.is_terminal and (symbol is None or order.symbol == symbol):
                order.status = OrderStatus.CANCELED
                count += 1
        return count

    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        self.leverage[symbol.upper()] = leverage
        return True

    async def fills(self, order_id: str, symbol: str | None = None) -> list[Fill]:
        order = self.orders.get(order_id)
        if order is None:
            return []
        return [
            Fill(
                order_id=f.order_id,
                symbol=f.symbol,
                side=f.side,
                quantity=f.quantity,
                price=f.price,
                fee=f.fee,
                ts=f.ts,
                trade_id=f"{f.order_id}-0",
                is_maker=f.is_maker,
            )
            for f in order.fills
        ]

    async def open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        return [
            self._as_exchange_order(o)
            for o in self.orders.values()
            if not o.status.is_terminal and (symbol is None or o.symbol == symbol)
        ]

    # -- funding ----------------------------------------------------------

    def accrue_funding(
        self,
        positions: Any,
        funding_rates: Mapping[str, float],
        now: int | None = None,
    ) -> float:
        """Charge funding on open positions when an interval has elapsed.

        Longs pay when funding is positive, shorts receive, and vice versa.
        Returns the net amount charged (negative means the account received).
        """

        now = int(now if now is not None else time.time())
        if now - self._last_funding_ts < self.funding_interval:
            return 0.0
        self._last_funding_ts = now

        total = 0.0
        for position in positions:
            rate = funding_rates.get(position.symbol)
            if rate is None:
                continue
            price = self.price_provider(position.symbol) or position.entry_price
            notional = position.quantity * position.contract_size * price
            payment = notional * rate * position.side.sign
            position.funding += payment
            self.funding_paid[position.symbol] = (
                self.funding_paid.get(position.symbol, 0.0) + payment
            )
            total += payment
        if total:
            log.info("paper funding settled: %+.4f", -total)
        return total

    def stats(self) -> dict[str, Any]:
        filled = [o for o in self.orders.values() if o.status is OrderStatus.FILLED]
        return {
            "orders": len(self.orders),
            "filled": len(filled),
            "rejected": self.rejected,
            "fees": round(sum(o.fee for o in filled), 6),
            "funding": round(sum(self.funding_paid.values()), 6),
        }


__all__ = ["PaperEngine", "PaperOrder", "PaperFill"]
