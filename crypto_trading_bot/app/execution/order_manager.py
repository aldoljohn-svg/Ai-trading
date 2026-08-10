"""Order lifecycle management.

Responsibilities:

* mint a deterministic **client order id** for every order, so a request that
  timed out can be identified on the venue instead of being blindly resent;
* persist the order before it is sent (write-ahead), so a crash between "sent"
  and "acknowledged" is recoverable;
* poll until the order reaches a terminal state, with a timeout;
* record fills and surface partial fills honestly.

The write-ahead ordering matters.  If the process dies immediately after
``place_order`` returns, the database already contains a ``new`` row with the
client order id, and reconciliation can find the real order on the exchange.
The opposite ordering would lose the order entirely.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.domain import (
    ExchangeOrder,
    Fill,
    OrderIntent,
    OrderStatus,
    OrderType,
    Side,
)
from app.exchange.base import ExchangeError, ExchangeNotSupported
from app.logger import get_logger

log = get_logger(__name__)


class Broker(Protocol):
    """The subset of the exchange interface that execution needs."""

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
    ) -> ExchangeOrder: ...

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> bool: ...

    async def order_status(self, order_id: str, symbol: str | None = None) -> ExchangeOrder: ...

    async def set_leverage(self, symbol: str, leverage: float) -> bool: ...

    async def fills(self, order_id: str, symbol: str | None = None) -> list[Fill]: ...


@dataclass(slots=True)
class TrackedOrder:
    client_order_id: str
    symbol: str
    side: Side
    intent: OrderIntent
    order_type: OrderType
    quantity: float
    price: float | None
    status: OrderStatus = OrderStatus.NEW
    exchange_order_id: str = ""
    filled_quantity: float = 0.0
    average_price: float = 0.0
    reduce_only: bool = False
    fees: float = 0.0
    error: str = ""
    created_at: int = 0
    updated_at: int = 0
    fills: list[Fill] = field(default_factory=list)

    @property
    def is_filled(self) -> bool:
        return self.status is OrderStatus.FILLED

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def remaining(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)


class OrderManager:
    def __init__(
        self,
        broker: Broker,
        repositories: Any = None,
        mode: str = "paper",
        poll_interval: float = 1.0,
        fill_timeout: float = 30.0,
        instance: str = "bot",
    ) -> None:
        self.broker = broker
        self.repositories = repositories
        self.mode = mode
        self.poll_interval = poll_interval
        self.fill_timeout = fill_timeout
        self.instance = instance
        self.orders: dict[str, TrackedOrder] = {}
        self.submitted = 0
        self.failures = 0

    # -- ids --------------------------------------------------------------

    def new_client_order_id(self, symbol: str, intent: OrderIntent) -> str:
        """Short, unique, and traceable back to this instance."""

        token = uuid.uuid4().hex[:10]
        prefix = "o" if intent is OrderIntent.OPEN else "c"
        # MEXC limits externalOid length, so keep it compact.
        return f"{prefix}{int(time.time()) % 10_000_000}{token}"[:32]

    # -- placement --------------------------------------------------------

    async def place(
        self,
        symbol: str,
        side: Side,
        intent: OrderIntent,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        leverage: float | None = None,
        reduce_only: bool = False,
        trade_id: int | None = None,
    ) -> TrackedOrder:
        client_order_id = self.new_client_order_id(symbol, intent)
        tracked = TrackedOrder(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            intent=intent,
            order_type=order_type,
            quantity=quantity,
            price=price,
            reduce_only=reduce_only or intent is not OrderIntent.OPEN,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )
        self.orders[client_order_id] = tracked

        # Write-ahead: the order exists in our records before it can exist on
        # the venue.
        self._persist(tracked, trade_id=trade_id)

        try:
            result = await self.broker.place_order(
                symbol=symbol,
                side=side,
                intent=intent,
                quantity=quantity,
                order_type=order_type,
                price=price,
                leverage=leverage,
                reduce_only=tracked.reduce_only,
                client_order_id=client_order_id,
            )
        except ExchangeNotSupported as exc:
            self.failures += 1
            tracked.status = OrderStatus.REJECTED
            tracked.error = str(exc)
            self._persist(tracked, trade_id=trade_id)
            log.error("order rejected by venue capability: %s", exc)
            raise
        except ExchangeError as exc:
            self.failures += 1
            tracked.status = OrderStatus.UNKNOWN
            tracked.error = str(exc)
            self._persist(tracked, trade_id=trade_id)
            log.error(
                "order %s failed in flight - reconciliation will resolve it: %s",
                client_order_id,
                exc,
            )
            raise

        self.submitted += 1
        tracked.exchange_order_id = result.order_id
        tracked.status = result.status
        tracked.filled_quantity = result.filled_quantity
        tracked.average_price = result.average_price
        tracked.updated_at = int(time.time())
        self._persist(tracked, trade_id=trade_id)
        return tracked

    # -- monitoring -------------------------------------------------------

    async def wait_for_fill(
        self, tracked: TrackedOrder, timeout: float | None = None
    ) -> TrackedOrder:
        """Poll until terminal or timeout.  Never raises on timeout."""

        if tracked.is_terminal:
            await self._collect_fills(tracked)
            return tracked

        deadline = time.monotonic() + (timeout if timeout is not None else self.fill_timeout)
        while time.monotonic() < deadline:
            await asyncio.sleep(self.poll_interval)
            try:
                status = await self.broker.order_status(
                    tracked.exchange_order_id, tracked.symbol
                )
            except ExchangeError as exc:
                log.warning("could not poll order %s: %s", tracked.client_order_id, exc)
                continue

            tracked.status = status.status
            tracked.filled_quantity = status.filled_quantity
            tracked.average_price = status.average_price or tracked.average_price
            tracked.updated_at = int(time.time())
            self._persist(tracked)

            if tracked.is_terminal:
                break

        if not tracked.is_terminal:
            log.warning(
                "order %s still open after %.0fs (filled %.4f/%.4f)",
                tracked.client_order_id,
                timeout if timeout is not None else self.fill_timeout,
                tracked.filled_quantity,
                tracked.quantity,
            )
        await self._collect_fills(tracked)
        return tracked

    async def _collect_fills(self, tracked: TrackedOrder) -> None:
        if not tracked.exchange_order_id:
            return
        try:
            fills = await self.broker.fills(tracked.exchange_order_id, tracked.symbol)
        except Exception as exc:  # noqa: BLE001 - fills are supplementary
            log.debug("no fill detail for %s: %s", tracked.client_order_id, exc)
            return
        tracked.fills = fills
        tracked.fees = sum(f.fee for f in fills)
        if fills and not tracked.average_price:
            total_quantity = sum(f.quantity for f in fills)
            if total_quantity > 0:
                tracked.average_price = (
                    sum(f.price * f.quantity for f in fills) / total_quantity
                )
        if self.repositories is not None:
            for fill in fills:
                try:
                    self.repositories.fills.record(
                        {
                            "order_client_id": tracked.client_order_id,
                            "exchange_trade_id": fill.trade_id,
                            "symbol": fill.symbol or tracked.symbol,
                            "side": fill.side.value,
                            "quantity": fill.quantity,
                            "price": fill.price,
                            "fee": fill.fee,
                            "is_maker": fill.is_maker,
                            "ts": fill.ts,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    log.debug("could not record fill: %s", exc)

    async def cancel(self, tracked: TrackedOrder) -> bool:
        if tracked.is_terminal or not tracked.exchange_order_id:
            return False
        ok = await self.broker.cancel_order(tracked.exchange_order_id, tracked.symbol)
        if ok:
            tracked.status = OrderStatus.CANCELED
            tracked.updated_at = int(time.time())
            self._persist(tracked)
        return ok

    async def cancel_all_open(self) -> int:
        count = 0
        for tracked in list(self.orders.values()):
            if not tracked.is_terminal:
                try:
                    if await self.cancel(tracked):
                        count += 1
                except ExchangeError as exc:
                    log.warning("cancel failed for %s: %s", tracked.client_order_id, exc)
        return count

    def open_orders(self) -> list[TrackedOrder]:
        return [o for o in self.orders.values() if not o.is_terminal]

    # -- persistence ------------------------------------------------------

    def _persist(self, tracked: TrackedOrder, trade_id: int | None = None) -> None:
        if self.repositories is None:
            return
        try:
            self.repositories.orders.create(
                {
                    "client_order_id": tracked.client_order_id,
                    "exchange_order_id": tracked.exchange_order_id or None,
                    "trade_id": trade_id,
                    "symbol": tracked.symbol,
                    "side": tracked.side.value,
                    "intent": tracked.intent.value,
                    "order_type": tracked.order_type.value,
                    "quantity": tracked.quantity,
                    "price": tracked.price,
                    "status": tracked.status.value,
                    "filled_quantity": tracked.filled_quantity,
                    "average_price": tracked.average_price,
                    "reduce_only": tracked.reduce_only,
                    "mode": self.mode,
                    "error": tracked.error or None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("could not persist order %s: %s", tracked.client_order_id, exc)

    def load_open_orders(self) -> list[dict[str, Any]]:
        if self.repositories is None:
            return []
        return self.repositories.orders.open_orders(mode=self.mode)

    def health(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "failures": self.failures,
            "open": len(self.open_orders()),
            "tracked": len(self.orders),
        }


__all__ = ["OrderManager", "TrackedOrder", "Broker"]
