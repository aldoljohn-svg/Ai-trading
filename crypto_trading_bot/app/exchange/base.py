"""Exchange abstraction.

Everything above this layer (scanner, signals, risk, execution) talks only to
:class:`BaseExchange`, which is why the same orchestrator drives live MEXC,
the paper engine and the deterministic synthetic feed used by the test suite.

Units and conventions
---------------------
* ``symbol`` is always the normalised form (``BTCUSDT``); adapters translate.
* ``quantity`` is always in **contracts**, not base currency.  Use
  :meth:`BaseExchange.contract_size` to convert.
* Prices are floats in quote currency.
* Timestamps are epoch **seconds** (UTC) at the boundary; adapters convert from
  whatever the venue uses.
"""

from __future__ import annotations

import abc
from typing import Any, Sequence

from app.domain import (
    Balance,
    Candle,
    ContractSpec,
    ExchangeOrder,
    ExchangePosition,
    Fill,
    OrderIntent,
    OrderType,
    OrderBook,
    Side,
    Ticker,
    Timeframe,
)


class ExchangeError(RuntimeError):
    """Base class for all exchange failures."""


class ExchangeAuthError(ExchangeError):
    """Credentials missing, invalid, or lacking the required permission."""


class ExchangeRateLimit(ExchangeError):
    """The venue asked us to slow down."""


class ExchangeUnavailable(ExchangeError):
    """Network failure, maintenance window, or 5xx from the venue."""


class ExchangeNotSupported(ExchangeError):
    """The venue does not expose this capability for this account."""


class BaseExchange(abc.ABC):
    """Interface every exchange adapter implements."""

    name: str = "base"
    supports_trading: bool = False

    # -- lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        """Open network resources.  Idempotent."""

    async def close(self) -> None:
        """Release network resources.  Idempotent."""

    # -- public market data ----------------------------------------------

    @abc.abstractmethod
    async def ping(self) -> float:
        """Round trip latency in milliseconds.  Raises on failure."""

    @abc.abstractmethod
    async def server_time(self) -> int:
        """Exchange server time, epoch seconds."""

    @abc.abstractmethod
    async def contracts(self) -> dict[str, ContractSpec]:
        """All tradable contracts keyed by normalised symbol."""

    @abc.abstractmethod
    async def tickers(self) -> dict[str, Ticker]:
        """Snapshot of every ticker, keyed by normalised symbol."""

    @abc.abstractmethod
    async def ticker(self, symbol: str) -> Ticker:
        ...

    @abc.abstractmethod
    async def order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        ...

    @abc.abstractmethod
    async def candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 500,
        end_ts: int | None = None,
    ) -> list[Candle]:
        """Closed candles, oldest first.  The forming bar is excluded."""

    async def funding_rate(self, symbol: str) -> float | None:
        """Current funding rate as a fraction, or ``None`` if unknown."""

        return None

    async def open_interest(self, symbol: str) -> float | None:
        return None

    # -- private ----------------------------------------------------------

    @abc.abstractmethod
    async def balance(self, currency: str = "USDT") -> Balance:
        ...

    @abc.abstractmethod
    async def positions(self) -> list[ExchangePosition]:
        ...

    @abc.abstractmethod
    async def open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        ...

    @abc.abstractmethod
    async def order_status(self, order_id: str, symbol: str | None = None) -> ExchangeOrder:
        ...

    async def fills(self, order_id: str, symbol: str | None = None) -> list[Fill]:
        return []

    # -- trading ----------------------------------------------------------

    @abc.abstractmethod
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
        ...

    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        ...

    async def cancel_all(self, symbol: str | None = None) -> int:
        count = 0
        for order in await self.open_orders(symbol):
            if await self.cancel_order(order.order_id, order.symbol):
                count += 1
        return count

    @abc.abstractmethod
    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        ...

    async def close_position(
        self, symbol: str, side: Side, quantity: float
    ) -> ExchangeOrder:
        """Reduce-only market order in the opposite direction."""

        return await self.place_order(
            symbol=symbol,
            side=side,
            intent=OrderIntent.CLOSE,
            quantity=quantity,
            order_type=OrderType.MARKET,
            reduce_only=True,
        )

    # -- helpers ----------------------------------------------------------

    async def contract_size(self, symbol: str) -> float:
        specs = await self.contracts()
        spec = specs.get(symbol)
        return spec.contract_size if spec else 1.0

    async def health(self) -> dict[str, Any]:
        try:
            latency = await self.ping()
            return {"ok": True, "latency_ms": round(latency, 2), "name": self.name}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "name": self.name}


def contracts_from_quantity(quantity_base: float, contract_size: float) -> float:
    """Convert a base-currency quantity into whole contracts."""

    if contract_size <= 0:
        return 0.0
    return quantity_base / contract_size


def base_from_contracts(contracts: float, contract_size: float) -> float:
    return contracts * contract_size


def notional(contracts: float, contract_size: float, price: float) -> float:
    return contracts * contract_size * price


__all__ = [
    "BaseExchange",
    "ExchangeError",
    "ExchangeAuthError",
    "ExchangeRateLimit",
    "ExchangeUnavailable",
    "ExchangeNotSupported",
    "contracts_from_quantity",
    "base_from_contracts",
    "notional",
]
