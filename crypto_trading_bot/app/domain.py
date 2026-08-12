"""Shared domain types.

Deliberately dependency-free: every analytical engine in this project consumes
these plain dataclasses rather than a DataFrame.  That keeps the quant core
exactly reproducible (no dtype coercion, no NaN propagation surprises, stable
tie-breaking) and makes the backtester bit-for-bit deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def seconds(self) -> int:
        return _TF_SECONDS[self]

    @property
    def minutes(self) -> int:
        return self.seconds // 60


_TF_SECONDS: dict[Timeframe, int] = {
    Timeframe.M1: 60,
    Timeframe.M5: 300,
    Timeframe.M15: 900,
    Timeframe.M30: 1800,
    Timeframe.H1: 3600,
    Timeframe.H4: 14400,
    Timeframe.D1: 86400,
}

#: Higher timeframes establish context; lower timeframes only time the entry.
CONTEXT_TIMEFRAMES: tuple[Timeframe, ...] = (Timeframe.D1, Timeframe.H4, Timeframe.H1)
EXECUTION_TIMEFRAMES: tuple[Timeframe, ...] = (Timeframe.M15, Timeframe.M5)
ALL_TIMEFRAMES: tuple[Timeframe, ...] = (
    Timeframe.M1,
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.M30,
    Timeframe.H1,
    Timeframe.H4,
    Timeframe.D1,
)


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1

    @property
    def opposite(self) -> "Side":
        return Side.SHORT if self is Side.LONG else Side.LONG


class Bias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    CONFLICT = "CONFLICT"


class Regime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        return self in {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }


class OrderIntent(str, Enum):
    """Why an order exists - drives reduce-only and reconciliation logic."""

    OPEN = "open"
    CLOSE = "close"
    REDUCE = "reduce"


class Sentiment(str, Enum):
    """Fundamental/news read.  UNKNOWN is never coerced to a direction."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    DANGER = "DANGER"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class Candle:
    """A single OHLCV bar.  ``ts`` is the bar *open* time, epoch seconds, UTC."""

    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    closed: bool = True

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def typical(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "closed": self.closed,
        }


Candles = Sequence[Candle]


def closes(candles: Candles) -> list[float]:
    return [c.close for c in candles]


def highs(candles: Candles) -> list[float]:
    return [c.high for c in candles]


def lows(candles: Candles) -> list[float]:
    return [c.low for c in candles]


def volumes(candles: Candles) -> list[float]:
    return [c.volume for c in candles]


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """Static exchange metadata for one futures contract."""

    symbol: str            # normalised, e.g. "BTCUSDT"
    exchange_symbol: str   # native, e.g. "BTC_USDT"
    base: str
    quote: str
    contract_size: float = 1.0
    price_scale: int = 2
    volume_scale: int = 0
    min_volume: float = 1.0
    max_volume: float = 1_000_000.0
    max_leverage: float = 20.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0006
    price_unit: float = 0.01
    active: bool = True

    def round_price(self, price: float) -> float:
        if self.price_unit > 0:
            steps = round(price / self.price_unit)
            return round(steps * self.price_unit, max(self.price_scale, 0))
        return round(price, max(self.price_scale, 0))

    def round_volume(self, volume: float) -> float:
        if self.volume_scale <= 0:
            return float(int(volume))
        factor = 10 ** self.volume_scale
        return int(volume * factor) / factor


@dataclass(frozen=True, slots=True)
class Ticker:
    symbol: str
    last: float
    bid: float
    ask: float
    volume_24h: float = 0.0
    quote_volume_24h: float = 0.0
    change_24h_pct: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    funding_rate: float | None = None
    open_interest: float | None = None
    ts: int = 0

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return max(self.ask - self.bid, 0.0)
        return 0.0

    @property
    def spread_pct(self) -> float:
        mid = self.mid
        return self.spread / mid if mid > 0 else float("inf")


@dataclass(frozen=True, slots=True)
class OrderBook:
    """A depth snapshot, best price first.

    Levels are ``(price, size)`` where **size is in base units**, not
    contracts -- every consumer treats ``price * size`` as a quote-currency
    notional.  Adapters whose venue quotes depth in contracts must multiply by
    the contract size before constructing this; see
    :meth:`app.exchange.mexc.MexcFuturesExchange.order_book`.
    """

    symbol: str
    bids: tuple[tuple[float, float], ...] = ()
    asks: tuple[tuple[float, float], ...] = ()
    ts: int = 0

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid(self) -> float:
        if self.bids and self.asks:
            return (self.best_bid + self.best_ask) / 2.0
        return 0.0

    @property
    def spread_pct(self) -> float:
        mid = self.mid
        if mid <= 0:
            return float("inf")
        return (self.best_ask - self.best_bid) / mid

    def depth_quote(self, side: str, pct: float = 0.005) -> float:
        """Quote-currency depth within ``pct`` of mid - a liquidity proxy."""

        mid = self.mid
        if mid <= 0:
            return 0.0
        total = 0.0
        levels = self.bids if side == "bid" else self.asks
        for price, size in levels:
            if abs(price - mid) / mid > pct:
                break
            total += price * size
        return total


@dataclass(frozen=True, slots=True)
class Balance:
    currency: str = "USDT"
    equity: float = 0.0
    available: float = 0.0
    used_margin: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def free_ratio(self) -> float:
        return self.available / self.equity if self.equity > 0 else 0.0


@dataclass(frozen=True, slots=True)
class ExchangePosition:
    """A position as reported by the exchange (source of truth)."""

    symbol: str
    side: Side
    quantity: float          # in contracts
    entry_price: float
    leverage: float = 1.0
    unrealized_pnl: float = 0.0
    margin: float = 0.0
    liquidation_price: float | None = None
    position_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, slots=True)
class ExchangeOrder:
    order_id: str
    symbol: str
    side: Side
    intent: OrderIntent
    order_type: OrderType
    quantity: float
    price: float | None
    status: OrderStatus
    filled_quantity: float = 0.0
    average_price: float = 0.0
    client_order_id: str = ""
    reduce_only: bool = False
    ts: int = 0
    raw: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    fee: float
    ts: int
    trade_id: str = ""
    is_maker: bool = False


class HealthState(str, Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"

    @property
    def emoji(self) -> str:
        return {
            HealthState.HEALTHY: "🟢",
            HealthState.WARNING: "🟡",
            HealthState.ERROR: "🔴",
            HealthState.UNKNOWN: "⚪",
        }[self]

    @property
    def rank(self) -> int:
        return {
            HealthState.HEALTHY: 0,
            HealthState.UNKNOWN: 1,
            HealthState.WARNING: 2,
            HealthState.ERROR: 3,
        }[self]


class BotState(str, Enum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    EMERGENCY = "EMERGENCY"

    @property
    def emoji(self) -> str:
        return {
            BotState.STARTING: "🟡",
            BotState.RUNNING: "🟢",
            BotState.PAUSED: "⏸",
            BotState.STOPPING: "🟠",
            BotState.STOPPED: "🔴",
            BotState.EMERGENCY: "🚨",
        }[self]


def normalise_symbol(symbol: str) -> str:
    """``BTC_USDT`` / ``btc-usdt`` -> ``BTCUSDT``."""

    return symbol.replace("_", "").replace("-", "").replace("/", "").upper()


def mexc_symbol(symbol: str, quote: str = "USDT") -> str:
    """``BTCUSDT`` -> ``BTC_USDT`` (MEXC contract native format)."""

    symbol = symbol.upper()
    if "_" in symbol:
        return symbol
    if symbol.endswith(quote):
        return f"{symbol[: -len(quote)]}_{quote}"
    return symbol


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or denominator != denominator:
        return default
    return numerator / denominator


def pct_change(new: float, old: float) -> float:
    return safe_div(new - old, abs(old), 0.0)


def dedupe_sorted_candles(candles: Iterable[Candle]) -> list[Candle]:
    """Sort by timestamp and keep the last observation for duplicate stamps."""

    by_ts: dict[int, Candle] = {}
    for candle in candles:
        by_ts[candle.ts] = candle
    return [by_ts[ts] for ts in sorted(by_ts)]
