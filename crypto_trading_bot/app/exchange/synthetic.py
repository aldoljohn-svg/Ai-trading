"""Deterministic synthetic exchange.

Purpose: run the *entire* pipeline - scanner, multi-timeframe analysis, ICT,
RTM, regime, ML, risk, sizing, execution, position management - without any
network access, and get identical results on every run.  It is what the test
suite and ``scripts/run_simulation.py`` drive, and it is also useful as an
offline demo of PAPER mode.

Every symbol has **one** underlying price path, generated at 5-minute
resolution; all higher timeframes are aggregated from it and 1-minute bars are
subdivided from it.  That coherence matters: if each timeframe were generated
independently, higher-timeframe "context" would be unrelated to the execution
chart and every multi-timeframe rule in the system would be tested against
nonsense.

The path itself is a seeded regime-switching random walk with fat tails and
volatility clustering, so it produces trends, ranges, breakouts and the
occasional shock rather than pure Brownian noise.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Iterable, Sequence

from app.domain import (
    Balance,
    Candle,
    ContractSpec,
    ExchangeOrder,
    ExchangePosition,
    OrderBook,
    Side,
    Ticker,
    Timeframe,
    dedupe_sorted_candles,
)
from app.exchange.base import BaseExchange, ExchangeError, ExchangeNotSupported

DEFAULT_SYMBOLS: tuple[tuple[str, float, float], ...] = (
    # (symbol, starting price, annualised volatility knob)
    ("BTCUSDT", 64_000.0, 0.55),
    ("ETHUSDT", 3_100.0, 0.65),
    ("SOLUSDT", 145.0, 0.95),
    ("AVAXUSDT", 32.0, 1.05),
    ("BNBUSDT", 580.0, 0.60),
    ("XRPUSDT", 0.52, 0.85),
    ("ADAUSDT", 0.42, 0.90),
    ("LINKUSDT", 14.5, 0.95),
    ("DOGEUSDT", 0.135, 1.20),
    ("MATICUSDT", 0.58, 1.00),
    ("ARBUSDT", 0.85, 1.15),
    ("OPUSDT", 1.75, 1.10),
)

#: The path is generated at this resolution and aggregated upwards.
BASE_TIMEFRAME = Timeframe.M5


def generate_path(
    symbol: str,
    start_price: float,
    annual_vol: float,
    bars: int,
    tf_seconds: int,
    end_ts: int,
    seed: int,
) -> list[Candle]:
    """Regime-switching GBM with volatility clustering and fat tails."""

    rng = random.Random(f"{seed}:{symbol}")
    bars_per_year = (365 * 24 * 3600) / tf_seconds
    sigma = annual_vol / math.sqrt(bars_per_year)

    price = start_price
    drift = 0.0
    vol_state = 1.0
    regime_left = rng.randint(200, 900)

    start_ts = (end_ts // tf_seconds) * tf_seconds - bars * tf_seconds
    candles: list[Candle] = []

    # Drift is expressed per bar but persists for hundreds of bars, so it must
    # stay small relative to sigma or it compounds into absurd daily moves.
    # These bounds put a trending regime at roughly 2-6% of daily range.
    for index in range(bars):
        if regime_left <= 0:
            roll = rng.random()
            if roll < 0.32:
                drift = sigma * rng.uniform(0.03, 0.12)       # trend up
            elif roll < 0.64:
                drift = -sigma * rng.uniform(0.03, 0.12)      # trend down
            else:
                drift = 0.0                                    # range
            vol_state = rng.choice([0.55, 0.8, 1.0, 1.4, 2.2])
            regime_left = rng.randint(250, 1400)
        regime_left -= 1

        vol_state += (1.0 - vol_state) * 0.02 + rng.gauss(0, 0.03)
        vol_state = max(0.3, min(vol_state, 3.5))

        step_sigma = sigma * vol_state
        shock = rng.gauss(0.0, 1.0)
        if rng.random() < 0.0015:
            shock *= rng.uniform(3.0, 6.0)

        open_price = price
        close_price = open_price * math.exp(drift + step_sigma * shock)

        wick = abs(step_sigma) * open_price * rng.uniform(0.3, 1.5)
        high = max(open_price, close_price) + wick * rng.random()
        low = max(min(open_price, close_price) - wick * rng.random(), 1e-9)

        volume = max(1.0, rng.lognormvariate(math.log(1_000), 0.6) * (1 + vol_state))
        ts = start_ts + index * tf_seconds
        candles.append(
            Candle(
                ts=ts,
                open=round(open_price, 10),
                high=round(high, 10),
                low=round(low, 10),
                close=round(close_price, 10),
                volume=round(volume, 4),
                quote_volume=round(volume * close_price, 4),
                closed=True,
            )
        )
        price = close_price

    return candles


def aggregate(candles: Sequence[Candle], target: Timeframe) -> list[Candle]:
    """Aggregate a fine series into a coarser one on the timeframe grid.

    Bars are bucketed by ``ts // step``, so the result lands exactly on the
    timeframe grid the validators expect.  A partially-filled trailing bucket
    is dropped: it would be a bar that has not closed yet.
    """

    step = target.seconds
    if not candles:
        return []

    source_step = candles[1].ts - candles[0].ts if len(candles) > 1 else step
    if source_step >= step:
        return list(candles)
    expected = step // source_step

    buckets: dict[int, list[Candle]] = {}
    for candle in candles:
        buckets.setdefault((candle.ts // step) * step, []).append(candle)

    out: list[Candle] = []
    for bucket_ts in sorted(buckets):
        group = buckets[bucket_ts]
        if len(group) < expected:
            continue                     # incomplete bar - not closed
        out.append(
            Candle(
                ts=bucket_ts,
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                volume=sum(c.volume for c in group),
                quote_volume=sum(c.quote_volume for c in group),
                closed=True,
            )
        )
    return out


def subdivide(candles: Sequence[Candle], target: Timeframe) -> list[Candle]:
    """Split coarse bars into finer ones that respect the parent's OHLC."""

    step = target.seconds
    if not candles:
        return []
    source_step = candles[1].ts - candles[0].ts if len(candles) > 1 else step
    parts = max(source_step // step, 1)
    if parts <= 1:
        return list(candles)

    out: list[Candle] = []
    for parent in candles:
        rng = random.Random(f"sub:{parent.ts}:{parent.close}")
        # Walk linearly from open to close, sprinkling the parent's extremes
        # into two of the sub-bars so the aggregate is exactly preserved.
        high_slot = rng.randrange(parts)
        low_slot = rng.randrange(parts)
        previous = parent.open
        for slot in range(parts):
            progress = (slot + 1) / parts
            close_price = parent.open + (parent.close - parent.open) * progress
            high = max(previous, close_price)
            low = min(previous, close_price)
            if slot == high_slot:
                high = parent.high
            if slot == low_slot:
                low = parent.low
            high = max(high, previous, close_price)
            low = min(low, previous, close_price)
            out.append(
                Candle(
                    ts=parent.ts + slot * step,
                    open=previous,
                    high=high,
                    low=low,
                    close=close_price,
                    volume=parent.volume / parts,
                    quote_volume=parent.quote_volume / parts,
                    closed=True,
                )
            )
            previous = close_price
    return out


class SyntheticExchange(BaseExchange):
    """Offline, reproducible market-data source."""

    name = "synthetic"
    supports_trading = False

    def __init__(
        self,
        symbols: Iterable[tuple[str, float, float]] = DEFAULT_SYMBOLS,
        seed: int = 20240817,
        base_bars: int = 60_000,
        end_ts: int | None = None,
        equity: float = 1000.0,
    ) -> None:
        self.seed = seed
        self.base_bars = base_bars
        # Align to the base grid.  Higher timeframes are bucketed on the
        # absolute epoch grid and drop their incomplete trailing bucket, which
        # is exactly how a real exchange behaves mid-session.
        raw_end = int(end_ts or time.time())
        self.end_ts = (raw_end // BASE_TIMEFRAME.seconds) * BASE_TIMEFRAME.seconds
        self.equity = equity
        self._spec_inputs = list(symbols)
        self._base: dict[str, list[Candle]] = {}
        self._cache: dict[tuple[str, str], list[Candle]] = {}
        self._contracts: dict[str, ContractSpec] = {}
        self._positions: list[ExchangePosition] = []

        for symbol, price, _vol in self._spec_inputs:
            scale = 2 if price >= 100 else (4 if price >= 1 else 6)
            self._contracts[symbol] = ContractSpec(
                symbol=symbol,
                exchange_symbol=symbol.replace("USDT", "_USDT"),
                base=symbol.replace("USDT", ""),
                quote="USDT",
                contract_size=_contract_size_for(price),
                price_scale=scale,
                volume_scale=0,
                min_volume=1.0,
                max_volume=1_000_000.0,
                max_leverage=20.0,
                price_unit=10 ** (-scale),
                active=True,
            )

    # -- series -----------------------------------------------------------

    def _base_series(self, symbol: str) -> list[Candle]:
        cached = self._base.get(symbol)
        if cached is not None:
            return cached
        entry = next((s for s in self._spec_inputs if s[0] == symbol), None)
        if entry is None:
            raise ExchangeError(f"unknown synthetic symbol {symbol}")
        _sym, price, vol = entry
        series = generate_path(
            symbol=symbol,
            start_price=price,
            annual_vol=vol,
            bars=self.base_bars,
            tf_seconds=BASE_TIMEFRAME.seconds,
            end_ts=self.end_ts,
            seed=self.seed,
        )
        self._base[symbol] = series
        return series

    def _series(self, symbol: str, timeframe: Timeframe) -> list[Candle]:
        key = (symbol, timeframe.value)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        base = self._base_series(symbol)
        if timeframe is BASE_TIMEFRAME:
            series = list(base)
        elif timeframe.seconds < BASE_TIMEFRAME.seconds:
            series = subdivide(base[-4000:], timeframe)
        else:
            series = aggregate(base, timeframe)
        series = dedupe_sorted_candles(series)
        self._cache[key] = series
        return series

    # -- market data ------------------------------------------------------

    async def ping(self) -> float:
        return 0.5

    async def server_time(self) -> int:
        return self.end_ts

    async def contracts(self) -> dict[str, ContractSpec]:
        return dict(self._contracts)

    async def candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 500,
        end_ts: int | None = None,
    ) -> list[Candle]:
        series = self._series(symbol, timeframe)
        if end_ts is not None:
            series = [c for c in series if c.ts + timeframe.seconds <= end_ts]
        return series[-limit:]

    async def ticker(self, symbol: str) -> Ticker:
        series = self._series(symbol, BASE_TIMEFRAME)
        if not series:
            raise ExchangeError(f"no synthetic data for {symbol}")
        last = series[-1].close
        day = series[-288:] if len(series) >= 288 else series
        rng = random.Random(f"spread:{symbol}:{self.seed}")
        spread_pct = rng.uniform(0.00005, 0.0006)
        half = last * spread_pct / 2
        quote_volume = sum(c.quote_volume for c in day)
        return Ticker(
            symbol=symbol,
            last=last,
            bid=last - half,
            ask=last + half,
            volume_24h=sum(c.volume for c in day),
            quote_volume_24h=quote_volume,
            change_24h_pct=(last - day[0].open) / day[0].open if day else 0.0,
            high_24h=max(c.high for c in day),
            low_24h=min(c.low for c in day),
            funding_rate=rng.uniform(-0.0004, 0.0004),
            open_interest=quote_volume * rng.uniform(0.2, 1.5),
            ts=series[-1].ts,
        )

    async def tickers(self) -> dict[str, Ticker]:
        out: dict[str, Ticker] = {}
        for symbol, _price, _vol in self._spec_inputs:
            try:
                out[symbol] = await self.ticker(symbol)
            except ExchangeError:
                continue
        return out

    async def order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        ticker = await self.ticker(symbol)
        rng = random.Random(f"book:{symbol}:{self.seed}")
        bids: list[tuple[float, float]] = []
        asks: list[tuple[float, float]] = []
        step = max(ticker.last * 0.0002, 1e-8)
        for level in range(depth):
            size = rng.uniform(0.5, 4.0) * (1000.0 / max(ticker.last, 1.0))
            bids.append((ticker.bid - level * step, size))
            asks.append((ticker.ask + level * step, size))
        return OrderBook(symbol=symbol, bids=tuple(bids), asks=tuple(asks), ts=ticker.ts)

    async def funding_rate(self, symbol: str) -> float | None:
        return (await self.ticker(symbol)).funding_rate

    async def open_interest(self, symbol: str) -> float | None:
        return (await self.ticker(symbol)).open_interest

    # -- account ----------------------------------------------------------

    async def balance(self, currency: str = "USDT") -> Balance:
        return Balance(
            currency=currency,
            equity=self.equity,
            available=self.equity,
            used_margin=0.0,
            unrealized_pnl=0.0,
        )

    async def positions(self) -> list[ExchangePosition]:
        return list(self._positions)

    async def open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        return []

    async def order_status(self, order_id: str, symbol: str | None = None) -> ExchangeOrder:
        raise ExchangeError(f"synthetic exchange has no order {order_id}")

    # -- trading (always refused) ----------------------------------------

    async def place_order(self, *args: Any, **kwargs: Any) -> ExchangeOrder:
        raise ExchangeNotSupported(
            "the synthetic exchange never places orders; use the paper engine"
        )

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        return False

    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        return True


def _contract_size_for(price: float) -> float:
    """Pick a plausible contract size so notionals land in a sane range."""

    if price >= 10_000:
        return 0.0001
    if price >= 1_000:
        return 0.001
    if price >= 100:
        return 0.01
    if price >= 1:
        return 1.0
    return 10.0


__all__ = [
    "SyntheticExchange",
    "DEFAULT_SYMBOLS",
    "generate_path",
    "aggregate",
    "subdivide",
]
