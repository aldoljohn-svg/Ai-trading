"""Market-data facade.

Single entry point for every consumer that needs candles, tickers or order
books.  Responsibilities:

* cache aggressively (:class:`~app.data.candle_store.CandleStore`) so the
  scanner does not re-request identical data every cycle;
* validate everything before it is handed out;
* persist closed candles so a restart does not start from an empty history;
* coalesce concurrent requests for the same series into one network call.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

from app.data.candle_store import CandleStore
from app.data.validators import CandleValidation, validate_candles
from app.domain import Candle, OrderBook, Ticker, Timeframe
from app.exchange.base import BaseExchange, ExchangeError
from app.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class MarketDataStats:
    fetches: int = 0
    failures: int = 0
    validation_failures: int = 0
    last_success_ts: float = 0.0
    last_error: str = ""


class MarketData:
    def __init__(
        self,
        exchange: BaseExchange,
        store: CandleStore | None = None,
        candle_repository: object | None = None,
        ticker_ttl: float = 5.0,
        book_ttl: float = 2.0,
        max_concurrency: int = 8,
    ) -> None:
        self.exchange = exchange
        self.store = store or CandleStore()
        self.candle_repository = candle_repository
        self.ticker_ttl = ticker_ttl
        self.book_ttl = book_ttl
        self.stats = MarketDataStats()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._inflight: dict[tuple[str, str], asyncio.Future] = {}
        self._tickers: dict[str, tuple[float, Ticker]] = {}
        self._tickers_all: tuple[float, dict[str, Ticker]] | None = None
        self._books: dict[str, tuple[float, OrderBook]] = {}
        self._validation: dict[tuple[str, str], CandleValidation] = {}

    # -- candles ----------------------------------------------------------

    async def candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 400,
        min_length: int = 60,
        allow_stale: bool = False,
    ) -> list[Candle]:
        """Return validated closed candles, oldest first.

        Raises :class:`~app.exchange.base.ExchangeError` when the series cannot
        be produced or fails validation - callers treat that as "do not trade
        this symbol", never as "assume flat".
        """

        cached = self.store.get(symbol, timeframe)
        if cached is not None and len(cached) >= min(limit, min_length):
            return cached[-limit:]

        key = (symbol.upper(), timeframe.value)
        inflight = self._inflight.get(key)
        if inflight is not None:
            return (await asyncio.shield(inflight))[-limit:]

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._inflight[key] = future
        try:
            candles = await self._fetch_candles(symbol, timeframe, limit, min_length, allow_stale)
            if not future.done():
                future.set_result(candles)
            return candles[-limit:]
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            self._inflight.pop(key, None)
            # Prevent "exception never retrieved" warnings when nobody awaited.
            if future.done() and future.exception() is not None:
                future.exception()

    async def _fetch_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int,
        min_length: int,
        allow_stale: bool,
    ) -> list[Candle]:
        async with self._semaphore:
            self.stats.fetches += 1
            try:
                fetched = await self.exchange.candles(symbol, timeframe, limit=limit)
            except ExchangeError as exc:
                self.stats.failures += 1
                self.stats.last_error = f"{symbol} {timeframe.value}: {exc}"
                raise

        result = validate_candles(
            fetched,
            timeframe,
            min_length=min_length,
            max_staleness_bars=1e9 if allow_stale else 3.0,
        )
        self._validation[(symbol.upper(), timeframe.value)] = result
        if not result.ok:
            self.stats.validation_failures += 1
            self.stats.last_error = f"{symbol} {timeframe.value}: {result.problems}"
            raise ExchangeError(
                f"{symbol} {timeframe.value} failed data validation: "
                + "; ".join(result.problems)
            )

        self.stats.last_success_ts = time.time()
        stored = self.store.put(symbol, timeframe, fetched)
        self._persist(symbol, timeframe, stored)
        return stored

    def _persist(self, symbol: str, timeframe: Timeframe, candles: Sequence[Candle]) -> None:
        if self.candle_repository is None or not candles:
            return
        try:
            self.candle_repository.save(symbol, timeframe, candles)
        except Exception as exc:  # persistence must never break trading
            log.warning("could not persist candles for %s: %s", symbol, exc)

    async def multi_timeframe(
        self,
        symbol: str,
        timeframes: Iterable[Timeframe],
        limit: int = 400,
        min_length: int = 60,
    ) -> dict[Timeframe, list[Candle]]:
        """Fetch several timeframes concurrently.

        A timeframe that fails is omitted; the caller decides whether the
        remaining context is sufficient (it usually is not, and the signal
        engine refuses to trade without its context timeframes).
        """

        timeframes = list(timeframes)
        results = await asyncio.gather(
            *(self.candles(symbol, tf, limit=limit, min_length=min_length) for tf in timeframes),
            return_exceptions=True,
        )
        out: dict[Timeframe, list[Candle]] = {}
        for timeframe, result in zip(timeframes, results):
            if isinstance(result, BaseException):
                log.debug("no %s data for %s: %s", timeframe.value, symbol, result)
                continue
            out[timeframe] = result
        return out

    def validation_for(self, symbol: str, timeframe: Timeframe) -> CandleValidation | None:
        return self._validation.get((symbol.upper(), timeframe.value))

    # -- tickers ----------------------------------------------------------

    async def ticker(self, symbol: str, max_age: float | None = None) -> Ticker:
        max_age = self.ticker_ttl if max_age is None else max_age
        now = time.time()
        cached = self._tickers.get(symbol.upper())
        if cached and now - cached[0] <= max_age:
            return cached[1]
        ticker = await self.exchange.ticker(symbol)
        self._tickers[symbol.upper()] = (now, ticker)
        return ticker

    async def all_tickers(self, max_age: float = 15.0) -> dict[str, Ticker]:
        now = time.time()
        if self._tickers_all and now - self._tickers_all[0] <= max_age:
            return self._tickers_all[1]
        tickers = await self.exchange.tickers()
        self._tickers_all = (now, tickers)
        for symbol, ticker in tickers.items():
            self._tickers[symbol] = (now, ticker)
        return tickers

    def ingest_ticker(self, ticker: Ticker) -> None:
        """Feed a ticker pushed over the WebSocket into the cache."""

        self._tickers[ticker.symbol] = (time.time(), ticker)

    def ingest_candle(self, symbol: str, timeframe: Timeframe, candle: Candle) -> None:
        if candle.closed:
            self.store.merge(symbol, timeframe, [candle])

    # -- order books ------------------------------------------------------

    async def order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        now = time.time()
        cached = self._books.get(symbol.upper())
        if cached and now - cached[0] <= self.book_ttl:
            return cached[1]
        book = await self.exchange.order_book(symbol, depth=depth)
        self._books[symbol.upper()] = (now, book)
        return book

    # -- diagnostics ------------------------------------------------------

    def health(self) -> dict[str, object]:
        age = time.time() - self.stats.last_success_ts if self.stats.last_success_ts else None
        return {
            "fetches": self.stats.fetches,
            "failures": self.stats.failures,
            "validation_failures": self.stats.validation_failures,
            "seconds_since_success": round(age, 1) if age is not None else None,
            "last_error": self.stats.last_error,
            "cache": self.store.stats(),
        }


__all__ = ["MarketData", "MarketDataStats"]
