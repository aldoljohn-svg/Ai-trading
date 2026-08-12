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
import statistics
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from app.data.candle_store import CandleStore
from app.data.validators import CandleValidation, validate_candles
from app.domain import Candle, OrderBook, Ticker, Timeframe
from app.exchange.base import BaseExchange, ExchangeError, ExchangeRateLimit
from app.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class MarketDataStats:
    fetches: int = 0
    failures: int = 0
    validation_failures: int = 0
    last_success_ts: float = 0.0
    last_error: str = ""
    #: Round-trip time of recent candle fetches, in milliseconds.  Kept as a
    #: short window so the median is a live reading rather than a lifetime
    #: average that hides a degradation happening right now.
    latencies_ms: list[float] = field(default_factory=list)

    def record_latency(self, milliseconds: float, window: int = 50) -> None:
        self.latencies_ms.append(milliseconds)
        if len(self.latencies_ms) > window:
            del self.latencies_ms[:-window]

    @property
    def median_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)


class MarketData:
    def __init__(
        self,
        exchange: BaseExchange,
        store: CandleStore | None = None,
        candle_repository: object | None = None,
        ticker_ttl: float = 5.0,
        book_ttl: float = 20.0,
        max_concurrency: int = 8,
        book_concurrency: int = 2,
        book_stale_ttl: float = 120.0,
    ) -> None:
        self.exchange = exchange
        self.store = store or CandleStore()
        self.candle_repository = candle_repository
        self.ticker_ttl = ticker_ttl
        #: How long a depth snapshot is served without refetching.  Depth is a
        #: per-symbol endpoint the scanner hits for every candidate, and a
        #: two-second window meant a wide scan re-requested books it had just
        #: received, which is what triggered MEXC's "Requests are too frequent".
        #: Twenty seconds is well inside one 15m execution bar and still fresh
        #: enough for spread and visible-depth sizing.
        self.book_ttl = book_ttl
        #: Upper bound on serving a cached book after a refetch *failed*.  The
        #: timestamp on the book is the real one, so a consumer that cares can
        #: still see the age; this only decides when we stop offering it.
        self.book_stale_ttl = max(book_stale_ttl, book_ttl)
        self.stats = MarketDataStats()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        #: Depth is throttled separately and much harder than candles: candle
        #: requests are cached across cycles, depth is not.
        self._book_semaphore = asyncio.Semaphore(max(1, book_concurrency))
        self._inflight: dict[tuple[str, str], asyncio.Future] = {}
        self._books_inflight: dict[str, asyncio.Future] = {}
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
            started = time.perf_counter()
            try:
                fetched = await self.exchange.candles(symbol, timeframe, limit=limit)
            except ExchangeError as exc:
                self.stats.failures += 1
                self.stats.last_error = f"{symbol} {timeframe.value}: {exc}"
                raise
            self.stats.record_latency((time.perf_counter() - started) * 1000.0)

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
        """Return a depth snapshot, throttled and coalesced.

        Three things happen here that did not before, all of them because the
        venue started answering "Requests are too frequent" during a wide scan:

        * concurrent callers asking for the same symbol share one request;
        * only :attr:`_book_semaphore` fetches run at a time;
        * when the fetch is refused for going too fast, a recent cached book is
          served rather than reporting that the symbol has no order book.  The
          two are not the same thing, and treating them the same benched liquid
          majors for half an hour over a mistake of ours.
        """

        key = symbol.upper()
        cached = self._books.get(key)
        if cached and time.time() - cached[0] <= self.book_ttl:
            return cached[1]

        inflight = self._books_inflight.get(key)
        if inflight is not None:
            return await asyncio.shield(inflight)

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._books_inflight[key] = future
        try:
            book = await self._fetch_order_book(key, symbol, depth)
            if not future.done():
                future.set_result(book)
            return book
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            self._books_inflight.pop(key, None)
            if future.done() and future.exception() is not None:
                future.exception()

    async def _fetch_order_book(self, key: str, symbol: str, depth: int) -> OrderBook:
        async with self._book_semaphore:
            # Re-check: while queueing behind the semaphore another caller may
            # have refreshed this symbol, and refetching it is exactly the
            # behaviour that provoked the throttling.
            now = time.time()
            cached = self._books.get(key)
            if cached and now - cached[0] <= self.book_ttl:
                return cached[1]
            try:
                book = await self.exchange.order_book(symbol, depth=depth)
            except ExchangeRateLimit as exc:
                fallback = self._stale_book(key, now)
                if fallback is None:
                    raise
                log.debug(
                    "serving %.0fs-old book for %s after a rate limit: %s",
                    now - self._books[key][0],
                    key,
                    exc,
                )
                return fallback
        self._books[key] = (time.time(), book)
        return book

    def _stale_book(self, key: str, now: float) -> OrderBook | None:
        cached = self._books.get(key)
        if cached is None or now - cached[0] > self.book_stale_ttl:
            return None
        return cached[1]

    # -- diagnostics ------------------------------------------------------

    def health(self) -> dict[str, object]:
        age = time.time() - self.stats.last_success_ts if self.stats.last_success_ts else None
        return {
            "fetches": self.stats.fetches,
            "failures": self.stats.failures,
            "validation_failures": self.stats.validation_failures,
            "seconds_since_success": round(age, 1) if age is not None else None,
            "median_latency_ms": round(self.stats.median_latency_ms, 1),
            "last_error": self.stats.last_error,
            "cache": self.store.stats(),
        }


__all__ = ["MarketData", "MarketDataStats"]
