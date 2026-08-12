"""Depth requests were being refused by the venue, and blamed on the symbol.

The live journal showed this on BTC, NEAR, DOT, SHIB, AVAX and CAP within one
cycle::

    execution risk 0.96: no order book: depth fetch failed:
    GET /api/v1/contract/depth/BTC_USDT: Requests are too frequent

Three separate defects stacked up to produce it:

1. MEXC reports throttling in the *body* of an HTTP 200 response, so the
   transport's 429 handling never saw it and it surfaced as a generic
   ``ExchangeError``.
2. The depth call was charged one token in the rate limiter while klines were
   charged two, and depth -- unlike candles -- is refetched every cycle.
3. The scanner benched any symbol whose depth fetch failed for 30 minutes.
   Applied to a rate limit, that punishes BTC for *our* request rate.

These tests pin each one.
"""

from __future__ import annotations

import asyncio

import pytest

from app.domain import OrderBook, Timeframe
from app.exchange.base import ExchangeError, ExchangeRateLimit


class TestTheVenueSaysSlowDownInTheBody:
    """MEXC returns HTTP 200 with ``success: false`` and a message."""

    def _unwrap(self, payload):
        from app.exchange.mexc import MexcFuturesExchange

        return MexcFuturesExchange._unwrap(payload, "GET /api/v1/contract/depth/BTC_USDT")

    def test_a_throttle_message_is_classified_as_a_rate_limit(self):
        with pytest.raises(ExchangeRateLimit):
            self._unwrap({"success": False, "code": 510, "message": "Requests are too frequent"})

    def test_the_other_phrasings_are_recognised_too(self):
        for message in (
            "Too many requests",
            "request frequency exceeds the limit",
            "rate limit reached",
        ):
            with pytest.raises(ExchangeRateLimit):
                self._unwrap({"success": False, "message": message})

    def test_an_ordinary_failure_is_still_an_ordinary_failure(self):
        """Widening the classifier must not swallow real errors."""

        with pytest.raises(ExchangeError) as caught:
            self._unwrap({"success": False, "code": 1002, "message": "contract not found"})
        assert not isinstance(caught.value, ExchangeRateLimit)

    def test_a_rate_limit_is_an_exchange_error(self):
        """Callers with a broad `except ExchangeError` keep working."""

        assert issubclass(ExchangeRateLimit, ExchangeError)


class TestDepthIsChargedLikeKlines:
    def test_the_depth_call_costs_two_tokens(self):
        """A per-symbol endpoint hit once per candidate cannot be free."""

        import inspect

        from app.exchange import mexc

        source = inspect.getsource(mexc.MexcFuturesExchange.order_book)
        assert "cost=2.0" in source, "depth must be weighted like klines"


class _CountingExchange:
    """Records every depth call and can be told to refuse them."""

    def __init__(self, refuse: bool = False, delay: float = 0.0):
        self.calls: list[str] = []
        self.refuse = refuse
        self.delay = delay
        self.concurrent = 0
        self.peak_concurrent = 0

    async def order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        self.concurrent += 1
        self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        try:
            self.calls.append(symbol.upper())
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.refuse:
                raise ExchangeRateLimit("Requests are too frequent")
            return OrderBook(
                symbol=symbol.upper(),
                bids=((100.0, 5.0),),
                asks=((100.1, 5.0),),
                ts=0,
            )
        finally:
            self.concurrent -= 1


class TestMarketDataAsksLessOften:
    def _market_data(self, exchange, **kwargs):
        from app.data.market_data import MarketData

        return MarketData(exchange, **kwargs)

    def test_concurrent_callers_for_one_symbol_share_a_request(self):
        exchange = _CountingExchange(delay=0.01)
        market = self._market_data(exchange)

        async def run():
            books = await asyncio.gather(
                *(market.order_book("BTCUSDT") for _ in range(6))
            )
            return books

        books = asyncio.run(run())
        assert len(exchange.calls) == 1, exchange.calls
        assert all(book.symbol == "BTCUSDT" for book in books)

    def test_depth_fetches_are_throttled_below_the_scan_width(self):
        exchange = _CountingExchange(delay=0.02)
        market = self._market_data(exchange, book_concurrency=2)
        symbols = [f"SYM{i}USDT" for i in range(10)]

        async def run():
            await asyncio.wait_for(
                asyncio.gather(*(market.order_book(s) for s in symbols)), timeout=5
            )

        asyncio.run(run())
        assert exchange.peak_concurrent <= 2, exchange.peak_concurrent
        assert len(exchange.calls) == 10

    def test_a_cached_book_is_reused_inside_the_ttl(self):
        exchange = _CountingExchange()
        market = self._market_data(exchange, book_ttl=60.0)

        async def run():
            await market.order_book("BTCUSDT")
            await market.order_book("BTCUSDT")
            await market.order_book("BTCUSDT")

        asyncio.run(run())
        assert len(exchange.calls) == 1

    def test_the_default_ttl_is_wide_enough_to_matter(self):
        """Two seconds meant a wide scan refetched books it had just received."""

        market = self._market_data(_CountingExchange())
        assert market.book_ttl >= 15.0

    def test_a_throttled_refresh_falls_back_to_the_recent_book(self):
        """A stale book is not the same as no book."""

        exchange = _CountingExchange()
        market = self._market_data(exchange, book_ttl=0.0, book_stale_ttl=120.0)

        async def run():
            first = await market.order_book("BTCUSDT")
            exchange.refuse = True
            second = await market.order_book("BTCUSDT")
            return first, second

        first, second = asyncio.run(run())
        assert second is first
        assert len(exchange.calls) == 2, "it did try to refresh"

    def test_without_a_recent_book_the_rate_limit_still_propagates(self):
        """The fallback must not invent depth we never had."""

        exchange = _CountingExchange(refuse=True)
        market = self._market_data(exchange)

        with pytest.raises(ExchangeRateLimit):
            asyncio.run(market.order_book("BTCUSDT"))

    def test_a_book_older_than_the_stale_limit_is_not_served(self):
        import time

        exchange = _CountingExchange()
        market = self._market_data(exchange, book_ttl=0.0, book_stale_ttl=30.0)

        async def run():
            await market.order_book("BTCUSDT")
            # Age the cached entry past the limit.
            ts, book = market._books["BTCUSDT"]
            market._books["BTCUSDT"] = (time.time() - 600.0, book)
            exchange.refuse = True
            await market.order_book("BTCUSDT")

        with pytest.raises(ExchangeRateLimit):
            asyncio.run(run())


class TestOurOwnRateLimitDoesNotBenchTheSymbol:
    """Benching BTC for 30 minutes because we asked too fast is a bug."""

    def _scanner(self, exchange_error: Exception):
        from app.data.market_data import MarketData
        from app.exchange.synthetic import SyntheticExchange
        from app.scanner.scanner import Scanner
        from app.scanner.universe import UniverseBuilder

        exchange = SyntheticExchange()

        class Refusing(MarketData):
            async def order_book(self, symbol, depth=20):
                raise exchange_error

        universe = UniverseBuilder(
            min_quote_volume=1000.0, max_spread_pct=0.01, max_symbols=4
        )
        return exchange, Refusing(exchange), universe, Scanner

    def _analyse(self, error: Exception):
        exchange, market_data, universe, Scanner = self._scanner(error)

        async def run():
            await exchange.connect()
            scanner = Scanner(market_data, universe, deep_analysis_count=1)
            analyses = await scanner.deep_analyse(await scanner.prescreen())
            await exchange.close()
            return analyses[0], universe

        return asyncio.run(run())

    def test_a_rate_limit_leaves_the_symbol_in_the_universe(self):
        analysis, universe = self._analyse(
            ExchangeRateLimit("GET /api/v1/contract/depth/BTC_USDT: Requests are too frequent")
        )
        assert universe.benched_symbols() == []
        assert "throttled" in analysis.book_problem

    def test_a_genuine_depth_failure_still_benches(self):
        """Fixing this must not disable the bench for symbols that deserve it."""

        analysis, universe = self._analyse(ExchangeError("no such contract"))
        assert universe.benched_symbols() == [analysis.symbol]
        assert "depth fetch failed" in analysis.book_problem


class TestConfiguration:
    def test_the_throttle_is_configurable(self, settings):
        assert settings.order_book_concurrency >= 1
        assert settings.order_book_ttl_seconds > 0

    def test_an_absurd_concurrency_is_refused(self):
        from app.config import ConfigError, build_settings
        from tests.conftest import TEST_ENV

        with pytest.raises(ConfigError):
            build_settings(env=dict(TEST_ENV, ORDER_BOOK_CONCURRENCY="0"))

    def test_it_is_wired_into_the_engine(self):
        import inspect

        from app import engine

        source = inspect.getsource(engine)
        assert "book_concurrency=settings.order_book_concurrency" in source
        assert "book_ttl=settings.order_book_ttl_seconds" in source
