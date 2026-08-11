"""Deep history must actually be deep.

Exchanges cap one request at 2000 bars.  Asking for 8000 and silently getting
2000 trained the direction model on a quarter of the intended sample and made
``--limit`` a lie.  These tests pin the pager's behaviour, including the cases
where it must stop rather than loop.
"""

from __future__ import annotations

import asyncio

import pytest

from app.data.history import (
    PAGE_SIZE,
    dedupe_sorted,
    fetch_history,
    fetch_history_many,
)
from app.domain import Candle, Timeframe


class PagedExchange:
    """An exchange with ``total`` bars that never returns more than the cap."""

    def __init__(self, total: int = 10_000, cap: int = PAGE_SIZE, step: int = 900):
        self.total = total
        self.cap = cap
        self.step = step
        self.requests: list[tuple[int, int | None]] = []
        self._all = [
            Candle(
                ts=i * step,
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=10.0,
            )
            for i in range(total)
        ]

    async def candles(self, symbol, timeframe, limit=500, end_ts=None):
        self.requests.append((limit, end_ts))
        series = self._all
        if end_ts is not None:
            series = [c for c in series if c.ts + self.step <= end_ts]
        return series[-min(limit, self.cap) :]


class TestFetchHistory:
    def test_a_small_request_does_not_page(self):
        exchange = PagedExchange()
        candles = asyncio.run(
            fetch_history(exchange, "BTCUSDT", Timeframe.M15, limit=500)
        )
        assert len(candles) == 500
        assert len(exchange.requests) == 1

    def test_a_large_request_pages_past_the_cap(self):
        """The whole point: 8000 requested must not come back as 2000."""

        exchange = PagedExchange(total=10_000)
        candles = asyncio.run(
            fetch_history(exchange, "BTCUSDT", Timeframe.M15, limit=8000)
        )
        assert len(candles) == 8000
        assert len(exchange.requests) > 1

    def test_results_are_ordered_and_unique(self):
        exchange = PagedExchange(total=10_000)
        candles = asyncio.run(
            fetch_history(exchange, "BTCUSDT", Timeframe.M15, limit=6000)
        )
        stamps = [c.ts for c in candles]
        assert stamps == sorted(stamps)
        assert len(set(stamps)) == len(stamps)

    def test_the_newest_bars_are_kept(self):
        """History is trimmed from the old end, never the recent end."""

        exchange = PagedExchange(total=10_000)
        candles = asyncio.run(
            fetch_history(exchange, "BTCUSDT", Timeframe.M15, limit=3000)
        )
        assert candles[-1].ts == exchange._all[-1].ts

    def test_a_short_listing_returns_what_exists(self):
        """A recent listing has less history; that is not an error."""

        exchange = PagedExchange(total=2_500)
        candles = asyncio.run(
            fetch_history(exchange, "NEWUSDT", Timeframe.M15, limit=8000)
        )
        assert len(candles) == 2_500

    def test_an_exchange_that_stops_going_older_terminates(self):
        """No progress must end the loop, not spin against the page cap."""

        class Stuck(PagedExchange):
            async def candles(self, symbol, timeframe, limit=500, end_ts=None):
                self.requests.append((limit, end_ts))
                # Always returns the same newest window regardless of end_ts.
                return self._all[-self.cap :]

        exchange = Stuck(total=10_000)
        candles = asyncio.run(
            fetch_history(exchange, "BTCUSDT", Timeframe.M15, limit=8000)
        )
        assert len(candles) == PAGE_SIZE
        assert len(exchange.requests) < 5, "must not keep asking for the same window"

    def test_an_empty_exchange_returns_nothing(self):
        exchange = PagedExchange(total=0)
        candles = asyncio.run(
            fetch_history(exchange, "BTCUSDT", Timeframe.M15, limit=8000)
        )
        assert candles == []

    def test_the_page_budget_is_bounded(self):
        exchange = PagedExchange(total=10_000_000)
        asyncio.run(
            fetch_history(
                exchange, "BTCUSDT", Timeframe.M15, limit=10_000_000, max_pages=3
            )
        )
        assert len(exchange.requests) <= 3


class TestFetchHistoryMany:
    def test_every_symbol_is_fetched(self):
        exchange = PagedExchange(total=6000)
        series = asyncio.run(
            fetch_history_many(
                exchange, ["BTCUSDT", "ETHUSDT", "SOLUSDT"], Timeframe.M15, 4000
            )
        )
        assert set(series) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
        assert all(len(c) == 4000 for c in series.values())

    def test_one_failing_symbol_does_not_abort_the_run(self):
        class Flaky(PagedExchange):
            async def candles(self, symbol, timeframe, limit=500, end_ts=None):
                if symbol == "BADUSDT":
                    raise RuntimeError("delisted")
                return await super().candles(symbol, timeframe, limit, end_ts)

        exchange = Flaky(total=3000)
        series = asyncio.run(
            fetch_history_many(
                exchange, ["BTCUSDT", "BADUSDT", "ETHUSDT"], Timeframe.M15, 2500
            )
        )
        assert set(series) == {"BTCUSDT", "ETHUSDT"}


class TestDedupe:
    def test_duplicates_collapse_keeping_the_last(self):
        a = Candle(ts=0, open=1, high=2, low=0.5, close=1.5, volume=1)
        b = Candle(ts=0, open=1, high=9, low=0.5, close=8.0, volume=1)
        c = Candle(ts=900, open=2, high=3, low=1.5, close=2.5, volume=1)
        out = dedupe_sorted([a, c, b])
        assert len(out) == 2
        assert out[0].close == 8.0
        assert [x.ts for x in out] == [0, 900]


class TestSampleYield:
    """The arithmetic that made the truncation visible in the first place."""

    @pytest.mark.parametrize(
        "bars,expected",
        [
            (2000, 394),      # what a clamped request actually produced
            (8000, 1894),     # what --limit 8000 should produce
        ],
    )
    def test_samples_per_symbol(self, bars, expected):
        warmup, horizon, stride = 400, 24, 4
        assert (bars - warmup - horizon) // stride == expected
