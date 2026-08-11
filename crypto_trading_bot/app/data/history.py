"""Paginated historical candle download.

Exchanges cap how many bars one request may return -- MEXC allows 2000.  The
adapter clamps to that cap silently, which is correct for the live scanner
(which never wants more than a few hundred bars) but wrong for training and
backtesting, where asking for 8000 bars and receiving 2000 quietly trains the
model on a quarter of the intended sample.

This module pages backwards through time until the requested depth is reached
or the exchange stops returning older data, so ``--limit`` means what it says.

It is deliberately not used by the live path: the scanner works from the candle
cache and never needs deep history, and paging would multiply its request count
for no benefit.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from app.domain import Candle, Timeframe
from app.logger import get_logger

log = get_logger(__name__)

#: Bars per request.  Below every exchange cap this project talks to, so a
#: single page never comes back truncated in a way we cannot detect.
PAGE_SIZE = 2000

#: Stop after this many pages regardless, so a misbehaving exchange that keeps
#: returning the same window cannot spin forever.
MAX_PAGES = 60


def dedupe_sorted(candles: list[Candle]) -> list[Candle]:
    """Sort by timestamp and drop duplicates, keeping the last seen bar."""

    by_ts: dict[int, Candle] = {}
    for candle in candles:
        by_ts[candle.ts] = candle
    return [by_ts[ts] for ts in sorted(by_ts)]


async def fetch_history(
    exchange: Any,
    symbol: str,
    timeframe: Timeframe,
    limit: int,
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
    progress: Callable[[str, int], None] | None = None,
) -> list[Candle]:
    """Download up to ``limit`` closed bars, paging backwards as needed.

    Returns bars oldest-first.  Fewer than ``limit`` bars means the exchange
    has no more history for that symbol, which is a normal outcome for a recent
    listing and not an error.
    """

    if limit <= page_size:
        return await exchange.candles(symbol, timeframe, limit=limit)

    collected: list[Candle] = []
    end_ts: int | None = None
    seen_oldest: int | None = None

    for _ in range(max_pages):
        page = await exchange.candles(
            symbol, timeframe, limit=page_size, end_ts=end_ts
        )
        if not page:
            break

        collected.extend(page)
        oldest = min(c.ts for c in page)

        # No progress means the exchange has nothing older; stop rather than
        # requesting the same window again.
        if seen_oldest is not None and oldest >= seen_oldest:
            break
        seen_oldest = oldest

        if progress is not None:
            progress(symbol, len(dedupe_sorted(collected)))

        if len(dedupe_sorted(collected)) >= limit:
            break

        # Next page ends where this one began.  One interval of overlap is
        # harmless because duplicates are removed, and it avoids a gap if the
        # exchange treats the bound as inclusive.
        end_ts = oldest

    ordered = dedupe_sorted(collected)
    return ordered[-limit:]


async def fetch_history_many(
    exchange: Any,
    symbols: list[str],
    timeframe: Timeframe,
    limit: int,
    concurrency: int = 4,
    progress: Callable[[str, int], None] | None = None,
) -> dict[str, list[Candle]]:
    """Fetch several symbols concurrently, skipping the ones that fail.

    A symbol that errors is omitted rather than aborting the run: one delisted
    contract must not cost an hour of downloading for everything else.
    """

    semaphore = asyncio.Semaphore(max(1, concurrency))
    out: dict[str, list[Candle]] = {}

    async def one(symbol: str) -> None:
        async with semaphore:
            try:
                candles = await fetch_history(
                    exchange, symbol, timeframe, limit, progress=progress
                )
            except Exception as exc:  # noqa: BLE001 - one bad symbol is not fatal
                log.warning("skipping %s: %s", symbol, exc)
                return
            if candles:
                out[symbol] = candles

    await asyncio.gather(*(one(s) for s in symbols))
    return out


__all__ = ["fetch_history", "fetch_history_many", "dedupe_sorted", "PAGE_SIZE"]
