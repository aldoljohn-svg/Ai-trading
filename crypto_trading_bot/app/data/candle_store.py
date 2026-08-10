"""In-memory candle cache with time-to-live keyed on the timeframe.

Re-requesting a 4H series every 60-second scan cycle would waste the entire
rate-limit budget, so a series is considered fresh until the next bar of that
timeframe is due to close.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

from app.domain import Candle, Timeframe, dedupe_sorted_candles


@dataclass(slots=True)
class _Entry:
    candles: list[Candle]
    fetched_at: float
    expires_at: float


class CandleStore:
    """Thread-safe LRU-ish cache of closed candle series."""

    def __init__(self, max_entries: int = 4000, max_candles: int = 1500) -> None:
        self.max_entries = max_entries
        self.max_candles = max_candles
        self._data: dict[tuple[str, str], _Entry] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(symbol: str, timeframe: Timeframe) -> tuple[str, str]:
        return (symbol.upper(), timeframe.value)

    @staticmethod
    def _ttl(timeframe: Timeframe, now: float) -> float:
        """Expire when the next bar of this timeframe closes (plus a grace)."""

        step = timeframe.seconds
        next_close = (int(now) // step + 1) * step
        return next_close + 2.0

    def get(self, symbol: str, timeframe: Timeframe, now: float | None = None) -> list[Candle] | None:
        now = now if now is not None else time.time()
        with self._lock:
            entry = self._data.get(self._key(symbol, timeframe))
            if entry is None or entry.expires_at <= now:
                self.misses += 1
                return None
            self.hits += 1
            return list(entry.candles)

    def put(
        self,
        symbol: str,
        timeframe: Timeframe,
        candles: Iterable[Candle],
        now: float | None = None,
    ) -> list[Candle]:
        now = now if now is not None else time.time()
        cleaned = dedupe_sorted_candles(c for c in candles if c.closed)
        cleaned = cleaned[-self.max_candles :]
        with self._lock:
            if len(self._data) >= self.max_entries:
                self._evict_oldest()
            self._data[self._key(symbol, timeframe)] = _Entry(
                candles=cleaned,
                fetched_at=now,
                expires_at=self._ttl(timeframe, now),
            )
        return list(cleaned)

    def merge(
        self,
        symbol: str,
        timeframe: Timeframe,
        candles: Iterable[Candle],
        now: float | None = None,
    ) -> list[Candle]:
        """Append/replace bars in an existing series (used by the WebSocket)."""

        with self._lock:
            entry = self._data.get(self._key(symbol, timeframe))
            existing = entry.candles if entry else []
            merged = dedupe_sorted_candles(list(existing) + list(candles))
        return self.put(symbol, timeframe, merged, now=now)

    def invalidate(self, symbol: str | None = None) -> int:
        with self._lock:
            if symbol is None:
                count = len(self._data)
                self._data.clear()
                return count
            upper = symbol.upper()
            keys = [k for k in self._data if k[0] == upper]
            for key in keys:
                del self._data[key]
            return len(keys)

    def _evict_oldest(self) -> None:
        if not self._data:
            return
        oldest = min(self._data.items(), key=lambda kv: kv[1].fetched_at)[0]
        del self._data[oldest]

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        with self._lock:
            return {
                "entries": len(self._data),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }


__all__ = ["CandleStore"]
