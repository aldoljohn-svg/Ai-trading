"""Macro / crypto-wide context.

Two classes of input:

**Derived from the exchange we are already connected to** - funding rate, open
interest, and the behaviour of BTC itself.  These are always available and are
the ones that actually gate trades.

**External providers** - BTC dominance, total market cap, Fear & Greed,
liquidation totals.  These require third-party HTTP endpoints which are
configured, not hard-coded.  With no provider configured the values are
``UNKNOWN`` and the engine says so.

A provider is described by a URL plus a dotted path into the JSON response, so
an operator can point the bot at whichever data source they are entitled to use
without touching code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from app.domain import Sentiment, Timeframe
from app.exchange.base import BaseExchange, ExchangeError
from app.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Declarative description of an external JSON endpoint."""

    name: str
    url: str
    json_path: str = ""          # e.g. "data.0.value"
    headers: Mapping[str, str] = field(default_factory=dict)
    ttl_seconds: int = 900

    def extract(self, payload: Any) -> float | None:
        node: Any = payload
        if not self.json_path:
            return _as_float(node)
        for part in self.json_path.split("."):
            if node is None:
                return None
            if isinstance(node, list):
                try:
                    node = node[int(part)]
                except (ValueError, IndexError):
                    return None
            elif isinstance(node, dict):
                node = node.get(part)
            else:
                return None
        return _as_float(node)


@dataclass(slots=True)
class MacroSnapshot:
    btc_dominance: float | None = None
    total_market_cap: float | None = None
    fear_greed: float | None = None
    liquidations_24h: float | None = None
    btc_trend: Sentiment = Sentiment.UNKNOWN
    btc_change_24h: float | None = None
    market_breadth: float | None = None      # fraction of scanned symbols up
    available: dict[str, bool] = field(default_factory=dict)
    ts: int = 0

    def summary(self) -> str:
        parts = []
        parts.append(
            f"BTC trend: {self.btc_trend.value}"
            + (f" ({self.btc_change_24h:+.2%})" if self.btc_change_24h is not None else "")
        )
        parts.append(
            f"Breadth: {self.market_breadth:.0%}"
            if self.market_breadth is not None
            else "Breadth: UNKNOWN"
        )
        parts.append(
            f"Dominance: {self.btc_dominance:.1f}%"
            if self.btc_dominance is not None
            else "Dominance: UNKNOWN"
        )
        parts.append(
            f"Fear&Greed: {self.fear_greed:.0f}"
            if self.fear_greed is not None
            else "Fear&Greed: UNKNOWN"
        )
        return " | ".join(parts)

    def as_features(self) -> dict[str, float]:
        return {
            "macro_btc_trend": _sentiment_value(self.btc_trend),
            "macro_btc_change": self.btc_change_24h or 0.0,
            "macro_breadth": self.market_breadth if self.market_breadth is not None else 0.5,
            "macro_fear_greed": (self.fear_greed or 50.0) / 100.0,
            "macro_dominance": (self.btc_dominance or 0.0) / 100.0,
        }


class MacroEngine:
    def __init__(
        self,
        exchange: BaseExchange,
        providers: Mapping[str, ProviderSpec] | None = None,
        http_client: Any = None,
    ) -> None:
        self.exchange = exchange
        self.providers = dict(providers or {})
        self.http_client = http_client
        self._cache: dict[str, tuple[float, float | None]] = {}
        self._snapshot: MacroSnapshot | None = None
        self._snapshot_at = 0.0

    async def snapshot(self, tickers: Mapping[str, Any] | None = None, ttl: float = 300.0) -> MacroSnapshot:
        now = time.time()
        if self._snapshot is not None and now - self._snapshot_at < ttl:
            return self._snapshot

        btc_trend, btc_change = await self._btc_context()
        breadth = _breadth(tickers) if tickers else None

        snapshot = MacroSnapshot(
            btc_dominance=await self._provider_value("btc_dominance"),
            total_market_cap=await self._provider_value("total_market_cap"),
            fear_greed=await self._provider_value("fear_greed"),
            liquidations_24h=await self._provider_value("liquidations_24h"),
            btc_trend=btc_trend,
            btc_change_24h=btc_change,
            market_breadth=breadth,
            ts=int(now),
        )
        snapshot.available = {
            "btc_dominance": snapshot.btc_dominance is not None,
            "total_market_cap": snapshot.total_market_cap is not None,
            "fear_greed": snapshot.fear_greed is not None,
            "liquidations_24h": snapshot.liquidations_24h is not None,
            "btc_trend": snapshot.btc_trend is not Sentiment.UNKNOWN,
            "market_breadth": breadth is not None,
        }
        self._snapshot = snapshot
        self._snapshot_at = now
        return snapshot

    async def _btc_context(self) -> tuple[Sentiment, float | None]:
        """BTC's own daily structure - the single most useful macro input."""

        try:
            candles = await self.exchange.candles("BTCUSDT", Timeframe.D1, limit=60)
        except (ExchangeError, Exception) as exc:  # noqa: BLE001 - context only
            log.debug("BTC macro context unavailable: %s", exc)
            return Sentiment.UNKNOWN, None
        if len(candles) < 25:
            return Sentiment.UNKNOWN, None

        closes = [c.close for c in candles]
        change = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] else 0.0
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else ma20

        if closes[-1] > ma20 > ma50:
            trend = Sentiment.BULLISH
        elif closes[-1] < ma20 < ma50:
            trend = Sentiment.BEARISH
        else:
            trend = Sentiment.NEUTRAL

        # A violent BTC day is a market-wide danger signal regardless of sign.
        if abs(change) >= 0.08:
            trend = Sentiment.DANGER
        return trend, change

    async def _provider_value(self, key: str) -> float | None:
        spec = self.providers.get(key)
        if spec is None or self.http_client is None:
            return None
        cached = self._cache.get(key)
        now = time.time()
        if cached and now - cached[0] < spec.ttl_seconds:
            return cached[1]
        try:
            payload = await self.http_client.get_json(spec.url, headers=dict(spec.headers))
            value = spec.extract(payload)
        except Exception as exc:  # noqa: BLE001 - external data is best effort
            log.warning("macro provider %s failed: %s", spec.name, exc)
            value = None
        self._cache[key] = (now, value)
        return value


def _breadth(tickers: Mapping[str, Any]) -> float | None:
    values = []
    for ticker in tickers.values():
        change = getattr(ticker, "change_24h_pct", None)
        if change is not None:
            values.append(change)
    if len(values) < 5:
        return None
    return sum(1 for v in values if v > 0) / len(values)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sentiment_value(sentiment: Sentiment) -> float:
    return {
        Sentiment.BULLISH: 1.0,
        Sentiment.BEARISH: -1.0,
        Sentiment.NEUTRAL: 0.0,
        Sentiment.DANGER: -0.5,
        Sentiment.UNKNOWN: 0.0,
    }[sentiment]


__all__ = ["MacroEngine", "MacroSnapshot", "ProviderSpec"]
