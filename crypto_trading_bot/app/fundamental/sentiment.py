"""Fundamental aggregation.

Combines the macro engine, the news engine and per-symbol derivatives data
(funding, open interest) into one :class:`FundamentalSnapshot`.

Two rules govern how this feeds the decision:

1. ``UNKNOWN`` is not bullish and not bearish.  A missing data point contributes
   a neutral 50 to the fundamental score and adds an explicit note; it never
   nudges direction.
2. Fundamentals can **veto** or **shrink** a trade but cannot **create** one.
   The directional thesis always comes from price.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from app.domain import Sentiment, Side
from app.fundamental.macro_engine import MacroEngine, MacroSnapshot
from app.fundamental.news_engine import NewsAssessment, NewsEngine
from app.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DataPoint:
    """A single fundamental measurement with explicit availability."""

    name: str
    value: float | None
    sentiment: Sentiment
    source: str
    available: bool

    @classmethod
    def unknown(cls, name: str, source: str = "not configured") -> "DataPoint":
        return cls(
            name=name,
            value=None,
            sentiment=Sentiment.UNKNOWN,
            source=source,
            available=False,
        )

    def describe(self) -> str:
        if not self.available:
            return f"{self.name}: UNKNOWN"
        if self.value is None:
            return f"{self.name}: {self.sentiment.value}"
        return f"{self.name}: {self.value:.6g} ({self.sentiment.value})"


@dataclass(slots=True)
class FundamentalSnapshot:
    symbol: str
    macro: MacroSnapshot
    news: NewsAssessment
    points: dict[str, DataPoint] = field(default_factory=dict)
    ts: int = 0

    # -- gating -----------------------------------------------------------

    @property
    def danger(self) -> bool:
        """A hard block on new entries."""

        return (
            self.news.blocks_entries
            or self.macro.btc_trend is Sentiment.DANGER
            or any(p.sentiment is Sentiment.DANGER for p in self.points.values())
        )

    @property
    def danger_reasons(self) -> list[str]:
        reasons = list(self.news.reasons) if self.news.blocks_entries else []
        if self.macro.btc_trend is Sentiment.DANGER:
            change = self.macro.btc_change_24h
            reasons.append(
                "BTC moved "
                + (f"{change:+.1%}" if change is not None else "violently")
                + " in a day - market-wide risk"
            )
        for point in self.points.values():
            if point.sentiment is Sentiment.DANGER:
                reasons.append(f"{point.name} is flagged dangerous")
        return reasons

    @property
    def unknown_fields(self) -> list[str]:
        unknown = [name for name, point in self.points.items() if not point.available]
        unknown += [name for name, ok in self.macro.available.items() if not ok]
        if self.news.sentiment is Sentiment.UNKNOWN:
            unknown.append("news")
        return sorted(set(unknown))

    @property
    def coverage(self) -> float:
        """Fraction of expected fundamental inputs that are actually available."""

        total = len(self.points) + len(self.macro.available) + 1
        if total == 0:
            return 0.0
        available = sum(1 for p in self.points.values() if p.available)
        available += sum(1 for ok in self.macro.available.values() if ok)
        available += 1 if self.news.sentiment is not Sentiment.UNKNOWN else 0
        return round(available / total, 4)

    # -- scoring ----------------------------------------------------------

    def score(self, side: Side) -> float:
        """0..100.  Exactly 50 when nothing is known - never a directional guess."""

        score = 50.0
        direction = 1 if side is Side.LONG else -1

        if self.macro.btc_trend is Sentiment.BULLISH:
            score += 8 * direction
        elif self.macro.btc_trend is Sentiment.BEARISH:
            score -= 8 * direction

        if self.macro.market_breadth is not None:
            # Breadth of 0.5 is neutral; scale +-10 around it.
            score += (self.macro.market_breadth - 0.5) * 20 * direction

        if self.news.sentiment is Sentiment.BULLISH:
            score += 6 * direction
        elif self.news.sentiment is Sentiment.BEARISH:
            score -= 6 * direction

        funding = self.points.get("funding_rate")
        if funding is not None and funding.available and funding.value is not None:
            # Crowded positioning is a headwind for the crowded side.
            crowding = max(-1.0, min(1.0, funding.value / 0.0008))
            score -= crowding * 8 * direction

        if self.macro.fear_greed is not None:
            # Extreme greed penalises longs, extreme fear penalises shorts.
            tilt = (self.macro.fear_greed - 50.0) / 50.0
            score -= tilt * 6 * direction

        if self.danger:
            score -= 25

        return max(0.0, min(100.0, score))

    def as_features(self) -> dict[str, float]:
        features = dict(self.macro.as_features())
        funding = self.points.get("funding_rate")
        oi = self.points.get("open_interest")
        features.update(
            {
                "fund_news_sentiment": _sentiment_value(self.news.sentiment),
                "fund_danger": 1.0 if self.danger else 0.0,
                "fund_coverage": self.coverage,
                "fund_funding_rate": (funding.value or 0.0) if funding else 0.0,
                "fund_open_interest_available": 1.0 if (oi and oi.available) else 0.0,
            }
        )
        return features

    def summary(self) -> str:
        lines = [self.macro.summary()]
        lines.append(f"News: {self.news.sentiment.value}")
        for point in self.points.values():
            lines.append(point.describe())
        if self.unknown_fields:
            lines.append("UNKNOWN: " + ", ".join(self.unknown_fields))
        return "\n".join(lines)


class FundamentalEngine:
    def __init__(
        self,
        macro: MacroEngine,
        news: NewsEngine,
        exchange: Any = None,
        funding_alert_threshold: float = 0.0015,
    ) -> None:
        self.macro = macro
        self.news = news
        self.exchange = exchange or macro.exchange
        self.funding_alert_threshold = funding_alert_threshold

    async def snapshot(
        self,
        symbol: str,
        tickers: Mapping[str, Any] | None = None,
        ticker: Any = None,
    ) -> FundamentalSnapshot:
        macro_snapshot = await self.macro.snapshot(tickers=tickers)
        news_assessment = await self.news.assess(symbol)

        points: dict[str, DataPoint] = {}

        funding = getattr(ticker, "funding_rate", None) if ticker is not None else None
        if funding is None:
            try:
                funding = await self.exchange.funding_rate(symbol)
            except Exception as exc:  # noqa: BLE001
                log.debug("funding rate unavailable for %s: %s", symbol, exc)
                funding = None
        if funding is None:
            points["funding_rate"] = DataPoint.unknown("funding_rate", "exchange")
        else:
            if abs(funding) >= self.funding_alert_threshold:
                sentiment = Sentiment.DANGER
            elif funding > 0.0003:
                sentiment = Sentiment.BEARISH      # longs paying shorts
            elif funding < -0.0003:
                sentiment = Sentiment.BULLISH
            else:
                sentiment = Sentiment.NEUTRAL
            points["funding_rate"] = DataPoint(
                name="funding_rate",
                value=float(funding),
                sentiment=sentiment,
                source="exchange",
                available=True,
            )

        open_interest = getattr(ticker, "open_interest", None) if ticker is not None else None
        if open_interest is None:
            try:
                open_interest = await self.exchange.open_interest(symbol)
            except Exception:  # noqa: BLE001
                open_interest = None
        points["open_interest"] = (
            DataPoint(
                name="open_interest",
                value=float(open_interest),
                sentiment=Sentiment.NEUTRAL,
                source="exchange",
                available=True,
            )
            if open_interest is not None
            else DataPoint.unknown("open_interest", "exchange")
        )

        for name in ("cpi", "fomc", "interest_rates", "employment", "etf_flows",
                     "token_unlocks", "exchange_announcements", "security_incidents"):
            points.setdefault(
                name,
                DataPoint.unknown(name, "economic calendar / news provider"),
            )
        calendar_events = self.news.upcoming(hours=48)
        if calendar_events:
            for event in calendar_events:
                key = _calendar_key(event.name)
                if key in points:
                    points[key] = DataPoint(
                        name=key,
                        value=float(event.ts),
                        sentiment=(
                            Sentiment.DANGER if event.is_high_impact else Sentiment.NEUTRAL
                        ),
                        source="economic calendar",
                        available=True,
                    )

        return FundamentalSnapshot(
            symbol=symbol,
            macro=macro_snapshot,
            news=news_assessment,
            points=points,
            ts=int(time.time()),
        )


def _calendar_key(name: str) -> str:
    upper = name.upper()
    if "CPI" in upper or "INFLATION" in upper:
        return "cpi"
    if "FOMC" in upper or "FED" in upper:
        return "fomc"
    if "RATE" in upper:
        return "interest_rates"
    if "NFP" in upper or "EMPLOY" in upper or "UNEMPLOY" in upper or "JOBS" in upper:
        return "employment"
    if "ETF" in upper:
        return "etf_flows"
    if "UNLOCK" in upper:
        return "token_unlocks"
    if "ANNOUNCE" in upper or "LISTING" in upper:
        return "exchange_announcements"
    if "HACK" in upper or "EXPLOIT" in upper or "SECURITY" in upper:
        return "security_incidents"
    return "other"


def _sentiment_value(sentiment: Sentiment) -> float:
    return {
        Sentiment.BULLISH: 1.0,
        Sentiment.BEARISH: -1.0,
        Sentiment.NEUTRAL: 0.0,
        Sentiment.DANGER: -0.5,
        Sentiment.UNKNOWN: 0.0,
    }[sentiment]


__all__ = [
    "DataPoint",
    "FundamentalEngine",
    "FundamentalSnapshot",
]
