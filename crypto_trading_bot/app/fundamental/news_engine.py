"""News and economic-calendar layer.

Two independent sources, both optional and both fail-safe:

**Economic calendar** - a local JSON file (``data/economic_calendar.json``) the
operator maintains or a scheduled job writes.  Entries describe CPI, FOMC,
rate decisions, employment prints, ETF decisions, token unlocks and so on.  The
engine only reads it; it never invents entries.  During a configurable blackout
window around a high-impact event, new entries are blocked.

**News feed** - an optional HTTP JSON endpoint.  Headlines are classified by
keyword into ``DANGER`` (exchange hack, exploit, insolvency, delisting, halt),
directional, or neutral.  Keyword classification is crude and is treated as
such: it can *block* or *reduce* risk, and it can never be the reason a trade
is taken.

With neither configured the engine reports ``UNKNOWN``, which downstream code
handles as "no information", not "no risk".
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.domain import Sentiment
from app.logger import get_logger

log = get_logger(__name__)

#: Words that indicate a market-wide hazard.  These block entries; they never
#: create a directional opinion.
DANGER_KEYWORDS: tuple[str, ...] = (
    "hack", "hacked", "exploit", "exploited", "drained", "stolen",
    "insolvency", "insolvent", "bankrupt", "bankruptcy", "halt", "halted",
    "suspend", "suspended", "withdrawal freeze", "freeze", "rug",
    "delist", "delisting", "security incident", "breach", "attack",
    "sec lawsuit", "lawsuit", "investigation", "seized", "sanction",
    "depeg", "depegged", "liquidation cascade", "emergency",
)

BULLISH_KEYWORDS: tuple[str, ...] = (
    "etf approved", "approval", "adoption", "partnership", "upgrade",
    "listing", "listed", "buyback", "burn", "institutional inflow",
    "record inflow", "rate cut",
)

BEARISH_KEYWORDS: tuple[str, ...] = (
    "outflow", "dump", "sell-off", "selloff", "crash", "unlock",
    "rate hike", "inflation surge", "ban", "restriction", "fine",
)

HIGH_IMPACT_EVENTS: tuple[str, ...] = (
    "CPI", "FOMC", "FED", "INTEREST RATE", "NFP", "NON-FARM",
    "UNEMPLOYMENT", "PPI", "PCE", "GDP", "ETF DECISION", "ETF RULING",
)


@dataclass(frozen=True, slots=True)
class EconomicEvent:
    name: str
    ts: int                     # epoch seconds, UTC
    impact: str                 # "high" | "medium" | "low"
    currency: str = "USD"
    detail: str = ""

    @property
    def is_high_impact(self) -> bool:
        if self.impact.lower() == "high":
            return True
        upper = self.name.upper()
        return any(token in upper for token in HIGH_IMPACT_EVENTS)

    def seconds_until(self, now: int | None = None) -> int:
        return self.ts - int(now if now is not None else time.time())


@dataclass(frozen=True, slots=True)
class NewsItem:
    title: str
    ts: int
    source: str = ""
    url: str = ""
    symbols: tuple[str, ...] = ()

    def classify(self) -> Sentiment:
        text = self.title.lower()
        if any(keyword in text for keyword in DANGER_KEYWORDS):
            return Sentiment.DANGER
        bullish = sum(1 for keyword in BULLISH_KEYWORDS if keyword in text)
        bearish = sum(1 for keyword in BEARISH_KEYWORDS if keyword in text)
        if bullish > bearish:
            return Sentiment.BULLISH
        if bearish > bullish:
            return Sentiment.BEARISH
        return Sentiment.NEUTRAL


@dataclass(slots=True)
class NewsAssessment:
    sentiment: Sentiment = Sentiment.UNKNOWN
    danger: bool = False
    reasons: list[str] = field(default_factory=list)
    blocking_event: EconomicEvent | None = None
    items_considered: int = 0
    calendar_loaded: bool = False
    feed_loaded: bool = False

    @property
    def blocks_entries(self) -> bool:
        return self.danger or self.blocking_event is not None


class NewsEngine:
    def __init__(
        self,
        calendar_path: Path | None = None,
        feed_url: str = "",
        http_client: Any = None,
        blackout_before_seconds: int = 30 * 60,
        blackout_after_seconds: int = 15 * 60,
        news_ttl_seconds: int = 600,
        max_news_age_seconds: int = 6 * 3600,
    ) -> None:
        self.calendar_path = calendar_path
        self.feed_url = feed_url
        self.http_client = http_client
        self.blackout_before = blackout_before_seconds
        self.blackout_after = blackout_after_seconds
        self.news_ttl = news_ttl_seconds
        self.max_news_age = max_news_age_seconds
        self._news_cache: tuple[float, list[NewsItem]] | None = None

    # -- calendar ---------------------------------------------------------

    def load_calendar(self) -> list[EconomicEvent]:
        """Read the operator-maintained calendar file.  Missing file = empty."""

        if self.calendar_path is None or not self.calendar_path.is_file():
            return []
        try:
            raw = json.loads(self.calendar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("could not read economic calendar: %s", exc)
            return []
        entries = raw.get("events", raw) if isinstance(raw, dict) else raw
        events: list[EconomicEvent] = []
        for item in entries or []:
            if not isinstance(item, dict):
                continue
            try:
                events.append(
                    EconomicEvent(
                        name=str(item["name"]),
                        ts=int(item["ts"]),
                        impact=str(item.get("impact", "medium")),
                        currency=str(item.get("currency", "USD")),
                        detail=str(item.get("detail", "")),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.debug("skipping malformed calendar entry: %s", exc)
        return sorted(events, key=lambda e: e.ts)

    def blocking_event(self, now: int | None = None) -> EconomicEvent | None:
        """A high-impact event inside the blackout window, if any."""

        now = int(now if now is not None else time.time())
        for event in self.load_calendar():
            if not event.is_high_impact:
                continue
            if -self.blackout_after <= (event.ts - now) <= self.blackout_before:
                return event
        return None

    def upcoming(self, hours: int = 24, now: int | None = None) -> list[EconomicEvent]:
        now = int(now if now is not None else time.time())
        horizon = now + hours * 3600
        return [e for e in self.load_calendar() if now <= e.ts <= horizon]

    # -- news feed --------------------------------------------------------

    async def fetch_news(self) -> list[NewsItem]:
        if not self.feed_url or self.http_client is None:
            return []
        now = time.time()
        if self._news_cache and now - self._news_cache[0] < self.news_ttl:
            return self._news_cache[1]
        try:
            payload = await self.http_client.get_json(self.feed_url)
        except Exception as exc:  # noqa: BLE001 - external feed is best effort
            log.warning("news feed request failed: %s", exc)
            return self._news_cache[1] if self._news_cache else []
        items = parse_news_payload(payload)
        self._news_cache = (now, items)
        return items

    # -- assessment -------------------------------------------------------

    async def assess(
        self, symbol: str | None = None, now: int | None = None
    ) -> NewsAssessment:
        now = int(now if now is not None else time.time())
        assessment = NewsAssessment()

        calendar = self.load_calendar()
        assessment.calendar_loaded = bool(calendar)
        event = self.blocking_event(now)
        if event is not None:
            assessment.blocking_event = event
            minutes = event.seconds_until(now) // 60
            assessment.reasons.append(
                f"high-impact event blackout: {event.name} in {minutes} min"
            )

        items = await self.fetch_news()
        assessment.feed_loaded = bool(self.feed_url and self.http_client)
        relevant = [
            item
            for item in items
            if now - item.ts <= self.max_news_age
            and (
                not symbol
                or not item.symbols
                or symbol.upper() in {s.upper() for s in item.symbols}
            )
        ]
        assessment.items_considered = len(relevant)

        if not calendar and not relevant and not assessment.feed_loaded:
            assessment.sentiment = Sentiment.UNKNOWN
            assessment.reasons.append(
                "no news provider configured - fundamentals are UNKNOWN"
            )
            return assessment

        scores = {Sentiment.BULLISH: 0, Sentiment.BEARISH: 0, Sentiment.NEUTRAL: 0}
        for item in relevant:
            classification = item.classify()
            if classification is Sentiment.DANGER:
                assessment.danger = True
                assessment.reasons.append(f"danger headline: {item.title[:110]}")
            elif classification in scores:
                scores[classification] += 1

        if assessment.danger:
            assessment.sentiment = Sentiment.DANGER
        elif not relevant:
            assessment.sentiment = Sentiment.UNKNOWN if not calendar else Sentiment.NEUTRAL
        elif scores[Sentiment.BULLISH] > scores[Sentiment.BEARISH] * 2:
            assessment.sentiment = Sentiment.BULLISH
        elif scores[Sentiment.BEARISH] > scores[Sentiment.BULLISH] * 2:
            assessment.sentiment = Sentiment.BEARISH
        else:
            assessment.sentiment = Sentiment.NEUTRAL
        return assessment


def parse_news_payload(payload: Any) -> list[NewsItem]:
    """Accept the common shapes a JSON news endpoint returns."""

    if isinstance(payload, dict):
        for key in ("data", "results", "articles", "items", "Data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = []
    if not isinstance(payload, list):
        return []

    items: list[NewsItem] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        title = entry.get("title") or entry.get("headline") or entry.get("name")
        if not title:
            continue
        ts = entry.get("published_at") or entry.get("publishedAt") or entry.get("ts")
        ts = ts or entry.get("published_on") or entry.get("time") or 0
        try:
            timestamp = int(float(ts))
        except (TypeError, ValueError):
            timestamp = int(time.time())
        if timestamp > 1e11:
            timestamp //= 1000
        symbols = entry.get("currencies") or entry.get("symbols") or ()
        if isinstance(symbols, list):
            resolved = tuple(
                str(s.get("code") if isinstance(s, dict) else s) for s in symbols
            )
        else:
            resolved = ()
        items.append(
            NewsItem(
                title=str(title),
                ts=timestamp,
                source=str(entry.get("source") or entry.get("domain") or ""),
                url=str(entry.get("url") or ""),
                symbols=resolved,
            )
        )
    return items


__all__ = [
    "NewsEngine",
    "NewsItem",
    "NewsAssessment",
    "EconomicEvent",
    "parse_news_payload",
    "DANGER_KEYWORDS",
]
