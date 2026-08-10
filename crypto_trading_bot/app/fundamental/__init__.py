"""Fundamental / macro / news layer.

Design rule for this whole package: **never fabricate a data point.**  If a
provider is not configured or a request fails, the value is ``UNKNOWN`` and
``UNKNOWN`` is never silently converted into bullish or bearish.  The most it
can do is reduce confidence, and - for scheduled high-impact events - block new
entries.
"""

from app.fundamental.macro_engine import MacroEngine, MacroSnapshot
from app.fundamental.news_engine import EconomicEvent, NewsEngine, NewsItem
from app.fundamental.sentiment import (
    DataPoint,
    FundamentalEngine,
    FundamentalSnapshot,
)

__all__ = [
    "DataPoint",
    "FundamentalEngine",
    "FundamentalSnapshot",
    "MacroEngine",
    "MacroSnapshot",
    "NewsEngine",
    "NewsItem",
    "EconomicEvent",
]
