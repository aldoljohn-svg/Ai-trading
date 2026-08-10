"""Universe construction and pre-screening.

The bot discovers what to trade instead of being told.  This module turns "all
MEXC futures contracts" into a short, ranked list of symbols worth spending
deep-analysis budget on.

Filters, in order (each rejection is recorded so ``/scanner`` can explain the
funnel):

1. contract inactive / not tradable
2. wrong quote currency
3. explicitly blacklisted
4. no usable ticker
5. 24h quote volume below the liquidity floor
6. spread wider than the configured maximum
7. degenerate price data (zero or crossed book)

Survivors are scored on volume, volatility and movement.  That pre-screen score
only decides *who gets analysed*, never who gets traded.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from app.domain import ContractSpec, Ticker
from app.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ScanCandidate:
    symbol: str
    ticker: Ticker
    spec: ContractSpec
    quote_volume: float
    spread_pct: float
    volatility_pct: float       # 24h high-low range relative to price
    change_pct: float
    prescreen_score: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "symbol": self.symbol,
            "quote_volume": round(self.quote_volume, 2),
            "spread_pct": round(self.spread_pct, 6),
            "volatility_pct": round(self.volatility_pct, 6),
            "change_pct": round(self.change_pct, 6),
            "prescreen_score": round(self.prescreen_score, 2),
        }


@dataclass(slots=True)
class UniverseReport:
    considered: int = 0
    accepted: int = 0
    rejections: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        parts = [f"{self.accepted}/{self.considered} symbols passed"]
        for reason, count in self.rejections.most_common():
            parts.append(f"{reason}: {count}")
        return " | ".join(parts)


class UniverseBuilder:
    def __init__(
        self,
        quote_currency: str = "USDT",
        min_quote_volume: float = 5_000_000.0,
        max_spread_pct: float = 0.0008,
        min_volatility_pct: float = 0.005,
        max_volatility_pct: float = 0.60,
        blacklist: Sequence[str] = (),
        max_symbols: int = 100,
    ) -> None:
        self.quote_currency = quote_currency.upper()
        self.min_quote_volume = min_quote_volume
        self.max_spread_pct = max_spread_pct
        self.min_volatility_pct = min_volatility_pct
        self.max_volatility_pct = max_volatility_pct
        self.blacklist = {s.upper() for s in blacklist}
        self.max_symbols = max_symbols
        self.last_report = UniverseReport()

    def build(
        self,
        contracts: Mapping[str, ContractSpec],
        tickers: Mapping[str, Ticker],
    ) -> list[ScanCandidate]:
        report = UniverseReport(considered=len(contracts))
        candidates: list[ScanCandidate] = []

        for symbol, spec in contracts.items():
            if not spec.active:
                report.rejections["inactive contract"] += 1
                continue
            if spec.quote and spec.quote.upper() != self.quote_currency:
                report.rejections["wrong quote currency"] += 1
                continue
            if symbol in self.blacklist:
                report.rejections["blacklisted"] += 1
                continue

            ticker = tickers.get(symbol)
            if ticker is None or ticker.last <= 0:
                report.rejections["no ticker"] += 1
                continue
            if ticker.bid <= 0 or ticker.ask <= 0 or ticker.ask < ticker.bid:
                report.rejections["bad quote"] += 1
                continue

            quote_volume = ticker.quote_volume_24h or (ticker.volume_24h * ticker.last)
            if quote_volume < self.min_quote_volume:
                report.rejections["low liquidity"] += 1
                continue

            spread = ticker.spread_pct
            if spread > self.max_spread_pct:
                report.rejections["wide spread"] += 1
                continue

            volatility = self._volatility(ticker)
            if volatility < self.min_volatility_pct:
                report.rejections["too quiet"] += 1
                continue
            if volatility > self.max_volatility_pct:
                report.rejections["too volatile"] += 1
                continue

            candidates.append(
                ScanCandidate(
                    symbol=symbol,
                    ticker=ticker,
                    spec=spec,
                    quote_volume=quote_volume,
                    spread_pct=spread,
                    volatility_pct=volatility,
                    change_pct=ticker.change_24h_pct,
                    prescreen_score=0.0,
                )
            )

        scored = self._score(candidates)
        scored.sort(key=lambda c: c.prescreen_score, reverse=True)
        selected = scored[: self.max_symbols]

        report.accepted = len(selected)
        self.last_report = report
        log.info("universe: %s", report.summary())
        return selected

    def _volatility(self, ticker: Ticker) -> float:
        if ticker.high_24h > 0 and ticker.low_24h > 0 and ticker.last > 0:
            return (ticker.high_24h - ticker.low_24h) / ticker.last
        return abs(ticker.change_24h_pct)

    def _score(self, candidates: list[ScanCandidate]) -> list[ScanCandidate]:
        """Rank by liquidity, healthy volatility, movement and tight spread."""

        if not candidates:
            return []

        volumes = [c.quote_volume for c in candidates]
        max_log_volume = max(math.log10(v) for v in volumes if v > 0) or 1.0
        min_log_volume = min(math.log10(v) for v in volumes if v > 0)
        volume_span = max(max_log_volume - min_log_volume, 1e-9)

        out: list[ScanCandidate] = []
        for candidate in candidates:
            volume_term = (
                (math.log10(candidate.quote_volume) - min_log_volume) / volume_span
                if candidate.quote_volume > 0
                else 0.0
            )
            # Volatility is desirable up to a point, then it is just risk.
            ideal = 0.06
            volatility_term = math.exp(
                -((candidate.volatility_pct - ideal) ** 2) / (2 * (0.05 ** 2))
            )
            movement_term = min(abs(candidate.change_pct) / 0.08, 1.0)
            spread_term = 1.0 - min(candidate.spread_pct / self.max_spread_pct, 1.0)

            score = 100.0 * (
                0.40 * volume_term
                + 0.25 * volatility_term
                + 0.20 * movement_term
                + 0.15 * spread_term
            )
            out.append(
                ScanCandidate(
                    symbol=candidate.symbol,
                    ticker=candidate.ticker,
                    spec=candidate.spec,
                    quote_volume=candidate.quote_volume,
                    spread_pct=candidate.spread_pct,
                    volatility_pct=candidate.volatility_pct,
                    change_pct=candidate.change_pct,
                    prescreen_score=round(score, 3),
                )
            )
        return out


__all__ = ["UniverseBuilder", "ScanCandidate", "UniverseReport"]
