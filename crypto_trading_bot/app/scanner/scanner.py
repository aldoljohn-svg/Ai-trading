"""Multi-timeframe deep analysis.

The scanner is the expensive half of the pipeline, so it is strictly staged:

1. :class:`~app.scanner.universe.UniverseBuilder` reduces every contract to a
   ranked shortlist using one cheap ``tickers`` call.
2. Only the top ``deep_analysis_count`` symbols get multi-timeframe candle
   fetches and the full analytical stack.
3. Everything runs concurrently with a bounded semaphore so the event loop is
   never blocked and the exchange rate limit is respected.

Higher timeframes establish context; lower timeframes only time the entry.  A
symbol whose context timeframes disagree is reported as ``CONFLICT`` and the
signal engine will not trade it.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from app.data.market_data import MarketData
from app.data.validators import detect_price_anomaly
from app.domain import (
    ALL_TIMEFRAMES,
    Bias,
    Candle,
    CONTEXT_TIMEFRAMES,
    EXECUTION_TIMEFRAMES,
    OrderBook,
    Regime,
    Ticker,
    Timeframe,
)
from app.fundamental.sentiment import FundamentalEngine, FundamentalSnapshot
from app.ict.ict_engine import ICTAnalysis, analyse_ict
from app.indicators.indicators import IndicatorSet, compute_indicators
from app.indicators.slopes import SlopeSet, compute_slopes
from app.logger import get_logger
from app.market_structure.structure import MarketStructure, analyse_structure
from app.regime.regime_detector import RegimeReading, detect_regime
from app.rtm.rtm_engine import RTMAnalysis, analyse_rtm
from app.scanner.universe import ScanCandidate, UniverseBuilder

log = get_logger(__name__)

#: Timeframes actually analysed.  1m is fetched for microstructure checks only.
ANALYSIS_TIMEFRAMES: tuple[Timeframe, ...] = (
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.M30,
    Timeframe.H1,
    Timeframe.H4,
    Timeframe.D1,
)

_MIN_BARS: dict[Timeframe, int] = {
    Timeframe.M1: 120,
    Timeframe.M5: 200,
    Timeframe.M15: 200,
    Timeframe.M30: 200,
    Timeframe.H1: 200,
    Timeframe.H4: 180,
    Timeframe.D1: 120,
}


@dataclass(slots=True)
class TimeframeAnalysis:
    """The complete analytical read for one symbol on one timeframe."""

    timeframe: Timeframe
    candles: list[Candle]
    indicators: IndicatorSet
    slopes: SlopeSet
    structure: MarketStructure
    ict: ICTAnalysis
    rtm: RTMAnalysis

    @property
    def close(self) -> float:
        return self.candles[-1].close

    @property
    def atr(self) -> float:
        return self.indicators.last_atr or 0.0

    def bias(self) -> Bias:
        """Blend the three structural readings into one directional view."""

        votes = [self.structure.bias, self.ict.bias(), self.rtm.bias()]
        bullish = sum(1 for v in votes if v is Bias.BULLISH)
        bearish = sum(1 for v in votes if v is Bias.BEARISH)
        conflict = sum(1 for v in votes if v is Bias.CONFLICT)

        stack = self.indicators.ma_stack
        if stack > 0:
            bullish += 1
        elif stack < 0:
            bearish += 1

        if conflict >= 2:
            return Bias.CONFLICT
        if bullish >= bearish + 2:
            return Bias.BULLISH
        if bearish >= bullish + 2:
            return Bias.BEARISH
        if bullish and bearish:
            return Bias.CONFLICT
        return Bias.NEUTRAL

    def as_features(self, prefix: str = "") -> dict[str, float]:
        features: dict[str, float] = {}
        for source in (
            self.indicators.as_features(),
            self.slopes.as_features(),
            self.structure.as_features(),
            self.ict.as_features(),
            self.rtm.as_features(),
        ):
            for key, value in source.items():
                features[f"{prefix}{key}"] = value
        return features


@dataclass(slots=True)
class SymbolAnalysis:
    """Everything known about one symbol at one moment in time."""

    symbol: str
    ticker: Ticker
    candidate: ScanCandidate
    timeframes: dict[Timeframe, TimeframeAnalysis]
    regime: RegimeReading
    htf_bias: Bias
    alignment: float                       # 0..1 agreement across context TFs
    fundamentals: FundamentalSnapshot | None
    order_book: OrderBook | None
    anomaly: str
    ts: int
    errors: list[str] = field(default_factory=list)

    # -- accessors --------------------------------------------------------

    @property
    def primary(self) -> TimeframeAnalysis | None:
        """The execution timeframe used for entry/stop geometry."""

        for timeframe in EXECUTION_TIMEFRAMES:
            if timeframe in self.timeframes:
                return self.timeframes[timeframe]
        return next(iter(self.timeframes.values()), None)

    @property
    def context(self) -> TimeframeAnalysis | None:
        for timeframe in (Timeframe.H1, Timeframe.H4, Timeframe.D1):
            if timeframe in self.timeframes:
                return self.timeframes[timeframe]
        return None

    @property
    def close(self) -> float:
        primary = self.primary
        return primary.close if primary else self.ticker.last

    @property
    def usable(self) -> bool:
        """Enough context to even consider a trade."""

        have_context = sum(1 for tf in CONTEXT_TIMEFRAMES if tf in self.timeframes)
        have_execution = any(tf in self.timeframes for tf in EXECUTION_TIMEFRAMES)
        return have_context >= 2 and have_execution and not self.anomaly

    def bias_for(self, timeframe: Timeframe) -> Bias:
        analysis = self.timeframes.get(timeframe)
        return analysis.bias() if analysis else Bias.NEUTRAL

    def features(self) -> dict[str, float]:
        """Flat feature vector for the ML layer and the audit trail."""

        features: dict[str, float] = {}
        for timeframe, analysis in self.timeframes.items():
            if timeframe in (Timeframe.M15, Timeframe.H1, Timeframe.H4):
                features.update(analysis.as_features(prefix=f"{timeframe.value}_"))
        features.update(self.regime.as_features())
        if self.fundamentals is not None:
            features.update(self.fundamentals.as_features())
        features["alignment"] = self.alignment
        features["htf_bias"] = _bias_value(self.htf_bias)
        features["spread_pct"] = self.candidate.spread_pct
        features["quote_volume_log"] = (
            math.log10(self.candidate.quote_volume)
            if self.candidate.quote_volume > 0
            else 0.0
        )
        return features


def _bias_value(bias: Bias) -> float:
    return {Bias.BULLISH: 1.0, Bias.BEARISH: -1.0, Bias.NEUTRAL: 0.0, Bias.CONFLICT: 0.0}[bias]


def combine_bias(biases: Sequence[Bias], weights: Sequence[float] | None = None) -> tuple[Bias, float]:
    """Combine timeframe biases into a single view plus an alignment score.

    Alignment is the weighted share held by the winning direction, so three
    agreeing timeframes give 1.0 and a straight disagreement gives ~0.5.
    """

    if not biases:
        return Bias.NEUTRAL, 0.0
    weights = list(weights) if weights else [1.0] * len(biases)
    total = sum(weights) or 1.0

    bullish = sum(w for b, w in zip(biases, weights) if b is Bias.BULLISH)
    bearish = sum(w for b, w in zip(biases, weights) if b is Bias.BEARISH)
    conflict = sum(w for b, w in zip(biases, weights) if b is Bias.CONFLICT)

    if conflict / total >= 0.5:
        return Bias.CONFLICT, round(1.0 - conflict / total, 4)
    if bullish > 0 and bearish > 0:
        dominant = max(bullish, bearish)
        if dominant / (bullish + bearish) < 0.7:
            return Bias.CONFLICT, round(dominant / total, 4)
    if bullish > bearish:
        return Bias.BULLISH, round(bullish / total, 4)
    if bearish > bullish:
        return Bias.BEARISH, round(bearish / total, 4)
    return Bias.NEUTRAL, round(1.0 - (bullish + bearish) / total, 4)


class Scanner:
    def __init__(
        self,
        market_data: MarketData,
        universe: UniverseBuilder,
        fundamentals: FundamentalEngine | None = None,
        deep_analysis_count: int = 15,
        concurrency: int = 6,
        timeframes: Sequence[Timeframe] = ANALYSIS_TIMEFRAMES,
    ) -> None:
        self.market_data = market_data
        self.universe = universe
        self.fundamentals = fundamentals
        self.deep_analysis_count = deep_analysis_count
        self.timeframes = tuple(timeframes)
        self._semaphore = asyncio.Semaphore(concurrency)
        self.last_scan_ts = 0.0
        self.last_candidates: list[ScanCandidate] = []
        self.last_errors: dict[str, str] = {}

    # -- stage 1 ----------------------------------------------------------

    async def prescreen(self) -> list[ScanCandidate]:
        contracts = await self.market_data.exchange.contracts()
        tickers = await self.market_data.all_tickers()
        candidates = self.universe.build(contracts, tickers)
        self.last_candidates = candidates
        self.last_scan_ts = time.time()
        return candidates

    # -- stage 2 ----------------------------------------------------------

    async def deep_analyse(
        self, candidates: Sequence[ScanCandidate], tickers: Mapping[str, Ticker] | None = None
    ) -> list[SymbolAnalysis]:
        selected = list(candidates)[: self.deep_analysis_count]
        self.last_errors = {}
        results = await asyncio.gather(
            *(self._analyse_symbol(candidate, tickers) for candidate in selected),
            return_exceptions=True,
        )
        out: list[SymbolAnalysis] = []
        for candidate, result in zip(selected, results):
            if isinstance(result, BaseException):
                self.last_errors[candidate.symbol] = str(result)
                log.warning("deep analysis failed for %s: %s", candidate.symbol, result)
                continue
            if result is not None:
                out.append(result)
        return out

    async def _analyse_symbol(
        self, candidate: ScanCandidate, tickers: Mapping[str, Ticker] | None
    ) -> SymbolAnalysis | None:
        async with self._semaphore:
            symbol = candidate.symbol
            series = await self.market_data.multi_timeframe(
                symbol,
                self.timeframes,
                limit=400,
                min_length=60,
            )
            if not series:
                raise RuntimeError("no timeframe data available")

            analyses: dict[Timeframe, TimeframeAnalysis] = {}
            errors: list[str] = []
            for timeframe, candles in series.items():
                minimum = _MIN_BARS.get(timeframe, 120)
                if len(candles) < min(minimum, 60):
                    errors.append(
                        f"{timeframe.value}: only {len(candles)} bars"
                    )
                    continue
                try:
                    analyses[timeframe] = _build_timeframe_analysis(timeframe, candles)
                except Exception as exc:  # noqa: BLE001 - isolate per timeframe
                    errors.append(f"{timeframe.value}: {exc}")

            if not analyses:
                raise RuntimeError("every timeframe failed analysis: " + "; ".join(errors))

            # Regime is judged on the highest available context timeframe that
            # still reacts within a trading session.
            regime_source = (
                analyses.get(Timeframe.H1)
                or analyses.get(Timeframe.H4)
                or next(iter(analyses.values()))
            )
            regime = detect_regime(
                regime_source.candles,
                regime_source.indicators,
                regime_source.slopes,
                regime_source.structure,
            )

            context_biases: list[Bias] = []
            weights: list[float] = []
            for timeframe, weight in (
                (Timeframe.D1, 3.0),
                (Timeframe.H4, 2.0),
                (Timeframe.H1, 1.5),
            ):
                analysis = analyses.get(timeframe)
                if analysis is not None:
                    context_biases.append(analysis.bias())
                    weights.append(weight)
            htf_bias, alignment = combine_bias(context_biases, weights)

            anomaly = ""
            execution = analyses.get(Timeframe.M15) or analyses.get(Timeframe.M5)
            if execution is not None:
                flagged, message = detect_price_anomaly(execution.candles)
                if flagged:
                    anomaly = message

            ticker = (tickers or {}).get(symbol) or candidate.ticker
            fundamentals = None
            if self.fundamentals is not None:
                try:
                    fundamentals = await self.fundamentals.snapshot(
                        symbol, tickers=tickers, ticker=ticker
                    )
                except Exception as exc:  # noqa: BLE001 - never block on macro
                    errors.append(f"fundamentals: {exc}")

            order_book = None
            try:
                order_book = await self.market_data.order_book(symbol, depth=20)
            except Exception as exc:  # noqa: BLE001 - depth is optional context
                errors.append(f"order book: {exc}")

            return SymbolAnalysis(
                symbol=symbol,
                ticker=ticker,
                candidate=candidate,
                timeframes=analyses,
                regime=regime,
                htf_bias=htf_bias,
                alignment=alignment,
                fundamentals=fundamentals,
                order_book=order_book,
                anomaly=anomaly,
                ts=int(time.time()),
                errors=errors,
            )

    # -- combined ---------------------------------------------------------

    async def scan(self) -> list[SymbolAnalysis]:
        candidates = await self.prescreen()
        if not candidates:
            log.warning("scanner produced no candidates: %s", self.universe.last_report.summary())
            return []
        tickers = await self.market_data.all_tickers()
        return await self.deep_analyse(candidates, tickers)

    def health(self) -> dict[str, object]:
        age = time.time() - self.last_scan_ts if self.last_scan_ts else None
        return {
            "candidates": len(self.last_candidates),
            "deep_errors": len(self.last_errors),
            "seconds_since_scan": round(age, 1) if age is not None else None,
            "universe": self.universe.last_report.summary(),
        }


def _build_timeframe_analysis(timeframe: Timeframe, candles: list[Candle]) -> TimeframeAnalysis:
    indicators = compute_indicators(candles)
    slopes = compute_slopes(indicators, [c.close for c in candles])
    structure = analyse_structure(candles, indicators.atr)
    ict = analyse_ict(candles, indicators.atr, structure)
    rtm = analyse_rtm(candles, indicators.atr)
    return TimeframeAnalysis(
        timeframe=timeframe,
        candles=candles,
        indicators=indicators,
        slopes=slopes,
        structure=structure,
        ict=ict,
        rtm=rtm,
    )


__all__ = [
    "Scanner",
    "SymbolAnalysis",
    "TimeframeAnalysis",
    "ANALYSIS_TIMEFRAMES",
    "combine_bias",
]
