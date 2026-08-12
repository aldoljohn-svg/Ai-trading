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
from app.exchange.base import ExchangeRateLimit
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


def directional_bias(
    structure: MarketStructure,
    ict: ICTAnalysis,
    rtm: RTMAnalysis,
    indicators: IndicatorSet,
) -> Bias:
    """Blend the structural readings into one directional view.

    A free function rather than only a method because the meta-label trainer
    needs the identical decision when replaying history.  If the two ever
    diverged, the model would be graded on an engine that does not exist.

    Three structural votes plus the moving-average stack.  A clear majority of
    two is required to call a direction; two conflicting reads, or one vote each
    way, is CONFLICT rather than a coin toss.
    """

    votes = [structure.bias, ict.bias(), rtm.bias()]
    bullish = sum(1 for v in votes if v is Bias.BULLISH)
    bearish = sum(1 for v in votes if v is Bias.BEARISH)
    conflict = sum(1 for v in votes if v is Bias.CONFLICT)

    stack = indicators.ma_stack
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

        return directional_bias(self.structure, self.ict, self.rtm, self.indicators)

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
class ScreenResult:
    """A cheap, single-timeframe read used to rank candidates for deep analysis."""

    candidate: ScanCandidate
    score: float
    trend: float = 0.0
    momentum: float = 0.0
    volatility: float = 0.0
    expansion: float = 0.0
    direction: int = 0
    note: str = ""

    @property
    def symbol(self) -> str:
        return self.candidate.symbol

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 2),
            "trend": round(self.trend, 1),
            "momentum": round(self.momentum, 1),
            "volatility": round(self.volatility, 1),
            "expansion": round(self.expansion, 1),
            "direction": self.direction,
            "note": self.note,
        }


def score_activity(
    candidate: ScanCandidate, candles: Sequence[Candle]
) -> ScreenResult:
    """Score how *interesting* a symbol is right now from one timeframe.

    A ticker says a symbol is liquid.  It does not say whether anything is
    happening on the chart, which is why ranking on 24h volume alone surfaces
    the same permanently-liquid majors every cycle while a coin in a clean trend
    with expanding range never gets looked at.

    Four components:

    * **trend** - a directional MA stack, weighted by how linear the Average
      line is.  A stack that exists only because price is chopping across the
      averages scores low.
    * **momentum** - ADX for strength, tempered when RSI says the move is
      already stretched and has less room left.
    * **volatility fit** - enough range to pay costs, not so much that stops
      have to be absurdly wide.
    * **expansion** - bandwidth at either extreme of its recent distribution.
      Both a fresh squeeze and a fresh expansion are worth attention; the dead
      middle is not.

    Used by the scanner's screen stage and by the trainer when choosing which
    symbols to learn from, so both mean the same thing by "worth looking at".
    """

    if len(candles) < 60:
        return ScreenResult(
            candidate=candidate,
            score=candidate.prescreen_score * 0.5,
            note="not enough history to screen",
        )

    indicators = compute_indicators(list(candles))
    slopes = compute_slopes(indicators, [c.close for c in candles])

    atr = indicators.last_atr or 0.0
    close = indicators.close or candles[-1].close
    adx = indicators.last_adx or 0.0
    rsi = indicators.last_rsi or 50.0
    stack = indicators.ma_stack

    trend = 0.0
    if stack != 0:
        trend = 60.0 + 40.0 * min(slopes.trend_quality, 1.0)
    elif slopes.aligned_bullish or slopes.aligned_bearish:
        trend = 45.0

    momentum = min(adx / 40.0, 1.0) * 100.0
    if rsi > 75 or rsi < 25:
        momentum *= 0.7

    atr_pct = (atr / close) if close > 0 else 0.0
    if atr_pct <= 0:
        volatility = 0.0
    elif atr_pct < 0.002:
        volatility = 100.0 * atr_pct / 0.002        # too quiet to pay costs
    elif atr_pct <= 0.03:
        volatility = 100.0
    else:
        volatility = max(0.0, 100.0 * (1.0 - (atr_pct - 0.03) / 0.07))

    bandwidth = indicators.last_bandwidth or 0.0
    history = [b for b in indicators.bb_bandwidth[-60:] if b is not None]
    expansion = 0.0
    if history and bandwidth > 0:
        ordered = sorted(history)
        rank = sum(1 for b in ordered if b <= bandwidth) / len(ordered)
        expansion = 100.0 * abs(rank - 0.5) * 2.0

    score = (
        0.35 * trend
        + 0.25 * momentum
        + 0.20 * volatility
        + 0.10 * expansion
        + 0.10 * candidate.prescreen_score
    )

    direction = stack if stack != 0 else (1 if slopes.average_slope > 0 else -1)
    note = (
        f"{'up' if direction > 0 else 'down'} trend {trend:.0f}, "
        f"ADX {adx:.0f}, ATR {atr_pct:.2%}"
    )
    return ScreenResult(
        candidate=candidate,
        score=score,
        trend=trend,
        momentum=momentum,
        volatility=volatility,
        expansion=expansion,
        direction=direction,
        note=note,
    )


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
    #: Why the order book is missing, when it is.  Empty when the book is fine.
    #: Carried so the decision journal can say "depth fetch failed: timeout"
    #: rather than the useless "no order book available".
    book_problem: str = ""

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
        screen_count: int = 0,
        screen_timeframe: Timeframe = Timeframe.H1,
        screen_concurrency: int = 12,
    ) -> None:
        self.market_data = market_data
        self.universe = universe
        self.fundamentals = fundamentals
        self.deep_analysis_count = deep_analysis_count
        self.timeframes = tuple(timeframes)
        self.screen_count = screen_count
        self.screen_timeframe = screen_timeframe
        self._semaphore = asyncio.Semaphore(concurrency)
        self._screen_semaphore = asyncio.Semaphore(screen_concurrency)
        self.last_scan_ts = 0.0
        self.last_candidates: list[ScanCandidate] = []
        self.last_screened: list[ScreenResult] = []
        self.last_errors: dict[str, str] = {}

    # -- stage 1 ----------------------------------------------------------

    async def prescreen(self) -> list[ScanCandidate]:
        contracts = await self.market_data.exchange.contracts()
        tickers = await self.market_data.all_tickers()
        candidates = self.universe.build(contracts, tickers)
        self.last_candidates = candidates
        self.last_scan_ts = time.time()
        return candidates

    # -- stage 1.5: cheap screen -------------------------------------------

    async def screen(
        self, candidates: Sequence[ScanCandidate]
    ) -> list[ScreenResult]:
        """Rank a wide set of symbols using one timeframe instead of six.

        The pre-screen only sees a ticker: volume, spread, 24h range.  That is
        enough to reject the obviously untradable but says nothing about
        whether anything is *happening* on the chart right now, so the top of
        that ranking is dominated by whatever is permanently liquid rather than
        whatever is interesting today.

        This stage pulls a single timeframe and asks a narrower question: is
        there a trend worth trading, is momentum behind it, is volatility
        expanding, and is price at a decision point?  One fetch per symbol
        instead of six means several times as many symbols can be examined for
        the same budget, and the candle cache makes it nearly free on repeat
        cycles.

        The score decides who gets deep analysis.  It never decides who gets
        traded -- every gate downstream still applies in full.
        """

        selected = list(candidates)[: self.screen_count or len(candidates)]
        results = await asyncio.gather(
            *(self._screen_symbol(candidate) for candidate in selected),
            return_exceptions=True,
        )

        out: list[ScreenResult] = []
        for candidate, result in zip(selected, results):
            if isinstance(result, BaseException):
                # A screen failure is not worth logging per symbol; the symbol
                # simply keeps its pre-screen ranking.
                out.append(
                    ScreenResult(
                        candidate=candidate,
                        score=candidate.prescreen_score * 0.5,
                        note=f"screen unavailable: {result}",
                    )
                )
                continue
            out.append(result)

        out.sort(key=lambda r: r.score, reverse=True)
        self.last_screened = out
        return out

    async def _screen_symbol(self, candidate: ScanCandidate) -> ScreenResult:
        async with self._screen_semaphore:
            candles = await self.market_data.candles(
                candidate.symbol,
                self.screen_timeframe,
                limit=240,
                min_length=200,
            )
        return score_activity(candidate, candles)

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
            book_problem = ""
            try:
                fetched = await self.market_data.order_book(symbol, depth=20)
            except ExchangeRateLimit as exc:
                # We asked too fast.  That is a fact about this process, not
                # about the instrument, and benching BTC for thirty minutes
                # because of our own request rate is not a judgement the
                # scanner is entitled to make.  Report it and try again next
                # cycle; the throttling itself is handled upstream.
                book_problem = f"depth throttled by the venue: {exc}"
                errors.append(f"order book: {exc}")
            except Exception as exc:  # noqa: BLE001 - depth is optional context
                book_problem = f"depth fetch failed: {exc}"
                errors.append(f"order book: {exc}")
                self.universe.bench(symbol, book_problem)
            else:
                if fetched.bids and fetched.asks:
                    order_book = fetched
                    self.universe.unbench(symbol)
                else:
                    # An empty book is not a transient error, it means nothing
                    # is quoting.  Bench it rather than paying for the analysis
                    # again next cycle only to reject it for the same reason.
                    book_problem = (
                        f"exchange returned {len(fetched.bids)} bids / "
                        f"{len(fetched.asks)} asks"
                    )
                    errors.append(f"order book is empty ({book_problem})")
                    self.universe.bench(symbol, "order book came back empty")

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
                book_problem=book_problem,
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

        if self.screen_count:
            screened = await self.screen(candidates)
            candidates = [result.candidate for result in screened]
            log.info(
                "screened %d symbols on %s, deep-analysing the top %d",
                len(screened),
                self.screen_timeframe.value,
                min(self.deep_analysis_count, len(candidates)),
            )

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
