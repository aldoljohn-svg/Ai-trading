"""Market regime detection and regime-specific strategy selection.

The regime answers a different question from the signal: not "which way" but
"what kind of market is this, and is my playbook even valid here?".  Trading a
mean-reversion setup in a violent trend, or a breakout setup inside a dead
range, is how well-built systems still lose money.

Classification uses four measurements, all normalised so they compare across
symbols:

``trend_strength``   ADX and the MA40/80/160 stack agreement
``directionality``   r^2 of the close regression - how linear the move is
``volatility_ratio`` current ATR% versus its own recent median
``expansion``        Bollinger bandwidth versus its own recent median

The result is a single :class:`~app.domain.Regime` plus a confidence, and an
explicit :class:`RegimeStrategy` describing what is allowed in that regime.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Sequence

from app.domain import Bias, Candles, Regime
from app.indicators.indicators import IndicatorSet
from app.indicators.slopes import SlopeSet, r_squared
from app.market_structure.structure import MarketStructure


@dataclass(frozen=True, slots=True)
class RegimeReading:
    regime: Regime
    confidence: float           # 0..1
    trend_strength: float       # 0..1
    directionality: float       # 0..1 (r^2)
    volatility_ratio: float     # current ATR% / median ATR%
    expansion: float            # current bandwidth / median bandwidth
    direction: int              # +1 up, -1 down, 0 none
    notes: list[str] = field(default_factory=list)

    @property
    def is_trending(self) -> bool:
        return self.regime in (Regime.TREND_UP, Regime.TREND_DOWN)

    @property
    def is_dangerous(self) -> bool:
        return self.regime is Regime.HIGH_VOLATILITY

    def as_features(self) -> dict[str, float]:
        return {
            "regime_trend_strength": self.trend_strength,
            "regime_directionality": self.directionality,
            "regime_volatility_ratio": self.volatility_ratio,
            "regime_expansion": self.expansion,
            "regime_direction": float(self.direction),
            "regime_confidence": self.confidence,
            "regime_is_trending": 1.0 if self.is_trending else 0.0,
            "regime_is_range": 1.0 if self.regime is Regime.RANGE else 0.0,
            "regime_is_high_vol": 1.0 if self.regime is Regime.HIGH_VOLATILITY else 0.0,
        }


@dataclass(frozen=True, slots=True)
class RegimeStrategy:
    """What the bot is permitted to do in a given regime."""

    name: str
    allow_entries: bool
    allow_counter_trend: bool
    risk_multiplier: float          # scales risk-per-trade
    min_rr_multiplier: float        # scales the required reward:risk
    confidence_penalty: float       # subtracted from the final confidence
    stop_atr_multiplier: float      # widens/tightens the ATR stop component
    preferred_playbook: str
    rationale: str


_STRATEGIES: dict[Regime, RegimeStrategy] = {
    Regime.TREND_UP: RegimeStrategy(
        name="trend-following-long",
        allow_entries=True,
        allow_counter_trend=False,
        risk_multiplier=1.0,
        min_rr_multiplier=1.0,
        confidence_penalty=0.0,
        stop_atr_multiplier=1.0,
        preferred_playbook="pullback continuation into demand / bullish OB",
        rationale="clean directional trend: trade with it, never against it",
    ),
    Regime.TREND_DOWN: RegimeStrategy(
        name="trend-following-short",
        allow_entries=True,
        allow_counter_trend=False,
        risk_multiplier=1.0,
        min_rr_multiplier=1.0,
        confidence_penalty=0.0,
        stop_atr_multiplier=1.0,
        preferred_playbook="pullback continuation into supply / bearish OB",
        rationale="clean directional trend: trade with it, never against it",
    ),
    Regime.RANGE: RegimeStrategy(
        name="range-fade",
        allow_entries=True,
        allow_counter_trend=True,
        risk_multiplier=0.7,
        min_rr_multiplier=1.0,
        confidence_penalty=0.05,
        stop_atr_multiplier=0.9,
        preferred_playbook="fade the edges, only from the outer third of the range",
        rationale="no trend to follow; edges are the only asymmetric locations",
    ),
    Regime.BREAKOUT: RegimeStrategy(
        name="breakout-continuation",
        allow_entries=True,
        allow_counter_trend=False,
        risk_multiplier=0.8,
        min_rr_multiplier=1.1,
        confidence_penalty=0.05,
        stop_atr_multiplier=1.2,
        preferred_playbook="join displacement, or the first retest of the broken level",
        rationale="expansion is real but fakeouts are common; demand more reward",
    ),
    Regime.HIGH_VOLATILITY: RegimeStrategy(
        name="capital-preservation",
        allow_entries=False,
        allow_counter_trend=False,
        risk_multiplier=0.0,
        min_rr_multiplier=2.0,
        confidence_penalty=0.5,
        stop_atr_multiplier=2.0,
        preferred_playbook="no new entries; manage and de-risk what is open",
        rationale="stops get skipped and slippage explodes; sitting out is the edge",
    ),
    Regime.LOW_VOLATILITY: RegimeStrategy(
        name="coil-watch",
        allow_entries=True,
        allow_counter_trend=False,
        risk_multiplier=0.6,
        min_rr_multiplier=1.2,
        confidence_penalty=0.10,
        stop_atr_multiplier=0.8,
        preferred_playbook="wait for expansion out of the coil; do not pre-position",
        rationale="tight ranges look safe but chop out stops before expanding",
    ),
    Regime.TRANSITION: RegimeStrategy(
        name="transition-caution",
        allow_entries=True,
        allow_counter_trend=False,
        risk_multiplier=0.5,
        min_rr_multiplier=1.25,
        confidence_penalty=0.12,
        stop_atr_multiplier=1.1,
        preferred_playbook="only A+ confluence with a confirmed MSS",
        rationale="structure is changing hands; most setups fail mid-transition",
    ),
    Regime.UNKNOWN: RegimeStrategy(
        name="stand-aside",
        allow_entries=False,
        allow_counter_trend=False,
        risk_multiplier=0.0,
        min_rr_multiplier=2.0,
        confidence_penalty=1.0,
        stop_atr_multiplier=1.5,
        preferred_playbook="none - insufficient information",
        rationale="an unknown regime is never assumed to be a favourable one",
    ),
}


def strategy_for_regime(regime: Regime) -> RegimeStrategy:
    return _STRATEGIES[regime]


def _median(values: Sequence[float]) -> float:
    cleaned = [v for v in values if v is not None]
    return statistics.median(cleaned) if cleaned else 0.0


def detect_regime(
    candles: Candles,
    indicators: IndicatorSet,
    slopes: SlopeSet,
    structure: MarketStructure,
    lookback: int = 60,
    vol_lookback: int = 100,
) -> RegimeReading:
    """Classify the current regime.

    Ordering matters: a dangerous volatility state overrides everything else,
    because "which direction" is irrelevant if the market is untradeable.
    """

    notes: list[str] = []

    if len(candles) < 40:
        return RegimeReading(
            regime=Regime.UNKNOWN,
            confidence=0.0,
            trend_strength=0.0,
            directionality=0.0,
            volatility_ratio=1.0,
            expansion=1.0,
            direction=0,
            notes=["not enough history to classify the regime"],
        )

    closes = [c.close for c in candles[-lookback:]]
    directionality = r_squared(closes)

    adx_value = indicators.last_adx or 0.0
    stack = indicators.ma_stack
    slope_direction = slopes.direction
    trend_strength = min(adx_value / 35.0, 1.0)
    if stack != 0 and stack == slope_direction:
        trend_strength = min(1.0, trend_strength + 0.2)

    # Volatility relative to the symbol's own recent behaviour, not an absolute.
    atr_series = [
        (a / c.close) if (a is not None and c.close > 0) else None
        for a, c in zip(indicators.atr[-vol_lookback:], candles[-vol_lookback:])
    ]
    median_atr_pct = _median([v for v in atr_series if v is not None])
    volatility_ratio = (
        indicators.atr_pct / median_atr_pct if median_atr_pct > 0 else 1.0
    )

    bandwidths = [b for b in indicators.bb_bandwidth[-vol_lookback:] if b is not None]
    median_bandwidth = _median(bandwidths)
    current_bandwidth = indicators.last_bandwidth or 0.0
    expansion = (
        current_bandwidth / median_bandwidth if median_bandwidth > 0 else 1.0
    )

    direction = 0
    if stack != 0:
        direction = stack
    elif slope_direction != 0:
        direction = slope_direction
    elif structure.bias is Bias.BULLISH:
        direction = 1
    elif structure.bias is Bias.BEARISH:
        direction = -1

    # --- classification -------------------------------------------------

    if volatility_ratio >= 2.2 or (volatility_ratio >= 1.8 and expansion >= 2.0):
        notes.append(
            f"ATR is {volatility_ratio:.1f}x its own median - stops are unreliable here"
        )
        return RegimeReading(
            regime=Regime.HIGH_VOLATILITY,
            confidence=round(min(1.0, volatility_ratio / 3.0), 4),
            trend_strength=round(trend_strength, 4),
            directionality=round(directionality, 4),
            volatility_ratio=round(volatility_ratio, 4),
            expansion=round(expansion, 4),
            direction=direction,
            notes=notes,
        )

    if structure.breakout_direction != 0 and expansion >= 1.3 and volatility_ratio >= 1.15:
        notes.append("price closed beyond the prior envelope with expanding bands")
        return RegimeReading(
            regime=Regime.BREAKOUT,
            confidence=round(min(1.0, 0.4 + 0.3 * expansion / 2 + 0.3 * trend_strength), 4),
            trend_strength=round(trend_strength, 4),
            directionality=round(directionality, 4),
            volatility_ratio=round(volatility_ratio, 4),
            expansion=round(expansion, 4),
            direction=structure.breakout_direction,
            notes=notes,
        )

    trending = (
        adx_value >= 22
        and directionality >= 0.45
        and direction != 0
        and slopes.strength >= 0.15
    )
    if trending:
        regime = Regime.TREND_UP if direction > 0 else Regime.TREND_DOWN
        confidence = round(
            min(1.0, 0.35 + 0.3 * trend_strength + 0.35 * directionality), 4
        )
        notes.append(
            f"ADX {adx_value:.0f}, r^2 {directionality:.2f}, MA stack {stack:+d}"
        )
        return RegimeReading(
            regime=regime,
            confidence=confidence,
            trend_strength=round(trend_strength, 4),
            directionality=round(directionality, 4),
            volatility_ratio=round(volatility_ratio, 4),
            expansion=round(expansion, 4),
            direction=direction,
            notes=notes,
        )

    if volatility_ratio <= 0.6 and expansion <= 0.75:
        notes.append("volatility compressed well below its median - coiling")
        return RegimeReading(
            regime=Regime.LOW_VOLATILITY,
            confidence=round(min(1.0, 1.0 - volatility_ratio), 4),
            trend_strength=round(trend_strength, 4),
            directionality=round(directionality, 4),
            volatility_ratio=round(volatility_ratio, 4),
            expansion=round(expansion, 4),
            direction=0,
            notes=notes,
        )

    if structure.in_range and adx_value < 22:
        notes.append("repeated touches of both envelope edges with a weak trend")
        return RegimeReading(
            regime=Regime.RANGE,
            confidence=round(min(1.0, 0.4 + 0.4 * (1.0 - directionality)), 4),
            trend_strength=round(trend_strength, 4),
            directionality=round(directionality, 4),
            volatility_ratio=round(volatility_ratio, 4),
            expansion=round(expansion, 4),
            direction=0,
            notes=notes,
        )

    # Structure has flipped recently but no new trend has established itself.
    last_event = structure.last_event
    if last_event is not None and last_event.type.value in {"CHOCH", "MSS"}:
        notes.append(f"recent {last_event.type.value} - structure changing hands")
        return RegimeReading(
            regime=Regime.TRANSITION,
            confidence=round(min(1.0, 0.35 + 0.4 * last_event.confidence), 4),
            trend_strength=round(trend_strength, 4),
            directionality=round(directionality, 4),
            volatility_ratio=round(volatility_ratio, 4),
            expansion=round(expansion, 4),
            direction=last_event.direction,
            notes=notes,
        )

    notes.append("no measurement crossed a classification threshold")
    return RegimeReading(
        regime=Regime.TRANSITION,
        confidence=0.3,
        trend_strength=round(trend_strength, 4),
        directionality=round(directionality, 4),
        volatility_ratio=round(volatility_ratio, 4),
        expansion=round(expansion, 4),
        direction=direction,
        notes=notes,
    )


__all__ = [
    "RegimeReading",
    "RegimeStrategy",
    "detect_regime",
    "strategy_for_regime",
]
