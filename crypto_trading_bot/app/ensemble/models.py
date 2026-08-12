"""The fifteen analytical models.

Each wraps an existing engine and expresses its view in the common
:class:`~app.ensemble.base.ModelOutput` vocabulary.  They are deliberately thin:
the analysis already lives in ``indicators/``, ``market_structure/``, ``ict/``,
``rtm/``, ``regime/``, ``fundamental/``, ``orderflow/`` and ``ml/``.  What is new
here is that each one now reports *independently*, so agreement and
disagreement become measurable rather than being averaged away inside a single
score.
"""

from __future__ import annotations

from typing import Sequence

from app.domain import Bias, Regime, Sentiment, Side, Timeframe
from app.ensemble.base import AnalyticalModel, ModelContext, ModelOutput, ModelSignal


# --------------------------------------------------------------------------
# Price-derived models
# --------------------------------------------------------------------------


class TechnicalModel(AnalyticalModel):
    """MA stack, RSI location, MACD, ADX."""

    name = "technical"
    family = "price"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        execution = context.execution
        if execution is None:
            return ModelOutput.no_signal(self.name, "no execution timeframe")

        indicators = execution.indicators
        reasons: list[str] = []
        score = 0.0

        stack = indicators.ma_stack
        if stack:
            score += 2.0 * stack
            reasons.append(f"MA40/80/160 stacked {'bullish' if stack > 0 else 'bearish'}")

        rsi = indicators.last_rsi
        if rsi is not None:
            if 50 <= rsi <= 68:
                score += 1.0
                reasons.append(f"RSI {rsi:.0f} in bullish working range")
            elif 32 <= rsi <= 50:
                score -= 1.0
                reasons.append(f"RSI {rsi:.0f} in bearish working range")
            elif rsi > 78:
                score -= 0.5
                reasons.append(f"RSI {rsi:.0f} overextended")
            elif rsi < 22:
                score += 0.5
                reasons.append(f"RSI {rsi:.0f} washed out")

        histogram = indicators.last_macd_hist
        if histogram is not None and histogram != 0:
            score += 1.0 if histogram > 0 else -1.0
            reasons.append("MACD histogram " + ("positive" if histogram > 0 else "negative"))

        adx = indicators.last_adx or 0.0
        strength = min(adx / 30.0, 1.0)
        if adx >= 25:
            reasons.append(f"ADX {adx:.0f} confirms a trending state")

        if score == 0:
            return ModelOutput(
                name=self.name,
                signal=ModelSignal.NEUTRAL,
                confidence=0.3,
                reasoning=["indicators disagree with each other"],
            )

        confidence = min(abs(score) / 4.0, 1.0) * (0.55 + 0.45 * strength)
        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(1 if score > 0 else -1),
            confidence=confidence,
            expected_move_atr=1.0 + 2.0 * strength,
            risk=0.5 - 0.2 * strength,
            reasoning=reasons,
            detail={"adx": adx, "rsi": rsi, "ma_stack": stack},
        )


class PriceActionModel(AnalyticalModel):
    """Swing labelling, retests, rejections, fakeouts, impulse/correction."""

    name = "price_action"
    family = "price"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        execution = context.execution
        if execution is None:
            return ModelOutput.no_signal(self.name, "no execution timeframe")

        structure = execution.structure
        if len(structure.swings) < 4:
            return ModelOutput.no_signal(
                self.name, "not enough confirmed swings", data_quality=0.4
            )

        score = 0.0
        reasons: list[str] = []

        bias_direction = {Bias.BULLISH: 1, Bias.BEARISH: -1}.get(structure.bias, 0)
        if bias_direction:
            score += 2.0 * bias_direction
            reasons.append(f"swing sequence is {structure.bias.value}")
        elif structure.bias is Bias.CONFLICT:
            reasons.append("swing sequence is in conflict")

        if structure.retest_direction:
            score += 1.5 * structure.retest_direction
            reasons.append("broken level retested and held")
        if structure.rejection_direction:
            score += 1.0 * structure.rejection_direction
            reasons.append("wick rejection on the last bar")
        if structure.fakeout_direction:
            # A failed break the other way supports us.
            score += -1.5 * structure.fakeout_direction
            reasons.append("failed breakout in the opposite direction")
        if structure.leg_kind == "impulse" and structure.leg_direction:
            score += 1.0 * structure.leg_direction
            reasons.append(f"impulsive leg of {structure.leg_atr:.1f} ATR")

        if score == 0:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.3,
                reasoning=reasons or ["no directional price action"],
            )

        confidence = min(abs(score) / 5.0, 1.0) * (0.6 + 0.4 * structure.trend_quality)
        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(1 if score > 0 else -1),
            confidence=confidence,
            expected_move_atr=max(structure.leg_atr, 1.0),
            risk=0.6 if structure.in_range else 0.4,
            reasoning=reasons,
            detail={"bias": structure.bias.value, "in_range": structure.in_range},
        )


class MarketStructureModel(AnalyticalModel):
    """BOS / CHOCH / MSS events specifically - the structural break itself."""

    name = "market_structure"
    family = "price"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        execution = context.execution
        if execution is None:
            return ModelOutput.no_signal(self.name, "no execution timeframe")

        structure = execution.structure
        event = structure.last_event
        if event is None:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.25,
                reasoning=["no structural break yet"],
            )

        bars_since = max(len(execution.candles) - 1 - event.index, 0)
        # A break loses force as it ages.
        recency = max(0.0, 1.0 - bars_since / 40.0)
        weight = {"MSS": 1.0, "CHOCH": 0.8, "BOS": 0.6}[event.type.value]

        confidence = event.confidence * weight * (0.4 + 0.6 * recency)
        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(event.direction),
            confidence=confidence,
            expected_move_atr=max(event.displacement_atr * 1.5, 1.0),
            risk=0.45,
            reasoning=[
                f"{event.type.value} {'up' if event.direction > 0 else 'down'} "
                f"at {event.level:.6g}",
                f"displacement {event.displacement_atr:.1f} ATR, {bars_since} bars ago",
            ],
            detail={"event": event.type.value, "bars_since": bars_since},
        )


class ICTModel(AnalyticalModel):
    name = "ict"
    family = "smart_money"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        execution = context.execution
        if execution is None:
            return ModelOutput.no_signal(self.name, "no execution timeframe")

        ict = execution.ict
        long_score = ict.score(1)
        short_score = ict.score(-1)
        edge = (long_score - short_score) / 100.0

        reasons: list[str] = []
        sweep = ict.last_sweep
        if sweep is not None:
            reasons.append(
                f"liquidity sweep {'up' if sweep.direction > 0 else 'down'} "
                f"({sweep.confidence:.0%})"
            )
        displacement = ict.last_displacement
        if displacement is not None:
            reasons.append(f"displacement {displacement.body_atr:.1f} ATR")
        if ict.zone != "equilibrium":
            reasons.append(f"price in {ict.zone}")
        blocks = len(ict.active_order_blocks(1 if edge > 0 else -1))
        if blocks:
            reasons.append(f"{blocks} unmitigated order block(s)")

        if abs(edge) < 0.05:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.3,
                reasoning=reasons or ["no ICT confluence either way"],
            )

        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(1 if edge > 0 else -1),
            confidence=min(abs(edge) * 2.2, 1.0),
            expected_move_atr=2.0,
            risk=0.45,
            reasoning=reasons,
            detail={"long": long_score, "short": short_score, "zone": ict.zone},
        )


class RTMModel(AnalyticalModel):
    name = "rtm"
    family = "smart_money"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        execution = context.execution
        if execution is None:
            return ModelOutput.no_signal(self.name, "no execution timeframe")

        rtm = execution.rtm
        if not rtm.zones:
            return ModelOutput.no_signal(
                self.name, "no RTM zones formed", data_quality=0.5
            )

        edge = (rtm.score(1) - rtm.score(-1)) / 100.0
        reasons: list[str] = []
        active = rtm.active_zone(rtm.close)
        if active is not None:
            reasons.append(
                f"price inside a {active.pattern} "
                f"{'demand' if active.direction > 0 else 'supply'} zone"
            )
        if rtm.engulf_direction:
            reasons.append("engulfing candle")
        if rtm.compression > 0.4:
            reasons.append(f"compression {rtm.compression:.0%}")

        if abs(edge) < 0.05:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.25,
                reasoning=reasons or ["no fresh zone in play"],
            )

        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(1 if edge > 0 else -1),
            confidence=min(abs(edge) * 2.0, 1.0),
            expected_move_atr=2.0,
            risk=0.5,
            reasoning=reasons,
            detail={"compression": rtm.compression},
        )


class MomentumModel(AnalyticalModel):
    """Normalised slopes and their linearity."""

    name = "momentum"
    family = "price"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        execution = context.execution
        if execution is None:
            return ModelOutput.no_signal(self.name, "no execution timeframe")

        slopes = execution.slopes
        direction = slopes.direction
        if direction == 0:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL,
                confidence=0.3 * (1 - slopes.trend_quality),
                reasoning=["moving-average slopes are not aligned"],
            )

        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(direction),
            confidence=min(slopes.strength * 1.2, 1.0),
            expected_move_atr=1.5 + 2.0 * slopes.strength,
            risk=0.4,
            reasoning=[
                f"all slopes aligned {'up' if direction > 0 else 'down'}",
                f"trend linearity r^2 {slopes.trend_quality:.2f}",
            ],
            detail={"strength": slopes.strength, "quality": slopes.trend_quality},
        )


class MeanReversionModel(AnalyticalModel):
    """Only speaks up in a range, and only from the outer third of it.

    Deliberately silent in a trend: fading a trend is how mean-reversion models
    lose more than they ever made.
    """

    name = "mean_reversion"
    family = "price"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        execution = context.execution
        if execution is None:
            return ModelOutput.no_signal(self.name, "no execution timeframe")

        structure = execution.structure
        indicators = execution.indicators

        if context.regime in (Regime.TREND_UP, Regime.TREND_DOWN, Regime.BREAKOUT):
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.1,
                reasoning=[f"stands aside in a {context.regime.value} regime"],
            )
        if not structure.in_range:
            return ModelOutput.no_signal(self.name, "market is not ranging", data_quality=0.8)

        position = structure.range_position
        rsi = indicators.last_rsi or 50.0

        if position <= 0.30 and rsi < 45:
            direction, edge = 1, (0.30 - position) / 0.30
            reasons = [f"at the lower {position:.0%} of the range with RSI {rsi:.0f}"]
        elif position >= 0.70 and rsi > 55:
            direction, edge = -1, (position - 0.70) / 0.30
            reasons = [f"at the upper {position:.0%} of the range with RSI {rsi:.0f}"]
        else:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.2,
                reasoning=["price is mid-range - no asymmetry"],
            )

        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(direction),
            confidence=min(0.35 + 0.5 * edge, 0.85),
            expected_move_atr=1.5,
            risk=0.55,
            reasoning=reasons,
            detail={"range_position": position},
        )


class MarketRegimeModel(AnalyticalModel):
    """Not a direction model so much as a permission model."""

    name = "regime"
    family = "context"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        analysis = context.analysis
        if analysis is None:
            return ModelOutput.no_signal(self.name, "no analysis")

        reading = analysis.regime
        regime = reading.regime

        if regime in (Regime.HIGH_VOLATILITY, Regime.UNKNOWN):
            return ModelOutput(
                name=self.name,
                signal=ModelSignal.NEUTRAL,
                confidence=reading.confidence,
                risk=0.95 if regime is Regime.HIGH_VOLATILITY else 0.8,
                reasoning=[f"{regime.value}: entries should be blocked"] + reading.notes[:2],
                detail={"regime": regime.value},
            )

        direction = reading.direction
        signal = ModelSignal.from_direction(direction)
        return ModelOutput(
            name=self.name,
            signal=signal,
            confidence=reading.confidence if direction else reading.confidence * 0.4,
            expected_move_atr=2.0 * reading.trend_strength,
            risk=_clip01(0.25 + 0.5 * max(reading.volatility_ratio - 1.0, 0.0)),
            reasoning=[f"regime {regime.value}"] + reading.notes[:2],
            detail={
                "regime": regime.value,
                "volatility_ratio": reading.volatility_ratio,
                "directionality": reading.directionality,
            },
        )


# --------------------------------------------------------------------------
# Flow / liquidity models
# --------------------------------------------------------------------------


class OrderFlowModel(AnalyticalModel):
    name = "order_flow"
    family = "flow"
    requires_order_flow = True

    def evaluate(self, context: ModelContext) -> ModelOutput:
        flow = context.order_flow
        if flow is None:
            # No flow read was built at all -- a structurally absent input, not
            # an abstention.  Counting it against participation caps the
            # ensemble's conviction for a reason unrelated to the setup.
            return ModelOutput.unavailable(self.name, "no order-flow data available")
        if flow.data_quality <= 0.2:
            return ModelOutput.no_signal(
                self.name, "order-flow data too incomplete", data_quality=flow.data_quality
            )

        direction = flow.direction
        if direction == 0:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL,
                confidence=0.3, data_quality=flow.data_quality,
                reasoning=flow.reasoning[:4] or ["balanced flow"],
            )

        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(direction),
            confidence=flow.confidence,
            expected_move_atr=1.5,
            risk=0.4,
            data_quality=flow.data_quality,
            reasoning=flow.reasoning[:4],
            detail={"score": flow.score, "cvd_slope": flow.cvd_slope},
        )


class LiquidityModel(AnalyticalModel):
    """Where is the fuel, and which way is price likely to reach for it?"""

    name = "liquidity"
    family = "flow"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        liquidity = context.liquidity
        if liquidity is None:
            return ModelOutput.unavailable(self.name, "no liquidity map")

        target = liquidity.dominant_target(context.price)
        if target is None:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.25,
                reasoning=["no clear liquidity concentration nearby"],
            )

        direction, probability, note = target
        if probability < 0.3:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.25,
                reasoning=[note],
            )
        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(direction),
            confidence=min(probability, 0.85),
            expected_move_atr=liquidity.distance_atr(context.price, direction),
            risk=0.5,
            data_quality=liquidity.data_quality,
            reasoning=[note],
            detail={"probability": probability},
        )


class DerivativesModel(AnalyticalModel):
    """Funding + open interest structure.  Never used in isolation."""

    name = "derivatives"
    family = "flow"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        derivatives = context.derivatives
        if derivatives is None:
            return ModelOutput.unavailable(self.name, "no funding/OI data")
        if not derivatives.available:
            return ModelOutput.no_signal(
                self.name, "funding/OI unavailable", data_quality=0.0
            )

        direction = derivatives.direction
        if direction == 0:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.3,
                data_quality=derivatives.data_quality,
                reasoning=derivatives.reasoning[:3] or ["funding and OI are unremarkable"],
            )
        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(direction),
            # Positioning is a headwind indicator, not a timing one - it never
            # gets to be highly confident on its own.
            confidence=min(derivatives.confidence, 0.6),
            expected_move_atr=1.5,
            risk=derivatives.risk,
            data_quality=derivatives.data_quality,
            reasoning=derivatives.reasoning[:3],
            detail={"regime": derivatives.regime},
        )


# --------------------------------------------------------------------------
# Context models
# --------------------------------------------------------------------------


class MacroModel(AnalyticalModel):
    name = "macro"
    family = "context"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        snapshot = context.fundamentals
        if snapshot is None:
            return ModelOutput.unavailable(self.name, "no fundamental snapshot")

        macro = snapshot.macro
        if macro.btc_trend is Sentiment.UNKNOWN and macro.market_breadth is None:
            return ModelOutput.no_signal(
                self.name, "no macro inputs available", data_quality=snapshot.coverage
            )

        score = 0.0
        reasons: list[str] = []

        if macro.btc_trend is Sentiment.BULLISH:
            score += 1.0
            reasons.append("BTC daily structure is bullish")
        elif macro.btc_trend is Sentiment.BEARISH:
            score -= 1.0
            reasons.append("BTC daily structure is bearish")
        elif macro.btc_trend is Sentiment.DANGER:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.8, risk=0.95,
                data_quality=snapshot.coverage,
                reasoning=["BTC made an extreme daily move - market-wide risk"],
            )

        if macro.market_breadth is not None:
            tilt = (macro.market_breadth - 0.5) * 2
            score += tilt
            reasons.append(f"market breadth {macro.market_breadth:.0%}")

        if abs(score) < 0.25:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.3,
                data_quality=snapshot.coverage, reasoning=reasons or ["macro is mixed"],
            )

        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(1 if score > 0 else -1),
            confidence=min(abs(score) / 2.0, 0.7),
            expected_move_atr=1.5,
            risk=0.5,
            data_quality=snapshot.coverage,
            reasoning=reasons,
        )


class NewsModel(AnalyticalModel):
    """News can veto or shrink.  It is capped so it can never drive a trade."""

    name = "news"
    family = "context"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        snapshot = context.fundamentals
        if snapshot is None:
            return ModelOutput.unavailable(self.name, "no news assessment")

        news = snapshot.news
        if news.sentiment is Sentiment.UNKNOWN:
            return ModelOutput.no_signal(
                self.name, "no news provider configured", data_quality=0.0
            )
        if news.blocks_entries:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.9, risk=1.0,
                reasoning=news.reasons[:3] or ["news blackout in effect"],
                detail={"blocking": True},
            )
        if news.sentiment is Sentiment.NEUTRAL:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.3,
                reasoning=[f"{news.items_considered} headlines, nothing directional"],
            )

        direction = 1 if news.sentiment is Sentiment.BULLISH else -1
        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(direction),
            confidence=0.35,          # keyword classification is crude; cap it hard
            expected_move_atr=1.0,
            risk=0.5,
            reasoning=[f"headline sentiment {news.sentiment.value}"],
        )


class SentimentModel(AnalyticalModel):
    """Positioning-based sentiment (funding as a crowding proxy) + Fear & Greed.

    Contrarian by construction: crowded positioning is a headwind for the
    crowded side.
    """

    name = "sentiment"
    family = "context"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        snapshot = context.fundamentals
        if snapshot is None:
            return ModelOutput.unavailable(self.name, "no sentiment inputs")

        score = 0.0
        reasons: list[str] = []
        quality = 0.0

        funding = snapshot.points.get("funding_rate")
        if funding is not None and funding.available and funding.value is not None:
            quality += 0.5
            crowding = max(-1.0, min(1.0, funding.value / 0.0008))
            score -= crowding          # longs paying -> headwind for longs
            if abs(crowding) > 0.3:
                reasons.append(
                    f"funding {funding.value:+.4%} - "
                    f"{'longs' if crowding > 0 else 'shorts'} are crowded"
                )

        fear_greed = snapshot.macro.fear_greed
        if fear_greed is not None:
            quality += 0.5
            tilt = (fear_greed - 50.0) / 50.0
            score -= tilt * 0.5
            reasons.append(f"Fear & Greed {fear_greed:.0f}")

        if quality <= 0:
            return ModelOutput.unavailable(self.name, "no sentiment data")
        if abs(score) < 0.2:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.25,
                data_quality=quality, reasoning=reasons or ["positioning is balanced"],
            )

        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(1 if score > 0 else -1),
            confidence=min(abs(score) * 0.6, 0.55),
            expected_move_atr=1.0,
            risk=0.5,
            data_quality=quality,
            reasoning=reasons,
        )


class MachineLearningModel(AnalyticalModel):
    name = "ml"
    family = "learned"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        predictor = context.predictor
        if predictor is None or not getattr(predictor, "ready", False):
            return ModelOutput.unavailable(self.name, "no trained model loaded")

        features = context.features or {}
        if not features:
            return ModelOutput.no_signal(self.name, "no feature vector", data_quality=0.0)

        p_long, p_short, p_no_trade = predictor.predict(features)
        if p_no_trade >= 1.0:
            return ModelOutput.no_signal(self.name, "model returned no opinion")

        if p_no_trade >= max(p_long, p_short):
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=p_no_trade,
                reasoning=[f"model favours NO_TRADE at {p_no_trade:.0%}"],
                detail={"p_long": p_long, "p_short": p_short, "p_no_trade": p_no_trade},
            )

        direction = 1 if p_long > p_short else -1
        return ModelOutput(
            name=self.name,
            signal=ModelSignal.from_direction(direction),
            confidence=max(p_long, p_short),
            expected_move_atr=2.0,
            risk=0.5,
            reasoning=[
                f"calibrated P(long)={p_long:.0%} P(short)={p_short:.0%} "
                f"P(no trade)={p_no_trade:.0%}"
            ],
            detail={"p_long": p_long, "p_short": p_short, "p_no_trade": p_no_trade},
        )


class PortfolioRiskModel(AnalyticalModel):
    """Views the trade from the portfolio's perspective, not the chart's."""

    name = "portfolio_risk"
    family = "risk"

    def evaluate(self, context: ModelContext) -> ModelOutput:
        state = context.meta.get("portfolio_state")
        portfolio_risk = context.meta.get("portfolio_risk")
        if state is None or portfolio_risk is None:
            return ModelOutput.unavailable(self.name, "no portfolio state")

        used = portfolio_risk.effective_risk_pct(state)
        cap = portfolio_risk.max_portfolio_risk
        headroom = 1.0 - (used / cap if cap > 0 else 1.0)
        reasons = [
            f"portfolio risk {used:.2%} of a {cap:.2%} cap",
            f"{state.open_positions} position(s) open",
        ]

        existing = {r.symbol.upper() for r in state.open_risks}
        if context.symbol.upper() in existing:
            return ModelOutput(
                name=self.name, signal=ModelSignal.NEUTRAL, confidence=0.9, risk=1.0,
                reasoning=[f"already holding {context.symbol}"],
            )

        # Correlation with what is already open is the real cost of a new trade.
        correlation = 0.0
        for risk in state.open_risks:
            correlation = max(
                correlation, portfolio_risk.matrix.correlation(risk.symbol, context.symbol)
            )
        if correlation >= portfolio_risk.correlation_threshold and state.open_risks:
            reasons.append(f"correlated {correlation:.2f} with an existing position")

        risk_level = _clip01(0.2 + 0.5 * (1 - headroom) + 0.3 * correlation)
        return ModelOutput(
            name=self.name,
            signal=ModelSignal.NEUTRAL,      # never directional by design
            confidence=0.5,
            risk=risk_level,
            reasoning=reasons,
            detail={
                "headroom": round(max(headroom, 0.0), 4),
                "max_correlation": round(correlation, 3),
            },
        )


# --------------------------------------------------------------------------

ALL_MODELS: tuple[type[AnalyticalModel], ...] = (
    TechnicalModel,
    PriceActionModel,
    MarketStructureModel,
    ICTModel,
    RTMModel,
    MomentumModel,
    MeanReversionModel,
    MarketRegimeModel,
    OrderFlowModel,
    LiquidityModel,
    DerivativesModel,
    MacroModel,
    NewsModel,
    SentimentModel,
    MachineLearningModel,
    PortfolioRiskModel,
)


def build_default_models() -> list[AnalyticalModel]:
    return [cls() for cls in ALL_MODELS]


def _clip01(value: float) -> float:
    return 0.0 if value < 0 else 1.0 if value > 1 else value


__all__ = ["ALL_MODELS", "build_default_models"] + [c.__name__ for c in ALL_MODELS]
