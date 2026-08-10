"""Component scoring.

Each component returns ``0..100`` for a *specific direction*.  Keeping them
separate (rather than collapsing early into one number) is what makes the bot
explainable: every trade record stores the full breakdown, so "why did it take
this?" and "why did it skip that?" are answerable after the fact.

All weights live in :data:`DEFAULT_WEIGHTS` so they can be tuned or
walk-forward optimised without touching logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from app.domain import Bias, Regime, Side, Timeframe

if TYPE_CHECKING:  # scanner imports nothing from signals; keep it that way
    from app.scanner.scanner import SymbolAnalysis, TimeframeAnalysis

#: Relative weight of each component in the final opportunity score.
DEFAULT_WEIGHTS: dict[str, float] = {
    "technical": 0.14,
    "structure": 0.16,
    "ict": 0.14,
    "rtm": 0.10,
    "momentum": 0.10,
    "volume": 0.06,
    "volatility": 0.06,
    "htf_alignment": 0.16,
    "fundamental": 0.04,
    "risk_reward": 0.04,
}


@dataclass(slots=True)
class ScoreBreakdown:
    technical: float = 50.0
    structure: float = 50.0
    ict: float = 50.0
    rtm: float = 50.0
    momentum: float = 50.0
    volume: float = 50.0
    volatility: float = 50.0
    htf_alignment: float = 50.0
    fundamental: float = 50.0
    risk_reward: float = 50.0
    ml_probability: float = 0.0
    #: Effective ML weight.  Zero means "no model opinion available", which is
    #: different from "the model says 0%" - a distinction that matters, because
    #: treating an absent model as a zero probability would silently suppress
    #: every score by ``ml_weight``.
    ml_weight: float = 0.0
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def components(self) -> dict[str, float]:
        return {
            "technical": self.technical,
            "structure": self.structure,
            "ict": self.ict,
            "rtm": self.rtm,
            "momentum": self.momentum,
            "volume": self.volume,
            "volatility": self.volatility,
            "htf_alignment": self.htf_alignment,
            "fundamental": self.fundamental,
            "risk_reward": self.risk_reward,
        }

    def weighted(self) -> float:
        """Weighted blend of the rule-based components, 0..100."""

        components = self.components()
        total_weight = sum(self.weights.get(k, 0.0) for k in components)
        if total_weight <= 0:
            return 50.0
        total = sum(components[k] * self.weights.get(k, 0.0) for k in components)
        return round(total / total_weight, 3)

    def blended(self, ml_weight: float | None = None) -> float:
        """Blend the rule-based score with the calibrated ML probability.

        The ML model is a *contributor*, never the decision maker: even at the
        maximum configured weight the rules retain the majority say.  When no
        model opinion is available the weight is zero and the rules stand alone.
        """

        effective = self.ml_weight if ml_weight is None else ml_weight
        effective = max(0.0, min(effective, 0.5))
        rules = self.weighted()
        if effective <= 0.0:
            return rules
        ml_component = self.ml_probability * 100.0
        return round(rules * (1.0 - effective) + ml_component * effective, 3)

    def as_dict(self) -> dict[str, float]:
        out = {k: round(v, 3) for k, v in self.components().items()}
        out["ml_probability"] = round(self.ml_probability, 4)
        out["ml_weight"] = round(self.ml_weight, 4)
        out["weighted"] = self.weighted()
        out["blended"] = self.blended()
        return out


# --------------------------------------------------------------------------
# Component scorers
# --------------------------------------------------------------------------


def score_technical(analysis: TimeframeAnalysis, direction: int) -> float:
    """Indicators: MA stack, RSI location, MACD, ADX."""

    indicators = analysis.indicators
    score = 50.0

    stack = indicators.ma_stack
    if stack == direction:
        score += 16
    elif stack == -direction:
        score -= 18

    rsi = indicators.last_rsi
    if rsi is not None:
        if direction > 0:
            # Strength is good; blow-off extension is not.
            if 50 <= rsi <= 68:
                score += 10
            elif 40 <= rsi < 50:
                score += 4
            elif rsi > 78:
                score -= 10
            elif rsi < 30:
                score -= 4
        else:
            if 32 <= rsi <= 50:
                score += 10
            elif 50 < rsi <= 60:
                score += 4
            elif rsi < 22:
                score -= 10
            elif rsi > 70:
                score -= 4

    histogram = indicators.last_macd_hist
    if histogram is not None:
        if histogram * direction > 0:
            score += 8
        else:
            score -= 6

    adx = indicators.last_adx
    if adx is not None:
        if adx >= 25:
            score += 8
        elif adx < 15:
            score -= 6

    return _clip(score)


def score_structure(analysis: TimeframeAnalysis, direction: int) -> float:
    structure = analysis.structure
    score = structure.score()

    bias_direction = {
        Bias.BULLISH: 1,
        Bias.BEARISH: -1,
        Bias.NEUTRAL: 0,
        Bias.CONFLICT: 0,
    }[structure.bias]
    if bias_direction == direction:
        score += 10
    elif bias_direction == -direction:
        score -= 20

    event = structure.last_event
    if event is not None and event.direction == direction:
        score += 6
    elif event is not None:
        score -= 8

    if structure.retest_direction == direction:
        score += 6
    if structure.rejection_direction == direction:
        score += 4
    if structure.fakeout_direction == -direction:
        # A failed break in the opposite direction supports us.
        score += 6
    elif structure.fakeout_direction == direction:
        score -= 8

    if structure.leg_kind == "impulse" and structure.leg_direction == direction:
        score += 5

    return _clip(score)


def score_ict(analysis: TimeframeAnalysis, direction: int) -> float:
    return analysis.ict.score(direction)


def score_rtm(analysis: TimeframeAnalysis, direction: int) -> float:
    return analysis.rtm.score(direction)


def score_momentum(analysis: TimeframeAnalysis, direction: int) -> float:
    slopes = analysis.slopes
    score = 50.0

    if slopes.direction == direction:
        score += 18 * slopes.strength + 6
    elif slopes.direction == -direction:
        score -= 22

    if slopes.rsi_slope * direction > 0:
        score += 8
    else:
        score -= 5

    if slopes.price_slope * direction > 0:
        score += 6

    score += 12 * slopes.trend_quality - 6
    return _clip(score)


def score_volume(analysis: TimeframeAnalysis, direction: int) -> float:
    metrics = analysis.indicators.volume
    score = 50.0

    relative = metrics.get("relative_volume", 1.0)
    if relative >= 1.6:
        score += 14
    elif relative >= 1.15:
        score += 8
    elif relative < 0.6:
        score -= 12

    trend = metrics.get("volume_trend", 0.0)
    score += max(-10.0, min(10.0, trend * 20))

    pressure = metrics.get("buy_pressure", 0.5)
    # 0.5 is neutral; map the deviation onto the traded direction.
    score += (pressure - 0.5) * 40 * direction

    return _clip(score)


def score_volatility(analysis: TimeframeAnalysis, min_atr_pct: float, max_atr_pct: float) -> float:
    """Directionless: is volatility in the band where our stops make sense?"""

    atr_pct = analysis.indicators.atr_pct
    if atr_pct <= 0:
        return 20.0
    if atr_pct < min_atr_pct:
        # Too quiet: the stop would sit inside the noise band.
        return _clip(50.0 * atr_pct / max(min_atr_pct, 1e-9))
    if atr_pct > max_atr_pct:
        return _clip(50.0 * max_atr_pct / atr_pct)

    # Reward the middle of the usable band.
    span = max_atr_pct - min_atr_pct
    position = (atr_pct - min_atr_pct) / span if span > 0 else 0.5
    return _clip(100.0 - abs(position - 0.35) * 90.0)


def score_htf_alignment(symbol_analysis: SymbolAnalysis, direction: int) -> float:
    """How strongly the daily / 4H / 1H context supports this direction."""

    wanted = Bias.BULLISH if direction > 0 else Bias.BEARISH
    opposite = Bias.BEARISH if direction > 0 else Bias.BULLISH

    score = 50.0
    weights = {Timeframe.D1: 3.0, Timeframe.H4: 2.0, Timeframe.H1: 1.5}
    total = 0.0
    agree = 0.0
    for timeframe, weight in weights.items():
        analysis = symbol_analysis.timeframes.get(timeframe)
        if analysis is None:
            continue
        total += weight
        bias = analysis.bias()
        if bias is wanted:
            agree += weight
        elif bias is opposite:
            agree -= weight
        elif bias is Bias.CONFLICT:
            agree -= weight * 0.5

    if total > 0:
        score += 45.0 * (agree / total)

    if symbol_analysis.htf_bias is wanted:
        score += 6
    elif symbol_analysis.htf_bias is opposite:
        score -= 14
    elif symbol_analysis.htf_bias is Bias.CONFLICT:
        score -= 10

    score += 10.0 * (symbol_analysis.alignment - 0.5)
    return _clip(score)


def score_fundamental(symbol_analysis: SymbolAnalysis, side: Side) -> float:
    """50 (perfectly neutral) whenever nothing is actually known."""

    snapshot = symbol_analysis.fundamentals
    if snapshot is None:
        return 50.0
    return snapshot.score(side)


def score_risk_reward(rr: float, min_rr: float) -> float:
    """Map achieved reward:risk onto 0..100 around the configured minimum."""

    if rr <= 0:
        return 0.0
    if rr < min_rr:
        return _clip(50.0 * rr / min_rr)
    # Above the minimum, reward diminishing returns - 4R is not twice as good
    # as 2R in practice because it is far less likely to be reached.
    excess = (rr - min_rr) / max(min_rr, 0.1)
    return _clip(50.0 + 50.0 * (1.0 - 1.0 / (1.0 + excess)))


def score_direction(
    symbol_analysis: SymbolAnalysis,
    side: Side,
    rr: float,
    min_rr: float,
    min_atr_pct: float,
    max_atr_pct: float,
    ml_probability: float = 0.0,
    ml_weight: float = 0.0,
    weights: Mapping[str, float] | None = None,
) -> ScoreBreakdown:
    """Full component breakdown for one candidate direction.

    ``ml_weight`` must be zero unless a trained model actually produced
    ``ml_probability``.
    """

    direction = side.sign
    execution = symbol_analysis.primary
    context = symbol_analysis.timeframes.get(Timeframe.H1) or execution

    if execution is None or context is None:
        return ScoreBreakdown(weights=dict(weights or DEFAULT_WEIGHTS))

    # Execution timeframe drives entry quality; context timeframe drives trend
    # quality.  Blending both avoids chasing a 5-minute pattern with no context
    # and avoids entering a great trend at a terrible location.
    return ScoreBreakdown(
        technical=_blend(
            score_technical(execution, direction), score_technical(context, direction)
        ),
        structure=_blend(
            score_structure(execution, direction), score_structure(context, direction)
        ),
        ict=_blend(score_ict(execution, direction), score_ict(context, direction)),
        rtm=_blend(score_rtm(execution, direction), score_rtm(context, direction)),
        momentum=_blend(
            score_momentum(execution, direction), score_momentum(context, direction)
        ),
        volume=score_volume(execution, direction),
        volatility=score_volatility(execution, min_atr_pct, max_atr_pct),
        htf_alignment=score_htf_alignment(symbol_analysis, direction),
        fundamental=score_fundamental(symbol_analysis, side),
        risk_reward=score_risk_reward(rr, min_rr),
        ml_probability=ml_probability,
        ml_weight=ml_weight,
        weights=dict(weights or DEFAULT_WEIGHTS),
    )


def _blend(execution: float, context: float, execution_weight: float = 0.55) -> float:
    return round(execution * execution_weight + context * (1 - execution_weight), 3)


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return low if value < low else high if value > high else value


__all__ = [
    "ScoreBreakdown",
    "score_direction",
    "score_technical",
    "score_structure",
    "score_ict",
    "score_rtm",
    "score_momentum",
    "score_volume",
    "score_volatility",
    "score_htf_alignment",
    "score_fundamental",
    "score_risk_reward",
    "DEFAULT_WEIGHTS",
]
