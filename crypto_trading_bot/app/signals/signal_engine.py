"""The hybrid decision engine.

Rules + statistics + machine learning + regime + risk, in that order of
authority.  The rule layer decides *whether a thesis exists*; the ML layer only
adjusts confidence; the regime layer can veto; the risk engine (separate
module) has the final word on size and permission.

Decision flow for one symbol
----------------------------
1. Is the data usable at all?  (context timeframes present, no anomaly)
2. Which direction, if any, does higher-timeframe context permit?
3. Where is the invalidation?  -> stop loss
4. Where is the market likely to travel?  -> TP1/TP2/TP3
5. Is the resulting reward:risk acceptable?
6. Score every component, fold in the calibrated ML probability.
7. Apply the regime playbook and every hard gate.

``NO_TRADE`` is the default outcome and requires no justification; entering
requires *all* gates to pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

from app.config import Settings
from app.domain import Bias, Regime, Side, Timeframe, clamp
from app.ict.ict_engine import ICTAnalysis
from app.logger import get_logger
from app.market_structure.structure import MarketStructure
from app.regime.regime_detector import RegimeStrategy, strategy_for_regime
from app.scanner.scanner import SymbolAnalysis, TimeframeAnalysis
from app.signals.scoring import ScoreBreakdown, score_direction
from app.signals.trade_proposal import Decision, TradeProposal

log = get_logger(__name__)

#: A structural level further than this is not treated as a take profit.
MAX_TARGET_R = 8.0
MAX_TARGET_PCT = 0.35


@dataclass(frozen=True, slots=True)
class StopPlan:
    price: float
    rationale: str
    source: str


@dataclass(frozen=True, slots=True)
class TargetPlan:
    prices: tuple[float, float, float]
    rationale: list[str]


class SignalEngine:
    def __init__(self, settings: Settings, predictor: object | None = None) -> None:
        self.settings = settings
        self.predictor = predictor

    # -- entry point ------------------------------------------------------

    def evaluate(self, analysis: SymbolAnalysis) -> TradeProposal:
        settings = self.settings
        proposal = TradeProposal(
            symbol=analysis.symbol,
            side=None,
            decision=Decision.NO_TRADE,
            regime=analysis.regime.regime,
            htf_bias=analysis.htf_bias,
            alignment=analysis.alignment,
            ts=int(time.time()),
        )

        # --- 1. data usability -------------------------------------------
        if not analysis.usable:
            if analysis.anomaly:
                return proposal.reject(f"market data anomaly: {analysis.anomaly}")
            return proposal.reject(
                "insufficient multi-timeframe context (need 2 of 1D/4H/1H plus an "
                "execution timeframe)"
            )

        execution = analysis.primary
        context = analysis.timeframes.get(Timeframe.H1) or execution
        if execution is None or context is None:
            return proposal.reject("no execution timeframe available")

        proposal.atr = execution.atr
        proposal.features = analysis.features()

        if execution.atr <= 0:
            return proposal.reject("ATR is zero - cannot size a stop")

        # --- 2. direction ------------------------------------------------
        strategy = strategy_for_regime(analysis.regime.regime)
        side = self._choose_side(analysis, strategy)
        if side is None:
            if analysis.htf_bias is Bias.CONFLICT:
                return proposal.reject(
                    "higher-timeframe context is in CONFLICT - no directional edge"
                )
            return proposal.reject(
                f"no directional thesis under the {strategy.name} playbook "
                f"({analysis.regime.regime.value})"
            )
        proposal.side = side

        # --- 3. stop loss -------------------------------------------------
        entry = self._entry_price(analysis, side)
        proposal.entry = entry
        stop = self.compute_stop(analysis, side, entry)
        if stop is None:
            return proposal.reject("could not locate a valid invalidation level")
        proposal.stop_loss = stop.price
        proposal.stop_rationale = stop.rationale
        proposal.stop_distance = abs(entry - stop.price)

        stop_pct = proposal.stop_distance / entry if entry > 0 else 0.0
        if stop_pct < settings.min_stop_pct:
            return proposal.reject(
                f"stop distance {stop_pct:.3%} is inside the noise band "
                f"(minimum {settings.min_stop_pct:.3%})"
            )
        if stop_pct > settings.max_stop_pct:
            return proposal.reject(
                f"stop distance {stop_pct:.2%} exceeds the maximum "
                f"{settings.max_stop_pct:.2%} - the setup is too wide to size safely"
            )

        # --- 4. targets ---------------------------------------------------
        targets = self.compute_targets(analysis, side, entry, stop.price)
        proposal.tp1, proposal.tp2, proposal.tp3 = targets.prices
        proposal.target_rationale = targets.rationale

        risk = proposal.stop_distance
        proposal.rr = round(abs(proposal.tp2 - entry) / risk, 3) if risk > 0 else 0.0
        proposal.rr_weighted = round(
            self._weighted_rr(entry, stop.price, targets.prices, side), 3
        )

        # --- 5. reward:risk gate -----------------------------------------
        required_rr = settings.min_rr * strategy.min_rr_multiplier
        if proposal.rr < required_rr:
            proposal.reject(
                f"reward:risk {proposal.rr:.2f} is below the {required_rr:.2f} "
                f"required in a {analysis.regime.regime.value} regime"
            )

        # --- 6. scoring + ML ---------------------------------------------
        probabilities = self._predict(analysis, proposal)
        proposal.p_long, proposal.p_short, proposal.p_no_trade = probabilities

        # A meta model answers "will this side reach its target first", which is
        # exactly the number wanted here.  A direction model's unconditional
        # p_long / p_short is the fallback.
        meta_probability = self._predict_meta(proposal, side)
        if meta_probability is not None:
            ml_probability = meta_probability
            ml_available = True
            if side is Side.LONG:
                proposal.p_long = meta_probability
            else:
                proposal.p_short = meta_probability
            proposal.p_no_trade = round(1.0 - meta_probability, 4)
        else:
            ml_probability = proposal.p_long if side is Side.LONG else proposal.p_short
            # An absent model contributes nothing rather than a zero probability.
            ml_available = proposal.p_no_trade < 1.0

        ml_weight = settings.ml_weight if ml_available else 0.0

        breakdown = score_direction(
            analysis,
            side,
            rr=proposal.rr,
            min_rr=settings.min_rr,
            min_atr_pct=settings.min_atr_pct,
            max_atr_pct=settings.max_atr_pct,
            ml_probability=ml_probability,
            ml_weight=ml_weight,
        )
        proposal.scores = breakdown

        confidence = breakdown.blended() / 100.0
        confidence -= strategy.confidence_penalty
        if analysis.htf_bias is Bias.CONFLICT:
            confidence -= 0.15
        confidence *= 0.6 + 0.4 * analysis.regime.confidence
        # The system never expresses certainty.
        proposal.confidence = round(clamp(confidence, 0.0, 0.95), 4)

        proposal.suggested_leverage = self._suggest_leverage(analysis, stop_pct)
        proposal.reasons = self._build_reasons(analysis, side, stop, targets, breakdown)

        # --- 7. hard gates -----------------------------------------------
        self._apply_gates(analysis, proposal, strategy, ml_probability)

        if not proposal.rejections:
            proposal.decision = Decision.ENTER
        return proposal

    # -- direction --------------------------------------------------------

    def _choose_side(
        self, analysis: SymbolAnalysis, strategy: RegimeStrategy
    ) -> Side | None:
        """Higher timeframes grant permission; lower timeframes never override."""

        if not strategy.allow_entries:
            return None
        if analysis.htf_bias is Bias.CONFLICT:
            return None

        regime = analysis.regime.regime
        execution = analysis.primary
        if execution is None:
            return None

        if regime is Regime.TREND_UP:
            return Side.LONG if analysis.htf_bias is not Bias.BEARISH else None
        if regime is Regime.TREND_DOWN:
            return Side.SHORT if analysis.htf_bias is not Bias.BULLISH else None

        if regime is Regime.BREAKOUT:
            direction = analysis.regime.direction
            if direction > 0 and analysis.htf_bias is not Bias.BEARISH:
                return Side.LONG
            if direction < 0 and analysis.htf_bias is not Bias.BULLISH:
                return Side.SHORT
            return None

        if regime is Regime.RANGE:
            # Only fade from the outer third of the range, and only in the
            # direction that has room to travel.
            structure = execution.structure
            if structure.range_position <= 0.33:
                return Side.LONG
            if structure.range_position >= 0.67:
                return Side.SHORT
            return None

        # LOW_VOLATILITY / TRANSITION: require the HTF to be unambiguous.
        if analysis.htf_bias is Bias.BULLISH:
            return Side.LONG
        if analysis.htf_bias is Bias.BEARISH:
            return Side.SHORT
        return None

    def _entry_price(self, analysis: SymbolAnalysis, side: Side) -> float:
        """Enter at the crossing price, not the mid - we pay the spread."""

        ticker = analysis.ticker
        if side is Side.LONG and ticker.ask > 0:
            return ticker.ask
        if side is Side.SHORT and ticker.bid > 0:
            return ticker.bid
        return analysis.close

    # -- stop loss --------------------------------------------------------

    def compute_stop(
        self, analysis: SymbolAnalysis, side: Side, entry: float
    ) -> StopPlan | None:
        """Place the stop where the *thesis* is wrong, then sanity-clamp it.

        Candidates, in order of preference:

        1. beyond the swing that defines the current leg
        2. beyond the demand/supply zone or order block being traded from
        3. beyond the liquidity pool that would be swept on invalidation
        4. a pure ATR distance (always available as a floor)

        We take the *furthest* structural candidate that still fits inside the
        configured maximum, because a stop that sits just inside obvious
        liquidity is a stop that gets taken for no reason.
        """

        settings = self.settings
        execution = analysis.primary
        if execution is None:
            return None

        atr = execution.atr
        if atr <= 0:
            return None

        structure = execution.structure
        ict = execution.ict
        buffer = settings.stop_structure_buffer_atr * atr
        candidates: list[tuple[float, str, str]] = []

        # 1. structural swing
        swing = _last_swing_beyond(structure, side, entry)
        if swing is not None:
            price = swing - buffer if side is Side.LONG else swing + buffer
            candidates.append((price, f"beyond the swing at {swing:.6g}", "swing"))

        # 2. order block / RTM zone being traded from
        blocks = ict.active_order_blocks(side.sign)
        if blocks:
            block = min(blocks, key=lambda b: abs(b.mid - entry))
            price = (
                block.bottom - buffer if side is Side.LONG else block.top + buffer
            )
            candidates.append((price, f"beyond the {side.value} order block", "order_block"))

        zone = execution.rtm.nearest_zone(side.sign, entry)
        if zone is not None and zone.distance_pct(entry) < 0.02:
            price = zone.bottom - buffer if side is Side.LONG else zone.top + buffer
            candidates.append((price, f"beyond the {zone.pattern} zone", "rtm_zone"))

        # 3. liquidity that would be swept if we are wrong
        pool = ict.nearest_liquidity(-side.sign, entry)
        if pool is not None:
            price = pool.price - buffer if side is Side.LONG else pool.price + buffer
            candidates.append(
                (price, f"beyond resting liquidity at {pool.price:.6g}", "liquidity")
            )

        # 4. ATR floor - always present
        strategy = strategy_for_regime(analysis.regime.regime)
        atr_distance = settings.stop_atr_mult * strategy.stop_atr_multiplier * atr
        atr_stop = entry - atr_distance if side is Side.LONG else entry + atr_distance
        candidates.append((atr_stop, f"{settings.stop_atr_mult:.2g}x ATR", "atr"))

        # Keep only candidates on the correct side of entry.
        valid = [
            c
            for c in candidates
            if (c[0] < entry if side is Side.LONG else c[0] > entry)
        ]
        if not valid:
            return None

        max_distance = settings.max_stop_pct * entry
        min_distance = max(settings.min_stop_pct * entry, 0.45 * atr)

        within = [
            c for c in valid if min_distance <= abs(entry - c[0]) <= max_distance
        ]
        if within:
            # Furthest acceptable structural stop.
            chosen = max(within, key=lambda c: abs(entry - c[0]))
        else:
            # Nothing fits: fall back to the ATR stop clamped into range.
            distance = clamp(atr_distance, min_distance, max_distance)
            price = entry - distance if side is Side.LONG else entry + distance
            chosen = (price, f"{settings.stop_atr_mult:.2g}x ATR (clamped)", "atr")

        return StopPlan(price=chosen[0], rationale=chosen[1], source=chosen[2])

    # -- targets ----------------------------------------------------------

    def compute_targets(
        self, analysis: SymbolAnalysis, side: Side, entry: float, stop: float
    ) -> TargetPlan:
        """Targets come from where the market is *likely to go*, not from a ratio.

        Structural targets (liquidity pools, support/resistance, fair value
        gaps, range boundaries) are collected first; R-multiples only fill the
        gaps.  This is what makes the reward:risk gate meaningful - if the only
        thing above us is 1.2R away, the trade is correctly rejected.
        """

        settings = self.settings
        execution = analysis.primary
        context = analysis.timeframes.get(Timeframe.H1) or execution
        risk = abs(entry - stop)
        rationale: list[str] = []

        if execution is None or risk <= 0:
            fallback = _r_multiple_targets(entry, risk, side, settings)
            return TargetPlan(prices=fallback, rationale=["R-multiple ladder"])

        raw: list[tuple[float, str]] = []
        direction = side.sign

        for source in (execution, context):
            if source is None:
                continue
            label = source.timeframe.value

            for pool in source.ict.liquidity:
                if pool.swept:
                    continue
                if _beyond(pool.price, entry, direction):
                    raw.append((pool.price, f"{label} {pool.kind} liquidity"))

            levels = (
                source.structure.resistance if direction > 0 else source.structure.support
            )
            for level in levels:
                if _beyond(level.price, entry, direction):
                    raw.append((level.price, f"{label} {level.kind} ({level.touches} touches)"))

            gap = source.ict.nearest_fvg(-direction, entry)
            if gap is not None and _beyond(gap.mid, entry, direction):
                raw.append((gap.mid, f"{label} opposing fair value gap"))

            envelope = (
                source.structure.range_high if direction > 0 else source.structure.range_low
            )
            if _beyond(envelope, entry, direction):
                raw.append((envelope, f"{label} range boundary"))

        # Order by distance from entry and drop anything too close to be a
        # meaningful target (fees and spread would eat it) or so far away that
        # quoting it as a target would be dishonest - a level 40R away is not a
        # take profit, it is a different market.
        minimum = max(0.6 * risk, 0.5 * execution.atr)
        maximum = min(MAX_TARGET_R * risk, MAX_TARGET_PCT * entry)
        ordered = sorted(
            {round(price, 10): note for price, note in raw}.items(),
            key=lambda item: abs(item[0] - entry),
        )
        usable = [
            (p, n) for p, n in ordered if minimum <= abs(p - entry) <= maximum
        ]

        # Deduplicate targets that sit within a fraction of ATR of each other.
        selected: list[tuple[float, str]] = []
        for price, note in usable:
            if all(abs(price - chosen) > 0.5 * execution.atr for chosen, _ in selected):
                selected.append((price, note))
            if len(selected) == 3:
                break

        ladder = _r_multiple_targets(entry, risk, side, settings)
        prices: list[float] = []
        for index in range(3):
            if index < len(selected):
                price, note = selected[index]
                prices.append(price)
                rationale.append(
                    f"TP{index + 1} {price:.6g}: {note} "
                    f"({abs(price - entry) / risk:.2f}R)"
                )
            else:
                price = ladder[index]
                prices.append(price)
                rationale.append(
                    f"TP{index + 1} {price:.6g}: R-multiple fallback "
                    f"({abs(price - entry) / risk:.2f}R)"
                )

        # Enforce monotonic ordering away from entry.
        prices = _enforce_monotonic(prices, entry, direction, risk)
        return TargetPlan(prices=(prices[0], prices[1], prices[2]), rationale=rationale)

    def _weighted_rr(
        self, entry: float, stop: float, targets: Sequence[float], side: Side
    ) -> float:
        """Expected R of the whole scaled exit plan, not just one target."""

        risk = abs(entry - stop)
        if risk <= 0:
            return 0.0
        settings = self.settings
        weights = (
            settings.tp1_close_pct,
            settings.tp2_close_pct,
            max(0.0, 1.0 - settings.tp1_close_pct - settings.tp2_close_pct),
        )
        total = 0.0
        for weight, target in zip(weights, targets):
            total += weight * (abs(target - entry) / risk)
        return total

    # -- ml ---------------------------------------------------------------

    def _predict(
        self, analysis: SymbolAnalysis, proposal: TradeProposal
    ) -> tuple[float, float, float]:
        """``(P_LONG, P_SHORT, P_NO_TRADE)`` from the calibrated model.

        With no model trained yet the predictor returns an explicitly
        uninformative distribution rather than a guess.
        """

        if self.predictor is None or not self.settings.ml_enabled:
            return 0.0, 0.0, 1.0
        try:
            return self.predictor.predict(proposal.features)
        except Exception as exc:  # noqa: BLE001 - never let ML break trading
            log.warning("ML prediction failed for %s: %s", analysis.symbol, exc)
            return 0.0, 0.0, 1.0

    def _predict_meta(self, proposal: TradeProposal, side: Side) -> float | None:
        """P(this side reaches its target first), or ``None`` for no opinion.

        ``None`` and a low probability mean different things -- one is silence,
        the other is a warning -- so they must not be collapsed.
        """

        if self.predictor is None or not self.settings.ml_enabled:
            return None
        predict_meta = getattr(self.predictor, "predict_meta", None)
        if not callable(predict_meta):
            return None
        try:
            return predict_meta(proposal.features, side.sign)
        except Exception as exc:  # noqa: BLE001 - never let ML break trading
            log.warning("meta prediction failed for %s: %s", proposal.symbol, exc)
            return None

    # -- leverage ---------------------------------------------------------

    def _suggest_leverage(self, analysis: SymbolAnalysis, stop_pct: float) -> float:
        """Leverage follows the stop distance, never the other way round.

        The position size is fixed by the risk budget; leverage only determines
        how much margin that position consumes.  A wide stop therefore needs
        *less* leverage, not more.
        """

        settings = self.settings
        if stop_pct <= 0:
            return settings.min_leverage
        # Keep the liquidation price at least 4x the stop distance away.
        safe = 1.0 / (stop_pct * 4.0)
        volatility_penalty = clamp(
            1.0 / max(analysis.regime.volatility_ratio, 0.5), 0.4, 1.0
        )
        leverage = safe * volatility_penalty
        return round(
            clamp(leverage, settings.min_leverage, settings.max_leverage), 2
        )

    # -- gates ------------------------------------------------------------

    def _apply_gates(
        self,
        analysis: SymbolAnalysis,
        proposal: TradeProposal,
        strategy: RegimeStrategy,
        ml_probability: float,
    ) -> None:
        """Every gate is checked; none can be bypassed."""

        settings = self.settings

        if proposal.confidence < settings.min_confidence:
            proposal.reject(
                f"confidence {proposal.confidence:.0%} is below the "
                f"{settings.min_confidence:.0%} minimum"
            )

        if not strategy.allow_entries:
            proposal.reject(
                f"{analysis.regime.regime.value} regime forbids new entries "
                f"({strategy.rationale})"
            )

        spread = analysis.candidate.spread_pct
        if spread > settings.max_spread_pct:
            proposal.reject(
                f"spread {spread:.4%} exceeds the {settings.max_spread_pct:.4%} limit"
            )

        if analysis.candidate.quote_volume < settings.min_24h_quote_volume:
            proposal.reject(
                f"24h volume ${analysis.candidate.quote_volume:,.0f} is below the "
                f"${settings.min_24h_quote_volume:,.0f} liquidity floor"
            )

        execution = analysis.primary
        if execution is not None:
            atr_pct = execution.indicators.atr_pct
            if atr_pct < settings.min_atr_pct:
                proposal.reject(
                    f"volatility {atr_pct:.3%} is too low to cover costs"
                )
            elif atr_pct > settings.max_atr_pct:
                proposal.reject(
                    f"volatility {atr_pct:.2%} is above the {settings.max_atr_pct:.2%} ceiling"
                )

        if analysis.anomaly:
            proposal.reject(f"price anomaly detected: {analysis.anomaly}")

        fundamentals = analysis.fundamentals
        if fundamentals is not None and fundamentals.danger:
            for reason in fundamentals.danger_reasons[:3]:
                proposal.reject(f"fundamental danger: {reason}")

        if analysis.order_book is not None:
            book = analysis.order_book
            if book.spread_pct > settings.max_spread_pct * 1.5:
                proposal.reject(
                    f"order book spread {book.spread_pct:.4%} is too wide to execute"
                )
            side_key = "ask" if proposal.side is Side.LONG else "bid"
            depth = book.depth_quote(side_key, pct=0.005)
            if depth > 0 and depth < settings.min_24h_quote_volume * 0.00005:
                proposal.reject(
                    f"only ${depth:,.0f} of depth within 0.5% - too thin to fill"
                )

        if settings.ml_enabled and self.predictor is not None and proposal.p_no_trade < 1.0:
            if proposal.p_no_trade >= 0.60:
                proposal.reject(
                    f"model assigns {proposal.p_no_trade:.0%} to NO_TRADE"
                )
            elif ml_probability > 0 and ml_probability < 0.40:
                proposal.reject(
                    f"model probability for {proposal.side.value if proposal.side else '?'} "
                    f"is only {ml_probability:.0%}"
                )

    # -- explanation ------------------------------------------------------

    def _build_reasons(
        self,
        analysis: SymbolAnalysis,
        side: Side,
        stop: StopPlan,
        targets: TargetPlan,
        breakdown: ScoreBreakdown,
    ) -> list[str]:
        """Human-readable WHY, stored with every trade."""

        reasons: list[str] = []
        execution = analysis.primary
        if execution is None:
            return reasons

        direction = side.sign
        structure = execution.structure
        ict = execution.ict
        rtm = execution.rtm

        event = structure.last_event
        if event is not None and event.direction == direction:
            reasons.append(f"{event.type.value} {('up' if direction > 0 else 'down')}")

        sweep = ict.last_sweep
        if sweep is not None and sweep.direction == direction:
            reasons.append(f"Liquidity Sweep ({sweep.confidence:.0%})")

        displacement = ict.last_displacement
        if displacement is not None and displacement.direction == direction:
            reasons.append(f"Displacement {displacement.body_atr:.1f} ATR")

        fresh = ict.fresh_bullish_fvgs if direction > 0 else ict.fresh_bearish_fvgs
        if fresh:
            reasons.append(f"Fresh FVG x{len(fresh)}")

        blocks = ict.active_order_blocks(direction)
        if blocks:
            reasons.append(f"{'Bullish' if direction > 0 else 'Bearish'} Order Block")

        if (direction > 0 and ict.zone == "discount") or (
            direction < 0 and ict.zone == "premium"
        ):
            reasons.append(f"Price in {ict.zone}")

        zones = rtm.fresh_demand if direction > 0 else rtm.fresh_supply
        if zones:
            reasons.append(f"Fresh RTM {zones[-1].pattern} zone")

        if rtm.engulf_direction == direction:
            reasons.append("Engulfing candle")

        if analysis.htf_bias in (Bias.BULLISH, Bias.BEARISH):
            reasons.append(f"HTF {analysis.htf_bias.value}")

        if structure.retest_direction == direction:
            reasons.append("Retest held")

        if execution.indicators.ma_stack == direction:
            reasons.append("MA40/80/160 stacked")

        if execution.slopes.direction == direction:
            reasons.append(f"Slopes aligned (quality {execution.slopes.trend_quality:.2f})")

        reasons.append(f"Regime {analysis.regime.regime.value}")
        reasons.append(f"Stop {stop.rationale}")
        return reasons


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _last_swing_beyond(
    structure: MarketStructure, side: Side, entry: float
) -> float | None:
    """The most recent swing that sits on the invalidation side of entry."""

    for swing in reversed(structure.swings):
        if side is Side.LONG and swing.is_low and swing.price < entry:
            return swing.price
        if side is Side.SHORT and swing.is_high and swing.price > entry:
            return swing.price
    return None


def _beyond(price: float, entry: float, direction: int) -> bool:
    return price > entry if direction > 0 else price < entry


def _r_multiple_targets(
    entry: float, risk: float, side: Side, settings: Settings
) -> tuple[float, float, float]:
    sign = side.sign
    return (
        entry + sign * settings.tp1_r * risk,
        entry + sign * settings.tp2_r * risk,
        entry + sign * settings.tp3_r * risk,
    )


def _enforce_monotonic(
    prices: list[float], entry: float, direction: int, risk: float
) -> list[float]:
    """Guarantee TP1 < TP2 < TP3 in the direction of travel."""

    out: list[float] = []
    for index, price in enumerate(prices):
        if index == 0:
            out.append(price)
            continue
        previous = out[-1]
        if direction > 0 and price <= previous:
            price = previous + max(0.5 * risk, abs(previous) * 1e-6)
        elif direction < 0 and price >= previous:
            price = previous - max(0.5 * risk, abs(previous) * 1e-6)
        out.append(price)
    return out


__all__ = ["SignalEngine", "StopPlan", "TargetPlan"]
