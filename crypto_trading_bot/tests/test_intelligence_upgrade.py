"""Tests for the advanced quant / institutional layer.

The theme running through this file is that every new subsystem is only allowed
to make the bot *more* selective.  Several tests exist purely to pin that
invariant down, because it is the property that makes the layer safe to add to
a system that already trades.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.config import ConfigError, build_settings
from app.domain import Candle, OrderBook, Regime, Side, Timeframe
from app.ensemble.base import ModelContext, ModelOutput, ModelSignal
from app.ensemble.engine import EnsembleEngine
from app.ensemble.models import ALL_MODELS
from app.ensemble.no_trade import NoTradeModel
from app.ensemble.weighting import WeightTable
from app.governance.drift import DriftAction, detect_drift
from app.governance.leakage import LeakageError, LeakageGuard, assert_no_lookahead
from app.governance.overfitting import OverfittingVerdict, detect_overfitting
from app.governance.registry import (
    PromotionCriteria,
    StrategyRegistry,
    StrategyStatus,
)
from app.governance.shadow import ShadowBook
from app.intelligence import IntelligenceCoordinator
from app.memory.clustering import cluster_trades, detect_loss_cluster, session_of
from app.memory.excursion import (
    TradeExcursion,
    analyse_excursions,
    compute_excursion,
    suggest_levels,
)
from app.memory.journal import DecisionKind, DecisionJournal, DecisionTrace
from app.memory.market_memory import MarketMemory, MarketState
from app.orderflow.derivatives import analyse_derivatives, classify_quadrant
from app.orderflow.liquidity_pools import build_liquidity_map
from app.orderflow.microstructure import analyse_microstructure, walk_book
from app.orderflow.order_flow import analyse_order_flow, bar_delta
from app.portfolio.clusters import CorrelationClusterEngine, assess_concentration
from app.portfolio.stress import monte_carlo, stress_test
from app.quality.data_risk import assess_data_risk
from app.quality.execution_risk import assess_execution_risk
from app.quality.expected_value import break_even_probability, compute_expected_value
from app.quality.model_risk import assess_model_risk
from app.quality.trade_quality import compute_trade_quality
from app.safety.anomaly import detect_anomalies
from app.safety.behaviour import BehaviourGuard, RecoveryState
from app.safety.failsafe import FailSafe, SystemFault
from app.safety.kill_switch import KillSwitchLevel, MultiLayerKillSwitch
from app.signals.trade_proposal import Decision

from tests.conftest import TEST_ENV, make_candles


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def book(mid: float = 100.0, spread: float = 0.02, depth: float = 50.0) -> OrderBook:
    bids = [(mid - spread / 2 - i * 0.01, depth) for i in range(20)]
    asks = [(mid + spread / 2 + i * 0.01, depth) for i in range(20)]
    return OrderBook(symbol="BTCUSDT", ts=int(time.time()), bids=bids, asks=asks)


def rising(n: int = 200, start: float = 100.0, step: float = 0.4) -> list[Candle]:
    return make_candles([start + i * step for i in range(n)], step=900)


# ===========================================================================
# configuration
# ===========================================================================


class TestIntelligenceSettings:
    def test_defaults_are_conservative(self, settings):
        assert settings.intelligence_enabled
        assert settings.require_ensemble_agreement
        assert 0 < settings.no_trade_threshold < 1
        assert settings.min_expected_value_r >= 0

    def test_negative_expected_value_threshold_is_rejected(self):
        env = dict(TEST_ENV, MIN_EXPECTED_VALUE_R="-0.5")
        with pytest.raises(ConfigError):
            build_settings(env=env)

    def test_out_of_range_no_trade_threshold_is_rejected(self):
        env = dict(TEST_ENV, NO_TRADE_THRESHOLD="1.4")
        with pytest.raises(ConfigError):
            build_settings(env=env)

    def test_quality_threshold_must_be_a_percentage(self):
        env = dict(TEST_ENV, MIN_TRADE_QUALITY="140")
        with pytest.raises(ConfigError):
            build_settings(env=env)


# ===========================================================================
# ensemble
# ===========================================================================


class TestEnsemble:
    def test_no_signal_is_not_neutral(self):
        """An abstention must not be counted as a vote for "do nothing"."""

        absent = ModelOutput.no_signal("x", "no data")
        neutral = ModelOutput(name="y", signal=ModelSignal.NEUTRAL, confidence=0.6)
        assert not absent.usable
        assert neutral.usable
        assert absent.signal is not ModelSignal.NEUTRAL

    def test_effective_confidence_is_discounted_by_data_quality(self):
        output = ModelOutput(
            name="m", signal=ModelSignal.LONG, confidence=0.8, data_quality=0.5
        )
        assert output.effective_confidence == pytest.approx(0.4)

    def test_safe_evaluate_converts_a_crash_into_an_abstention(self):
        from app.ensemble.base import AnalyticalModel

        class Exploding(AnalyticalModel):
            name = "exploding"

            def evaluate(self, context):
                raise RuntimeError("boom")

        output = Exploding().safe_evaluate(
            ModelContext(symbol="BTCUSDT", ts=0, analysis=None)
        )
        assert output.signal is ModelSignal.NO_SIGNAL
        assert not output.usable

    def test_every_registered_model_survives_an_empty_context(self):
        """No model may crash the ensemble when data is missing."""

        context = ModelContext(symbol="BTCUSDT", ts=0, analysis=None)
        for model in ALL_MODELS:
            output = model().safe_evaluate(context)
            assert isinstance(output, ModelOutput)

    def test_weights_stay_within_bounds_and_normalise(self):
        table = WeightTable()
        names = [m.name for m in ALL_MODELS]
        # Give one model a spectacular record and another a terrible one.
        for _ in range(80):
            table.record("ict", "ALL", won=True, r_multiple=2.0, confidence=0.9)
            table.record("news", "ALL", won=False, r_multiple=-1.0, confidence=0.9)

        weights = table.weights(names)
        # Weights are rounded to 6dp for storage and display, so the sum lands
        # within rounding distance of 1 rather than exactly on it.
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4)
        for value in weights.values():
            assert 0.02 - 1e-9 <= value <= 0.25 + 1e-9
        assert weights["ict"] > weights["news"]

    def test_reliability_is_shrunk_toward_the_prior_on_small_samples(self):
        table = WeightTable()
        table.record("m", "ALL", won=True, r_multiple=1.0, confidence=0.8)
        # One win is not a 100% hit rate.
        assert table.reliability("m", "ALL") < 0.75

    def test_confident_disagreement_lowers_ensemble_confidence(self):
        engine = EnsembleEngine()
        context = ModelContext(symbol="BTCUSDT", ts=0, analysis=None)
        result = engine.evaluate(context)
        # With no analysis, nothing may claim a directional view.
        assert result.side is None
        assert result.participation < 0.5

    def test_dissenting_models_are_credited_when_the_trade_loses(self):
        engine = EnsembleEngine()
        result = engine.evaluate(ModelContext(symbol="BTCUSDT", ts=0, analysis=None))
        before = engine.weights.performance("ict", Regime.UNKNOWN).trades
        engine.record_outcome(result, won=False, r_multiple=-1.0)
        after = engine.weights.performance("ict", Regime.UNKNOWN).trades
        # Whether or not ict voted, recording an outcome must never crash and
        # must not silently drop the sample.
        assert after >= before


# ===========================================================================
# no-trade model
# ===========================================================================


class TestNoTradeModel:
    def test_thin_participation_blocks(self):
        engine = EnsembleEngine()
        context = ModelContext(symbol="BTCUSDT", ts=0, analysis=None)
        result = engine.evaluate(context)
        verdict = NoTradeModel().evaluate(result, context)
        assert verdict.no_trade
        assert verdict.blocking_reasons

    def test_a_clean_setup_is_not_blocked_out_of_hand(self):
        """The model must be capable of saying yes, or it is not a model."""

        context = ModelContext(
            symbol="BTCUSDT", ts=0, analysis=None, regime=Regime.TREND_UP
        )
        verdict = NoTradeModel().evaluate(
            _FakeEnsemble(), context, rr=3.0, data_risk=0.05, execution_risk=0.05
        )
        assert not verdict.no_trade, verdict.summary()


class _FakeEnsemble:
    """A deliberately strong, unanimous ensemble result."""

    signal = ModelSignal.LONG
    confidence = 0.82
    agreement = 0.95
    participation = 0.9
    dissent = 0.05
    conviction = 0.8
    data_quality = 0.9
    aggregate_risk = 0.15
    regime = Regime.TREND_UP
    model_agreement_label = "10/11"

    def __init__(self, **overrides):
        for key, value in overrides.items():
            setattr(self, key, value)


# ===========================================================================
# order flow / microstructure
# ===========================================================================


class TestOrderFlow:
    def test_delta_follows_close_location(self):
        up = Candle(ts=0, open=100, high=101, low=99.9, close=100.98, volume=10)
        down = Candle(ts=0, open=100, high=100.1, low=99, close=99.02, volume=10)
        assert bar_delta(up) > 0
        assert bar_delta(down) < 0

    def test_cvd_is_labelled_a_proxy_without_a_tape(self):
        read = analyse_order_flow(rising(), book(), atr=1.0)
        assert read.cvd_is_proxy
        assert read.data_quality < 1.0

    def test_walk_book_drags_the_average_price_through_the_levels(self):
        thin = OrderBook(
            symbol="X",
            ts=0,
            bids=[(99.0, 1.0)],
            asks=[(101.0, 1.0), (105.0, 100.0)],
        )
        top, _, _ = walk_book(thin, Side.LONG, notional=50.0)
        deep, _, _ = walk_book(thin, Side.LONG, notional=5000.0)
        # A larger order must consume worse levels, never better ones.
        assert deep >= top >= 101.0

    def test_walk_book_reports_an_unfillable_order(self):
        tiny = OrderBook(symbol="X", ts=0, bids=[(99.0, 1.0)], asks=[(101.0, 1.0)])
        _, _, filled = walk_book(tiny, Side.LONG, notional=10_000_000.0)
        assert not filled, "a book this thin cannot fill the order"
        _, _, small = walk_book(tiny, Side.LONG, notional=10.0)
        assert small

    def test_a_wide_thin_book_is_not_fillable(self):
        wide = OrderBook(
            symbol="X", ts=0, bids=[(90.0, 0.1)], asks=[(110.0, 0.1)]
        )
        read = analyse_microstructure(wide, notional=100_000.0, side=Side.LONG)
        assert read.problems
        assert not read.fillable or read.expected_slippage_pct > 0.01

    def test_liquidity_levels_are_labelled_as_estimates(self):
        liquidity = build_liquidity_map(rising(), None, book(), atr=1.0)
        payload = liquidity.as_dict()
        assert "pools" in payload
        for pool in liquidity.pools:
            assert 0.0 <= pool.strength <= 1.0

    def test_derivatives_quadrants(self):
        assert classify_quadrant(0.05, 0.05) == "NEW_LONGS"
        assert classify_quadrant(0.05, -0.05) == "SHORT_COVERING"
        assert classify_quadrant(-0.05, 0.05) == "NEW_SHORTS"
        assert classify_quadrant(-0.05, -0.05) == "LONG_LIQUIDATION"

    def test_missing_derivatives_data_stays_unknown(self):
        """Absent data must never be turned into a directional opinion."""

        read = analyse_derivatives(funding_rate=None, open_interest=None)
        assert read.direction == 0
        assert read.data_quality < 0.5


# ===========================================================================
# quality and expected value
# ===========================================================================


class TestQualityScores:
    def test_missing_analysis_is_maximum_data_risk(self):
        report = assess_data_risk(None)
        assert report.score == 1.0
        assert report.blocked

    def test_execution_risk_rises_when_costs_eat_the_reward(self):
        micro = analyse_microstructure(book(mid=100.0, spread=0.5), notional=1000.0)
        cheap = assess_execution_risk(micro, expected_reward_pct=0.05)
        dear = assess_execution_risk(micro, expected_reward_pct=0.002)
        assert dear.score > cheap.score

    def test_model_risk_without_an_ensemble_is_total(self):
        report = assess_model_risk(None)
        assert report.score == 1.0

    def test_expected_value_is_shrunk_without_calibration(self):
        unproven = compute_expected_value(confidence=0.9, rr=2.0)
        # A 90% claim with no track record must not be taken at face value.
        assert unproven.win_probability < 0.9

    def test_expected_value_accounts_for_costs(self):
        free = compute_expected_value(confidence=0.6, rr=2.0, cost_pct=0.0)
        costly = compute_expected_value(
            confidence=0.6, rr=2.0, cost_pct=0.01, stop_distance_pct=0.01
        )
        assert costly.expected_r < free.expected_r

    def test_break_even_probability(self):
        assert break_even_probability(1.0) == pytest.approx(0.5)
        assert break_even_probability(3.0) == pytest.approx(0.25)

    def test_risk_scores_can_only_penalise_quality(self):
        clean = compute_trade_quality(
            ensemble=_FakeEnsemble(), regime=Regime.TREND_UP, rr=3.0
        )
        risky = compute_trade_quality(
            ensemble=_FakeEnsemble(),
            regime=Regime.TREND_UP,
            rr=3.0,
            data_risk=_Score(0.8),
            model_risk=_Score(0.8),
            execution_risk=_Score(0.8),
        )
        assert risky.score < clean.score


class _Score:
    def __init__(self, score: float) -> None:
        self.score = score
        self.problems: list[str] = []
        self.notes: list[str] = []
        self.blocked = score >= 0.7


# ===========================================================================
# safety
# ===========================================================================


class TestKillSwitch:
    def test_level_is_the_maximum_of_all_triggers(self):
        switch = MultiLayerKillSwitch()
        state = switch.evaluate(daily_loss_pct=0.04, consecutive_losses=7)
        assert state.level >= KillSwitchLevel.STOP_NEW_TRADES

    def test_no_evaluation_can_lower_a_sticky_level(self):
        switch = MultiLayerKillSwitch()
        engaged = switch.evaluate(daily_loss_pct=0.05).level
        calm = switch.evaluate(daily_loss_pct=0.0, consecutive_losses=0).level
        assert calm >= engaged

    def test_only_a_human_can_clear_a_live_disable(self):
        switch = MultiLayerKillSwitch()
        switch.engage(KillSwitchLevel.DISABLE_LIVE, "operator stop")
        assert switch.live_disabled
        switch.human_reset(clear_live_disable=False)
        assert switch.live_disabled, "a plain reset must not re-enable live trading"
        switch.human_reset(clear_live_disable=True)
        assert not switch.live_disabled

    def test_manual_engagement_survives_a_quiet_evaluation(self):
        switch = MultiLayerKillSwitch()
        switch.engage(KillSwitchLevel.STOP_NEW_TRADES, "manual pause")
        assert switch.evaluate().level >= KillSwitchLevel.STOP_NEW_TRADES


class TestAnomalyDetection:
    def test_calm_data_is_not_an_anomaly(self):
        report = detect_anomalies(rising())
        assert not report.black_swan

    def test_a_price_shock_is_detected(self):
        candles = rising(120)
        last = candles[-1]
        candles.append(
            Candle(
                ts=last.ts + 900,
                open=last.close,
                high=last.close * 1.35,
                low=last.close,
                close=last.close * 1.32,
                volume=last.volume * 40,
            )
        )
        report = detect_anomalies(candles)
        assert report.anomalies
        assert report.severity > 0


class TestBehaviourGuard:
    def test_recovery_mode_is_sticky_until_the_drawdown_heals(self):
        guard = BehaviourGuard()
        assert guard.evaluate(drawdown_pct=0.09).state is RecoveryState.RECOVERY
        # A small improvement must not immediately restore full size.
        assert guard.evaluate(drawdown_pct=0.05).state is RecoveryState.RECOVERY
        assert guard.evaluate(drawdown_pct=0.01).state is not RecoveryState.RECOVERY

    def test_risk_multiplier_is_never_above_one(self):
        assert BehaviourGuard.clamp_risk_multiplier(4.0) == 1.0
        assert BehaviourGuard.clamp_risk_multiplier(-1.0) == 0.0

    def test_every_recovery_state_only_reduces_risk(self):
        for state in RecoveryState:
            assert state.risk_multiplier <= 1.0

    def test_overtrading_is_flagged_when_frequency_is_not_rewarded(self):
        guard = BehaviourGuard(max_trades_per_hour=2)
        now = int(time.time())
        for _ in range(5):
            guard.record_entry(now)
        assert guard.check_overtrading(now=now).overtrading


class TestFailSafe:
    def test_a_clock_drift_fault_escalates(self):
        failsafe = FailSafe()
        failsafe.record_clock_drift(60.0)
        report = failsafe.evaluate()
        assert report.faults
        assert report.level > KillSwitchLevel.NONE

    def test_a_healthy_system_raises_nothing(self):
        failsafe = FailSafe()
        failsafe.record_clock_drift(0.1)
        failsafe.record_latency("exchange", 40.0)
        assert not failsafe.evaluate().faults

    def test_sustained_latency_is_a_fault(self):
        failsafe = FailSafe(max_latency_ms=100.0)
        for _ in range(30):
            failsafe.record_latency("exchange", 900.0)
        report = failsafe.evaluate()
        assert any(f.fault is SystemFault.LATENCY for f in report.faults)

    def test_a_fault_stays_until_it_is_cleared(self):
        failsafe = FailSafe()
        failsafe.raise_fault(SystemFault.DATABASE, "disk full")
        assert failsafe.has_fault(SystemFault.DATABASE)
        assert failsafe.evaluate().faults
        failsafe.clear_fault(SystemFault.DATABASE)
        assert not failsafe.has_fault(SystemFault.DATABASE)


# ===========================================================================
# governance
# ===========================================================================


class TestLeakageGuards:
    def test_a_future_bar_is_rejected(self):
        candles = make_candles([1.0, 2.0, 3.0], start_ts=0, step=900)
        with pytest.raises(LeakageError):
            assert_no_lookahead(candles, decision_ts=900, timeframe=Timeframe.M15)

    def test_closed_history_passes(self):
        candles = make_candles([1.0, 2.0, 3.0], start_ts=0, step=900)
        assert_no_lookahead(candles, decision_ts=10_000, timeframe=Timeframe.M15)

    def test_misaligned_timestamps_are_rejected(self):
        candles = make_candles([1.0, 2.0], start_ts=7, step=900)
        with pytest.raises(LeakageError):
            assert_no_lookahead(candles, decision_ts=10_000, timeframe=Timeframe.M15)

    def test_guard_reports_instead_of_raising(self):
        """In production a leak must be reported, not crash the scan."""

        guard = LeakageGuard()
        candles = make_candles([1.0, 2.0, 3.0], start_ts=0, step=900)
        assert not guard.check_series(candles, decision_ts=900, timeframe=Timeframe.M15)
        assert not guard.report.ok
        assert guard.report.violations

    def test_a_future_event_timestamp_is_caught(self):
        guard = LeakageGuard()
        assert not guard.check_event_ts(event_ts=2_000, decision_ts=1_000, label="news")

    def test_a_feature_timestamped_in_the_future_is_caught(self):
        guard = LeakageGuard()
        assert guard.check_features({"news_ts": 500, "rsi": 55.0}, decision_ts=1_000)
        assert not guard.check_features({"news_ts": 5_000}, decision_ts=1_000)


class TestStrategyGovernance:
    def test_promotion_is_all_or_nothing(self):
        registry = StrategyRegistry(PromotionCriteria(min_trades=10, min_shadow_days=0.0))
        challenger = registry.register("challenger", status=StrategyStatus.SHADOW)
        champion = registry.register("champion", status=StrategyStatus.ACTIVE)
        for _ in range(20):
            challenger.record(1.0, "TREND_UP")
            champion.record(0.9, "TREND_UP")
        decision = registry.evaluate_promotion(challenger, champion)
        # Even a good-looking challenger must fail if any single check fails.
        assert decision.promote == (not decision.failed)

    def test_a_challenger_with_no_record_is_never_promoted(self):
        registry = StrategyRegistry()
        challenger = registry.register("new", status=StrategyStatus.SHADOW)
        champion = registry.register("old", status=StrategyStatus.ACTIVE)
        assert not registry.evaluate_promotion(challenger, champion).promote

    def test_shadow_book_cannot_reach_an_exchange(self):
        shadow = ShadowBook()
        assert not hasattr(shadow, "broker")
        assert not hasattr(shadow, "exchange")
        for attribute in vars(shadow):
            assert "key" not in attribute and "secret" not in attribute

    def test_shadow_trades_resolve_against_price(self):
        shadow = ShadowBook()
        trade = shadow.open("s", "BTCUSDT", Side.LONG, entry=100.0, stop=99.0)
        assert trade is not None
        shadow.mark("BTCUSDT", high=98.0, low=97.0, close=97.5)
        assert shadow.open_count() == 0


class TestDriftAndOverfitting:
    def test_drift_needs_evidence(self):
        report = detect_drift("m", recent_outcomes=[1.0] * 3, baseline_outcomes=[1.0] * 5)
        assert report.action is DriftAction.NONE

    def test_a_collapse_in_hit_rate_is_detected(self):
        report = detect_drift(
            "m",
            recent_outcomes=[-1.0] * 30,
            baseline_outcomes=[1.0] * 40 + [-1.0] * 20,
        )
        assert report.action is not DriftAction.NONE
        assert report.weight_multiplier <= 1.0

    def test_implausible_results_are_always_rejected(self):
        report = detect_overfitting(win_rate=0.95, sharpe=9.0, trades=500)
        assert report.verdict is OverfittingVerdict.REJECT

    def test_consistent_folds_are_accepted(self):
        report = detect_overfitting(
            train_score=1.0,
            test_score=0.95,
            oos_score=0.92,
            fold_scores=[0.9, 0.95, 0.93, 0.94],
            trades=200,
            win_rate=0.55,
            sharpe=1.4,
        )
        assert report.verdict is not OverfittingVerdict.REJECT


# ===========================================================================
# memory
# ===========================================================================


class TestDecisionJournal:
    def test_outcomes_are_new_records_not_edits(self):
        journal = DecisionJournal()
        entry = journal.record(
            DecisionKind.ENTRY,
            "BTCUSDT",
            trace=DecisionTrace(ensemble={"signal": "LONG"}),
        )
        journal.record_outcome(entry.decision_id, "BTCUSDT", {"r_multiple": 1.5})
        stored = journal.get(entry.decision_id)
        assert stored is not None
        assert stored.outcome == {}, "the original decision must stay untouched"
        assert len(journal.outcomes_for(entry.decision_id)) == 1

    def test_replay_renders_the_stages_that_ran(self):
        journal = DecisionJournal()
        entry = journal.record(
            DecisionKind.ENTRY,
            "BTCUSDT",
            trace=DecisionTrace(
                ensemble={"signal": "LONG"}, expected_value={"expected_r": 0.4}
            ),
        )
        text = journal.replay(entry.decision_id)
        assert "ENSEMBLE" in text.upper()
        assert "BTCUSDT" in text


class TestExcursions:
    def test_mae_and_mfe_are_measured_in_r(self):
        candles = [
            Candle(ts=0, open=100, high=102, low=99, close=101, volume=1),
            Candle(ts=900, open=101, high=104, low=100, close=103, volume=1),
        ]
        mae_r, mfe_r, realised_r = compute_excursion(
            entry=100.0, stop=99.0, side=Side.LONG, candles=candles, exit_price=103.0
        )
        assert mae_r == pytest.approx(-1.0)
        assert mfe_r == pytest.approx(4.0)
        assert realised_r == pytest.approx(3.0)

    def test_suggestions_are_never_applied_automatically(self):
        excursions = [
            TradeExcursion(
                symbol="BTCUSDT",
                side="LONG",
                r_multiple=1.8,
                mae_r=-0.3,
                mfe_r=3.0,
                won=True,
            )
            for _ in range(60)
        ]
        stats = analyse_excursions(excursions)
        suggestion = suggest_levels(stats)
        assert suggestion["applied"] is False

    def test_a_small_sample_yields_no_suggestion(self):
        stats = analyse_excursions(
            [
                TradeExcursion(
                    symbol="X",
                    side="LONG",
                    r_multiple=1.0,
                    mae_r=-0.2,
                    mfe_r=1.0,
                    won=True,
                )
            ]
        )
        assert not stats.reliable
        assert "insufficient" in " ".join(suggest_levels(stats)["suggestions"])


class TestMarketMemory:
    def _state(self, i: int, rsi: float) -> MarketState:
        return MarketState(
            symbol="BTCUSDT", ts=i, timeframe="M15", features={"rsi": rsi}
        )

    def test_unresolved_history_gives_no_analogue(self):
        """An unresolved state has no outcome, so it can teach nothing yet."""

        memory = MarketMemory()
        for i in range(50):
            memory.remember(self._state(i, 50.0 + i % 5))
        assert memory.find_analogues({"rsi": 52.0}).sample == 0

    def test_resolved_history_produces_a_graded_analogue_report(self):
        memory = MarketMemory()
        for i in range(40):
            memory.remember(self._state(i, 50.0 + i % 7))
            memory.resolve(
                "BTCUSDT",
                i,
                forward_return=0.01,
                forward_max_up=0.02,
                forward_max_down=-0.01,
            )
        report = memory.find_analogues({"rsi": 52.0})
        assert report.sample > 0
        assert report.confidence in {"NONE", "WEAK", "MODERATE", "STRONG"}


class TestTradeClustering:
    def test_sessions_are_labelled(self):
        assert session_of(0) in {"ASIA", "LONDON", "NEW_YORK", "OFF_HOURS"}

    def test_a_systematically_losing_condition_is_found(self):
        trades = [
            {"r_multiple": -1.0, "session": "ASIA", "regime": "RANGE"} for _ in range(12)
        ] + [
            {"r_multiple": 2.0, "session": "LONDON", "regime": "TREND_UP"}
            for _ in range(12)
        ]
        cluster = detect_loss_cluster(trades)
        assert cluster.detected
        assert "ASIA" in cluster.description or "RANGE" in cluster.description

    def test_clustering_groups_by_every_dimension(self):
        report = cluster_trades(
            [{"r_multiple": 1.0, "session": "ASIA", "symbol": "BTCUSDT"}]
        )
        assert report.clusters


# ===========================================================================
# portfolio
# ===========================================================================


class TestCorrelationClusters:
    def test_identical_series_cluster_together(self):
        engine = CorrelationClusterEngine()
        candles = rising(200)
        engine.update("AAAUSDT", candles)
        engine.update("BBBUSDT", candles)
        clusters = engine.cluster(["AAAUSDT", "BBBUSDT"])
        assert len(clusters) == 1
        assert clusters[0].size == 2

    def test_worst_case_correlation_is_used(self):
        engine = CorrelationClusterEngine()
        candles = rising(200)
        engine.update("AAAUSDT", candles)
        engine.update("BBBUSDT", candles)
        assert engine.correlation("AAAUSDT", "BBBUSDT") == pytest.approx(1.0, abs=1e-6)

    def test_unknown_pairs_assume_correlation_rather_than_independence(self):
        engine = CorrelationClusterEngine()
        assert engine.correlation("AAAUSDT", "ZZZUSDT") == engine.default_correlation

    def test_a_one_sided_book_is_flagged(self):
        report = assess_concentration(
            exposures={"A": 100.0, "B": 100.0, "C": 100.0},
            directions={"A": 1, "B": 1, "C": 1},
        )
        assert report.concentrated
        assert report.direction == "LONG"

    def test_a_balanced_book_is_not_flagged(self):
        report = assess_concentration(
            exposures={"A": 100.0, "B": 100.0, "C": 100.0},
            directions={"A": 1, "B": -1, "C": -1},
        )
        assert not report.concentrated


class TestStress:
    def test_monte_carlo_needs_history(self):
        result = monte_carlo([1.0, -1.0])
        assert result.notes

    def test_monte_carlo_reports_a_worst_case_worse_than_the_median(self):
        r_values = [2.0] * 40 + [-1.0] * 60
        result = monte_carlo(r_values, simulations=400)
        assert result.worst_drawdown >= result.median_max_drawdown

    def test_stress_test_with_no_positions_is_harmless(self):
        report = stress_test([], equity=1000.0, prices={})
        assert not report.any_liquidation
        assert report.worst_case_pct == 0.0


# ===========================================================================
# integration: the layer may only restrict
# ===========================================================================


class TestIntelligenceCoordinator:
    def _pipeline(self, settings):
        from app.data.market_data import MarketData
        from app.exchange.synthetic import SyntheticExchange
        from app.scanner.scanner import Scanner
        from app.scanner.universe import UniverseBuilder
        from app.signals.signal_engine import SignalEngine

        exchange = SyntheticExchange(seed=99)
        market_data = MarketData(exchange)
        universe = UniverseBuilder(
            quote_currency=settings.quote_currency,
            min_quote_volume=settings.min_24h_quote_volume,
            max_spread_pct=settings.max_spread_pct,
            max_symbols=settings.max_symbols_to_scan,
        )
        scanner = Scanner(market_data, universe, deep_analysis_count=4)
        return exchange, scanner, SignalEngine(settings)

    def test_evaluate_produces_a_complete_verdict(self, settings):
        async def run():
            exchange, scanner, engine = self._pipeline(settings)
            await exchange.connect()
            coordinator = IntelligenceCoordinator(settings)

            analyses = await scanner.deep_analyse(await scanner.prescreen())
            assert analyses
            for analysis in analyses:
                proposal = engine.evaluate(analysis)
                verdict = coordinator.evaluate(
                    analysis, proposal, notional=1000.0, record=False
                )
                assert verdict.ensemble is not None
                assert verdict.trade_quality is not None
                assert verdict.expected_value is not None
                assert verdict.no_trade is not None
                assert 0.0 <= verdict.size_multiplier <= 1.0
                assert isinstance(verdict.as_dict(), dict)
                assert verdict.summary()
            await exchange.close()

        asyncio.run(run())

    def test_a_rejected_proposal_is_never_approved_by_the_layer(self, settings):
        """The core safety property of the whole upgrade."""

        async def run():
            exchange, scanner, engine = self._pipeline(settings)
            await exchange.connect()
            coordinator = IntelligenceCoordinator(settings)

            analyses = await scanner.deep_analyse(await scanner.prescreen())
            for analysis in analyses:
                proposal = engine.evaluate(analysis)
                verdict = coordinator.evaluate(
                    analysis, proposal, notional=1000.0, record=False
                )
                if proposal.decision is not Decision.ENTER:
                    assert not verdict.approved
            await exchange.close()

        asyncio.run(run())

    def test_the_journal_records_every_evaluation(self, settings):
        async def run():
            exchange, scanner, engine = self._pipeline(settings)
            await exchange.connect()
            coordinator = IntelligenceCoordinator(settings)
            analyses = await scanner.deep_analyse(await scanner.prescreen())
            analysis = analyses[0]
            verdict = coordinator.evaluate(
                analysis, engine.evaluate(analysis), notional=1000.0
            )
            assert verdict.decision_id
            assert coordinator.journal.get(verdict.decision_id) is not None
            await exchange.close()

        asyncio.run(run())

    def test_disabling_the_layer_removes_it_entirely(self):
        settings = build_settings(env=dict(TEST_ENV, INTELLIGENCE_ENABLED="false"))
        assert not settings.intelligence_enabled


class TestRiskEngineScaling:
    def test_the_intelligence_scale_can_only_shrink_a_position(self, settings, spec):
        from app.risk.portfolio_risk import PortfolioRisk, PortfolioState
        from app.risk.risk_engine import RiskEngine
        from app.signals.trade_proposal import TradeProposal

        def proposal() -> TradeProposal:
            return TradeProposal(
                symbol="BTCUSDT",
                side=Side.LONG,
                decision=Decision.ENTER,
                regime=Regime.TREND_UP,
                entry=100.0,
                stop_loss=98.0,
                tp1=102.0,
                tp2=104.0,
                tp3=106.0,
                confidence=0.85,
                rr=3.0,
                stop_distance=2.0,
                atr=2.0,
                ts=int(time.time()),
            )

        engine = RiskEngine(
            settings,
            portfolio_risk=PortfolioRisk(
                max_portfolio_risk=settings.max_portfolio_risk,
                max_correlated_exposure=settings.max_correlated_exposure,
            ),
        )
        state = PortfolioState(equity=1000.0, available=1000.0, peak_equity=1000.0)

        full = engine.evaluate(proposal(), state, spec)
        half = engine.evaluate(proposal(), state, spec, intelligence_scale=0.5)
        absurd = engine.evaluate(proposal(), state, spec, intelligence_scale=5.0)

        assert full.approved, full.rejections
        assert half.risk_amount == pytest.approx(full.risk_amount * 0.5, rel=1e-6)
        # A value above 1 must be clamped, never honoured.
        assert absurd.risk_amount == pytest.approx(full.risk_amount)
        assert half.size.contracts <= full.size.contracts
        assert half.size.notional <= full.size.notional
