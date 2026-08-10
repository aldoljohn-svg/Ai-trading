"""Risk engine, position sizing, portfolio risk and circuit breakers.

These are the tests that stand between a bug and a blown account.
"""

from __future__ import annotations

import pytest

from app.domain import ContractSpec, Regime, Side
from app.risk.circuit_breaker import BreakerState, CircuitBreaker, TripReason
from app.risk.portfolio_risk import (
    CorrelationMatrix,
    OpenRisk,
    PortfolioRisk,
    PortfolioState,
    pearson,
)
from app.risk.position_sizing import (
    estimate_liquidation,
    max_safe_leverage,
    size_position,
    stop_is_safe_from_liquidation,
)
from app.risk.risk_engine import RiskEngine
from app.signals.trade_proposal import Decision, TradeProposal


class TestPositionSizing:
    def test_risk_matches_the_budget(self, spec):
        result = size_position(
            equity=10_000, available_margin=10_000, entry=100.0, stop=98.0,
            side=Side.LONG, spec=ContractSpec("X", "X", "X", "USDT", contract_size=1.0,
                                              min_volume=1, volume_scale=2, price_unit=0.01),
            risk_pct=0.005, leverage=3, max_leverage=3, fee_rate=0.0,
        )
        assert result.ok
        # $50 budget / $2 stop distance = 25 units.
        assert result.base_quantity == pytest.approx(25.0, rel=0.01)
        assert result.actual_risk == pytest.approx(50.0, rel=0.02)

    def test_size_is_independent_of_leverage(self):
        contract = ContractSpec("X", "X", "X", "USDT", contract_size=1.0,
                                min_volume=1, volume_scale=2, price_unit=0.01)
        low = size_position(10_000, 10_000, 100.0, 98.0, Side.LONG, contract,
                            0.005, 1, 20, fee_rate=0.0)
        high = size_position(10_000, 10_000, 100.0, 98.0, Side.LONG, contract,
                             0.005, 10, 20, fee_rate=0.0)
        # Leverage changes margin, never the quantity or the risk.
        assert low.base_quantity == pytest.approx(high.base_quantity)
        assert low.actual_risk == pytest.approx(high.actual_risk)
        assert high.margin < low.margin

    def test_wider_stop_means_smaller_size(self):
        contract = ContractSpec("X", "X", "X", "USDT", contract_size=1.0,
                                min_volume=1, volume_scale=2, price_unit=0.01)
        tight = size_position(10_000, 10_000, 100.0, 99.0, Side.LONG, contract, 0.005, 3, 3)
        wide = size_position(10_000, 10_000, 100.0, 95.0, Side.LONG, contract, 0.005, 3, 3)
        assert wide.base_quantity < tight.base_quantity
        assert wide.actual_risk == pytest.approx(tight.actual_risk, rel=0.05)

    def test_rounding_never_exceeds_the_budget(self, spec):
        for equity in (100, 517, 1842.5, 20_000):
            result = size_position(equity, equity, 118_420, 117_610, Side.LONG,
                                   spec, 0.005, 3, 3)
            if result.ok:
                assert result.actual_risk <= result.risk_amount * 1.05

    def test_rejects_wrong_side_stops(self, spec):
        assert not size_position(1000, 1000, 100, 105, Side.LONG, spec, 0.005, 3, 3).ok
        assert not size_position(1000, 1000, 100, 95, Side.SHORT, spec, 0.005, 3, 3).ok

    def test_rejects_zero_stop_distance(self, spec):
        assert not size_position(1000, 1000, 100, 100, Side.LONG, spec, 0.005, 3, 3).ok

    def test_rejects_zero_equity(self, spec):
        assert not size_position(0, 0, 100, 98, Side.LONG, spec, 0.005, 3, 3).ok

    def test_refuses_when_minimum_contract_is_too_large(self):
        chunky = ContractSpec("X", "X", "X", "USDT", contract_size=1.0,
                              min_volume=1, volume_scale=0, price_unit=0.01)
        result = size_position(100, 100, 100_000, 99_000, Side.LONG, chunky, 0.005, 3, 3)
        assert not result.ok
        assert "minimum" in result.reasons[0] or "budget" in result.reasons[0]

    def test_scales_down_to_available_margin(self):
        contract = ContractSpec("X", "X", "X", "USDT", contract_size=1.0,
                                min_volume=1, volume_scale=2, price_unit=0.01)
        result = size_position(10_000, 50, 100.0, 99.0, Side.LONG, contract,
                               0.005, 1, 3, fee_rate=0.0)
        if result.ok:
            assert result.margin <= 50
            assert any("margin" in r for r in result.reasons)

    def test_fees_are_charged_against_the_budget(self):
        contract = ContractSpec("X", "X", "X", "USDT", contract_size=1.0,
                                min_volume=1, volume_scale=4, price_unit=0.01)
        free = size_position(10_000, 10_000, 100, 98, Side.LONG, contract,
                             0.005, 3, 3, fee_rate=0.0)
        costly = size_position(10_000, 10_000, 100, 98, Side.LONG, contract,
                               0.005, 3, 3, fee_rate=0.001)
        assert costly.base_quantity < free.base_quantity


class TestLeverageSafety:
    def test_liquidation_moves_closer_as_leverage_rises(self):
        low = estimate_liquidation(100, 2, Side.LONG)
        high = estimate_liquidation(100, 20, Side.LONG)
        assert low < high < 100

    def test_short_liquidation_is_above_entry(self):
        assert estimate_liquidation(100, 5, Side.SHORT) > 100

    def test_unsafe_leverage_is_rejected(self):
        ok, message = stop_is_safe_from_liquidation(100, 99, leverage=100, side=Side.LONG)
        assert not ok and "liquidation" in message

    def test_conservative_leverage_is_accepted(self):
        assert stop_is_safe_from_liquidation(100, 95, leverage=3, side=Side.LONG)[0]

    def test_max_safe_leverage_respects_the_cap(self):
        assert max_safe_leverage(100, 99, hard_cap=3.0) <= 3.0
        assert max_safe_leverage(100, 50) >= 1.0


class TestCorrelation:
    def test_pearson_extremes(self):
        assert pearson([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
        assert pearson([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_unknown_pairs_assume_correlation(self):
        matrix = CorrelationMatrix(default=0.6)
        assert matrix.correlation("AAAUSDT", "BBBUSDT") == 0.6
        assert matrix.correlation("AAAUSDT", "AAAUSDT") == 1.0

    def test_correlated_longs_are_not_independent(self):
        risk = PortfolioRisk(0.02, 0.012, 0.7, 3)
        risks = [
            OpenRisk("BTCUSDT", Side.LONG, 5.0, 300),
            OpenRisk("ETHUSDT", Side.LONG, 5.0, 300),
            OpenRisk("SOLUSDT", Side.LONG, 5.0, 300),
        ]
        independent = (3 * 25) ** 0.5          # sqrt of sum of squares
        assert risk.effective_risk(risks) > independent

    def test_opposite_directions_offset(self):
        risk = PortfolioRisk(0.02, 0.012, 0.7, 3)
        same = [OpenRisk("BTCUSDT", Side.LONG, 5.0, 300), OpenRisk("ETHUSDT", Side.LONG, 5.0, 300)]
        hedged = [OpenRisk("BTCUSDT", Side.LONG, 5.0, 300), OpenRisk("ETHUSDT", Side.SHORT, 5.0, 300)]
        assert risk.effective_risk(hedged) < risk.effective_risk(same)

    def test_same_direction_forms_one_cluster(self):
        risk = PortfolioRisk(0.02, 0.012, 0.5, 3)
        clusters = risk.clusters(
            [OpenRisk("BTCUSDT", Side.LONG, 5, 300), OpenRisk("ETHUSDT", Side.LONG, 5, 300)]
        )
        assert len(clusters) == 1

    def test_cluster_cap_blocks_concentration(self):
        risk = PortfolioRisk(max_portfolio_risk=0.10, max_correlated_exposure=0.012,
                             correlation_threshold=0.5, max_open_positions=5)
        state = PortfolioState(equity=1000, available=900, open_risks=[
            OpenRisk("BTCUSDT", Side.LONG, 5.0, 300),
            OpenRisk("ETHUSDT", Side.LONG, 5.0, 300),
        ])
        ok, problems, _ = risk.can_add(state, OpenRisk("SOLUSDT", Side.LONG, 5.0, 300))
        assert not ok
        assert any("cluster" in p for p in problems)

    def test_position_count_cap(self):
        risk = PortfolioRisk(0.10, 0.10, 0.7, max_open_positions=2)
        state = PortfolioState(equity=1000, available=900, open_risks=[
            OpenRisk("BTCUSDT", Side.LONG, 1.0, 100),
            OpenRisk("ETHUSDT", Side.LONG, 1.0, 100),
        ])
        ok, problems, _ = risk.can_add(state, OpenRisk("SOLUSDT", Side.LONG, 1.0, 100))
        assert not ok and any("maximum" in p for p in problems)

    def test_duplicate_symbol_is_refused(self):
        risk = PortfolioRisk(0.10, 0.10, 0.7, 5)
        state = PortfolioState(equity=1000, available=900,
                               open_risks=[OpenRisk("BTCUSDT", Side.LONG, 1.0, 100)])
        ok, problems, _ = risk.can_add(state, OpenRisk("BTCUSDT", Side.SHORT, 1.0, 100))
        assert not ok and any("already holding" in p for p in problems)

    def test_remaining_budget_shrinks(self):
        risk = PortfolioRisk(0.02, 0.02, 0.7, 5)
        empty = PortfolioState(equity=1000, available=1000)
        used = PortfolioState(equity=1000, available=900,
                              open_risks=[OpenRisk("BTCUSDT", Side.LONG, 10.0, 300)])
        assert risk.remaining_risk_budget(empty) == pytest.approx(20.0)
        assert risk.remaining_risk_budget(used) < 20.0


class TestCircuitBreakers:
    def test_clean_state(self):
        breaker = CircuitBreaker(0.02, 0.10, 4)
        assert breaker.evaluate(1000, 0, 1000, 0).state is BreakerState.OK

    def test_daily_loss_trips_and_cannot_be_reset_manually(self):
        breaker = CircuitBreaker(0.02, 0.10, 4)
        report = breaker.evaluate(980, -20, 1000, 0)
        assert report.blocked
        assert any(t.reason is TripReason.DAILY_LOSS for t in report.trips)
        breaker.reset_manual()
        assert breaker.evaluate(980, -20, 1000, 0).blocked

    def test_warning_before_the_limit(self):
        report = CircuitBreaker(0.02, 0.10, 4).evaluate(986, -14, 1000, 0)
        assert report.state is BreakerState.WARNING
        assert not report.blocked

    def test_drawdown_and_streak_trip(self):
        breaker = CircuitBreaker(0.02, 0.10, 4)
        assert any(
            t.reason is TripReason.MAX_DRAWDOWN
            for t in breaker.evaluate(890, 0, 1000, 0).trips
        )
        assert any(
            t.reason is TripReason.CONSECUTIVE_LOSSES
            for t in breaker.evaluate(1000, 0, 1000, 5).trips
        )

    def test_api_error_rate_trips(self):
        breaker = CircuitBreaker(0.02, 0.10, 4, max_api_error_rate=0.25)
        for _ in range(20):
            breaker.record_api_result(False)
        assert any(
            t.reason is TripReason.API_ERRORS
            for t in breaker.evaluate(1000, 0, 1000, 0).trips
        )

    def test_data_and_reconciliation_problems_trip(self):
        breaker = CircuitBreaker(0.02, 0.10, 4)
        breaker.set_data_problem("stale candles")
        assert breaker.evaluate(1000, 0, 1000, 0).blocked
        breaker.clear_data_problem()
        breaker.set_reconciliation_problem("mismatch")
        assert breaker.evaluate(1000, 0, 1000, 0).blocked

    def test_kill_switch_file(self, tmp_path):
        path = tmp_path / "KILL"
        breaker = CircuitBreaker(0.02, 0.10, 4, kill_switch_file=path)
        assert not breaker.evaluate(1000, 0, 1000, 0).blocked
        path.write_text("stop")
        report = breaker.evaluate(1000, 0, 1000, 0)
        assert report.blocked
        assert any(t.reason is TripReason.KILL_SWITCH for t in report.trips)

    def test_emergency_requires_flatten(self):
        breaker = CircuitBreaker(0.02, 0.10, 4)
        breaker.trigger_emergency("test")
        assert breaker.evaluate(1000, 0, 1000, 0).requires_flatten


def _proposal(**overrides) -> TradeProposal:
    proposal = TradeProposal(
        symbol="BTCUSDT", side=Side.LONG, decision=Decision.ENTER,
        entry=100.0, stop_loss=98.0, tp1=102.0, tp2=104.0, tp3=107.0,
        rr=2.0, confidence=0.8, atr=1.0, suggested_leverage=3.0,
        regime=Regime.TREND_UP,
    )
    for key, value in overrides.items():
        setattr(proposal, key, value)
    return proposal


class TestRiskEngine:
    @pytest.fixture
    def contract(self) -> ContractSpec:
        return ContractSpec("BTCUSDT", "BTC_USDT", "BTC", "USDT", contract_size=1.0,
                            min_volume=1, volume_scale=2, max_leverage=20, price_unit=0.01)

    def test_approves_a_clean_trade(self, settings, contract):
        engine = RiskEngine(settings)
        state = PortfolioState(equity=10_000, available=10_000, peak_equity=10_000)
        decision = engine.evaluate(_proposal(), state, contract)
        assert decision.approved
        assert decision.risk_pct <= settings.default_risk_per_trade * 1.05
        assert decision.leverage <= settings.max_leverage

    def test_rejects_low_confidence(self, settings, contract):
        engine = RiskEngine(settings)
        state = PortfolioState(equity=10_000, available=10_000, peak_equity=10_000)
        decision = engine.evaluate(_proposal(confidence=0.5), state, contract)
        assert not decision.approved
        assert any("confidence" in r for r in decision.rejections)

    def test_rejects_poor_reward_risk(self, settings, contract):
        engine = RiskEngine(settings)
        state = PortfolioState(equity=10_000, available=10_000, peak_equity=10_000)
        decision = engine.evaluate(_proposal(rr=1.0), state, contract)
        assert not decision.approved
        assert any("reward:risk" in r for r in decision.rejections)

    def test_rejects_a_non_entry_proposal(self, settings, contract):
        engine = RiskEngine(settings)
        state = PortfolioState(equity=10_000, available=10_000, peak_equity=10_000)
        decision = engine.evaluate(
            _proposal(decision=Decision.NO_TRADE), state, contract
        )
        assert not decision.approved

    def test_tripped_breaker_blocks_everything(self, settings, contract):
        engine = RiskEngine(settings)
        engine.breaker.trip_manual("testing")
        state = PortfolioState(equity=10_000, available=10_000, peak_equity=10_000)
        decision = engine.evaluate(_proposal(), state, contract)
        assert not decision.approved
        assert any("circuit breaker" in r for r in decision.rejections)

    def test_risk_is_cut_after_losses_never_raised(self, settings, contract):
        engine = RiskEngine(settings)
        calm = PortfolioState(equity=10_000, available=10_000, peak_equity=10_000)
        bruised = PortfolioState(equity=10_000, available=10_000, peak_equity=10_000,
                                 consecutive_losses=3)
        base = engine.evaluate(_proposal(), calm, contract)
        reduced = engine.evaluate(_proposal(), bruised, contract)
        assert reduced.risk_amount < base.risk_amount

    def test_drawdown_reduces_risk(self, settings, contract):
        engine = RiskEngine(settings)
        calm = PortfolioState(equity=10_000, available=10_000, peak_equity=10_000)
        drawn = PortfolioState(equity=9_400, available=9_400, peak_equity=10_000)
        assert (
            engine.evaluate(_proposal(), drawn, contract).risk_amount
            < engine.evaluate(_proposal(), calm, contract).risk_amount
        )

    def test_risk_reduction_factor_escalates(self, settings):
        engine = RiskEngine(settings)
        engine.breaker.trigger_emergency("x")
        state = PortfolioState(equity=1000, available=1000, peak_equity=1000)
        engine.check_breakers(state)
        assert engine.risk_reduction_factor(state) == 1.0

    def test_describe_exposes_limits(self, settings):
        engine = RiskEngine(settings)
        data = engine.describe(PortfolioState(equity=1000, available=1000, peak_equity=1000))
        assert data["max_portfolio_risk"] == settings.max_portfolio_risk
        assert "breaker_state" in data
