"""Paper engine, order manager, execution gating and position management."""

from __future__ import annotations

import asyncio

import pytest

from app.config import TradingMode, build_settings
from app.domain import (
    Bias,
    Candle,
    ContractSpec,
    OrderIntent,
    OrderStatus,
    OrderType,
    Regime,
    Side,
)
from app.exchange.base import ExchangeError
from app.execution.execution_engine import ExecutionEngine, TradingModeViolation
from app.execution.order_manager import OrderManager
from app.execution.order_reconciliation import Reconciler
from app.paper.paper_engine import PaperEngine
from app.portfolio.portfolio_manager import ManagedPosition, PortfolioManager
from app.position_manager.manager import (
    ActionKind,
    ManagementContext,
    PositionManager,
)
from app.position_manager.stop_manager import StopManager, StopUpdate
from app.position_manager.target_manager import TargetManager
from app.position_manager.trailing import TrailingStop
from app.risk.risk_engine import RiskDecision
from app.risk.position_sizing import PositionSize
from app.signals.trade_proposal import Decision, TradeProposal
from tests.conftest import TEST_ENV


CONTRACT = ContractSpec(
    "BTCUSDT", "BTC_USDT", "BTC", "USDT",
    contract_size=1.0, min_volume=1, volume_scale=0, max_leverage=20, price_unit=0.01,
)


@pytest.fixture
def prices() -> dict[str, float]:
    return {"BTCUSDT": 100.0}


@pytest.fixture
def paper(prices) -> PaperEngine:
    return PaperEngine(
        price_provider=lambda s: prices.get(s, 0.0),
        contracts={"BTCUSDT": CONTRACT},
        taker_fee=0.0006,
        slippage_pct=0.001,
        latency_ms=0,
    )


def position(**overrides) -> ManagedPosition:
    base = dict(
        symbol="BTCUSDT", side=Side.LONG, quantity=10.0, entry_price=100.0,
        contract_size=1.0, stop_loss=98.0, initial_stop=98.0,
        tp1=102.0, tp2=104.0, tp3=107.0, initial_quantity=10.0, risk_amount=20.0,
    )
    base.update(overrides)
    return ManagedPosition(**base)


class TestPaperEngine:
    def test_market_buy_pays_slippage(self, paper):
        order = asyncio.run(paper.place_order(
            "BTCUSDT", Side.LONG, OrderIntent.OPEN, 10, OrderType.MARKET))
        assert order.status is OrderStatus.FILLED
        assert order.average_price > 100.0          # bought above mid

    def test_market_sell_receives_less(self, paper):
        order = asyncio.run(paper.place_order(
            "BTCUSDT", Side.LONG, OrderIntent.CLOSE, 10, OrderType.MARKET))
        assert order.average_price < 100.0

    def test_short_open_sells_below_mid(self, paper):
        order = asyncio.run(paper.place_order(
            "BTCUSDT", Side.SHORT, OrderIntent.OPEN, 10, OrderType.MARKET))
        assert order.average_price < 100.0

    def test_fees_are_charged(self, paper):
        asyncio.run(paper.place_order("BTCUSDT", Side.LONG, OrderIntent.OPEN, 10))
        assert paper.stats()["fees"] > 0

    def test_below_minimum_is_rejected(self, paper):
        with pytest.raises(ExchangeError):
            asyncio.run(paper.place_order("BTCUSDT", Side.LONG, OrderIntent.OPEN, 0.4))

    def test_limit_order_rests_until_crossed(self, paper, prices):
        order = asyncio.run(paper.place_order(
            "BTCUSDT", Side.LONG, OrderIntent.OPEN, 10, OrderType.LIMIT, price=95.0))
        assert order.status is OrderStatus.NEW
        prices["BTCUSDT"] = 94.0
        refreshed = asyncio.run(paper.order_status(order.order_id))
        assert refreshed.status is OrderStatus.FILLED

    def test_cancel(self, paper):
        order = asyncio.run(paper.place_order(
            "BTCUSDT", Side.LONG, OrderIntent.OPEN, 10, OrderType.LIMIT, price=90.0))
        assert asyncio.run(paper.cancel_order(order.order_id))
        assert not asyncio.run(paper.cancel_order(order.order_id))

    def test_funding_charges_longs_when_positive(self, paper):
        import time

        now = int(time.time())
        pos = position()
        # Nothing is due before a full interval has elapsed.
        assert paper.accrue_funding([pos], {"BTCUSDT": 0.001}, now=now) == 0.0
        charged = paper.accrue_funding([pos], {"BTCUSDT": 0.001}, now=now + 9 * 3600)
        assert charged > 0 and pos.funding > 0

    def test_funding_pays_shorts_when_positive(self, paper):
        import time

        now = int(time.time())
        pos = position(side=Side.SHORT, stop_loss=102.0, initial_stop=102.0)
        paper.accrue_funding([pos], {"BTCUSDT": 0.001}, now=now)
        charged = paper.accrue_funding([pos], {"BTCUSDT": 0.001}, now=now + 9 * 3600)
        assert charged < 0 and pos.funding < 0

    def test_no_price_means_no_fill(self):
        engine = PaperEngine(price_provider=lambda s: 0.0, contracts={"BTCUSDT": CONTRACT})
        with pytest.raises(ExchangeError):
            asyncio.run(engine.place_order("BTCUSDT", Side.LONG, OrderIntent.OPEN, 10))


class TestOrderManager:
    def test_client_ids_are_unique_and_bounded(self, paper):
        manager = OrderManager(paper)
        ids = {manager.new_client_order_id("BTCUSDT", OrderIntent.OPEN) for _ in range(200)}
        assert len(ids) == 200
        assert all(len(i) <= 32 for i in ids)

    def test_places_and_fills(self, paper):
        manager = OrderManager(paper, poll_interval=0.01, fill_timeout=0.05)
        tracked = asyncio.run(manager.place(
            "BTCUSDT", Side.LONG, OrderIntent.OPEN, 10))
        tracked = asyncio.run(manager.wait_for_fill(tracked))
        assert tracked.is_filled and tracked.filled_quantity == 10

    def test_close_orders_are_reduce_only(self, paper):
        manager = OrderManager(paper)
        tracked = asyncio.run(manager.place("BTCUSDT", Side.LONG, OrderIntent.CLOSE, 5))
        assert tracked.reduce_only


class TestModeGating:
    def _engine(self, mode: str, paper: PaperEngine) -> ExecutionEngine:
        env = dict(TEST_ENV)
        env["TRADING_MODE"] = mode
        if mode == "live":
            env.update({
                "DATA_SOURCE": "mexc",
                "MEXC_ACCESS_KEY": "k" * 20, "MEXC_SECRET_KEY": "s" * 20,
                "TELEGRAM_BOT_TOKEN": "t" * 40, "TELEGRAM_CHAT_ID": "1",
                "TELEGRAM_ALLOWED_USER_IDS": "1",
                "LIVE_CONFIRM_PHRASE": "I UNDERSTAND THE RISK",
            })
        settings = build_settings(env=env)
        return ExecutionEngine(settings, paper, PortfolioManager(mode=mode))

    def test_backtest_mode_refuses_to_execute(self, paper):
        engine = self._engine("backtest", paper)
        with pytest.raises(TradingModeViolation):
            asyncio.run(engine.close_position(position(), 1.0))

    def test_live_mode_refuses_until_enabled(self, paper):
        engine = self._engine("live", paper)
        assert not engine.live_enabled
        with pytest.raises(TradingModeViolation):
            asyncio.run(engine.close_position(position(), 1.0))

    def test_paper_mode_executes(self, paper):
        engine = self._engine("paper", paper)
        engine.portfolio.balance = 1000
        engine.portfolio.equity = 1000
        engine.portfolio.add(position())
        result = asyncio.run(engine.close_position(engine.portfolio.get("BTCUSDT"), 1.0))
        assert result.ok

    def test_paper_broker_is_never_the_exchange(self, paper):
        engine = self._engine("paper", paper)
        assert isinstance(engine.broker, PaperEngine)
        assert not hasattr(engine.broker, "_secret_key")


class TestStopManager:
    def test_stops_only_ever_tighten(self):
        manager = StopManager()
        long_position = position()
        assert manager.is_improvement(long_position, 99.0)
        assert not manager.is_improvement(long_position, 97.0)

        short_position = position(side=Side.SHORT, stop_loss=102.0, initial_stop=102.0)
        assert manager.is_improvement(short_position, 101.0)
        assert not manager.is_improvement(short_position, 103.0)

    def test_widening_is_refused_even_when_asked(self):
        manager = StopManager()
        pos = position()
        assert not manager.apply(pos, StopUpdate(90.0, "wider", "structure"), 105.0)
        assert pos.stop_loss == 98.0

    def test_stop_cannot_be_placed_through_price(self):
        manager = StopManager()
        pos = position()
        assert not manager.is_valid(pos, 105.0, price=104.0)

    def test_breakeven_only_after_the_trigger(self):
        manager = StopManager(breakeven_at_r=1.0, breakeven_offset_r=0.1)
        pos = position()
        assert manager.breakeven(pos, price=101.0) is None      # 0.5R
        update = manager.breakeven(pos, price=102.5)            # 1.25R
        assert update is not None and update.new_stop > pos.entry_price

    def test_protective_stop_only_for_unprotected_positions(self):
        manager = StopManager()
        assert manager.protective(position(), 100.0, 1.0) is None
        naked = position(stop_loss=0.0, initial_stop=0.0)
        update = manager.protective(naked, 100.0, 1.0)
        assert update is not None and update.new_stop < 100.0


class TestTrailing:
    def test_does_not_activate_too_early(self):
        trailing = TrailingStop(activate_r=1.8)
        assert not trailing.should_activate(position(), price=101.0)

    def test_activates_after_the_threshold(self):
        trailing = TrailingStop(activate_r=1.8)
        assert trailing.should_activate(position(), price=104.5)

    def test_tightens_after_partials(self):
        trailing = TrailingStop(atr_multiplier=2.0, tighten_after_tp2=1.0)
        assert trailing.active_multiplier(position(tp1_done=True, tp2_done=True)) == 1.0
        assert trailing.active_multiplier(position()) == 2.0

    def test_never_trails_past_price(self):
        trailing = TrailingStop(activate_r=0.0, atr_multiplier=0.0)
        update = trailing.compute(position(), price=105.0, atr=1.0, candles=[])
        assert update is not None and update.new_stop < 105.0


class TestTargets:
    def test_ladder_fires_in_order(self):
        manager = TargetManager(0.4, 0.35)
        pos = position()
        hit = manager.check(pos, 102.5)
        assert hit and hit.level == 1
        manager.mark(pos, hit)
        assert manager.check(pos, 102.5) is None
        hit2 = manager.check(pos, 104.5)
        assert hit2 and hit2.level == 2
        manager.mark(pos, hit2)
        hit3 = manager.check(pos, 107.5)
        assert hit3 and hit3.level == 3 and hit3.full_close

    def test_uses_the_bar_high_not_just_the_close(self):
        manager = TargetManager()
        pos = position()
        assert manager.check(pos, price=101.0, high=102.5, low=100.0) is not None

    def test_short_targets_use_the_low(self):
        manager = TargetManager()
        pos = position(side=Side.SHORT, stop_loss=102.0, initial_stop=102.0,
                       tp1=98.0, tp2=96.0, tp3=93.0)
        assert manager.check(pos, price=99.0, high=99.5, low=97.5) is not None

    def test_a_runner_always_remains_after_tp1_and_tp2(self):
        manager = TargetManager(0.4, 0.35)
        pos = position()
        first = manager.check(pos, 102.5)
        manager.mark(pos, first)
        remaining = 1 - first.close_fraction
        second = manager.check(pos, 104.5)
        remaining *= 1 - second.close_fraction
        assert remaining > 0.05

    def test_stop_wins_when_a_bar_spans_both(self):
        manager = TargetManager()
        pos = position()
        bar = Candle(ts=0, open=100, high=103, low=97, close=100, volume=1)
        assert manager.stop_hit_first(pos, bar)


class TestPositionManager:
    @pytest.fixture
    def manager(self, settings) -> PositionManager:
        return PositionManager(settings)

    def test_stop_hit_closes_everything(self, manager):
        actions = manager.evaluate(position(), ManagementContext(price=97.0, atr=1.0,
                                                                 high=99.0, low=97.0))
        assert actions[0].kind is ActionKind.STOP_HIT
        assert actions[0].close_fraction == 1.0

    def test_stop_is_checked_before_targets(self, manager):
        # A bar spanning stop and target must resolve as a stop.
        actions = manager.evaluate(
            position(), ManagementContext(price=103.0, atr=1.0, high=103.0, low=97.0)
        )
        assert actions[0].kind is ActionKind.STOP_HIT

    def test_target_hit_produces_a_partial(self, manager):
        actions = manager.evaluate(
            position(), ManagementContext(price=102.5, atr=1.0, high=102.5, low=102.0)
        )
        kinds = [a.kind for a in actions]
        assert ActionKind.TARGET_HIT in kinds
        target = next(a for a in actions if a.kind is ActionKind.TARGET_HIT)
        assert 0 < target.close_fraction < 1

    def test_structure_flip_closes_a_losing_position(self, manager):
        actions = manager.evaluate(
            position(),
            ManagementContext(price=99.5, atr=1.0, high=100.0, low=99.0,
                              structure_bias=Bias.BEARISH),
        )
        assert actions[0].kind is ActionKind.CLOSE
        assert "structure" in actions[0].reason

    def test_fundamental_danger_closes(self, manager):
        actions = manager.evaluate(
            position(),
            ManagementContext(price=100.5, atr=1.0, high=101.0, low=100.0,
                              fundamental_danger=True, fundamental_reason="exchange halt"),
        )
        assert actions[0].kind is ActionKind.CLOSE
        assert "exchange halt" in actions[0].reason

    def test_emergency_closes_regardless(self, manager):
        actions = manager.evaluate(
            position(), ManagementContext(price=100.5, atr=1.0), risk_reduction=1.0
        )
        assert actions[0].kind is ActionKind.CLOSE

    def test_partial_derisk(self, manager):
        actions = manager.evaluate(
            position(), ManagementContext(price=100.5, atr=1.0), risk_reduction=0.5
        )
        assert any(a.kind is ActionKind.REDUCE_RISK for a in actions)

    def test_nothing_to_do_when_quiet(self, manager):
        actions = manager.evaluate(
            position(breakeven_done=True, stop_loss=100.2),
            ManagementContext(price=100.5, atr=0.05, high=100.6, low=100.4),
        )
        assert all(a.kind is not ActionKind.STOP_HIT for a in actions)


class TestPortfolioManager:
    def test_open_risk_falls_to_zero_at_breakeven(self):
        pos = position()
        assert pos.open_risk() == pytest.approx(20.0)
        pos.stop_loss = 100.0
        assert pos.open_risk() == 0.0

    def test_r_multiple(self):
        pos = position()
        assert pos.r_multiple(102.0) == pytest.approx(1.0)
        assert pos.r_multiple(96.0) == pytest.approx(-2.0)

    def test_short_pnl_direction(self):
        pos = position(side=Side.SHORT, stop_loss=102.0, initial_stop=102.0)
        assert pos.unrealized_pnl(95.0) > 0
        assert pos.unrealized_pnl(105.0) < 0

    def test_round_trip_through_the_database(self, repositories):
        manager = PortfolioManager(repositories=repositories, mode="paper")
        manager.add(position(meta={"confidence": 0.8}))
        reloaded = PortfolioManager(repositories=repositories, mode="paper")
        assert reloaded.load() == 1
        restored = reloaded.get("BTCUSDT")
        assert restored.entry_price == 100.0
        assert restored.meta["confidence"] == 0.8

    def test_realised_pnl_tracks_streaks(self):
        manager = PortfolioManager(mode="paper", starting_equity=1000)
        manager.apply_realized(-10)
        manager.apply_realized(-10)
        assert manager.consecutive_losses == 2
        manager.apply_realized(30)
        assert manager.consecutive_losses == 0
        assert manager.balance == pytest.approx(1010)


class TestReconciliation:
    class _Exchange:
        def __init__(self, positions=(), orders=()):
            self._positions = list(positions)
            self._orders = list(orders)

        async def positions(self):
            return self._positions

        async def open_orders(self, symbol=None):
            return self._orders

    def test_in_sync_is_healthy(self):
        portfolio = PortfolioManager(mode="paper")
        reconciler = Reconciler(self._Exchange(), portfolio)
        report = asyncio.run(reconciler.reconcile())
        assert report.healthy and not report.mismatches

    def test_missing_position_is_dropped_and_flagged(self):
        portfolio = PortfolioManager(mode="paper")
        portfolio.add(position())
        reconciler = Reconciler(self._Exchange(), portfolio)
        report = asyncio.run(reconciler.reconcile(apply=True))
        assert report.blocking
        assert "BTCUSDT" in report.dropped
        assert portfolio.get("BTCUSDT") is None

    def test_unknown_position_is_adopted_without_a_stop(self):
        from app.domain import ExchangePosition

        portfolio = PortfolioManager(mode="paper")
        exchange = self._Exchange(positions=[
            ExchangePosition(symbol="ETHUSDT", side=Side.LONG, quantity=5,
                             entry_price=2000.0, leverage=3)
        ])
        reconciler = Reconciler(exchange, portfolio)
        report = asyncio.run(reconciler.reconcile(apply=True))
        assert report.blocking
        adopted = portfolio.get("ETHUSDT")
        assert adopted is not None
        assert adopted.stop_loss == 0.0            # we cannot invent a thesis
        assert adopted.meta.get("adopted") is True

    def test_exchange_error_is_reported_not_swallowed(self):
        class Broken:
            async def positions(self):
                raise ExchangeError("network down")

            async def open_orders(self, symbol=None):
                return []

        reconciler = Reconciler(Broken(), PortfolioManager(mode="paper"))
        report = asyncio.run(reconciler.reconcile())
        assert report.error and report.blocking
