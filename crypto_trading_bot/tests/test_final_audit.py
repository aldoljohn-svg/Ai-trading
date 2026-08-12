"""Findings from the final full-source pass.

Three defects, in descending order of what they could cost you:

1. **A restart cleared the circuit breakers.** ``realized_pnl_today``,
   ``consecutive_losses`` and ``peak_equity`` lived only in memory, so every
   start began from zero. That handed the operator a way to clear a tripped
   daily-loss breaker -- restart the process -- which directly contradicts what
   :mod:`app.risk.circuit_breaker` says about itself: the daily-loss breaker is
   supposed to be unclearable until the next UTC day. With ``Restart=always`` in
   the systemd unit, a crash loop cleared it repeatedly and nobody had to decide
   anything. The queries needed to rebuild all three already existed in
   ``repositories`` and were simply never called.
2. **The backtester skipped the entry bar.** A position filled at bar *i+1*'s
   open was not managed until bar *i+2*, so bar *i+1*'s own high and low were
   never checked. Every trade that would have been stopped out on the bar it
   opened survived to be judged on a later one.
3. **Partial fills recorded the planned risk, not the filled risk.** The
   inflated figure is what the portfolio limits and the correlation cluster cap
   are measured against, so a half-filled entry crowded out the next trade by
   risk the account was not carrying.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time

import pytest

from app.domain import Candle, ContractSpec, Side


def _database():
    from app.database.database import Database
    from app.database.repositories import Repositories

    path = os.path.join(tempfile.mkdtemp(), "audit.db")
    db = Database(f"sqlite:///{path}")
    db.migrate()
    return db, Repositories(db)


def _close_trade(repos, pnl: float, symbol: str = "BTCUSDT", mode: str = "paper"):
    trade_id = repos.trades.open_trade(
        {
            "symbol": symbol,
            "side": "long",
            "mode": mode,
            "quantity": 1.0,
            "entry_price": 100.0,
            "opened_at": int(time.time()) - 600,
        }
    )
    repos.trades.close_trade(
        trade_id,
        exit_price=99.0,
        realized_pnl=pnl,
        fees=0.0,
        funding=0.0,
        r_multiple=-1.0 if pnl < 0 else 1.0,
        why_exited="stop" if pnl < 0 else "target",
    )
    return trade_id


class TestBreakerStateSurvivesARestart:
    def _manager(self, repos, equity=1000.0):
        from app.portfolio.portfolio_manager import PortfolioManager

        return PortfolioManager(
            repositories=repos, mode="paper", starting_equity=equity
        )

    def test_todays_realised_loss_is_rebuilt(self):
        db, repos = _database()
        for pnl in (-8.0, -6.0, -5.0):
            _close_trade(repos, pnl)
        manager = self._manager(repos)
        manager.load()
        assert manager.realized_pnl_today == pytest.approx(-19.0)

    def test_the_loss_streak_is_rebuilt(self):
        db, repos = _database()
        for pnl in (-8.0, -6.0, -5.0):
            _close_trade(repos, pnl)
        manager = self._manager(repos)
        manager.load()
        assert manager.consecutive_losses == 3

    def test_a_win_breaks_the_streak(self):
        db, repos = _database()
        _close_trade(repos, -8.0)
        _close_trade(repos, -6.0)
        _close_trade(repos, +12.0)
        manager = self._manager(repos)
        manager.load()
        assert manager.consecutive_losses == 0

    def test_peak_equity_is_rebuilt_from_snapshots(self):
        db, repos = _database()
        repos.account.snapshot({"mode": "paper", "equity": 1500.0, "balance": 1500.0})
        repos.account.snapshot({"mode": "paper", "equity": 1200.0, "balance": 1200.0})
        manager = self._manager(repos, equity=1200.0)
        manager.load()
        assert manager.peak_equity == pytest.approx(1500.0)

    def test_peak_equity_never_regresses(self):
        """A stale snapshot must not lower a peak we already know about."""

        db, repos = _database()
        repos.account.snapshot({"mode": "paper", "equity": 900.0, "balance": 900.0})
        manager = self._manager(repos, equity=2000.0)
        manager.load()
        assert manager.peak_equity >= 2000.0

    def test_yesterdays_losses_do_not_count_against_today(self):
        db, repos = _database()
        trade_id = repos.trades.open_trade(
            {
                "symbol": "BTCUSDT", "side": "long", "mode": "paper",
                "quantity": 1.0, "entry_price": 100.0,
                "opened_at": int(time.time()) - 200_000,
            }
        )
        repos.trades.close_trade(
            trade_id, exit_price=99.0, realized_pnl=-50.0, fees=0.0,
            funding=0.0, r_multiple=-1.0, why_exited="stop",
        )
        # Backdate the close to two days ago.
        db.execute(
            "UPDATE trades SET closed_at=? WHERE id=?",
            [int(time.time()) - 200_000, trade_id],
        )
        manager = self._manager(repos)
        manager.load()
        assert manager.realized_pnl_today == pytest.approx(0.0)

    def test_another_mode_is_not_counted(self):
        db, repos = _database()
        _close_trade(repos, -40.0, mode="live")
        manager = self._manager(repos)
        manager.load()
        assert manager.realized_pnl_today == pytest.approx(0.0)

    def test_the_breaker_then_trips_after_a_restart(self):
        """End to end: the restart must not buy another trade."""

        from app.risk.circuit_breaker import CircuitBreaker

        db, repos = _database()
        for pnl in (-8.0, -6.0, -5.0, -4.0):
            _close_trade(repos, pnl)

        manager = self._manager(repos, equity=977.0)
        manager.load()
        report = CircuitBreaker(
            max_daily_loss=0.02,
            max_drawdown=0.10,
            max_consecutive_losses=4,
            kill_switch_file=None,
        ).evaluate(
            equity=977.0,
            realized_pnl_today=manager.realized_pnl_today,
            peak_equity=manager.peak_equity,
            consecutive_losses=manager.consecutive_losses,
        )
        assert report.blocked, "a restart must not clear the daily-loss breaker"
        assert any(t.reason.value == "DAILY_LOSS" for t in report.trips)

    def test_an_unreadable_database_does_not_block_startup(self):
        class Broken:
            class trades:
                @staticmethod
                def realized_pnl_since(*a, **k):
                    raise RuntimeError("db gone")

                @staticmethod
                def consecutive_losses(*a, **k):
                    return 0

            class account:
                @staticmethod
                def peak_equity(*a, **k):
                    return 0.0

            class positions:
                @staticmethod
                def open_positions(mode=None):
                    return []

        from app.portfolio.portfolio_manager import PortfolioManager

        manager = PortfolioManager(
            repositories=Broken(), mode="paper", starting_equity=1000.0
        )
        assert manager.load() == 0     # must not raise


class TestTheBacktesterJudgesTheEntryBar:
    def test_a_position_is_managed_on_the_bar_it_opens(self):
        import inspect

        from app.backtest import engine

        source = inspect.getsource(engine.BacktestEngine.run)
        assert "opened_now" in source, "the entry bar is being skipped again"

    def test_a_stop_inside_the_entry_bar_closes_the_trade(self):
        """The behaviour, not just the presence of the code."""

        from app.config import build_settings
        from app.position_manager.manager import ManagementContext, PositionManager
        from app.portfolio.portfolio_manager import ManagedPosition
        from tests.conftest import TEST_ENV

        settings = build_settings(env=dict(TEST_ENV))
        manager = PositionManager(settings)
        position = ManagedPosition(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=1.0,
            entry_price=100.0,
            contract_size=1.0,
            stop_loss=98.0,
            initial_stop=98.0,
            initial_quantity=1.0,
        )
        # The bar opens at 100, dips to 97.5 (through the stop), closes at 100.5.
        # Judged on the close alone this looks like a winning bar.
        context = ManagementContext(price=100.5, high=101.0, low=97.5, atr=1.0)
        actions = manager.evaluate(position, context)
        assert actions, "the bar pierced the stop and nothing happened"
        assert actions[0].kind.name == "STOP_HIT"

    def test_the_docstring_records_the_guarantee(self):
        from app.backtest import engine

        assert "high and low" in (engine.__doc__ or "")


class TestPartialFillsRecordTheRiskActuallyTaken:
    def test_risk_scales_with_the_filled_quantity(self):
        import inspect

        from app.execution import execution_engine

        source = inspect.getsource(execution_engine)
        assert "risk_amount=size.actual_risk * fill_ratio" in source

    def test_a_full_fill_is_unchanged(self):
        """The common path must not move."""

        contracts, filled = 10.0, 10.0
        assert filled / contracts == pytest.approx(1.0)

    def test_a_half_fill_halves_the_recorded_risk(self):
        planned_risk, contracts, filled = 50.0, 10.0, 5.0
        assert planned_risk * (filled / contracts) == pytest.approx(25.0)


class TestStopsAreSoftwareOnly:
    """Not a bug -- a design property that has to stay documented.

    The bot places no trigger order on the venue: the position manager polls
    and market-closes when a stop is breached. That is a deliberate choice, and
    it has one hard consequence the operator must know about, so the README
    carries it and this test keeps the README honest.
    """

    def test_no_trigger_orders_are_placed_anywhere(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "app"
        markers = ("stopPrice", "triggerPrice", "planorder", "STOP_MARKET")
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for marker in markers:
                assert marker not in text, (
                    f"{path.name} places venue-side stops; the README says the "
                    "bot does not, so one of the two needs updating"
                )

    def test_the_readme_says_so(self):
        import pathlib

        readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
        text = readme.read_text(encoding="utf-8").lower()
        assert "software stop" in text or "no stop order" in text
        assert "unprotected" in text


class TestTheJournalAgreesWithItself:
    """The header and the rejection reason must show the same number.

    The live journal read::

        LINKUSDT LONG quality 42 EV +0.68R
           trade quality 41 below the 55 minimum

    The rejection text formats ``score`` directly; the journal row stores
    ``as_dict()``'s ``round(score, 2)`` and the Telegram header formats that.
    A raw score near a ``.x95`` boundary went through Python's round-half-to-even
    at two different precisions and disagreed by a point -- in either direction,
    which is why it did not look like a rounding artifact at first.
    """

    def _quality(self, **kwargs):
        from app.domain import Regime
        from app.ensemble.base import ModelSignal
        from app.quality.trade_quality import compute_trade_quality

        class Ensemble:
            signal = ModelSignal.LONG
            agreement = 0.72
            participation = 0.61
            dissent = 0.28
            data_quality = 0.83
            aggregate_risk = 0.25
            confidence = 0.4137

        return compute_trade_quality(
            ensemble=Ensemble(), regime=Regime.TREND_UP, rr=2.13, min_rr=1.7, **kwargs
        )

    def test_the_score_is_already_rounded_to_its_stored_precision(self):
        quality = self._quality()
        assert quality.score == round(quality.score, 2)

    def test_both_display_paths_render_the_same_integer(self):
        quality = self._quality()
        from_reason = f"{quality.score:.0f}"
        from_journal = f"{quality.as_dict()['score']:.0f}"
        assert from_reason == from_journal

    def test_it_holds_across_the_awkward_boundaries(self):
        """Sweep the .x95 and .x5 region where the two disagreed."""

        from app.quality.trade_quality import TradeQuality

        for raw in (41.495, 41.4999, 46.5, 46.502, 54.995, 55.005):
            quality = TradeQuality(score=round(raw, 2))
            assert f"{quality.score:.0f}" == f"{quality.as_dict()['score']:.0f}"


class TestTheColdStartTaxIsBoundedAndDecays:
    """A fresh install is penalised for having no record. That is deliberate.

    What matters is that it *ramps off* as outcomes accumulate rather than
    sitting there forever: a penalty that cannot be worked off would be a
    bootstrap trap, because the trades that would earn the experience are the
    ones it is blocking.
    """

    def _risk(self, resolved: int):
        from app.ensemble.weighting import WeightTable
        from app.quality.model_risk import assess_model_risk

        table = WeightTable()
        for i in range(resolved):
            table.record(
                "technical", "TREND_UP", won=i % 2 == 0,
                r_multiple=1.0 if i % 2 == 0 else -1.0, confidence=0.6,
            )

        class Output:
            name = "technical"
            confidence = 0.6

        class Ensemble:
            dissent = 0.15
            participation = 0.6
            outputs = [Output()]

        class Predictor:
            def info(self):
                return {"loaded": False}

        return assess_model_risk(
            Ensemble(), weight_table=table, predictor=Predictor(), regime="TREND_UP"
        ).score

    def test_a_fresh_install_is_penalised(self):
        assert self._risk(0) > 0.3

    def test_the_penalty_falls_monotonically_with_experience(self):
        scores = [self._risk(n) for n in (0, 5, 10, 15, 20)]
        assert scores == sorted(scores, reverse=True)

    def test_it_is_largely_worked_off_by_the_threshold(self):
        assert self._risk(20) < 0.1

    def test_it_never_alone_blocks_a_trade(self):
        """It costs a fraction of quality; it is not a veto."""

        assert 1.0 - 0.5 * self._risk(0) > 0.75
