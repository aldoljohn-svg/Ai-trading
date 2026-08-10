"""Persistence, Telegram commands, health monitoring and engine integration."""

from __future__ import annotations

import asyncio

import pytest

from app.config import build_settings
from app.database.database import Database, dumps, loads
from app.database.models import TABLES, schema_sql
from app.domain import BotState, Candle, HealthState, Side, Timeframe
from app.engine import TradingEngine
from app.health.monitor import HealthMonitor
from app.health.preflight import run_preflight
from app.telegram.commands import COMMANDS, CommandRouter
from app.telegram.keyboards import emergency_confirm, main_menu
from app.telegram.notifications import Notifier
from tests.conftest import TEST_ENV


class TestDatabase:
    def test_schema_covers_every_required_table(self, database):
        names = {t.name for t in TABLES}
        required = {
            "candles", "features", "signals", "opportunities", "orders", "fills",
            "positions", "trades", "account_snapshots", "risk_events",
            "system_events", "model_versions", "backtest_results",
        }
        assert required <= names

    def test_ddl_renders_for_both_dialects(self):
        assert any("AUTOINCREMENT" in s for s in schema_sql("sqlite"))
        assert any("BIGSERIAL" in s for s in schema_sql("postgres"))

    def test_migrate_is_idempotent(self, database):
        database.migrate()
        database.migrate()
        assert database.health()["ok"]

    def test_json_helpers(self):
        assert loads(dumps({"a": 1})) == {"a": 1}
        assert loads(None, default=[]) == []
        assert loads("not json", default={}) == {}

    def test_candle_upsert_is_idempotent(self, repositories):
        candles = [Candle(ts=i * 3600, open=1, high=2, low=0.5, close=1.5, volume=10)
                   for i in range(10)]
        repositories.candles.save("BTCUSDT", Timeframe.H1, candles)
        repositories.candles.save("BTCUSDT", Timeframe.H1, candles)
        assert repositories.candles.count() == 10

    def test_candles_round_trip_in_order(self, repositories):
        candles = [Candle(ts=i * 3600, open=1, high=2, low=0.5, close=float(i), volume=10)
                   for i in range(10)]
        repositories.candles.save("BTCUSDT", Timeframe.H1, candles)
        loaded = repositories.candles.load("BTCUSDT", Timeframe.H1, limit=5)
        assert [c.ts for c in loaded] == sorted(c.ts for c in loaded)
        assert loaded[-1].close == 9.0

    def test_trade_lifecycle(self, repositories):
        trade_id = repositories.trades.open_trade({
            "symbol": "BTCUSDT", "side": "long", "entry_price": 100.0,
            "quantity": 1.0, "why_entered": ["BOS", "FVG"],
        })
        repositories.trades.close_trade(trade_id, 110.0, 10.0, 0.5, 0.1, 1.0, ["TP2"])
        trade = repositories.trades.get(trade_id)
        assert trade["status"] == "closed"
        assert trade["realized_pnl"] == 10.0
        assert trade["why_entered"] == ["BOS", "FVG"]
        assert trade["why_exited"] == ["TP2"]

    def test_consecutive_loss_counting(self, repositories):
        for pnl in (5.0, -3.0, -4.0):
            tid = repositories.trades.open_trade({
                "symbol": "X", "side": "long", "entry_price": 1, "quantity": 1})
            repositories.trades.close_trade(tid, 1, pnl, 0, 0, 0, [])
        assert repositories.trades.consecutive_losses() == 2

    def test_signal_audit_trail(self, repositories):
        repositories.signals.record({
            "symbol": "BTCUSDT", "decision": "NO_TRADE", "confidence": 0.4,
            "rejections": ["confidence too low"], "reasons": [], "scores": {"technical": 60},
        })
        recent = repositories.signals.recent(limit=5)
        assert recent[0]["rejections"] == ["confidence too low"]
        assert recent[0]["scores"]["technical"] == 60

    def test_risk_and_system_events(self, repositories):
        repositories.events.risk_event("DAILY_LOSS", "limit hit", severity="ERROR")
        repositories.events.system_event("engine", "started")
        assert repositories.events.recent_risk()[0]["kind"] == "DAILY_LOSS"
        assert repositories.events.recent_system()[0]["component"] == "engine"

    def test_order_status_updates(self, repositories):
        repositories.orders.create({
            "client_order_id": "abc", "symbol": "BTCUSDT", "side": "long",
            "intent": "open", "order_type": "market", "quantity": 1.0, "status": "new",
        })
        repositories.orders.update_status("abc", "filled", filled_quantity=1.0,
                                          average_price=100.0)
        order = repositories.orders.get("abc")
        assert order["status"] == "filled" and order["average_price"] == 100.0
        assert repositories.orders.open_orders() == []

    def test_account_snapshots_and_peak(self, repositories):
        for equity in (1000, 1100, 1050):
            repositories.account.snapshot({"mode": "paper", "equity": equity,
                                           "balance": equity})
        assert repositories.account.peak_equity() == 1100

    def test_model_registry_activation(self, repositories):
        repositories.models.register({"name": "m", "version": "1", "algorithm": "x"})
        repositories.models.register({"name": "m", "version": "2", "algorithm": "x"})
        repositories.models.activate("m", "2")
        assert repositories.models.active("m")["version"] == "2"

    def test_unsupported_url_is_rejected(self):
        from app.database.database import DatabaseError

        with pytest.raises(DatabaseError):
            Database("mysql://localhost/db")


class TestHealthMonitor:
    def test_aggregates_to_the_worst_state(self):
        monitor = HealthMonitor()
        monitor.register("good", lambda: {"ok": True})
        monitor.register("bad", lambda: {"ok": False, "error": "down"})
        report = asyncio.run(monitor.check())
        assert report.state is HealthState.ERROR
        assert not report.healthy
        assert report.by_name("bad").detail == "down"

    def test_all_healthy(self):
        monitor = HealthMonitor()
        monitor.register("a", lambda: {"ok": True})
        assert asyncio.run(monitor.check()).state is HealthState.HEALTHY

    def test_a_raising_probe_becomes_an_error(self):
        monitor = HealthMonitor()

        def broken():
            raise RuntimeError("boom")

        monitor.register("broken", broken)
        report = asyncio.run(monitor.check())
        assert report.state is HealthState.ERROR
        assert "boom" in report.by_name("broken").detail

    def test_async_probes_are_supported(self):
        monitor = HealthMonitor()

        async def probe():
            return {"state": "WARNING", "detail": "degraded"}

        monitor.register("async", probe)
        assert asyncio.run(monitor.check()).state is HealthState.WARNING

    def test_consecutive_error_tracking(self):
        monitor = HealthMonitor()
        monitor.register("flaky", lambda: {"ok": False})
        asyncio.run(monitor.check())
        asyncio.run(monitor.check())
        assert monitor.consecutive_errors["flaky"] == 2


class _FakeEngine:
    """Minimal engine surface the router needs."""

    def __init__(self):
        self.paused = False
        self.stopped = False
        self.emergency_called = False
        self.portfolio = None
        self.execution = None

    def status(self):
        return {
            "state": "PAUSED" if self.paused else "RUNNING",
            "emoji": "🟢", "mode": "paper", "scans": 3, "entries": 1, "exits": 0,
            "errors": 0, "last_error": "", "uptime_seconds": 3600, "health": "HEALTHY",
            "pause_reason": "", "account": {"equity": 1000.0, "open_positions": 0},
        }

    def scanner_view(self, limit=20):
        return [{"symbol": "BTCUSDT", "opportunity_score": 80.0, "trend": "BULLISH",
                 "confidence": 0.8, "rr": 2.5, "regime": "TREND_UP",
                 "status": "BEST SETUP", "reasons": ["BOS"], "rejections": []}]

    def positions_view(self):
        return []

    def orders_view(self, limit=20):
        return []

    def account_view(self):
        return {"mode": "paper", "equity": 1000.0, "risk": {}}

    def risk_view(self):
        return {"risk_per_trade": 0.005, "breaker_state": "OK", "clusters": []}

    def ai_view(self):
        return {"model": {"loaded": False}, "ml_weight": 0.35,
                "min_confidence": 0.7, "recent_signals": []}

    def performance_view(self, days=30):
        return {"trades": 0}

    def trades_view(self, limit=15):
        return []

    def signals_view(self, limit=15):
        return []

    def health_view(self):
        return {"state": "HEALTHY", "components": []}

    def preflight_view(self):
        return {"passed": True, "checks": []}

    def pause(self, reason=""):
        self.paused = True

    def resume(self):
        self.paused = False

    async def stop(self, reason=""):
        self.stopped = True

    async def emergency_stop(self, reason=""):
        self.emergency_called = True
        return {"closed": 2, "failed": 0, "cancelled": 1}


class TestTelegramRouter:
    @pytest.fixture
    def router(self, settings):
        env = dict(TEST_ENV)
        env["TELEGRAM_CHAT_ID"] = "555"
        env["TELEGRAM_ALLOWED_USER_IDS"] = "42"
        return CommandRouter(_FakeEngine(), build_settings(env=env))

    def test_only_allowlisted_users_may_control(self, router):
        assert router.is_authorised(42)
        assert not router.is_authorised(43)
        assert not router.is_authorised(0)

    def test_chat_fallback_when_no_allowlist(self):
        env = dict(TEST_ENV)
        env["TELEGRAM_CHAT_ID"] = "555"
        router = CommandRouter(_FakeEngine(), build_settings(env=env))
        assert router.is_authorised(1, chat_id="555")
        assert not router.is_authorised(1, chat_id="999")

    @pytest.mark.parametrize("command", sorted(COMMANDS))
    def test_every_documented_command_responds(self, router, command):
        result = asyncio.run(router.handle(f"/{command}", user_id=42))
        assert result.handled and result.text

    def test_unknown_command(self, router):
        result = asyncio.run(router.handle("/nope", user_id=42))
        assert "Unknown command" in result.text

    def test_callback_data_is_routed_like_a_command(self, router):
        assert asyncio.run(router.handle("cmd:status", user_id=42)).text

    def test_pause_and_resume(self, router):
        asyncio.run(router.handle("/pause", user_id=42))
        assert router.engine.paused
        asyncio.run(router.handle("/resume", user_id=42))
        assert not router.engine.paused

    def test_emergency_requires_confirmation(self, router):
        first = asyncio.run(router.handle("/emergency", user_id=42))
        assert not router.engine.emergency_called
        assert "CLOSE ALL" in str(first.keyboard)

        confirmed = asyncio.run(router.handle("confirm:emergency", user_id=42))
        assert router.engine.emergency_called
        assert "EMERGENCY STOP EXECUTED" in confirmed.text

    def test_stop_requires_confirmation(self, router):
        asyncio.run(router.handle("/stop", user_id=42))
        assert not router.engine.stopped
        asyncio.run(router.handle("confirm:stop", user_id=42))
        assert router.engine.stopped

    def test_menu_has_every_control(self):
        buttons = str(main_menu())
        for label in ("LIVE REPORT", "MARKET SCANNER", "POSITIONS", "START",
                      "PAUSE", "RESUME", "STOP", "EMERGENCY STOP", "ACCOUNT",
                      "PERFORMANCE", "AI", "RISK"):
            assert label in buttons

    def test_emergency_keyboard_offers_a_cancel(self):
        assert "CANCEL" in str(emergency_confirm())

    def test_output_is_html_escaped(self, router):
        result = asyncio.run(router.handle("/<script>", user_id=42))
        assert "<script>" not in result.text


class TestNotifier:
    def test_disabled_notifier_is_a_no_op(self):
        notifier = Notifier(client=None, chat_id="", enabled=False)
        assert not asyncio.run(notifier.send("hello"))
        assert notifier.stats.suppressed == 1

    def test_secrets_are_scrubbed_before_sending(self):
        sent: list[str] = []

        class Client:
            async def send_message(self, chat_id, text, keyboard=None, silent=False):
                sent.append(text)
                return {}

        from app.logger import register_secret

        register_secret("supersecrettelegramvalue")
        notifier = Notifier(Client(), "1", min_interval=0.0)
        asyncio.run(notifier.send("token supersecrettelegramvalue here"))
        assert "supersecrettelegramvalue" not in sent[0]

    def test_send_failure_is_swallowed(self):
        class Broken:
            async def send_message(self, *a, **k):
                raise RuntimeError("network")

        notifier = Notifier(Broken(), "1", min_interval=0.0)
        assert not asyncio.run(notifier.send("x"))
        assert notifier.stats.failed == 1


class TestEngineIntegration:
    def _settings(self, **overrides):
        env = dict(TEST_ENV)
        env.update(overrides)
        return build_settings(env=env)

    def test_full_paper_cycle(self):
        async def run():
            settings = self._settings(DEEP_ANALYSIS_COUNT="6")
            engine = TradingEngine(settings)
            await engine.build()
            await engine.start()
            assert engine.state is BotState.RUNNING

            opportunities = await engine.scan_once()
            assert opportunities

            # Every view must render without an exception.
            assert engine.status()["mode"] == "paper"
            assert isinstance(engine.scanner_view(), list)
            assert isinstance(engine.positions_view(), list)
            assert isinstance(engine.risk_view(), dict)
            assert isinstance(engine.ai_view(), dict)
            assert isinstance(engine.trades_view(), list)

            await engine.manage_positions()
            report = await engine.health.check()
            assert report.components

            engine.pause("test")
            assert engine.state is BotState.PAUSED
            engine.resume()
            assert engine.state is BotState.RUNNING

            await engine.stop("test complete")
            assert engine.state is BotState.STOPPED

        asyncio.run(run())

    def test_paper_mode_never_enables_live_trading(self):
        async def run():
            engine = TradingEngine(self._settings())
            await engine.build()
            await engine.start()
            assert not engine.live_enabled
            assert not engine.execution.live_enabled
            from app.paper.paper_engine import PaperEngine

            assert isinstance(engine.execution.broker, PaperEngine)
            await engine.stop()

        asyncio.run(run())

    def test_emergency_stop_flattens_and_stops(self):
        async def run():
            engine = TradingEngine(self._settings())
            await engine.build()
            await engine.start()
            result = await engine.emergency_stop("test")
            assert "closed" in result
            assert engine.state is BotState.STOPPED
            assert engine.risk_engine.breaker.emergency_active

        asyncio.run(run())

    def test_preflight_reports_every_check(self):
        async def run():
            settings = self._settings()
            engine = TradingEngine(settings)
            await engine.build()
            report = await run_preflight(
                settings, engine.exchange, engine.database, None, engine.reconciler
            )
            names = {c.name for c in report.checks}
            for expected in ("MEXC REST", "market data", "database",
                             "symbol validation", "risk configuration"):
                assert expected in names
            await engine.exchange.close()

        asyncio.run(run())

    def test_dashboard_routes_expose_no_secrets(self):
        async def run():
            from app.dashboard.api import DashboardRoutes

            settings = self._settings(
                MEXC_SECRET_KEY="s" * 30, TELEGRAM_BOT_TOKEN="t" * 40
            )
            engine = TradingEngine(settings)
            await engine.build()
            await engine.start()

            routes = DashboardRoutes(engine, settings)
            assert routes.health()["status"] == "ok"
            config = routes.config()
            assert "s" * 30 not in str(config)
            assert "t" * 40 not in str(config)
            snapshot = routes.snapshot()
            assert {"status", "account", "positions", "scanner"} <= set(snapshot)
            await engine.stop()

        asyncio.run(run())
