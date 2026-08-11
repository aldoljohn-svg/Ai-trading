"""The dashboard is an accessory. Losing it must not stop the bot.

A stale instance holding port 8000 took the whole process down: uvicorn calls
``sys.exit(1)`` when it cannot bind, which inside a task raises ``SystemExit``
and propagates out of the event loop. A plain ``except Exception`` does not
catch that.

The engine manages open positions. A bot that exits because a web UI could not
bind is a bot that leaves those positions unmanaged, which is far worse than
losing the UI.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from app.config import build_settings
from app.main import Application

from tests.conftest import TEST_ENV


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def app_env(**overrides) -> dict[str, str]:
    return dict(
        TEST_ENV,
        DEEP_ANALYSIS_COUNT="3",
        SCREEN_COUNT="6",
        MAX_SYMBOLS_TO_SCAN="20",
        **overrides,
    )


class TestDashboardFailureIsSurvivable:
    def test_a_port_clash_does_not_stop_the_engine(self):
        port = free_port()
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("0.0.0.0", port))
        blocker.listen(5)

        async def run():
            settings = build_settings(
                env=app_env(DASHBOARD_ENABLED="true", DASHBOARD_PORT=str(port))
            )
            app = Application(settings)
            await app.start()
            try:
                # The whole point: the engine is alive and working.
                assert app.engine is not None
                opportunities = await app.engine.scan_once()
                assert opportunities
                assert app.dashboard_server is None
            finally:
                await app.stop()

        try:
            asyncio.run(run())
        finally:
            blocker.close()

    def test_the_failure_is_reported_as_degraded_health(self):
        port = free_port()
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("0.0.0.0", port))
        blocker.listen(5)

        async def run():
            settings = build_settings(
                env=app_env(DASHBOARD_ENABLED="true", DASHBOARD_PORT=str(port))
            )
            app = Application(settings)
            await app.start()
            try:
                report = await app.engine.health.check()
                dashboard = [c for c in report.components if c.name == "Dashboard"]
                assert dashboard, "a dead dashboard must be visible in /health"
                assert dashboard[0].state.value == "WARNING"
                assert "bind" in dashboard[0].detail
            finally:
                await app.stop()

        try:
            asyncio.run(run())
        finally:
            blocker.close()

    def test_a_dashboard_that_raises_on_start_is_survived(self, monkeypatch):
        """Any startup failure, not only a bind clash."""

        async def run():
            settings = build_settings(env=app_env(DASHBOARD_ENABLED="true"))
            app = Application(settings)

            async def explode() -> None:
                raise RuntimeError("frontend assets missing")

            monkeypatch.setattr(app, "_start_uvicorn_dashboard", explode)
            monkeypatch.setattr(
                "app.dashboard.server.serve_dashboard",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")),
            )
            await app.start()
            try:
                assert app.dashboard_server is None
                assert app.engine.state.value == "RUNNING"
            finally:
                await app.stop()

        asyncio.run(run())

    def test_a_disabled_dashboard_is_not_an_error(self):
        async def run():
            settings = build_settings(env=app_env(DASHBOARD_ENABLED="false"))
            app = Application(settings)
            await app.start()
            try:
                assert app.dashboard_server is None
                assert app.dashboard_task is None
                assert app.engine.state.value == "RUNNING"
            finally:
                await app.stop()

        asyncio.run(run())

    def test_shutdown_is_clean_after_a_failed_dashboard(self):
        port = free_port()
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("0.0.0.0", port))
        blocker.listen(5)

        async def run():
            settings = build_settings(
                env=app_env(DASHBOARD_ENABLED="true", DASHBOARD_PORT=str(port))
            )
            app = Application(settings)
            await app.start()
            await app.stop()
            if app.dashboard_task is not None:
                assert app.dashboard_task.done()

        try:
            asyncio.run(run())
        finally:
            blocker.close()


class TestDashboardWorksWhenThePortIsFree:
    def test_it_starts_normally(self):
        async def run():
            settings = build_settings(
                env=app_env(DASHBOARD_ENABLED="true", DASHBOARD_PORT=str(free_port()))
            )
            app = Application(settings)
            await app.start()
            try:
                assert app.dashboard_server is not None
            finally:
                await app.stop()

        asyncio.run(run())
