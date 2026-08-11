"""Application entry point.

Starts, in order:

1. configuration + logging (with secret redaction installed first)
2. the trading engine (which does crash recovery and pre-flight itself)
3. the Telegram control interface, if configured
4. the dashboard - FastAPI/uvicorn when available, the stdlib server otherwise

Shutdown is graceful on SIGINT/SIGTERM: loops stop, positions are persisted,
and the exchange connection is closed.  Open positions are **left open on the
exchange** with their stops in place - stopping the bot is not an instruction
to liquidate.  Use ``/emergency`` for that.

Run with::

    python -m app.main
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
from typing import Any

from app.compat import HAVE_FASTAPI, HAVE_UVICORN, capabilities
from app.config import ConfigError, Settings, TradingMode, get_settings
from app.engine import TradingEngine
from app.logger import get_logger, setup_logging

log = get_logger(__name__)


class Application:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: TradingEngine | None = None
        self.telegram: Any = None
        self.notifier: Any = None
        self.dashboard_server: Any = None
        self.dashboard_task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

    # -- wiring -----------------------------------------------------------

    def _build_notifier(self) -> Any:
        from app.engine import NullNotifier
        from app.telegram.bot import TelegramClient, TelegramError
        from app.telegram.notifications import Notifier

        if not self.settings.has_telegram:
            log.info("Telegram is not configured - notifications disabled")
            return NullNotifier()
        try:
            client = TelegramClient(self.settings.telegram_bot_token)
        except TelegramError as exc:
            log.error("could not create the Telegram client: %s", exc)
            return NullNotifier()
        return Notifier(client, self.settings.telegram_chat_id)

    async def start(self) -> None:
        settings = self.settings

        log.info("=" * 68)
        log.info("Crypto Trading Bot starting")
        log.info("  mode        : %s", settings.trading_mode.value.upper())
        log.info("  data source : %s", settings.data_source)
        log.info("  instance    : %s", settings.instance_name)
        log.info("  risk/trade  : %.2f%%", settings.default_risk_per_trade * 100)
        log.info("  max positions: %d   max leverage: %gx",
                 settings.max_open_positions, settings.max_leverage)
        available = [k for k, v in capabilities().items() if v]
        log.info("  optional deps: %s", ", ".join(available) or "none")
        log.info("=" * 68)

        self.notifier = self._build_notifier()
        self.engine = TradingEngine(settings, notifier=self.notifier)
        await self.engine.build()
        await self.engine.start()

        await self._start_telegram()
        await self._start_dashboard()

        with contextlib.suppress(Exception):
            await self.notifier.startup(settings, self.engine.status())

    async def _start_telegram(self) -> None:
        if not self.settings.has_telegram:
            return
        from app.telegram.bot import TelegramBot

        self.telegram = TelegramBot(self.settings, self.engine, notifier=self.notifier)
        await self.telegram.start()
        if self.engine is not None and self.telegram.enabled:
            self.engine.health.register("Telegram", self.telegram.health)

    async def _start_dashboard(self) -> None:
        """Start the dashboard, or carry on without it.

        The dashboard is an accessory.  The engine manages open positions, and a
        bot that exits because a web UI could not bind a port is a bot that
        leaves those positions unmanaged -- which is far worse than losing the
        UI.  Every failure here is logged and swallowed.

        Uvicorn calls ``sys.exit(1)`` when it cannot bind, which inside a task
        raises ``SystemExit``; that propagates out of the event loop and takes
        the process down, so it has to be caught explicitly rather than relying
        on a plain ``except Exception``.
        """

        settings = self.settings
        if not settings.dashboard_enabled:
            log.info("dashboard disabled by configuration")
            return

        try:
            if HAVE_FASTAPI and HAVE_UVICORN:
                await self._start_uvicorn_dashboard()
            else:
                from app.dashboard.server import serve_dashboard

                self.dashboard_server = serve_dashboard(
                    self.engine, settings, loop=asyncio.get_running_loop()
                )
                log.info("dashboard on %s", self.dashboard_server.url)
        except OSError as exc:
            self._dashboard_unavailable(
                f"could not bind {settings.dashboard_host}:{settings.dashboard_port}"
                f" ({exc}). Another instance is probably already running -"
                " check with: ss -ltnp | grep :%d" % settings.dashboard_port
            )
        except Exception as exc:  # noqa: BLE001 - the UI is never worth a crash
            self._dashboard_unavailable(f"failed to start: {exc}")

    async def _start_uvicorn_dashboard(self) -> None:
        import uvicorn  # type: ignore

        from app.dashboard.api import create_app

        settings = self.settings
        app = create_app(self.engine, settings)
        config = uvicorn.Config(
            app,
            host=settings.dashboard_host,
            port=settings.dashboard_port,
            log_level=settings.log_level.value.lower(),
            access_log=False,
            lifespan="on",
        )
        server = uvicorn.Server(config)
        # Uvicorn installs its own signal handlers; we own shutdown.
        server.install_signal_handlers = lambda: None  # type: ignore[assignment]
        self.dashboard_server = server

        async def supervise() -> None:
            try:
                await server.serve()
            except asyncio.CancelledError:
                raise
            except SystemExit as exc:
                # uvicorn's way of reporting a failed bind.
                self._dashboard_unavailable(
                    f"could not bind {settings.dashboard_host}:"
                    f"{settings.dashboard_port} (exit {exc.code}). Another "
                    "instance is probably already running."
                )
            except Exception as exc:  # noqa: BLE001
                self._dashboard_unavailable(f"stopped unexpectedly: {exc}")

        self.dashboard_task = asyncio.create_task(supervise(), name="dashboard")

        # Give the bind a moment to fail so the "dashboard on ..." line is not
        # printed for a server that is already dead.
        await asyncio.sleep(0.5)
        if self.dashboard_server is None:
            return
        log.info(
            "dashboard on http://%s:%d",
            "localhost" if settings.dashboard_host == "0.0.0.0" else settings.dashboard_host,
            settings.dashboard_port,
        )

    def _dashboard_unavailable(self, detail: str) -> None:
        """Record that the dashboard is down without stopping the engine."""

        self.dashboard_server = None
        log.error(
            "DASHBOARD UNAVAILABLE - %s\n"
            "    The trading engine is unaffected and keeps running. "
            "Telegram control still works.",
            detail,
        )
        if self.engine is not None:
            self.engine.health.register(
                "Dashboard",
                lambda: {"state": "WARNING", "detail": detail[:180]},
            )

    # -- lifecycle --------------------------------------------------------

    async def run(self) -> None:
        await self.start()
        await self._shutdown.wait()
        await self.stop()

    def request_shutdown(self, signal_name: str = "signal") -> None:
        log.info("shutdown requested (%s)", signal_name)
        self._shutdown.set()

    async def stop(self) -> None:
        log.info("shutting down…")

        if self.telegram is not None:
            with contextlib.suppress(Exception):
                await self.telegram.stop()

        if self.engine is not None:
            with contextlib.suppress(Exception):
                await self.engine.stop("process shutdown")

        if self.dashboard_task is not None and self.dashboard_server is not None:
            self.dashboard_server.should_exit = True
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self.dashboard_task, timeout=10)
        elif self.dashboard_server is not None:
            with contextlib.suppress(Exception):
                self.dashboard_server.stop()
        elif self.dashboard_task is not None and not self.dashboard_task.done():
            # The dashboard failed to start; its supervisor may still be
            # unwinding.  Do not leave the task dangling.
            self.dashboard_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.dashboard_task

        log.info("shutdown complete")


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="crypto-trading-bot",
        description=(
            "Autonomous MEXC futures trading bot. Defaults to PAPER mode; LIVE "
            "requires TRADING_MODE=live plus passing pre-flight checks."
        ),
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate configuration and exit",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true", help="do not start the web dashboard"
    )
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        settings = get_settings()
    except ConfigError as exc:
        # Logging may not be configured yet, so print as well.
        print(f"CONFIGURATION ERROR\n\n{exc}", file=sys.stderr)
        return 2

    setup_logging(
        level=settings.log_level.value,
        log_dir=settings.resolve_path(settings.log_dir),
        secrets=settings.secret_values(),
    )

    if args.check_config:
        log.info("configuration is valid")
        for key, value in sorted(settings.redacted().items()):
            log.info("  %-28s %s", key, value)
        return 0

    if args.no_dashboard:
        settings = settings.with_overrides(dashboard_enabled=False)

    application = Application(settings)

    loop = asyncio.get_running_loop()
    for signal_name in ("SIGINT", "SIGTERM"):
        signal_number = getattr(signal, signal_name, None)
        if signal_number is None:
            continue
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(
                signal_number, application.request_shutdown, signal_name
            )

    try:
        await application.run()
    except Exception as exc:  # noqa: BLE001 - top level safety net
        log.exception("fatal error: %s", exc)
        with contextlib.suppress(Exception):
            await application.stop()
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
