"""Standard-library dashboard server.

Used when FastAPI/uvicorn are not installed.  It serves exactly the same routes
as :func:`app.dashboard.api.create_app` by delegating to the same
:class:`~app.dashboard.api.DashboardRoutes` object, so ``/health`` and the JSON
API behave identically either way.  There is no WebSocket here - the frontend
falls back to polling.

Runs in a background thread with a threaded HTTP server; handlers only read
already-computed state, so they never block the trading event loop.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from app.dashboard.api import FRONTEND_DIR, DashboardRoutes
from app.logger import get_logger

log = get_logger(__name__)


def _make_handler(routes: DashboardRoutes, loop: asyncio.AbstractEventLoop | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TradingBotDashboard/1.0"

        # -- plumbing -----------------------------------------------------

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            log.debug("dashboard %s - %s", self.address_string(), format % args)

        def _send(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, text: str, status: int = 200) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # -- routing ------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)

            try:
                if path == "/":
                    index = FRONTEND_DIR / "index.html"
                    if index.is_file():
                        self._send_html(index.read_text(encoding="utf-8"))
                    else:
                        self._send_html("<h1>Frontend missing</h1>", 500)
                    return
                if path == "/health":
                    data = routes.health()
                    self._send(data, 200 if data["status"] == "ok" else 503)
                    return

                mapping = {
                    "/api/status": routes.status,
                    "/api/account": routes.account,
                    "/api/positions": routes.positions,
                    "/api/scanner": routes.scanner,
                    "/api/signals": routes.signals,
                    "/api/trades": routes.trades,
                    "/api/orders": routes.orders,
                    "/api/risk": routes.risk,
                    "/api/ai": routes.ai,
                    "/api/preflight": routes.preflight,
                    "/api/config": routes.config,
                    "/api/snapshot": routes.snapshot,
                }
                handler = mapping.get(path)
                if handler is not None:
                    self._send(handler())
                    return
                if path == "/api/performance":
                    days = int(query.get("days", ["30"])[0])
                    self._send(routes.performance(max(1, min(days, 365))))
                    return
                if path == "/api/equity":
                    limit = int(query.get("limit", ["500"])[0])
                    self._send(routes.equity_curve(max(10, min(limit, 5000))))
                    return

                self._send({"error": "not found", "path": path}, 404)
            except Exception as exc:  # noqa: BLE001 - never kill the server thread
                log.exception("dashboard GET %s failed", path)
                self._send({"error": str(exc)}, 500)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")
            token = self.headers.get("X-Dashboard-Token") or parse_qs(
                parsed.query
            ).get("token", [None])[0]

            if not routes.authorised(token):
                self._send({"error": "dashboard token required"}, 401)
                return
            if loop is None:
                self._send({"error": "control endpoints need a running engine"}, 503)
                return

            actions = {
                "/api/control/pause": routes.pause,
                "/api/control/resume": routes.resume,
                "/api/control/emergency": routes.emergency,
            }
            action = actions.get(path)
            if action is None:
                self._send({"error": "not found", "path": path}, 404)
                return
            try:
                # The engine lives on the asyncio loop; hop onto it safely.
                future = asyncio.run_coroutine_threadsafe(action(), loop)
                self._send(future.result(timeout=120))
            except Exception as exc:  # noqa: BLE001
                log.exception("dashboard POST %s failed", path)
                self._send({"error": str(exc)}, 500)

    return Handler


class FallbackDashboardServer:
    def __init__(
        self,
        routes: DashboardRoutes,
        host: str = "0.0.0.0",
        port: int = 8000,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.routes = routes
        self.host = host
        self.port = port
        self.loop = loop
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = _make_handler(self.routes, self.loop)
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="dashboard-http",
            daemon=True,
        )
        self._thread.start()
        log.warning(
            "FastAPI is not installed - serving the dashboard with the "
            "standard-library fallback on http://%s:%d (no WebSocket)",
            self.host,
            self.port,
        )

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}"


def serve_dashboard(
    engine: Any,
    settings: Any,
    loop: asyncio.AbstractEventLoop | None = None,
) -> FallbackDashboardServer:
    routes = DashboardRoutes(engine, settings)
    server = FallbackDashboardServer(
        routes, settings.dashboard_host, settings.dashboard_port, loop=loop
    )
    server.start()
    return server


__all__ = ["FallbackDashboardServer", "serve_dashboard"]
