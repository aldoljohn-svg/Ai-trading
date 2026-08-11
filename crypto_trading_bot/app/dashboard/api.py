"""Dashboard HTTP API.

``DashboardRoutes`` holds every handler as a plain method returning JSON-ready
data.  ``create_app`` wraps them in FastAPI when it is available;
:mod:`app.dashboard.server` wraps the same object in a standard-library server
when it is not.  Neither wrapper contains business logic.

Security note: the dashboard is **read-only by default** and never exposes
secrets - :meth:`config` returns ``Settings.redacted()``.  The only mutating
endpoints are pause/resume/emergency, and they require the dashboard token when
one is configured.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.compat import HAVE_FASTAPI, fastapi
from app.logger import get_logger

log = get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


class DashboardRoutes:
    """All dashboard data, independent of the web framework."""

    def __init__(self, engine: Any, settings: Any) -> None:
        self.engine = engine
        self.settings = settings
        self.started_at = time.time()

    # -- read-only --------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Liveness + component health.  Always available."""

        health = self.engine.health_view() if self.engine else {"state": "UNKNOWN"}
        state = health.get("state", "UNKNOWN")
        return {
            "status": "ok" if state in ("HEALTHY", "WARNING", "UNKNOWN") else "degraded",
            "state": state,
            "mode": self.settings.trading_mode.value,
            "uptime_seconds": int(time.time() - self.started_at),
            "components": health.get("components", []),
            "version": _version(),
        }

    def status(self) -> dict[str, Any]:
        return self.engine.status() if self.engine else {}

    def account(self) -> dict[str, Any]:
        return self.engine.account_view() if self.engine else {}

    def positions(self) -> dict[str, Any]:
        return {"positions": self.engine.positions_view() if self.engine else []}

    def scanner(self) -> dict[str, Any]:
        return {"opportunities": self.engine.scanner_view(limit=25) if self.engine else []}

    def signals(self) -> dict[str, Any]:
        return {"signals": self.engine.signals_view(limit=25) if self.engine else []}

    def trades(self) -> dict[str, Any]:
        return {"trades": self.engine.trades_view(limit=50) if self.engine else []}

    def orders(self) -> dict[str, Any]:
        return {"orders": self.engine.orders_view(limit=50) if self.engine else []}

    def performance(self, days: int = 30) -> dict[str, Any]:
        return self.engine.performance_view(days=days) if self.engine else {}

    def risk(self) -> dict[str, Any]:
        return self.engine.risk_view() if self.engine else {}

    def ai(self) -> dict[str, Any]:
        return self.engine.ai_view() if self.engine else {}

    def preflight(self) -> dict[str, Any]:
        return self.engine.preflight_view() if self.engine else {}

    # -- intelligence layer -----------------------------------------------

    def _view(self, name: str, *args: Any, default: Any = None) -> Any:
        """Call an engine view defensively; the dashboard is read-only."""

        view = getattr(self.engine, name, None) if self.engine else None
        if not callable(view):
            return default
        try:
            return view(*args)
        except Exception:  # noqa: BLE001 - a broken panel must not 500 the page
            return default

    def intelligence(self) -> dict[str, Any]:
        return self._view("intelligence_view", default={"enabled": False})

    def verdicts(self, limit: int = 10) -> dict[str, Any]:
        return {"verdicts": self._view("verdicts_view", limit, default=[]) or []}

    def flow(self, limit: int = 10) -> dict[str, Any]:
        return {"symbols": self._view("flow_view", limit, default=[]) or []}

    def journal(self, limit: int = 20) -> dict[str, Any]:
        return {"entries": self._view("journal_view", limit, default=[]) or []}

    def learning(self) -> dict[str, Any]:
        return self._view("learning_view", default={"enabled": False})

    def decision(self, symbol: str) -> dict[str, Any]:
        return self._view("verdict_view", symbol, default={}) or {}

    def config(self) -> dict[str, Any]:
        """Redacted configuration - secrets are masked, never returned."""

        return self.settings.redacted()

    def equity_curve(self, limit: int = 500) -> dict[str, Any]:
        repositories = getattr(self.engine, "repositories", None)
        if repositories is None:
            return {"points": []}
        history = repositories.account.history(
            limit=limit, mode=self.settings.trading_mode.value
        )
        return {
            "points": [
                {
                    "ts": row["ts"],
                    "equity": row["equity"],
                    "drawdown": row["drawdown"],
                    "open_positions": row["open_positions"],
                }
                for row in history
            ]
        }

    def snapshot(self) -> dict[str, Any]:
        """Everything the dashboard needs in one round trip."""

        return {
            "ts": int(time.time()),
            "status": self.status(),
            "account": self.account(),
            "positions": self.positions()["positions"],
            "scanner": self.scanner()["opportunities"],
            "risk": self.risk(),
            "health": self.engine.health_view() if self.engine else {},
            "performance": self.performance(),
            "intelligence": self.intelligence(),
            "verdicts": self.verdicts(6)["verdicts"],
            "flow": self.flow(6)["symbols"],
            "learning": self.learning(),
        }

    # -- mutating ---------------------------------------------------------

    def authorised(self, token: str | None) -> bool:
        expected = getattr(self.settings, "dashboard_token", "")
        if not expected:
            return True
        return bool(token) and token == expected

    async def pause(self) -> dict[str, Any]:
        self.engine.pause("paused from the dashboard")
        return {"ok": True, "state": self.engine.status()["state"]}

    async def resume(self) -> dict[str, Any]:
        self.engine.resume()
        return {"ok": True, "state": self.engine.status()["state"]}

    async def emergency(self) -> dict[str, Any]:
        result = await self.engine.emergency_stop("emergency stop from the dashboard")
        return {"ok": True, **result}


def _version() -> str:
    try:
        from app import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------
# FastAPI wrapper
# --------------------------------------------------------------------------


def create_app(engine: Any, settings: Any) -> Any:
    """Build the FastAPI application.  Raises if FastAPI is not installed."""

    if not HAVE_FASTAPI:
        raise RuntimeError(
            "FastAPI is not installed; app.dashboard.server provides the "
            "standard-library fallback"
        )

    from fastapi import FastAPI, HTTPException, Query, Request  # type: ignore
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # type: ignore

    routes = DashboardRoutes(engine, settings)
    app = FastAPI(
        title="Crypto Trading Bot",
        version=_version(),
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.routes = routes
    app.state.engine = engine

    def _guard(request: Request) -> None:
        token = request.headers.get("X-Dashboard-Token") or request.query_params.get(
            "token"
        )
        if not routes.authorised(token):
            raise HTTPException(status_code=401, detail="dashboard token required")

    @app.get("/health")
    async def health() -> Any:
        data = routes.health()
        return JSONResponse(data, status_code=200 if data["status"] == "ok" else 503)

    @app.get("/api/status")
    async def status() -> Any:
        return routes.status()

    @app.get("/api/account")
    async def account() -> Any:
        return routes.account()

    @app.get("/api/positions")
    async def positions() -> Any:
        return routes.positions()

    @app.get("/api/scanner")
    async def scanner() -> Any:
        return routes.scanner()

    @app.get("/api/signals")
    async def signals() -> Any:
        return routes.signals()

    @app.get("/api/trades")
    async def trades() -> Any:
        return routes.trades()

    @app.get("/api/orders")
    async def orders() -> Any:
        return routes.orders()

    @app.get("/api/performance")
    async def performance(days: int = Query(30, ge=1, le=365)) -> Any:
        return routes.performance(days)

    @app.get("/api/risk")
    async def risk() -> Any:
        return routes.risk()

    @app.get("/api/ai")
    async def ai() -> Any:
        return routes.ai()

    @app.get("/api/preflight")
    async def preflight() -> Any:
        return routes.preflight()

    @app.get("/api/config")
    async def config() -> Any:
        return routes.config()

    @app.get("/api/intelligence")
    async def intelligence() -> Any:
        return routes.intelligence()

    @app.get("/api/verdicts")
    async def verdicts(limit: int = Query(10, ge=1, le=60)) -> Any:
        return routes.verdicts(limit)

    @app.get("/api/flow")
    async def flow(limit: int = Query(10, ge=1, le=60)) -> Any:
        return routes.flow(limit)

    @app.get("/api/journal")
    async def journal(limit: int = Query(20, ge=1, le=100)) -> Any:
        return routes.journal(limit)

    @app.get("/api/learning")
    async def learning() -> Any:
        return routes.learning()

    @app.get("/api/decision/{symbol}")
    async def decision(symbol: str) -> Any:
        return routes.decision(symbol)

    @app.get("/api/equity")
    async def equity(limit: int = Query(500, ge=10, le=5000)) -> Any:
        return routes.equity_curve(limit)

    @app.get("/api/snapshot")
    async def snapshot() -> Any:
        return routes.snapshot()

    @app.post("/api/control/pause")
    async def pause(request: Request) -> Any:
        _guard(request)
        return await routes.pause()

    @app.post("/api/control/resume")
    async def resume(request: Request) -> Any:
        _guard(request)
        return await routes.resume()

    @app.post("/api/control/emergency")
    async def emergency(request: Request) -> Any:
        _guard(request)
        return await routes.emergency()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> Any:
        index_file = FRONTEND_DIR / "index.html"
        if index_file.is_file():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Dashboard frontend is missing</h1>", status_code=500)

    from app.dashboard.websocket import register_websocket

    register_websocket(app, routes)
    return app


__all__ = ["DashboardRoutes", "create_app", "FRONTEND_DIR"]
