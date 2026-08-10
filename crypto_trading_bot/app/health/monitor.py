"""Runtime health monitoring.

Each component reports 🟢 HEALTHY / 🟡 WARNING / 🔴 ERROR.  The overall state is
the worst component state, and a 🔴 on market data, the exchange or the database
feeds the circuit breaker - degraded infrastructure is a trading risk, not just
an ops problem.

Components watched: MEXC REST, MEXC WebSocket, database, Telegram, scanner,
AI/ML, risk engine, execution, market data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.domain import HealthState
from app.logger import get_logger

log = get_logger(__name__)

Probe = Callable[[], Any]


@dataclass(slots=True)
class ComponentHealth:
    name: str
    state: HealthState = HealthState.UNKNOWN
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    checked_at: int = 0

    def line(self) -> str:
        return f"{self.state.emoji} {self.name}: {self.detail or self.state.value}"


@dataclass(slots=True)
class HealthReport:
    components: list[ComponentHealth] = field(default_factory=list)
    ts: int = 0

    @property
    def state(self) -> HealthState:
        if not self.components:
            return HealthState.UNKNOWN
        return max((c.state for c in self.components), key=lambda s: s.rank)

    @property
    def healthy(self) -> bool:
        return self.state in (HealthState.HEALTHY, HealthState.UNKNOWN)

    def by_name(self, name: str) -> ComponentHealth | None:
        for component in self.components:
            if component.name == name:
                return component
        return None

    def errors(self) -> list[ComponentHealth]:
        return [c for c in self.components if c.state is HealthState.ERROR]

    def summary(self) -> str:
        lines = [f"{self.state.emoji} SYSTEM {self.state.value}", ""]
        lines.extend(c.line() for c in self.components)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "ts": self.ts,
            "components": [
                {
                    "name": c.name,
                    "state": c.state.value,
                    "detail": c.detail,
                    "metrics": c.metrics,
                }
                for c in self.components
            ],
        }


class HealthMonitor:
    """Collects health from every registered component."""

    def __init__(self) -> None:
        self._probes: dict[str, Probe] = {}
        self.last_report: HealthReport | None = None
        self.consecutive_errors: dict[str, int] = {}

    def register(self, name: str, probe: Probe) -> None:
        self._probes[name] = probe

    async def check(self) -> HealthReport:
        report = HealthReport(ts=int(time.time()))
        for name, probe in self._probes.items():
            component = ComponentHealth(name=name, checked_at=report.ts)
            try:
                result = probe()
                if hasattr(result, "__await__"):
                    result = await result
                self._interpret(component, result)
            except Exception as exc:  # noqa: BLE001 - a probe must never crash us
                component.state = HealthState.ERROR
                component.detail = f"{type(exc).__name__}: {exc}"

            if component.state is HealthState.ERROR:
                self.consecutive_errors[name] = self.consecutive_errors.get(name, 0) + 1
            else:
                self.consecutive_errors[name] = 0
            report.components.append(component)

        self.last_report = report
        if not report.healthy:
            log.warning("health degraded:\n%s", report.summary())
        return report

    @staticmethod
    def _interpret(component: ComponentHealth, result: Any) -> None:
        """Translate a component's own dict into a health state."""

        if isinstance(result, HealthState):
            component.state = result
            return
        if isinstance(result, bool):
            component.state = HealthState.HEALTHY if result else HealthState.ERROR
            return
        if not isinstance(result, dict):
            component.state = HealthState.UNKNOWN
            component.detail = str(result)
            return

        component.metrics = {
            k: v for k, v in result.items() if k not in {"state", "detail"}
        }

        explicit = result.get("state")
        if explicit:
            try:
                component.state = HealthState(str(explicit).upper())
            except ValueError:
                component.state = HealthState.UNKNOWN
        elif "ok" in result:
            component.state = (
                HealthState.HEALTHY if result["ok"] else HealthState.ERROR
            )
        else:
            component.state = HealthState.HEALTHY

        detail = result.get("detail") or result.get("error") or result.get("reason")
        if detail:
            component.detail = str(detail)
        elif component.state is HealthState.HEALTHY:
            parts = []
            for key in ("latency_ms", "messages", "entries", "open", "predictions"):
                if key in result:
                    parts.append(f"{key}={result[key]}")
            component.detail = ", ".join(parts) or "ok"


def market_data_health(market_data: Any, max_age: float = 180.0) -> dict[str, Any]:
    """Health probe for :class:`~app.data.market_data.MarketData`."""

    data = market_data.health()
    age = data.get("seconds_since_success")
    failures = data.get("failures", 0)
    fetches = max(data.get("fetches", 1), 1)
    error_rate = failures / fetches

    if age is None:
        return {"state": "UNKNOWN", "detail": "no successful fetch yet", **data}
    if age > max_age:
        return {
            "state": "ERROR",
            "detail": f"no good market data for {age:.0f}s",
            **data,
        }
    if error_rate > 0.25:
        return {
            "state": "WARNING",
            "detail": f"{error_rate:.0%} of fetches failed",
            **data,
        }
    return {"state": "HEALTHY", "detail": f"fresh ({age:.0f}s ago)", **data}


def websocket_health(socket: Any) -> dict[str, Any]:
    data = socket.health()
    if not data.get("ok"):
        reason = data.get("reason") or data.get("last_error") or "disconnected"
        # The socket is an optimisation, never a requirement: REST still works.
        return {"state": "WARNING", "detail": f"{reason} (running REST-only)", **data}
    return {"state": "HEALTHY", "detail": f"{data.get('messages', 0)} messages", **data}


def scanner_health(scanner: Any, interval: float) -> dict[str, Any]:
    data = scanner.health()
    age = data.get("seconds_since_scan")
    if age is None:
        return {"state": "UNKNOWN", "detail": "has not run yet", **data}
    if age > interval * 4:
        return {"state": "ERROR", "detail": f"last scan {age:.0f}s ago", **data}
    if age > interval * 2:
        return {"state": "WARNING", "detail": f"last scan {age:.0f}s ago", **data}
    return {
        "state": "HEALTHY",
        "detail": f"{data.get('candidates', 0)} candidates, {age:.0f}s ago",
        **data,
    }


def predictor_health(predictor: Any) -> dict[str, Any]:
    info = predictor.info()
    if not info.get("loaded"):
        # Not an error: the bot is designed to work without a model.
        return {"state": "HEALTHY", "detail": "rule-based only (no model loaded)", **info}
    if info.get("failures", 0) > 10:
        return {"state": "WARNING", "detail": f"{info['failures']} prediction failures", **info}
    return {"state": "HEALTHY", "detail": f"{info.get('model')} loaded", **info}


def risk_health(risk_engine: Any, state: Any) -> dict[str, Any]:
    report = risk_engine.last_report or risk_engine.check_breakers(state)
    if report.state.value == "TRIPPED":
        return {
            "state": "WARNING",   # a tripped breaker is correct behaviour, not a fault
            "detail": "; ".join(t.reason.value for t in report.trips),
            "trips": [t.reason.value for t in report.trips],
        }
    if report.warnings:
        return {"state": "WARNING", "detail": "; ".join(report.warnings)}
    return {"state": "HEALTHY", "detail": "all breakers clear"}


__all__ = [
    "HealthMonitor",
    "HealthReport",
    "ComponentHealth",
    "market_data_health",
    "websocket_health",
    "scanner_health",
    "predictor_health",
    "risk_health",
]
