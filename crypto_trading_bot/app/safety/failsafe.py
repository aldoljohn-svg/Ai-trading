"""Fail-safe supervision.

The governing principle: **when a critical subsystem fails, the safe state is
"no new trades", never "carry on".**

Faults are registered by the components that detect them and cleared by the
components that recover. The supervisor turns the set of active faults into a
kill-switch level, so a database outage produces the same disciplined response
as a loss limit.

Also covers latency and clock drift, since both silently corrupt execution:
a stale decision and a rejected signature look like different problems and are
both "we are not synchronised with reality".
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Iterable
from collections import deque

from app.logger import get_logger
from app.safety.kill_switch import KillSwitchLevel

log = get_logger(__name__)

#: MEXC rejects signed requests drifting more than a few seconds.
MAX_CLOCK_DRIFT_SECONDS = 5.0


class SystemFault(str, Enum):
    DATABASE = "DATABASE"
    EXCHANGE_REST = "EXCHANGE_REST"
    EXCHANGE_WS = "EXCHANGE_WS"
    MARKET_DATA = "MARKET_DATA"
    RISK_ENGINE = "RISK_ENGINE"
    MODEL = "MODEL"
    CLOCK_DRIFT = "CLOCK_DRIFT"
    LATENCY = "LATENCY"
    RECONCILIATION = "RECONCILIATION"
    UNKNOWN_ORDER_STATE = "UNKNOWN_ORDER_STATE"

    @property
    def level(self) -> KillSwitchLevel:
        """How severely this fault restricts trading."""

        return {
            # Losing the record of what we did is worse than losing prices.
            SystemFault.DATABASE: KillSwitchLevel.STOP_NEW_TRADES,
            SystemFault.EXCHANGE_REST: KillSwitchLevel.STOP_NEW_TRADES,
            # The socket is an optimisation; REST still works.
            SystemFault.EXCHANGE_WS: KillSwitchLevel.NONE,
            SystemFault.MARKET_DATA: KillSwitchLevel.STOP_NEW_TRADES,
            SystemFault.RISK_ENGINE: KillSwitchLevel.CLOSE_RISKY,
            SystemFault.MODEL: KillSwitchLevel.NONE,
            SystemFault.CLOCK_DRIFT: KillSwitchLevel.STOP_NEW_TRADES,
            SystemFault.LATENCY: KillSwitchLevel.STOP_NEW_TRADES,
            SystemFault.RECONCILIATION: KillSwitchLevel.STOP_NEW_TRADES,
            SystemFault.UNKNOWN_ORDER_STATE: KillSwitchLevel.STOP_NEW_TRADES,
        }[self]


@dataclass(frozen=True, slots=True)
class Fault:
    fault: SystemFault
    detail: str
    ts: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fault": self.fault.value,
            "detail": self.detail,
            "ts": self.ts,
            "level": int(self.fault.level),
        }


@dataclass(slots=True)
class FailSafeReport:
    faults: list[Fault] = field(default_factory=list)
    level: KillSwitchLevel = KillSwitchLevel.NONE
    latency_ms: float = 0.0
    clock_drift_seconds: float = 0.0

    @property
    def safe_to_trade(self) -> bool:
        return self.level < KillSwitchLevel.STOP_NEW_TRADES

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": int(self.level),
            "safe_to_trade": self.safe_to_trade,
            "latency_ms": round(self.latency_ms, 1),
            "clock_drift_seconds": round(self.clock_drift_seconds, 3),
            "faults": [f.as_dict() for f in self.faults],
        }

    def summary(self) -> str:
        if not self.faults:
            return "🟢 no system faults"
        lines = [f"🔴 {len(self.faults)} system fault(s) - trading restricted"]
        for fault in self.faults:
            lines.append(f"  {fault.fault.value}: {fault.detail}")
        return "\n".join(lines)


class FailSafe:
    def __init__(
        self,
        max_latency_ms: float = 3000.0,
        max_clock_drift: float = MAX_CLOCK_DRIFT_SECONDS,
        latency_window: int = 50,
    ) -> None:
        self.max_latency_ms = max_latency_ms
        self.max_clock_drift = max_clock_drift
        self._faults: dict[SystemFault, Fault] = {}
        self._latencies: dict[str, Deque[float]] = {}
        self.clock_drift_seconds = 0.0
        self.latency_window = latency_window

    # -- faults -----------------------------------------------------------

    def raise_fault(self, fault: SystemFault, detail: str) -> None:
        if fault not in self._faults:
            log.error("system fault raised: %s - %s", fault.value, detail)
        self._faults[fault] = Fault(fault, detail, int(time.time()))

    def clear_fault(self, fault: SystemFault) -> None:
        if self._faults.pop(fault, None) is not None:
            log.info("system fault cleared: %s", fault.value)

    def has_fault(self, fault: SystemFault) -> bool:
        return fault in self._faults

    # -- latency -----------------------------------------------------------

    def record_latency(self, channel: str, milliseconds: float) -> None:
        window = self._latencies.setdefault(channel, deque(maxlen=self.latency_window))
        window.append(max(0.0, milliseconds))

    def latency(self, channel: str) -> float:
        """Median latency - robust to the occasional slow call."""

        window = self._latencies.get(channel)
        if not window:
            return 0.0
        return statistics.median(window)

    def worst_latency(self) -> tuple[str, float]:
        worst = ("", 0.0)
        for channel in self._latencies:
            value = self.latency(channel)
            if value > worst[1]:
                worst = (channel, value)
        return worst

    def check_latency(self) -> None:
        channel, value = self.worst_latency()
        if value > self.max_latency_ms:
            self.raise_fault(
                SystemFault.LATENCY,
                f"{channel} median latency {value:.0f}ms exceeds "
                f"{self.max_latency_ms:.0f}ms",
            )
        else:
            self.clear_fault(SystemFault.LATENCY)

    # -- clock --------------------------------------------------------------

    def record_clock_drift(self, drift_seconds: float) -> None:
        self.clock_drift_seconds = drift_seconds
        if abs(drift_seconds) > self.max_clock_drift:
            self.raise_fault(
                SystemFault.CLOCK_DRIFT,
                f"system clock is {drift_seconds:+.2f}s from the exchange - "
                "signed requests will be rejected. Enable NTP.",
            )
        else:
            self.clear_fault(SystemFault.CLOCK_DRIFT)

    # -- supervision ---------------------------------------------------------

    def evaluate(self) -> FailSafeReport:
        self.check_latency()
        faults = list(self._faults.values())
        level = max((f.fault.level for f in faults), default=KillSwitchLevel.NONE)
        channel, latency = self.worst_latency()
        return FailSafeReport(
            faults=faults,
            level=level,
            latency_ms=latency,
            clock_drift_seconds=self.clock_drift_seconds,
        )

    def observe_health(self, health_report: Any) -> None:
        """Translate a :class:`~app.health.monitor.HealthReport` into faults."""

        if health_report is None:
            return
        mapping = {
            "Database": SystemFault.DATABASE,
            "MEXC REST": SystemFault.EXCHANGE_REST,
            "MEXC WebSocket": SystemFault.EXCHANGE_WS,
            "Market Data": SystemFault.MARKET_DATA,
            "Risk Engine": SystemFault.RISK_ENGINE,
            "AI": SystemFault.MODEL,
        }
        for component in getattr(health_report, "components", []):
            fault = mapping.get(component.name)
            if fault is None:
                continue
            if component.state.value == "ERROR":
                self.raise_fault(fault, component.detail or "component in ERROR state")
            else:
                self.clear_fault(fault)


__all__ = [
    "FailSafe",
    "FailSafeReport",
    "SystemFault",
    "Fault",
    "MAX_CLOCK_DRIFT_SECONDS",
]
