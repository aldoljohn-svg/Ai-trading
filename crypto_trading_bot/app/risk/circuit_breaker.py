"""Circuit breakers - automatic, unconditional trading halts.

A breaker is not a suggestion.  When one trips, new entries stop; depending on
severity, open positions are de-risked or flattened.  Nothing in the system can
bypass a tripped breaker except an explicit operator reset, and the daily-loss
breaker cannot be reset at all until the next UTC day - "just one more trade to
make it back" is precisely the behaviour this exists to prevent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from app.logger import get_logger

log = get_logger(__name__)


class BreakerState(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    TRIPPED = "TRIPPED"

    @property
    def emoji(self) -> str:
        return {
            BreakerState.OK: "🟢",
            BreakerState.WARNING: "🟡",
            BreakerState.TRIPPED: "🔴",
        }[self]


class TripReason(str, Enum):
    DAILY_LOSS = "DAILY_LOSS"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    API_ERRORS = "API_ERRORS"
    DATA_QUALITY = "DATA_QUALITY"
    RECONCILIATION = "RECONCILIATION"
    KILL_SWITCH = "KILL_SWITCH"
    MANUAL = "MANUAL"
    EMERGENCY = "EMERGENCY"

    @property
    def action(self) -> str:
        """What tripping this breaker does to *existing* positions."""

        return {
            TripReason.DAILY_LOSS: "block new entries; manage open positions normally",
            TripReason.MAX_DRAWDOWN: "block new entries and reduce open risk",
            TripReason.CONSECUTIVE_LOSSES: "block new entries until reset",
            TripReason.API_ERRORS: "block new entries; do not touch positions blindly",
            TripReason.DATA_QUALITY: "block new entries; manage on last good data",
            TripReason.RECONCILIATION: "block everything until state is verified",
            TripReason.KILL_SWITCH: "block new entries",
            TripReason.MANUAL: "block new entries",
            TripReason.EMERGENCY: "close everything and stop",
        }[self]


@dataclass(frozen=True, slots=True)
class Trip:
    reason: TripReason
    message: str
    ts: int
    until_ts: int | None = None       # None = until manually reset

    def active(self, now: int | None = None) -> bool:
        if self.until_ts is None:
            return True
        return int(now if now is not None else time.time()) < self.until_ts


@dataclass(slots=True)
class BreakerReport:
    state: BreakerState
    trips: list[Trip] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.state is BreakerState.TRIPPED

    @property
    def requires_flatten(self) -> bool:
        return any(t.reason is TripReason.EMERGENCY for t in self.trips)

    @property
    def requires_derisk(self) -> bool:
        return any(
            t.reason in (TripReason.MAX_DRAWDOWN, TripReason.RECONCILIATION)
            for t in self.trips
        )

    def summary(self) -> str:
        if not self.trips and not self.warnings:
            return f"{self.state.emoji} all breakers clear"
        lines = [f"{self.state.emoji} {self.state.value}"]
        for trip in self.trips:
            lines.append(f"  🔴 {trip.reason.value}: {trip.message}")
        for warning in self.warnings:
            lines.append(f"  🟡 {warning}")
        return "\n".join(lines)


class CircuitBreaker:
    def __init__(
        self,
        max_daily_loss: float = 0.02,
        max_drawdown: float = 0.10,
        max_consecutive_losses: int = 4,
        max_api_error_rate: float = 0.25,
        api_error_window: int = 40,
        kill_switch_file: Path | None = None,
    ) -> None:
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.max_api_error_rate = max_api_error_rate
        self.api_error_window = api_error_window
        self.kill_switch_file = kill_switch_file

        self._manual_trips: list[Trip] = []
        self._api_results: list[bool] = []
        self._data_problem: str = ""
        self._reconciliation_problem: str = ""
        self._emergency = False

    # -- external signals -------------------------------------------------

    def record_api_result(self, ok: bool) -> None:
        self._api_results.append(ok)
        if len(self._api_results) > self.api_error_window:
            self._api_results.pop(0)

    def set_data_problem(self, message: str) -> None:
        self._data_problem = message

    def clear_data_problem(self) -> None:
        self._data_problem = ""

    def set_reconciliation_problem(self, message: str) -> None:
        self._reconciliation_problem = message

    def clear_reconciliation_problem(self) -> None:
        self._reconciliation_problem = ""

    def trip_manual(self, message: str, until_ts: int | None = None) -> Trip:
        trip = Trip(
            reason=TripReason.MANUAL,
            message=message,
            ts=int(time.time()),
            until_ts=until_ts,
        )
        self._manual_trips.append(trip)
        log.warning("manual circuit breaker: %s", message)
        return trip

    def trigger_emergency(self, message: str = "emergency stop requested") -> Trip:
        self._emergency = True
        trip = Trip(reason=TripReason.EMERGENCY, message=message, ts=int(time.time()))
        self._manual_trips.append(trip)
        log.error("EMERGENCY breaker: %s", message)
        return trip

    def reset_manual(self) -> int:
        count = len(self._manual_trips)
        self._manual_trips.clear()
        self._emergency = False
        return count

    @property
    def emergency_active(self) -> bool:
        return self._emergency

    def kill_switch_active(self) -> bool:
        return bool(self.kill_switch_file and self.kill_switch_file.exists())

    # -- evaluation -------------------------------------------------------

    def evaluate(
        self,
        equity: float,
        realized_pnl_today: float,
        peak_equity: float,
        consecutive_losses: int,
        now: int | None = None,
    ) -> BreakerReport:
        now = int(now if now is not None else time.time())
        trips: list[Trip] = [t for t in self._manual_trips if t.active(now)]
        warnings: list[str] = []
        metrics: dict[str, float] = {}

        # --- daily loss ---------------------------------------------------
        daily_loss = max(0.0, -realized_pnl_today) / equity if equity > 0 else 0.0
        metrics["daily_loss"] = round(daily_loss, 6)
        if daily_loss >= self.max_daily_loss:
            trips.append(
                Trip(
                    reason=TripReason.DAILY_LOSS,
                    message=(
                        f"lost {daily_loss:.2%} today, at or beyond the "
                        f"{self.max_daily_loss:.2%} limit - no new entries until "
                        "the next UTC day"
                    ),
                    ts=now,
                    until_ts=_next_utc_midnight(now),
                )
            )
        elif daily_loss >= self.max_daily_loss * 0.7:
            warnings.append(
                f"daily loss {daily_loss:.2%} is approaching the "
                f"{self.max_daily_loss:.2%} limit"
            )

        # --- drawdown -----------------------------------------------------
        peak = max(peak_equity, equity)
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        metrics["drawdown"] = round(drawdown, 6)
        if drawdown >= self.max_drawdown:
            trips.append(
                Trip(
                    reason=TripReason.MAX_DRAWDOWN,
                    message=(
                        f"drawdown {drawdown:.2%} has reached the "
                        f"{self.max_drawdown:.2%} limit - reduce risk and reset manually"
                    ),
                    ts=now,
                )
            )
        elif drawdown >= self.max_drawdown * 0.7:
            warnings.append(f"drawdown {drawdown:.2%} is deepening")

        # --- losing streak -------------------------------------------------
        metrics["consecutive_losses"] = float(consecutive_losses)
        if consecutive_losses >= self.max_consecutive_losses:
            trips.append(
                Trip(
                    reason=TripReason.CONSECUTIVE_LOSSES,
                    message=(
                        f"{consecutive_losses} losses in a row - pausing so the "
                        "system is reviewed rather than allowed to keep firing"
                    ),
                    ts=now,
                )
            )

        # --- exchange health ------------------------------------------------
        if len(self._api_results) >= 10:
            failures = sum(1 for ok in self._api_results if not ok)
            rate = failures / len(self._api_results)
            metrics["api_error_rate"] = round(rate, 4)
            if rate >= self.max_api_error_rate:
                trips.append(
                    Trip(
                        reason=TripReason.API_ERRORS,
                        message=(
                            f"{rate:.0%} of the last {len(self._api_results)} exchange "
                            "calls failed - the venue or the link is unhealthy"
                        ),
                        ts=now,
                        until_ts=now + 300,
                    )
                )
            elif rate >= self.max_api_error_rate * 0.6:
                warnings.append(f"exchange error rate {rate:.0%}")

        # --- data quality ---------------------------------------------------
        if self._data_problem:
            trips.append(
                Trip(
                    reason=TripReason.DATA_QUALITY,
                    message=self._data_problem,
                    ts=now,
                    until_ts=now + 120,
                )
            )

        # --- reconciliation -------------------------------------------------
        if self._reconciliation_problem:
            trips.append(
                Trip(
                    reason=TripReason.RECONCILIATION,
                    message=self._reconciliation_problem,
                    ts=now,
                )
            )

        # --- kill switch file -----------------------------------------------
        if self.kill_switch_active():
            trips.append(
                Trip(
                    reason=TripReason.KILL_SWITCH,
                    message=(
                        f"kill switch file present at {self.kill_switch_file} - "
                        "delete it to resume"
                    ),
                    ts=now,
                )
            )

        state = (
            BreakerState.TRIPPED
            if trips
            else (BreakerState.WARNING if warnings else BreakerState.OK)
        )
        return BreakerReport(state=state, trips=trips, warnings=warnings, metrics=metrics)


def _next_utc_midnight(now: int) -> int:
    return (now // 86400 + 1) * 86400


__all__ = [
    "CircuitBreaker",
    "BreakerState",
    "BreakerReport",
    "TripReason",
    "Trip",
]
