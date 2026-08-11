"""Multi-layer kill switch.

Five escalating levels, each strictly more restrictive than the last::

    L1  STOP_NEW_TRADES     no new entries; existing positions managed normally
    L2  REDUCE_SIZE         new entries allowed at reduced size
    L3  CLOSE_RISKY         close positions that are losing or unprotected
    L4  CLOSE_ALL           flatten everything
    L5  DISABLE_LIVE        flatten and refuse to trade until a human resets

Design rules:

* The active level is the **maximum** demanded by any trigger. Levels never
  cancel each other out.
* De-escalation only happens when the triggering condition clears *and* the
  trigger is not sticky. Loss-based triggers are sticky by design: recovering
  the equity does not undo the fact that the day hit its limit.
* ``L5`` can only be cleared by an explicit human action, never automatically.
* The AI has no method here that lowers a level. It can only *raise* one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable

from app.logger import get_logger

log = get_logger(__name__)


class KillSwitchLevel(IntEnum):
    NONE = 0
    STOP_NEW_TRADES = 1
    REDUCE_SIZE = 2
    CLOSE_RISKY = 3
    CLOSE_ALL = 4
    DISABLE_LIVE = 5

    @property
    def label(self) -> str:
        return {
            KillSwitchLevel.NONE: "NORMAL",
            KillSwitchLevel.STOP_NEW_TRADES: "L1 STOP NEW TRADES",
            KillSwitchLevel.REDUCE_SIZE: "L2 REDUCE SIZE",
            KillSwitchLevel.CLOSE_RISKY: "L3 CLOSE RISKY POSITIONS",
            KillSwitchLevel.CLOSE_ALL: "L4 CLOSE ALL",
            KillSwitchLevel.DISABLE_LIVE: "L5 LIVE TRADING DISABLED",
        }[self]

    @property
    def emoji(self) -> str:
        return ["🟢", "🟡", "🟠", "🔴", "🚨", "⛔"][int(self)]

    @property
    def blocks_new_entries(self) -> bool:
        return self >= KillSwitchLevel.STOP_NEW_TRADES

    @property
    def requires_flatten(self) -> bool:
        return self >= KillSwitchLevel.CLOSE_ALL

    @property
    def size_multiplier(self) -> float:
        if self >= KillSwitchLevel.STOP_NEW_TRADES:
            return 0.0
        return 1.0


@dataclass(frozen=True, slots=True)
class Trigger:
    code: str
    level: KillSwitchLevel
    message: str
    ts: int
    sticky: bool = False        # survives the condition clearing
    manual_only: bool = False   # only a human can clear it

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "level": int(self.level),
            "level_label": self.level.label,
            "message": self.message,
            "ts": self.ts,
            "sticky": self.sticky,
            "manual_only": self.manual_only,
        }


@dataclass(slots=True)
class KillSwitchState:
    level: KillSwitchLevel = KillSwitchLevel.NONE
    triggers: list[Trigger] = field(default_factory=list)
    size_multiplier: float = 1.0

    @property
    def blocks_new_entries(self) -> bool:
        return self.level.blocks_new_entries

    @property
    def requires_flatten(self) -> bool:
        return self.level.requires_flatten

    @property
    def requires_derisk(self) -> bool:
        return self.level >= KillSwitchLevel.CLOSE_RISKY

    def reasons(self) -> list[str]:
        return [f"{t.code}: {t.message}" for t in self.triggers]

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": int(self.level),
            "level_label": self.level.label,
            "emoji": self.level.emoji,
            "size_multiplier": round(self.size_multiplier, 4),
            "blocks_new_entries": self.blocks_new_entries,
            "requires_flatten": self.requires_flatten,
            "triggers": [t.as_dict() for t in self.triggers],
        }

    def summary(self) -> str:
        if self.level is KillSwitchLevel.NONE:
            return "🟢 kill switch clear"
        lines = [f"{self.level.emoji} {self.level.label}"]
        for trigger in sorted(self.triggers, key=lambda t: -int(t.level)):
            lines.append(f"  [{int(trigger.level)}] {trigger.code}: {trigger.message}")
        return "\n".join(lines)


@dataclass(slots=True)
class KillSwitchThresholds:
    daily_loss_l1: float = 0.02
    daily_loss_l3: float = 0.035
    weekly_loss_l1: float = 0.05
    weekly_loss_l4: float = 0.08
    drawdown_l2: float = 0.06
    drawdown_l3: float = 0.10
    drawdown_l4: float = 0.15
    api_error_rate_l1: float = 0.25
    api_error_rate_l3: float = 0.5
    consecutive_losses_l1: int = 4
    consecutive_losses_l2: int = 6
    model_drift_l2: float = 0.6
    volatility_ratio_l1: float = 2.5
    volatility_ratio_l3: float = 4.0


class MultiLayerKillSwitch:
    """Evaluates every trigger and reports the highest level demanded."""

    def __init__(
        self,
        thresholds: KillSwitchThresholds | None = None,
        size_reduction: float = 0.5,
    ) -> None:
        self.thresholds = thresholds or KillSwitchThresholds()
        self.size_reduction = size_reduction
        self._manual: list[Trigger] = []
        self._sticky: list[Trigger] = []
        self._live_disabled = False

    # -- human controls (the only things that can lower a level) ----------

    def engage(
        self,
        level: KillSwitchLevel,
        message: str,
        code: str = "MANUAL",
        manual_only: bool = True,
    ) -> Trigger:
        trigger = Trigger(
            code=code,
            level=level,
            message=message,
            ts=int(time.time()),
            sticky=True,
            manual_only=manual_only,
        )
        self._manual.append(trigger)
        if level >= KillSwitchLevel.DISABLE_LIVE:
            self._live_disabled = True
        log.warning("kill switch engaged at %s: %s", level.label, message)
        return trigger

    def human_reset(self, clear_live_disable: bool = False) -> int:
        """Only a human calls this. Clears manual and sticky triggers."""

        count = len(self._manual) + len(self._sticky)
        self._manual.clear()
        self._sticky.clear()
        if clear_live_disable:
            self._live_disabled = False
        log.warning("kill switch reset by operator (%d trigger(s) cleared)", count)
        return count

    @property
    def live_disabled(self) -> bool:
        return self._live_disabled

    # -- evaluation --------------------------------------------------------

    def evaluate(
        self,
        daily_loss_pct: float = 0.0,
        weekly_loss_pct: float = 0.0,
        drawdown_pct: float = 0.0,
        consecutive_losses: int = 0,
        api_error_rate: float = 0.0,
        data_risk: float = 0.0,
        model_drift: float = 0.0,
        volatility_ratio: float = 1.0,
        reconciliation_failed: bool = False,
        exchange_unhealthy: bool = False,
        anomaly_severity: float = 0.0,
        now: int | None = None,
    ) -> KillSwitchState:
        now = int(now if now is not None else time.time())
        thresholds = self.thresholds
        triggers: list[Trigger] = list(self._manual) + list(self._sticky)

        def add(
            code: str,
            level: KillSwitchLevel,
            message: str,
            sticky: bool = False,
        ) -> None:
            trigger = Trigger(code, level, message, now, sticky=sticky)
            triggers.append(trigger)
            if sticky and not any(t.code == code for t in self._sticky):
                self._sticky.append(trigger)

        # --- capital losses (sticky: recovering does not undo the breach) ---
        if daily_loss_pct >= thresholds.daily_loss_l3:
            add("DAILY_LOSS", KillSwitchLevel.CLOSE_RISKY,
                f"daily loss {daily_loss_pct:.2%} is severe", sticky=True)
        elif daily_loss_pct >= thresholds.daily_loss_l1:
            add("DAILY_LOSS", KillSwitchLevel.STOP_NEW_TRADES,
                f"daily loss limit {daily_loss_pct:.2%} reached", sticky=True)

        if weekly_loss_pct >= thresholds.weekly_loss_l4:
            add("WEEKLY_LOSS", KillSwitchLevel.CLOSE_ALL,
                f"weekly loss {weekly_loss_pct:.2%} is critical", sticky=True)
        elif weekly_loss_pct >= thresholds.weekly_loss_l1:
            add("WEEKLY_LOSS", KillSwitchLevel.STOP_NEW_TRADES,
                f"weekly loss {weekly_loss_pct:.2%} reached the limit", sticky=True)

        if drawdown_pct >= thresholds.drawdown_l4:
            add("DRAWDOWN", KillSwitchLevel.CLOSE_ALL,
                f"drawdown {drawdown_pct:.2%} is critical", sticky=True)
        elif drawdown_pct >= thresholds.drawdown_l3:
            add("DRAWDOWN", KillSwitchLevel.CLOSE_RISKY,
                f"drawdown {drawdown_pct:.2%} exceeded the hard limit", sticky=True)
        elif drawdown_pct >= thresholds.drawdown_l2:
            add("DRAWDOWN", KillSwitchLevel.REDUCE_SIZE,
                f"drawdown {drawdown_pct:.2%} - reducing size")

        # --- losing streak ----------------------------------------------------
        if consecutive_losses >= thresholds.consecutive_losses_l2:
            add("LOSS_STREAK", KillSwitchLevel.REDUCE_SIZE,
                f"{consecutive_losses} consecutive losses", sticky=True)
        elif consecutive_losses >= thresholds.consecutive_losses_l1:
            add("LOSS_STREAK", KillSwitchLevel.STOP_NEW_TRADES,
                f"{consecutive_losses} consecutive losses - pausing for review",
                sticky=True)

        # --- infrastructure (not sticky: it recovers when the link does) -----
        if api_error_rate >= thresholds.api_error_rate_l3:
            add("API_ERRORS", KillSwitchLevel.CLOSE_RISKY,
                f"{api_error_rate:.0%} of exchange calls failing")
        elif api_error_rate >= thresholds.api_error_rate_l1:
            add("API_ERRORS", KillSwitchLevel.STOP_NEW_TRADES,
                f"{api_error_rate:.0%} of exchange calls failing")

        if exchange_unhealthy:
            add("EXCHANGE_HEALTH", KillSwitchLevel.STOP_NEW_TRADES,
                "exchange connectivity is unreliable")

        if data_risk >= 0.7:
            add("DATA_RISK", KillSwitchLevel.STOP_NEW_TRADES,
                f"data risk score {data_risk:.2f}")

        if reconciliation_failed:
            add("RECONCILIATION", KillSwitchLevel.STOP_NEW_TRADES,
                "internal state does not match the exchange", sticky=True)

        # --- model health -------------------------------------------------------
        if model_drift >= thresholds.model_drift_l2:
            add("MODEL_DRIFT", KillSwitchLevel.REDUCE_SIZE,
                f"model drift severity {model_drift:.2f}")

        # --- market conditions ---------------------------------------------------
        if volatility_ratio >= thresholds.volatility_ratio_l3:
            add("VOLATILITY", KillSwitchLevel.CLOSE_RISKY,
                f"volatility {volatility_ratio:.1f}x normal")
        elif volatility_ratio >= thresholds.volatility_ratio_l1:
            add("VOLATILITY", KillSwitchLevel.STOP_NEW_TRADES,
                f"volatility {volatility_ratio:.1f}x normal")

        if anomaly_severity >= 0.8:
            add("ANOMALY", KillSwitchLevel.CLOSE_RISKY,
                f"severe market anomaly (severity {anomaly_severity:.2f})")
        elif anomaly_severity >= 0.5:
            add("ANOMALY", KillSwitchLevel.STOP_NEW_TRADES,
                f"market anomaly detected (severity {anomaly_severity:.2f})")

        if self._live_disabled:
            triggers.append(
                Trigger("LIVE_DISABLED", KillSwitchLevel.DISABLE_LIVE,
                        "live trading disabled - operator reset required",
                        now, sticky=True, manual_only=True)
            )

        level = max((t.level for t in triggers), default=KillSwitchLevel.NONE)
        state = KillSwitchState(level=level, triggers=triggers)

        if level is KillSwitchLevel.REDUCE_SIZE:
            state.size_multiplier = self.size_reduction
        elif level >= KillSwitchLevel.STOP_NEW_TRADES:
            state.size_multiplier = 0.0
        else:
            state.size_multiplier = 1.0
        return state


__all__ = [
    "MultiLayerKillSwitch",
    "KillSwitchLevel",
    "KillSwitchState",
    "KillSwitchThresholds",
    "Trigger",
]
