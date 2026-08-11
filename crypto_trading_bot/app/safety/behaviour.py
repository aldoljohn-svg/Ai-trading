"""Behavioural safeguards.

The bot has no emotions, but it can reproduce every behaviour that emotions
cause in humans - and for the same structural reason: a losing streak changes
the statistics the system is conditioning on. These guards make the failure
modes explicit and impossible rather than merely discouraged.

* **Anti-revenge**: risk may only ever be *cut* after losses.
* **Overtrading**: trade frequency must be justified by realised edge.
* **Capital preservation mode**: a graduated de-risking state.
* **Recovery mode**: after a drawdown, become *more* selective, not less.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


class RecoveryState(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    CAPITAL_PRESERVATION = "CAPITAL_PRESERVATION"
    RECOVERY = "RECOVERY"

    @property
    def risk_multiplier(self) -> float:
        return {
            RecoveryState.NORMAL: 1.0,
            RecoveryState.CAUTION: 0.7,
            RecoveryState.CAPITAL_PRESERVATION: 0.4,
            RecoveryState.RECOVERY: 0.5,
        }[self]

    @property
    def confidence_premium(self) -> float:
        """Added to the minimum confidence - selectivity rises under stress."""

        return {
            RecoveryState.NORMAL: 0.0,
            RecoveryState.CAUTION: 0.03,
            RecoveryState.CAPITAL_PRESERVATION: 0.08,
            RecoveryState.RECOVERY: 0.06,
        }[self]

    @property
    def max_positions_multiplier(self) -> float:
        return {
            RecoveryState.NORMAL: 1.0,
            RecoveryState.CAUTION: 1.0,
            RecoveryState.CAPITAL_PRESERVATION: 0.5,
            RecoveryState.RECOVERY: 0.67,
        }[self]


@dataclass(slots=True)
class OvertradingReport:
    trades_last_hour: int = 0
    trades_last_day: int = 0
    expectancy_r: float = 0.0
    overtrading: bool = False
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "trades_last_hour": self.trades_last_hour,
            "trades_last_day": self.trades_last_day,
            "expectancy_r": round(self.expectancy_r, 4),
            "overtrading": self.overtrading,
            "detail": self.detail,
        }


@dataclass(slots=True)
class BehaviourVerdict:
    state: RecoveryState = RecoveryState.NORMAL
    risk_multiplier: float = 1.0
    confidence_premium: float = 0.0
    max_positions_multiplier: float = 1.0
    overtrading: OvertradingReport = field(default_factory=OvertradingReport)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "risk_multiplier": round(self.risk_multiplier, 4),
            "confidence_premium": round(self.confidence_premium, 4),
            "max_positions_multiplier": round(self.max_positions_multiplier, 4),
            "overtrading": self.overtrading.as_dict(),
            "reasons": self.reasons,
        }

    def summary(self) -> str:
        if self.state is RecoveryState.NORMAL and not self.reasons:
            return "behaviour normal"
        lines = [
            f"{self.state.value}: risk x{self.risk_multiplier:.2f}, "
            f"confidence +{self.confidence_premium:.0%}"
        ]
        lines.extend(f"  {reason}" for reason in self.reasons)
        return "\n".join(lines)


class BehaviourGuard:
    def __init__(
        self,
        max_trades_per_hour: int = 4,
        max_trades_per_day: int = 15,
        drawdown_caution: float = 0.04,
        drawdown_preservation: float = 0.07,
        recovery_exit_drawdown: float = 0.02,
    ) -> None:
        self.max_trades_per_hour = max_trades_per_hour
        self.max_trades_per_day = max_trades_per_day
        self.drawdown_caution = drawdown_caution
        self.drawdown_preservation = drawdown_preservation
        self.recovery_exit_drawdown = recovery_exit_drawdown
        self._in_recovery = False
        self._entry_times: list[int] = []

    # -- recording --------------------------------------------------------

    def record_entry(self, ts: int | None = None) -> None:
        self._entry_times.append(int(ts if ts is not None else time.time()))
        cutoff = int(time.time()) - 7 * 86400
        self._entry_times = [t for t in self._entry_times if t >= cutoff]

    def check_overtrading(
        self, expectancy_r: float = 0.0, now: int | None = None
    ) -> OvertradingReport:
        now = int(now if now is not None else time.time())
        last_hour = sum(1 for t in self._entry_times if now - t <= 3600)
        last_day = sum(1 for t in self._entry_times if now - t <= 86400)

        report = OvertradingReport(
            trades_last_hour=last_hour,
            trades_last_day=last_day,
            expectancy_r=expectancy_r,
        )

        if last_hour > self.max_trades_per_hour:
            report.overtrading = True
            report.detail = (
                f"{last_hour} entries in the last hour exceeds the "
                f"{self.max_trades_per_hour} limit"
            )
        elif last_day > self.max_trades_per_day:
            report.overtrading = True
            report.detail = (
                f"{last_day} entries today exceeds the {self.max_trades_per_day} limit"
            )
        elif last_day >= self.max_trades_per_day * 0.7 and expectancy_r <= 0:
            # Frequency rising while the edge is not there is the real signature.
            report.overtrading = True
            report.detail = (
                f"{last_day} trades today with an expectancy of "
                f"{expectancy_r:+.3f}R - frequency is not being rewarded"
            )
        return report

    # -- evaluation --------------------------------------------------------

    def evaluate(
        self,
        drawdown_pct: float = 0.0,
        consecutive_losses: int = 0,
        expectancy_r: float = 0.0,
        data_risk: float = 0.0,
        model_drift: float = 0.0,
        volatility_ratio: float = 1.0,
        now: int | None = None,
    ) -> BehaviourVerdict:
        verdict = BehaviourVerdict()
        reasons: list[str] = []
        state = RecoveryState.NORMAL

        # --- recovery mode is sticky until the drawdown genuinely heals ---
        if drawdown_pct >= self.drawdown_preservation:
            self._in_recovery = True
        elif self._in_recovery and drawdown_pct <= self.recovery_exit_drawdown:
            self._in_recovery = False
            reasons.append("drawdown recovered - leaving recovery mode")

        if self._in_recovery:
            state = RecoveryState.RECOVERY
            reasons.append(
                f"recovery mode: {drawdown_pct:.1%} below peak - trading smaller and "
                "more selectively, not bigger"
            )
        elif drawdown_pct >= self.drawdown_preservation:
            state = RecoveryState.CAPITAL_PRESERVATION
            reasons.append(f"capital preservation: {drawdown_pct:.1%} drawdown")
        elif drawdown_pct >= self.drawdown_caution:
            state = RecoveryState.CAUTION
            reasons.append(f"caution: {drawdown_pct:.1%} drawdown")

        # --- other stressors escalate the state ---------------------------
        stressors: list[str] = []
        if consecutive_losses >= 3:
            stressors.append(f"{consecutive_losses} consecutive losses")
        if data_risk >= 0.5:
            stressors.append(f"data risk {data_risk:.2f}")
        if model_drift >= 0.5:
            stressors.append(f"model drift {model_drift:.2f}")
        if volatility_ratio >= 2.0:
            stressors.append(f"volatility {volatility_ratio:.1f}x normal")

        if stressors and state is RecoveryState.NORMAL:
            state = RecoveryState.CAUTION
        elif len(stressors) >= 2 and state is RecoveryState.CAUTION:
            state = RecoveryState.CAPITAL_PRESERVATION
        reasons.extend(stressors)

        verdict.state = state
        verdict.risk_multiplier = state.risk_multiplier
        verdict.confidence_premium = state.confidence_premium
        verdict.max_positions_multiplier = state.max_positions_multiplier

        # --- overtrading ----------------------------------------------------
        verdict.overtrading = self.check_overtrading(expectancy_r, now)
        if verdict.overtrading.overtrading:
            verdict.risk_multiplier *= 0.5
            verdict.confidence_premium += 0.05
            reasons.append(verdict.overtrading.detail)

        verdict.reasons = reasons
        return verdict

    @staticmethod
    def clamp_risk_multiplier(multiplier: float) -> float:
        """Anti-revenge invariant: the guard may only ever reduce risk.

        Every caller routes through this, so there is no code path by which a
        behavioural adjustment can increase position size.
        """

        return max(0.0, min(multiplier, 1.0))


__all__ = [
    "BehaviourGuard",
    "BehaviourVerdict",
    "OvertradingReport",
    "RecoveryState",
]
