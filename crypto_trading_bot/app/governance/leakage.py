"""Data-leakage protection.

Lookahead bias is the single most expensive bug class in a trading system,
because it produces *better* results and no error. These guards make the common
forms structurally detectable:

* **future candle leakage** - a bar whose close time is after the decision time
* **unclosed bars** - using the currently-forming candle
* **timestamp misalignment** - bars off the timeframe grid
* **future news/fundamentals** - an event dated after the decision
* **feature timestamps** - any feature carrying a stamp later than the decision

The guard is cheap enough to run in production, and the backtester and dataset
builder call it on every window they hand to a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from app.domain import Candle, Timeframe


class LeakageError(AssertionError):
    """Raised when data from after the decision point is detected."""


def assert_no_lookahead(
    candles: Sequence[Candle],
    decision_ts: int,
    timeframe: Timeframe,
    label: str = "series",
) -> None:
    """Every bar must have *closed* at or before ``decision_ts``.

    A bar opening at ``ts`` closes at ``ts + timeframe.seconds``; using a bar
    whose close is in the future means the model saw price action that had not
    happened yet.
    """

    if not candles:
        return
    step = timeframe.seconds
    for candle in candles:
        close_ts = candle.ts + step
        if close_ts > decision_ts:
            raise LeakageError(
                f"{label}: bar opening at {candle.ts} closes at {close_ts}, "
                f"after the decision timestamp {decision_ts} - lookahead bias"
            )
        if not candle.closed:
            raise LeakageError(
                f"{label}: bar at {candle.ts} is still forming and must not be used"
            )
        if candle.ts % step != 0:
            raise LeakageError(
                f"{label}: bar timestamp {candle.ts} is not aligned to the "
                f"{timeframe.value} grid - timestamps are unreliable"
            )


@dataclass(slots=True)
class LeakageReport:
    ok: bool = True
    violations: list[str] = field(default_factory=list)
    checked: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": self.checked,
            "violations": self.violations,
        }


class LeakageGuard:
    """Collects violations instead of raising - for auditing a whole run."""

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self.report = LeakageReport()

    def check_series(
        self,
        candles: Sequence[Candle],
        decision_ts: int,
        timeframe: Timeframe,
        label: str = "series",
    ) -> bool:
        self.report.checked += 1
        try:
            assert_no_lookahead(candles, decision_ts, timeframe, label)
            return True
        except LeakageError as exc:
            self.report.ok = False
            self.report.violations.append(str(exc))
            if self.strict:
                raise
            return False

    def check_event_ts(self, event_ts: int, decision_ts: int, label: str) -> bool:
        """News, calendar entries and fundamentals must predate the decision."""

        self.report.checked += 1
        if event_ts > decision_ts:
            message = (
                f"{label}: event timestamped {event_ts} is after the decision "
                f"timestamp {decision_ts}"
            )
            self.report.ok = False
            self.report.violations.append(message)
            if self.strict:
                raise LeakageError(message)
            return False
        return True

    def check_features(
        self, features: Mapping[str, Any], decision_ts: int, label: str = "features"
    ) -> bool:
        """Any feature carrying an explicit timestamp must not be in the future."""

        self.report.checked += 1
        ok = True
        for name, value in features.items():
            if not name.endswith("_ts"):
                continue
            try:
                stamp = int(value)
            except (TypeError, ValueError):
                continue
            if stamp > decision_ts:
                message = (
                    f"{label}: feature {name}={stamp} is after the decision "
                    f"timestamp {decision_ts}"
                )
                self.report.ok = False
                self.report.violations.append(message)
                ok = False
                if self.strict:
                    raise LeakageError(message)
        return ok

    def check_analysis(self, analysis: Any, decision_ts: int) -> bool:
        """Validate every timeframe on a ``SymbolAnalysis``."""

        ok = True
        for timeframe, tf_analysis in (getattr(analysis, "timeframes", {}) or {}).items():
            candles = getattr(tf_analysis, "candles", [])
            if not self.check_series(
                candles, decision_ts, timeframe, f"{analysis.symbol} {timeframe.value}"
            ):
                ok = False
        return ok


__all__ = [
    "LeakageGuard",
    "LeakageError",
    "LeakageReport",
    "assert_no_lookahead",
]
