"""DATA_RISK_SCORE - is the information we are about to act on trustworthy?

Bad data is the failure mode that looks exactly like a good opportunity: a
stale candle series produces a confident, well-formed, completely fictional
setup. This scores staleness, missing timeframes, validation failures, provider
health, timestamp alignment and internal contradiction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.domain import Timeframe

#: Above this the trade is blocked outright.
HARD_LIMIT = 0.70


@dataclass(slots=True)
class DataRiskReport:
    score: float = 0.0
    stale_seconds: float = 0.0
    missing_timeframes: list[str] = field(default_factory=list)
    validation_failures: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    coverage: float = 1.0

    @property
    def blocked(self) -> bool:
        return self.score >= HARD_LIMIT

    @property
    def level(self) -> str:
        if self.score >= HARD_LIMIT:
            return "CRITICAL"
        if self.score >= 0.4:
            return "ELEVATED"
        return "OK"

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "level": self.level,
            "blocked": self.blocked,
            "stale_seconds": round(self.stale_seconds, 1),
            "missing_timeframes": self.missing_timeframes,
            "validation_failures": self.validation_failures,
            "coverage": round(self.coverage, 4),
            "problems": self.problems,
            "notes": self.notes,
        }


def assess_data_risk(
    analysis: Any,
    required_timeframes: Sequence[Timeframe] = (Timeframe.H1, Timeframe.H4, Timeframe.D1),
    market_data_health: Mapping[str, Any] | None = None,
    now: int | None = None,
    max_stale_bars: float = 3.0,
) -> DataRiskReport:
    """Score how much the data itself is a risk."""

    report = DataRiskReport()
    now = int(now if now is not None else time.time())
    penalties: list[float] = []

    if analysis is None:
        report.score = 1.0
        report.problems.append("no analysis available")
        return report

    timeframes = getattr(analysis, "timeframes", {}) or {}

    # --- missing context -------------------------------------------------
    missing = [tf.value for tf in required_timeframes if tf not in timeframes]
    report.missing_timeframes = missing
    if missing:
        share = len(missing) / max(len(required_timeframes), 1)
        penalties.append(0.55 * share)
        report.problems.append(f"missing context timeframes: {', '.join(missing)}")

    execution = getattr(analysis, "primary", None)
    if execution is None:
        report.score = 1.0
        report.problems.append("no execution timeframe")
        return report

    # --- staleness --------------------------------------------------------
    candles = getattr(execution, "candles", []) or []
    if candles:
        timeframe = getattr(execution, "timeframe", Timeframe.M15)
        step = timeframe.seconds
        age = now - (candles[-1].ts + step)
        report.stale_seconds = max(age, 0)
        bars_behind = age / step if step else 0
        if bars_behind > max_stale_bars:
            penalties.append(min(0.9, 0.3 + 0.15 * (bars_behind - max_stale_bars)))
            report.problems.append(
                f"execution data is {bars_behind:.1f} bars stale"
            )
        elif bars_behind > 1.5:
            penalties.append(0.15)
            report.notes.append(f"data {bars_behind:.1f} bars behind")
    else:
        report.score = 1.0
        report.problems.append("execution timeframe has no candles")
        return report

    # --- per-timeframe history depth --------------------------------------
    thin = [
        tf.value
        for tf, tf_analysis in timeframes.items()
        if len(getattr(tf_analysis, "candles", [])) < 100
    ]
    if thin:
        penalties.append(min(0.3, 0.08 * len(thin)))
        report.notes.append(f"short history on {', '.join(sorted(thin))}")

    # --- analysis-reported errors -----------------------------------------
    errors = list(getattr(analysis, "errors", []) or [])
    if errors:
        report.validation_failures = errors[:5]
        penalties.append(min(0.4, 0.1 * len(errors)))
        report.problems.append(f"{len(errors)} analysis warning(s)")

    # --- price anomaly ------------------------------------------------------
    anomaly = getattr(analysis, "anomaly", "")
    if anomaly:
        penalties.append(0.85)
        report.problems.append(f"price anomaly: {anomaly}")

    # --- provider health ----------------------------------------------------
    if market_data_health:
        failures = float(market_data_health.get("failures", 0) or 0)
        fetches = max(float(market_data_health.get("fetches", 1) or 1), 1.0)
        error_rate = failures / fetches
        if error_rate > 0.25:
            penalties.append(min(0.6, error_rate))
            report.problems.append(f"{error_rate:.0%} of data fetches failed")
        elif error_rate > 0.1:
            penalties.append(0.15)
            report.notes.append(f"data fetch error rate {error_rate:.0%}")

        validation_failures = float(
            market_data_health.get("validation_failures", 0) or 0
        )
        if validation_failures > 0:
            penalties.append(min(0.3, 0.05 * validation_failures))

    # --- fundamental coverage ------------------------------------------------
    fundamentals = getattr(analysis, "fundamentals", None)
    if fundamentals is not None:
        report.coverage = getattr(fundamentals, "coverage", 1.0)
        if report.coverage < 0.3:
            # Missing macro is normal without providers, so this is a soft note,
            # never a block - UNKNOWN is a valid state, not a data fault.
            report.notes.append(
                f"fundamental coverage only {report.coverage:.0%} (providers not configured)"
            )

    report.score = _noisy_or(penalties)
    return report


def _noisy_or(penalties: Sequence[float]) -> float:
    """Independent problems accumulate but never exceed 1."""

    survival = 1.0
    for penalty in penalties:
        survival *= 1.0 - max(0.0, min(penalty, 1.0))
    return 1.0 - survival


__all__ = ["DataRiskReport", "assess_data_risk", "HARD_LIMIT"]
