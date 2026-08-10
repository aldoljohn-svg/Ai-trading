"""Market-data integrity checks.

Corrupt or stale candles are one of the few failure modes that can silently
destroy an account: a single bad print moves ATR, invalidates a stop distance
and can size a position ten times too large.  Every candle series is therefore
validated before it reaches an analytical engine, and a series that fails hard
checks is refused rather than "cleaned up".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.domain import Candle, Timeframe


@dataclass(slots=True)
class CandleValidation:
    ok: bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gaps: int = 0
    duplicates: int = 0
    age_seconds: int = 0

    def __bool__(self) -> bool:
        return self.ok


def validate_candles(
    candles: list[Candle],
    timeframe: Timeframe,
    min_length: int = 60,
    now: int | None = None,
    max_staleness_bars: float = 3.0,
    max_gap_ratio: float = 0.02,
) -> CandleValidation:
    """Validate a closed-candle series.

    Hard failures (``ok=False``):
      * fewer than ``min_length`` bars
      * non-positive or non-finite prices
      * ``high < low`` or a close outside ``[low, high]``
      * unsorted or duplicated timestamps
      * timestamps not aligned to the timeframe grid
      * data staler than ``max_staleness_bars`` bars
      * more than ``max_gap_ratio`` of bars missing
    """

    problems: list[str] = []
    warnings: list[str] = []
    now = int(now if now is not None else time.time())

    if not candles:
        return CandleValidation(ok=False, problems=["empty candle series"])

    if len(candles) < min_length:
        problems.append(
            f"only {len(candles)} bars, need at least {min_length}"
        )

    step = timeframe.seconds
    duplicates = 0
    gaps = 0
    previous: Candle | None = None

    for candle in candles:
        if not _finite(candle.open, candle.high, candle.low, candle.close):
            problems.append(f"non-finite price at ts={candle.ts}")
            break
        if min(candle.open, candle.high, candle.low, candle.close) <= 0:
            problems.append(f"non-positive price at ts={candle.ts}")
            break
        if candle.high < candle.low:
            problems.append(f"high < low at ts={candle.ts}")
            break
        if not (candle.low - 1e-12 <= candle.close <= candle.high + 1e-12):
            problems.append(f"close outside [low, high] at ts={candle.ts}")
            break
        if not (candle.low - 1e-12 <= candle.open <= candle.high + 1e-12):
            problems.append(f"open outside [low, high] at ts={candle.ts}")
            break
        if candle.volume < 0:
            problems.append(f"negative volume at ts={candle.ts}")
            break
        if candle.ts % step != 0:
            problems.append(
                f"timestamp {candle.ts} is not aligned to the {timeframe.value} grid"
            )
            break

        if previous is not None:
            delta = candle.ts - previous.ts
            if delta <= 0:
                duplicates += 1
                problems.append(
                    f"timestamps not strictly increasing at ts={candle.ts}"
                )
                break
            if delta != step:
                missing = delta // step - 1
                if missing > 0:
                    gaps += int(missing)
        previous = candle

    if previous is not None:
        # ``ts`` is the bar open; the bar closed one step later.
        age = now - (previous.ts + step)
        if age > max_staleness_bars * step:
            problems.append(
                f"stale data: newest bar closed {age}s ago "
                f"({age / step:.1f} bars behind)"
            )
    else:
        age = 0

    expected = len(candles) + gaps
    if expected > 0 and gaps / expected > max_gap_ratio:
        problems.append(
            f"{gaps} missing bars out of {expected} exceeds the "
            f"{max_gap_ratio:.1%} tolerance"
        )
    elif gaps:
        warnings.append(f"{gaps} missing bars tolerated")

    flat = sum(1 for c in candles if c.high == c.low)
    if flat > len(candles) * 0.2:
        warnings.append(f"{flat} zero-range bars - the symbol may be illiquid")

    return CandleValidation(
        ok=not problems,
        problems=problems,
        warnings=warnings,
        gaps=gaps,
        duplicates=duplicates,
        age_seconds=max(age, 0),
    )


def _finite(*values: float) -> bool:
    for value in values:
        if value != value or value in (float("inf"), float("-inf")):
            return False
    return True


def validate_ticker_spread(bid: float, ask: float, max_spread_pct: float) -> tuple[bool, str]:
    if bid <= 0 or ask <= 0:
        return False, "missing bid/ask"
    if ask < bid:
        return False, "crossed book (ask < bid)"
    mid = (bid + ask) / 2
    spread = (ask - bid) / mid
    if spread > max_spread_pct:
        return False, f"spread {spread:.4%} exceeds limit {max_spread_pct:.4%}"
    return True, ""


def detect_price_anomaly(
    candles: list[Candle], sigma_threshold: float = 8.0, lookback: int = 100
) -> tuple[bool, str]:
    """Flag an implausible single-bar move relative to recent behaviour.

    Used by the capital-protection layer: a genuine flash move and a corrupt
    print look the same from one bar, so both suspend new entries.
    """

    window = candles[-(lookback + 1) :]
    if len(window) < 30:
        return False, ""
    returns = []
    for previous, current in zip(window, window[1:]):
        if previous.close > 0:
            returns.append((current.close - previous.close) / previous.close)
    if len(returns) < 20:
        return False, ""
    body = returns[:-1]
    mean = sum(body) / len(body)
    variance = sum((r - mean) ** 2 for r in body) / max(len(body) - 1, 1)
    std = variance ** 0.5
    if std <= 0:
        return False, ""
    latest = returns[-1]
    z = abs(latest - mean) / std
    if z > sigma_threshold:
        return True, (
            f"last bar moved {latest:+.2%}, {z:.1f} sigma versus its own "
            "recent distribution"
        )
    return False, ""


__all__ = [
    "CandleValidation",
    "validate_candles",
    "validate_ticker_spread",
    "detect_price_anomaly",
]
