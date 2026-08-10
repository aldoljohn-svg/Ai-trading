"""Swing (fractal) detection.

A swing high at index ``i`` is a bar whose high is strictly greater than the
highs of the ``strength`` bars on each side.  Strictness on both sides makes
the result unambiguous: no two adjacent bars can both be swing highs, and there
is no tie-breaking rule to get wrong.

**No lookahead**: a swing at index ``i`` cannot be known until ``i + strength``
bars exist.  :func:`detect_swings` records that as ``confirmed_at`` and, by
default, only returns swings that are already confirmed at the end of the
series.  Every consumer in this project relies on that guarantee - it is what
keeps the backtester honest.

An optional ATR filter removes swings whose excursion from the previous
opposite swing is insignificant, which prevents micro-noise from being treated
as structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from app.domain import Candle, Candles


class SwingType(str, Enum):
    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class Swing:
    index: int
    ts: int
    price: float
    type: SwingType
    strength: int
    confirmed_at: int      # index at which this swing became knowable

    @property
    def is_high(self) -> bool:
        return self.type is SwingType.HIGH

    @property
    def is_low(self) -> bool:
        return self.type is SwingType.LOW


def detect_swings(
    candles: Candles,
    strength: int = 3,
    confirmed_only: bool = True,
    atr_values: Sequence[float | None] | None = None,
    min_atr_excursion: float = 0.0,
) -> list[Swing]:
    """Return swings ordered by index.

    Parameters
    ----------
    strength:
        Number of bars that must be lower (for a high) on each side.
    confirmed_only:
        Drop swings whose confirmation bar does not exist yet.
    atr_values / min_atr_excursion:
        When both are supplied, alternating swings whose price excursion is
        smaller than ``min_atr_excursion * ATR`` are discarded as noise.
    """

    size = len(candles)
    if strength < 1 or size < strength * 2 + 1:
        return []

    swings: list[Swing] = []
    last_confirmable = size - 1

    for i in range(strength, size - strength):
        candle = candles[i]
        left = candles[i - strength : i]
        right = candles[i + 1 : i + 1 + strength]

        if all(candle.high > c.high for c in left) and all(
            candle.high > c.high for c in right
        ):
            confirmed_at = i + strength
            if not confirmed_only or confirmed_at <= last_confirmable:
                swings.append(
                    Swing(
                        index=i,
                        ts=candle.ts,
                        price=candle.high,
                        type=SwingType.HIGH,
                        strength=strength,
                        confirmed_at=confirmed_at,
                    )
                )

        if all(candle.low < c.low for c in left) and all(
            candle.low < c.low for c in right
        ):
            confirmed_at = i + strength
            if not confirmed_only or confirmed_at <= last_confirmable:
                swings.append(
                    Swing(
                        index=i,
                        ts=candle.ts,
                        price=candle.low,
                        type=SwingType.LOW,
                        strength=strength,
                        confirmed_at=confirmed_at,
                    )
                )

    swings.sort(key=lambda s: (s.index, 0 if s.is_high else 1))

    if atr_values is not None and min_atr_excursion > 0:
        swings = _filter_by_excursion(swings, atr_values, min_atr_excursion)

    return swings


def _filter_by_excursion(
    swings: list[Swing],
    atr_values: Sequence[float | None],
    min_atr_excursion: float,
) -> list[Swing]:
    """Drop swings that do not move far enough from the previous opposite swing."""

    kept: list[Swing] = []
    for swing in swings:
        previous_opposite = None
        for candidate in reversed(kept):
            if candidate.type is not swing.type:
                previous_opposite = candidate
                break
        if previous_opposite is None:
            kept.append(swing)
            continue
        atr = _atr_at(atr_values, swing.index)
        if atr is None or atr <= 0:
            kept.append(swing)
            continue
        if abs(swing.price - previous_opposite.price) >= min_atr_excursion * atr:
            kept.append(swing)
    return kept


def _atr_at(atr_values: Sequence[float | None], index: int) -> float | None:
    if index < len(atr_values) and atr_values[index] is not None:
        return atr_values[index]
    for value in reversed(atr_values[: index + 1]):
        if value is not None:
            return value
    return None


def alternate(swings: Sequence[Swing]) -> list[Swing]:
    """Collapse consecutive same-type swings, keeping the most extreme one.

    Structure logic (HH/HL, BOS, CHOCH) is only meaningful on an alternating
    high/low sequence.
    """

    out: list[Swing] = []
    for swing in swings:
        if out and out[-1].type is swing.type:
            previous = out[-1]
            better = (
                swing.price > previous.price
                if swing.is_high
                else swing.price < previous.price
            )
            if better:
                out[-1] = swing
            continue
        out.append(swing)
    return out


def last_swing(swings: Sequence[Swing], swing_type: SwingType) -> Swing | None:
    for swing in reversed(swings):
        if swing.type is swing_type:
            return swing
    return None


def swing_highs(swings: Sequence[Swing]) -> list[Swing]:
    return [s for s in swings if s.is_high]


def swing_lows(swings: Sequence[Swing]) -> list[Swing]:
    return [s for s in swings if s.is_low]


__all__ = [
    "Swing",
    "SwingType",
    "detect_swings",
    "alternate",
    "last_swing",
    "swing_highs",
    "swing_lows",
]
