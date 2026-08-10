"""RTM-inspired pattern engine.

RTM as taught is substantially discretionary.  This module implements only the
parts that can be stated as measurements, and it is deliberately explicit about
that: a "fresh zone" here means *no candle has traded back into the zone since
it formed*, not a chart-reading judgement.  Where a concept has no objective
definition it is simply not implemented rather than approximated with something
that would look precise and behave arbitrarily.

Implemented
-----------
Legs (Rally / Drop), Bases (consolidation), the four base patterns
(Rally-Base-Rally, Drop-Base-Drop, Rally-Base-Drop, Drop-Base-Rally),
Compression, Fresh vs Reaction zones, Departure strength, and Engulf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from app.domain import Bias, Candle, Candles


LegKind = str          # "rally" | "drop"
PatternKind = str      # "RBR" | "DBD" | "RBD" | "DBR"


@dataclass(frozen=True, slots=True)
class Leg:
    kind: LegKind
    start: int
    end: int
    start_price: float
    end_price: float
    size_atr: float
    bars: int

    @property
    def direction(self) -> int:
        return 1 if self.kind == "rally" else -1

    @property
    def speed(self) -> float:
        """ATR of travel per bar - how explosive the leg was."""

        return self.size_atr / max(self.bars, 1)


@dataclass(frozen=True, slots=True)
class Base:
    """A consolidation: a run of bars held inside a narrow band."""

    start: int
    end: int
    top: float
    bottom: float
    bars: int
    height_atr: float
    compression: float      # 0..1, how tight relative to surrounding volatility

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass(frozen=True, slots=True)
class Zone:
    """A tradable supply/demand zone produced by a base."""

    direction: int          # +1 demand (buy), -1 supply (sell)
    top: float
    bottom: float
    base: Base
    pattern: PatternKind
    departure_atr: float    # size of the leg leaving the base
    fresh: bool             # never revisited since forming
    touches: int
    confidence: float

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def distance_pct(self, price: float) -> float:
        return abs(self.mid - price) / price if price > 0 else 0.0


@dataclass(frozen=True, slots=True)
class RTMPattern:
    kind: PatternKind
    zone: Zone
    leg_in: Leg
    leg_out: Leg
    index: int
    ts: int
    confidence: float

    @property
    def direction(self) -> int:
        return self.zone.direction

    @property
    def is_continuation(self) -> bool:
        return self.kind in {"RBR", "DBD"}


@dataclass(slots=True)
class RTMAnalysis:
    legs: list[Leg]
    bases: list[Base]
    patterns: list[RTMPattern]
    zones: list[Zone]
    compression: float          # current compression reading, 0..1
    engulf_direction: int       # +1 bullish engulf on the last bar
    close: float
    atr: float

    @property
    def fresh_demand(self) -> list[Zone]:
        return [z for z in self.zones if z.direction > 0 and z.fresh]

    @property
    def fresh_supply(self) -> list[Zone]:
        return [z for z in self.zones if z.direction < 0 and z.fresh]

    def nearest_zone(self, direction: int, price: float) -> Zone | None:
        candidates = [z for z in self.zones if z.direction == direction]
        if not candidates:
            return None
        return min(candidates, key=lambda z: abs(z.mid - price))

    def active_zone(self, price: float) -> Zone | None:
        """The zone price is currently trading inside, if any."""

        inside = [z for z in self.zones if z.contains(price)]
        if not inside:
            return None
        return max(inside, key=lambda z: z.confidence)

    def bias(self) -> Bias:
        bullish = sum(z.confidence for z in self.fresh_demand[-3:])
        bearish = sum(z.confidence for z in self.fresh_supply[-3:])
        if self.engulf_direction > 0:
            bullish += 0.5
        elif self.engulf_direction < 0:
            bearish += 0.5
        if bullish >= bearish * 1.4 and bullish >= 0.8:
            return Bias.BULLISH
        if bearish >= bullish * 1.4 and bearish >= 0.8:
            return Bias.BEARISH
        if bullish >= 0.8 and bearish >= 0.8:
            return Bias.CONFLICT
        return Bias.NEUTRAL

    def score(self, direction: int) -> float:
        """0..100 - how much RTM confluence supports ``direction``."""

        score = 40.0
        zones = self.fresh_demand if direction > 0 else self.fresh_supply
        score += min(len(zones), 3) * 6.0

        active = self.active_zone(self.close)
        if active is not None:
            if active.direction == direction:
                score += 16 * active.confidence
            else:
                score -= 12 * active.confidence

        nearest = self.nearest_zone(direction, self.close)
        if nearest is not None:
            proximity = max(0.0, 1.0 - nearest.distance_pct(self.close) / 0.02)
            score += 8 * proximity * nearest.confidence

        continuation = [
            p for p in self.patterns if p.is_continuation and p.direction == direction
        ]
        score += min(len(continuation), 2) * 5.0

        if self.engulf_direction == direction:
            score += 6
        elif self.engulf_direction == -direction:
            score -= 6

        # Compression is directionless: it raises the odds of *a* move, so it
        # only adds a small amount and never picks a side.
        score += 4 * self.compression

        return max(0.0, min(100.0, score))

    def as_features(self) -> dict[str, float]:
        active = self.active_zone(self.close)
        return {
            "rtm_fresh_demand": float(len(self.fresh_demand)),
            "rtm_fresh_supply": float(len(self.fresh_supply)),
            "rtm_compression": self.compression,
            "rtm_engulf": float(self.engulf_direction),
            "rtm_in_zone": float(active.direction) if active else 0.0,
            "rtm_zone_confidence": active.confidence if active else 0.0,
            "rtm_pattern_count": float(len(self.patterns)),
            "rtm_leg_direction": float(self.legs[-1].direction) if self.legs else 0.0,
            "rtm_leg_speed": self.legs[-1].speed if self.legs else 0.0,
        }


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def detect_bases(
    candles: Candles,
    atr_values: Sequence[float | None],
    min_bars: int = 2,
    max_bars: int = 8,
    max_height_atr: float = 1.6,
) -> list[Base]:
    """Find runs of bars that stay inside a narrow band.

    Bases are grown greedily from left to right: a run is extended while it
    stays under ``max_height_atr``, then emitted if it is long enough.  Greedy
    growth makes the segmentation unique for a given series.
    """

    bases: list[Base] = []
    size = len(candles)
    i = 0

    while i < size:
        atr = _atr_at(atr_values, i)
        if atr is None or atr <= 0:
            i += 1
            continue

        top = candles[i].high
        bottom = candles[i].low
        end = i

        for j in range(i + 1, min(i + max_bars, size)):
            new_top = max(top, candles[j].high)
            new_bottom = min(bottom, candles[j].low)
            if (new_top - new_bottom) / atr > max_height_atr:
                break
            top, bottom, end = new_top, new_bottom, j

        bars = end - i + 1
        if bars >= min_bars:
            height_atr = (top - bottom) / atr
            # Tightness relative to the space a random walk of this length
            # would normally cover.
            expected = max(bars ** 0.5, 1.0)
            compression = max(0.0, min(1.0, 1.0 - height_atr / (expected * max_height_atr)))
            bases.append(
                Base(
                    start=i,
                    end=end,
                    top=top,
                    bottom=bottom,
                    bars=bars,
                    height_atr=round(height_atr, 4),
                    compression=round(compression, 4),
                )
            )
            i = end + 1
        else:
            i += 1

    return bases


def detect_legs(
    candles: Candles,
    atr_values: Sequence[float | None],
    bases: Sequence[Base],
    min_size_atr: float = 1.0,
) -> list[Leg]:
    """The moves *between* bases are the rallies and drops."""

    legs: list[Leg] = []
    if not bases:
        return legs

    boundaries: list[tuple[int, int]] = []
    for previous, current in zip(bases, bases[1:]):
        if current.start > previous.end + 1:
            boundaries.append((previous.end, current.start))
    if bases and bases[-1].end < len(candles) - 1:
        boundaries.append((bases[-1].end, len(candles) - 1))

    for start, end in boundaries:
        atr = _atr_at(atr_values, end)
        if atr is None or atr <= 0:
            continue
        start_price = candles[start].close
        end_price = candles[end].close
        move = end_price - start_price
        size_atr = abs(move) / atr
        if size_atr < min_size_atr:
            continue
        legs.append(
            Leg(
                kind="rally" if move > 0 else "drop",
                start=start,
                end=end,
                start_price=start_price,
                end_price=end_price,
                size_atr=round(size_atr, 4),
                bars=max(end - start, 1),
            )
        )
    return legs


def build_patterns(
    candles: Candles,
    bases: Sequence[Base],
    legs: Sequence[Leg],
    atr: float,
    max_patterns: int = 10,
) -> tuple[list[RTMPattern], list[Zone]]:
    """Assemble Rally/Drop-Base-Rally/Drop patterns and their zones."""

    patterns: list[RTMPattern] = []
    zones: list[Zone] = []
    size = len(candles)

    for base in bases:
        leg_in = _leg_ending_before(legs, base.start)
        leg_out = _leg_starting_after(legs, base.end)
        if leg_in is None or leg_out is None:
            continue

        kind = _pattern_kind(leg_in.kind, leg_out.kind)
        direction = 1 if leg_out.kind == "rally" else -1

        after = candles[leg_out.end + 1 :]
        touches = sum(
            1
            for c in after
            if c.low <= base.top and c.high >= base.bottom
        )
        fresh = touches == 0

        recency = base.end / max(size, 1)
        departure = leg_out.size_atr
        confidence = _clip(
            0.25
            + 0.25 * min(departure / 3.0, 1.0)
            + 0.20 * base.compression
            + 0.15 * recency
            + (0.15 if fresh else 0.0)
        )

        zone = Zone(
            direction=direction,
            top=base.top,
            bottom=base.bottom,
            base=base,
            pattern=kind,
            departure_atr=departure,
            fresh=fresh,
            touches=touches,
            confidence=round(confidence, 4),
        )
        zones.append(zone)
        patterns.append(
            RTMPattern(
                kind=kind,
                zone=zone,
                leg_in=leg_in,
                leg_out=leg_out,
                index=base.end,
                ts=candles[base.end].ts,
                confidence=round(confidence, 4),
            )
        )

    return patterns[-max_patterns:], zones[-max_patterns:]


def _pattern_kind(leg_in: LegKind, leg_out: LegKind) -> PatternKind:
    return {
        ("rally", "rally"): "RBR",
        ("drop", "drop"): "DBD",
        ("rally", "drop"): "RBD",
        ("drop", "rally"): "DBR",
    }[(leg_in, leg_out)]


def _leg_ending_before(legs: Sequence[Leg], index: int) -> Leg | None:
    candidates = [leg for leg in legs if leg.end <= index]
    return candidates[-1] if candidates else None


def _leg_starting_after(legs: Sequence[Leg], index: int) -> Leg | None:
    candidates = [leg for leg in legs if leg.start >= index]
    return candidates[0] if candidates else None


def detect_engulf(candles: Candles, atr: float, min_body_atr: float = 0.6) -> int:
    """Classic engulfing on the last closed bar.

    +1 bullish engulf, -1 bearish engulf, 0 none.
    """

    if len(candles) < 2 or atr <= 0:
        return 0
    previous, current = candles[-2], candles[-1]
    if current.body / atr < min_body_atr:
        return 0
    if (
        current.is_bullish
        and previous.is_bearish
        and current.close >= previous.open
        and current.open <= previous.close
    ):
        return 1
    if (
        current.is_bearish
        and previous.is_bullish
        and current.close <= previous.open
        and current.open >= previous.close
    ):
        return -1
    return 0


def current_compression(
    candles: Candles, atr_values: Sequence[float | None], lookback: int = 12
) -> float:
    """How tightly price is coiling right now, in ``[0, 1]``.

    Compares the recent envelope against the average true range: a range worth
    only a couple of ATRs after a dozen bars is a coil.
    """

    if len(candles) < lookback + 1:
        return 0.0
    atr = _atr_at(atr_values, len(candles) - 1)
    if atr is None or atr <= 0:
        return 0.0
    window = candles[-lookback:]
    height = max(c.high for c in window) - min(c.low for c in window)
    height_atr = height / atr
    # A random walk of ``lookback`` bars typically spans ~sqrt(lookback) ATR.
    expected = max(lookback ** 0.5, 1.0)
    return round(_clip(1.0 - height_atr / (expected * 1.2)), 4)


def _atr_at(atr_values: Sequence[float | None], index: int) -> float | None:
    if index < len(atr_values) and atr_values[index] is not None:
        return atr_values[index]
    for value in reversed(atr_values[: min(index + 1, len(atr_values))]):
        if value is not None:
            return value
    return None


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else high if value > high else value


def analyse_rtm(
    candles: Candles,
    atr_values: Sequence[float | None],
    lookback: int = 150,
) -> RTMAnalysis:
    """Full RTM read for one symbol/timeframe."""

    if not candles:
        raise ValueError("cannot analyse an empty candle series")

    window = list(candles[-lookback:])
    offset = len(candles) - len(window)
    window_atr = list(atr_values[offset:]) if offset else list(atr_values)

    atr = _atr_at(window_atr, len(window) - 1) or 0.0
    close = window[-1].close

    bases = detect_bases(window, window_atr)
    legs = detect_legs(window, window_atr, bases)
    patterns, zones = build_patterns(window, bases, legs, atr)

    return RTMAnalysis(
        legs=legs,
        bases=bases,
        patterns=patterns,
        zones=zones,
        compression=current_compression(window, window_atr),
        engulf_direction=detect_engulf(window, atr),
        close=close,
        atr=atr,
    )


__all__ = [
    "RTMAnalysis",
    "RTMPattern",
    "Base",
    "Leg",
    "Zone",
    "analyse_rtm",
    "detect_bases",
    "detect_legs",
    "build_patterns",
    "detect_engulf",
    "current_compression",
]
