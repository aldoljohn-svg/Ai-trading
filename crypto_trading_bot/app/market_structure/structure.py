"""Price-action / market-structure engine.

Everything here is deterministic and backtestable: the same candles always
produce the same labels, and no rule reads a bar that would not have existed at
decision time.

Vocabulary implemented
----------------------
``HH`` / ``HL`` / ``LH`` / ``LL``
    Swing labels relative to the previous swing of the same type.
``BOS`` (Break of Structure)
    A close beyond the most recent unbroken swing **in the direction of the
    prevailing structure** - continuation.
``CHOCH`` (Change of Character)
    The first close beyond the most recent unbroken swing **against** the
    prevailing structure - the structure flips.
``MSS`` (Market Structure Shift)
    A CHOCH that arrives with displacement (an outsized, decisive candle).
    Every MSS is also a CHOCH; not every CHOCH is an MSS.
Support / Resistance
    Clusters of swing prices, scored by touch count and recency.
Range / Breakout / Fakeout / Retest / Rejection / Impulse / Correction
    Measured from the range envelope and the most recent structural break.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from app.domain import Bias, Candle, Candles
from app.indicators.slopes import r_squared
from app.market_structure.swing_detector import (
    Swing,
    SwingType,
    alternate,
    detect_swings,
)


class SwingLabel(str, Enum):
    HH = "HH"
    HL = "HL"
    LH = "LH"
    LL = "LL"
    UNKNOWN = "UNKNOWN"


class StructureEventType(str, Enum):
    BOS = "BOS"
    CHOCH = "CHOCH"
    MSS = "MSS"


@dataclass(frozen=True, slots=True)
class StructureEvent:
    type: StructureEventType
    direction: int              # +1 bullish, -1 bearish
    index: int                  # candle index where the break closed
    ts: int
    level: float                # the swing price that was broken
    close: float
    displacement_atr: float     # size of the breaking candle body in ATR
    confidence: float           # 0..1

    @property
    def is_bullish(self) -> bool:
        return self.direction > 0


@dataclass(frozen=True, slots=True)
class Level:
    """A clustered support or resistance level."""

    price: float
    touches: int
    last_index: int
    kind: str                   # "support" | "resistance"
    strength: float             # 0..1

    def distance_pct(self, price: float) -> float:
        return abs(self.price - price) / price if price > 0 else 0.0


@dataclass(slots=True)
class MarketStructure:
    swings: list[Swing]
    labels: list[tuple[Swing, SwingLabel]]
    events: list[StructureEvent]
    bias: Bias
    support: list[Level]
    resistance: list[Level]
    in_range: bool
    range_high: float
    range_low: float
    range_position: float           # 0 at range low, 1 at range high
    breakout_direction: int         # +1 up, -1 down, 0 none
    fakeout_direction: int          # +1 failed up-break, -1 failed down-break
    retest_direction: int           # +1 bullish retest held, -1 bearish
    rejection_direction: int        # +1 rejection of lows, -1 rejection of highs
    leg_kind: str                   # "impulse" | "correction" | "unknown"
    leg_direction: int
    leg_atr: float                  # size of the current leg in ATR
    trend_quality: float            # r^2 of closes over the lookback
    close: float
    atr: float

    # -- derived ----------------------------------------------------------

    @property
    def last_event(self) -> StructureEvent | None:
        return self.events[-1] if self.events else None

    @property
    def last_bos(self) -> StructureEvent | None:
        return _last_of(self.events, StructureEventType.BOS)

    @property
    def last_choch(self) -> StructureEvent | None:
        return _last_of(self.events, StructureEventType.CHOCH)

    @property
    def last_mss(self) -> StructureEvent | None:
        return _last_of(self.events, StructureEventType.MSS)

    @property
    def label_sequence(self) -> list[str]:
        return [label.value for _swing, label in self.labels]

    def nearest_level(self, price: float, kind: str) -> Level | None:
        levels = self.support if kind == "support" else self.resistance
        if not levels:
            return None
        if kind == "support":
            below = [lv for lv in levels if lv.price < price]
            return max(below, key=lambda lv: lv.price) if below else None
        above = [lv for lv in levels if lv.price > price]
        return min(above, key=lambda lv: lv.price) if above else None

    def score(self) -> float:
        """0..100 structural clarity score used by the ranking engine."""

        score = 50.0
        if self.bias is Bias.BULLISH:
            score += 12
        elif self.bias is Bias.BEARISH:
            score += 12
        elif self.bias is Bias.CONFLICT:
            score -= 12

        score += 20.0 * self.trend_quality

        event = self.last_event
        if event is not None:
            weight = {
                StructureEventType.MSS: 12.0,
                StructureEventType.CHOCH: 8.0,
                StructureEventType.BOS: 6.0,
            }[event.type]
            score += weight * event.confidence

        if self.leg_kind == "impulse":
            score += 6
        if self.fakeout_direction:
            score += 4
        if self.retest_direction:
            score += 6
        if self.in_range:
            score -= 8
        return max(0.0, min(100.0, score))

    def as_features(self) -> dict[str, float]:
        event = self.last_event
        return {
            "structure_bias": _bias_value(self.bias),
            "trend_quality": self.trend_quality,
            "in_range": 1.0 if self.in_range else 0.0,
            "range_position": self.range_position,
            "breakout_direction": float(self.breakout_direction),
            "fakeout_direction": float(self.fakeout_direction),
            "retest_direction": float(self.retest_direction),
            "rejection_direction": float(self.rejection_direction),
            "leg_direction": float(self.leg_direction),
            "leg_atr": self.leg_atr,
            "is_impulse": 1.0 if self.leg_kind == "impulse" else 0.0,
            "last_event_direction": float(event.direction) if event else 0.0,
            "last_event_is_mss": (
                1.0 if event and event.type is StructureEventType.MSS else 0.0
            ),
            "last_event_displacement": event.displacement_atr if event else 0.0,
            "swing_count": float(len(self.swings)),
        }


def _last_of(
    events: Sequence[StructureEvent], kind: StructureEventType
) -> StructureEvent | None:
    for event in reversed(events):
        if event.type is kind:
            return event
    return None


def _bias_value(bias: Bias) -> float:
    return {
        Bias.BULLISH: 1.0,
        Bias.BEARISH: -1.0,
        Bias.NEUTRAL: 0.0,
        Bias.CONFLICT: 0.0,
    }[bias]


# --------------------------------------------------------------------------
# Labelling
# --------------------------------------------------------------------------


def label_swings(swings: Sequence[Swing]) -> list[tuple[Swing, SwingLabel]]:
    """Label each swing HH/LH (highs) or HL/LL (lows)."""

    labels: list[tuple[Swing, SwingLabel]] = []
    previous_high: Swing | None = None
    previous_low: Swing | None = None
    for swing in swings:
        if swing.is_high:
            if previous_high is None:
                label = SwingLabel.UNKNOWN
            else:
                label = SwingLabel.HH if swing.price > previous_high.price else SwingLabel.LH
            previous_high = swing
        else:
            if previous_low is None:
                label = SwingLabel.UNKNOWN
            else:
                label = SwingLabel.HL if swing.price > previous_low.price else SwingLabel.LL
            previous_low = swing
        labels.append((swing, label))
    return labels


def bias_from_labels(labels: Sequence[tuple[Swing, SwingLabel]], window: int = 4) -> Bias:
    """Derive structural bias from the most recent swing labels."""

    recent = [label for _swing, label in labels if label is not SwingLabel.UNKNOWN][-window:]
    if len(recent) < 2:
        return Bias.NEUTRAL
    bullish = sum(1 for label in recent if label in (SwingLabel.HH, SwingLabel.HL))
    bearish = sum(1 for label in recent if label in (SwingLabel.LH, SwingLabel.LL))
    if bullish == len(recent):
        return Bias.BULLISH
    if bearish == len(recent):
        return Bias.BEARISH
    if bullish and bearish:
        # A mixed sequence with both a higher high and a lower low is a genuine
        # conflict, not merely a neutral drift.
        has_hh = any(label is SwingLabel.HH for label in recent)
        has_ll = any(label is SwingLabel.LL for label in recent)
        if has_hh and has_ll:
            return Bias.CONFLICT
    if bullish > bearish:
        return Bias.BULLISH
    if bearish > bullish:
        return Bias.BEARISH
    return Bias.NEUTRAL


# --------------------------------------------------------------------------
# Structural events
# --------------------------------------------------------------------------


def detect_events(
    candles: Candles,
    swings: Sequence[Swing],
    atr_values: Sequence[float | None],
    displacement_atr: float = 1.2,
) -> list[StructureEvent]:
    """Walk the series and emit BOS / CHOCH / MSS events.

    The state machine tracks the most recent *unbroken* swing high and low that
    were already confirmed at the time of the break, so an event is never
    generated from information that arrived later.
    """

    events: list[StructureEvent] = []
    if not candles or not swings:
        return events

    state = 0                      # prevailing structure: +1 bullish, -1 bearish
    pending_high: Swing | None = None
    pending_low: Swing | None = None
    swing_pos = 0

    for index, candle in enumerate(candles):
        # Promote every swing that became confirmed at or before this bar.
        while swing_pos < len(swings) and swings[swing_pos].confirmed_at <= index:
            swing = swings[swing_pos]
            if swing.is_high:
                if pending_high is None or swing.price > pending_high.price:
                    pending_high = swing
                elif pending_high.index < swing.index:
                    pending_high = swing
            else:
                if pending_low is None or swing.price < pending_low.price:
                    pending_low = swing
                elif pending_low.index < swing.index:
                    pending_low = swing
            swing_pos += 1

        atr = _atr_at(atr_values, index)
        if atr is None or atr <= 0:
            continue

        body_atr = candle.body / atr

        if pending_high is not None and candle.close > pending_high.price:
            direction = 1
            if state == -1:
                kind = (
                    StructureEventType.MSS
                    if body_atr >= displacement_atr
                    else StructureEventType.CHOCH
                )
            else:
                kind = StructureEventType.BOS
            events.append(
                StructureEvent(
                    type=kind,
                    direction=direction,
                    index=index,
                    ts=candle.ts,
                    level=pending_high.price,
                    close=candle.close,
                    displacement_atr=round(body_atr, 4),
                    confidence=_event_confidence(
                        body_atr, candle, direction, displacement_atr
                    ),
                )
            )
            state = 1
            pending_high = None

        elif pending_low is not None and candle.close < pending_low.price:
            direction = -1
            if state == 1:
                kind = (
                    StructureEventType.MSS
                    if body_atr >= displacement_atr
                    else StructureEventType.CHOCH
                )
            else:
                kind = StructureEventType.BOS
            events.append(
                StructureEvent(
                    type=kind,
                    direction=direction,
                    index=index,
                    ts=candle.ts,
                    level=pending_low.price,
                    close=candle.close,
                    displacement_atr=round(body_atr, 4),
                    confidence=_event_confidence(
                        body_atr, candle, direction, displacement_atr
                    ),
                )
            )
            state = -1
            pending_low = None

    return events


def _event_confidence(
    body_atr: float, candle: Candle, direction: int, displacement_atr: float
) -> float:
    """0..1 - how decisive the break was."""

    size_term = min(body_atr / max(displacement_atr, 0.1), 1.5) / 1.5
    span = candle.range
    if span <= 0:
        close_term = 0.5
    else:
        location = (candle.close - candle.low) / span
        close_term = location if direction > 0 else 1.0 - location
    body_ratio = candle.body / span if span > 0 else 0.0
    return round(max(0.0, min(1.0, 0.5 * size_term + 0.3 * close_term + 0.2 * body_ratio)), 4)


def _atr_at(atr_values: Sequence[float | None], index: int) -> float | None:
    if index < len(atr_values) and atr_values[index] is not None:
        return atr_values[index]
    for value in reversed(atr_values[: min(index + 1, len(atr_values))]):
        if value is not None:
            return value
    return None


# --------------------------------------------------------------------------
# Levels
# --------------------------------------------------------------------------


def cluster_levels(
    swings: Sequence[Swing],
    kind: str,
    tolerance: float,
    total_bars: int,
    max_levels: int = 6,
) -> list[Level]:
    """Cluster swing prices into support/resistance levels."""

    wanted = SwingType.HIGH if kind == "resistance" else SwingType.LOW
    points = [s for s in swings if s.type is wanted]
    if not points or tolerance <= 0:
        return []

    clusters: list[list[Swing]] = []
    for swing in sorted(points, key=lambda s: s.price):
        if clusters and abs(swing.price - clusters[-1][-1].price) <= tolerance:
            clusters[-1].append(swing)
        else:
            clusters.append([swing])

    levels: list[Level] = []
    for cluster in clusters:
        price = sum(s.price for s in cluster) / len(cluster)
        last_index = max(s.index for s in cluster)
        recency = last_index / total_bars if total_bars else 0.0
        touch_term = min(len(cluster) / 4.0, 1.0)
        levels.append(
            Level(
                price=price,
                touches=len(cluster),
                last_index=last_index,
                kind=kind,
                strength=round(0.65 * touch_term + 0.35 * recency, 4),
            )
        )
    levels.sort(key=lambda lv: lv.strength, reverse=True)
    return levels[:max_levels]


# --------------------------------------------------------------------------
# Range / breakout / fakeout / retest / rejection / legs
# --------------------------------------------------------------------------


def _range_envelope(candles: Candles, lookback: int) -> tuple[float, float]:
    window = candles[-lookback:]
    return max(c.high for c in window), min(c.low for c in window)


def detect_fakeout(
    candles: Candles, range_high: float, range_low: float, window: int = 12
) -> int:
    """+1 = failed *upside* break (bearish), -1 = failed downside break (bullish).

    A fakeout is a bar that traded beyond the envelope but closed back inside,
    with price still inside the envelope now.
    """

    if len(candles) < window + 1:
        return 0
    recent = candles[-window:]
    last_close = candles[-1].close

    for candle in reversed(recent):
        if candle.high > range_high and candle.close < range_high:
            if last_close < range_high:
                return 1
            return 0
        if candle.low < range_low and candle.close > range_low:
            if last_close > range_low:
                return -1
            return 0
    return 0


def detect_retest(
    candles: Candles,
    event: StructureEvent | None,
    atr: float,
    window: int = 15,
) -> int:
    """Did price return to the broken level and hold it?

    Returns +1 for a held bullish retest, -1 for a held bearish retest, else 0.
    """

    if event is None or atr <= 0:
        return 0
    tail = candles[event.index + 1 :][:window]
    if len(tail) < 2:
        return 0
    tolerance = 0.5 * atr
    touched = False
    for candle in tail:
        if event.is_bullish:
            if candle.low <= event.level + tolerance:
                touched = True
            elif touched and candle.close > event.level + tolerance:
                return 1
        else:
            if candle.high >= event.level - tolerance:
                touched = True
            elif touched and candle.close < event.level - tolerance:
                return -1
    # A retest that is currently holding also counts.
    last = candles[-1]
    if touched and event.is_bullish and last.close > event.level:
        return 1
    if touched and not event.is_bullish and last.close < event.level:
        return -1
    return 0


def detect_rejection(candles: Candles, atr: float, wick_ratio: float = 2.0) -> int:
    """Wick rejection on the most recent closed bar.

    +1 = lower wick rejected (bullish), -1 = upper wick rejected (bearish).
    """

    if not candles or atr <= 0:
        return 0
    candle = candles[-1]
    body = max(candle.body, atr * 0.05)
    if candle.lower_wick >= wick_ratio * body and candle.lower_wick >= 0.4 * atr:
        return 1
    if candle.upper_wick >= wick_ratio * body and candle.upper_wick >= 0.4 * atr:
        return -1
    return 0


def classify_leg(
    candles: Candles, swings: Sequence[Swing], atr: float
) -> tuple[str, int, float]:
    """Classify the move since the last swing as impulse or correction.

    Returns ``(kind, direction, size_in_atr)``.  An impulse covers ground
    quickly with committed bodies; a correction drifts.
    """

    if not candles or atr <= 0 or not swings:
        return "unknown", 0, 0.0
    origin = swings[-1]
    tail = candles[origin.index :]
    if len(tail) < 2:
        return "unknown", 0, 0.0

    move = candles[-1].close - origin.price
    size_atr = abs(move) / atr
    direction = 1 if move > 0 else (-1 if move < 0 else 0)
    bars = len(tail)
    speed = size_atr / bars

    total_range = sum(c.range for c in tail) or 1e-12
    efficiency = abs(move) / total_range     # 1.0 = straight line

    if size_atr >= 2.0 and speed >= 0.28 and efficiency >= 0.35:
        return "impulse", direction, round(size_atr, 3)
    return "correction", direction, round(size_atr, 3)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def analyse_structure(
    candles: Candles,
    atr_values: Sequence[float | None],
    swing_strength: int = 3,
    range_lookback: int = 60,
    displacement_atr: float = 1.2,
) -> MarketStructure:
    """Full structural read for one symbol/timeframe."""

    if not candles:
        raise ValueError("cannot analyse an empty candle series")

    atr = _atr_at(atr_values, len(candles) - 1) or 0.0
    close = candles[-1].close

    raw_swings = detect_swings(
        candles,
        strength=swing_strength,
        confirmed_only=True,
        atr_values=atr_values,
        min_atr_excursion=0.6,
    )
    swings = alternate(raw_swings)
    labels = label_swings(swings)
    bias = bias_from_labels(labels)
    events = detect_events(candles, swings, atr_values, displacement_atr)

    tolerance = max(atr * 0.5, close * 0.0015) if close > 0 else atr * 0.5
    support = cluster_levels(swings, "support", tolerance, len(candles))
    resistance = cluster_levels(swings, "resistance", tolerance, len(candles))

    lookback = min(range_lookback, len(candles))
    range_high, range_low = _range_envelope(candles, lookback)
    width = range_high - range_low
    range_position = (close - range_low) / width if width > 0 else 0.5

    window_closes = [c.close for c in candles[-lookback:]]
    trend_quality = r_squared(window_closes) if len(window_closes) >= 3 else 0.0

    band = width * 0.15
    touches_high = sum(1 for c in candles[-lookback:] if c.high >= range_high - band)
    touches_low = sum(1 for c in candles[-lookback:] if c.low <= range_low + band)
    width_atr = width / atr if atr > 0 else 0.0
    in_range = (
        touches_high >= 2
        and touches_low >= 2
        and trend_quality < 0.35
        and 0 < width_atr < 14
    )

    # Breakout is measured against the envelope *excluding* the current bar so
    # that "we just broke out" is a statement about new information.
    prior_high, prior_low = _range_envelope(candles[:-1], min(lookback, len(candles) - 1)) if len(candles) > 1 else (range_high, range_low)
    breakout_direction = 0
    if close > prior_high:
        breakout_direction = 1
    elif close < prior_low:
        breakout_direction = -1

    fakeout_direction = detect_fakeout(candles, prior_high, prior_low)
    retest_direction = detect_retest(candles, events[-1] if events else None, atr)
    rejection_direction = detect_rejection(candles, atr)
    leg_kind, leg_direction, leg_atr = classify_leg(candles, swings, atr)

    return MarketStructure(
        swings=swings,
        labels=labels,
        events=events,
        bias=bias,
        support=support,
        resistance=resistance,
        in_range=in_range,
        range_high=range_high,
        range_low=range_low,
        range_position=round(max(0.0, min(1.0, range_position)), 4),
        breakout_direction=breakout_direction,
        fakeout_direction=fakeout_direction,
        retest_direction=retest_direction,
        rejection_direction=rejection_direction,
        leg_kind=leg_kind,
        leg_direction=leg_direction,
        leg_atr=leg_atr,
        trend_quality=round(trend_quality, 4),
        close=close,
        atr=atr,
    )


__all__ = [
    "MarketStructure",
    "StructureEvent",
    "StructureEventType",
    "SwingLabel",
    "Level",
    "analyse_structure",
    "label_swings",
    "bias_from_labels",
    "detect_events",
    "cluster_levels",
    "detect_fakeout",
    "detect_retest",
    "detect_rejection",
    "classify_leg",
]
