"""ICT-inspired concepts, expressed as deterministic algorithms.

These are *algorithmic interpretations* of the published concepts, not the
discretionary judgement a human applies to a chart.  Each detection carries an
explicit ``confidence`` in ``[0, 1]`` derived from measurable properties (size
relative to ATR, how cleanly the level was respected, recency), so downstream
scoring can weight a marginal pattern differently from a textbook one.

Implemented
-----------
Fair Value Gap / Imbalance, Order Block, Breaker Block, Buy-side & Sell-side
liquidity, Equal Highs / Equal Lows, Liquidity Sweep, Displacement,
Premium / Discount / Equilibrium, and MSS (delegated to the market-structure
engine so there is exactly one definition of it in the code base).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from app.domain import Bias, Candle, Candles
from app.market_structure.structure import (
    MarketStructure,
    StructureEvent,
    StructureEventType,
)
from app.market_structure.swing_detector import Swing


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FairValueGap:
    """Three-candle imbalance: candle 1's wick never overlaps candle 3's wick."""

    direction: int          # +1 bullish gap (support), -1 bearish gap (resistance)
    top: float
    bottom: float
    index: int              # index of the third candle
    ts: int
    size_atr: float
    filled_pct: float       # 0 = untouched, 1 = fully traded through
    confidence: float

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def is_fresh(self) -> bool:
        return self.filled_pct < 0.5

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass(frozen=True, slots=True)
class OrderBlock:
    """Last opposing candle before a displacement that broke structure."""

    direction: int          # +1 bullish OB (demand), -1 bearish OB (supply)
    top: float
    bottom: float
    index: int
    ts: int
    displacement_atr: float
    mitigated: bool         # price has traded back into it
    broken: bool            # price closed decisively through it
    confidence: float

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def is_breaker(self) -> bool:
        """A broken order block flips polarity and becomes a breaker block."""

        return self.broken

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass(frozen=True, slots=True)
class LiquidityPool:
    """Resting liquidity above a high (buy side) or below a low (sell side)."""

    kind: str               # "buy_side" | "sell_side"
    price: float
    index: int
    ts: int
    touches: int            # 2+ means equal highs / equal lows
    equal: bool
    swept: bool
    confidence: float

    @property
    def direction(self) -> int:
        """Direction price must travel to take this liquidity."""

        return 1 if self.kind == "buy_side" else -1


@dataclass(frozen=True, slots=True)
class LiquiditySweep:
    """A wick beyond a liquidity pool followed by a close back inside."""

    direction: int          # +1 swept lows then reversed up (bullish)
    level: float
    index: int
    ts: int
    penetration_atr: float
    rejection_ratio: float  # wick beyond the level / total bar range
    confidence: float


@dataclass(frozen=True, slots=True)
class Displacement:
    direction: int
    index: int
    ts: int
    body_atr: float
    close_location: float   # 0..1, where the close sat in the bar
    confidence: float


@dataclass(slots=True)
class ICTAnalysis:
    fair_value_gaps: list[FairValueGap]
    order_blocks: list[OrderBlock]
    liquidity: list[LiquidityPool]
    sweeps: list[LiquiditySweep]
    displacements: list[Displacement]
    dealing_range_high: float
    dealing_range_low: float
    premium_discount: float          # 0 at range low, 1 at range high
    zone: str                        # "premium" | "discount" | "equilibrium"
    mss: StructureEvent | None
    close: float
    atr: float

    # -- convenience ------------------------------------------------------

    @property
    def fresh_bullish_fvgs(self) -> list[FairValueGap]:
        return [g for g in self.fair_value_gaps if g.direction > 0 and g.is_fresh]

    @property
    def fresh_bearish_fvgs(self) -> list[FairValueGap]:
        return [g for g in self.fair_value_gaps if g.direction < 0 and g.is_fresh]

    @property
    def last_sweep(self) -> LiquiditySweep | None:
        return self.sweeps[-1] if self.sweeps else None

    @property
    def last_displacement(self) -> Displacement | None:
        return self.displacements[-1] if self.displacements else None

    def active_order_blocks(self, direction: int) -> list[OrderBlock]:
        return [
            ob
            for ob in self.order_blocks
            if ob.direction == direction and not ob.broken
        ]

    def breaker_blocks(self, direction: int) -> list[OrderBlock]:
        """Broken blocks flip polarity: a broken bullish OB becomes resistance."""

        return [ob for ob in self.order_blocks if ob.broken and ob.direction == -direction]

    def nearest_fvg(self, direction: int, price: float) -> FairValueGap | None:
        candidates = [
            g
            for g in self.fair_value_gaps
            if g.direction == direction and g.is_fresh
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda g: abs(g.mid - price))

    def nearest_liquidity(self, direction: int, price: float) -> LiquidityPool | None:
        wanted = "buy_side" if direction > 0 else "sell_side"
        pools = [
            p
            for p in self.liquidity
            if p.kind == wanted
            and not p.swept
            and ((p.price > price) if direction > 0 else (p.price < price))
        ]
        if not pools:
            return None
        return min(pools, key=lambda p: abs(p.price - price))

    # -- aggregate --------------------------------------------------------

    def bias(self) -> Bias:
        bullish = 0.0
        bearish = 0.0
        sweep = self.last_sweep
        if sweep is not None:
            if sweep.direction > 0:
                bullish += sweep.confidence * 2
            else:
                bearish += sweep.confidence * 2
        displacement = self.last_displacement
        if displacement is not None:
            if displacement.direction > 0:
                bullish += displacement.confidence
            else:
                bearish += displacement.confidence
        bullish += 0.5 * len(self.fresh_bullish_fvgs[-3:])
        bearish += 0.5 * len(self.fresh_bearish_fvgs[-3:])
        bullish += 0.6 * len(self.active_order_blocks(1)[-2:])
        bearish += 0.6 * len(self.active_order_blocks(-1)[-2:])
        if self.mss is not None:
            if self.mss.direction > 0:
                bullish += 1.5 * self.mss.confidence
            else:
                bearish += 1.5 * self.mss.confidence
        # Discount favours longs, premium favours shorts.
        if self.zone == "discount":
            bullish += 0.5
        elif self.zone == "premium":
            bearish += 0.5

        if bullish >= bearish * 1.5 and bullish >= 1.0:
            return Bias.BULLISH
        if bearish >= bullish * 1.5 and bearish >= 1.0:
            return Bias.BEARISH
        if bullish >= 1.0 and bearish >= 1.0:
            return Bias.CONFLICT
        return Bias.NEUTRAL

    def score(self, direction: int) -> float:
        """0..100 - how much ICT confluence supports ``direction``."""

        score = 40.0
        sweep = self.last_sweep
        if sweep is not None and sweep.direction == direction:
            score += 18 * sweep.confidence
        elif sweep is not None:
            score -= 8 * sweep.confidence

        displacement = self.last_displacement
        if displacement is not None and displacement.direction == direction:
            score += 12 * displacement.confidence

        if self.mss is not None and self.mss.direction == direction:
            score += 14 * self.mss.confidence

        fresh = self.fresh_bullish_fvgs if direction > 0 else self.fresh_bearish_fvgs
        score += min(len(fresh), 3) * 4.0

        blocks = self.active_order_blocks(direction)
        score += min(len(blocks), 2) * 5.0

        if direction > 0 and self.zone == "discount":
            score += 8
        elif direction < 0 and self.zone == "premium":
            score += 8
        elif self.zone == "equilibrium":
            score += 2
        else:
            score -= 6

        target = self.nearest_liquidity(direction, self.close)
        if target is not None:
            score += 5 * target.confidence

        return max(0.0, min(100.0, score))

    def as_features(self) -> dict[str, float]:
        sweep = self.last_sweep
        displacement = self.last_displacement
        return {
            "ict_premium_discount": self.premium_discount,
            "ict_zone_discount": 1.0 if self.zone == "discount" else 0.0,
            "ict_zone_premium": 1.0 if self.zone == "premium" else 0.0,
            "ict_sweep_direction": float(sweep.direction) if sweep else 0.0,
            "ict_sweep_confidence": sweep.confidence if sweep else 0.0,
            "ict_displacement_direction": (
                float(displacement.direction) if displacement else 0.0
            ),
            "ict_displacement_atr": displacement.body_atr if displacement else 0.0,
            "ict_bull_fvg_count": float(len(self.fresh_bullish_fvgs)),
            "ict_bear_fvg_count": float(len(self.fresh_bearish_fvgs)),
            "ict_bull_ob_count": float(len(self.active_order_blocks(1))),
            "ict_bear_ob_count": float(len(self.active_order_blocks(-1))),
            "ict_mss_direction": float(self.mss.direction) if self.mss else 0.0,
            "ict_equal_levels": float(sum(1 for p in self.liquidity if p.equal)),
        }


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------


def detect_fair_value_gaps(
    candles: Candles,
    atr_values: Sequence[float | None],
    min_size_atr: float = 0.12,
    lookback: int = 120,
    max_gaps: int = 12,
) -> list[FairValueGap]:
    """Three-candle FVG / imbalance detection with fill tracking."""

    gaps: list[FairValueGap] = []
    size = len(candles)
    start = max(2, size - lookback)

    for i in range(start, size):
        first = candles[i - 2]
        third = candles[i]
        atr = _atr_at(atr_values, i)
        if atr is None or atr <= 0:
            continue

        if third.low > first.high:              # bullish imbalance
            top, bottom, direction = third.low, first.high, 1
        elif third.high < first.low:            # bearish imbalance
            top, bottom, direction = first.low, third.high, -1
        else:
            continue

        gap_size = top - bottom
        size_atr = gap_size / atr
        if size_atr < min_size_atr:
            continue

        filled = _fill_fraction(candles[i + 1 :], top, bottom, direction)
        # Displacement of the middle candle adds conviction.
        middle_body_atr = candles[i - 1].body / atr
        confidence = _clip(
            0.35
            + 0.30 * min(size_atr / 1.0, 1.0)
            + 0.20 * min(middle_body_atr / 1.5, 1.0)
            + 0.15 * (1.0 - filled)
        )
        gaps.append(
            FairValueGap(
                direction=direction,
                top=top,
                bottom=bottom,
                index=i,
                ts=third.ts,
                size_atr=round(size_atr, 4),
                filled_pct=round(filled, 4),
                confidence=round(confidence, 4),
            )
        )

    return gaps[-max_gaps:]


def _fill_fraction(after: Candles, top: float, bottom: float, direction: int) -> float:
    """How much of the gap has been traded back into, in ``[0, 1]``."""

    height = top - bottom
    if height <= 0:
        return 1.0
    deepest = 0.0
    for candle in after:
        if direction > 0:
            penetration = top - candle.low
        else:
            penetration = candle.high - bottom
        deepest = max(deepest, min(penetration, height))
    return max(0.0, min(1.0, deepest / height))


def detect_displacements(
    candles: Candles,
    atr_values: Sequence[float | None],
    min_body_atr: float = 1.2,
    lookback: int = 60,
    max_items: int = 10,
) -> list[Displacement]:
    """Outsized, decisive candles - the engine of ICT entry models."""

    out: list[Displacement] = []
    size = len(candles)
    for i in range(max(0, size - lookback), size):
        candle = candles[i]
        atr = _atr_at(atr_values, i)
        if atr is None or atr <= 0:
            continue
        body_atr = candle.body / atr
        if body_atr < min_body_atr:
            continue
        span = candle.range
        if span <= 0:
            continue
        location = (candle.close - candle.low) / span
        direction = 1 if candle.is_bullish else -1
        commitment = location if direction > 0 else 1.0 - location
        body_ratio = candle.body / span
        confidence = _clip(
            0.3 + 0.35 * min(body_atr / 2.5, 1.0) + 0.2 * commitment + 0.15 * body_ratio
        )
        out.append(
            Displacement(
                direction=direction,
                index=i,
                ts=candle.ts,
                body_atr=round(body_atr, 4),
                close_location=round(location, 4),
                confidence=round(confidence, 4),
            )
        )
    return out[-max_items:]


def detect_order_blocks(
    candles: Candles,
    atr_values: Sequence[float | None],
    displacements: Sequence[Displacement],
    max_blocks: int = 10,
) -> list[OrderBlock]:
    """The last opposing candle immediately before each displacement."""

    blocks: list[OrderBlock] = []
    size = len(candles)

    for displacement in displacements:
        origin = None
        for j in range(displacement.index - 1, max(displacement.index - 8, -1), -1):
            candle = candles[j]
            if displacement.direction > 0 and candle.is_bearish:
                origin = j
                break
            if displacement.direction < 0 and candle.is_bullish:
                origin = j
                break
        if origin is None:
            continue

        candle = candles[origin]
        top, bottom = candle.high, candle.low
        after = candles[displacement.index + 1 :]

        mitigated = any(
            (c.low <= top if displacement.direction > 0 else c.high >= bottom)
            for c in after
        )
        # "Broken" means a close through the far side of the block.
        broken = any(
            (c.close < bottom if displacement.direction > 0 else c.close > top)
            for c in after
        )

        age = size - origin
        recency = max(0.0, 1.0 - age / max(size, 1))
        confidence = _clip(
            0.3
            + 0.35 * displacement.confidence
            + 0.2 * recency
            + (0.15 if not mitigated else 0.0)
        )
        blocks.append(
            OrderBlock(
                direction=displacement.direction,
                top=top,
                bottom=bottom,
                index=origin,
                ts=candle.ts,
                displacement_atr=displacement.body_atr,
                mitigated=mitigated,
                broken=broken,
                confidence=round(confidence, 4),
            )
        )

    # De-duplicate blocks that share an origin candle.
    unique: dict[int, OrderBlock] = {}
    for block in blocks:
        current = unique.get(block.index)
        if current is None or block.confidence > current.confidence:
            unique[block.index] = block
    return sorted(unique.values(), key=lambda b: b.index)[-max_blocks:]


def detect_liquidity(
    candles: Candles,
    swings: Sequence[Swing],
    atr: float,
    equal_tolerance_atr: float = 0.15,
    max_pools: int = 12,
) -> list[LiquidityPool]:
    """Swing highs are buy-side liquidity; swing lows are sell-side liquidity."""

    if not swings or atr <= 0:
        return []

    tolerance = equal_tolerance_atr * atr
    pools: list[LiquidityPool] = []

    for kind, is_high in (("buy_side", True), ("sell_side", False)):
        points = [s for s in swings if s.is_high == is_high]
        used: set[int] = set()
        for i, swing in enumerate(points):
            if i in used:
                continue
            cluster = [swing]
            for j in range(i + 1, len(points)):
                if j in used:
                    continue
                if abs(points[j].price - swing.price) <= tolerance:
                    cluster.append(points[j])
                    used.add(j)
            used.add(i)

            price = sum(s.price for s in cluster) / len(cluster)
            last_index = max(s.index for s in cluster)
            after = candles[last_index + 1 :]
            swept = any(
                (c.high > price + tolerance * 0.5)
                if is_high
                else (c.low < price - tolerance * 0.5)
                for c in after
            )
            equal = len(cluster) >= 2
            recency = last_index / max(len(candles), 1)
            confidence = _clip(
                0.35 + (0.25 if equal else 0.0) + 0.25 * recency + 0.15 * min(len(cluster) / 3, 1.0)
            )
            pools.append(
                LiquidityPool(
                    kind=kind,
                    price=price,
                    index=last_index,
                    ts=candles[last_index].ts if last_index < len(candles) else 0,
                    touches=len(cluster),
                    equal=equal,
                    swept=swept,
                    confidence=round(confidence, 4),
                )
            )

    pools.sort(key=lambda p: p.index)
    return pools[-max_pools:]


def detect_sweeps(
    candles: Candles,
    pools: Sequence[LiquidityPool],
    atr_values: Sequence[float | None],
    lookback: int = 30,
    max_sweeps: int = 6,
) -> list[LiquiditySweep]:
    """A bar that pierces a liquidity pool and closes back on the other side.

    Sweeping *sell-side* liquidity (running the lows) and closing back up is a
    bullish sweep, and vice versa - the direction is the direction of the
    expected reaction, not of the wick.
    """

    sweeps: list[LiquiditySweep] = []
    size = len(candles)
    start = max(0, size - lookback)

    for i in range(start, size):
        candle = candles[i]
        atr = _atr_at(atr_values, i)
        if atr is None or atr <= 0 or candle.range <= 0:
            continue
        for pool in pools:
            if pool.index >= i:
                continue
            if pool.kind == "sell_side":
                if candle.low < pool.price and candle.close > pool.price:
                    penetration = (pool.price - candle.low) / atr
                    rejection = (pool.price - candle.low) / candle.range
                    direction = 1
                else:
                    continue
            else:
                if candle.high > pool.price and candle.close < pool.price:
                    penetration = (candle.high - pool.price) / atr
                    rejection = (candle.high - pool.price) / candle.range
                    direction = -1
                else:
                    continue

            if penetration < 0.05:
                continue
            confidence = _clip(
                0.3
                + 0.25 * min(penetration / 0.8, 1.0)
                + 0.25 * min(rejection / 0.6, 1.0)
                + 0.2 * pool.confidence
            )
            sweeps.append(
                LiquiditySweep(
                    direction=direction,
                    level=pool.price,
                    index=i,
                    ts=candle.ts,
                    penetration_atr=round(penetration, 4),
                    rejection_ratio=round(rejection, 4),
                    confidence=round(confidence, 4),
                )
            )

    # Keep the strongest sweep per bar.
    best: dict[int, LiquiditySweep] = {}
    for sweep in sweeps:
        current = best.get(sweep.index)
        if current is None or sweep.confidence > current.confidence:
            best[sweep.index] = sweep
    return sorted(best.values(), key=lambda s: s.index)[-max_sweeps:]


def dealing_range(
    candles: Candles, swings: Sequence[Swing], lookback: int = 90
) -> tuple[float, float]:
    """The reference range used for premium/discount.

    Prefers the most recent confirmed swing high/low pair; falls back to the
    lookback envelope when structure is not yet established.
    """

    highs = [s for s in swings if s.is_high]
    lows = [s for s in swings if s.is_low]
    if highs and lows:
        return highs[-1].price, lows[-1].price
    window = candles[-lookback:]
    return max(c.high for c in window), min(c.low for c in window)


def classify_zone(position: float) -> str:
    if position > 0.55:
        return "premium"
    if position < 0.45:
        return "discount"
    return "equilibrium"


def _atr_at(atr_values: Sequence[float | None], index: int) -> float | None:
    if index < len(atr_values) and atr_values[index] is not None:
        return atr_values[index]
    for value in reversed(atr_values[: min(index + 1, len(atr_values))]):
        if value is not None:
            return value
    return None


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else high if value > high else value


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def analyse_ict(
    candles: Candles,
    atr_values: Sequence[float | None],
    structure: MarketStructure,
) -> ICTAnalysis:
    """Run the full ICT read for one symbol/timeframe."""

    if not candles:
        raise ValueError("cannot analyse an empty candle series")

    atr = structure.atr
    close = candles[-1].close

    displacements = detect_displacements(candles, atr_values)
    gaps = detect_fair_value_gaps(candles, atr_values)
    blocks = detect_order_blocks(candles, atr_values, displacements)
    pools = detect_liquidity(candles, structure.swings, atr)
    sweeps = detect_sweeps(candles, pools, atr_values)

    range_high, range_low = dealing_range(candles, structure.swings)
    width = range_high - range_low
    position = (close - range_low) / width if width > 0 else 0.5
    position = _clip(position)

    mss = None
    for event in reversed(structure.events):
        if event.type is StructureEventType.MSS:
            mss = event
            break

    return ICTAnalysis(
        fair_value_gaps=gaps,
        order_blocks=blocks,
        liquidity=pools,
        sweeps=sweeps,
        displacements=displacements,
        dealing_range_high=range_high,
        dealing_range_low=range_low,
        premium_discount=round(position, 4),
        zone=classify_zone(position),
        mss=mss,
        close=close,
        atr=atr,
    )


__all__ = [
    "ICTAnalysis",
    "FairValueGap",
    "OrderBlock",
    "LiquidityPool",
    "LiquiditySweep",
    "Displacement",
    "analyse_ict",
    "detect_fair_value_gaps",
    "detect_order_blocks",
    "detect_liquidity",
    "detect_sweeps",
    "detect_displacements",
    "dealing_range",
    "classify_zone",
]
