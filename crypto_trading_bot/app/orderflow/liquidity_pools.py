"""Liquidity pool intelligence.

Builds a probabilistic map of where resting liquidity most likely sits, from
sources that actually exist in the data we have:

* prior swing highs/lows and equal highs/lows (from the ICT engine)
* high-volume price nodes (a volume-by-price histogram over recent bars)
* order-book concentration (unusually large resting levels)
* estimated liquidation clusters, derived from *typical* leverage bands around
  recent swing entries - explicitly a model, not observed data

**Nothing here assumes a pool will be reached.** Each has a probability derived
from distance, size and freshness, and the consuming model treats a low
probability as no signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.domain import Candle, OrderBook


@dataclass(frozen=True, slots=True)
class LiquidityPool:
    price: float
    side: str                 # "above" | "below"
    kind: str                 # swing | equal | volume_node | book | liquidation
    strength: float           # 0..1 relative size/importance
    probability: float        # 0..1 estimated chance of being traded into
    note: str = ""

    @property
    def direction(self) -> int:
        return 1 if self.side == "above" else -1


@dataclass(slots=True)
class LiquidityMap:
    pools: list[LiquidityPool] = field(default_factory=list)
    data_quality: float = 0.0
    price: float = 0.0
    atr: float = 0.0

    def above(self) -> list[LiquidityPool]:
        return sorted([p for p in self.pools if p.side == "above"], key=lambda p: p.price)

    def below(self) -> list[LiquidityPool]:
        return sorted(
            [p for p in self.pools if p.side == "below"], key=lambda p: p.price, reverse=True
        )

    def pull(self, direction: int) -> float:
        """Total probability-weighted strength on one side."""

        side = "above" if direction > 0 else "below"
        return sum(p.strength * p.probability for p in self.pools if p.side == side)

    def dominant_target(self, price: float) -> tuple[int, float, str] | None:
        """The side liquidity is most concentrated on.

        Returns ``(direction, probability, explanation)``. Returns ``None`` when
        neither side is meaningfully heavier - liquidity being *somewhere* is
        not a signal.
        """

        up = self.pull(1)
        down = self.pull(-1)
        total = up + down
        if total <= 0:
            return None

        share = max(up, down) / total
        if share < 0.6:
            return None

        direction = 1 if up > down else -1
        pools = self.above() if direction > 0 else self.below()
        if not pools:
            return None
        nearest = pools[0]
        probability = min(share * nearest.probability * 1.2, 0.9)
        note = (
            f"{share:.0%} of nearby liquidity sits "
            f"{'above' if direction > 0 else 'below'}; nearest is "
            f"{nearest.kind} at {nearest.price:.6g}"
        )
        return direction, probability, note

    def distance_atr(self, price: float, direction: int) -> float:
        pools = self.above() if direction > 0 else self.below()
        if not pools or self.atr <= 0:
            return 0.0
        return abs(pools[0].price - price) / self.atr

    def as_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "data_quality": round(self.data_quality, 4),
            "pull_up": round(self.pull(1), 4),
            "pull_down": round(self.pull(-1), 4),
            "pools": [
                {
                    "price": round(p.price, 8),
                    "side": p.side,
                    "kind": p.kind,
                    "strength": round(p.strength, 3),
                    "probability": round(p.probability, 3),
                    "note": p.note,
                }
                for p in sorted(self.pools, key=lambda p: -p.strength)[:12]
            ],
        }


def volume_nodes(
    candles: Sequence[Candle], bins: int = 40, top: int = 4
) -> list[tuple[float, float]]:
    """High-volume price nodes from a volume-by-price histogram.

    Volume is spread uniformly across each bar's range, which is the standard
    approximation when intrabar distribution is unknown.
    """

    if len(candles) < 20:
        return []
    low = min(c.low for c in candles)
    high = max(c.high for c in candles)
    if high <= low:
        return []

    width = (high - low) / bins
    histogram = [0.0] * bins
    for candle in candles:
        span = candle.high - candle.low
        if span <= 0:
            index = min(int((candle.close - low) / width), bins - 1)
            histogram[index] += candle.volume
            continue
        start = max(int((candle.low - low) / width), 0)
        end = min(int((candle.high - low) / width), bins - 1)
        share = candle.volume / max(end - start + 1, 1)
        for index in range(start, end + 1):
            histogram[index] += share

    peak = max(histogram) or 1.0
    ranked = sorted(range(bins), key=lambda i: histogram[i], reverse=True)[:top]
    return [(low + (i + 0.5) * width, histogram[i] / peak) for i in ranked]


def book_concentrations(
    book: OrderBook | None, multiple: float = 3.0, top: int = 3
) -> list[tuple[float, float, str]]:
    """Resting levels far larger than their neighbours."""

    if book is None:
        return []
    out: list[tuple[float, float, str]] = []
    for levels, side in ((book.bids, "below"), (book.asks, "above")):
        if len(levels) < 5:
            continue
        sizes = [size for _price, size in levels]
        average = sum(sizes) / len(sizes)
        if average <= 0:
            continue
        for price, size in levels:
            if size >= average * multiple:
                out.append((price, min(size / (average * multiple), 1.0), side))
    out.sort(key=lambda item: item[1], reverse=True)
    return out[:top]


def estimated_liquidation_levels(
    candles: Sequence[Candle], price: float, leverages: Sequence[float] = (10, 25, 50)
) -> list[tuple[float, str]]:
    """Approximate where leveraged positions would be liquidated.

    This is a **model, not data**: it assumes traders entered near recent swing
    extremes at common leverage bands. It is labelled as such everywhere it
    surfaces, and is given low strength so it can never dominate the map.
    """

    if len(candles) < 30 or price <= 0:
        return []
    window = candles[-60:]
    recent_high = max(c.high for c in window)
    recent_low = min(c.low for c in window)

    out: list[tuple[float, str]] = []
    for leverage in leverages:
        move = 1.0 / leverage
        # Longs entered near the recent low get liquidated below it.
        out.append((recent_low * (1 - move), f"~{leverage:g}x longs from the recent low"))
        # Shorts entered near the recent high get liquidated above it.
        out.append((recent_high * (1 + move), f"~{leverage:g}x shorts from the recent high"))
    return out


def build_liquidity_map(
    candles: Sequence[Candle],
    ict_analysis: Any = None,
    book: OrderBook | None = None,
    atr: float = 0.0,
    include_estimated_liquidations: bool = True,
    max_distance_atr: float = 12.0,
) -> LiquidityMap:
    """Assemble the probabilistic liquidity map."""

    if not candles:
        return LiquidityMap()

    price = candles[-1].close
    liquidity_map = LiquidityMap(price=price, atr=atr)
    quality_parts: list[float] = []

    def add(price_level: float, kind: str, strength: float, note: str) -> None:
        if price_level <= 0 or price_level == price:
            return
        distance = abs(price_level - price)
        if atr > 0:
            distance_atr = distance / atr
            if distance_atr > max_distance_atr:
                return
            # Nearby liquidity is far more likely to be traded into.
            probability = math.exp(-distance_atr / 4.0)
        else:
            probability = 0.4
        liquidity_map.pools.append(
            LiquidityPool(
                price=price_level,
                side="above" if price_level > price else "below",
                kind=kind,
                strength=_clip01(strength),
                probability=_clip01(probability * (0.4 + 0.6 * strength)),
                note=note,
            )
        )

    # --- swing / equal highs and lows from the ICT engine ----------------
    if ict_analysis is not None and getattr(ict_analysis, "liquidity", None):
        quality_parts.append(1.0)
        for pool in ict_analysis.liquidity:
            if pool.swept:
                continue
            add(
                pool.price,
                "equal" if pool.equal else "swing",
                0.5 + 0.4 * pool.confidence + (0.1 if pool.equal else 0.0),
                f"{'equal' if pool.equal else 'swing'} {pool.kind.replace('_', ' ')}"
                f" ({pool.touches} touch{'es' if pool.touches != 1 else ''})",
            )

    # --- volume nodes ------------------------------------------------------
    nodes = volume_nodes(candles)
    if nodes:
        quality_parts.append(1.0)
        for node_price, strength in nodes:
            add(node_price, "volume_node", strength * 0.8, "high-volume price node")

    # --- book concentrations ------------------------------------------------
    concentrations = book_concentrations(book)
    if concentrations:
        quality_parts.append(1.0)
        for level_price, strength, _side in concentrations:
            add(level_price, "book", strength * 0.7, "unusually large resting order")
    elif book is not None:
        quality_parts.append(0.5)

    # --- estimated liquidations ---------------------------------------------
    if include_estimated_liquidations:
        for level_price, note in estimated_liquidation_levels(candles, price):
            # Deliberately weak: this is inference, not observation.
            add(level_price, "liquidation", 0.3, f"estimated liquidations, {note}")

    liquidity_map.data_quality = (
        sum(quality_parts) / max(len(quality_parts), 1) if quality_parts else 0.2
    )
    return liquidity_map


def _clip01(value: float) -> float:
    if value != value:
        return 0.0
    return 0.0 if value < 0 else 1.0 if value > 1 else value


__all__ = [
    "LiquidityMap",
    "LiquidityPool",
    "build_liquidity_map",
    "volume_nodes",
    "book_concentrations",
    "estimated_liquidation_levels",
]
