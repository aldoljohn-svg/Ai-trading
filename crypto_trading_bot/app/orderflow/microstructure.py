"""Market microstructure and execution-quality assessment.

Answers one question: *can this size actually be traded here, right now, at a
cost that does not eat the edge?*

Estimates expected slippage by walking the visible book, which is the only
honest way to do it - a fixed slippage assumption is fine for a backtest and
useless when the book is thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain import OrderBook, Side


@dataclass(slots=True)
class MicrostructureRead:
    spread_pct: float = 0.0
    depth_quote_bid: float = 0.0
    depth_quote_ask: float = 0.0
    imbalance: float = 0.0
    expected_slippage_pct: float = 0.0
    market_impact_pct: float = 0.0
    fillable: bool = True
    levels_consumed: int = 0
    thin_liquidity: bool = False
    spread_expanded: bool = False
    data_quality: float = 0.0
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def tradable(self) -> bool:
        return self.fillable and not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "spread_pct": round(self.spread_pct, 6),
            "depth_bid": round(self.depth_quote_bid, 2),
            "depth_ask": round(self.depth_quote_ask, 2),
            "imbalance": round(self.imbalance, 4),
            "expected_slippage_pct": round(self.expected_slippage_pct, 6),
            "market_impact_pct": round(self.market_impact_pct, 6),
            "fillable": self.fillable,
            "levels_consumed": self.levels_consumed,
            "thin_liquidity": self.thin_liquidity,
            "spread_expanded": self.spread_expanded,
            "data_quality": round(self.data_quality, 4),
            "problems": self.problems,
            "notes": self.notes,
        }


def walk_book(
    book: OrderBook, side: Side, notional: float
) -> tuple[float, int, bool]:
    """Walk the book to fill ``notional``.

    Returns ``(volume-weighted average price, levels consumed, fully filled)``.
    Buying consumes asks; selling consumes bids.
    """

    levels = book.asks if side is Side.LONG else book.bids
    if not levels or notional <= 0:
        return 0.0, 0, False

    remaining = notional
    cost = 0.0
    quantity = 0.0
    consumed = 0

    for price, size in levels:
        if price <= 0 or size <= 0:
            continue
        level_notional = price * size
        consumed += 1
        if level_notional >= remaining:
            take = remaining / price
            cost += take * price
            quantity += take
            remaining = 0.0
            break
        cost += level_notional
        quantity += size
        remaining -= level_notional

    if quantity <= 0:
        return 0.0, consumed, False
    return cost / quantity, consumed, remaining <= 1e-9


def analyse_microstructure(
    book: OrderBook | None,
    notional: float,
    side: Side | None = None,
    max_spread_pct: float = 0.0008,
    max_slippage_pct: float = 0.003,
    min_depth_multiple: float = 3.0,
    reference_spread_pct: float | None = None,
) -> MicrostructureRead:
    """Assess whether ``notional`` can be executed acceptably.

    ``min_depth_multiple`` requires the visible near-touch depth to be at least
    this many times the order size - filling more than a third of the visible
    book is how a "market" order becomes a market event.
    """

    read = MicrostructureRead()

    if book is None or not book.bids or not book.asks:
        read.problems.append("no order book available")
        read.fillable = False
        read.data_quality = 0.0
        return read

    mid = book.mid
    if mid <= 0:
        read.problems.append("order book has no valid mid price")
        read.fillable = False
        return read

    read.spread_pct = book.spread_pct
    read.depth_quote_bid = book.depth_quote("bid", 0.005)
    read.depth_quote_ask = book.depth_quote("ask", 0.005)
    total_depth = read.depth_quote_bid + read.depth_quote_ask
    if total_depth > 0:
        read.imbalance = (read.depth_quote_bid - read.depth_quote_ask) / total_depth

    levels = min(len(book.bids), len(book.asks))
    read.data_quality = min(levels / 10.0, 1.0)

    # --- spread ---------------------------------------------------------
    if read.spread_pct > max_spread_pct:
        read.problems.append(
            f"spread {read.spread_pct:.4%} exceeds the {max_spread_pct:.4%} limit"
        )
    if reference_spread_pct and reference_spread_pct > 0:
        if read.spread_pct > reference_spread_pct * 2.5:
            read.spread_expanded = True
            read.notes.append(
                f"spread is {read.spread_pct / reference_spread_pct:.1f}x its usual level"
            )

    # --- depth ------------------------------------------------------------
    if side is not None and notional > 0:
        relevant_depth = (
            read.depth_quote_ask if side is Side.LONG else read.depth_quote_bid
        )
        if relevant_depth > 0 and notional > relevant_depth / min_depth_multiple:
            read.thin_liquidity = True
            read.problems.append(
                f"order of ${notional:,.0f} is large against ${relevant_depth:,.0f} "
                f"of visible depth"
            )

        average_price, consumed, filled = walk_book(book, side, notional)
        read.levels_consumed = consumed
        read.fillable = filled
        if not filled:
            read.problems.append("visible book cannot absorb this size")
        if average_price > 0:
            slip = (average_price - mid) / mid
            # Buying pays up, selling gets hit down; express both as a cost.
            read.expected_slippage_pct = abs(slip)
            read.market_impact_pct = max(
                0.0, read.expected_slippage_pct - read.spread_pct / 2
            )
            if read.expected_slippage_pct > max_slippage_pct:
                read.problems.append(
                    f"estimated slippage {read.expected_slippage_pct:.3%} exceeds the "
                    f"{max_slippage_pct:.3%} limit"
                )

    if read.data_quality < 0.3:
        read.notes.append("shallow book snapshot - estimates are rough")

    return read


__all__ = ["MicrostructureRead", "analyse_microstructure", "walk_book"]
