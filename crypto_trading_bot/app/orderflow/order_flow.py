"""Order flow analysis.

Inputs available from MEXC's public API:

* an order-book snapshot (bids/asks with sizes)
* OHLCV candles
* ticker-level open interest and funding

Inputs **not** available: a per-trade tape with aggressor flags. Real cumulative
volume delta requires knowing whether each trade hit the bid or lifted the
offer. Without it, this module computes a *proxy* delta from where each bar
closed inside its own range weighted by volume, and reports
``cvd_is_proxy=True`` with reduced ``data_quality``.

That distinction is load-bearing: a proxy CVD is a reasonable directional hint
and a terrible precision instrument, and the models that read it are told which
one they are getting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from app.domain import Candle, OrderBook

#: Quality assigned when delta is derived from bars rather than a real tape.
PROXY_QUALITY = 0.55


@dataclass(slots=True)
class OrderFlowRead:
    score: float = 0.0                  # -1 (sell pressure) .. +1 (buy pressure)
    state: str = "UNCERTAIN"            # BUY_PRESSURE | SELL_PRESSURE | NEUTRAL | UNCERTAIN
    confidence: float = 0.0
    data_quality: float = 0.0
    book_imbalance: float = 0.0         # -1 .. +1
    delta: float = 0.0                  # proxy aggressive buy - sell volume
    cvd: float = 0.0                    # cumulative over the window
    cvd_slope: float = 0.0              # normalised trend of CVD
    cvd_is_proxy: bool = True
    volume_ratio: float = 1.0
    absorption: bool = False            # heavy volume, little price progress
    divergence: int = 0                 # +1 bullish, -1 bearish, 0 none
    reasoning: list[str] = field(default_factory=list)

    @property
    def direction(self) -> int:
        if self.state == "BUY_PRESSURE":
            return 1
        if self.state == "SELL_PRESSURE":
            return -1
        return 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "state": self.state,
            "confidence": round(self.confidence, 4),
            "data_quality": round(self.data_quality, 4),
            "book_imbalance": round(self.book_imbalance, 4),
            "cvd": round(self.cvd, 4),
            "cvd_slope": round(self.cvd_slope, 4),
            "cvd_is_proxy": self.cvd_is_proxy,
            "volume_ratio": round(self.volume_ratio, 3),
            "absorption": self.absorption,
            "divergence": self.divergence,
            "reasoning": self.reasoning,
        }


def bar_delta(candle: Candle) -> float:
    """Volume-weighted close location - the proxy for aggressor imbalance.

    A bar closing at its high with heavy volume implies buyers were lifting
    offers; closing at its low implies sellers were hitting bids. This is a
    coarse but unbiased estimator when no tape is available.
    """

    span = candle.high - candle.low
    if span <= 0:
        return 0.0
    location = (candle.close - candle.low) / span      # 0..1
    return candle.volume * (2.0 * location - 1.0)      # -volume .. +volume


def book_imbalance(book: OrderBook | None, depth_pct: float = 0.005) -> tuple[float, float]:
    """Returns ``(imbalance in [-1, 1], quality in [0, 1])``.

    Only levels within ``depth_pct`` of mid are counted: quotes far from the
    touch are frequently pulled and say little about immediate pressure.
    """

    if book is None or not book.bids or not book.asks:
        return 0.0, 0.0
    mid = book.mid
    if mid <= 0:
        return 0.0, 0.0

    bid_volume = book.depth_quote("bid", depth_pct)
    ask_volume = book.depth_quote("ask", depth_pct)
    total = bid_volume + ask_volume
    if total <= 0:
        return 0.0, 0.0

    imbalance = (bid_volume - ask_volume) / total
    # Quality scales with how many levels we actually saw.
    levels = min(len(book.bids), len(book.asks))
    quality = min(levels / 10.0, 1.0)
    return imbalance, quality


def analyse_order_flow(
    candles: Sequence[Candle],
    book: OrderBook | None = None,
    lookback: int = 30,
    atr: float = 0.0,
    trades: Sequence[Any] | None = None,
) -> OrderFlowRead:
    """Build an order-flow read.

    ``trades`` is accepted for venues that do expose an aggressor tape; when
    supplied with ``side`` and ``quantity`` attributes it replaces the proxy and
    ``cvd_is_proxy`` becomes ``False``.
    """

    read = OrderFlowRead()
    reasons: list[str] = []

    if len(candles) < 10:
        read.state = "UNCERTAIN"
        read.reasoning = ["not enough bars for a flow read"]
        return read

    window = list(candles[-lookback:])

    # --- delta / CVD ---------------------------------------------------
    if trades:
        deltas = []
        for trade in trades:
            side = getattr(trade, "side", None)
            quantity = float(getattr(trade, "quantity", 0.0) or 0.0)
            sign = 1.0 if str(getattr(side, "value", side)).lower() in ("long", "buy") else -1.0
            deltas.append(sign * quantity)
        read.cvd_is_proxy = False
        delta_series = deltas
        base_quality = 1.0
        reasons.append(f"true aggressor tape ({len(deltas)} trades)")
    else:
        delta_series = [bar_delta(c) for c in window]
        read.cvd_is_proxy = True
        base_quality = PROXY_QUALITY
        reasons.append("delta estimated from bar close location (no tape available)")

    read.delta = delta_series[-1] if delta_series else 0.0
    read.cvd = sum(delta_series)

    total_volume = sum(abs(d) for d in delta_series) or 1e-12
    normalised_cvd = read.cvd / total_volume            # -1 .. +1

    # CVD slope over the second half versus the first.
    half = max(len(delta_series) // 2, 1)
    first = sum(delta_series[:half])
    second = sum(delta_series[half:])
    read.cvd_slope = (second - first) / total_volume

    # --- book imbalance -------------------------------------------------
    imbalance, book_quality = book_imbalance(book)
    read.book_imbalance = imbalance
    if book_quality > 0:
        reasons.append(
            f"book imbalance {imbalance:+.0%} "
            f"({'bid' if imbalance > 0 else 'ask'}-heavy near the touch)"
        )

    # --- relative volume --------------------------------------------------
    recent_volume = sum(c.volume for c in window[-5:]) / 5
    baseline = sum(c.volume for c in window[:-5]) / max(len(window) - 5, 1)
    read.volume_ratio = recent_volume / baseline if baseline > 0 else 1.0

    # --- absorption: heavy volume, no progress ---------------------------
    price_move = abs(window[-1].close - window[-5].close) if len(window) >= 5 else 0.0
    if atr > 0 and read.volume_ratio > 1.5 and price_move < 0.5 * atr:
        read.absorption = True
        reasons.append(
            f"absorption: volume {read.volume_ratio:.1f}x normal but price moved "
            f"only {price_move / atr:.2f} ATR"
        )

    # --- divergence: price and flow disagree ------------------------------
    price_direction = 0
    if len(window) >= 10:
        change = window[-1].close - window[-10].close
        if atr > 0 and abs(change) > 0.5 * atr:
            price_direction = 1 if change > 0 else -1
    flow_direction = 1 if read.cvd_slope > 0.08 else (-1 if read.cvd_slope < -0.08 else 0)
    if price_direction and flow_direction and price_direction != flow_direction:
        read.divergence = flow_direction
        reasons.append(
            f"flow/price divergence: price {'up' if price_direction > 0 else 'down'} "
            f"while flow is {'buying' if flow_direction > 0 else 'selling'}"
        )

    # --- combine ----------------------------------------------------------
    # Book imbalance is immediate but shallow; CVD slope is slower but deeper.
    components = [
        (normalised_cvd, 0.30),
        (read.cvd_slope, 0.35),
        (imbalance, 0.35 if book_quality > 0 else 0.0),
    ]
    weight_total = sum(w for _v, w in components) or 1.0
    read.score = sum(v * w for v, w in components) / weight_total

    if read.divergence:
        read.score = 0.6 * read.score + 0.4 * read.divergence

    read.data_quality = _clip01(
        base_quality * (0.6 + 0.4 * min(len(window) / lookback, 1.0))
        + 0.25 * book_quality
    )

    magnitude = abs(read.score)
    if magnitude < 0.12:
        read.state = "NEUTRAL"
        read.confidence = 0.25
        reasons.append("buy and sell pressure are balanced")
    else:
        read.state = "BUY_PRESSURE" if read.score > 0 else "SELL_PRESSURE"
        # Certainty in the read, NOT discounted by data quality.  Every consumer
        # discounts by `data_quality` itself -- `ModelOutput.effective_confidence`
        # multiplies the two, and `compute_trade_quality` scales the order-flow
        # component by it as well -- so applying it here too meant the same
        # haircut landed twice, and three times on the way into trade quality.
        # With no aggressor tape (MEXC exposes none) base quality is 0.55, so a
        # full read scored 0.80: the double application turned that into 0.64
        # for no reason connected to the flow itself.
        read.confidence = _clip01(min(magnitude * 1.8, 1.0))
        reasons.append(
            f"net {'buying' if read.score > 0 else 'selling'} pressure "
            f"(score {read.score:+.2f})"
        )

    if read.data_quality < 0.2:
        read.state = "UNCERTAIN"
        read.confidence = 0.0

    read.reasoning = reasons
    return read


def _clip01(value: float) -> float:
    if value != value:
        return 0.0
    return 0.0 if value < 0 else 1.0 if value > 1 else value


__all__ = ["OrderFlowRead", "analyse_order_flow", "bar_delta", "book_imbalance"]
