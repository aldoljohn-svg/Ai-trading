"""Slope analysis.

A moving average tells you *where* price is; its slope tells you *what the
trend is doing right now*.  All slopes are computed with ordinary least squares
over a lookback window and then **normalised**, so a slope is comparable across
symbols priced at $0.42 and $64,000:

* price-based series (MA40/MA80/MA160/Average) are normalised by the series
  mean and expressed as *fraction of price per bar*;
* the RSI slope is normalised by the RSI scale (points per bar / 100).

The sign convention is the intuitive one: positive means rising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

Number = float | None


def linear_regression_slope(values: Sequence[float]) -> float:
    """OLS slope of ``values`` against ``0..n-1``.  Units: value per bar."""

    n = len(values)
    if n < 2:
        return 0.0
    # x is 0..n-1, so the sums have closed forms.
    sum_x = n * (n - 1) / 2.0
    sum_xx = (n - 1) * n * (2 * n - 1) / 6.0
    sum_y = 0.0
    sum_xy = 0.0
    for i, value in enumerate(values):
        sum_y += value
        sum_xy += i * value
    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denominator


def r_squared(values: Sequence[float]) -> float:
    """Goodness of fit of the linear trend - how *clean* the move is."""

    n = len(values)
    if n < 3:
        return 0.0
    slope = linear_regression_slope(values)
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    intercept = mean_y - slope * mean_x
    ss_total = sum((v - mean_y) ** 2 for v in values)
    if ss_total <= 0:
        return 0.0
    ss_residual = sum((v - (slope * i + intercept)) ** 2 for i, v in enumerate(values))
    return max(0.0, min(1.0, 1.0 - ss_residual / ss_total))


def _defined_tail(series: Sequence[Number], lookback: int) -> list[float]:
    values = [v for v in series if v is not None]
    return values[-lookback:] if len(values) >= 2 else []


def normalised_slope(
    series: Sequence[Number], lookback: int = 20, scale: float | None = None
) -> float:
    """Slope per bar, divided by ``scale`` (defaults to the window mean)."""

    window = _defined_tail(series, lookback)
    if len(window) < 2:
        return 0.0
    slope = linear_regression_slope(window)
    if scale is None:
        scale = sum(abs(v) for v in window) / len(window)
    if not scale:
        return 0.0
    return slope / scale


def slope_series(
    series: Sequence[Number], lookback: int = 20, scale: float | None = None
) -> list[Number]:
    """Rolling normalised slope aligned with the input series."""

    out: list[Number] = [None] * len(series)
    buffer: list[float] = []
    positions: list[int] = []
    for index, value in enumerate(series):
        if value is None:
            continue
        buffer.append(value)
        positions.append(index)
        if len(buffer) < lookback:
            continue
        window = buffer[-lookback:]
        window_scale = scale
        if window_scale is None:
            window_scale = sum(abs(v) for v in window) / len(window)
        if window_scale:
            out[index] = linear_regression_slope(window) / window_scale
    return out


@dataclass(slots=True)
class SlopeSet:
    """Normalised slopes of the project's core trend references."""

    rsi_slope: float
    ma40_slope: float
    ma80_slope: float
    ma160_slope: float
    average_slope: float
    price_slope: float
    trend_quality: float          # r^2 of the Average line, 0..1
    lookback: int

    @property
    def aligned_bullish(self) -> bool:
        return (
            self.ma40_slope > 0
            and self.ma80_slope > 0
            and self.ma160_slope >= 0
            and self.average_slope > 0
        )

    @property
    def aligned_bearish(self) -> bool:
        return (
            self.ma40_slope < 0
            and self.ma80_slope < 0
            and self.ma160_slope <= 0
            and self.average_slope < 0
        )

    @property
    def direction(self) -> int:
        if self.aligned_bullish:
            return 1
        if self.aligned_bearish:
            return -1
        return 0

    @property
    def strength(self) -> float:
        """0..1 blend of slope magnitude and linearity of the Average line."""

        magnitude = min(abs(self.average_slope) / 0.004, 1.0)
        return round(magnitude * (0.4 + 0.6 * self.trend_quality), 4)

    def as_features(self) -> dict[str, float]:
        return {
            "rsi_slope": self.rsi_slope,
            "ma40_slope": self.ma40_slope,
            "ma80_slope": self.ma80_slope,
            "ma160_slope": self.ma160_slope,
            "average_slope": self.average_slope,
            "price_slope": self.price_slope,
            "trend_quality": self.trend_quality,
            "slope_direction": float(self.direction),
            "slope_strength": self.strength,
        }


def compute_slopes(indicators, closes: Sequence[float], lookback: int = 20) -> SlopeSet:
    """Build a :class:`SlopeSet` from an :class:`~app.indicators.indicators.IndicatorSet`."""

    average_window = _defined_tail(indicators.average, lookback)
    return SlopeSet(
        # RSI lives on a fixed 0..100 scale, so normalise by 100 rather than by
        # its own mean - otherwise a low-RSI regime would exaggerate the slope.
        rsi_slope=normalised_slope(indicators.rsi, lookback, scale=100.0),
        ma40_slope=normalised_slope(indicators.ma40, lookback),
        ma80_slope=normalised_slope(indicators.ma80, lookback),
        ma160_slope=normalised_slope(indicators.ma160, lookback),
        average_slope=normalised_slope(indicators.average, lookback),
        price_slope=normalised_slope(list(closes), lookback),
        trend_quality=r_squared(average_window) if len(average_window) >= 3 else 0.0,
        lookback=lookback,
    )


__all__ = [
    "SlopeSet",
    "compute_slopes",
    "linear_regression_slope",
    "normalised_slope",
    "slope_series",
    "r_squared",
]
