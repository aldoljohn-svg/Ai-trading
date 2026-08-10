"""Technical indicators.

Every function returns a list the *same length* as the input, padded with
``None`` where the indicator is not yet defined.  Keeping the alignment
explicit removes a whole class of off-by-one bugs when several indicators are
combined, and makes it impossible to accidentally read a value that depends on
bars that did not exist yet (a common source of lookahead bias).

Wilder's smoothing is used for RSI, ATR and ADX, matching what charting
platforms display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from app.domain import Candle, Candles

Number = float | None


# --------------------------------------------------------------------------
# Moving averages
# --------------------------------------------------------------------------


def sma(values: Sequence[float], period: int) -> list[Number]:
    """Simple moving average."""

    if period <= 0:
        raise ValueError("period must be positive")
    out: list[Number] = [None] * len(values)
    if len(values) < period:
        return out
    window = sum(values[:period])
    out[period - 1] = window / period
    for i in range(period, len(values)):
        window += values[i] - values[i - period]
        out[i] = window / period
    return out


def ema(values: Sequence[float], period: int) -> list[Number]:
    """Exponential moving average seeded with the first SMA."""

    if period <= 0:
        raise ValueError("period must be positive")
    out: list[Number] = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    current = sum(values[:period]) / period
    out[period - 1] = current
    for i in range(period, len(values)):
        current = values[i] * alpha + current * (1 - alpha)
        out[i] = current
    return out


def wilder_smooth(values: Sequence[float], period: int) -> list[Number]:
    """Wilder's RMA - an EMA with ``alpha = 1 / period``."""

    out: list[Number] = [None] * len(values)
    if len(values) < period or period <= 0:
        return out
    current = sum(values[:period]) / period
    out[period - 1] = current
    for i in range(period, len(values)):
        current = (current * (period - 1) + values[i]) / period
        out[i] = current
    return out


# --------------------------------------------------------------------------
# RSI / ATR
# --------------------------------------------------------------------------


def rsi(values: Sequence[float], period: int = 14) -> list[Number]:
    """Relative Strength Index (Wilder)."""

    out: list[Number] = [None] * len(values)
    if len(values) <= period or period <= 0:
        return out

    gains: list[float] = []
    losses: list[float] = []
    for previous, current in zip(values, values[1:]):
        change = current - previous
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out[period] = _rsi_value(avg_gain, avg_loss)

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(candles: Candles) -> list[Number]:
    out: list[Number] = [None] * len(candles)
    if not candles:
        return out
    out[0] = candles[0].high - candles[0].low
    for i in range(1, len(candles)):
        current, previous = candles[i], candles[i - 1]
        out[i] = max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        )
    return out


def atr(candles: Candles, period: int = 14) -> list[Number]:
    """Average True Range (Wilder)."""

    ranges = true_range(candles)
    values = [r for r in ranges if r is not None]
    if len(values) < period:
        return [None] * len(candles)
    smoothed = wilder_smooth(values, period)
    # ``true_range`` is defined from index 0, so alignment is 1:1.
    return list(smoothed)


# --------------------------------------------------------------------------
# MACD / ADX / VWAP / Bollinger
# --------------------------------------------------------------------------


def macd(
    values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list[Number], list[Number], list[Number]]:
    """Returns ``(macd_line, signal_line, histogram)``."""

    fast_ema = ema(values, fast)
    slow_ema = ema(values, slow)
    line: list[Number] = [
        (f - s) if (f is not None and s is not None) else None
        for f, s in zip(fast_ema, slow_ema)
    ]
    defined = [v for v in line if v is not None]
    signal_line: list[Number] = [None] * len(values)
    if len(defined) >= signal:
        offset = len(line) - len(defined)
        smoothed = ema(defined, signal)
        for i, value in enumerate(smoothed):
            signal_line[offset + i] = value
    histogram: list[Number] = [
        (m - s) if (m is not None and s is not None) else None
        for m, s in zip(line, signal_line)
    ]
    return line, signal_line, histogram


def adx(candles: Candles, period: int = 14) -> tuple[list[Number], list[Number], list[Number]]:
    """Returns ``(adx, plus_di, minus_di)`` using Wilder smoothing."""

    size = len(candles)
    empty: list[Number] = [None] * size
    if size < period * 2 + 1:
        return empty, list(empty), list(empty)

    plus_dm: list[float] = [0.0]
    minus_dm: list[float] = [0.0]
    for i in range(1, size):
        up = candles[i].high - candles[i - 1].high
        down = candles[i - 1].low - candles[i].low
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)

    ranges = [r if r is not None else 0.0 for r in true_range(candles)]

    smooth_tr = wilder_smooth(ranges, period)
    smooth_plus = wilder_smooth(plus_dm, period)
    smooth_minus = wilder_smooth(minus_dm, period)

    plus_di: list[Number] = [None] * size
    minus_di: list[Number] = [None] * size
    dx: list[float] = []
    dx_index: list[int] = []

    for i in range(size):
        tr_value = smooth_tr[i]
        if tr_value is None or tr_value == 0:
            continue
        p = smooth_plus[i]
        m = smooth_minus[i]
        if p is None or m is None:
            continue
        pdi = 100.0 * p / tr_value
        mdi = 100.0 * m / tr_value
        plus_di[i] = pdi
        minus_di[i] = mdi
        total = pdi + mdi
        if total > 0:
            dx.append(100.0 * abs(pdi - mdi) / total)
            dx_index.append(i)

    adx_values: list[Number] = [None] * size
    if len(dx) >= period:
        smoothed = wilder_smooth(dx, period)
        for position, value in enumerate(smoothed):
            if value is not None:
                adx_values[dx_index[position]] = value
    return adx_values, plus_di, minus_di


def vwap(candles: Candles, period: int | None = None) -> list[Number]:
    """Volume weighted average price - rolling when ``period`` is given."""

    size = len(candles)
    out: list[Number] = [None] * size
    if size == 0:
        return out
    if period is None:
        cum_pv = 0.0
        cum_v = 0.0
        for i, candle in enumerate(candles):
            cum_pv += candle.typical * candle.volume
            cum_v += candle.volume
            out[i] = cum_pv / cum_v if cum_v > 0 else candle.close
        return out

    for i in range(period - 1, size):
        window = candles[i - period + 1 : i + 1]
        volume_sum = sum(c.volume for c in window)
        if volume_sum > 0:
            out[i] = sum(c.typical * c.volume for c in window) / volume_sum
        else:
            out[i] = sum(c.typical for c in window) / period
    return out


def bollinger(
    values: Sequence[float], period: int = 20, deviations: float = 2.0
) -> tuple[list[Number], list[Number], list[Number]]:
    """Returns ``(upper, middle, lower)``."""

    middle = sma(values, period)
    upper: list[Number] = [None] * len(values)
    lower: list[Number] = [None] * len(values)
    for i in range(period - 1, len(values)):
        mean = middle[i]
        if mean is None:
            continue
        window = values[i - period + 1 : i + 1]
        variance = sum((v - mean) ** 2 for v in window) / period
        std = math.sqrt(variance)
        upper[i] = mean + deviations * std
        lower[i] = mean - deviations * std
    return upper, middle, lower


def bollinger_bandwidth(
    upper: Sequence[Number], middle: Sequence[Number], lower: Sequence[Number]
) -> list[Number]:
    out: list[Number] = []
    for u, m, l in zip(upper, middle, lower):
        if u is None or m is None or l is None or m == 0:
            out.append(None)
        else:
            out.append((u - l) / m)
    return out


def volume_profile_metrics(candles: Candles, lookback: int = 20) -> dict[str, float]:
    """Relative-volume statistics used by the scanner and scoring."""

    if len(candles) < lookback + 1:
        return {
            "relative_volume": 1.0,
            "volume_trend": 0.0,
            "buy_pressure": 0.5,
            "quote_volume": 0.0,
        }
    window = candles[-lookback:]
    recent = window[-1]
    average = sum(c.volume for c in window[:-1]) / max(len(window) - 1, 1)
    relative = recent.volume / average if average > 0 else 1.0

    first_half = window[: lookback // 2]
    second_half = window[lookback // 2 :]
    first_sum = sum(c.volume for c in first_half) or 1e-12
    second_sum = sum(c.volume for c in second_half)
    volume_trend = (second_sum - first_sum) / first_sum

    # Approximate buy pressure from where each bar closed within its range.
    weighted = 0.0
    total_volume = 0.0
    for candle in window:
        span = candle.high - candle.low
        location = 0.5 if span <= 0 else (candle.close - candle.low) / span
        weighted += location * candle.volume
        total_volume += candle.volume
    buy_pressure = weighted / total_volume if total_volume > 0 else 0.5

    return {
        "relative_volume": relative,
        "volume_trend": volume_trend,
        "buy_pressure": buy_pressure,
        "quote_volume": sum(c.quote_volume or c.volume * c.close for c in window),
    }


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------


@dataclass(slots=True)
class IndicatorSet:
    """All indicator values for one symbol/timeframe, plus latest readings."""

    rsi: list[Number]
    atr: list[Number]
    ma40: list[Number]
    ma80: list[Number]
    ma160: list[Number]
    average: list[Number]
    ema21: list[Number]
    macd_line: list[Number]
    macd_signal: list[Number]
    macd_hist: list[Number]
    adx: list[Number]
    plus_di: list[Number]
    minus_di: list[Number]
    vwap: list[Number]
    bb_upper: list[Number]
    bb_middle: list[Number]
    bb_lower: list[Number]
    bb_bandwidth: list[Number]
    volume: dict[str, float]
    close: float
    atr_pct: float

    # -- latest helpers ---------------------------------------------------

    @staticmethod
    def _last(series: Sequence[Number]) -> float | None:
        for value in reversed(series):
            if value is not None:
                return value
        return None

    @property
    def last_rsi(self) -> float | None:
        return self._last(self.rsi)

    @property
    def last_atr(self) -> float | None:
        return self._last(self.atr)

    @property
    def last_ma40(self) -> float | None:
        return self._last(self.ma40)

    @property
    def last_ma80(self) -> float | None:
        return self._last(self.ma80)

    @property
    def last_ma160(self) -> float | None:
        return self._last(self.ma160)

    @property
    def last_average(self) -> float | None:
        return self._last(self.average)

    @property
    def last_adx(self) -> float | None:
        return self._last(self.adx)

    @property
    def last_macd_hist(self) -> float | None:
        return self._last(self.macd_hist)

    @property
    def last_bandwidth(self) -> float | None:
        return self._last(self.bb_bandwidth)

    @property
    def ma_stack(self) -> int:
        """+1 fully bullish stack, -1 fully bearish stack, 0 mixed."""

        ma40, ma80, ma160 = self.last_ma40, self.last_ma80, self.last_ma160
        if None in (ma40, ma80, ma160):
            return 0
        if self.close > ma40 > ma80 > ma160:
            return 1
        if self.close < ma40 < ma80 < ma160:
            return -1
        return 0

    def as_features(self) -> dict[str, float]:
        """Flat numeric view used by the ML feature builder and the audit log."""

        def value(x: float | None, default: float = 0.0) -> float:
            return float(x) if x is not None else default

        close = self.close or 1.0
        return {
            "rsi": value(self.last_rsi, 50.0),
            "atr_pct": self.atr_pct,
            "ma40_dist": (close - value(self.last_ma40, close)) / close,
            "ma80_dist": (close - value(self.last_ma80, close)) / close,
            "ma160_dist": (close - value(self.last_ma160, close)) / close,
            "average_dist": (close - value(self.last_average, close)) / close,
            "ma_stack": float(self.ma_stack),
            "adx": value(self.last_adx, 0.0),
            "macd_hist_norm": value(self.last_macd_hist, 0.0) / close,
            "bb_bandwidth": value(self.last_bandwidth, 0.0),
            "relative_volume": self.volume.get("relative_volume", 1.0),
            "volume_trend": self.volume.get("volume_trend", 0.0),
            "buy_pressure": self.volume.get("buy_pressure", 0.5),
        }


def compute_indicators(
    candles: Candles,
    rsi_period: int = 14,
    atr_period: int = 14,
    ma_periods: tuple[int, int, int] = (40, 80, 160),
) -> IndicatorSet:
    """Compute the full indicator bundle for a candle series.

    The MA40/MA80/MA160 triple and their *average* are the project's core trend
    reference; everything else is supporting context.
    """

    if not candles:
        raise ValueError("cannot compute indicators on an empty series")

    close_values = [c.close for c in candles]
    p40, p80, p160 = ma_periods

    ma40 = sma(close_values, p40)
    ma80 = sma(close_values, p80)
    ma160 = sma(close_values, p160)
    average: list[Number] = []
    for a, b, c in zip(ma40, ma80, ma160):
        if a is None or b is None or c is None:
            average.append(None)
        else:
            average.append((a + b + c) / 3.0)

    macd_line, macd_signal, macd_hist = macd(close_values)
    adx_values, plus_di, minus_di = adx(candles, 14)
    bb_upper, bb_middle, bb_lower = bollinger(close_values, 20, 2.0)
    atr_values = atr(candles, atr_period)

    last_close = close_values[-1]
    last_atr = next((v for v in reversed(atr_values) if v is not None), None)
    atr_pct = (last_atr / last_close) if (last_atr and last_close > 0) else 0.0

    return IndicatorSet(
        rsi=rsi(close_values, rsi_period),
        atr=atr_values,
        ma40=ma40,
        ma80=ma80,
        ma160=ma160,
        average=average,
        ema21=ema(close_values, 21),
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        adx=adx_values,
        plus_di=plus_di,
        minus_di=minus_di,
        vwap=vwap(candles, period=20),
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        bb_bandwidth=bollinger_bandwidth(bb_upper, bb_middle, bb_lower),
        volume=volume_profile_metrics(candles),
        close=last_close,
        atr_pct=atr_pct,
    )


__all__ = [
    "IndicatorSet",
    "compute_indicators",
    "sma",
    "ema",
    "wilder_smooth",
    "rsi",
    "atr",
    "true_range",
    "macd",
    "adx",
    "vwap",
    "bollinger",
    "bollinger_bandwidth",
    "volume_profile_metrics",
]
