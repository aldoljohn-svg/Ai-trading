"""Indicators and slopes - verified against hand-computable cases."""

from __future__ import annotations

import math

import pytest

from app.indicators.indicators import (
    adx,
    atr,
    bollinger,
    compute_indicators,
    ema,
    macd,
    rsi,
    sma,
    true_range,
    volume_profile_metrics,
    vwap,
    wilder_smooth,
)
from app.indicators.slopes import (
    compute_slopes,
    linear_regression_slope,
    normalised_slope,
    r_squared,
    slope_series,
)
from tests.conftest import make_candles


class TestMovingAverages:
    def test_sma_known_values(self):
        result = sma([1, 2, 3, 4, 5], 3)
        assert result[:2] == [None, None]
        assert result[2] == pytest.approx(2.0)
        assert result[4] == pytest.approx(4.0)

    def test_sma_alignment_is_one_to_one(self):
        values = list(range(50))
        assert len(sma(values, 10)) == len(values)

    def test_sma_too_short_is_all_none(self):
        assert sma([1, 2], 5) == [None, None]

    def test_ema_seeded_with_sma_then_weights_recent(self):
        values = [1.0] * 10 + [2.0] * 10
        result = ema(values, 5)
        assert result[4] == pytest.approx(1.0)
        assert result[-1] > result[10]
        assert result[-1] < 2.0

    def test_ema_of_constant_is_constant(self):
        result = ema([5.0] * 30, 10)
        assert result[-1] == pytest.approx(5.0)

    def test_wilder_smoothing_is_slower_than_ema(self):
        values = [1.0] * 10 + [10.0] * 10
        assert wilder_smooth(values, 5)[-1] < ema(values, 5)[-1]

    def test_rejects_bad_period(self):
        with pytest.raises(ValueError):
            sma([1, 2, 3], 0)


class TestRSI:
    def test_all_gains_is_100(self):
        assert rsi([float(i) for i in range(1, 40)], 14)[-1] == pytest.approx(100.0)

    def test_all_losses_is_0(self):
        assert rsi([float(i) for i in range(40, 1, -1)], 14)[-1] == pytest.approx(0.0)

    def test_flat_series_is_neutral(self):
        assert rsi([10.0] * 40, 14)[-1] == pytest.approx(50.0)

    def test_bounded_and_aligned(self):
        values = [10 + math.sin(i / 3) * 5 for i in range(120)]
        result = rsi(values, 14)
        assert len(result) == len(values)
        assert result[13] is None and result[14] is not None
        assert all(0 <= v <= 100 for v in result if v is not None)


class TestATR:
    def test_true_range_uses_previous_close(self):
        candles = make_candles([100, 110], spread=0.0)
        # Second bar: high 110, low 100, previous close 100 -> range 10.
        assert true_range(candles)[1] == pytest.approx(10.0)

    def test_atr_of_constant_range_equals_that_range(self):
        candles = make_candles([100.0] * 40, spread=1.0)
        assert atr(candles, 14)[-1] == pytest.approx(2.0, rel=0.05)

    def test_atr_is_positive_and_aligned(self, candles_h1):
        result = atr(candles_h1, 14)
        assert len(result) == len(candles_h1)
        assert all(v > 0 for v in result if v is not None)


class TestOtherIndicators:
    def test_macd_histogram_is_line_minus_signal(self):
        values = [100 + i * 0.5 for i in range(120)]
        line, signal, hist = macd(values)
        index = next(i for i, v in enumerate(hist) if v is not None)
        assert hist[index] == pytest.approx(line[index] - signal[index])

    def test_adx_directional_indicators_follow_the_trend(self):
        up = make_candles([100 + i for i in range(120)], spread=0.5)
        down = make_candles([220 - i for i in range(120)], spread=0.5)

        adx_up, plus_up, minus_up = adx(up)
        adx_down, plus_down, minus_down = adx(down)

        assert plus_up[-1] > minus_up[-1]      # uptrend: +DI dominates
        assert minus_down[-1] > plus_down[-1]  # downtrend: -DI dominates
        assert adx_up[-1] > 25 and adx_down[-1] > 25

    def test_adx_is_lower_on_a_random_walk_than_on_a_clean_trend(self):
        import random

        rng = random.Random(7)
        walk = [100.0]
        for _ in range(200):
            walk.append(max(1.0, walk[-1] * (1 + rng.gauss(0, 0.01))))
        trending = make_candles([100 + i for i in range(200)], spread=0.5)
        assert (adx(make_candles(walk, spread=0.5))[0][-1] or 0) < (
            adx(trending)[0][-1] or 0
        )

    def test_bollinger_ordering(self, candles_h1):
        upper, middle, lower = bollinger([c.close for c in candles_h1], 20, 2.0)
        assert upper[-1] > middle[-1] > lower[-1]

    def test_vwap_sits_inside_the_range(self, candles_h1):
        result = vwap(candles_h1, period=20)
        window = candles_h1[-20:]
        assert min(c.low for c in window) <= result[-1] <= max(c.high for c in window)

    def test_volume_metrics_bounds(self, candles_h1):
        metrics = volume_profile_metrics(candles_h1)
        assert metrics["relative_volume"] > 0
        assert 0.0 <= metrics["buy_pressure"] <= 1.0


class TestIndicatorSet:
    def test_computes_the_core_reference(self, indicators_h1):
        assert indicators_h1.last_rsi is not None
        assert indicators_h1.last_atr > 0
        assert indicators_h1.last_ma40 is not None
        assert indicators_h1.last_ma80 is not None
        assert indicators_h1.last_ma160 is not None
        assert indicators_h1.last_average == pytest.approx(
            (indicators_h1.last_ma40 + indicators_h1.last_ma80 + indicators_h1.last_ma160) / 3
        )

    def test_ma_stack_detects_alignment(self):
        rising = make_candles([100 + i * 0.5 for i in range(300)], spread=0.2)
        falling = make_candles([250 - i * 0.5 for i in range(300)], spread=0.2)
        assert compute_indicators(rising).ma_stack == 1
        assert compute_indicators(falling).ma_stack == -1

    def test_atr_pct_is_relative(self, indicators_h1):
        assert 0 < indicators_h1.atr_pct < 1

    def test_features_are_finite(self, indicators_h1):
        for key, value in indicators_h1.as_features().items():
            assert isinstance(value, float)
            assert value == value, key

    def test_empty_series_raises(self):
        with pytest.raises(ValueError):
            compute_indicators([])


class TestSlopes:
    def test_regression_slope_exact(self):
        assert linear_regression_slope([0, 1, 2, 3, 4]) == pytest.approx(1.0)
        assert linear_regression_slope([4, 3, 2, 1, 0]) == pytest.approx(-1.0)
        assert linear_regression_slope([5, 5, 5]) == pytest.approx(0.0)

    def test_r_squared_of_a_line_is_one(self):
        assert r_squared([0, 1, 2, 3, 4]) == pytest.approx(1.0)

    def test_r_squared_of_noise_is_low(self):
        assert r_squared([1, 0, 1, 0, 1, 0, 1, 0]) < 0.2

    def test_normalisation_makes_slopes_comparable(self):
        cheap = [1.0 + i * 0.01 for i in range(40)]
        expensive = [x * 1000 for x in cheap]
        assert normalised_slope(cheap, 20) == pytest.approx(
            normalised_slope(expensive, 20), rel=1e-6
        )

    def test_slope_series_is_aligned(self):
        values = [float(i) for i in range(60)]
        result = slope_series(values, 20)
        assert len(result) == len(values)
        assert result[18] is None and result[-1] is not None

    def test_slope_set_direction(self):
        rising = make_candles([100 + i * 0.5 for i in range(300)], spread=0.2)
        indicators = compute_indicators(rising)
        slopes = compute_slopes(indicators, [c.close for c in rising])
        assert slopes.direction == 1
        assert slopes.aligned_bullish
        assert slopes.trend_quality > 0.9
        assert 0.0 <= slopes.strength <= 1.0

    def test_rsi_slope_scaled_by_rsi_range(self, indicators_h1, candles_h1):
        slopes = compute_slopes(indicators_h1, [c.close for c in candles_h1])
        # An RSI slope normalised by 100 can never be a huge number.
        assert abs(slopes.rsi_slope) < 1.0
