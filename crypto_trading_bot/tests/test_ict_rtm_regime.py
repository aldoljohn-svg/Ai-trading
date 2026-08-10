"""ICT, RTM and market-regime engines."""

from __future__ import annotations

import pytest

from app.domain import Bias, Candle, Regime
from app.ict.ict_engine import (
    analyse_ict,
    classify_zone,
    dealing_range,
    detect_displacements,
    detect_fair_value_gaps,
    detect_liquidity,
    detect_order_blocks,
    detect_sweeps,
)
from app.indicators.indicators import atr, compute_indicators
from app.indicators.slopes import compute_slopes
from app.market_structure.structure import analyse_structure
from app.market_structure.swing_detector import detect_swings
from app.regime.regime_detector import detect_regime, strategy_for_regime
from app.rtm.rtm_engine import (
    analyse_rtm,
    current_compression,
    detect_bases,
    detect_engulf,
)
from tests.test_market_structure import bars, zigzag


class TestFairValueGaps:
    def test_detects_a_bullish_gap(self):
        # Candle 3's low never overlaps candle 1's high.
        candles = [
            Candle(ts=0, open=100, high=101, low=99, close=100.5, volume=10),
            Candle(ts=3600, open=100.5, high=110, low=100.4, close=109, volume=10),
            Candle(ts=7200, open=109, high=112, low=105, close=111, volume=10),
        ] + bars([111.0] * 5)[1:]
        gaps = detect_fair_value_gaps(candles, atr(candles, 2), min_size_atr=0.0)
        bullish = [g for g in gaps if g.direction > 0]
        assert bullish
        assert bullish[0].bottom == pytest.approx(101)
        assert bullish[0].top == pytest.approx(105)

    def test_detects_a_bearish_gap(self):
        candles = [
            Candle(ts=0, open=110, high=112, low=109, close=110, volume=10),
            Candle(ts=3600, open=110, high=110, low=100, close=101, volume=10),
            Candle(ts=7200, open=101, high=105, low=99, close=100, volume=10),
        ] + bars([100.0] * 5)[1:]
        gaps = detect_fair_value_gaps(candles, atr(candles, 2), min_size_atr=0.0)
        assert any(g.direction < 0 for g in gaps)

    def test_no_gap_when_wicks_overlap(self):
        candles = bars([100, 101, 102, 103, 104], wick=2.0)
        assert detect_fair_value_gaps(candles, atr(candles, 2), min_size_atr=0.0) == []

    def test_fill_tracking_marks_stale_gaps(self):
        candles = [
            Candle(ts=0, open=100, high=101, low=99, close=100.5, volume=10),
            Candle(ts=3600, open=100.5, high=110, low=100.4, close=109, volume=10),
            Candle(ts=7200, open=109, high=112, low=105, close=111, volume=10),
        ] + bars([100.0] * 6)[1:]     # price trades all the way back down
        gaps = detect_fair_value_gaps(candles, atr(candles, 2), min_size_atr=0.0)
        bullish = [g for g in gaps if g.direction > 0]
        assert bullish and bullish[0].filled_pct > 0.9
        assert not bullish[0].is_fresh

    def test_confidence_is_bounded(self, candles_h1, indicators_h1):
        gaps = detect_fair_value_gaps(candles_h1, indicators_h1.atr)
        assert all(0.0 <= g.confidence <= 1.0 for g in gaps)


class TestDisplacementAndOrderBlocks:
    def test_displacement_needs_a_large_body(self):
        quiet = bars([100 + i * 0.01 for i in range(40)], wick=0.02)
        assert detect_displacements(quiet, atr(quiet, 14)) == []

    def test_order_block_is_the_last_opposing_candle(self):
        # ``bars`` produces doji-like candles (open == close), so the
        # displacement and its origin bar are constructed explicitly here.
        candles = bars([100.0] * 10, wick=0.3)
        candles.append(  # index 10: the bearish origin candle
            Candle(ts=10 * 3600, open=100.5, high=100.6, low=98.9, close=99.0, volume=10)
        )
        candles.append(  # index 11: the bullish displacement
            Candle(ts=11 * 3600, open=99.0, high=110.2, low=98.9, close=110.0, volume=10)
        )
        candles.extend(
            Candle(ts=(12 + i) * 3600, open=110.0, high=110.5, low=109.5, close=110.2, volume=10)
            for i in range(5)
        )
        values = atr(candles, 5)
        displacements = detect_displacements(candles, values, min_body_atr=0.5)
        assert any(d.index == 11 and d.direction > 0 for d in displacements)

        blocks = detect_order_blocks(candles, values, displacements)
        assert any(b.direction > 0 and b.index == 10 for b in blocks)

    def test_broken_block_becomes_a_breaker(self, candles_h1, indicators_h1):
        structure = analyse_structure(candles_h1, indicators_h1.atr)
        ict = analyse_ict(candles_h1, indicators_h1.atr, structure)
        for block in ict.order_blocks:
            assert block.is_breaker == block.broken


class TestLiquidity:
    def test_equal_highs_form_one_pool(self):
        candles = zigzag([100, 120, 105, 120.05, 100], steps=5)
        values = atr(candles, 5)
        swings = detect_swings(candles, strength=2)
        pools = detect_liquidity(candles, swings, values[-1] or 1.0, equal_tolerance_atr=5.0)
        assert any(p.equal and p.kind == "buy_side" for p in pools)

    def test_sweep_requires_a_close_back_inside(self):
        candles = bars([100.0] * 20, wick=0.2)
        # Wick below a prior low, closing back above it.
        candles[15] = Candle(ts=candles[15].ts, open=100, high=100.2, low=90, close=99.9, volume=10)
        values = atr(candles, 5)
        swings = detect_swings(candles, strength=2)
        pools = detect_liquidity(candles, swings, values[-1] or 1.0)
        sweeps = detect_sweeps(candles, pools, values)
        assert all(0.0 <= s.confidence <= 1.0 for s in sweeps)

    def test_premium_discount_classification(self):
        assert classify_zone(0.9) == "premium"
        assert classify_zone(0.1) == "discount"
        assert classify_zone(0.5) == "equilibrium"

    def test_dealing_range_contains_price(self, candles_h1, indicators_h1):
        structure = analyse_structure(candles_h1, indicators_h1.atr)
        high, low = dealing_range(candles_h1, structure.swings)
        assert high > low


class TestICTAnalysis:
    def test_full_analysis(self, candles_h1, indicators_h1):
        structure = analyse_structure(candles_h1, indicators_h1.atr)
        ict = analyse_ict(candles_h1, indicators_h1.atr, structure)
        assert 0.0 <= ict.premium_discount <= 1.0
        assert ict.zone in {"premium", "discount", "equilibrium"}
        assert 0.0 <= ict.score(1) <= 100.0
        assert 0.0 <= ict.score(-1) <= 100.0
        assert isinstance(ict.bias(), Bias)

    def test_features_are_finite(self, candles_h1, indicators_h1):
        structure = analyse_structure(candles_h1, indicators_h1.atr)
        ict = analyse_ict(candles_h1, indicators_h1.atr, structure)
        assert all(v == v for v in ict.as_features().values())

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            analyse_ict([], [], None)


class TestRTM:
    def test_bases_are_tight_by_construction(self):
        candles = bars([100 + (i % 2) * 0.1 for i in range(40)], wick=0.05)
        values = atr(candles, 5)
        bases = detect_bases(candles, values, max_height_atr=3.0)
        assert bases
        for base in bases:
            assert base.top >= base.bottom
            assert base.bars >= 2

    def test_engulfing_detection(self):
        candles = [
            Candle(ts=0, open=105, high=105.5, low=99, close=100, volume=10),
            Candle(ts=3600, open=99.5, high=107, low=99, close=106, volume=10),
        ]
        assert detect_engulf(candles, atr=1.0) == 1

        bearish = [
            Candle(ts=0, open=100, high=106, low=99, close=105, volume=10),
            Candle(ts=3600, open=105.5, high=106, low=98, close=99, volume=10),
        ]
        assert detect_engulf(bearish, atr=1.0) == -1

    def test_no_engulf_on_a_small_body(self):
        candles = [
            Candle(ts=0, open=100, high=101, low=99, close=100.1, volume=10),
            Candle(ts=3600, open=100.05, high=100.2, low=100, close=100.15, volume=10),
        ]
        assert detect_engulf(candles, atr=10.0) == 0

    def test_compression_is_bounded(self, candles_h1, indicators_h1):
        assert 0.0 <= current_compression(candles_h1, indicators_h1.atr) <= 1.0

    def test_full_analysis(self, candles_h1, indicators_h1):
        rtm = analyse_rtm(candles_h1, indicators_h1.atr)
        assert 0.0 <= rtm.score(1) <= 100.0
        assert isinstance(rtm.bias(), Bias)
        for pattern in rtm.patterns:
            assert pattern.kind in {"RBR", "DBD", "RBD", "DBR"}
            assert 0.0 <= pattern.confidence <= 1.0

    def test_zone_direction_matches_departure(self, candles_h1, indicators_h1):
        rtm = analyse_rtm(candles_h1, indicators_h1.atr)
        for pattern in rtm.patterns:
            expected = 1 if pattern.leg_out.kind == "rally" else -1
            assert pattern.zone.direction == expected


class TestRegime:
    def _read(self, candles):
        indicators = compute_indicators(candles)
        slopes = compute_slopes(indicators, [c.close for c in candles])
        structure = analyse_structure(candles, indicators.atr)
        return detect_regime(candles, indicators, slopes, structure)

    def test_uptrend_is_detected(self):
        candles = bars([100 + i * 0.6 for i in range(300)], wick=0.15)
        reading = self._read(candles)
        assert reading.regime is Regime.TREND_UP
        assert reading.direction == 1

    def test_downtrend_is_detected(self):
        candles = bars([300 - i * 0.6 for i in range(300)], wick=0.15)
        reading = self._read(candles)
        assert reading.regime is Regime.TREND_DOWN
        assert reading.direction == -1

    def test_insufficient_history_is_unknown(self):
        assert self._read(bars([100.0] * 20)).regime is Regime.UNKNOWN

    def test_high_volatility_blocks_entries(self):
        strategy = strategy_for_regime(Regime.HIGH_VOLATILITY)
        assert not strategy.allow_entries
        assert strategy.risk_multiplier == 0.0

    def test_unknown_regime_stands_aside(self):
        strategy = strategy_for_regime(Regime.UNKNOWN)
        assert not strategy.allow_entries

    def test_every_regime_has_a_strategy(self):
        for regime in Regime:
            strategy = strategy_for_regime(regime)
            assert strategy.name
            assert strategy.rationale
            assert 0.0 <= strategy.risk_multiplier <= 1.5

    def test_trending_regimes_forbid_counter_trend(self):
        assert not strategy_for_regime(Regime.TREND_UP).allow_counter_trend
        assert not strategy_for_regime(Regime.TREND_DOWN).allow_counter_trend
        assert strategy_for_regime(Regime.RANGE).allow_counter_trend

    def test_features_are_finite(self, candles_h1, indicators_h1):
        slopes = compute_slopes(indicators_h1, [c.close for c in candles_h1])
        structure = analyse_structure(candles_h1, indicators_h1.atr)
        reading = detect_regime(candles_h1, indicators_h1, slopes, structure)
        assert all(v == v for v in reading.as_features().values())
