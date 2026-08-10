"""Swing detection and price-action structure."""

from __future__ import annotations

import pytest

from app.domain import Bias, Candle
from app.indicators.indicators import atr, compute_indicators
from app.market_structure.structure import (
    SwingLabel,
    StructureEventType,
    analyse_structure,
    bias_from_labels,
    classify_leg,
    cluster_levels,
    detect_events,
    detect_fakeout,
    detect_rejection,
    label_swings,
)
from app.market_structure.swing_detector import (
    Swing,
    SwingType,
    alternate,
    detect_swings,
)
from tests.conftest import make_candles


def bars(closes: list[float], wick: float = 0.1, step: int = 3600) -> list[Candle]:
    """Bars whose extremes come from their *own* close.

    ``make_candles`` opens each bar at the previous close, which makes two
    adjacent bars share the same high at a turning point - and a strict fractal
    can never form on a tie.  Real candles rarely behave that way; these
    independent bars give clean, unambiguous swings for structure tests.
    """

    return [
        Candle(
            ts=index * step,
            open=close,
            high=close + wick,
            low=max(close - wick, 0.01),
            close=close,
            volume=100.0,
            quote_volume=100.0 * close,
        )
        for index, close in enumerate(closes)
    ]


def zigzag(pivots: list[float], steps: int = 6) -> list[Candle]:
    """A series that walks linearly between the given pivot prices."""

    prices: list[float] = [pivots[0]]
    for start, end in zip(pivots, pivots[1:]):
        for i in range(1, steps + 1):
            prices.append(start + (end - start) * i / steps)
    return bars(prices, wick=0.05)


class TestSwingDetection:
    def test_finds_an_obvious_peak(self):
        candles = bars([1, 2, 3, 4, 10, 4, 3, 2, 1])
        highs = [s for s in detect_swings(candles, strength=2, confirmed_only=False) if s.is_high]
        assert any(s.index == 4 for s in highs)

    def test_finds_an_obvious_trough(self):
        candles = bars([10, 9, 8, 7, 1, 7, 8, 9, 10])
        lows = [s for s in detect_swings(candles, strength=2, confirmed_only=False) if s.is_low]
        assert any(s.index == 4 for s in lows)

    def test_no_swings_in_a_monotonic_series(self):
        candles = bars([float(i) for i in range(30)])
        assert not [s for s in detect_swings(candles, strength=3) if s.is_high]

    def test_confirmation_requires_bars_to_the_right(self):
        """The core no-lookahead guarantee."""

        candles = bars([1, 2, 3, 10, 3, 2])
        # With strength 3 the peak at index 3 needs bars up to index 6, which
        # do not exist, so nothing may be reported.
        assert detect_swings(candles, strength=3, confirmed_only=True) == []

    def test_confirmed_at_is_recorded(self):
        candles = bars([1, 2, 3, 10, 3, 2, 1, 0.2, 0.5])
        swings = detect_swings(candles, strength=2, confirmed_only=True)
        for swing in swings:
            assert swing.confirmed_at == swing.index + swing.strength
            assert swing.confirmed_at <= len(candles) - 1

    def test_alternate_collapses_same_type_runs(self):
        swings = [
            Swing(0, 0, 10.0, SwingType.HIGH, 2, 2),
            Swing(2, 2, 12.0, SwingType.HIGH, 2, 4),
            Swing(4, 4, 5.0, SwingType.LOW, 2, 6),
        ]
        result = alternate(swings)
        assert len(result) == 2
        assert result[0].price == 12.0        # the more extreme high wins

    def test_atr_filter_drops_noise(self):
        candles = zigzag([100, 101, 100, 101, 100, 130], steps=4)
        values = atr(candles, 14)
        loose = detect_swings(candles, strength=2, atr_values=values, min_atr_excursion=0.0)
        strict = detect_swings(candles, strength=2, atr_values=values, min_atr_excursion=3.0)
        assert len(strict) <= len(loose)


class TestLabelling:
    def test_labels_higher_highs_and_higher_lows(self):
        swings = [
            Swing(0, 0, 10.0, SwingType.LOW, 2, 2),
            Swing(2, 2, 20.0, SwingType.HIGH, 2, 4),
            Swing(4, 4, 12.0, SwingType.LOW, 2, 6),
            Swing(6, 6, 25.0, SwingType.HIGH, 2, 8),
        ]
        labels = [label for _s, label in label_swings(swings)]
        assert labels[2] is SwingLabel.HL
        assert labels[3] is SwingLabel.HH

    def test_labels_lower_highs_and_lower_lows(self):
        swings = [
            Swing(0, 0, 30.0, SwingType.HIGH, 2, 2),
            Swing(2, 2, 20.0, SwingType.LOW, 2, 4),
            Swing(4, 4, 28.0, SwingType.HIGH, 2, 6),
            Swing(6, 6, 15.0, SwingType.LOW, 2, 8),
        ]
        labels = [label for _s, label in label_swings(swings)]
        assert labels[2] is SwingLabel.LH
        assert labels[3] is SwingLabel.LL

    def test_bias_from_uniform_labels(self):
        swings = [Swing(i, i, float(i), SwingType.HIGH, 2, i + 2) for i in range(4)]
        bullish = [(s, SwingLabel.HH) for s in swings]
        bearish = [(s, SwingLabel.LL) for s in swings]
        assert bias_from_labels(bullish) is Bias.BULLISH
        assert bias_from_labels(bearish) is Bias.BEARISH

    def test_conflict_when_both_hh_and_ll_present(self):
        swings = [Swing(i, i, float(i), SwingType.HIGH, 2, i + 2) for i in range(4)]
        mixed = [
            (swings[0], SwingLabel.HH),
            (swings[1], SwingLabel.LL),
            (swings[2], SwingLabel.HH),
            (swings[3], SwingLabel.LL),
        ]
        assert bias_from_labels(mixed) is Bias.CONFLICT

    def test_insufficient_history_is_neutral(self):
        assert bias_from_labels([]) is Bias.NEUTRAL


class TestStructureEvents:
    def test_bos_on_continuation(self):
        candles = zigzag([100, 120, 110, 140], steps=6)
        values = atr(candles, 14)
        swings = detect_swings(candles, strength=2, confirmed_only=True)
        events = detect_events(candles, swings, values)
        assert any(e.type is StructureEventType.BOS and e.is_bullish for e in events)

    def test_choch_when_structure_flips(self):
        # Up, up, then a decisive break of the last higher low.
        candles = zigzag([100, 120, 112, 135, 95], steps=6)
        values = atr(candles, 14)
        swings = detect_swings(candles, strength=2, confirmed_only=True)
        events = detect_events(candles, swings, values)
        kinds = {(e.type, e.direction) for e in events}
        assert any(
            kind in kinds
            for kind in (
                (StructureEventType.CHOCH, -1),
                (StructureEventType.MSS, -1),
            )
        )

    def test_mss_requires_displacement(self):
        candles = zigzag([100, 120, 112, 135, 95], steps=6)
        values = atr(candles, 14)
        swings = detect_swings(candles, strength=2, confirmed_only=True)
        gentle = detect_events(candles, swings, values, displacement_atr=99.0)
        assert not any(e.type is StructureEventType.MSS for e in gentle)

    def test_events_carry_confidence_in_range(self, candles_h1, indicators_h1):
        events = detect_events(
            candles_h1, detect_swings(candles_h1, strength=3), indicators_h1.atr
        )
        assert all(0.0 <= e.confidence <= 1.0 for e in events)


class TestLevelsAndPatterns:
    def test_clusters_nearby_swings(self):
        swings = [
            Swing(0, 0, 100.0, SwingType.HIGH, 2, 2),
            Swing(4, 4, 100.4, SwingType.HIGH, 2, 6),
            Swing(8, 8, 140.0, SwingType.HIGH, 2, 10),
        ]
        levels = cluster_levels(swings, "resistance", tolerance=1.0, total_bars=20)
        prices = sorted(round(level.price, 1) for level in levels)
        assert prices == [100.2, 140.0]
        strongest = max(levels, key=lambda level: level.touches)
        assert strongest.touches == 2

    def test_fakeout_detects_a_failed_upside_break(self):
        # A bar that *wicks* above the range and closes back inside, with price
        # still inside now - the textbook failed breakout.
        candles = bars([100.0] * 12, wick=0.2)
        candles[-3] = Candle(
            ts=candles[-3].ts, open=100, high=108, low=99.5, close=99.8, volume=100
        )
        assert detect_fakeout(candles, range_high=105, range_low=95, window=10) == 1

    def test_no_fakeout_when_the_break_held(self):
        candles = bars([100.0] * 10 + [106.0, 107.0], wick=0.2)
        assert detect_fakeout(candles, range_high=105, range_low=95, window=10) == 0

    def test_rejection_needs_a_long_wick(self):
        normal = bars([100, 100.5], wick=0.1)
        assert detect_rejection(normal, atr=1.0) == 0

        wick = list(normal)
        wick[-1] = Candle(
            ts=wick[-1].ts, open=100.0, high=100.2, low=95.0, close=100.1, volume=1
        )
        assert detect_rejection(wick, atr=1.0) == 1

    def test_impulse_versus_correction(self):
        impulse = bars([100 + i * 3 for i in range(30)], wick=0.1)
        swings = detect_swings(impulse, strength=2, confirmed_only=False) or [
            Swing(0, 0, 100.0, SwingType.LOW, 2, 2)
        ]
        kind, direction, size = classify_leg(impulse, swings, atr=1.0)
        assert kind == "impulse" and direction == 1 and size > 2


class TestAnalyseStructure:
    def test_full_analysis_on_real_series(self, candles_h1, indicators_h1):
        structure = analyse_structure(candles_h1, indicators_h1.atr)
        assert structure.swings
        assert structure.close == candles_h1[-1].close
        assert 0.0 <= structure.range_position <= 1.0
        assert 0.0 <= structure.trend_quality <= 1.0
        assert 0.0 <= structure.score() <= 100.0
        assert structure.range_low <= structure.close <= structure.range_high

    def test_trending_series_is_bullish(self):
        candles = zigzag([100, 120, 112, 140, 130, 165, 155, 190], steps=6)
        structure = analyse_structure(candles, atr(candles, 14))
        assert structure.bias in (Bias.BULLISH, Bias.NEUTRAL)
        assert structure.leg_direction >= 0

    def test_features_are_finite(self, candles_h1, indicators_h1):
        features = analyse_structure(candles_h1, indicators_h1.atr).as_features()
        assert all(v == v for v in features.values())

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            analyse_structure([], [])
