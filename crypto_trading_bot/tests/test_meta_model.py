"""Meta-labelling, and the feature-namespace bug it exposed.

The direction model could not beat a constant predictor on real 15m data. While
reframing the problem it turned out the ML path was also silently dead at
inference: the trainer emitted bare feature names (``rsi``) while the live
scanner emits timeframe-prefixed ones (``15m_rsi``), and ``FeatureSpec`` fills
anything it cannot find with the training mean. Every symbol therefore produced
an identical all-zero vector and an identical "prediction".

Nothing raised. The tests below exist so that cannot happen again.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from app.domain import Bias, Candle, Side, Timeframe
from app.ml.features import FeatureSpec
from app.ml.meta import (
    LABEL_LOSS,
    LABEL_WIN,
    SIDE_FEATURE,
    MetaStats,
    analyse_window,
    barrier_outcomes,
    build_meta_dataset,
    meta_class_distribution,
)
from app.ml.model_registry import ModelArtifact, ModelRegistry
from app.ml.predict import Predictor
from app.ml.train import _decile_lift, expectancy_r, train_model
from app.scanner.scanner import directional_bias

from tests.conftest import make_candles


def trending(n: int = 500, start: float = 100.0, step: float = 0.25) -> list[Candle]:
    """An uptrend with pullbacks, and enough history for the 400-bar window.

    The oscillation matters.  A perfectly linear ramp has no swing points, so
    the structure and ICT reads produce no bias and only the MA stack votes --
    which is correctly reported as NEUTRAL, and yields no meta rows at all.
    A real trend retraces.
    """

    return make_candles(
        [start + i * step + 3.0 * math.sin(i / 9.0) for i in range(n)], step=900
    )


def choppy(n: int = 500, start: float = 100.0) -> list[Candle]:
    prices = [start + (3.0 if i % 2 else -3.0) for i in range(n)]
    return make_candles(prices, step=900)


# ===========================================================================
# the namespace bug
# ===========================================================================


class TestFeatureNamespace:
    def test_coverage_reports_the_overlap(self):
        spec = FeatureSpec.fit([{"a": 1.0, "b": 2.0, "c": 3.0}])
        assert spec.coverage({"a": 1.0, "b": 2.0, "c": 3.0}) == pytest.approx(1.0)
        assert spec.coverage({"a": 1.0}) == pytest.approx(1 / 3)
        assert spec.coverage({"x": 1.0}) == 0.0

    def test_an_empty_spec_has_no_coverage(self):
        assert FeatureSpec.fit([{}]).coverage({"a": 1.0}) == 0.0

    def test_prefix_reaches_every_feature(self):
        bare, _ = analyse_window(trending())
        prefixed, _ = analyse_window(trending(), prefix="15m_")
        assert bare
        assert len(prefixed) == len(bare)
        assert all(name.startswith("15m_") for name in prefixed)
        assert {f"15m_{n}" for n in bare} == set(prefixed)

    def test_training_names_match_the_live_scanner(self):
        """The regression test for the bug that made every prediction identical.

        If SymbolAnalysis.features() ever changes how it namespaces a timeframe,
        this fails here rather than silently producing a useless model.
        """

        from app.data.market_data import MarketData
        from app.exchange.synthetic import SyntheticExchange
        from app.scanner.scanner import Scanner
        from app.scanner.universe import UniverseBuilder

        async def run() -> dict[str, float]:
            exchange = SyntheticExchange()
            await exchange.connect()
            universe = UniverseBuilder(
                min_quote_volume=1000.0, max_spread_pct=0.01, max_symbols=4
            )
            scanner = Scanner(MarketData(exchange), universe, deep_analysis_count=1)
            analyses = await scanner.deep_analyse(await scanner.prescreen())
            await exchange.close()
            return analyses[0].features()

        live = asyncio.run(run())
        trained, _ = analyse_window(trending(), prefix="15m_")

        overlap = set(trained) & set(live)
        assert overlap, (
            "training features share no names with the live vector - a model "
            "trained this way returns a constant for every symbol"
        )
        # Every trained name must be live; extra live features are fine.
        missing = set(trained) - set(live)
        assert not missing, f"training emits names the scanner never sends: {sorted(missing)[:5]}"


class TestCoverageGuard:
    def _artifact(self, names: list[str]) -> ModelArtifact:
        spec = FeatureSpec.fit([{n: 1.0 for n in names}])
        artifact = ModelArtifact(
            name="t", version="1", algorithm="stub", trained_at=0,
            rows=100, feature_spec=spec, model_payload={},
        )
        artifact.attach_runtime(_ConstantModel())
        return artifact

    def test_a_mismatched_row_is_refused_not_answered(self):
        predictor = Predictor(artifact=self._artifact(["rsi", "adx", "atr_pct"]))
        assert predictor.ready
        result = predictor.predict({"15m_rsi": 50.0, "15m_adx": 20.0})
        assert result == (0.0, 0.0, 1.0), "must abstain rather than emit a constant"
        assert not predictor.feature_coverage_ok

    def test_a_matching_row_is_answered(self):
        predictor = Predictor(artifact=self._artifact(["rsi", "adx", "atr_pct"]))
        result = predictor.predict({"rsi": 50.0, "adx": 20.0, "atr_pct": 0.01})
        assert result != (0.0, 0.0, 1.0)
        assert predictor.feature_coverage_ok

    def test_one_missing_feature_is_tolerated(self):
        """A single absent value is normal; only wholesale mismatch is fatal."""

        predictor = Predictor(artifact=self._artifact(["a", "b", "c", "d"]))
        assert predictor.predict({"a": 1.0, "b": 1.0, "c": 1.0}) != (0.0, 0.0, 1.0)

    def test_the_meta_path_is_guarded_too(self):
        artifact = self._artifact(["rsi", "adx", "atr_pct"])
        artifact.kind = "meta"
        predictor = Predictor(artifact=artifact)
        assert predictor.predict_meta({"15m_rsi": 50.0}, side=1) is None


class _ConstantModel:
    def predict_proba(self, rows):
        return [[0.3, 0.4, 0.3] for _ in rows]


# ===========================================================================
# labelling
# ===========================================================================


class TestBarrierOutcomes:
    def _atr(self, n: int, value: float = 1.0) -> list[float]:
        return [value] * n

    def test_a_clean_rally_wins_long_and_loses_short(self):
        candles = make_candles([100.0 + i for i in range(40)], step=900)
        outcomes = barrier_outcomes(candles, self._atr(len(candles)), horizon=10)
        long_won, short_won = outcomes[0]
        assert long_won is True
        assert short_won is False

    def test_a_bar_spanning_both_barriers_is_resolved_pessimistically(self):
        """Optimism here is how a backtest stops reproducing in live trading."""

        candles = [
            Candle(ts=0, open=100, high=100.5, low=99.5, close=100, volume=1),
            # This bar reaches +3 ATR and -2 ATR; both barriers are inside it.
            Candle(ts=900, open=100, high=103, low=98, close=101, volume=1),
            Candle(ts=1800, open=101, high=101.5, low=100.5, close=101, volume=1),
        ]
        outcomes = barrier_outcomes(candles, self._atr(3), horizon=2)
        long_won, short_won = outcomes[0]
        assert long_won is False, "the stop must be assumed to have hit first"
        assert short_won is False

    def test_reaching_the_vertical_barrier_is_not_a_win(self):
        flat = make_candles([100.0] * 30, step=900, spread=0.01)
        outcomes = barrier_outcomes(flat, self._atr(30), horizon=10)
        assert outcomes[0] == (False, False)

    def test_bars_without_a_full_horizon_are_unresolved(self):
        candles = make_candles([100.0 + i for i in range(20)], step=900)
        outcomes = barrier_outcomes(candles, self._atr(20), horizon=10)
        assert outcomes[-1] == (None, None)


class TestMetaDataset:
    def test_rows_are_binary_and_carry_the_side(self):
        stats = MetaStats()
        samples = build_meta_dataset(
            "BTCUSDT", Timeframe.M15, trending(600),
            warmup=400, stride=8, stats=stats, prefix="15m_",
        )
        assert samples
        assert {s.label for s in samples} <= {LABEL_LOSS, LABEL_WIN}
        assert all(SIDE_FEATURE in s.features for s in samples)
        assert all(abs(s.features[SIDE_FEATURE]) == 1.0 for s in samples)

    def test_bars_with_no_directional_read_are_dropped(self):
        """The filter is the point: structureless bars teach nothing."""

        stats = MetaStats()
        build_meta_dataset(
            "X", Timeframe.M15, choppy(600),
            warmup=400, stride=4, stats=stats, prefix="15m_",
        )
        assert stats.bars_examined > 0
        assert stats.no_direction > 0
        assert stats.selectivity > 0

    def test_the_stats_explain_a_thin_dataset(self):
        stats = MetaStats()
        build_meta_dataset(
            "X", Timeframe.M15, trending(600),
            warmup=400, stride=8, stats=stats, prefix="15m_",
        )
        text = stats.summary()
        assert "rows from" in text
        assert "base win rate" in text
        assert stats.rows == stats.long_rows + stats.short_rows

    def test_a_short_series_yields_nothing(self):
        assert build_meta_dataset("X", Timeframe.M15, trending(50)) == []

    def test_class_distribution_uses_win_loss_names(self):
        samples = build_meta_dataset(
            "BTCUSDT", Timeframe.M15, trending(600),
            warmup=400, stride=8, prefix="15m_",
        )
        distribution = meta_class_distribution(samples)
        assert set(distribution) == {"WIN", "LOSS"}


class TestSharedDirection:
    def test_the_trainer_and_the_scanner_agree(self):
        """One definition of "the rules say LONG", used by both."""

        from app.ict.ict_engine import analyse_ict
        from app.indicators.indicators import compute_indicators
        from app.market_structure.structure import analyse_structure
        from app.rtm.rtm_engine import analyse_rtm
        from app.scanner.scanner import TimeframeAnalysis
        from app.indicators.slopes import compute_slopes

        candles = trending(500)
        indicators = compute_indicators(candles)
        slopes = compute_slopes(indicators, [c.close for c in candles])
        structure = analyse_structure(candles, indicators.atr)
        ict = analyse_ict(candles, indicators.atr, structure)
        rtm = analyse_rtm(candles, indicators.atr)

        analysis = TimeframeAnalysis(
            timeframe=Timeframe.M15, candles=candles, indicators=indicators,
            slopes=slopes, structure=structure, ict=ict, rtm=rtm,
        )
        _, from_meta = analyse_window(candles)
        assert analysis.bias() is from_meta
        assert directional_bias(structure, ict, rtm, indicators) is from_meta


# ===========================================================================
# training and acceptance
# ===========================================================================


class TestEconomicAcceptance:
    """A model can rank trades genuinely and still be worth nothing.

    Filtering a negative edge harder produces a smaller negative edge, not a
    positive one.  These pin the check that says so out loud, drawn from a real
    training run: 32.3% base win rate at a 2:1 payoff, where break-even is
    33.3%.
    """

    def _slices(self, top: float, bottom: float, n: int = 8198):
        """Scores and labels that realise the given tail win rates exactly."""

        size = int(n * 0.2)
        scores = [i / n for i in range(n)]
        labels = [0] * n
        for i in range(size):                       # bottom tail
            if i < round(bottom * size):
                labels[i] = 1
        for i in range(n - size, n):                # top tail
            if i - (n - size) < round(top * size):
                labels[i] = 1
        return scores, labels

    def test_the_z_score_scales_with_sample_size(self):
        """A fixed lift threshold cannot tell 200 rows from 20,000."""

        small = _decile_lift(*self._slices(0.34, 0.29, n=400))
        large = _decile_lift(*self._slices(0.34, 0.29, n=20000))
        assert large["lift"] == pytest.approx(small["lift"], abs=0.05)
        assert large["separation_z"] > small["separation_z"]

    def test_break_even_is_the_payoff_reciprocal(self):
        assert expectancy_r(1 / 3, 2.0) == pytest.approx(0.0, abs=1e-9)
        assert expectancy_r(0.25, 3.0) == pytest.approx(0.0, abs=1e-9)

    def test_a_real_but_unprofitable_edge_is_refused(self):
        """The exact shape of a real run: significant, still loses money."""

        top, bottom = 0.3368, 0.2935
        metrics = _decile_lift(*self._slices(top, bottom))
        assert metrics["separation_z"] >= 1.96, "the separation is genuine"

        gross = expectancy_r(top, 2.0)
        assert gross > 0, "it beats break-even before costs"
        assert gross - 0.09 < 0, "but not after costs"

    def test_a_profitable_edge_clears(self):
        top = 0.42
        assert expectancy_r(top, 2.0) - 0.09 > 0

    def test_the_rejection_explains_which_check_failed(self):
        import random
        from app.ml.dataset import LabelledSample

        rng = random.Random(5)
        samples = []
        for i in range(4000):
            # A genuinely rankable but unprofitable setup: the base rate sits
            # just under break-even and the signal moves it barely above.
            signal = rng.random()
            win = rng.random() < (0.30 + 0.06 * signal)
            samples.append(
                LabelledSample(
                    symbol="X", timeframe="15m", ts=i * 900,
                    features={"signal": signal, "noise": rng.random()},
                    label=LABEL_WIN if win else LABEL_LOSS,
                    horizon=24,
                )
            )
        result = train_model(samples, min_rows=200, kind="meta", reward=2.0)
        assert not result.accepted
        assert "net_expectancy_r" in result.metrics
        assert "break_even_win_rate" in result.metrics
        # Whichever check failed, the message must name it.
        assert "rank" in result.reason or "loses money" in result.reason

    def test_the_reward_ratio_reaches_the_metrics(self):
        import random
        from app.ml.dataset import LabelledSample

        rng = random.Random(9)
        samples = [
            LabelledSample(
                symbol="X", timeframe="15m", ts=i * 900,
                features={"signal": rng.random()},
                label=LABEL_WIN if i % 3 == 0 else LABEL_LOSS,
                horizon=24,
            )
            for i in range(1500)
        ]
        result = train_model(samples, min_rows=200, kind="meta", reward=3.0)
        assert result.metrics["reward_r"] == 3.0
        assert result.metrics["break_even_win_rate"] == pytest.approx(0.25)


class TestDecileLift:
    def test_a_perfect_ranker_lifts(self):
        scores = [i / 100 for i in range(100)]
        labels = [0] * 50 + [1] * 50
        result = _decile_lift(scores, labels)
        assert result["top_win_rate"] == 1.0
        assert result["bottom_win_rate"] == 0.0
        assert result["lift"] > 1.15

    def test_a_useless_ranker_does_not(self):
        scores = [0.5] * 100
        labels = [0, 1] * 50
        assert _decile_lift(scores, labels)["lift"] == pytest.approx(1.0, abs=0.3)

    def test_the_ratio_is_guarded_against_a_zero_denominator(self):
        result = _decile_lift([i / 10 for i in range(10)], [0] * 10)
        assert result["lift"] == 0.0


class TestMetaAcceptance:
    def _samples(self, count: int, separable: bool):
        """Rows whose features either predict the label or carry nothing.

        The unseparable case uses features with *no* relationship to the label,
        so any lift is sampling noise -- which is exactly what the acceptance
        test has to reject.
        """

        import random
        from app.ml.dataset import LabelledSample

        rng = random.Random(7)
        out = []
        for i in range(count):
            win = i % 3 == 0
            if separable:
                features = {
                    "signal": (0.9 if win else 0.1) + rng.gauss(0, 0.05),
                    "noise": rng.random(),
                }
            else:
                features = {"signal": rng.random(), "noise": rng.random()}
            out.append(
                LabelledSample(
                    symbol="X", timeframe="15m", ts=i * 900,
                    features=features,
                    label=LABEL_WIN if win else LABEL_LOSS,
                    horizon=24,
                )
            )
        return out

    def test_a_model_that_cannot_rank_is_refused(self):
        result = train_model(
            self._samples(6000, separable=False), min_rows=200, kind="meta"
        )
        assert not result.accepted, (
            f"noise was accepted with lift {result.metrics.get('lift')}"
        )
        assert "rank" in result.reason

    def test_a_model_that_ranks_is_accepted_and_marked_meta(self):
        result = train_model(
            self._samples(1200, separable=True), min_rows=200, kind="meta"
        )
        assert result.accepted, result.reason
        assert result.artifact is not None
        assert result.artifact.kind == "meta"
        assert result.metrics["lift"] >= 1.15

    def test_the_kind_survives_a_serialisation_round_trip(self, tmp_path):
        result = train_model(
            self._samples(1200, separable=True), min_rows=200, kind="meta"
        )
        assert result.artifact is not None
        registry = ModelRegistry(tmp_path)
        registry.save(result.artifact, activate=True)
        loaded = registry.active(result.artifact.name)
        assert loaded is not None
        assert loaded.kind == "meta"

    def test_an_old_artifact_without_a_kind_is_a_direction_model(self):
        data = {
            "name": "legacy", "version": "1", "algorithm": "x", "trained_at": 0,
            "rows": 10, "feature_spec": {"names": ["a"], "means": [0.0], "stds": [1.0]},
            "model": {}, "metrics": {},
        }
        assert ModelArtifact.from_dict(data).kind == "direction"


class TestMetaInference:
    def test_a_meta_model_abstains_from_a_directional_view(self):
        """It has no unconditional opinion; reporting one would be invented."""

        spec = FeatureSpec.fit([{"a": 1.0}])
        artifact = ModelArtifact(
            name="m", version="1", algorithm="stub", trained_at=0,
            rows=100, feature_spec=spec, model_payload={}, kind="meta",
        )
        artifact.attach_runtime(_ConstantModel())
        predictor = Predictor(artifact=artifact)
        assert predictor.is_meta
        assert predictor.predict({"a": 1.0}) == (0.0, 0.0, 1.0)

    def test_side_zero_has_no_answer(self):
        spec = FeatureSpec.fit([{"a": 1.0}])
        artifact = ModelArtifact(
            name="m", version="1", algorithm="stub", trained_at=0,
            rows=100, feature_spec=spec, model_payload={}, kind="meta",
        )
        artifact.attach_runtime(_ConstantModel())
        assert Predictor(artifact=artifact).predict_meta({"a": 1.0}, side=0) is None

    def test_no_model_means_no_opinion_not_a_low_probability(self):
        predictor = Predictor()
        assert predictor.predict_meta({"a": 1.0}, side=1) is None
        assert predictor.predict({"a": 1.0}) == (0.0, 0.0, 1.0)
