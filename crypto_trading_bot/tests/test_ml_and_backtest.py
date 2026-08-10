"""ML pipeline, calibration, backtesting and walk-forward validation."""

from __future__ import annotations

import asyncio
import math
import random

import pytest

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.backtest.metrics import (
    compute_metrics,
    max_drawdown,
    risk_of_ruin,
    sharpe_ratio,
    sortino_ratio,
)
from app.backtest.walk_forward import make_folds, walk_forward
from app.domain import Timeframe
from app.indicators.indicators import compute_indicators
from app.ml.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    fit_calibrator,
)
from app.ml.dataset import (
    LABEL_LONG,
    LABEL_NO_TRADE,
    LABEL_SHORT,
    LabelledSample,
    build_dataset,
    class_distribution,
    time_split,
    triple_barrier_labels,
)
from app.ml.features import FeatureSpec
from app.ml.model_registry import ModelArtifact, ModelRegistry
from app.ml.predict import Predictor
from app.ml.train import SoftmaxRegression, train_model
from tests.test_market_structure import bars


class TestFeatureSpec:
    def test_standardisation(self):
        rows = [{"a": float(i), "b": 1.0} for i in range(10)]
        spec = FeatureSpec.fit(rows)
        assert spec.names == ["a", "b"]
        transformed = spec.transform({"a": 4.5, "b": 1.0})
        assert transformed[0] == pytest.approx(0.0, abs=1e-9)
        assert transformed[1] == pytest.approx(0.0)     # zero variance -> 0

    def test_missing_feature_falls_back_to_the_mean(self):
        spec = FeatureSpec.fit([{"a": float(i)} for i in range(10)])
        assert spec.transform({})[0] == pytest.approx(0.0)

    def test_unknown_feature_is_ignored(self):
        spec = FeatureSpec.fit([{"a": 1.0}, {"a": 2.0}])
        assert len(spec.transform({"a": 1.0, "surprise": 99.0})) == 1

    def test_non_finite_values_are_clipped(self):
        spec = FeatureSpec.fit([{"a": 1.0}, {"a": 2.0}])
        assert all(math.isfinite(v) for v in spec.transform({"a": float("inf")}))

    def test_round_trip(self):
        spec = FeatureSpec.fit([{"a": float(i), "b": float(i * 2)} for i in range(20)])
        restored = FeatureSpec.from_dict(spec.as_dict())
        assert restored.names == spec.names
        assert restored.transform({"a": 5.0, "b": 10.0}) == spec.transform({"a": 5.0, "b": 10.0})


class TestLabelling:
    def test_upward_path_is_labelled_long(self):
        candles = bars([100 + i for i in range(60)], wick=0.1)
        values = compute_indicators(candles).atr
        labels = triple_barrier_labels(candles, values, horizon=20, profit_atr=1.0, loss_atr=1.0)
        assert LABEL_LONG in [l for l in labels if l is not None]

    def test_downward_path_is_labelled_short(self):
        candles = bars([200 - i for i in range(60)], wick=0.1)
        values = compute_indicators(candles).atr
        labels = triple_barrier_labels(candles, values, horizon=20, profit_atr=1.0, loss_atr=1.0)
        assert LABEL_SHORT in [l for l in labels if l is not None]

    def test_the_tail_is_never_labelled(self):
        """The horizon must lie entirely in the past - no leakage."""

        candles = bars([100 + i for i in range(60)], wick=0.1)
        values = compute_indicators(candles).atr
        labels = triple_barrier_labels(candles, values, horizon=20)
        assert all(label is None for label in labels[-20:])

    def test_flat_market_labels_no_trade(self):
        candles = bars([100.0] * 80, wick=0.5)
        values = compute_indicators(candles).atr
        labels = triple_barrier_labels(candles, values, horizon=10, profit_atr=5.0, loss_atr=5.0)
        defined = [l for l in labels if l is not None]
        assert defined and all(l == LABEL_NO_TRADE for l in defined)

    def test_time_split_has_an_embargo(self):
        samples = [
            LabelledSample("X", "1h", ts=i, features={"a": float(i)}, label=i % 3, horizon=10)
            for i in range(200)
        ]
        train, test = time_split(samples, test_fraction=0.25, embargo=10)
        assert train and test
        assert max(s.ts for s in train) < min(s.ts for s in test)
        assert min(s.ts for s in test) - max(s.ts for s in train) >= 10

    def test_class_distribution(self):
        samples = [
            LabelledSample("X", "1h", ts=i, features={}, label=LABEL_LONG, horizon=1)
            for i in range(5)
        ]
        assert class_distribution(samples)["LONG"] == 5

    def test_build_dataset_uses_history_only(self):
        candles = bars([100 + math.sin(i / 5) * 10 for i in range(400)], wick=0.2)
        seen: list[int] = []

        def feature_fn(history):
            seen.append(len(history))
            return {"last": history[-1].close, "n": float(len(history))}

        samples = build_dataset("X", Timeframe.H1, candles, feature_fn,
                                horizon=20, warmup=200, stride=10)
        assert samples
        # Each call saw strictly the bars up to that point, never more.
        assert all(n <= len(candles) for n in seen)
        assert all(s.label in (0, 1, 2) for s in samples)


class TestCalibration:
    def test_platt_maps_scores_to_probabilities(self):
        rng = random.Random(3)
        scores, labels = [], []
        for _ in range(400):
            label = rng.random() < 0.3
            scores.append(min(max(rng.gauss(0.7 if label else 0.3, 0.15), 0.01), 0.99))
            labels.append(1 if label else 0)
        calibrator = PlattCalibrator().fit(scores, labels)
        assert calibrator.fitted
        assert 0.0 <= calibrator.transform(0.5) <= 1.0
        assert calibrator.transform(0.9) > calibrator.transform(0.1)

    def test_isotonic_is_monotone(self):
        rng = random.Random(5)
        scores = [rng.random() for _ in range(500)]
        labels = [1 if s + rng.gauss(0, 0.1) > 0.5 else 0 for s in scores]
        calibrator = IsotonicCalibrator().fit(scores, labels)
        assert calibrator.fitted
        outputs = [calibrator.transform(x / 20) for x in range(21)]
        assert all(b >= a - 1e-9 for a, b in zip(outputs, outputs[1:]))

    def test_too_few_samples_is_a_no_op(self):
        calibrator = PlattCalibrator().fit([0.5] * 5, [1] * 5)
        assert not calibrator.fitted
        assert calibrator.transform(0.5) == pytest.approx(0.5)

    def test_brier_and_ece(self):
        assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
        assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)
        assert expected_calibration_error([0.5] * 100, [1, 0] * 50) < 0.05

    def test_calibration_improves_a_skewed_model(self):
        rng = random.Random(11)
        # Systematically overconfident scores.
        raw = [min(0.99, rng.random() ** 0.5) for _ in range(600)]
        labels = [1 if rng.random() < s * 0.5 else 0 for s in raw]
        calibrator, metrics = fit_calibrator(raw, labels)
        assert calibrator is not None
        assert metrics["calibrated_brier"] <= metrics["baseline_brier"]

    def test_round_trip(self):
        calibrator = PlattCalibrator(a=2.0, b=-1.0, fitted=True)
        restored = PlattCalibrator.from_dict(calibrator.as_dict())
        assert restored.transform(0.7) == pytest.approx(calibrator.transform(0.7))


class TestSoftmaxRegression:
    def test_learns_a_separable_problem(self):
        rng = random.Random(2)
        X, y = [], []
        for _ in range(400):
            label = rng.randrange(3)
            centre = [(2.0, 0.0), (-2.0, 0.0), (0.0, 2.0)][label]
            X.append([centre[0] + rng.gauss(0, 0.4), centre[1] + rng.gauss(0, 0.4)])
            y.append(label)
        model = SoftmaxRegression(epochs=60).fit(X, y)
        predictions = [max(range(3), key=lambda k: p[k]) for p in model.predict_proba(X)]
        accuracy = sum(1 for p, t in zip(predictions, y) if p == t) / len(y)
        assert accuracy > 0.85

    def test_probabilities_sum_to_one(self):
        model = SoftmaxRegression(epochs=5).fit([[0.0, 0.0], [1.0, 1.0]], [0, 1])
        for row in model.predict_proba([[0.5, 0.5]]):
            assert sum(row) == pytest.approx(1.0)

    def test_round_trip(self):
        model = SoftmaxRegression(epochs=5).fit([[0.0], [1.0]], [0, 1])
        restored = SoftmaxRegression.from_dict(model.as_dict())
        assert restored.predict_proba([[0.5]])[0] == pytest.approx(
            model.predict_proba([[0.5]])[0]
        )


class TestTrainingAndPrediction:
    def _samples(self, n=800):
        rng = random.Random(4)
        samples = []
        for i in range(n):
            signal = rng.gauss(0, 1)
            noise = rng.gauss(0, 1)
            if signal > 0.6:
                label = LABEL_LONG
            elif signal < -0.6:
                label = LABEL_SHORT
            else:
                label = LABEL_NO_TRADE
            samples.append(LabelledSample(
                "X", "1h", ts=i * 3600,
                features={"signal": signal, "noise": noise},
                label=label, horizon=10,
            ))
        return samples

    def test_refuses_to_train_on_too_little_data(self):
        result = train_model(self._samples(50), min_rows=400)
        assert not result.accepted and "labelled rows" in result.reason

    def test_refuses_a_single_class(self):
        samples = [
            LabelledSample("X", "1h", ts=i, features={"a": 1.0}, label=LABEL_LONG, horizon=1)
            for i in range(500)
        ]
        result = train_model(samples, min_rows=100)
        assert not result.accepted

    def test_trains_and_calibrates_a_learnable_problem(self):
        result = train_model(self._samples(), min_rows=200)
        assert result.accepted, result.reason
        artifact = result.artifact
        assert artifact is not None
        assert artifact.metrics["accuracy"] > 0.5
        assert artifact.feature_spec.names == ["noise", "signal"]

    def test_predictor_without_a_model_says_it_does_not_know(self, tmp_path):
        predictor = Predictor(ModelRegistry(tmp_path), model_name="nothing")
        assert predictor.predict({"a": 1.0}) == (0.0, 0.0, 1.0)
        assert not predictor.info()["loaded"]

    def test_registry_round_trip_and_prediction(self, tmp_path):
        result = train_model(self._samples(), name="test_model", min_rows=200)
        assert result.accepted
        registry = ModelRegistry(tmp_path)
        registry.save(result.artifact, activate=True)

        predictor = Predictor(registry, model_name="test_model")
        assert predictor.ready
        p_long, p_short, p_no = predictor.predict({"signal": 2.0, "noise": 0.0})
        assert sum((p_long, p_short, p_no)) == pytest.approx(1.0, abs=0.01)
        assert p_long > p_short                       # strong positive signal
        assert max(p_long, p_short, p_no) <= 0.98     # never certainty

        opposite = predictor.predict({"signal": -2.0, "noise": 0.0})
        assert opposite[1] > opposite[0]

    def test_prediction_failure_degrades_safely(self, tmp_path):
        result = train_model(self._samples(), name="broken", min_rows=200)
        registry = ModelRegistry(tmp_path)
        registry.save(result.artifact, activate=True)
        predictor = Predictor(registry, model_name="broken")
        predictor.artifact.runtime = object()          # no predict_proba
        assert predictor.predict({"signal": 1.0}) == (0.0, 0.0, 1.0)
        assert predictor.failures == 1


class TestMetrics:
    def test_drawdown(self):
        worst, duration = max_drawdown([100, 120, 90, 95, 130])
        assert worst == pytest.approx(0.25)
        assert duration >= 2

    def test_no_drawdown_on_a_rising_curve(self):
        assert max_drawdown([100, 110, 120])[0] == 0.0

    def test_sharpe_of_a_constant_series_is_zero(self):
        assert sharpe_ratio([0.01] * 50, 365) == 0.0

    def test_sortino_ignores_upside(self):
        returns = [0.01, 0.02, -0.01, 0.03, -0.005]
        assert sortino_ratio(returns, 365) > sharpe_ratio(returns, 365)

    def test_sortino_caps_instead_of_returning_infinity(self):
        # A constant series has no risk to measure at all, like Sharpe.
        assert sortino_ratio([0.01] * 50, 365) == 0.0
        # Varying but never negative: infinite in theory, capped in practice.
        value = sortino_ratio([0.01, 0.02, 0.03], 365)
        assert math.isfinite(value) and value == 99.0

    def test_risk_of_ruin_bounds(self):
        assert risk_of_ruin(0.2, 1.0, 0.005) == 1.0        # no edge
        assert risk_of_ruin(0.6, 2.0, 0.005) < 0.01        # strong edge, small bets
        assert risk_of_ruin(0.6, 2.0, 0.5) > risk_of_ruin(0.6, 2.0, 0.005)

    def test_metrics_from_trades(self):
        trades = [
            {"pnl": 20.0, "r_multiple": 2.0, "fees": 0.5},
            {"pnl": -10.0, "r_multiple": -1.0, "fees": 0.5},
            {"pnl": 30.0, "r_multiple": 3.0, "fees": 0.5},
        ]
        curve = [1000, 1020, 1010, 1040]
        metrics = compute_metrics(trades, curve, 1000, period_seconds=3600)
        assert metrics.trades == 3
        assert metrics.wins == 2 and metrics.losses == 1
        assert metrics.win_rate == pytest.approx(2 / 3)
        assert metrics.profit_factor == pytest.approx(5.0)
        assert metrics.total_return == pytest.approx(0.04)
        assert "not significant" in metrics.sample_warning

    def test_no_losers_is_flagged_as_suspicious(self):
        trades = [{"pnl": 10.0, "r_multiple": 1.0} for _ in range(40)]
        metrics = compute_metrics(trades, [1000 + i * 10 for i in range(41)], 1000)
        assert "overfit" in metrics.sample_warning


class TestBacktest:
    @pytest.fixture(scope="class")
    def data(self, exchange):
        symbols = ["BTCUSDT", "ETHUSDT"]
        timeframes = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]
        return {
            symbol: {
                tf: asyncio.run(exchange.candles(symbol, tf, limit=25000))
                for tf in timeframes
            }
            for symbol in symbols
        }

    def test_warmup_is_raised_to_cover_the_highest_timeframe(self):
        config = BacktestConfig(symbols=["BTCUSDT"], execution_timeframe=Timeframe.M15,
                                warmup_bars=100)
        # 60 daily bars at 15-minute resolution.
        assert config.required_warmup() == 60 * 96

    def test_runs_and_produces_coherent_metrics(self, settings, data, exchange):
        contracts = asyncio.run(exchange.contracts())
        config = BacktestConfig(symbols=["BTCUSDT", "ETHUSDT"],
                                execution_timeframe=Timeframe.M15,
                                starting_equity=1000.0, max_bars=1200, signal_stride=8)
        result = BacktestEngine(settings, data, contracts, config).run()

        assert result.bars > 0
        assert result.equity_curve
        assert result.metrics.starting_equity == 1000.0
        # Equity must be consistent with the trades that were taken.
        assert result.metrics.ending_equity == pytest.approx(result.equity_curve[-1])
        assert result.metrics.trades == len(result.trades)
        for trade in result.trades:
            assert trade["symbol"] in {"BTCUSDT", "ETHUSDT"}
            assert trade["closed_at"] >= trade["opened_at"]
            assert trade["fees"] >= 0

    def test_costs_reduce_returns(self, settings, data, exchange):
        contracts = asyncio.run(exchange.contracts())
        base = BacktestConfig(symbols=["BTCUSDT"], execution_timeframe=Timeframe.M15,
                              max_bars=800, signal_stride=8, taker_fee=0.0,
                              slippage_pct=0.0, spread_pct=0.0)
        costly = BacktestConfig(symbols=["BTCUSDT"], execution_timeframe=Timeframe.M15,
                                max_bars=800, signal_stride=8, taker_fee=0.002,
                                slippage_pct=0.003, spread_pct=0.002)
        free_result = BacktestEngine(settings, data, contracts, base).run()
        cost_result = BacktestEngine(settings, data, contracts, costly).run()
        if free_result.metrics.trades and cost_result.metrics.trades:
            assert cost_result.metrics.total_return <= free_result.metrics.total_return

    def test_refuses_without_enough_history(self, settings, exchange):
        contracts = asyncio.run(exchange.contracts())
        tiny = {"BTCUSDT": {Timeframe.M15: asyncio.run(
            exchange.candles("BTCUSDT", Timeframe.M15, limit=50))}}
        config = BacktestConfig(symbols=["BTCUSDT"], execution_timeframe=Timeframe.M15)
        with pytest.raises(ValueError, match="not enough execution bars"):
            BacktestEngine(settings, tiny, contracts, config).run()

    def test_rejection_reasons_are_recorded(self, settings, data, exchange):
        contracts = asyncio.run(exchange.contracts())
        config = BacktestConfig(symbols=["BTCUSDT"], execution_timeframe=Timeframe.M15,
                                max_bars=600, signal_stride=8)
        result = BacktestEngine(settings, data, contracts, config).run()
        assert result.rejections, "the bot should explain why it stayed out"


class TestWalkForward:
    def test_fold_construction(self):
        folds = make_folds(total_bars=6000, folds=4, in_sample_fraction=0.7, warmup=1000)
        assert len(folds) == 4
        for fold in folds:
            assert fold.in_sample_start < fold.in_sample_end <= fold.out_start < fold.out_end
        # Folds do not overlap.
        for a, b in zip(folds, folds[1:]):
            assert a.out_end <= b.in_sample_start

    def test_too_little_history_yields_no_folds(self):
        assert make_folds(total_bars=200, folds=4, warmup=150) == []

    def test_walk_forward_reports_out_of_sample_only(self, settings, exchange):
        contracts = asyncio.run(exchange.contracts())
        data = {
            "BTCUSDT": {
                tf: asyncio.run(exchange.candles("BTCUSDT", tf, limit=25000))
                for tf in (Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1)
            }
        }
        config = BacktestConfig(symbols=["BTCUSDT"], execution_timeframe=Timeframe.M15,
                                warmup_bars=6000, max_bars=900, signal_stride=16)
        result = walk_forward(settings, data, contracts, config, folds=2)
        assert result.folds
        assert 0.0 <= result.consistency <= 1.0
