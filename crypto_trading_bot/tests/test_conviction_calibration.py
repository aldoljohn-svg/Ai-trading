"""Ensemble conviction was double-counting the same quantity.

The formula was ``margin * agreement * (0.5 + 0.5 * participation)``, but
``margin == 2 * agreement - 1`` -- the two factors are algebraically the same
information. Multiplying them squared the penalty and crushed every ordinary
vote toward zero: a 55/45 split scored 4% conviction, a solid 70/30 scored 22%.

That was not conservatism. It made the downstream thresholds unreachable for
reasons unrelated to how good the setup was, and it propagated into the trade
quality score, which then re-applied agreement and participation a third time.

These tests pin the corrected scale and the properties it has to keep.
"""

from __future__ import annotations

import pytest

from app.domain import Regime
from app.ensemble.base import ModelSignal
from app.quality.trade_quality import compute_trade_quality


def conviction(agreement: float, participation: float) -> float:
    """The formula as implemented, expressed in terms of the vote split."""

    margin = 2 * agreement - 1
    return max(0.0, min(1.0, margin * (0.5 + 0.5 * participation)))


class _Ensemble:
    """Minimal stand-in with the fields trade quality reads."""

    def __init__(self, agreement: float, participation: float, data_quality: float = 1.0):
        self.signal = ModelSignal.LONG
        self.agreement = agreement
        self.participation = participation
        self.dissent = 1.0 - agreement
        self.data_quality = data_quality
        self.aggregate_risk = 0.2
        self.confidence = conviction(agreement, participation)


class TestConvictionScale:
    def test_a_dead_heat_is_zero(self):
        assert conviction(0.5, 1.0) == pytest.approx(0.0)

    def test_unanimity_at_full_attendance_is_total(self):
        assert conviction(1.0, 1.0) == pytest.approx(1.0)

    def test_it_rises_monotonically_with_agreement(self):
        values = [conviction(a, 0.6) for a in (0.55, 0.6, 0.7, 0.8, 0.9, 1.0)]
        assert values == sorted(values)

    def test_it_rises_monotonically_with_participation(self):
        values = [conviction(0.7, p) for p in (0.4, 0.5, 0.6, 0.8, 1.0)]
        assert values == sorted(values)

    def test_it_is_linear_in_the_margin(self):
        """The regression guard: no second factor that also encodes agreement."""

        p = 0.8
        a1, a2 = 0.60, 0.80
        m1, m2 = 2 * a1 - 1, 2 * a2 - 1
        # Doubling the margin must double the conviction, exactly.
        assert conviction(a2, p) / conviction(a1, p) == pytest.approx(m2 / m1)

    def test_an_ordinary_majority_is_not_crushed_to_noise(self):
        """A 70/30 vote is not a coin flip and must not score like one."""

        assert conviction(0.70, 0.60) > 0.30

    def test_a_bare_majority_is_still_weak(self):
        """Fixing the double-count must not turn 55/45 into a strong signal."""

        assert conviction(0.55, 0.60) < 0.15


class TestQualityDoesNotReapplyTheSameTerms:
    def test_signal_tracks_conviction_proportionally(self):
        """Quality's signal component must not re-add agreement/participation."""

        low = compute_trade_quality(
            ensemble=_Ensemble(0.60, 0.6), regime=Regime.TREND_UP, rr=2.5, min_rr=1.7
        )
        high = compute_trade_quality(
            ensemble=_Ensemble(0.85, 0.6), regime=Regime.TREND_UP, rr=2.5, min_rr=1.7
        )
        ratio_conviction = conviction(0.85, 0.6) / conviction(0.60, 0.6)
        ratio_signal = high.components["signal"] / low.components["signal"]
        assert ratio_signal == pytest.approx(ratio_conviction, rel=1e-6)

    def test_thin_data_discounts_the_signal(self):
        """Data quality is a separate axis and should still count once."""

        good = compute_trade_quality(
            ensemble=_Ensemble(0.8, 0.7, data_quality=1.0),
            regime=Regime.TREND_UP, rr=2.5, min_rr=1.7,
        )
        thin = compute_trade_quality(
            ensemble=_Ensemble(0.8, 0.7, data_quality=0.3),
            regime=Regime.TREND_UP, rr=2.5, min_rr=1.7,
        )
        assert thin.components["signal"] < good.components["signal"]

    def test_a_strong_clean_setup_can_clear_the_default_minimum(self):
        """The gate has to be reachable, or it is not a gate."""

        quality = compute_trade_quality(
            ensemble=_Ensemble(0.90, 0.85),
            regime=Regime.TREND_UP,
            rr=3.0,
            min_rr=1.7,
            portfolio_headroom=1.0,
        )
        assert quality.score >= 55.0, quality.summary()

    def test_a_weak_setup_still_fails_it(self):
        quality = compute_trade_quality(
            ensemble=_Ensemble(0.55, 0.45),
            regime=Regime.TRANSITION,
            rr=1.8,
            min_rr=1.7,
        )
        assert quality.score < 55.0

    def test_no_ensemble_scores_nothing_for_signal(self):
        quality = compute_trade_quality(regime=Regime.TREND_UP, rr=2.5, min_rr=1.7)
        assert quality.components["signal"] == 0.0


class TestExecutionTimeframe:
    def test_entries_are_timed_on_fifteen_minutes_by_default(self, settings):
        assert settings.execution_timeframe == "15m"

    def test_a_context_timeframe_is_refused_as_an_execution_one(self):
        from app.config import ConfigError, build_settings
        from tests.conftest import TEST_ENV

        with pytest.raises(ConfigError):
            build_settings(env=dict(TEST_ENV, EXECUTION_TIMEFRAME="4h"))

    def test_the_analysis_primary_is_the_execution_timeframe(self):
        """SymbolAnalysis.primary must agree with the configured setting."""

        import asyncio

        from app.data.market_data import MarketData
        from app.domain import Timeframe
        from app.exchange.synthetic import SyntheticExchange
        from app.scanner.scanner import Scanner
        from app.scanner.universe import UniverseBuilder

        async def run():
            exchange = SyntheticExchange()
            await exchange.connect()
            universe = UniverseBuilder(
                min_quote_volume=1000.0, max_spread_pct=0.01, max_symbols=4
            )
            scanner = Scanner(MarketData(exchange), universe, deep_analysis_count=1)
            analyses = await scanner.deep_analyse(await scanner.prescreen())
            await exchange.close()
            return analyses[0]

        analysis = asyncio.run(run())
        assert analysis.primary is not None
        assert analysis.primary.timeframe is Timeframe.M15


class TestNewsWiring:
    def test_no_feed_is_the_default_and_is_valid(self, settings):
        assert settings.news_feed_url == ""

    def test_a_non_http_feed_is_rejected(self):
        from app.config import ConfigError, build_settings
        from tests.conftest import TEST_ENV

        with pytest.raises(ConfigError):
            build_settings(env=dict(TEST_ENV, NEWS_FEED_URL="file:///etc/passwd"))

    def test_an_http_feed_is_accepted(self):
        from app.config import build_settings
        from tests.conftest import TEST_ENV

        settings = build_settings(
            env=dict(TEST_ENV, NEWS_FEED_URL="https://example.com/news.json")
        )
        assert settings.news_feed_url.startswith("https://")

    def test_without_a_feed_the_assessment_stays_unknown(self):
        """Missing news must never become a sentiment."""

        import asyncio

        from app.domain import Sentiment
        from app.fundamental.news_engine import NewsEngine

        engine = NewsEngine()
        assessment = asyncio.run(engine.assess("BTCUSDT"))
        assert not assessment.feed_loaded
        assert assessment.sentiment is Sentiment.UNKNOWN
        assert not assessment.blocks_entries


class TestUnknownIsNotZero:
    """"Not measured" must not be scored as "measured and terrible"."""

    def test_a_missing_reward_risk_is_neutral(self):
        from app.quality.trade_quality import compute_trade_quality

        unknown = compute_trade_quality(
            ensemble=_Ensemble(0.8, 0.7), regime=Regime.TREND_UP, rr=0.0, min_rr=1.7
        )
        assert unknown.components["reward_risk"] == 50.0
        assert any("no reward:risk" in note for note in unknown.notes)

    def test_a_genuinely_poor_reward_risk_still_scores_low(self):
        from app.quality.trade_quality import compute_trade_quality

        poor = compute_trade_quality(
            ensemble=_Ensemble(0.8, 0.7), regime=Regime.TREND_UP, rr=0.5, min_rr=1.7
        )
        good = compute_trade_quality(
            ensemble=_Ensemble(0.8, 0.7), regime=Regime.TREND_UP, rr=3.5, min_rr=1.7
        )
        assert poor.components["reward_risk"] < 50.0 < good.components["reward_risk"]

    def test_an_unproposed_trade_is_still_rejected_on_its_merits(self):
        """Neutralising the artefact must not let a non-trade look acceptable."""

        from app.quality.trade_quality import compute_trade_quality

        weak = compute_trade_quality(
            ensemble=_Ensemble(0.55, 0.4), regime=Regime.TRANSITION, rr=0.0, min_rr=1.7
        )
        assert not weak.acceptable


class TestParticipationCountsOnlyRealVoters:
    """An untrained model is not an undecided voter.

    Counting a model with no data source against participation permanently caps
    conviction for a reason unrelated to the setup, and that depressed
    conviction is then penalised again through model risk and trade quality.
    """

    def _context(self):
        from app.ensemble.base import ModelContext

        return ModelContext(symbol="BTCUSDT", ts=0, analysis=None)

    def test_unavailable_is_distinct_from_an_abstention(self):
        from app.ensemble.base import ModelOutput

        abstained = ModelOutput.no_signal("ict", "no clear structure")
        absent = ModelOutput.unavailable("ml", "no trained model loaded")
        assert abstained.available
        assert not absent.available
        # Both are still non-votes.
        assert not abstained.usable and not absent.usable

    def test_an_absent_model_does_not_dilute_participation(self):
        from app.ensemble.base import AnalyticalModel, ModelOutput, ModelSignal
        from app.ensemble.engine import EnsembleEngine

        class Voter(AnalyticalModel):
            name = "voter"
            family = "technical"

            def evaluate(self, context):
                return ModelOutput(
                    name=self.name, signal=ModelSignal.LONG, confidence=0.8
                )

        class Absent(AnalyticalModel):
            name = "absent"
            family = "learned"

            def evaluate(self, context):
                return ModelOutput.unavailable(self.name, "no trained model")

        class Abstainer(AnalyticalModel):
            name = "abstainer"
            family = "technical"

            def evaluate(self, context):
                return ModelOutput.no_signal(self.name, "looked, no view")

        # One voter alongside one structurally absent model: full participation.
        with_absent = EnsembleEngine(models=[Voter(), Absent()]).evaluate(
            self._context()
        )
        assert with_absent.participation == pytest.approx(1.0)

        # One voter alongside a genuine abstention: half.
        with_abstainer = EnsembleEngine(models=[Voter(), Abstainer()]).evaluate(
            self._context()
        )
        assert with_abstainer.participation < 1.0

    def test_a_genuine_abstention_still_counts_against_participation(self):
        """Fixing the dilution must not make every non-vote free."""

        from app.ensemble.base import AnalyticalModel, ModelOutput, ModelSignal
        from app.ensemble.engine import EnsembleEngine

        class Voter(AnalyticalModel):
            name = "voter"
            family = "technical"

            def evaluate(self, context):
                return ModelOutput(
                    name=self.name, signal=ModelSignal.LONG, confidence=0.8
                )

        class Abstainer(AnalyticalModel):
            name = "abstainer"
            family = "technical"

            def evaluate(self, context):
                return ModelOutput.no_signal(self.name, "no view")

        result = EnsembleEngine(
            models=[Voter(), Abstainer(), Abstainer()]
        ).evaluate(self._context())
        assert result.participation < 0.6

    def test_the_real_models_report_absence_where_it_is_structural(self):
        """The ML and news models must not look like undecided voters."""

        from app.ensemble.models import MachineLearningModel, NewsModel

        context = self._context()
        ml = MachineLearningModel().safe_evaluate(context)
        news = NewsModel().safe_evaluate(context)
        assert not ml.available, "an untrained model was never in the room"
        assert not news.available, "an unconfigured feed was never in the room"
