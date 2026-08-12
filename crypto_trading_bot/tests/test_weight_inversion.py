"""Weight was accruing to models for *not* participating.

The live dashboard, after a single resolved losing trade::

    mean_reversion   10%  50%   0    -      <- never voted
    derivatives      10%  50%   0    -      <- never voted
    news             10%  50%   0    -      <- never voted
    sentiment        10%  50%   0    -      <- never voted
    ml               10%  50%   0    -      <- cannot vote at all
    portfolio_risk   10%  50%   0    -      <- never votes by design
    technical         4%  43%   1   0%      <- had an opinion, lost
    ... nine more at 4%

Six models that said nothing held 61% of the vote; the ten that actually
formed a view held 39% between them.

The mechanism was the weight transform, ``max(reliability - 0.35, 0.01) ** 1.5``.
Beta shrinkage did its job -- one loss moved reliability from 0.5000 to 0.4286,
correctly modest -- and then subtracting 0.35 from a value living in a narrow
band around 0.5 turned that 14% difference into a 2.64x difference in weight.

It compounded: participation is directional weight over *available* weight, so
loading the denominator with models that never speak drove participation to
~30% and the ensemble refused entries on ``thin_participation``.
"""

from __future__ import annotations

import pytest

from app.ensemble.weighting import PRIOR_STRENGTH, WeightTable

MODELS = [
    "technical", "price_action", "market_structure", "ict", "rtm", "momentum",
    "regime", "order_flow", "liquidity", "macro",          # these voted
    "mean_reversion", "derivatives", "news", "sentiment", "ml", "portfolio_risk",
]
VOTED = MODELS[:10]
SILENT = MODELS[10:]


def _table_after_one_loss() -> WeightTable:
    table = WeightTable()
    for name in VOTED:
        table.record(name, "ALL", won=False, r_multiple=-1.0, confidence=0.6)
    return table


class TestSilenceIsNotRewarded:
    def test_one_loss_does_not_invert_the_weighting(self):
        weights = _table_after_one_loss().weights(MODELS, "ALL")
        silent = weights["ml"]
        voted = weights["technical"]
        assert silent / voted < 1.6, (
            f"a model that never spoke carries {silent / voted:.2f}x the weight "
            "of one that did; the shipped transform made this 2.64x"
        )

    def test_the_silent_minority_does_not_hold_the_majority_of_the_vote(self):
        weights = _table_after_one_loss().weights(MODELS, "ALL")
        assert sum(weights[n] for n in SILENT) < 0.55

    def test_the_shrinkage_is_not_undone_by_the_transform(self):
        """Weight must move roughly in proportion to reliability, not explode."""

        table = _table_after_one_loss()
        weights = table.weights(MODELS, "ALL")
        reliability_ratio = (
            table.reliability("ml", "ALL") / table.reliability("technical", "ALL")
        )
        weight_ratio = weights["ml"] / weights["technical"]
        # Squaring is the intended curvature; anything beyond it means the
        # transform is amplifying small-sample noise again.
        assert weight_ratio == pytest.approx(reliability_ratio**2, rel=0.05)


class TestRealSkillIsStillRewarded:
    def _table(self, records):
        table = WeightTable()
        for name, trades, wins in records:
            for i in range(trades):
                table.record(
                    name, "ALL", won=i < wins, r_multiple=1.0 if i < wins else -1.0,
                    confidence=0.6,
                )
        return table

    def test_a_proven_model_outweighs_a_mediocre_one(self):
        table = self._table([("good", 30, 20), ("poor", 30, 10)])
        weights = table.weights(MODELS + ["good", "poor"], "ALL")
        assert weights["good"] > weights["poor"] * 1.8

    def test_a_proven_model_outweighs_an_unproven_one(self):
        table = self._table([("good", 30, 20)])
        weights = table.weights(MODELS + ["good", "unproven"], "ALL")
        assert weights["good"] > weights["unproven"]

    def test_evidence_beats_the_prior_over_time(self):
        """The differentiation has to grow with sample size, not shrink."""

        early = self._table([("m", 4, 3)])
        late = self._table([("m", 40, 30)])
        early_w = early.weights(MODELS + ["m"], "ALL")["m"]
        late_w = late.weights(MODELS + ["m"], "ALL")["m"]
        assert late_w > early_w

    def test_no_model_can_dominate(self):
        table = self._table([("star", 60, 60)])
        weights = table.weights(MODELS + ["star"], "ALL")
        assert weights["star"] <= 0.25 + 1e-6

    def test_no_model_is_silenced_by_weight_alone(self):
        table = self._table([("dud", 40, 2)])
        weights = table.weights(MODELS + ["dud"], "ALL")
        assert weights["dud"] > 0.0

    def test_weights_still_sum_to_one(self):
        weights = _table_after_one_loss().weights(MODELS, "ALL")
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)

    def test_an_all_fresh_table_is_uniform(self):
        weights = WeightTable().weights(MODELS, "ALL")
        assert len(set(round(w, 6) for w in weights.values())) == 1

    def test_a_governance_override_still_damps(self):
        table = _table_after_one_loss()
        before = table.weights(MODELS, "ALL")["technical"]
        table.set_override("technical", 0.2)
        after = table.weights(MODELS, "ALL")["technical"]
        assert after < before


class TestAnUnconfiguredNewsFeedIsAbsentNotUndecided:
    """NEWS_FEED_URL is empty by default, so this fired every cycle."""

    def _context(self, sentiment):
        from app.ensemble.base import ModelContext
        from app.fundamental.macro_engine import MacroSnapshot
        from app.fundamental.news_engine import NewsAssessment
        from app.fundamental.sentiment import FundamentalSnapshot

        snapshot = FundamentalSnapshot(
            symbol="BTCUSDT",
            macro=MacroSnapshot(),
            news=NewsAssessment(sentiment=sentiment),
        )
        return ModelContext(
            symbol="BTCUSDT", ts=0, analysis=None, fundamentals=snapshot
        )

    def test_unknown_sentiment_reports_unavailable(self):
        from app.domain import Sentiment
        from app.ensemble.models import NewsModel

        output = NewsModel().safe_evaluate(self._context(Sentiment.UNKNOWN))
        assert not output.available

    def test_a_configured_feed_with_nothing_to_say_still_participates(self):
        """A feed that ran and found nothing directional is a real abstention."""

        from app.domain import Sentiment
        from app.ensemble.models import NewsModel

        output = NewsModel().safe_evaluate(self._context(Sentiment.NEUTRAL))
        assert output.available

    def test_it_does_not_dilute_participation(self):
        from app.domain import Sentiment
        from app.ensemble.base import AnalyticalModel, ModelOutput, ModelSignal
        from app.ensemble.engine import EnsembleEngine
        from app.ensemble.models import NewsModel

        class Voter(AnalyticalModel):
            name = "voter"
            family = "technical"

            def evaluate(self, context):
                return ModelOutput(
                    name=self.name, signal=ModelSignal.LONG, confidence=0.8
                )

        result = EnsembleEngine(models=[Voter(), NewsModel()]).evaluate(
            self._context(Sentiment.UNKNOWN)
        )
        assert result.participation == pytest.approx(1.0)


class TestTheArithmeticThatWasWrong:
    """Pin the exact numbers, so the regression is recognisable if it returns."""

    def _reliability(self, trades, wins):
        a = b = PRIOR_STRENGTH * 0.5
        return (wins + a) / (trades + a + b)

    def test_shrinkage_itself_was_never_the_problem(self):
        assert self._reliability(0, 0) == pytest.approx(0.5)
        assert self._reliability(1, 0) == pytest.approx(3 / 7)

    def test_the_old_transform_amplified_a_14_percent_gap_to_164_percent(self):
        old = lambda r: max(r - 0.35, 0.01) ** 1.5      # noqa: E731
        silent, voted = self._reliability(0, 0), self._reliability(1, 0)
        assert silent / voted == pytest.approx(1.167, rel=0.01)
        assert old(silent) / old(voted) == pytest.approx(2.64, rel=0.01)

    def test_the_new_transform_is_proportionate(self):
        new = lambda r: max(r, 0.05) ** 2               # noqa: E731
        silent, voted = self._reliability(0, 0), self._reliability(1, 0)
        assert new(silent) / new(voted) == pytest.approx(1.36, rel=0.01)
