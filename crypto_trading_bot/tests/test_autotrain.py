"""Autonomous retraining.

The bot retrains itself and grades each candidate against its own realised
trades. The tests below are mostly about the *refusals*: automated retraining
is only safe if promotion is hard, so most of this file pins the conditions
under which a challenger is turned away.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.config import ConfigError, build_settings
from app.database.database import Database
from app.database.repositories import Repositories
from app.memory.journal import DecisionJournal, DecisionKind, DecisionTrace
from app.ml.autotrain import (
    DEFAULT_MIN_LIVE_SAMPLES,
    THIN_EVIDENCE_MARGIN,
    AutoTrainer,
    LiveOutcome,
    LiveScore,
    TrainingCycle,
    collect_live_outcomes,
    score_on_outcomes,
)
from app.ml.features import FeatureSpec
from app.ml.model_registry import ModelArtifact

from tests.conftest import TEST_ENV

AUTO_ENV = dict(
    TEST_ENV,
    AUTO_TRAIN_ENABLED="true",
    AUTO_TRAIN_SYMBOLS="4",
    AUTO_TRAIN_BARS="4000",
    AUTO_TRAIN_MIN_NEW_TRADES="0",
    ML_MIN_TRAINING_ROWS="300",
)


@pytest.fixture
def settings_auto():
    return build_settings(env=dict(AUTO_ENV))


@pytest.fixture
def repos():
    db = Database("sqlite:///:memory:")
    db.migrate()
    yield Repositories(db)
    db.close()


def seed_history(repos, count: int = 60, resolve: bool = True) -> DecisionJournal:
    """Write a plausible trading history into the journal."""

    journal = DecisionJournal(repos)
    for i in range(count):
        signal = (i % 10) / 10.0
        record = journal.record(
            DecisionKind.ENTRY,
            "BTCUSDT",
            side="LONG" if i % 2 else "SHORT",
            trace=DecisionTrace(
                features={"15m_rsi": 40.0 + signal * 30, "15m_adx": signal * 50}
            ),
        )
        if resolve:
            won = signal > 0.5
            journal.record_outcome(
                record.decision_id,
                "BTCUSDT",
                {"r_multiple": 2.0 if won else -1.0, "won": won},
            )
    return journal


def artifact(names: list[str], lift: float = 1.5, kind: str = "meta") -> ModelArtifact:
    spec = FeatureSpec.fit([{n: 1.0 for n in names}])
    art = ModelArtifact(
        name="direction_v1", version="v", algorithm="stub", trained_at=0,
        rows=1000, feature_spec=spec, model_payload={},
        metrics={"lift": lift}, kind=kind,
    )
    art.attach_runtime(_RankingModel())
    return art


class _RankingModel:
    """Scores on the first feature, so ranking quality is controllable."""

    def predict_proba(self, rows):
        out = []
        for row in rows:
            p = 1.0 / (1.0 + pow(2.718, -row[0]))
            out.append([1.0 - p, p])
        return out


# ===========================================================================
# reading the bot's own history
# ===========================================================================


class TestCollectLiveOutcomes:
    def test_resolved_entries_are_joined_to_their_outcome(self, repos):
        seed_history(repos, 40)
        outcomes = collect_live_outcomes(repos, mode="paper")
        assert len(outcomes) == 40
        assert all(isinstance(o, LiveOutcome) for o in outcomes)
        assert all(o.features for o in outcomes)
        assert {o.side for o in outcomes} == {1, -1}

    def test_unresolved_entries_are_excluded(self, repos):
        """A decision with no outcome yet teaches nothing."""

        seed_history(repos, 20, resolve=False)
        assert collect_live_outcomes(repos, mode="paper") == []

    def test_entries_without_features_are_skipped_not_guessed(self, repos):
        journal = DecisionJournal(repos)
        record = journal.record(
            DecisionKind.ENTRY, "BTCUSDT", side="LONG", trace=DecisionTrace()
        )
        journal.record_outcome(
            record.decision_id, "BTCUSDT", {"r_multiple": 1.0, "won": True}
        )
        assert collect_live_outcomes(repos, mode="paper") == []

    def test_an_entry_without_a_side_is_skipped(self, repos):
        journal = DecisionJournal(repos)
        record = journal.record(
            DecisionKind.ENTRY, "BTCUSDT",
            trace=DecisionTrace(features={"15m_rsi": 50.0}),
        )
        journal.record_outcome(
            record.decision_id, "BTCUSDT", {"r_multiple": 1.0, "won": True}
        )
        assert collect_live_outcomes(repos, mode="paper") == []

    def test_resolved_count_matches(self, repos):
        seed_history(repos, 25)
        assert repos.journal.resolved_count(mode="paper") == 25

    def test_a_broken_repository_returns_nothing_rather_than_raising(self):
        class Broken:
            class journal:
                @staticmethod
                def resolved_entries(**kwargs):
                    raise RuntimeError("database gone")

        assert collect_live_outcomes(Broken()) == []

    def test_no_repository_is_tolerated(self):
        assert collect_live_outcomes(None) == []


# ===========================================================================
# grading a model on real trades
# ===========================================================================


class TestScoreOnOutcomes:
    def _outcomes(self, n: int = 60) -> list[LiveOutcome]:
        out = []
        for i in range(n):
            signal = (i / n) * 6 - 3
            out.append(
                LiveOutcome(
                    decision_id=f"d{i}", ts=i, symbol="BTCUSDT", side=1,
                    features={"a": signal}, won=signal > 0, r_multiple=1.0,
                )
            )
        return out

    def test_a_ranking_model_lifts(self):
        score = score_on_outcomes(artifact(["a"]), self._outcomes())
        assert score.usable
        assert score.lift > 1.0
        assert score.top_win_rate > score.bottom_win_rate

    def test_no_model_scores_nothing(self):
        score = score_on_outcomes(None, self._outcomes())
        assert not score.usable
        assert score.lift == 0.0

    def test_no_outcomes_scores_nothing(self):
        assert not score_on_outcomes(artifact(["a"]), []).usable

    def test_a_feature_mismatch_scores_nothing_rather_than_a_constant(self):
        """The coverage guard must reach this path too."""

        score = score_on_outcomes(artifact(["completely", "different", "names"]),
                                  self._outcomes())
        assert score.scored == 0
        assert not score.usable

    def test_too_few_scored_rows_are_not_trusted(self):
        score = score_on_outcomes(artifact(["a"]), self._outcomes(5))
        assert score.lift == 0.0


# ===========================================================================
# the promotion decision -- where the safety lives
# ===========================================================================


class TestPromotionDecision:
    def _trainer(self, settings_auto, repos) -> AutoTrainer:
        return AutoTrainer(settings_auto, registry=_FakeRegistry(), repositories=repos)

    def _cycle(self, challenger_lift=1.5, champ_live=None, chal_live=None, rows=1000):
        return TrainingCycle(
            rows=rows,
            challenger_lift=challenger_lift,
            champion_live=champ_live,
            challenger_live=chal_live,
            live_samples=(chal_live.scored if chal_live else 0),
            metrics={"base_win_rate": 0.4, "top_win_rate": 0.55},
        )

    def _score(self, lift: float, scored: int = 60) -> LiveScore:
        return LiveScore(samples=scored, scored=scored, lift=lift, brier=0.2)

    def test_with_no_champion_the_acceptance_test_is_the_whole_bar(
        self, settings_auto, repos
    ):
        promote, reason = self._trainer(settings_auto, repos)._decide(
            None, self._cycle()
        )
        assert promote
        assert "no active model" in reason

    def test_beating_the_champion_on_real_trades_promotes(
        self, settings_auto, repos
    ):
        cycle = self._cycle(
            champ_live=self._score(1.10), chal_live=self._score(1.40)
        )
        promote, reason = self._trainer(settings_auto, repos)._decide(
            artifact(["a"]), cycle
        )
        assert promote
        assert "real trades" in reason

    def test_losing_on_real_trades_holds_the_champion(self, settings_auto, repos):
        cycle = self._cycle(
            champ_live=self._score(1.60), chal_live=self._score(1.20)
        )
        promote, reason = self._trainer(settings_auto, repos)._decide(
            artifact(["a"]), cycle
        )
        assert not promote
        assert "did not beat" in reason

    def test_a_tie_holds_the_champion(self, settings_auto, repos):
        """The incumbent has been tested with real money; ties go to it."""

        cycle = self._cycle(
            champ_live=self._score(1.30), chal_live=self._score(1.30)
        )
        promote, _ = self._trainer(settings_auto, repos)._decide(
            artifact(["a"]), cycle
        )
        assert not promote

    def test_a_margin_below_the_threshold_is_not_enough(
        self, settings_auto, repos
    ):
        trainer = self._trainer(settings_auto, repos)
        cycle = self._cycle(
            champ_live=self._score(1.30),
            chal_live=self._score(1.30 + trainer.promotion_margin - 0.01),
        )
        promote, _ = trainer._decide(artifact(["a"]), cycle)
        assert not promote

    def test_thin_live_evidence_demands_a_wider_history_gap(
        self, settings_auto, repos
    ):
        trainer = self._trainer(settings_auto, repos)
        champion = artifact(["a"], lift=1.20)

        # Not enough live outcomes to judge on, small history gap -> hold.
        narrow = self._cycle(
            challenger_lift=1.30, chal_live=self._score(1.9, scored=5)
        )
        promote, reason = trainer._decide(champion, narrow)
        assert not promote
        assert "tested with real money" in reason

        # Same thin evidence, but a decisive history gap -> promote.
        wide = self._cycle(
            challenger_lift=1.20 + THIN_EVIDENCE_MARGIN + 0.01,
            chal_live=self._score(1.9, scored=5),
        )
        promote, reason = trainer._decide(champion, wide)
        assert promote
        assert "clears history" in reason

    def test_an_overfit_challenger_is_refused_outright(
        self, settings_auto, repos
    ):
        cycle = self._cycle(
            champ_live=self._score(1.0), chal_live=self._score(9.0)
        )
        # Implausible held-out performance on a tiny sample.
        cycle.metrics = {"base_win_rate": 0.99, "top_win_rate": 0.05}
        cycle.rows = 5
        promote, _ = self._trainer(settings_auto, repos)._decide(
            artifact(["a"]), cycle
        )
        assert not promote


class _FakeRegistry:
    def __init__(self) -> None:
        self.saved: list[ModelArtifact] = []
        self._active: ModelArtifact | None = None

    def active(self, name: str):
        return self._active

    def save(self, artifact, activate: bool = True):
        self.saved.append(artifact)
        if activate:
            self._active = artifact
        return None


# ===========================================================================
# scheduling
# ===========================================================================


class TestScheduling:
    def test_the_first_run_is_always_due(self, settings_auto, repos):
        assert AutoTrainer(settings_auto, _FakeRegistry(), repos).due()

    def test_the_interval_is_respected(self, settings_auto, repos):
        trainer = AutoTrainer(settings_auto, _FakeRegistry(), repos)
        now = time.time()
        trainer.last_run_ts = now
        assert not trainer.due(now=now + 60)
        interval = settings_auto.auto_train_interval_hours * 3600
        assert trainer.due(now=now + interval + 1)

    def test_retraining_waits_for_new_evidence(self, repos):
        """Retraining with nothing new to learn from only chases noise."""

        settings = build_settings(env=dict(AUTO_ENV, AUTO_TRAIN_MIN_NEW_TRADES="10"))
        trainer = AutoTrainer(settings, _FakeRegistry(), repos)
        now = time.time()
        trainer.last_run_ts = now - settings.auto_train_interval_hours * 3600 - 10

        assert not trainer.due(now=now), "no new resolved trades yet"
        seed_history(repos, 12)
        assert trainer.due(now=now)


# ===========================================================================
# a full cycle
# ===========================================================================


class TestFullCycle:
    def test_a_cycle_runs_and_is_recorded(self, settings_auto, repos, tmp_path):
        from app.exchange.synthetic import SyntheticExchange
        from app.ml.model_registry import ModelRegistry

        async def run():
            exchange = SyntheticExchange()
            await exchange.connect()
            trainer = AutoTrainer(
                settings_auto, ModelRegistry(tmp_path), repos
            )
            seed_history(repos, 50)
            cycle = await trainer.run_cycle(exchange, trigger="test")
            await exchange.close()
            return cycle, trainer

        cycle, trainer = asyncio.run(run())
        assert cycle.rows > 0
        assert cycle.symbols > 0
        assert cycle.live_samples == 50
        assert trainer.cycles == 1
        assert repos.training.recent(), "the attempt must be recorded either way"
        assert cycle.summary()

    def test_a_failure_is_recorded_rather_than_raised(
        self, settings_auto, repos, tmp_path
    ):
        """Learning must never take down the trading loop."""

        from app.ml.model_registry import ModelRegistry

        class BrokenExchange:
            async def contracts(self):
                raise RuntimeError("exchange unreachable")

        async def run():
            trainer = AutoTrainer(settings_auto, ModelRegistry(tmp_path), repos)
            return await trainer.run_cycle(BrokenExchange(), trigger="test")

        cycle = asyncio.run(run())
        assert not cycle.promoted
        assert "could not build a dataset" in cycle.reason
        assert repos.training.recent()

    def test_the_lineage_counts_holds_as_well_as_promotions(self, repos):
        repos.training.record(TrainingCycle(promoted=True).as_row())
        repos.training.record(TrainingCycle(promoted=False).as_row())
        repos.training.record(TrainingCycle(promoted=False).as_row())
        stats = repos.training.statistics()
        assert stats == {"runs": 3, "promoted": 1, "held": 2}
        assert repos.training.last_promotion() is not None


# ===========================================================================
# configuration
# ===========================================================================


class TestAutoTrainSettings:
    def test_it_is_off_by_default(self, settings):
        assert not settings.auto_train_enabled

    def test_a_negative_margin_is_rejected(self):
        with pytest.raises(ConfigError):
            build_settings(env=dict(AUTO_ENV, AUTO_TRAIN_PROMOTION_MARGIN="-0.5"))

    def test_sub_hourly_retraining_is_rejected(self):
        with pytest.raises(ConfigError):
            build_settings(env=dict(AUTO_ENV, AUTO_TRAIN_INTERVAL_HOURS="0.25"))

    def test_too_few_live_samples_is_rejected(self):
        with pytest.raises(ConfigError):
            build_settings(env=dict(AUTO_ENV, AUTO_TRAIN_MIN_LIVE_SAMPLES="3"))

    def test_it_requires_ml_to_be_enabled(self):
        with pytest.raises(ConfigError):
            build_settings(env=dict(AUTO_ENV, ML_ENABLED="false"))

    def test_an_unknown_timeframe_is_rejected(self):
        with pytest.raises(ConfigError):
            build_settings(env=dict(AUTO_ENV, AUTO_TRAIN_TIMEFRAME="9h"))

    def test_the_defaults_are_validated_when_enabled(self, settings_auto):
        assert settings_auto.auto_train_enabled
        assert settings_auto.auto_train_min_live_samples >= 10
        assert settings_auto.auto_train_promotion_margin >= 0
