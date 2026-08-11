"""Autonomous retraining: learn from the trades the bot actually took.

A model trained once and never revisited decays, because the market it learned
stops being the market it trades.  This module retrains on a schedule and, more
importantly, **grades every candidate against the bot's own realised results**
rather than only against historical simulation.

The loop
--------

1. Collect the decisions that have since resolved.  The journal stores the
   feature vector exactly as it was at decision time, the side taken and what
   happened, so the bot's trading history is already a labelled dataset -- one
   drawn from the distribution it actually faces, not a backtest's idea of it.
2. Build a fresh training set from recent market data and fit a challenger.
3. Score the **champion and the challenger on the same live outcomes**.  This is
   the comparison that matters: not "does the new model look good on history"
   but "would it have ranked our real trades better than the model that took
   them".
4. Promote only if the challenger wins by a margin, on evidence, every check.

Why promotion is deliberately hard
----------------------------------

Retraining on recent data and deploying whatever comes out is how a system
chases noise into a drawdown.  Every barrier below has to clear:

* the challenger must pass the same acceptance test as a manual run -- it has
  to *rank* trades, not merely score well on argmax;
* the overfitting detector must not reject it;
* it must beat the champion on live outcomes by ``promotion_margin``, or beat
  it on held-out history by a **wider** margin when live evidence is thin;
* it must not be built on fewer live outcomes than ``min_live_samples`` unless
  there is no champion at all to compare against.

A tie leaves the champion in place.  The incumbent has the advantage precisely
because it has already been tested by real money, and the burden of proof sits
with the challenger.

What this can and cannot change
-------------------------------

It swaps which model informs *confidence*.  That is all.  The model's influence
is capped at ``ML_WEIGHT`` (itself capped at 0.5) and it can never create a
trade, raise a position size, or touch a risk limit -- those sit above it in the
hierarchy and are not writable from here.  Automating the retraining does not
widen what the model is allowed to do.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.domain import Timeframe
from app.logger import get_logger
from app.ml.meta import LABEL_LOSS, LABEL_WIN, SIDE_FEATURE, build_meta_dataset
from app.ml.model_registry import ModelArtifact
from app.ml.train import _decile_lift, train_model
from app.ml.calibration import brier_score

log = get_logger(__name__)

#: A challenger must beat the champion's live lift by this much to take over.
DEFAULT_PROMOTION_MARGIN = 0.10

#: Live outcomes needed before their verdict is trusted on its own.
DEFAULT_MIN_LIVE_SAMPLES = 40

#: With less live evidence than that, a challenger must clear a wider gap on
#: held-out history instead.
THIN_EVIDENCE_MARGIN = 0.35


@dataclass(slots=True)
class LiveOutcome:
    """One resolved decision: what the bot saw, what it did, what happened."""

    decision_id: str
    ts: int
    symbol: str
    side: int
    features: dict[str, float]
    won: bool
    r_multiple: float

    @property
    def label(self) -> int:
        return LABEL_WIN if self.won else LABEL_LOSS


@dataclass(slots=True)
class LiveScore:
    """How a model would have ranked a set of real trades."""

    samples: int = 0
    lift: float = 0.0
    brier: float = 0.0
    top_win_rate: float = 0.0
    bottom_win_rate: float = 0.0
    scored: int = 0

    @property
    def usable(self) -> bool:
        return self.scored > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "scored": self.scored,
            "lift": round(self.lift, 3),
            "brier": round(self.brier, 5),
            "top_win_rate": round(self.top_win_rate, 4),
            "bottom_win_rate": round(self.bottom_win_rate, 4),
        }


@dataclass(slots=True)
class TrainingCycle:
    """The record of one attempt, promoted or not."""

    ts: int = field(default_factory=lambda: int(time.time()))
    mode: str = "meta"
    trigger: str = "scheduled"
    timeframe: str = ""
    symbols: int = 0
    rows: int = 0
    challenger_version: str = ""
    challenger_lift: float = 0.0
    challenger_brier: float = 0.0
    live_samples: int = 0
    champion_live: LiveScore | None = None
    challenger_live: LiveScore | None = None
    promoted: bool = False
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "mode": self.mode,
            "trigger": self.trigger,
            "timeframe": self.timeframe,
            "symbols": self.symbols,
            "rows": self.rows,
            "challenger_version": self.challenger_version,
            "challenger_lift": self.challenger_lift,
            "challenger_brier": self.challenger_brier,
            "live_samples": self.live_samples,
            "champion_live_lift": (
                self.champion_live.lift if self.champion_live else None
            ),
            "challenger_live_lift": (
                self.challenger_live.lift if self.challenger_live else None
            ),
            "champion_live_brier": (
                self.champion_live.brier if self.champion_live else None
            ),
            "challenger_live_brier": (
                self.challenger_live.brier if self.challenger_live else None
            ),
            "promoted": 1 if self.promoted else 0,
            "reason": self.reason[:500],
            "metrics": self.metrics,
        }

    def summary(self) -> str:
        verdict = "PROMOTED" if self.promoted else "HELD"
        lines = [
            f"{verdict}: {self.rows} rows from {self.symbols} symbols "
            f"on {self.timeframe}"
        ]
        if self.challenger_lift:
            lines.append(
                f"  challenger lift {self.challenger_lift:.2f} on held-out history"
            )
        if self.champion_live and self.challenger_live:
            lines.append(
                f"  on {self.live_samples} real trades: champion "
                f"{self.champion_live.lift:.2f} vs challenger "
                f"{self.challenger_live.lift:.2f}"
            )
        if self.reason:
            lines.append(f"  {self.reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# reading the bot's own history
# ---------------------------------------------------------------------------


def collect_live_outcomes(
    repositories: Any, since: int = 0, mode: str | None = None, limit: int = 5000
) -> list[LiveOutcome]:
    """Turn resolved journal entries into labelled rows.

    Entries without a stored feature vector or without a usable outcome are
    skipped rather than guessed at -- a fabricated row here would poison the
    only honest evaluation set the system has.
    """

    if repositories is None:
        return []
    try:
        rows = repositories.journal.resolved_entries(
            since=since, mode=mode, limit=limit
        )
    except Exception as exc:  # noqa: BLE001 - learning never breaks trading
        log.warning("could not read resolved decisions: %s", exc)
        return []

    out: list[LiveOutcome] = []
    for row in rows:
        trace = row.get("trace") or {}
        features = trace.get("features") or {}
        if not features:
            continue

        outcome = row.get("outcome") or {}
        if "r_multiple" not in outcome and "won" not in outcome:
            continue

        side_text = str(row.get("side") or "").upper()
        if side_text.startswith("LONG"):
            side = 1
        elif side_text.startswith("SHORT"):
            side = -1
        else:
            continue

        r_multiple = float(outcome.get("r_multiple", 0.0) or 0.0)
        won = bool(outcome.get("won", r_multiple > 0))

        out.append(
            LiveOutcome(
                decision_id=str(row.get("decision_id", "")),
                ts=int(row.get("ts", 0)),
                symbol=str(row.get("symbol", "")),
                side=side,
                features={k: float(v) for k, v in features.items() if _finite(v)},
                won=won,
                r_multiple=r_multiple,
            )
        )
    return out


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


# ---------------------------------------------------------------------------
# grading a model against real trades
# ---------------------------------------------------------------------------


def score_on_outcomes(
    artifact: ModelArtifact | None,
    outcomes: Sequence[LiveOutcome],
    predictor_factory: Any = None,
) -> LiveScore:
    """Ask a model to rank trades it never saw, and measure whether it could.

    The model is given each decision's feature vector and the side that was
    taken, and asked for P(win).  We then check whether the trades it scored
    highest actually won more often than the ones it scored lowest.  That is the
    only property a meta model needs to be worth deploying.
    """

    score = LiveScore(samples=len(outcomes))
    if artifact is None or not outcomes:
        return score

    from app.ml.predict import Predictor

    predictor = (predictor_factory or Predictor)(artifact=artifact)
    if not predictor.ready:
        return score

    scores: list[float] = []
    labels: list[int] = []
    for outcome in outcomes:
        if artifact.kind == "meta":
            probability = predictor.predict_meta(outcome.features, outcome.side)
        else:
            p_long, p_short, _ = predictor.predict(outcome.features)
            probability = p_long if outcome.side > 0 else p_short
        if probability is None:
            continue
        scores.append(float(probability))
        labels.append(1 if outcome.won else 0)

    score.scored = len(scores)
    if len(scores) < 10:
        return score

    lift = _decile_lift(scores, labels)
    score.lift = float(lift["lift"])
    score.top_win_rate = float(lift["top_win_rate"])
    score.bottom_win_rate = float(lift["bottom_win_rate"])
    score.brier = brier_score(scores, labels)
    return score


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------


class AutoTrainer:
    def __init__(
        self,
        settings: Any,
        registry: Any,
        repositories: Any = None,
        promotion_margin: float = DEFAULT_PROMOTION_MARGIN,
        min_live_samples: int = DEFAULT_MIN_LIVE_SAMPLES,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.repositories = repositories
        self.promotion_margin = promotion_margin
        self.min_live_samples = min_live_samples
        self.last_run_ts = 0.0
        self.cycles = 0
        self.promotions = 0
        self.last_cycle: TrainingCycle | None = None

    # -- scheduling --------------------------------------------------------

    def due(self, now: float | None = None) -> bool:
        """Whether enough time *and* enough new evidence have accumulated."""

        now = now if now is not None else time.time()
        interval = self.settings.auto_train_interval_hours * 3600
        if self.last_run_ts and now - self.last_run_ts < interval:
            return False

        # Retraining with nothing new to learn from just burns CPU and risks
        # swapping a tested model for an untested one on noise.
        needed = self.settings.auto_train_min_new_trades
        if needed > 0 and self.repositories is not None:
            try:
                resolved = self.repositories.journal.resolved_count(
                    since=int(self.last_run_ts), mode=self.settings.trading_mode.value
                )
            except Exception:  # noqa: BLE001
                resolved = 0
            if self.last_run_ts and resolved < needed:
                return False
        return True

    # -- one cycle ---------------------------------------------------------

    async def run_cycle(
        self, exchange: Any, trigger: str = "scheduled"
    ) -> TrainingCycle:
        """Train a challenger, grade it honestly, promote it only if it earns it."""

        settings = self.settings
        timeframe = Timeframe(settings.auto_train_timeframe)
        cycle = TrainingCycle(
            mode="meta", trigger=trigger, timeframe=timeframe.value
        )
        self.cycles += 1
        self.last_run_ts = time.time()

        champion = self.registry.active(settings.ml_model_name)

        # --- the bot's own results, which is the evidence that matters ----
        outcomes = collect_live_outcomes(
            self.repositories, mode=settings.trading_mode.value
        )
        cycle.live_samples = len(outcomes)

        # --- build a challenger -------------------------------------------
        try:
            samples, symbols = await self._build_dataset(exchange, timeframe)
        except Exception as exc:  # noqa: BLE001 - never break the trading loop
            cycle.reason = f"could not build a dataset: {exc}"
            log.warning("auto-train aborted: %s", exc)
            self._record(cycle)
            return cycle

        cycle.symbols = symbols
        cycle.rows = len(samples)
        if len(samples) < settings.ml_min_training_rows:
            cycle.reason = (
                f"only {len(samples)} rows, need {settings.ml_min_training_rows}"
            )
            self._record(cycle)
            return cycle

        result = train_model(
            samples,
            name=settings.ml_model_name,
            min_rows=settings.ml_min_training_rows,
            kind="meta",
            reward=settings.auto_train_profit_atr / settings.auto_train_loss_atr,
        )
        cycle.metrics = dict(result.metrics)
        cycle.challenger_lift = float(result.metrics.get("lift", 0.0) or 0.0)
        cycle.challenger_brier = float(result.metrics.get("brier_win", 0.0) or 0.0)

        if not result.accepted or result.artifact is None:
            cycle.reason = f"challenger refused: {result.reason}"
            log.info("auto-train: %s", cycle.reason)
            self._record(cycle)
            return cycle

        challenger = result.artifact
        cycle.challenger_version = challenger.version

        # --- grade both on the same real trades ---------------------------
        cycle.challenger_live = score_on_outcomes(challenger, outcomes)
        cycle.champion_live = score_on_outcomes(champion, outcomes)

        promote, reason = self._decide(champion, cycle)
        cycle.promoted = promote
        cycle.reason = reason

        if promote:
            self.registry.save(challenger, activate=True)
            self.promotions += 1
            log.warning(
                "auto-train PROMOTED %s: %s", challenger.version, reason
            )
            self._audit(cycle, challenger)
        else:
            log.info("auto-train held the champion: %s", reason)

        self.last_cycle = cycle
        self._record(cycle)
        return cycle

    # -- the promotion decision -------------------------------------------

    def _decide(
        self, champion: ModelArtifact | None, cycle: TrainingCycle
    ) -> tuple[bool, str]:
        """All-or-nothing.  A tie leaves the champion in place."""

        challenger_live = cycle.challenger_live
        champion_live = cycle.champion_live

        # --- overfitting -------------------------------------------------
        from app.governance.overfitting import OverfittingVerdict, detect_overfitting

        overfitting = detect_overfitting(
            train_score=cycle.metrics.get("base_win_rate"),
            test_score=cycle.metrics.get("top_win_rate"),
            trades=cycle.rows,
            min_trades=self.settings.ml_min_training_rows,
        )
        if overfitting.verdict is OverfittingVerdict.REJECT:
            return False, f"overfitting detector rejected it: {overfitting.summary()[:160]}"

        # --- no champion: the acceptance test is the whole bar ------------
        if champion is None:
            return True, "no active model to compare against; challenger passed on its own"

        # --- enough live evidence: that decides it ------------------------
        if (
            challenger_live is not None
            and champion_live is not None
            and challenger_live.usable
            and champion_live.usable
            and challenger_live.scored >= self.min_live_samples
        ):
            gap = challenger_live.lift - champion_live.lift
            if gap >= self.promotion_margin:
                return True, (
                    f"beat the champion on {challenger_live.scored} real trades "
                    f"(lift {challenger_live.lift:.2f} vs {champion_live.lift:.2f})"
                )
            return False, (
                f"did not beat the champion on {challenger_live.scored} real trades "
                f"(lift {challenger_live.lift:.2f} vs {champion_live.lift:.2f}, "
                f"needed +{self.promotion_margin:.2f})"
            )

        # --- thin live evidence: demand a wider gap on history ------------
        champion_lift = float(champion.metrics.get("lift", 0.0) or 0.0)
        gap = cycle.challenger_lift - champion_lift
        scored = challenger_live.scored if challenger_live else 0
        if gap >= THIN_EVIDENCE_MARGIN:
            return True, (
                f"only {scored} live outcomes to judge on, but the challenger "
                f"clears history by {gap:.2f} (>= {THIN_EVIDENCE_MARGIN:.2f})"
            )
        return False, (
            f"only {scored} live outcomes (need {self.min_live_samples}) and the "
            f"challenger leads on history by just {gap:.2f}; keeping the "
            "champion, which has already been tested with real money"
        )

    # -- dataset -----------------------------------------------------------

    async def _build_dataset(
        self, exchange: Any, timeframe: Timeframe
    ) -> tuple[list[Any], int]:
        from app.data.history import fetch_history_many
        from app.ml.meta import MetaStats
        from app.scanner.instruments import parse_allowed_classes
        from app.scanner.scanner import score_activity
        from app.scanner.universe import UniverseBuilder

        settings = self.settings
        count = settings.auto_train_symbols

        contracts = await exchange.contracts()
        tickers = await exchange.tickers()
        universe = UniverseBuilder(
            quote_currency=settings.quote_currency,
            min_quote_volume=settings.min_24h_quote_volume,
            max_spread_pct=settings.max_spread_pct,
            blacklist=settings.symbol_blacklist,
            max_symbols=max(count * 3, count),
            allowed_classes=parse_allowed_classes(
                settings.allowed_instrument_classes
            ),
        )
        candidates = universe.build(contracts, tickers)
        if not candidates:
            return [], 0

        # Rank by activity, as the manual trainer does: a symbol grinding
        # sideways contributes bars whose barriers never resolve.
        screen = await fetch_history_many(
            exchange, [c.symbol for c in candidates], timeframe, 240, concurrency=4
        )
        ranked = sorted(
            (
                (score_activity(c, screen[c.symbol]).score, c.symbol)
                for c in candidates
                if c.symbol in screen
            ),
            reverse=True,
        )
        symbols = [s for _, s in ranked[:count]]
        if not symbols:
            return [], 0

        series = await fetch_history_many(
            exchange, symbols, timeframe, settings.auto_train_bars, concurrency=3
        )

        prefix = f"{timeframe.value}_"
        stats = MetaStats()
        samples: list[Any] = []
        for symbol, candles in series.items():
            samples.extend(
                build_meta_dataset(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=candles,
                    horizon=settings.auto_train_horizon,
                    profit_atr=settings.auto_train_profit_atr,
                    loss_atr=settings.auto_train_loss_atr,
                    warmup=400,
                    stride=4,
                    stats=stats,
                    prefix=prefix,
                )
            )
        log.info("auto-train dataset: %s", stats.summary())
        return samples, len(series)

    # -- persistence -------------------------------------------------------

    def _record(self, cycle: TrainingCycle) -> None:
        self.last_cycle = cycle
        if self.repositories is None:
            return
        try:
            self.repositories.training.record(cycle.as_row())
        except Exception as exc:  # noqa: BLE001
            log.debug("could not record training run: %s", exc)

    def _audit(self, cycle: TrainingCycle, artifact: ModelArtifact) -> None:
        if self.repositories is None:
            return
        try:
            self.repositories.audit.record(
                action="MODEL_PROMOTED",
                actor="auto-trainer",
                detail={
                    "version": artifact.version,
                    "algorithm": artifact.algorithm,
                    "rows": cycle.rows,
                    "live_samples": cycle.live_samples,
                    "reason": cycle.reason,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("could not audit promotion: %s", exc)

    # -- view --------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        if self.repositories is not None:
            try:
                history = self.repositories.training.recent(limit=10)
            except Exception:  # noqa: BLE001
                history = []
        return {
            "enabled": self.settings.auto_train_enabled,
            "cycles": self.cycles,
            "promotions": self.promotions,
            "last_run_ts": int(self.last_run_ts),
            "interval_hours": self.settings.auto_train_interval_hours,
            "min_new_trades": self.settings.auto_train_min_new_trades,
            "promotion_margin": self.promotion_margin,
            "min_live_samples": self.min_live_samples,
            "last_cycle": self.last_cycle.as_row() if self.last_cycle else None,
            "history": history,
        }


__all__ = [
    "AutoTrainer",
    "LiveOutcome",
    "LiveScore",
    "TrainingCycle",
    "collect_live_outcomes",
    "score_on_outcomes",
    "DEFAULT_PROMOTION_MARGIN",
    "DEFAULT_MIN_LIVE_SAMPLES",
    "THIN_EVIDENCE_MARGIN",
]
