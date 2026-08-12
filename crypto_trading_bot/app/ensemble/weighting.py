"""Dynamic model weighting.

Models do **not** get equal weight. Each model's weight comes from its measured
hit rate *in the current regime*, blended toward a prior so a model with three
lucky calls cannot take over the vote.

The estimator is a Beta-Binomial shrinkage:

    reliability = (wins + a) / (trades + a + b)

with ``a``/``b`` chosen so an unproven model sits at the 0.5 prior and needs
real evidence to move. Weights are then bounded to ``[min_weight, max_weight]``
and normalised, so:

* no model can ever dominate, however good its record looks;
* no model is ever fully silenced by weight alone (silencing is governance's
  job, via :mod:`app.governance`, which is a deliberate, recorded action).

Performance is tracked per ``(model, regime)`` because a model that reads trends
well is often exactly the model that misreads ranges.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from app.domain import Regime

#: Beta prior. 6 pseudo-observations at 50% - enough to need ~10 real trades
#: before the estimate moves meaningfully.
PRIOR_STRENGTH = 6.0
PRIOR_MEAN = 0.5


@dataclass(slots=True)
class ModelPerformance:
    """Rolling record for one ``(model, regime)`` pair."""

    model: str
    regime: str = "ALL"
    trades: int = 0
    wins: int = 0
    r_sum: float = 0.0
    confidence_sum: float = 0.0
    last_updated: int = 0

    @property
    def hit_rate(self) -> float:
        return self.wins / self.trades if self.trades else PRIOR_MEAN

    @property
    def expectancy_r(self) -> float:
        return self.r_sum / self.trades if self.trades else 0.0

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.trades if self.trades else 0.0

    @property
    def reliability(self) -> float:
        """Shrunk hit rate - the number the weighting actually uses."""

        a = PRIOR_STRENGTH * PRIOR_MEAN
        b = PRIOR_STRENGTH * (1 - PRIOR_MEAN)
        return (self.wins + a) / (self.trades + a + b)

    @property
    def calibration_gap(self) -> float:
        """How far stated confidence sits from realised hit rate.

        Positive means the model is overconfident, which is the dangerous
        direction.
        """

        if self.trades < 10:
            return 0.0
        return self.mean_confidence - self.hit_rate

    def record(self, won: bool, r_multiple: float, confidence: float) -> None:
        self.trades += 1
        self.wins += 1 if won else 0
        self.r_sum += r_multiple
        self.confidence_sum += confidence
        self.last_updated = int(time.time())

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "regime": self.regime,
            "trades": self.trades,
            "wins": self.wins,
            "hit_rate": round(self.hit_rate, 4),
            "reliability": round(self.reliability, 4),
            "expectancy_r": round(self.expectancy_r, 4),
            "calibration_gap": round(self.calibration_gap, 4),
            "last_updated": self.last_updated,
        }


class WeightTable:
    """Maps ``(model, regime)`` to a bounded, normalised weight."""

    def __init__(
        self,
        min_weight: float = 0.02,
        max_weight: float = 0.25,
        regime_blend: float = 0.6,
        min_regime_trades: int = 15,
    ) -> None:
        self.min_weight = min_weight
        self.max_weight = max_weight
        #: How much the regime-specific record counts versus the overall one,
        #: once there is enough regime-specific evidence.
        self.regime_blend = regime_blend
        self.min_regime_trades = min_regime_trades
        self._records: dict[tuple[str, str], ModelPerformance] = {}
        #: Governance can damp a model without removing it from the vote.
        self._overrides: dict[str, float] = {}

    # -- recording --------------------------------------------------------

    def record(
        self,
        model: str,
        regime: Regime | str,
        won: bool,
        r_multiple: float,
        confidence: float,
    ) -> None:
        regime_key = regime.value if isinstance(regime, Regime) else str(regime)
        # dict.fromkeys keeps insertion order and drops the duplicate when the
        # caller passes the literal regime "ALL" -- otherwise the aggregate
        # record was written twice for the same trade, inflating its trade count
        # and pulling its shrunk reliability further from the prior than one
        # outcome warrants.
        for key in dict.fromkeys(((model, "ALL"), (model, regime_key))):
            record = self._records.get(key)
            if record is None:
                record = ModelPerformance(model=key[0], regime=key[1])
                self._records[key] = record
            record.record(won, r_multiple, confidence)

    def performance(self, model: str, regime: Regime | str = "ALL") -> ModelPerformance:
        regime_key = regime.value if isinstance(regime, Regime) else str(regime)
        return self._records.get(
            (model, regime_key), ModelPerformance(model=model, regime=regime_key)
        )

    def set_override(self, model: str, multiplier: float) -> None:
        """Governance hook: scale a model's weight (e.g. on drift)."""

        self._overrides[model] = max(0.0, min(multiplier, 1.0))

    def clear_override(self, model: str) -> None:
        self._overrides.pop(model, None)

    # -- weighting --------------------------------------------------------

    def reliability(self, model: str, regime: Regime | str) -> float:
        overall = self.performance(model, "ALL")
        specific = self.performance(model, regime)

        if specific.trades >= self.min_regime_trades:
            blended = (
                self.regime_blend * specific.reliability
                + (1 - self.regime_blend) * overall.reliability
            )
        else:
            blended = overall.reliability

        # Overconfident models are damped: if a model claims 80% and hits 50%,
        # its influence should fall even if 50% is a decent hit rate.
        gap = max(overall.calibration_gap, specific.calibration_gap)
        if gap > 0.1:
            blended *= max(0.5, 1.0 - gap)
        return blended

    def weights(
        self,
        models: Iterable[str],
        regime: Regime | str = "ALL",
    ) -> dict[str, float]:
        """Normalised weights for the given models in the given regime."""

        names = list(models)
        if not names:
            return {}

        raw: dict[str, float] = {}
        for name in names:
            reliability = self.reliability(name, regime)
            # Weight rises with reliability, proportionately.
            #
            # This used to be `max(reliability - 0.35, 0.01) ** 1.5`, and the
            # offset was the problem.  Shrunk reliabilities sit in a narrow band
            # around the 0.5 prior for small samples, so subtracting 0.35 turned
            # that band into an enormous lever: the Beta shrinkage moved a model
            # with one loss from 0.5000 to 0.4286 -- correctly modest -- and the
            # transform then blew that 14% difference up into a 2.64x difference
            # in weight.
            #
            # The consequence was perverse.  After a single losing trade the ten
            # models that had an opinion were demoted to 3.9% each while the six
            # that stayed silent -- including one that cannot vote at all and one
            # that never votes on direction by design -- rose to 10.2% each and
            # held 61% of the vote between them.  Weight accrued to models for
            # not participating, and because participation is measured against
            # total available weight, it dragged the ensemble's participation
            # down to ~30% and blocked entries on `thin_participation`.
            #
            # Squaring keeps real skill rewarded -- 0.65 outweighs 0.40 by 2.6x,
            # the same spread as before but now requiring actual evidence -- while
            # one trade moves weight by 1.36x instead of 2.64x.
            score = max(reliability, 0.05) ** 2
            raw[name] = score * self._overrides.get(name, 1.0)

        total = sum(raw.values())
        if total <= 0:
            equal = 1.0 / len(names)
            return {name: equal for name in names}

        weights = {name: value / total for name, value in raw.items()}
        return self._bound(weights)

    def _bound(self, weights: dict[str, float]) -> dict[str, float]:
        """Clamp into ``[min, max]`` and renormalise, iterating to a fixed point."""

        names = list(weights)
        if len(names) == 1:
            return {names[0]: 1.0}

        # With n models, min_weight * n must leave room; shrink the floor if not.
        floor = min(self.min_weight, 0.9 / len(names))
        ceiling = max(self.max_weight, 1.0 / len(names))

        bounded = dict(weights)
        for _ in range(12):
            clipped = {n: min(max(w, floor), ceiling) for n, w in bounded.items()}
            total = sum(clipped.values())
            if total <= 0:
                break
            renormalised = {n: w / total for n, w in clipped.items()}
            if all(
                abs(renormalised[n] - clipped[n]) < 1e-9 for n in names
            ):
                bounded = renormalised
                break
            bounded = renormalised
        return {n: round(w, 6) for n, w in bounded.items()}

    # -- persistence ------------------------------------------------------

    def as_rows(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self._records.values()]

    def load_rows(self, rows: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        for row in rows:
            try:
                record = ModelPerformance(
                    model=str(row["model"]),
                    regime=str(row.get("regime", "ALL")),
                    trades=int(row.get("trades", 0)),
                    wins=int(row.get("wins", 0)),
                    r_sum=float(row.get("r_sum", 0.0)),
                    confidence_sum=float(row.get("confidence_sum", 0.0)),
                    last_updated=int(row.get("last_updated", 0)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            self._records[(record.model, record.regime)] = record
            count += 1
        return count

    def describe(self, models: Iterable[str], regime: Regime | str = "ALL") -> list[dict[str, Any]]:
        weights = self.weights(models, regime)
        out = []
        for name, weight in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
            performance = self.performance(name, regime)
            out.append(
                {
                    "model": name,
                    "weight": weight,
                    "reliability": round(self.reliability(name, regime), 4),
                    "trades": performance.trades,
                    "hit_rate": round(performance.hit_rate, 4),
                    "expectancy_r": round(performance.expectancy_r, 4),
                    "calibration_gap": round(performance.calibration_gap, 4),
                    "override": self._overrides.get(name, 1.0),
                }
            )
        return out


__all__ = ["WeightTable", "ModelPerformance", "PRIOR_STRENGTH"]
