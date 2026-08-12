"""The ensemble combiner.

Takes every model's :class:`~app.ensemble.base.ModelOutput` and produces one
directional view plus the *disagreement* structure, which downstream layers
treat as first-class information rather than noise to be averaged away.

How the vote works
------------------
1. Each usable, directional model contributes ``weight x effective_confidence``
   to its side.
2. ``agreement`` is the winning side's share of the *directional* vote.
3. ``participation`` is how much of the total available weight actually
   produced a directional opinion - low participation means the ensemble is
   deciding on thin evidence even if the models present agree.
4. The final confidence multiplies the vote margin by agreement and
   participation, so three loud models out of fifteen cannot look like a
   consensus.

Risk-family models (``portfolio_risk``) never vote on direction; they
contribute only to the aggregate risk reading.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.domain import Regime, Side
from app.ensemble.base import AnalyticalModel, ModelContext, ModelOutput, ModelSignal
from app.ensemble.weighting import WeightTable
from app.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class EnsembleResult:
    symbol: str
    ts: int
    signal: ModelSignal = ModelSignal.NO_SIGNAL
    confidence: float = 0.0
    agreement: float = 0.0              # winning share of the directional vote
    participation: float = 0.0          # share of weight that had an opinion
    dissent: float = 0.0                # weight voting the other way
    expected_move_atr: float = 0.0
    aggregate_risk: float = 0.5
    data_quality: float = 0.0
    outputs: list[ModelOutput] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    regime: Regime = Regime.UNKNOWN

    # -- derived ----------------------------------------------------------

    @property
    def side(self) -> Side | None:
        if self.signal is ModelSignal.LONG:
            return Side.LONG
        if self.signal is ModelSignal.SHORT:
            return Side.SHORT
        return None

    @property
    def usable_models(self) -> list[ModelOutput]:
        return [o for o in self.outputs if o.usable]

    @property
    def directional_models(self) -> list[ModelOutput]:
        return [o for o in self.outputs if o.signal.is_directional]

    @property
    def agreeing_models(self) -> list[ModelOutput]:
        return [o for o in self.outputs if o.signal is self.signal]

    @property
    def opposing_models(self) -> list[ModelOutput]:
        if not self.signal.is_directional:
            return []
        opposite = ModelSignal.SHORT if self.signal is ModelSignal.LONG else ModelSignal.LONG
        return [o for o in self.outputs if o.signal is opposite]

    @property
    def model_agreement_label(self) -> str:
        """``8/11`` style label - agreeing over models that had any opinion."""

        return f"{len(self.agreeing_models)}/{len(self.usable_models)}"

    def top_reasons(self, limit: int = 6) -> list[str]:
        """Reasoning from the highest-weighted agreeing models."""

        agreeing = sorted(
            self.agreeing_models,
            key=lambda o: self.weights.get(o.name, 0.0) * o.effective_confidence,
            reverse=True,
        )
        reasons: list[str] = []
        for output in agreeing:
            for reason in output.reasoning:
                if reason not in reasons:
                    reasons.append(reason)
                if len(reasons) >= limit:
                    return reasons
        return reasons

    def objections(self, limit: int = 4) -> list[str]:
        out: list[str] = []
        for output in sorted(
            self.opposing_models,
            key=lambda o: self.weights.get(o.name, 0.0) * o.effective_confidence,
            reverse=True,
        ):
            out.append(f"{output.name}: {output.reasoning[0] if output.reasoning else output.signal.value}")
            if len(out) >= limit:
                break
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ts": self.ts,
            "signal": self.signal.value,
            "confidence": round(self.confidence, 4),
            "agreement": round(self.agreement, 4),
            "participation": round(self.participation, 4),
            "dissent": round(self.dissent, 4),
            "model_agreement": self.model_agreement_label,
            "expected_move_atr": round(self.expected_move_atr, 4),
            "aggregate_risk": round(self.aggregate_risk, 4),
            "data_quality": round(self.data_quality, 4),
            "regime": self.regime.value,
            "weights": self.weights,
            "models": [o.as_dict() for o in self.outputs],
            "reasons": self.top_reasons(),
            "objections": self.objections(),
        }

    def summary(self) -> str:
        lines = [
            f"{self.symbol}: {self.signal.value} @ {self.confidence:.0%} "
            f"(agreement {self.model_agreement_label}, participation {self.participation:.0%})"
        ]
        for output in sorted(
            self.outputs, key=lambda o: self.weights.get(o.name, 0.0), reverse=True
        ):
            weight = self.weights.get(output.name, 0.0)
            lines.append(f"  [{weight:.3f}] {output.summary()}")
        return "\n".join(lines)


class EnsembleEngine:
    def __init__(
        self,
        models: Sequence[AnalyticalModel] | None = None,
        weights: WeightTable | None = None,
        min_participation: float = 0.35,
    ) -> None:
        from app.ensemble.models import build_default_models

        self.models = list(models) if models is not None else build_default_models()
        self.weights = weights or WeightTable()
        #: Below this share of weight actually voting, the ensemble refuses to
        #: express a direction at all.
        self.min_participation = min_participation

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self.models]

    def evaluate(self, context: ModelContext) -> EnsembleResult:
        regime = context.regime
        outputs = [model.safe_evaluate(context) for model in self.models]
        weights = self.weights.weights(self.model_names, regime)

        result = EnsembleResult(
            symbol=context.symbol,
            ts=context.ts or int(time.time()),
            outputs=outputs,
            weights=weights,
            regime=regime,
        )

        long_vote = 0.0
        short_vote = 0.0
        directional_weight = 0.0
        available_weight = 0.0
        risk_weighted = 0.0
        risk_total = 0.0
        quality_weighted = 0.0
        quality_total = 0.0
        move_weighted = 0.0
        move_total = 0.0

        for output in outputs:
            weight = weights.get(output.name, 0.0)
            if weight <= 0:
                continue

            # Risk and data quality count from every model that ran, including
            # the ones with no directional view - a model saying "this is
            # dangerous" matters even when it will not pick a side.
            if not output.error:
                risk_weighted += weight * output.risk
                risk_total += weight
                quality_weighted += weight * output.data_quality
                quality_total += weight

            # Only models that *could* vote count toward participation.
            # Excluded: the risk family, which never takes a side, and any
            # model reporting itself unavailable because it has no data source
            # at all.  An untrained ML model is not an undecided voter.
            model = next((m for m in self.models if m.name == output.name), None)
            votes_on_direction = model is None or model.family != "risk"
            if votes_on_direction and output.available:
                available_weight += weight

            if not output.usable or not output.signal.is_directional:
                continue

            contribution = weight * output.effective_confidence
            if output.signal is ModelSignal.LONG:
                long_vote += contribution
            else:
                short_vote += contribution
            directional_weight += weight
            move_weighted += contribution * output.expected_move_atr
            move_total += contribution

        result.aggregate_risk = risk_weighted / risk_total if risk_total else 0.5
        result.data_quality = quality_weighted / quality_total if quality_total else 0.0
        result.participation = (
            directional_weight / available_weight if available_weight else 0.0
        )
        result.expected_move_atr = move_weighted / move_total if move_total else 0.0

        total_vote = long_vote + short_vote
        if total_vote <= 0:
            result.signal = ModelSignal.NEUTRAL
            result.confidence = 0.0
            return result

        if long_vote >= short_vote:
            result.signal = ModelSignal.LONG
            winning, losing = long_vote, short_vote
        else:
            result.signal = ModelSignal.SHORT
            winning, losing = short_vote, long_vote

        result.agreement = winning / total_vote
        result.dissent = losing / total_vote

        if result.participation < self.min_participation:
            # Not enough of the ensemble had an opinion to call this a view.
            result.signal = ModelSignal.NEUTRAL
            result.confidence = 0.0
            return result

        # Margin of victory, scaled by how well-attended the vote was.  A 51/49
        # split at full participation is still a coin flip, and the margin says
        # so on its own: it runs from 0 at a dead heat to 1 at unanimity.
        #
        # It is deliberately NOT multiplied by `agreement` as well.  The two are
        # the same quantity -- margin == 2 * agreement - 1 -- so multiplying
        # them squares the penalty and crushes every ordinary vote toward zero.
        # A 55/45 split scored 4% conviction under that formula and 8% under
        # this one; a solid 70/30 went from 22% to 32%.  The old numbers were
        # not conservatism, they were double-counting.
        margin = (winning - losing) / total_vote
        result.confidence = _clip01(margin * (0.5 + 0.5 * result.participation))
        return result

    # -- learning ---------------------------------------------------------

    def record_outcome(
        self,
        result: EnsembleResult,
        won: bool,
        r_multiple: float,
        regime: Regime | str | None = None,
    ) -> None:
        """Attribute a closed trade back to the models that voted for it.

        Only directional models are scored, and only on the side they actually
        took: a model that voted against the trade is credited when the trade
        loses, which is what keeps contrarian value visible.
        """

        regime = regime or result.regime
        for output in result.outputs:
            if not output.signal.is_directional:
                continue
            agreed = output.signal is result.signal
            model_won = won if agreed else not won
            model_r = r_multiple if agreed else -r_multiple
            self.weights.record(
                model=output.name,
                regime=regime,
                won=model_won,
                r_multiple=model_r,
                confidence=output.confidence,
            )

    def describe_weights(self, regime: Regime | str = "ALL") -> list[dict[str, Any]]:
        return self.weights.describe(self.model_names, regime)


def _clip01(value: float) -> float:
    if value != value:
        return 0.0
    return 0.0 if value < 0 else 1.0 if value > 1 else value


__all__ = ["EnsembleEngine", "EnsembleResult"]
