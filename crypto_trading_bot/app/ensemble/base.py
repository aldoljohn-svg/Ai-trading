"""Contracts shared by every analytical model.

A model is a pure function of a :class:`ModelContext` (which carries only
information that existed at the decision timestamp) to a :class:`ModelOutput`.
Purity matters: it is what lets the same model run in live, paper, backtest and
shadow mode, and what makes the decision replay in
:mod:`app.memory.journal` reproduce exactly what happened.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from app.domain import Bias, Regime, Side, Timeframe


class ModelSignal(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    #: The model could not form a view - missing data, insufficient history, or
    #: a precondition it needs was absent.  Distinct from NEUTRAL, which is an
    #: actual "no directional edge" opinion.
    NO_SIGNAL = "NO_SIGNAL"

    @property
    def direction(self) -> int:
        return {
            ModelSignal.LONG: 1,
            ModelSignal.SHORT: -1,
            ModelSignal.NEUTRAL: 0,
            ModelSignal.NO_SIGNAL: 0,
        }[self]

    @property
    def is_directional(self) -> bool:
        return self in (ModelSignal.LONG, ModelSignal.SHORT)

    @classmethod
    def from_direction(cls, direction: int) -> "ModelSignal":
        if direction > 0:
            return cls.LONG
        if direction < 0:
            return cls.SHORT
        return cls.NEUTRAL

    @classmethod
    def from_bias(cls, bias: Bias) -> "ModelSignal":
        return {
            Bias.BULLISH: cls.LONG,
            Bias.BEARISH: cls.SHORT,
            Bias.NEUTRAL: cls.NEUTRAL,
            Bias.CONFLICT: cls.NEUTRAL,
        }[bias]


@dataclass(slots=True)
class ModelContext:
    """Everything a model may look at.

    Constructed once per symbol per cycle by
    :class:`~app.ensemble.engine.EnsembleEngine`.  Nothing in here may contain
    information from after ``ts`` - the ``SymbolAnalysis`` it wraps is built
    from closed candles only, and the derived engines inherit that guarantee.
    """

    symbol: str
    ts: int
    analysis: Any                       # app.scanner.scanner.SymbolAnalysis
    regime: Regime = Regime.UNKNOWN
    order_flow: Any = None              # app.orderflow.order_flow.OrderFlowRead
    microstructure: Any = None          # app.orderflow.microstructure.MicrostructureRead
    derivatives: Any = None             # app.orderflow.derivatives.DerivativesRead
    liquidity: Any = None               # app.orderflow.liquidity_pools.LiquidityMap
    fundamentals: Any = None            # app.fundamental.sentiment.FundamentalSnapshot
    session: Any = None                 # app.sessions.session_intel.SessionRead
    predictor: Any = None               # app.ml.predict.Predictor
    features: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def execution(self) -> Any:
        """The execution-timeframe analysis, or ``None``."""

        return getattr(self.analysis, "primary", None)

    @property
    def context_tf(self) -> Any:
        analysis = self.analysis
        if analysis is None:
            return None
        return analysis.timeframes.get(Timeframe.H1) or self.execution

    @property
    def price(self) -> float:
        analysis = self.analysis
        return getattr(analysis, "close", 0.0) if analysis else 0.0

    @property
    def atr(self) -> float:
        execution = self.execution
        return getattr(execution, "atr", 0.0) if execution else 0.0


@dataclass(slots=True)
class ModelOutput:
    """One model's view.

    ``confidence`` is the model's own certainty in ``signal`` and must be in
    ``[0, 1]``.  It is *not* a probability of profit - the ensemble and the
    calibration layer are what turn these into something with a frequentist
    meaning.
    """

    name: str
    signal: ModelSignal = ModelSignal.NO_SIGNAL
    confidence: float = 0.0
    expected_move_atr: float = 0.0      # magnitude only, in ATR
    risk: float = 0.5                   # 0 = benign, 1 = hostile
    data_quality: float = 1.0           # 0 = unusable, 1 = complete
    reasoning: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    #: False when the model has no data source and could never have voted.
    #: Excluded from the participation denominator; see `unavailable`.
    available: bool = True

    def __post_init__(self) -> None:
        self.confidence = _clip(self.confidence)
        self.risk = _clip(self.risk)
        self.data_quality = _clip(self.data_quality)
        if self.signal is ModelSignal.NO_SIGNAL:
            self.confidence = 0.0

    @property
    def usable(self) -> bool:
        return self.signal is not ModelSignal.NO_SIGNAL and not self.error

    @property
    def direction(self) -> int:
        return self.signal.direction

    @property
    def effective_confidence(self) -> float:
        """Confidence discounted by data quality - a model that is confident on
        incomplete data does not get to be as loud as one that is not."""

        return self.confidence * self.data_quality

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signal": self.signal.value,
            "confidence": round(self.confidence, 4),
            "effective_confidence": round(self.effective_confidence, 4),
            "expected_move_atr": round(self.expected_move_atr, 4),
            "risk": round(self.risk, 4),
            "data_quality": round(self.data_quality, 4),
            "reasoning": self.reasoning[:6],
            "error": self.error,
        }

    def summary(self) -> str:
        if not self.usable:
            return f"{self.name}: NO SIGNAL ({self.error or 'insufficient data'})"
        return f"{self.name}: {self.signal.value} {self.confidence:.0%}"

    @classmethod
    def no_signal(cls, name: str, reason: str, data_quality: float = 0.0) -> "ModelOutput":
        """The model looked at this bar and formed no view.

        A genuine abstention.  It counts against participation, because a model
        that could have spoken and did not is telling you something.
        """

        return cls(
            name=name,
            signal=ModelSignal.NO_SIGNAL,
            data_quality=data_quality,
            error=reason,
        )

    @classmethod
    def unavailable(cls, name: str, reason: str) -> "ModelOutput":
        """The model has no data source at all and was never able to vote.

        Structurally different from an abstention, and the difference matters.
        An untrained ML model or an unconfigured news feed is not *undecided* --
        it was never in the room.  Counting it against participation
        permanently caps the ensemble's conviction for a reason that has
        nothing to do with the setup being looked at, and then that depressed
        conviction is penalised again through model risk and trade quality.

        These outputs are excluded from the participation denominator entirely,
        so participation measures "how many of the models that *could* vote
        did" rather than "how much of the system is configured".
        """

        return cls(
            name=name,
            signal=ModelSignal.NO_SIGNAL,
            data_quality=0.0,
            error=reason,
            available=False,
        )


class AnalyticalModel(abc.ABC):
    """Base class for every ensemble member."""

    #: Stable identifier used for weight lookup and persistence.
    name: str = "model"
    #: Human-readable family, used for grouping in the dashboard.
    family: str = "general"
    #: Whether this model needs order-flow data to say anything.
    requires_order_flow: bool = False

    @abc.abstractmethod
    def evaluate(self, context: ModelContext) -> ModelOutput:
        """Return this model's view.  Must never raise."""

    def safe_evaluate(self, context: ModelContext) -> ModelOutput:
        """Wrapper used by the ensemble so one broken model cannot stop a cycle."""

        try:
            output = self.evaluate(context)
        except Exception as exc:  # noqa: BLE001 - isolate model failures
            return ModelOutput.no_signal(self.name, f"{type(exc).__name__}: {exc}")
        if output.name != self.name:
            output.name = self.name
        return output


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    if value != value:                  # NaN
        return low
    return low if value < low else high if value > high else value


__all__ = [
    "AnalyticalModel",
    "ModelContext",
    "ModelOutput",
    "ModelSignal",
]
