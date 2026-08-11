"""Multi-model ensemble decision layer.

Instead of one model deciding, fifteen *independent* analytical models each
produce a signal, a confidence, an expected move, a risk estimate, reasoning and
a data-quality figure.  An ensemble combines them with weights that adapt to the
current regime based on measured historical reliability.

Design rules that the rest of the system depends on:

* A model that has no opinion returns ``ModelSignal.NO_SIGNAL`` with
  ``data_quality`` describing why.  It contributes nothing rather than voting
  neutral, which is a different thing.
* Disagreement between models is itself information: it feeds the no-trade model
  and lowers trade quality.
* Weights are bounded and normalised, so no single model can dominate the vote
  even if its measured reliability is extreme.
* The ensemble produces a *proposal input*, never an order. Risk still has veto.
"""

from app.ensemble.base import (
    AnalyticalModel,
    ModelContext,
    ModelOutput,
    ModelSignal,
)
from app.ensemble.engine import EnsembleEngine, EnsembleResult
from app.ensemble.models import ALL_MODELS, build_default_models
from app.ensemble.no_trade import NoTradeModel, NoTradeVerdict
from app.ensemble.weighting import ModelPerformance, WeightTable

__all__ = [
    "AnalyticalModel",
    "ModelContext",
    "ModelOutput",
    "ModelSignal",
    "EnsembleEngine",
    "EnsembleResult",
    "ALL_MODELS",
    "build_default_models",
    "NoTradeModel",
    "NoTradeVerdict",
    "WeightTable",
    "ModelPerformance",
]
