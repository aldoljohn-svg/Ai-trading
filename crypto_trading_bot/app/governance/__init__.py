"""Model and strategy governance.

Nothing reaches production by being interesting. The promotion path is::

    BACKTEST -> OUT-OF-SAMPLE -> WALK-FORWARD -> SHADOW -> PAPER -> APPROVAL

and every gate is objective, recorded and re-checkable.

Contains:

* :mod:`app.governance.registry` - strategy/model registry with performance
  statistics, lifecycle status and champion/challenger comparison
* :mod:`app.governance.shadow` - shadow-mode recording of hypothetical trades
* :mod:`app.governance.drift` - detects decay against the long-run record
* :mod:`app.governance.overfitting` - train/test gap, parameter sensitivity,
  performance instability
* :mod:`app.governance.leakage` - assertions that catch lookahead before it
  reaches a model
"""

from app.governance.drift import DriftReport, detect_drift
from app.governance.leakage import LeakageError, LeakageGuard, assert_no_lookahead
from app.governance.overfitting import OverfittingReport, detect_overfitting
from app.governance.registry import (
    PromotionDecision,
    StrategyRecord,
    StrategyRegistry,
    StrategyStatus,
)
from app.governance.shadow import ShadowBook, ShadowTrade

__all__ = [
    "StrategyRegistry",
    "StrategyRecord",
    "StrategyStatus",
    "PromotionDecision",
    "ShadowBook",
    "ShadowTrade",
    "DriftReport",
    "detect_drift",
    "OverfittingReport",
    "detect_overfitting",
    "LeakageGuard",
    "LeakageError",
    "assert_no_lookahead",
]
