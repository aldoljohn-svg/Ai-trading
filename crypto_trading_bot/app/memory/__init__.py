"""Experience memory.

* :mod:`app.memory.journal` - append-only decision journal with the full
  decision trace, so any past decision can be replayed and explained
* :mod:`app.memory.excursion` - MAE/MFE analysis feeding stop and target
  placement back from realised outcomes
* :mod:`app.memory.market_memory` - historical analogues: "when conditions
  looked like this before, what actually happened?"
* :mod:`app.memory.clustering` - trade clustering and loss-cluster detection

The rule that governs this package: history informs priors, it never
guarantees outcomes. Every analogue result is reported as a *distribution* with
its sample size, never as a prediction.
"""

from app.memory.clustering import ClusterReport, LossCluster, cluster_trades, detect_loss_cluster
from app.memory.excursion import ExcursionStats, analyse_excursions, suggest_levels
from app.memory.journal import DecisionJournal, DecisionRecord, DecisionTrace
from app.memory.market_memory import Analogue, MarketMemory, MarketState

__all__ = [
    "DecisionJournal",
    "DecisionRecord",
    "DecisionTrace",
    "ExcursionStats",
    "analyse_excursions",
    "suggest_levels",
    "MarketMemory",
    "MarketState",
    "Analogue",
    "cluster_trades",
    "detect_loss_cluster",
    "ClusterReport",
    "LossCluster",
]
