"""Safety layers.

The risk hierarchy this package enforces, highest authority first::

    HUMAN EMERGENCY CONTROL
      -> HARD RISK LIMITS
      -> KILL SWITCH
      -> PORTFOLIO RISK
      -> EXECUTION RISK
      -> STRATEGY
      -> AI SIGNAL

Nothing below a level may relax anything above it. In particular no model,
score or ensemble can raise a hard limit, clear a kill switch, or override a
human control - the code paths for those simply do not exist.
"""

from app.safety.anomaly import AnomalyReport, AnomalyType, detect_anomalies
from app.safety.behaviour import (
    BehaviourGuard,
    OvertradingReport,
    RecoveryState,
)
from app.safety.failsafe import FailSafe, FailSafeReport, SystemFault
from app.safety.kill_switch import (
    KillSwitchLevel,
    KillSwitchState,
    MultiLayerKillSwitch,
    Trigger,
)

__all__ = [
    "MultiLayerKillSwitch",
    "KillSwitchLevel",
    "KillSwitchState",
    "Trigger",
    "AnomalyReport",
    "AnomalyType",
    "detect_anomalies",
    "FailSafe",
    "FailSafeReport",
    "SystemFault",
    "BehaviourGuard",
    "OvertradingReport",
    "RecoveryState",
]
