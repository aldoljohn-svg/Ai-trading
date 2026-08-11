"""Risk scoring and trade quality.

Four independent risk scores, each in ``[0, 1]`` where higher is worse:

``DATA_RISK_SCORE``       is the information we are acting on trustworthy?
``MODEL_RISK_SCORE``      is the machinery producing the view trustworthy?
``EXECUTION_RISK_SCORE``  can this actually be traded at an acceptable cost?
``PORTFOLIO`` (existing)  can the book absorb it?

They combine with signal quality into ``TRADE_QUALITY_SCORE`` (0-100), and the
expected-value engine turns the whole picture into a number that must clear a
configured threshold before anything is traded.

Each score has **veto authority**: above its hard limit the trade is blocked
regardless of how attractive the signal is. That ordering is the point - a
beautiful setup on stale data is not a beautiful setup.
"""

from app.quality.data_risk import DataRiskReport, assess_data_risk
from app.quality.execution_risk import ExecutionRiskReport, assess_execution_risk
from app.quality.expected_value import ExpectedValue, compute_expected_value
from app.quality.model_risk import ModelRiskReport, assess_model_risk
from app.quality.trade_quality import TradeQuality, compute_trade_quality

__all__ = [
    "DataRiskReport",
    "assess_data_risk",
    "ModelRiskReport",
    "assess_model_risk",
    "ExecutionRiskReport",
    "assess_execution_risk",
    "TradeQuality",
    "compute_trade_quality",
    "ExpectedValue",
    "compute_expected_value",
]
