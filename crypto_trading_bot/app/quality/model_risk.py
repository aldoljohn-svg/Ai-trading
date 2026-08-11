"""MODEL_RISK_SCORE - is the machinery producing this view trustworthy?

Distinct from data risk. The data can be perfect while the models are drifting,
poorly calibrated, disagreeing with each other, or operating in a regime none of
them has a track record in.

Contributors:

* **disagreement** - the ensemble is split
* **thin participation** - few models had an opinion
* **calibration gap** - models claim more confidence than they historically earn
* **drift** - recent performance has decayed against the long-run record
* **inexperience** - little or no history in the current regime
* **ML absence/failure** - the learned model is missing or erroring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain import Regime

HARD_LIMIT = 0.75


@dataclass(slots=True)
class ModelRiskReport:
    score: float = 0.0
    disagreement: float = 0.0
    participation: float = 1.0
    calibration_gap: float = 0.0
    drift: float = 0.0
    regime_experience: int = 0
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.score >= HARD_LIMIT

    @property
    def level(self) -> str:
        if self.score >= HARD_LIMIT:
            return "CRITICAL"
        if self.score >= 0.45:
            return "ELEVATED"
        return "OK"

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "level": self.level,
            "blocked": self.blocked,
            "disagreement": round(self.disagreement, 4),
            "participation": round(self.participation, 4),
            "calibration_gap": round(self.calibration_gap, 4),
            "drift": round(self.drift, 4),
            "regime_experience": self.regime_experience,
            "problems": self.problems,
            "notes": self.notes,
        }


def assess_model_risk(
    ensemble: Any,
    weight_table: Any = None,
    drift_report: Any = None,
    predictor: Any = None,
    regime: Regime | str = "ALL",
    min_regime_trades: int = 20,
) -> ModelRiskReport:
    report = ModelRiskReport()
    penalties: list[float] = []

    if ensemble is None:
        report.score = 1.0
        report.problems.append("no ensemble result")
        return report

    # --- disagreement -----------------------------------------------------
    report.disagreement = getattr(ensemble, "dissent", 0.0)
    if report.disagreement > 0.35:
        penalties.append(min(0.7, report.disagreement))
        report.problems.append(
            f"{report.disagreement:.0%} of the vote is against this direction"
        )
    elif report.disagreement > 0.2:
        penalties.append(0.15)
        report.notes.append(f"minority dissent {report.disagreement:.0%}")

    # --- participation ------------------------------------------------------
    report.participation = getattr(ensemble, "participation", 1.0)
    if report.participation < 0.4:
        penalties.append(0.5 * (1 - report.participation / 0.4))
        report.problems.append(
            f"only {report.participation:.0%} of model weight had an opinion"
        )

    # --- calibration ---------------------------------------------------------
    if weight_table is not None:
        gaps: list[float] = []
        experience = 0
        for output in getattr(ensemble, "outputs", []):
            performance = weight_table.performance(output.name, regime)
            experience = max(experience, performance.trades)
            if performance.trades >= 10:
                gaps.append(performance.calibration_gap)
        report.regime_experience = experience
        if gaps:
            report.calibration_gap = max(gaps)
            if report.calibration_gap > 0.15:
                penalties.append(min(0.6, report.calibration_gap * 2))
                report.problems.append(
                    f"models are overconfident by {report.calibration_gap:.0%} "
                    "against their realised hit rate"
                )

        if experience < min_regime_trades:
            # Little evidence in this regime: the weights are still priors.
            shortfall = 1 - experience / max(min_regime_trades, 1)
            penalties.append(0.30 * shortfall)
            report.notes.append(
                f"only {experience} recorded outcome(s) in this regime - "
                "weights are still near their priors"
            )

    # --- drift ---------------------------------------------------------------
    if drift_report is not None:
        report.drift = float(getattr(drift_report, "severity", 0.0) or 0.0)
        if report.drift > 0.5:
            penalties.append(min(0.8, report.drift))
            report.problems.append(
                f"model drift detected (severity {report.drift:.2f})"
            )
        elif report.drift > 0.25:
            penalties.append(0.2)
            report.notes.append(f"mild model drift ({report.drift:.2f})")

    # --- learned model health -------------------------------------------------
    if predictor is not None:
        info = predictor.info() if hasattr(predictor, "info") else {}
        if not info.get("loaded"):
            # Not a fault - the system is designed to run on rules - but it is
            # one fewer independent check.
            report.notes.append("no trained ML model loaded; running on rules")
            penalties.append(0.08)
        else:
            failures = int(info.get("failures", 0) or 0)
            if failures > 5:
                penalties.append(min(0.5, failures / 50))
                report.problems.append(f"{failures} ML prediction failures")

    report.score = _noisy_or(penalties)
    return report


def _noisy_or(penalties: list[float]) -> float:
    survival = 1.0
    for penalty in penalties:
        survival *= 1.0 - max(0.0, min(penalty, 1.0))
    return 1.0 - survival


__all__ = ["ModelRiskReport", "assess_model_risk", "HARD_LIMIT"]
