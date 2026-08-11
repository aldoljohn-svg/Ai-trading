"""Trade clustering and loss-cluster detection.

Groups closed trades by the conditions they were taken in (strategy, regime,
symbol, session, side) and looks for *conditions* that systematically lose,
rather than for streaks.

The distinction matters: five losses in a row across five different regimes is
variance. Five losses in a row all in `TRANSITION` on the same symbol is a
broken assumption, and the response is investigation, not a bigger position.

Explicitly **not** a revenge-trading trigger: the only actions this can motivate
are reducing risk, blocking a condition, or asking a human to look.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(slots=True)
class Cluster:
    key: str
    dimension: str
    trades: int = 0
    wins: int = 0
    total_r: float = 0.0
    r_values: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def expectancy_r(self) -> float:
        return self.total_r / self.trades if self.trades else 0.0

    @property
    def worst_streak(self) -> int:
        streak = 0
        worst = 0
        for r in self.r_values:
            if r <= 0:
                streak += 1
                worst = max(worst, streak)
            else:
                streak = 0
        return worst

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "dimension": self.dimension,
            "trades": self.trades,
            "win_rate": round(self.win_rate, 4),
            "expectancy_r": round(self.expectancy_r, 4),
            "total_r": round(self.total_r, 3),
            "worst_streak": self.worst_streak,
        }


@dataclass(slots=True)
class ClusterReport:
    clusters: list[Cluster] = field(default_factory=list)

    def by_dimension(self, dimension: str) -> list[Cluster]:
        return sorted(
            [c for c in self.clusters if c.dimension == dimension],
            key=lambda c: c.expectancy_r,
        )

    def worst(self, min_trades: int = 5, limit: int = 5) -> list[Cluster]:
        eligible = [c for c in self.clusters if c.trades >= min_trades]
        return sorted(eligible, key=lambda c: c.expectancy_r)[:limit]

    def best(self, min_trades: int = 5, limit: int = 5) -> list[Cluster]:
        eligible = [c for c in self.clusters if c.trades >= min_trades]
        return sorted(eligible, key=lambda c: c.expectancy_r, reverse=True)[:limit]

    def as_dict(self) -> dict[str, Any]:
        return {
            "clusters": [c.as_dict() for c in self.clusters],
            "worst": [c.as_dict() for c in self.worst()],
            "best": [c.as_dict() for c in self.best()],
        }


@dataclass(slots=True)
class LossCluster:
    detected: bool = False
    dimension: str = ""
    key: str = ""
    trades: int = 0
    expectancy_r: float = 0.0
    win_rate: float = 0.0
    description: str = ""
    recommendation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "dimension": self.dimension,
            "key": self.key,
            "trades": self.trades,
            "expectancy_r": round(self.expectancy_r, 4),
            "win_rate": round(self.win_rate, 4),
            "description": self.description,
            "recommendation": self.recommendation,
        }


DIMENSIONS = ("strategy", "regime", "symbol", "session", "side")


def cluster_trades(
    trades: Sequence[Mapping[str, Any]],
    dimensions: Sequence[str] = DIMENSIONS,
) -> ClusterReport:
    """Group closed trades along each dimension."""

    buckets: dict[tuple[str, str], Cluster] = {}
    for trade in trades:
        r_multiple = float(trade.get("r_multiple", 0.0) or 0.0)
        for dimension in dimensions:
            value = trade.get(dimension)
            if value in (None, ""):
                continue
            key = (dimension, str(value))
            cluster = buckets.get(key)
            if cluster is None:
                cluster = Cluster(key=str(value), dimension=dimension)
                buckets[key] = cluster
            cluster.trades += 1
            cluster.total_r += r_multiple
            cluster.r_values.append(r_multiple)
            if r_multiple > 0:
                cluster.wins += 1

    return ClusterReport(clusters=list(buckets.values()))


def detect_loss_cluster(
    trades: Sequence[Mapping[str, Any]],
    min_trades: int = 5,
    expectancy_threshold: float = -0.25,
    win_rate_threshold: float = 0.25,
    recent_window: int = 30,
) -> LossCluster:
    """Find a *condition* that is systematically losing.

    Looks only at the most recent ``recent_window`` trades: a condition that
    lost badly six months ago and has since recovered is not a live problem.
    """

    recent = list(trades)[-recent_window:]
    if len(recent) < min_trades:
        return LossCluster(
            description=f"only {len(recent)} recent trades - not enough to judge"
        )

    report = cluster_trades(recent)
    worst: Cluster | None = None
    for cluster in report.clusters:
        if cluster.trades < min_trades:
            continue
        if cluster.expectancy_r > expectancy_threshold:
            continue
        if cluster.win_rate > win_rate_threshold:
            continue
        if worst is None or cluster.expectancy_r < worst.expectancy_r:
            worst = cluster

    if worst is None:
        return LossCluster(description="no losing condition cluster detected")

    recommendation = {
        "regime": "review whether the strategy is valid in this regime at all",
        "symbol": "consider blacklisting this symbol until it is understood",
        "session": "review session-specific liquidity and spread assumptions",
        "strategy": "move this strategy to shadow mode and investigate",
        "side": "check for a directional bias in the signal or in execution",
    }.get(worst.dimension, "investigate before trading this condition again")

    return LossCluster(
        detected=True,
        dimension=worst.dimension,
        key=worst.key,
        trades=worst.trades,
        expectancy_r=worst.expectancy_r,
        win_rate=worst.win_rate,
        description=(
            f"{worst.trades} recent trades where {worst.dimension}={worst.key} "
            f"averaged {worst.expectancy_r:+.2f}R at a {worst.win_rate:.0%} win rate"
        ),
        recommendation=recommendation,
    )


def session_of(timestamp: int) -> str:
    """UTC trading session label."""

    hour = (timestamp // 3600) % 24
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 12:
        return "LONDON"
    if 12 <= hour < 16:
        return "LONDON_NY_OVERLAP"
    if 16 <= hour < 21:
        return "NEW_YORK"
    return "LATE_US"


def weekday_of(timestamp: int) -> str:
    import time as _time

    return _time.strftime("%A", _time.gmtime(timestamp)).upper()


__all__ = [
    "cluster_trades",
    "detect_loss_cluster",
    "ClusterReport",
    "Cluster",
    "LossCluster",
    "session_of",
    "weekday_of",
    "DIMENSIONS",
]
