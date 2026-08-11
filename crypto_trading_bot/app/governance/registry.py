"""Strategy / model registry with champion-challenger promotion.

Every strategy carries a lifecycle status and a performance record. A challenger
is promoted **only** when it clears every objective criterion simultaneously -
never because it looks better over a short sample, which is how noise gets
promoted into production.

Promotion criteria (all must hold):

1. minimum out-of-sample trade count
2. better expectancy than the champion by a required margin
3. profit factor above an absolute floor
4. drawdown no worse than the champion by more than a tolerance
5. positive expectancy in the *majority* of regimes it has traded
6. a completed shadow period of minimum length
7. no active overfitting or drift flag
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class StrategyStatus(str, Enum):
    CANDIDATE = "CANDIDATE"      # exists, not yet evaluated
    SHADOW = "SHADOW"            # generating signals, cannot execute
    CHALLENGER = "CHALLENGER"    # competing against the champion
    ACTIVE = "ACTIVE"            # champion, in production
    DEGRADED = "DEGRADED"        # underperforming, weight reduced
    DISABLED = "DISABLED"        # removed from the vote

    @property
    def can_execute(self) -> bool:
        return self is StrategyStatus.ACTIVE

    @property
    def emoji(self) -> str:
        return {
            StrategyStatus.CANDIDATE: "⚪",
            StrategyStatus.SHADOW: "👤",
            StrategyStatus.CHALLENGER: "🥈",
            StrategyStatus.ACTIVE: "🥇",
            StrategyStatus.DEGRADED: "🟠",
            StrategyStatus.DISABLED: "⛔",
        }[self]


@dataclass(slots=True)
class StrategyRecord:
    name: str
    version: str = "1"
    status: StrategyStatus = StrategyStatus.CANDIDATE
    market: str = "*"
    timeframe: str = "*"
    trades: int = 0
    wins: int = 0
    gross_profit_r: float = 0.0
    gross_loss_r: float = 0.0
    r_values: list[float] = field(default_factory=list)
    regime_r: dict[str, list[float]] = field(default_factory=dict)
    max_drawdown_r: float = 0.0
    peak_r: float = 0.0
    cumulative_r: float = 0.0
    created_at: int = field(default_factory=lambda: int(time.time()))
    shadow_since: int = 0
    promoted_at: int = 0
    notes: list[str] = field(default_factory=list)

    # -- statistics -------------------------------------------------------

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def expectancy_r(self) -> float:
        return self.cumulative_r / self.trades if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss_r > 0:
            return self.gross_profit_r / self.gross_loss_r
        return float("inf") if self.gross_profit_r > 0 else 0.0

    @property
    def sharpe_like(self) -> float:
        """R-based Sharpe analogue. Not annualised - a comparison figure only."""

        if len(self.r_values) < 10:
            return 0.0
        stdev = statistics.pstdev(self.r_values)
        if stdev <= 0:
            return 0.0
        return statistics.fmean(self.r_values) / stdev

    @property
    def sortino_like(self) -> float:
        if len(self.r_values) < 10:
            return 0.0
        downside = [r for r in self.r_values if r < 0]
        if not downside:
            return 0.0
        deviation = (sum(r * r for r in downside) / len(self.r_values)) ** 0.5
        if deviation <= 0:
            return 0.0
        return statistics.fmean(self.r_values) / deviation

    @property
    def shadow_days(self) -> float:
        if not self.shadow_since:
            return 0.0
        return (time.time() - self.shadow_since) / 86400

    def regime_expectancy(self) -> dict[str, float]:
        return {
            regime: statistics.fmean(values) if values else 0.0
            for regime, values in self.regime_r.items()
        }

    def positive_regime_share(self, min_trades: int = 5) -> float:
        """Share of regimes (with enough evidence) where expectancy is positive."""

        eligible = {
            regime: values
            for regime, values in self.regime_r.items()
            if len(values) >= min_trades
        }
        if not eligible:
            return 0.0
        positive = sum(
            1 for values in eligible.values() if statistics.fmean(values) > 0
        )
        return positive / len(eligible)

    # -- recording ---------------------------------------------------------

    def record(self, r_multiple: float, regime: str = "ALL") -> None:
        self.trades += 1
        self.r_values.append(r_multiple)
        self.cumulative_r += r_multiple
        if r_multiple > 0:
            self.wins += 1
            self.gross_profit_r += r_multiple
        else:
            self.gross_loss_r += abs(r_multiple)
        self.regime_r.setdefault(regime, []).append(r_multiple)

        self.peak_r = max(self.peak_r, self.cumulative_r)
        drawdown = self.peak_r - self.cumulative_r
        self.max_drawdown_r = max(self.max_drawdown_r, drawdown)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "emoji": self.status.emoji,
            "market": self.market,
            "timeframe": self.timeframe,
            "trades": self.trades,
            "win_rate": round(self.win_rate, 4),
            "expectancy_r": round(self.expectancy_r, 4),
            "profit_factor": (
                round(self.profit_factor, 3)
                if self.profit_factor != float("inf")
                else None
            ),
            "cumulative_r": round(self.cumulative_r, 3),
            "max_drawdown_r": round(self.max_drawdown_r, 3),
            "sharpe_like": round(self.sharpe_like, 3),
            "sortino_like": round(self.sortino_like, 3),
            "regime_expectancy": {
                k: round(v, 3) for k, v in self.regime_expectancy().items()
            },
            "shadow_days": round(self.shadow_days, 2),
            "notes": self.notes[-3:],
        }


@dataclass(slots=True)
class PromotionDecision:
    promote: bool
    challenger: str
    champion: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "challenger": self.challenger,
            "champion": self.champion,
            "passed": self.passed,
            "failed": self.failed,
        }

    def summary(self) -> str:
        verdict = "PROMOTE" if self.promote else "HOLD"
        lines = [f"{verdict}: {self.challenger} vs champion {self.champion}"]
        for item in self.passed:
            lines.append(f"  ✅ {item}")
        for item in self.failed:
            lines.append(f"  ❌ {item}")
        return "\n".join(lines)


@dataclass(slots=True)
class PromotionCriteria:
    min_trades: int = 40
    min_shadow_days: float = 14.0
    expectancy_margin: float = 0.05      # challenger must beat champion by this R
    min_profit_factor: float = 1.2
    max_drawdown_tolerance: float = 1.5  # multiple of the champion's drawdown
    min_positive_regime_share: float = 0.6


class StrategyRegistry:
    def __init__(self, criteria: PromotionCriteria | None = None) -> None:
        self.criteria = criteria or PromotionCriteria()
        self._records: dict[str, StrategyRecord] = {}

    # -- lifecycle --------------------------------------------------------

    def register(
        self,
        name: str,
        version: str = "1",
        status: StrategyStatus = StrategyStatus.CANDIDATE,
        **kwargs: Any,
    ) -> StrategyRecord:
        key = f"{name}:{version}"
        record = self._records.get(key)
        if record is None:
            record = StrategyRecord(name=name, version=version, status=status, **kwargs)
            self._records[key] = record
        return record

    def get(self, name: str, version: str = "1") -> StrategyRecord | None:
        return self._records.get(f"{name}:{version}")

    def all(self) -> list[StrategyRecord]:
        return list(self._records.values())

    def by_status(self, status: StrategyStatus) -> list[StrategyRecord]:
        return [r for r in self._records.values() if r.status is status]

    def champion(self, name: str) -> StrategyRecord | None:
        candidates = [
            r
            for r in self._records.values()
            if r.name == name and r.status is StrategyStatus.ACTIVE
        ]
        return candidates[0] if candidates else None

    def set_status(self, name: str, version: str, status: StrategyStatus) -> None:
        record = self.get(name, version)
        if record is None:
            return
        record.status = status
        if status is StrategyStatus.SHADOW and not record.shadow_since:
            record.shadow_since = int(time.time())
        if status is StrategyStatus.ACTIVE:
            record.promoted_at = int(time.time())
        record.notes.append(f"status -> {status.value} at {int(time.time())}")

    def record_outcome(
        self, name: str, version: str, r_multiple: float, regime: str = "ALL"
    ) -> None:
        record = self.get(name, version) or self.register(name, version)
        record.record(r_multiple, regime)

    # -- champion / challenger ---------------------------------------------

    def evaluate_promotion(
        self,
        challenger: StrategyRecord,
        champion: StrategyRecord | None,
        drift: Any = None,
        overfitting: Any = None,
    ) -> PromotionDecision:
        """Objective, all-or-nothing promotion test."""

        criteria = self.criteria
        decision = PromotionDecision(
            promote=False,
            challenger=f"{challenger.name}:{challenger.version}",
            champion=(
                f"{champion.name}:{champion.version}" if champion else "<none>"
            ),
        )

        def check(passed: bool, description: str) -> None:
            (decision.passed if passed else decision.failed).append(description)

        # 1. sample size
        check(
            challenger.trades >= criteria.min_trades,
            f"trade count {challenger.trades} >= {criteria.min_trades}",
        )

        # 2. shadow period
        check(
            challenger.shadow_days >= criteria.min_shadow_days,
            f"shadow period {challenger.shadow_days:.1f}d >= {criteria.min_shadow_days}d",
        )

        # 3. expectancy versus champion
        if champion is None:
            check(
                challenger.expectancy_r > 0,
                f"expectancy {challenger.expectancy_r:+.3f}R > 0 (no incumbent)",
            )
        else:
            required = champion.expectancy_r + criteria.expectancy_margin
            check(
                challenger.expectancy_r >= required,
                f"expectancy {challenger.expectancy_r:+.3f}R >= "
                f"{required:+.3f}R (champion + margin)",
            )

        # 4. profit factor floor
        profit_factor = challenger.profit_factor
        check(
            profit_factor >= criteria.min_profit_factor,
            f"profit factor {profit_factor:.2f} >= {criteria.min_profit_factor}",
        )

        # 5. drawdown
        if champion is not None and champion.max_drawdown_r > 0:
            limit = champion.max_drawdown_r * criteria.max_drawdown_tolerance
            check(
                challenger.max_drawdown_r <= limit,
                f"max drawdown {challenger.max_drawdown_r:.2f}R <= {limit:.2f}R",
            )

        # 6. regime breadth
        share = challenger.positive_regime_share()
        check(
            share >= criteria.min_positive_regime_share,
            f"positive in {share:.0%} of regimes >= "
            f"{criteria.min_positive_regime_share:.0%}",
        )

        # 7. governance flags
        if drift is not None and getattr(drift, "drifting", False):
            check(False, "no active drift flag")
        else:
            check(True, "no active drift flag")

        if overfitting is not None and getattr(overfitting, "overfitted", False):
            check(False, "no active overfitting flag")
        else:
            check(True, "no active overfitting flag")

        decision.promote = not decision.failed
        return decision

    def promote(self, challenger: StrategyRecord) -> None:
        """Swap the challenger in and demote the incumbent to challenger."""

        incumbent = self.champion(challenger.name)
        if incumbent is not None and incumbent is not challenger:
            incumbent.status = StrategyStatus.CHALLENGER
            incumbent.notes.append("demoted - a challenger met every criterion")
        challenger.status = StrategyStatus.ACTIVE
        challenger.promoted_at = int(time.time())
        challenger.notes.append("promoted to champion")

    # -- ranking -------------------------------------------------------------

    def ranked(self, min_trades: int = 10) -> list[StrategyRecord]:
        """Rank by expectancy, but only strategies with enough evidence."""

        eligible = [r for r in self._records.values() if r.trades >= min_trades]
        return sorted(eligible, key=lambda r: r.expectancy_r, reverse=True)

    def describe(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in sorted(
            self._records.values(), key=lambda r: (-r.trades, r.name)
        )]


__all__ = [
    "StrategyRegistry",
    "StrategyRecord",
    "StrategyStatus",
    "PromotionDecision",
    "PromotionCriteria",
]
