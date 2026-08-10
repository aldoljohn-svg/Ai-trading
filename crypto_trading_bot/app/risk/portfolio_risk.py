"""Portfolio-level risk, with correlation treated as first-class.

Crypto majors and their alts routinely move together with correlations above
0.8.  Three "independent" 0.5% risks on BTC, ETH and SOL longs are not 1.5% of
diversified risk - on a bad day they are one 1.5% bet on the same thing.  This
module makes that explicit:

* correlation is **measured** from recent returns, not assumed;
* positions are grouped into clusters by correlation;
* each cluster consumes a combined risk budget;
* the portfolio risk figure reported everywhere is the *correlation-adjusted*
  one, computed as ``sqrt(w' C w)`` over the open risks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from app.domain import Candle, Side


@dataclass(slots=True)
class OpenRisk:
    """One position's contribution to portfolio risk."""

    symbol: str
    side: Side
    risk_amount: float           # currency at risk if the stop is hit
    notional: float
    leverage: float = 1.0

    @property
    def signed_risk(self) -> float:
        return self.risk_amount * self.side.sign


@dataclass(slots=True)
class PortfolioState:
    equity: float
    available: float
    open_risks: list[OpenRisk] = field(default_factory=list)
    realized_pnl_today: float = 0.0
    unrealized_pnl: float = 0.0
    peak_equity: float = 0.0
    consecutive_losses: int = 0

    @property
    def open_positions(self) -> int:
        return len(self.open_risks)

    @property
    def gross_risk(self) -> float:
        """Naive sum of risks - correct only if nothing is correlated."""

        return sum(r.risk_amount for r in self.open_risks)

    @property
    def gross_risk_pct(self) -> float:
        return self.gross_risk / self.equity if self.equity > 0 else 0.0

    @property
    def gross_notional(self) -> float:
        return sum(r.notional for r in self.open_risks)

    @property
    def gross_leverage(self) -> float:
        return self.gross_notional / self.equity if self.equity > 0 else 0.0

    @property
    def drawdown(self) -> float:
        peak = max(self.peak_equity, self.equity)
        return (peak - self.equity) / peak if peak > 0 else 0.0

    @property
    def daily_loss_pct(self) -> float:
        """Positive number meaning "fraction of equity lost today"."""

        if self.equity <= 0:
            return 0.0
        return max(0.0, -self.realized_pnl_today) / self.equity

    def symbols(self) -> set[str]:
        return {r.symbol for r in self.open_risks}


class CorrelationMatrix:
    """Pairwise return correlation estimated from recent candles."""

    def __init__(self, lookback: int = 120, default: float = 0.6) -> None:
        self.lookback = lookback
        #: Used when a pair has no overlapping history.  Crypto pairs are
        #: correlated far more often than not, so the safe default is high.
        self.default = default
        self._returns: dict[str, list[float]] = {}
        self._cache: dict[tuple[str, str], float] = {}

    def update(self, symbol: str, candles: Sequence[Candle]) -> None:
        window = candles[-(self.lookback + 1) :]
        if len(window) < 20:
            return
        returns = [
            (b.close - a.close) / a.close
            for a, b in zip(window, window[1:])
            if a.close > 0
        ]
        self._returns[symbol.upper()] = returns
        self._cache = {k: v for k, v in self._cache.items() if symbol.upper() not in k}

    def update_many(self, series: Mapping[str, Sequence[Candle]]) -> None:
        for symbol, candles in series.items():
            self.update(symbol, candles)

    def correlation(self, a: str, b: str) -> float:
        a, b = a.upper(), b.upper()
        if a == b:
            return 1.0
        key = (a, b) if a < b else (b, a)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        series_a = self._returns.get(a)
        series_b = self._returns.get(b)
        if not series_a or not series_b:
            return self.default

        n = min(len(series_a), len(series_b))
        if n < 20:
            return self.default
        x = series_a[-n:]
        y = series_b[-n:]
        value = pearson(x, y)
        self._cache[key] = value
        return value

    def known_symbols(self) -> set[str]:
        return set(self._returns)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x, y = x[-n:], y[-n:]
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    covariance = 0.0
    var_x = 0.0
    var_y = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        covariance += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    denominator = math.sqrt(var_x * var_y)
    if denominator <= 0:
        return 0.0
    return max(-1.0, min(1.0, covariance / denominator))


class PortfolioRisk:
    def __init__(
        self,
        max_portfolio_risk: float,
        max_correlated_exposure: float,
        correlation_threshold: float = 0.7,
        max_open_positions: int = 3,
        max_gross_leverage: float = 6.0,
    ) -> None:
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlated_exposure = max_correlated_exposure
        self.correlation_threshold = correlation_threshold
        self.max_open_positions = max_open_positions
        self.max_gross_leverage = max_gross_leverage
        self.matrix = CorrelationMatrix()

    # -- measurement ------------------------------------------------------

    def effective_risk(self, risks: Sequence[OpenRisk]) -> float:
        """Correlation-adjusted portfolio risk: ``sqrt(w' C w)``.

        Same-direction correlated positions add; opposite-direction correlated
        positions partially cancel, which is why the *signed* risk is used.
        """

        if not risks:
            return 0.0
        total = 0.0
        for i, a in enumerate(risks):
            for j, b in enumerate(risks):
                correlation = 1.0 if i == j else self.matrix.correlation(a.symbol, b.symbol)
                total += a.signed_risk * b.signed_risk * correlation
        return math.sqrt(max(total, 0.0))

    def effective_risk_pct(self, state: PortfolioState) -> float:
        if state.equity <= 0:
            return 0.0
        return self.effective_risk(state.open_risks) / state.equity

    def clusters(self, risks: Sequence[OpenRisk]) -> list[list[OpenRisk]]:
        """Group positions that move together into correlation clusters."""

        remaining = list(risks)
        groups: list[list[OpenRisk]] = []
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            rest: list[OpenRisk] = []
            for candidate in remaining:
                correlated = any(
                    self.matrix.correlation(member.symbol, candidate.symbol)
                    >= self.correlation_threshold
                    and member.side is candidate.side
                    for member in group
                )
                if correlated:
                    group.append(candidate)
                else:
                    rest.append(candidate)
            remaining = rest
            groups.append(group)
        return groups

    def cluster_exposure(self, risks: Sequence[OpenRisk], equity: float) -> float:
        """Largest single-cluster risk as a fraction of equity."""

        if equity <= 0 or not risks:
            return 0.0
        return max(
            sum(r.risk_amount for r in group) / equity
            for group in self.clusters(risks)
        )

    # -- admission control ------------------------------------------------

    def can_add(
        self, state: PortfolioState, candidate: OpenRisk
    ) -> tuple[bool, list[str], dict[str, float]]:
        """Would adding ``candidate`` breach any portfolio-level limit?"""

        problems: list[str] = []
        metrics: dict[str, float] = {}

        if candidate.symbol.upper() in {r.symbol.upper() for r in state.open_risks}:
            problems.append(f"already holding a position in {candidate.symbol}")

        if state.open_positions >= self.max_open_positions:
            problems.append(
                f"already at the maximum of {self.max_open_positions} open positions"
            )

        combined = list(state.open_risks) + [candidate]

        effective = self.effective_risk(combined)
        effective_pct = effective / state.equity if state.equity > 0 else 1.0
        metrics["effective_risk_pct"] = round(effective_pct, 6)
        metrics["current_risk_pct"] = round(self.effective_risk_pct(state), 6)
        if effective_pct > self.max_portfolio_risk:
            problems.append(
                f"correlation-adjusted portfolio risk would reach "
                f"{effective_pct:.2%}, above the {self.max_portfolio_risk:.2%} cap"
            )

        cluster_pct = self.cluster_exposure(combined, state.equity)
        metrics["cluster_risk_pct"] = round(cluster_pct, 6)
        if cluster_pct > self.max_correlated_exposure:
            correlated = [
                r.symbol
                for r in state.open_risks
                if self.matrix.correlation(r.symbol, candidate.symbol)
                >= self.correlation_threshold
                and r.side is candidate.side
            ]
            detail = f" (correlated with {', '.join(correlated)})" if correlated else ""
            problems.append(
                f"correlated cluster risk would reach {cluster_pct:.2%}, above the "
                f"{self.max_correlated_exposure:.2%} cap{detail}"
            )

        gross_notional = state.gross_notional + candidate.notional
        gross_leverage = gross_notional / state.equity if state.equity > 0 else 0.0
        metrics["gross_leverage"] = round(gross_leverage, 4)
        if gross_leverage > self.max_gross_leverage:
            problems.append(
                f"gross portfolio leverage would reach {gross_leverage:.2f}x, above "
                f"the {self.max_gross_leverage:.2f}x cap"
            )

        return (not problems), problems, metrics

    def remaining_risk_budget(self, state: PortfolioState) -> float:
        """Currency amount of risk still available under the portfolio cap."""

        used = self.effective_risk(state.open_risks)
        cap = state.equity * self.max_portfolio_risk
        return max(0.0, cap - used)

    def describe(self, state: PortfolioState) -> dict[str, object]:
        groups = self.clusters(state.open_risks)
        return {
            "open_positions": state.open_positions,
            "gross_risk_pct": round(state.gross_risk_pct, 6),
            "effective_risk_pct": round(self.effective_risk_pct(state), 6),
            "cluster_risk_pct": round(
                self.cluster_exposure(state.open_risks, state.equity), 6
            ),
            "gross_leverage": round(state.gross_leverage, 4),
            "drawdown": round(state.drawdown, 6),
            "clusters": [[r.symbol for r in group] for group in groups],
            "remaining_risk_budget": round(self.remaining_risk_budget(state), 2),
        }


__all__ = [
    "PortfolioRisk",
    "PortfolioState",
    "OpenRisk",
    "CorrelationMatrix",
    "pearson",
]
