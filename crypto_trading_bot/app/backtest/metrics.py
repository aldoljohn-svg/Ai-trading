"""Performance metrics.

Deliberately blunt about what these numbers do and do not mean:

* Sharpe and Sortino are annualised from the equity curve's own sampling
  period.  On a short sample they are noise; the report includes the sample
  size so a "3.2 Sharpe" from 40 trades is visibly untrustworthy.
* Risk of ruin uses the observed win rate and payoff, which assumes the future
  resembles the sample.  It is a sanity check, not a guarantee.
* Max drawdown is measured on the equity curve including open positions, not
  only on closed trades - the latter flatters every strategy.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

#: Periods per year for annualisation, keyed by the equity-curve step.
_SECONDS_PER_YEAR = 365 * 24 * 3600


@dataclass(slots=True)
class PerformanceMetrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    total_pnl: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0            # currency per trade
    expectancy_r: float = 0.0          # R per trade
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0     # in equity-curve steps
    risk_of_ruin: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_consecutive_losses: int = 0
    average_r: float = 0.0
    total_fees: float = 0.0
    total_funding: float = 0.0
    exposure: float = 0.0              # fraction of time with a position open
    starting_equity: float = 0.0
    ending_equity: float = 0.0
    sample_warning: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "total_return": round(self.total_return, 6),
            "total_pnl": round(self.total_pnl, 4),
            "average_win": round(self.average_win, 4),
            "average_loss": round(self.average_loss, 4),
            "profit_factor": round(self.profit_factor, 4),
            "expectancy": round(self.expectancy, 4),
            "expectancy_r": round(self.expectancy_r, 4),
            "sharpe": round(self.sharpe, 4),
            "sortino": round(self.sortino, 4),
            "calmar": round(self.calmar, 4),
            "max_drawdown": round(self.max_drawdown, 6),
            "max_drawdown_duration": self.max_drawdown_duration,
            "risk_of_ruin": round(self.risk_of_ruin, 6),
            "largest_win": round(self.largest_win, 4),
            "largest_loss": round(self.largest_loss, 4),
            "max_consecutive_losses": self.max_consecutive_losses,
            "average_r": round(self.average_r, 4),
            "total_fees": round(self.total_fees, 4),
            "total_funding": round(self.total_funding, 4),
            "exposure": round(self.exposure, 4),
            "starting_equity": round(self.starting_equity, 4),
            "ending_equity": round(self.ending_equity, 4),
            "sample_warning": self.sample_warning,
        }

    def summary(self) -> str:
        lines = [
            f"Trades: {self.trades}  (W {self.wins} / L {self.losses})",
            f"Win rate: {self.win_rate:.1%}",
            f"Return: {self.total_return:+.2%}  (${self.total_pnl:+,.2f})",
            f"Profit factor: {self.profit_factor:.2f}",
            f"Expectancy: {self.expectancy_r:+.3f}R  (${self.expectancy:+,.2f})",
            f"Sharpe: {self.sharpe:.2f}   Sortino: {self.sortino:.2f}",
            f"Max drawdown: {self.max_drawdown:.2%}",
            f"Risk of ruin: {self.risk_of_ruin:.2%}",
        ]
        if self.sample_warning:
            lines.append(f"⚠️ {self.sample_warning}")
        return "\n".join(lines)


def max_drawdown(equity_curve: Sequence[float]) -> tuple[float, int]:
    """Returns ``(max drawdown fraction, longest underwater run in steps)``."""

    if not equity_curve:
        return 0.0, 0
    peak = equity_curve[0]
    worst = 0.0
    underwater = 0
    longest = 0
    for value in equity_curve:
        if value > peak:
            peak = value
            underwater = 0
        else:
            underwater += 1
            longest = max(longest, underwater)
        if peak > 0:
            drop = (peak - value) / peak
            worst = max(worst, drop)
    return worst, longest


def sharpe_ratio(returns: Sequence[float], periods_per_year: float) -> float:
    if len(returns) < 3:
        return 0.0
    mean = statistics.fmean(returns)
    stdev = statistics.pstdev(returns)
    if stdev <= 0:
        return 0.0
    return (mean / stdev) * math.sqrt(periods_per_year)


def sortino_ratio(returns: Sequence[float], periods_per_year: float) -> float:
    """Like Sharpe but only penalises downside deviation."""

    if len(returns) < 3:
        return 0.0
    mean = statistics.fmean(returns)
    if statistics.pstdev(returns) <= 0:
        # A perfectly constant series has no risk to measure, exactly as in
        # ``sharpe_ratio``.  Reporting a huge ratio here would be meaningless.
        return 0.0
    downside = [r for r in returns if r < 0]
    if not downside:
        # No losing period at all: the ratio is mathematically infinite, which
        # would be misread as skill.  Report a capped sentinel instead.
        return 0.0 if mean <= 0 else 99.0
    deviation = math.sqrt(sum(r * r for r in downside) / len(returns))
    if deviation <= 0:
        return 0.0
    return (mean / deviation) * math.sqrt(periods_per_year)


def risk_of_ruin(
    win_rate: float, payoff_ratio: float, risk_per_trade: float, ruin_fraction: float = 0.5
) -> float:
    """Probability of losing ``ruin_fraction`` of capital, given the edge.

    Uses the standard gambler's-ruin approximation for a fixed-fractional
    bettor.  It assumes independent trades with a stable edge - both optimistic
    assumptions, so treat the output as a floor, not a forecast.
    """

    if win_rate <= 0 or risk_per_trade <= 0:
        return 1.0
    if win_rate >= 1:
        return 0.0
    loss_rate = 1.0 - win_rate
    # Edge per unit risked.
    edge = win_rate * payoff_ratio - loss_rate
    if edge <= 0:
        return 1.0
    # Units of risk available before ruin.
    units = ruin_fraction / risk_per_trade
    # a = probability ratio per unit for an asymmetric bet.
    ratio = loss_rate / (win_rate * payoff_ratio)
    if ratio >= 1:
        return 1.0
    return min(1.0, max(0.0, ratio ** units))


def compute_metrics(
    trades: Sequence[dict[str, Any]],
    equity_curve: Sequence[float],
    starting_equity: float,
    period_seconds: float = 3600.0,
    risk_per_trade: float = 0.005,
    bars_with_position: int = 0,
    total_bars: int = 0,
) -> PerformanceMetrics:
    """Build the full metric set from closed trades and an equity curve."""

    metrics = PerformanceMetrics(starting_equity=starting_equity)
    metrics.trades = len(trades)
    if equity_curve:
        metrics.ending_equity = equity_curve[-1]
    else:
        metrics.ending_equity = starting_equity

    pnls = [float(t.get("pnl", 0.0)) for t in trades]
    r_multiples = [float(t.get("r_multiple", 0.0)) for t in trades]
    metrics.total_pnl = sum(pnls)
    metrics.total_fees = sum(float(t.get("fees", 0.0)) for t in trades)
    metrics.total_funding = sum(float(t.get("funding", 0.0)) for t in trades)
    metrics.total_return = (
        (metrics.ending_equity - starting_equity) / starting_equity
        if starting_equity > 0
        else 0.0
    )

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    metrics.wins = len(wins)
    metrics.losses = len(losses)
    metrics.win_rate = len(wins) / len(pnls) if pnls else 0.0
    metrics.average_win = statistics.fmean(wins) if wins else 0.0
    metrics.average_loss = statistics.fmean(losses) if losses else 0.0
    metrics.largest_win = max(wins) if wins else 0.0
    metrics.largest_loss = min(losses) if losses else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0:
        metrics.profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        metrics.profit_factor = float("inf")
    else:
        metrics.profit_factor = 0.0

    metrics.expectancy = statistics.fmean(pnls) if pnls else 0.0
    metrics.average_r = statistics.fmean(r_multiples) if r_multiples else 0.0
    metrics.expectancy_r = metrics.average_r

    streak = 0
    for pnl in pnls:
        if pnl < 0:
            streak += 1
            metrics.max_consecutive_losses = max(metrics.max_consecutive_losses, streak)
        else:
            streak = 0

    returns: list[float] = []
    for previous, current in zip(equity_curve, equity_curve[1:]):
        if previous > 0:
            returns.append((current - previous) / previous)

    periods_per_year = _SECONDS_PER_YEAR / max(period_seconds, 1.0)
    metrics.sharpe = sharpe_ratio(returns, periods_per_year)
    metrics.sortino = sortino_ratio(returns, periods_per_year)

    metrics.max_drawdown, metrics.max_drawdown_duration = max_drawdown(equity_curve)
    if metrics.max_drawdown > 0:
        years = (len(equity_curve) * period_seconds) / _SECONDS_PER_YEAR
        if years > 0:
            annualised = (
                (metrics.ending_equity / starting_equity) ** (1 / years) - 1
                if starting_equity > 0 and metrics.ending_equity > 0
                else 0.0
            )
            metrics.calmar = annualised / metrics.max_drawdown

    payoff = (
        abs(metrics.average_win / metrics.average_loss)
        if metrics.average_loss != 0
        else (2.0 if metrics.average_win > 0 else 0.0)
    )
    metrics.risk_of_ruin = risk_of_ruin(metrics.win_rate, payoff, risk_per_trade)

    if total_bars > 0:
        metrics.exposure = bars_with_position / total_bars

    warnings: list[str] = []
    if metrics.trades < 30:
        warnings.append(
            f"only {metrics.trades} trades - these statistics are not significant"
        )
    if len(returns) < 100:
        warnings.append("short equity curve; Sharpe/Sortino are unreliable")
    if metrics.profit_factor == float("inf"):
        warnings.append("no losing trades in the sample - almost certainly overfit")
    metrics.sample_warning = "; ".join(warnings)

    return metrics


__all__ = [
    "PerformanceMetrics",
    "compute_metrics",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "risk_of_ruin",
]
