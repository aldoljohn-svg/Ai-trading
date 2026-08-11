"""Portfolio stress testing and Monte Carlo risk simulation.

Two distinct tools with different jobs:

* **Stress tests** are deterministic "what if" scenarios - BTC −20%, a dollar
  spike, a liquidity drain - applied to the *current* book. They answer: would
  we survive this?
* **Monte Carlo** resamples the *historical trade distribution* to estimate the
  spread of outcomes: drawdown distribution, loss streaks, risk of ruin.

Monte Carlo here is emphatically **not** a price predictor. It is bootstrap
resampling of realised R multiples, which answers "given this edge and this bet
size, how bad can a normal run of luck get?".
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from app.domain import Side


# --------------------------------------------------------------------------
# Stress testing
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    #: Move applied to BTC; other assets move by ``beta x this``.
    btc_move: float = 0.0
    #: Correlation assumed *during* the stress - crises push this toward 1.
    stress_correlation: float = 0.9
    #: Multiplier applied to spreads and slippage.
    liquidity_multiplier: float = 1.0
    description: str = ""


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("btc_-10", -0.10, 0.85, 1.5, "BTC falls 10%"),
    Scenario("btc_-20", -0.20, 0.92, 2.5, "BTC falls 20%"),
    Scenario("btc_-30", -0.30, 0.96, 4.0, "BTC falls 30% - cascade conditions"),
    Scenario("btc_+20", 0.20, 0.90, 2.0, "BTC rallies 20% - short squeeze"),
    Scenario("risk_off", -0.15, 0.95, 3.0, "broad risk-off across all assets"),
    Scenario("liquidity_drain", -0.05, 0.90, 6.0, "liquidity collapse, modest price move"),
)


@dataclass(slots=True)
class ScenarioResult:
    scenario: str
    description: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    equity_after: float = 0.0
    margin_after: float = 0.0
    positions_liquidated: list[str] = field(default_factory=list)
    stops_breached: list[str] = field(default_factory=list)
    survivable: bool = True
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "description": self.description,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "equity_after": round(self.equity_after, 2),
            "margin_after": round(self.margin_after, 2),
            "positions_liquidated": self.positions_liquidated,
            "stops_breached": self.stops_breached,
            "survivable": self.survivable,
            "notes": self.notes,
        }


@dataclass(slots=True)
class StressReport:
    results: list[ScenarioResult] = field(default_factory=list)
    worst_case_pct: float = 0.0
    worst_scenario: str = ""
    any_liquidation: bool = False

    @property
    def survivable(self) -> bool:
        return all(r.survivable for r in self.results)

    def as_dict(self) -> dict[str, Any]:
        return {
            "survivable": self.survivable,
            "worst_case_pct": round(self.worst_case_pct, 4),
            "worst_scenario": self.worst_scenario,
            "any_liquidation": self.any_liquidation,
            "results": [r.as_dict() for r in self.results],
        }

    def summary(self) -> str:
        if not self.results:
            return "no positions to stress test"
        lines = [
            f"Stress test: worst case {self.worst_case_pct:+.2%} "
            f"({self.worst_scenario})"
        ]
        for result in sorted(self.results, key=lambda r: r.pnl_pct):
            icon = "✅" if result.survivable else "🔴"
            lines.append(
                f"  {icon} {result.description}: {result.pnl_pct:+.2%}"
                + (
                    f" - liquidations: {', '.join(result.positions_liquidated)}"
                    if result.positions_liquidated
                    else ""
                )
            )
        return "\n".join(lines)


def stress_test(
    positions: Sequence[Any],
    equity: float,
    prices: Mapping[str, float],
    betas: Mapping[str, float] | None = None,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
    ruin_threshold: float = 0.5,
) -> StressReport:
    """Apply each scenario to the current book.

    ``positions`` are :class:`~app.portfolio.portfolio_manager.ManagedPosition`
    objects. ``betas`` maps symbol to sensitivity versus BTC; anything missing
    defaults to 1.2, since most alts move more than BTC, not less.
    """

    report = StressReport()
    if equity <= 0:
        return report

    betas = dict(betas or {})

    for scenario in scenarios:
        result = ScenarioResult(scenario=scenario.name, description=scenario.description)
        total_pnl = 0.0
        margin_used = 0.0

        for position in positions:
            symbol = position.symbol
            price = prices.get(symbol, position.entry_price)
            if price <= 0:
                continue

            beta = betas.get(symbol, 1.0 if symbol.startswith("BTC") else 1.2)
            # Under stress, idiosyncratic behaviour disappears and everything
            # tracks the market factor.
            move = scenario.btc_move * beta * scenario.stress_correlation
            stressed_price = price * (1 + move)

            base_quantity = position.quantity * position.contract_size
            pnl = (stressed_price - position.entry_price) * position.side.sign * base_quantity

            # Stops help, but under stress they slip.
            stop = getattr(position, "stop_loss", 0.0)
            if stop > 0:
                breached = (
                    stressed_price <= stop
                    if position.side is Side.LONG
                    else stressed_price >= stop
                )
                if breached:
                    result.stops_breached.append(symbol)
                    slippage = 0.002 * scenario.liquidity_multiplier
                    fill = stop * (
                        1 - slippage if position.side is Side.LONG else 1 + slippage
                    )
                    pnl = (fill - position.entry_price) * position.side.sign * base_quantity

            total_pnl += pnl
            notional = base_quantity * price
            leverage = max(getattr(position, "leverage", 1.0), 1.0)
            margin_used += notional / leverage

            # Liquidation check against the position's own margin.
            position_margin = notional / leverage
            if position_margin > 0 and pnl < -position_margin * 0.85:
                result.positions_liquidated.append(symbol)

        result.pnl = total_pnl
        result.pnl_pct = total_pnl / equity
        result.equity_after = equity + total_pnl
        result.margin_after = margin_used
        result.survivable = (
            result.equity_after > equity * ruin_threshold
            and not result.positions_liquidated
        )

        if result.positions_liquidated:
            result.notes.append(
                f"{len(result.positions_liquidated)} position(s) would be liquidated"
            )
            report.any_liquidation = True
        if result.margin_after > result.equity_after:
            result.notes.append("margin requirement would exceed remaining equity")
            result.survivable = False

        report.results.append(result)

    if report.results:
        worst = min(report.results, key=lambda r: r.pnl_pct)
        report.worst_case_pct = worst.pnl_pct
        report.worst_scenario = worst.description
    return report


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------


@dataclass(slots=True)
class MonteCarloResult:
    simulations: int = 0
    trades_per_run: int = 0
    median_return: float = 0.0
    mean_return: float = 0.0
    p5_return: float = 0.0
    p95_return: float = 0.0
    median_max_drawdown: float = 0.0
    p95_max_drawdown: float = 0.0
    worst_drawdown: float = 0.0
    risk_of_ruin: float = 0.0
    median_longest_losing_streak: int = 0
    p95_longest_losing_streak: int = 0
    probability_of_loss: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "simulations": self.simulations,
            "trades_per_run": self.trades_per_run,
            "median_return": round(self.median_return, 4),
            "mean_return": round(self.mean_return, 4),
            "p5_return": round(self.p5_return, 4),
            "p95_return": round(self.p95_return, 4),
            "median_max_drawdown": round(self.median_max_drawdown, 4),
            "p95_max_drawdown": round(self.p95_max_drawdown, 4),
            "worst_drawdown": round(self.worst_drawdown, 4),
            "risk_of_ruin": round(self.risk_of_ruin, 5),
            "median_longest_losing_streak": self.median_longest_losing_streak,
            "p95_longest_losing_streak": self.p95_longest_losing_streak,
            "probability_of_loss": round(self.probability_of_loss, 4),
            "notes": self.notes,
        }

    def summary(self) -> str:
        lines = [
            f"Monte Carlo: {self.simulations} runs of {self.trades_per_run} trades",
            f"  return    median {self.median_return:+.1%}  "
            f"5th {self.p5_return:+.1%}  95th {self.p95_return:+.1%}",
            f"  drawdown  median {self.median_max_drawdown:.1%}  "
            f"95th {self.p95_max_drawdown:.1%}  worst {self.worst_drawdown:.1%}",
            f"  longest losing streak: median {self.median_longest_losing_streak}, "
            f"95th {self.p95_longest_losing_streak}",
            f"  probability of ending down: {self.probability_of_loss:.1%}",
            f"  risk of ruin: {self.risk_of_ruin:.2%}",
        ]
        lines.extend(f"  ⚠️ {note}" for note in self.notes)
        return "\n".join(lines)


def monte_carlo(
    r_multiples: Sequence[float],
    risk_per_trade: float = 0.005,
    simulations: int = 2000,
    trades_per_run: int | None = None,
    ruin_threshold: float = 0.5,
    seed: int = 42,
) -> MonteCarloResult:
    """Bootstrap the historical R distribution.

    Resamples *with replacement*, which preserves the distribution's shape while
    destroying its ordering - the point being to ask how bad an unlucky
    *sequence* of the same trades could be.
    """

    result = MonteCarloResult()
    cleaned = [float(r) for r in r_multiples if r == r]
    if len(cleaned) < 10:
        result.notes.append(
            f"only {len(cleaned)} historical trades - the resampled distribution "
            "cannot be trusted"
        )
        return result

    trades_per_run = trades_per_run or max(len(cleaned), 100)
    rng = random.Random(seed)

    returns: list[float] = []
    drawdowns: list[float] = []
    streaks: list[int] = []
    ruins = 0

    for _ in range(simulations):
        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        streak = 0
        longest_streak = 0
        ruined = False

        for _trade in range(trades_per_run):
            r = cleaned[rng.randrange(len(cleaned))]
            # Fixed-fractional: risk a constant share of *current* equity.
            equity *= 1.0 + r * risk_per_trade
            if equity <= 0:
                equity = 0.0
                ruined = True
                break
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
            if equity <= ruin_threshold:
                ruined = True
                break
            if r <= 0:
                streak += 1
                longest_streak = max(longest_streak, streak)
            else:
                streak = 0

        returns.append(equity - 1.0)
        drawdowns.append(max_drawdown)
        streaks.append(longest_streak)
        if ruined:
            ruins += 1

    def percentile(values: Sequence[float], q: float) -> float:
        ordered = sorted(values)
        index = min(int(q * (len(ordered) - 1)), len(ordered) - 1)
        return ordered[index]

    result.simulations = simulations
    result.trades_per_run = trades_per_run
    result.median_return = statistics.median(returns)
    result.mean_return = statistics.fmean(returns)
    result.p5_return = percentile(returns, 0.05)
    result.p95_return = percentile(returns, 0.95)
    result.median_max_drawdown = statistics.median(drawdowns)
    result.p95_max_drawdown = percentile(drawdowns, 0.95)
    result.worst_drawdown = max(drawdowns)
    result.risk_of_ruin = ruins / simulations
    result.median_longest_losing_streak = int(statistics.median(streaks))
    result.p95_longest_losing_streak = int(percentile([float(s) for s in streaks], 0.95))
    result.probability_of_loss = sum(1 for r in returns if r < 0) / len(returns)

    if len(cleaned) < 50:
        result.notes.append(
            f"based on only {len(cleaned)} historical trades - wide error bars"
        )
    if result.risk_of_ruin > 0.01:
        result.notes.append(
            f"risk of ruin {result.risk_of_ruin:.1%} is material at "
            f"{risk_per_trade:.2%} per trade - consider reducing size"
        )
    if result.p95_max_drawdown > 0.3:
        result.notes.append(
            f"1 run in 20 sees a {result.p95_max_drawdown:.0%} drawdown - "
            "make sure that is survivable psychologically and financially"
        )
    return result


__all__ = [
    "Scenario",
    "ScenarioResult",
    "StressReport",
    "stress_test",
    "DEFAULT_SCENARIOS",
    "MonteCarloResult",
    "monte_carlo",
]
