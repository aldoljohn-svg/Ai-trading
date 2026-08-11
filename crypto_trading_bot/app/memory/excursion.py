"""MAE / MFE analysis.

For every closed trade:

``MAE`` maximum adverse excursion - how far it went against us before resolving
``MFE`` maximum favourable excursion - the best it ever offered

Both in R, so they are comparable across symbols and account sizes.

What they are used for:

* **Stop placement** - if winners rarely draw down more than 0.6R, a 1.0R stop
  is donating the difference. If losers routinely exceed the stop by a wide
  margin, the stop is inside the noise.
* **Target placement** - if MFE routinely reaches 2.5R but exits average 1.4R,
  the exit is early. If MFE rarely reaches TP3, TP3 is decoration.
* **Exit quality** - realised R against the MFE that was actually available.

Suggestions are *evidence for a human*, never applied automatically: a stop
tuned to fit history is a stop fitted to noise.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from app.domain import Candle, Side


@dataclass(slots=True)
class TradeExcursion:
    symbol: str
    side: str
    r_multiple: float
    mae_r: float
    mfe_r: float
    won: bool
    exit_efficiency: float = 0.0     # realised R / MFE R
    regime: str = "ALL"

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "r_multiple": round(self.r_multiple, 4),
            "mae_r": round(self.mae_r, 4),
            "mfe_r": round(self.mfe_r, 4),
            "won": self.won,
            "exit_efficiency": round(self.exit_efficiency, 4),
            "regime": self.regime,
        }


@dataclass(slots=True)
class ExcursionStats:
    sample: int = 0
    winner_mae_median: float = 0.0
    winner_mae_p90: float = 0.0
    loser_mae_median: float = 0.0
    winner_mfe_median: float = 0.0
    winner_mfe_p90: float = 0.0
    loser_mfe_median: float = 0.0
    exit_efficiency_median: float = 0.0
    stop_too_tight: bool = False
    stop_too_wide: bool = False
    exits_too_early: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        return self.sample >= 30

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample,
            "reliable": self.reliable,
            "winner_mae_median": round(self.winner_mae_median, 4),
            "winner_mae_p90": round(self.winner_mae_p90, 4),
            "loser_mae_median": round(self.loser_mae_median, 4),
            "winner_mfe_median": round(self.winner_mfe_median, 4),
            "winner_mfe_p90": round(self.winner_mfe_p90, 4),
            "loser_mfe_median": round(self.loser_mfe_median, 4),
            "exit_efficiency_median": round(self.exit_efficiency_median, 4),
            "stop_too_tight": self.stop_too_tight,
            "stop_too_wide": self.stop_too_wide,
            "exits_too_early": self.exits_too_early,
            "notes": self.notes,
        }

    def summary(self) -> str:
        if not self.sample:
            return "no excursion data yet"
        lines = [f"MAE/MFE over {self.sample} trades"
                 + ("" if self.reliable else " (too few to be reliable)")]
        lines.append(
            f"  winners: MAE median {self.winner_mae_median:.2f}R "
            f"(p90 {self.winner_mae_p90:.2f}R), MFE median {self.winner_mfe_median:.2f}R"
        )
        lines.append(
            f"  losers:  MAE median {self.loser_mae_median:.2f}R, "
            f"MFE median {self.loser_mfe_median:.2f}R"
        )
        lines.append(f"  exit efficiency median {self.exit_efficiency_median:.0%}")
        for note in self.notes:
            lines.append(f"  · {note}")
        return "\n".join(lines)


def compute_excursion(
    entry: float,
    stop: float,
    side: Side,
    candles: Sequence[Candle],
    exit_price: float,
) -> tuple[float, float, float]:
    """Returns ``(mae_r, mfe_r, realised_r)`` from the bars the trade was open."""

    risk = abs(entry - stop)
    if risk <= 0 or not candles:
        return 0.0, 0.0, 0.0

    sign = side.sign
    mae_r = 0.0
    mfe_r = 0.0
    for candle in candles:
        favourable = candle.high if side is Side.LONG else candle.low
        adverse = candle.low if side is Side.LONG else candle.high
        mfe_r = max(mfe_r, (favourable - entry) * sign / risk)
        mae_r = min(mae_r, (adverse - entry) * sign / risk)

    realised_r = (exit_price - entry) * sign / risk
    return mae_r, mfe_r, realised_r


def analyse_excursions(
    excursions: Sequence[TradeExcursion],
    stop_r: float = 1.0,
    tp2_r: float = 2.0,
) -> ExcursionStats:
    """Turn a set of excursions into actionable statistics."""

    stats = ExcursionStats(sample=len(excursions))
    if not excursions:
        return stats

    winners = [e for e in excursions if e.won]
    losers = [e for e in excursions if not e.won]

    def median(values: Sequence[float]) -> float:
        return statistics.median(values) if values else 0.0

    def percentile(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(int(q * (len(ordered) - 1)), len(ordered) - 1)
        return ordered[index]

    # MAE is negative; use magnitude for readability.
    stats.winner_mae_median = abs(median([e.mae_r for e in winners]))
    stats.winner_mae_p90 = abs(percentile([e.mae_r for e in winners], 0.10))
    stats.loser_mae_median = abs(median([e.mae_r for e in losers]))
    stats.winner_mfe_median = median([e.mfe_r for e in winners])
    stats.winner_mfe_p90 = percentile([e.mfe_r for e in winners], 0.90)
    stats.loser_mfe_median = median([e.mfe_r for e in losers])
    stats.exit_efficiency_median = median(
        [e.exit_efficiency for e in winners if e.exit_efficiency > 0]
    )

    if not stats.reliable:
        stats.notes.append(
            f"only {stats.sample} trades - treat these figures as indicative"
        )
        return stats

    # --- stop diagnostics -------------------------------------------------
    if stats.winner_mae_p90 < stop_r * 0.55:
        stats.stop_too_wide = True
        stats.notes.append(
            f"90% of winners never drew down beyond {stats.winner_mae_p90:.2f}R - "
            f"a {stop_r:.2f}R stop may be wider than necessary"
        )
    if losers and stats.loser_mfe_median >= 0.75:
        stats.stop_too_tight = True
        stats.notes.append(
            f"losing trades reached {stats.loser_mfe_median:.2f}R in profit before "
            "failing - the stop or the management may be too tight"
        )

    # --- target diagnostics -------------------------------------------------
    if stats.exit_efficiency_median and stats.exit_efficiency_median < 0.55:
        stats.exits_too_early = True
        stats.notes.append(
            f"winners capture only {stats.exit_efficiency_median:.0%} of the move "
            "that was available - exits are early"
        )
    if winners and stats.winner_mfe_p90 < tp2_r:
        stats.notes.append(
            f"90% of winners peaked below {stats.winner_mfe_p90:.2f}R - "
            f"TP2 at {tp2_r:.2f}R is rarely reached"
        )

    return stats


def suggest_levels(
    stats: ExcursionStats, current_stop_r: float = 1.0, current_tp2_r: float = 2.0
) -> dict[str, Any]:
    """Evidence-based suggestions for a human to review.

    Deliberately returns suggestions, not settings. Nothing in this system
    applies them automatically.
    """

    suggestions: dict[str, Any] = {
        "applied": False,
        "reliable": stats.reliable,
        "suggestions": [],
    }
    if not stats.reliable:
        suggestions["suggestions"].append(
            "insufficient sample - no changes should be considered yet"
        )
        return suggestions

    if stats.stop_too_wide:
        # Leave clear headroom above the observed p90 rather than fitting it.
        proposed = round(max(stats.winner_mae_p90 * 1.35, 0.4), 2)
        if proposed < current_stop_r:
            suggestions["suggestions"].append(
                f"consider a stop near {proposed:.2f}R (currently {current_stop_r:.2f}R); "
                f"90% of winners stayed within {stats.winner_mae_p90:.2f}R"
            )
    if stats.stop_too_tight:
        suggestions["suggestions"].append(
            "consider widening the stop or loosening break-even: losers reached "
            f"{stats.loser_mfe_median:.2f}R before failing"
        )
    if stats.exits_too_early and stats.winner_mfe_median > current_tp2_r:
        suggestions["suggestions"].append(
            f"winners typically reach {stats.winner_mfe_median:.2f}R; TP2 at "
            f"{current_tp2_r:.2f}R may be leaving money on the table"
        )
    if not suggestions["suggestions"]:
        suggestions["suggestions"].append("current levels look consistent with the data")
    return suggestions


__all__ = [
    "TradeExcursion",
    "ExcursionStats",
    "compute_excursion",
    "analyse_excursions",
    "suggest_levels",
]
