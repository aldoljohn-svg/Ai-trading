"""Shadow mode.

A shadow strategy sees exactly what production sees and records exactly what it
*would* have done - entry, stop, targets, size - but can never place an order.
The book is then marked to market against real subsequent prices, so the
comparison against the champion uses the same price path production experienced.

The guarantee is structural: :class:`ShadowBook` holds no broker, no exchange
client and no credentials. There is no code path from here to an order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from app.domain import Side


@dataclass(slots=True)
class ShadowTrade:
    strategy: str
    version: str
    symbol: str
    side: Side
    entry: float
    stop: float
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    confidence: float = 0.0
    regime: str = "UNKNOWN"
    opened_at: int = field(default_factory=lambda: int(time.time()))
    closed_at: int = 0
    exit_price: float = 0.0
    r_multiple: float = 0.0
    mae_r: float = 0.0            # worst excursion against, in R
    mfe_r: float = 0.0            # best excursion in favour, in R
    status: str = "open"
    exit_reason: str = ""
    reasons: list[str] = field(default_factory=list)

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)

    def r_at(self, price: float) -> float:
        risk = self.risk_per_unit
        if risk <= 0:
            return 0.0
        return (price - self.entry) * self.side.sign / risk

    def update(self, high: float, low: float, close: float) -> bool:
        """Mark to market against one bar. Returns ``True`` when it closes.

        Uses the same pessimistic convention as the backtester: if a bar spans
        both the stop and a target, the stop is assumed to have been hit first.
        """

        if self.status != "open":
            return True

        favourable = high if self.side is Side.LONG else low
        adverse = low if self.side is Side.LONG else high
        self.mfe_r = max(self.mfe_r, self.r_at(favourable))
        self.mae_r = min(self.mae_r, self.r_at(adverse))

        stop_hit = (
            low <= self.stop if self.side is Side.LONG else high >= self.stop
        )
        if stop_hit:
            self.close(self.stop, "stop loss")
            return True

        for level, target in ((3, self.tp3), (2, self.tp2), (1, self.tp1)):
            if target <= 0:
                continue
            reached = (
                high >= target if self.side is Side.LONG else low <= target
            )
            if reached and level == 3:
                self.close(target, "TP3")
                return True
        return False

    def close(self, price: float, reason: str) -> None:
        self.exit_price = price
        self.r_multiple = self.r_at(price)
        self.closed_at = int(time.time())
        self.status = "closed"
        self.exit_reason = reason

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "version": self.version,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry": self.entry,
            "stop": self.stop,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "confidence": round(self.confidence, 4),
            "regime": self.regime,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "exit_price": self.exit_price,
            "r_multiple": round(self.r_multiple, 4),
            "mae_r": round(self.mae_r, 4),
            "mfe_r": round(self.mfe_r, 4),
            "status": self.status,
            "exit_reason": self.exit_reason,
        }


class ShadowBook:
    """Records hypothetical trades. Holds no broker - it structurally cannot trade."""

    def __init__(self, repositories: Any = None, max_open_per_strategy: int = 10) -> None:
        self.repositories = repositories
        self.max_open_per_strategy = max_open_per_strategy
        self.trades: list[ShadowTrade] = []

    # -- recording --------------------------------------------------------

    def open(
        self,
        strategy: str,
        symbol: str,
        side: Side,
        entry: float,
        stop: float,
        version: str = "1",
        **kwargs: Any,
    ) -> ShadowTrade | None:
        if entry <= 0 or stop <= 0 or entry == stop:
            return None
        if self.open_count(strategy) >= self.max_open_per_strategy:
            return None
        if any(
            t.status == "open" and t.strategy == strategy and t.symbol == symbol
            for t in self.trades
        ):
            return None

        trade = ShadowTrade(
            strategy=strategy,
            version=version,
            symbol=symbol,
            side=side,
            entry=entry,
            stop=stop,
            **kwargs,
        )
        self.trades.append(trade)
        self._persist(trade)
        return trade

    def open_count(self, strategy: str | None = None) -> int:
        return sum(
            1
            for t in self.trades
            if t.status == "open" and (strategy is None or t.strategy == strategy)
        )

    def open_trades(self, symbol: str | None = None) -> list[ShadowTrade]:
        return [
            t
            for t in self.trades
            if t.status == "open" and (symbol is None or t.symbol == symbol)
        ]

    def mark(self, symbol: str, high: float, low: float, close: float) -> list[ShadowTrade]:
        """Mark every open shadow trade on ``symbol`` against one bar."""

        closed: list[ShadowTrade] = []
        for trade in self.open_trades(symbol):
            if trade.update(high, low, close):
                closed.append(trade)
                self._persist(trade)
        return closed

    def close_all(self, symbol: str, price: float, reason: str = "forced") -> int:
        count = 0
        for trade in self.open_trades(symbol):
            trade.close(price, reason)
            self._persist(trade)
            count += 1
        return count

    # -- statistics ---------------------------------------------------------

    def performance(self, strategy: str | None = None) -> dict[str, Any]:
        closed = [
            t
            for t in self.trades
            if t.status == "closed" and (strategy is None or t.strategy == strategy)
        ]
        if not closed:
            return {"trades": 0, "note": "no closed shadow trades yet"}

        r_values = [t.r_multiple for t in closed]
        wins = [r for r in r_values if r > 0]
        losses = [r for r in r_values if r <= 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        return {
            "strategy": strategy or "*",
            "trades": len(closed),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(closed), 4),
            "expectancy_r": round(sum(r_values) / len(r_values), 4),
            "profit_factor": (
                round(gross_profit / gross_loss, 3) if gross_loss > 0 else None
            ),
            "cumulative_r": round(sum(r_values), 3),
            "avg_mae_r": round(sum(t.mae_r for t in closed) / len(closed), 4),
            "avg_mfe_r": round(sum(t.mfe_r for t in closed) / len(closed), 4),
            "open": self.open_count(strategy),
        }

    def compare(self, challenger: str, champion: str) -> dict[str, Any]:
        a = self.performance(challenger)
        b = self.performance(champion)
        if not a.get("trades") or not b.get("trades"):
            return {"comparable": False, "challenger": a, "champion": b}
        return {
            "comparable": True,
            "challenger": a,
            "champion": b,
            "expectancy_delta": round(
                a["expectancy_r"] - b["expectancy_r"], 4
            ),
            "win_rate_delta": round(a["win_rate"] - b["win_rate"], 4),
        }

    def strategies(self) -> list[str]:
        return sorted({t.strategy for t in self.trades})

    def _persist(self, trade: ShadowTrade) -> None:
        if self.repositories is None:
            return
        try:
            self.repositories.shadow.save(trade.as_dict())
        except Exception:  # noqa: BLE001 - shadow persistence is best effort
            pass


__all__ = ["ShadowBook", "ShadowTrade"]
