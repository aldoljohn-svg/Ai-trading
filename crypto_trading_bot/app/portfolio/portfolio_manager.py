"""Portfolio manager.

Owns the in-memory picture of open positions and keeps it in step with the
database.  The *exchange* is always the source of truth for what actually
exists; this class holds the bot's intent (stops, targets, the plan) which the
exchange does not store.  :mod:`app.execution.order_reconciliation` is what
reconciles the two.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from app.domain import Candle, ExchangePosition, Side
from app.logger import get_logger
from app.risk.portfolio_risk import OpenRisk, PortfolioRisk, PortfolioState

log = get_logger(__name__)


@dataclass(slots=True)
class ManagedPosition:
    """An open position plus the plan attached to it."""

    symbol: str
    side: Side
    quantity: float                  # contracts currently held
    entry_price: float
    leverage: float = 1.0
    contract_size: float = 1.0
    stop_loss: float = 0.0
    initial_stop: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    initial_quantity: float = 0.0
    risk_amount: float = 0.0
    opened_at: int = 0
    updated_at: int = 0
    mode: str = "paper"
    state: str = "open"
    tp1_done: bool = False
    tp2_done: bool = False
    breakeven_done: bool = False
    trailing_active: bool = False
    realized_pnl: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    id: int | None = None
    trade_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # -- geometry ---------------------------------------------------------

    @property
    def base_quantity(self) -> float:
        return self.quantity * self.contract_size

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_price - self.initial_stop)

    def notional(self, price: float | None = None) -> float:
        return self.base_quantity * (price if price is not None else self.entry_price)

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.side.sign * self.base_quantity

    def unrealized_pct(self, price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (price - self.entry_price) * self.side.sign / self.entry_price

    def r_multiple(self, price: float) -> float:
        risk = self.risk_per_unit
        if risk <= 0:
            return 0.0
        return (price - self.entry_price) * self.side.sign / risk

    def open_risk(self, price: float | None = None) -> float:
        """Currency still at risk if the *current* stop is hit.

        Once the stop is at or beyond break-even this is zero or negative, and
        the position stops consuming portfolio risk budget - which is the whole
        point of moving stops to break-even.
        """

        if self.stop_loss <= 0:
            return self.risk_amount
        per_unit = (self.entry_price - self.stop_loss) * self.side.sign
        return max(0.0, per_unit * self.base_quantity)

    def to_open_risk(self, price: float | None = None) -> OpenRisk:
        return OpenRisk(
            symbol=self.symbol,
            side=self.side,
            risk_amount=self.open_risk(price),
            notional=self.notional(price),
            leverage=self.leverage,
        )

    def is_stopped(self, price: float) -> bool:
        if self.stop_loss <= 0:
            return False
        return (
            price <= self.stop_loss if self.side is Side.LONG else price >= self.stop_loss
        )

    def target_hit(self, price: float, target: float) -> bool:
        if target <= 0:
            return False
        return price >= target if self.side is Side.LONG else price <= target

    # -- persistence ------------------------------------------------------

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "leverage": self.leverage,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "initial_stop": self.initial_stop,
            "initial_quantity": self.initial_quantity,
            "risk_amount": self.risk_amount,
            "state": self.state,
            "breakeven_done": self.breakeven_done,
            "trailing_active": self.trailing_active,
            "tp1_done": self.tp1_done,
            "tp2_done": self.tp2_done,
            "opened_at": self.opened_at,
            "mode": self.mode,
            "meta": {
                **self.meta,
                "contract_size": self.contract_size,
                "realized_pnl": self.realized_pnl,
                "fees": self.fees,
                "funding": self.funding,
            },
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ManagedPosition":
        meta = dict(row.get("meta") or {})
        return cls(
            id=int(row["id"]) if row.get("id") is not None else None,
            trade_id=int(row["trade_id"]) if row.get("trade_id") is not None else None,
            symbol=str(row["symbol"]),
            side=Side(str(row["side"])),
            quantity=float(row["quantity"]),
            entry_price=float(row["entry_price"]),
            leverage=float(row.get("leverage") or 1.0),
            contract_size=float(meta.get("contract_size", 1.0)),
            stop_loss=float(row.get("stop_loss") or 0.0),
            initial_stop=float(row.get("initial_stop") or row.get("stop_loss") or 0.0),
            tp1=float(row.get("tp1") or 0.0),
            tp2=float(row.get("tp2") or 0.0),
            tp3=float(row.get("tp3") or 0.0),
            initial_quantity=float(row.get("initial_quantity") or row.get("quantity") or 0.0),
            risk_amount=float(row.get("risk_amount") or 0.0),
            opened_at=int(row.get("opened_at") or 0),
            updated_at=int(row.get("updated_at") or 0),
            mode=str(row.get("mode") or "paper"),
            state=str(row.get("state") or "open"),
            tp1_done=bool(row.get("tp1_done")),
            tp2_done=bool(row.get("tp2_done")),
            breakeven_done=bool(row.get("breakeven_done")),
            trailing_active=bool(row.get("trailing_active")),
            realized_pnl=float(meta.get("realized_pnl", 0.0)),
            fees=float(meta.get("fees", 0.0)),
            funding=float(meta.get("funding", 0.0)),
            meta=meta,
        )

    def summary(self, price: float | None = None) -> dict[str, Any]:
        price = price if price is not None else self.entry_price
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry": self.entry_price,
            "current": price,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "leverage": self.leverage,
            "notional": round(self.notional(price), 4),
            "unrealized_pnl": round(self.unrealized_pnl(price), 4),
            "unrealized_pct": round(self.unrealized_pct(price), 6),
            "r_multiple": round(self.r_multiple(price), 3),
            "open_risk": round(self.open_risk(price), 4),
            "risk_amount": round(self.risk_amount, 4),
            "breakeven": self.breakeven_done,
            "trailing": self.trailing_active,
            "tp1_done": self.tp1_done,
            "tp2_done": self.tp2_done,
            "opened_at": self.opened_at,
            "confidence": self.meta.get("confidence"),
            "regime": self.meta.get("regime"),
        }


class PortfolioManager:
    def __init__(
        self,
        repositories: Any = None,
        portfolio_risk: PortfolioRisk | None = None,
        mode: str = "paper",
        starting_equity: float = 0.0,
    ) -> None:
        self.repositories = repositories
        self.portfolio_risk = portfolio_risk
        self.mode = mode
        self.positions: dict[str, ManagedPosition] = {}
        self.equity = starting_equity
        self.balance = starting_equity
        self.available = starting_equity
        self.used_margin = 0.0
        self.peak_equity = starting_equity
        self.realized_pnl_today = 0.0
        self.consecutive_losses = 0
        self._day = _utc_day(time.time())
        self._prices: dict[str, float] = {}

    # -- lifecycle --------------------------------------------------------

    def load(self) -> int:
        """Rebuild in-memory positions from the database (crash recovery)."""

        if self.repositories is None:
            return 0
        rows = self.repositories.positions.open_positions(mode=self.mode)
        self.positions = {}
        for row in rows:
            position = ManagedPosition.from_row(row)
            self.positions[position.symbol] = position
        log.info("loaded %d open positions from the database", len(self.positions))
        return len(self.positions)

    def add(self, position: ManagedPosition) -> ManagedPosition:
        position.mode = self.mode
        if position.initial_quantity <= 0:
            position.initial_quantity = position.quantity
        if position.initial_stop <= 0:
            position.initial_stop = position.stop_loss
        position.opened_at = position.opened_at or int(time.time())
        position.updated_at = int(time.time())
        self.positions[position.symbol] = position
        self.persist(position)
        return position

    def remove(self, symbol: str) -> ManagedPosition | None:
        position = self.positions.pop(symbol, None)
        if position is not None and self.repositories is not None and position.id:
            try:
                self.repositories.positions.close(position.id)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not mark position %s closed: %s", symbol, exc)
        return position

    def get(self, symbol: str) -> ManagedPosition | None:
        return self.positions.get(symbol)

    def all(self) -> list[ManagedPosition]:
        return list(self.positions.values())

    def persist(self, position: ManagedPosition) -> None:
        if self.repositories is None:
            return
        try:
            position.updated_at = int(time.time())
            position_id = self.repositories.positions.save(position.to_row())
            if position.id is None:
                position.id = position_id
        except Exception as exc:  # noqa: BLE001 - persistence must not stop trading
            log.warning("could not persist position %s: %s", position.symbol, exc)

    def persist_all(self) -> None:
        for position in self.positions.values():
            self.persist(position)

    # -- prices -----------------------------------------------------------

    def set_price(self, symbol: str, price: float) -> None:
        if price > 0:
            self._prices[symbol.upper()] = price

    def price(self, symbol: str) -> float:
        position = self.positions.get(symbol)
        return self._prices.get(symbol.upper(), position.entry_price if position else 0.0)

    def mark_to_market(self, prices: Mapping[str, float] | None = None) -> float:
        """Refresh equity from the latest prices; returns unrealised PnL."""

        if prices:
            for symbol, price in prices.items():
                self.set_price(symbol, price)
        unrealized = 0.0
        used_margin = 0.0
        for position in self.positions.values():
            price = self.price(position.symbol)
            if price > 0:
                unrealized += position.unrealized_pnl(price)
            used_margin += position.notional(price) / max(position.leverage, 1.0)
        self.used_margin = used_margin
        self.equity = self.balance + unrealized
        self.available = max(0.0, self.balance - used_margin)
        self.peak_equity = max(self.peak_equity, self.equity)
        return unrealized

    # -- accounting -------------------------------------------------------

    def apply_realized(self, pnl: float, fees: float = 0.0, funding: float = 0.0) -> None:
        self._roll_day()
        net = pnl - fees - funding
        self.balance += net
        self.realized_pnl_today += net
        self.equity = self.balance
        self.peak_equity = max(self.peak_equity, self.equity)
        if net < 0:
            self.consecutive_losses += 1
        elif net > 0:
            self.consecutive_losses = 0

    def sync_balance(self, equity: float, available: float, used_margin: float = 0.0) -> None:
        """Adopt the exchange's account figures (LIVE mode)."""

        self.equity = equity
        self.available = available
        self.used_margin = used_margin
        unrealized = sum(
            p.unrealized_pnl(self.price(p.symbol))
            for p in self.positions.values()
            if self.price(p.symbol) > 0
        )
        self.balance = equity - unrealized
        self.peak_equity = max(self.peak_equity, equity)

    def _roll_day(self) -> None:
        today = _utc_day(time.time())
        if today != self._day:
            log.info(
                "new UTC day: resetting daily PnL (previous day %+.2f)",
                self.realized_pnl_today,
            )
            self._day = today
            self.realized_pnl_today = 0.0

    def refresh_day(self) -> None:
        self._roll_day()

    # -- state ------------------------------------------------------------

    def state(self) -> PortfolioState:
        self._roll_day()
        risks = [
            position.to_open_risk(self.price(position.symbol))
            for position in self.positions.values()
        ]
        return PortfolioState(
            equity=self.equity,
            available=self.available,
            open_risks=risks,
            realized_pnl_today=self.realized_pnl_today,
            unrealized_pnl=sum(
                p.unrealized_pnl(self.price(p.symbol))
                for p in self.positions.values()
                if self.price(p.symbol) > 0
            ),
            peak_equity=max(self.peak_equity, self.equity),
            consecutive_losses=self.consecutive_losses,
        )

    def update_correlations(self, series: Mapping[str, Sequence[Candle]]) -> None:
        if self.portfolio_risk is not None:
            self.portfolio_risk.matrix.update_many(series)

    def snapshot(self) -> dict[str, Any]:
        state = self.state()
        risk = (
            self.portfolio_risk.describe(state)
            if self.portfolio_risk is not None
            else {}
        )
        return {
            "mode": self.mode,
            "equity": round(self.equity, 4),
            "balance": round(self.balance, 4),
            "available": round(self.available, 4),
            "used_margin": round(self.used_margin, 4),
            "unrealized_pnl": round(state.unrealized_pnl, 4),
            "realized_pnl_day": round(self.realized_pnl_today, 4),
            "open_positions": len(self.positions),
            "peak_equity": round(self.peak_equity, 4),
            "drawdown": round(state.drawdown, 6),
            "consecutive_losses": self.consecutive_losses,
            "positions": [
                p.summary(self.price(p.symbol)) for p in self.positions.values()
            ],
            "risk": risk,
        }

    def record_snapshot(self) -> None:
        if self.repositories is None:
            return
        state = self.state()
        try:
            self.repositories.account.snapshot(
                {
                    "mode": self.mode,
                    "equity": self.equity,
                    "balance": self.balance,
                    "available": self.available,
                    "used_margin": self.used_margin,
                    "unrealized_pnl": state.unrealized_pnl,
                    "realized_pnl_day": self.realized_pnl_today,
                    "open_positions": len(self.positions),
                    "portfolio_risk": (
                        self.portfolio_risk.effective_risk_pct(state)
                        if self.portfolio_risk
                        else 0.0
                    ),
                    "drawdown": state.drawdown,
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("could not record account snapshot: %s", exc)


def _utc_day(timestamp: float) -> int:
    return int(timestamp) // 86400


__all__ = ["PortfolioManager", "ManagedPosition"]
