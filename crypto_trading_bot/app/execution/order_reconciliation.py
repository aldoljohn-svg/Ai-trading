"""Reconciliation between our records and the exchange.

Our database records *intent*; the exchange records *reality*.  They can
diverge for perfectly ordinary reasons - a request that timed out after the
venue accepted it, a manual close from the MEXC app, a liquidation, a partial
fill we never saw.  Divergence is not itself an emergency, but acting on a
stale picture is.

Policy:

* **Adopt** positions the exchange has that we do not know about, flat and
  unmanaged, and immediately alert - we cannot invent the original stop.
* **Drop** positions we think we have that the exchange does not, after
  recording the outcome.
* **Correct** quantity drift silently when it is within rounding, loudly
  otherwise.
* Any *material* mismatch pauses new entries until an operator confirms.

This runs at startup (crash recovery) and periodically thereafter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from app.domain import ExchangeOrder, ExchangePosition, OrderStatus, Side
from app.exchange.base import BaseExchange, ExchangeError
from app.logger import get_logger
from app.portfolio.portfolio_manager import ManagedPosition, PortfolioManager

log = get_logger(__name__)


class MismatchKind(str, Enum):
    UNKNOWN_POSITION = "UNKNOWN_POSITION"      # venue has it, we do not
    MISSING_POSITION = "MISSING_POSITION"      # we have it, venue does not
    QUANTITY_DRIFT = "QUANTITY_DRIFT"
    SIDE_MISMATCH = "SIDE_MISMATCH"
    ORPHAN_ORDER = "ORPHAN_ORDER"              # open order with no position
    STALE_ORDER = "STALE_ORDER"                # our record says open, venue disagrees


@dataclass(frozen=True, slots=True)
class Mismatch:
    kind: MismatchKind
    symbol: str
    message: str
    severity: str = "WARNING"            # "INFO" | "WARNING" | "CRITICAL"
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_material(self) -> bool:
        return self.severity == "CRITICAL"


@dataclass(slots=True)
class ReconciliationReport:
    ts: int
    checked_positions: int = 0
    checked_orders: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    adopted: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    corrected: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def healthy(self) -> bool:
        return not self.error and not any(m.is_material for m in self.mismatches)

    @property
    def blocking(self) -> bool:
        return bool(self.error) or any(m.is_material for m in self.mismatches)

    def summary(self) -> str:
        if self.error:
            return f"🔴 reconciliation failed: {self.error}"
        if not self.mismatches:
            return (
                f"🟢 in sync ({self.checked_positions} positions, "
                f"{self.checked_orders} orders)"
            )
        lines = [f"🟡 {len(self.mismatches)} mismatch(es):"]
        for mismatch in self.mismatches:
            icon = "🔴" if mismatch.is_material else "🟡"
            lines.append(f"  {icon} {mismatch.symbol}: {mismatch.message}")
        return "\n".join(lines)


class Reconciler:
    def __init__(
        self,
        exchange: BaseExchange,
        portfolio: PortfolioManager,
        repositories: Any = None,
        quantity_tolerance: float = 0.02,
        adopt_unknown: bool = True,
    ) -> None:
        self.exchange = exchange
        self.portfolio = portfolio
        self.repositories = repositories
        self.quantity_tolerance = quantity_tolerance
        self.adopt_unknown = adopt_unknown
        self.last_report: ReconciliationReport | None = None

    async def reconcile(self, apply: bool = True) -> ReconciliationReport:
        report = ReconciliationReport(ts=int(time.time()))
        try:
            venue_positions = await self.exchange.positions()
            venue_orders = await self.exchange.open_orders()
        except ExchangeError as exc:
            report.error = str(exc)
            self.last_report = report
            log.error("reconciliation could not read the exchange: %s", exc)
            return report

        local = {p.symbol.upper(): p for p in self.portfolio.all()}
        remote = {p.symbol.upper(): p for p in venue_positions}
        report.checked_positions = len(set(local) | set(remote))
        report.checked_orders = len(venue_orders)

        # --- positions the venue has that we do not -----------------------
        for symbol, position in remote.items():
            if symbol in local:
                continue
            message = (
                f"exchange reports {position.side.value} {position.quantity:g} "
                f"contracts @ {position.entry_price:.6g} that this bot did not open"
            )
            report.mismatches.append(
                Mismatch(
                    kind=MismatchKind.UNKNOWN_POSITION,
                    symbol=symbol,
                    message=message,
                    severity="CRITICAL",
                    detail={
                        "side": position.side.value,
                        "quantity": position.quantity,
                        "entry": position.entry_price,
                    },
                )
            )
            if apply and self.adopt_unknown:
                self._adopt(position)
                report.adopted.append(symbol)

        # --- positions we have that the venue does not --------------------
        for symbol, position in local.items():
            if symbol in remote:
                continue
            message = (
                "this bot holds a position the exchange does not report - it was "
                "closed externally, liquidated, or never opened"
            )
            report.mismatches.append(
                Mismatch(
                    kind=MismatchKind.MISSING_POSITION,
                    symbol=symbol,
                    message=message,
                    severity="CRITICAL",
                    detail={"quantity": position.quantity, "side": position.side.value},
                )
            )
            if apply:
                self._drop(position)
                report.dropped.append(symbol)

        # --- positions both sides have ------------------------------------
        for symbol in set(local) & set(remote):
            ours = local[symbol]
            theirs = remote[symbol]

            if ours.side is not theirs.side:
                report.mismatches.append(
                    Mismatch(
                        kind=MismatchKind.SIDE_MISMATCH,
                        symbol=symbol,
                        message=(
                            f"we think {ours.side.value}, the exchange says "
                            f"{theirs.side.value}"
                        ),
                        severity="CRITICAL",
                        detail={"ours": ours.side.value, "theirs": theirs.side.value},
                    )
                )
                if apply:
                    self._adopt(theirs, existing=ours)
                    report.corrected.append(symbol)
                continue

            if ours.quantity <= 0:
                continue
            drift = abs(ours.quantity - theirs.quantity) / ours.quantity
            if drift > self.quantity_tolerance:
                severity = "CRITICAL" if drift > 0.10 else "WARNING"
                report.mismatches.append(
                    Mismatch(
                        kind=MismatchKind.QUANTITY_DRIFT,
                        symbol=symbol,
                        message=(
                            f"quantity differs by {drift:.1%} "
                            f"(ours {ours.quantity:g}, exchange {theirs.quantity:g}) "
                            "- adopting the exchange figure"
                        ),
                        severity=severity,
                        detail={"ours": ours.quantity, "theirs": theirs.quantity},
                    )
                )
                if apply:
                    ours.quantity = theirs.quantity
                    if theirs.entry_price > 0:
                        ours.entry_price = theirs.entry_price
                    self.portfolio.persist(ours)
                    report.corrected.append(symbol)

        # --- orphan orders --------------------------------------------------
        for order in venue_orders:
            symbol = order.symbol.upper()
            if symbol not in remote and order.reduce_only:
                report.mismatches.append(
                    Mismatch(
                        kind=MismatchKind.ORPHAN_ORDER,
                        symbol=symbol,
                        message=(
                            f"reduce-only order {order.order_id} is open with no "
                            "position behind it"
                        ),
                        severity="WARNING",
                        detail={"order_id": order.order_id},
                    )
                )

        # --- our open-order records versus the venue -------------------------
        if self.repositories is not None:
            remote_ids = {o.order_id for o in venue_orders}
            try:
                for row in self.repositories.orders.open_orders(
                    mode=self.portfolio.mode
                ):
                    exchange_id = row.get("exchange_order_id")
                    if not exchange_id:
                        report.mismatches.append(
                            Mismatch(
                                kind=MismatchKind.STALE_ORDER,
                                symbol=str(row.get("symbol", "")),
                                message=(
                                    f"order {row.get('client_order_id')} was written "
                                    "ahead but never acknowledged by the exchange"
                                ),
                                severity="CRITICAL",
                                detail={"client_order_id": row.get("client_order_id")},
                            )
                        )
                    elif exchange_id not in remote_ids:
                        if apply:
                            self.repositories.orders.update_status(
                                str(row["client_order_id"]),
                                OrderStatus.UNKNOWN.value,
                                error="not present on the exchange at reconciliation",
                            )
                        report.mismatches.append(
                            Mismatch(
                                kind=MismatchKind.STALE_ORDER,
                                symbol=str(row.get("symbol", "")),
                                message=(
                                    f"order {exchange_id} is recorded as open but the "
                                    "exchange does not list it"
                                ),
                                severity="WARNING",
                                detail={"exchange_order_id": exchange_id},
                            )
                        )
            except Exception as exc:  # noqa: BLE001
                log.warning("could not check order records: %s", exc)

        self.last_report = report
        self._record(report)
        if report.mismatches:
            log.warning("reconciliation: %s", report.summary())
        else:
            log.info("reconciliation: %s", report.summary())
        return report

    # -- actions ----------------------------------------------------------

    def _adopt(
        self, position: ExchangePosition, existing: ManagedPosition | None = None
    ) -> ManagedPosition:
        """Take ownership of an externally-created position.

        We cannot know the original thesis, so the position is adopted with
        **no stop and no targets**, flagged as unmanaged.  The position manager
        will place a protective ATR stop on its next pass, and the operator is
        alerted immediately.
        """

        contract_size = 1.0
        adopted = ManagedPosition(
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            leverage=position.leverage or 1.0,
            contract_size=contract_size,
            stop_loss=0.0,
            initial_stop=0.0,
            initial_quantity=position.quantity,
            risk_amount=0.0,
            opened_at=int(time.time()),
            mode=self.portfolio.mode,
            id=existing.id if existing else None,
            trade_id=existing.trade_id if existing else None,
            meta={
                "adopted": True,
                "adopted_at": int(time.time()),
                "source": "reconciliation",
                "note": (
                    "position was not opened by this bot; it has no original stop "
                    "or target plan"
                ),
            },
        )
        self.portfolio.positions[adopted.symbol] = adopted
        self.portfolio.persist(adopted)
        log.warning("adopted unmanaged position %s from the exchange", adopted.symbol)
        return adopted

    def _drop(self, position: ManagedPosition) -> None:
        position.state = "closed"
        self.portfolio.remove(position.symbol)
        if self.repositories is not None and position.trade_id is not None:
            try:
                self.repositories.trades.close_trade(
                    trade_id=position.trade_id,
                    exit_price=self.portfolio.price(position.symbol) or position.entry_price,
                    realized_pnl=position.realized_pnl,
                    fees=position.fees,
                    funding=position.funding,
                    r_multiple=0.0,
                    why_exited=["closed externally - detected by reconciliation"],
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("could not close trade record: %s", exc)
        log.warning("dropped %s - the exchange does not report it", position.symbol)

    def _record(self, report: ReconciliationReport) -> None:
        if self.repositories is None:
            return
        try:
            if report.mismatches or report.error:
                self.repositories.events.risk_event(
                    kind="RECONCILIATION",
                    message=report.summary(),
                    severity="ERROR" if report.blocking else "WARNING",
                    detail={
                        "adopted": report.adopted,
                        "dropped": report.dropped,
                        "corrected": report.corrected,
                        "mismatches": [
                            {
                                "kind": m.kind.value,
                                "symbol": m.symbol,
                                "message": m.message,
                                "severity": m.severity,
                            }
                            for m in report.mismatches
                        ],
                    },
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("could not record reconciliation event: %s", exc)


__all__ = [
    "Reconciler",
    "ReconciliationReport",
    "Mismatch",
    "MismatchKind",
]
