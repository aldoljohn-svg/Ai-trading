"""Outbound Telegram notifications.

Every message passes through :func:`app.logger.redact` before it is sent, so a
credential that somehow reached a reason string or an exception message cannot
be published to a chat.

Notifications are best-effort: a Telegram outage must never stop trading or
crash a loop, so all sending failures are swallowed and counted.
"""

from __future__ import annotations

import asyncio
import html
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.logger import get_logger, redact

log = get_logger(__name__)


@dataclass(slots=True)
class NotifierStats:
    sent: int = 0
    failed: int = 0
    suppressed: int = 0
    last_error: str = ""


class Notifier:
    def __init__(
        self,
        client: Any,
        chat_id: str,
        enabled: bool = True,
        min_interval: float = 0.4,
    ) -> None:
        self.client = client
        self.chat_id = chat_id
        self.enabled = enabled and bool(chat_id) and client is not None
        self.min_interval = min_interval
        self.stats = NotifierStats()
        self._last_sent = 0.0
        self._lock = asyncio.Lock()

    # -- transport --------------------------------------------------------

    async def send(
        self,
        text: str,
        keyboard: dict[str, Any] | None = None,
        chat_id: str | None = None,
        silent: bool = False,
    ) -> bool:
        if not self.enabled:
            self.stats.suppressed += 1
            return False
        safe = redact(text)
        async with self._lock:
            # Telegram rate-limits aggressively; a small floor avoids 429s.
            elapsed = time.monotonic() - self._last_sent
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            try:
                await self.client.send_message(
                    chat_id or self.chat_id,
                    safe,
                    keyboard=keyboard,
                    silent=silent,
                )
                self.stats.sent += 1
                self._last_sent = time.monotonic()
                return True
            except Exception as exc:  # noqa: BLE001 - never break the caller
                self.stats.failed += 1
                self.stats.last_error = str(exc)
                log.warning("telegram send failed: %s", exc)
                return False

    async def alert(self, text: str, keyboard: dict[str, Any] | None = None) -> bool:
        return await self.send(text, keyboard=keyboard)

    async def test_connection(self) -> bool:
        if not self.enabled:
            return False
        try:
            return bool(await self.client.get_me())
        except Exception as exc:  # noqa: BLE001
            self.stats.last_error = str(exc)
            return False

    # -- trade lifecycle ---------------------------------------------------

    async def trade_opened(
        self,
        position: Any,
        proposal: Any,
        decision: Any,
        equity: float,
        notes: Sequence[str] = (),
    ) -> bool:
        side_icon = "🟢" if position.side.value == "long" else "🔴"
        risk_pct = decision.risk_pct if decision else 0.0
        notional = position.notional()

        lines = [
            f"{side_icon} <b>NEW TRADE</b>",
            "",
            f"<b>{html.escape(position.symbol)}</b>",
            "",
            f"<b>{position.side.value.upper()}</b>",
            "",
            f"Equity:\n<b>${equity:,.2f}</b>",
            "",
            f"Risk:\n<b>{risk_pct:.2%}</b>",
            "",
            f"Risk Amount:\n<b>${decision.risk_amount:,.2f}</b>",
            "",
            f"Position:\n<b>{position.quantity:g} contracts</b>",
            "",
            f"Notional:\n<b>${notional:,.2f}</b>",
            "",
            f"Leverage:\n<b>{position.leverage:g}x</b>",
            "",
            f"Entry:\n<b>{position.entry_price:.8g}</b>",
            "",
            f"SL:\n<b>{position.stop_loss:.8g}</b>",
            "",
            f"TP1:\n{position.tp1:.8g}",
            f"TP2:\n{position.tp2:.8g}",
            f"TP3:\n{position.tp3:.8g}",
            "",
            f"R:R:\n<b>1:{proposal.rr:.1f}</b>",
            "",
            f"Confidence:\n<b>{proposal.confidence:.0%}</b>",
            "",
            f"Regime:\n{proposal.regime.value}",
            f"HTF: {proposal.htf_bias.value} (alignment {proposal.alignment:.0%})",
            "",
            "Reason:",
            "",
            html.escape("\n+\n".join(proposal.reasons[:8])),
        ]
        if proposal.stop_rationale:
            lines += ["", f"<i>Stop: {html.escape(proposal.stop_rationale)}</i>"]
        if notes:
            lines += ["", "⚠️ " + html.escape("; ".join(notes))]
        lines += ["", "<i>Status: OPEN - managed automatically</i>"]
        return await self.send("\n".join(lines))

    async def position_update(
        self,
        position: Any,
        event: str,
        detail: str = "",
        price: float = 0.0,
    ) -> bool:
        icons = {
            "TARGET_HIT": "🎯",
            "MOVE_STOP": "🛡",
            "stop moved": "🛡",
            "REDUCE_RISK": "🪄",
            "PARTIAL_CLOSE": "💰",
        }
        icon = icons.get(event, "ℹ️")
        r_multiple = position.r_multiple(price or position.entry_price)
        lines = [
            f"{icon} <b>{html.escape(position.symbol)}</b> - {html.escape(event)}",
            "",
            html.escape(detail),
            "",
            f"Price: {price:.8g}" if price else "",
            f"Stop: <b>{position.stop_loss:.8g}</b>",
            f"Remaining: {position.quantity:g} contracts",
            f"Open R: {r_multiple:+.2f}R",
            f"Risk left: ${position.open_risk(price or position.entry_price):,.2f}",
        ]
        if position.breakeven_done:
            lines.append("✅ Break-even active")
        if position.trailing_active:
            lines.append("📈 Trailing active")
        return await self.send("\n".join(line for line in lines if line))

    async def trade_closed(
        self, position: Any, reason: str, exit_price: float, pnl: float
    ) -> bool:
        icon = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
        risk = position.risk_amount or 1e-12
        held = max(0, int(time.time()) - position.opened_at)
        lines = [
            f"{icon} <b>TRADE CLOSED</b>",
            "",
            f"<b>{html.escape(position.symbol)}</b> {position.side.value.upper()}",
            "",
            f"Entry: {position.entry_price:.8g}",
            f"Exit: {exit_price:.8g}",
            "",
            f"PnL: <b>{pnl:+,.2f}</b>",
            f"R: <b>{pnl / risk:+.2f}R</b>",
            f"Fees: ${position.fees:,.4f}",
            f"Funding: {position.funding:+,.4f}",
            f"Held: {_duration(held)}",
            "",
            f"Why exited:\n{html.escape(reason)}",
        ]
        return await self.send("\n".join(lines))

    async def risk_event(self, kind: str, message: str, severity: str = "WARNING") -> bool:
        icon = {"INFO": "ℹ️", "WARNING": "🟡", "ERROR": "🔴", "CRITICAL": "🚨"}.get(
            severity, "🟡"
        )
        return await self.send(
            f"{icon} <b>RISK: {html.escape(kind)}</b>\n\n{html.escape(message)}"
        )

    async def health_alert(self, report: Any) -> bool:
        errors = report.errors() if hasattr(report, "errors") else []
        if not errors:
            return False
        lines = ["🔴 <b>HEALTH DEGRADED</b>", ""]
        for component in errors:
            lines.append(
                f"🔴 <b>{html.escape(component.name)}</b>: "
                f"{html.escape(str(component.detail))[:140]}"
            )
        return await self.send("\n".join(lines))

    async def startup(self, settings: Any, status: dict[str, Any]) -> bool:
        mode = settings.trading_mode.value.upper()
        warning = (
            "\n\n🔴 <b>LIVE MODE - REAL MONEY AT RISK</b>"
            if mode == "LIVE"
            else "\n\n<i>No real orders will be placed in this mode.</i>"
        )
        return await self.send(
            f"🤖 <b>BOT STARTED</b>\n\n"
            f"Mode: <b>{mode}</b>\n"
            f"Data source: {settings.data_source}\n"
            f"Instance: {html.escape(settings.instance_name)}\n"
            f"Risk/trade: {settings.default_risk_per_trade:.2%}\n"
            f"Max positions: {settings.max_open_positions}\n"
            f"Max leverage: {settings.max_leverage:g}x"
            f"{warning}",
        )


def _duration(seconds: int) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, _ = divmod(remainder, 60)
    if hours >= 24:
        return f"{hours // 24}d {hours % 24}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


__all__ = ["Notifier", "NotifierStats"]
