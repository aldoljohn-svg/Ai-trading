"""Command router.

The router is a plain async class with no network dependency: it takes a
command string and returns text plus an optional keyboard.  That makes every
command unit-testable against a fake engine, which matters because these are
the controls an operator reaches for when something is going wrong.

Destructive commands (``/stop``, ``/emergency``, closing a position) always
require a second confirmation.
"""

from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.logger import get_logger, redact
from app.telegram import keyboards

log = get_logger(__name__)

COMMANDS: dict[str, str] = {
    "start": "show the control panel",
    "menu": "show the control panel",
    "status": "live report: state, equity, positions, health",
    "pause": "stop opening new trades (open positions keep being managed)",
    "resume": "resume opening new trades",
    "stop": "stop the bot (requires confirmation)",
    "emergency": "close every position and stop (requires confirmation)",
    "positions": "open positions with live PnL",
    "orders": "recent orders",
    "account": "balance, margin, exposure and drawdown",
    "performance": "win rate, profit factor, expectancy",
    "scanner": "ranked market scan results",
    "signals": "recent signals including the rejected ones",
    "risk": "risk limits, current usage and circuit breakers",
    "ai": "model status, calibration and recent probabilities",
    "trades": "recent trade history",
    "health": "component health",
    "help": "this list",
}


@dataclass(slots=True)
class CommandResult:
    text: str
    keyboard: dict[str, Any] | None = None
    alert: str = ""
    handled: bool = True

    def __bool__(self) -> bool:
        return self.handled


class CommandRouter:
    def __init__(self, engine: Any, settings: Any) -> None:
        self.engine = engine
        self.settings = settings
        self._pending_confirmations: dict[int, tuple[str, float]] = {}

    # -- authorisation ----------------------------------------------------

    def is_authorised(self, user_id: int, chat_id: int | str = "") -> bool:
        """Only explicitly listed Telegram user ids may control the bot."""

        allowed = set(self.settings.telegram_allowed_user_ids)
        if allowed:
            return int(user_id) in allowed
        # With no allowlist, fall back to the configured chat id.  Live mode
        # refuses to start without an allowlist, so this only affects paper.
        configured = str(self.settings.telegram_chat_id or "")
        return bool(configured) and str(chat_id) == configured

    # -- dispatch ---------------------------------------------------------

    async def handle(self, raw: str, user_id: int = 0) -> CommandResult:
        text = (raw or "").strip()
        if not text:
            return CommandResult(text="", handled=False)

        if text.startswith("cmd:"):
            command, _, args = text[4:].partition(":")
        elif text.startswith("confirm:"):
            return await self._handle_confirmation(text[len("confirm:") :], user_id)
        elif text.startswith("close:"):
            symbol = text[len("close:") :].upper()
            return CommandResult(
                text=f"Close <b>{html.escape(symbol)}</b> at market?",
                keyboard=keyboards.close_confirm(symbol),
            )
        else:
            body = text.lstrip("/")
            command, _, args = body.partition(" ")
            command = command.split("@")[0]

        command = command.lower().strip()
        handler = getattr(self, f"_cmd_{command}", None)
        if handler is None:
            return CommandResult(
                text=f"Unknown command <code>{html.escape(command)}</code>. Try /help.",
                keyboard=keyboards.back_to_menu(),
            )
        return await handler(args.strip(), user_id)

    async def _handle_confirmation(self, payload: str, user_id: int) -> CommandResult:
        action, _, argument = payload.partition(":")
        if action == "emergency":
            result = await self.engine.emergency_stop("Telegram emergency stop")
            return CommandResult(
                text=(
                    "🚨 <b>EMERGENCY STOP EXECUTED</b>\n\n"
                    f"Positions closed: {result.get('closed', 0)}\n"
                    f"Failed: {result.get('failed', 0)}\n"
                    f"Orders cancelled: {result.get('cancelled', 0)}\n\n"
                    "The bot is stopped. Restart the process to trade again."
                ),
                alert="Emergency stop executed",
            )
        if action == "stop":
            await self.engine.stop("stopped from Telegram")
            return CommandResult(
                text="🛑 <b>Bot stopped.</b>\n\nOpen positions were left untouched.",
                alert="Stopped",
            )
        if action == "close":
            symbol = argument.upper()
            position = self.engine.portfolio.get(symbol) if self.engine.portfolio else None
            if position is None:
                return CommandResult(text=f"No open position in {html.escape(symbol)}.")
            result = await self.engine.execution.close_position(
                position, 1.0, reason="closed from Telegram"
            )
            if not result.ok:
                return CommandResult(text=f"❌ Could not close {symbol}: {result.error}")
            return CommandResult(
                text=f"✅ Closed <b>{html.escape(symbol)}</b> at {result.average_price:.6g}",
                keyboard=keyboards.back_to_menu(),
                alert=f"{symbol} closed",
            )
        return CommandResult(text="Unknown confirmation.", handled=False)

    # -- commands ---------------------------------------------------------

    async def _cmd_start(self, args: str, user_id: int) -> CommandResult:
        return await self._cmd_menu(args, user_id)

    async def _cmd_menu(self, args: str, user_id: int) -> CommandResult:
        status = self.engine.status()
        account = status.get("account", {})
        text = (
            "🤖 <b>TRADING BOT</b>\n\n"
            f"Status:\n{status['emoji']} <b>{status['state']}</b>\n\n"
            f"Mode: <b>{status['mode'].upper()}</b>"
            + (" 🔴 REAL MONEY" if status["mode"] == "live" else "")
            + "\n"
            f"Equity: <b>${_num(account.get('equity'))}</b>\n"
            f"Open positions: <b>{account.get('open_positions', 0)}</b>\n"
            f"Health: {status.get('health', 'UNKNOWN')}\n"
        )
        if status.get("pause_reason"):
            text += f"\n⏸ {html.escape(status['pause_reason'])}"
        return CommandResult(text=text, keyboard=keyboards.main_menu())

    async def _cmd_help(self, args: str, user_id: int) -> CommandResult:
        lines = ["<b>Commands</b>", ""]
        for name, description in COMMANDS.items():
            lines.append(f"/{name} - {description}")
        return CommandResult(text="\n".join(lines), keyboard=keyboards.back_to_menu())

    async def _cmd_status(self, args: str, user_id: int) -> CommandResult:
        status = self.engine.status()
        account = status.get("account", {})
        positions = self.engine.positions_view()

        lines = [
            f"{status['emoji']} <b>{status['state']}</b>  |  {status['mode'].upper()}",
            "",
            f"Equity: <b>${_num(account.get('equity'))}</b>",
            f"Balance: ${_num(account.get('balance'))}",
            f"Available: ${_num(account.get('available'))}",
            f"Unrealised: {_signed(account.get('unrealized_pnl'))}",
            f"Today: {_signed(account.get('realized_pnl_day'))}",
            f"Drawdown: {_pct(account.get('drawdown'))}",
            "",
            f"Positions: <b>{len(positions)}</b>",
        ]
        for position in positions[:5]:
            lines.append(
                f"  {_side_icon(position['side'])} {position['symbol']} "
                f"{_signed(position['unrealized_pnl'])} "
                f"({position['r_multiple']:+.2f}R)"
            )
        lines += [
            "",
            f"Scans: {status['scans']}  Entries: {status['entries']}  Exits: {status['exits']}",
            f"Uptime: {_duration(status['uptime_seconds'])}",
            f"Health: {status.get('health')}",
        ]
        if status.get("last_error"):
            lines.append(f"\nLast error: <code>{html.escape(redact(status['last_error'])[:180])}</code>")
        return CommandResult(text="\n".join(lines), keyboard=keyboards.refresh_menu("status"))

    async def _cmd_scanner(self, args: str, user_id: int) -> CommandResult:
        rows = self.engine.scanner_view(limit=12)
        if not rows:
            return CommandResult(
                text="No scan results yet. The first scan runs shortly after start.",
                keyboard=keyboards.refresh_menu("scanner"),
            )
        lines = ["📈 <b>MARKET SCANNER</b>", ""]
        for row in rows:
            lines.append(
                f"<b>{row['symbol']}</b>  {row['opportunity_score']:.0f}\n"
                f"  {row['trend']} | conf {_pct(row.get('confidence'))} | "
                f"R:R {row.get('rr', 0):.1f} | {row['regime']}\n"
                f"  <i>{row['status']}</i>"
            )
        lines.append("")
        lines.append("<i>A high score is not permission to trade.</i>")
        return CommandResult(text="\n".join(lines), keyboard=keyboards.refresh_menu("scanner"))

    async def _cmd_positions(self, args: str, user_id: int) -> CommandResult:
        positions = self.engine.positions_view()
        if not positions:
            return CommandResult(
                text="💼 No open positions.", keyboard=keyboards.refresh_menu("positions")
            )
        lines = ["💼 <b>OPEN POSITIONS</b>", ""]
        for position in positions:
            lines.append(
                f"{_side_icon(position['side'])} <b>{position['symbol']}</b> "
                f"{position['side'].upper()} {position['leverage']:g}x\n"
                f"  Entry: {position['entry']:.6g}\n"
                f"  Now: {position['current']:.6g}\n"
                f"  Stop: {position['stop_loss']:.6g}"
                + ("  (BE)" if position.get("breakeven") else "")
                + ("  (trailing)" if position.get("trailing") else "")
                + "\n"
                f"  TP: {position['tp1']:.6g} / {position['tp2']:.6g} / {position['tp3']:.6g}\n"
                f"  PnL: {_signed(position['unrealized_pnl'])} "
                f"({position['r_multiple']:+.2f}R)\n"
                f"  Risk left: ${_num(position['open_risk'])}"
            )
            lines.append("")
        return CommandResult(
            text="\n".join(lines),
            keyboard=keyboards.positions_menu([p["symbol"] for p in positions]),
        )

    async def _cmd_orders(self, args: str, user_id: int) -> CommandResult:
        orders = self.engine.orders_view(limit=12)
        if not orders:
            return CommandResult(text="No orders recorded yet.", keyboard=keyboards.back_to_menu())
        lines = ["📋 <b>RECENT ORDERS</b>", ""]
        for order in orders:
            lines.append(
                f"{order['symbol']} {order['side']} {order['intent']} "
                f"{order['quantity']:g} - <b>{order['status']}</b>"
                + (f"\n  filled {order['filled_quantity']:g} @ {order['average_price']:.6g}"
                   if order.get("filled_quantity") else "")
            )
        return CommandResult(text="\n".join(lines), keyboard=keyboards.refresh_menu("orders"))

    async def _cmd_account(self, args: str, user_id: int) -> CommandResult:
        snapshot = self.engine.account_view()
        risk = snapshot.get("risk", {})
        lines = [
            "💰 <b>ACCOUNT</b>",
            "",
            f"Mode: <b>{snapshot.get('mode', '?').upper()}</b>",
            f"Equity: <b>${_num(snapshot.get('equity'))}</b>",
            f"Balance: ${_num(snapshot.get('balance'))}",
            f"Available margin: ${_num(snapshot.get('available'))}",
            f"Used margin: ${_num(snapshot.get('used_margin'))}",
            f"Unrealised PnL: {_signed(snapshot.get('unrealized_pnl'))}",
            f"Realised today: {_signed(snapshot.get('realized_pnl_day'))}",
            f"Peak equity: ${_num(snapshot.get('peak_equity'))}",
            f"Drawdown: {_pct(snapshot.get('drawdown'))}",
            "",
            f"Open positions: {snapshot.get('open_positions', 0)}",
            f"Portfolio risk: {_pct(risk.get('effective_risk_pct'))}",
            f"Gross leverage: {risk.get('gross_leverage', 0):.2f}x",
        ]
        return CommandResult(text="\n".join(lines), keyboard=keyboards.refresh_menu("account"))

    async def _cmd_performance(self, args: str, user_id: int) -> CommandResult:
        days = 30
        if args.isdigit():
            days = max(1, min(int(args), 365))
        data = self.engine.performance_view(days=days)
        if not data or data.get("trades", 0) == 0:
            return CommandResult(
                text="📊 No closed trades yet.", keyboard=keyboards.back_to_menu()
            )
        profit_factor = data.get("profit_factor")
        lines = [
            f"📊 <b>PERFORMANCE</b> ({days}d)",
            "",
            f"Trades: <b>{data['trades']}</b>  (W {data['wins']} / L {data['losses']})",
            f"Win rate: <b>{_pct(data['win_rate'])}</b>",
            f"Net PnL: {_signed(data['total_pnl'])}",
            f"Profit factor: {profit_factor if profit_factor is not None else 'n/a'}",
            f"Expectancy: <b>{data['expectancy_r']:+.3f}R</b> per trade",
            f"Average win: {_signed(data['average_win'])}",
            f"Average loss: {_signed(data['average_loss'])}",
            f"Best: {_signed(data['best'])}   Worst: {_signed(data['worst'])}",
            f"Fees paid: ${_num(data['fees'])}",
        ]
        if data["trades"] < 30:
            lines.append("")
            lines.append("<i>⚠️ Small sample - these numbers are not yet meaningful.</i>")
        return CommandResult(text="\n".join(lines), keyboard=keyboards.refresh_menu("performance"))

    async def _cmd_risk(self, args: str, user_id: int) -> CommandResult:
        risk = self.engine.risk_view()
        if not risk:
            return CommandResult(text="Risk engine is not ready.", keyboard=keyboards.back_to_menu())
        lines = [
            "⚙️ <b>RISK</b>",
            "",
            "<b>Limits</b>",
            f"  Risk per trade: {_pct(risk.get('risk_per_trade'))}",
            f"  Max daily loss: {_pct(risk.get('max_daily_loss'))}",
            f"  Max portfolio risk: {_pct(risk.get('max_portfolio_risk'))}",
            f"  Max positions: {risk.get('max_open_positions')}",
            f"  Max leverage: {risk.get('max_leverage')}x",
            "",
            "<b>Current</b>",
            f"  Open positions: {risk.get('open_positions')}",
            f"  Portfolio risk: {_pct(risk.get('effective_risk_pct'))}",
            f"  Largest cluster: {_pct(risk.get('cluster_risk_pct'))}",
            f"  Gross leverage: {risk.get('gross_leverage', 0):.2f}x",
            f"  Daily loss: {_pct(risk.get('daily_loss_pct'))}",
            f"  Drawdown: {_pct(risk.get('drawdown'))}",
            f"  Losing streak: {risk.get('consecutive_losses')}",
            f"  Budget left: ${_num(risk.get('remaining_risk_budget'))}",
        ]
        clusters = risk.get("clusters") or []
        if clusters:
            lines.append("")
            lines.append("<b>Correlation clusters</b>")
            for cluster in clusters:
                lines.append(f"  {' + '.join(cluster)}")
        lines.append("")
        lines.append(f"<b>Breakers:</b> {risk.get('breaker_state')}")
        summary = risk.get("breaker_summary", "")
        if summary:
            lines.append(f"<pre>{html.escape(summary)}</pre>")
        return CommandResult(text="\n".join(lines), keyboard=keyboards.refresh_menu("risk"))

    async def _cmd_ai(self, args: str, user_id: int) -> CommandResult:
        data = self.engine.ai_view()
        model = data.get("model", {})
        lines = ["🧠 <b>AI ENGINE</b>", ""]
        if not model.get("loaded"):
            lines.append("Model: <b>none loaded</b>")
            lines.append("<i>Decisions are rule-based only. Train one with")
            lines.append("<code>python scripts/train_model.py</code></i>")
        else:
            lines += [
                f"Model: <b>{model.get('model')}</b>",
                f"Algorithm: {model.get('algorithm')}",
                f"Training rows: {model.get('rows')}",
                f"Features: {model.get('features')}",
                f"Accuracy: {model.get('accuracy')}",
                f"Brier (long): {model.get('brier_long')}",
                f"Calibration error: {model.get('ece_long')}",
                f"Calibrated: {'yes' if model.get('calibrated') else 'no'}",
                f"Predictions: {model.get('predictions')}",
            ]
        lines += [
            "",
            f"ML weight in the blend: {_pct(data.get('ml_weight'))}",
            f"Minimum confidence: {_pct(data.get('min_confidence'))}",
            "",
            "<b>Recent decisions</b>",
        ]
        for signal in data.get("recent_signals", [])[:6]:
            icon = "✅" if signal["decision"] == "ENTER" else "⛔"
            lines.append(
                f"{icon} <b>{signal['symbol']}</b> conf {_pct(signal.get('confidence'))}"
            )
            if signal["decision"] == "ENTER" and signal.get("reasons"):
                lines.append(f"   {' + '.join(signal['reasons'][:3])}")
            elif signal.get("rejections"):
                lines.append(f"   <i>{html.escape(str(signal['rejections'][0])[:90])}</i>")
        lines += [
            "",
            "<i>Probabilities are estimates, not predictions. The system never "
            "claims to know what happens next.</i>",
        ]
        return CommandResult(text="\n".join(lines), keyboard=keyboards.refresh_menu("ai"))

    async def _cmd_signals(self, args: str, user_id: int) -> CommandResult:
        signals = self.engine.signals_view(limit=12)
        if not signals:
            return CommandResult(text="No signals recorded yet.", keyboard=keyboards.back_to_menu())
        lines = ["🎯 <b>RECENT SIGNALS</b>", ""]
        for signal in signals:
            icon = "✅" if signal["decision"] == "ENTER" else "⛔"
            lines.append(
                f"{icon} <b>{signal['symbol']}</b> "
                f"{(signal.get('side') or '-').upper()} "
                f"conf {_pct(signal.get('confidence'))} R:R {signal.get('rr', 0):.1f}"
            )
            reasons = signal.get("reasons") or []
            rejections = signal.get("rejections") or []
            if signal["decision"] == "ENTER" and reasons:
                lines.append(f"   {html.escape(' + '.join(reasons[:3]))}")
            elif rejections:
                lines.append(f"   <i>{html.escape(str(rejections[0])[:100])}</i>")
        return CommandResult(text="\n".join(lines), keyboard=keyboards.refresh_menu("signals"))

    async def _cmd_trades(self, args: str, user_id: int) -> CommandResult:
        trades = self.engine.trades_view(limit=12)
        if not trades:
            return CommandResult(text="No trades yet.", keyboard=keyboards.back_to_menu())
        lines = ["📜 <b>RECENT TRADES</b>", ""]
        for trade in trades:
            status = trade.get("status")
            pnl = float(trade.get("realized_pnl") or 0.0)
            icon = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
            if status == "open":
                icon = "🔵"
            lines.append(
                f"{icon} <b>{trade['symbol']}</b> {trade['side'].upper()} "
                + (
                    f"{_signed(pnl)} ({float(trade.get('r_multiple') or 0):+.2f}R)"
                    if status == "closed"
                    else "OPEN"
                )
            )
            exits = trade.get("why_exited") or []
            if exits:
                lines.append(f"   <i>{html.escape(str(exits[0])[:80])}</i>")
        return CommandResult(text="\n".join(lines), keyboard=keyboards.refresh_menu("trades"))

    async def _cmd_health(self, args: str, user_id: int) -> CommandResult:
        health = self.engine.health_view()
        lines = [f"<b>SYSTEM HEALTH: {health.get('state', 'UNKNOWN')}</b>", ""]
        for component in health.get("components", []):
            emoji = {
                "HEALTHY": "🟢",
                "WARNING": "🟡",
                "ERROR": "🔴",
                "UNKNOWN": "⚪",
            }.get(component["state"], "⚪")
            lines.append(
                f"{emoji} <b>{component['name']}</b>: "
                f"{html.escape(redact(str(component.get('detail', ''))))[:90]}"
            )
        preflight = self.engine.preflight_view()
        if preflight.get("passed") is not None:
            lines.append("")
            lines.append(
                f"Pre-flight: {'✅ passed' if preflight['passed'] else '🚫 failed'}"
            )
        return CommandResult(text="\n".join(lines), keyboard=keyboards.refresh_menu("health"))

    # -- control ----------------------------------------------------------

    async def _cmd_pause(self, args: str, user_id: int) -> CommandResult:
        self.engine.pause(f"paused from Telegram by user {user_id}")
        return CommandResult(
            text=(
                "⏸ <b>PAUSED</b>\n\nNo new trades will be opened.\n"
                "Open positions are still fully managed (stops, targets, trailing)."
            ),
            keyboard=keyboards.main_menu(),
            alert="Paused",
        )

    async def _cmd_resume(self, args: str, user_id: int) -> CommandResult:
        self.engine.resume()
        return CommandResult(
            text="▶️ <b>RESUMED</b>\n\nThe bot will consider new trades again.",
            keyboard=keyboards.main_menu(),
            alert="Resumed",
        )

    async def _cmd_start_trading(self, args: str, user_id: int) -> CommandResult:
        state = self.engine.status()["state"]
        if state == "PAUSED":
            return await self._cmd_resume(args, user_id)
        if state == "RUNNING":
            return CommandResult(
                text="The bot is already running.", keyboard=keyboards.main_menu()
            )
        await self.engine.start()
        return CommandResult(
            text="▶️ <b>STARTED</b>", keyboard=keyboards.main_menu(), alert="Started"
        )

    async def _cmd_stop(self, args: str, user_id: int) -> CommandResult:
        return CommandResult(
            text=(
                "🛑 <b>STOP THE BOT?</b>\n\n"
                "This stops scanning and management. Open positions are <b>left "
                "open on the exchange</b> with their stops in place.\n\n"
                "Use /emergency instead if you want everything closed."
            ),
            keyboard=keyboards.stop_confirm(),
        )

    async def _cmd_emergency(self, args: str, user_id: int) -> CommandResult:
        positions = self.engine.positions_view()
        return CommandResult(
            text=(
                "⚠️ <b>EMERGENCY STOP</b>\n\n"
                f"This will <b>market-close all {len(positions)} open position(s)</b>, "
                "cancel every order, and stop the bot.\n\n"
                "Closing at market crystallises losses immediately.\n\n"
                "Do you want to:"
            ),
            keyboard=keyboards.emergency_confirm(),
        )


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------


def _num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _signed(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{'+' if number >= 0 else ''}{number:,.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def _side_icon(side: str) -> str:
    return "🟢" if str(side).lower() == "long" else "🔴"


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, _ = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


__all__ = ["CommandRouter", "CommandResult", "COMMANDS"]
