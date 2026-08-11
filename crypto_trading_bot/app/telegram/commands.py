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
    "models": "per-model weights, reliability and calibration",
    "flow": "order flow, book depth and derivatives per symbol",
    "liquidity": "liquidity pools and where price is being pulled",
    "regime": "current regime per scanned symbol",
    "journal": "recent decisions with their reasons",
    "why": "/why SYMBOL - the full decision trace for one symbol",
    "memory": "what the market memory has learned so far",
    "learning": "autonomous retraining: what was tried, what won, what was held",
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

    # -- intelligence layer ------------------------------------------------

    def _intelligence(self) -> dict[str, Any]:
        view = getattr(self.engine, "intelligence_view", None)
        return view() if callable(view) else {"enabled": False}

    async def _cmd_models(self, args: str, user_id: int) -> CommandResult:
        data = self._intelligence()
        if not data.get("enabled"):
            return CommandResult(
                text="The intelligence layer is disabled "
                "(<code>INTELLIGENCE_ENABLED=false</code>).",
                keyboard=keyboards.back_to_menu(),
            )

        lines = ["🗳 <b>MODEL ENSEMBLE</b>", ""]
        models = data.get("models", [])
        if not models:
            lines.append("No model weights recorded yet.")
        else:
            lines.append("<code>model            wt   rel  n   hit</code>")
            for entry in models[:20]:
                lines.append(
                    "<code>{:<15s} {:>4.0%} {:>4.0%} {:>3d} {:>4}</code>".format(
                        str(entry["model"])[:15],
                        entry["weight"],
                        entry["reliability"],
                        int(entry["trades"]),
                        f"{entry['hit_rate']:.0%}" if entry["trades"] else "-",
                    )
                )
            drifted = [
                e for e in models if abs(float(e.get("calibration_gap") or 0)) > 0.1
            ]
            if drifted:
                lines += ["", "<b>Poorly calibrated</b>"]
                for entry in drifted[:5]:
                    lines.append(
                        f"  {entry['model']}: states "
                        f"{entry['calibration_gap']:+.0%} more confidence than it earns"
                    )

        thresholds = data.get("thresholds", {})
        lines += [
            "",
            f"Minimum agreement: {_pct(thresholds.get('min_model_agreement'))}",
            f"Minimum quality: {thresholds.get('min_trade_quality', '-')}/100",
            f"Minimum EV: {thresholds.get('min_expected_value_r', 0):+.3f}R",
            "",
            "<i>Weights are earned from realised outcomes and are bounded, so no "
            "single model can dominate the vote.</i>",
        ]
        return CommandResult(
            text="\n".join(lines), keyboard=keyboards.refresh_menu("models")
        )

    async def _cmd_flow(self, args: str, user_id: int) -> CommandResult:
        rows = self._call_view("flow_view", limit=6)
        if not rows:
            return CommandResult(
                text="No order flow captured yet - run a scan first.",
                keyboard=keyboards.back_to_menu(),
            )

        lines = ["🌊 <b>ORDER FLOW</b>", ""]
        for row in rows:
            flow = row.get("order_flow") or {}
            micro = row.get("microstructure") or {}
            deriv = row.get("derivatives") or {}
            lines.append(f"<b>{html.escape(str(row['symbol']))}</b>")
            if flow:
                proxy = " <i>(proxy)</i>" if flow.get("cvd_is_proxy") else ""
                lines.append(
                    f"  {flow.get('state', '-')} score {flow.get('score', 0):+.2f}"
                    f" imbalance {_pct(flow.get('book_imbalance'))}{proxy}"
                )
            if micro:
                lines.append(
                    f"  spread {_pct(micro.get('spread_pct'), 3)}"
                    f" slip {_pct(micro.get('expected_slippage_pct'), 3)}"
                    f" {'fillable' if micro.get('fillable') else '<b>NOT fillable</b>'}"
                )
                for problem in (micro.get("problems") or [])[:1]:
                    lines.append(f"  ⚠️ {html.escape(str(problem)[:90])}")
            if deriv.get("available"):
                lines.append(
                    f"  {deriv.get('regime', '-')}"
                    f" funding {_pct(deriv.get('funding_rate'), 4)}"
                    + (" 🔥 extreme" if deriv.get("funding_extreme") else "")
                )
            lines.append("")

        lines.append(
            "<i>Without an aggressor tape, CVD is estimated from close location "
            "and volume. It is labelled a proxy wherever that is the case.</i>"
        )
        return CommandResult(
            text="\n".join(lines), keyboard=keyboards.refresh_menu("flow")
        )

    async def _cmd_liquidity(self, args: str, user_id: int) -> CommandResult:
        rows = self._call_view("flow_view", limit=6)
        if not rows:
            return CommandResult(
                text="No liquidity map captured yet - run a scan first.",
                keyboard=keyboards.back_to_menu(),
            )

        lines = ["💧 <b>LIQUIDITY MAP</b>", ""]
        for row in rows:
            liquidity = row.get("liquidity") or {}
            if not liquidity:
                continue
            up = float(liquidity.get("pull_up") or 0.0)
            down = float(liquidity.get("pull_down") or 0.0)
            lines.append(
                f"<b>{html.escape(str(row['symbol']))}</b> "
                f"↑{up:.2f} ↓{down:.2f}"
            )
            for pool in (liquidity.get("pools") or [])[:3]:
                arrow = "↑" if pool.get("side") == "above" else "↓"
                lines.append(
                    f"  {arrow} {pool.get('kind', '?')} @ {pool.get('price')} "
                    f"(strength {pool.get('strength', 0):.2f})"
                )
            lines.append("")

        lines.append(
            "<i>Liquidation levels are modelled from price and open interest, not "
            "read from the exchange. Treat them as estimates.</i>"
        )
        return CommandResult(
            text="\n".join(lines), keyboard=keyboards.refresh_menu("liquidity")
        )

    async def _cmd_regime(self, args: str, user_id: int) -> CommandResult:
        opportunities = self.engine.scanner_view(limit=15)
        if not opportunities:
            return CommandResult(
                text="No scan results yet.", keyboard=keyboards.back_to_menu()
            )
        counts: dict[str, int] = {}
        lines = ["🧭 <b>MARKET REGIME</b>", ""]
        for row in opportunities:
            regime = str(row.get("regime") or "UNKNOWN")
            counts[regime] = counts.get(regime, 0) + 1
            lines.append(
                f"<b>{html.escape(str(row.get('symbol')))}</b> {regime}"
                f" — score {float(row.get('opportunity_score') or 0):.0f}"
            )
        lines += ["", "<b>Distribution</b>"]
        for regime, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {regime}: {count}")
        return CommandResult(
            text="\n".join(lines), keyboard=keyboards.refresh_menu("regime")
        )

    async def _cmd_journal(self, args: str, user_id: int) -> CommandResult:
        entries = self._call_view("journal_view", limit=10)
        if not entries:
            return CommandResult(
                text="The decision journal is empty.",
                keyboard=keyboards.back_to_menu(),
            )

        lines = ["📓 <b>DECISION JOURNAL</b>", ""]
        for entry in entries:
            icon = "✅" if entry.get("kind") == "ENTRY" else "⛔"
            lines.append(
                f"{icon} <b>{html.escape(str(entry.get('symbol')))}</b> "
                f"{(entry.get('side') or '-').upper()} "
                f"quality {entry.get('trade_quality', 0):.0f} "
                f"EV {entry.get('expected_r', 0):+.2f}R"
            )
            detail = (entry.get("rejections") or entry.get("reasoning") or [])[:1]
            if detail:
                lines.append(f"   <i>{html.escape(str(detail[0])[:110])}</i>")
            lines.append(f"   <code>{entry.get('decision_id', '')}</code>")

        lines += [
            "",
            "<i>Every decision - including the refusals - is recorded and can be "
            "replayed with /why SYMBOL.</i>",
        ]
        return CommandResult(
            text="\n".join(lines), keyboard=keyboards.refresh_menu("journal")
        )

    async def _cmd_why(self, args: str, user_id: int) -> CommandResult:
        symbol = (args or "").strip().upper()
        if not symbol:
            return CommandResult(
                text="Usage: <code>/why BTCUSDT</code>",
                keyboard=keyboards.back_to_menu(),
            )

        verdict = self._call_view("verdict_view", symbol) or {}
        if not verdict:
            return CommandResult(
                text=f"No recent decision recorded for <b>{html.escape(symbol)}</b>.",
                keyboard=keyboards.back_to_menu(),
            )

        ensemble = verdict.get("ensemble") or {}
        quality = verdict.get("trade_quality") or {}
        ev = verdict.get("expected_value") or {}
        lines = [
            f"🔍 <b>{html.escape(symbol)}</b> — "
            f"{'APPROVED' if verdict.get('approved') else 'NOT TAKEN'}",
            "",
            f"Ensemble: {ensemble.get('signal', '-')} "
            f"({ensemble.get('model_agreement', '-')} agreeing)",
            f"Confidence: {_pct(ensemble.get('confidence'))}   "
            f"Participation: {_pct(ensemble.get('participation'))}",
            f"Quality: {quality.get('score', 0):.0f}/100 ({quality.get('grade', '-')})",
            f"Expected value: {ev.get('expected_r', 0):+.3f}R "
            f"at P(win) {_pct(ev.get('win_probability'))}",
            f"Size influence: {_pct(verdict.get('size_multiplier'))}",
        ]

        for label, key in (
            ("Data risk", "data_risk"),
            ("Model risk", "model_risk"),
            ("Execution risk", "execution_risk"),
        ):
            report = verdict.get(key) or {}
            if report:
                lines.append(f"{label}: {report.get('score', 0):.2f}")

        vetoes = verdict.get("veto_reasons") or []
        rejections = verdict.get("signal_rejections") or []
        if vetoes or rejections:
            lines += ["", "<b>Why it was not taken</b>"]
            for reason in list(rejections)[:3]:
                lines.append(f"  ✗ signal engine: {html.escape(str(reason)[:120])}")
            for reason in vetoes[:5]:
                lines.append(f"  ✗ {html.escape(str(reason)[:130])}")
        elif verdict.get("approved"):
            lines += ["", "No objection from any layer."]

        if verdict.get("decision_id"):
            lines += ["", f"<code>{verdict['decision_id']}</code>"]
        return CommandResult(text="\n".join(lines), keyboard=keyboards.back_to_menu())

    async def _cmd_memory(self, args: str, user_id: int) -> CommandResult:
        data = self._intelligence()
        if not data.get("enabled"):
            return CommandResult(
                text="The intelligence layer is disabled.",
                keyboard=keyboards.back_to_menu(),
            )

        memory = data.get("memory", {})
        journal = data.get("journal", {})
        selectivity = journal.get("selectivity")
        lines = [
            "🧠 <b>MARKET MEMORY</b>",
            "",
            f"States remembered: {memory.get('states', 0)}",
            f"Resolved (outcome known): {memory.get('resolved', 0)}",
            f"Awaiting their horizon: {memory.get('pending', 0)}",
            f"Symbols: {memory.get('symbols', 0)}",
            "",
            "<b>Decision journal</b>",
            f"Records: {journal.get('records', 0)}",
            f"Refused: {_pct(selectivity) if selectivity is not None else '-'}",
            f"Average quality of entries: {journal.get('avg_trade_quality') or '-'}",
            "",
            "<i>Only resolved states can teach anything; pending ones are held "
            "back until their outcome is known.</i>",
        ]
        return CommandResult(
            text="\n".join(lines), keyboard=keyboards.refresh_menu("memory")
        )

    async def _cmd_learning(self, args: str, user_id: int) -> CommandResult:
        data = self._call_view("learning_view") or {"enabled": False}
        if not data.get("enabled"):
            return CommandResult(
                text=(
                    "🧠 <b>AUTONOMOUS LEARNING</b>\n\n"
                    "Disabled. Set <code>AUTO_TRAIN_ENABLED=true</code> to let the "
                    "bot retrain itself and grade each candidate against its own "
                    "realised trades."
                ),
                keyboard=keyboards.back_to_menu(),
            )

        last_run = data.get("last_run_ts") or 0
        lines = [
            "🧠 <b>AUTONOMOUS LEARNING</b>",
            "",
            f"Cycles run: {data.get('cycles', 0)}",
            f"Models promoted: {data.get('promotions', 0)}",
            f"Every: {data.get('interval_hours', 0):g}h, "
            f"after {data.get('min_new_trades', 0)} new resolved trades",
            f"Promotion margin: +{data.get('promotion_margin', 0):.2f} lift "
            f"on {data.get('min_live_samples', 0)}+ real trades",
        ]
        if last_run:
            lines.append(f"Last run: {_duration(time.time() - last_run)} ago")

        last = data.get("last_cycle")
        if last:
            verdict = "✅ PROMOTED" if last.get("promoted") else "⏸ HELD"
            lines += [
                "",
                f"<b>Last cycle — {verdict}</b>",
                f"  {last.get('rows', 0)} rows from {last.get('symbols', 0)} symbols",
            ]
            champion = last.get("champion_live_lift")
            challenger = last.get("challenger_live_lift")
            if champion is not None and challenger is not None:
                lines.append(
                    f"  on {last.get('live_samples', 0)} real trades: "
                    f"champion {champion:.2f} vs challenger {challenger:.2f}"
                )
            if last.get("reason"):
                lines.append(f"  <i>{html.escape(str(last['reason'])[:160])}</i>")

        history = data.get("history") or []
        if len(history) > 1:
            lines += ["", "<b>Recent cycles</b>"]
            for row in history[:6]:
                icon = "✅" if row.get("promoted") else "⏸"
                lift = row.get("challenger_live_lift") or row.get("challenger_lift") or 0
                lines.append(
                    f"{icon} {row.get('rows', 0)} rows, lift {float(lift):.2f}"
                )

        lines += [
            "",
            "<i>A promoted model only informs confidence. It cannot create a "
            "trade, raise a size, or change a risk limit.</i>",
        ]
        return CommandResult(
            text="\n".join(lines), keyboard=keyboards.refresh_menu("learning")
        )

    def _call_view(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call an engine view if it exists, tolerating an older engine."""

        view = getattr(self.engine, name, None)
        if not callable(view):
            return None
        try:
            return view(*args, **kwargs)
        except Exception:  # noqa: BLE001 - a broken view must not break Telegram
            return None

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
