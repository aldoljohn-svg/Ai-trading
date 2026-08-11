"""Pre-flight checks for LIVE trading.

LIVE mode is blocked until every **required** check passes.  There is no
override flag, no ``--force``, and no way to enable live trading from Telegram
or the dashboard - it requires editing ``.env`` and restarting, which is a
deliberate speed bump.

Checks
------
1. configuration is valid and explicitly set to live
2. required dependencies for live trading are installed
3. MEXC REST connectivity
4. API credentials are accepted (a signed, read-only call)
5. account is readable and has a non-zero balance
6. market data is fetchable and passes validation
7. the system clock agrees with the exchange
8. the configured symbols exist and are tradable
9. the database is writable
10. Telegram is reachable (so alerts and the kill switch work)
11. existing positions have been reconciled
12. risk configuration is sane for the account size
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.compat import missing_for_live
from app.config import Settings, TradingMode
from app.domain import Timeframe
from app.logger import get_logger

log = get_logger(__name__)

MAX_CLOCK_DRIFT_SECONDS = 5.0


@dataclass(slots=True)
class PreflightCheck:
    name: str
    required: bool
    passed: bool = False
    detail: str = ""
    skipped: bool = False

    @property
    def icon(self) -> str:
        if self.skipped:
            return "⚪"
        if self.passed:
            return "🟢"
        return "🔴" if self.required else "🟡"

    def line(self) -> str:
        return f"{self.icon} {self.name}: {self.detail}"


@dataclass(slots=True)
class PreflightReport:
    checks: list[PreflightCheck] = field(default_factory=list)
    ts: int = 0

    @property
    def passed(self) -> bool:
        return all(c.passed or c.skipped for c in self.checks if c.required)

    @property
    def failures(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.required and not c.passed and not c.skipped]

    @property
    def warnings(self) -> list[PreflightCheck]:
        return [c for c in self.checks if not c.required and not c.passed and not c.skipped]

    def summary(self) -> str:
        header = (
            "✅ PRE-FLIGHT PASSED - live trading permitted"
            if self.passed
            else "🚫 PRE-FLIGHT FAILED - LIVE MODE BLOCKED"
        )
        lines = [header, ""]
        lines.extend(check.line() for check in self.checks)
        if not self.passed:
            lines.append("")
            lines.append("Fix the 🔴 items and restart. The bot stays in PAPER mode.")
        return "\n".join(lines)


async def _check(
    report: PreflightReport,
    name: str,
    required: bool,
    fn: Callable[[], Awaitable[tuple[bool, str]]],
) -> PreflightCheck:
    check = PreflightCheck(name=name, required=required)
    try:
        ok, detail = await fn()
        check.passed = ok
        check.detail = detail
    except Exception as exc:  # noqa: BLE001 - a failing check must not crash startup
        check.passed = False
        check.detail = f"{type(exc).__name__}: {exc}"
    report.checks.append(check)
    return check


async def run_preflight(
    settings: Settings,
    exchange: Any,
    database: Any = None,
    notifier: Any = None,
    reconciler: Any = None,
    probe_symbol: str = "BTCUSDT",
) -> PreflightReport:
    """Run every check.  Safe to call in any mode; only LIVE gates on it."""

    report = PreflightReport(ts=int(time.time()))

    # --- 1. configuration -------------------------------------------------
    async def check_mode() -> tuple[bool, str]:
        if settings.trading_mode is not TradingMode.LIVE:
            return True, f"mode is {settings.trading_mode.value} (live checks advisory)"
        return True, "TRADING_MODE=live is explicitly set"

    await _check(report, "trading mode", True, check_mode)

    # --- 2. dependencies --------------------------------------------------
    async def check_deps() -> tuple[bool, str]:
        missing = missing_for_live()
        if missing:
            return False, f"missing packages required for live trading: {', '.join(missing)}"
        return True, "httpx and websockets are installed"

    await _check(report, "dependencies", settings.is_live, check_deps)

    # --- 3. REST connectivity ---------------------------------------------
    async def check_rest() -> tuple[bool, str]:
        latency = await exchange.ping()
        if latency > 5000:
            return False, f"exchange responded but latency is {latency:.0f}ms"
        return True, f"reachable, {latency:.0f}ms round trip"

    await _check(report, "MEXC REST", True, check_rest)

    # --- 4. credentials ---------------------------------------------------
    async def check_credentials() -> tuple[bool, str]:
        if not settings.has_mexc_credentials:
            return False, "MEXC_ACCESS_KEY / MEXC_SECRET_KEY are not set"
        await exchange.balance("USDT")
        return True, "signed request accepted"

    await _check(report, "API credentials", settings.is_live, check_credentials)

    # --- 5. account -------------------------------------------------------
    async def check_account() -> tuple[bool, str]:
        balance = await exchange.balance("USDT")
        if balance.equity <= 0:
            return False, "account equity is zero - fund the futures wallet"
        minimum = 50.0
        if balance.equity < minimum:
            return False, (
                f"equity ${balance.equity:.2f} is below ${minimum:.0f}; position "
                "sizing cannot respect the risk budget at this size"
            )
        return True, f"equity ${balance.equity:.2f}, available ${balance.available:.2f}"

    await _check(report, "account balance", settings.is_live, check_account)

    # --- 6. market data ---------------------------------------------------
    async def check_market_data() -> tuple[bool, str]:
        from app.data.validators import validate_candles

        candles = await exchange.candles(probe_symbol, Timeframe.H1, limit=200)
        result = validate_candles(candles, Timeframe.H1, min_length=60)
        if not result.ok:
            return False, f"{probe_symbol} 1h data failed validation: {result.problems[:2]}"
        return True, f"{len(candles)} validated {probe_symbol} 1h bars"

    await _check(report, "market data", True, check_market_data)

    # --- 7. clock ---------------------------------------------------------
    async def check_clock() -> tuple[bool, str]:
        server = await exchange.server_time()
        drift = abs(server - time.time())
        if drift > MAX_CLOCK_DRIFT_SECONDS:
            return False, (
                f"system clock is {drift:.1f}s from the exchange - signed requests "
                "will be rejected. Enable NTP (e.g. `timedatectl set-ntp true`)"
            )
        return True, f"clock drift {drift:.2f}s"

    await _check(report, "clock sync", settings.is_live, check_clock)

    # --- 8. symbols -------------------------------------------------------
    async def check_symbols() -> tuple[bool, str]:
        contracts = await exchange.contracts()
        active = sum(1 for c in contracts.values() if c.active)
        if active < 5:
            return False, f"only {active} tradable contracts discovered"
        if probe_symbol not in contracts:
            return False, f"{probe_symbol} is not in the contract list"
        return True, f"{active} tradable {settings.quote_currency} contracts"

    await _check(report, "symbol validation", True, check_symbols)

    # --- 9. database ------------------------------------------------------
    async def check_database() -> tuple[bool, str]:
        if database is None:
            return False, "no database configured"
        health = database.health()
        if not health.get("ok"):
            return False, str(health.get("error", "database unreachable"))
        database.execute(
            "INSERT INTO system_events (ts, component, level, message, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            (int(time.time()), "preflight", "INFO", "write test", "{}"),
        )
        return True, f"{health['dialect']} writable, {health['latency_ms']}ms"

    await _check(report, "database", True, check_database)

    # --- 10. telegram -----------------------------------------------------
    async def check_telegram() -> tuple[bool, str]:
        if notifier is None or not settings.has_telegram:
            return False, "Telegram is not configured - alerts would be lost"
        ok = await notifier.test_connection()
        if not ok:
            return False, "could not reach the Telegram API with this token"
        return True, "bot token accepted and chat reachable"

    await _check(report, "Telegram", settings.is_live, check_telegram)

    # --- 11. reconciliation -----------------------------------------------
    async def check_reconciliation() -> tuple[bool, str]:
        if reconciler is None:
            return False, "no reconciler available"
        if not settings.is_live:
            # Outside live mode our positions are simulated and do not exist on
            # the venue, so a comparison would be meaningless (and destructive
            # if applied).
            return True, "not applicable outside live mode"
        result = await reconciler.reconcile(apply=True)
        if result.error:
            return False, f"could not read exchange state: {result.error}"
        if result.blocking:
            return False, (
                f"state mismatch on startup - {result.summary()}. Resolve it "
                "before trading live."
            )
        return True, result.summary()

    await _check(report, "position reconciliation", settings.is_live, check_reconciliation)

    # --- 12. risk configuration -------------------------------------------
    async def check_risk() -> tuple[bool, str]:
        problems: list[str] = []
        if settings.default_risk_per_trade > 0.02:
            problems.append(
                f"risk per trade {settings.default_risk_per_trade:.1%} is aggressive"
            )
        if settings.max_leverage > 5:
            problems.append(f"max leverage {settings.max_leverage:g}x is high")
        if settings.max_daily_loss > 0.05:
            problems.append(f"daily loss limit {settings.max_daily_loss:.1%} is wide")

        if settings.is_live:
            try:
                balance = await exchange.balance("USDT")
                risk_amount = balance.equity * settings.default_risk_per_trade
                if risk_amount < 1.0:
                    problems.append(
                        f"risk per trade is only ${risk_amount:.2f}; most contracts "
                        "will round to zero size"
                    )
            except Exception:  # noqa: BLE001 - covered by the account check
                pass

        if problems:
            return False, "; ".join(problems)
        return True, (
            f"{settings.default_risk_per_trade:.2%}/trade, "
            f"{settings.max_daily_loss:.1%} daily cap, "
            f"{settings.max_leverage:g}x max, "
            f"{settings.max_open_positions} positions"
        )

    await _check(report, "risk configuration", False, check_risk)

    # --- 13. intelligence layer --------------------------------------------
    async def check_intelligence() -> tuple[bool, str]:
        if not getattr(settings, "intelligence_enabled", False):
            return False, (
                "the intelligence layer is disabled - trades are gated by the "
                "signal engine alone, with no ensemble, no-trade model or "
                "trade-quality check"
            )
        return True, (
            f"quality >= {settings.min_trade_quality:.0f}/100, "
            f"EV >= {settings.min_expected_value_r:+.3f}R, "
            f"agreement >= {settings.min_model_agreement:.0%}, "
            f"no-trade threshold {settings.no_trade_threshold:.2f}"
        )

    await _check(report, "intelligence layer", False, check_intelligence)

    if settings.is_live:
        if report.passed:
            log.warning(
                "PRE-FLIGHT PASSED - LIVE TRADING IS ENABLED. Real orders will be sent."
            )
        else:
            log.error("PRE-FLIGHT FAILED - live trading blocked:\n%s", report.summary())
    return report


__all__ = ["run_preflight", "PreflightReport", "PreflightCheck"]
