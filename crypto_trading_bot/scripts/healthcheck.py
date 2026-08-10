#!/usr/bin/env python3
"""Check configuration and connectivity without starting the bot.

    python scripts/healthcheck.py

Runs the same pre-flight checks live mode gates on, so you can verify a
deployment before switching ``TRADING_MODE`` to ``live``.  Exit code 0 means
every required check passed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.compat import capabilities, missing_for_live  # noqa: E402
from app.config import ConfigError, get_settings  # noqa: E402
from app.database.database import get_database  # noqa: E402
from app.engine import TradingEngine  # noqa: E402
from app.health.preflight import run_preflight  # noqa: E402
from app.logger import setup_logging  # noqa: E402


async def main() -> int:
    try:
        settings = get_settings()
    except ConfigError as exc:
        print("🔴 CONFIGURATION INVALID\n")
        print(exc)
        return 2

    setup_logging(level="WARNING", secrets=settings.secret_values())

    print("=" * 74)
    print("CONFIGURATION")
    print("=" * 74)
    print(f"  mode            {settings.trading_mode.value.upper()}")
    print(f"  data source     {settings.data_source}")
    print(f"  database        {settings.database_url}")
    print(f"  risk per trade  {settings.default_risk_per_trade:.2%}")
    print(f"  max daily loss  {settings.max_daily_loss:.2%}")
    print(f"  max positions   {settings.max_open_positions}")
    print(f"  max leverage    {settings.max_leverage:g}x")
    print(f"  min confidence  {settings.min_confidence:.0%}")
    print(f"  min R:R         {settings.min_rr:.1f}")
    print(f"  MEXC creds      {'set' if settings.has_mexc_credentials else 'NOT SET'}")
    print(f"  Telegram        {'set' if settings.has_telegram else 'NOT SET'}")

    print()
    print("=" * 74)
    print("OPTIONAL DEPENDENCIES")
    print("=" * 74)
    for name, present in capabilities().items():
        print(f"  {'🟢' if present else '⚪'} {name}")
    missing = missing_for_live()
    if missing:
        print(f"\n  ⚠️ live trading additionally requires: {', '.join(missing)}")

    print()
    print("=" * 74)
    print("PRE-FLIGHT")
    print("=" * 74)

    engine = TradingEngine(settings)
    try:
        await engine.build()
        report = await run_preflight(
            settings=settings,
            exchange=engine.exchange,
            database=engine.database,
            notifier=None,
            reconciler=engine.reconciler,
        )
        print(report.summary())
        print()
        if settings.is_live:
            if report.passed:
                print("✅ This configuration is permitted to trade live.")
            else:
                print("🚫 LIVE MODE WOULD BE BLOCKED. Fix the 🔴 items above.")
        else:
            print(
                f"Mode is {settings.trading_mode.value.upper()}; no real orders "
                "will be placed regardless of the results above."
            )
        return 0 if report.passed else 1
    finally:
        if engine.exchange is not None:
            await engine.exchange.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
