#!/usr/bin/env python3
"""Drive the whole bot offline against the deterministic synthetic feed.

    python scripts/run_simulation.py --cycles 3

This exercises the real scanner, signal engine, risk engine, paper broker and
position manager with no network access at all.  It is the fastest way to see
what the bot actually does, and it is what CI runs as an integration smoke
test.

Prices are simulated. Nothing here is evidence about real markets.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import build_settings, set_settings  # noqa: E402
from app.engine import TradingEngine  # noqa: E402
from app.exchange.synthetic import SyntheticExchange  # noqa: E402
from app.logger import setup_logging  # noqa: E402


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Offline end-to-end simulation")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--equity", type=float, default=2000.0)
    parser.add_argument("--symbols", type=int, default=12)
    parser.add_argument("--db", default="sqlite:///data/simulation.db")
    args = parser.parse_args()

    settings = build_settings(env={
        "TRADING_MODE": "paper",
        "DATA_SOURCE": "synthetic",
        "DATABASE_URL": args.db,
        "MIN_24H_QUOTE_VOLUME": "1000",
        "MAX_SPREAD_PCT": "0.002",
        "DEEP_ANALYSIS_COUNT": str(args.symbols),
        "PAPER_STARTING_EQUITY": str(args.equity),
        "DASHBOARD_ENABLED": "false",
        "LOG_LEVEL": "WARNING",
    })
    set_settings(settings)
    setup_logging(level="WARNING")

    engine = TradingEngine(settings)
    await engine.build()
    # Swap in a seeded feed so runs are reproducible.
    engine.exchange = SyntheticExchange(seed=args.seed, equity=args.equity)
    engine.market_data.exchange = engine.exchange
    engine.reconciler.exchange = engine.exchange
    engine.paper.set_contracts(await engine.exchange.contracts())
    await engine.start()

    try:
        for cycle in range(1, args.cycles + 1):
            banner(f"CYCLE {cycle}")
            opportunities = await engine.scan_once()

            print(f"{'SYMBOL':10s} {'SCORE':>6s} {'TREND':9s} {'CONF':>6s} "
                  f"{'R:R':>5s} {'REGIME':16s} STATUS")
            print("-" * 78)
            for opportunity in opportunities[:12]:
                print(
                    f"{opportunity.symbol:10s} {opportunity.opportunity_score:6.1f} "
                    f"{opportunity.trend.value:9s} {opportunity.confidence:6.0%} "
                    f"{opportunity.rr:5.2f} {opportunity.regime.value:16s} "
                    f"{opportunity.status}"
                )

            best = opportunities[0] if opportunities else None
            if best is not None:
                print()
                print(f"Top candidate {best.symbol}:")
                if best.proposal.reasons:
                    print("  why:  " + " + ".join(best.proposal.reasons[:6]))
                if best.proposal.rejections:
                    print("  why not: " + best.proposal.rejections[0])

            positions = engine.positions_view()
            if positions:
                print()
                print("OPEN POSITIONS")
                for position in positions:
                    print(
                        f"  {position['symbol']:10s} {position['side']:5s} "
                        f"qty={position['quantity']:g} entry={position['entry']:.6g} "
                        f"stop={position['stop_loss']:.6g} "
                        f"pnl={position['unrealized_pnl']:+.2f} "
                        f"({position['r_multiple']:+.2f}R)"
                    )

            await engine.manage_positions()

        banner("ACCOUNT")
        account = engine.account_view()
        for key in ("equity", "balance", "available", "used_margin",
                    "unrealized_pnl", "realized_pnl_day", "open_positions", "drawdown"):
            print(f"  {key:20s} {account.get(key)}")

        banner("RISK")
        for key, value in engine.risk_view().items():
            if key != "breaker_summary":
                print(f"  {key:26s} {value}")

        banner("HEALTH")
        report = await engine.health.check()
        print(report.summary())

        banner("PAPER BROKER")
        print(f"  {engine.paper.stats()}")

        print()
        print("Simulated prices. This tells you the machinery works; it tells you")
        print("nothing about profitability in a real market.")
        return 0
    finally:
        await engine.stop("simulation complete")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
