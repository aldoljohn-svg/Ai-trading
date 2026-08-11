#!/usr/bin/env python3
"""Run a backtest.

    python scripts/backtest.py --symbols BTCUSDT,ETHUSDT --bars 4000
    python scripts/backtest.py --source synthetic --walk-forward --folds 4

Results are printed and stored in the ``backtest_results`` table.

Read the caveats the report prints. A backtest is an estimate of how a
strategy *would have* behaved on one sample of the past, with assumed costs.
It is not a forecast.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.backtest.engine import BacktestConfig, BacktestEngine  # noqa: E402
from app.backtest.walk_forward import walk_forward  # noqa: E402
from app.config import ConfigError, get_settings  # noqa: E402
from app.data.history import fetch_history  # noqa: E402
from app.database.database import get_database  # noqa: E402
from app.database.repositories import Repositories  # noqa: E402
from app.domain import Timeframe  # noqa: E402
from app.exchange.mexc import MexcFuturesExchange  # noqa: E402
from app.exchange.synthetic import SyntheticExchange  # noqa: E402
from app.logger import get_logger, setup_logging  # noqa: E402

log = get_logger("backtest")

CONTEXT = (Timeframe.H1, Timeframe.H4, Timeframe.D1)


async def load_data(exchange, symbols, execution_tf, limit):
    data = {}
    for symbol in symbols:
        series = {}
        for timeframe in (execution_tf, *CONTEXT):
            # Context timeframes need far fewer bars than the execution one to
            # cover the same span, and asking for 25000 daily bars would page
            # pointlessly back to before the asset existed.
            scale = execution_tf.seconds / timeframe.seconds
            wanted = max(200, int(limit * scale)) if scale < 1 else limit
            try:
                candles = await fetch_history(
                    exchange, symbol, timeframe, limit=wanted
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("could not load %s %s: %s", symbol, timeframe.value, exc)
                continue
            if candles:
                series[timeframe] = candles
        if execution_tf in series:
            data[symbol] = series
            log.info(
                "%s: %s",
                symbol,
                ", ".join(f"{tf.value}={len(c)}" for tf, c in series.items()),
            )
        else:
            log.warning("skipping %s - no %s data", symbol, execution_tf.value)
    return data


async def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest the trading strategy")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--timeframe", default="15m", choices=[t.value for t in Timeframe])
    parser.add_argument("--bars", type=int, default=4000, help="execution bars to replay")
    parser.add_argument("--limit", type=int, default=25000, help="candles to fetch")
    parser.add_argument("--equity", type=float, default=1000.0)
    parser.add_argument("--stride", type=int, default=4, help="evaluate entries every N bars")
    parser.add_argument("--source", choices=["mexc", "synthetic"], default=None)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--save", action="store_true", help="store the result in the database")
    args = parser.parse_args()

    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n\n{exc}", file=sys.stderr)
        return 2

    setup_logging(level=settings.log_level.value, secrets=settings.secret_values())

    source = args.source or settings.data_source
    if source == "synthetic":
        exchange = SyntheticExchange()
        log.warning("using SYNTHETIC data - results say nothing about real markets")
    else:
        exchange = MexcFuturesExchange(
            access_key=settings.mexc_access_key,
            secret_key=settings.mexc_secret_key,
            base_url=settings.mexc_base_url,
            quote=settings.quote_currency,
        )
        await exchange.connect()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    execution_tf = Timeframe(args.timeframe)

    try:
        data = await load_data(exchange, symbols, execution_tf, args.limit)
        if not data:
            log.error("no usable data - aborting")
            return 1

        contracts = await exchange.contracts()
        missing = [s for s in data if s not in contracts]
        for symbol in missing:
            log.warning("no contract metadata for %s - skipping", symbol)
            data.pop(symbol)
        if not data:
            return 1

        config = BacktestConfig(
            symbols=list(data),
            execution_timeframe=execution_tf,
            starting_equity=args.equity,
            max_bars=args.bars,
            signal_stride=args.stride,
            taker_fee=settings.taker_fee,
            slippage_pct=settings.slippage_pct,
            funding_interval_hours=settings.funding_interval_hours,
        )

        if args.walk_forward:
            result = walk_forward(settings, data, contracts, config, folds=args.folds)
            print()
            print(result.summary())
            return 0

        engine = BacktestEngine(settings, data, contracts, config)
        result = engine.run()
        print()
        print(result.summary())
        print()
        print(
            "NOTE: this is a simulation over one historical sample with assumed\n"
            "fees, slippage and funding. Past behaviour does not predict future\n"
            "results, and a small trade count means the statistics are noise.\n"
            "Validate with --walk-forward before trusting any configuration."
        )

        if args.save:
            database = get_database(settings.database_url, root=settings.resolve_path("."))
            Repositories(database).backtests.save(result.as_dict())
            log.info("saved as run_id=%s", result.config.run_id)
        return 0
    finally:
        await exchange.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
