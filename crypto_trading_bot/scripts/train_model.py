#!/usr/bin/env python3
"""Train and register the direction model.

    python scripts/train_model.py --symbols BTCUSDT,ETHUSDT,SOLUSDT

Labels come from the triple-barrier method; features are rebuilt bar by bar
from history only.  The model is calibrated on a chronologically held-out
slice and is **refused** if it fails to beat a naive baseline - an
uninformative model that ships is worse than no model, because the signal
engine would weight its noise.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import ConfigError, get_settings  # noqa: E402
from app.data.history import fetch_history_many  # noqa: E402
from app.database.database import get_database  # noqa: E402
from app.database.repositories import Repositories  # noqa: E402
from app.domain import Timeframe  # noqa: E402
from app.exchange.mexc import MexcFuturesExchange  # noqa: E402
from app.exchange.synthetic import SyntheticExchange  # noqa: E402
from app.indicators.indicators import compute_indicators  # noqa: E402
from app.indicators.slopes import compute_slopes  # noqa: E402
from app.logger import get_logger, setup_logging  # noqa: E402
from app.market_structure.structure import analyse_structure  # noqa: E402
from app.ict.ict_engine import analyse_ict  # noqa: E402
from app.ml.dataset import build_dataset, class_distribution  # noqa: E402
from app.ml.model_registry import ModelRegistry  # noqa: E402
from app.ml.train import train_model  # noqa: E402
from app.rtm.rtm_engine import analyse_rtm  # noqa: E402
from app.scanner.instruments import parse_allowed_classes  # noqa: E402
from app.scanner.universe import UniverseBuilder  # noqa: E402

log = get_logger("train")


def feature_builder(history):
    """Features for one bar, computed from ``history`` only.

    The window is capped so training uses the same amount of context the live
    scanner does - a model trained on 5000 bars of history would see a
    different world at inference time.
    """

    window = list(history[-400:])
    if len(window) < 200:
        return {}
    indicators = compute_indicators(window)
    slopes = compute_slopes(indicators, [c.close for c in window])
    structure = analyse_structure(window, indicators.atr)
    ict = analyse_ict(window, indicators.atr, structure)
    rtm = analyse_rtm(window, indicators.atr)

    features: dict[str, float] = {}
    features.update(indicators.as_features())
    features.update(slopes.as_features())
    features.update(structure.as_features())
    features.update(ict.as_features())
    features.update(rtm.as_features())
    return features


async def _liquid_symbols(exchange, settings, count: int) -> list[str]:
    """The ``count`` most liquid contracts the scanner would actually consider.

    Reuses the live universe filter so training and inference see the same kind
    of instrument.  A model trained on tokenised equities and then applied to
    perpetual crypto has learned the wrong market.
    """

    contracts = await exchange.contracts()
    tickers = await exchange.tickers()
    universe = UniverseBuilder(
        quote_currency=settings.quote_currency,
        min_quote_volume=settings.min_24h_quote_volume,
        max_spread_pct=settings.max_spread_pct,
        blacklist=settings.symbol_blacklist,
        max_symbols=max(count, 1),
        allowed_classes=parse_allowed_classes(settings.allowed_instrument_classes),
    )
    candidates = universe.build(contracts, tickers)
    ordered = sorted(candidates, key=lambda c: c.quote_volume, reverse=True)
    return [c.symbol for c in ordered[:count]]


async def main() -> int:
    parser = argparse.ArgumentParser(description="Train the direction model")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help=(
            "ignore --symbols and train on the N most liquid tradable contracts. "
            "Three symbols is far too narrow a sample to generalise from; 30-60 "
            "is a more honest basis for a model the scanner applies to hundreds."
        ),
    )
    parser.add_argument("--timeframe", default="15m", choices=[t.value for t in Timeframe])
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="parallel candle downloads (keep modest to respect rate limits)",
    )
    parser.add_argument("--horizon", type=int, default=24, help="bars to the vertical barrier")
    parser.add_argument("--profit-atr", type=float, default=2.0)
    parser.add_argument("--loss-atr", type=float, default=1.0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--source", choices=["mexc", "synthetic"], default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--dry-run", action="store_true", help="train but do not register")
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
        log.warning(
            "training on SYNTHETIC data - the resulting model has learned a "
            "simulator, not a market. Use it for plumbing checks only."
        )
    else:
        exchange = MexcFuturesExchange(
            access_key=settings.mexc_access_key,
            secret_key=settings.mexc_secret_key,
            base_url=settings.mexc_base_url,
            quote=settings.quote_currency,
        )
        await exchange.connect()

    timeframe = Timeframe(args.timeframe)
    samples = []

    try:
        if args.top:
            symbols = await _liquid_symbols(exchange, settings, args.top)
            log.info("training on the %d most liquid contracts", len(symbols))
        else:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

        # Exchanges cap one request at 2000 bars; fetch_history pages backwards
        # so --limit means what it says instead of being silently truncated.
        log.info(
            "downloading up to %d %s bars for %d symbols...",
            args.limit,
            timeframe.value,
            len(symbols),
        )
        series = await fetch_history_many(
            exchange, symbols, timeframe, args.limit, concurrency=args.concurrency
        )

        short = {s: len(c) for s, c in series.items() if len(c) < args.limit * 0.6}
        if short:
            log.warning(
                "%d symbol(s) returned much less history than requested "
                "(recent listings): %s",
                len(short),
                ", ".join(f"{s}={n}" for s, n in sorted(short.items())[:6]),
            )

        for symbol, candles in sorted(series.items()):
            log.info("building dataset for %s (%d bars)...", symbol, len(candles))
            built = build_dataset(
                symbol=symbol,
                timeframe=timeframe,
                candles=candles,
                feature_fn=feature_builder,
                horizon=args.horizon,
                profit_atr=args.profit_atr,
                loss_atr=args.loss_atr,
                warmup=400,
                stride=args.stride,
            )
            log.info("  %d labelled samples", len(built))
            samples.extend(built)
    finally:
        await exchange.close()

    if not samples:
        log.error("no samples were built - nothing to train on")
        return 1

    log.info("total samples: %d", len(samples))
    log.info("class distribution: %s", class_distribution(samples))

    result = train_model(
        samples,
        name=args.name or settings.ml_model_name,
        min_rows=settings.ml_min_training_rows,
    )

    print()
    for key, value in result.metrics.items():
        if key != "reliability_long":
            print(f"  {key:24s} {value}")
    print()

    if not result.accepted:
        print(f"MODEL REJECTED: {result.reason}")
        print("The bot continues to run on rules alone, which is the safe outcome.")
        return 1

    if args.dry_run:
        print("dry run - model not registered")
        return 0

    database = get_database(settings.database_url, root=settings.resolve_path("."))
    registry = ModelRegistry(
        settings.resolve_path(settings.models_dir),
        repository=Repositories(database).models,
    )
    path = registry.save(result.artifact, activate=True)
    print(f"MODEL ACCEPTED and activated: {path}")
    print("Restart the bot (or it will pick the model up on next start).")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
