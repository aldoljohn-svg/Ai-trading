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
from app.ml.meta import (  # noqa: E402
    MetaStats,
    analyse_window,
    build_meta_dataset,
    meta_class_distribution,
)
from app.ml.model_registry import ModelRegistry  # noqa: E402
from app.ml.train import train_model  # noqa: E402
from app.rtm.rtm_engine import analyse_rtm  # noqa: E402
from app.scanner.instruments import parse_allowed_classes  # noqa: E402
from app.scanner.scanner import score_activity  # noqa: E402
from app.scanner.universe import UniverseBuilder  # noqa: E402

log = get_logger("train")


def make_feature_builder(prefix: str):
    """Features for one bar, computed from ``history`` only.

    ``prefix`` must be the timeframe prefix the live scanner emits ("15m_",
    "1h_", ...).  SymbolAnalysis.features() namespaces every feature by
    timeframe, and FeatureSpec silently substitutes the training mean for any
    name it cannot find - so a model trained on bare names and asked about
    prefixed ones sees an all-zero vector and returns the same constant for
    every symbol, with no error raised anywhere.

    The window is capped so training uses the same amount of context the live
    scanner does; a model trained on 5000 bars of history would see a different
    world at inference time.
    """

    def feature_builder(history):
        features, _ = analyse_window(history, prefix=prefix)
        return features

    return feature_builder


async def _training_symbols(
    exchange,
    settings,
    count: int,
    timeframe: Timeframe,
    select: str = "activity",
    min_volume: float | None = None,
) -> list[str]:
    """Choose which symbols to learn from.

    Reuses the live universe filter, so training never sees tokenised equities
    or anything else the scanner would refuse -- a model trained on instruments
    it will never be asked about has learned the wrong market.

    ``select`` decides how the survivors are ranked:

    ``volume``
        Straight 24h quote volume.  Simple, but it returns the same
        permanently-liquid majors every time, including the ones that have done
        nothing for a week.

    ``activity`` (default)
        The same measure the scanner's screen stage uses: trend quality,
        momentum, volatility fit and range expansion, on top of the liquidity
        pre-screen.  A symbol that is liquid *and* actually moving produces far
        more resolved barrier outcomes per bar than one grinding sideways, so
        the dataset carries more decided trades and fewer timeouts.

    A liquidity floor still applies underneath either ranking -- ``activity``
    reorders symbols that already passed it, it does not admit illiquid ones.
    """

    contracts = await exchange.contracts()
    tickers = await exchange.tickers()
    universe = UniverseBuilder(
        quote_currency=settings.quote_currency,
        min_quote_volume=(
            min_volume if min_volume is not None else settings.min_24h_quote_volume
        ),
        max_spread_pct=settings.max_spread_pct,
        blacklist=settings.symbol_blacklist,
        # Screen a wider pool than we need so the ranking has something to
        # choose between rather than just accepting whatever passed.
        max_symbols=max(count * 3, count),
        allowed_classes=parse_allowed_classes(settings.allowed_instrument_classes),
    )
    candidates = universe.build(contracts, tickers)
    if not candidates:
        log.error(
            "no symbol passed the universe filter: %s", universe.last_report.summary()
        )
        return []

    if select == "volume":
        ordered = sorted(candidates, key=lambda c: c.quote_volume, reverse=True)
        return [c.symbol for c in ordered[:count]]

    log.info(
        "ranking %d liquid symbols by trading activity on %s...",
        len(candidates),
        timeframe.value,
    )
    series = await fetch_history_many(
        exchange, [c.symbol for c in candidates], timeframe, 240, concurrency=6
    )

    scored: list[tuple[float, str, str]] = []
    for candidate in candidates:
        candles = series.get(candidate.symbol)
        if not candles:
            continue
        result = score_activity(candidate, candles)
        scored.append((result.score, candidate.symbol, result.note))

    if not scored:
        log.warning("activity ranking produced nothing; falling back to volume")
        ordered = sorted(candidates, key=lambda c: c.quote_volume, reverse=True)
        return [c.symbol for c in ordered[:count]]

    scored.sort(reverse=True)
    for score, symbol, note in scored[:count]:
        log.info("  %-14s activity %5.1f  %s", symbol, score, note)
    if len(scored) > count:
        dropped = [s for _, s, _ in scored[count : count + 6]]
        log.info("  ... dropped as too quiet: %s", ", ".join(dropped))

    return [symbol for _, symbol, _ in scored[:count]]


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
    parser.add_argument(
        "--select",
        choices=["activity", "volume"],
        default="activity",
        help=(
            "how --top ranks candidates. 'activity' (default) prefers symbols "
            "that are trending and moving, not merely liquid; 'volume' ranks on "
            "24h quote volume alone."
        ),
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=None,
        help="override the 24h quote-volume floor for training symbol selection",
    )
    parser.add_argument(
        "--mode",
        choices=["meta", "direction"],
        default="meta",
        help=(
            "'meta' (default) learns whether the rule engine's chosen side "
            "reaches its target first -- a binary question, trained only on "
            "bars where the rules take a view. 'direction' learns "
            "LONG/SHORT/NO_TRADE from every bar, which is a much harder problem "
            "and on short-horizon crypto usually fails to beat its baseline."
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
            symbols = await _training_symbols(
                exchange,
                settings,
                args.top,
                timeframe,
                select=args.select,
                min_volume=args.min_volume,
            )
            log.info(
                "training on %d symbols selected by %s", len(symbols), args.select
            )
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

        # Namespace features exactly as the live scanner does for this
        # timeframe, or the model will be unusable at inference time.
        prefix = f"{timeframe.value}_"
        feature_builder = make_feature_builder(prefix)
        meta_stats = MetaStats()
        for symbol, candles in sorted(series.items()):
            log.info("building dataset for %s (%d bars)...", symbol, len(candles))
            if args.mode == "meta":
                built = build_meta_dataset(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=candles,
                    feature_fn=feature_builder,
                    horizon=args.horizon,
                    profit_atr=args.profit_atr,
                    loss_atr=args.loss_atr,
                    warmup=400,
                    stride=args.stride,
                    stats=meta_stats,
                    prefix=prefix,
                )
            else:
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
    if args.mode == "meta":
        log.info("filter: %s", meta_stats.summary())
        log.info("class distribution: %s", meta_class_distribution(samples))
        if meta_stats.win_rate < 0.05 or meta_stats.win_rate > 0.95:
            log.warning(
                "base win rate %.1%% is nearly degenerate - the barriers may be "
                "mis-scaled for this timeframe",
                meta_stats.win_rate * 100,
            )
    else:
        log.info("class distribution: %s", class_distribution(samples))

    result = train_model(
        samples,
        name=args.name or settings.ml_model_name,
        min_rows=settings.ml_min_training_rows,
        kind=args.mode,
        reward=args.profit_atr / max(args.loss_atr, 1e-9),
    )

    print()
    for key, value in result.metrics.items():
        if key not in ("reliability_long", "reliability_win"):
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
