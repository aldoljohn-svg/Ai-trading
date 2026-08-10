"""Walk-forward validation.

A single backtest over one period tells you almost nothing: parameters that
looked good were chosen with knowledge of that period.  Walk-forward splits the
history into consecutive **in-sample / out-of-sample** folds, optimises only on
in-sample data, and reports only out-of-sample results.

The honest number is the aggregate of the out-of-sample folds.  The
``efficiency`` figure (out-of-sample return divided by in-sample return) is the
overfitting detector: well below 1.0 means the parameters do not generalise.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from app.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from app.backtest.metrics import PerformanceMetrics, compute_metrics
from app.config import Settings
from app.domain import Candle, ContractSpec, Timeframe
from app.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class Fold:
    index: int
    in_sample_start: int
    in_sample_end: int
    out_start: int
    out_end: int
    in_sample: BacktestResult | None = None
    out_sample: BacktestResult | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def efficiency(self) -> float:
        """Out-of-sample return relative to in-sample - the overfit detector."""

        if not self.in_sample or not self.out_sample:
            return 0.0
        in_return = self.in_sample.metrics.total_return
        if in_return <= 0:
            return 0.0
        return self.out_sample.metrics.total_return / in_return


@dataclass(slots=True)
class WalkForwardResult:
    folds: list[Fold] = field(default_factory=list)
    combined: PerformanceMetrics | None = None
    efficiency: float = 0.0
    consistency: float = 0.0        # fraction of folds profitable out of sample

    def summary(self) -> str:
        lines = [f"Walk-forward: {len(self.folds)} folds"]
        for fold in self.folds:
            out = fold.out_sample
            if out is None:
                lines.append(f"  fold {fold.index}: no out-of-sample result")
                continue
            lines.append(
                f"  fold {fold.index}: OOS return {out.metrics.total_return:+.2%} "
                f"({out.metrics.trades} trades, PF {out.metrics.profit_factor:.2f}, "
                f"efficiency {fold.efficiency:.2f})"
            )
        lines.append("")
        lines.append(f"Walk-forward efficiency: {self.efficiency:.2f}")
        lines.append(f"Fold consistency: {self.consistency:.0%}")
        if self.combined is not None:
            lines.append("")
            lines.append("Combined out-of-sample:")
            lines.append(self.combined.summary())
        if self.efficiency < 0.4:
            lines.append("")
            lines.append(
                "⚠️ Efficiency below 0.4 - the parameters do not generalise. "
                "Do not trade this configuration."
            )
        return "\n".join(lines)


def make_folds(
    total_bars: int,
    folds: int = 4,
    in_sample_fraction: float = 0.7,
    warmup: int = 220,
) -> list[Fold]:
    """Consecutive anchored folds with no overlap between IS and OOS."""

    usable = total_bars - warmup
    if usable <= 0 or folds < 1:
        return []
    window = usable // folds
    if window < 100:
        return []

    out: list[Fold] = []
    for index in range(folds):
        start = warmup + index * window
        in_end = start + int(window * in_sample_fraction)
        out_end = start + window
        out.append(
            Fold(
                index=index,
                in_sample_start=start,
                in_sample_end=in_end,
                out_start=in_end,
                out_end=out_end,
            )
        )
    return out


def walk_forward(
    settings: Settings,
    data: Mapping[str, Mapping[Timeframe, Sequence[Candle]]],
    contracts: Mapping[str, ContractSpec],
    config: BacktestConfig,
    folds: int = 4,
    in_sample_fraction: float = 0.7,
    optimiser: Callable[[Settings, BacktestResult], Settings] | None = None,
) -> WalkForwardResult:
    """Run walk-forward validation.

    ``optimiser`` receives the in-sample result and returns the settings to use
    out of sample.  With no optimiser the same settings are used throughout,
    which still answers the important question: does performance hold up on data
    the parameters were not chosen on?
    """

    timeframe = config.execution_timeframe
    lengths = [
        len(data[symbol].get(timeframe) or [])
        for symbol in config.symbols
        if symbol in data
    ]
    if not lengths:
        return WalkForwardResult()
    total = min(lengths)

    plan = make_folds(total, folds, in_sample_fraction, config.warmup_bars)
    if not plan:
        log.warning("not enough history for %d walk-forward folds", folds)
        return WalkForwardResult()

    all_trades: list[dict[str, Any]] = []
    combined_curve: list[float] = []
    equity = config.starting_equity

    for fold in plan:
        in_config = _slice_config(config, fold.in_sample_start, fold.in_sample_end)
        try:
            in_engine = BacktestEngine(settings, _slice_data(data, timeframe, fold.in_sample_end), contracts, in_config)
            fold.in_sample = in_engine.run()
        except ValueError as exc:
            log.warning("fold %d in-sample skipped: %s", fold.index, exc)
            continue

        fold_settings = settings
        if optimiser is not None:
            try:
                fold_settings = optimiser(settings, fold.in_sample)
                fold.parameters = fold_settings.redacted()
            except Exception as exc:  # noqa: BLE001
                log.warning("optimiser failed on fold %d: %s", fold.index, exc)

        out_config = _slice_config(
            config, fold.out_start, fold.out_end, starting_equity=equity
        )
        try:
            out_engine = BacktestEngine(
                fold_settings,
                _slice_data(data, timeframe, fold.out_end),
                contracts,
                out_config,
            )
            fold.out_sample = out_engine.run()
        except ValueError as exc:
            log.warning("fold %d out-of-sample skipped: %s", fold.index, exc)
            continue

        all_trades.extend(fold.out_sample.trades)
        # Chain the equity curves so the combined result compounds.
        scale = equity / max(fold.out_sample.equity_curve[0], 1e-9) if fold.out_sample.equity_curve else 1.0
        combined_curve.extend(v * scale for v in fold.out_sample.equity_curve)
        if combined_curve:
            equity = combined_curve[-1]

    completed = [f for f in plan if f.out_sample is not None]
    efficiencies = [f.efficiency for f in completed if f.in_sample and f.in_sample.metrics.total_return > 0]
    profitable = sum(
        1 for f in completed if f.out_sample and f.out_sample.metrics.total_return > 0
    )

    combined = None
    if combined_curve:
        combined = compute_metrics(
            trades=all_trades,
            equity_curve=combined_curve,
            starting_equity=config.starting_equity,
            period_seconds=timeframe.seconds,
            risk_per_trade=settings.default_risk_per_trade,
        )

    return WalkForwardResult(
        folds=plan,
        combined=combined,
        efficiency=round(statistics.fmean(efficiencies), 4) if efficiencies else 0.0,
        consistency=round(profitable / len(completed), 4) if completed else 0.0,
    )


def _slice_config(
    config: BacktestConfig,
    start: int,
    end: int,
    starting_equity: float | None = None,
) -> BacktestConfig:
    return BacktestConfig(
        symbols=list(config.symbols),
        execution_timeframe=config.execution_timeframe,
        context_timeframes=config.context_timeframes,
        starting_equity=(
            starting_equity if starting_equity is not None else config.starting_equity
        ),
        warmup_bars=start,
        max_bars=max(end - start, 1),
        signal_stride=config.signal_stride,
        taker_fee=config.taker_fee,
        slippage_pct=config.slippage_pct,
        spread_pct=config.spread_pct,
        funding_interval_hours=config.funding_interval_hours,
        funding_rate=config.funding_rate,
        latency_bars=config.latency_bars,
        run_id=f"{config.run_id}-{start}-{end}",
    )


def _slice_data(
    data: Mapping[str, Mapping[Timeframe, Sequence[Candle]]],
    timeframe: Timeframe,
    end_index: int,
) -> dict[str, dict[Timeframe, list[Candle]]]:
    """Truncate the execution series so a fold cannot see past its window.

    Context timeframes are truncated by timestamp against the execution cutoff,
    which is what keeps higher-timeframe lookahead out of the folds too.
    """

    out: dict[str, dict[Timeframe, list[Candle]]] = {}
    for symbol, series in data.items():
        execution = list(series.get(timeframe) or [])[:end_index]
        if not execution:
            continue
        cutoff = execution[-1].ts + timeframe.seconds
        symbol_data: dict[Timeframe, list[Candle]] = {timeframe: execution}
        for other, candles in series.items():
            if other is timeframe:
                continue
            symbol_data[other] = [
                c for c in candles if c.ts + other.seconds <= cutoff
            ]
        out[symbol] = symbol_data
    return out


__all__ = ["walk_forward", "WalkForwardResult", "Fold", "make_folds"]
