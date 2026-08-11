"""Meta-labelling: learn whether the rules are right, not what the market does.

The direction model asks the hardest possible question -- "which way will price
go from this bar?" -- at every bar in the series, including the overwhelming
majority where no setup exists and the answer is close to a coin flip.  Trained
that way on 15m crypto bars it does not beat a constant predictor, which is the
expected outcome rather than a surprise: short-horizon directional return is
close to a martingale, and averaging over thousands of structureless bars
buries whatever conditional signal exists.

Meta-labelling reframes it.  The rule engine already picks a direction.  The
model's job is the narrower, genuinely learnable question:

    given that the rules say LONG here, does this trade reach its target
    before its stop?

Two things change, and both matter:

* **The sample is filtered.**  Only bars where the rules produce a directional
  read become training rows.  Structureless bars are dropped rather than
  diluting the signal.
* **The label is binary and matches what a trade actually experiences** -- did
  the profit barrier come first -- instead of a three-way direction guess.

This also matches how the output is already consumed.  ``SignalEngine`` reads
``p_long`` when the side is LONG and ``p_short`` when it is SHORT; it never uses
both.  Training an unconditional model and then discarding half its output was
always wasteful.

The direction here is derived from :func:`app.scanner.scanner.directional_bias`
-- the same function the live scanner uses -- so the model cannot be trained on
one definition of "the rules say LONG" and then applied to another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from app.domain import Bias, Candle, Timeframe
from app.ict.ict_engine import analyse_ict
from app.indicators.indicators import compute_indicators
from app.indicators.slopes import compute_slopes
from app.logger import get_logger
from app.market_structure.structure import analyse_structure
from app.ml.dataset import LabelledSample, _atr_at
from app.rtm.rtm_engine import analyse_rtm

log = get_logger(__name__)

LABEL_LOSS = 0
LABEL_WIN = 1
META_LABEL_NAMES = {LABEL_LOSS: "LOSS", LABEL_WIN: "WIN"}

#: Feature carrying the direction the rules chose, so one model serves both.
SIDE_FEATURE = "rule_side"


#: Bars of context each analysis sees.  Capped to what the live scanner uses,
#: so the model is not trained on a longer view than it gets at inference time.
ANALYSIS_WINDOW = 400
MIN_WINDOW = 200


def analyse_window(
    history: Sequence[Candle], prefix: str = ""
) -> tuple[dict[str, float], Bias]:
    """Features and the rule direction from one pass over the window.

    Returns ``({}, Bias.NEUTRAL)`` when there is not enough history yet.  Both
    outputs come from the same analysis so they cannot describe different bars,
    and the direction is produced by
    :func:`app.scanner.scanner.directional_bias` -- the same function the live
    scanner calls.

    ``prefix`` must match what :meth:`app.scanner.scanner.SymbolAnalysis.features`
    emits for this timeframe -- ``"15m_"``, ``"1h_"`` and so on.  Getting this
    wrong is silent and fatal: :class:`~app.ml.features.FeatureSpec` fills any
    name it cannot find with the training mean, so a model trained on ``rsi``
    and asked about ``15m_rsi`` receives an all-zero vector and returns the same
    constant for every symbol, forever, with no error anywhere.
    """

    from app.scanner.scanner import directional_bias

    window = list(history[-ANALYSIS_WINDOW:])
    if len(window) < MIN_WINDOW:
        return {}, Bias.NEUTRAL

    indicators = compute_indicators(window)
    slopes = compute_slopes(indicators, [c.close for c in window])
    # These take the ATR *series*, not the latest value -- passing a scalar
    # silently yields nothing usable.
    structure = analyse_structure(window, indicators.atr)
    ict = analyse_ict(window, indicators.atr, structure)
    rtm = analyse_rtm(window, indicators.atr)

    features: dict[str, float] = {}
    for source in (
        indicators.as_features(),
        slopes.as_features(),
        structure.as_features(),
        ict.as_features(),
        rtm.as_features(),
    ):
        for key, value in source.items():
            features[f"{prefix}{key}"] = value

    return features, directional_bias(structure, ict, rtm, indicators)


@dataclass(slots=True)
class MetaStats:
    """What the filter did, so a thin dataset explains itself."""

    bars_examined: int = 0
    no_direction: int = 0
    unresolved: int = 0
    feature_failures: int = 0
    long_rows: int = 0
    short_rows: int = 0
    wins: int = 0

    @property
    def rows(self) -> int:
        return self.long_rows + self.short_rows

    @property
    def win_rate(self) -> float:
        return self.wins / self.rows if self.rows else 0.0

    @property
    def selectivity(self) -> float:
        """Share of bars the rules declined to take a view on."""

        return self.no_direction / self.bars_examined if self.bars_examined else 0.0

    def summary(self) -> str:
        return (
            f"{self.rows} rows from {self.bars_examined} bars "
            f"({self.long_rows} long / {self.short_rows} short), "
            f"base win rate {self.win_rate:.1%}; "
            f"{self.no_direction} bars had no directional read, "
            f"{self.unresolved} could not be resolved"
        )


def barrier_outcomes(
    candles: Sequence[Candle],
    atr_values: Sequence[float | None],
    horizon: int = 24,
    profit_atr: float = 2.0,
    loss_atr: float = 1.0,
) -> list[tuple[bool | None, bool | None]]:
    """For each bar, whether a long and a short would have won.

    Returns ``(long_won, short_won)`` per bar, ``None`` where the horizon runs
    past the end of the series or ATR is unavailable.

    A bar that spans both barriers is resolved **pessimistically** -- the stop
    is assumed to have been hit first.  Being optimistic here is how a model
    learns a win rate that live trading never reproduces.
    """

    size = len(candles)
    out: list[tuple[bool | None, bool | None]] = [(None, None)] * size

    for i in range(size):
        if i + horizon >= size:
            continue
        atr = _atr_at(atr_values, i)
        if atr is None or atr <= 0:
            continue

        entry = candles[i].close
        long_target = entry + profit_atr * atr
        long_stop = entry - loss_atr * atr
        short_target = entry - profit_atr * atr
        short_stop = entry + loss_atr * atr

        long_won: bool | None = None
        short_won: bool | None = None

        for j in range(i + 1, i + horizon + 1):
            candle = candles[j]
            if long_won is None:
                if candle.low <= long_stop:
                    long_won = False
                elif candle.high >= long_target:
                    long_won = True
            if short_won is None:
                if candle.high >= short_stop:
                    short_won = False
                elif candle.low <= short_target:
                    short_won = True
            if long_won is not None and short_won is not None:
                break

        # Reaching the vertical barrier without touching either horizontal one
        # is not a win: the trade tied up risk and returned nothing.
        out[i] = (
            long_won if long_won is not None else False,
            short_won if short_won is not None else False,
        )
    return out


def build_meta_dataset(
    symbol: str,
    timeframe: Timeframe,
    candles: Sequence[Candle],
    feature_fn: Callable[[list[Candle]], dict[str, float]] | None = None,
    horizon: int = 24,
    profit_atr: float = 2.0,
    loss_atr: float = 1.0,
    warmup: int = 400,
    stride: int = 4,
    stats: MetaStats | None = None,
    prefix: str = "",
) -> list[LabelledSample]:
    """Build WIN/LOSS rows for the bars where the rules take a directional view.

    Every row is built from ``candles[: i + 1]`` and is therefore structurally
    incapable of seeing the future, exactly as in the direction dataset.

    ``feature_fn`` is accepted for signature compatibility with
    :func:`app.ml.dataset.build_dataset` and is ignored: the meta builder needs
    the features and the rule direction to come from the same analysis, which
    :func:`analyse_window` guarantees.
    """

    stats = stats if stats is not None else MetaStats()
    if len(candles) <= warmup + horizon + 5:
        return []

    series = list(candles)
    indicators = compute_indicators(series)
    outcomes = barrier_outcomes(
        series, indicators.atr, horizon=horizon,
        profit_atr=profit_atr, loss_atr=loss_atr,
    )

    samples: list[LabelledSample] = []
    for i in range(warmup, len(series) - horizon, max(stride, 1)):
        stats.bars_examined += 1

        long_won, short_won = outcomes[i]
        if long_won is None or short_won is None:
            stats.unresolved += 1
            continue

        history = series[: i + 1]

        # Features and the rule direction come from ONE analysis of the window.
        # Computing them separately would double the cost of an already long
        # run, and worse, would let the direction the model is graded on drift
        # away from the features it is given.
        try:
            features, bias = analyse_window(history, prefix=prefix)
        except Exception as exc:  # noqa: BLE001 - one bad bar is not fatal
            log.debug("window analysis failed at %s[%d]: %s", symbol, i, exc)
            stats.feature_failures += 1
            continue

        if not features:
            stats.feature_failures += 1
            continue

        if bias is Bias.BULLISH:
            side = 1
            won = long_won
        elif bias is Bias.BEARISH:
            side = -1
            won = short_won
        else:
            # No directional read: the bot would not have traded this bar, so
            # it teaches nothing about whether the rules are right.
            stats.no_direction += 1
            continue

        features = dict(features)
        features[SIDE_FEATURE] = float(side)

        if side > 0:
            stats.long_rows += 1
        else:
            stats.short_rows += 1
        if won:
            stats.wins += 1

        samples.append(
            LabelledSample(
                symbol=symbol,
                timeframe=timeframe.value,
                ts=series[i].ts,
                features=features,
                label=LABEL_WIN if won else LABEL_LOSS,
                horizon=horizon,
            )
        )

    return samples


def meta_class_distribution(samples: Sequence[LabelledSample]) -> dict[str, int]:
    counts = {name: 0 for name in META_LABEL_NAMES.values()}
    for sample in samples:
        counts[META_LABEL_NAMES[sample.label]] += 1
    return counts


__all__ = [
    "LABEL_LOSS",
    "LABEL_WIN",
    "META_LABEL_NAMES",
    "SIDE_FEATURE",
    "MetaStats",
    "barrier_outcomes",
    "build_meta_dataset",
    "meta_class_distribution",
]
