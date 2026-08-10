"""Dataset construction with strict leakage control.

Labels come from the **triple-barrier method**: from each bar, walk forward and
see which barrier is hit first - an upper barrier at ``+k * ATR``, a lower
barrier at ``-k * ATR``, or the vertical barrier at ``horizon`` bars.

Three properties matter and all three are enforced here:

* **No lookahead in the features.** Features for bar ``i`` are computed from
  candles ``[0 .. i]`` only.
* **No leakage in the labels.** A bar is only labelled once its whole horizon
  lies in the past; the final ``horizon`` bars of a series are dropped.
* **No overlap between train and test.** ``time_split`` leaves an embargo gap
  of ``horizon`` bars so a training label cannot overlap a test window.

Labels: ``0`` = NO_TRADE, ``1`` = LONG, ``2`` = SHORT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from app.domain import Candle, Candles, Timeframe
from app.indicators.indicators import compute_indicators
from app.logger import get_logger

log = get_logger(__name__)

LABEL_NO_TRADE = 0
LABEL_LONG = 1
LABEL_SHORT = 2
LABEL_NAMES = {LABEL_NO_TRADE: "NO_TRADE", LABEL_LONG: "LONG", LABEL_SHORT: "SHORT"}


@dataclass(slots=True)
class LabelledSample:
    symbol: str
    timeframe: str
    ts: int
    features: dict[str, float]
    label: int
    horizon: int


def triple_barrier_labels(
    candles: Candles,
    atr_values: Sequence[float | None],
    horizon: int = 24,
    profit_atr: float = 2.0,
    loss_atr: float = 1.0,
) -> list[int | None]:
    """Label each bar by which barrier its future path touches first.

    The asymmetry (``profit_atr`` > ``loss_atr``) matters: it teaches the model
    to recognise setups that offer favourable reward:risk, not merely setups
    that drift the right way.  A bar is ``None`` when its horizon extends past
    the end of the series - those rows are dropped, never guessed.
    """

    size = len(candles)
    labels: list[int | None] = [None] * size

    for i in range(size):
        if i + horizon >= size:
            continue
        atr = _atr_at(atr_values, i)
        if atr is None or atr <= 0:
            continue

        entry = candles[i].close
        upper = entry + profit_atr * atr
        lower = entry - loss_atr * atr
        short_upper = entry + loss_atr * atr
        short_lower = entry - profit_atr * atr

        long_hit = None
        short_hit = None
        for j in range(i + 1, i + horizon + 1):
            candle = candles[j]
            if long_hit is None:
                # A bar that spans both barriers is resolved pessimistically:
                # assume the stop was hit first.  Being optimistic here is how
                # backtests end up unreproducible in live trading.
                if candle.low <= lower:
                    long_hit = False
                elif candle.high >= upper:
                    long_hit = True
            if short_hit is None:
                if candle.high >= short_upper:
                    short_hit = False
                elif candle.low <= short_lower:
                    short_hit = True
            if long_hit is not None and short_hit is not None:
                break

        if long_hit:
            labels[i] = LABEL_LONG
        elif short_hit:
            labels[i] = LABEL_SHORT
        else:
            labels[i] = LABEL_NO_TRADE

    return labels


def _atr_at(atr_values: Sequence[float | None], index: int) -> float | None:
    if index < len(atr_values) and atr_values[index] is not None:
        return atr_values[index]
    for value in reversed(atr_values[: min(index + 1, len(atr_values))]):
        if value is not None:
            return value
    return None


def build_dataset(
    symbol: str,
    timeframe: Timeframe,
    candles: Candles,
    feature_fn: Callable[[list[Candle]], dict[str, float]],
    horizon: int = 24,
    profit_atr: float = 2.0,
    loss_atr: float = 1.0,
    warmup: int = 200,
    stride: int = 1,
) -> list[LabelledSample]:
    """Walk the series bar by bar, building features from history only.

    ``feature_fn`` receives ``candles[: i + 1]`` - it is structurally incapable
    of seeing the future, which is the whole point of doing it this way rather
    than computing everything vectorised and slicing afterwards.
    """

    if len(candles) <= warmup + horizon + 5:
        return []

    indicators = compute_indicators(candles)
    labels = triple_barrier_labels(
        candles, indicators.atr, horizon=horizon, profit_atr=profit_atr, loss_atr=loss_atr
    )

    samples: list[LabelledSample] = []
    for i in range(warmup, len(candles) - horizon, max(stride, 1)):
        label = labels[i]
        if label is None:
            continue
        history = list(candles[: i + 1])
        try:
            features = feature_fn(history)
        except Exception as exc:  # noqa: BLE001 - one bad bar must not stop the run
            log.debug("feature build failed at %s[%d]: %s", symbol, i, exc)
            continue
        if not features:
            continue
        samples.append(
            LabelledSample(
                symbol=symbol,
                timeframe=timeframe.value,
                ts=candles[i].ts,
                features=features,
                label=label,
                horizon=horizon,
            )
        )
    return samples


def time_split(
    samples: Sequence[LabelledSample],
    test_fraction: float = 0.25,
    embargo: int | None = None,
) -> tuple[list[LabelledSample], list[LabelledSample]]:
    """Chronological split with an embargo gap between train and test."""

    if not samples:
        return [], []
    ordered = sorted(samples, key=lambda s: s.ts)
    split_at = int(len(ordered) * (1.0 - test_fraction))
    gap = embargo if embargo is not None else (ordered[0].horizon if ordered else 0)
    train = ordered[: max(split_at - gap, 0)]
    test = ordered[split_at:]
    return train, test


def class_distribution(samples: Iterable[LabelledSample]) -> dict[str, int]:
    counts = {name: 0 for name in LABEL_NAMES.values()}
    for sample in samples:
        counts[LABEL_NAMES[sample.label]] += 1
    return counts


def balance_weights(samples: Sequence[LabelledSample]) -> dict[int, float]:
    """Inverse-frequency weights so a rare class is not simply ignored."""

    counts: dict[int, int] = {}
    for sample in samples:
        counts[sample.label] = counts.get(sample.label, 0) + 1
    if not counts:
        return {}
    total = sum(counts.values())
    classes = len(counts)
    return {label: total / (classes * count) for label, count in counts.items()}


__all__ = [
    "LabelledSample",
    "build_dataset",
    "triple_barrier_labels",
    "time_split",
    "class_distribution",
    "balance_weights",
    "LABEL_NO_TRADE",
    "LABEL_LONG",
    "LABEL_SHORT",
    "LABEL_NAMES",
]
