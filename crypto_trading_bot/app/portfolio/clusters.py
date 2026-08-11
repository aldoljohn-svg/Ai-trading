"""Dynamic correlation clustering and concentration risk.

Categories like "L1", "DeFi" or "meme" are **not** hard-coded. They are derived
from measured rolling correlation, because those labels stop describing reality
exactly when it matters: in a liquidation cascade every category becomes one
trade.

Clustering is agglomerative with a correlation-distance threshold, over several
rolling windows (30/60/90/180 bars). Using multiple windows catches the case
where a pair looks uncorrelated recently but is structurally linked.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from app.domain import Candle
from app.risk.portfolio_risk import pearson

#: Rolling windows, in bars.
DEFAULT_WINDOWS: tuple[int, ...] = (30, 60, 90, 180)


@dataclass(slots=True)
class CorrelationCluster:
    label: str
    members: list[str] = field(default_factory=list)
    average_correlation: float = 0.0

    @property
    def size(self) -> int:
        return len(self.members)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "members": self.members,
            "size": self.size,
            "average_correlation": round(self.average_correlation, 4),
        }


@dataclass(slots=True)
class ConcentrationReport:
    largest_symbol_share: float = 0.0
    largest_symbol: str = ""
    largest_cluster_share: float = 0.0
    largest_cluster: list[str] = field(default_factory=list)
    directional_share: float = 0.0
    direction: str = "FLAT"
    breaches: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def concentrated(self) -> bool:
        return bool(self.breaches)

    def as_dict(self) -> dict[str, Any]:
        return {
            "largest_symbol": self.largest_symbol,
            "largest_symbol_share": round(self.largest_symbol_share, 4),
            "largest_cluster": self.largest_cluster,
            "largest_cluster_share": round(self.largest_cluster_share, 4),
            "direction": self.direction,
            "directional_share": round(self.directional_share, 4),
            "concentrated": self.concentrated,
            "breaches": self.breaches,
            "warnings": self.warnings,
        }


class CorrelationClusterEngine:
    def __init__(
        self,
        windows: Sequence[int] = DEFAULT_WINDOWS,
        threshold: float = 0.7,
        default_correlation: float = 0.6,
    ) -> None:
        self.windows = list(windows)
        self.threshold = threshold
        self.default_correlation = default_correlation
        self._returns: dict[str, list[float]] = {}

    # -- data -------------------------------------------------------------

    def update(self, symbol: str, candles: Sequence[Candle]) -> None:
        longest = max(self.windows) + 1
        window = list(candles[-longest:])
        if len(window) < 20:
            return
        self._returns[symbol.upper()] = [
            (b.close - a.close) / a.close
            for a, b in zip(window, window[1:])
            if a.close > 0
        ]

    def update_many(self, series: Mapping[str, Sequence[Candle]]) -> None:
        for symbol, candles in series.items():
            self.update(symbol, candles)

    # -- correlation --------------------------------------------------------

    def correlation(self, a: str, b: str) -> float:
        """Worst-case correlation across all windows.

        Taking the maximum rather than the mean is deliberate: for risk purposes
        the relevant number is how linked two assets can be, not how linked they
        are on average.
        """

        a, b = a.upper(), b.upper()
        if a == b:
            return 1.0
        series_a = self._returns.get(a)
        series_b = self._returns.get(b)
        if not series_a or not series_b:
            return self.default_correlation

        values: list[float] = []
        for window in self.windows:
            if len(series_a) < window or len(series_b) < window:
                continue
            values.append(pearson(series_a[-window:], series_b[-window:]))
        if not values:
            n = min(len(series_a), len(series_b))
            if n < 20:
                return self.default_correlation
            values.append(pearson(series_a[-n:], series_b[-n:]))
        return max(values)

    def matrix(self, symbols: Sequence[str]) -> dict[str, dict[str, float]]:
        return {
            a: {b: round(self.correlation(a, b), 4) for b in symbols} for a in symbols
        }

    # -- clustering ----------------------------------------------------------

    def cluster(self, symbols: Sequence[str]) -> list[CorrelationCluster]:
        """Agglomerative single-linkage clustering on correlation distance."""

        remaining = [s.upper() for s in symbols]
        clusters: list[list[str]] = [[s] for s in remaining]

        merged = True
        while merged and len(clusters) > 1:
            merged = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Single linkage: merge if ANY pair across the two clusters
                    # is correlated above the threshold.
                    linked = any(
                        self.correlation(a, b) >= self.threshold
                        for a in clusters[i]
                        for b in clusters[j]
                    )
                    if linked:
                        clusters[i] = clusters[i] + clusters[j]
                        clusters.pop(j)
                        merged = True
                        break
                if merged:
                    break

        out: list[CorrelationCluster] = []
        for index, members in enumerate(clusters):
            pairs = [
                self.correlation(a, b)
                for i, a in enumerate(members)
                for b in members[i + 1 :]
            ]
            average = statistics.fmean(pairs) if pairs else 1.0
            # Name the cluster after its most-connected member rather than an
            # invented sector label.
            label = members[0]
            if len(members) > 1:
                label = max(
                    members,
                    key=lambda s: sum(self.correlation(s, o) for o in members if o != s),
                )
            out.append(
                CorrelationCluster(
                    label=label, members=sorted(members), average_correlation=average
                )
            )
        return sorted(out, key=lambda c: c.size, reverse=True)

    def cluster_of(self, symbol: str, symbols: Sequence[str]) -> CorrelationCluster | None:
        for cluster in self.cluster(symbols):
            if symbol.upper() in cluster.members:
                return cluster
        return None

    def statistics(self) -> dict[str, Any]:
        return {
            "symbols": len(self._returns),
            "windows": self.windows,
            "threshold": self.threshold,
        }


def assess_concentration(
    exposures: Mapping[str, float],
    directions: Mapping[str, int] | None = None,
    clusters: Sequence[CorrelationCluster] | None = None,
    max_symbol_share: float = 0.5,
    max_cluster_share: float = 0.65,
    max_directional_share: float = 0.85,
) -> ConcentrationReport:
    """Check whether exposure is piled into one place.

    ``exposures`` maps symbol to a positive magnitude (notional or risk).
    """

    report = ConcentrationReport()
    total = sum(abs(v) for v in exposures.values())
    if total <= 0:
        return report

    # --- single symbol ---------------------------------------------------
    largest_symbol = max(exposures.items(), key=lambda kv: abs(kv[1]))
    report.largest_symbol = largest_symbol[0]
    report.largest_symbol_share = abs(largest_symbol[1]) / total
    if report.largest_symbol_share > max_symbol_share and len(exposures) > 1:
        report.breaches.append(
            f"{report.largest_symbol} is {report.largest_symbol_share:.0%} of exposure "
            f"(limit {max_symbol_share:.0%})"
        )

    # --- cluster ------------------------------------------------------------
    if clusters:
        best_share = 0.0
        best_members: list[str] = []
        for cluster in clusters:
            share = sum(abs(exposures.get(m, 0.0)) for m in cluster.members) / total
            if share > best_share:
                best_share = share
                best_members = cluster.members
        report.largest_cluster_share = best_share
        report.largest_cluster = best_members
        if best_share > max_cluster_share and len(exposures) > 1:
            report.breaches.append(
                f"correlated cluster {', '.join(best_members)} is {best_share:.0%} "
                f"of exposure (limit {max_cluster_share:.0%})"
            )
        elif best_share > max_cluster_share * 0.8 and len(exposures) > 1:
            report.warnings.append(
                f"cluster {', '.join(best_members)} approaching the concentration limit"
            )

    # --- direction -----------------------------------------------------------
    if directions:
        long_exposure = sum(
            abs(v) for s, v in exposures.items() if directions.get(s, 0) > 0
        )
        short_exposure = sum(
            abs(v) for s, v in exposures.items() if directions.get(s, 0) < 0
        )
        directional_total = long_exposure + short_exposure
        if directional_total > 0:
            if long_exposure >= short_exposure:
                report.direction = "LONG"
                report.directional_share = long_exposure / directional_total
            else:
                report.direction = "SHORT"
                report.directional_share = short_exposure / directional_total
            if (
                report.directional_share > max_directional_share
                and len(exposures) >= 3
            ):
                report.breaches.append(
                    f"{report.directional_share:.0%} of exposure is {report.direction} - "
                    "the book is effectively one directional bet"
                )

    return report


__all__ = [
    "CorrelationClusterEngine",
    "CorrelationCluster",
    "ConcentrationReport",
    "assess_concentration",
    "DEFAULT_WINDOWS",
]
