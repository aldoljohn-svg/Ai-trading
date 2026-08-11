"""Market anomaly and black-swan detection.

Every detector compares a current reading against **that instrument's own recent
distribution**, not an absolute threshold: a 4% hourly move is unremarkable for
a small-cap alt and a five-sigma event for BTC.

Severity is ``0..1``. The kill switch escalates on it; the models see it as
elevated risk.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from app.domain import Candle


class AnomalyType(str, Enum):
    PRICE_SHOCK = "PRICE_SHOCK"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    VOLATILITY_EXPLOSION = "VOLATILITY_EXPLOSION"
    SPREAD_BLOWOUT = "SPREAD_BLOWOUT"
    LIQUIDITY_COLLAPSE = "LIQUIDITY_COLLAPSE"
    FUNDING_SHOCK = "FUNDING_SHOCK"
    OI_SHOCK = "OI_SHOCK"
    CORRELATION_BREAKDOWN = "CORRELATION_BREAKDOWN"
    STALE_DATA = "STALE_DATA"
    FLASH_CRASH = "FLASH_CRASH"


@dataclass(frozen=True, slots=True)
class Anomaly:
    type: AnomalyType
    severity: float           # 0..1
    detail: str
    sigma: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "severity": round(self.severity, 4),
            "sigma": round(self.sigma, 2),
            "detail": self.detail,
        }


@dataclass(slots=True)
class AnomalyReport:
    anomalies: list[Anomaly] = field(default_factory=list)

    @property
    def severity(self) -> float:
        """The worst single anomaly, escalated slightly when several coincide.

        Simultaneous independent anomalies are what a black swan actually looks
        like from inside the data.
        """

        if not self.anomalies:
            return 0.0
        worst = max(a.severity for a in self.anomalies)
        if len(self.anomalies) >= 3:
            worst = min(1.0, worst + 0.15)
        elif len(self.anomalies) == 2:
            worst = min(1.0, worst + 0.07)
        return worst

    @property
    def black_swan(self) -> bool:
        """Multiple severe, independent anomalies at once."""

        severe = [a for a in self.anomalies if a.severity >= 0.6]
        return len(severe) >= 2 or any(a.severity >= 0.9 for a in self.anomalies)

    @property
    def types(self) -> list[str]:
        return [a.type.value for a in self.anomalies]

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": round(self.severity, 4),
            "black_swan": self.black_swan,
            "count": len(self.anomalies),
            "anomalies": [a.as_dict() for a in self.anomalies],
        }

    def summary(self) -> str:
        if not self.anomalies:
            return "no anomalies detected"
        header = "🚨 BLACK SWAN CONDITIONS" if self.black_swan else "⚠️ anomalies detected"
        lines = [f"{header} (severity {self.severity:.2f})"]
        for anomaly in sorted(self.anomalies, key=lambda a: -a.severity):
            lines.append(f"  {anomaly.type.value}: {anomaly.detail}")
        return "\n".join(lines)


def _sigma_of(value: float, history: Sequence[float]) -> tuple[float, float]:
    """Returns ``(sigma, severity)`` of ``value`` against ``history``."""

    cleaned = [v for v in history if v == v]
    if len(cleaned) < 20:
        return 0.0, 0.0
    mean = statistics.fmean(cleaned)
    stdev = statistics.pstdev(cleaned)
    if stdev <= 0:
        return 0.0, 0.0
    sigma = abs(value - mean) / stdev
    # 3 sigma is notable, 6+ sigma is a different market.
    severity = max(0.0, min((sigma - 3.0) / 5.0, 1.0))
    return sigma, severity


def detect_anomalies(
    candles: Sequence[Candle],
    spread_pct: float | None = None,
    reference_spread_pct: float | None = None,
    book_depth: float | None = None,
    reference_depth: float | None = None,
    funding_rate: float | None = None,
    funding_history: Sequence[float] | None = None,
    oi_change_pct: float | None = None,
    correlation_shift: float | None = None,
    data_age_seconds: float | None = None,
    timeframe_seconds: int = 900,
    lookback: int = 120,
) -> AnomalyReport:
    """Run every detector that has enough data to run."""

    report = AnomalyReport()

    if len(candles) >= 30:
        window = list(candles[-lookback:])

        # --- price shock -------------------------------------------------
        returns = [
            (b.close - a.close) / a.close
            for a, b in zip(window, window[1:])
            if a.close > 0
        ]
        if len(returns) >= 20:
            sigma, severity = _sigma_of(returns[-1], returns[:-1])
            if severity > 0:
                report.anomalies.append(
                    Anomaly(
                        AnomalyType.PRICE_SHOCK,
                        severity,
                        f"last bar moved {returns[-1]:+.2%}, {sigma:.1f} sigma",
                        sigma,
                    )
                )

            # --- flash crash: a large move that substantially retraced -----
            if len(window) >= 5:
                recent = window[-5:]
                low = min(c.low for c in recent)
                high = max(c.high for c in recent)
                start = recent[0].open
                if start > 0 and (high - low) / start > 0.08:
                    retrace = abs(recent[-1].close - start) / max(high - low, 1e-9)
                    if retrace < 0.4:
                        report.anomalies.append(
                            Anomaly(
                                AnomalyType.FLASH_CRASH,
                                0.85,
                                f"{(high - low) / start:.1%} range in 5 bars that "
                                f"largely retraced",
                            )
                        )

        # --- volume spike --------------------------------------------------
        volumes = [c.volume for c in window]
        if len(volumes) >= 20 and volumes[-1] > 0:
            sigma, severity = _sigma_of(volumes[-1], volumes[:-1])
            if severity > 0:
                report.anomalies.append(
                    Anomaly(
                        AnomalyType.VOLUME_SPIKE,
                        severity * 0.8,
                        f"volume {sigma:.1f} sigma above its recent distribution",
                        sigma,
                    )
                )

        # --- volatility explosion --------------------------------------------
        ranges = [(c.high - c.low) / c.close for c in window if c.close > 0]
        if len(ranges) >= 20:
            recent = statistics.fmean(ranges[-5:])
            baseline = statistics.median(ranges[:-5])
            if baseline > 0:
                ratio = recent / baseline
                if ratio >= 2.5:
                    report.anomalies.append(
                        Anomaly(
                            AnomalyType.VOLATILITY_EXPLOSION,
                            min((ratio - 2.5) / 3.0 + 0.5, 1.0),
                            f"bar ranges are {ratio:.1f}x their median",
                        )
                    )

    # --- spread blowout ---------------------------------------------------
    if spread_pct is not None and reference_spread_pct and reference_spread_pct > 0:
        ratio = spread_pct / reference_spread_pct
        if ratio >= 3.0:
            report.anomalies.append(
                Anomaly(
                    AnomalyType.SPREAD_BLOWOUT,
                    min((ratio - 3.0) / 5.0 + 0.5, 1.0),
                    f"spread is {ratio:.1f}x its usual width",
                )
            )

    # --- liquidity collapse ------------------------------------------------
    if book_depth is not None and reference_depth and reference_depth > 0:
        ratio = book_depth / reference_depth
        if ratio <= 0.35:
            report.anomalies.append(
                Anomaly(
                    AnomalyType.LIQUIDITY_COLLAPSE,
                    min((0.35 - ratio) / 0.35 + 0.45, 1.0),
                    f"visible depth is {ratio:.0%} of normal",
                )
            )

    # --- funding shock -------------------------------------------------------
    if funding_rate is not None and funding_history:
        sigma, severity = _sigma_of(funding_rate, funding_history)
        if severity > 0:
            report.anomalies.append(
                Anomaly(
                    AnomalyType.FUNDING_SHOCK,
                    severity,
                    f"funding {funding_rate:+.4%} is {sigma:.1f} sigma from normal",
                    sigma,
                )
            )

    # --- open interest shock --------------------------------------------------
    if oi_change_pct is not None and abs(oi_change_pct) >= 0.25:
        report.anomalies.append(
            Anomaly(
                AnomalyType.OI_SHOCK,
                min(abs(oi_change_pct) / 0.6, 1.0),
                f"open interest changed {oi_change_pct:+.0%} - likely a liquidation cascade",
            )
        )

    # --- correlation breakdown -------------------------------------------------
    if correlation_shift is not None and abs(correlation_shift) >= 0.5:
        report.anomalies.append(
            Anomaly(
                AnomalyType.CORRELATION_BREAKDOWN,
                min(abs(correlation_shift), 1.0),
                f"cross-asset correlation moved {correlation_shift:+.2f} - "
                "diversification assumptions no longer hold",
            )
        )

    # --- stale data -------------------------------------------------------------
    if data_age_seconds is not None and timeframe_seconds > 0:
        bars_behind = data_age_seconds / timeframe_seconds
        if bars_behind >= 4:
            report.anomalies.append(
                Anomaly(
                    AnomalyType.STALE_DATA,
                    min(bars_behind / 12.0, 1.0),
                    f"market data is {bars_behind:.1f} bars stale",
                )
            )

    return report


__all__ = ["AnomalyReport", "Anomaly", "AnomalyType", "detect_anomalies"]
