"""Funding rate and open-interest intelligence.

The classic four-quadrant read of price against open interest:

===============  ===========================================================
Price ↑ / OI ↑   new longs - a genuine trend, but crowded as funding rises
Price ↑ / OI ↓   short covering - a rally with no new commitment behind it
Price ↓ / OI ↑   new shorts - a genuine down-trend
Price ↓ / OI ↓   long liquidation/capitulation - often exhaustion
===============  ===========================================================

Plus funding extremes (crowded positioning, a contrarian headwind) and
funding/OI divergence.

None of this is a timing signal on its own, which is why the consuming model
caps its confidence at 0.6. It is context: it tells you what the crowd has
already done, not what price will do next.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(slots=True)
class DerivativesRead:
    available: bool = False
    funding_rate: float | None = None
    open_interest: float | None = None
    oi_change_pct: float | None = None
    price_change_pct: float | None = None
    regime: str = "UNKNOWN"        # the four-quadrant label
    funding_percentile: float | None = None
    funding_extreme: bool = False
    divergence: bool = False
    squeeze_risk: str = "NONE"     # NONE | LONG_SQUEEZE | SHORT_SQUEEZE
    direction: int = 0
    confidence: float = 0.0
    risk: float = 0.5
    data_quality: float = 0.0
    reasoning: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "funding_rate": self.funding_rate,
            "open_interest": self.open_interest,
            "oi_change_pct": self.oi_change_pct,
            "price_change_pct": self.price_change_pct,
            "regime": self.regime,
            "funding_percentile": self.funding_percentile,
            "funding_extreme": self.funding_extreme,
            "divergence": self.divergence,
            "squeeze_risk": self.squeeze_risk,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "risk": round(self.risk, 4),
            "data_quality": round(self.data_quality, 4),
            "reasoning": self.reasoning,
        }


def classify_quadrant(price_change: float, oi_change: float, threshold: float = 0.002) -> str:
    """Label the price/OI quadrant, or ``FLAT`` when neither moved."""

    price_up = price_change > threshold
    price_down = price_change < -threshold
    oi_up = oi_change > threshold
    oi_down = oi_change < -threshold

    if price_up and oi_up:
        return "NEW_LONGS"
    if price_up and oi_down:
        return "SHORT_COVERING"
    if price_down and oi_up:
        return "NEW_SHORTS"
    if price_down and oi_down:
        return "LONG_LIQUIDATION"
    return "FLAT"


def percentile_of(value: float, history: Sequence[float]) -> float | None:
    """Where ``value`` sits within its own history, in ``[0, 1]``."""

    cleaned = [v for v in history if v is not None]
    if len(cleaned) < 10:
        return None
    below = sum(1 for v in cleaned if v < value)
    return below / len(cleaned)


def analyse_derivatives(
    funding_rate: float | None,
    open_interest: float | None,
    price_change_pct: float | None = None,
    oi_change_pct: float | None = None,
    funding_history: Sequence[float] | None = None,
    extreme_funding: float = 0.0008,
) -> DerivativesRead:
    """Build the derivatives read from whatever is actually available."""

    read = DerivativesRead(
        funding_rate=funding_rate,
        open_interest=open_interest,
        price_change_pct=price_change_pct,
        oi_change_pct=oi_change_pct,
    )
    reasons: list[str] = []
    quality = 0.0
    score = 0.0

    if funding_rate is None and open_interest is None:
        read.reasoning = ["no funding or open-interest data available"]
        return read

    read.available = True

    # --- funding --------------------------------------------------------
    if funding_rate is not None:
        quality += 0.5
        if funding_history:
            read.funding_percentile = percentile_of(funding_rate, funding_history)

        if abs(funding_rate) >= extreme_funding:
            read.funding_extreme = True
            # Crowded positioning is a headwind for the crowded side.
            score -= 1.0 if funding_rate > 0 else -1.0
            reasons.append(
                f"funding {funding_rate:+.4%} is extreme - "
                f"{'longs' if funding_rate > 0 else 'shorts'} are paying heavily"
            )
            read.squeeze_risk = "LONG_SQUEEZE" if funding_rate > 0 else "SHORT_SQUEEZE"
            read.risk = 0.75
        elif abs(funding_rate) >= extreme_funding * 0.4:
            score -= 0.4 if funding_rate > 0 else -0.4
            reasons.append(f"funding {funding_rate:+.4%} shows mild crowding")
        else:
            reasons.append(f"funding {funding_rate:+.4%} is neutral")

    # --- open interest quadrant ------------------------------------------
    if price_change_pct is not None and oi_change_pct is not None:
        quality += 0.5
        read.regime = classify_quadrant(price_change_pct, oi_change_pct)

        if read.regime == "NEW_LONGS":
            score += 0.8
            reasons.append("price up with rising OI - new longs, trend has commitment")
        elif read.regime == "SHORT_COVERING":
            score += 0.2
            reasons.append("price up with falling OI - short covering, not new demand")
            read.risk = max(read.risk, 0.6)
        elif read.regime == "NEW_SHORTS":
            score -= 0.8
            reasons.append("price down with rising OI - new shorts, trend has commitment")
        elif read.regime == "LONG_LIQUIDATION":
            score -= 0.2
            reasons.append("price down with falling OI - long liquidation, often exhaustion")
            read.risk = max(read.risk, 0.65)
        else:
            reasons.append("price and open interest are both flat")

        # Divergence: price making progress while positioning goes the other way.
        if price_change_pct > 0.005 and (funding_rate or 0) < -0.0002:
            read.divergence = True
            reasons.append("price rising while funding is negative - shorts are wrong-footed")
            score += 0.5
        elif price_change_pct < -0.005 and (funding_rate or 0) > 0.0002:
            read.divergence = True
            reasons.append("price falling while funding is positive - longs are wrong-footed")
            score -= 0.5

    read.data_quality = quality
    read.direction = 1 if score > 0.3 else (-1 if score < -0.3 else 0)
    read.confidence = min(abs(score) / 1.5, 1.0) * quality
    read.reasoning = reasons
    return read


__all__ = [
    "DerivativesRead",
    "analyse_derivatives",
    "classify_quadrant",
    "percentile_of",
]
