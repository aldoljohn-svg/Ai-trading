"""Position sizing.

The central rule: **size is derived from the distance to the stop, never from
leverage.**  Leverage only decides how much margin the resulting position
consumes; it does not change how much can be lost, because the loss is capped
by the stop.

.. code-block:: text

    risk_amount   = equity x risk_per_trade x regime_multiplier
    base_quantity = risk_amount / |entry - stop|
    contracts     = round_down(base_quantity / contract_size)
    notional      = contracts x contract_size x entry
    margin        = notional / leverage

Rounding to whole contracts always rounds **down**, so the realised risk is at
or below the budget, never above it.  The realised risk is then recomputed from
the rounded size and re-checked - a small contract on an expensive symbol can
otherwise quietly blow through the budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain import ContractSpec, Side, clamp


@dataclass(slots=True)
class PositionSize:
    contracts: float = 0.0
    base_quantity: float = 0.0
    notional: float = 0.0
    margin: float = 0.0
    leverage: float = 1.0
    risk_amount: float = 0.0          # requested risk budget
    actual_risk: float = 0.0          # risk after rounding to whole contracts
    risk_pct: float = 0.0             # actual risk as a fraction of equity
    liquidation_price: float = 0.0
    ok: bool = False
    reasons: list[str] = field(default_factory=list)

    def reject(self, reason: str) -> "PositionSize":
        self.reasons.append(reason)
        self.ok = False
        return self


def size_position(
    equity: float,
    available_margin: float,
    entry: float,
    stop: float,
    side: Side,
    spec: ContractSpec,
    risk_pct: float,
    leverage: float,
    max_leverage: float,
    min_leverage: float = 1.0,
    max_notional_pct_of_equity: float = 1.0,
    margin_buffer: float = 0.75,
    risk_multiplier: float = 1.0,
    fee_rate: float = 0.0006,
) -> PositionSize:
    """Compute a position size that respects every constraint simultaneously."""

    result = PositionSize()

    if equity <= 0:
        return result.reject("equity is zero or negative")
    if entry <= 0:
        return result.reject("invalid entry price")

    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return result.reject("stop distance is zero - cannot size the position")

    # Direction sanity: a long's stop must be below entry.
    if side is Side.LONG and stop >= entry:
        return result.reject("long stop must sit below the entry price")
    if side is Side.SHORT and stop <= entry:
        return result.reject("short stop must sit above the entry price")

    leverage = clamp(leverage, min_leverage, min(max_leverage, spec.max_leverage))
    result.leverage = round(leverage, 2)

    risk_amount = equity * risk_pct * max(risk_multiplier, 0.0)
    if risk_amount <= 0:
        return result.reject("risk budget for this trade is zero")
    result.risk_amount = risk_amount

    # Round-trip fees are paid out of the same budget as the stop loss.
    fee_per_unit = entry * fee_rate * 2
    effective_risk_per_unit = stop_distance + fee_per_unit
    base_quantity = risk_amount / effective_risk_per_unit

    if spec.contract_size <= 0:
        return result.reject(f"invalid contract size for {spec.symbol}")

    contracts = spec.round_volume(base_quantity / spec.contract_size)

    if contracts < spec.min_volume:
        return result.reject(
            f"risk budget ${risk_amount:.2f} only affords {contracts:g} contracts; "
            f"{spec.symbol} requires at least {spec.min_volume:g}. "
            "Either the account is too small for this symbol or the stop is too wide."
        )
    contracts = min(contracts, spec.max_volume)

    notional = contracts * spec.contract_size * entry
    margin = notional / leverage

    # --- constraint: margin must fit inside the available balance --------
    usable_margin = available_margin * margin_buffer
    if margin > usable_margin:
        # Scale down rather than refuse outright.
        affordable_notional = usable_margin * leverage
        contracts = spec.round_volume(
            affordable_notional / (spec.contract_size * entry)
        )
        if contracts < spec.min_volume:
            return result.reject(
                f"available margin ${available_margin:.2f} cannot support the "
                f"minimum {spec.min_volume:g} contracts of {spec.symbol} at "
                f"{leverage:g}x"
            )
        notional = contracts * spec.contract_size * entry
        margin = notional / leverage
        result.reasons.append("size reduced to fit available margin")

    # --- constraint: notional exposure cap -------------------------------
    max_notional = equity * max_notional_pct_of_equity * leverage
    if notional > max_notional:
        contracts = spec.round_volume(max_notional / (spec.contract_size * entry))
        if contracts < spec.min_volume:
            return result.reject(
                f"exposure cap allows only {contracts:g} contracts, below the "
                f"{spec.min_volume:g} minimum"
            )
        notional = contracts * spec.contract_size * entry
        margin = notional / leverage
        result.reasons.append("size reduced to respect the exposure cap")

    # --- realised risk after all rounding --------------------------------
    base_quantity = contracts * spec.contract_size
    actual_risk = base_quantity * effective_risk_per_unit

    if actual_risk > risk_amount * 1.05:
        # One contract is already too large for the budget.
        return result.reject(
            f"the smallest tradable size risks ${actual_risk:.2f}, which exceeds "
            f"the ${risk_amount:.2f} budget by more than 5%"
        )

    result.contracts = contracts
    result.base_quantity = base_quantity
    result.notional = notional
    result.margin = margin
    result.actual_risk = actual_risk
    result.risk_pct = actual_risk / equity if equity > 0 else 0.0
    result.liquidation_price = estimate_liquidation(entry, leverage, side)
    result.ok = True
    return result


def estimate_liquidation(
    entry: float, leverage: float, side: Side, maintenance_margin_rate: float = 0.005
) -> float:
    """Approximate isolated-margin liquidation price.

    This is an estimate for *safety checks only* - the exchange's own tiered
    maintenance-margin schedule is authoritative.  It is deliberately
    conservative (it liquidates sooner than the venue would).
    """

    if leverage <= 0 or entry <= 0:
        return 0.0
    move = (1.0 / leverage) - maintenance_margin_rate
    move = max(move, 0.0)
    return entry * (1 - move) if side is Side.LONG else entry * (1 + move)


def stop_is_safe_from_liquidation(
    entry: float,
    stop: float,
    leverage: float,
    side: Side,
    safety_factor: float = 1.5,
) -> tuple[bool, str]:
    """The stop must trigger comfortably before liquidation would.

    If liquidation sits between entry and the stop, the "risk per trade" number
    is fiction: the position dies first and takes the whole margin with it.
    """

    liquidation = estimate_liquidation(entry, leverage, side)
    if liquidation <= 0:
        return True, ""
    stop_distance = abs(entry - stop)
    liquidation_distance = abs(entry - liquidation)
    if liquidation_distance < stop_distance * safety_factor:
        return False, (
            f"estimated liquidation at {liquidation:.6g} is only "
            f"{liquidation_distance / max(stop_distance, 1e-12):.2f}x the stop "
            f"distance away at {leverage:g}x - reduce leverage"
        )
    return True, ""


def max_safe_leverage(
    entry: float,
    stop: float,
    safety_factor: float = 1.5,
    maintenance_margin_rate: float = 0.005,
    hard_cap: float = 20.0,
) -> float:
    """Highest leverage that keeps liquidation ``safety_factor`` beyond the stop."""

    if entry <= 0:
        return 1.0
    stop_pct = abs(entry - stop) / entry
    required_move = stop_pct * safety_factor + maintenance_margin_rate
    if required_move <= 0:
        return hard_cap
    return max(1.0, min(hard_cap, 1.0 / required_move))


__all__ = [
    "PositionSize",
    "size_position",
    "estimate_liquidation",
    "stop_is_safe_from_liquidation",
    "max_safe_leverage",
]
