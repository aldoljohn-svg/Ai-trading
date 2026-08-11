"""Order flow, market microstructure, liquidity and derivatives intelligence.

Everything here degrades honestly. MEXC's public API gives a depth snapshot,
klines and ticker-level funding/open-interest - it does **not** give a
tick-by-tick aggressor tape. So:

* True CVD needs per-trade aggressor flags. Where those are unavailable, this
  layer computes a **bar-derived delta proxy** and labels it as such via
  ``data_quality``, rather than presenting an approximation as the real thing.
* Every read carries ``data_quality`` in ``[0, 1]``, and the models that consume
  it discount their confidence accordingly.
* Nothing here fabricates liquidation data. If a provider is not configured,
  liquidation intelligence reports ``available=False``.
"""

from app.orderflow.derivatives import DerivativesRead, analyse_derivatives
from app.orderflow.liquidity_pools import LiquidityMap, build_liquidity_map
from app.orderflow.microstructure import MicrostructureRead, analyse_microstructure
from app.orderflow.order_flow import OrderFlowRead, analyse_order_flow

__all__ = [
    "OrderFlowRead",
    "analyse_order_flow",
    "MicrostructureRead",
    "analyse_microstructure",
    "LiquidityMap",
    "build_liquidity_map",
    "DerivativesRead",
    "analyse_derivatives",
]
