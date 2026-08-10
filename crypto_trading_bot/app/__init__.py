"""Autonomous crypto trading bot for MEXC Futures.

The package is deliberately layered so that the *analytical and decision core*
(indicators, market structure, ICT, RTM, regime, risk, sizing, backtest, paper)
depends only on the Python standard library.  Heavy third-party libraries are
used at the boundaries (HTTP, dashboard, ML acceleration) and are always
optional at import time - see :mod:`app.compat`.

Nothing in this package guarantees profit.  Trading leveraged derivatives can
lose more than the deposited capital.  The system is built around capital
preservation and risk-adjusted performance, and "NO TRADE" is a first class,
frequently-chosen outcome.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
