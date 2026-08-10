"""Shared fixtures.

Every test runs offline against :class:`~app.exchange.synthetic.SyntheticExchange`
and an in-memory SQLite database, so the suite is deterministic and needs no
credentials or network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings, build_settings  # noqa: E402
from app.database.database import Database  # noqa: E402
from app.database.repositories import Repositories  # noqa: E402
from app.domain import Candle, ContractSpec, Timeframe  # noqa: E402
from app.exchange.synthetic import SyntheticExchange  # noqa: E402
from app.indicators.indicators import compute_indicators  # noqa: E402

TEST_ENV = {
    "TRADING_MODE": "paper",
    "DATA_SOURCE": "synthetic",
    "DATABASE_URL": "sqlite:///:memory:",
    "MIN_24H_QUOTE_VOLUME": "1000",
    "MAX_SPREAD_PCT": "0.002",
    "DEEP_ANALYSIS_COUNT": "6",
    "PAPER_STARTING_EQUITY": "1000",
    "DASHBOARD_ENABLED": "false",
}


@pytest.fixture
def settings() -> Settings:
    return build_settings(env=dict(TEST_ENV))


@pytest.fixture
def database() -> Database:
    db = Database("sqlite:///:memory:")
    db.migrate()
    yield db
    db.close()


@pytest.fixture
def repositories(database: Database) -> Repositories:
    return Repositories(database)


@pytest.fixture(scope="session")
def exchange() -> SyntheticExchange:
    return SyntheticExchange(seed=20240817)


@pytest.fixture(scope="session")
def candles_h1(exchange: SyntheticExchange) -> list[Candle]:
    import asyncio

    return asyncio.run(exchange.candles("BTCUSDT", Timeframe.H1, limit=400))


@pytest.fixture(scope="session")
def candles_m15(exchange: SyntheticExchange) -> list[Candle]:
    import asyncio

    return asyncio.run(exchange.candles("BTCUSDT", Timeframe.M15, limit=400))


@pytest.fixture(scope="session")
def indicators_h1(candles_h1: list[Candle]):
    return compute_indicators(candles_h1)


@pytest.fixture
def spec() -> ContractSpec:
    return ContractSpec(
        symbol="BTCUSDT",
        exchange_symbol="BTC_USDT",
        base="BTC",
        quote="USDT",
        contract_size=0.0001,
        price_scale=2,
        volume_scale=0,
        min_volume=1.0,
        max_volume=1_000_000.0,
        max_leverage=20.0,
        price_unit=0.01,
    )


def make_candles(
    prices: list[float], start_ts: int = 0, step: int = 3600, spread: float = 0.5
) -> list[Candle]:
    """Build a hand-specified series for deterministic unit tests."""

    out: list[Candle] = []
    previous = prices[0]
    for index, price in enumerate(prices):
        high = max(previous, price) + spread
        low = min(previous, price) - spread
        out.append(
            Candle(
                ts=start_ts + index * step,
                open=previous,
                high=high,
                low=max(low, 0.01),
                close=price,
                volume=100.0,
                quote_volume=100.0 * price,
            )
        )
        previous = price
    return out
