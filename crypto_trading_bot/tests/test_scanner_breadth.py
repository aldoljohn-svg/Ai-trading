"""Tests for instrument filtering, the order-book bench and the screen stage.

These three changes exist to answer one complaint: the scanner was spending its
deep-analysis budget on instruments that could never be traded -- tokenised
equities with no order book -- while genuinely interesting crypto symbols never
got looked at.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.config import ConfigError, build_settings
from app.domain import ContractSpec, OrderBook, Ticker, Timeframe
from app.data.market_data import MarketData
from app.exchange.synthetic import SyntheticExchange
from app.scanner.instruments import (
    InstrumentClass,
    base_of,
    classify_symbol,
    parse_allowed_classes,
)
from app.scanner.scanner import Scanner, ScreenResult
from app.scanner.universe import UniverseBuilder

from tests.conftest import TEST_ENV


def ticker(symbol: str, price: float = 100.0, volume: float = 50_000_000.0) -> Ticker:
    return Ticker(
        symbol=symbol,
        last=price,
        bid=price * 0.9999,
        ask=price * 1.0001,
        high_24h=price * 1.04,
        low_24h=price * 0.96,
        volume_24h=volume / price,
        quote_volume_24h=volume,
        change_24h_pct=0.02,
        ts=int(time.time()),
    )


def contract(symbol: str) -> ContractSpec:
    return ContractSpec(
        symbol=symbol,
        exchange_symbol=symbol.replace("USDT", "_USDT"),
        base=symbol.replace("USDT", ""),
        quote="USDT",
        contract_size=0.001,
        price_scale=4,
        volume_scale=0,
        min_volume=1.0,
        max_volume=1_000_000.0,
        max_leverage=20.0,
        price_unit=0.0001,
        active=True,
    )


# ===========================================================================
# instrument classification
# ===========================================================================


class TestInstrumentClassification:
    @pytest.mark.parametrize(
        "symbol",
        ["SNDKSTOCKUSDT", "SKHYNIXSTOCKUSDT", "MUSTOCKUSDT", "NVDASTOCKUSDT"],
    )
    def test_tokenised_equities_are_recognised(self, symbol):
        assert classify_symbol(symbol) is InstrumentClass.TOKENISED_EQUITY

    @pytest.mark.parametrize("symbol", ["SOXLUSDT", "TQQQUSDT", "SPXUSDT", "VIXUSDT"])
    def test_indices_and_leveraged_etfs_are_recognised(self, symbol):
        assert classify_symbol(symbol) is InstrumentClass.INDEX

    @pytest.mark.parametrize("symbol", ["XAUUSDT", "XAGUSDT", "XAUTUSDT", "PAXGUSDT"])
    def test_commodities_are_recognised(self, symbol):
        assert classify_symbol(symbol) is InstrumentClass.COMMODITY

    @pytest.mark.parametrize("symbol", ["EURUSDT", "JPYUSDT", "GBPUSDT"])
    def test_fx_is_recognised(self, symbol):
        assert classify_symbol(symbol) is InstrumentClass.FX

    @pytest.mark.parametrize("symbol", ["USDCUSDT", "DAIUSDT", "FDUSDUSDT"])
    def test_stablecoins_are_recognised(self, symbol):
        assert classify_symbol(symbol) is InstrumentClass.STABLECOIN

    @pytest.mark.parametrize(
        "symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "PEPEUSDT", "WIFUSDT"]
    )
    def test_crypto_is_crypto(self, symbol):
        assert classify_symbol(symbol) is InstrumentClass.CRYPTO

    def test_leveraged_spot_tokens_are_excluded(self):
        assert classify_symbol("BTC3LUSDT") is not InstrumentClass.CRYPTO
        assert classify_symbol("ETH5SUSDT") is not InstrumentClass.CRYPTO

    def test_an_unknown_listing_defaults_to_crypto(self):
        """A new token this module has never heard of must not be dropped."""

        assert classify_symbol("ZZZQQQFOOUSDT") is InstrumentClass.CRYPTO

    def test_base_extraction(self):
        assert base_of("BTCUSDT") == "BTC"
        assert base_of("BTC_USDT") == "BTC"
        assert base_of("USDT") == "USDT"        # do not strip to nothing

    def test_parse_allowed_classes_always_includes_crypto(self):
        assert InstrumentClass.CRYPTO in parse_allowed_classes([])
        assert InstrumentClass.CRYPTO in parse_allowed_classes(["COMMODITY"])

    def test_parse_allowed_classes_accepts_both_spellings(self):
        allowed = parse_allowed_classes(["TOKENIZED_EQUITY"])
        assert InstrumentClass.TOKENISED_EQUITY in allowed

    def test_all_disables_the_filter(self):
        assert parse_allowed_classes(["ALL"]) == frozenset(InstrumentClass)

    def test_unknown_names_are_ignored_not_fatal(self):
        assert parse_allowed_classes(["NONSENSE"]) == frozenset(
            {InstrumentClass.CRYPTO}
        )


# ===========================================================================
# universe filtering
# ===========================================================================


class TestUniverseFiltering:
    def _universe(self, **kwargs) -> UniverseBuilder:
        return UniverseBuilder(min_quote_volume=1000.0, max_spread_pct=0.01, **kwargs)

    def _market(self) -> tuple[dict, dict]:
        symbols = [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "SNDKSTOCKUSDT",
            "MUSTOCKUSDT",
            "SOXLUSDT",
            "XAUUSDT",
            "USDCUSDT",
        ]
        return (
            {s: contract(s) for s in symbols},
            {s: ticker(s) for s in symbols},
        )

    def test_non_crypto_is_excluded_by_default(self):
        contracts, tickers = self._market()
        built = self._universe().build(contracts, tickers)
        symbols = {c.symbol for c in built}
        assert symbols == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}

    def test_the_rejection_reason_is_reported(self):
        contracts, tickers = self._market()
        universe = self._universe()
        universe.build(contracts, tickers)
        reasons = universe.last_report.rejections
        assert reasons["tokenised equity"] == 2
        assert reasons["equity index or ETF"] == 1
        assert reasons["commodity"] == 1
        assert reasons["stablecoin"] == 1

    def test_classes_can_be_re_admitted(self):
        contracts, tickers = self._market()
        universe = self._universe(
            allowed_classes=parse_allowed_classes(["COMMODITY"])
        )
        symbols = {c.symbol for c in universe.build(contracts, tickers)}
        assert "XAUUSDT" in symbols
        assert "SNDKSTOCKUSDT" not in symbols

    def test_all_admits_everything(self):
        contracts, tickers = self._market()
        universe = self._universe(allowed_classes=parse_allowed_classes(["ALL"]))
        assert len(universe.build(contracts, tickers)) == 8


# ===========================================================================
# order-book bench
# ===========================================================================


class TestOrderBookBench:
    def test_a_benched_symbol_is_dropped_from_the_universe(self):
        contracts = {s: contract(s) for s in ("BTCUSDT", "ETHUSDT")}
        tickers = {s: ticker(s) for s in contracts}
        universe = UniverseBuilder(min_quote_volume=1000.0, max_spread_pct=0.01)

        assert len(universe.build(contracts, tickers)) == 2
        universe.bench("ETHUSDT", "no order book")
        symbols = {c.symbol for c in universe.build(contracts, tickers)}
        assert symbols == {"BTCUSDT"}
        assert universe.last_report.rejections["no usable order book"] == 1

    def test_the_bench_expires(self):
        universe = UniverseBuilder(bench_seconds=60.0)
        now = time.time()
        universe.bench("ETHUSDT", now=now)
        assert universe.is_benched("ETHUSDT", now=now + 30)
        assert not universe.is_benched("ETHUSDT", now=now + 61)

    def test_a_recovered_symbol_can_be_released_early(self):
        universe = UniverseBuilder()
        universe.bench("ETHUSDT")
        universe.unbench("ETHUSDT")
        assert not universe.is_benched("ETHUSDT")

    def test_benching_is_idempotent(self):
        universe = UniverseBuilder()
        universe.bench("ETHUSDT")
        universe.bench("ETHUSDT")
        assert universe.benched_symbols() == ["ETHUSDT"]

    def test_an_empty_book_benches_the_symbol(self):
        """The scanner must not pay for the same rejection every cycle."""

        class EmptyBookExchange(SyntheticExchange):
            async def order_book(self, symbol, depth=20):
                return OrderBook(symbol=symbol, ts=int(time.time()), bids=(), asks=())

        async def run():
            exchange = EmptyBookExchange()
            await exchange.connect()
            market_data = MarketData(exchange)
            universe = UniverseBuilder(
                min_quote_volume=1000.0, max_spread_pct=0.01, max_symbols=4
            )
            scanner = Scanner(market_data, universe, deep_analysis_count=2)

            candidates = await scanner.prescreen()
            analyses = await scanner.deep_analyse(candidates)
            assert analyses

            for analysis in analyses:
                assert analysis.order_book is None
                assert universe.is_benched(analysis.symbol)
            await exchange.close()

        asyncio.run(run())

    def test_a_healthy_book_does_not_bench(self):
        async def run():
            exchange = SyntheticExchange()
            await exchange.connect()
            market_data = MarketData(exchange)
            universe = UniverseBuilder(
                min_quote_volume=1000.0, max_spread_pct=0.01, max_symbols=4
            )
            scanner = Scanner(market_data, universe, deep_analysis_count=2)

            analyses = await scanner.deep_analyse(await scanner.prescreen())
            assert analyses
            assert universe.benched_symbols() == []
            for analysis in analyses:
                assert analysis.order_book is not None
            await exchange.close()

        asyncio.run(run())


# ===========================================================================
# the screen stage
# ===========================================================================


class TestScreenStage:
    def _scanner(self, **kwargs) -> tuple[SyntheticExchange, Scanner]:
        exchange = SyntheticExchange()
        market_data = MarketData(exchange)
        universe = UniverseBuilder(
            min_quote_volume=1000.0, max_spread_pct=0.01, max_symbols=50
        )
        return exchange, Scanner(market_data, universe, **kwargs)

    def test_screening_ranks_every_candidate(self):
        async def run():
            exchange, scanner = self._scanner(
                deep_analysis_count=3, screen_count=12
            )
            await exchange.connect()
            candidates = await scanner.prescreen()
            screened = await scanner.screen(candidates)

            assert len(screened) == min(12, len(candidates))
            assert all(isinstance(r, ScreenResult) for r in screened)
            # Sorted best first.
            scores = [r.score for r in screened]
            assert scores == sorted(scores, reverse=True)
            await exchange.close()

        asyncio.run(run())

    def test_screening_reorders_what_gets_deep_analysis(self):
        """The whole point: chart activity, not just 24h volume, picks the list."""

        async def run():
            exchange, scanner = self._scanner(
                deep_analysis_count=3, screen_count=12
            )
            await exchange.connect()
            candidates = await scanner.prescreen()
            prescreen_order = [c.symbol for c in candidates[:12]]
            screened = [r.symbol for r in await scanner.screen(candidates)]

            assert set(screened) == set(prescreen_order)
            await exchange.close()

        asyncio.run(run())

    def test_a_screen_failure_keeps_the_symbol_with_a_penalty(self):
        """One bad series must not remove a symbol from consideration entirely."""

        class FlakyExchange(SyntheticExchange):
            async def candles(self, symbol, timeframe, limit=500, end_ts=None):
                if symbol == "ETHUSDT" and timeframe is Timeframe.H1:
                    raise RuntimeError("simulated outage")
                return await super().candles(symbol, timeframe, limit, end_ts)

        async def run():
            exchange = FlakyExchange()
            await exchange.connect()
            market_data = MarketData(exchange)
            universe = UniverseBuilder(
                min_quote_volume=1000.0, max_spread_pct=0.01, max_symbols=50
            )
            scanner = Scanner(
                market_data, universe, deep_analysis_count=3, screen_count=12
            )
            screened = await scanner.screen(await scanner.prescreen())
            symbols = {r.symbol for r in screened}
            assert "ETHUSDT" in symbols
            eth = next(r for r in screened if r.symbol == "ETHUSDT")
            assert "unavailable" in eth.note
            await exchange.close()

        asyncio.run(run())

    def test_scan_runs_end_to_end_with_screening(self):
        async def run():
            exchange, scanner = self._scanner(
                deep_analysis_count=3, screen_count=10
            )
            await exchange.connect()
            analyses = await scanner.scan()
            assert 0 < len(analyses) <= 3
            assert scanner.last_screened
            await exchange.close()

        asyncio.run(run())

    def test_screen_count_zero_disables_the_stage(self):
        async def run():
            exchange, scanner = self._scanner(deep_analysis_count=3, screen_count=0)
            await exchange.connect()
            analyses = await scanner.scan()
            assert analyses
            assert scanner.last_screened == []
            await exchange.close()

        asyncio.run(run())


# ===========================================================================
# configuration
# ===========================================================================


class TestScannerSettings:
    def test_defaults(self, settings):
        assert settings.screen_count >= settings.deep_analysis_count
        assert settings.allowed_instrument_classes == ("CRYPTO",)
        assert settings.order_book_bench_minutes > 0

    def test_screening_fewer_than_are_analysed_is_rejected(self):
        env = dict(TEST_ENV, SCREEN_COUNT="5", DEEP_ANALYSIS_COUNT="20")
        with pytest.raises(ConfigError):
            build_settings(env=env)

    def test_deep_analysis_beyond_the_universe_is_rejected(self):
        env = dict(TEST_ENV, MAX_SYMBOLS_TO_SCAN="10", DEEP_ANALYSIS_COUNT="40")
        with pytest.raises(ConfigError):
            build_settings(env=env)

    def test_absurd_concurrency_is_rejected(self):
        env = dict(TEST_ENV, SCREEN_CONCURRENCY="500")
        with pytest.raises(ConfigError):
            build_settings(env=env)

    def test_an_unknown_screen_timeframe_is_rejected(self):
        env = dict(TEST_ENV, SCREEN_TIMEFRAME="7h")
        with pytest.raises(ConfigError):
            build_settings(env=env)

    def test_screening_can_be_disabled(self):
        settings = build_settings(env=dict(TEST_ENV, SCREEN_COUNT="0"))
        assert settings.screen_count == 0

    def test_wider_scanning_is_accepted(self):
        settings = build_settings(
            env=dict(
                TEST_ENV,
                MAX_SYMBOLS_TO_SCAN="200",
                SCREEN_COUNT="120",
                DEEP_ANALYSIS_COUNT="40",
                SCREEN_CONCURRENCY="16",
            )
        )
        assert settings.screen_count == 120
        assert settings.deep_analysis_count == 40


class TestOrderBookDiagnostics:
    """"Could not fetch" and "nobody is quoting" are different diagnoses.

    Collapsing them into one "no order book available" message made the cause
    impossible to see from the decision journal, which is where the operator
    actually looks.
    """

    def test_a_failed_fetch_reports_the_exception(self):
        from app.orderflow.microstructure import analyse_microstructure

        read = analyse_microstructure(
            None, notional=1000.0, unavailable_reason="depth fetch failed: timeout"
        )
        assert read.problems
        assert "timeout" in read.problems[0]

    def test_an_empty_book_says_nothing_is_quoting(self):
        from app.domain import OrderBook
        from app.orderflow.microstructure import analyse_microstructure

        empty = OrderBook(symbol="X", ts=0, bids=(), asks=())
        read = analyse_microstructure(empty, notional=1000.0)
        assert "quoting" in read.problems[0]

    def test_a_missing_book_with_no_reason_still_explains_itself(self):
        from app.orderflow.microstructure import analyse_microstructure

        read = analyse_microstructure(None, notional=1000.0)
        assert read.problems
        assert read.problems[0] != "no order book available"

    def test_the_scanner_records_why_the_book_was_unusable(self):
        class EmptyBookExchange(SyntheticExchange):
            async def order_book(self, symbol, depth=20):
                return OrderBook(symbol=symbol, ts=int(time.time()), bids=(), asks=())

        async def run():
            exchange = EmptyBookExchange()
            await exchange.connect()
            universe = UniverseBuilder(
                min_quote_volume=1000.0, max_spread_pct=0.01, max_symbols=4
            )
            scanner = Scanner(MarketData(exchange), universe, deep_analysis_count=2)
            analyses = await scanner.deep_analyse(await scanner.prescreen())
            await exchange.close()
            return analyses

        for analysis in asyncio.run(run()):
            assert analysis.book_problem, "the reason must be carried, not dropped"
            assert "bids" in analysis.book_problem or "asks" in analysis.book_problem

    def test_a_healthy_book_leaves_no_problem_recorded(self):
        async def run():
            exchange = SyntheticExchange()
            await exchange.connect()
            universe = UniverseBuilder(
                min_quote_volume=1000.0, max_spread_pct=0.01, max_symbols=4
            )
            scanner = Scanner(MarketData(exchange), universe, deep_analysis_count=2)
            analyses = await scanner.deep_analyse(await scanner.prescreen())
            await exchange.close()
            return analyses

        for analysis in asyncio.run(run()):
            assert analysis.book_problem == ""
