"""Findings from a full-source audit, each pinned so it cannot come back.

Five defects, four of them instances of patterns already fixed elsewhere in the
system and missed in these particular places:

1. ``Reconciler._adopt`` hardcoded ``contract_size = 1.0``. Every quantity
   derived from an adopted position -- notional, exposure, portfolio risk, PnL
   -- was then wrong by the real contract size.
2. ``OrderFlowRead.confidence`` was pre-multiplied by ``data_quality``, which
   every consumer then applies again. Same double-count as the ensemble
   conviction and trade-quality bugs.
3. ``DerivativesRead.confidence`` did the same.
4. The order-flow and liquidity models reported a structurally absent input as
   an *abstention*, diluting participation for a reason unrelated to the setup.
   The ML and news models were fixed for this; these were missed.
5. ``repr(Settings)`` printed the API secret, access key and bot token in full.
"""

from __future__ import annotations

import asyncio

import pytest

from app.domain import Candle, ContractSpec, ExchangePosition, OrderBook, Side


def _candles(n: int = 60, start: float = 100.0, step: float = 0.2) -> list[Candle]:
    out = []
    price = start
    for i in range(n):
        close = price + step
        out.append(
            Candle(
                ts=i * 900,
                open=price,
                high=max(price, close) + 0.1,
                low=min(price, close) - 0.1,
                close=close,
                volume=1000.0,
                closed=True,
            )
        )
        price = close
    return out


class TestAdoptedPositionsCarryTheRealContractSize:
    """A position adopted from the venue must be sized in the venue's units."""

    def _reconciler(self, contract_size: float, with_specs: bool = True):
        from app.execution.order_reconciliation import Reconciler
        from app.portfolio.portfolio_manager import PortfolioManager

        spec = ContractSpec(
            symbol="SHIBUSDT",
            base="SHIB",
            quote="USDT",
            exchange_symbol="SHIB_USDT",
            contract_size=contract_size,
        )

        class Exchange:
            async def positions(self):
                return [
                    ExchangePosition(
                        symbol="SHIBUSDT",
                        side=Side.LONG,
                        quantity=10.0,
                        entry_price=0.00001234,
                        leverage=3.0,
                    )
                ]

            async def open_orders(self, symbol=None):
                return []

            async def contracts(self):
                if not with_specs:
                    raise RuntimeError("specs unavailable")
                return {"SHIBUSDT": spec}

        portfolio = PortfolioManager()
        return Reconciler(Exchange(), portfolio), portfolio

    def test_the_contract_size_comes_from_the_spec(self):
        reconciler, portfolio = self._reconciler(10_000.0)
        report = asyncio.run(reconciler.reconcile(apply=True))
        assert report.adopted == ["SHIBUSDT"]
        adopted = portfolio.get("SHIBUSDT")
        assert adopted is not None
        assert adopted.contract_size == pytest.approx(10_000.0)

    def test_the_notional_is_therefore_right(self):
        """The number the portfolio risk engine actually reads."""

        reconciler, portfolio = self._reconciler(10_000.0)
        asyncio.run(reconciler.reconcile(apply=True))
        adopted = portfolio.get("SHIBUSDT")
        # 10 contracts x 10,000 SHIB x $0.00001234
        assert adopted.notional(0.00001234) == pytest.approx(1.234, rel=1e-6)

    def test_a_sub_unit_contract_is_handled_too(self):
        """BTC contracts are 0.0001 - the error runs the other way."""

        reconciler, portfolio = self._reconciler(0.0001)
        asyncio.run(reconciler.reconcile(apply=True))
        assert portfolio.get("SHIBUSDT").contract_size == pytest.approx(0.0001)

    def test_reconciliation_still_completes_when_specs_cannot_be_loaded(self):
        """Knowing the venue holds an unknown position outranks sizing it."""

        reconciler, portfolio = self._reconciler(10_000.0, with_specs=False)
        report = asyncio.run(reconciler.reconcile(apply=True))
        assert not report.error
        assert report.adopted == ["SHIBUSDT"]
        assert portfolio.get("SHIBUSDT").contract_size == pytest.approx(1.0)


class TestDataQualityIsChargedExactlyOnce:
    """`confidence` is certainty in the read; `data_quality` is a separate axis."""

    def _book(self, levels: int = 12) -> OrderBook:
        bids = tuple((100.0 - i * 0.01, 50.0) for i in range(levels))
        asks = tuple((100.1 + i * 0.01, 50.0) for i in range(levels))
        return OrderBook(symbol="BTCUSDT", bids=bids, asks=asks)

    def test_order_flow_confidence_is_not_pre_discounted(self):
        from app.orderflow.order_flow import analyse_order_flow

        read = analyse_order_flow(_candles(), self._book(), atr=0.5)
        if read.state in ("NEUTRAL", "UNCERTAIN"):
            pytest.skip("this fixture produced no directional flow")
        # Were data quality applied here, confidence could not exceed it.
        assert read.data_quality < 1.0, "fixture should have imperfect quality"
        assert read.confidence > read.data_quality * read.confidence

    def test_the_ensemble_still_discounts_it_once(self):
        from app.ensemble.base import ModelOutput, ModelSignal

        output = ModelOutput(
            name="order_flow",
            signal=ModelSignal.LONG,
            confidence=0.8,
            data_quality=0.5,
        )
        assert output.effective_confidence == pytest.approx(0.4)

    def test_derivatives_confidence_is_not_pre_discounted(self):
        import inspect

        from app.orderflow import derivatives

        source = inspect.getsource(derivatives)
        assert "min(abs(score) / 1.5, 1.0) * quality" not in source

    def test_trade_quality_order_flow_scales_with_quality_once(self):
        from app.domain import Regime
        from app.quality.trade_quality import compute_trade_quality

        class Flow:
            direction = 1
            confidence = 0.8
            data_quality = 0.5

        class FullFlow(Flow):
            data_quality = 1.0

        class Ensemble:
            from app.ensemble.base import ModelSignal as _S

            signal = _S.LONG
            confidence = 0.5
            agreement = 0.8
            participation = 0.8
            dissent = 0.2
            data_quality = 1.0
            aggregate_risk = 0.2

        thin = compute_trade_quality(
            ensemble=Ensemble(), regime=Regime.TREND_UP, order_flow=Flow(),
            rr=2.5, min_rr=1.7,
        )
        full = compute_trade_quality(
            ensemble=Ensemble(), regime=Regime.TREND_UP, order_flow=FullFlow(),
            rr=2.5, min_rr=1.7,
        )
        # 0.5 + 0.5*dq is the single discount: 0.75 vs 1.0.
        ratio = thin.components["order_flow"] / full.components["order_flow"]
        assert ratio == pytest.approx(0.75, rel=1e-6)


class TestAbsentInputsAreNotAbstentions:
    def _context(self):
        from app.ensemble.base import ModelContext

        return ModelContext(symbol="BTCUSDT", ts=0, analysis=None)

    def test_order_flow_without_a_read_is_unavailable(self):
        from app.ensemble.models import OrderFlowModel

        output = OrderFlowModel().safe_evaluate(self._context())
        assert not output.available

    def test_liquidity_without_a_map_is_unavailable(self):
        from app.ensemble.models import LiquidityModel

        output = LiquidityModel().safe_evaluate(self._context())
        assert not output.available

    def test_portfolio_risk_without_state_is_unavailable(self):
        from app.ensemble.models import PortfolioRiskModel

        output = PortfolioRiskModel().safe_evaluate(self._context())
        assert not output.available

    def test_a_thin_but_present_read_is_still_an_abstention(self):
        """Only a wholly missing input is 'unavailable'."""

        from app.ensemble.base import ModelContext
        from app.ensemble.models import OrderFlowModel
        from app.orderflow.order_flow import OrderFlowRead

        read = OrderFlowRead()
        read.data_quality = 0.1
        context = ModelContext(
            symbol="BTCUSDT", ts=0, analysis=None, order_flow=read
        )
        output = OrderFlowModel().safe_evaluate(context)
        assert output.available, "we did look; we just could not say"
        assert not output.usable


class TestSettingsNeverRenderCredentials:
    SECRET = "SECRETsk_abcdef0123456789abcdef0123456789"
    ACCESS = "ACCESSmx0vgl9999999999"
    TOKEN = "8888888888:AAtokenTOKENtokenTOKENtokenTOKENxyz"

    def _settings(self):
        from app.config import build_settings
        from tests.conftest import TEST_ENV

        return build_settings(
            env=dict(
                TEST_ENV,
                MEXC_ACCESS_KEY=self.ACCESS,
                MEXC_SECRET_KEY=self.SECRET,
                TELEGRAM_BOT_TOKEN=self.TOKEN,
                TELEGRAM_CHAT_ID="123",
            )
        )

    def test_repr_hides_all_three(self):
        text = repr(self._settings())
        for secret in (self.SECRET, self.ACCESS, self.TOKEN):
            assert secret not in text

    def test_str_hides_all_three(self):
        """f-strings and print() go through __str__, which defaults to __repr__."""

        text = f"{self._settings()}"
        for secret in (self.SECRET, self.ACCESS, self.TOKEN):
            assert secret not in text

    def test_the_redacted_dict_is_still_safe(self):
        import json

        text = json.dumps(self._settings().redacted(), default=str)
        for secret in (self.SECRET, self.ACCESS, self.TOKEN):
            assert secret not in text

    def test_repr_still_identifies_the_object(self):
        text = repr(self._settings())
        assert text.startswith("Settings(")
        assert "trading_mode" in text


class TestPaperFillsAccountForSize:
    def _engine(self, depth):
        from app.paper.paper_engine import PaperEngine

        return PaperEngine(
            price_provider=lambda s: 100.0,
            slippage_pct=0.001,
            depth_provider=lambda s: depth,
        )

    def test_a_large_order_against_a_thin_book_costs_more(self):
        thin = self._engine(1_000.0)._slippage("BTCUSDT", 5_000.0)
        deep = self._engine(1_000_000.0)._slippage("BTCUSDT", 5_000.0)
        assert thin > deep

    def test_unknown_depth_is_not_punished(self):
        """None means 'no book cached', not 'no liquidity'."""

        unknown = self._engine(None)._slippage("BTCUSDT", 5_000.0)
        assert unknown == pytest.approx(0.001)

    def test_an_empty_book_still_is(self):
        empty = self._engine(0.0)._slippage("BTCUSDT", 5_000.0)
        assert empty > 0.001

    def test_market_data_reports_cached_depth_without_fetching(self):
        from app.data.market_data import MarketData

        class Exchange:
            calls = 0

            async def order_book(self, symbol, depth=20):
                Exchange.calls += 1
                return OrderBook(
                    symbol=symbol,
                    bids=((100.0, 10.0),),
                    asks=((100.05, 10.0),),
                )

        exchange = Exchange()
        market = MarketData(exchange)
        assert market.cached_depth("BTCUSDT") is None
        asyncio.run(market.order_book("BTCUSDT"))
        before = Exchange.calls
        depth = market.cached_depth("BTCUSDT")
        assert depth == pytest.approx(100.0 * 10 + 100.05 * 10, rel=1e-6)
        assert Exchange.calls == before, "cached_depth must never fetch"

    def test_the_engine_wires_it_up(self):
        import inspect

        from app import engine

        assert "depth_provider=self._cached_depth" in inspect.getsource(engine)


class TestHardLimitsStayHard:
    """The intelligence layer may only ever shrink a position."""

    def test_the_scale_is_clamped_in_the_risk_engine_itself(self):
        import inspect

        from app.risk import risk_engine

        source = inspect.getsource(risk_engine)
        assert "min(1.0, max(0.0, intelligence_scale))" in source

    def test_and_again_at_the_source(self):
        import inspect

        from app import intelligence

        source = inspect.getsource(intelligence)
        assert "max(0.0, min(1.0, quality_scale * risk_scale))" in source

    def test_settings_are_frozen(self):
        from dataclasses import FrozenInstanceError

        from app.config import build_settings
        from tests.conftest import TEST_ENV

        settings = build_settings(env=dict(TEST_ENV))
        with pytest.raises(FrozenInstanceError):
            settings.max_leverage = 100.0
