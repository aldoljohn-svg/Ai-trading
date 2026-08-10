"""Universe filtering, scanning, scoring, ranking and the signal engine."""

from __future__ import annotations

import asyncio

import pytest

from app.data.market_data import MarketData
from app.data.validators import (
    detect_price_anomaly,
    validate_candles,
    validate_ticker_spread,
)
from app.domain import Bias, Candle, ContractSpec, Regime, Side, Ticker, Timeframe
from app.exchange.base import ExchangeError
from app.fundamental.macro_engine import MacroEngine
from app.fundamental.news_engine import EconomicEvent, NewsEngine, NewsItem
from app.fundamental.sentiment import FundamentalEngine
from app.scanner.ranking import rank_opportunities, status_for
from app.scanner.scanner import Scanner, combine_bias
from app.scanner.universe import UniverseBuilder
from app.signals.scoring import ScoreBreakdown, score_risk_reward
from app.signals.signal_engine import SignalEngine
from app.signals.trade_proposal import Decision


def ticker(symbol="BTCUSDT", last=100.0, spread=0.0002, volume=50_000_000, change=0.03):
    half = last * spread / 2
    return Ticker(
        symbol=symbol, last=last, bid=last - half, ask=last + half,
        quote_volume_24h=volume, volume_24h=volume / last,
        change_24h_pct=change, high_24h=last * 1.04, low_24h=last * 0.97,
    )


def contract(symbol="BTCUSDT", active=True, quote="USDT"):
    return ContractSpec(symbol, symbol, symbol.replace("USDT", ""), quote,
                        contract_size=1.0, min_volume=1, volume_scale=0, price_unit=0.01,
                        active=active)


class TestValidators:
    def test_accepts_a_clean_series(self, candles_h1):
        result = validate_candles(candles_h1, Timeframe.H1, min_length=60)
        assert result.ok, result.problems

    def test_rejects_a_short_series(self):
        result = validate_candles([], Timeframe.H1)
        assert not result.ok

    def test_rejects_negative_prices(self):
        candles = [Candle(ts=i * 3600, open=1, high=2, low=-1, close=1, volume=1)
                   for i in range(80)]
        assert not validate_candles(candles, Timeframe.H1).ok

    def test_rejects_high_below_low(self):
        candles = [Candle(ts=i * 3600, open=1, high=0.5, low=2, close=1, volume=1)
                   for i in range(80)]
        assert not validate_candles(candles, Timeframe.H1).ok

    def test_rejects_close_outside_the_range(self):
        candles = [Candle(ts=i * 3600, open=1, high=2, low=1, close=5, volume=1)
                   for i in range(80)]
        assert not validate_candles(candles, Timeframe.H1).ok

    def test_rejects_misaligned_timestamps(self):
        candles = [Candle(ts=i * 3600 + 7, open=1, high=2, low=1, close=1.5, volume=1)
                   for i in range(80)]
        result = validate_candles(candles, Timeframe.H1)
        assert not result.ok and any("aligned" in p for p in result.problems)

    def test_rejects_stale_data(self, candles_h1):
        result = validate_candles(candles_h1, Timeframe.H1,
                                  now=candles_h1[-1].ts + 100 * 3600)
        assert not result.ok and any("stale" in p for p in result.problems)

    def test_detects_missing_bars(self):
        candles = [Candle(ts=i * 3600, open=1, high=2, low=1, close=1.5, volume=1)
                   for i in range(80) if i % 5]
        result = validate_candles(candles, Timeframe.H1, now=79 * 3600 + 3600)
        assert result.gaps > 0

    def test_spread_gate(self):
        assert validate_ticker_spread(99.9, 100.1, 0.01)[0]
        assert not validate_ticker_spread(99.0, 101.0, 0.001)[0]
        assert not validate_ticker_spread(101.0, 99.0, 0.5)[0]      # crossed book

    def test_price_anomaly_detection(self):
        calm = [Candle(ts=i * 3600, open=100, high=101, low=99, close=100 + (i % 3) * 0.1,
                       volume=1) for i in range(60)]
        assert not detect_price_anomaly(calm)[0]
        spiked = calm + [Candle(ts=60 * 3600, open=100, high=200, low=100, close=180, volume=1)]
        flagged, message = detect_price_anomaly(spiked)
        assert flagged and "sigma" in message


class TestUniverse:
    def test_filters_the_obvious_rejects(self):
        builder = UniverseBuilder(min_quote_volume=1_000_000, max_spread_pct=0.001,
                                  blacklist=["BADUSDT"])
        contracts = {
            "GOODUSDT": contract("GOODUSDT"),
            "DEADUSDT": contract("DEADUSDT", active=False),
            "BADUSDT": contract("BADUSDT"),
            "THINUSDT": contract("THINUSDT"),
            "WIDEUSDT": contract("WIDEUSDT"),
            "BTCUSD": contract("BTCUSD", quote="USD"),
        }
        tickers = {
            "GOODUSDT": ticker("GOODUSDT"),
            "DEADUSDT": ticker("DEADUSDT"),
            "BADUSDT": ticker("BADUSDT"),
            "THINUSDT": ticker("THINUSDT", volume=1000),
            "WIDEUSDT": ticker("WIDEUSDT", spread=0.05),
            "BTCUSD": ticker("BTCUSD"),
        }
        symbols = {c.symbol for c in builder.build(contracts, tickers)}
        assert symbols == {"GOODUSDT"}
        reasons = builder.last_report.rejections
        assert reasons["inactive contract"] == 1
        assert reasons["blacklisted"] == 1
        assert reasons["low liquidity"] == 1
        assert reasons["wide spread"] == 1
        assert reasons["wrong quote currency"] == 1

    def test_missing_ticker_is_rejected(self):
        builder = UniverseBuilder(min_quote_volume=1000, max_spread_pct=0.01)
        assert builder.build({"XUSDT": contract("XUSDT")}, {}) == []

    def test_caps_the_symbol_count(self):
        builder = UniverseBuilder(min_quote_volume=1000, max_spread_pct=0.01, max_symbols=3)
        contracts = {f"C{i}USDT": contract(f"C{i}USDT") for i in range(10)}
        tickers = {f"C{i}USDT": ticker(f"C{i}USDT", volume=10_000_000 * (i + 1))
                   for i in range(10)}
        assert len(builder.build(contracts, tickers)) == 3

    def test_liquidity_dominates_the_prescreen(self):
        builder = UniverseBuilder(min_quote_volume=1000, max_spread_pct=0.01)
        contracts = {"BIGUSDT": contract("BIGUSDT"), "SMALLUSDT": contract("SMALLUSDT")}
        tickers = {
            "BIGUSDT": ticker("BIGUSDT", volume=500_000_000),
            "SMALLUSDT": ticker("SMALLUSDT", volume=10_000),
        }
        result = builder.build(contracts, tickers)
        assert result[0].symbol == "BIGUSDT"


class TestBiasCombination:
    def test_unanimous(self):
        bias, alignment = combine_bias([Bias.BULLISH] * 3, [3, 2, 1.5])
        assert bias is Bias.BULLISH and alignment == pytest.approx(1.0)

    def test_disagreement_is_conflict(self):
        bias, _ = combine_bias([Bias.BULLISH, Bias.BEARISH], [1, 1])
        assert bias is Bias.CONFLICT

    def test_weighting_favours_the_higher_timeframe(self):
        bias, _ = combine_bias([Bias.BEARISH, Bias.BULLISH, Bias.BULLISH], [10, 1, 1])
        assert bias is Bias.BEARISH

    def test_empty_is_neutral(self):
        assert combine_bias([])[0] is Bias.NEUTRAL


class TestScoring:
    def test_no_model_means_rules_stand_alone(self):
        breakdown = ScoreBreakdown(technical=80, structure=80, ict=80, rtm=80,
                                   momentum=80, volume=80, volatility=80,
                                   htf_alignment=80, fundamental=80, risk_reward=80,
                                   ml_probability=0.0, ml_weight=0.0)
        assert breakdown.blended() == pytest.approx(80.0, abs=0.5)

    def test_model_shifts_but_never_dominates(self):
        breakdown = ScoreBreakdown(technical=80, structure=80, ict=80, rtm=80,
                                   momentum=80, volume=80, volatility=80,
                                   htf_alignment=80, fundamental=80, risk_reward=80,
                                   ml_probability=0.0, ml_weight=0.5)
        # Even a maximally pessimistic model leaves half the rule score.
        assert breakdown.blended() == pytest.approx(40.0, abs=0.5)

    def test_reward_risk_scoring_is_monotone(self):
        assert score_risk_reward(1.0, 2.0) < score_risk_reward(2.0, 2.0)
        assert score_risk_reward(2.0, 2.0) < score_risk_reward(4.0, 2.0)
        assert score_risk_reward(0.0, 2.0) == 0.0

    def test_neutral_default(self):
        assert ScoreBreakdown().weighted() == pytest.approx(50.0)


class TestFundamentals:
    def test_unconfigured_providers_report_unknown(self, exchange):
        engine = FundamentalEngine(MacroEngine(exchange), NewsEngine())
        snapshot = asyncio.run(engine.snapshot("BTCUSDT"))
        # No macro provider and no news feed is configured, so these must be
        # explicitly UNKNOWN rather than silently defaulted.
        for field in ("cpi", "fomc", "etf_flows", "btc_dominance", "fear_greed"):
            assert field in snapshot.unknown_fields
        assert snapshot.coverage < 1.0

    def test_a_fully_unknown_snapshot_scores_exactly_neutral(self):
        """UNKNOWN must never become bullish or bearish."""

        from app.domain import Sentiment
        from app.fundamental.macro_engine import MacroSnapshot
        from app.fundamental.news_engine import NewsAssessment
        from app.fundamental.sentiment import DataPoint, FundamentalSnapshot

        snapshot = FundamentalSnapshot(
            symbol="BTCUSDT",
            macro=MacroSnapshot(btc_trend=Sentiment.UNKNOWN),
            news=NewsAssessment(sentiment=Sentiment.UNKNOWN),
            points={
                "funding_rate": DataPoint.unknown("funding_rate"),
                "open_interest": DataPoint.unknown("open_interest"),
            },
        )
        assert snapshot.score(Side.LONG) == pytest.approx(50.0)
        assert snapshot.score(Side.SHORT) == pytest.approx(50.0)
        assert not snapshot.danger

    def test_scoring_is_symmetric_between_sides(self, exchange):
        """Whatever tilt exists must be mirrored, never biased long."""

        engine = FundamentalEngine(MacroEngine(exchange), NewsEngine())
        snapshot = asyncio.run(engine.snapshot("BTCUSDT"))
        assert snapshot.score(Side.LONG) + snapshot.score(Side.SHORT) == pytest.approx(
            100.0, abs=0.01
        )

    def test_danger_headline_blocks(self):
        item = NewsItem(title="Major exchange hacked, withdrawals halted", ts=0)
        from app.domain import Sentiment

        assert item.classify() is Sentiment.DANGER

    def test_neutral_headline(self):
        from app.domain import Sentiment

        assert NewsItem(title="Bitcoin trades sideways", ts=0).classify() is Sentiment.NEUTRAL

    def test_calendar_blackout(self, tmp_path):
        import json
        import time

        now = int(time.time())
        path = tmp_path / "cal.json"
        path.write_text(json.dumps({"events": [
            {"name": "US CPI", "ts": now + 600, "impact": "high"}
        ]}), encoding="utf-8")
        engine = NewsEngine(calendar_path=path)
        blocking = engine.blocking_event(now)
        assert blocking is not None and "CPI" in blocking.name

    def test_calendar_outside_the_window_does_not_block(self, tmp_path):
        import json
        import time

        now = int(time.time())
        path = tmp_path / "cal.json"
        path.write_text(json.dumps({"events": [
            {"name": "US CPI", "ts": now + 86400, "impact": "high"}
        ]}), encoding="utf-8")
        assert NewsEngine(calendar_path=path).blocking_event(now) is None

    def test_missing_calendar_is_not_an_error(self, tmp_path):
        assert NewsEngine(calendar_path=tmp_path / "nope.json").load_calendar() == []

    def test_high_impact_inferred_from_the_name(self):
        assert EconomicEvent(name="FOMC statement", ts=0, impact="low").is_high_impact


class TestScannerPipeline:
    @pytest.fixture(scope="class")
    def scanned(self, exchange):
        market_data = MarketData(exchange)
        universe = UniverseBuilder(min_quote_volume=1000, max_spread_pct=0.002,
                                   max_symbols=50)
        fundamentals = FundamentalEngine(MacroEngine(exchange), NewsEngine())
        scanner = Scanner(market_data, universe, fundamentals, deep_analysis_count=6)
        return asyncio.run(scanner.scan())

    def test_produces_analyses(self, scanned):
        assert scanned
        for analysis in scanned:
            assert analysis.timeframes
            assert analysis.close > 0
            assert 0.0 <= analysis.alignment <= 1.0

    def test_context_timeframes_are_present(self, scanned):
        for analysis in scanned:
            assert Timeframe.H1 in analysis.timeframes or Timeframe.H4 in analysis.timeframes

    def test_features_are_finite(self, scanned):
        for analysis in scanned:
            for key, value in analysis.features().items():
                assert value == value, key

    def test_signal_engine_always_produces_a_verdict(self, scanned, settings):
        engine = SignalEngine(settings)
        for analysis in scanned:
            proposal = engine.evaluate(analysis)
            assert proposal.decision in (Decision.ENTER, Decision.NO_TRADE)
            if proposal.decision is Decision.NO_TRADE:
                assert proposal.rejections, "a rejection must always be explained"
            else:
                assert proposal.side is not None
                assert proposal.reasons
                assert proposal.rr >= settings.min_rr
                assert proposal.confidence >= settings.min_confidence

    def test_entry_geometry_is_coherent(self, scanned, settings):
        engine = SignalEngine(settings)
        for analysis in scanned:
            proposal = engine.evaluate(analysis)
            if not proposal.is_entry:
                continue
            if proposal.side is Side.LONG:
                assert proposal.stop_loss < proposal.entry < proposal.tp1 < proposal.tp2 < proposal.tp3
            else:
                assert proposal.stop_loss > proposal.entry > proposal.tp1 > proposal.tp2 > proposal.tp3

    def test_confidence_never_reaches_certainty(self, scanned, settings):
        engine = SignalEngine(settings)
        for analysis in scanned:
            assert engine.evaluate(analysis).confidence <= 0.95

    def test_ranking_puts_tradable_first(self, scanned, settings):
        engine = SignalEngine(settings)
        opportunities = rank_opportunities([(a, engine.evaluate(a)) for a in scanned])
        tradable = [i for i, o in enumerate(opportunities) if o.tradable]
        blocked = [i for i, o in enumerate(opportunities) if not o.tradable]
        if tradable and blocked:
            assert max(tradable) < min(blocked)

    def test_status_labels(self):
        from app.signals.trade_proposal import TradeProposal

        entered = TradeProposal(symbol="X", side=Side.LONG, decision=Decision.ENTER)
        assert status_for(entered, 0) == "BEST SETUP"
        assert status_for(entered, 3) == "TRADABLE"
        rejected = TradeProposal(symbol="X", side=None, decision=Decision.NO_TRADE,
                                 rejections=["confidence 40% is below the 70% minimum"])
        assert status_for(rejected, 0) == "LOW CONFIDENCE"

    def test_no_trade_is_a_common_outcome(self, scanned, settings):
        """The system must be willing to sit out."""

        engine = SignalEngine(settings)
        decisions = [engine.evaluate(a).decision for a in scanned]
        assert any(d is Decision.NO_TRADE for d in decisions)
