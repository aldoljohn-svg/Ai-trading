"""Three defects that rejected trades the market had not actually refused.

The journal showed the pattern plainly::

    BTCUSDT  reward:risk 0.88 below the 1.70 minimum
    DOTUSDT  reward:risk 0.89 below the 1.70 minimum
    AVAXUSDT reward:risk 0.98 below the 1.70 minimum
    SHIBUSDT order of $565 is large against $433 of visible depth

Sub-1.0 reward:risk on three different symbols in one cycle is not a market
condition, it is a geometry bug; and $433 of visible depth on SHIB, one of the
deepest books on the venue, is not a liquidity condition.

1. **Depth was read in the wrong unit.** MEXC quotes order-book volumes in
   contracts, like every other quantity on its contract API. ``OrderBook``
   levels are base units because consumers compute ``price * size`` as a
   notional. Every symbol whose ``contractSize`` is not 1 had its visible depth
   understated by exactly that factor.
2. **The target ladder took the three *nearest* structural levels.** An active
   chart always has several levels close by, so the whole exit plan was
   squeezed into the first ~1R while further structure sat unused.
3. **The reward:risk gate judged TP2 alone.** The position closes in three
   parts; TP2 is one arbitrary point of the ladder.
"""

from __future__ import annotations

import pytest

from app.domain import ContractSpec, OrderBook, Side


class TestDepthIsReadInBaseUnits:
    def test_levels_are_scaled_by_the_contract_size(self):
        from app.exchange.mexc import _parse_levels

        raw = [["0.00001234", "500", 3], ["0.00001233", "800", 5]]
        # A SHIB-style contract carrying 10,000 base units per lot.
        levels = _parse_levels(raw, 10_000.0)
        assert levels[0][1] == pytest.approx(5_000_000.0)
        assert levels[1][1] == pytest.approx(8_000_000.0)

    def test_a_unit_contract_is_unchanged(self):
        from app.exchange.mexc import _parse_levels

        levels = _parse_levels([["100.0", "2.5", 1]], 1.0)
        assert levels == [(100.0, 2.5)]

    def test_the_default_does_not_silently_rescale(self):
        from app.exchange.mexc import _parse_levels

        assert _parse_levels([["100.0", "2.5"]]) == [(100.0, 2.5)]

    def test_the_notional_a_book_reports_reflects_the_contract_size(self):
        """The end-to-end quantity that the thin-liquidity check compares."""

        small = OrderBook(
            symbol="SHIBUSDT",
            bids=((0.00001234, 50_000.0),),
            asks=((0.00001235, 50_000.0),),
        )
        correct = OrderBook(
            symbol="SHIBUSDT",
            bids=((0.00001234, 50_000.0 * 10_000),),
            asks=((0.00001235, 50_000.0 * 10_000),),
        )
        # The understated book cannot absorb a $565 order; the real one can,
        # ten thousand times over.
        assert small.depth_quote("ask") < 565.0
        assert correct.depth_quote("ask") > 565.0

    def test_the_adapter_applies_it(self):
        """Regression guard on the call site, not just the helper."""

        import asyncio

        from app.exchange.mexc import MexcFuturesExchange

        exchange = MexcFuturesExchange()
        spec = ContractSpec(
            symbol="SHIBUSDT",
            base="SHIB",
            quote="USDT",
            exchange_symbol="SHIB_USDT",
            contract_size=10_000.0,
        )
        exchange._contracts = {"SHIBUSDT": spec}
        exchange._contracts_fetched_at = 9e18       # never refresh

        async def fake_public(path, params=None, cost=1.0):
            return {
                "bids": [["0.00001234", "500", 2]],
                "asks": [["0.00001235", "500", 2]],
                "timestamp": 1_700_000_000_000,
            }

        exchange._public = fake_public
        book = asyncio.run(exchange.order_book("SHIBUSDT"))
        assert book.bids[0][1] == pytest.approx(5_000_000.0)


class TestTheLadderSpansTheLevelsThatExist:
    def _levels(self, entry, distances):
        return [(entry + d, f"level at {d}") for d in distances]

    def test_three_or_fewer_levels_are_taken_as_they_are(self):
        from app.signals.signal_engine import _spread_targets

        levels = self._levels(100.0, [1.0, 2.0, 3.0])
        assert _spread_targets(levels, 100.0) == levels

    def test_the_furthest_level_is_reached_for(self):
        """The bug: with eight levels the plan stopped at the third."""

        from app.signals.signal_engine import _spread_targets

        levels = self._levels(100.0, [0.7, 0.9, 1.2, 2.0, 3.0, 4.4, 6.0, 8.0])
        chosen = _spread_targets(levels, 100.0)
        assert [price for price, _ in chosen] == [100.7, 104.4, 108.0]

    def test_it_works_for_a_short(self):
        from app.signals.signal_engine import _spread_targets

        levels = self._levels(100.0, [-0.7, -0.9, -1.2, -2.0, -3.0, -4.4, -8.0])
        chosen = _spread_targets(levels, 100.0)
        assert chosen[0][0] == pytest.approx(99.3)
        assert chosen[-1][0] == pytest.approx(92.0)

    def test_the_middle_target_sits_near_the_midpoint(self):
        from app.signals.signal_engine import _spread_targets

        levels = self._levels(100.0, [1.0, 1.1, 1.2, 5.0, 8.9, 9.0])
        chosen = _spread_targets(levels, 100.0)
        # Midpoint of 1.0 and 9.0 is 5.0, and that level exists.
        assert chosen[1][0] == pytest.approx(105.0)

    def test_it_never_invents_a_level(self):
        from app.signals.signal_engine import _spread_targets

        levels = self._levels(100.0, [0.7, 0.9, 1.2, 2.0, 3.0])
        chosen = _spread_targets(levels, 100.0)
        assert all(item in levels for item in chosen)

    def test_it_never_exceeds_three(self):
        from app.signals.signal_engine import _spread_targets

        levels = self._levels(100.0, [float(i) for i in range(1, 30)])
        assert len(_spread_targets(levels, 100.0)) == 3

    def test_an_empty_set_stays_empty(self):
        from app.signals.signal_engine import _spread_targets

        assert _spread_targets([], 100.0) == []


class TestTheGateJudgesThePlanNotOneTarget:
    def _proposal(self, tp_r, stop_distance=1.0, entry=100.0):
        from app.signals.trade_proposal import Decision, TradeProposal

        return TradeProposal(
            symbol="BTCUSDT",
            side=Side.LONG,
            decision=Decision.ENTER,
            entry=entry,
            stop_loss=entry - stop_distance,
            stop_distance=stop_distance,
            tp1=entry + tp_r[0] * stop_distance,
            tp2=entry + tp_r[1] * stop_distance,
            tp3=entry + tp_r[2] * stop_distance,
        )

    def test_rr_plan_prefers_the_weighted_figure(self):
        proposal = self._proposal((1.0, 2.0, 3.0))
        proposal.rr = 2.0
        proposal.rr_weighted = 1.85
        assert proposal.rr_plan == pytest.approx(1.85)

    def test_rr_plan_falls_back_when_the_ladder_was_never_computed(self):
        """A hand-built or replayed proposal must not be rejected at 0.00."""

        proposal = self._proposal((1.0, 2.0, 3.0))
        proposal.rr = 2.4
        proposal.rr_weighted = 0.0
        assert proposal.rr_plan == pytest.approx(2.4)

    def test_the_weighted_figure_punishes_a_near_first_target(self):
        """Not a looser gate: banking 40% at 0.7R scores worse, not better."""

        from app.config import build_settings
        from app.signals.signal_engine import SignalEngine
        from tests.conftest import TEST_ENV

        settings = build_settings(env=dict(TEST_ENV))
        engine = SignalEngine(settings)
        near = engine._weighted_rr(100.0, 99.0, (100.7, 103.0, 104.0), Side.LONG)
        even = engine._weighted_rr(100.0, 99.0, (102.0, 103.0, 104.0), Side.LONG)
        assert near < even
        assert near < 3.0, "TP2 alone would have said 3.00"

    def test_both_gates_use_the_same_measure(self):
        """The risk engine must not silently become the binding one."""

        import inspect

        from app.risk import risk_engine
        from app.signals import signal_engine

        assert "proposal.rr_plan < settings.min_rr" in inspect.getsource(risk_engine)
        assert "proposal.rr_plan < required_rr" in inspect.getsource(signal_engine)

    def test_the_no_trade_model_and_quality_score_agree_with_the_gate(self):
        import inspect

        from app import intelligence

        source = inspect.getsource(intelligence)
        assert "rr=proposal.rr," not in source, "still judging TP2 alone somewhere"
        assert source.count("rr=proposal.rr_plan,") >= 3

    def test_a_genuinely_poor_plan_is_still_refused(self):
        """Every level within ~1R must remain a rejection."""

        from app.config import build_settings
        from app.signals.signal_engine import SignalEngine
        from tests.conftest import TEST_ENV

        settings = build_settings(env=dict(TEST_ENV))
        engine = SignalEngine(settings)
        weighted = engine._weighted_rr(100.0, 99.0, (100.7, 100.9, 101.2), Side.LONG)
        assert weighted < settings.min_rr


class TestExpectedValueUsesTheRealLadder:
    def test_the_ev_ladder_is_built_from_the_proposed_targets(self):
        import inspect

        from app import intelligence

        source = inspect.getsource(intelligence)
        assert "abs(target - proposal.entry) / risk_distance" in source, (
            "expected value was valuing every symbol on the configured "
            "TP1_R/TP2_R/TP3_R rather than the targets actually proposed"
        )

    def test_a_closer_ladder_is_worth_less(self):
        from app.quality.expected_value import compute_expected_value

        far = compute_expected_value(
            confidence=0.7, rr=3.0, partial_ladder=[(0.4, 1.0), (0.35, 3.0), (0.25, 5.0)]
        )
        near = compute_expected_value(
            confidence=0.7, rr=3.0, partial_ladder=[(0.4, 0.7), (0.35, 0.9), (0.25, 1.2)]
        )
        assert near.expected_r < far.expected_r
