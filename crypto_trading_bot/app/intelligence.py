"""Intelligence coordinator - the advanced-quant layer.

Sits **on top of** the existing signal engine rather than replacing it. The
signal engine still owns entry/stop/target geometry, which is well tested and
works; this layer adds the parts that make the decision institutional:

    SymbolAnalysis (existing)
        -> order flow / microstructure / liquidity / derivatives
        -> ensemble of 15 models with regime-aware weights
        -> data risk / model risk / execution risk
        -> trade quality + expected value
        -> no-trade model
        -> market memory analogue
        -> verdict + full decision trace -> journal

The verdict can only ever be **more** restrictive than the signal engine.
It has veto power; it has no power to create a trade the signal engine
rejected, and no power to raise size or relax a limit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.config import Settings
from app.domain import Regime, Side, Timeframe
from app.ensemble.base import ModelContext, ModelSignal
from app.ensemble.engine import EnsembleEngine, EnsembleResult
from app.ensemble.no_trade import NoTradeModel, NoTradeVerdict
from app.logger import get_logger
from app.memory.clustering import session_of
from app.memory.journal import DecisionJournal, DecisionKind, DecisionTrace
from app.memory.market_memory import MarketMemory, MarketState
from app.orderflow.derivatives import DerivativesRead, analyse_derivatives
from app.orderflow.liquidity_pools import LiquidityMap, build_liquidity_map
from app.orderflow.microstructure import MicrostructureRead, analyse_microstructure
from app.orderflow.order_flow import OrderFlowRead, analyse_order_flow
from app.quality.data_risk import DataRiskReport, assess_data_risk
from app.quality.execution_risk import ExecutionRiskReport, assess_execution_risk
from app.quality.expected_value import ExpectedValue, compute_expected_value
from app.quality.model_risk import ModelRiskReport, assess_model_risk
from app.quality.trade_quality import TradeQuality, compute_trade_quality
from app.safety.anomaly import AnomalyReport, detect_anomalies
from app.signals.trade_proposal import Decision, TradeProposal

log = get_logger(__name__)


@dataclass(slots=True)
class IntelligenceVerdict:
    """The advanced layer's opinion on a proposal from the signal engine."""

    symbol: str
    ts: int
    approved: bool = False
    veto_reasons: list[str] = field(default_factory=list)
    #: Why the *signal engine* declined, when it did.  Carried through so a
    #: "not taken" verdict always explains itself, even when this layer had no
    #: objection of its own.
    signal_rejections: list[str] = field(default_factory=list)
    ensemble: EnsembleResult | None = None
    order_flow: OrderFlowRead | None = None
    microstructure: MicrostructureRead | None = None
    liquidity: LiquidityMap | None = None
    derivatives: DerivativesRead | None = None
    data_risk: DataRiskReport | None = None
    model_risk: ModelRiskReport | None = None
    execution_risk: ExecutionRiskReport | None = None
    trade_quality: TradeQuality | None = None
    expected_value: ExpectedValue | None = None
    no_trade: NoTradeVerdict | None = None
    anomaly: AnomalyReport | None = None
    analogue: Any = None
    decision_id: str = ""
    #: Multiplier the risk engine applies on top of its own sizing. Never > 1.
    size_multiplier: float = 1.0
    recommended_order_type: str = "market"

    @property
    def quality_score(self) -> float:
        return self.trade_quality.score if self.trade_quality else 0.0

    @property
    def model_agreement(self) -> str:
        return self.ensemble.model_agreement_label if self.ensemble else "0/0"

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ts": self.ts,
            "approved": self.approved,
            "veto_reasons": self.veto_reasons,
            "signal_rejections": self.signal_rejections,
            "decision_id": self.decision_id,
            "size_multiplier": round(self.size_multiplier, 4),
            "recommended_order_type": self.recommended_order_type,
            "trade_quality": self.trade_quality.as_dict() if self.trade_quality else None,
            "expected_value": (
                self.expected_value.as_dict() if self.expected_value else None
            ),
            "ensemble": self.ensemble.as_dict() if self.ensemble else None,
            "order_flow": self.order_flow.as_dict() if self.order_flow else None,
            "microstructure": (
                self.microstructure.as_dict() if self.microstructure else None
            ),
            "liquidity": self.liquidity.as_dict() if self.liquidity else None,
            "derivatives": self.derivatives.as_dict() if self.derivatives else None,
            "data_risk": self.data_risk.as_dict() if self.data_risk else None,
            "model_risk": self.model_risk.as_dict() if self.model_risk else None,
            "execution_risk": (
                self.execution_risk.as_dict() if self.execution_risk else None
            ),
            "no_trade": self.no_trade.as_dict() if self.no_trade else None,
            "anomaly": self.anomaly.as_dict() if self.anomaly else None,
            "analogue": self.analogue.as_dict() if self.analogue else None,
        }

    def summary(self) -> str:
        lines = [
            f"{self.symbol}: {'APPROVED' if self.approved else 'VETOED'} "
            f"quality {self.quality_score:.0f}/100 agreement {self.model_agreement}"
        ]
        if self.expected_value:
            lines.append(f"  {self.expected_value.summary()}")
        for reason in self.veto_reasons[:4]:
            lines.append(f"  ✗ {reason}")
        return "\n".join(lines)


class IntelligenceCoordinator:
    def __init__(
        self,
        settings: Settings,
        ensemble: EnsembleEngine | None = None,
        journal: DecisionJournal | None = None,
        memory: MarketMemory | None = None,
        predictor: Any = None,
        repositories: Any = None,
    ) -> None:
        self.settings = settings
        self.ensemble = ensemble or EnsembleEngine()
        self.journal = journal or DecisionJournal(repositories)
        self.memory = memory or MarketMemory(repositories)
        self.predictor = predictor
        self.repositories = repositories
        self.no_trade_model = NoTradeModel(
            threshold=settings.no_trade_threshold,
            min_agreement=settings.min_model_agreement,
        )
        #: Rolling reference values per symbol, used to spot abnormality.
        self._spread_reference: dict[str, list[float]] = {}
        self._depth_reference: dict[str, list[float]] = {}
        self._funding_history: dict[str, list[float]] = {}

    # -- context building --------------------------------------------------

    def build_context(
        self,
        analysis: Any,
        portfolio_state: Any = None,
        portfolio_risk: Any = None,
        notional: float = 0.0,
        side: Side | None = None,
    ) -> ModelContext:
        """Assemble everything the models are allowed to look at."""

        execution = analysis.primary
        atr = execution.atr if execution else 0.0
        candles = execution.candles if execution else []

        order_flow = analyse_order_flow(candles, analysis.order_book, atr=atr)
        microstructure = analyse_microstructure(
            analysis.order_book,
            notional=notional,
            side=side,
            max_spread_pct=self.settings.max_spread_pct,
            max_slippage_pct=self.settings.max_slippage_pct,
            unavailable_reason=getattr(analysis, "book_problem", "") or "",
            reference_spread_pct=self._reference(
                self._spread_reference, analysis.symbol, analysis.candidate.spread_pct
            ),
        )
        liquidity = build_liquidity_map(
            candles, execution.ict if execution else None, analysis.order_book, atr=atr
        )
        derivatives = self._derivatives(analysis, candles)

        return ModelContext(
            symbol=analysis.symbol,
            ts=analysis.ts,
            analysis=analysis,
            regime=analysis.regime.regime,
            order_flow=order_flow,
            microstructure=microstructure,
            derivatives=derivatives,
            liquidity=liquidity,
            fundamentals=analysis.fundamentals,
            predictor=self.predictor,
            features=analysis.features(),
            meta={
                "portfolio_state": portfolio_state,
                "portfolio_risk": portfolio_risk,
            },
        )

    def _derivatives(self, analysis: Any, candles: Sequence[Any]) -> DerivativesRead:
        ticker = analysis.ticker
        funding = getattr(ticker, "funding_rate", None)
        if funding is not None:
            history = self._funding_history.setdefault(analysis.symbol, [])
            history.append(funding)
            if len(history) > 200:
                del history[:-200]

        price_change = None
        if len(candles) >= 30:
            reference = candles[-30].close
            if reference > 0:
                price_change = (candles[-1].close - reference) / reference

        return analyse_derivatives(
            funding_rate=funding,
            open_interest=getattr(ticker, "open_interest", None),
            price_change_pct=price_change,
            # MEXC's ticker gives a point-in-time OI; a change series needs
            # history we do not persist yet, so this stays None rather than
            # being fabricated from one observation.
            oi_change_pct=None,
            funding_history=self._funding_history.get(analysis.symbol),
        )

    @staticmethod
    def _reference(
        store: dict[str, list[float]], symbol: str, value: float, window: int = 200
    ) -> float | None:
        """Rolling median of a metric, used as its own normality baseline."""

        if value is None or value <= 0:
            return None
        history = store.setdefault(symbol, [])
        history.append(value)
        if len(history) > window:
            del history[:-window]
        if len(history) < 20:
            return None
        ordered = sorted(history)
        return ordered[len(ordered) // 2]

    # -- evaluation ---------------------------------------------------------

    def evaluate(
        self,
        analysis: Any,
        proposal: TradeProposal,
        portfolio_state: Any = None,
        portfolio_risk: Any = None,
        market_data_health: Mapping[str, Any] | None = None,
        exchange_health: Any = None,
        latency_ms: float = 0.0,
        notional: float = 0.0,
        loss_cluster: Any = None,
        drift_report: Any = None,
        record: bool = True,
    ) -> IntelligenceVerdict:
        """Assess a proposal. Can only ever restrict, never permit."""

        settings = self.settings
        verdict = IntelligenceVerdict(symbol=analysis.symbol, ts=int(time.time()))

        context = self.build_context(
            analysis,
            portfolio_state=portfolio_state,
            portfolio_risk=portfolio_risk,
            notional=notional,
            side=proposal.side,
        )
        verdict.order_flow = context.order_flow
        verdict.microstructure = context.microstructure
        verdict.liquidity = context.liquidity
        verdict.derivatives = context.derivatives

        # --- ensemble -----------------------------------------------------
        ensemble = self.ensemble.evaluate(context)
        verdict.ensemble = ensemble

        # --- anomalies -----------------------------------------------------
        execution = analysis.primary
        candles = execution.candles if execution else []
        verdict.anomaly = detect_anomalies(
            candles,
            spread_pct=analysis.candidate.spread_pct,
            reference_spread_pct=self._reference(
                self._spread_reference, analysis.symbol, analysis.candidate.spread_pct
            ),
            funding_rate=getattr(analysis.ticker, "funding_rate", None),
            funding_history=self._funding_history.get(analysis.symbol),
            timeframe_seconds=(
                execution.timeframe.seconds if execution else 900
            ),
        )

        # --- risk scores ----------------------------------------------------
        verdict.data_risk = assess_data_risk(
            analysis, market_data_health=market_data_health
        )
        verdict.model_risk = assess_model_risk(
            ensemble,
            weight_table=self.ensemble.weights,
            drift_report=drift_report,
            predictor=self.predictor,
            regime=analysis.regime.regime,
        )

        stop_distance_pct = (
            proposal.stop_distance / proposal.entry if proposal.entry > 0 else 0.0
        )
        expected_reward_pct = stop_distance_pct * proposal.rr if proposal.rr else 0.0
        verdict.execution_risk = assess_execution_risk(
            context.microstructure,
            taker_fee=settings.taker_fee,
            maker_fee=settings.maker_fee,
            expected_reward_pct=expected_reward_pct,
            latency_ms=latency_ms,
            exchange_health=exchange_health,
        )
        verdict.recommended_order_type = verdict.execution_risk.recommended_order_type

        # --- no-trade model ---------------------------------------------------
        verdict.no_trade = self.no_trade_model.evaluate(
            ensemble,
            context,
            rr=proposal.rr,
            min_rr=settings.min_rr,
            loss_cluster=loss_cluster,
            execution_risk=verdict.execution_risk.score,
            data_risk=verdict.data_risk.score,
        )

        # --- portfolio headroom ------------------------------------------------
        headroom = 1.0
        if portfolio_state is not None and portfolio_risk is not None:
            used = portfolio_risk.effective_risk_pct(portfolio_state)
            cap = portfolio_risk.max_portfolio_risk
            headroom = max(0.0, 1.0 - (used / cap if cap > 0 else 0.0))

        # --- trade quality -------------------------------------------------------
        verdict.trade_quality = compute_trade_quality(
            ensemble=ensemble,
            regime=analysis.regime.regime,
            order_flow=context.order_flow,
            liquidity=context.liquidity,
            rr=proposal.rr,
            min_rr=settings.min_rr,
            data_risk=verdict.data_risk,
            model_risk=verdict.model_risk,
            execution_risk=verdict.execution_risk,
            portfolio_headroom=headroom,
            minimum=settings.min_trade_quality,
        )

        # --- expected value --------------------------------------------------------
        ladder = [
            (settings.tp1_close_pct, settings.tp1_r),
            (settings.tp2_close_pct, settings.tp2_r),
            (
                max(0.0, 1 - settings.tp1_close_pct - settings.tp2_close_pct),
                settings.tp3_r,
            ),
        ]
        cost_pct = verdict.execution_risk.total_cost_pct
        verdict.expected_value = compute_expected_value(
            confidence=proposal.confidence,
            rr=proposal.rr,
            cost_pct=cost_pct,
            stop_distance_pct=stop_distance_pct,
            partial_ladder=ladder,
            threshold_r=settings.min_expected_value_r,
        )

        # --- market memory ----------------------------------------------------------
        state_features = self._state_features(analysis, context)
        verdict.analogue = self.memory.find_analogues(
            state_features, regime=analysis.regime.regime.value
        )

        # --- vetoes -------------------------------------------------------------------
        vetoes: list[str] = []

        if verdict.data_risk.blocked:
            vetoes.append(f"data risk {verdict.data_risk.score:.2f}: "
                          + "; ".join(verdict.data_risk.problems[:2]))
        if verdict.model_risk.blocked:
            vetoes.append(f"model risk {verdict.model_risk.score:.2f}: "
                          + "; ".join(verdict.model_risk.problems[:2]))
        if verdict.execution_risk.blocked:
            vetoes.append(f"execution risk {verdict.execution_risk.score:.2f}: "
                          + "; ".join(verdict.execution_risk.problems[:2]))
        if verdict.no_trade.no_trade:
            vetoes.append("no-trade model: " + "; ".join(
                verdict.no_trade.blocking_reasons[:2]))
        if not verdict.trade_quality.acceptable:
            vetoes.append(
                f"trade quality {verdict.trade_quality.score:.0f} below the "
                f"{settings.min_trade_quality:.0f} minimum"
            )
        if not verdict.expected_value.acceptable:
            vetoes.append(
                f"expected value {verdict.expected_value.expected_r:+.3f}R below the "
                f"{settings.min_expected_value_r:+.3f}R threshold"
            )
        if verdict.anomaly.black_swan:
            vetoes.append(f"black-swan conditions: {', '.join(verdict.anomaly.types[:3])}")
        elif verdict.anomaly.severity >= settings.max_anomaly_severity:
            vetoes.append(
                f"market anomaly severity {verdict.anomaly.severity:.2f}"
            )

        # The ensemble must actually agree with the direction the signal engine
        # chose. Disagreement here is the whole point of running an ensemble.
        if proposal.side is not None and ensemble.signal.is_directional:
            if ensemble.signal.direction != proposal.side.sign:
                vetoes.append(
                    f"ensemble says {ensemble.signal.value} but the signal engine "
                    f"proposed {proposal.side.value.upper()}"
                )
        elif proposal.side is not None and settings.require_ensemble_agreement:
            vetoes.append(
                f"ensemble has no directional view "
                f"(participation {ensemble.participation:.0%})"
            )

        verdict.veto_reasons = vetoes
        verdict.signal_rejections = list(getattr(proposal, "rejections", []) or [])
        verdict.approved = not vetoes and proposal.decision is Decision.ENTER

        # --- sizing influence (only ever downward) --------------------------------
        quality_scale = min(1.0, verdict.trade_quality.score / 100.0 + 0.35)
        risk_scale = 1.0 - 0.5 * max(
            verdict.data_risk.score,
            verdict.model_risk.score,
            verdict.execution_risk.score,
        )
        verdict.size_multiplier = max(0.0, min(1.0, quality_scale * risk_scale))

        # --- journal ----------------------------------------------------------------
        if record:
            verdict.decision_id = self._record(analysis, proposal, verdict, context)
        return verdict

    # -- journalling ---------------------------------------------------------

    def _record(
        self,
        analysis: Any,
        proposal: TradeProposal,
        verdict: IntelligenceVerdict,
        context: ModelContext,
    ) -> str:
        execution = analysis.primary
        trace = DecisionTrace(
            data={
                "price": analysis.close,
                "spread_pct": analysis.candidate.spread_pct,
                "quote_volume": analysis.candidate.quote_volume,
                "timeframes": sorted(tf.value for tf in analysis.timeframes),
                "anomaly": verdict.anomaly.as_dict() if verdict.anomaly else {},
            },
            features=context.features,
            regime={
                "regime": analysis.regime.regime.value,
                "confidence": analysis.regime.confidence,
                "volatility_ratio": analysis.regime.volatility_ratio,
                "htf_bias": analysis.htf_bias.value,
                "alignment": analysis.alignment,
            },
            models=[o.as_dict() for o in (verdict.ensemble.outputs if verdict.ensemble else [])],
            ensemble=verdict.ensemble.as_dict() if verdict.ensemble else {},
            expected_value=(
                verdict.expected_value.as_dict() if verdict.expected_value else {}
            ),
            risk={
                "data_risk": verdict.data_risk.as_dict() if verdict.data_risk else {},
                "model_risk": verdict.model_risk.as_dict() if verdict.model_risk else {},
                "execution_risk": (
                    verdict.execution_risk.as_dict() if verdict.execution_risk else {}
                ),
                "no_trade": verdict.no_trade.as_dict() if verdict.no_trade else {},
                "trade_quality": (
                    verdict.trade_quality.as_dict() if verdict.trade_quality else {}
                ),
            },
            execution={
                "order_flow": verdict.order_flow.as_dict() if verdict.order_flow else {},
                "microstructure": (
                    verdict.microstructure.as_dict() if verdict.microstructure else {}
                ),
                "recommended_order_type": verdict.recommended_order_type,
                "size_multiplier": verdict.size_multiplier,
            },
        )

        record = self.journal.record(
            kind=DecisionKind.ENTRY if verdict.approved else DecisionKind.NO_TRADE,
            symbol=analysis.symbol,
            mode=self.settings.trading_mode.value,
            timeframe=execution.timeframe.value if execution else "",
            side=proposal.side.value if proposal.side else None,
            decision="ENTER" if verdict.approved else "NO_TRADE",
            confidence=proposal.confidence,
            entry=proposal.entry,
            stop=proposal.stop_loss,
            tp1=proposal.tp1,
            tp2=proposal.tp2,
            tp3=proposal.tp3,
            trade_quality=verdict.quality_score,
            expected_r=(
                verdict.expected_value.expected_r if verdict.expected_value else 0.0
            ),
            reasoning=(
                verdict.ensemble.top_reasons() if verdict.ensemble else proposal.reasons
            ),
            rejections=verdict.veto_reasons + proposal.rejections,
            trace=trace,
        )
        return record.decision_id

    # -- memory -------------------------------------------------------------

    def _state_features(self, analysis: Any, context: ModelContext) -> dict[str, float]:
        execution = analysis.primary
        indicators = execution.indicators if execution else None
        structure = execution.structure if execution else None
        ict = execution.ict if execution else None

        return {
            "atr_pct": indicators.atr_pct if indicators else 0.0,
            "rsi": (indicators.last_rsi or 50.0) if indicators else 50.0,
            "ma_stack": float(indicators.ma_stack) if indicators else 0.0,
            "trend_quality": structure.trend_quality if structure else 0.0,
            "range_position": structure.range_position if structure else 0.5,
            "regime_direction": float(analysis.regime.direction),
            "volatility_ratio": analysis.regime.volatility_ratio,
            "relative_volume": (
                indicators.volume.get("relative_volume", 1.0) if indicators else 1.0
            ),
            "funding_rate": getattr(analysis.ticker, "funding_rate", 0.0) or 0.0,
            "premium_discount": ict.premium_discount if ict else 0.5,
            "order_flow_score": (
                context.order_flow.score if context.order_flow else 0.0
            ),
            "htf_bias": {
                "BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0, "CONFLICT": 0.0
            }.get(analysis.htf_bias.value, 0.0),
        }

    def remember(self, analysis: Any, context: ModelContext, horizon_bars: int = 24) -> None:
        """Store the current state so it can become a future analogue."""

        execution = analysis.primary
        if execution is None:
            return
        self.memory.remember(
            MarketState(
                symbol=analysis.symbol,
                ts=execution.candles[-1].ts if execution.candles else analysis.ts,
                timeframe=execution.timeframe.value,
                features=self._state_features(analysis, context),
                regime=analysis.regime.regime.value,
                horizon_bars=horizon_bars,
            )
        )

    # -- learning ------------------------------------------------------------

    def record_outcome(
        self,
        decision_id: str,
        symbol: str,
        ensemble: EnsembleResult | None,
        r_multiple: float,
        regime: Regime | str = "ALL",
        lesson: str = "",
    ) -> None:
        """Attribute a closed trade back to the models and the journal."""

        won = r_multiple > 0
        if ensemble is not None:
            self.ensemble.record_outcome(ensemble, won, r_multiple, regime)
            self._persist_weights()

        if decision_id:
            self.journal.record_outcome(
                parent_id=decision_id,
                symbol=symbol,
                outcome={
                    "r_multiple": round(r_multiple, 4),
                    "won": won,
                    "regime": regime.value if isinstance(regime, Regime) else regime,
                },
                lesson=lesson,
            )

    def _persist_weights(self) -> None:
        if self.repositories is None:
            return
        try:
            rows = []
            for record in self.ensemble.weights._records.values():  # noqa: SLF001
                rows.append(
                    {
                        "model": record.model,
                        "regime": record.regime,
                        "trades": record.trades,
                        "wins": record.wins,
                        "r_sum": record.r_sum,
                        "confidence_sum": record.confidence_sum,
                        "last_updated": record.last_updated,
                    }
                )
            self.repositories.model_performance.save_many(rows)
        except Exception as exc:  # noqa: BLE001
            log.debug("could not persist model weights: %s", exc)

    def load_weights(self) -> int:
        if self.repositories is None:
            return 0
        try:
            return self.ensemble.weights.load_rows(
                self.repositories.model_performance.all()
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("could not load model weights: %s", exc)
            return 0

    # -- views ----------------------------------------------------------------

    def describe(self, regime: Regime | str = "ALL") -> dict[str, Any]:
        return {
            "models": self.ensemble.describe_weights(regime),
            "journal": self.journal.statistics(),
            "memory": self.memory.statistics(),
            "no_trade_threshold": self.no_trade_model.threshold,
        }


__all__ = ["IntelligenceCoordinator", "IntelligenceVerdict"]
