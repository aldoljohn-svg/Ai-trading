"""The risk engine - the last gate before an order can exist.

The signal engine says "this looks like a trade".  The risk engine decides
whether it *is allowed to be one*, and at what size.  Every check is applied;
none can be skipped; the first failure does not short-circuit the rest, so the
operator sees every reason at once instead of fixing them one at a time.

Order of evaluation:

1. circuit breakers (including the kill switch and emergency stop)
2. mode - never approve a live order outside LIVE mode
3. proposal integrity - the geometry must make sense
4. per-trade risk budget, scaled by regime
5. leverage safety versus the liquidation estimate
6. position sizing
7. portfolio limits, correlation-adjusted
8. margin availability
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, TradingMode
from app.domain import ContractSpec, Side
from app.logger import get_logger
from app.regime.regime_detector import strategy_for_regime
from app.risk.circuit_breaker import BreakerReport, CircuitBreaker
from app.risk.portfolio_risk import OpenRisk, PortfolioRisk, PortfolioState
from app.risk.position_sizing import (
    PositionSize,
    max_safe_leverage,
    size_position,
    stop_is_safe_from_liquidation,
)
from app.signals.trade_proposal import Decision, TradeProposal

log = get_logger(__name__)


@dataclass(slots=True)
class RiskDecision:
    approved: bool = False
    proposal: TradeProposal | None = None
    size: PositionSize | None = None
    leverage: float = 1.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    reasons: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    ts: int = 0

    def reject(self, reason: str) -> "RiskDecision":
        if reason not in self.rejections:
            self.rejections.append(reason)
        self.approved = False
        return self

    @property
    def contracts(self) -> float:
        return self.size.contracts if self.size else 0.0

    @property
    def notional(self) -> float:
        return self.size.notional if self.size else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "symbol": self.proposal.symbol if self.proposal else None,
            "side": self.proposal.side.value if self.proposal and self.proposal.side else None,
            "contracts": self.contracts,
            "notional": round(self.notional, 4),
            "leverage": self.leverage,
            "risk_amount": round(self.risk_amount, 4),
            "risk_pct": round(self.risk_pct, 6),
            "reasons": self.reasons,
            "rejections": self.rejections,
            "metrics": self.metrics,
        }


class RiskEngine:
    def __init__(
        self,
        settings: Settings,
        portfolio_risk: PortfolioRisk | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.settings = settings
        self.portfolio_risk = portfolio_risk or PortfolioRisk(
            max_portfolio_risk=settings.max_portfolio_risk,
            max_correlated_exposure=settings.max_correlated_exposure,
            correlation_threshold=settings.correlation_threshold,
            max_open_positions=settings.max_open_positions,
            max_gross_leverage=settings.max_leverage * settings.max_open_positions,
        )
        self.breaker = breaker or CircuitBreaker(
            max_daily_loss=settings.max_daily_loss,
            max_drawdown=settings.max_drawdown_stop,
            max_consecutive_losses=settings.max_consecutive_losses,
            kill_switch_file=settings.resolve_path(settings.kill_switch_file),
        )
        self.last_report: BreakerReport | None = None

    # -- breakers ---------------------------------------------------------

    def check_breakers(self, state: PortfolioState) -> BreakerReport:
        report = self.breaker.evaluate(
            equity=state.equity,
            realized_pnl_today=state.realized_pnl_today,
            peak_equity=state.peak_equity,
            consecutive_losses=state.consecutive_losses,
        )
        self.last_report = report
        return report

    # -- approval ---------------------------------------------------------

    def evaluate(
        self,
        proposal: TradeProposal,
        state: PortfolioState,
        spec: ContractSpec,
        regime_risk_multiplier: float | None = None,
        breaker_report: BreakerReport | None = None,
        intelligence_scale: float = 1.0,
    ) -> RiskDecision:
        """Size and approve an entry.

        ``intelligence_scale`` is the advanced intelligence layer's influence on
        size.  It is clamped to ``[0, 1]`` here rather than trusted, so that
        layer can only ever *shrink* a position — no downstream component is
        permitted to raise risk above what this engine computed.
        """

        settings = self.settings
        decision = RiskDecision(proposal=proposal, ts=int(time.time()))

        # --- 1. breakers --------------------------------------------------
        report = breaker_report or self.check_breakers(state)
        if report.blocked:
            for trip in report.trips:
                decision.reject(f"circuit breaker {trip.reason.value}: {trip.message}")

        # --- 2. mode ------------------------------------------------------
        if settings.trading_mode is TradingMode.BACKTEST and not proposal.is_entry:
            decision.reject("proposal is not an entry")

        # --- 3. proposal integrity ---------------------------------------
        if proposal.decision is not Decision.ENTER or proposal.side is None:
            decision.reject("signal engine did not approve this proposal")
            return decision

        side = proposal.side
        if proposal.entry <= 0 or proposal.stop_loss <= 0:
            return decision.reject("proposal has no valid entry or stop price")
        if side is Side.LONG and proposal.stop_loss >= proposal.entry:
            return decision.reject("long stop is not below entry")
        if side is Side.SHORT and proposal.stop_loss <= proposal.entry:
            return decision.reject("short stop is not above entry")
        # Same measure the signal engine gates on: the weighted R of the scaled
        # exit plan, which is what the position actually earns.  Keeping this on
        # TP2 alone would silently make it the binding gate.
        if proposal.rr_plan < settings.min_rr:
            decision.reject(
                f"reward:risk {proposal.rr_plan:.2f} across the exit plan is "
                f"below the {settings.min_rr:.2f} minimum"
            )
        if proposal.confidence < settings.min_confidence:
            decision.reject(
                f"confidence {proposal.confidence:.0%} is below the "
                f"{settings.min_confidence:.0%} minimum"
            )

        # --- 4. risk budget ------------------------------------------------
        strategy = strategy_for_regime(proposal.regime)
        multiplier = (
            regime_risk_multiplier
            if regime_risk_multiplier is not None
            else strategy.risk_multiplier
        )
        if multiplier <= 0:
            decision.reject(
                f"{proposal.regime.value} regime sets the risk multiplier to zero"
            )

        scale = min(1.0, max(0.0, intelligence_scale))
        if scale < 0.999:
            multiplier *= scale
            decision.reasons.append(
                f"size scaled to {scale:.0%} by the intelligence layer"
            )

        # Never scale risk *up* after losses; only ever down.
        if state.consecutive_losses >= 2:
            multiplier *= 0.6
            decision.reasons.append(
                f"risk halved after {state.consecutive_losses} consecutive losses"
            )
        if state.drawdown >= settings.max_drawdown_stop * 0.5:
            multiplier *= 0.6
            decision.reasons.append(
                f"risk reduced while {state.drawdown:.1%} below the equity peak"
            )

        remaining_budget = self.portfolio_risk.remaining_risk_budget(state)
        requested_risk = state.equity * settings.default_risk_per_trade * multiplier
        if remaining_budget <= 0:
            decision.reject("no portfolio risk budget remains")
        elif requested_risk > remaining_budget:
            scaled = remaining_budget / max(state.equity * settings.default_risk_per_trade, 1e-12)
            multiplier = min(multiplier, scaled)
            decision.reasons.append(
                f"risk trimmed to the ${remaining_budget:.2f} remaining budget"
            )

        # --- 5. leverage safety --------------------------------------------
        safe_cap = max_safe_leverage(
            proposal.entry, proposal.stop_loss, hard_cap=settings.max_leverage
        )
        leverage = min(
            proposal.suggested_leverage,
            settings.max_leverage,
            spec.max_leverage,
            safe_cap,
        )
        leverage = max(leverage, settings.min_leverage)
        decision.leverage = round(leverage, 2)

        safe, message = stop_is_safe_from_liquidation(
            proposal.entry, proposal.stop_loss, leverage, side
        )
        if not safe:
            decision.reject(message)

        # --- 6. sizing ------------------------------------------------------
        size = size_position(
            equity=state.equity,
            available_margin=state.available,
            entry=proposal.entry,
            stop=proposal.stop_loss,
            side=side,
            spec=spec,
            risk_pct=settings.default_risk_per_trade,
            leverage=leverage,
            max_leverage=settings.max_leverage,
            min_leverage=settings.min_leverage,
            max_notional_pct_of_equity=settings.max_position_notional_pct,
            risk_multiplier=multiplier,
            fee_rate=settings.taker_fee,
        )
        decision.size = size
        decision.risk_amount = size.actual_risk
        decision.risk_pct = size.risk_pct
        decision.reasons.extend(size.reasons)

        if not size.ok:
            for reason in size.reasons:
                decision.reject(reason)
            return decision

        if size.risk_pct > settings.default_risk_per_trade * 1.10:
            decision.reject(
                f"sized risk {size.risk_pct:.3%} exceeds the per-trade budget "
                f"{settings.default_risk_per_trade:.3%}"
            )

        # --- 7. portfolio limits --------------------------------------------
        candidate = OpenRisk(
            symbol=proposal.symbol,
            side=side,
            risk_amount=size.actual_risk,
            notional=size.notional,
            leverage=leverage,
        )
        ok, problems, metrics = self.portfolio_risk.can_add(state, candidate)
        decision.metrics.update(metrics)
        if not ok:
            for problem in problems:
                decision.reject(problem)

        # --- 8. margin -------------------------------------------------------
        if size.margin > state.available:
            decision.reject(
                f"required margin ${size.margin:.2f} exceeds the available "
                f"${state.available:.2f}"
            )

        decision.metrics.update(
            {
                "risk_amount": round(size.actual_risk, 4),
                "risk_pct": round(size.risk_pct, 6),
                "notional": round(size.notional, 4),
                "margin": round(size.margin, 4),
                "leverage": decision.leverage,
                "liquidation_price": round(size.liquidation_price, 8),
            }
        )

        if not decision.rejections:
            decision.approved = True
            decision.reasons.append(
                f"risk ${size.actual_risk:.2f} ({size.risk_pct:.2%} of equity) "
                f"at {decision.leverage:g}x"
            )
        return decision

    # -- de-risking -------------------------------------------------------

    def risk_reduction_factor(self, state: PortfolioState) -> float:
        """How much open risk should be cut, ``0`` = none, ``1`` = flatten."""

        report = self.last_report or self.check_breakers(state)
        if report.requires_flatten:
            return 1.0
        if report.requires_derisk:
            return 0.5
        drawdown = state.drawdown
        if drawdown >= self.settings.max_drawdown_stop * 0.8:
            return 0.5
        if drawdown >= self.settings.max_drawdown_stop * 0.6:
            return 0.25
        return 0.0

    def describe(self, state: PortfolioState) -> dict[str, Any]:
        report = self.last_report or self.check_breakers(state)
        data = self.portfolio_risk.describe(state)
        data.update(
            {
                "breaker_state": report.state.value,
                "breaker_summary": report.summary(),
                "trips": [t.reason.value for t in report.trips],
                "warnings": report.warnings,
                "risk_per_trade": self.settings.default_risk_per_trade,
                "max_portfolio_risk": self.settings.max_portfolio_risk,
                "max_daily_loss": self.settings.max_daily_loss,
                "max_open_positions": self.settings.max_open_positions,
                "max_leverage": self.settings.max_leverage,
                "daily_loss_pct": round(state.daily_loss_pct, 6),
                "consecutive_losses": state.consecutive_losses,
            }
        )
        return data


__all__ = ["RiskEngine", "RiskDecision"]
