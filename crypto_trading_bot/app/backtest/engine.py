"""Event-driven backtester.

Design goals, in priority order: **no lookahead**, **same code as live**, then
speed.

No lookahead
------------
* A signal computed on the close of bar ``i`` is filled at the **open of bar
  i+1**, never at bar ``i``'s close.
* Higher-timeframe series are sliced to bars that had already *closed* at the
  moment the execution bar closed - ``htf_close_ts <= execution_close_ts``.
* When one bar spans both the stop and a target, the stop is assumed to have
  been hit first.  From OHLC alone the order is unknowable, and the optimistic
  assumption is what makes backtests stop matching live results.
* Fees, spread crossing, slippage and funding are all charged.

Same code as live
-----------------
The signal engine, risk engine, position sizing, stop manager, target manager
and trailing logic are the *same objects* the live bot uses.  Only the fill
comes from the bar data instead of an exchange.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from app.backtest.metrics import PerformanceMetrics, compute_metrics
from app.config import Settings
from app.domain import (
    Bias,
    Candle,
    ContractSpec,
    Regime,
    Side,
    Ticker,
    Timeframe,
)
from app.logger import get_logger
from app.portfolio.portfolio_manager import ManagedPosition
from app.position_manager.manager import (
    ActionKind,
    ManagementContext,
    PositionManager,
)
from app.risk.portfolio_risk import OpenRisk, PortfolioRisk, PortfolioState
from app.risk.position_sizing import size_position
from app.scanner.scanner import SymbolAnalysis, TimeframeAnalysis, _build_timeframe_analysis, combine_bias
from app.scanner.universe import ScanCandidate
from app.signals.signal_engine import SignalEngine
from app.signals.trade_proposal import Decision
from app.regime.regime_detector import detect_regime, strategy_for_regime

log = get_logger(__name__)


@dataclass(slots=True)
class BacktestConfig:
    symbols: list[str] = field(default_factory=list)
    execution_timeframe: Timeframe = Timeframe.M15
    context_timeframes: tuple[Timeframe, ...] = (
        Timeframe.H1,
        Timeframe.H4,
        Timeframe.D1,
    )
    starting_equity: float = 1000.0
    warmup_bars: int = 220
    max_bars: int | None = None
    signal_stride: int = 1               # evaluate entries every N bars
    taker_fee: float = 0.0006
    slippage_pct: float = 0.0005
    spread_pct: float = 0.0002
    funding_interval_hours: int = 8
    funding_rate: float = 0.0001
    latency_bars: int = 0                # extra bars before a fill lands
    #: Bars of history handed to the analytical engines.  Must match what the
    #: live scanner uses (``MarketData.candles(limit=...)``) or the backtest
    #: measures a different strategy from the one that will trade.
    analysis_window: int = 400
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = f"bt-{int(time.time())}-{uuid.uuid4().hex[:6]}"

    def required_warmup(self, min_context_bars: int = 60) -> int:
        """Execution bars needed before every context timeframe has history.

        Starting the replay earlier than this produces a long stretch where the
        daily and 4H series are still empty, so the bot correctly refuses to
        trade and the "backtest" measures nothing but its own warmup.
        """

        if not self.context_timeframes:
            return self.warmup_bars
        highest = max(tf.seconds for tf in self.context_timeframes)
        needed = (highest * min_context_bars) // self.execution_timeframe.seconds
        return max(self.warmup_bars, int(needed))


@dataclass(slots=True)
class BacktestResult:
    config: BacktestConfig
    metrics: PerformanceMetrics
    trades: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[int] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    start_ts: int = 0
    end_ts: int = 0
    bars: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.config.run_id,
            "created_at": int(time.time()),
            "symbols": self.config.symbols,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "config": {
                "execution_timeframe": self.config.execution_timeframe.value,
                "starting_equity": self.config.starting_equity,
                "taker_fee": self.config.taker_fee,
                "slippage_pct": self.config.slippage_pct,
                "spread_pct": self.config.spread_pct,
                "funding_rate": self.config.funding_rate,
                "warmup_bars": self.config.warmup_bars,
                "signal_stride": self.config.signal_stride,
            },
            "metrics": self.metrics.as_dict(),
            "equity_curve": [round(v, 4) for v in self.equity_curve[-2000:]],
            "trades": len(self.trades),
        }

    def summary(self) -> str:
        lines = [
            f"Backtest {self.config.run_id}",
            f"Symbols: {', '.join(self.config.symbols)}",
            f"Bars: {self.bars} on {self.config.execution_timeframe.value}",
            "",
            self.metrics.summary(),
        ]
        if self.rejections:
            lines.append("")
            lines.append("Top reasons no trade was taken:")
            for reason, count in sorted(
                self.rejections.items(), key=lambda kv: kv[1], reverse=True
            )[:8]:
                lines.append(f"  {count:5d}  {reason}")
        return "\n".join(lines)


class BacktestEngine:
    def __init__(
        self,
        settings: Settings,
        data: Mapping[str, Mapping[Timeframe, Sequence[Candle]]],
        contracts: Mapping[str, ContractSpec],
        config: BacktestConfig,
    ) -> None:
        self.settings = settings
        self.data = data
        self.contracts = contracts
        self.config = config
        self.signal_engine = SignalEngine(settings, predictor=None)
        self.position_manager = PositionManager(settings)
        self.portfolio_risk = PortfolioRisk(
            max_portfolio_risk=settings.max_portfolio_risk,
            max_correlated_exposure=settings.max_correlated_exposure,
            correlation_threshold=settings.correlation_threshold,
            max_open_positions=settings.max_open_positions,
            max_gross_leverage=settings.max_leverage * settings.max_open_positions,
        )

        self.equity = config.starting_equity
        self.balance = config.starting_equity
        self.peak_equity = config.starting_equity
        self.positions: dict[str, ManagedPosition] = {}
        self.closed_trades: list[dict[str, Any]] = []
        self.equity_curve: list[float] = []
        self.timestamps: list[int] = []
        self.rejections: dict[str, int] = {}
        self.realized_today = 0.0
        self._day = 0
        self._last_funding_ts = 0
        self._bars_with_position = 0

    # -- data helpers -----------------------------------------------------

    def _slice(
        self, symbol: str, timeframe: Timeframe, close_ts: int
    ) -> list[Candle]:
        """Bars of ``timeframe`` that had closed by ``close_ts``.

        This single function is what prevents higher-timeframe lookahead.
        """

        series = self.data.get(symbol, {}).get(timeframe) or []
        # Binary search for the last bar that had closed by ``close_ts``.
        low, high = 0, len(series)
        while low < high:
            mid = (low + high) // 2
            if series[mid].ts + timeframe.seconds <= close_ts:
                low = mid + 1
            else:
                high = mid
        return list(series[max(0, low - self.config.analysis_window) : low])

    def _build_analysis(
        self, symbol: str, close_ts: int, execution: list[Candle]
    ) -> SymbolAnalysis | None:
        timeframes: dict[Timeframe, TimeframeAnalysis] = {}
        try:
            timeframes[self.config.execution_timeframe] = _build_timeframe_analysis(
                self.config.execution_timeframe, execution
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("execution analysis failed for %s: %s", symbol, exc)
            return None

        for timeframe in self.config.context_timeframes:
            candles = self._slice(symbol, timeframe, close_ts)
            if len(candles) < 60:
                continue
            try:
                timeframes[timeframe] = _build_timeframe_analysis(timeframe, candles)
            except Exception as exc:  # noqa: BLE001
                log.debug("%s %s analysis failed: %s", symbol, timeframe.value, exc)

        source = (
            timeframes.get(Timeframe.H1)
            or timeframes.get(Timeframe.H4)
            or timeframes[self.config.execution_timeframe]
        )
        regime = detect_regime(
            source.candles, source.indicators, source.slopes, source.structure
        )

        biases: list[Bias] = []
        weights: list[float] = []
        for timeframe, weight in (
            (Timeframe.D1, 3.0),
            (Timeframe.H4, 2.0),
            (Timeframe.H1, 1.5),
        ):
            analysis = timeframes.get(timeframe)
            if analysis is not None:
                biases.append(analysis.bias())
                weights.append(weight)
        htf_bias, alignment = combine_bias(biases, weights)

        last = execution[-1]
        spec = self.contracts[symbol]
        half_spread = last.close * self.config.spread_pct / 2
        ticker = Ticker(
            symbol=symbol,
            last=last.close,
            bid=last.close - half_spread,
            ask=last.close + half_spread,
            quote_volume_24h=max(
                self.settings.min_24h_quote_volume * 2,
                sum(c.quote_volume for c in execution[-96:]),
            ),
            high_24h=max(c.high for c in execution[-96:]),
            low_24h=min(c.low for c in execution[-96:]),
            ts=last.ts,
        )
        candidate = ScanCandidate(
            symbol=symbol,
            ticker=ticker,
            spec=spec,
            quote_volume=ticker.quote_volume_24h,
            spread_pct=self.config.spread_pct,
            volatility_pct=(ticker.high_24h - ticker.low_24h) / max(last.close, 1e-9),
            change_pct=0.0,
            prescreen_score=50.0,
        )
        return SymbolAnalysis(
            symbol=symbol,
            ticker=ticker,
            candidate=candidate,
            timeframes=timeframes,
            regime=regime,
            htf_bias=htf_bias,
            alignment=alignment,
            fundamentals=None,
            order_book=None,
            anomaly="",
            ts=last.ts,
        )

    # -- accounting -------------------------------------------------------

    def _mark(self, prices: Mapping[str, float]) -> None:
        unrealized = 0.0
        for position in self.positions.values():
            price = prices.get(position.symbol)
            if price:
                unrealized += position.unrealized_pnl(price)
        self.equity = self.balance + unrealized
        self.peak_equity = max(self.peak_equity, self.equity)

    def _state(self, prices: Mapping[str, float]) -> PortfolioState:
        risks = [
            p.to_open_risk(prices.get(p.symbol, p.entry_price))
            for p in self.positions.values()
        ]
        used_margin = sum(
            p.notional(prices.get(p.symbol, p.entry_price)) / max(p.leverage, 1.0)
            for p in self.positions.values()
        )
        return PortfolioState(
            equity=self.equity,
            available=max(0.0, self.balance - used_margin),
            open_risks=risks,
            realized_pnl_today=self.realized_today,
            peak_equity=self.peak_equity,
            consecutive_losses=self._consecutive_losses(),
        )

    def _consecutive_losses(self) -> int:
        streak = 0
        for trade in reversed(self.closed_trades):
            if trade["pnl"] < 0:
                streak += 1
            else:
                break
        return streak

    def _roll_day(self, ts: int) -> None:
        day = ts // 86400
        if day != self._day:
            self._day = day
            self.realized_today = 0.0

    # -- fills ------------------------------------------------------------

    def _entry_fill(self, side: Side, bar: Candle) -> float:
        """Fill at the next bar's open, crossing the spread and slipping."""

        base = bar.open
        cost = self.config.spread_pct / 2 + self.config.slippage_pct
        return base * (1 + cost) if side is Side.LONG else base * (1 - cost)

    def _exit_fill(self, side: Side, price: float) -> float:
        cost = self.config.spread_pct / 2 + self.config.slippage_pct
        return price * (1 - cost) if side is Side.LONG else price * (1 + cost)

    def _close(
        self,
        position: ManagedPosition,
        fraction: float,
        raw_price: float,
        reason: str,
        ts: int,
    ) -> None:
        fraction = max(0.0, min(fraction, 1.0))
        quantity = position.quantity * fraction
        if quantity <= 0:
            return
        spec = self.contracts[position.symbol]
        quantity = spec.round_volume(quantity)
        if quantity <= 0 or position.quantity - quantity < spec.min_volume:
            quantity = position.quantity

        price = self._exit_fill(position.side, raw_price)
        base_quantity = quantity * position.contract_size
        pnl = (price - position.entry_price) * position.side.sign * base_quantity
        fee = price * base_quantity * self.config.taker_fee

        position.quantity -= quantity
        position.realized_pnl += pnl
        position.fees += fee
        self.balance += pnl - fee
        self.realized_today += pnl - fee

        if position.quantity <= 1e-12:
            risk = position.risk_amount or 1e-12
            self.closed_trades.append(
                {
                    "symbol": position.symbol,
                    "side": position.side.value,
                    "entry": position.entry_price,
                    "exit": price,
                    "pnl": position.realized_pnl - position.fees - position.funding,
                    "gross_pnl": position.realized_pnl,
                    "fees": position.fees,
                    "funding": position.funding,
                    "r_multiple": (
                        position.realized_pnl - position.fees - position.funding
                    )
                    / risk,
                    "risk_amount": position.risk_amount,
                    "opened_at": position.opened_at,
                    "closed_at": ts,
                    "reason": reason,
                    "confidence": position.meta.get("confidence"),
                    "regime": position.meta.get("regime"),
                }
            )
            self.positions.pop(position.symbol, None)

    def _charge_funding(self, ts: int, prices: Mapping[str, float]) -> None:
        interval = self.config.funding_interval_hours * 3600
        if self._last_funding_ts == 0:
            self._last_funding_ts = ts
            return
        if ts - self._last_funding_ts < interval:
            return
        self._last_funding_ts = ts
        for position in self.positions.values():
            price = prices.get(position.symbol, position.entry_price)
            notional = position.quantity * position.contract_size * price
            payment = notional * self.config.funding_rate * position.side.sign
            position.funding += payment
            self.balance -= payment
            self.realized_today -= payment

    # -- main loop --------------------------------------------------------

    def run(self) -> BacktestResult:
        config = self.config
        timeframe = config.execution_timeframe
        symbols = [s for s in config.symbols if s in self.data]
        if not symbols:
            raise ValueError("no symbols with data to backtest")

        # Align on the execution series of the first symbol, then require every
        # symbol to have a bar at that timestamp.
        series: dict[str, list[Candle]] = {
            symbol: list(self.data[symbol].get(timeframe) or []) for symbol in symbols
        }
        length = min(len(s) for s in series.values())
        warmup = config.required_warmup()
        if warmup > config.warmup_bars:
            log.info(
                "warmup raised from %d to %d bars so the %s context has history",
                config.warmup_bars,
                warmup,
                max(config.context_timeframes, key=lambda tf: tf.seconds).value,
            )
        if length <= warmup + 5:
            raise ValueError(
                f"not enough execution bars ({length}) for a {warmup}-bar warmup; "
                f"either supply more history or drop the highest context timeframe"
            )
        end = length - 1
        if config.max_bars:
            end = min(end, warmup + config.max_bars)

        pending: list[tuple[int, str, Any, Any]] = []   # (fill_index, symbol, decision, spec)

        for index in range(warmup, end):
            bar_ts = series[symbols[0]][index].ts
            close_ts = bar_ts + timeframe.seconds
            self._roll_day(bar_ts)

            prices = {s: series[s][index].close for s in symbols}
            self._charge_funding(bar_ts, prices)

            # --- 1. manage open positions against THIS bar ----------------
            for symbol in list(self.positions):
                position = self.positions.get(symbol)
                if position is None:
                    continue
                bar = series[symbol][index]
                self._manage(position, bar, symbol, index, close_ts)

            if self.positions:
                self._bars_with_position += 1

            # --- 2. execute fills scheduled from previous bars ------------
            still_pending: list[tuple[int, str, Any, Any]] = []
            for fill_index, symbol, decision, spec in pending:
                if fill_index > index:
                    still_pending.append((fill_index, symbol, decision, spec))
                    continue
                self._open(decision, spec, series[symbol][index], bar_ts)
            pending = still_pending

            # --- 3. look for new entries ----------------------------------
            if (index - warmup) % max(config.signal_stride, 1) == 0:
                self._mark(prices)
                for symbol in symbols:
                    if symbol in self.positions:
                        continue
                    if any(p[1] == symbol for p in pending):
                        continue
                    if len(self.positions) >= self.settings.max_open_positions:
                        break
                    decision = self._evaluate(symbol, index, close_ts, series, prices)
                    if decision is not None:
                        pending.append(
                            (
                                index + 1 + config.latency_bars,
                                symbol,
                                decision,
                                self.contracts[symbol],
                            )
                        )

            self._mark(prices)
            self.equity_curve.append(self.equity)
            self.timestamps.append(bar_ts)

        # Close whatever is still open at the final bar.
        final_index = end - 1
        for symbol in list(self.positions):
            position = self.positions[symbol]
            self._close(
                position,
                1.0,
                series[symbol][final_index].close,
                "backtest ended",
                series[symbol][final_index].ts,
            )
        self._mark({s: series[s][final_index].close for s in symbols})
        self.equity_curve.append(self.equity)

        metrics = compute_metrics(
            trades=self.closed_trades,
            equity_curve=self.equity_curve,
            starting_equity=config.starting_equity,
            period_seconds=timeframe.seconds,
            risk_per_trade=self.settings.default_risk_per_trade,
            bars_with_position=self._bars_with_position,
            total_bars=max(end - warmup, 1),
        )
        return BacktestResult(
            config=config,
            metrics=metrics,
            trades=self.closed_trades,
            equity_curve=self.equity_curve,
            timestamps=self.timestamps,
            rejections=self.rejections,
            start_ts=series[symbols[0]][warmup].ts,
            end_ts=series[symbols[0]][final_index].ts,
            bars=end - warmup,
        )

    # -- steps ------------------------------------------------------------

    def _manage(
        self,
        position: ManagedPosition,
        bar: Candle,
        symbol: str,
        index: int,
        close_ts: int,
    ) -> None:
        context = ManagementContext(
            price=bar.close,
            atr=float(position.meta.get("atr") or 0.0),
            high=bar.high,
            low=bar.low,
            candles=self._slice(symbol, self.config.execution_timeframe, close_ts)[-60:],
        )
        actions = self.position_manager.evaluate(position, context)
        for action in actions:
            if action.kind is ActionKind.STOP_HIT:
                self._close(
                    position, 1.0, position.stop_loss, _stop_label(position), bar.ts
                )
                return
            if action.kind is ActionKind.TARGET_HIT:
                self.position_manager.mark_target(action)
                self._close(
                    position,
                    action.close_fraction,
                    action.price,
                    f"TP{action.target_level}",
                    bar.ts,
                )
                if position.quantity <= 1e-12:
                    return
            elif action.kind is ActionKind.CLOSE:
                self._close(position, 1.0, bar.close, action.reason, bar.ts)
                return
            elif action.kind is ActionKind.MOVE_STOP:
                self.position_manager.apply_stop(action)

    def _evaluate(
        self,
        symbol: str,
        index: int,
        close_ts: int,
        series: Mapping[str, list[Candle]],
        prices: Mapping[str, float],
    ) -> Any:
        window = self.config.analysis_window
        execution = series[symbol][max(0, index + 1 - window) : index + 1]
        analysis = self._build_analysis(symbol, close_ts, execution)
        if analysis is None:
            return None

        proposal = self.signal_engine.evaluate(analysis)
        if proposal.decision is not Decision.ENTER or proposal.side is None:
            for reason in proposal.rejections[:1]:
                key = reason.split("(")[0].strip()[:70]
                self.rejections[key] = self.rejections.get(key, 0) + 1
            return None

        state = self._state(prices)
        strategy = strategy_for_regime(proposal.regime)
        spec = self.contracts[symbol]

        size = size_position(
            equity=state.equity,
            available_margin=state.available,
            entry=proposal.entry,
            stop=proposal.stop_loss,
            side=proposal.side,
            spec=spec,
            risk_pct=self.settings.default_risk_per_trade,
            leverage=min(proposal.suggested_leverage, self.settings.max_leverage),
            max_leverage=self.settings.max_leverage,
            min_leverage=self.settings.min_leverage,
            max_notional_pct_of_equity=self.settings.max_position_notional_pct,
            risk_multiplier=strategy.risk_multiplier,
            fee_rate=self.config.taker_fee,
        )
        if not size.ok:
            key = (size.reasons[0] if size.reasons else "sizing failed")[:70]
            self.rejections[key] = self.rejections.get(key, 0) + 1
            return None

        candidate = OpenRisk(
            symbol=symbol,
            side=proposal.side,
            risk_amount=size.actual_risk,
            notional=size.notional,
            leverage=min(proposal.suggested_leverage, self.settings.max_leverage),
        )
        ok, problems, _metrics = self.portfolio_risk.can_add(state, candidate)
        if not ok:
            key = problems[0][:70]
            self.rejections[key] = self.rejections.get(key, 0) + 1
            return None

        return (proposal, size)

    def _open(self, decision: Any, spec: ContractSpec, bar: Candle, ts: int) -> None:
        proposal, size = decision
        if proposal.side is None:
            return
        fill = self._entry_fill(proposal.side, bar)

        # The stop must still be on the correct side after slippage.
        if proposal.side is Side.LONG and fill <= proposal.stop_loss:
            return
        if proposal.side is Side.SHORT and fill >= proposal.stop_loss:
            return

        base_quantity = size.contracts * spec.contract_size
        entry_fee = fill * base_quantity * self.config.taker_fee
        self.balance -= entry_fee
        self.realized_today -= entry_fee

        position = ManagedPosition(
            symbol=proposal.symbol,
            side=proposal.side,
            quantity=size.contracts,
            entry_price=fill,
            leverage=size.leverage,
            contract_size=spec.contract_size,
            stop_loss=proposal.stop_loss,
            initial_stop=proposal.stop_loss,
            tp1=proposal.tp1,
            tp2=proposal.tp2,
            tp3=proposal.tp3,
            initial_quantity=size.contracts,
            risk_amount=size.actual_risk,
            opened_at=ts,
            mode="backtest",
            fees=entry_fee,
            meta={
                "confidence": proposal.confidence,
                "regime": proposal.regime.value,
                "atr": proposal.atr,
                "rr": proposal.rr,
                "reasons": proposal.reasons,
            },
        )
        self.positions[position.symbol] = position


def _stop_label(position: ManagedPosition) -> str:
    """Distinguish a genuine loss from a protected exit in the audit trail."""

    if position.trailing_active:
        return "trailing stop"
    if position.breakeven_done:
        return "break-even stop"
    return "stop loss"


__all__ = ["BacktestEngine", "BacktestConfig", "BacktestResult"]
