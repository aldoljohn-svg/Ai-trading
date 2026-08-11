"""The trading engine - the orchestrator that owns every component and loop.

Deliberately independent of FastAPI and of Telegram: ``app.main`` mounts the
dashboard on top of this, and the Telegram bot talks to it through the same
public methods the dashboard uses.  That separation is what lets the whole
system be driven headlessly by ``scripts/run_simulation.py`` and by the tests.

Loops
-----
``scanner``      discover, analyse, rank and (maybe) enter - every
                 ``SCANNER_INTERVAL_SECONDS``
``positions``    manage everything open - every few seconds
``health``       component health + circuit-breaker evaluation
``reconcile``    compare our records against the exchange

Crash recovery runs before any loop starts: positions are reloaded from the
database, reconciled against the exchange, and the engine refuses to resume
trading if the reconciled state is not healthy.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.config import Settings, TradingMode
from app.data.candle_store import CandleStore
from app.data.market_data import MarketData
from app.data.websocket import MexcWebSocket
from app.database.database import Database, get_database
from app.database.repositories import Repositories
from app.domain import (
    Bias,
    BotState,
    ContractSpec,
    HealthState,
    Regime,
    Side,
    Timeframe,
)
from app.exchange.base import BaseExchange, ExchangeError
from app.exchange.mexc import MexcFuturesExchange
from app.exchange.synthetic import SyntheticExchange
from app.execution.execution_engine import ExecutionEngine
from app.execution.order_reconciliation import Reconciler
from app.fundamental.macro_engine import MacroEngine
from app.fundamental.news_engine import NewsEngine
from app.fundamental.sentiment import FundamentalEngine
from app.health.monitor import (
    HealthMonitor,
    HealthReport,
    market_data_health,
    predictor_health,
    risk_health,
    scanner_health,
    websocket_health,
)
from app.health.preflight import PreflightReport, run_preflight
from app.logger import get_logger
from app.ml.model_registry import ModelRegistry
from app.ml.predict import Predictor
from app.paper.paper_engine import PaperEngine
from app.portfolio.portfolio_manager import ManagedPosition, PortfolioManager
from app.position_manager.manager import (
    ActionKind,
    ManagementContext,
    PositionAction,
    PositionManager,
)
from app.risk.portfolio_risk import PortfolioRisk
from app.risk.risk_engine import RiskEngine
from app.scanner.ranking import Opportunity, rank_opportunities
from app.scanner.scanner import Scanner, SymbolAnalysis
from app.scanner.universe import UniverseBuilder
from app.intelligence import IntelligenceCoordinator, IntelligenceVerdict
from app.signals.signal_engine import SignalEngine
from app.signals.trade_proposal import Decision, TradeProposal

log = get_logger(__name__)


class NullNotifier:
    """No-op notifier so the engine never has to check for ``None``."""

    async def send(self, text: str, **kwargs: Any) -> None:
        return None

    async def trade_opened(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def position_update(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def trade_closed(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def alert(self, text: str, **kwargs: Any) -> None:
        return None

    async def test_connection(self) -> bool:
        return False


@dataclass(slots=True)
class EngineStats:
    started_at: float = 0.0
    scans: int = 0
    proposals: int = 0
    entries: int = 0
    exits: int = 0
    errors: int = 0
    last_scan_duration: float = 0.0
    last_error: str = ""


class TradingEngine:
    def __init__(self, settings: Settings, notifier: Any | None = None) -> None:
        self.settings = settings
        self.notifier = notifier or NullNotifier()
        self.state = BotState.STOPPED
        self.stats = EngineStats()

        self.database: Database | None = None
        self.repositories: Repositories | None = None
        self.exchange: BaseExchange | None = None
        self.market_data: MarketData | None = None
        self.websocket: MexcWebSocket | None = None
        self.scanner: Scanner | None = None
        self.signal_engine: SignalEngine | None = None
        self.risk_engine: RiskEngine | None = None
        self.portfolio: PortfolioManager | None = None
        self.execution: ExecutionEngine | None = None
        self.position_manager: PositionManager | None = None
        self.reconciler: Reconciler | None = None
        self.predictor: Predictor | None = None
        self.intelligence: IntelligenceCoordinator | None = None
        self.health = HealthMonitor()
        self.paper: PaperEngine | None = None
        #: Latest intelligence verdict per symbol, for Telegram / dashboard.
        self.last_verdicts: dict[str, IntelligenceVerdict] = {}
        #: Verdict that authorised each currently-open position, so a close can
        #: be attributed back to the models that voted for it.
        self._entry_verdicts: dict[str, IntelligenceVerdict] = {}
        #: Proposal timestamp each cached verdict was computed for.
        self._verdict_for: dict[str, int] = {}

        self.preflight_report: PreflightReport | None = None
        self.last_health: HealthReport | None = None
        self.last_opportunities: list[Opportunity] = []
        self.last_analyses: dict[str, SymbolAnalysis] = {}
        self.live_enabled = False

        self._tasks: list[asyncio.Task] = []
        self._stopping = asyncio.Event()
        self._pause_reason = ""

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    async def build(self) -> None:
        settings = self.settings

        self.database = get_database(
            settings.database_url, root=settings.resolve_path(".")
        )
        self.repositories = Repositories(self.database)

        self.exchange = self._build_exchange()
        await self.exchange.connect()

        self.market_data = MarketData(
            self.exchange,
            store=CandleStore(),
            candle_repository=self.repositories.candles,
        )

        universe = UniverseBuilder(
            quote_currency=settings.quote_currency,
            min_quote_volume=settings.min_24h_quote_volume,
            max_spread_pct=settings.max_spread_pct,
            blacklist=settings.symbol_blacklist,
            max_symbols=settings.max_symbols_to_scan,
        )

        http_client = getattr(self.exchange, "http", None)
        macro = MacroEngine(self.exchange, http_client=http_client)
        news = NewsEngine(
            calendar_path=settings.resolve_path(settings.data_dir)
            / "economic_calendar.json",
        )
        fundamentals = FundamentalEngine(macro, news, exchange=self.exchange)

        self.scanner = Scanner(
            market_data=self.market_data,
            universe=universe,
            fundamentals=fundamentals,
            deep_analysis_count=settings.deep_analysis_count,
        )

        registry = ModelRegistry(
            settings.resolve_path(settings.models_dir),
            repository=self.repositories.models,
        )
        self.predictor = Predictor(registry, model_name=settings.ml_model_name)
        self.signal_engine = SignalEngine(settings, predictor=self.predictor)

        if settings.intelligence_enabled:
            self.intelligence = IntelligenceCoordinator(
                settings,
                predictor=self.predictor,
                repositories=self.repositories,
            )
            loaded = self.intelligence.load_weights()
            if loaded:
                log.info("restored performance weights for %d models", loaded)

        portfolio_risk = PortfolioRisk(
            max_portfolio_risk=settings.max_portfolio_risk,
            max_correlated_exposure=settings.max_correlated_exposure,
            correlation_threshold=settings.correlation_threshold,
            max_open_positions=settings.max_open_positions,
            max_gross_leverage=settings.max_leverage * settings.max_open_positions,
        )
        self.risk_engine = RiskEngine(settings, portfolio_risk=portfolio_risk)

        self.portfolio = PortfolioManager(
            repositories=self.repositories,
            portfolio_risk=portfolio_risk,
            mode=settings.trading_mode.value,
            starting_equity=settings.paper_starting_equity,
        )
        self.position_manager = PositionManager(settings)

        broker = await self._build_broker()
        self.execution = ExecutionEngine(
            settings=settings,
            broker=broker,
            portfolio=self.portfolio,
            repositories=self.repositories,
            notifier=self.notifier,
            live_enabled=False,
        )

        self.reconciler = Reconciler(
            exchange=self.exchange,
            portfolio=self.portfolio,
            repositories=self.repositories,
        )

        if settings.data_source == "mexc":
            self.websocket = MexcWebSocket(
                url=settings.mexc_ws_url,
                access_key=settings.mexc_access_key,
                secret_key=settings.mexc_secret_key,
            )
            self.websocket.on_ticker(self._on_ticker)

        self._register_health_probes()
        log.info(
            "engine built: mode=%s data=%s db=%s",
            settings.trading_mode.value,
            settings.data_source,
            self.database.path,
        )

    def _build_exchange(self) -> BaseExchange:
        settings = self.settings
        if settings.data_source == "synthetic":
            log.warning(
                "using the SYNTHETIC data source - prices are simulated, not real"
            )
            return SyntheticExchange(equity=settings.paper_starting_equity)
        return MexcFuturesExchange(
            access_key=settings.mexc_access_key,
            secret_key=settings.mexc_secret_key,
            base_url=settings.mexc_base_url,
            recv_window=settings.mexc_recv_window,
            quote=settings.quote_currency,
            allow_trading=False,     # flipped on only after pre-flight passes
        )

    async def _build_broker(self) -> Any:
        settings = self.settings
        if settings.trading_mode is TradingMode.LIVE:
            return self.exchange
        contracts: Mapping[str, ContractSpec] = {}
        try:
            contracts = await self.exchange.contracts()
        except ExchangeError as exc:
            log.warning("could not preload contracts for the paper engine: %s", exc)
        self.paper = PaperEngine(
            price_provider=self._price_for,
            contracts=contracts,
            taker_fee=settings.taker_fee,
            maker_fee=settings.maker_fee,
            slippage_pct=settings.slippage_pct,
            latency_ms=settings.latency_ms,
            funding_interval_hours=settings.funding_interval_hours,
        )
        return self.paper

    def _price_for(self, symbol: str) -> float:
        if self.portfolio is None:
            return 0.0
        price = self.portfolio.price(symbol)
        if price > 0:
            return price
        analysis = self.last_analyses.get(symbol)
        return analysis.close if analysis else 0.0

    def _on_ticker(self, ticker: Any) -> None:
        if self.market_data is not None:
            self.market_data.ingest_ticker(ticker)
        if self.portfolio is not None:
            self.portfolio.set_price(ticker.symbol, ticker.last)

    def _register_health_probes(self) -> None:
        settings = self.settings
        self.health.register("MEXC REST", lambda: self.exchange.health())
        if self.websocket is not None:
            self.health.register("MEXC WebSocket", lambda: websocket_health(self.websocket))
        self.health.register("Database", lambda: self.database.health())
        self.health.register("Telegram", self._telegram_health)
        self.health.register(
            "Scanner",
            lambda: scanner_health(self.scanner, settings.scanner_interval_seconds),
        )
        self.health.register("AI", lambda: predictor_health(self.predictor))
        self.health.register(
            "Risk Engine",
            lambda: risk_health(self.risk_engine, self.portfolio.state()),
        )
        self.health.register("Execution", lambda: self.execution.health())
        self.health.register("Market Data", lambda: market_data_health(self.market_data))

    async def _telegram_health(self) -> dict[str, Any]:
        if not self.settings.has_telegram:
            return {"state": "WARNING", "detail": "not configured"}
        ok = await self.notifier.test_connection()
        return {
            "state": "HEALTHY" if ok else "ERROR",
            "detail": "reachable" if ok else "unreachable",
        }

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self.state in (BotState.RUNNING, BotState.STARTING):
            return
        self.state = BotState.STARTING
        self.stats.started_at = time.time()
        settings = self.settings

        if self.exchange is None:
            await self.build()

        assert self.portfolio and self.repositories and self.execution

        # --- first-start safety banner ----------------------------------
        if settings.trading_mode is TradingMode.LIVE:
            log.warning("=" * 72)
            log.warning("TRADING_MODE=live - REAL ORDERS WILL BE PLACED WITH REAL MONEY")
            log.warning("=" * 72)
            await self.notifier.alert(
                "⚠️ <b>LIVE MODE CONFIGURED</b>\n\n"
                "The bot is starting in LIVE mode. Real orders will be placed once "
                "pre-flight checks pass."
            )
        else:
            log.info(
                "starting in %s mode - no real orders will be placed",
                settings.trading_mode.value.upper(),
            )

        # --- crash recovery ---------------------------------------------
        recovered = self.portfolio.load()
        if recovered:
            log.info("crash recovery: restored %d positions", recovered)

        await self._seed_prices()

        # Reconciliation compares our records against the *exchange*, which is
        # only meaningful when the exchange is where the positions actually
        # live.  Paper fills never reach MEXC, so reconciling them would report
        # every simulated position as missing and discard it.
        if settings.trading_mode is TradingMode.LIVE:
            report = await self.reconciler.reconcile(apply=True)
            if report.blocking:
                self.risk_engine.breaker.set_reconciliation_problem(report.summary())
                await self.notifier.alert(
                    "🚨 <b>STATE MISMATCH ON STARTUP</b>\n\n"
                    f"<pre>{report.summary()}</pre>\n\n"
                    "New entries are blocked until this is resolved."
                )
            else:
                self.risk_engine.breaker.clear_reconciliation_problem()
        elif recovered:
            log.info(
                "%s mode: %d recovered position(s) restored from the database "
                "(exchange reconciliation does not apply)",
                settings.trading_mode.value,
                recovered,
            )

        # --- pre-flight --------------------------------------------------
        self.preflight_report = await run_preflight(
            settings=settings,
            exchange=self.exchange,
            database=self.database,
            notifier=self.notifier,
            reconciler=self.reconciler,
        )
        if settings.trading_mode is TradingMode.LIVE:
            if self.preflight_report.passed:
                self.live_enabled = True
                self.execution.live_enabled = True
                if isinstance(self.exchange, MexcFuturesExchange):
                    self.exchange.allow_trading = True
                    await self.exchange.sync_clock()
                await self.notifier.alert(
                    "🟢 <b>LIVE TRADING ENABLED</b>\n\n"
                    f"<pre>{self.preflight_report.summary()}</pre>"
                )
            else:
                log.error("LIVE MODE BLOCKED - staying flat")
                await self.notifier.alert(
                    "🚫 <b>LIVE MODE BLOCKED</b>\n\n"
                    f"<pre>{self.preflight_report.summary()}</pre>"
                )
                self.risk_engine.breaker.trip_manual(
                    "live pre-flight checks failed - trading disabled"
                )

        # --- account -----------------------------------------------------
        await self._sync_account()

        if self.websocket is not None:
            await self.websocket.start()

        self.state = BotState.RUNNING
        self._stopping.clear()
        self._tasks = [
            asyncio.create_task(self._scanner_loop(), name="scanner"),
            asyncio.create_task(self._position_loop(), name="positions"),
            asyncio.create_task(self._health_loop(), name="health"),
            asyncio.create_task(self._reconcile_loop(), name="reconcile"),
        ]
        self.repositories.events.system_event(
            "engine", f"started in {settings.trading_mode.value} mode"
        )
        log.info("engine running (%s)", settings.trading_mode.value)

    async def stop(self, reason: str = "shutdown") -> None:
        if self.state is BotState.STOPPED:
            return
        self.state = BotState.STOPPING
        self._stopping.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks = []

        if self.websocket is not None:
            await self.websocket.stop()
        if self.portfolio is not None:
            self.portfolio.persist_all()
        if self.exchange is not None:
            await self.exchange.close()
        if self.repositories is not None:
            self.repositories.events.system_event("engine", f"stopped: {reason}")
        self.state = BotState.STOPPED
        log.info("engine stopped (%s)", reason)

    def pause(self, reason: str = "paused by operator") -> None:
        if self.state is BotState.RUNNING:
            self.state = BotState.PAUSED
            self._pause_reason = reason
            log.warning("engine paused: %s", reason)
            if self.repositories:
                self.repositories.events.system_event("engine", f"paused: {reason}")

    def resume(self) -> None:
        if self.state is BotState.PAUSED:
            self.state = BotState.RUNNING
            self._pause_reason = ""
            log.info("engine resumed")
            if self.repositories:
                self.repositories.events.system_event("engine", "resumed")

    async def emergency_stop(self, reason: str = "operator emergency stop") -> dict[str, Any]:
        """Close everything, cancel everything, and stop trading."""

        log.error("EMERGENCY STOP: %s", reason)
        self.state = BotState.EMERGENCY
        self.risk_engine.breaker.trigger_emergency(reason)

        results: list[Any] = []
        cancelled = 0
        try:
            cancelled = await self.execution.cancel_all_orders()
            results = await self.execution.close_all(reason="emergency stop")
        except Exception as exc:  # noqa: BLE001 - must still report
            log.error("emergency close encountered an error: %s", exc)
            self.stats.last_error = str(exc)

        closed = sum(1 for r in results if getattr(r, "ok", False))
        failed = [r for r in results if not getattr(r, "ok", False)]

        if self.repositories:
            self.repositories.events.risk_event(
                kind="EMERGENCY_STOP",
                message=reason,
                severity="CRITICAL",
                detail={"closed": closed, "failed": len(failed)},
            )
        await self.notifier.alert(
            "🚨 <b>EMERGENCY STOP EXECUTED</b>\n\n"
            f"Closed: {closed}\nFailed: {len(failed)}\nOrders cancelled: {cancelled}\n\n"
            f"Reason: {reason}"
        )
        await self.stop(reason="emergency stop")
        return {"closed": closed, "failed": len(failed), "cancelled": cancelled}

    async def run_forever(self) -> None:
        await self.start()
        try:
            await self._stopping.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # loops
    # ------------------------------------------------------------------

    async def _scanner_loop(self) -> None:
        interval = self.settings.scanner_interval_seconds
        while not self._stopping.is_set():
            try:
                if self.state is BotState.RUNNING:
                    started = time.perf_counter()
                    await self.scan_once()
                    self.stats.last_scan_duration = time.perf_counter() - started
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                self.stats.errors += 1
                self.stats.last_error = str(exc)
                log.exception("scanner loop error: %s", exc)
            await self._sleep(interval)

    async def _position_loop(self) -> None:
        interval = self.settings.position_manage_interval_seconds
        while not self._stopping.is_set():
            try:
                if self.state in (BotState.RUNNING, BotState.PAUSED):
                    # Paused stops *new* entries; open positions are always managed.
                    await self.manage_positions()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.stats.errors += 1
                self.stats.last_error = str(exc)
                log.exception("position loop error: %s", exc)
            await self._sleep(interval)

    async def _health_loop(self) -> None:
        interval = self.settings.health_interval_seconds
        while not self._stopping.is_set():
            try:
                report = await self.health.check()
                self.last_health = report
                self._feed_breaker(report)
                if self.portfolio is not None:
                    self.portfolio.record_snapshot()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("health loop error: %s", exc)
            await self._sleep(interval)

    async def _reconcile_loop(self) -> None:
        interval = self.settings.reconcile_interval_seconds
        while not self._stopping.is_set():
            await self._sleep(interval)
            if self._stopping.is_set():
                break
            try:
                if self.settings.trading_mode is not TradingMode.LIVE:
                    continue
                report = await self.reconciler.reconcile(apply=True)
                if report.blocking:
                    self.risk_engine.breaker.set_reconciliation_problem(report.summary())
                    await self.notifier.alert(
                        "🚨 <b>RECONCILIATION MISMATCH</b>\n\n"
                        f"<pre>{report.summary()}</pre>\n\nNew entries are blocked."
                    )
                else:
                    self.risk_engine.breaker.clear_reconciliation_problem()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("reconcile loop error: %s", exc)

    async def _sleep(self, seconds: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)

    def _feed_breaker(self, report: HealthReport) -> None:
        breaker = self.risk_engine.breaker
        market = report.by_name("Market Data")
        if market is not None and market.state is HealthState.ERROR:
            breaker.set_data_problem(market.detail or "market data unhealthy")
        else:
            breaker.clear_data_problem()

        rest = report.by_name("MEXC REST")
        if rest is not None:
            breaker.record_api_result(rest.state is not HealthState.ERROR)

    # ------------------------------------------------------------------
    # the actual work
    # ------------------------------------------------------------------

    async def _seed_prices(self) -> None:
        """Populate prices for recovered positions before anything is managed."""

        if not self.portfolio or not self.portfolio.all():
            return
        for position in self.portfolio.all():
            try:
                ticker = await self.market_data.ticker(position.symbol)
                self.portfolio.set_price(position.symbol, ticker.last)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not price recovered position %s: %s", position.symbol, exc)

    async def _sync_account(self) -> None:
        if self.settings.trading_mode is TradingMode.LIVE:
            try:
                balance = await self.exchange.balance(self.settings.quote_currency)
                self.portfolio.sync_balance(
                    balance.equity, balance.available, balance.used_margin
                )
                log.info(
                    "live account: equity %.2f available %.2f",
                    balance.equity,
                    balance.available,
                )
            except ExchangeError as exc:
                log.error("could not read the live account: %s", exc)
        else:
            self.portfolio.mark_to_market()

    async def scan_once(self) -> list[Opportunity]:
        """One full cycle: discover, analyse, rank, and act on the best."""

        assert self.scanner and self.signal_engine and self.portfolio

        analyses = await self.scanner.scan()
        self.stats.scans += 1
        self.last_analyses = {a.symbol: a for a in analyses}

        # Feed correlation with the context series we already fetched.
        series = {
            a.symbol: a.timeframes[Timeframe.H1].candles
            for a in analyses
            if Timeframe.H1 in a.timeframes
        }
        self.portfolio.update_correlations(series)
        for analysis in analyses:
            self.portfolio.set_price(analysis.symbol, analysis.ticker.last)

        proposals: list[tuple[SymbolAnalysis, TradeProposal]] = []
        for analysis in analyses:
            proposal = self.signal_engine.evaluate(analysis)
            proposals.append((analysis, proposal))
            self.stats.proposals += 1
            if self.repositories:
                try:
                    self.repositories.signals.record(proposal.as_record())
                except Exception as exc:  # noqa: BLE001
                    log.debug("could not record signal: %s", exc)

        opportunities = rank_opportunities(proposals)
        self.last_opportunities = opportunities
        if self.repositories:
            try:
                self.repositories.opportunities.record_batch(
                    [o.as_row() for o in opportunities]
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("could not record opportunities: %s", exc)

        await self._sync_account()
        await self._assess_opportunities(opportunities)
        await self._consider_entries(opportunities)
        # Re-mark after any entry so margin and exposure reflect what was opened.
        self.portfolio.mark_to_market()
        return opportunities

    async def _assess_opportunities(
        self, opportunities: Sequence[Opportunity], limit: int = 8
    ) -> None:
        """Run the intelligence layer over the best-ranked opportunities.

        This happens on every scan, not only when something is tradable.  Two
        reasons: the operator wants to see *why* nothing was taken, and the
        decision journal is only honest if it records the refusals as well as
        the entries.
        """

        if self.intelligence is None or not self.portfolio:
            return

        state = self.portfolio.state()
        for opportunity in list(opportunities)[:limit]:
            await self._intelligence_verdict(opportunity, state)

    async def _intelligence_verdict(
        self, opportunity: Opportunity, state: Any
    ) -> IntelligenceVerdict | None:
        """Run the advanced layer over an opportunity, tolerating its failure.

        A crash inside the intelligence layer must never *enable* a trade that
        would otherwise be blocked, and must never block the whole scan.  On
        failure we return ``None``, which leaves the pre-upgrade behaviour --
        the signal engine and risk engine decide on their own.
        """

        if self.intelligence is None:
            return None

        analysis = self.last_analyses.get(opportunity.symbol)
        if analysis is None:
            return None

        proposal = opportunity.proposal

        # The scan already assessed the top opportunities.  Re-using that
        # verdict keeps the entry decision consistent with what the operator
        # was shown, and stops the journal recording the same decision twice.
        cached = self.last_verdicts.get(opportunity.symbol)
        if cached is not None and self._verdict_for.get(opportunity.symbol) == proposal.ts:
            return cached

        notional = 0.0
        if proposal.entry > 0 and proposal.stop_loss > 0:
            stop_distance = abs(proposal.entry - proposal.stop_loss)
            if stop_distance > 0:
                risk_cash = state.equity * self.settings.default_risk_per_trade
                notional = risk_cash / stop_distance * proposal.entry

        try:
            verdict = self.intelligence.evaluate(
                analysis=analysis,
                proposal=proposal,
                portfolio_state=state,
                portfolio_risk=self.portfolio.portfolio_risk if self.portfolio else None,
                market_data_health=(
                    self.market_data.health() if self.market_data else None
                ),
                exchange_health=self.last_health,
                latency_ms=(
                    self.market_data.stats.median_latency_ms if self.market_data else 0.0
                ),
                notional=notional,
            )
        except Exception as exc:  # noqa: BLE001
            self.stats.errors += 1
            log.exception(
                "intelligence layer failed for %s: %s", opportunity.symbol, exc
            )
            return None

        self.last_verdicts[opportunity.symbol] = verdict
        self._verdict_for[opportunity.symbol] = proposal.ts
        if len(self.last_verdicts) > 60:
            for symbol in list(self.last_verdicts)[:-60]:
                del self.last_verdicts[symbol]
                self._verdict_for.pop(symbol, None)
        return verdict

    async def _consider_entries(self, opportunities: Sequence[Opportunity]) -> None:
        assert self.risk_engine and self.portfolio and self.execution

        if self.state is not BotState.RUNNING:
            return

        state = self.portfolio.state()
        breaker = self.risk_engine.check_breakers(state)
        if breaker.blocked:
            log.info("entries blocked: %s", breaker.summary().replace("\n", " | "))
            return

        contracts = await self.exchange.contracts()

        for opportunity in opportunities:
            if not opportunity.tradable:
                continue
            if len(self.portfolio.all()) >= self.settings.max_open_positions:
                break

            spec = contracts.get(opportunity.symbol)
            if spec is None:
                continue

            state = self.portfolio.state()

            # The intelligence layer runs *before* sizing so that a veto costs
            # nothing, and so its size opinion can only shrink what the risk
            # engine would otherwise have allowed.
            verdict = await self._intelligence_verdict(opportunity, state)
            if verdict is not None and not verdict.approved:
                log.info(
                    "intelligence declined %s: %s",
                    opportunity.symbol,
                    "; ".join(verdict.veto_reasons[:2]) or "proposal not an entry",
                )
                if self.repositories:
                    self.repositories.events.risk_event(
                        kind="INTELLIGENCE_VETO",
                        message=f"{opportunity.symbol}: "
                        + "; ".join(verdict.veto_reasons[:2]),
                        severity="INFO",
                        symbol=opportunity.symbol,
                        detail={
                            "decision_id": verdict.decision_id,
                            "quality": round(verdict.quality_score, 1),
                            "agreement": verdict.model_agreement,
                            "veto_reasons": verdict.veto_reasons,
                        },
                    )
                continue

            decision = self.risk_engine.evaluate(
                proposal=opportunity.proposal,
                state=state,
                spec=spec,
                breaker_report=breaker,
                intelligence_scale=(
                    verdict.size_multiplier if verdict is not None else 1.0
                ),
            )
            if not decision.approved:
                log.info(
                    "risk engine declined %s: %s",
                    opportunity.symbol,
                    "; ".join(decision.rejections[:2]),
                )
                if self.repositories:
                    self.repositories.events.risk_event(
                        kind="TRADE_DECLINED",
                        message=f"{opportunity.symbol}: {decision.rejections[:1]}",
                        severity="INFO",
                        symbol=opportunity.symbol,
                        detail=decision.as_dict(),
                    )
                continue

            analysis = self.last_analyses.get(opportunity.symbol)
            atr = analysis.primary.atr if analysis and analysis.primary else 0.0

            try:
                result = await self.execution.open_position(decision, spec, atr=atr)
            except Exception as exc:  # noqa: BLE001
                self.stats.errors += 1
                log.exception("could not open %s: %s", opportunity.symbol, exc)
                await self.notifier.alert(
                    f"🔴 Failed to open {opportunity.symbol}: {exc}"
                )
                continue

            if not result.ok or result.position is None:
                log.warning("entry failed for %s: %s", opportunity.symbol, result.error)
                continue

            self.stats.entries += 1
            if verdict is not None:
                self._entry_verdicts[opportunity.symbol] = verdict
            if self.intelligence is not None and analysis is not None:
                try:
                    context = self.intelligence.build_context(analysis)
                    self.intelligence.remember(analysis, context)
                except Exception as exc:  # noqa: BLE001
                    log.debug("could not store market memory: %s", exc)

            await self.notifier.trade_opened(
                position=result.position,
                proposal=opportunity.proposal,
                decision=decision,
                equity=state.equity,
                notes=result.notes,
            )

    async def manage_positions(self) -> list[PositionAction]:
        assert self.portfolio and self.position_manager and self.execution

        positions = self.portfolio.all()
        if not positions:
            return []

        contexts = await self._build_contexts(positions)
        prices = {
            symbol: context.price for symbol, context in contexts.items() if context.price > 0
        }
        self.portfolio.mark_to_market(prices)

        state = self.portfolio.state()
        breaker = self.risk_engine.check_breakers(state)
        reduction = self.risk_engine.risk_reduction_factor(state)

        actions = self.position_manager.evaluate_all(self.portfolio, contexts, reduction)
        contracts = await self.exchange.contracts()

        for action in actions:
            position = action.position
            spec = contracts.get(position.symbol)
            try:
                if action.kind is ActionKind.MOVE_STOP:
                    if self.position_manager.apply_stop(action):
                        self.portfolio.persist(position)
                        if action.notify:
                            await self.notifier.position_update(
                                position=position,
                                event="stop moved",
                                detail=action.reason,
                                price=action.price,
                            )
                    continue

                if not action.closes_anything:
                    continue

                if action.kind is ActionKind.TARGET_HIT:
                    self.position_manager.mark_target(action)

                result = await self.execution.close_position(
                    position,
                    fraction=action.close_fraction,
                    reason=action.reason,
                    spec=spec,
                )
                if not result.ok:
                    log.warning(
                        "could not close %s: %s", position.symbol, result.error
                    )
                    continue

                self.stats.exits += 1
                if position.quantity <= 1e-12:
                    self._attribute_close(position, action.reason)
                    await self.notifier.trade_closed(
                        position=position,
                        reason=action.reason,
                        exit_price=result.average_price,
                        pnl=position.realized_pnl - position.fees - position.funding,
                    )
                else:
                    await self.notifier.position_update(
                        position=position,
                        event=action.kind.value,
                        detail=action.reason,
                        price=result.average_price,
                    )
            except Exception as exc:  # noqa: BLE001 - keep managing the rest
                self.stats.errors += 1
                log.exception("position action failed for %s: %s", position.symbol, exc)

        if breaker.requires_flatten and self.state is not BotState.EMERGENCY:
            await self.emergency_stop("circuit breaker demanded a flatten")

        return actions

    def _attribute_close(self, position: Any, reason: str) -> None:
        """Feed a finished trade back into the models that voted for it.

        Attribution is what makes the dynamic weighting mean anything: a model
        that keeps being right on trades it voted for earns weight, one that
        does not loses it.  Failure here is logged and swallowed -- learning is
        never allowed to interfere with closing a position.
        """

        verdict = self._entry_verdicts.pop(position.symbol, None)
        if self.intelligence is None or verdict is None:
            return

        net = position.realized_pnl - position.fees - position.funding
        risk = position.risk_amount
        if risk <= 0:
            return
        r_multiple = net / risk

        try:
            self.intelligence.record_outcome(
                decision_id=verdict.decision_id,
                symbol=position.symbol,
                ensemble=verdict.ensemble,
                r_multiple=r_multiple,
                regime=(
                    verdict.ensemble.regime if verdict.ensemble is not None else "ALL"
                ),
                lesson=reason,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("could not attribute %s outcome: %s", position.symbol, exc)

    async def _build_contexts(
        self, positions: Sequence[ManagedPosition]
    ) -> dict[str, ManagementContext]:
        contexts: dict[str, ManagementContext] = {}
        for position in positions:
            symbol = position.symbol
            analysis = self.last_analyses.get(symbol)
            price = 0.0
            try:
                ticker = await self.market_data.ticker(symbol)
                price = ticker.last
            except Exception as exc:  # noqa: BLE001
                log.warning("no fresh price for %s: %s", symbol, exc)
                price = self.portfolio.price(symbol)

            atr = 0.0
            candles: Sequence[Any] = []
            structure_bias = Bias.NEUTRAL
            regime = Regime.UNKNOWN
            swing = None
            volatility_ratio = 1.0
            anomaly = ""
            fundamental_danger = False
            fundamental_reason = ""

            if analysis is not None and analysis.primary is not None:
                primary = analysis.primary
                atr = primary.atr
                candles = primary.candles
                structure_bias = primary.structure.bias
                regime = analysis.regime.regime
                volatility_ratio = analysis.regime.volatility_ratio
                anomaly = analysis.anomaly
                if analysis.fundamentals is not None and analysis.fundamentals.danger:
                    fundamental_danger = True
                    reasons = analysis.fundamentals.danger_reasons
                    fundamental_reason = reasons[0] if reasons else ""
                swing = _protective_swing(primary, position.side)
            else:
                atr = float(position.meta.get("atr") or 0.0)

            high = candles[-1].high if candles else price
            low = candles[-1].low if candles else price

            contexts[symbol] = ManagementContext(
                price=price,
                atr=atr,
                high=max(high, price),
                low=min(low, price),
                candles=list(candles[-80:]),
                structure_bias=structure_bias,
                regime=regime,
                protective_swing=swing,
                fundamental_danger=fundamental_danger,
                fundamental_reason=fundamental_reason,
                anomaly=anomaly,
                volatility_ratio=volatility_ratio,
            )
        return contexts

    # ------------------------------------------------------------------
    # views (shared by Telegram and the dashboard)
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        uptime = time.time() - self.stats.started_at if self.stats.started_at else 0.0
        snapshot = self.portfolio.snapshot() if self.portfolio else {}
        return {
            "state": self.state.value,
            "emoji": self.state.emoji,
            "mode": self.settings.trading_mode.value,
            "data_source": self.settings.data_source,
            "live_enabled": self.live_enabled,
            "pause_reason": self._pause_reason,
            "uptime_seconds": int(uptime),
            "instance": self.settings.instance_name,
            "scans": self.stats.scans,
            "entries": self.stats.entries,
            "exits": self.stats.exits,
            "errors": self.stats.errors,
            "last_error": self.stats.last_error,
            "last_scan_duration": round(self.stats.last_scan_duration, 2),
            "health": self.last_health.state.value if self.last_health else "UNKNOWN",
            "account": {
                k: snapshot.get(k)
                for k in (
                    "equity",
                    "balance",
                    "available",
                    "used_margin",
                    "unrealized_pnl",
                    "realized_pnl_day",
                    "open_positions",
                    "drawdown",
                )
            },
        }

    def scanner_view(self, limit: int = 20) -> list[dict[str, Any]]:
        return [o.as_dict() for o in self.last_opportunities[:limit]]

    def positions_view(self) -> list[dict[str, Any]]:
        if not self.portfolio:
            return []
        return [
            p.summary(self.portfolio.price(p.symbol)) for p in self.portfolio.all()
        ]

    def account_view(self) -> dict[str, Any]:
        return self.portfolio.snapshot() if self.portfolio else {}

    def risk_view(self) -> dict[str, Any]:
        if not (self.risk_engine and self.portfolio):
            return {}
        return self.risk_engine.describe(self.portfolio.state())

    def ai_view(self) -> dict[str, Any]:
        info = self.predictor.info() if self.predictor else {}
        recent = []
        if self.repositories:
            try:
                recent = self.repositories.signals.recent(limit=10)
            except Exception:  # noqa: BLE001
                recent = []
        return {
            "model": info,
            "ml_enabled": self.settings.ml_enabled,
            "ml_weight": self.settings.ml_weight,
            "min_confidence": self.settings.min_confidence,
            "recent_signals": [
                {
                    "symbol": r.get("symbol"),
                    "decision": r.get("decision"),
                    "confidence": r.get("confidence"),
                    "p_long": r.get("p_long"),
                    "p_short": r.get("p_short"),
                    "p_no_trade": r.get("p_no_trade"),
                    "reasons": r.get("reasons", [])[:4],
                    "rejections": r.get("rejections", [])[:2],
                }
                for r in recent
            ],
        }

    # -- intelligence views ------------------------------------------------

    def intelligence_view(self) -> dict[str, Any]:
        """Model weights, journal and memory statistics."""

        if self.intelligence is None:
            return {"enabled": False}
        view = self.intelligence.describe()
        view["enabled"] = True
        view["thresholds"] = {
            "min_trade_quality": self.settings.min_trade_quality,
            "min_expected_value_r": self.settings.min_expected_value_r,
            "min_model_agreement": self.settings.min_model_agreement,
            "max_anomaly_severity": self.settings.max_anomaly_severity,
            "max_slippage_pct": self.settings.max_slippage_pct,
        }
        return view

    def verdicts_view(self, limit: int = 10) -> list[dict[str, Any]]:
        """Most recent intelligence verdicts, newest first."""

        verdicts = sorted(
            self.last_verdicts.values(), key=lambda v: v.ts, reverse=True
        )
        return [v.as_dict() for v in verdicts[:limit]]

    def verdict_view(self, symbol: str) -> dict[str, Any]:
        verdict = self.last_verdicts.get(symbol.upper())
        return verdict.as_dict() if verdict else {}

    def flow_view(self, limit: int = 10) -> list[dict[str, Any]]:
        """Order flow, microstructure and derivatives per recently-seen symbol."""

        out: list[dict[str, Any]] = []
        verdicts = sorted(
            self.last_verdicts.values(), key=lambda v: v.ts, reverse=True
        )
        for verdict in verdicts[:limit]:
            out.append(
                {
                    "symbol": verdict.symbol,
                    "order_flow": (
                        verdict.order_flow.as_dict() if verdict.order_flow else None
                    ),
                    "microstructure": (
                        verdict.microstructure.as_dict()
                        if verdict.microstructure
                        else None
                    ),
                    "liquidity": (
                        verdict.liquidity.as_dict() if verdict.liquidity else None
                    ),
                    "derivatives": (
                        verdict.derivatives.as_dict() if verdict.derivatives else None
                    ),
                }
            )
        return out

    def journal_view(self, limit: int = 15) -> list[dict[str, Any]]:
        if self.intelligence is None:
            return []
        return [entry.as_dict() for entry in self.intelligence.journal.recent(limit)]

    def performance_view(self, days: int = 30) -> dict[str, Any]:
        if not self.repositories:
            return {}
        since = int(time.time()) - days * 86400
        trades = self.repositories.trades.closed_since(
            since, mode=self.settings.trading_mode.value
        )
        if not trades:
            return {"trades": 0, "note": "no closed trades yet"}

        pnls = [float(t.get("realized_pnl") or 0.0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        r_values = [float(t.get("r_multiple") or 0.0) for t in trades]

        return {
            "period_days": days,
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades), 4),
            "total_pnl": round(sum(pnls), 4),
            "profit_factor": (
                round(gross_profit / gross_loss, 3) if gross_loss > 0 else None
            ),
            "expectancy_r": round(sum(r_values) / len(r_values), 4) if r_values else 0.0,
            "average_win": round(sum(wins) / len(wins), 4) if wins else 0.0,
            "average_loss": round(sum(losses) / len(losses), 4) if losses else 0.0,
            "best": round(max(pnls), 4),
            "worst": round(min(pnls), 4),
            "fees": round(sum(float(t.get("fees") or 0.0) for t in trades), 4),
        }

    def trades_view(self, limit: int = 15) -> list[dict[str, Any]]:
        if not self.repositories:
            return []
        return self.repositories.trades.recent(
            limit=limit, mode=self.settings.trading_mode.value
        )

    def orders_view(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.repositories:
            return []
        return self.repositories.orders.recent(limit=limit)

    def signals_view(self, limit: int = 15) -> list[dict[str, Any]]:
        if not self.repositories:
            return []
        return self.repositories.signals.recent(limit=limit)

    def health_view(self) -> dict[str, Any]:
        if self.last_health is None:
            return {"state": "UNKNOWN", "components": []}
        return self.last_health.as_dict()

    def preflight_view(self) -> dict[str, Any]:
        if self.preflight_report is None:
            return {"passed": None, "checks": []}
        return {
            "passed": self.preflight_report.passed,
            "summary": self.preflight_report.summary(),
            "checks": [
                {
                    "name": c.name,
                    "required": c.required,
                    "passed": c.passed,
                    "detail": c.detail,
                }
                for c in self.preflight_report.checks
            ],
        }


def _protective_swing(primary: Any, side: Side) -> float | None:
    """Most recent swing that could carry a risk-reducing stop."""

    structure = primary.structure
    for swing in reversed(structure.swings):
        if side is Side.LONG and swing.is_low:
            return swing.price
        if side is Side.SHORT and swing.is_high:
            return swing.price
    return None


__all__ = ["TradingEngine", "NullNotifier", "EngineStats"]
