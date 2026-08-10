"""MEXC contract WebSocket client.

Endpoint and message shapes follow the official MEXC contract WebSocket docs:

* connect to ``wss://contract.mexc.com/edge``
* subscribe with ``{"method": "sub.ticker", "param": {"symbol": "BTC_USDT"}}``
* keep alive with ``{"method": "ping"}`` roughly every 15 seconds
* private streams require ``{"method": "login", "param": {...}}`` first, where
  the signature is ``HMAC_SHA256(secret, apiKey + reqTime)``

The client reconnects with exponential backoff and republishes subscriptions.
It is strictly a *latency optimisation*: everything it delivers is also
available over REST, so the bot degrades to REST-only rather than stopping when
the socket is down (the health monitor reports the degradation).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import random
import time
from typing import Any, Awaitable, Callable, Iterable

from app.compat import HAVE_WEBSOCKETS, websockets
from app.domain import Candle, Ticker, Timeframe, mexc_symbol, normalise_symbol
from app.logger import get_logger

log = get_logger(__name__)

_WS_INTERVALS: dict[Timeframe, str] = {
    Timeframe.M1: "Min1",
    Timeframe.M5: "Min5",
    Timeframe.M15: "Min15",
    Timeframe.M30: "Min30",
    Timeframe.H1: "Min60",
    Timeframe.H4: "Hour4",
    Timeframe.D1: "Day1",
}

TickerHandler = Callable[[Ticker], Any]
CandleHandler = Callable[[str, Timeframe, Candle], Any]


class MexcWebSocket:
    """Resilient market-data socket."""

    def __init__(
        self,
        url: str = "wss://contract.mexc.com/edge",
        access_key: str = "",
        secret_key: str = "",
        ping_interval: float = 15.0,
        max_backoff: float = 60.0,
    ) -> None:
        self.url = url
        self._access_key = access_key
        self._secret_key = secret_key
        self.ping_interval = ping_interval
        self.max_backoff = max_backoff

        self._connection: Any = None
        self._task: asyncio.Task | None = None
        self._ping_task: asyncio.Task | None = None
        self._running = False
        self._subscriptions: set[tuple[str, str, str]] = set()
        self._ticker_handlers: list[TickerHandler] = []
        self._candle_handlers: list[CandleHandler] = []

        self.connected = False
        self.last_message_at = 0.0
        self.reconnects = 0
        self.messages = 0
        self.last_error = ""

    # -- handlers ---------------------------------------------------------

    def on_ticker(self, handler: TickerHandler) -> None:
        self._ticker_handlers.append(handler)

    def on_candle(self, handler: CandleHandler) -> None:
        self._candle_handlers.append(handler)

    # -- lifecycle --------------------------------------------------------

    @property
    def available(self) -> bool:
        return HAVE_WEBSOCKETS

    async def start(self) -> None:
        if not HAVE_WEBSOCKETS:
            log.warning(
                "the 'websockets' package is not installed - running REST-only; "
                "install it for lower-latency price updates"
            )
            return
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="mexc-ws")

    async def stop(self) -> None:
        self._running = False
        for task in (self._ping_task, self._task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._ping_task = None
        self._task = None
        if self._connection is not None:
            with contextlib.suppress(Exception):
                await self._connection.close()
            self._connection = None
        self.connected = False

    # -- subscriptions ----------------------------------------------------

    async def subscribe_tickers(self, symbols: Iterable[str]) -> None:
        for symbol in symbols:
            native = mexc_symbol(symbol)
            self._subscriptions.add(("sub.ticker", native, ""))
        await self._flush_subscriptions()

    async def subscribe_candles(self, symbol: str, timeframe: Timeframe) -> None:
        interval = _WS_INTERVALS.get(timeframe)
        if interval is None:
            return
        self._subscriptions.add(("sub.kline", mexc_symbol(symbol), interval))
        await self._flush_subscriptions()

    async def _flush_subscriptions(self) -> None:
        if self._connection is None or not self.connected:
            return
        for method, symbol, interval in sorted(self._subscriptions):
            param: dict[str, Any] = {"symbol": symbol}
            if interval:
                param["interval"] = interval
            await self._send({"method": method, "param": param})

    # -- internals --------------------------------------------------------

    async def _send(self, message: dict[str, Any]) -> None:
        if self._connection is None:
            return
        try:
            await self._connection.send(json.dumps(message, separators=(",", ":")))
        except Exception as exc:
            self.last_error = str(exc)
            log.debug("websocket send failed: %s", exc)

    async def _login(self) -> None:
        if not (self._access_key and self._secret_key):
            return
        request_time = str(int(time.time() * 1000))
        signature = hmac.new(
            self._secret_key.encode("utf-8"),
            f"{self._access_key}{request_time}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        await self._send(
            {
                "method": "login",
                "param": {
                    "apiKey": self._access_key,
                    "reqTime": request_time,
                    "signature": signature,
                },
            }
        )

    async def _heartbeat(self) -> None:
        while self._running and self.connected:
            await asyncio.sleep(self.ping_interval)
            await self._send({"method": "ping"})

    async def _run(self) -> None:
        attempt = 0
        while self._running:
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=None,       # MEXC uses an application-level ping
                    close_timeout=5,
                    max_queue=512,
                ) as connection:
                    self._connection = connection
                    self.connected = True
                    attempt = 0
                    log.info("websocket connected: %s", self.url)
                    await self._login()
                    await self._flush_subscriptions()
                    self._ping_task = asyncio.create_task(self._heartbeat())
                    async for raw in connection:
                        self.messages += 1
                        self.last_message_at = time.time()
                        self._dispatch(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc)
                log.warning("websocket disconnected: %s", exc)
            finally:
                self.connected = False
                self._connection = None
                if self._ping_task is not None:
                    self._ping_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._ping_task
                    self._ping_task = None

            if not self._running:
                break
            self.reconnects += 1
            delay = min(self.max_backoff, 2 ** min(attempt, 6))
            delay = random.uniform(delay * 0.5, delay)
            attempt += 1
            log.info("reconnecting websocket in %.1fs", delay)
            await asyncio.sleep(delay)

    def _dispatch(self, raw: str | bytes) -> None:
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return
        if not isinstance(message, dict):
            return
        channel = str(message.get("channel", ""))
        data = message.get("data")
        symbol = message.get("symbol")

        if channel in {"pong", "rs.login", "rs.error"}:
            if channel == "rs.error":
                self.last_error = str(data)
                log.warning("websocket error frame: %s", data)
            return

        if channel == "push.ticker" and isinstance(data, dict):
            ticker = parse_ws_ticker(data)
            if ticker is not None:
                self._emit_ticker(ticker)
            return

        if channel == "push.kline" and isinstance(data, dict):
            parsed = parse_ws_kline(data, symbol)
            if parsed is not None:
                sym, timeframe, candle = parsed
                for handler in self._candle_handlers:
                    _safe_call(handler, sym, timeframe, candle)

    def _emit_ticker(self, ticker: Ticker) -> None:
        for handler in self._ticker_handlers:
            _safe_call(handler, ticker)

    # -- diagnostics ------------------------------------------------------

    def health(self) -> dict[str, Any]:
        if not HAVE_WEBSOCKETS:
            return {"ok": False, "reason": "websockets package not installed"}
        age = time.time() - self.last_message_at if self.last_message_at else None
        return {
            "ok": self.connected and (age is None or age < 60),
            "connected": self.connected,
            "messages": self.messages,
            "reconnects": self.reconnects,
            "seconds_since_message": round(age, 1) if age is not None else None,
            "subscriptions": len(self._subscriptions),
            "last_error": self.last_error,
        }


def _safe_call(handler: Callable, *args: Any) -> None:
    try:
        result = handler(*args)
        if asyncio.iscoroutine(result):
            asyncio.create_task(result)
    except Exception as exc:  # a bad handler must not kill the socket
        log.warning("websocket handler failed: %s", exc)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_ws_ticker(data: dict[str, Any]) -> Ticker | None:
    symbol = data.get("symbol")
    last = _as_float(data.get("lastPrice"))
    if not symbol or last <= 0:
        return None
    bid = _as_float(data.get("bid1")) or last
    ask = _as_float(data.get("ask1")) or last
    timestamp = _as_float(data.get("timestamp"))
    return Ticker(
        symbol=normalise_symbol(str(symbol)),
        last=last,
        bid=bid,
        ask=ask,
        volume_24h=_as_float(data.get("volume24")),
        quote_volume_24h=_as_float(data.get("amount24")),
        change_24h_pct=_as_float(data.get("riseFallRate")),
        high_24h=_as_float(data.get("high24Price")),
        low_24h=_as_float(data.get("lower24Price")),
        funding_rate=_as_float(data.get("fundingRate")) or None,
        open_interest=_as_float(data.get("holdVol")) or None,
        ts=int(timestamp / 1000) if timestamp > 1e11 else int(timestamp or time.time()),
    )


_INTERVAL_TO_TF = {value: key for key, value in _WS_INTERVALS.items()}


def parse_ws_kline(
    data: dict[str, Any], symbol: Any = None
) -> tuple[str, Timeframe, Candle] | None:
    interval = str(data.get("interval", ""))
    timeframe = _INTERVAL_TO_TF.get(interval)
    sym = symbol or data.get("symbol")
    if timeframe is None or not sym:
        return None
    ts = int(_as_float(data.get("t") or data.get("time")))
    if ts > 1e11:
        ts //= 1000
    if ts <= 0:
        return None
    close_price = _as_float(data.get("c") or data.get("close"))
    if close_price <= 0:
        return None
    # The bar is closed once its window has fully elapsed.
    closed = ts + timeframe.seconds <= int(time.time())
    candle = Candle(
        ts=ts,
        open=_as_float(data.get("o") or data.get("open")),
        high=_as_float(data.get("h") or data.get("high")),
        low=_as_float(data.get("l") or data.get("low")),
        close=close_price,
        volume=_as_float(data.get("q") or data.get("vol")),
        quote_volume=_as_float(data.get("a") or data.get("amount")),
        closed=closed,
    )
    return normalise_symbol(str(sym)), timeframe, candle


__all__ = ["MexcWebSocket", "parse_ws_ticker", "parse_ws_kline"]
