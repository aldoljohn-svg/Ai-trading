"""MEXC Futures (contract) API adapter.

Only endpoints documented in the official MEXC Contract V1 API are used.

Authentication (per MEXC contract docs)
--------------------------------------
Every private request carries these headers::

    ApiKey:       <access key>
    Request-Time: <epoch milliseconds>
    Recv-Window:  <seconds, optional>
    Signature:    HMAC_SHA256(secret, accessKey + reqTime + paramString)
    Content-Type: application/json

where ``paramString`` is

* for **GET/DELETE**: the query string with parameters sorted by key ascending,
  rendered as ``k=v&k2=v2`` (no leading ``?``);
* for **POST**: the exact JSON body string that is sent.

Operational note
----------------
MEXC has, for extended periods, restricted *futures order placement* over the
API to whitelisted accounts, returning a "maintenance"/contract-system error
for everyone else while leaving market data and read-only account endpoints
working.  This adapter surfaces that condition as
:class:`~app.exchange.base.ExchangeNotSupported` rather than pretending the
order succeeded.  PAPER and BACKTEST modes are unaffected.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import random
import time
from typing import Any, Mapping

from app.domain import (
    Balance,
    Candle,
    ContractSpec,
    ExchangeOrder,
    ExchangePosition,
    Fill,
    OrderBook,
    OrderIntent,
    OrderStatus,
    OrderType,
    Side,
    Ticker,
    Timeframe,
    mexc_symbol,
    normalise_symbol,
)
from app.exchange.base import (
    BaseExchange,
    ExchangeAuthError,
    ExchangeError,
    ExchangeNotSupported,
    ExchangeRateLimit,
    ExchangeUnavailable,
)
from app.exchange.http_client import AsyncHttpClient
from app.logger import get_logger, register_secret

log = get_logger(__name__)

#: MEXC contract kline interval names.
INTERVALS: dict[Timeframe, str] = {
    Timeframe.M1: "Min1",
    Timeframe.M5: "Min5",
    Timeframe.M15: "Min15",
    Timeframe.M30: "Min30",
    Timeframe.H1: "Min60",
    Timeframe.H4: "Hour4",
    Timeframe.D1: "Day1",
}

# MEXC contract order side codes.
SIDE_OPEN_LONG = 1
SIDE_CLOSE_SHORT = 2
SIDE_OPEN_SHORT = 3
SIDE_CLOSE_LONG = 4

# MEXC contract order type codes.
TYPE_LIMIT = 1
TYPE_POST_ONLY = 2
TYPE_IOC = 3
TYPE_FOK = 4
TYPE_MARKET = 5

OPEN_TYPE_ISOLATED = 1
OPEN_TYPE_CROSS = 2

# MEXC contract order states.
_ORDER_STATE: dict[int, OrderStatus] = {
    1: OrderStatus.NEW,             # uninformed / not yet matched
    2: OrderStatus.PARTIALLY_FILLED,
    3: OrderStatus.FILLED,
    4: OrderStatus.CANCELED,
    5: OrderStatus.REJECTED,        # invalid
}

# MEXC contract "state" for a symbol: 0 enabled, others not tradable.
_TRADABLE_STATE = 0

_MAINTENANCE_MARKERS = (
    "maintenance",
    "system busy",
    "contract system",
    "not open yet",
    "temporarily",
)

#: MEXC reports throttling in the *body* of an HTTP 200 response rather than
#: with a 429, so the transport's rate-limit handling in
#: :class:`~app.exchange.http_client.AsyncHttpClient` never sees it.  Without
#: these markers "Requests are too frequent" arrived as a generic
#: :class:`ExchangeError`, indistinguishable from "this symbol has no book" --
#: which made the scanner bench liquid majors for half an hour because *we*
#: asked too fast.
_RATE_LIMIT_MARKERS = (
    "too frequent",
    "too many request",
    "rate limit",
    "frequency limit",
    "request frequency",
)

#: Extra attempts made when MEXC says "slow down".  Deliberately small: the
#: real fix is asking less often (see the token-bucket costs below and the
#: order-book throttle in :mod:`app.data.market_data`), not retrying harder.
_RATE_LIMIT_RETRIES = 2
_RATE_LIMIT_BACKOFF = 0.6


class MexcFuturesExchange(BaseExchange):
    """Adapter for ``https://contract.mexc.com``."""

    name = "mexc-futures"
    supports_trading = True

    def __init__(
        self,
        access_key: str = "",
        secret_key: str = "",
        base_url: str = "https://contract.mexc.com",
        recv_window: int = 30,
        timeout: float = 15.0,
        quote: str = "USDT",
        allow_trading: bool = False,
    ) -> None:
        self._access_key = access_key
        self._secret_key = secret_key
        self.recv_window = recv_window
        self.quote = quote
        #: Hard gate: even a fully authenticated client refuses to send orders
        #: unless the orchestrator explicitly enabled trading for LIVE mode.
        self.allow_trading = allow_trading
        self.http = AsyncHttpClient(
            base_url=base_url,
            timeout=timeout,
            rate_limit_per_second=15.0,
            burst=20,
        )
        self._contracts: dict[str, ContractSpec] = {}
        self._contracts_fetched_at = 0.0
        self._time_offset_ms = 0
        if secret_key:
            register_secret(secret_key)
        if access_key:
            register_secret(access_key)

    # -- lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        await self.http.connect()

    async def close(self) -> None:
        await self.http.close()

    @property
    def authenticated(self) -> bool:
        return bool(self._access_key and self._secret_key)

    # -- signing ----------------------------------------------------------

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000) + self._time_offset_ms

    def _param_string_get(self, params: Mapping[str, Any]) -> str:
        if not params:
            return ""
        items = sorted((str(k), _stringify(v)) for k, v in params.items() if v is not None)
        return "&".join(f"{k}={v}" for k, v in items)

    def sign(self, param_string: str, timestamp_ms: int) -> str:
        """HMAC-SHA256 over ``accessKey + reqTime + paramString``."""

        if not self._secret_key:
            raise ExchangeAuthError("MEXC secret key is not configured")
        payload = f"{self._access_key}{timestamp_ms}{param_string}"
        return hmac.new(
            self._secret_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self, param_string: str) -> dict[str, str]:
        if not self.authenticated:
            raise ExchangeAuthError(
                "this endpoint requires MEXC_ACCESS_KEY and MEXC_SECRET_KEY"
            )
        timestamp = self._timestamp_ms()
        return {
            "ApiKey": self._access_key,
            "Request-Time": str(timestamp),
            "Recv-Window": str(self.recv_window),
            "Signature": self.sign(param_string, timestamp),
            "Content-Type": "application/json",
        }

    # -- envelope ---------------------------------------------------------

    @staticmethod
    def _unwrap(payload: Any, context: str) -> Any:
        """Validate the ``{success, code, data}`` envelope MEXC returns."""

        if not isinstance(payload, dict):
            raise ExchangeError(f"{context}: unexpected response type {type(payload)}")
        if payload.get("success") is True or payload.get("code") in (0, 200):
            return payload.get("data")
        message = str(payload.get("message") or payload.get("msg") or payload)
        lowered = message.lower()
        if any(marker in lowered for marker in _RATE_LIMIT_MARKERS):
            raise ExchangeRateLimit(f"{context}: {message}")
        if any(marker in lowered for marker in _MAINTENANCE_MARKERS):
            raise ExchangeNotSupported(
                f"{context}: MEXC reports this endpoint unavailable for this "
                f"account ({message}). Futures order placement over the API is "
                "restricted to whitelisted accounts on MEXC."
            )
        if payload.get("code") in (401, 403, 602, 1002, 10007):
            raise ExchangeAuthError(f"{context}: {message}")
        raise ExchangeError(f"{context}: {message}")

    async def _public(
        self, path: str, params: Mapping[str, Any] | None = None, cost: float = 1.0
    ) -> Any:
        last: ExchangeRateLimit | None = None
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            payload = await self.http.get_json(path, params=params, cost=cost)
            try:
                return self._unwrap(payload, f"GET {path}")
            except ExchangeRateLimit as exc:
                last = exc
                if attempt == _RATE_LIMIT_RETRIES:
                    break
                # Jittered so a burst of symbols rate-limited together does not
                # come back in lockstep and trip the same limit again.
                delay = _RATE_LIMIT_BACKOFF * (2**attempt)
                await asyncio.sleep(delay * (0.5 + random.random()))
        assert last is not None
        raise last

    async def _private_get(
        self, path: str, params: Mapping[str, Any] | None = None, cost: float = 1.0
    ) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        headers = self._auth_headers(self._param_string_get(params))
        payload = await self.http.get_json(path, params=params, headers=headers, cost=cost)
        return self._unwrap(payload, f"GET {path}")

    async def _private_post(
        self, path: str, body: Mapping[str, Any], cost: float = 2.0
    ) -> Any:
        cleaned = {k: v for k, v in body.items() if v is not None}
        body_text = json.dumps(cleaned, separators=(",", ":"))
        headers = self._auth_headers(body_text)
        payload = await self.http.post_json(
            path, body=body_text, headers=headers, retry=False, cost=cost
        )
        return self._unwrap(payload, f"POST {path}")

    # -- public market data ----------------------------------------------

    async def ping(self) -> float:
        started = time.perf_counter()
        await self._public("/api/v1/contract/ping")
        return (time.perf_counter() - started) * 1000.0

    async def server_time(self) -> int:
        data = await self._public("/api/v1/contract/ping")
        # ``ping`` returns the server timestamp in milliseconds.
        if isinstance(data, (int, float)):
            return int(data) // 1000
        if isinstance(data, dict):
            for key in ("serverTime", "timestamp", "time"):
                if key in data:
                    return int(data[key]) // 1000
        return int(time.time())

    async def sync_clock(self) -> float:
        """Align local time with the venue.  Returns the drift in seconds."""

        local_ms = int(time.time() * 1000)
        server_seconds = await self.server_time()
        drift = server_seconds - local_ms / 1000.0
        # Only compensate for meaningful drift; small jitter is measurement noise.
        self._time_offset_ms = int(drift * 1000) if abs(drift) > 1.0 else 0
        return drift

    async def contracts(self, force: bool = False) -> dict[str, ContractSpec]:
        now = time.monotonic()
        if self._contracts and not force and now - self._contracts_fetched_at < 3600:
            return self._contracts
        data = await self._public("/api/v1/contract/detail", cost=2.0)
        specs: dict[str, ContractSpec] = {}
        for item in data or []:
            try:
                native = str(item["symbol"])
                quote = str(item.get("quoteCoin", "")).upper()
                if self.quote and quote and quote != self.quote:
                    continue
                symbol = normalise_symbol(native)
                price_unit = _as_float(item.get("priceUnit"), 0.0)
                price_scale = int(item.get("priceScale") or 2)
                if price_unit <= 0:
                    price_unit = 10 ** (-price_scale)
                specs[symbol] = ContractSpec(
                    symbol=symbol,
                    exchange_symbol=native,
                    base=str(item.get("baseCoin", "")).upper(),
                    quote=quote or self.quote,
                    contract_size=_as_float(item.get("contractSize"), 1.0) or 1.0,
                    price_scale=price_scale,
                    volume_scale=int(item.get("volScale") or 0),
                    min_volume=_as_float(item.get("minVol"), 1.0) or 1.0,
                    max_volume=_as_float(item.get("maxVol"), 1e9) or 1e9,
                    max_leverage=_as_float(item.get("maxLeverage"), 20.0) or 20.0,
                    maker_fee=_as_float(item.get("makerFeeRate"), 0.0002),
                    taker_fee=_as_float(item.get("takerFeeRate"), 0.0006),
                    price_unit=price_unit,
                    active=int(item.get("state", 0)) == _TRADABLE_STATE,
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.debug("skipping malformed contract entry: %s", exc)
        if not specs:
            raise ExchangeError("contract detail returned no usable symbols")
        self._contracts = specs
        self._contracts_fetched_at = now
        log.info("loaded %d MEXC %s contracts", len(specs), self.quote)
        return specs

    async def _native(self, symbol: str) -> str:
        specs = await self.contracts()
        spec = specs.get(normalise_symbol(symbol))
        return spec.exchange_symbol if spec else mexc_symbol(symbol, self.quote)

    async def tickers(self) -> dict[str, Ticker]:
        data = await self._public("/api/v1/contract/ticker", cost=2.0)
        out: dict[str, Ticker] = {}
        for item in data or []:
            ticker = _parse_ticker(item)
            if ticker is not None:
                out[ticker.symbol] = ticker
        return out

    async def ticker(self, symbol: str) -> Ticker:
        native = await self._native(symbol)
        data = await self._public("/api/v1/contract/ticker", params={"symbol": native})
        if isinstance(data, list):
            data = data[0] if data else None
        ticker = _parse_ticker(data) if data else None
        if ticker is None:
            raise ExchangeError(f"no ticker data for {symbol}")
        return ticker

    async def order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        native = await self._native(symbol)
        # Weighted like klines: depth is a per-symbol endpoint the scanner hits
        # once for every candidate, so charging it a single token let a wide
        # scan drain the venue's budget and come back "Requests are too
        # frequent" for majors that trade perfectly well.
        data = await self._public(
            f"/api/v1/contract/depth/{native}", params={"limit": depth}, cost=2.0
        )
        if not isinstance(data, dict):
            raise ExchangeError(f"unexpected depth payload for {symbol}")
        return OrderBook(
            symbol=normalise_symbol(symbol),
            bids=tuple(_parse_levels(data.get("bids"))),
            asks=tuple(_parse_levels(data.get("asks"))),
            ts=int(_as_float(data.get("timestamp"), 0.0) / 1000) or int(time.time()),
        )

    async def candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 500,
        end_ts: int | None = None,
    ) -> list[Candle]:
        interval = INTERVALS.get(timeframe)
        if interval is None:
            raise ExchangeError(f"timeframe {timeframe} is not supported by MEXC")
        native = await self._native(symbol)
        limit = max(1, min(int(limit), 2000))
        end = int(end_ts or time.time())
        # MEXC returns [start, end] inclusive; request one extra bar because the
        # newest bar is still forming and will be discarded below.
        start = end - timeframe.seconds * (limit + 2)
        data = await self._public(
            f"/api/v1/contract/kline/{native}",
            params={"interval": interval, "start": start, "end": end},
            cost=2.0,
        )
        return _parse_klines(data, timeframe, limit)

    async def funding_rate(self, symbol: str) -> float | None:
        native = await self._native(symbol)
        try:
            data = await self._public(f"/api/v1/contract/funding_rate/{native}")
        except ExchangeError as exc:
            log.debug("funding rate unavailable for %s: %s", symbol, exc)
            return None
        if isinstance(data, dict) and "fundingRate" in data:
            return _as_float(data["fundingRate"], 0.0)
        return None

    async def open_interest(self, symbol: str) -> float | None:
        try:
            ticker = await self.ticker(symbol)
        except ExchangeError:
            return None
        return ticker.open_interest

    # -- private ----------------------------------------------------------

    async def balance(self, currency: str = "USDT") -> Balance:
        data = await self._private_get(f"/api/v1/private/account/asset/{currency}")
        if not isinstance(data, dict):
            raise ExchangeError("unexpected asset payload")
        equity = _as_float(data.get("equity"), 0.0)
        available = _as_float(data.get("availableBalance"), 0.0)
        frozen = _as_float(data.get("frozenBalance"), 0.0)
        position_margin = _as_float(data.get("positionMargin"), 0.0)
        unrealized = _as_float(data.get("unrealized"), 0.0)
        if equity <= 0:
            equity = _as_float(data.get("cashBalance"), 0.0) + unrealized
        return Balance(
            currency=currency,
            equity=equity,
            available=available,
            used_margin=position_margin + frozen,
            unrealized_pnl=unrealized,
        )

    async def positions(self) -> list[ExchangePosition]:
        data = await self._private_get("/api/v1/private/position/open_positions")
        out: list[ExchangePosition] = []
        for item in data or []:
            try:
                volume = _as_float(item.get("holdVol"), 0.0)
                if volume <= 0:
                    continue
                # state: 1 holding, 2 system holding, 3 closed
                if int(item.get("state", 1)) == 3:
                    continue
                side = Side.LONG if int(item.get("positionType", 1)) == 1 else Side.SHORT
                out.append(
                    ExchangePosition(
                        symbol=normalise_symbol(str(item["symbol"])),
                        side=side,
                        quantity=volume,
                        entry_price=_as_float(item.get("holdAvgPrice"), 0.0),
                        leverage=_as_float(item.get("leverage"), 1.0) or 1.0,
                        unrealized_pnl=_as_float(item.get("realised"), 0.0),
                        margin=_as_float(item.get("im"), 0.0),
                        liquidation_price=_as_float(item.get("liquidatePrice"), 0.0) or None,
                        position_id=str(item.get("positionId", "")),
                        raw=dict(item),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("skipping malformed position entry: %s", exc)
        return out

    async def open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        native = await self._native(symbol) if symbol else None
        path = (
            f"/api/v1/private/order/list/open_orders/{native}"
            if native
            else "/api/v1/private/order/list/open_orders"
        )
        data = await self._private_get(path, params={"page_num": 1, "page_size": 100})
        return [o for o in (_parse_order(item) for item in data or []) if o is not None]

    async def order_status(self, order_id: str, symbol: str | None = None) -> ExchangeOrder:
        data = await self._private_get(f"/api/v1/private/order/get/{order_id}")
        order = _parse_order(data) if data else None
        if order is None:
            raise ExchangeError(f"order {order_id} not found")
        return order

    async def order_by_client_id(self, client_order_id: str) -> ExchangeOrder | None:
        """Look an order up by our own external id (used by reconciliation)."""

        try:
            data = await self._private_get(
                f"/api/v1/private/order/external/{client_order_id}"
            )
        except ExchangeError as exc:
            log.debug("client-id lookup failed for %s: %s", client_order_id, exc)
            return None
        return _parse_order(data) if data else None

    async def fills(self, order_id: str, symbol: str | None = None) -> list[Fill]:
        try:
            data = await self._private_get(
                f"/api/v1/private/order/deal_details/{order_id}"
            )
        except ExchangeError as exc:
            log.debug("fill lookup failed for %s: %s", order_id, exc)
            return []
        out: list[Fill] = []
        for item in data or []:
            try:
                out.append(
                    Fill(
                        order_id=str(item.get("orderId", order_id)),
                        symbol=normalise_symbol(str(item.get("symbol", symbol or ""))),
                        side=Side.LONG if int(item.get("side", 1)) in (1, 2) else Side.SHORT,
                        quantity=_as_float(item.get("vol"), 0.0),
                        price=_as_float(item.get("price"), 0.0),
                        fee=_as_float(item.get("fee"), 0.0),
                        ts=int(_as_float(item.get("timestamp"), 0.0) / 1000) or int(time.time()),
                        trade_id=str(item.get("id", "")),
                        is_maker=int(item.get("taker", 1)) == 0,
                    )
                )
            except (TypeError, ValueError):
                continue
        return out

    async def leverage(self, symbol: str) -> float | None:
        native = await self._native(symbol)
        try:
            data = await self._private_get(
                "/api/v1/private/position/leverage", params={"symbol": native}
            )
        except ExchangeError:
            return None
        if isinstance(data, list) and data:
            return _as_float(data[0].get("leverage"), 0.0) or None
        if isinstance(data, dict):
            return _as_float(data.get("leverage"), 0.0) or None
        return None

    # -- trading ----------------------------------------------------------

    def _assert_trading_allowed(self) -> None:
        if not self.allow_trading:
            raise ExchangeNotSupported(
                "live order placement is disabled on this exchange client; "
                "orders are only sent when TRADING_MODE=live and every "
                "pre-flight check has passed"
            )
        if not self.authenticated:
            raise ExchangeAuthError("cannot trade without MEXC API credentials")

    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        self._assert_trading_allowed()
        native = await self._native(symbol)
        specs = await self.contracts()
        spec = specs.get(normalise_symbol(symbol))
        capped = max(1, int(min(leverage, spec.max_leverage if spec else leverage)))
        try:
            await self._private_post(
                "/api/v1/private/position/change_leverage",
                {
                    "symbol": native,
                    "leverage": capped,
                    "openType": OPEN_TYPE_ISOLATED,
                    "positionType": 1,
                },
            )
            await self._private_post(
                "/api/v1/private/position/change_leverage",
                {
                    "symbol": native,
                    "leverage": capped,
                    "openType": OPEN_TYPE_ISOLATED,
                    "positionType": 2,
                },
            )
            return True
        except ExchangeError as exc:
            log.warning("could not set leverage for %s: %s", symbol, exc)
            return False

    async def place_order(
        self,
        symbol: str,
        side: Side,
        intent: OrderIntent,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        leverage: float | None = None,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> ExchangeOrder:
        self._assert_trading_allowed()

        specs = await self.contracts()
        spec = specs.get(normalise_symbol(symbol))
        if spec is None:
            raise ExchangeError(f"unknown contract {symbol}")
        if not spec.active:
            raise ExchangeError(f"contract {symbol} is not tradable right now")

        volume = spec.round_volume(quantity)
        if volume < spec.min_volume:
            raise ExchangeError(
                f"quantity {quantity} rounds to {volume} which is below the "
                f"{symbol} minimum of {spec.min_volume} contracts"
            )
        if volume > spec.max_volume:
            raise ExchangeError(
                f"quantity {volume} exceeds the {symbol} maximum of {spec.max_volume}"
            )

        if intent is OrderIntent.OPEN:
            side_code = SIDE_OPEN_LONG if side is Side.LONG else SIDE_OPEN_SHORT
        else:
            # Closing a long sells; closing a short buys.
            side_code = SIDE_CLOSE_LONG if side is Side.LONG else SIDE_CLOSE_SHORT

        if order_type is OrderType.MARKET:
            type_code = TYPE_MARKET
            order_price = None
        else:
            if price is None or price <= 0:
                raise ExchangeError("limit orders require a positive price")
            type_code = TYPE_LIMIT
            order_price = spec.round_price(price)

        body: dict[str, Any] = {
            "symbol": spec.exchange_symbol,
            "vol": volume,
            "side": side_code,
            "type": type_code,
            "openType": OPEN_TYPE_ISOLATED,
            "leverage": int(max(1, leverage or 1)),
        }
        if order_price is not None:
            body["price"] = order_price
        if client_order_id:
            body["externalOid"] = client_order_id[:32]
        if reduce_only or intent in (OrderIntent.CLOSE, OrderIntent.REDUCE):
            body["reduceOnly"] = True

        data = await self._private_post("/api/v1/private/order/submit", body)
        order_id = _extract_order_id(data)
        log.info(
            "submitted %s %s %s %s contracts (order id %s)",
            intent.value,
            side.value,
            symbol,
            volume,
            order_id,
        )
        return ExchangeOrder(
            order_id=order_id,
            symbol=normalise_symbol(symbol),
            side=side,
            intent=intent,
            order_type=order_type,
            quantity=volume,
            price=order_price,
            status=OrderStatus.NEW,
            client_order_id=client_order_id or "",
            reduce_only=bool(body.get("reduceOnly")),
            ts=int(time.time()),
            raw={"request": body, "response": data},
        )

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        self._assert_trading_allowed()
        try:
            # The cancel endpoint takes a JSON array of order ids.
            body_text = json.dumps([str(order_id)], separators=(",", ":"))
            headers = self._auth_headers(body_text)
            payload = await self.http.post_json(
                "/api/v1/private/order/cancel",
                body=body_text,
                headers=headers,
                retry=False,
                cost=2.0,
            )
            self._unwrap(payload, "POST /api/v1/private/order/cancel")
            return True
        except ExchangeError as exc:
            log.warning("cancel failed for order %s: %s", order_id, exc)
            return False

    async def cancel_all(self, symbol: str | None = None) -> int:
        self._assert_trading_allowed()
        body: dict[str, Any] = {}
        if symbol:
            body["symbol"] = await self._native(symbol)
        try:
            await self._private_post("/api/v1/private/order/cancel_all", body)
            return 1
        except ExchangeError as exc:
            log.warning("cancel_all failed: %s", exc)
            return 0

    # -- diagnostics ------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name, "authenticated": self.authenticated}
        try:
            result["latency_ms"] = round(await self.ping(), 2)
            result["ok"] = True
        except Exception as exc:
            result["ok"] = False
            result["error"] = str(exc)
            return result
        result["http"] = dict(self.http.stats)
        return result


# --------------------------------------------------------------------------
# Parsing helpers (module-level so they are unit testable without network)
# --------------------------------------------------------------------------


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_levels(levels: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for level in levels or []:
        try:
            price = float(level[0])
            size = float(level[1])
        except (TypeError, ValueError, IndexError):
            continue
        if price > 0 and size > 0:
            out.append((price, size))
    return out


def _parse_ticker(item: Any) -> Ticker | None:
    if not isinstance(item, dict) or "symbol" not in item:
        return None
    last = _as_float(item.get("lastPrice"), 0.0)
    if last <= 0:
        return None
    bid = _as_float(item.get("bid1"), 0.0)
    ask = _as_float(item.get("ask1"), 0.0)
    timestamp = _as_float(item.get("timestamp"), 0.0)
    return Ticker(
        symbol=normalise_symbol(str(item["symbol"])),
        last=last,
        bid=bid or last,
        ask=ask or last,
        volume_24h=_as_float(item.get("volume24"), 0.0),
        quote_volume_24h=_as_float(item.get("amount24"), 0.0),
        change_24h_pct=_as_float(item.get("riseFallRate"), 0.0),
        high_24h=_as_float(item.get("high24Price"), 0.0),
        low_24h=_as_float(item.get("lower24Price"), 0.0),
        funding_rate=_as_float(item.get("fundingRate"), 0.0) or None,
        open_interest=_as_float(item.get("holdVol"), 0.0) or None,
        ts=int(timestamp / 1000) if timestamp > 1e11 else int(timestamp or time.time()),
    )


def _parse_klines(data: Any, timeframe: Timeframe, limit: int) -> list[Candle]:
    """MEXC returns klines column-wise: ``{time, open, close, high, low, vol, amount}``."""

    if not isinstance(data, dict):
        return []
    times = data.get("time") or []
    opens = data.get("open") or []
    closes = data.get("close") or []
    highs = data.get("high") or []
    lows = data.get("low") or []
    vols = data.get("vol") or []
    amounts = data.get("amount") or []

    count = min(len(times), len(opens), len(closes), len(highs), len(lows))
    if count == 0:
        return []

    now = int(time.time())
    candles: list[Candle] = []
    for i in range(count):
        try:
            ts = int(times[i])
        except (TypeError, ValueError):
            continue
        if ts > 1e11:  # milliseconds
            ts //= 1000
        # Exclude the bar that is still forming - using it would leak the
        # future into every backtest and signal.
        if ts + timeframe.seconds > now:
            continue
        candles.append(
            Candle(
                ts=ts,
                open=_as_float(opens[i]),
                high=_as_float(highs[i]),
                low=_as_float(lows[i]),
                close=_as_float(closes[i]),
                volume=_as_float(vols[i]) if i < len(vols) else 0.0,
                quote_volume=_as_float(amounts[i]) if i < len(amounts) else 0.0,
                closed=True,
            )
        )
    candles.sort(key=lambda c: c.ts)
    return candles[-limit:]


def _parse_order(item: Any) -> ExchangeOrder | None:
    if not isinstance(item, dict):
        return None
    order_id = str(item.get("orderId") or item.get("id") or "")
    if not order_id:
        return None
    side_code = int(item.get("side", 1) or 1)
    # 1 open long / 4 close long -> the position is long
    side = Side.LONG if side_code in (SIDE_OPEN_LONG, SIDE_CLOSE_LONG) else Side.SHORT
    intent = (
        OrderIntent.OPEN
        if side_code in (SIDE_OPEN_LONG, SIDE_OPEN_SHORT)
        else OrderIntent.CLOSE
    )
    state = int(item.get("state", 1) or 1)
    order_type_code = int(item.get("orderType", TYPE_LIMIT) or TYPE_LIMIT)
    filled = _as_float(item.get("dealVol"), 0.0)
    total = _as_float(item.get("vol"), 0.0)
    status = _ORDER_STATE.get(state, OrderStatus.UNKNOWN)
    if status is OrderStatus.NEW and 0 < filled < total:
        status = OrderStatus.PARTIALLY_FILLED
    timestamp = _as_float(item.get("updateTime") or item.get("createTime"), 0.0)
    return ExchangeOrder(
        order_id=order_id,
        symbol=normalise_symbol(str(item.get("symbol", ""))),
        side=side,
        intent=intent,
        order_type=OrderType.MARKET if order_type_code == TYPE_MARKET else OrderType.LIMIT,
        quantity=total,
        price=_as_float(item.get("price"), 0.0) or None,
        status=status,
        filled_quantity=filled,
        average_price=_as_float(item.get("dealAvgPrice"), 0.0),
        client_order_id=str(item.get("externalOid", "") or ""),
        reduce_only=bool(item.get("reduceOnly", False)),
        ts=int(timestamp / 1000) if timestamp > 1e11 else int(timestamp or time.time()),
        raw=dict(item),
    )


def _extract_order_id(data: Any) -> str:
    if data is None:
        raise ExchangeError("order submit returned no order id")
    if isinstance(data, (str, int)):
        return str(data)
    if isinstance(data, dict):
        for key in ("orderId", "order_id", "id"):
            if key in data:
                return str(data[key])
    raise ExchangeError(f"could not read order id from response: {data!r}")


__all__ = [
    "MexcFuturesExchange",
    "INTERVALS",
    "_parse_klines",
    "_parse_ticker",
    "_parse_order",
]
