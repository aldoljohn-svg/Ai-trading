"""Async HTTP transport with rate limiting, timeouts and exponential backoff.

Uses ``httpx.AsyncClient`` when available and otherwise falls back to
``urllib.request`` executed in a worker thread, so read-only diagnostics still
work in a minimal environment.  LIVE trading requires httpx (enforced by
:func:`app.compat.missing_for_live`).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from app.compat import HAVE_HTTPX, httpx
from app.exchange.base import (
    ExchangeAuthError,
    ExchangeError,
    ExchangeRateLimit,
    ExchangeUnavailable,
)
from app.logger import get_logger

log = get_logger(__name__)


class TokenBucket:
    """Simple async token bucket used to stay under the venue rate limits."""

    def __init__(self, rate_per_second: float, burst: int) -> None:
        self.rate = max(rate_per_second, 0.1)
        self.burst = max(burst, 1)
        self._tokens = float(self.burst)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: float = 1.0) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._updated
                self._updated = now
                self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                wait = (cost - self._tokens) / self.rate
                await asyncio.sleep(min(wait, 5.0))

    @property
    def available(self) -> float:
        elapsed = time.monotonic() - self._updated
        return min(self.burst, self._tokens + elapsed * self.rate)


@dataclass(slots=True)
class HttpResponse:
    status: int
    body: str
    headers: Mapping[str, str]

    def json(self) -> Any:
        try:
            return json.loads(self.body)
        except ValueError as exc:
            raise ExchangeError(
                f"non-JSON response (status {self.status}): {self.body[:200]!r}"
            ) from exc


class AsyncHttpClient:
    """Shared HTTP client with retry and backoff.

    Retries are applied only to *idempotent* situations: network errors,
    timeouts, 5xx and 429.  A ``POST`` that reached the venue is never blindly
    replayed - :mod:`app.execution.order_manager` uses client order ids and
    reconciliation for that case instead.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 15.0,
        max_retries: int = 4,
        rate_limit_per_second: float = 15.0,
        burst: int = 20,
        user_agent: str = "crypto-trading-bot/1.0",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.bucket = TokenBucket(rate_limit_per_second, burst)
        self._client: Any = None
        self._closed = False
        self.stats = {"requests": 0, "retries": 0, "errors": 0, "rate_limited": 0}

    async def connect(self) -> None:
        if HAVE_HTTPX and self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
        self._closed = False

    async def close(self) -> None:
        self._closed = True
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    # -- request ----------------------------------------------------------

    async def request(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        body: str | None = None,
        headers: Mapping[str, str] | None = None,
        retry: bool = True,
        cost: float = 1.0,
    ) -> HttpResponse:
        attempts = self.max_retries if retry else 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            await self.bucket.acquire(cost)
            self.stats["requests"] += 1
            try:
                response = await self._send(method, path, params, body, headers)
            except (ExchangeUnavailable, ExchangeRateLimit) as exc:
                last_error = exc
                self.stats["errors"] += 1
                if attempt == attempts - 1:
                    break
                await self._backoff(attempt, exc)
                self.stats["retries"] += 1
                continue

            if response.status == 429 or response.status == 418:
                self.stats["rate_limited"] += 1
                last_error = ExchangeRateLimit(
                    f"rate limited by venue (HTTP {response.status})"
                )
                if attempt == attempts - 1:
                    break
                await self._backoff(attempt, last_error, response.headers)
                self.stats["retries"] += 1
                continue

            if response.status in {401, 403}:
                raise ExchangeAuthError(
                    f"authentication rejected (HTTP {response.status}); "
                    "check API key, permissions and IP allowlist"
                )

            if 500 <= response.status < 600:
                last_error = ExchangeUnavailable(
                    f"venue error HTTP {response.status}: {response.body[:200]}"
                )
                if attempt == attempts - 1:
                    break
                await self._backoff(attempt, last_error)
                self.stats["retries"] += 1
                continue

            return response

        assert last_error is not None
        raise last_error

    async def _backoff(
        self,
        attempt: int,
        error: Exception,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        delay = min(2.0 ** attempt, 30.0)
        if headers:
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
        # Full jitter avoids synchronised retries across symbols.
        delay = random.uniform(delay * 0.5, delay)
        log.warning(
            "http retry %d in %.2fs after: %s", attempt + 1, delay, error
        )
        await asyncio.sleep(delay)

    async def _send(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None,
        body: str | None,
        headers: Mapping[str, str] | None,
    ) -> HttpResponse:
        if self._client is not None:
            return await self._send_httpx(method, path, params, body, headers)
        return await asyncio.to_thread(
            self._send_urllib, method, path, params, body, headers
        )

    async def _send_httpx(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None,
        body: str | None,
        headers: Mapping[str, str] | None,
    ) -> HttpResponse:
        try:
            response = await self._client.request(
                method,
                path,
                params=dict(params) if params else None,
                content=body.encode("utf-8") if body else None,
                headers=dict(headers) if headers else None,
            )
        except httpx.TimeoutException as exc:
            raise ExchangeUnavailable(f"timeout after {self.timeout}s: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ExchangeUnavailable(f"network error: {exc}") from exc
        return HttpResponse(
            status=response.status_code,
            body=response.text,
            headers=dict(response.headers),
        )

    def _send_urllib(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None,
        body: str | None,
        headers: Mapping[str, str] | None,
    ) -> HttpResponse:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(dict(params))}"
        data = body.encode("utf-8") if body else None
        request = urllib.request.Request(url=url, data=data, method=method.upper())
        request.add_header("User-Agent", self.user_agent)
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return HttpResponse(
                    status=response.status,
                    body=response.read().decode("utf-8", errors="replace"),
                    headers=dict(response.headers.items()),
                )
        except urllib.error.HTTPError as exc:
            return HttpResponse(
                status=exc.code,
                body=exc.read().decode("utf-8", errors="replace"),
                headers=dict(exc.headers.items()) if exc.headers else {},
            )
        except urllib.error.URLError as exc:
            raise ExchangeUnavailable(f"network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ExchangeUnavailable(f"timeout after {self.timeout}s") from exc

    # -- convenience ------------------------------------------------------

    async def get_json(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        cost: float = 1.0,
    ) -> Any:
        response = await self.request("GET", path, params=params, headers=headers, cost=cost)
        return response.json()

    async def post_json(
        self,
        path: str,
        body: str,
        headers: Mapping[str, str] | None = None,
        retry: bool = False,
        cost: float = 1.0,
    ) -> Any:
        response = await self.request(
            "POST", path, body=body, headers=headers, retry=retry, cost=cost
        )
        return response.json()
