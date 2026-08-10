"""Telegram Bot API client and long-polling loop.

Written directly against the HTTP API.  The token never appears in a log line:
:mod:`app.logger` registers it as a secret at construction time and the
URL-shaped pattern ``/bot<token>/`` is scrubbed as well.

Authorisation is enforced on **every** update.  An unauthorised user gets a
flat refusal and the attempt is logged with the user id, because an unexpected
id trying to drive a trading bot is worth knowing about.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

from app.compat import HAVE_HTTPX, httpx
from app.logger import get_logger, register_secret
from app.telegram.commands import CommandResult, CommandRouter
from app.telegram.keyboards import main_menu

log = get_logger(__name__)


class TelegramError(RuntimeError):
    pass


class TelegramClient:
    """Minimal async Bot API client."""

    def __init__(self, token: str, timeout: float = 35.0) -> None:
        if not token:
            raise TelegramError("a Telegram bot token is required")
        register_secret(token)
        self._token = token
        self.timeout = timeout
        self.base_url = f"https://api.telegram.org/bot{token}"
        self._client: Any = None

    async def connect(self) -> None:
        if not HAVE_HTTPX:
            raise TelegramError(
                "the 'httpx' package is required for Telegram support"
            )
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _call(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        await self.connect()
        try:
            response = await self._client.post(
                f"{self.base_url}/{method}", json=payload or {}
            )
        except Exception as exc:  # noqa: BLE001
            raise TelegramError(f"{method} request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise TelegramError(f"{method} returned non-JSON") from exc

        if not data.get("ok"):
            raise TelegramError(
                f"{method} failed: {data.get('description', 'unknown error')}"
            )
        return data.get("result")

    async def get_me(self) -> dict[str, Any]:
        return await self._call("getMe")

    async def send_message(
        self,
        chat_id: str | int,
        text: str,
        keyboard: dict[str, Any] | None = None,
        silent: bool = False,
        parse_mode: str = "HTML",
    ) -> dict[str, Any]:
        # Telegram rejects messages over 4096 characters.
        chunks = _split_message(text)
        result: dict[str, Any] = {}
        for index, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
                "disable_notification": silent,
            }
            if keyboard and index == len(chunks) - 1:
                payload["reply_markup"] = json.dumps(keyboard)
            result = await self._call("sendMessage", payload)
        return result

    async def edit_message(
        self,
        chat_id: str | int,
        message_id: int,
        text: str,
        keyboard: dict[str, Any] | None = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text[:4096],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if keyboard:
            payload["reply_markup"] = json.dumps(keyboard)
        return await self._call("editMessageText", payload)

    async def answer_callback(
        self, callback_id: str, text: str = "", show_alert: bool = False
    ) -> Any:
        return await self._call(
            "answerCallbackQuery",
            {"callback_query_id": callback_id, "text": text[:200], "show_alert": show_alert},
        )

    async def get_updates(self, offset: int, timeout: int = 25) -> list[dict[str, Any]]:
        return await self._call(
            "getUpdates",
            {
                "offset": offset,
                "timeout": timeout,
                "allowed_updates": ["message", "callback_query"],
            },
        ) or []

    async def set_commands(self, commands: dict[str, str]) -> Any:
        return await self._call(
            "setMyCommands",
            {
                "commands": [
                    {"command": name, "description": description[:250]}
                    for name, description in commands.items()
                ]
            },
        )


class TelegramBot:
    def __init__(self, settings: Any, engine: Any, notifier: Any | None = None) -> None:
        self.settings = settings
        self.engine = engine
        self.client = (
            TelegramClient(settings.telegram_bot_token)
            if settings.telegram_bot_token
            else None
        )
        self.notifier = notifier
        self.router = CommandRouter(engine, settings)
        self._offset = 0
        self._task: asyncio.Task | None = None
        self._running = False
        self.updates_handled = 0
        self.unauthorised_attempts = 0

    @property
    def enabled(self) -> bool:
        return self.client is not None and bool(self.settings.telegram_chat_id)

    async def start(self) -> None:
        if not self.enabled:
            log.info("Telegram is not configured - control interface disabled")
            return
        await self.client.connect()
        try:
            me = await self.client.get_me()
            log.info("telegram bot connected: @%s", me.get("username", "?"))
        except TelegramError as exc:
            log.error("could not connect to Telegram: %s", exc)
            return

        from app.telegram.commands import COMMANDS

        with contextlib.suppress(TelegramError):
            await self.client.set_commands(COMMANDS)

        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="telegram-poll")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self.client is not None:
            await self.client.close()

    # -- polling ----------------------------------------------------------

    async def _poll_loop(self) -> None:
        backoff = 1.0
        while self._running:
            try:
                updates = await self.client.get_updates(
                    self._offset, timeout=self.settings.telegram_poll_timeout
                )
                backoff = 1.0
                for update in updates:
                    self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
                    await self._dispatch(update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - polling must survive anything
                log.warning("telegram poll error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _dispatch(self, update: dict[str, Any]) -> None:
        self.updates_handled += 1

        message = update.get("message")
        callback = update.get("callback_query")

        if callback:
            user = callback.get("from", {})
            chat = (callback.get("message") or {}).get("chat", {})
            payload = callback.get("data", "")
            callback_id = callback.get("id", "")
        elif message:
            user = message.get("from", {})
            chat = message.get("chat", {})
            payload = message.get("text", "")
            callback_id = ""
        else:
            return

        user_id = int(user.get("id", 0))
        chat_id = chat.get("id", "")

        if not self.router.is_authorised(user_id, chat_id):
            self.unauthorised_attempts += 1
            log.warning(
                "unauthorised Telegram command from user id %s (chat %s): %r",
                user_id,
                chat_id,
                str(payload)[:60],
            )
            with contextlib.suppress(TelegramError):
                await self.client.send_message(
                    chat_id,
                    "⛔ You are not authorised to control this bot.",
                )
            return

        try:
            result = await self.router.handle(payload, user_id=user_id)
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            log.exception("command handler failed: %s", exc)
            result = CommandResult(text=f"❌ Command failed: {exc}")

        if callback_id:
            with contextlib.suppress(TelegramError):
                await self.client.answer_callback(callback_id, result.alert)

        if not result.handled or not result.text:
            return

        with contextlib.suppress(TelegramError):
            await self.client.send_message(
                chat_id, result.text, keyboard=result.keyboard
            )

    # -- diagnostics ------------------------------------------------------

    def health(self) -> dict[str, Any]:
        if not self.enabled:
            return {"state": "WARNING", "detail": "not configured"}
        return {
            "state": "HEALTHY" if self._running else "ERROR",
            "detail": f"{self.updates_handled} updates handled",
            "updates": self.updates_handled,
            "unauthorised": self.unauthorised_attempts,
        }


def _split_message(text: str, limit: int = 4000) -> list[str]:
    """Split on line boundaries so HTML tags are not cut in half."""

    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in text.split("\n"):
        if length + len(line) + 1 > limit and current:
            chunks.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


__all__ = ["TelegramBot", "TelegramClient", "TelegramError"]
