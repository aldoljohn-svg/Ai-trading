"""The Telegram control panel replaces itself instead of stacking copies.

Pressing REFRESH used to leave the old panel above the new one, so a few
minutes of use buried the chat in dead panels.  The rule now is: a panel
replaces the previous panel, and nothing else is ever deleted -- trade alerts
and reports are the record and must survive a refresh.
"""

from __future__ import annotations

import asyncio

import pytest

from app.config import build_settings
from app.telegram.bot import TelegramBot, TelegramError
from app.telegram.commands import CommandResult

from tests.conftest import TEST_ENV


class FakeClient:
    """Records what the bot asked Telegram to do."""

    def __init__(self, fail_send: bool = False, fail_delete: bool = False) -> None:
        self.sent: list[tuple[str, str]] = []
        self.deleted: list[int] = []
        self.answered: list[str] = []
        self.fail_send = fail_send
        self.fail_delete = fail_delete
        self._next_id = 100

    async def send_message_chunks(self, chat_id, text, keyboard=None, **kwargs):
        if self.fail_send:
            raise TelegramError("simulated send failure")
        # Mirror the real client: long text becomes several messages.
        chunks = [text[i : i + 4000] for i in range(0, max(len(text), 1), 4000)]
        results = []
        for chunk in chunks:
            self._next_id += 1
            self.sent.append((str(chat_id), chunk))
            results.append({"message_id": self._next_id, "chat": {"id": chat_id}})
        return results

    async def send_message(self, chat_id, text, keyboard=None, **kwargs):
        results = await self.send_message_chunks(chat_id, text, keyboard, **kwargs)
        return results[-1]

    async def delete_message(self, chat_id, message_id):
        if self.fail_delete:
            return False
        self.deleted.append(int(message_id))
        return True

    async def answer_callback(self, callback_id, text="", show_alert=False):
        self.answered.append(callback_id)


class StubRouter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def is_authorised(self, user_id, chat_id=""):
        return True

    async def handle(self, raw, user_id=0):
        self.calls.append(raw)
        return CommandResult(text=f"panel for {raw}", keyboard={"inline_keyboard": []})


def make_bot(client: FakeClient, **overrides) -> TelegramBot:
    settings = build_settings(
        env=dict(
            TEST_ENV,
            TELEGRAM_BOT_TOKEN="123456:" + "x" * 35,
            TELEGRAM_CHAT_ID="555",
            TELEGRAM_ALLOWED_USER_IDS="42",
            **overrides,
        )
    )
    bot = TelegramBot(settings, engine=None)
    bot.client = client
    bot.router = StubRouter()
    return bot


def callback(data: str, chat_id: int = 555, user_id: int = 42) -> dict:
    return {
        "callback_query": {
            "id": "cb1",
            "from": {"id": user_id},
            "message": {"chat": {"id": chat_id}},
            "data": data,
        }
    }


def typed(text: str, chat_id: int = 555, user_id: int = 42) -> dict:
    return {
        "message": {
            "from": {"id": user_id},
            "chat": {"id": chat_id},
            "text": text,
        }
    }


class TestPanelReplacement:
    def test_the_first_panel_deletes_nothing(self):
        client = FakeClient()
        bot = make_bot(client)
        asyncio.run(bot._dispatch(typed("/status")))
        assert len(client.sent) == 1
        assert client.deleted == []

    def test_a_refresh_deletes_the_previous_panel(self):
        client = FakeClient()
        bot = make_bot(client)

        async def run():
            await bot._dispatch(typed("/status"))
            first_id = bot._panels["555"][0]
            await bot._dispatch(callback("cmd:status"))
            return first_id

        first_id = asyncio.run(run())
        assert len(client.sent) == 2
        assert client.deleted == [first_id]

    def test_only_one_panel_is_ever_tracked(self):
        client = FakeClient()
        bot = make_bot(client)

        async def run():
            for _ in range(5):
                await bot._dispatch(callback("cmd:status"))

        asyncio.run(run())
        assert len(client.sent) == 5
        # Four deletions for five sends: every panel but the live one.
        assert len(client.deleted) == 4
        assert len(bot._panels["555"]) == 1

    def test_switching_panels_also_replaces(self):
        """MENU -> RISK -> MENU must leave exactly one message on screen."""

        client = FakeClient()
        bot = make_bot(client)

        async def run():
            await bot._dispatch(callback("cmd:menu"))
            await bot._dispatch(callback("cmd:risk"))
            await bot._dispatch(callback("cmd:menu"))

        asyncio.run(run())
        assert len(client.deleted) == 2

    def test_every_chunk_of_a_long_panel_is_deleted(self):
        """A split panel must not leave orphaned halves behind."""

        client = FakeClient()
        bot = make_bot(client)

        class LongRouter(StubRouter):
            async def handle(self, raw, user_id=0):
                return CommandResult(text="x" * 9000)

        bot.router = LongRouter()

        async def run():
            await bot._dispatch(callback("cmd:journal"))
            first = list(bot._panels["555"])
            await bot._dispatch(callback("cmd:journal"))
            return first

        first = asyncio.run(run())
        assert len(first) == 3                      # 9000 chars -> 3 messages
        assert sorted(client.deleted) == sorted(first)

    def test_separate_chats_keep_separate_panels(self):
        client = FakeClient()
        bot = make_bot(client)

        async def run():
            await bot._dispatch(callback("cmd:status", chat_id=555))
            await bot._dispatch(callback("cmd:status", chat_id=777))
            # Refreshing one must not touch the other's panel.
            await bot._dispatch(callback("cmd:status", chat_id=555))

        asyncio.run(run())
        assert set(bot._panels) == {"555", "777"}
        assert len(client.deleted) == 1


class TestPanelFailureModes:
    def test_a_failed_send_keeps_the_old_panel(self):
        """Worst case must be two panels, never zero."""

        client = FakeClient()
        bot = make_bot(client)
        asyncio.run(bot._dispatch(callback("cmd:status")))
        original = list(bot._panels["555"])

        client.fail_send = True
        asyncio.run(bot._dispatch(callback("cmd:status")))

        assert bot._panels["555"] == original
        assert client.deleted == [], "nothing may be deleted when the send failed"

    def test_a_failed_delete_does_not_break_the_panel(self):
        """Telegram refuses to delete messages older than 48 hours."""

        client = FakeClient(fail_delete=True)
        bot = make_bot(client)

        async def run():
            await bot._dispatch(callback("cmd:status"))
            await bot._dispatch(callback("cmd:status"))

        asyncio.run(run())
        assert len(client.sent) == 2
        assert len(bot._panels["555"]) == 1

    def test_forget_panel_stops_tracking(self):
        client = FakeClient()
        bot = make_bot(client)
        asyncio.run(bot._dispatch(callback("cmd:status")))
        bot.forget_panel(555)
        asyncio.run(bot._dispatch(callback("cmd:status")))
        assert client.deleted == []

    def test_an_empty_result_sends_nothing(self):
        client = FakeClient()
        bot = make_bot(client)

        class SilentRouter(StubRouter):
            async def handle(self, raw, user_id=0):
                return CommandResult(text="", handled=False)

        bot.router = SilentRouter()
        asyncio.run(bot._dispatch(callback("cmd:noop")))
        assert client.sent == []
        assert client.deleted == []

    def test_an_unauthorised_user_gets_no_panel_tracked(self):
        client = FakeClient()
        bot = make_bot(client)

        class ClosedRouter(StubRouter):
            def is_authorised(self, user_id, chat_id=""):
                return False

        bot.router = ClosedRouter()
        asyncio.run(bot._dispatch(callback("cmd:status", user_id=999)))
        assert bot._panels == {}
        assert client.deleted == []


class TestSinglePanelToggle:
    def test_disabling_it_restores_the_old_behaviour(self):
        client = FakeClient()
        bot = make_bot(client, TELEGRAM_SINGLE_PANEL="false")

        async def run():
            await bot._dispatch(callback("cmd:status"))
            await bot._dispatch(callback("cmd:status"))

        asyncio.run(run())
        assert len(client.sent) == 2
        assert client.deleted == []

    def test_it_defaults_to_on(self, settings):
        assert settings.telegram_single_panel


class TestRiskRewardDefault:
    def test_the_default_is_seventeen(self, settings):
        assert settings.min_rr == pytest.approx(1.7)

    def test_break_even_expectation_is_documented(self, settings):
        """1.7R needs a 37% win rate before costs; keep that honest."""

        break_even = 1.0 / (1.0 + settings.min_rr)
        assert 0.36 < break_even < 0.38

    def test_sub_one_is_still_rejected(self):
        from app.config import ConfigError

        with pytest.raises(ConfigError):
            build_settings(env=dict(TEST_ENV, MIN_RR="0.8"))
