"""Telegram control centre.

Implemented directly against the Bot API over HTTPS - no third-party Telegram
library - so the dependency surface stays small and the command router is a
pure function that can be unit tested without a network.
"""

from app.telegram.bot import TelegramBot, TelegramClient
from app.telegram.commands import COMMANDS, CommandRouter, CommandResult
from app.telegram.keyboards import main_menu, emergency_confirm
from app.telegram.notifications import Notifier

__all__ = [
    "TelegramBot",
    "TelegramClient",
    "CommandRouter",
    "CommandResult",
    "COMMANDS",
    "Notifier",
    "main_menu",
    "emergency_confirm",
]
