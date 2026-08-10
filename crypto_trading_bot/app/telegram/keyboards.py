"""Inline keyboard layouts.

Every button carries a ``callback_data`` string that the router treats exactly
like the equivalent typed command, so there is one code path for both.
"""

from __future__ import annotations

from typing import Any


def _button(text: str, data: str) -> dict[str, str]:
    return {"text": text, "callback_data": data}


def main_menu() -> dict[str, Any]:
    """The main control panel."""

    return {
        "inline_keyboard": [
            [
                _button("📊 LIVE REPORT", "cmd:status"),
                _button("📈 MARKET SCANNER", "cmd:scanner"),
            ],
            [_button("💼 POSITIONS", "cmd:positions")],
            [
                _button("▶️ START", "cmd:start_trading"),
                _button("⏸ PAUSE", "cmd:pause"),
            ],
            [
                _button("▶️ RESUME", "cmd:resume"),
                _button("🛑 STOP", "cmd:stop"),
            ],
            [_button("🚨 EMERGENCY STOP", "cmd:emergency")],
            [
                _button("💰 ACCOUNT", "cmd:account"),
                _button("📊 PERFORMANCE", "cmd:performance"),
            ],
            [
                _button("🧠 AI", "cmd:ai"),
                _button("⚙️ RISK", "cmd:risk"),
            ],
        ]
    }


def emergency_confirm() -> dict[str, Any]:
    """Emergency stop always requires a second, explicit confirmation."""

    return {
        "inline_keyboard": [
            [_button("🚨 CLOSE ALL + STOP", "confirm:emergency")],
            [_button("❌ CANCEL", "cmd:menu")],
        ]
    }


def stop_confirm() -> dict[str, Any]:
    return {
        "inline_keyboard": [
            [_button("🛑 CONFIRM STOP", "confirm:stop")],
            [_button("❌ CANCEL", "cmd:menu")],
        ]
    }


def back_to_menu() -> dict[str, Any]:
    return {"inline_keyboard": [[_button("⬅️ MENU", "cmd:menu")]]}


def refresh_menu(command: str) -> dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                _button("🔄 REFRESH", f"cmd:{command}"),
                _button("⬅️ MENU", "cmd:menu"),
            ]
        ]
    }


def positions_menu(symbols: list[str]) -> dict[str, Any]:
    """Per-position close buttons, capped so the keyboard stays usable."""

    rows = [
        [_button(f"❌ Close {symbol}", f"close:{symbol}")] for symbol in symbols[:6]
    ]
    rows.append([_button("🔄 REFRESH", "cmd:positions"), _button("⬅️ MENU", "cmd:menu")])
    return {"inline_keyboard": rows}


def close_confirm(symbol: str) -> dict[str, Any]:
    return {
        "inline_keyboard": [
            [_button(f"✅ CLOSE {symbol}", f"confirm:close:{symbol}")],
            [_button("❌ CANCEL", "cmd:positions")],
        ]
    }


__all__ = [
    "main_menu",
    "emergency_confirm",
    "stop_confirm",
    "back_to_menu",
    "refresh_menu",
    "positions_menu",
    "close_confirm",
]
