"""Configuration validation and secret handling.

These are the tests that matter most for safety: a misconfigured risk limit or a
leaked key is worse than any strategy bug.
"""

from __future__ import annotations

import logging

import pytest

from app.config import (
    ConfigError,
    Settings,
    TradingMode,
    build_settings,
    load_yaml_file,
    parse_env_file,
    parse_simple_yaml,
)
from app.logger import SecretRedactingFilter, redact, register_secret


class TestDefaults:
    def test_defaults_to_paper_when_mode_missing(self):
        assert build_settings(env={}).trading_mode is TradingMode.PAPER

    def test_empty_mode_still_defaults_to_paper(self):
        assert build_settings(env={"TRADING_MODE": ""}).trading_mode is TradingMode.PAPER

    def test_conservative_risk_defaults(self):
        s = build_settings(env={})
        assert s.default_risk_per_trade == 0.005
        assert s.max_daily_loss == 0.02
        assert s.max_portfolio_risk == 0.02
        assert s.max_open_positions == 3
        assert s.max_leverage == 3.0
        assert s.min_confidence == 0.70
        assert s.min_rr == 2.0
        assert s.max_symbols_to_scan == 100
        assert s.scanner_interval_seconds == 60

    def test_unknown_env_keys_are_ignored(self):
        s = build_settings(env={"TOTALLY_UNRELATED": "x"})
        assert s.trading_mode is TradingMode.PAPER


class TestLiveGating:
    def test_live_requires_credentials(self):
        with pytest.raises(ConfigError, match="MEXC_ACCESS_KEY"):
            build_settings(env={"TRADING_MODE": "live"})

    def test_live_requires_telegram_and_allowlist_and_phrase(self):
        env = {
            "TRADING_MODE": "live",
            "MEXC_ACCESS_KEY": "k" * 20,
            "MEXC_SECRET_KEY": "s" * 20,
        }
        with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
            build_settings(env=env)

        env |= {"TELEGRAM_BOT_TOKEN": "t" * 40, "TELEGRAM_CHAT_ID": "123"}
        with pytest.raises(ConfigError, match="TELEGRAM_ALLOWED_USER_IDS"):
            build_settings(env=env)

        env |= {"TELEGRAM_ALLOWED_USER_IDS": "111,222"}
        with pytest.raises(ConfigError, match="LIVE_CONFIRM_PHRASE"):
            build_settings(env=env)

        env |= {"LIVE_CONFIRM_PHRASE": "I UNDERSTAND THE RISK"}
        s = build_settings(env=env)
        assert s.is_live and s.telegram_allowed_user_ids == (111, 222)

    def test_live_rejects_synthetic_data(self):
        env = {
            "TRADING_MODE": "live",
            "DATA_SOURCE": "synthetic",
            "MEXC_ACCESS_KEY": "k" * 20,
            "MEXC_SECRET_KEY": "s" * 20,
            "TELEGRAM_BOT_TOKEN": "t" * 40,
            "TELEGRAM_CHAT_ID": "1",
            "TELEGRAM_ALLOWED_USER_IDS": "1",
            "LIVE_CONFIRM_PHRASE": "I UNDERSTAND THE RISK",
        }
        with pytest.raises(ConfigError, match="data_source=mexc"):
            build_settings(env=env)


class TestValidation:
    @pytest.mark.parametrize(
        "env,message",
        [
            ({"MAX_LEVERAGE": "500"}, "max_leverage"),
            ({"DEFAULT_RISK_PER_TRADE": "0.5"}, "outside allowed range"),
            ({"MIN_RR": "0.5"}, "min_rr"),
            ({"MAX_OPEN_POSITIONS": "0"}, "max_open_positions"),
            ({"SCANNER_INTERVAL_SECONDS": "1"}, "rate limits"),
            ({"MAX_SYMBOLS_TO_SCAN": "5000"}, "rate limits"),
            ({"DASHBOARD_PORT": "99999"}, "dashboard_port"),
            ({"TP1_R": "3", "TP2_R": "2"}, "take profit ladder"),
            ({"TP1_CLOSE_PCT": "0.7", "TP2_CLOSE_PCT": "0.5"}, "runner remains"),
            ({"DATA_SOURCE": "bogus"}, "data_source"),
        ],
    )
    def test_rejects_unsafe_values(self, env, message):
        with pytest.raises(ConfigError, match=message):
            build_settings(env=env)

    def test_risk_per_trade_cannot_exceed_portfolio_cap(self):
        with pytest.raises(ConfigError, match="max_portfolio_risk"):
            build_settings(
                env={"DEFAULT_RISK_PER_TRADE": "0.03", "MAX_PORTFOLIO_RISK": "0.01"}
            )

    def test_rejects_non_numeric(self):
        with pytest.raises(ConfigError, match="expected a number"):
            build_settings(env={"DEFAULT_RISK_PER_TRADE": "aggressive"})

    def test_accepts_valid_tightening(self):
        s = build_settings(
            env={"DEFAULT_RISK_PER_TRADE": "0.0025", "MAX_LEVERAGE": "2", "MIN_RR": "3"}
        )
        assert s.default_risk_per_trade == 0.0025
        assert s.max_leverage == 2.0


class TestParsing:
    def test_env_file_parsing(self):
        parsed = parse_env_file(
            "\n".join(
                [
                    "# comment",
                    "TRADING_MODE=paper",
                    'MEXC_SECRET_KEY="quoted secret"',
                    "EMPTY=",
                    "export EXPORTED=1",
                    "WITH_COMMENT=value # trailing",
                    "malformed line",
                ]
            )
        )
        assert parsed["TRADING_MODE"] == "paper"
        assert parsed["MEXC_SECRET_KEY"] == "quoted secret"
        assert parsed["EXPORTED"] == "1"
        assert parsed["WITH_COMMENT"] == "value"
        assert "malformed line" not in parsed

    def test_simple_yaml_parser(self):
        data = parse_simple_yaml(
            "risk:\n  per_trade: 0.005\n  max_leverage: 3\nscanner:\n  symbols: [BTC, ETH]\n"
        )
        assert data["risk"]["per_trade"] == 0.005
        assert data["scanner"]["symbols"] == ["BTC", "ETH"]

    def test_yaml_may_not_contain_secrets(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("exchange:\n  api_key: leaked\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="must not contain secrets"):
            load_yaml_file(path)


class TestSecretHandling:
    def test_redacted_masks_every_secret_field(self):
        s = build_settings(
            env={
                "MEXC_ACCESS_KEY": "mx0aBCDEFGHIJKLMNOP",
                "MEXC_SECRET_KEY": "supersecretvalue1234",
                "TELEGRAM_BOT_TOKEN": "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw",
            }
        )
        redacted = s.redacted()
        for field in ("mexc_access_key", "mexc_secret_key", "telegram_bot_token"):
            assert "secret" not in redacted[field].lower()
            assert redacted[field] != getattr(s, field)
        assert "supersecretvalue1234" not in str(redacted)

    def test_registered_secret_is_scrubbed(self):
        register_secret("hunter2isaverylongsecret")
        assert "hunter2isaverylongsecret" not in redact(
            "connecting with hunter2isaverylongsecret now"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "apiKey=abcdef123456",
            '{"secret": "abcdef123456"}',
            "https://api.telegram.org/bot123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw/getMe",
            "Signature: deadbeefdeadbeef",
        ],
    )
    def test_secret_shaped_text_is_scrubbed(self, text):
        assert "<redacted" in redact(text)

    def test_log_filter_scrubs_records(self):
        f = SecretRedactingFilter(["averylongsecretvalue"])
        record = logging.LogRecord(
            "t", logging.INFO, __file__, 1, "key averylongsecretvalue", (), None
        )
        f.filter(record)
        assert "averylongsecretvalue" not in record.getMessage()

    def test_short_values_are_not_registered_as_secrets(self):
        # Registering a 3-character "secret" would redact it everywhere.
        f = SecretRedactingFilter()
        f.add_secret("abc")
        assert f.scrub("abc def") == "abc def"
