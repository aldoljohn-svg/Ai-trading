"""`.env.example` has to be readable by more than one parser.

The bot's own parser strips inline comments. systemd's ``EnvironmentFile=``
does not -- it passes everything after the ``=`` straight through, so

    MAX_LEVERAGE=3    # conservative

arrives as the string ``"3    # conservative"`` and the bot refuses to start
with "expected a number". A restart loop follows, and the cause is nowhere near
obvious from the error.

The shipped unit file avoids ``EnvironmentFile=`` entirely, but people copy
these files around, so the example stays compatible with both.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.config import build_settings, parse_env_file

ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = ROOT / ".env.example"
UNIT_FILE = ROOT / "deploy" / "trading-bot.service"

ASSIGNMENT = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")


def systemd_style_parse(text: str) -> dict[str, str]:
    """How systemd reads an EnvironmentFile: no inline-comment stripping."""

    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ASSIGNMENT.match(stripped)
        if match:
            out[match.group(1)] = match.group(2)
    return out


class TestEnvExampleIsSystemdSafe:
    def test_no_value_carries_an_inline_comment(self):
        offenders = [
            line
            for line in ENV_EXAMPLE.read_text().splitlines()
            if (m := ASSIGNMENT.match(line.strip())) and "#" in m.group(2)
        ]
        assert not offenders, (
            "these lines break systemd's EnvironmentFile=; move the comment to "
            f"its own line: {offenders}"
        )

    def test_both_parsers_agree_on_every_value(self):
        text = ENV_EXAMPLE.read_text()
        ours = parse_env_file(text)
        theirs = systemd_style_parse(text)

        assert set(ours) == set(theirs)
        for key, value in ours.items():
            assert value == theirs[key].strip(), (
                f"{key} is read differently by the two parsers: "
                f"{value!r} vs {theirs[key]!r}"
            )

    def test_the_example_still_builds_valid_settings(self):
        """Whatever the file says must actually start the bot."""

        env = systemd_style_parse(ENV_EXAMPLE.read_text())
        # Paper mode with no credentials is the documented default.
        env.setdefault("DATABASE_URL", "sqlite:///:memory:")
        settings = build_settings(env=env)
        assert settings.trading_mode.value == "paper"

    def test_numeric_settings_survive_the_systemd_path(self):
        """The exact failure that was seen: a float arriving with a comment."""

        env = systemd_style_parse(ENV_EXAMPLE.read_text())
        env["DATABASE_URL"] = "sqlite:///:memory:"
        settings = build_settings(env=env)
        assert settings.tp1_close_pct == pytest.approx(0.4)
        assert settings.tp2_close_pct == pytest.approx(0.35)
        assert settings.min_atr_pct == pytest.approx(0.0015)
        assert settings.max_correlated_exposure == pytest.approx(0.012)
        assert settings.min_rr == pytest.approx(1.7)


class TestOurParserIsStillForgiving:
    """The application's own parser must keep handling the friendlier forms."""

    def test_inline_comments_are_stripped(self):
        parsed = parse_env_file("MAX_LEVERAGE=3    # conservative\n")
        assert parsed["MAX_LEVERAGE"] == "3"

    def test_quoted_values_keep_their_hashes(self):
        parsed = parse_env_file('NOTE="a # inside quotes"\n')
        assert parsed["NOTE"] == "a # inside quotes"

    def test_export_prefixes_are_accepted(self):
        assert parse_env_file("export FOO=bar\n")["FOO"] == "bar"


class TestShippedUnitFile:
    def test_it_exists(self):
        assert UNIT_FILE.is_file(), "deploy/trading-bot.service should be shipped"

    def test_it_does_not_use_environmentfile(self):
        """The bot loads .env itself; two parsers is one too many."""

        active = [
            line.strip()
            for line in UNIT_FILE.read_text().splitlines()
            if not line.strip().startswith("#")
        ]
        assert not any(line.startswith("EnvironmentFile=") for line in active)

    def test_it_runs_in_utc(self):
        """The daily-loss breaker rolls on the UTC day."""

        assert "Environment=TZ=UTC" in UNIT_FILE.read_text()

    def test_it_bounds_the_restart_loop(self):
        """A configuration error will never fix itself by restarting."""

        text = UNIT_FILE.read_text()
        assert "StartLimitBurst=" in text
        assert "StartLimitIntervalSec=" in text

    def test_it_allows_a_graceful_stop(self):
        """Positions are persisted and the exchange connection closed on stop."""

        text = UNIT_FILE.read_text()
        assert "KillSignal=SIGTERM" in text
        timeout = int(re.search(r"TimeoutStopSec=(\d+)", text).group(1))
        assert timeout >= 60
