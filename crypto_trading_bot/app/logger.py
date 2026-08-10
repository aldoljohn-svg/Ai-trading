"""Logging with mandatory secret redaction.

Every handler installed by :func:`setup_logging` carries a
:class:`SecretRedactingFilter`.  The filter rewrites the *formatted* message so
that a credential cannot reach a log file even if it is interpolated into an
exception message, a URL query string, or a repr of a request object.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys
from pathlib import Path
from typing import Any, Iterable

# Patterns that catch credentials even when they were never registered - for
# example an API key echoed back inside an error body from the exchange.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # key=value in query strings / form bodies
    (
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?key|secret[_-]?key|secret|token|"
            r"signature|password|passphrase)\b(\s*[=:]\s*)([^\s,&;'\"]+)"
        ),
        r"\1\2<redacted>",
    ),
    # JSON: "apiKey": "...."
    (
        re.compile(
            r'(?i)"(api_?key|access_?key|secret_?key|secret|token|signature|'
            r'password)"(\s*:\s*)"([^"]*)"'
        ),
        r'"\1"\2"<redacted>"',
    ),
    # Telegram bot tokens have a very distinctive shape: 123456789:AA...
    (re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b"), "<redacted-telegram-token>"),
    # bot<token>/ inside API URLs
    (re.compile(r"/bot[^/\s]+/"), "/bot<redacted>/"),
)


class SecretRedactingFilter(logging.Filter):
    """Scrub known secret values and secret-shaped substrings from records."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets: list[str] = []
        for secret in secrets:
            self.add_secret(secret)

    def add_secret(self, secret: str | None) -> None:
        if secret and isinstance(secret, str) and len(secret) >= 8:
            if secret not in self._secrets:
                self._secrets.append(secret)

    def scrub(self, text: str) -> str:
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, "<redacted>")
        for pattern, replacement in _PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - broken %-formatting
            message = str(record.msg)
        scrubbed = self.scrub(message)
        if scrubbed != message:
            record.msg = scrubbed
            record.args = ()
        if record.exc_text:
            record.exc_text = self.scrub(record.exc_text)
        return True


_redactor = SecretRedactingFilter()


def register_secret(secret: str | None) -> None:
    """Register a secret so it is scrubbed from all future log output."""

    _redactor.add_secret(secret)


def redact(text: str) -> str:
    """Public helper - also used by Telegram notifications."""

    return _redactor.scrub(text)


class _SafeFormatter(logging.Formatter):
    """Formatter that applies redaction to tracebacks as well."""

    def format(self, record: logging.LogRecord) -> str:
        return _redactor.scrub(super().format(record))


_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    secrets: Iterable[str] = (),
    console: bool = True,
) -> logging.Logger:
    """Configure root logging.  Safe to call more than once."""

    global _CONFIGURED

    for secret in secrets:
        register_secret(secret)

    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    if _CONFIGURED:
        return root

    fmt = _SafeFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        stream.addFilter(_redactor)
        root.addHandler(stream)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            directory / "bot.log",
            maxBytes=20 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        file_handler.addFilter(_redactor)
        root.addHandler(file_handler)

        errors = logging.handlers.RotatingFileHandler(
            directory / "errors.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        errors.setLevel(logging.WARNING)
        errors.setFormatter(fmt)
        errors.addFilter(_redactor)
        root.addHandler(errors)

    # Third-party libraries are noisy at DEBUG.
    for noisy in ("httpx", "httpcore", "websockets", "asyncio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.addFilter(_redactor)
    return logger


def log_dict(logger: logging.Logger, level: int, title: str, data: dict[str, Any]) -> None:
    lines = [f"{title}:"]
    for key in sorted(data):
        lines.append(f"    {key} = {data[key]}")
    logger.log(level, "\n".join(lines))
