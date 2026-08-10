"""Configuration loading and strict validation.

Precedence (highest first):

1. Process environment variables (including anything exported by Docker).
2. ``.env`` file in the project root.
3. ``config.yaml`` in the project root (non-secret tuning only).
4. Built-in conservative defaults.

Secrets (API keys, bot tokens) are **only** ever read from the environment or
``.env``.  ``config.yaml`` is explicitly forbidden from carrying secrets so it
can be committed safely; if a secret-looking key appears there the loader
raises.

Validation is strict and fails closed: an invalid risk configuration must stop
the process rather than silently fall back to something dangerous.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any

from app.compat import HAVE_YAML, yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keys that must never appear in config.yaml.
_SECRET_KEY_PATTERN = re.compile(
    r"(secret|token|api[_-]?key|access[_-]?key|password|passphrase|private[_-]?key)",
    re.IGNORECASE,
)


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed or unsafe."""


class TradingMode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# --------------------------------------------------------------------------
# .env parsing (no python-dotenv dependency; the format we support is the
# common ``KEY=VALUE`` subset with ``#`` comments and optional quoting).
# --------------------------------------------------------------------------


def parse_env_file(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # Strip inline comments only for unquoted values.
        if value[:1] in {'"', "'"}:
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end > 0 else value[1:]
        else:
            hash_pos = value.find(" #")
            if hash_pos >= 0:
                value = value[:hash_pos].rstrip()
        out[key] = value
    return out


def load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return parse_env_file(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Minimal YAML subset parser used when PyYAML is unavailable.
# Supports nested mappings, scalars, and inline ``[a, b]`` lists - which is all
# ``config.yaml`` uses.
# --------------------------------------------------------------------------


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    low = token.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"null", "none", "~", ""}:
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    # stack of (indent, mapping)
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if line.startswith("- "):
            # List items are only supported inline in config.yaml; a block list
            # here means the file uses YAML we do not model.
            raise ConfigError(
                "config.yaml uses block lists which require PyYAML; install "
                "pyyaml or use inline [a, b] lists"
            )
        if ":" not in line:
            raise ConfigError(f"cannot parse config.yaml line: {raw!r}")
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ConfigError("inconsistent indentation in config.yaml")
        parent = stack[-1][1]
        if rest == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(rest)
    return root


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if HAVE_YAML else parse_simple_yaml(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError("config.yaml must contain a mapping at the top level")
    _assert_no_secrets(data)
    return data


def _assert_no_secrets(data: dict[str, Any], prefix: str = "") -> None:
    for key, value in data.items():
        path = f"{prefix}{key}"
        if _SECRET_KEY_PATTERN.search(str(key)):
            raise ConfigError(
                f"config.yaml must not contain secrets (found {path!r}); "
                "put credentials in .env instead"
            )
        if isinstance(value, dict):
            _assert_no_secrets(value, prefix=f"{path}.")


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested YAML into ``SECTION_KEY`` style env-like names."""

    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}".upper()
        if isinstance(value, dict):
            out.update(flatten(value, prefix=f"{name}_"))
        else:
            out[name] = value
    return out


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

SECRET_FIELDS = frozenset(
    {"mexc_access_key", "mexc_secret_key", "telegram_bot_token"}
)


@dataclass(frozen=True, slots=True)
class Settings:
    """Fully validated runtime configuration."""

    # --- mode / identity -------------------------------------------------
    trading_mode: TradingMode = TradingMode.PAPER
    #: "mexc" uses real MEXC market data (public endpoints need no credentials);
    #: "synthetic" runs the offline deterministic feed for demos and tests.
    data_source: str = "mexc"
    instance_name: str = "mexc-bot"
    log_level: LogLevel = LogLevel.INFO
    log_dir: str = "logs"
    data_dir: str = "data"
    models_dir: str = "models"

    # --- credentials (never logged) --------------------------------------
    mexc_access_key: str = ""
    mexc_secret_key: str = ""
    mexc_base_url: str = "https://contract.mexc.com"
    mexc_ws_url: str = "wss://contract.mexc.com/edge"
    mexc_recv_window: int = 30
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_allowed_user_ids: tuple[int, ...] = ()

    # --- database --------------------------------------------------------
    database_url: str = "sqlite:///data/trading_bot.db"

    # --- risk ------------------------------------------------------------
    default_risk_per_trade: float = 0.005
    max_daily_loss: float = 0.02
    max_portfolio_risk: float = 0.02
    max_open_positions: int = 3
    max_leverage: float = 3.0
    min_leverage: float = 1.0
    max_position_notional_pct: float = 1.0
    max_correlated_exposure: float = 0.012
    correlation_threshold: float = 0.7
    max_consecutive_losses: int = 4
    max_drawdown_stop: float = 0.10

    # --- signal gating ---------------------------------------------------
    min_confidence: float = 0.70
    min_rr: float = 2.0
    min_atr_pct: float = 0.0015
    max_atr_pct: float = 0.15
    max_spread_pct: float = 0.0008
    min_24h_quote_volume: float = 5_000_000.0

    # --- scanner ---------------------------------------------------------
    max_symbols_to_scan: int = 100
    scanner_interval_seconds: int = 60
    deep_analysis_count: int = 15
    symbol_blacklist: tuple[str, ...] = ()
    quote_currency: str = "USDT"

    # --- execution / costs ----------------------------------------------
    taker_fee: float = 0.0006
    maker_fee: float = 0.0002
    slippage_pct: float = 0.0005
    latency_ms: int = 250
    funding_interval_hours: int = 8
    order_poll_seconds: float = 3.0
    reconcile_interval_seconds: int = 60

    # --- position management --------------------------------------------
    tp1_r: float = 1.0
    tp2_r: float = 2.0
    tp3_r: float = 3.5
    tp1_close_pct: float = 0.4
    tp2_close_pct: float = 0.35
    breakeven_at_r: float = 1.0
    breakeven_offset_r: float = 0.1
    trailing_activate_r: float = 1.8
    trailing_atr_mult: float = 1.6
    stop_atr_mult: float = 1.5
    stop_structure_buffer_atr: float = 0.25
    max_stop_pct: float = 0.08
    min_stop_pct: float = 0.0015

    # --- ml --------------------------------------------------------------
    ml_enabled: bool = True
    ml_model_name: str = "direction_v1"
    ml_min_training_rows: int = 400
    ml_weight: float = 0.35

    # --- paper -----------------------------------------------------------
    paper_starting_equity: float = 1000.0

    # --- dashboard -------------------------------------------------------
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000
    dashboard_enabled: bool = True
    #: When set, the dashboard's pause/resume/emergency endpoints require this
    #: token in the ``X-Dashboard-Token`` header.  Read-only endpoints stay open.
    dashboard_token: str = ""

    # --- loop cadence ----------------------------------------------------
    position_manage_interval_seconds: float = 5.0
    health_interval_seconds: float = 30.0
    telegram_poll_timeout: int = 25

    # --- safety ----------------------------------------------------------
    live_confirm_phrase: str = ""
    kill_switch_file: str = "data/KILL_SWITCH"

    # provenance, not user configurable
    source_files: tuple[str, ...] = field(default=(), compare=False)

    # -- helpers ----------------------------------------------------------

    @property
    def is_live(self) -> bool:
        return self.trading_mode is TradingMode.LIVE

    @property
    def is_paper(self) -> bool:
        return self.trading_mode is TradingMode.PAPER

    @property
    def is_backtest(self) -> bool:
        return self.trading_mode is TradingMode.BACKTEST

    @property
    def has_mexc_credentials(self) -> bool:
        return bool(self.mexc_access_key and self.mexc_secret_key)

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    def secret_values(self) -> list[str]:
        """Secret strings that must be scrubbed from logs / notifications."""

        values = []
        for name in SECRET_FIELDS:
            value = getattr(self, name, "")
            if isinstance(value, str) and len(value) >= 8:
                values.append(value)
        return values

    def redacted(self) -> dict[str, Any]:
        """A dict safe to log, print, serve over HTTP or send to Telegram."""

        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "source_files":
                continue
            value = getattr(self, f.name)
            if f.name in SECRET_FIELDS:
                out[f.name] = _mask(value)
            elif isinstance(value, Enum):
                out[f.name] = value.value
            elif isinstance(value, tuple):
                out[f.name] = list(value)
            else:
                out[f.name] = value
        return out

    def with_overrides(self, **kwargs: Any) -> "Settings":
        return replace(self, **kwargs)

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _mask(value: Any) -> str:
    if not value:
        return "<unset>"
    text = str(value)
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:3]}***{text[-2:]} (len={len(text)})"


# --------------------------------------------------------------------------
# Coercion + validation
# --------------------------------------------------------------------------


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name}: expected a boolean, got {value!r}")


def _as_float(value: Any, name: str) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        raise ConfigError(f"{name}: expected a number, got {value!r}") from None


def _as_int(value: Any, name: str) -> int:
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        # Tolerate "3.0" for integer options.
        try:
            as_float = float(text)
        except ValueError:
            raise ConfigError(f"{name}: expected an integer, got {value!r}") from None
        if as_float.is_integer():
            return int(as_float)
        raise ConfigError(f"{name}: expected an integer, got {value!r}") from None


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    else:
        items = [part.strip() for part in str(value).split(",")]
    return tuple(item for item in items if item)


def _as_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    out = []
    for item in _as_str_tuple(value):
        try:
            out.append(int(item))
        except ValueError:
            raise ConfigError(f"{name}: {item!r} is not a valid integer id") from None
    return tuple(out)


def _coerce(name: str, raw: Any, current: Any) -> Any:
    if name == "trading_mode":
        text = str(raw).strip().lower()
        try:
            return TradingMode(text)
        except ValueError:
            raise ConfigError(
                f"TRADING_MODE must be one of backtest/paper/live, got {raw!r}"
            ) from None
    if name == "log_level":
        text = str(raw).strip().upper()
        try:
            return LogLevel(text)
        except ValueError:
            raise ConfigError(
                f"LOG_LEVEL must be one of DEBUG/INFO/WARNING/ERROR, got {raw!r}"
            ) from None
    if name == "telegram_allowed_user_ids":
        return _as_int_tuple(raw, name)
    if name in {"symbol_blacklist"}:
        return tuple(s.upper() for s in _as_str_tuple(raw))
    if isinstance(current, bool):
        return _as_bool(raw, name)
    if isinstance(current, int) and not isinstance(current, bool):
        return _as_int(raw, name)
    if isinstance(current, float):
        return _as_float(raw, name)
    return str(raw).strip()


_FRACTION_FIELDS = {
    "default_risk_per_trade": (1e-6, 0.05),
    "max_daily_loss": (1e-4, 0.5),
    "max_portfolio_risk": (1e-4, 0.5),
    "max_correlated_exposure": (1e-4, 0.5),
    "max_drawdown_stop": (1e-3, 0.9),
    "min_confidence": (0.0, 1.0),
    "correlation_threshold": (0.0, 1.0),
    "tp1_close_pct": (0.0, 1.0),
    "tp2_close_pct": (0.0, 1.0),
    "taker_fee": (0.0, 0.01),
    "maker_fee": (-0.001, 0.01),
    "slippage_pct": (0.0, 0.05),
    "max_spread_pct": (0.0, 0.05),
    "min_atr_pct": (0.0, 1.0),
    "max_atr_pct": (0.0, 1.0),
    "max_stop_pct": (0.0, 1.0),
    "min_stop_pct": (0.0, 1.0),
    "ml_weight": (0.0, 1.0),
    "max_position_notional_pct": (0.0, 10.0),
}


def validate(settings: Settings) -> Settings:
    """Validate cross-field invariants.  Raises :class:`ConfigError`."""

    errors: list[str] = []

    for name, (low, high) in _FRACTION_FIELDS.items():
        value = getattr(settings, name)
        if not (low <= value <= high):
            errors.append(f"{name}={value} outside allowed range [{low}, {high}]")

    if settings.default_risk_per_trade > settings.max_portfolio_risk:
        errors.append(
            "default_risk_per_trade must not exceed max_portfolio_risk "
            f"({settings.default_risk_per_trade} > {settings.max_portfolio_risk})"
        )
    if settings.max_portfolio_risk > settings.max_daily_loss * 2:
        errors.append(
            "max_portfolio_risk should not exceed 2x max_daily_loss; "
            "the daily loss limit would be unreachable before the portfolio cap"
        )
    if settings.max_correlated_exposure > settings.max_portfolio_risk:
        errors.append(
            "max_correlated_exposure must not exceed max_portfolio_risk"
        )
    if settings.max_open_positions < 1:
        errors.append("max_open_positions must be >= 1")
    if settings.max_open_positions > 20:
        errors.append("max_open_positions > 20 is not supported by the risk model")
    if settings.max_leverage < 1:
        errors.append("max_leverage must be >= 1")
    if settings.max_leverage > 20:
        errors.append(
            "max_leverage > 20 is rejected: the risk engine is designed for "
            "conservative leverage and unlimited leverage is never permitted"
        )
    if settings.min_leverage > settings.max_leverage:
        errors.append("min_leverage must not exceed max_leverage")
    if settings.min_rr < 1.0:
        errors.append("min_rr must be >= 1.0; sub-1R targets are rejected")
    if settings.min_rr > 20:
        errors.append("min_rr > 20 is unrealistic and would block all trades")
    if settings.min_atr_pct >= settings.max_atr_pct:
        errors.append("min_atr_pct must be < max_atr_pct")
    if settings.min_stop_pct >= settings.max_stop_pct:
        errors.append("min_stop_pct must be < max_stop_pct")
    if not (settings.tp1_r < settings.tp2_r < settings.tp3_r):
        errors.append("take profit ladder must satisfy tp1_r < tp2_r < tp3_r")
    if settings.tp1_close_pct + settings.tp2_close_pct >= 1.0:
        errors.append(
            "tp1_close_pct + tp2_close_pct must be < 1.0 so a runner remains"
        )
    if settings.max_symbols_to_scan < 1:
        errors.append("max_symbols_to_scan must be >= 1")
    if settings.max_symbols_to_scan > 1000:
        errors.append("max_symbols_to_scan > 1000 will breach exchange rate limits")
    if settings.deep_analysis_count < 1:
        errors.append("deep_analysis_count must be >= 1")
    if settings.deep_analysis_count > settings.max_symbols_to_scan:
        errors.append("deep_analysis_count must not exceed max_symbols_to_scan")
    if settings.scanner_interval_seconds < 10:
        errors.append(
            "scanner_interval_seconds < 10 would breach MEXC rate limits"
        )
    if settings.paper_starting_equity <= 0:
        errors.append("paper_starting_equity must be > 0")
    if not (1 <= settings.dashboard_port <= 65535):
        errors.append("dashboard_port must be between 1 and 65535")
    if settings.mexc_recv_window < 5 or settings.mexc_recv_window > 60:
        errors.append("mexc_recv_window must be between 5 and 60 seconds")
    if not settings.database_url:
        errors.append("database_url must not be empty")
    if settings.data_source not in {"mexc", "synthetic"}:
        errors.append("data_source must be 'mexc' or 'synthetic'")
    if settings.trading_mode is TradingMode.LIVE and settings.data_source != "mexc":
        errors.append("live trading requires data_source=mexc")

    # LIVE mode has extra, non-negotiable configuration requirements.  These
    # are checked here so an operator cannot start the process at all with a
    # half-configured live setup; runtime pre-flight checks in
    # ``app.safety.preflight`` then verify connectivity.
    if settings.trading_mode is TradingMode.LIVE:
        if not settings.has_mexc_credentials:
            errors.append(
                "TRADING_MODE=live requires MEXC_ACCESS_KEY and MEXC_SECRET_KEY"
            )
        if not settings.has_telegram:
            errors.append(
                "TRADING_MODE=live requires TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID so that alerts and the kill switch work"
            )
        if not settings.telegram_allowed_user_ids:
            errors.append(
                "TRADING_MODE=live requires TELEGRAM_ALLOWED_USER_IDS so that "
                "only known operators can control the bot"
            )
        if settings.live_confirm_phrase != "I UNDERSTAND THE RISK":
            errors.append(
                "TRADING_MODE=live requires LIVE_CONFIRM_PHRASE='I UNDERSTAND THE RISK'"
            )

    if errors:
        raise ConfigError(
            "invalid configuration:\n  - " + "\n  - ".join(errors)
        )
    return settings


def build_settings(
    env: dict[str, str] | None = None,
    yaml_data: dict[str, Any] | None = None,
) -> Settings:
    """Build settings from raw sources without touching the filesystem."""

    merged: dict[str, Any] = {}
    if yaml_data:
        merged.update(flatten(yaml_data))
    if env:
        merged.update(env)

    base = Settings()
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(Settings)} - {"source_files"}
    for name in known:
        key = name.upper()
        if key in merged and merged[key] is not None and merged[key] != "":
            kwargs[name] = _coerce(name, merged[key], getattr(base, name))

    # A missing TRADING_MODE defaults to PAPER - never to LIVE.
    settings = replace(base, **kwargs)
    return validate(settings)


def load_settings(
    root: Path | None = None,
    env: dict[str, str] | None = None,
) -> Settings:
    """Load settings from ``.env``, ``config.yaml`` and the process env."""

    root = root or PROJECT_ROOT
    sources: list[str] = []

    yaml_path = root / "config.yaml"
    yaml_data = load_yaml_file(yaml_path)
    if yaml_data:
        sources.append(str(yaml_path))

    file_env = load_env_file(root / ".env")
    if file_env:
        sources.append(str(root / ".env"))

    process_env = dict(os.environ) if env is None else dict(env)
    combined = {**file_env, **process_env}

    settings = build_settings(env=combined, yaml_data=yaml_data)
    return replace(settings, source_files=tuple(sources))


_cached: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    global _cached
    if _cached is None or reload:
        _cached = load_settings()
    return _cached


def set_settings(settings: Settings) -> None:
    """Install a settings object (used by tests and the backtest CLI)."""

    global _cached
    _cached = settings
