"""Optional-dependency handling.

The bot must start and make correct decisions even when optional scientific /
web libraries are missing.  Every optional import in the code base goes through
this module so that a single place records what is available, and the health
monitor can report it.

Hard requirements: CPython 3.12+ standard library only.
"""

from __future__ import annotations

import importlib
from typing import Any


def _try(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:  # pragma: no cover - depends on the environment
        return None


numpy = _try("numpy")
pandas = _try("pandas")
scipy = _try("scipy")
sklearn = _try("sklearn")
lightgbm = _try("lightgbm")
xgboost = _try("xgboost")
torch = _try("torch")
httpx = _try("httpx")
aiohttp = _try("aiohttp")
websockets = _try("websockets")
fastapi = _try("fastapi")
pydantic = _try("pydantic")
uvicorn = _try("uvicorn")
yaml = _try("yaml")

HAVE_NUMPY = numpy is not None
HAVE_PANDAS = pandas is not None
HAVE_SCIPY = scipy is not None
HAVE_SKLEARN = sklearn is not None
HAVE_LIGHTGBM = lightgbm is not None
HAVE_XGBOOST = xgboost is not None
HAVE_TORCH = torch is not None
HAVE_HTTPX = httpx is not None
HAVE_AIOHTTP = aiohttp is not None
HAVE_WEBSOCKETS = websockets is not None
HAVE_FASTAPI = fastapi is not None
HAVE_PYDANTIC = pydantic is not None
HAVE_UVICORN = uvicorn is not None
HAVE_YAML = yaml is not None


def capabilities() -> dict[str, bool]:
    """Return the optional-dependency matrix, for /health and diagnostics."""

    return {
        "numpy": HAVE_NUMPY,
        "pandas": HAVE_PANDAS,
        "scipy": HAVE_SCIPY,
        "scikit-learn": HAVE_SKLEARN,
        "lightgbm": HAVE_LIGHTGBM,
        "xgboost": HAVE_XGBOOST,
        "torch": HAVE_TORCH,
        "httpx": HAVE_HTTPX,
        "aiohttp": HAVE_AIOHTTP,
        "websockets": HAVE_WEBSOCKETS,
        "fastapi": HAVE_FASTAPI,
        "pydantic": HAVE_PYDANTIC,
        "uvicorn": HAVE_UVICORN,
        "pyyaml": HAVE_YAML,
    }


def missing_for_live() -> list[str]:
    """Dependencies that must be present before LIVE trading is allowed.

    LIVE trading talks to MEXC over REST and WebSocket.  The pure-stdlib HTTP
    fallback is fine for read-only smoke tests but we require the real async
    clients before risking capital.
    """

    missing: list[str] = []
    if not HAVE_HTTPX:
        missing.append("httpx")
    if not HAVE_WEBSOCKETS:
        missing.append("websockets")
    return missing


__all__ = [name for name in dir() if name.startswith("HAVE_")] + [
    "capabilities",
    "missing_for_live",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "lightgbm",
    "xgboost",
    "torch",
    "httpx",
    "aiohttp",
    "websockets",
    "fastapi",
    "pydantic",
    "uvicorn",
    "yaml",
]
