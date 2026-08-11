"""Instrument classification.

MEXC's futures list is not only crypto.  It also carries tokenised equities
(``SNDKSTOCKUSDT``, ``MUSTOCKUSDT``), leveraged equity ETFs (``SOXLUSDT``),
commodities (``XAUUSDT``), FX pairs and index products.

Those instruments pass a naive liquidity screen -- the exchange reports a 24h
volume for them -- but they are structurally unsuitable for this bot:

* their order books are frequently empty or unquotable outside the underlying
  market's hours, which is exactly the ``no order book available`` execution
  risk that fills the decision journal;
* they gap across sessions rather than trading continuously, so an ATR-based
  stop computed from crypto-style bars is meaningless;
* the analytical stack (ICT, RTM, liquidity sweeps) is built around a
  24/7 order-driven market.

Each one that reaches deep analysis costs a slot a tradable crypto symbol could
have used.  Classification is deliberately conservative: anything not
recognised stays ``CRYPTO`` rather than being silently dropped.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Iterable

#: Bases that are metals, energy or agricultural products.
_COMMODITY_BASES: frozenset[str] = frozenset(
    {
        "XAU", "XAG", "XPT", "XPD",              # metals
        "GOLD", "SILVER", "PLATINUM",
        "WTI", "BRENT", "USOIL", "UKOIL", "OIL", "NG", "NGAS",
        "COPPER", "CORN", "WHEAT", "SUGAR", "COCOA", "COFFEE",
    }
)

#: Equity indices and the leveraged ETFs that track them.
_INDEX_BASES: frozenset[str] = frozenset(
    {
        "SPX", "SPX500", "US500", "SPY", "ES",
        "NDX", "NAS100", "QQQ", "NQ",
        "DJI", "US30", "DIA",
        "RUT", "IWM", "VIX",
        "DAX", "FTSE", "NIKKEI", "HSI",
        # Leveraged / inverse sector ETFs.
        "SOXL", "SOXS", "TQQQ", "SQQQ", "SPXL", "SPXS",
        "LABU", "LABD", "TNA", "TZA", "FAS", "FAZ", "YINN", "YANG",
        "UVXY", "SVXY", "TMF", "TMV", "ARKK",
    }
)

#: Fiat currencies.  As a base against USDT these are FX, not crypto.
_FX_BASES: frozenset[str] = frozenset(
    {"EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNH", "CNY", "MXN", "TRY"}
)

#: Stablecoins.  Trading one against another has no directional thesis and the
#: stop distance collapses below anything the risk engine can size.
_STABLE_BASES: frozenset[str] = frozenset(
    {
        "USDC", "USDT", "DAI", "TUSD", "FDUSD", "USDE", "USDD", "PYUSD",
        "BUSD", "USDP", "GUSD", "LUSD", "FRAX", "SUSD", "EURT", "EURS",
    }
)

#: Substrings that mark a tokenised equity on MEXC.
_EQUITY_MARKERS: tuple[str, ...] = ("STOCK", "SHARES", "EQUITY")

#: Wrapped or synthetic commodity tokens.  Crypto assets, but they track a
#: commodity, so they inherit its session-driven behaviour.
_COMMODITY_TOKENS: frozenset[str] = frozenset({"XAUT", "PAXG", "KAU", "KAG"})


class InstrumentClass(str, Enum):
    CRYPTO = "CRYPTO"
    TOKENISED_EQUITY = "TOKENISED_EQUITY"
    INDEX = "INDEX"
    COMMODITY = "COMMODITY"
    FX = "FX"
    STABLECOIN = "STABLECOIN"

    @property
    def is_crypto(self) -> bool:
        return self is InstrumentClass.CRYPTO

    @property
    def description(self) -> str:
        return {
            InstrumentClass.CRYPTO: "crypto",
            InstrumentClass.TOKENISED_EQUITY: "tokenised equity",
            InstrumentClass.INDEX: "equity index or ETF",
            InstrumentClass.COMMODITY: "commodity",
            InstrumentClass.FX: "foreign exchange",
            InstrumentClass.STABLECOIN: "stablecoin",
        }[self]


#: Everything that is not continuously traded crypto.  These are excluded by
#: default; ``ALLOWED_INSTRUMENT_CLASSES`` can re-admit any of them.
NON_CRYPTO: frozenset[InstrumentClass] = frozenset(
    c for c in InstrumentClass if c is not InstrumentClass.CRYPTO
)


def base_of(symbol: str, quote: str = "USDT") -> str:
    """Strip the quote currency and any separator from a symbol."""

    cleaned = re.sub(r"[^A-Z0-9]", "", symbol.upper())
    quote = quote.upper()
    if quote and cleaned.endswith(quote) and len(cleaned) > len(quote):
        cleaned = cleaned[: -len(quote)]
    return cleaned


def classify_symbol(symbol: str, quote: str = "USDT") -> InstrumentClass:
    """Classify a contract from its symbol.

    Unknown bases are reported as :attr:`InstrumentClass.CRYPTO`, because a new
    listing this module has never heard of is far more likely to be a token
    than a tokenised share of a semiconductor company.
    """

    base = base_of(symbol, quote)
    if not base:
        return InstrumentClass.CRYPTO

    if any(marker in base for marker in _EQUITY_MARKERS):
        return InstrumentClass.TOKENISED_EQUITY
    if base in _INDEX_BASES:
        return InstrumentClass.INDEX
    if base in _COMMODITY_BASES or base in _COMMODITY_TOKENS:
        return InstrumentClass.COMMODITY
    if base in _FX_BASES:
        return InstrumentClass.FX
    if base in _STABLE_BASES:
        return InstrumentClass.STABLECOIN

    # Leveraged spot tokens (``BTC3L``, ``ETH5S``) decay by construction and
    # are not a directional instrument on any timeframe this bot trades.
    if re.fullmatch(r".+?\d+[LS]", base):
        return InstrumentClass.INDEX

    return InstrumentClass.CRYPTO


def parse_allowed_classes(values: Iterable[str]) -> frozenset[InstrumentClass]:
    """Turn configured names into a set, ignoring anything unrecognised."""

    allowed: set[InstrumentClass] = {InstrumentClass.CRYPTO}
    for raw in values:
        name = str(raw).strip().upper().replace("-", "_")
        if not name:
            continue
        if name in {"ALL", "*"}:
            return frozenset(InstrumentClass)
        # Accept both spellings so the setting is forgiving.
        name = name.replace("TOKENIZED", "TOKENISED")
        try:
            allowed.add(InstrumentClass(name))
        except ValueError:
            continue
    return frozenset(allowed)


__all__ = [
    "InstrumentClass",
    "NON_CRYPTO",
    "base_of",
    "classify_symbol",
    "parse_allowed_classes",
]
