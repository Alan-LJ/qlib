"""Ticker symbol normalisation utilities shared across data generators."""

from __future__ import annotations

from typing import Iterable, List


def _extract_digits(token: str) -> str:
    digits = "".join(ch for ch in token if ch.isdigit())
    if not digits:
        raise ValueError(f"Unable to parse numeric part from ticker '{token}'.")
    if len(digits) > 6:
        digits = digits[-6:]
    return digits.zfill(6)


def _infer_exchange(token: str, digits: str) -> str:
    upper = token.upper()
    if upper.endswith(".SH") or upper.startswith("SH"):
        return "SH"
    if upper.endswith(".SZ") or upper.startswith("SZ"):
        return "SZ"
    if digits.startswith(("0", "3")):
        return "SZ"
    if digits.startswith("6"):
        return "SH"
    raise ValueError(f"Cannot infer exchange for ticker '{token}'.")


def as_qlib_code(token: str) -> str:
    """Return ticker in Qlib format (e.g. ``000001.SZ``)."""

    stripped = token.strip()
    if not stripped:
        raise ValueError("Ticker string is empty.")
    digits = _extract_digits(stripped)
    exchange = _infer_exchange(stripped, digits)
    return f"{digits}.{exchange}"


def to_tencent_symbol(token: str) -> str:
    """Convert ticker to Tencent API symbol (e.g. ``sz000001``)."""

    qlib_code = as_qlib_code(token)
    digits, exchange = qlib_code.split(".")
    prefix = exchange.lower()
    return f"{prefix}{digits}"


def normalise_tickers(tickers: Iterable[str]) -> List[str]:
    """Normalise an iterable of ticker strings into Qlib codes."""

    return [as_qlib_code(item) for item in tickers]


__all__ = ["as_qlib_code", "to_tencent_symbol", "normalise_tickers"]
