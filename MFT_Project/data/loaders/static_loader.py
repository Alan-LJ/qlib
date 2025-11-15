"""Helper utilities for loading static artefacts used by the MFT project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


class StaticDataError(RuntimeError):
    """Raised when static artefacts do not satisfy expected constraints."""


@dataclass
class StaticMarketDataLoader:
    """Load and validate synthetic market data stored as a pickle."""

    feature_path: Path

    def load(self) -> pd.DataFrame:
        frame = pd.read_pickle(self.feature_path)
        _validate_multiindex(frame.index)
        _validate_column_hierarchy(frame.columns)
        return frame


@dataclass
class StaticTextFeatureLoader:
    """Load embedding features aligned with the market static data."""

    feature_path: Path

    def load(self) -> pd.DataFrame:
        frame = pd.read_pickle(self.feature_path)
        _validate_multiindex(frame.index)
        if isinstance(frame.columns, pd.MultiIndex):
            raise StaticDataError("Text feature frame should use a single-level column index.")
        return frame


def _validate_multiindex(index: pd.Index) -> None:
    if not isinstance(index, pd.MultiIndex):
        raise StaticDataError("Expected multi-index (datetime, instrument).")
    if list(index.names) != ["datetime", "instrument"]:
        raise StaticDataError("Index levels must be named 'datetime' and 'instrument'.")


def _validate_column_hierarchy(columns: pd.Index) -> None:
    if not isinstance(columns, pd.MultiIndex):
        raise StaticDataError("Market feature frame must use a hierarchical column index.")
    expected_top_levels = {"feature", "label"}
    top_levels = set(columns.get_level_values(0))
    missing = expected_top_levels - top_levels
    if missing:
        raise StaticDataError(f"Missing top-level column groups: {sorted(missing)}")


def load_market_and_text(market_path: Path, text_path: Optional[Path]) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Convenience wrapper returning both market and text frames."""

    market = StaticMarketDataLoader(market_path=market_path).load()
    text = None
    if text_path is not None:
        text = StaticTextFeatureLoader(feature_path=text_path).load()
        _align_indices(market, text)
    return market, text


def _align_indices(market: pd.DataFrame, text: pd.DataFrame) -> None:
    if not market.index.equals(text.index):
        raise StaticDataError("Market and text feature indices do not align.")
