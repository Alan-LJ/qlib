"""Multi-modal dataset definitions for the MFT project.

This module bridges Qlib's handler-based dataset pipeline with additional text
features stored as static artefacts. The resulting dataset returns a
dictionary-like structure providing both market and text views that can be fed
into multi-input neural networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP

from MFT_Project.data import StaticDataError, StaticTextFeatureLoader


@dataclass
class TextView:
    """Container framing text features for a specific market slice."""

    data: pd.DataFrame

    def to_numpy(self) -> pd.DataFrame:
        """Return the underlying dataframe (kept for API symmetry)."""

        return self.data


@dataclass
class MarketView:
    """Container framing market features and labels for a specific slice."""

    data: pd.DataFrame

    def features(self) -> pd.DataFrame:
        return self.data.xs("feature", level=0, axis=1)

    def labels(self) -> pd.DataFrame:
        return self.data.xs("label", level=0, axis=1)

    def to_pandas(self) -> pd.DataFrame:
        return self.data


@dataclass
class MultiModalBatch:
    """Return type for :class:`MultiModalDataset.prepare`."""

    market: MarketView
    text: Optional[TextView]

    def __iter__(self):
        yield self.market
        yield self.text

    def as_dict(self) -> Mapping[str, Optional[pd.DataFrame]]:
        return {
            "market": self.market.data,
            "text": None if self.text is None else self.text.data,
        }


class MultiModalDataset(DatasetH):
    """Dataset that augments Qlib market data with auxiliary text features."""

    def __init__(
        self,
        handler: Union[dict, DataHandler],
        segments: Mapping[str, Tuple[str, str]],
        text_feature_path: Optional[Union[str, Path]] = None,
        lazy_text_loading: bool = True,
        **kwargs,
    ):
        self.text_feature_path = Path(text_feature_path) if text_feature_path else None
        self.lazy_text_loading = lazy_text_loading
        self._text_cache: Optional[pd.DataFrame] = None
        super().__init__(handler=handler, segments=segments, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def setup_data(self, handler_kwargs: dict = None, **kwargs):  # type: ignore[override]
        super().setup_data(handler_kwargs=handler_kwargs, **kwargs)
        if self.text_feature_path and not self.lazy_text_loading:
            self._text_cache = self._load_text_features(self.text_feature_path)

    def _load_text_features(self, path: Path) -> pd.DataFrame:
        loader = StaticTextFeatureLoader(feature_path=path)
        return loader.load()

    def _get_text_frame(self) -> Optional[pd.DataFrame]:
        if self.text_feature_path is None:
            return None
        if self._text_cache is not None:
            return self._text_cache
        self._text_cache = self._load_text_features(self.text_feature_path)
        return self._text_cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare(
        self,
        segments: Union[List[str], Tuple[str, ...], str, slice, pd.Index],
        col_set: str = DataHandler.CS_ALL,
        data_key: str = DataHandlerLP.DK_I,
        **kwargs,
    ) -> Union[MultiModalBatch, List[MultiModalBatch]]:  # type: ignore[override]
        market_data = super().prepare(segments=segments, col_set=col_set, data_key=data_key, **kwargs)
        if isinstance(market_data, list):
            return [self._bundle(frame) for frame in market_data]
        return self._bundle(market_data)

    def prepare_market(self, *args, **kwargs) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Expose the underlying market dataframe(s) without text augmentation."""

        return super().prepare(*args, **kwargs)

    def prepare_with_text(
        self,
        segments: Union[List[str], Tuple[str, ...], str, slice, pd.Index],
        col_set: str = DataHandler.CS_ALL,
        data_key: str = DataHandlerLP.DK_I,
        **kwargs,
    ) -> Union[Mapping[str, Optional[pd.DataFrame]], List[Mapping[str, Optional[pd.DataFrame]]]]:
        """Compatibility wrapper returning dictionary payloads."""

        batches = self.prepare(segments, col_set=col_set, data_key=data_key, **kwargs)
        if isinstance(batches, list):
            return [batch.as_dict() for batch in batches]
        return batches.as_dict()

    # ------------------------------------------------------------------
    # Packaging helpers
    # ------------------------------------------------------------------
    def _bundle(self, frame: pd.DataFrame) -> MultiModalBatch:
        text_frame = self._slice_text(frame.index)
        return MultiModalBatch(
            market=MarketView(frame),
            text=None if text_frame is None else TextView(text_frame),
        )

    def _slice_text(self, index: pd.MultiIndex) -> Optional[pd.DataFrame]:
        text_frame = self._get_text_frame()
        if text_frame is None:
            return None
        try:
            return text_frame.loc[index]
        except KeyError as exc:  # pragma: no cover - defensive guard
            missing = index.difference(text_frame.index)
            raise StaticDataError(
                f"Text features missing rows for indices: {missing[:5]} (total {len(missing)})"
            ) from exc


__all__ = [
    "TextView",
    "MarketView",
    "MultiModalBatch",
    "MultiModalDataset",
]
