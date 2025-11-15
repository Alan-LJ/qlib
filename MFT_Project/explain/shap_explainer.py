"""SHAP explainability utilities tailored for the fusion model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from qlib.data.dataset.handler import DataHandlerLP

from MFT_Project.dataset import MultiModalDataset
from MFT_Project.model import FusionModel

try:  # pragma: no cover - optional dependency
    import shap  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    shap = None  # type: ignore


@dataclass
class ShapResult:
    """Container for SHAP outputs."""

    values: pd.DataFrame
    base_value: float


class FusionShapExplainer:
    """Compute SHAP values for :class:`FusionModel` predictions."""

    def __init__(
        self,
        model: FusionModel,
        dataset: MultiModalDataset,
        background_segment: str = "train",
        background_size: int = 200,
    ) -> None:
        if shap is None:  # pragma: no cover - defensive guard
            raise ImportError("shap package is required for FusionShapExplainer. Please install shap>=0.41.")
        if not model._fitted:  # type: ignore[attr-defined]
            raise ValueError("模型尚未训练，无法计算 SHAP。请先调用 model.fit。")

        self.model = model
        self.dataset = dataset
        self.market_dim = model.market_dim
        self.text_dim = model.text_dim or 0
        self.device = model.get_device()
        self.network = model.get_backbone().to(self.device).eval()
        self.feature_names: List[str] = []

        background_data, _ = self._prepare_data(background_segment, limit=background_size)
        if background_data.shape[0] == 0:
            raise ValueError("背景样本为空，无法初始化 SHAP 解释器。")
        self.explainer = shap.KernelExplainer(self._predict, background_data)
        self.base_value = float(self.explainer.expected_value)

    # ------------------------------------------------------------------
    def explain(self, segment: str = "test", sample_size: Optional[int] = None) -> ShapResult:
        """Return SHAP values for a given segment as a dataframe."""

        data, index = self._prepare_data(segment, limit=sample_size)
        if data.shape[0] == 0:
            raise ValueError(f"分段 {segment} 中没有可用于 SHAP 的样本。")

        shap_values = self.explainer.shap_values(data)
        if isinstance(shap_values, list):  # pragma: no cover - regression returns array, but guard anyway
            shap_values = shap_values[0]
        shap_df = pd.DataFrame(shap_values, columns=self.feature_names, index=index)
        return ShapResult(values=shap_df, base_value=self.base_value)

    # ------------------------------------------------------------------
    def _predict(self, samples: np.ndarray) -> np.ndarray:
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        market_arr = arr[:, : self.market_dim]
        text_arr = arr[:, self.market_dim : self.market_dim + self.text_dim] if self.text_dim > 0 else None

        market_tensor = torch.from_numpy(market_arr).to(self.device)
        text_tensor = torch.from_numpy(text_arr).to(self.device) if text_arr is not None else None

        with torch.no_grad():
            outputs = self.network(market_tensor, text_tensor)
        return outputs.detach().cpu().numpy()

    # ------------------------------------------------------------------
    def _prepare_data(self, segment: str, limit: Optional[int] = None) -> Tuple[np.ndarray, pd.Index]:
        payload = self.dataset.prepare_with_text(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        if isinstance(payload, list):  # pragma: no cover - dataset.prepare may return list, guard for clarity
            raise ValueError("多段分片返回列表，请一次只解释单个分段。")
        market_frame = payload["market"]
        if market_frame is None:
            raise ValueError("缺少市场数据，无法构建 SHAP 输入。")
        if isinstance(market_frame.columns, pd.MultiIndex):
            feature_frame = market_frame.xs("feature", level=0, axis=1)
        else:
            feature_frame = market_frame
        text_frame = payload.get("text")

        combined, names = self._combine(feature_frame, text_frame)
        if limit is not None:
            combined = combined[:limit]
            index = feature_frame.index[:limit]
        else:
            index = feature_frame.index

        self.feature_names = names
        return combined, index

    def _combine(self, feature_frame: pd.DataFrame, text_frame: Optional[pd.DataFrame]) -> Tuple[np.ndarray, List[str]]:
        feature_values = feature_frame.to_numpy(dtype=np.float32, copy=True)
        if isinstance(feature_frame.columns, pd.MultiIndex):
            feature_names = ["/".join(str(part) for part in col if part != "" and part is not None) for col in feature_frame.columns]
        else:
            feature_names = feature_frame.columns.tolist()

        if feature_values.shape[1] != self.market_dim:
            raise ValueError(
                f"市场特征维度 {feature_values.shape[1]} 与模型配置 {self.market_dim} 不一致。"
            )

        if self.text_dim > 0:
            if text_frame is None:
                raise ValueError("模型启用了文本塔，但未提供文本特征。")
            if text_frame.shape[1] != self.text_dim:
                raise ValueError(
                    f"文本特征维度 {text_frame.shape[1]} 与模型配置 {self.text_dim} 不一致。"
                )
            text_values = text_frame.to_numpy(dtype=np.float32, copy=True)
            combined = np.hstack([feature_values, text_values])
            names = feature_names + text_frame.columns.tolist()
        else:
            combined = feature_values
            names = feature_names
        return combined, names


__all__ = ["FusionShapExplainer", "ShapResult"]
