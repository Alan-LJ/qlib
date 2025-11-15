"""PyTorch-based multi-modal fusion model for the MFT project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from qlib.data.dataset import Dataset
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.workflow import R

from MFT_Project.dataset import MultiModalBatch, MultiModalDataset


# ---------------------------------------------------------------------------
# Neural network building blocks
# ---------------------------------------------------------------------------


def _as_sequence(hidden_dims: Union[int, Sequence[int]]) -> Tuple[int, ...]:
    if isinstance(hidden_dims, Iterable) and not isinstance(hidden_dims, (int, float)):
        return tuple(int(h) for h in hidden_dims)
    return (int(hidden_dims),)


def _build_mlp(input_dim: int, hidden_dims: Sequence[int], activation: nn.Module, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    return nn.Sequential(*layers)


class MarketTower(nn.Module):
    """Encodes market features via a configurable MLP."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], activation: nn.Module, dropout: float):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dims, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextTower(nn.Module):
    """Encodes text embeddings via a configurable MLP."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], activation: nn.Module, dropout: float):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dims, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionHead(nn.Module):
    """Combines market and text representations to produce a scalar score."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], activation: nn.Module, dropout: float):
        super().__init__()
        if hidden_dims:
            self.mlp = _build_mlp(input_dim, hidden_dims, activation, dropout)
            final_dim = hidden_dims[-1]
        else:
            self.mlp = nn.Identity()
            final_dim = input_dim
        self.output = nn.Linear(final_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.mlp(x)
        return self.output(hidden).squeeze(-1)


class FusionRegressor(nn.Module):
    """Full fusion network orchestrating individual towers and the head."""

    def __init__(
        self,
        market_input_dim: int,
        market_hidden_dims: Sequence[int],
        fusion_hidden_dims: Sequence[int],
        activation: nn.Module,
        dropout: float,
        text_input_dim: Optional[int] = None,
        text_hidden_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.market_tower = MarketTower(market_input_dim, market_hidden_dims, activation, dropout)
        self.has_text = text_input_dim is not None and text_input_dim > 0
        if self.has_text:
            assert text_hidden_dims is not None and len(text_hidden_dims) > 0, "Text tower requires hidden dims"
            self.text_tower = TextTower(text_input_dim, text_hidden_dims, activation, dropout)
            fusion_input_dim = market_hidden_dims[-1] + text_hidden_dims[-1]
        else:
            self.text_tower = None
            fusion_input_dim = market_hidden_dims[-1]
        self.fusion_head = FusionHead(fusion_input_dim, fusion_hidden_dims, activation, dropout)

    def forward(self, market: torch.Tensor, text: Optional[torch.Tensor] = None) -> torch.Tensor:
        market_repr = self.market_tower(market)
        if self.has_text and text is not None:
            text_repr = self.text_tower(text)
            fused = torch.cat([market_repr, text_repr], dim=1)
        else:
            fused = market_repr
        return self.fusion_head(fused)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _mask_and_stack(
    market: pd.DataFrame,
    text: Optional[pd.DataFrame],
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, pd.Index]:
    """Convert pandas frames to numpy arrays while dropping NaNs."""

    feature_frame = market.xs("feature", level=0, axis=1)
    label_frame = market.xs("label", level=0, axis=1)
    mask = feature_frame.notna().all(axis=1)
    mask &= label_frame.notna().all(axis=1)
    text_array: Optional[np.ndarray]
    if text is not None:
        mask &= text.notna().all(axis=1)
        text_array = text.loc[mask].to_numpy(dtype=np.float32, copy=True)
    else:
        text_array = None
    features = feature_frame.loc[mask].to_numpy(dtype=np.float32, copy=True)
    labels_raw = label_frame.loc[mask].to_numpy(dtype=np.float32, copy=True)
    if labels_raw.ndim == 2:
        if labels_raw.shape[1] != 1:
            raise ValueError("FusionModel currently supports a single label column.")
        labels = labels_raw[:, 0]
    else:
        labels = labels_raw
    index = feature_frame.loc[mask].index
    return features, text_array, labels, index


@dataclass
class TrainingBatch:
    features: torch.Tensor
    labels: torch.Tensor
    text: Optional[torch.Tensor]
    index: pd.Index


def _to_training_batch(
    payload: Mapping[str, Optional[pd.DataFrame]],
    device: torch.device,
) -> TrainingBatch:
    market_frame = payload["market"]
    if market_frame is None:
        raise ValueError("Market data is required for training.")
    text_frame = payload.get("text")
    features_np, text_np, labels_np, index = _mask_and_stack(market_frame, text_frame)
    features_tensor = torch.from_numpy(features_np).to(device)
    labels_tensor = torch.from_numpy(labels_np).to(device)
    text_tensor = torch.from_numpy(text_np).to(device) if text_np is not None else None
    return TrainingBatch(features_tensor, labels_tensor, text_tensor, index)


# ---------------------------------------------------------------------------
# Fusion model orchestrator
# ---------------------------------------------------------------------------


class FusionModel(Model):
    """Qlib model wrapping the multi-modal fusion network."""

    def __init__(
        self,
        market_dim: int,
        text_dim: Optional[int] = None,
        market_hidden_dims: Union[int, Sequence[int]] = (256, 128),
        text_hidden_dims: Optional[Union[int, Sequence[int]]] = (256,),
        fusion_hidden_dims: Union[int, Sequence[int]] = 128,
        activation: str = "relu",
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        n_epochs: int = 50,
        batch_size: int = 4096,
        patience: int = 5,
        loss: str = "mse",
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.logger = get_module_logger("FusionModel")
        self.market_dim = market_dim
        self.text_dim = text_dim
        self.market_hidden_dims = _as_sequence(market_hidden_dims)
        self.text_hidden_dims = _as_sequence(text_hidden_dims) if text_hidden_dims is not None else None
        self.fusion_hidden_dims = _as_sequence(fusion_hidden_dims) if fusion_hidden_dims else ()
        if len(self.market_hidden_dims) == 0:
            raise ValueError("market_hidden_dims must contain at least one layer size.")
        if self.text_dim and (self.text_hidden_dims is None or len(self.text_hidden_dims) == 0):
            raise ValueError("text_hidden_dims must be provided when text_dim is set.")
        self.dropout = float(dropout)
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.loss_name = loss
        self.seed = seed
        self.activation = activation

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("Using device %s", self.device)

        activation_module = self._get_activation()
        self.network = FusionRegressor(
            market_input_dim=market_dim,
            market_hidden_dims=self.market_hidden_dims,
            text_input_dim=text_dim,
            text_hidden_dims=self.text_hidden_dims,
            fusion_hidden_dims=self.fusion_hidden_dims,
            activation=activation_module,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.MSELoss() if loss == "mse" else nn.L1Loss()
        self._fitted = False

    # ------------------------------------------------------------------
    def _get_activation(self) -> nn.Module:
        if self.activation.lower() == "relu":
            return nn.ReLU()
        if self.activation.lower() == "gelu":
            return nn.GELU()
        if self.activation.lower() == "elu":
            return nn.ELU()
        raise ValueError(f"Unsupported activation: {self.activation}")

    # ------------------------------------------------------------------
    def fit(self, dataset: Dataset, reweighter=None):  # type: ignore[override]
        if not isinstance(dataset, MultiModalDataset):
            raise TypeError("FusionModel expects a MultiModalDataset instance.")

        train_payload = dataset.prepare_with_text(
            "train",
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        valid_payload = None
        if "valid" in dataset.segments:
            valid_payload = dataset.prepare_with_text(
                "valid",
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )

        train_batch = _to_training_batch(train_payload, self.device)
        valid_batch = _to_training_batch(valid_payload, self.device) if valid_payload else None

        if train_batch.features.numel() == 0:
            raise ValueError("Training set is empty after filtering NaNs.")

        recorder = None
        try:
            recorder = R.get_recorder()
        except Exception:  # pragma: no cover - recorder may not exist in unit tests
            recorder = None

        best_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None

        indices = torch.arange(train_batch.features.size(0))
        for epoch in range(self.n_epochs):
            permutation = indices[torch.randperm(len(indices))]
            epoch_loss = 0.0
            self.network.train()
            for start in range(0, len(permutation), self.batch_size):
                excerpt = permutation[start : start + self.batch_size]
                features = train_batch.features[excerpt]
                labels = train_batch.labels[excerpt]
                text = train_batch.text[excerpt] if train_batch.text is not None else None

                self.optimizer.zero_grad()
                prediction = self.network(features, text)
                loss = self.criterion(prediction, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
                self.optimizer.step()

                epoch_loss += loss.item() * len(excerpt)

            epoch_loss /= len(permutation)
            valid_loss = None
            if valid_batch is not None and valid_batch.features.numel() > 0:
                valid_loss = self._evaluate(valid_batch)
                if valid_loss + 1e-8 < best_loss:
                    best_loss = valid_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            else:
                if epoch_loss + 1e-8 < best_loss:
                    best_loss = epoch_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}

            if recorder is not None:
                metrics = {"train_loss": epoch_loss}
                if valid_loss is not None:
                    metrics["valid_loss"] = valid_loss
                recorder.log_metrics(step=epoch, **metrics)

            if self.patience and epochs_without_improvement >= self.patience:
                self.logger.info("Early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            self.network.load_state_dict(best_state)

        self._fitted = True
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def _evaluate(self, batch: TrainingBatch) -> float:
        self.network.eval()
        with torch.no_grad():
            predictions = self.network(batch.features, batch.text)
            loss = self.criterion(predictions, batch.labels)
        return float(loss.item())

    # ------------------------------------------------------------------
    def predict(self, dataset: Dataset, segment: Union[str, slice] = "test") -> pd.Series:  # type: ignore[override]
        if not self._fitted:
            raise ValueError("Model has not been fitted yet.")
        if not isinstance(dataset, MultiModalDataset):
            raise TypeError("FusionModel expects a MultiModalDataset instance.")

        payload = dataset.prepare_with_text(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        if isinstance(payload, list):
            raise ValueError("Prediction expects a single segment, not a list.")
        market_frame = payload["market"]
        if market_frame is None:
            raise ValueError("Market data is required for prediction.")
        if isinstance(market_frame.columns, pd.MultiIndex):
            features_frame = market_frame.xs("feature", level=0, axis=1)
        else:
            features_frame = market_frame
        text_frame = payload.get("text")

        mask = features_frame.notna().all(axis=1)
        if text_frame is not None:
            mask &= text_frame.notna().all(axis=1)
        selected_index = features_frame.loc[mask].index
        if len(selected_index) == 0:
            return pd.Series(dtype=float, index=selected_index)

        features = torch.from_numpy(features_frame.loc[mask].to_numpy(dtype=np.float32, copy=True)).to(self.device)
        text_tensor = (
            torch.from_numpy(text_frame.loc[mask].to_numpy(dtype=np.float32, copy=True)).to(self.device)
            if text_frame is not None
            else None
        )

        self.network.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                excerpt = slice(start, start + self.batch_size)
                batch_features = features[excerpt]
                batch_text = text_tensor[excerpt] if text_tensor is not None else None
                output = self.network(batch_features, batch_text)
                preds.append(output.cpu().numpy())
        all_preds = np.concatenate(preds)
        return pd.Series(all_preds, index=selected_index)

    # ------------------------------------------------------------------
    def get_backbone(self) -> nn.Module:
        """Expose the PyTorch module for advanced users (e.g. SHAP)."""

        return self.network

    def get_device(self) -> torch.device:
        return self.device


__all__ = ["FusionModel", "FusionRegressor", "MarketTower", "TextTower", "FusionHead"]
