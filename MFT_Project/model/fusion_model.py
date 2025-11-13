import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from qlib.model.base import Model
from qlib.data.dataset.handler import DataHandler, DataHandlerLP


class FusionTransformer(Model, nn.Module):
    """融合行情与文本特征的多层感知器模型。"""

    def __init__(
        self,
        market_dim,
        text_dim,
        hidden_dim,
        dropout=0.1,
        lr=1e-3,
        num_epochs=5,
        batch_size=256,
        **kwargs,
    ):
        Model.__init__(self)
        nn.Module.__init__(self)

        self.market_dim = market_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.market_proj = nn.Sequential(
            nn.Linear(market_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.prediction_layer = nn.Linear(hidden_dim, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self._fitted = False
        self._feature_columns = None
        self._text_columns = None

    def _split_market_frame(self, market_df: pd.DataFrame):
        if isinstance(market_df.columns, pd.MultiIndex):
            feature_df = market_df.xs("feature", axis=1, level=0)
            label_df = (
                market_df.xs("label", axis=1, level=0)
                if "label" in market_df.columns.get_level_values(0)
                else None
            )
        else:
            feature_df = market_df
            label_df = None
        if label_df is not None and not label_df.empty:
            label_series = label_df.iloc[:, 0]
        else:
            label_series = pd.Series(0.0, index=feature_df.index)
        return feature_df, label_series

    def _prepare_tensors(self, pack, require_label=True):
        market_df = pack["market_data"]
        feature_df, label_series = self._split_market_frame(market_df)
        text_df = pack.get("text_data")
        if text_df is None or text_df.empty:
            text_df = pd.DataFrame(
                np.zeros((len(feature_df), self.text_dim), dtype=np.float32),
                index=feature_df.index,
                columns=[f"text_feat_{i}" for i in range(self.text_dim)],
            )
        text_df = text_df.reindex(feature_df.index).fillna(0.0)

        market_tensor = torch.tensor(feature_df.values, dtype=torch.float32)
        text_tensor = torch.tensor(text_df.values, dtype=torch.float32)
        if require_label:
            labels = torch.tensor(label_series.values, dtype=torch.float32).unsqueeze(1)
        else:
            labels = None

        if self._feature_columns is None:
            self._feature_columns = feature_df.columns.tolist()
        if self._text_columns is None:
            self._text_columns = text_df.columns.tolist()

        return market_tensor, text_tensor, labels, feature_df.index

    def forward(self, market_tensor, text_tensor):
        market_feat = self.market_proj(market_tensor)
        text_feat = self.text_proj(text_tensor)
        fused = torch.cat([market_feat, text_feat], dim=-1)
        fused = self.fusion_layer(fused)
        return self.prediction_layer(fused)

    def fit(self, dataset, reweighter=None):
        pack = dataset.prepare_with_text("train", col_set=DataHandler.CS_RAW, data_key=DataHandlerLP.DK_L)
        market_tensor, text_tensor, labels, _ = self._prepare_tensors(pack, require_label=True)

        tensor_dataset = TensorDataset(market_tensor, text_tensor, labels)
        loader = DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.train()
        for _ in range(self.num_epochs):
            for market_batch, text_batch, label_batch in loader:
                market_batch = market_batch.to(self.device)
                text_batch = text_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                optimizer.zero_grad()
                preds = self.forward(market_batch, text_batch)
                loss = criterion(preds, label_batch)
                loss.backward()
                optimizer.step()

        self.eval()
        self._fitted = True

    def predict(self, dataset, segment="test"):
        if not self._fitted:
            raise RuntimeError("Model is not fitted yet")

        pack = dataset.prepare_with_text(segment, col_set=DataHandler.CS_RAW, data_key=DataHandlerLP.DK_I)
        market_tensor, text_tensor, _, index = self._prepare_tensors(pack, require_label=False)
        market_tensor = market_tensor.to(self.device)
        text_tensor = text_tensor.to(self.device)
        with torch.no_grad():
            preds = self.forward(market_tensor, text_tensor).cpu().squeeze()
        return pd.Series(preds.numpy(), index=index)

# 使用示例：
# model = FusionTransformer(market_dim=32, text_dim=768, hidden_dim=128)
# output = model({'market_data': market_tensor, 'text_data': text_tensor})
