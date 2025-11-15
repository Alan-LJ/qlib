import qlib
import pandas as pd

from MFT_Project.data import GenerationConfig, TextFeatureConfig, generate_market_data, generate_text_features
from MFT_Project.dataset import MultiModalDataset
from MFT_Project.model import FusionModel


def _build_dataset(tmp_path):
    start, end = "2023-01-02", "2023-01-12"
    market_cfg = GenerationConfig(
        start=start,
        end=end,
        ticker_count=6,
        feature_count=5,
        seed=7,
        output_root=tmp_path,
    )
    text_cfg = TextFeatureConfig(
        start=start,
        end=end,
        ticker_count=6,
        embedding_dim=8,
        seed=8,
        output_root=tmp_path,
    )
    generate_market_data(market_cfg)
    generate_text_features(text_cfg)

    qlib.init(provider_uri=str(tmp_path), region="cn", reload=True)

    handler = {
        "class": "qlib.data.dataset.handler.DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "instruments": None,
            "start_time": market_cfg.start,
            "end_time": market_cfg.end,
            "data_loader": {
                "class": "StaticDataLoader",
                "module_path": "qlib.data.dataset.loader",
                "kwargs": {"config": str(market_cfg.market_path)},
            },
        },
    }
    segments = {
        "train": (market_cfg.start, "2023-01-09"),
        "valid": ("2023-01-10", "2023-01-11"),
        "test": ("2023-01-12", "2023-01-12"),
    }
    dataset = MultiModalDataset(
        handler=handler,
        segments=segments,
        text_feature_path=str(text_cfg.output_path),
        lazy_text_loading=True,
    )
    return dataset, market_cfg, text_cfg


def test_fusion_model_training_and_prediction(tmp_path):
    dataset, market_cfg, text_cfg = _build_dataset(tmp_path)

    model = FusionModel(
        market_dim=market_cfg.feature_count,
        text_dim=text_cfg.embedding_dim,
        market_hidden_dims=(32,),
        text_hidden_dims=(32,),
        fusion_hidden_dims=(16,),
        dropout=0.0,
        lr=5e-3,
        n_epochs=3,
        batch_size=128,
        patience=2,
        loss="mse",
        seed=42,
    )

    model.fit(dataset)
    preds = model.predict(dataset, segment="test")

    assert isinstance(preds, pd.Series)
    assert not preds.empty
    assert preds.index.names == ["datetime", "instrument"]
    assert preds.notna().all()
