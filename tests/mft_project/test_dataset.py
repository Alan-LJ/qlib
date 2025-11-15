import pandas as pd
import qlib
from qlib.data.dataset.handler import DataHandlerLP

from MFT_Project.data import GenerationConfig, TextFeatureConfig, generate_market_data, generate_text_features
from MFT_Project.dataset import MultiModalDataset


def _setup_synthetic_environment(tmp_path, start="2023-01-02", end="2023-01-10"):
    market_cfg = GenerationConfig(
        start=start,
        end=end,
        ticker_count=4,
        feature_count=4,
        seed=1,
        output_root=tmp_path,
    )
    text_cfg = TextFeatureConfig(
        start=start,
        end=end,
        ticker_count=4,
        embedding_dim=6,
        seed=2,
        output_root=tmp_path,
    )
    generate_market_data(market_cfg)
    generate_text_features(text_cfg)
    return market_cfg, text_cfg


def test_multimodal_dataset_prepare(tmp_path):
    market_cfg, text_cfg = _setup_synthetic_environment(tmp_path)
    qlib.init(provider_uri=str(tmp_path), region="cn", reload=True)

    handler_config = {
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
        "train": (market_cfg.start, market_cfg.end),
        "valid": (market_cfg.start, market_cfg.end),
        "test": (market_cfg.start, market_cfg.end),
    }

    dataset = MultiModalDataset(
        handler=handler_config,
        segments=segments,
        text_feature_path=str(text_cfg.output_path),
        lazy_text_loading=True,
    )

    payload = dataset.prepare_with_text("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    assert "market" in payload and "text" in payload

    market_frame = payload["market"]
    text_frame = payload["text"]
    assert isinstance(market_frame, pd.DataFrame)
    assert isinstance(text_frame, pd.DataFrame)
    assert market_frame.xs("feature", level=0, axis=1).shape[1] == market_cfg.feature_count
    assert text_frame.shape[1] == text_cfg.embedding_dim
