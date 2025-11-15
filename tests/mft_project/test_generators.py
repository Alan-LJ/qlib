import pandas as pd

from MFT_Project.data import GenerationConfig, TextFeatureConfig, generate_market_data, generate_text_features


def test_generate_market_and_text_data(tmp_path):
    start, end = "2023-01-02", "2023-01-10"
    market_cfg = GenerationConfig(
        start=start,
        end=end,
        ticker_count=5,
        feature_count=4,
        seed=123,
        output_root=tmp_path,
    )
    text_cfg = TextFeatureConfig(
        start=start,
        end=end,
        ticker_count=5,
        embedding_dim=8,
        seed=456,
        output_root=tmp_path,
    )

    generate_market_data(market_cfg)
    generate_text_features(text_cfg)

    market_frame = pd.read_pickle(market_cfg.market_path)
    text_frame = pd.read_pickle(text_cfg.output_path)

    assert market_frame.index.names == ["datetime", "instrument"]
    assert ("feature", "$feat_0") in market_frame.columns
    assert ("label", "LABEL0") in market_frame.columns
    assert text_frame.shape[1] == text_cfg.embedding_dim
    assert market_frame.index.equals(text_frame.index)
