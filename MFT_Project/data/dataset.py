import os
import pickle
import pandas as pd
from qlib.data.dataset import DatasetH


class MultiModalDataset(DatasetH):
    """融合行情数据与文本特征的数据集。"""

    def __init__(self, *args, text_feature_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_feature_path = text_feature_path
        self.text_feature_df = self._load_text_features(text_feature_path) if text_feature_path else None

    def _load_text_features(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"文本特征文件不存在: {path}")
        with open(path, "rb") as f:
            df = pickle.load(f)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("加载的文本特征不是 DataFrame 类型")
        return df

    def _slice_text_feature(self, slc):
        if self.text_feature_df is None:
            return None
        try:
            if isinstance(slc, tuple):
                start, end = slc
                start = None if start is None else start
                end = None if end is None else end
                return self.text_feature_df.loc[pd.IndexSlice[start:end, :]]
            if isinstance(slc, slice):
                return self.text_feature_df.loc[slc]
            return self.text_feature_df.loc[slc]
        except KeyError:
            return pd.DataFrame(columns=self.text_feature_df.columns)

    def get_text_segment(self, segment):
        if isinstance(segment, str) and segment in self.segments:
            slc = self.segments[segment]
        else:
            slc = segment
        return self._slice_text_feature(slc)

    def prepare_with_text(self, segments, **kwargs):
        if isinstance(segments, (list, tuple)):
            market_list = super().prepare(segments, **kwargs)
            return [
                {
                    "market_data": market_df,
                    "text_data": self.get_text_segment(seg),
                }
                for market_df, seg in zip(market_list, segments)
            ]
        market_df = super().prepare(segments, **kwargs)
        return {
            "market_data": market_df,
            "text_data": self.get_text_segment(segments),
        }

# 使用示例：
# dataset = MultiModalDataset(text_feature_path='path/to/text_features.pkl', ...qlib参数...)
