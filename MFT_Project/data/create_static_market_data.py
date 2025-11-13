import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path("p:/PyCharm/ai4finance/qlib/MFT_Project/data")
CALENDAR_PATH = DATA_ROOT / "calendars" / "day.txt"
INSTRUMENT_PATH = DATA_ROOT / "instruments" / "all.txt"
OUTPUT_PATH = DATA_ROOT / "static_market_data.pkl"

# 配置生成数据所用的时间范围
START_DATE = "2023-01-02"
END_DATE = "2023-01-06"

# 特征列名称，首层 MultiIndex 使用 "feature"，第二层以 $ 开头与 Qlib 保持一致
FEATURE_NAMES = ["$open", "$high", "$low", "$close", "$volume", "$vwap", "$factor"]
LABEL_NAME = "LABEL0"


def load_calendar(start_date: str, end_date: str) -> pd.DatetimeIndex:
    with CALENDAR_PATH.open() as f:
        all_days = [line.strip() for line in f if line.strip()]
    days = [day for day in all_days if start_date <= day <= end_date]
    if not days:
        raise ValueError("指定时间范围内没有交易日，请确认 day.txt 覆盖该区间")
    return pd.to_datetime(days)


def load_instruments() -> list:
    codes = []
    with INSTRUMENT_PATH.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            code = parts[0]
            codes.append(code)
    if not codes:
        raise ValueError("all.txt 中没有股票代码")
    return codes


def build_dataframe(trade_days: pd.DatetimeIndex, instruments: list) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([trade_days, instruments], names=["datetime", "instrument"])
    feature_data = {}
    for feat in FEATURE_NAMES:
        feature_data[feat] = np.random.randn(len(index)).astype(np.float32)
    feature_df = pd.DataFrame(feature_data, index=index)
    feature_df.columns = pd.MultiIndex.from_product([["feature"], FEATURE_NAMES])

    label_values = np.random.randn(len(index)).astype(np.float32)
    label_df = pd.DataFrame({LABEL_NAME: label_values}, index=index)
    label_df.columns = pd.MultiIndex.from_product([["label"], [LABEL_NAME]])

    df = pd.concat([feature_df, label_df], axis=1)
    return df


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    trade_days = load_calendar(START_DATE, END_DATE)
    instruments = load_instruments()
    df = build_dataframe(trade_days, instruments)
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(df, f)
    print(f"静态市场数据已生成，共 {len(trade_days)} 个交易日，{len(instruments)} 只股票。")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
