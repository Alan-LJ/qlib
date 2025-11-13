import numpy as np
import pandas as pd
from pathlib import Path

DATA_ROOT = Path("p:/PyCharm/ai4finance/qlib/MFT_Project/data")
CALENDAR_PATH = DATA_ROOT / "calendars" / "day.txt"
INSTRUMENT_PATH = DATA_ROOT / "instruments" / "all.txt"
OUTPUT_PATH = DATA_ROOT / "dummy_text_features.pkl"

START_DATE = "2023-01-02"
END_DATE = "2023-01-06"
TEXT_DIM = 256  # 与 FusionTransformer 配置的 text_dim 对齐


def load_calendar(start_date: str, end_date: str):
    with CALENDAR_PATH.open() as f:
        all_days = [line.strip() for line in f if line.strip()]
    days = [day for day in all_days if start_date <= day <= end_date]
    if not days:
        raise ValueError("指定时间范围内无交易日，请检查 calendars/day.txt")
    return pd.to_datetime(days)


def load_instruments():
    codes = []
    with INSTRUMENT_PATH.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                codes.append(parts[0])
    if not codes:
        raise ValueError("instruments/all.txt 中无股票代码")
    return codes


def main():
    trade_days = load_calendar(START_DATE, END_DATE)
    instruments = load_instruments()
    index = pd.MultiIndex.from_product([trade_days, instruments], names=["datetime", "instrument"])
    num_rows = len(index)
    data = np.random.randn(num_rows, TEXT_DIM).astype(np.float32)
    columns = [f"text_feat_{i}" for i in range(TEXT_DIM)]
    df = pd.DataFrame(data, index=index, columns=columns)
    df.to_pickle(OUTPUT_PATH)
    print(f"Dummy text features saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
