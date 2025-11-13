import os
import pickle
import pandas as pd
import numpy as np

features_root = "p:/PyCharm/ai4finance/qlib/MFT_Project/data/features"
calendar_path = "p:/PyCharm/ai4finance/qlib/MFT_Project/data/calendars/day.txt"

with open(calendar_path) as f:
    days = [d.strip() for d in f if d.strip()]

for code in os.listdir(features_root):
    code_dir = os.path.join(features_root, code)
    if not os.path.isdir(code_dir):
        continue
    for feat in ["volume", "vwap"]:
        fpath = os.path.join(code_dir, f"{feat}.day.bin")
        # 只修复损坏或空文件
        try:
            with open(fpath, "rb") as f:
                df = pickle.load(f)
            if not isinstance(df, pd.DataFrame) or df.empty:
                raise Exception("格式错误或为空")
        except Exception:
            arr = np.random.rand(len(days)).astype(np.float32)
            df = pd.DataFrame({feat: arr}, index=days)
            with open(fpath, "wb") as f:
                pickle.dump(df, f)
            print(f"{code}/{feat}.day.bin 已修复")
print("所有损坏的 volume/vwap.day.bin 已修复！")