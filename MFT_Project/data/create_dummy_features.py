import os
import numpy as np
import pandas as pd
import pickle

# 配置参数
feature_names = [
    "adjclose", "amount", "change", "close", "factor", "high", "low", "open"
]
start_date = "2023-01-01"
end_date = "2025-09-30"
calendar_path = "MFT_Project/data/calendars/day.txt"
instruments_path = "MFT_Project/data/instruments/all.txt"
features_root = "MFT_Project/data/features"

# 获取交易日
with open(calendar_path) as f:
    all_days = [d.strip() for d in f if d.strip()]
days = [d for d in all_days if start_date <= d <= end_date]

# 获取股票代码
with open(instruments_path) as f:
    codes = [line.split('\t')[0].strip() for line in f if line.strip()]

# 生成特征文件
for code in codes:
    code_dir = os.path.join(features_root, code)
    os.makedirs(code_dir, exist_ok=True)
    for feat in feature_names:
        arr = np.random.rand(len(days)).astype(np.float32)
        df = pd.DataFrame({feat: arr}, index=days)
        # 保存为 .day.bin（Qlib 兼容 pickle 格式）
        with open(os.path.join(code_dir, f"{feat}.day.bin"), "wb") as f:
            pickle.dump(df, f)
print("Dummy 特征文件已全部生成！")