import os
import pickle
import pandas as pd

# 配置路径
features_root = "p:/PyCharm/ai4finance/qlib/MFT_Project/data/features"
instruments_path = "p:/PyCharm/ai4finance/qlib/MFT_Project/data/instruments/all.txt"
calendar_path = "p:/PyCharm/ai4finance/qlib/MFT_Project/data/calendars/day.txt"

# 获取股票代码（all.txt，首列，保留大小写）
codes = []
with open(instruments_path) as f:
    for line in f:
        if line.strip():
            code = line.split('\t')[0].strip()
            codes.append(code)

# 获取交易日（day.txt）
days = []
with open(calendar_path) as f:
    for line in f:
        d = line.strip()
        if d:
            days.append(d)

# 检查每只股票的特征文件夹和文件
report = []
for code in codes:
    code_dir = os.path.join(features_root, code)
    exists = os.path.isdir(code_dir)
    files = []
    if exists:
        files = os.listdir(code_dir)
    else:
        report.append(f"股票 {code} 缺少特征文件夹！")
        continue
    # 检查每个 .day.bin 文件
    for fname in files:
        if fname.endswith(".day.bin"):
            fpath = os.path.join(code_dir, fname)
            try:
                with open(fpath, "rb") as f:
                    df = pickle.load(f)
                # 检查 DataFrame 格式
                if not isinstance(df, pd.DataFrame):
                    report.append(f"{code}/{fname} 不是 DataFrame！")
                elif df.empty:
                    report.append(f"{code}/{fname} DataFrame 为空！")
                elif not all(isinstance(i, str) for i in df.index):
                    report.append(f"{code}/{fname} index 不是字符串日期！")
                elif not all(day in days for day in df.index):
                    report.append(f"{code}/{fname} index 有不在 day.txt 的日期！")
            except Exception as e:
                report.append(f"{code}/{fname} 加载失败: {e}")
    # 检查是否有至少一个有效特征文件
    valid_files = [fname for fname in files if fname.endswith(".day.bin")]
    if not valid_files:
        report.append(f"{code} 没有任何 .day.bin 特征文件！")

# 输出报告
if not report:
    print("所有特征文件格式和路径均正确！")
else:
    print("特征文件检查报告：")
    for r in report:
        print(r)
