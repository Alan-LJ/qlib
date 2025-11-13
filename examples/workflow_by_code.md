# Qlib 量化交易工作流完整代码指南

本文档包含了 `workflow_by_code.ipynb` 的完整代码工作流，展示了使用 Qlib 进行模型训练、回测和分析的完整流程。

## 1. 安装和环境设置

```python
import sys, site
from pathlib import Path

################################# 重要说明 #################################
#  如果 Colab 安装了最新的 numpy 和 pyqlib，                              #
#  用户应该重新启动运行时，以便成功运行后续单元格。                        #
########################################################################

try:
    import qlib
except ImportError:
    # 安装 qlib
    ! pip install --upgrade numpy
    ! pip install pyqlib
    if "google.colab" in sys.modules:
        # Google Colab 环境较旧，需要降级 pyyaml 以兼容其他包
        ! pip install pyyaml==5.4.1
    # 重新加载
    site.main()

scripts_dir = Path.cwd().parent.joinpath("scripts")
if not scripts_dir.joinpath("get_data.py").exists():
    # 下载 get_data.py 脚本
    scripts_dir = Path("~/tmp/qlib_code/scripts").expanduser().resolve()
    scripts_dir.mkdir(parents=True, exist_ok=True)
    import requests

    with requests.get("https://raw.githubusercontent.com/microsoft/qlib/main/scripts/get_data.py", timeout=10) as resp:
        with open(scripts_dir.joinpath("get_data.py"), "wb") as fp:
            fp.write(resp.content)
```

## 2. 导入必要的库

```python
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
```

**库说明**：
- `qlib`: 主要库
- `pandas`: 数据处理
- `REG_CN`: 中国地区常量
- `exists_qlib_data`: 检查数据存在性
- `init_instance_by_config`: 从配置初始化实例
- `R`: 实验记录器
- `SignalRecord`: 信号记录
- `PortAnaRecord`: 投资组合分析记录
- `flatten_dict`: 平展字典

## 3. 数据初始化

```python
# 使用默认数据
# 注意：需要从远程下载数据: python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data
provider_uri = "~/.qlib/qlib_data/cn_data"  # 目标目录
if not exists_qlib_data(provider_uri):
    print(f"在 {provider_uri} 找不到 Qlib 数据")
    sys.path.append(str(scripts_dir))
    from get_data import GetData

    GetData().qlib_data(target_dir=provider_uri, region=REG_CN)
qlib.init(provider_uri=provider_uri, region=REG_CN)
```

**功能说明**：
- 检查本地是否存在 Qlib 数据
- 如果不存在，自动从远程下载
- 初始化 Qlib，指定中国数据区域

## 4. 设置市场和基准

```python
market = "csi300"          # 沪深 300 指数
benchmark = "SH000300"     # 基准指数代码
```

## 5. 模型训练

本部分演示如何在历史数据上训练 LightGBM 模型。

```python
###################################
# 训练模型
###################################
data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
}

task = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",                    # 均方误差损失函数
            "colsample_bytree": 0.8879,       # 每棵树的特征采样比例
            "learning_rate": 0.0421,          # 学习率
            "subsample": 0.8789,              # 样本采样比例
            "lambda_l1": 205.6999,            # L1 正则化系数
            "lambda_l2": 580.9768,            # L2 正则化系数
            "max_depth": 8,                   # 最大树深度
            "num_leaves": 210,                # 最大叶子节点数
            "num_threads": 20,                # 线程数
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),    # 训练集
                "valid": ("2015-01-01", "2016-12-31"),    # 验证集
                "test": ("2017-01-01", "2020-08-01"),     # 测试集
            },
        },
    },
}

# 模型初始化
model = init_instance_by_config(task["model"])
dataset = init_instance_by_config(task["dataset"])

# 启动实验来训练模型
with R.start(experiment_name="train_model"):
    R.log_params(**flatten_dict(task))      # 记录参数
    model.fit(dataset)                      # 拟合模型
    R.save_objects(trained_model=model)     # 保存模型
    rid = R.get_recorder().id               # 获取实验 ID
```

### 训练配置说明：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 时间周期 | 2008-01-01 至 2020-08-01 | 完整数据时间范围 |
| 训练数据 | 2008-01-01 至 2014-12-31 | 7年训练数据 |
| 验证数据 | 2015-01-01 至 2016-12-31 | 2年验证数据 |
| 测试数据 | 2017-01-01 至 2020-08-01 | 4年测试数据 |
| 模型类型 | LightGBM | 梯度提升树模型 |
| 特征集 | Alpha158 | 158个机器学习因子 |
| 交易市场 | CSI300 | 沪深 300 指数成分股 |

## 6. 预测、回测与分析

本部分使用训练好的模型进行回测，采用投资组合策略。

```python
###################################
# 预测、回测与分析
###################################
port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "model": model,
            "dataset": dataset,
            "topk": 50,                       # 选择排名前50的股票
            "n_drop": 5,                      # 随机剔除5只股票
        },
    },
    "backtest": {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,                 # 初始账户: 1亿
        "benchmark": benchmark,
        "exchange_kwargs": {
            "freq": "day",                    # 交易频率: 日频
            "limit_threshold": 0.095,         # 涨跌停限制: 9.5%
            "deal_price": "close",            # 成交价格: 收盘价
            "open_cost": 0.0005,              # 开仓成本: 0.05%
            "close_cost": 0.0015,             # 平仓成本: 0.15%
            "min_cost": 5,                    # 最小成本: 5元
        },
    },
}

# 执行回测与分析
with R.start(experiment_name="backtest_analysis"):
    # 加载之前训练好的模型
    recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
    model = recorder.load_object("trained_model")

    # 生成预测信号
    recorder = R.get_recorder()
    ba_rid = recorder.id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # 执行回测与投资组合分析
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()
```

### 回测配置详解：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 策略类型 | TopK-Dropout | 选择top-k股票，随机剔除n只 |
| 初始资金 | 1亿 | 初始账户金额 |
| 回测期间 | 2017-01-01 至 2020-08-01 | 4年回测期间 |
| 交易频率 | 日频 | 每天调整一次仓位 |
| 开仓成本 | 0.05% | 买入手续费 |
| 平仓成本 | 0.15% | 卖出手续费 |
| 最小成本 | 5元 | 单笔最小手续费 |
| 涨跌停限制 | 9.5% | 最大日涨跌限制 |

## 7. 加载并分析结果

```python
from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D

# 获取回测结果记录器
recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_analysis")
print(recorder)

# 加载各项分析对象
pred_df = recorder.load_object("pred.pkl")                                    # 预测结果
report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")  # 投资组合报告
positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")      # 持仓信息
analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")      # 投资组合分析
```

### 加载对象说明：

| 对象名称 | 说明 |
|---------|------|
| `pred_df` | 模型在测试期间的所有预测结果 |
| `report_normal_df` | 投资组合性能报告（收益、波动率等） |
| `positions` | 整个回测期间的股票持仓详情 |
| `analysis_df` | 详细的投资组合分析指标 |

## 8. 投资组合分析

### 8.1 投资组合报告

```python
analysis_position.report_graph(report_normal_df)
```

**功能**：生成投资组合性能报告的可视化图表，包括：
- 净值曲线
- 累计收益率
- 年化收益率
- 夏普比率
- 最大回撤

### 8.2 风险分析

```python
analysis_position.risk_analysis_graph(analysis_df, report_normal_df)
```

**功能**：生成风险分析图表，包括：
- 波动率分析
- 回撤分析
- 风险调整后收益指标
- VaR（风险价值）分析
- 下行风险指标

## 9. 模型分析

### 9.1 准备标签数据

```python
label_df = dataset.prepare("test", col_set="label")
label_df.columns = ["label"]
```

**说明**：从测试集中提取真实标签数据，用于与预测值比较。

### 9.2 信息系数（IC）分析

```python
pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
analysis_position.score_ic_graph(pred_label)
```

**功能**：
- 合并预测值和真实标签
- 计算信息系数（Information Coefficient）
- 分析预测与实际的相关性
- 生成 IC 曲线和统计图表

### 9.3 模型性能评估

```python
analysis_model.model_performance_graph(pred_label)
```

**功能**：
- 生成模型性能综合评估图表
- 显示预测准确度
- 分析模型的预测能力
- 性能排序和对比分析

---

## 工作流完整流程总结

### 阶段一：环境准备
1. 安装必要的依赖包（numpy、pyqlib）
2. 初始化 Qlib 环境
3. 下载必要的数据脚本

### 阶段二：数据处理
1. 检查本地数据，自动下载缺失数据
2. 初始化 Qlib 数据提供商
3. 指定中国市场数据区域

### 阶段三：特征工程与模型训练
1. 加载 Alpha158 特征集（158个机器学习因子）
2. 划分训练、验证、测试数据集
3. 使用 LightGBM 模型进行训练
4. 记录模型参数和结果

### 阶段四：策略回测
1. 使用 TopK-Dropout 策略进行选股
2. 模拟实际交易过程（包含手续费、涨跌停等）
3. 计算投资组合指标

### 阶段五：结果分析
1. **投资组合分析**：评估投资组合的收益和风险
2. **风险分析**：深入分析波动率、回撤等风险指标
3. **模型分析**：评估模型的预测能力（IC 分析）
4. **性能评估**：综合评估模型的整体表现

## 核心概念说明

### Alpha158 因子
158 个机器学习因子，基于常见的金融特征（价格、成交量、基本面等）计算而得，用于捕捉股票的异常超额收益。

### TopK-Dropout 策略
- **TopK**：选择预测收益排名前 K（50）的股票进行投资
- **Dropout**：在每个时期随机剔除 N（5）只持仓股票，增加策略的多样性

### 信息系数（IC）
衡量预测值与实际值之间相关性的指标，取值范围 [-1, 1]，绝对值越大说明预测能力越强。

### 夏普比率（Sharpe Ratio）
衡量投资收益与风险的比率，值越高说明单位风险获得的超额收益越高。

### 最大回撤（Max Drawdown）
从历史高点到后续最低点的最大跌幅，反映投资的最坏情况下的损失。

---

## 快速参考

| 步骤 | 主要函数/类 | 输入 | 输出 |
|------|-----------|------|------|
| 模型初始化 | `init_instance_by_config()` | 配置字典 | 模型实例 |
| 数据集初始化 | `init_instance_by_config()` | 配置字典 | 数据集实例 |
| 模型训练 | `model.fit()` | 数据集 | 训练好的模型 |
| 信号生成 | `SignalRecord.generate()` | 模型+数据集 | 预测信号 |
| 投资组合分析 | `PortAnaRecord.generate()` | 配置+数据集 | 回测报告 |

---

这个完整的工作流展示了如何使用 Qlib 框架构建、训练、回测和评估一个实际的量化交易策略。
