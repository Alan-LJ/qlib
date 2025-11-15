import yaml
import shap
import qlib
from MFT_Project.dataset import MultiModalDataset
from MFT_Project.model import FusionModel
from MFT_Project.explain import FusionShapExplainer
from qlib.utils import init_instance_by_config

# 读取 workflow 配置
with open("MFT_Project/configs/workflow_config_multi_modal.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 初始化 Qlib，构建数据集和模型
qlib.init(**cfg["qlib_init"])
dataset = init_instance_by_config(cfg["task"]["dataset"])
model = init_instance_by_config(cfg["task"]["model"])

# 训练模型（如已训练可替换为 load 模型逻辑）
model.fit(dataset)

# 创建 SHAP 解释器，使用训练集前 50 个样本作为背景
explainer = FusionShapExplainer(model, dataset, background_segment="train", background_size=50)

# 针对 test 分段抽取 20 个样本做解释
result = explainer.explain(segment="test", sample_size=20)

print("SHAP base value:", result.base_value)
print("SHAP values preview:")
print(result.values.head())
