import yaml
import shap
import matplotlib.pyplot as plt
import qlib
from pathlib import Path

from qlib.utils import init_instance_by_config
from MFT_Project.explain import FusionShapExplainer

# ---------- 读取配置并初始化 ----------
with open("MFT_Project/configs/workflow_config_multi_modal.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

qlib.init(**cfg["qlib_init"])
dataset = init_instance_by_config(cfg["task"]["dataset"])
model = init_instance_by_config(cfg["task"]["model"])
model.fit(dataset)

# ---------- 计算 SHAP ----------
explainer = FusionShapExplainer(model, dataset, background_segment="train", background_size=100)
result = explainer.explain(segment="test", sample_size=50)

print("SHAP base value:", result.base_value)
print(result.values.head())

# ---------- 导出 CSV ----------
output_dir = Path("MFT_Project/explain")
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / "shap_values_test.csv"
result.values.to_csv(csv_path, encoding="utf-8")
print(f"SHAP 值已导出至 {csv_path}")

# ---------- 绘制 summary_plot ----------
plt.figure(figsize=(12, 6))
shap.summary_plot(
    result.values.to_numpy(),
    feature_names=result.values.columns.tolist(),
    show=False  # 防止自动弹窗，便于保存
)
plt.tight_layout()
summary_path = output_dir / "shap_summary.png"
plt.savefig(summary_path, dpi=200)
plt.close()
print(f"SHAP summary 图已保存到 {summary_path}")