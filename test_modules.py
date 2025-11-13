

import torch
from MFT_Project.data.dataset import MultiModalDataset
from MFT_Project.model.fusion_model import FusionTransformer

# mock handler，返回随机数据
class MockHandler:
    def __init__(self):
        pass
    def get_data(self):
        return {
            'feature1': torch.randn(4, 1),
            'feature2': torch.randn(4, 1),
        }



# 创建 handler 配置字典
handler_config = {
    'class': 'MockHandler',
    'module_path': '__main__',  # MockHandler 定义在当前脚本中
    'kwargs': {}
}
text_feature_path = "MFT_Project/data/dummy_text_features.pkl"
segments = {
    'train': ('2021-01-01', '2021-01-20'),
    'valid': ('2021-01-21', '2021-01-25'),
    'test': ('2021-01-26', '2021-01-31')
}
dataset = MultiModalDataset(handler=handler_config, text_feature_path=text_feature_path, segments=segments)

# 获取第一个批次数据
batch = dataset[0]

print("数据批次类型:", type(batch))
if isinstance(batch, dict):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"特征: {k}, 类型: {type(v)}, 形状: {v.shape}")
        else:
            print(f"特征: {k}, 类型: {type(v)}")
else:
    print("批次不是字典类型，请检查数据集实现。")

# 创建 FusionTransformer 实例（参数请根据实际模型定义调整）
model = FusionTransformer(market_dim=2, text_dim=770, num_layers=2, num_heads=2, hidden_dim=8, dropout=0.1)

# 喂数据到模型
try:
    output = model(batch)
    print("模型 forward 成功，输出形状:", output.shape)
except Exception as e:
    print("模型 forward 失败，错误信息:", str(e))
