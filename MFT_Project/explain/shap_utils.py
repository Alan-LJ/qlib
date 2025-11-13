import shap
import torch
import numpy as np

def generate_shap_explanation(model, data_samples, feature_names):
    """
    使用SHAP DeepExplainer为FusionTransformer模型生成特征重要性解释。
    参数：
        model: 训练好的FusionTransformer模型实例。
        data_samples: 需要解释的数据样本（自定义字典格式，每个key对应一个特征，value为numpy数组）。
        feature_names: 所有特征名称列表。
    返回：
        shap_values: SHAP值对象
        summary_plot: SHAP summary plot图像对象
    """
    # 包装函数，将numpy数组转换为模型forward所需的字典格式（PyTorch张量）
    def model_wrapper(np_array):
        # np_array: shape [batch_size, num_features]
        # 将numpy数组按特征拆分，转换为字典，每个key对应一个特征张量
        batch_size = np_array.shape[0]
        input_dict = {}
        for i, feat in enumerate(feature_names):
            # 取每个特征的所有样本，转为torch张量
            input_dict[feat] = torch.from_numpy(np_array[:, i]).float().reshape(batch_size, -1)
        # 放到模型设备
        for k in input_dict:
            input_dict[k] = input_dict[k].to(next(model.parameters()).device)
        # 返回模型输出
        with torch.no_grad():
            output = model(input_dict)
        # 如果输出是张量，转为numpy
        if isinstance(output, torch.Tensor):
            return output.cpu().numpy()
        return output

    # 构造输入数据矩阵（shape: [batch_size, num_features]）
    # 假设data_samples是字典，每个key是特征名，value是numpy数组（shape: [batch_size, 1] 或 [batch_size]）
    X = []
    for feat in feature_names:
        arr = data_samples[feat]
        arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr
        X.append(arr)
    X = np.concatenate(X, axis=1)  # shape: [batch_size, num_features]

    # 初始化DeepExplainer
    explainer = shap.DeepExplainer(model_wrapper, X)
    shap_values = explainer.shap_values(X)

    # 绘制summary plot
    summary_plot = shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    return shap_values, summary_plot
