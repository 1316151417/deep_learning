"""
Kaggle House Prices - PyTorch 简版实现
参考: https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

torch.manual_seed(42)


# =========================================
# 1. 加载数据
# =========================================

DATA_DIR = "zj_test/kaggle/input/house-prices-advanced-regression-techniques"

train_data = pd.read_csv(f"{DATA_DIR}/train.csv")
test_data = pd.read_csv(f"{DATA_DIR}/test.csv")

print(f"训练集: {train_data.shape}, 测试集: {test_data.shape}")


# =========================================
# 2. 数据预处理
# =========================================

# 去掉 Id 列，拼接 train/test 统一做特征工程
all_features = pd.concat([train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]], axis=0)

# 把 "NA" 字符串替换为真正的 NaN，再尝试转数值
all_features = all_features.replace("NA", np.nan)
for col in all_features.columns:
    try:
        all_features[col] = pd.to_numeric(all_features[col])
    except (ValueError, TypeError):
        pass

# 数值特征: 标准化 (x - mean) / std，缺失值填 0
numeric_features = all_features.select_dtypes(include=["number"]).columns
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 类别特征: one-hot 编码
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.astype(float)  # bool+float 混合时 numpy 会生成 object，需要统一类型

# 转为 tensor
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    np.log(train_data.SalePrice.values.reshape(-1, 1)), dtype=torch.float32
)

print(f"特征维度: {train_features.shape[1]}")


# =========================================
# 3. 定义模型
# =========================================

in_features = train_features.shape[1]

net = nn.Sequential(
    nn.Linear(in_features, 1)
)


# =========================================
# 4. 损失函数 & 优化器
# =========================================

loss_fn = nn.MSELoss()
trainer = torch.optim.Adam(net.parameters(), lr=0.01)


# =========================================
# 5. 评估指标: log RMSE (标签已在 log 空间，直接算 MSE 的 sqrt 即 RMSLE)
# =========================================

def log_rmse(net, features, labels):
    with torch.no_grad():
        return torch.sqrt(loss_fn(net(features), labels)).item()


# =========================================
# 6. mini-batch
# =========================================

def batch_iterator(X, y, batch_size):
    indices = torch.randperm(len(X))
    for start in range(0, len(X), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


# =========================================
# 7. 超参数
# =========================================

batch_size = 64
epochs = 100


# =========================================
# 8. 训练
# =========================================

for epoch in range(epochs):

    total_loss = 0
    num_batches = 0

    for X_batch, y_batch in batch_iterator(train_features, train_labels, batch_size):

        y_hat = net(X_batch)
        l = loss_fn(y_hat, y_batch)

        total_loss += l.item()
        num_batches += 1

        trainer.zero_grad()
        l.backward()
        trainer.step()

    train_rmse = log_rmse(net, train_features, train_labels)

    print(
        f"epoch={epoch+1:03d} "
        f"loss={total_loss / num_batches:.4f} "
        f"train_log_rmse={train_rmse:.5f}"
    )


# =========================================
# 9. 预测 & 生成提交文件
# =========================================

with torch.no_grad():
    preds = torch.exp(net(test_features)).numpy()  # log 空间 -> 原始价格

submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": preds.reshape(-1),
})
submission.to_csv("zj_test/kaggle/submission.csv", index=False)

print("\n提交文件已保存到 zj_test/kaggle/submission.csv")
