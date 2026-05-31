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

DATA_DIR = "data/house-prices"

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


def get_net():
    return nn.Sequential(nn.Linear(in_features, 1))


# =========================================
# 4. 损失函数 & 优化器
# =========================================

loss_fn = nn.MSELoss()


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
# 7. 训练函数
# =========================================

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, batch_size):
    trainer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X_batch, y_batch in batch_iterator(train_features, train_labels, batch_size):
            trainer.zero_grad()
            l = loss_fn(net(X_batch), y_batch)
            l.backward()
            trainer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# =========================================
# 8. K 折交叉验证
# =========================================

def get_k_fold_data(k, i, X, y):
    fold_size = X.shape[0] // k
    val_start, val_end = i * fold_size, (i + 1) * fold_size
    idx = list(range(X.shape[0]))
    val_idx = idx[val_start:val_end]
    train_idx = idx[:val_start] + idx[val_end:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def k_fold(k, X_train, y_train, num_epochs, learning_rate, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f"折 {i+1}: 训练 log_rmse={train_ls[-1]:.5f}, "
              f"验证 log_rmse={valid_ls[-1]:.5f}")
    return train_l_sum / k, valid_l_sum / k


# =========================================
# 9. 超参数
# =========================================

k = 5
batch_size = 64
epochs = 100
lr = 0.01


# =========================================
# 10. K 折验证
# =========================================

print(f"\n开始 {k} 折交叉验证...")
avg_train_l, avg_valid_l = k_fold(
    k, train_features, train_labels, epochs, lr, batch_size
)
print(f"\n{k} 折结果: 平均训练 log_rmse={avg_train_l:.5f}, "
      f"平均验证 log_rmse={avg_valid_l:.5f}")


# =========================================
# 11. 全量训练 & 预测
# =========================================

print("\n在全部训练集上训练并预测...")
net = get_net()
train_ls, _ = train(net, train_features, train_labels, None, None,
                    epochs, lr, batch_size)
print(f"最终训练 log_rmse={train_ls[-1]:.5f}")

with torch.no_grad():
    preds = torch.exp(net(test_features)).numpy()  # log 空间 -> 原始价格

submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": preds.reshape(-1),
})
submission.to_csv("data/house-prices/submission.csv", index=False)

print("\n提交文件已保存到 data/house-prices/submission.csv")
