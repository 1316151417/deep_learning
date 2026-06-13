"""
序列模型 (Sequence Models) — 参考 d2l §8.1
https://zh-v2.d2l.ai/chapter_recurrent-neural-networks/sequence.html

演示内容:
1. 生成正弦序列数据 x_t = sin(0.01·t) + ε，观察"动力学不变、数值随时间变化"
2. 用嵌入维度 τ 构造特征-标签对: X_t = [x_{t-τ},...,x_{t-1}], y_t = x_t (自回归)
3. 训练极简 MLP (两层全连接 + ReLU) 做自回归预测
4. 单步预测 (onestep) vs 多步预测 (multistep): 后者用自己的预测当输入
5. k 步预测 (1/4/16/64): 直观展示预测越远, 误差累积越快

全部使用 PyTorch 官方 API (TensorDataset/DataLoader/nn.Sequential/Adam/MSELoss),
不自己实现训练循环之外的任何东西。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────
# 配置 (与 d2l 教程一致, 便于复现)
# ──────────────────────────────────────────────────────
torch.manual_seed(42)

T = 1000          # 序列总长度
TAU = 4           # 嵌入维度: 用过去 τ 个观测预测当前值
BATCH_SIZE = 16
N_TRAIN = 600     # 前 N_TRAIN 个样本用于训练
NUM_EPOCHS = 5
LR = 0.01

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"使用设备: {device}")


# ──────────────────────────────────────────────────────
# 1. 生成数据: x_t = sin(0.01·t) + 高斯噪声
# ──────────────────────────────────────────────────────
def make_data():
    """正弦序列 + 可加性噪声 (d2l 的标准玩具数据)"""
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    return time, x


# ──────────────────────────────────────────────────────
# 2. 构造特征-标签对 (滑窗, 自回归)
# ──────────────────────────────────────────────────────
def make_features(x):
    """
    把一维序列滑窗成 (T-τ, τ) 的特征矩阵:
        features[t] = [x_t, x_{t+1}, ..., x_{t+τ-1}]   预测 x_{t+τ}
    因此 labels = x[τ:].reshape(-1, 1)
    """
    features = torch.zeros((T - TAU, TAU))
    for i in range(TAU):
        features[:, i] = x[i: T - TAU + i]
    labels = x[TAU:].reshape((-1, 1))
    return features, labels


# ──────────────────────────────────────────────────────
# 3. 极简 MLP: 两层全连接 + ReLU (复用 nn.Sequential)
# ──────────────────────────────────────────────────────
def get_net():
    net = nn.Sequential(
        nn.Linear(TAU, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
    )
    return net


# ──────────────────────────────────────────────────────
# 4. 训练 (复用 Adam + MSELoss, 标准训练循环)
# ──────────────────────────────────────────────────────
def train(net, train_iter, features, labels):
    """reduction='none' + .sum().backward() 与 d2l 一致"""
    loss_fn = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    history = []

    feat_tr = features[:N_TRAIN].to(device)
    lab_tr = labels[:N_TRAIN].to(device)

    for epoch in range(NUM_EPOCHS):
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            l = loss_fn(net(X), y)
            l.sum().backward()
            optimizer.step()

        with torch.no_grad():
            train_loss = loss_fn(net(feat_tr), lab_tr).mean().item()
        history.append(train_loss)
        print(f"epoch {epoch + 1}, loss {train_loss:.6f}")
    return history


# ──────────────────────────────────────────────────────
# 5. 预测
# ──────────────────────────────────────────────────────
@torch.no_grad()
def one_step_pred(net, features):
    """单步预测: 每一步都用真实历史作输入 → 效果好"""
    return net(features.to(device)).reshape(-1).cpu()


@torch.no_grad()
def multi_step_pred(net, x):
    """
    多步预测: 超过 n_train+τ 后没有真实数据, 只能把自己的预测当输入。
    预测点会像滚雪球一样累积误差, 很快塌缩成常数。
    """
    preds = torch.zeros(T)
    preds[: N_TRAIN + TAU] = x[: N_TRAIN + TAU]
    for i in range(N_TRAIN + TAU, T):
        inp = preds[i - TAU: i].reshape((1, -1)).to(device)
        preds[i] = net(inp).item()
    return preds


@torch.no_grad()
def k_step_pred(net, x, max_steps=64):
    """
    k 步预测: 把整条序列铺成 (N, τ+max_steps) 的特征矩阵,
    第 i 列 (i>=τ) 用前 τ 列的预测填充, 从而对每个起点同时得到 1..max_steps 步预测。
    """
    n = T - TAU - max_steps + 1
    feats = torch.zeros((n, TAU + max_steps))
    # 前 τ 列是真实观测
    for i in range(TAU):
        feats[:, i] = x[i: i + n]
    # 之后的列逐个用模型预测填充
    for i in range(TAU, TAU + max_steps):
        inp = feats[:, i - TAU: i].to(device)
        feats[:, i] = net(inp).reshape(-1).cpu()
    return feats


# ──────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────
def plot_results(time, x, onestep, multistep, feats, history):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (a) 原始序列 + 单步预测
    axes[0, 0].plot(time, x, label="data", alpha=0.7)
    axes[0, 0].plot(time[TAU:], onestep, label="1-step preds", lw=1.5)
    axes[0, 0].set_title("(a) One-step prediction (real history)")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # (b) 单步 vs 多步 (重点看 604 之后)
    axes[0, 1].plot(time, x, label="data", alpha=0.6)
    axes[0, 1].plot(time[TAU:], onestep, label="1-step preds", lw=1.5)
    axes[0, 1].plot(time[N_TRAIN + TAU:], multistep[N_TRAIN + TAU:],
                    label="multistep preds", lw=1.5)
    axes[0, 1].axvline(N_TRAIN + TAU, color="red", ls="--", alpha=0.5,
                       label="train cutoff")
    axes[0, 1].set_title("(b) One-step vs Multi-step")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # (c) k 步预测 (1, 4, 16, 64)
    steps = (1, 4, 16, 64)
    n = feats.shape[0]
    for k in steps:
        t_axis = time[TAU + k - 1: T - 64 + k]  # 与 feats 行对齐
        axes[1, 0].plot(t_axis[:n], feats[:, TAU + k - 1], label=f"{k}-step preds")
    axes[1, 0].set_title("(c) k-step prediction (error accumulation)")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # (d) 训练损失
    axes[1, 1].plot(range(1, len(history) + 1), history, marker="o", color="tab:red")
    axes[1, 1].set_title("(d) Training loss"); axes[1, 1].set_xlabel("epoch")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Part 1: 生成数据 + 构造特征-标签对")
    print("=" * 60)
    time, x = make_data()
    features, labels = make_features(x)
    print(f"序列长度 T={T}, 嵌入维度 τ={TAU}")
    print(f"特征矩阵 shape={tuple(features.shape)}, 标签 shape={tuple(labels.shape)}")
    print(f"前 {N_TRAIN} 个样本用于训练, 后 {T - TAU - N_TRAIN} 个用于测试/预测")

    print("\n" + "=" * 60)
    print("Part 2: 训练极简 MLP")
    print("=" * 60)
    # 复用框架的 TensorDataset + DataLoader, 自动分批 + 打乱
    train_iter = DataLoader(
        TensorDataset(features[:N_TRAIN], labels[:N_TRAIN]),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    net = get_net().to(device)
    history = train(net, train_iter, features, labels)

    print("\n" + "=" * 60)
    print("Part 3: 预测对比")
    print("=" * 60)
    onestep = one_step_pred(net, features)
    multistep = multi_step_pred(net, x)
    feats = k_step_pred(net, x)

    # 量化测试段 (604~1000) 的误差, 展示多步预测的误差累积
    # onestep 长度为 T-τ, onestep[j] 对应 x[j+τ], 故测试段下标左移 τ
    onestep_mse = ((onestep[N_TRAIN:] - x[N_TRAIN + TAU:]) ** 2).mean().item()
    multistep_mse = ((multistep[N_TRAIN + TAU:] - x[N_TRAIN + TAU:]) ** 2).mean().item()
    print(f"测试段 ({N_TRAIN + TAU}~{T}):")
    print(f"  单步预测 MSE = {onestep_mse:.4f}")
    print(f"  多步预测 MSE = {multistep_mse:.4f}   ← 远大于单步, 误差累积")

    plot_results(time, x, onestep, multistep, feats, history)


if __name__ == "__main__":
    main()
