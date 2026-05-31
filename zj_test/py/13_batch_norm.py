"""
Batch Normalization (批量归一化) - 在 MNIST 上训练和测试
基于 d2l 教材实现,参照原始论文 (Ioffe & Szegedy, 2015)

批量归一化创新点:
1. 在每次训练迭代中对输入进行标准化
2. 应用可学习的缩放(γ)和偏移(β)参数
3. 加速收敛，允许使用更大的学习率
4. 减少对参数初始化的敏感性

实现两种版本:
1. 从零实现 BatchNorm
2. 使用 PyTorch 内置 nn.BatchNorm1d/nn.BatchNorm2d
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# 设备检测
device = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ──────────────────────────────────────────────────────
# 从零实现批量归一化
# ──────────────────────────────────────────────────────
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    批量归一化核心函数

    公式: BN(x) = γ * (x - μ_B) / sqrt(σ_B² + ε) + β

    Args:
        X: 输入张量
        gamma: 缩放参数 (可学习)
        beta: 偏移参数 (可学习)
        moving_mean: 移动平均均值 (用于预测)
        moving_var: 移动平均方差 (用于预测)
        eps: 防止除零的小常数
        momentum: 移动平均动量
    Returns:
        Y: 归一化后的输出
        moving_mean: 更新后的移动平均均值
        moving_var: 更新后的移动平均方差
    """
    if not torch.is_grad_enabled():
        # 预测模式：使用移动平均的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 训练模式
        assert len(X.shape) in (2, 4)

        if len(X.shape) == 2:
            # 全连接层：特征维上计算 (dim=0)
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 卷积层：通道维上计算，保持形状用于广播
            # 在每个输出通道的 m*p*q 个元素上计算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

        X_hat = (X - mean) / torch.sqrt(var + eps)

        # 更新移动平均 (训练时累积统计量供预测使用)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var

    # 应用可学习的缩放和偏移
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    """
    从零实现的批量归一化层

    特点:
    - 支持全连接层(2D)和卷积层(4D)
    - 训练时使用小批量统计量
    - 预测时使用移动平均统计量
    - γ和β是可学习参数
    """
    def __init__(self, num_features, num_dims):
        """
        Args:
            num_features: 特征数量(通道数)
            num_dims: 输入维度数 (2=全连接, 4=卷积)
        """
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        # 非模型参数(移动平均)
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 确保移动平均在正确的设备上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# ──────────────────────────────────────────────────────
# 使用自定义 BatchNorm 的 LeNet
# ──────────────────────────────────────────────────────
class LeNet_CustomBN(nn.Module):
    """
    LeNet + 自定义批量归一化

    结构:
    - Conv1 → BN → Sigmoid → AvgPool
    - Conv2 → BN → Sigmoid → AvgPool
    - FC1 → BN → Sigmoid
    - FC2 → BN → Sigmoid
    - FC3 (输出)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 卷积层1
            nn.Conv2d(1, 6, kernel_size=5),
            BatchNorm(6, num_dims=4),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 卷积层2
            nn.Conv2d(6, 16, kernel_size=5),
            BatchNorm(16, num_dims=4),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            # 全连接层1
            nn.Linear(16 * 4 * 4, 120),
            BatchNorm(120, num_dims=2),
            nn.Sigmoid(),

            # 全连接层2
            nn.Linear(120, 84),
            BatchNorm(84, num_dims=2),
            nn.Sigmoid(),

            # 输出层
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────
# 使用 PyTorch 内置 BatchNorm 的 LeNet
# ──────────────────────────────────────────────────────
class LeNet_PyTorchBN(nn.Module):
    """
    LeNet + PyTorch内置批量归一化

    使用 nn.BatchNorm2d 和 nn.BatchNorm1d
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 卷积层1
            nn.Conv2d(1, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 卷积层2
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            # 全连接层1
            nn.Linear(256, 120),
            nn.BatchNorm1d(120),
            nn.Sigmoid(),

            # 全连接层2
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),

            # 输出层
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────
# 不使用 BatchNorm 的原始 LeNet (用于对比)
# ──────────────────────────────────────────────────────
class LeNet_Original(nn.Module):
    """
    原始 LeNet (无批量归一化)
    用于对比 BatchNorm 的效果
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────
# 数据加载: MNIST
# ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# MNIST 类别名
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# ──────────────────────────────────────────────────────
# 训练和测试函数
# ──────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 每 50 个 batch 打印一次
        if (batch_idx + 1) % 50 == 0:
            print(f"  batch {batch_idx+1}/{len(loader)}: "
                  f"loss={loss.item():.4f}, "
                  f"acc={100.*correct/total:.2f}%")

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return running_loss / total, correct / total


# ──────────────────────────────────────────────────────
# 模型选择
# ──────────────────────────────────────────────────────
# 选择模型: 'custom_bn', 'pytorch_bn', 'original'
MODEL_TYPE = 'custom_bn'

if MODEL_TYPE == 'custom_bn':
    model = LeNet_CustomBN().to(device)
    model_name = "LeNet + 自定义BatchNorm"
elif MODEL_TYPE == 'pytorch_bn':
    model = LeNet_PyTorchBN().to(device)
    model_name = "LeNet + PyTorch BatchNorm"
else:
    model = LeNet_Original().to(device)
    model_name = "LeNet 原始(无BN)"

# 训练设置 - BN允许使用更大的学习率
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1.0 if MODEL_TYPE != 'original' else 0.1)
num_epochs = 10

# 打印模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型架构: {model_name}")
print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
print("=" * 60)


# ──────────────────────────────────────────────────────
# 训练循环
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"开始训练 {model_name}, 共 {num_epochs} 个 epoch")
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    print(f"Batch size: 256, 学习率: {optimizer.param_groups[0]['lr']}")
    print("=" * 60)

    best_test_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device)
        elapsed = time.time() - start_time

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f"13_{MODEL_TYPE}_best.pth")

        print(f"Epoch {epoch}/{num_epochs} [{elapsed:.1f}s] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    print("=" * 60)
    print(f"最终测试准确率: {test_acc * 100:.2f}%")
    print(f"最佳测试准确率: {best_test_acc * 100:.2f}%")
    torch.save(model.state_dict(), f"13_{MODEL_TYPE}_final.pth")
    print(f"模型已保存: 13_{MODEL_TYPE}_final.pth")


# ──────────────────────────────────────────────────────
# 模型架构说明
# ──────────────────────────────────────────────────────
"""
批量归一化 (Batch Normalization) 原理:

┌─────────────────────────────────────────────────────────────────┐
│ 公式: BN(x) = γ * (x - μ_B) / sqrt(σ_B² + ε) + β             │
├─────────────────────────────────────────────────────────────────┤
│ μ_B: 小批量均值                                                │
│ σ_B²: 小批量方差                                               │
│ ε: 防止除零的小常数 (1e-5)                                     │
│ γ: 缩放参数 (可学习)                                           │
│ β: 偏移参数 (可学习)                                           │
└─────────────────────────────────────────────────────────────────┘

全连接层 vs 卷积层的批量归一化:

┌─────────────────────────────────────────────────────────────────┐
│ 层类型      │ 计算维度          │ γ/β形状                      │
├─────────────────────────────────────────────────────────────────┤
│ 全连接层    │ 特征维 (dim=0)    │ (1, num_features)            │
│ 卷积层      │ 通道维+空间位置   │ (1, num_features, 1, 1)      │
└─────────────────────────────────────────────────────────────────┘

训练模式 vs 预测模式:

┌─────────────────────────────────────────────────────────────────┐
│ 模式      │ 均值/方差来源        │ 特点                        │
├─────────────────────────────────────────────────────────────────┤
│ 训练模式  │ 当前小批量统计量     │ 有噪声，正则化效果          │
│ 预测模式  │ 移动平均统计量       │ 确定性输出                  │
└─────────────────────────────────────────────────────────────────┘

移动平均更新公式:
μ ← momentum * μ + (1 - momentum) * μ_batch
σ² ← momentum * σ² + (1 - momentum) * σ²_batch

BatchNorm 的优势:
1. 加速收敛: 允许使用更大的学习率
2. 减少对初始化的敏感性
3. 提供正则化效果(训练时的噪声)
4. 缓解梯度消失/爆炸问题

本实现说明:
- 提供三种模型: 自定义BN、PyTorch内置BN、原始LeNet(无BN)
- 可通过 MODEL_TYPE 变量切换
- 使用学习率1.0(BN允许更大lr) vs 0.1(无BN)
- Batch size=256 (BN对batch大小敏感)
"""
