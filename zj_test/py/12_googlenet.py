"""
GoogLeNet (Inception v1) - 在 MNIST 上训练和测试
基于 d2l 教材实现,参照原始 GoogLeNet (Szegedy et al., 2015) 设计

GoogLeNet 创新点:
1. Inception块: 4条并行路径(1×1, 3×3, 5×5卷积 + 池化)
2. 1×1卷积降维，减少计算量
3. 全局平均汇聚层替代全连接层
4. 网络深度达22层(含池化层)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# 设备检测
device = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ──────────────────────────────────────────────────────
# Inception 块定义
# 包含4条并行路径，各自提取不同空间尺度的特征
# ──────────────────────────────────────────────────────
class Inception(nn.Module):
    """
    Inception块实现
    4条并行路径:
    - 路径1: 单个1×1卷积层
    - 路径2: 1×1卷积(降维) → 3×3卷积
    - 路径3: 1×1卷积(降维) → 5×5卷积
    - 路径4: 3×3最大汇聚层 → 1×1卷积

    所有路径使用合适填充保持空间尺寸一致，最终在通道维度上拼接输出

    Args:
        in_channels: 输入通道数
        c1: 路径1输出通道数
        c2: 路径2输出通道数元组 (c2a, c2b)
        c3: 路径3输出通道数元组 (c3a, c3b)
        c4: 路径4输出通道数
    """
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super().__init__(**kwargs)

        # 路径1：单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        # 路径2：1x1卷积后接3x3卷积
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        # 路径3：1x1卷积后接5x5卷积
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # 路径4：3x3最大汇聚后接1x1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上拼接
        return torch.cat((p1, p2, p3, p4), dim=1)


# ──────────────────────────────────────────────────────
# GoogLeNet 网络定义
# 由9个Inception块堆叠而成，分为5个模块(b1-b5)
# ──────────────────────────────────────────────────────
class GoogLeNet(nn.Module):
    """
    GoogLeNet网络实现
    原始输入: 1×224×224, 这里适配 MNIST (28×28 resize 到 96×96)

    网络结构:
    - 模块1(b1): 7×7卷积 + 最大汇聚
    - 模块2(b2): 1×1卷积 + 3×3卷积 + 最大汇聚
    - 模块3(b3): 2个Inception块 + 最大汇聚
    - 模块4(b4): 5个Inception块 + 最大汇聚
    - 模块5(b5): 2个Inception块 + 全局平均汇聚
    """
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()

        # 模块1：初始卷积+池化
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 模块2：双卷积+池化
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 模块3：2个Inception块 + 池化
        # Inception 1: 输出 64+128+32+32=256 通道
        # Inception 2: 输出 128+192+96+64=480 通道
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 模块4：5个Inception块 + 池化
        # 输出通道依次为 512, 512, 512, 528, 832
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 模块5：2个Inception块 + 全局平均汇聚
        # 输出 832 和 1024 通道
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 全连接输出层
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.fc(x)
        return x


# ──────────────────────────────────────────────────────
# 数据加载: MNIST, resize 到 96×96 以适配 GoogLeNet
# ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# MNIST 类别名
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# ──────────────────────────────────────────────────────
# 训练设置
# ──────────────────────────────────────────────────────
model = GoogLeNet(in_channels=1, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
num_epochs = 10

# 打印模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型架构: GoogLeNet (Inception v1)")
print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
print("=" * 60)


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

        # 每 100 个 batch 打印一次
        if (batch_idx + 1) % 100 == 0:
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
# 训练循环
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"开始训练 GoogLeNet, 共 {num_epochs} 个 epoch")
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    print(f"Batch size: 64, 学习率: 0.1, 动量: 0.9")
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
            torch.save(model.state_dict(), "12_googlenet_best.pth")

        print(f"Epoch {epoch}/{num_epochs} [{elapsed:.1f}s] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    print("=" * 60)
    print(f"最终测试准确率: {test_acc * 100:.2f}%")
    print(f"最佳测试准确率: {best_test_acc * 100:.2f}%")
    torch.save(model.state_dict(), "12_googlenet_final.pth")
    print("模型已保存: 12_googlenet_final.pth")


# ──────────────────────────────────────────────────────
# 模型架构说明
# ──────────────────────────────────────────────────────
"""
GoogLeNet 原论文模型架构 (输入: 1×96×96):

┌─────────────────────────────────────────────────────────────────┐
│ 模块      │ 层                    │ 输出形状                   │
├─────────────────────────────────────────────────────────────────┤
│ b1        │ Conv7×7(64, s=2)      │ [B, 64, 48, 48]           │
│           │ MaxPool3×3(s=2)       │ [B, 64, 24, 24]           │
├─────────────────────────────────────────────────────────────────┤
│ b2        │ Conv1×1(64)           │ [B, 64, 24, 24]           │
│           │ Conv3×3(192)          │ [B, 192, 24, 24]          │
│           │ MaxPool3×3(s=2)       │ [B, 192, 12, 12]          │
├─────────────────────────────────────────────────────────────────┤
│ b3        │ Inception(256)        │ [B, 256, 12, 12]          │
│           │ Inception(480)        │ [B, 480, 12, 12]          │
│           │ MaxPool3×3(s=2)       │ [B, 480, 6, 6]            │
├─────────────────────────────────────────────────────────────────┤
│ b4        │ Inception(512) ×5     │ [B, 832, 6, 6]            │
│           │ MaxPool3×3(s=2)       │ [B, 832, 3, 3]            │
├─────────────────────────────────────────────────────────────────┤
│ b5        │ Inception(1024) ×2    │ [B, 1024, 3, 3]           │
│           │ AdaptiveAvgPool       │ [B, 1024, 1, 1]           │
│           │ Flatten               │ [B, 1024]                 │
├─────────────────────────────────────────────────────────────────┤
│ 输出层    │ Linear(1024, 10)      │ [B, 10]                   │
└─────────────────────────────────────────────────────────────────┘

Inception块结构 (4条并行路径):

┌─────────────────────────────────────────────────────────────────┐
│ 路径1: 1×1卷积                                                  │
│ 路径2: 1×1卷积(降维) → 3×3卷积                                  │
│ 路径3: 1×1卷积(降维) → 5×5卷积                                  │
│ 路径4: 3×3 MaxPool → 1×1卷积                                    │
│ 输出: 4条路径在通道维度拼接                                      │
└─────────────────────────────────────────────────────────────────┘

Inception参数格式: Inception(输入通道, c1, (c2a, c2b), (c3a, c3b), c4)

GoogLeNet的创新点:
1. Inception块: 多尺度特征提取(1×1, 3×3, 5×5)
2. 1×1卷积降维: 减少计算量，增加非线性
3. 全局平均汇聚: 替代全连接层，减少参数
4. 网络深度: 22层(含池化层)，参数量仅约500万

本实现适配说明:
- 输入从224×224调整为MNIST的28×28(通过Resize到96×96)
- 输出层调整为10类(MNIST类别数)
- 保持原始GoogLeNet的核心设计理念
"""
