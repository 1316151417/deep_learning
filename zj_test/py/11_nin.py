"""
NiN (Network in Network) - 在 MNIST 上训练和测试
基于 d2l 教材实现,参照原始 NiN (Lin et al., 2013) 设计

NiN 创新点:
1. NiN块: 卷积层 + 两个1×1卷积层（逐像素MLP）
2. 完全移除全连接层，使用全局平均汇聚层替代
3. 显著减少模型参数，降低过拟合风险
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
# NiN 块定义
# 由一个标准卷积层 + 两个1×1卷积层组成，每层使用ReLU激活
# 1×1卷积层"充当带有ReLU激活函数的逐像素全连接层"
# ──────────────────────────────────────────────────────
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    """
    构建单个NiN块
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 第一层卷积核大小
        strides: 第一层卷积步幅
        padding: 第一层卷积填充
    Returns:
        nn.Sequential: NiN块
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True)
    )


# ──────────────────────────────────────────────────────
# NiN 网络定义
# 借鉴AlexNet的卷积层配置，但完全取消全连接层
# 使用全局平均汇聚层替代全连接层进行分类
# ──────────────────────────────────────────────────────
class NiN(nn.Module):
    """
    NiN网络实现
    原始输入: 1×224×224, 这里适配 MNIST (28×28 resize 到 224×224)

    关键特性:
    - 无全连接层，参数量显著减少
    - 使用AdaptiveAvgPool2d进行全局平均汇聚
    - 输出通道数直接等于类别数
    """
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # NiN块1: 11×11卷积，步幅4
            # 224×224 -> 54×54
            nin_block(in_channels, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),  # -> 26×26

            # NiN块2: 5×5卷积，步幅1
            nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),  # -> 12×12

            # NiN块3: 3×3卷积，步幅1
            nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),  # -> 5×5

            nn.Dropout(p=0.5),

            # NiN块4: 输出通道数 = 类别数
            # 这是NiN的关键设计：最后一个NiN块输出通道数等于类别数
            nin_block(384, num_classes, kernel_size=3, strides=1, padding=1),
        )

        # 全局平均汇聚层：对每个通道的所有空间位置求平均
        # 输出形状: (batch_size, num_classes, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, num_classes)
        return x


# ──────────────────────────────────────────────────────
# 数据加载: MNIST, resize 到 224×224 以适配 NiN
# ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(224),
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
model = NiN(in_channels=1, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
num_epochs = 10

# 打印模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型架构: NiN (Network in Network)")
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
    print(f"开始训练 NiN, 共 {num_epochs} 个 epoch")
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
            torch.save(model.state_dict(), "11_nin_best.pth")

        print(f"Epoch {epoch}/{num_epochs} [{elapsed:.1f}s] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    print("=" * 60)
    print(f"最终测试准确率: {test_acc * 100:.2f}%")
    print(f"最佳测试准确率: {best_test_acc * 100:.2f}%")
    torch.save(model.state_dict(), "11_nin_final.pth")
    print("模型已保存: 11_nin_final.pth")


# ──────────────────────────────────────────────────────
# 模型架构说明
# ──────────────────────────────────────────────────────
"""
NiN 原论文模型架构:

┌─────────────────────────────────────────────────────────────┐
│ 层              │ 输出形状        │ 说明                    │
├─────────────────────────────────────────────────────────────┤
│ NiN块1          │ [B, 96, 54, 54] │ Conv11×11(s=4) + 2×1×1 │
│ MaxPool         │ [B, 96, 26, 26] │ 3×3, stride=2          │
├─────────────────────────────────────────────────────────────┤
│ NiN块2          │ [B, 256, 26, 26]│ Conv5×5(s=1) + 2×1×1   │
│ MaxPool         │ [B, 256, 12, 12]│ 3×3, stride=2          │
├─────────────────────────────────────────────────────────────┤
│ NiN块3          │ [B, 384, 12, 12]│ Conv3×3(s=1) + 2×1×1   │
│ MaxPool         │ [B, 384, 5, 5]  │ 3×3, stride=2          │
├─────────────────────────────────────────────────────────────┤
│ Dropout(0.5)    │ [B, 384, 5, 5]  │ 防止过拟合              │
├─────────────────────────────────────────────────────────────┤
│ NiN块4          │ [B, 10, 5, 5]   │ Conv3×3(s=1) + 2×1×1   │
│ AdaptiveAvgPool │ [B, 10, 1, 1]   │ 全局平均汇聚            │
│ Flatten         │ [B, 10]         │ 输出logits              │
└─────────────────────────────────────────────────────────────┘

NiN vs 传统CNN (AlexNet/VGG) 的关键区别:

┌─────────────────────────────────────────────────────────────┐
│ 特征          │ 传统CNN           │ NiN                   │
├─────────────────────────────────────────────────────────────┤
│ 块内结构      │ 卷积+池化堆叠     │ 卷积+2个1×1卷积(MLP)  │
│ 全连接层      │ 有(参数量大)      │ 完全移除              │
│ 输出层        │ FC层输出logits    │ 全局平均汇聚输出      │
│ 参数量        │ 多(FC层贡献)      │ 显著减少              │
│ 过拟合倾向    │ 高(FC层易过拟合)  │ 低                    │
└─────────────────────────────────────────────────────────────┘

NiN的两大创新:
1. NiN块中使用1×1卷积实现逐像素非线性变换
   - 将空间维度每个像素视为一个样本
   - 通道维度视为不同特征
   - 1×1卷积"充当带有ReLU激活函数的逐像素全连接层"

2. 全局平均汇聚层替代全连接层
   - 对所有空间位置求和
   - 通道数等于类别数
   - "显著减少了模型所需参数的数量"
   - "影响了许多后续卷积神经网络的设计"

本实现适配说明:
- 输入从224×224调整为MNIST的28×28(通过Resize)
- 输出层调整为10类(MNIST类别数)
- 保持原始NiN的核心设计理念
"""
