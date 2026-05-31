"""
DenseNet (稠密连接网络) - 在 MNIST 上训练和测试
基于 d2l 教材实现,参照原始 DenseNet (Huang et al., 2017)

DenseNet 创新点:
1. 稠密连接: 在通道维度上拼接输入与输出(而非相加)
2. 增长率: 每个卷积块输出通道数固定
3. 过渡层: 压缩通道数，降低复杂度
4. 特征复用: 所有前面层的特征都可被后续层访问

实现:
- 稠密块 (Dense Block)
- 过渡层 (Transition Layer)
- 完整 DenseNet 架构
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
# 卷积块定义
# BN → ReLU → Conv3×3
# ──────────────────────────────────────────────────────
def conv_block(input_channels, num_channels):
    """
    构建卷积块 (BN → ReLU → Conv)

    Args:
        input_channels: 输入通道数
        num_channels: 输出通道数(增长率)
    Returns:
        nn.Sequential: 卷积块
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )


# ──────────────────────────────────────────────────────
# 稠密块定义
# 核心思想: 在通道维度上拼接输入与输出
# ──────────────────────────────────────────────────────
class DenseBlock(nn.Module):
    """
    稠密块实现

    包含多个卷积块，前向传播时将每一层的输入X和输出Y在通道维上拼接
    使后续层的输入通道数逐层增长

    第i个卷积块的输入通道数 = input_channels + i * growth_rate

    Args:
        num_convs: 卷积块数量
        input_channels: 输入通道数
        num_channels: 增长率(每个卷积块的输出通道数)
    """
    def __init__(self, num_convs, input_channels, num_channels):
        super().__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 稠密连接: 在通道维度上拼接输入与输出
            X = torch.cat((X, Y), dim=1)
        return X


# ──────────────────────────────────────────────────────
# 过渡层定义
# 用于压缩通道数和降低空间分辨率
# ──────────────────────────────────────────────────────
def transition_block(input_channels, num_channels):
    """
    构建过渡层

    结构: BN → ReLU → Conv1×1(压缩通道) → AvgPool2×2(降低分辨率)

    Args:
        input_channels: 输入通道数
        num_channels: 输出通道数(通常为输入的一半)
    Returns:
        nn.Sequential: 过渡层
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


# ──────────────────────────────────────────────────────
# DenseNet 网络定义
# ──────────────────────────────────────────────────────
class DenseNet(nn.Module):
    """
    DenseNet网络实现

    架构:
    - 前端(b1): 7×7 Conv + BN + ReLU + MaxPool
    - 主体: 4个稠密块 + 3个过渡层交替排列
    - 尾部: BN + ReLU + AdaptiveAvgPool + Linear

    Args:
        growth_rate: 增长率(每个卷积块的输出通道数)
        num_convs_in_dense_blocks: 每个稠密块中的卷积块数量
        in_channels: 输入通道数
        num_classes: 分类数
    """
    def __init__(self, growth_rate=32, num_convs_in_dense_blocks=[4, 4, 4, 4],
                 in_channels=1, num_classes=10):
        super().__init__()

        # 前端: 初始卷积层
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 主体: 4个稠密块 + 3个过渡层
        num_channels = 64
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            # 添加稠密块
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            # 更新通道数: 每个卷积块增加 growth_rate 个通道
            num_channels += num_convs * growth_rate

            # 添加过渡层(最后一个稠密块后不添加)
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        self.blks = nn.Sequential(*blks)

        # 尾部: 全局平均汇聚 + 全连接层
        self.head = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, num_classes)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.blks(x)
        x = self.head(x)
        return x


# ──────────────────────────────────────────────────────
# DenseNet 变体配置
# ──────────────────────────────────────────────────────

def densenet121(num_classes=10):
    """DenseNet-121: growth_rate=32, blocks=[6, 12, 24, 16]"""
    return DenseNet(growth_rate=32,
                    num_convs_in_dense_blocks=[6, 12, 24, 16],
                    num_classes=num_classes)

def densenet169(num_classes=10):
    """DenseNet-169: growth_rate=32, blocks=[6, 12, 32, 32]"""
    return DenseNet(growth_rate=32,
                    num_convs_in_dense_blocks=[6, 12, 32, 32],
                    num_classes=num_classes)

def densenet201(num_classes=10):
    """DenseNet-201: growth_rate=32, blocks=[6, 12, 48, 32]"""
    return DenseNet(growth_rate=32,
                    num_convs_in_dense_blocks=[6, 12, 48, 32],
                    num_classes=num_classes)

def densenet264(num_classes=10):
    """DenseNet-264: growth_rate=32, blocks=[6, 12, 64, 48]"""
    return DenseNet(growth_rate=32,
                    num_convs_in_dense_blocks=[6, 12, 64, 48],
                    num_classes=num_classes)


# ──────────────────────────────────────────────────────
# 数据加载: MNIST, resize 到 96×96 以适配 DenseNet
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
# 模型选择
# ──────────────────────────────────────────────────────
# 选择模型: 'densenet121', 'densenet169', 'densenet201', 'densenet264'
MODEL_TYPE = 'densenet121'

model_map = {
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'densenet264': densenet264,
}

model = model_map[MODEL_TYPE](num_classes=10).to(device)

# 训练设置
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
num_epochs = 10

# 打印模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型架构: {MODEL_TYPE.upper()}")
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
    print(f"开始训练 {MODEL_TYPE.upper()}, 共 {num_epochs} 个 epoch")
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
            torch.save(model.state_dict(), f"15_{MODEL_TYPE}_best.pth")

        print(f"Epoch {epoch}/{num_epochs} [{elapsed:.1f}s] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    print("=" * 60)
    print(f"最终测试准确率: {test_acc * 100:.2f}%")
    print(f"最佳测试准确率: {best_test_acc * 100:.2f}%")
    torch.save(model.state_dict(), f"15_{MODEL_TYPE}_final.pth")
    print(f"模型已保存: 15_{MODEL_TYPE}_final.pth")


# ──────────────────────────────────────────────────────
# 模型架构说明
# ──────────────────────────────────────────────────────
"""
DenseNet 原论文模型架构 (输入: 1×96×96):

┌─────────────────────────────────────────────────────────────────┐
│ 阶段    │ 组件                      │ 通道数变化               │
├─────────────────────────────────────────────────────────────────┤
│ b1      │ Conv7×7(64, s=2) + BN     │ 1 → 64                  │
│ (Stem)  │ ReLU + MaxPool3×3(s=2)    │                         │
├─────────────────────────────────────────────────────────────────┤
│ Dense1  │ 4个卷积块                  │ 64 → 64+4×32=192       │
│         │ (growth_rate=32)          │                         │
├─────────────────────────────────────────────────────────────────┤
│ Trans1  │ BN+ReLU+Conv1×1+AvgPool   │ 192 → 96               │
│         │ (通道减半，尺寸减半)       │                         │
├─────────────────────────────────────────────────────────────────┤
│ Dense2  │ 4个卷积块                  │ 96 → 96+4×32=224       │
├─────────────────────────────────────────────────────────────────┤
│ Trans2  │ BN+ReLU+Conv1×1+AvgPool   │ 224 → 112              │
├─────────────────────────────────────────────────────────────────┤
│ Dense3  │ 4个卷积块                  │ 112 → 112+4×32=240     │
├─────────────────────────────────────────────────────────────────┤
│ Trans3  │ BN+ReLU+Conv1×1+AvgPool   │ 240 → 120              │
├─────────────────────────────────────────────────────────────────┤
│ Dense4  │ 4个卷积块                  │ 120 → 120+4×32=248     │
├─────────────────────────────────────────────────────────────────┤
│ Head    │ BN+ReLU+AvgPool+Linear    │ 248 → 10               │
└─────────────────────────────────────────────────────────────────┘

稠密块结构:

┌─────────────────────────────────────────────────────────────────┐
│ 输入 X₀ (通道数: C)                                             │
│   │                                                             │
│   ├────────────────────────────────────────────────→ 拼接       │
│   │                                                             │
│   ↓                                                             │
│ Conv Block 1: BN→ReLU→Conv3×3 → X₁ (通道数: growth_rate)      │
│   │                                                             │
│   ├────────────────────────────────────────────────→ 拼接       │
│   │                                                             │
│   ↓                                                             │
│ [X₀, X₁] (通道数: C + growth_rate)                             │
│   │                                                             │
│   ↓                                                             │
│ Conv Block 2: BN→ReLU→Conv3×3 → X₂ (通道数: growth_rate)      │
│   │                                                             │
│   ↓                                                             │
│ [X₀, X₁, X₂] (通道数: C + 2*growth_rate)                       │
│   ...                                                           │
└─────────────────────────────────────────────────────────────────┘

DenseNet 变体配置:

┌─────────────────────────────────────────────────────────────────┐
│ 模型         │ growth_rate │ blocks          │ 参数量          │
├─────────────────────────────────────────────────────────────────┤
│ DenseNet-121 │ 32          │ [6,12,24,16]    │ ~8M             │
│ DenseNet-169 │ 32          │ [6,12,32,32]    │ ~14M            │
│ DenseNet-201 │ 32          │ [6,12,48,32]    │ ~20M            │
│ DenseNet-264 │ 32          │ [6,12,64,48]    │ ~34M            │
└─────────────────────────────────────────────────────────────────┘

DenseNet vs ResNet:

┌─────────────────────────────────────────────────────────────────┐
│ 特性          │ ResNet            │ DenseNet                  │
├─────────────────────────────────────────────────────────────────┤
│ 跨层连接方式  │ 相加              │ 通道维拼接                │
│ 通道数变化    │ 保持不变          │ 逐层增长                  │
│ 下采样方式    │ 残差块(stride=2)  │ 过渡层(1×1Conv+AvgPool)   │
│ 参数量        │ 较大              │ 较小                      │
│ 特征复用      │ 有限              │ 充分复用所有前面层特征    │
└─────────────────────────────────────────────────────────────────┘

DenseNet 的优势:
1. 特征复用: 所有前面层的特征都可被后续层访问
2. 参数效率: 更少的参数达到相似或更好的性能
3. 缓解梯度消失: 稠密连接提供多条梯度传播路径
4. 正则化效果: 特征复用减少过拟合风险

本实现说明:
- 输入从224×224调整为MNIST的28×28(通过Resize到96×96)
- 输出层调整为10类(MNIST类别数)
- 支持DenseNet-121/169/201/264
- 通过MODEL_TYPE变量切换模型
"""
