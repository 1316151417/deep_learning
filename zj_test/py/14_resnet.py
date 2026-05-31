"""
ResNet (残差网络) - 在 MNIST 上训练和测试
基于 d2l 教材实现,参照原始 ResNet (He et al., 2015) 设计

ResNet 创新点:
1. 残差学习: 学习残差映射 f(x) - x 而非直接学习 f(x)
2. 跨层连接: 输入可通过残余连接更快地向前传播
3. 解决深层网络的退化问题
4. Batch Normalization + 全局平均汇聚层

实现:
- 残差块 (Residual Block)
- ResNet-18/34/50/101/152 配置
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
# 残差块定义
# 核心设计: 让网络学习残差映射 f(x) - x
# ──────────────────────────────────────────────────────
class Residual(nn.Module):
    """
    残差块实现

    两种变体:
    - use_1x1conv=False: 输入输出形状一致，直接相加
    - use_1x1conv=True: 使用1×1卷积调整通道数和分辨率

    结构:
    Conv3×3 → BN → ReLU → Conv3×3 → BN → (+shortcut) → ReLU

    Args:
        input_channels: 输入通道数
        num_channels: 输出通道数
        use_1x1conv: 是否使用1×1卷积调整shortcut
        strides: 第一个卷积的步幅(用于下采样)
    """
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()

        # 主路径: 两个3×3卷积
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)

        # shortcut路径: 可选的1×1卷积
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        # 批量归一化
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # 主路径
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        # shortcut路径
        if self.conv3:
            X = self.conv3(X)

        # 残差连接: 主路径 + shortcut
        Y += X
        return F.relu(Y)


# ──────────────────────────────────────────────────────
# 残差块组构建函数
# ──────────────────────────────────────────────────────
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    """
    构建残差块组

    Args:
        input_channels: 输入通道数
        num_channels: 输出通道数
        num_residuals: 残差块数量
        first_block: 是否为第一个模块(b2)
    Returns:
        残差块组成的列表
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 第一个残差块(非b2模块): 通道翻倍，空间尺寸减半
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            # 其他残差块: 通道数不变
            blk.append(Residual(num_channels, num_channels))
    return blk


# ──────────────────────────────────────────────────────
# ResNet 网络定义
# ──────────────────────────────────────────────────────
class ResNet(nn.Module):
    """
    ResNet网络实现

    架构 (ResNet-18):
    - b1 (Stem): 7×7 Conv + BN + ReLU + MaxPool
    - b2 (Stage 1): 2个残差块, 64通道
    - b3 (Stage 2): 2个残差块, 128通道
    - b4 (Stage 3): 2个残差块, 256通道
    - b5 (Stage 4): 2个残差块, 512通道
    - Head: AdaptiveAvgPool + Linear

    Args:
        arch: 每个阶段的残差块数量列表 [b2, b3, b4, b5]
        in_channels: 输入通道数
        num_classes: 分类数
    """
    def __init__(self, arch=[2, 2, 2, 2], in_channels=1, num_classes=10):
        super().__init__()

        # b1: Stem层
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # b2-b5: 4个残差模块
        self.b2 = nn.Sequential(*resnet_block(64, 64, arch[0], first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, arch[1]))
        self.b4 = nn.Sequential(*resnet_block(128, 256, arch[2]))
        self.b5 = nn.Sequential(*resnet_block(256, 512, arch[3]))

        # 输出层
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.head(x)
        return x


# ──────────────────────────────────────────────────────
# ResNet 变体配置
# ──────────────────────────────────────────────────────

def resnet18(num_classes=10):
    """ResNet-18: 每个模块2个残差块"""
    return ResNet(arch=[2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=10):
    """ResNet-34: b2=3, b3=4, b4=6, b5=3"""
    return ResNet(arch=[3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=10):
    """ResNet-50: 使用Bottleneck架构"""
    return ResNetBottleneck(arch=[3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes=10):
    """ResNet-101: 使用Bottleneck架构"""
    return ResNetBottleneck(arch=[3, 4, 23, 3], num_classes=num_classes)

def resnet152(num_classes=10):
    """ResNet-152: 使用Bottleneck架构"""
    return ResNetBottleneck(arch=[3, 8, 36, 3], num_classes=num_classes)


# ──────────────────────────────────────────────────────
# Bottleneck残差块 (用于ResNet-50/101/152)
# ──────────────────────────────────────────────────────
class Bottleneck(nn.Module):
    """
    Bottleneck残差块

    结构: 1×1 Conv(降维) → 3×3 Conv → 1×1 Conv(升维)

    用于更深的网络(50层+)，降低计算复杂度

    Args:
        input_channels: 输入通道数
        bottleneck_channels: 瓶颈层通道数
        output_channels: 输出通道数
        use_1x1conv: 是否使用1×1卷积调整shortcut
        strides: 步幅(用于下采样)
    """
    def __init__(self, input_channels, bottleneck_channels, output_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()

        # 1×1卷积: 降维
        self.conv1 = nn.Conv2d(input_channels, bottleneck_channels,
                               kernel_size=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        # 3×3卷积: 特征提取
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        # 1×1卷积: 升维
        self.conv3 = nn.Conv2d(bottleneck_channels, output_channels,
                               kernel_size=1)
        self.bn3 = nn.BatchNorm2d(output_channels)

        # shortcut路径
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, output_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv4 = None

    def forward(self, X):
        # 主路径
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))

        # shortcut路径
        if self.conv4:
            X = self.conv4(X)

        # 残差连接
        Y += X
        return F.relu(Y)


class ResNetBottleneck(nn.Module):
    """
    使用Bottleneck架构的ResNet

    适用于ResNet-50/101/152
    """
    def __init__(self, arch=[3, 4, 6, 3], in_channels=1, num_classes=10):
        super().__init__()

        # b1: Stem层
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # b2-b5: 使用Bottleneck块
        self.b2 = self._make_layer(64, 64, 256, arch[0], first_block=True)
        self.b3 = self._make_layer(256, 128, 512, arch[1])
        self.b4 = self._make_layer(512, 256, 1024, arch[2])
        self.b5 = self._make_layer(1024, 512, 2048, arch[3])

        # 输出层
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

    def _make_layer(self, input_channels, bottleneck_channels, output_channels,
                    num_residuals, first_block=False):
        """构建Bottleneck残差块组"""
        layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                layers.append(Bottleneck(input_channels, bottleneck_channels,
                                         output_channels, use_1x1conv=True, strides=2))
            else:
                layers.append(Bottleneck(output_channels, bottleneck_channels,
                                         output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.head(x)
        return x


# ──────────────────────────────────────────────────────
# 数据加载: MNIST, resize 到 96×96 以适配 ResNet
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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# MNIST 类别名
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# ──────────────────────────────────────────────────────
# 模型选择
# ──────────────────────────────────────────────────────
# 选择模型: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
MODEL_TYPE = 'resnet18'

model_map = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}

model = model_map[MODEL_TYPE](num_classes=10).to(device)

# 训练设置
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
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
    print(f"Batch size: 128, 学习率: 0.05, 动量: 0.9")
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
            torch.save(model.state_dict(), f"14_{MODEL_TYPE}_best.pth")

        print(f"Epoch {epoch}/{num_epochs} [{elapsed:.1f}s] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    print("=" * 60)
    print(f"最终测试准确率: {test_acc * 100:.2f}%")
    print(f"最佳测试准确率: {best_test_acc * 100:.2f}%")
    torch.save(model.state_dict(), f"14_{MODEL_TYPE}_final.pth")
    print(f"模型已保存: 14_{MODEL_TYPE}_final.pth")


# ──────────────────────────────────────────────────────
# 模型架构说明
# ──────────────────────────────────────────────────────
"""
ResNet 原论文模型架构 (输入: 1×96×96):

┌─────────────────────────────────────────────────────────────────┐
│ 阶段    │ 组件                      │ 输出形状               │
├─────────────────────────────────────────────────────────────────┤
│ b1      │ Conv7×7(64, s=2) + BN     │ [B, 64, 48, 48]       │
│ (Stem)  │ ReLU + MaxPool3×3(s=2)    │ [B, 64, 24, 24]       │
├─────────────────────────────────────────────────────────────────┤
│ b2      │ 2个残差块                  │ [B, 64, 24, 24]       │
│ Stage1  │ (64→64, 无下采样)          │                        │
├─────────────────────────────────────────────────────────────────┤
│ b3      │ 2个残差块                  │ [B, 128, 12, 12]      │
│ Stage2  │ (64→128, 下采样)           │                        │
├─────────────────────────────────────────────────────────────────┤
│ b4      │ 2个残差块                  │ [B, 256, 6, 6]        │
│ Stage3  │ (128→256, 下采样)          │                        │
├─────────────────────────────────────────────────────────────────┤
│ b5      │ 2个残差块                  │ [B, 512, 3, 3]        │
│ Stage4  │ (256→512, 下采样)          │                        │
├─────────────────────────────────────────────────────────────────┤
│ Head    │ AdaptiveAvgPool            │ [B, 512, 1, 1]        │
│         │ Flatten + Linear(512→10)   │ [B, 10]               │
└─────────────────────────────────────────────────────────────────┘

残差块结构:

┌─────────────────────────────────────────────────────────────────┐
│ 输入 X                                                          │
│   │                                                             │
│   ├─────────────────────────────────────────→ shortcut          │
│   │                                         (可选1×1卷积)       │
│   ↓                                                             │
│ Conv3×3 → BN → ReLU → Conv3×3 → BN                             │
│   │                                                             │
│   ↓                                                             │
│ (+ shortcut) → ReLU → 输出                                     │
└─────────────────────────────────────────────────────────────────┘

ResNet变体配置:

┌─────────────────────────────────────────────────────────────────┐
│ 模型        │ b2  │ b3  │ b4  │ b5  │ 总层数 │ 参数量        │
├─────────────────────────────────────────────────────────────────┤
│ ResNet-18   │ 2   │ 2   │ 2   │ 2   │ 18    │ ~11M          │
│ ResNet-34   │ 3   │ 4   │ 6   │ 3   │ 34    │ ~21M          │
│ ResNet-50   │ 3   │ 4   │ 6   │ 3   │ 50    │ ~25M          │
│ ResNet-101  │ 3   │ 4   │ 23  │ 3   │ 101   │ ~44M          │
│ ResNet-152  │ 3   │ 8   │ 36  │ 3   │ 152   │ ~60M          │
└─────────────────────────────────────────────────────────────────┘

ResNet的设计哲学:
1. 残差学习: 学习 f(x) - x 比学习 f(x) 更容易
2. 跨层连接: 解决深层网络的退化问题
3. Batch Normalization: 加速收敛，稳定训练
4. 全局平均汇聚: 替代全连接层，减少参数

本实现说明:
- 输入从224×224调整为MNIST的28×28(通过Resize到96×96)
- 输出层调整为10类(MNIST类别数)
- 支持ResNet-18/34/50/101/152
- 通过MODEL_TYPE变量切换模型
"""
