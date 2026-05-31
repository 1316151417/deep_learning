"""
VGG - 在 MNIST 上训练和测试
基于 d2l 教材实现,参照原始 VGG (Simonyan & Zisserman, 2014) 设计
实现 VGG-11, VGG-16, VGG-19 三种架构
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
# VGG 块定义
# 每个VGG块由多个3×3卷积层串联 + 2×2最大汇聚层组成
# ──────────────────────────────────────────────────────
def vgg_block(num_convs, in_channels, out_channels):
    """
    构建单个VGG块
    Args:
        num_convs: 卷积层数量
        in_channels: 输入通道数
        out_channels: 输出通道数
    Returns:
        nn.Sequential: VGG块
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────
# VGG 网络定义
# ──────────────────────────────────────────────────────
class VGG(nn.Module):
    """
    VGG网络实现
    原始输入: 1×224×224, 这里适配 MNIST (28×28 resize 到 224×224)
    """
    def __init__(self, conv_arch, in_channels=1, num_classes=10):
        """
        Args:
            conv_arch: 卷积块配置，如 ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
            in_channels: 输入通道数
            num_classes: 分类数
        """
        super().__init__()

        # 卷积层部分
        conv_blks = []
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*conv_blks)

        # 全连接层部分
        # 经过5个VGG块后，224×224 会变成 7×7
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────────────
# VGG 架构配置 (来自原论文 Table 1)
# ──────────────────────────────────────────────────────

# VGG-11: 8个卷积层 + 3个全连接层 = 11层
vgg_11_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

# VGG-16: 13个卷积层 + 3个全连接层 = 16层
vgg_16_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))

# VGG-19: 16个卷积层 + 3个全连接层 = 19层
vgg_19_arch = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))


def get_vgg_model(arch_name='vgg_11', in_channels=1, num_classes=10):
    """
    获取VGG模型
    Args:
        arch_name: 架构名称，可选 'vgg_11', 'vgg_16', 'vgg_19'
        in_channels: 输入通道数
        num_classes: 分类数
    Returns:
        VGG模型实例
    """
    arch_map = {
        'vgg_11': vgg_11_arch,
        'vgg_16': vgg_16_arch,
        'vgg_19': vgg_19_arch,
    }

    if arch_name not in arch_map:
        raise ValueError(f"不支持的架构: {arch_name}, 可选: {list(arch_map.keys())}")

    conv_arch = arch_map[arch_name]

    # 为了在MNIST上可训练，将通道数缩小4倍
    ratio = 4
    small_conv_arch = [(num_convs, out_channels // ratio)
                       for num_convs, out_channels in conv_arch]

    return VGG(small_conv_arch, in_channels, num_classes)


# ──────────────────────────────────────────────────────
# 数据加载: MNIST, resize 到 224×224 以适配 VGG
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
# 选择模型架构: 'vgg_11', 'vgg_16', 'vgg_19'
ARCH_NAME = 'vgg_11'

model = get_vgg_model(ARCH_NAME).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
num_epochs = 10

# 打印模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型架构: {ARCH_NAME}")
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
    print(f"开始训练 VGG ({ARCH_NAME}), 共 {num_epochs} 个 epoch")
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    print(f"Batch size: 64, 学习率: 0.05, 动量: 0.9")
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
            torch.save(model.state_dict(), f"10_{ARCH_NAME}_best.pth")

        print(f"Epoch {epoch}/{num_epochs} [{elapsed:.1f}s] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    print("=" * 60)
    print(f"最终测试准确率: {test_acc * 100:.2f}%")
    print(f"最佳测试准确率: {best_test_acc * 100:.2f}%")
    torch.save(model.state_dict(), f"10_{ARCH_NAME}_final.pth")
    print(f"模型已保存: 10_{ARCH_NAME}_final.pth")


# ──────────────────────────────────────────────────────
# 模型架构说明
# ──────────────────────────────────────────────────────
"""
VGG 原论文模型架构对比:

┌─────────────────────────────────────────────────────────────┐
│ VGG-11 (A)    │ VGG-16 (D)    │ VGG-19 (E)    │ 输出尺寸    │
├─────────────────────────────────────────────────────────────┤
│ Conv3-64      │ Conv3-64      │ Conv3-64      │ 224×224     │
│               │ Conv3-64      │ Conv3-64      │             │
├─────────────────────────────────────────────────────────────┤
│ MaxPool       │ MaxPool       │ MaxPool       │ 112×112     │
├─────────────────────────────────────────────────────────────┤
│ Conv3-128     │ Conv3-128     │ Conv3-128     │ 112×112     │
│               │ Conv3-128     │ Conv3-128     │             │
├─────────────────────────────────────────────────────────────┤
│ MaxPool       │ MaxPool       │ MaxPool       │ 56×56       │
├─────────────────────────────────────────────────────────────┤
│ Conv3-256     │ Conv3-256     │ Conv3-256     │ 56×56       │
│ Conv3-256     │ Conv3-256     │ Conv3-256     │             │
│               │ Conv3-256     │ Conv3-256     │             │
│               │               │ Conv3-256     │             │
├─────────────────────────────────────────────────────────────┤
│ MaxPool       │ MaxPool       │ MaxPool       │ 28×28       │
├─────────────────────────────────────────────────────────────┤
│ Conv3-512     │ Conv3-512     │ Conv3-512     │ 28×28       │
│ Conv3-512     │ Conv3-512     │ Conv3-512     │             │
│               │ Conv3-512     │ Conv3-512     │             │
│               │               │ Conv3-512     │             │
├─────────────────────────────────────────────────────────────┤
│ MaxPool       │ MaxPool       │ MaxPool       │ 14×14       │
├─────────────────────────────────────────────────────────────┤
│ Conv3-512     │ Conv3-512     │ Conv3-512     │ 14×14       │
│ Conv3-512     │ Conv3-512     │ Conv3-512     │             │
│               │ Conv3-512     │ Conv3-512     │             │
│               │               │ Conv3-512     │             │
├─────────────────────────────────────────────────────────────┤
│ MaxPool       │ MaxPool       │ MaxPool       │ 7×7         │
├─────────────────────────────────────────────────────────────┤
│ FC-4096       │ FC-4096       │ FC-4096       │ 4096        │
│ FC-4096       │ FC-4096       │ FC-4096       │ 4096        │
│ FC-1000       │ FC-1000       │ FC-1000       │ 1000        │
└─────────────────────────────────────────────────────────────┘

关键设计原则:
1. 所有卷积层使用3×3卷积核，padding=1
2. 每个VGG块后接2×2最大汇聚层，步幅为2
3. 深层窄卷积(3×3)比浅层宽卷积更有效
4. 网络深度是影响性能的关键因素

本实现适配说明:
- 输入从224×224调整为MNIST的28×28(通过Resize)
- 通道数缩小4倍以适应MNIST数据集
- 输出层调整为10类(MNIST类别数)
"""
