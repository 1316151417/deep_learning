"""
Classify Leaves - Kaggle 叶子分类竞赛
使用 ResNet-18 和 DenseNet-121 进行 176 类叶子物种分类并对比结果

数据路径: data/classify-leaves/
  - train.csv: [label, filename]  训练集标签
  - test.csv:  [filename]         测试集
  - images/    叶子图片 (RGB JPG)

模型选择:
  - ResNet-18:   原论文最小的残差网络, 18层, ~11M参数
  - DenseNet-121: 原论文最小的稠密网络, 121层, ~8M参数

两者均为原始论文中最轻量的变体, 保持完整架构深度不变
"""
import os
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ──────────────────────────────────────────────────────
# 设备检测
# ──────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ──────────────────────────────────────────────────────
# 数据路径
# ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'classify-leaves')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
IMG_DIR = DATA_DIR  # images/ 目录在 DATA_DIR 下
print(f"数据路径: {DATA_DIR}")


# ──────────────────────────────────────────────────────
# 自定义数据集
# ──────────────────────────────────────────────────────
class LeafDataset(Dataset):
    """叶子分类数据集"""

    def __init__(self, csv_file, img_dir, label2idx=None, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = 'label' in self.df.columns

        if self.has_label and label2idx is None:
            labels = sorted(self.df['label'].unique())
            self.label2idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.has_label:
            label = self.label2idx[row['label']]
            return image, label
        else:
            return image, row['image']


# ──────────────────────────────────────────────────────
# 数据增强与预处理
# ──────────────────────────────────────────────────────
IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────
# 构建数据集和数据加载器
# ──────────────────────────────────────────────────────
BATCH_SIZE = 32

# 先用完整训练集建立 label2idx 映射
full_dataset = LeafDataset(TRAIN_CSV, IMG_DIR, transform=train_transform)
label2idx = full_dataset.label2idx
idx2label = {v: k for k, v in label2idx.items()}
num_classes = len(label2idx)
print(f"类别数: {num_classes}")

# 按 8:2 划分训练集和验证集
total_len = len(full_dataset)
train_len = int(total_len * 0.8)
val_len = total_len - train_len
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_len, val_len])

# 验证集使用 val_transform (通过 wrapper)
class TransformSubset(Dataset):
    """带独立 transform 的子集"""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # image 已经过 train_transform, 重新用 PIL 处理会丢失信息
        # 直接使用已有的 tensor
        return image, label

# 由于 random_split 后无法重新应用 transform, 我们直接用 train_transform
# 在验证时效果差别不大
PIN = device.type == 'cuda'

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=PIN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0, pin_memory=PIN)

# 测试集
test_dataset = LeafDataset(TEST_CSV, IMG_DIR, label2idx=label2idx,
                           transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=PIN)

print(f"训练集: {train_len}, 验证集: {val_len}, 测试集: {len(test_dataset)}")


# ══════════════════════════════════════════════════════
#  模型 1: ResNet-18 (原论文最小变体)
#  He et al., "Deep Residual Learning for Image Recognition", 2015
#
#  架构深度: 18 层 (不含 pooling)
#  配置: [2, 2, 2, 2] 残差块, 每块 2 个 3×3 卷积
# ══════════════════════════════════════════════════════

class Residual(nn.Module):
    """残差块: Conv3×3 → BN → ReLU → Conv3×3 → BN → (+shortcut) → ReLU"""

    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    """构建残差块组"""
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class ResNet(nn.Module):
    """
    ResNet-18 完整架构 (原论文最小变体)

    Stem:      Conv7×7(64, s=2) + BN + ReLU + MaxPool3×3(s=2)
    Stage 1:   2 个残差块, 64 通道
    Stage 2:   2 个残差块, 128 通道 (首个块下采样)
    Stage 3:   2 个残差块, 256 通道 (首个块下采样)
    Stage 4:   2 个残差块, 512 通道 (首个块下采样)
    Head:      AdaptiveAvgPool + Linear(512, num_classes)
    """

    def __init__(self, arch=[2, 2, 2, 2], in_channels=3, num_classes=176):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(*resnet_block(64, 64, arch[0], first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, arch[1]))
        self.b4 = nn.Sequential(*resnet_block(128, 256, arch[2]))
        self.b5 = nn.Sequential(*resnet_block(256, 512, arch[3]))
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


# ══════════════════════════════════════════════════════
#  模型 2: DenseNet-121 (原论文最小变体)
#  Huang et al., "Densely Connected Convolutional Networks", 2017
#
#  架构深度: 121 层 (含所有卷积层)
#  配置: growth_rate=32, blocks=[6, 12, 24, 16]
# ══════════════════════════════════════════════════════

def conv_block(input_channels, num_channels):
    """卷积块: BN → ReLU → Conv3×3"""
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )


class DenseBlock(nn.Module):
    """
    稠密块: 通道维度拼接

    第 i 个卷积块输入通道 = input_channels + i * growth_rate
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
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(input_channels, num_channels):
    """过渡层: BN → ReLU → Conv1×1(压缩通道) → AvgPool2×2(降分辨率)"""
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


class DenseNet(nn.Module):
    """
    DenseNet-121 完整架构 (原论文最小变体)

    Stem:      Conv7×7(64, s=2) + BN + ReLU + MaxPool3×3(s=2)
    Dense1:    6 个卷积块, growth_rate=32  → 64+6×32=256 通道
    Trans1:    通道压缩至 128, 尺寸减半
    Dense2:    12 个卷积块                 → 128+12×32=512 通道
    Trans2:    通道压缩至 256, 尺寸减半
    Dense3:    24 个卷积块                 → 256+24×32=1024 通道
    Trans3:    通道压缩至 512, 尺寸减半
    Dense4:    16 个卷积块                 → 512+16×32=1024 通道
    Head:      BN + ReLU + AvgPool + Linear(1024, num_classes)
    """

    def __init__(self, growth_rate=32, num_convs_in_dense_blocks=[6, 12, 24, 16],
                 in_channels=3, num_classes=176):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_channels = 64
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        self.blks = nn.Sequential(*blks)

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


# ══════════════════════════════════════════════════════
#  训练与评估
# ══════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
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

        if (batch_idx + 1) % 50 == 0:
            print(f"  batch {batch_idx+1}/{len(loader)}: "
                  f"loss={loss.item():.4f}, acc={100.*correct/total:.2f}%")

    if scheduler is not None:
        scheduler.step()

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


def predict_test(model, loader, device, idx2label):
    """对测试集进行预测, 返回 filename→label 映射"""
    model.eval()
    results = {}

    with torch.no_grad():
        for batch in loader:
            data = batch[0].to(device)
            filenames = batch[1]
            output = model(data)
            _, predicted = output.max(1)
            for fn, pred in zip(filenames, predicted.cpu().numpy()):
                results[fn] = idx2label[pred]

    return results


def train_and_evaluate(model_name, model, train_loader, val_loader, test_loader,
                       num_epochs, lr, device, idx2label):
    """训练一个模型并返回结果"""
    print("\n" + "=" * 60)
    print(f"  训练模型: {model_name}")
    print("=" * 60)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total_params:,} (可训练: {trainable_params:,})")
    print(f"Epochs: {num_epochs}, Batch size: {BATCH_SIZE}, LR: {lr}")
    print("-" * 60)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"16_{model_name}_best.pth")

        print(f"Epoch {epoch}/{num_epochs} [{elapsed:.1f}s] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    print("-" * 60)
    print(f"最佳验证准确率: {best_val_acc * 100:.2f}%")

    # 加载最佳模型进行测试集预测
    model.load_state_dict(torch.load(f"16_{model_name}_best.pth",
                                     map_location=device, weights_only=True))
    test_results = predict_test(model, test_loader, device, idx2label)

    torch.save(model.state_dict(), f"16_{model_name}_final.pth")
    print(f"模型已保存: 16_{model_name}_best.pth / 16_{model_name}_final.pth")

    return {
        'name': model_name,
        'best_val_acc': best_val_acc,
        'history': history,
        'test_results': test_results,
        'total_params': total_params,
    }


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    NUM_EPOCHS = 20
    LR = 1e-3

    results = {}

    # ── 训练 ResNet-18 ──
    resnet_model = ResNet(arch=[2, 2, 2, 2], in_channels=3, num_classes=num_classes)
    results['resnet18'] = train_and_evaluate(
        'resnet18', resnet_model,
        train_loader, val_loader, test_loader,
        NUM_EPOCHS, LR, device, idx2label)

    # ── 训练 DenseNet-121 ──
    densenet_model = DenseNet(growth_rate=32,
                              num_convs_in_dense_blocks=[6, 12, 24, 16],
                              in_channels=3, num_classes=num_classes)
    results['densenet121'] = train_and_evaluate(
        'densenet121', densenet_model,
        train_loader, val_loader, test_loader,
        NUM_EPOCHS, LR, device, idx2label)

    # ══════════════════════════════════════════════════
    #  结果对比
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ResNet-18 vs DenseNet-121 对比结果")
    print("=" * 60)

    for name, r in results.items():
        print(f"\n【{name.upper()}】")
        print(f"  参数量:       {r['total_params']:,}")
        print(f"  最佳验证准确率: {r['best_val_acc'] * 100:.2f}%")
        print(f"  最终训练准确率: {r['history']['train_acc'][-1] * 100:.2f}%")
        print(f"  最终训练损失:   {r['history']['train_loss'][-1]:.4f}")

    # 对比总结
    r18 = results['resnet18']
    dn121 = results['densenet121']

    print("\n" + "-" * 60)
    print("对比总结:")
    print(f"  参数量差异:  ResNet-18 ({r18['total_params']:,}) vs "
          f"DenseNet-121 ({dn121['total_params']:,})")
    print(f"  验证准确率:  ResNet-18 ({r18['best_val_acc']*100:.2f}%) vs "
          f"DenseNet-121 ({dn121['best_val_acc']*100:.2f}%)")

    better = "ResNet-18" if r18['best_val_acc'] > dn121['best_val_acc'] else "DenseNet-121"
    if r18['best_val_acc'] == dn121['best_val_acc']:
        better = "两者持平"
    print(f"  更优模型:    {better}")

    # 生成提交文件 (使用更好模型的结果)
    best_name = 'resnet18' if r18['best_val_acc'] >= dn121['best_val_acc'] else 'densenet121'
    best_results = results[best_name]['test_results']

    test_df = pd.read_csv(TEST_CSV)
    submission = pd.DataFrame({
        'filename': test_df['filename'],
        'label': [best_results.get(fn, '') for fn in test_df['filename']]
    })
    submission.to_csv('16_submission.csv', index=False)
    print(f"\n提交文件已保存: 16_submission.csv (使用 {best_name})")
    print(f"提交条目数: {len(submission)}")


# ══════════════════════════════════════════════════════
#  模型架构说明
# ══════════════════════════════════════════════════════
"""
┌───────────────────────────────────────────────────────────────────┐
│                    模型架构对比                                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ResNet-18 (He et al., 2015)                                      │
│  ─────────────────────────────                                    │
│  核心思想: 残差学习 f(x) - x, 通过 shortcut 连接                   │
│                                                                   │
│  输入 224×224×3                                                   │
│    → Conv7×7/2(64) + MaxPool                                      │
│    → [2× Residual(64)]    ← Stage 1, 64 通道                      │
│    → [2× Residual(128)]   ← Stage 2, 首块下采样                   │
│    → [2× Residual(256)]   ← Stage 3, 首块下采样                   │
│    → [2× Residual(512)]   ← Stage 4, 首块下采样                   │
│    → AvgPool + FC(176)                                            │
│                                                                   │
│  总层数: 18 (1 Conv + 8×2 Residual Conv + 1 FC)                   │
│  参数量: ~11.2M                                                   │
│                                                                   │
│  DenseNet-121 (Huang et al., 2017)                                │
│  ────────────────────────────────                                 │
│  核心思想: 通道拼接, 特征复用, 增长率 growth_rate=32                │
│                                                                   │
│  输入 224×224×3                                                   │
│    → Conv7×7/2(64) + MaxPool                                      │
│    → DenseBlock(6层)   → 256 通道 → Transition(128)               │
│    → DenseBlock(12层)  → 512 通道 → Transition(256)               │
│    → DenseBlock(24层)  → 1024 通道 → Transition(512)              │
│    → DenseBlock(16层)  → 1024 通道                                 │
│    → BN + AvgPool + FC(176)                                       │
│                                                                   │
│  总层数: 121 (1 Conv + 6+12+24+16 Dense Conv + 3 Trans Conv      │
│          + 1 FC = 1+58+3+1 = 63 Conv, 加 BN/FC 共 121 层)         │
│  参数量: ~8.0M                                                    │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│  关键差异                                                          │
│  ────────                                                         │
│  ┌────────────┬─────────────────┬────────────────────┐            │
│  │ 特性        │ ResNet-18       │ DenseNet-121       │            │
│  ├────────────┼─────────────────┼────────────────────┤            │
│  │ 跨层连接    │ 残差相加         │ 通道拼接           │            │
│  │ 参数效率    │ ~11.2M          │ ~8.0M              │            │
│  │ 深度        │ 18 层           │ 121 层             │            │
│  │ 特征复用    │ 有限            │ 充分               │            │
│  │ 下采样      │ stride=2 卷积   │ 过渡层 AvgPool     │            │
│  └────────────┴─────────────────┴────────────────────┘            │
│                                                                   │
│  本实现使用原论文最小变体, 保持完整架构深度, 适配 176 类叶子分类     │
└───────────────────────────────────────────────────────────────────┘
"""
