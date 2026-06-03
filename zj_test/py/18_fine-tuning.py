"""
微调 (Fine-Tuning) — 参考 d2l §13.2
https://zh-v2.d2l.ai/chapter_computer-vision/fine-tuning.html

演示内容:
1. 下载热狗识别数据集，可视化样本
2. 使用 torchvision 预训练 ResNet-18 微调 (输出层用 10× 学习率)
3. 对比: 微调 vs 从头训练 vs 冻结骨干仅训练输出层

全部使用 PyTorch / torchvision 官方 API，无自定义实现。
"""

import os
import zipfile
import urllib.request

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import time

# ──────────────────────────────────────────────────────
# 设备
# ──────────────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"使用设备: {device}")

DATA_DIR = "./data"
HOTDOG_DIR = os.path.join(DATA_DIR, "hotdog")
HOTDOG_URL = "http://d2l-data.s3-accelerate.amazonaws.com/hotdog.zip"

BATCH_SIZE = 128
NUM_EPOCHS = 10
LR_FINETUNE = 5e-5
LR_SCRATCH = 5e-4


# ──────────────────────────────────────────────────────
# 数据准备
# ──────────────────────────────────────────────────────
def download_hotdog():
    """下载并解压热狗数据集"""
    if os.path.exists(os.path.join(HOTDOG_DIR, "train")):
        print("热狗数据集已存在，跳过下载。")
        return
    zip_path = os.path.join(DATA_DIR, "hotdog.zip")
    print(f"下载热狗数据集 → {zip_path}")
    os.makedirs(DATA_DIR, exist_ok=True)
    urllib.request.urlretrieve(HOTDOG_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    print("解压完成。")


def show_samples(dataset, n=8):
    """显示数据集前 n 张正类和最后 n 张负类图片"""
    pos = [dataset[i][0] for i in range(n)]
    neg = [dataset[-i - 1][0] for i in range(n)]
    images = pos + neg
    # PIL → Tensor
    to_tensor = transforms.ToTensor()
    tensors = [to_tensor(img) for img in images]
    grid = vutils.make_grid(tensors, nrow=n, padding=2, normalize=True)
    plt.figure(figsize=(n * 1.8, 3.5))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title("Row 1: hotdog  |  Row 2: not-hotdog", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────
# 增广 & 数据加载
# ──────────────────────────────────────────────────────
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

train_augs = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

test_augs = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


def load_data(augs, is_train=True):
    """使用 ImageFolder 加载热狗数据集"""
    folder = os.path.join(HOTDOG_DIR, "train" if is_train else "test")
    ds = datasets.ImageFolder(folder, transform=augs)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=is_train, num_workers=0)


# ──────────────────────────────────────────────────────
# 模型构建
# ──────────────────────────────────────────────────────
def build_finetune_net():
    """加载 ImageNet 预训练 ResNet-18，替换输出层为 2 类"""
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 2)
    nn.init.xavier_uniform_(net.fc.weight)
    return net


def build_scratch_net():
    """随机初始化 ResNet-18，输出层为 2 类"""
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    nn.init.xavier_uniform_(net.fc.weight)
    return net


def build_frozen_net():
    """冻结骨干网络，仅训练输出层"""
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # 冻结所有参数
    for param in net.parameters():
        param.requires_grad = False
    # 替换输出层（新层默认 requires_grad=True）
    net.fc = nn.Linear(net.fc.in_features, 2)
    nn.init.xavier_uniform_(net.fc.weight)
    return net


# ──────────────────────────────────────────────────────
# 训练
# ──────────────────────────────────────────────────────
def train_model(net, train_iter, test_iter, lr, num_epochs=NUM_EPOCHS,
                param_group=False, label=""):
    """
    训练模型。

    param_group=True 时，输出层使用 10× 学习率（微调模式）。
    """
    net = nn.DataParallel(net).to(device)

    # 参数分组: 骨干小学习率, 输出层大学习率
    if param_group:
        params_backbone = []
        params_fc = []
        for name, param in net.named_parameters():
            if param.requires_grad:
                if "fc" in name:
                    params_fc.append(param)
                else:
                    params_backbone.append(param)
        optimizer = optim.SGD([
            {"params": params_backbone},
            {"params": params_fc, "lr": lr * 10},
        ], lr=lr, weight_decay=0.001)
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr, weight_decay=0.001
        )

    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    timer = time.time()

    print(f"\n--- {label} ---")
    for epoch in range(num_epochs):
        # 训练
        net.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_correct += (y_hat.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # 测试
        net.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                pred = net(X).argmax(dim=1)
                test_correct += (pred == y).sum().item()
                test_total += y.size(0)
        test_acc = test_correct / test_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"  epoch {epoch + 1:2d}/{num_epochs} | "
              f"loss {train_loss:.4f} | "
              f"train acc {train_acc:.3f} | "
              f"test acc {test_acc:.3f}")

    elapsed = time.time() - timer
    print(f"  耗时 {elapsed:.1f}s, "
          f"速度 {total_samples * num_epochs / elapsed:.0f} samples/sec on {device}")
    return history


def plot_histories(histories, labels):
    """绘制多条训练曲线做对比"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for key, ax, ylabel in [
        ("train_loss", axes[0], "Train Loss"),
        ("train_acc", axes[1], "Train Acc"),
        ("test_acc", axes[2], "Test Acc"),
    ]:
        for h, label in zip(histories, labels):
            ax.plot(range(1, len(h[key]) + 1), h[key], marker="o", label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────
def main():
    # 1. 下载数据
    download_hotdog()

    # 2. 可视化样本
    print("\n" + "=" * 60)
    print("Part 1: 数据集概览")
    print("=" * 60)
    train_raw = datasets.ImageFolder(os.path.join(HOTDOG_DIR, "train"))
    test_raw = datasets.ImageFolder(os.path.join(HOTDOG_DIR, "test"))
    print(f"训练集: {len(train_raw)} 张, 类别: {train_raw.classes}")
    print(f"测试集: {len(test_raw)} 张, 类别: {test_raw.classes}")
    show_samples(train_raw)

    # 3. 加载数据
    train_iter = load_data(train_augs, is_train=True)
    test_iter = load_data(test_augs, is_train=False)

    # 4. 三种训练策略对比
    print("\n" + "=" * 60)
    print("Part 2: 微调 vs 从头训练 vs 冻结骨干")
    print("=" * 60)

    # 4.1 微调 (预训练权重 + 输出层 10× 学习率)
    h_finetune = train_model(
        build_finetune_net(), train_iter, test_iter,
        lr=LR_FINETUNE, param_group=True,
        label=f"微调 (lr={LR_FINETUNE}, fc lr={LR_FINETUNE * 10})",
    )

    # 4.2 从头训练 (随机初始化)
    h_scratch = train_model(
        build_scratch_net(), train_iter, test_iter,
        lr=LR_SCRATCH, param_group=False,
        label=f"从头训练 (lr={LR_SCRATCH})",
    )

    # 4.3 冻结骨干，仅训练输出层
    h_frozen = train_model(
        build_frozen_net(), train_iter, test_iter,
        lr=LR_SCRATCH, param_group=False,
        label=f"冻结骨干 (lr={LR_SCRATCH})",
    )

    # 5. 对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"  微调     → 最终 test acc: {h_finetune['test_acc'][-1]:.3f}")
    print(f"  从头训练 → 最终 test acc: {h_scratch['test_acc'][-1]:.3f}")
    print(f"  冻结骨干 → 最终 test acc: {h_frozen['test_acc'][-1]:.3f}")

    plot_histories(
        [h_finetune, h_scratch, h_frozen],
        [
            f"Fine-Tune (lr={LR_FINETUNE})",
            f"Scratch (lr={LR_SCRATCH})",
            f"Frozen Backbone (lr={LR_SCRATCH})",
        ],
    )


if __name__ == "__main__":
    main()
