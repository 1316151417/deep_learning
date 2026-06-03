"""
图像增广 (Image Augmentation) — 参考 d2l §13.1
https://zh-v2.d2l.ai/chapter_computer-vision/image-augmentation.html

演示内容:
1. 常用增广方法可视化: 翻转、随机裁剪、颜色变换、组合增广
2. 在 Fashion-MNIST 上使用增广训练 ResNet-18 (torchvision 内置模型)

全部使用 PyTorch / torchvision 官方 API，无自定义实现。
"""

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


# ──────────────────────────────────────────────────────
# 可视化辅助
# ──────────────────────────────────────────────────────
def show_augmented_images(img, aug, title, num_rows=2, num_cols=4):
    """对 PIL 图像多次应用增广 aug 并可视化"""
    augmented = [aug(img) for _ in range(num_rows * num_cols)]
    grid = vutils.make_grid(
        torch.stack(augmented) if isinstance(augmented[0], torch.Tensor)
        else torch.stack([transforms.ToTensor()(a) for a in augmented]),
        nrow=num_cols, padding=2, normalize=True
    )
    plt.figure(figsize=(num_cols * 2, num_rows * 2))
    plt.imshow(grid.permute(1, 2, 0).squeeze().numpy(), cmap="gray")
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────
# Part 1: 增广方法可视化
# ──────────────────────────────────────────────────────
def demo_augmentations():
    """在 Fashion-MNIST 样本上展示各种增广效果"""
    print("=" * 60)
    print("Part 1: 图像增广方法可视化")
    print("=" * 60)

    # 下载 Fashion-MNIST，取一张图做演示
    raw = datasets.FashionMNIST(root="./data", train=True, download=True)
    img = raw[0][0]  # PIL Image, 28×28 灰度图

    # ---- 1.1 随机水平翻转 ----
    aug_flip_h = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    show_augmented_images(img, aug_flip_h, "RandomHorizontalFlip")

    # ---- 1.2 随机垂直翻转 ----
    aug_flip_v = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    show_augmented_images(img, aug_flip_v, "RandomVerticalFlip")

    # ---- 1.3 随机裁剪 + 缩放 ----
    aug_crop = transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.1, 1.0), ratio=(0.5, 2.0)),
        transforms.ToTensor(),
    ])
    show_augmented_images(img, aug_crop, "RandomResizedCrop(scale=0.1~1.0)")

    # ---- 1.4 随机旋转 ----
    aug_rotate = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
    ])
    show_augmented_images(img, aug_rotate, "RandomRotation(±30°)")

    # ---- 1.5 随机仿射变换 ----
    aug_affine = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
    ])
    show_augmented_images(img, aug_affine, "RandomAffine(translate=(0.2, 0.2))")

    # ---- 1.6 高斯模糊 ----
    aug_blur = transforms.Compose([
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
    ])
    show_augmented_images(img, aug_blur, "GaussianBlur(kernel_size=3)")

    # ---- 1.7 随机擦除 (需要先 ToTensor) ----
    aug_erase = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.8, scale=(0.02, 0.2)),
    ])
    show_augmented_images(img, aug_erase, "RandomErasing(p=0.8)")

    # ---- 1.8 组合多种增广 ----
    aug_combined = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
    ])
    show_augmented_images(img, aug_combined, "Combined: Flip + Rotation + Crop + Erasing")

    print("增广方法演示完成。\n")


# ──────────────────────────────────────────────────────
# Part 2: Fashion-MNIST 增广训练对比
# ──────────────────────────────────────────────────────

# 测试集增广: 仅 ToTensor（不做随机操作）
test_augs = transforms.Compose([
    transforms.ToTensor(),
])

# 基础增广: 随机翻转
train_augs = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# 丰富增广: 翻转 + 旋转 + 裁剪 + 擦除
train_augs_rich = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
])

BATCH_SIZE = 256
NUM_EPOCHS = 10
LR = 0.001


def load_fashion_mnist(is_train, augs, batch_size=BATCH_SIZE):
    """加载 Fashion-MNIST 并应用增广"""
    dataset = datasets.FashionMNIST(
        root="./data", train=is_train, transform=augs, download=True
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train,
        num_workers=0,  # macOS 兼容
    )


def build_resnet18_mnist():
    """构建适配 Fashion-MNIST (1通道, 28×28) 的 ResNet-18"""
    net = models.resnet18(weights=None, num_classes=10)
    # 修改首层: 1 通道, 小 kernel, 无 maxpool
    net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


def train(net, train_iter, test_iter, num_epochs=NUM_EPOCHS, lr=LR):
    """训练并评估模型，返回每个 epoch 的指标"""
    net = nn.DataParallel(net).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    timer = time.time()

    for epoch in range(num_epochs):
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


def demo_training():
    """对比: 无增广 vs 基础增广 vs 丰富增广 训练 Fashion-MNIST"""
    print("=" * 60)
    print("Part 2: Fashion-MNIST 增广训练对比")
    print("=" * 60)

    test_iter = load_fashion_mnist(False, test_augs)

    # ---- 2.1 无增广 ----
    print("\n[1/3] 无增广训练")
    train_iter_no_aug = load_fashion_mnist(True, test_augs)
    h_no_aug = train(build_resnet18_mnist(), train_iter_no_aug, test_iter)

    # ---- 2.2 基础增广 (随机翻转) ----
    print("\n[2/3] 基础增广 (RandomHorizontalFlip)")
    train_iter_aug = load_fashion_mnist(True, train_augs)
    h_aug = train(build_resnet18_mnist(), train_iter_aug, test_iter)

    # ---- 2.3 丰富增广 (裁剪+翻转+旋转+擦除) ----
    print("\n[3/3] 丰富增广 (Crop + Flip + Rotation + Erasing)")
    train_iter_rich = load_fashion_mnist(True, train_augs_rich)
    h_rich = train(build_resnet18_mnist(), train_iter_rich, test_iter)

    # ---- 对比 ----
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"  无增广    → 最终 test acc: {h_no_aug['test_acc'][-1]:.3f}")
    print(f"  基础增广  → 最终 test acc: {h_aug['test_acc'][-1]:.3f}")
    print(f"  丰富增广  → 最终 test acc: {h_rich['test_acc'][-1]:.3f}")

    plot_histories(
        [h_no_aug, h_aug, h_rich],
        ["No Augmentation", "HorizontalFlip", "Crop + Flip + Rotation + Erasing"],
    )


# ──────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Part 1: 可视化各种增广方法
    demo_augmentations()

    # Part 2: 在 Fashion-MNIST 上对比训练
    demo_training()
