"""
AlexNet - 在 MNIST 上训练和测试
基于 d2l 教材实现,参照原始 AlexNet (Krizhevsky et al., 2012) 设计
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
# AlexNet 模型定义
# 原始输入: 1×224×224, 这里适配 MNIST (28×28 resize 到 224×224)
# ──────────────────────────────────────────────────────
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 第 1 层卷积: 1×224×224 -> 96×54×54
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 96×26×26

            # 第 2 层卷积: 96×26×26 -> 256×26×26
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 256×12×12

            # 第 3-5 层卷积
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # -> 384×12×12
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # -> 384×12×12
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # -> 256×12×12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),           # -> 256×5×5
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────────────
# 数据加载: MNIST, resize 到 224×224 以适配 AlexNet
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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# MNIST 类别名
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# ──────────────────────────────────────────────────────
# 训练设置
# ──────────────────────────────────────────────────────
model = AlexNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 10

# 打印模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,}")
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
    print(f"开始训练 AlexNet, 共 {num_epochs} 个 epoch")
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    print(f"Batch size: 128, 学习率: 0.01")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch}/{num_epochs} [{elapsed:.1f}s] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    print("=" * 60)
    print(f"最终测试准确率: {test_acc * 100:.2f}%")
    torch.save(model.state_dict(), "09_alexnet.pth")
    print("模型已保存: 09_alexnet.pth")
