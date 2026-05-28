# LeNet-5 (1998) 极简复刻版
# 复刻目标：
# - 数据集：MNIST
# - 网络结构：LeNet-5
# - 激活函数：Tanh
# - 平均池化：AvgPool
# - 输入尺寸：32x32
# - 优化器：SGD
# - 学习率：0.01
# - Batch Size：64
# - Epoch：20
#
# pip install torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------------
# 超参数（尽量贴近原论文）
# ----------------------------
BATCH_SIZE = 64
LR = 0.01
EPOCHS = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ----------------------------
# 数据预处理
# 原始MNIST是28x28
# LeNet-5要求32x32
# ----------------------------
transform = transforms.Compose([
    transforms.Pad(2),                 # 28x28 -> 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ----------------------------
# LeNet-5
# ----------------------------
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # C1: 1x32x32 -> 6x28x28
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),

            # S2: 6x28x28 -> 6x14x14
            nn.AvgPool2d(kernel_size=2, stride=2),

            # C3: 6x14x14 -> 16x10x10
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),

            # S4: 16x10x10 -> 16x5x5
            nn.AvgPool2d(kernel_size=2, stride=2),

            # C5: 16x5x5 -> 120x1x1
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),

            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)

        # flatten
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x

model = LeNet5().to(DEVICE)

# ----------------------------
# 损失函数 + 优化器
# ----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=LR
)

# ----------------------------
# 训练
# ----------------------------
for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {total_loss / len(train_loader):.4f}")

# ----------------------------
# 测试
# ----------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")