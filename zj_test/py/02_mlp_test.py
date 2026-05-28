import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# =========================================
# 1. 加载数据
# =========================================

data = load_digits()

X = data.data
y = data.target

# 像素归一化
X = X / 16.0

# 转 tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# 划分训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================
# 2. 定义 MLP 模型
# =========================================

net = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)


# 权重初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)


# =========================================
# 3. 损失函数 & 优化器
# =========================================

loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


# =========================================
# 4. accuracy
# =========================================

def accuracy(y_hat, y):

    preds = y_hat.argmax(dim=1)

    return (preds == y).float().mean()


# =========================================
# 5. mini-batch
# =========================================

def batch_iterator(X, y, batch_size):

    indices = torch.randperm(len(X))

    for start in range(0, len(X), batch_size):

        end = start + batch_size

        batch_idx = indices[start:end]

        yield X[batch_idx], y[batch_idx]


# =========================================
# 6. 超参数
# =========================================

batch_size = 64
epochs = 100


# =========================================
# 7. 训练
# =========================================

for epoch in range(epochs):

    total_loss = 0

    for X_batch, y_batch in batch_iterator(
        X_train,
        y_train,
        batch_size
    ):

        # forward
        y_hat = net(X_batch)
        l = loss(y_hat, y_batch)

        total_loss += l.item()

        # backward
        trainer.zero_grad()
        l.backward()

        # SGD 更新
        trainer.step()

    # test
    with torch.no_grad():

        test_preds = net(X_test)

        test_acc = accuracy(
            test_preds,
            y_test
        )

    print(
        f"epoch={epoch+1:03d} "
        f"loss={total_loss:.4f} "
        f"test_acc={test_acc:.4f}"
    )


# =========================================
# 8. 预测示例
# =========================================

with torch.no_grad():

    sample = X_test[:5]

    preds = net(sample).argmax(dim=1)

print("\n预测:", preds.tolist())
print("真实:", y_test[:5].tolist())
