import torch
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
# 2. 初始化参数（叶子张量）
# =========================================

input_dim = 64
num_classes = 10

# 正确写法：
# 先完成初始化计算，再开启 requires_grad
#
W = (
    torch.randn(input_dim, num_classes) * 0.01
).detach().requires_grad_(True)

b = torch.zeros(
    num_classes
).detach().requires_grad_(True)


# 超参数
lr = 0.1
epochs = 100
batch_size = 64


# =========================================
# 3. softmax
# =========================================

def softmax(logits):

    # 数值稳定性
    logits = logits - logits.max(
        dim=1,
        keepdim=True
    ).values

    exp = torch.exp(logits)

    probs = exp / exp.sum(
        dim=1,
        keepdim=True
    )

    return probs


# =========================================
# 4. cross entropy
# =========================================

def cross_entropy(probs, labels):

    batch_size = probs.shape[0]

    # 真实类别概率
    correct_probs = probs[
        torch.arange(batch_size),
        labels
    ]

    # 防止 log(0)
    loss = -torch.log(correct_probs + 1e-12)

    return loss.mean()


# =========================================
# 5. accuracy
# =========================================

def accuracy(probs, labels):

    preds = probs.argmax(dim=1)

    return (preds == labels).float().mean()


# =========================================
# 6. mini-batch
# =========================================

def batch_iterator(X, y, batch_size):

    indices = torch.randperm(len(X))

    for start in range(0, len(X), batch_size):

        end = start + batch_size

        batch_idx = indices[start:end]

        yield X[batch_idx], y[batch_idx]


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

        # =================================
        # forward
        # =================================

        # logits
        #
        # (batch,64) @ (64,10)
        # =>
        # (batch,10)
        #
        logits = X_batch @ W + b

        # softmax 概率
        probs = softmax(logits)

        # cross entropy
        loss = cross_entropy(
            probs,
            y_batch
        )

        total_loss += loss.item()

        # =================================
        # backward
        # =================================

        loss.backward()

        # =================================
        # SGD 更新
        # =================================

        with torch.no_grad():

            W -= lr * W.grad
            b -= lr * b.grad

            # PyTorch 梯度默认累加
            # 所以每轮必须清零
            W.grad.zero_()
            b.grad.zero_()

    # =================================
    # test
    # =================================

    with torch.no_grad():

        test_logits = X_test @ W + b

        test_probs = softmax(test_logits)

        test_acc = accuracy(
            test_probs,
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

    logits = sample @ W + b

    probs = softmax(logits)

    preds = probs.argmax(dim=1)

print("\n预测:", preds.tolist())
print("真实:", y_test[:5].tolist())