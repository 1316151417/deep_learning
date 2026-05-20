import json
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# /// script
# dependencies = ["torch", "scikit-learn"]
# ///

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
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 2. 初始化参数
# =========================================

input_dim = 64
num_classes = 10

W = (torch.randn(input_dim, num_classes) * 0.01).detach().requires_grad_(True)
b = torch.zeros(num_classes).detach().requires_grad_(True)

lr = 0.1
epochs = 100
batch_size = 64

# =========================================
# 3. softmax
# =========================================

def softmax(logits):
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp = torch.exp(logits)
    return exp / exp.sum(dim=1, keepdim=True)

# =========================================
# 4. cross entropy
# =========================================

def cross_entropy(probs, labels):
    batch_size = probs.shape[0]
    correct_probs = probs[torch.arange(batch_size), labels]
    return -torch.log(correct_probs + 1e-12).mean()

# =========================================
# 5. accuracy
# =========================================

def accuracy(probs, labels):
    return (probs.argmax(dim=1) == labels).float().mean()

# =========================================
# 6. mini-batch
# =========================================

def batch_iterator(X, y, batch_size):
    indices = torch.randperm(len(X))
    for start in range(0, len(X), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]

# =========================================
# 7. 训练
# =========================================

for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_batch in batch_iterator(X_train, y_train, batch_size):
        logits = X_batch @ W + b
        probs = softmax(logits)
        loss = cross_entropy(probs, y_batch)
        total_loss += loss.item()
        loss.backward()

        with torch.no_grad():
            W -= lr * W.grad
            b -= lr * b.grad
            W.grad.zero_()
            b.grad.zero_()

    with torch.no_grad():
        test_logits = X_test @ W + b
        test_probs = softmax(test_logits)
        test_acc = accuracy(test_probs, y_test)

    print(f"epoch={epoch+1:03d} loss={total_loss:.4f} test_acc={test_acc:.4f}")

# =========================================
# 8. 导出参数
# =========================================

params = {
    "W": W.detach().tolist(),
    "b": b.detach().tolist(),
}

output_path = "zj_test/shouxie_w_params.json"
with open(output_path, "w") as f:
    json.dump(params, f)

print(f"\n参数已导出到 {output_path}")
print(f"W shape: {list(W.shape)}")
print(f"b shape: {list(b.shape)}")
