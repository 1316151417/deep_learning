import torch
from torch import nn


# =========================================
# 1. 生成数据（高维 + 少量样本 → 容易过拟合）
# =========================================

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w = torch.ones((num_inputs, 1)) * 0.01
true_b = 0.05


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个 PyTorch 数据迭代器"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=is_train
    )


train_data = synthetic_data(true_w, true_b, n_train)
train_iter = load_array(train_data, batch_size)

test_data = synthetic_data(true_w, true_b, n_test)
test_iter = load_array(test_data, batch_size, is_train=False)


# =========================================
# 2. 从零实现
# =========================================

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    """L2 惩罚项"""
    return torch.sum(w.pow(2)) / 2


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def evaluate_loss(net, data_iter, loss):
    """评估损失"""
    total_loss, n = 0.0, 0
    for X, y in data_iter:
        l = loss(net(X), y)
        total_loss += l.sum().item()
        n += y.numel()
    return total_loss / n


def train_scratch(lambd):
    """从零实现：带权重衰减的训练"""
    w, b = init_params()
    net = lambda X: torch.matmul(X, w) + b
    num_epochs, lr = 100, 0.003

    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 损失 = MSE + λ * L2惩罚
            l = squared_loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)

        if (epoch + 1) % 20 == 0:
            train_loss = evaluate_loss(net, train_iter, squared_loss)
            test_loss = evaluate_loss(net, test_iter, squared_loss)
            print(
                f"  epoch={epoch+1:03d}  "
                f"train_loss={train_loss:.6f}  "
                f"test_loss={test_loss:.6f}"
            )

    print(f"  w的L2范数: {torch.norm(w).item():.4f}\n")


# =========================================
# 3. 简洁实现（使用 PyTorch 内置 weight_decay）
# =========================================

def train_concise(wd):
    """简洁实现：使用 optimizer 的 weight_decay 参数"""
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    # 初始化权重
    net[0].weight.data.normal_()
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003

    # 关键：weight_decay 只应用于权重，不应用于偏置
    trainer = torch.optim.SGD([
        {"params": net[0].weight, "weight_decay": wd},
        {"params": net[0].bias}
    ], lr=lr)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()

        if (epoch + 1) % 20 == 0:
            train_loss = evaluate_loss(net, train_iter, loss)
            test_loss = evaluate_loss(net, test_iter, loss)
            print(
                f"  epoch={epoch+1:03d}  "
                f"train_loss={train_loss:.6f}  "
                f"test_loss={test_loss:.6f}"
            )

    print(f"  w的L2范数: {net[0].weight.norm().item():.4f}\n")


# =========================================
# 4. 运行实验
# =========================================

if __name__ == "__main__":
    print("=" * 50)
    print("从零实现")
    print("=" * 50)

    print("\n【不使用权重衰减 (λ=0)】— 观察过拟合")
    train_scratch(lambd=0)

    print("\n【使用权重衰减 (λ=3)】— 缓解过拟合")
    train_scratch(lambd=3)

    print("=" * 50)
    print("简洁实现")
    print("=" * 50)

    print("\n【不使用权重衰减 (wd=0)】")
    train_concise(wd=0)

    print("\n【使用权重衰减 (wd=3)】")
    train_concise(wd=3)
