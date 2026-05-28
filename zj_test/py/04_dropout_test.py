import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# =========================================
# 1. 加载数据
# =========================================

data = load_digits()
X = data.data / 16.0
y = data.target

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================
# 2. 从零实现：dropout_layer
# =========================================

def dropout_layer(X, dropout):
    """以概率 dropout 丢弃元素，保留部分做 1/(1-p) 缩放"""
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)


# =========================================
# 3. 从零实现：带 dropout 的 MLP
# =========================================

dropout1, dropout2 = 0.2, 0.5


class ScratchNet(nn.Module):
    def __init__(self, num_inputs, num_outputs,
                 num_hiddens1, num_hiddens2, is_training=True):
        super().__init__()
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X))
        if self.training:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout2)
        return self.lin3(H2)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean()


def batch_iterator(X, y, batch_size):
    indices = torch.randperm(len(X))
    for start in range(0, len(X), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


# =========================================
# 4. 训练函数（从零实现）
# =========================================

def train_scratch(net, epochs, lr, batch_size):
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in batch_iterator(X_train, y_train, batch_size):
            y_hat = net(X_batch)
            l = loss(y_hat, y_batch)
            total_loss += l.item()
            trainer.zero_grad()
            l.backward()
            trainer.step()

        # 测试：关闭 dropout
        net.eval()
        with torch.no_grad():
            test_acc = accuracy(net(X_test), y_test)
        net.train()

        if (epoch + 1) % 20 == 0:
            print(
                f"  epoch={epoch+1:03d}  "
                f"loss={total_loss:.4f}  "
                f"test_acc={test_acc:.4f}"
            )


# =========================================
# 5. 简洁实现：使用 nn.Dropout
# =========================================

def build_concise_net(dropout1, dropout2):
    return nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(256, 10)
    )


def train_concise(net, epochs, lr, batch_size):
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in batch_iterator(X_train, y_train, batch_size):
            # nn.Dropout 在 train() 模式下自动生效
            y_hat = net(X_batch)
            l = loss(y_hat, y_batch)
            total_loss += l.item()
            trainer.zero_grad()
            l.backward()
            trainer.step()

        # 测试：net.eval() 自动关闭 dropout
        net.eval()
        with torch.no_grad():
            test_acc = accuracy(net(X_test), y_test)
        net.train()

        if (epoch + 1) % 20 == 0:
            print(
                f"  epoch={epoch+1:03d}  "
                f"loss={total_loss:.4f}  "
                f"test_acc={test_acc:.4f}"
            )


# =========================================
# 6. 运行实验
# =========================================

if __name__ == "__main__":
    epochs, lr, batch_size = 100, 0.1, 64

    print("=" * 55)
    print("从零实现")
    print("=" * 55)

    print("\n【不使用 dropout】")
    net_no = ScratchNet(64, 10, 256, 256, is_training=False)
    net_no.apply(init_weights)
    train_scratch(net_no, epochs, lr, batch_size)

    print("\n【使用 dropout (0.2, 0.5)】")
    net_do = ScratchNet(64, 10, 256, 256, is_training=True)
    net_do.apply(init_weights)
    train_scratch(net_do, epochs, lr, batch_size)

    print("\n" + "=" * 55)
    print("简洁实现 (nn.Dropout)")
    print("=" * 55)

    print("\n【不使用 dropout】")
    net_concise_no = build_concise_net(0.0, 0.0)
    net_concise_no.apply(init_weights)
    train_concise(net_concise_no, epochs, lr, batch_size)

    print("\n【使用 dropout (0.2, 0.5)】")
    net_concise_do = build_concise_net(0.2, 0.5)
    net_concise_do.apply(init_weights)
    train_concise(net_concise_do, epochs, lr, batch_size)
