"""Nadaraya-Watson 核回归 (注意力汇聚) — 参考 d2l §10.2"""
import torch
from torch import nn

F = nn.functional


def f(x):
    return 2 * torch.sin(x) + x ** 0.8


def synthetic(n_train=50):
    x = torch.sort(torch.rand(n_train) * 5)[0]
    y = f(x) + torch.normal(0.0, 0.5, (n_train,))
    x_test = torch.arange(0, 5, 0.1)
    return x, y, x_test, f(x_test)


def average_pool(y_train, n_test):
    return torch.repeat_interleave(y_train.mean(), n_test)


def nonparam_attn(x_train, y_train, x_test):
    X = x_test.repeat_interleave(len(x_train)).reshape(-1, len(x_train))
    w = F.softmax(-(X - x_train) ** 2 / 2, dim=1)
    return w @ y_train, w


class NWKernelRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.rand(1))

    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape(-1, keys.shape[1])
        self.attention_weights = F.softmax(-((queries - keys) * self.w) ** 2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)


def leave_one_out(x_train, y_train):
    n = len(x_train)
    mask = (1 - torch.eye(n)).type(torch.bool)
    keys = x_train.repeat((n, 1))[mask].reshape(n, -1)
    values = y_train.repeat((n, 1))[mask].reshape(n, -1)
    return keys, values


def train_param(x_train, y_train, n_epochs=5, lr=0.5):
    keys, values = leave_one_out(x_train, y_train)
    net = NWKernelRegression()
    opt = torch.optim.SGD(net.parameters(), lr)
    loss_fn = nn.MSELoss(reduction='none')
    history = []
    for _ in range(n_epochs):
        opt.zero_grad()
        l = loss_fn(net(x_train, keys, values), y_train)
        l.sum().backward()
        opt.step()
        history.append(l.mean().item())
    return net, history


def main():
    torch.manual_seed(42)
    x_train, y_train, x_test, y_truth = synthetic()
    n_test = len(x_test)
    mse = lambda yh: float(((yh - y_truth) ** 2).mean())

    y_avg = average_pool(y_train, n_test)
    y_np, w_np = nonparam_attn(x_train, y_train, x_test)

    net, history = train_param(x_train, y_train)
    keys = x_train.repeat((n_test, 1))
    values = y_train.repeat((n_test, 1))
    y_param = net(x_test, keys, values).detach()

    print(f'n_train={len(x_train)}, n_test={n_test}')
    print('\n三种汇聚的测试 MSE (越低越好):')
    print(f'  平均汇聚      : {mse(y_avg):.4f}')
    print(f'  非参数注意力  : {mse(y_np):.4f}')
    print(f'  带参数注意力  : {mse(y_param):.4f}')
    print('\n带参数模型训练 loss (每轮):')
    for i, l in enumerate(history):
        print(f'  epoch {i + 1}: {l:.4f}')
    print(f'\n学到的参数 w = {net.w.item():.4f} (>>1 说明注意力比标准高斯核更尖锐)')


if __name__ == '__main__':
    main()
