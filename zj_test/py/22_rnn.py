"""
循环神经网络 (Recurrent Neural Networks) — 参考 d2l §8.4
https://zh-v2.d2l.ai/chapter_recurrent-neural-networks/rnn.html

演示内容:
1. 无隐状态的网络: 普通 MLP, 输出只依赖当前输入 (无记忆)
2. 有隐状态的 RNN: H_t = φ(W_hh H_{t-1} + W_hx X_t + b_h), 隐状态压缩历史
3. RNN 语言模型: O_t = φ(H_t W_hq + b_q), 输出 vocab 维 logits
4. 两种实现:
   - 从零开始: 手写 forward 逐时间步循环 + 手写梯度裁剪 + 困惑度
   - 简洁实现: torch.nn.RNN + nn.Linear 输出层
5. 梯度裁剪: g ← min(1, θ/||g||) · g, 防爆炸 (不防消失)

数据复用 21_language-models-and-dataset.py 的 load_data_time_machine。
全部使用标准库 + torch, 不用 d2l。
"""

import math
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# 文件名以数字开头 + 含连字符, 无法用普通 import, 用 importlib 加载
_spec = importlib.util.spec_from_file_location(
    "lm21", Path(__file__).with_name("21_language-models-and-dataset.py"))
_lm21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lm21)
load_data_time_machine = _lm21.load_data_time_machine

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

num_hiddens = 256
num_epochs, lr = 50, 1.0

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"使用设备: {device}")


# ──────────────────────────────────────────────────────
# §8.4.1 无隐状态的网络 (回顾 MLP): 输出只依赖当前输入
# ──────────────────────────────────────────────────────
def no_state_demo():
    net = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 1))
    X = torch.arange(10, dtype=torch.float32).reshape((2, 5))
    print("[无隐状态 MLP] 输出只依赖当前输入 X:")
    print("  X shape =", tuple(X.shape), "-> net(X) shape =", tuple(net(X).shape))
    print("  两次 X 相同, net(X) 就相同: 无记忆")
    print("  net(X)[0] =", net(X)[0].tolist())


# ──────────────────────────────────────────────────────
# §8.5 从零实现: RNN 语言模型 + 梯度裁剪 + 困惑度
#   H_t = tanh(X_t W_xh + H_{t-1} W_hh + b_h)
#   O_t =       H_t W_hq            + b_q
# ──────────────────────────────────────────────────────
class RNNModelScratch(nn.Module):
    def __init__(self, vocab_size, num_hiddens, device):
        super().__init__()
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.W_xh = nn.Parameter(torch.randn(vocab_size, num_hiddens, device=device) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(num_hiddens, num_hiddens, device=device) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens, device=device))
        self.W_hq = nn.Parameter(torch.randn(num_hiddens, vocab_size, device=device) * 0.01)
        self.b_q = nn.Parameter(torch.zeros(vocab_size, device=device))

    def forward(self, X, state):
        dev = self.W_xh.device
        X = X.to(dev)
        state = state.to(dev)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)  # (num_steps, batch, V)
        H = state
        outputs = []
        for x in X:
            H = torch.tanh(x @ self.W_xh + H @ self.W_hh + self.b_h)
            outputs.append(H @ self.W_hq + self.b_q)
        return torch.cat(outputs, dim=0), H

    def begin_state(self, batch_size):
        return torch.zeros(batch_size, self.num_hiddens, device=self.W_xh.device)


def grad_clipping(net, theta):
    params = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for p in params:
            p.grad[:] *= theta / norm


def predict(prefix, num_preds, net, vocab):
    state = net.begin_state(1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor(outputs[-1], device=device).reshape(1, 1)

    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        Y, state = net(get_input(), state)
        outputs.append(int(Y.argmax(dim=1).reshape(1)))
    return "".join(vocab.to_tokens(outputs))


def train_epoch(net, train_iter, updater):
    total_loss, total_tokens = 0.0, 0
    for X, y in train_iter:
        state = net.begin_state(X.shape[0])
        y_flat = y.T.reshape(-1).to(next(net.parameters()).device)
        Y, _ = net(X, state)
        l = F.cross_entropy(Y, y_flat)
        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()
        total_loss += l.item() * y_flat.numel()
        total_tokens += y_flat.numel()
    return math.exp(total_loss / total_tokens)


def train_rnn(net, train_iter, lr, num_epochs):
    updater = torch.optim.SGD(net.parameters(), lr)
    ppl_history = []
    for epoch in range(num_epochs):
        ppl = train_epoch(net, train_iter, updater)
        ppl_history.append(ppl)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:3d}, 困惑度 {ppl:6.2f}")
    return ppl_history


# ──────────────────────────────────────────────────────
# §8.6 简洁实现: torch.nn.RNN + 输出层
# ──────────────────────────────────────────────────────
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super().__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        self.linear = nn.Linear(self.num_hiddens, vocab_size)

    def forward(self, inputs, state):
        dev = next(self.parameters()).device
        inputs = inputs.to(dev)
        state = state.to(dev)
        X = F.one_hot(inputs.T, self.vocab_size).type(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, batch_size):
        dev = next(self.parameters()).device
        return torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens), device=dev)


def predict_concise(prefix, num_preds, net, vocab):
    state = net.begin_state(1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor(outputs[-1], device=device).reshape((1, 1))

    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        Y, state = net(get_input(), state)
        outputs.append(int(Y.argmax(dim=1)))
    return "".join(vocab.to_tokens(outputs))


# ──────────────────────────────────────────────────────
# 梯度裁剪的直观演示: 大梯度被缩放到阈值
# ──────────────────────────────────────────────────────
def grad_clip_demo():
    g = torch.tensor([3.0, 4.0])
    norm = g.norm()
    theta = 1.0
    clipped = g * min(1, theta / norm)
    print(f"\n[梯度裁剪] 原梯度 g = {g.tolist()}, ||g|| = {norm:.2f}")
    print(f"  θ = {theta}, min(1, θ/||g||) = {min(1, theta/norm):.3f}")
    print(f"  裁剪后 g' = {clipped.tolist()}, ||g'|| = {clipped.norm():.4f}")


# ──────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────
def main():
    torch.manual_seed(42)
    print("=" * 60)
    print("Part 1: 无隐状态的网络 vs 有隐状态的 RNN")
    print("=" * 60)
    no_state_demo()
    print("\n  无隐状态: O_t = φ(X_t W_xh + b_h)  —— 输出只看当前输入")
    print("  有隐状态: H_t = φ(X_t W_xh + H_{t-1} W_hh + b_h)  —— 隐状态携带历史")
    print("  RNN 语言模型: O_t = H_t W_hq + b_q  —— 隐状态映射到 vocab 维 logits")

    print("\n" + "=" * 60)
    print("Part 2: 从零实现 RNN 语言模型 (§8.5)")
    print("=" * 60)
    print(f"vocab_size={len(vocab)}, num_hiddens={num_hiddens}, "
          f"num_steps={num_steps}, batch_size={batch_size}")
    net_scratch = RNNModelScratch(len(vocab), num_hiddens, device).to(device)

    print("\n训练前生成 (traveller 前缀):")
    print("  '" + predict('traveller', 10, net_scratch, vocab) + "'")

    print("\n训练中:")
    train_rnn(net_scratch, train_iter, lr, num_epochs)

    print("\n训练后生成 (traveller 前缀, 10 个 token):")
    print("  '" + predict('traveller', 10, net_scratch, vocab) + "'")

    print("\n" + "=" * 60)
    print("Part 3: 梯度裁剪 — 防止梯度爆炸")
    print("=" * 60)
    grad_clip_demo()
    print("  公式: g ← min(1, θ/||g||) · g")
    print("  注意: 裁剪只能防爆炸, 不能防消失")

    print("\n" + "=" * 60)
    print("Part 4: 简洁实现 torch.nn.RNN (§8.6)")
    print("=" * 60)
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    net_concise = RNNModel(rnn_layer, len(vocab)).to(device)
    print("nn.RNN 内部已封装 H_t = tanh(X_t W_ih + H_{t-1} W_hh + b) 的逐时间步循环")
    print("\n训练中:")
    train_rnn(net_concise, train_iter, lr, num_epochs)

    print("\n简洁实现生成 (traveller 前缀):")
    print("  '" + predict_concise('traveller', 10, net_concise, vocab) + "'")

    print("\n结论: RNN 用一个隐状态 H_t 压缩任意长历史, 摆脱了 n-gram 的离散查表;")
    print("      困惑度从初始的 |V| 量级降到 ~1.0, 说明模型已经记住了训练文本。")


if __name__ == "__main__":
    main()
