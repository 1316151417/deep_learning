"""GPT-2 训练：仅无监督语言模型预训练 (无下游微调)，以「困惑度 perplexity」为主指标。

与 GPT-1 的区别：GPT-2 不做任务微调 —— 任务能力靠零样本提示激发 (见 data.py / main.py)。

训练目标 (无监督语言模型，与 GPT-1 的 L1 相同)：
    L = - Σ_i log P(u_i | u_{i-k}, ..., u_{i-1})

优化器与训练配方 (对齐论文)：
  * Adam，β1=0.9, β2=0.999, ε=1e-8 (GPT-1 预训练用 β2=0.98)。
  * 权重衰减 0.01 (GPT-2 显式引入，作用于权重不作用于 bias/LayerNorm)。
  * 线性 warmup，之后余弦衰减；梯度范数裁剪 1.0。
评估指标用困惑度 PPL = exp(平均 token 负对数似然)，是 GPT-2 论文的核心报告指标。
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from data import lm_batch


# ---------------------------------------------------------------------------
# 学习率调度：线性 warmup + 余弦衰减到 0
# ---------------------------------------------------------------------------
def _cosine_lr(it, warmup, max_iters):
    if it < warmup:
        return it / max(1, warmup)                          # 线性 warmup
    progress = (it - warmup) / max(1, max_iters - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))       # 余弦衰减到 0


def make_scheduler(optimizer, warmup: int, max_iters: int):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda it: _cosine_lr(it, warmup, max_iters))


def _split_decay_groups(model: nn.Module, weight_decay: float):
    """把参数分为「需权重衰减」(2D 权重) 与「不衰减」(bias / LayerNorm) 两组。

    GPT-2 的标准做法：LayerNorm 与 bias 不施加权重衰减。
    """
    decay, no_decay = [], []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if name.endswith("bias") or isinstance(module, nn.LayerNorm):
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# 无监督预训练 (语言模型)
# ---------------------------------------------------------------------------
def pretrain(model, token_ids, *, block_size, batch_size, epochs, lr,
             weight_decay, warmup_ratio, device, log_every=50):
    """语言模型预训练循环。返回每个记录点的 (步数, 平均损失) 列表。"""
    model.to(device).train()
    # GPT-2: Adam β2=0.999, ε=1e-8，并按参数组施加权重衰减
    param_groups = _split_decay_groups(model, weight_decay)
    optimizer = torch.optim.Adam(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    iters_per_epoch = max(1, 100)                            # 每 epoch 采样步数
    max_iters = epochs * iters_per_epoch
    warmup = max(1, int(warmup_ratio * max_iters))
    scheduler = make_scheduler(optimizer, warmup, max_iters)
    gen = torch.Generator(device="cpu").manual_seed(0)
    history, running, steps = [], 0.0, 0

    for _ in range(epochs):
        for _ in range(iters_per_epoch):
            x, y = lm_batch(token_ids, block_size, batch_size, device, gen)
            logits = model(x)                                # (B, T, V)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            steps += 1
            if steps % log_every == 0:
                history.append((steps, running / log_every))
                running = 0.0
    return history


# ---------------------------------------------------------------------------
# 困惑度 (perplexity)：GPT-2 的核心评估指标
# ---------------------------------------------------------------------------
@torch.no_grad()
def perplexity(model, token_ids, *, block_size: int, device: torch.device) -> float:
    """在 token 流上滑动非重叠窗口，计算困惑度 PPL = exp(平均 token NLL)。

    窗口大小自适应为 min(block_size, n-1)，以便在短序列上也能评估。
    """
    model.to(device).eval()
    n = len(token_ids)
    if n < 2:
        return float("inf")
    win = min(block_size, n - 1)                              # 自适应窗口长度
    total_nll, n_tokens = 0.0, 0
    for i in range(0, n - win, win):                          # 非重叠滑动窗口
        x = torch.tensor([token_ids[i:i + win]], dtype=torch.long, device=device)
        y = torch.tensor([token_ids[i + 1:i + 1 + win]], dtype=torch.long, device=device)
        logits = model(x)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_nll += nll.item()
        n_tokens += y.numel()
    if n_tokens == 0:                                         # 序列仅比窗口长 1 token
        x = torch.tensor([token_ids[:win]], dtype=torch.long, device=device)
        y = torch.tensor([token_ids[1:1 + win]], dtype=torch.long, device=device)
        logits = model(x)
        total_nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum").item()
        n_tokens = y.numel()
    return math.exp(total_nll / max(1, n_tokens))
