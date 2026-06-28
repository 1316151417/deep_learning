"""预训练 (无监督 LM) 与微调 (有监督 + 辅助 LM 损失) 训练循环。

预训练目标 (论文 L1，无监督语言模型)：
    L1 = - Σ_i log P(u_i | u_{i-k}, ..., u_{i-1})
微调目标 (论文 L3，λ = 0.5，在下游任务上引入语言模型作为辅助目标)：
    L3 = L2 + λ · L1
其中 L2 为下游任务的有监督损失 (这里用分类交叉熵)，L1 为同一序列上的语言模型损失。
辅助 LM 目标有助于在微调时保留通用语言能力、加速收敛、提升泛化。

优化器与训练配方 (对齐论文)：
  * Adam：预训练 β=(0.9, 0.98), ε=1e-9；微调 β=(0.9, 0.999)。
  * 学习率：线性 warmup，之后预训练用余弦衰减、微调用线性衰减。
  * 梯度范数裁剪 (clip = 1.0)。
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from data import lm_batch, collate_classification as collate


# ---------------------------------------------------------------------------
# 学习率调度
# ---------------------------------------------------------------------------
def _cosine_lr(it, warmup, max_iters):
    if it < warmup:
        return it / max(1, warmup)                       # 线性 warmup
    progress = (it - warmup) / max(1, max_iters - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))    # 余弦衰减到 0


def _linear_lr(it, warmup, max_iters):
    if it < warmup:
        return it / max(1, warmup)                        # 线性 warmup
    return max(0.0, 1.0 - (it - warmup) / max(1, max_iters - warmup))  # 线性衰减到 0


def make_scheduler(optimizer, kind: str, warmup: int, max_iters: int):
    """构造 LambdaLR 调度器。kind='cosine' 用于预训练，'linear' 用于微调。"""
    fn = _cosine_lr if kind == "cosine" else _linear_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: fn(it, warmup, max_iters))


# ---------------------------------------------------------------------------
# 无监督预训练
# ---------------------------------------------------------------------------
def pretrain(model, token_ids, *, block_size, batch_size, epochs, lr,
             warmup_ratio, device, log_every=50):
    """语言模型预训练循环。返回每个记录点的 (步数, 平均损失) 列表。"""
    model.to(device).train()
    # 预训练用 β2=0.98 (论文设定)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    iters_per_epoch = max(1, 100)                          # 每 epoch 采样步数
    max_iters = epochs * iters_per_epoch
    warmup = max(1, int(warmup_ratio * max_iters))
    scheduler = make_scheduler(optimizer, "cosine", warmup, max_iters)
    gen = torch.Generator(device="cpu").manual_seed(0)
    history, running, steps = [], 0.0, 0

    for _ in range(epochs):
        for _ in range(iters_per_epoch):
            x, y = lm_batch(token_ids, block_size, batch_size, device, gen)
            logits = model(x)                              # (B, T, V)
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
# 有监督微调 (分类 + 辅助 LM 损失)
# ---------------------------------------------------------------------------
def finetune(model, classifier, samples, tok, *, n_ctx, batch_size, epochs, lr,
             lm_weight, warmup_ratio, device):
    """在分类任务上微调 GPT。返回 (训练历史, 评估准确率)。

    total_loss = L2(分类) + lm_weight * L1(同序列语言模型)
    """
    model.to(device).train()
    classifier.to(device).train()
    params = list(model.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    steps_per_epoch = max(1, len(samples) // batch_size)
    max_iters = epochs * steps_per_epoch
    warmup = max(1, int(warmup_ratio * max_iters))
    scheduler = make_scheduler(optimizer, "linear", warmup, max_iters)
    history = []

    for epoch in range(epochs):
        # 每 epoch 打乱顺序
        order = torch.randperm(len(samples)).tolist()
        epoch_loss = 0.0
        for b in range(steps_per_epoch):
            batch = [samples[i] for i in order[b * batch_size:(b + 1) * batch_size]]
            x, extract_pos, labels, valid = collate(batch, tok, n_ctx)
            x, extract_pos, labels, valid = (t.to(device) for t in (x, extract_pos, labels, valid))

            hidden = model.hidden_states(x)               # (B, T, n_embd)
            logits = classifier(hidden, extract_pos)      # (B, n_classes)
            l2 = F.cross_entropy(logits, labels)          # 有监督分类损失

            # 辅助 LM 损失：仅在被 padding 之外的真实位置计算
            lm_logits = model.lm_head(hidden)             # (B, T, V)
            shift_logits = lm_logits[:, :-1, :].reshape(-1, lm_logits.size(-1))
            shift_targets = x[:, 1:].reshape(-1)
            shift_valid = valid[:, 1:].reshape(-1)
            if shift_valid.any():
                lm_losses = F.cross_entropy(shift_logits, shift_targets, reduction="none")
                l1 = (lm_losses * shift_valid).sum() / shift_valid.sum()
            else:
                l1 = torch.tensor(0.0, device=device)

            loss = l2 + lm_weight * l1
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        history.append((epoch + 1, epoch_loss / steps_per_epoch))
    return history


@torch.no_grad()
def evaluate(model, classifier, samples, tok, *, n_ctx, batch_size, device):
    """在分类数据上计算准确率。"""
    model.to(device).eval()
    classifier.to(device).eval()
    correct = total = 0
    for b in range(0, len(samples), batch_size):
        batch = samples[b:b + batch_size]
        x, extract_pos, labels, _ = collate(batch, tok, n_ctx)
        x, extract_pos, labels = (t.to(device) for t in (x, extract_pos, labels))
        hidden = model.hidden_states(x)
        preds = classifier(hidden, extract_pos).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)
