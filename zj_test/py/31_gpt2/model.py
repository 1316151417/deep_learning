"""GPT-2 模型实现 (论文: Language Models are Unsupervised Multitask Learners, Radford et al. 2019)。

架构对齐 OpenAI 官方实现 (openai/gpt-2) 与论文的 4 种规模 (Small 124M / Medium 355M /
Large 774M / XL 1558M)。相对 GPT-1 的关键变化：
  * 字节级词表 (byte-level)，配合外部的 byte-level BPE 分词器，天然无 OOV、支持任意 Unicode。
  * 上下文长度 n_ctx = 1024 (GPT-1 为 512)。
  * GELU 使用 tanh 近似 (GPT-2 官方实现的具体公式)，而非精确 erf。
  * 残差初始化缩放：注意力/前馈的输出投影 (c_proj) 权重按 1/sqrt(2*n_layer) 额外缩放，稳定深层训练。
  * 仍是 Pre-LN 解码器 + 末层 ln_f + LM 头与 token 嵌入权重绑定。
  * 不再有任务头 / 微调阶段 —— GPT-2 主张零样本 (zero-shot)，任务能力由提示词激发 (见 data.py / main.py)。
权重初始化：Linear/Embedding ~ N(0, 0.02)，bias 置 0，LayerNorm gamma=1 / beta=0。
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """GPT-2 超参数 (默认为可快速运行的教学小规模，括号内为论文 Small 配置)。"""
    vocab_size: int = 256        # 字节级词表大小 (由 byte-level BPE 决定)
    n_ctx: int = 1024            # 上下文长度 / block size (论文 1024，GPT-1 为 512)
    n_embd: int = 128            # 嵌入与隐藏维度 (论文 Small 768)
    n_layer: int = 4             # Transformer 层数 (论文 Small 12)
    n_head: int = 4              # 注意力头数 (论文 Small 12)
    embd_pdrop: float = 0.1      # 嵌入层 dropout
    resid_pdrop: float = 0.1     # 残差 / FFN dropout
    attn_pdrop: float = 0.1      # 注意力权重 dropout
    layer_norm_epsilon: float = 1e-5

    # 论文 Table 2.1 的 4 种规模 (n_layer / n_embd / n_head / 约参数量)
    @staticmethod
    def gpt2_small(vocab_size: int) -> "GPTConfig":
        return GPTConfig(vocab_size=vocab_size, n_ctx=1024, n_embd=768, n_layer=12, n_head=12)   # ~124M

    @staticmethod
    def gpt2_medium(vocab_size: int) -> "GPTConfig":
        return GPTConfig(vocab_size=vocab_size, n_ctx=1024, n_embd=1024, n_layer=24, n_head=16)  # ~355M

    @staticmethod
    def gpt2_large(vocab_size: int) -> "GPTConfig":
        return GPTConfig(vocab_size=vocab_size, n_ctx=1024, n_embd=1280, n_layer=36, n_head=20)  # ~774M

    @staticmethod
    def gpt2_xl(vocab_size: int) -> "GPTConfig":
        return GPTConfig(vocab_size=vocab_size, n_ctx=1024, n_embd=1600, n_layer=48, n_head=25)  # ~1558M


class GELU(nn.Module):
    """高斯误差线性单元，GPT-2 采用 tanh 近似 (官方实现的精确公式)。

    gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))
    GPT-1 用的 PyTorch 默认 erf 精确版；GPT-2 改用此 tanh 近似以匹配官方权重。
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """多头因果自注意力 (对应 GPT-2 的 Conv1D 模块 c_attn / c_proj)。

    qkv 用单个线性层 (c_attn) 融合投影到 3*n_embd，再 split 成 q/k/v；
    输出投影 c_proj 的权重在初始化阶段会被残差缩放。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd 必须能被 n_head 整除"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)      # 融合 QKV
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)          # 输出投影 (残差缩放)
        self.attn_drop = nn.Dropout(cfg.attn_pdrop)
        self.resid_drop = nn.Dropout(cfg.resid_pdrop)
        # 因果掩码 (上三角为 0)，注册为 buffer 以便随设备迁移
        mask = torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx)).view(1, 1, cfg.n_ctx, cfg.n_ctx)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # 重排为 (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # 缩放点积注意力 + 因果掩码
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                               # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))                    # 输出投影 + 残差 dropout


class MLP(nn.Module):
    """位置前馈网络 (GPT-2 的 c_fc → gelu → c_proj)，内层维度 4 * n_embd。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)       # 输出投影 (残差缩放)
        self.gelu = GELU()
        self.drop = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """GPT-2 解码块 (Pre-LN，结构与 GPT-1 相同)：

        x = x + attn(ln_1(x))     # 掩码自注意力子层
        x = x + mlp(ln_2(x))      # 前馈子层
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    """GPT-2 主体：token 嵌入 + 位置嵌入 → 多层 Block → 末层 LayerNorm (ln_f)。

    forward 返回每个位置的隐藏向量 (B, T, n_embd)，供 LM 头使用。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)            # token 嵌入
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.n_embd)                 # 学习的位置嵌入
        self.drop = nn.Dropout(cfg.embd_pdrop)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])  # 论文记作 h
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.apply(self._init_weights)
        # 残差缩放初始化 (GPT-2)：注意力/前馈的输出投影按 1/sqrt(2*n_layer) 额外缩放，
        # 使深层残差路径在堆叠时方差不至于发散。
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.cfg.n_ctx, f"序列长度 {T} 超过 n_ctx={self.cfg.n_ctx}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))                  # 不缩放 token 嵌入
        for block in self.h:
            x = block(x)
        return self.ln_f(x)                                           # (B, T, n_embd)


class LMHead(nn.Module):
    """语言模型头：将隐藏向量映射回字节级词表 logits，与 token 嵌入权重绑定。"""

    def __init__(self, wte: nn.Embedding):
        super().__init__()
        self.wte = wte

    def forward(self, hidden):
        return hidden @ self.wte.weight.t()                           # (B, T, vocab_size)


class GPT(nn.Module):
    """GPT-2：主体 + 语言模型头 (权重绑定)。

    * forward(idx) 返回 LM logits；
    * hidden_states(idx) 取隐藏表示 (供零样本打分等扩展使用)。

    与 GPT-1 不同：不内置任何任务头，所有任务都通过提示词以零样本方式完成。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = GPTModel(cfg)
        self.lm_head = LMHead(self.transformer.wte)                   # 权重绑定

    def hidden_states(self, idx):
        return self.transformer(idx)

    def forward(self, idx):
        return self.lm_head(self.transformer(idx))                    # LM logits

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
