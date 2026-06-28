"""GPT-1 模型实现 (论文: Improving Language Understanding by Generative Pre-Training, Radford et al. 2018)。

架构细节对齐 OpenAI 官方实现 (openai/finetune-transformer-lm)：
  * 仅解码器 (decoder-only) Transformer：论文 12 层 / 768 维 / 12 头 (本文件可缩放为教学小规模)。
  * 学习的位置编码 (learned positional embeddings)，而非正弦。
  * GELU 激活的位置前馈网络，内层维度 = 4 * n_embd。
  * Pre-LN 残差结构 (LayerNorm 在子层之前)，并在堆叠末尾再加一层 LayerNorm (ln_f)。
  * token 嵌入不乘以 sqrt(d_model) (与原始 Transformer 不同，GPT 不做缩放)。
  * 语言模型头与 token 嵌入权重绑定 (weight tying)。
权重初始化：Linear/Embedding ~ N(0, 0.02)，bias 置 0，LayerNorm gamma=1 / beta=0。
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """GPT 超参数 (默认值为可快速运行的教学小规模，括号内为论文配置)。"""
    vocab_size: int = 256        # 词表大小 (由 tokenizer 决定)
    n_ctx: int = 64              # 最大序列长度 / block size (论文 512)
    n_embd: int = 128            # 嵌入与隐藏维度 (论文 768)
    n_layer: int = 4             # Transformer 层数 (论文 12)
    n_head: int = 4              # 注意力头数 (论文 12)
    embd_pdrop: float = 0.1      # 嵌入层 dropout
    resid_pdrop: float = 0.1     # 残差 / FFN dropout
    attn_pdrop: float = 0.1      # 注意力权重 dropout
    layer_norm_epsilon: float = 1e-5


class GELU(nn.Module):
    """高斯误差线性单元 (GELU)，GPT 用它替代 ReLU。"""

    def forward(self, x):
        return F.gelu(x)


class CausalSelfAttention(nn.Module):
    """多头因果自注意力 (masked self-attention)。

    解码器中每个位置只能关注自身及之前的位置，通过对注意力分数矩阵施加
    上三角 -inf 掩码实现。qkv 用一个线性层一次性投影，再 split 成 q/k/v。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd 必须能被 n_head 整除"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.attn_pdrop)
        self.resid_drop = nn.Dropout(cfg.resid_pdrop)
        # 因果掩码 (上三角为 0)，注册为 buffer 以便随设备迁移
        mask = torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx)).view(1, 1, cfg.n_ctx, cfg.n_ctx)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # 重排为 (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # 缩放点积注意力 + 因果掩码
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))         # 输出投影 + 残差 dropout


class FeedForward(nn.Module):
    """位置前馈网络：两层线性夹 GELU，内层维度 4 * n_embd。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.gelu = GELU()
        self.drop = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        return self.drop(self.fc2(self.gelu(self.fc1(x))))


class Block(nn.Module):
    """GPT 解码块 (Pre-LN)：

        x = x + attn(ln_1(x))     # 掩码自注意力子层
        x = x + ffn(ln_2(x))      # 前馈子层
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    """GPT 主体：token 嵌入 + 位置嵌入 → 多层 Block → 末层 LayerNorm。

    forward 返回每个位置的隐藏向量 (B, T, n_embd)，供 LM 头或任务头使用。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)            # token 嵌入
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.n_embd)                 # 学习的位置嵌入
        self.drop = nn.Dropout(cfg.embd_pdrop)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.apply(self._init_weights)

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
        x = self.drop(self.wte(idx) + self.wpe(pos))                   # 不缩放 token 嵌入
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)                                           # (B, T, n_embd)


class LMHead(nn.Module):
    """语言模型头：将隐藏向量映射回词表 logits，与 token 嵌入权重绑定。"""

    def __init__(self, wte: nn.Embedding):
        super().__init__()
        self.wte = wte

    def forward(self, hidden):
        return hidden @ self.wte.weight.t()                           # (B, T, vocab_size)


class GPT(nn.Module):
    """GPT-1：主体 + 语言模型头 (权重绑定)。

    * 预训练：forward(idx) 得到 LM logits。
    * 微调：hidden_states(idx) 取隐藏表示供任务头使用。
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


class ClassificationHead(nn.Module):
    """分类头：取特殊 token [Extract] 处的隐藏向量 → 线性分类。

    论文做法 (Fig.2 分类任务)：输入 [Start] text [Extract]，用 [Extract] 位置的
    隐藏表示送入线性层预测类别。分类、蕴含任务都用此头 (区别只在序列如何拼接)。
    """

    def __init__(self, n_embd: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_embd, n_classes)

    def forward(self, hidden, extract_pos):
        # hidden: (B, T, n_embd)；extract_pos: (B,) 指定每个样本取哪个位置
        B = hidden.size(0)
        idx = extract_pos.view(B, 1, 1).expand(B, 1, hidden.size(-1))
        pooled = hidden.gather(1, idx).squeeze(1)                     # (B, n_embd)
        return self.linear(pooled)                                    # (B, n_classes)
