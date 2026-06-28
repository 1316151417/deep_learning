# GPT-1 复现：生成式预训练 + 下游任务微调

复刻论文 *Improving Language Understanding by Generative Pre-Training* (Radford et al., 2018)
的核心方法：**无监督预训练 → 有监督微调** 的两阶段范式。

> 说明：为可在普通电脑上离线、快速运行，本实现采用 **教学小规模**（4 层 / 128 维 / 4 头，
> 内置小语料与情感数据集）。真实复现需用 BooksCorpus 等大规模语料 + 论文 12 层 / 768 维配置。
> 架构、损失函数与训练配方均对齐论文与 OpenAI 官方实现 `openai/finetune-transformer-lm`。

## 论文要点与本实现的对应

| 论文要点 | 实现位置 | 说明 |
| --- | --- | --- |
| 仅解码器 Transformer（12 层/768 维/12 头） | `model.py: GPTModel` | **Post-LN 残差**（LayerNorm 在残差相加之后，对齐官方 `block`），无末层 ln_f |
| 学习的位置编码 + GELU 前馈 | `model.py` | 非“Attention is All You Need”的正弦编码/ReLU |
| 权重初始化 N(0, 0.02) | `GPTModel._init_weights` | 对齐 GPT 官方实现 |
| LM 头与 token 嵌入权重绑定 | `model.py: LMHead` | `hidden @ wte.T` |
| **无监督预训练目标 L1**（下一个 token 语言模型） | `train.py: pretrain` | L1 = −Σ log P(uᵢ \| uᵢ₋ₖ..uᵢ₋₁) |
| **有监督微调 L2 + 辅助 LM 目标 L1**（λ=0.5） | `train.py: finetune` | L3 = L2 + λ·L1，微调时保留语言能力 |
| Adam (预训练 β2=0.98)，线性 warmup | `train.py` | 预训练余弦衰减、微调线性衰减，梯度裁剪 1.0 |
| **Fig.2 四种任务输入变换** | `data.py` | 分类 / 蕴含 / 相似度 / 多选，含 `[Start]` `[Delim]` `[Extract]` |
| BPE 子词分词 | `tokenizer.py` | 自包含最小 BPE，无需外部依赖 |

### 特殊 token
- `[Start]` 序列起始
- `[Delim]` 分隔两段文本（前提/假设、上下文/问题等）
- `[Extract]` 其位置隐藏表示送入任务分类头（`data.py` 中各变换返回该位置下标）

### 四种任务输入变换（论文 Figure 2）
```
分类:    [Start] text                          [Extract]   → 用 [Extract] 分类
蕴含:    [Start] premise [Delim] hypothesis     [Extract]   → 用 [Extract] 分类
相似度:  [Start] s1 [Delim] s2 [Extract]  与  [Start] s2 [Delim] s1 [Extract]  → 两顺序求和（对称化）
多选:    [Start] context [Delim] Q+A_k [Extract]  对每个候选答案各一条 → softmax
```

## 文件结构
- `model.py` — GPT 架构与任务头（`GPT`, `ClassificationHead`）
- `tokenizer.py` — 自包含 BPE 分词器
- `data.py` — 语料、LM 批数据、Fig.2 四种任务输入变换、分类批整理
- `train.py` — 预训练 / 微调 / 评估循环与学习率调度
- `main.py` — 入口：预训练 → 保存 → 微调（对比 预训练初始化 vs 从零训练）→ 续写与任务变换演示

## 运行
```bash
cd zj_test/py/30_gpt1
python main.py
# 可选环境变量：PRETRAIN_EPOCHS / FINETUNE_EPOCHS / BPE_VOCAB
PRETRAIN_EPOCHS=20 FINETUNE_EPOCHS=40 python main.py
```
运行后会在本目录生成：
- `gpt1_pretrained.pth` — 预训练权重（被 `.gitignore` 忽略）
- `pretrain_loss.png` — 预训练损失曲线
- `finetune_compare.png` — 预训练初始化 vs 从零训练 的验证准确率对比

输出会演示：预训练 LM 损失下降、模型对提示词的续写、情感分类微调，
以及「预训练初始化」相对「从零训练」的收益（在小数据上收益有限，详见论文对大规模语料的依赖）。
