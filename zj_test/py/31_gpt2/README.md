# GPT-2 复现：无监督多任务学习 (Zero-Shot)

复刻论文 *Language Models are Unsupervised Multitask Learners* (Radford et al., 2019) 的
核心思想：**一个纯语言模型，无需任何下游微调，仅靠「把任务写成文本续写」就能零样本完成
翻译 / 问答 / 摘要等多种任务** —— 即「无监督多任务学习」。

> 说明：为可在普通电脑上离线、快速运行，本实现采用 **教学小规模**（4 层 / 128 维 / 4 头，
> 内置小语料）。真实复现需用 **WebText（~40GB）** + 论文 4 种规模配置。
> 架构、分词器、训练配方与生成方式均对齐论文与 OpenAI 官方实现 `openai/gpt-2`。

## 与 GPT-1 的核心区别（本实现重点体现）

| 维度 | GPT-1 | **GPT-2** | 实现位置 |
| --- | --- | --- | --- |
| 分词 | 字符/词级 BPE | **字节级 BPE**（无 OOV、支持任意 Unicode，空格→`Ġ`） | `tokenizer.py` |
| 任务方式 | 无监督预训练 → **有监督微调**（任务头 + 辅助 LM） | **无微调**，任务=文本续写（零样本） | 无任务头；`data.py` 提示模板 |
| 上下文长度 | 512 | **1024** | `model.py: GPTConfig` |
| 激活函数 | erf 精确 GELU | **tanh 近似 GELU** | `model.py: GELU` |
| 初始化 | N(0, 0.02) | 输出投影 **按 1/√(2·n_layer) 额外缩放** | `GPTModel.__init__` |
| 评估指标 | 分类准确率 | **困惑度 PPL** | `train.py: perplexity` |
| 模型规模 | 1 个（117M） | **4 个**（124M / 355M / 774M / 1558M） | `GPTConfig.gpt2_{small,medium,large,xl}` |
| 数据 | BooksCorpus | **WebText**（Reddit karma ≥ 3 抓取的网页） | `data.py: WEBTEXT_CORPUS` |
| 优化器 | Adam β2=0.98 | Adam β2=0.999 + **权重衰减 0.01** | `train.py: pretrain` |

## 论文要点与本实现的对应

| 论文要点 | 实现位置 | 说明 |
| --- | --- | --- |
| 仅解码器 Transformer（Pre-LN + ln_f） | `model.py: GPTModel/Block` | 与 GPT-1 同构，可缩放 |
| **字节级 BPE**（256 字节基底 + 合并 + `<\|endoftext\|>`） | `tokenizer.py` | `bytes_to_unicode` 把字节映射为可见字符，空格→`Ġ`(U+0120) |
| **tanh 近似 GELU** | `model.py: GELU` | 匹配 OpenAI 官方权重所用公式 |
| **残差缩放初始化** | `GPTModel.__init__` | `c_proj` 权重按 `1/√(2·n_layer)` 缩放，稳定深层训练 |
| LM 头与 token 嵌入权重绑定 | `model.py: LMHead` | `hidden @ wte.T` |
| 学习的位置编码（1024 位置） | `GPTModel.wpe` | 非正弦 |
| **无监督语言模型目标**（下一 token） | `train.py: pretrain` | L = −Σ log P(uᵢ \| uᵢ₋ₖ..uᵢ₋₁) |
| **困惑度 PPL**（核心指标） | `train.py: perplexity` | PPL = exp(平均 token NLL) |
| Adam（β2=0.999, ε=1e-8）+ 权重衰减 0.01 | `train.py` | LayerNorm/bias 不衰减；线性 warmup + 余弦衰减；梯度裁剪 1.0 |
| **零样本任务：翻译 / 问答 / 摘要** | `data.py` 提示模板 | 任务即文本续写，`translate … :` / `question … answer :` / `… tl;dr :` |
| top-k 采样生成 | `main.py: generate` | temperature + top-k=40，遇 `<\|endoftext\|>` 停止 |
| 论文 4 种规模 | `GPTConfig.gpt2_{small,medium,large,xl}` | 12/24/36/48 层，124M~1558M |

### 字节级分词示例
```
encode("the cat")  ->  ['the', 'Ġcat']        # 'Ġ' 是前导空格 (bytes_to_unicode 把 0x20 映射为 U+0120)
decode(['the','Ġcat'])  ->  'the cat'          # 任意 Unicode 都可无损往返，无 <unk>
```

### 零样本任务格式（论文核心：任务 = 文本续写）
```
翻译:   translate to french , the cat :          ->  le chat
问答:   question : what is the capital of france ? answer :   ->  paris
摘要:   a fast red car drove down the street . tl ; dr :       ->  a fast car drove .
```

## 文件结构
- `model.py` — GPT-2 架构（`GPT`，4 种规模预设，tanh-GELU，残差缩放初始化）
- `tokenizer.py` — 字节级 BPE 分词器（`bytes_to_unicode` + GPT-2 正则切分）
- `data.py` — WebText 风格语料、LM 批数据、零样本任务提示模板与评估集
- `train.py` — 无监督预训练循环、困惑度评估、学习率调度
- `main.py` — 入口：分词器训练 → 预训练 → PPL → 续写 → 零样本任务演示

## 运行
```bash
cd zj_test/py/30_gpt2
python main.py
# 可选环境变量：PRETRAIN_EPOCHS / BPE_VOCAB
PRETRAIN_EPOCHS=20 BPE_VOCAB=1000 python main.py
# 可选：安装 regex 以完全对齐 GPT-2 的 Unicode 正则切分
pip install regex
```
运行后会在本目录生成：
- `gpt2_pretrained.pth` — 预训练权重（被 `.gitignore` 忽略）
- `pretrain_loss.png` — 预训练损失曲线

输出会演示：字节级分词（`Ġ` 表示）、4 种论文规模的参数量、语言模型损失下降与验证困惑度、
top-k 续写，以及翻译/问答/摘要的零样本提示续写。

> 关于零样本效果：教学小语料下，模型能学会训练分布内的「提示→补全」关联（演示机制成立）；
> 对训练时未见过的全新任务/语言的真正零样本泛化，依赖 WebText 量级（~40GB）数据才会涌现 ——
> 这正是 GPT-2 通过扩大数据与模型规模所验证的核心结论。
