GPT-2 论文（Radford et al., 2019）为同一套架构定义了四种规模档位——Small (124M)、Medium (355M)、Large (774M)、XL (1558M)——它们共享完全相同的网络拓扑、上下文长度和训练配方，仅通过三个超参数（层数 `n_layer`、隐藏维度 `n_embd`、注意力头数 `n_head`）的协同放大来实现参数量的指数级跃升。本项目将这些预设编码为 `GPTConfig` 的四个静态工厂方法，以最小认知成本实现规模切换。本文将逐一拆解各档位的配置细节、设计不变量、参数分布规律，以及规模相关的初始化行为。

Sources: [model.py](model.py#L1-L49)

## 配置入口：GPTConfig 数据类与工厂方法模式

所有模型规模信息都集中在 `GPTConfig` 这一 dataclass 中。它采用了一种清晰的**两层配置策略**：默认值面向教学演示（极小规模，几秒内可运行），而四种论文预设则以静态方法形式提供，调用时注入标准超参数。

```python
@dataclass
class GPTConfig:
    vocab_size: int = 256        # 字节级词表大小
    n_ctx: int = 1024            # 上下文长度（论文 1024，GPT-1 为 512）
    n_embd: int = 128            # 嵌入与隐藏维度（论文 Small 768）
    n_layer: int = 4             # Transformer 层数（论文 Small 12）
    n_head: int = 4              # 注意力头数（论文 Small 12）
    embd_pdrop: float = 0.1      # 嵌入层 dropout
    resid_pdrop: float = 0.1     # 残差 / FFN dropout
    attn_pdrop: float = 0.1      # 注意力权重 dropout
    layer_norm_epsilon: float = 1e-5
```

dataclass 的默认值对应的是教学配置（`n_embd=128, n_layer=4, n_head=4`），括号注释标明了论文 Small 配置的真实值。四个静态工厂方法（`gpt2_small` / `gpt2_medium` / `gpt2_large` / `gpt2_xl`）各自覆盖 `n_layer`、`n_embd`、`n_head` 三个维度，同时统一设置 `n_ctx=1024`，其余字段（dropout 系列、LayerNorm epsilon）继承默认值不变。值得注意的是，`vocab_size` 作为唯一的外部依赖参数，由调用者根据实际训练的分词器传入——这体现了词表大小与模型架构之间的解耦。

Sources: [model.py](model.py#L22-L49)

## 四档位完整参数对比

下表汇总了四种预设的全部超参数，以及由此推导出的关键衍生量：

| 属性 | Small | Medium | Large | XL | 教学默认 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **n_layer** | 12 | 24 | 36 | 48 | 4 |
| **n_embd** | 768 | 1024 | 1280 | 1600 | 128 |
| **n_head** | 12 | 16 | 20 | 25 | 4 |
| **head_dim** (`n_embd/n_head`) | **64** | **64** | **64** | **64** | 32 |
| **n_ctx** | 1024 | 1024 | 1024 | 1024 | 1024 |
| **vocab_size**（论文标准） | 50257 | 50257 | 50257 | 50257 | 256 |
| **≈ 参数量** | 124M | 355M | 774M | 1558M | ~1M |
| **残差缩放因子** `1/√(2·n_layer)` | 0.204 | 0.144 | 0.118 | 0.102 | 0.354 |
| **相邻倍率** | 1.0× | 2.9× | 2.2× | 2.0× | — |

表中数据可通过 `main.py` 的运行输出直接验证——它遍历四个工厂方法，实例化模型并打印参数量：

```python
GPT2_VOCAB = 50257
for name, preset in [("Small", GPTConfig.gpt2_small), ("Medium", GPTConfig.gpt2_medium),
                     ("Large", GPTConfig.gpt2_large), ("XL", GPTConfig.gpt2_xl)]:
    pcfg = preset(vocab_size=GPT2_VOCAB)
    pcount = sum(p.numel() for p in GPT(pcfg).parameters())
    print(f"GPT-2 {name}: {pcfg.n_layer}/{pcfg.n_embd}/{pcfg.n_head} -> {pcount/1e6:,.0f}M")
```

Sources: [model.py](model.py#L34-L49), [main.py](main.py#L139-L145)

## 核心设计不变量：head_dim 恒等于 64

四种预设中最容易被忽视、却最具设计意图的约束是：**所有档位的 head_dim 都恰好等于 64**（`n_embd / n_head = 64`）。这不是巧合，而是 GPT-2 架构团队有意识的工程选择：

- `CausalSelfAttention.__init__` 中的断言 `assert cfg.n_embd % cfg.n_head == 0` 确保了该约束的硬性约束
- 每个注意力头的计算量、QKV 矩阵的维度划分在所有规模下保持一致，意味着**注意力机制的表达力瓶颈由头数决定**，而非单头维度
- 当模型从 Small 扩展到 XL 时，`n_head` 从 12 增加到 25（≈2.1×），`n_embd` 从 768 增加到 1600（≈2.1×），两者保持同步线性增长，维持了 `head_dim = 64` 的不变量

这一约束直接影响注意力的计算复杂度：每个头的 `QKᵀ` 操作在序列长度 `n_ctx=1024` 上固定为 `O(1024² × 64)`，总注意力开销与 `n_head` 成正比，与 `n_embd` 无关。

Sources: [model.py](model.py#L64-L99)

## 参数分布：从嵌入主导到 Block 主导

随着规模扩大，参数在不同组件之间的分布比例发生根本性变化。通过解析各模块的参数公式，可以精确量化这一趋势：

```
总参数 = token嵌入(vocab×n_embd) + 位置嵌入(n_ctx×n_embd)
       + n_layer × [每Block参数] + 末层LayerNorm(2×n_embd)

每Block参数 = 2×n_embd (LN₁)
            + n_embd×3n_embd + 3n_embd (c_attn)
            + n_embd×n_embd + n_embd (c_attn的c_proj)
            + 2×n_embd (LN₂)
            + n_embd×4n_embd + 4n_embd (c_fc)
            + 4n_embd×n_embd + n_embd (c_proj)
            ≈ 12 × n_embd²
```

由于 `vocab_size=50257` 远大于 `n_embd`，token 嵌入参数量为 `50257 × n_embd`，而每层 Block 参数量 ≈ `12 × n_embd²`。两者比值决定了哪部分占主导：

| 规模 | token 嵌入占比 | Block 参数占比 | 位置嵌入占比 | 末层 LN 占比 |
|:---|:---:|:---:|:---:|:---:|
| **Small** | 31.0% | 68.3% | 0.6% | <0.1% |
| **Medium** | 14.5% | 85.2% | 0.3% | <0.1% |
| **Large** | 8.3% | 91.5% | 0.2% | <0.1% |
| **XL** | 5.2% | 94.7% | 0.1% | <0.1% |

这一分布规律揭示了两个关键洞察：其一，token 嵌入层在 Small 中占据了近三分之一参数，但到 XL 时已降至 5%——这意味着**权重绑定机制（LM Head 复用 token 嵌入）在大模型中的参数节省效果递减**，但对 Small 而言节省了约 38M 参数（非可忽略）。其二，Block 参数占比从 68% 升至 95%，说明大模型的计算与存储开销几乎完全由 Transformer 深度和宽度决定，嵌入层不再是瓶颈。

Sources: [model.py](model.py#L136-L155), [model.py](model.py#L179-L187)

## 规模相关的初始化行为：残差路径缩放

残差缩放因子 `1/√(2·n_layer)` 是**唯一随规模变化的初始化超参数**——它不在 `GPTConfig` 中显式声明，而是由 `n_layer` 在运行时自动推导。在 `GPTModel.__init__` 中，所有残差路径的输出投影权重（`c_proj.weight`）被额外缩放：

```python
for name, param in self.named_parameters():
    if name.endswith("c_proj.weight"):
        nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
```

每个 Block 包含两个子层（注意力和前馈），各有一个 `c_proj` 输出投影，因此残差路径在每层中有两次"贡献注入"。当层数从 12 增加到 48 时，残差方差的累积风险急剧上升，缩放因子相应从 0.204 降至 0.102，确保深层网络在初始阶段的信号传播方差稳定。下表展示了四种规模下的具体缩放行为：

| 规模 | `n_layer` | `2·n_layer` | `1/√(2·n_layer)` | c_proj 初始 std |
|:---|:---:|:---:|:---:|:---:|
| **Small** | 12 | 24 | 0.2041 | 0.00408 |
| **Medium** | 24 | 48 | 0.1443 | 0.00289 |
| **Large** | 36 | 72 | 0.1179 | 0.00236 |
| **XL** | 48 | 96 | 0.1021 | 0.00204 |

其他所有参数（包括 `c_attn`、`c_fc` 的权重）统一使用 `std=0.02` 的正态初始化，bias 置零，LayerNorm 初始化 gamma=1/beta=0，与规模无关。

Sources: [model.py](model.py#L142-L167)

## 不变量分析：哪些超参数跨规模保持恒定

理解规模预设时，区分"变化"与"不变"同等重要。以下超参数在所有四种预设中**完全相同**：

| 超参数 | 值 | 代码位置 | 设计理由 |
|:---|:---|:---|:---|
| `n_ctx` | 1024 | 工厂方法显式设置 | 上下文窗口由训练效率与长程依赖折中决定 |
| `embd_pdrop` | 0.1 | dataclass 默认 | 正则化强度与模型规模解耦 |
| `resid_pdrop` | 0.1 | dataclass 默认 | 同上 |
| `attn_pdrop` | 0.1 | dataclass 默认 | 同上 |
| `layer_norm_epsilon` | 1e-5 | dataclass 默认 | 数值稳定性常数 |
| `vocab_size` | 外部传入（论文 50257） | 工厂方法参数 | 由分词器决定，与架构无关 |
| GELU 公式 | tanh 近似 | `GELU` 类 | GPT-2 全系列统一采用 |
| Pre-LN 结构 | 所有 Block | `Block` 类 | 架构拓扑不随规模变化 |
| 权重绑定 | `LMHead(wte)` | `GPT.__init__` | LM Head 复用 token 嵌入 |

这意味着从 Small 切换到 XL 时，代码中**唯一的架构变化**是 `n_layer`、`n_embd`、`n_head` 三个值——这正是工厂方法模式的价值所在：它将规模选择从架构理解的认知负担中完全剥离。

Sources: [model.py](model.py#L22-L49), [model.py](model.py#L190-L213)

## 实践指南：如何选择和切换规模

在实际使用中，规模切换仅需一行代码：

```python
# 论文标准 Small（~124M 参数，需 GPU 训练）
cfg = GPTConfig.gpt2_small(vocab_size=tok.vocab_size)

# 论文标准 XL（~1558M 参数，需多 GPU）
cfg = GPTConfig.gpt2_xl(vocab_size=tok.vocab_size)

# 自定义规模（如快速消融实验）
cfg = GPTConfig(vocab_size=tok.vocab_size, n_ctx=256, n_embd=256, n_layer=6, n_head=4)
```

本项目的 `main.py` 默认使用**教学配置**（`n_embd=128, n_layer=4, n_head=4, n_ctx=128`），参数量约 1M，可在 CPU 上数秒内完成训练。同时，它在运行时打印四种论文预设的参数量，方便开发者对照确认。需要注意的是，教学配置的 `head_dim = 128/4 = 32`，与论文预设的 64 不同——这意味着教学模型在注意力机制上的表达力特征与正式预设存在差异，仅适用于验证代码流程正确性，不适合进行架构消融分析。

Sources: [main.py](main.py#L136-L145), [model.py](model.py#L136-L137)

## 推荐阅读路径

理解了四种规模预设之后，以下方向可深入探索：

- 残差缩放初始化的完整原理分析 → [残差路径缩放初始化：1/√(2·n_layer) 的作用与原理](8-can-chai-lu-jing-suo-fang-chu-shi-hua-1-2-n_layer-de-zuo-yong-yu-yuan-li)
- 权重绑定如何影响参数量计算 → [语言模型头与 Token 嵌入权重绑定机制](9-yu-yan-mo-xing-tou-yu-token-qian-ru-quan-zhong-bang-ding-ji-zhi)
- 各规模如何影响训练配置 → [Adam 优化器配置：权重衰减分组与 β2=0.999 的选择](15-adam-you-hua-qi-pei-zhi-quan-zhong-shuai-jian-fen-zu-yu-b2-0-999-de-xuan-ze)
- 整体架构如何承载这些预设 → [解码器 Transformer 整体架构：嵌入、Block 堆叠与末层归一化](5-jie-ma-qi-transformer-zheng-ti-jia-gou-qian-ru-block-dui-die-yu-mo-ceng-gui-hua)