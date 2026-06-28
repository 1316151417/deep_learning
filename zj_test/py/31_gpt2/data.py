"""GPT-2 数据：WebText 风格语料、语言模型批数据、零样本 (zero-shot) 任务提示模板。

GPT-2 的核心思想：不再为每个下游任务做有监督微调，而是把「任务」表达成「文本续写」，
让在 WebText 上纯语言建模训练出的模型，零样本地完成翻译 / 问答 / 摘要等任务 ——
即「无监督多任务学习」(unsupervised multitask learner)。

特殊 token：<|endoftext|> 用作文档边界与序列结束符 (见 tokenizer.py)。
"""
import random
from typing import List, Tuple

import torch

# ---------------------------------------------------------------------------
# WebText 风格预训练语料 (用于无监督语言模型训练)。
# WebText = OpenAI 从 Reddit (karma >= 3) 抓取的网页文本，规模约 40GB。
# 为可离线快速运行，这里内置一份小型英文 + 少量多语言/问答/摘要范例，
# 以便 byte-level 分词器与零样本提示共享词汇分布。
# 真实复现需用 WebText 等大规模语料 (几十 GB)。
# ---------------------------------------------------------------------------
WEBTEXT_CORPUS = """
the cat sat on the mat and looked at the warm sun through the window .
a young woman walked into the quiet library and borrowed a thick history book .
the weather was cold and rainy all through the long autumn weekend .
she opened the small wooden box and found a silver ring inside it .
the old fisherman told stories about the deep sea to the curious children .
a fast red car drove down the empty street at midnight and vanished .
we had a delicious dinner at the new restaurant near the river last night .
the movie was long and boring and many people left the cinema early .
he spent the whole afternoon reading a fascinating book about ancient rome .
the garden was full of bright flowers and busy bees in the warm summer .
scientists study the stars and the planets to understand the vast universe .
water boils at one hundred degrees celsius and turns into steam .
the capital of france is paris and the capital of japan is tokyo .
the capital of italy is rome and the capital of england is london .
a dog is an animal and a cat is an animal too .
the sun rises in the east and sets in the west every single day .
honey is sweet and lemon is sour but both taste good together .
she speaks english at home and french at school every day .
the train arrived late and the tired passengers waited on the platform .
a baker makes bread and a farmer grows crops for the whole town .
"""
# 少量「任务即文本」范例：让模型在纯语言建模中自然见到 翻译 / 问答 / 摘要 的格式，
# 从而展示 GPT-2 「无监督多任务」的机制 (真实零样本能力依赖 WebText 量级的数据)。
WEBTEXT_TASK_EXAMPLES = """
the capital of france is : paris
the capital of japan is : tokyo
the capital of italy is : rome
translate to french , the cat : le chat
translate to french , the house : la maison
translate to french , the book : le livre
translate to german , the cat : die katze
question : what is the capital of france ? answer : paris
question : what is the capital of japan ? answer : tokyo
question : what boils at one hundred degrees ? answer : water
a long and boring movie that many people left early . tl ; dr : a boring movie .
a delicious dinner at the new restaurant near the river . tl ; dr : a good dinner .
a fast red car drove down the empty street . tl ; dr : a fast car drove .
"""


def full_corpus() -> str:
    """拼接通用语料与任务范例，作为预训练输入。"""
    return WEBTEXT_CORPUS + WEBTEXT_TASK_EXAMPLES


# ---------------------------------------------------------------------------
# 语言模型批数据
# ---------------------------------------------------------------------------
def lm_batch(token_ids: List[int], block_size: int, batch_size: int,
             device: torch.device, generator: torch.Generator):
    """从扁平 token id 流中随机采样一个 batch 的语言模型数据。

    返回 (x, y)：x 为 (B, block_size) 输入，y 为 x 右移一位的下一个 token。
    """
    n = len(token_ids)
    idx = torch.randint(0, max(1, n - block_size - 1), (batch_size,), generator=generator)
    x = torch.stack([torch.tensor(token_ids[i:i + block_size], dtype=torch.long) for i in idx])
    y = torch.stack([torch.tensor(token_ids[i + 1:i + 1 + block_size], dtype=torch.long) for i in idx])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# 零样本任务提示模板 (论文核心：把任务写成自然语言续写)
# 每个函数返回提示字符串；模型只需「接着写下去」即完成任务，无需任何参数微调。
# ---------------------------------------------------------------------------
def translate_prompt(text: str, target_lang: str) -> str:
    """翻译任务：translate to french , the cat : le chat  (格式同训练范例)。"""
    return f"translate to {target_lang} , {text} :"


def qa_prompt(question: str) -> str:
    """问答任务：question : ... ? answer : (模型续写答案)。"""
    return f"question : {question} answer :"


def summarize_prompt(text: str) -> str:
    """摘要任务：用论文风格的 'tl ; dr :' 触发模型生成摘要。"""
    return f"{text} tl ; dr :"


def complete_prompt(prefix: str) -> str:
    """通用补全任务：任意前缀续写。"""
    return prefix


# ---------------------------------------------------------------------------
# 零样本评估：给定 (prompt, 期望的续写前缀)，统计 argmax 续写是否命中。
# 这是 LAMBADA / Children's Book Test 这类「预测下一个词」指标的最小化演示 ——
# 真实 GPT-2 在这些基准上靠 WebText 训练得到显著优于随机的零样本表现。
# ---------------------------------------------------------------------------
ZERO_SHOT_EVAL: List[Tuple[str, str]] = [
    (qa_prompt("what is the capital of france ?"), " paris"),
    (qa_prompt("what is the capital of japan ?"), " tokyo"),
    (qa_prompt("what is the capital of italy ?"), " rome"),
    (translate_prompt("the cat", "french"), " le"),
    (translate_prompt("the house", "french"), " la"),
    ("the capital of france is :", " paris"),
    ("the capital of japan is :", " tokyo"),
    ("the sun rises in the", " east"),
    ("honey is sweet and lemon is", " sour"),
    ("a dog is an animal and a cat is an animal", " too"),
]


def split_corpus(token_ids: List[int], frac: float = 0.9) -> Tuple[List[int], List[int]]:
    """按 token 流切分训练/验证：前 frac 用于训练，后 (1-frac) 用于计算困惑度。"""
    k = int(len(token_ids) * frac)
    return token_ids[:k], token_ids[k:]


def seed_everything(seed: int = 42):
    """固定随机种子以保证可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
