"""数据构建：语料、语言模型批数据、以及论文 Figure 2 的 4 种任务输入变换。

特殊 token 约定：
  [Start]   序列起始标记
  [Delim]   分隔两段文本 (前提/假设、上下文/问题等)
  [Extract] 其所在位置的隐藏表示送入任务分类头 (本文件返回该位置的下标)
"""
import random
from typing import List, Tuple

import torch

# ---------------------------------------------------------------------------
# 预训练语料 (用于无监督语言模型预训练)。
# 为可离线快速运行，这里内置一份小型英文语料，含陈述句与评论句，
# 以便与下游情感分类任务共享词汇分布。
# 真实复现需用 BooksCorpus 等大规模语料。
# ---------------------------------------------------------------------------
PRETRAIN_CORPUS = """
the cat sat on the mat and looked at the warm sun .
a young woman walked into the quiet library and borrowed a book .
the weather was cold and rainy all through the long weekend .
she opened the small box and found a silver ring inside .
the old man told stories about the sea to the children .
a fast red car drove down the empty street at midnight .
we had a delicious dinner at the new restaurant last night .
the movie was long and boring and many people left early .
he spent the whole afternoon reading a thick history book .
the garden was full of bright flowers and busy bees in summer .
they climbed the tall mountain and enjoyed the beautiful view .
the teacher explained the difficult problem with great patience .
a small dog ran across the green park chasing a yellow ball .
the train arrived late and the passengers were very tired .
she painted the bedroom walls a soft and calm blue color .
the concert was amazing and the crowd cheered for a long time .
this food is absolutely delicious and i love every bite of it .
the hotel room was dirty and the service was truly terrible .
what a wonderful day to take a long walk in the park .
the book is fascinating and i could not put it down at all .
the coffee tasted bitter and cold after sitting for an hour .
he bought a new laptop and it works fast and really well .
the cake was sweet and moist and everyone asked for more .
the movie plot was confusing and the ending made no sense .
the beach was clean and the water was warm and crystal clear .
the lecture was clear and the examples were very helpful .
a heavy rain flooded the narrow streets of the old town .
she learned to play the piano when she was only six years old .
the team played well and won the final match of the season .
the soup was bland and needed more salt and fresh pepper .
we watched the sunset from the top of the green hill .
the phone battery died quickly and the screen cracked easily .
the museum exhibit was impressive and full of ancient art .
a gentle breeze moved the leaves of the tall oak tree .
the flight was smooth and the crew was very friendly and kind .
the shoes were uncomfortable and hurt my feet after an hour .
the children laughed loudly at the funny clown in the show .
the city lights shone brightly across the dark river at night .
the bread was fresh and soft and smelled like a warm bakery .
he forgot his umbrella and got soaked in the sudden storm .
the novel was thrilling and kept me awake until the morning .
the shopkeeper smiled and gave the child a free candy .
mountains rose sharply against the clear blue morning sky .
the music was loud and the lyrics were hard to understand .
the dessert was rich and creamy and worth every single penny .
the road was icy and several cars slid into the deep ditch .
she found an old photograph of her grandparents in the attic .
the play was witty and the actors delivered a great show .
the rain stopped and a bright rainbow appeared over the field .
the sandwich was stale and the cheese had a strange smell .
he fixed the broken chair with a hammer and some strong nails .
the garden smelled of roses and fresh grass after the rain .
the exam was hard but she had studied for many long weeks .
the puppy was playful and chewed on every slipper it found .
the river flowed quietly past the small wooden fishing boats .
the pizza was hot and cheesy and arrived in just ten minutes .
the hotel breakfast was cold and the juice was clearly sour .
she watered the plants and swept the floor of the sunny room .
the crowd roared as the singer stepped onto the bright stage .
the jacket was thin and offered no warmth in the cold wind .
the children built a tall sandcastle near the calm sea shore .
"""

# ---------------------------------------------------------------------------
# 下游任务演示数据：二分类情感 (label 1 = 正面, 0 = 负面)。
# 同样为可离线运行而内置小数据集；真实复现应在 SST-2 等数据上微调。
# ---------------------------------------------------------------------------
SENTIMENT_DATA: List[Tuple[str, int]] = [
    ("the food was delicious and i loved every bite", 1),
    ("what a wonderful and memorable evening", 1),
    ("this book is fascinating and beautifully written", 1),
    ("the service was excellent and the staff was friendly", 1),
    ("i really enjoyed the show it was amazing", 1),
    ("the hotel was clean and the room was comfortable", 1),
    ("a fantastic film with great acting and a smart plot", 1),
    ("the coffee was rich and warm and perfectly brewed", 1),
    ("she gave a clear and helpful explanation of the topic", 1),
    ("the dessert was sweet creamy and absolutely perfect", 1),
    ("what a lovely sunny day for a walk in the park", 1),
    ("the team played well and deserved to win the match", 1),
    ("the garden was full of bright and beautiful flowers", 1),
    ("the new laptop is fast and works really well", 1),
    ("the music was lovely and the singer had a great voice", 1),
    ("the museum exhibit was impressive and well organized", 1),
    ("the bread was fresh soft and smelled wonderful", 1),
    ("the flight was smooth and the crew was very kind", 1),
    ("the puppy was playful and brought us so much joy", 1),
    ("the pizza arrived hot cheesy and on time", 1),
    ("the staff greeted us with warm smiles and quick service", 1),
    ("the view from the balcony was stunning and worth the trip", 1),
    ("the teacher was patient and made the hard topic simple", 1),
    ("the soup was hearty and full of rich flavor", 1),
    ("she danced gracefully and the crowd cheered loudly", 1),
    ("the new phone has a bright screen and a long battery", 1),
    ("the kids laughed and played happily in the safe yard", 1),
    ("the ride was smooth quiet and very comfortable", 1),
    ("the gift was thoughtful and made her smile all day", 1),
    ("the concert was thrilling and the band sounded superb", 1),
    ("the market was lively and full of fresh produce", 1),
    ("the novel ended with a brilliant and moving twist", 1),
    ("the breakfast was warm tasty and generously served", 1),
    ("the guide was friendly and shared many fun stories", 1),
    ("the cake was moist sweet and beautifully decorated", 1),
    ("the weather was mild and perfect for a long hike", 1),
    ("the repair was quick and the price was very fair", 1),
    ("the puppy was healthy playful and full of energy", 1),
    ("the show was witty clever and truly entertaining", 1),
    ("the room was spacious bright and spotlessly clean", 1),
    ("the meal was cold bland and truly disappointing", 0),
    ("the hotel room was dirty and the service was terrible", 0),
    ("the movie was long boring and made no sense", 0),
    ("the shoes were uncomfortable and hurt my feet", 0),
    ("the coffee tasted bitter and was served cold", 0),
    ("the plot was confusing and the ending was awful", 0),
    ("the soup was bland and needed much more flavor", 0),
    ("the phone battery died quickly and the screen cracked", 0),
    ("the jacket was thin and offered no warmth at all", 0),
    ("the breakfast was cold and the juice was sour", 0),
    ("the road was icy and the drive was dangerous", 0),
    ("the sandwich was stale and the cheese smelled bad", 0),
    ("the lecture was dull and the examples were unclear", 0),
    ("the concert was too loud and the sound was poor", 0),
    ("the shop was messy and the staff was rude to me", 0),
    ("the exam was unfair and far too long for anyone", 0),
    ("the play was boring and the actors forgot their lines", 0),
    ("the cake was dry hard and had no flavor at all", 0),
    ("the rain ruined our trip and we stayed inside all day", 0),
    ("the traffic was awful and we missed the whole flight", 0),
    ("the waiter was slow and got our order completely wrong", 0),
    ("the bed was lumpy and the sheets were torn and dirty", 0),
    ("the price was absurd and the quality was shockingly poor", 0),
    ("the app kept crashing and lost all of my saved work", 0),
    ("the bus was late crowded and smelled terribly of smoke", 0),
    ("the movie ticket was pricey and the film was a letdown", 0),
    ("the meeting dragged on and covered nothing useful at all", 0),
    ("the salad was wilted and the dressing was far too salty", 0),
    ("the heater broke and the room stayed freezing all night", 0),
    ("the delivery was delayed and the package arrived damaged", 0),
    ("the teacher was harsh and refused to answer any questions", 0),
    ("the hike was exhausting and the trail was poorly marked", 0),
    ("the soup was watery and tasted like dirty dish water", 0),
    ("the venue was cramped and the view was fully blocked", 0),
    ("the laptop overheated and shut down during my presentation", 0),
    ("the movie was so dull that several people fell asleep", 0),
    ("the staff ignored us and we waited an hour for a menu", 0),
    ("the brakes were faulty and the car felt unsafe to drive", 0),
    ("the hotel charged extra and the wifi never once worked", 0),
    ("the trip was a disaster and nothing went as we planned", 0),
]


# ---------------------------------------------------------------------------
# 预训练语言模型批数据
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
# 论文 Figure 2：四种下游任务的输入变换
# 每个函数返回 (ids, extract_index)，extract_index 指明任务头读取的位置。
# ---------------------------------------------------------------------------
def _special(tok, name: str) -> int:
    return tok.special_to_id[name]


def classification_input(tok, text: str) -> Tuple[List[int], int]:
    """分类任务：[Start] text [Extract] -> 用 [Extract] 位置做分类。"""
    ids = [_special(tok, "[Start]")] + tok.encode(text) + [_special(tok, "[Extract]")]
    return ids, len(ids) - 1


def entailment_input(tok, premise: str, hypothesis: str) -> Tuple[List[int], int]:
    """文本蕴含：[Start] 前提 [Delim] 假设 [Extract] -> 用 [Extract] 位置分类 (3 类)。"""
    ids = ([_special(tok, "[Start]")] + tok.encode(premise) + [_special(tok, "[Delim]")]
           + tok.encode(hypothesis) + [_special(tok, "[Extract]")])
    return ids, len(ids) - 1


def similarity_inputs(tok, sent1: str, sent2: str):
    """语义相似度：拼接两种顺序，各自过模型后求和 (对称化)，再做回归/分类。

    返回两个 (ids, extract_index)，使用方分别前向后相加。
    """
    return [
        entailment_input(tok, sent1, sent2),
        entailment_input(tok, sent2, sent1),
    ]


def multiple_choice_input(tok, context: str, question: str, answers: List[str]):
    """多项选择 (阅读理解)：对每个候选答案构造一条序列，分别打分后 softmax。

        [Start] context [Delim] question + answer_k [Extract]
    返回 answers 条 (ids, extract_index)。
    """
    seqs = []
    for ans in answers:
        body = tok.encode(question + " " + ans)
        ids = ([_special(tok, "[Start]")] + tok.encode(context) + [_special(tok, "[Delim]")]
               + body + [_special(tok, "[Extract]")])
        seqs.append((ids, len(ids) - 1))
    return seqs


# ---------------------------------------------------------------------------
# 分类任务的批整理 (padding 到 n_ctx，并记录有效位置与 [Extract] 下标)
# ---------------------------------------------------------------------------
def collate_classification(samples, tok, n_ctx: int):
    """把 (text, label) 样本整理为一个分类 batch。

    返回 x (B, n_ctx) padding 后的 id、extract_pos (B,)、labels (B,)、valid (B, n_ctx)。
    """
    pad = tok.pad_id
    seqs, positions, labels = [], [], []
    for text, label in samples:
        ids, ext = classification_input(tok, text)
        ids = ids[:n_ctx]
        ext = min(ext, n_ctx - 1)
        seqs.append(ids)
        positions.append(ext)
        labels.append(label)
    x = torch.full((len(seqs), n_ctx), pad, dtype=torch.long)
    valid = torch.zeros((len(seqs), n_ctx), dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        valid[i, :len(s)] = True
    return (x, torch.tensor(positions, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long), valid)


def split_data(data, frac=0.75, seed=42):
    """按比例划分训练/验证集 (固定随机种子保证可复现)。"""
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)
    k = int(len(shuffled) * frac)
    return shuffled[:k], shuffled[k:]
