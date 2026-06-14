"""
语言模型和数据集 (Language Models and Datasets) — 参考 d2l §8.3
https://zh-v2.d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html

演示内容:
1. 复用 §8.2 的语料加载 / 分词 / 词表, 把《时光机器》整本变成 token 索引流 corpus
2. n-gram 频率统计: unigram / bigram / trigram —— 看马尔可夫假设如何把
   P(x_t | x_{<t}) 简化为 P(x_t | x_{t-n+1}, ..., x_{t-1})
3. 语言模型评估: 交叉熵 / 困惑度 perplexity = exp(平均负对数似然)
4. Zipf 律在 n-gram 上依旧成立 (n 越大, 长尾越陡)
5. 序列采样:
   - 随机采样 seq_data_iter_random —— 每个小批的子序列起点随机, 适合 RNN
   - 顺序分区 seq_data_iter_sequential —— 起点固定步长, 小批间首尾相接
6. d2l 的 SeqDataLoader: 把整本小说切成 num_steps 的子序列, 供 RNN 训练

全部使用标准库 + torch, 不用 d2l / torchtext。
词表实现沿用 20_text_preprocessing.py 的 Vocab, 保证两节衔接。
"""

import os
import re
import math
import collections
import random
import requests
import torch

URL = "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
CACHE = "data/timemachine.txt"


# ──────────────────────────────────────────────────────
# §8.2 复用: 读语料 + 分词 + 词表 (与 20 节完全一致)
# ──────────────────────────────────────────────────────
def read_time_machine():
    if not os.path.exists(CACHE):
        with open(CACHE, "w", encoding="utf-8") as f:
            f.write(requests.get(URL, timeout=30).text)
    with open(CACHE, encoding="utf-8") as f:
        lines = f.readlines()
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]


def tokenize(lines, token="word"):
    if token == "word":
        return [line.split() for line in lines]
    if token == "char":
        return [list(line) for line in lines]
    raise ValueError(f"未知 token 类型: {token}")


def count_corpus(tokens):
    return collections.Counter(tok for line in tokens for tok in line)


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        tokens = tokens or []
        reserved_tokens = reserved_tokens or []
        counter = count_corpus(tokens)
        self._freqs = sorted(counter.items(), key=lambda x: x[0])
        self._freqs.sort(key=lambda x: x[1], reverse=True)
        self.unk = 0
        uniq = ["<unk>"] + reserved_tokens
        uniq += [tok for tok, freq in self._freqs
                 if freq >= min_freq and tok not in uniq]
        self.idx_to_token, self.token_to_idx = [], {}
        for tok in uniq:
            self.idx_to_token.append(tok)
            self.token_to_idx[tok] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(t) for t in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[i] for i in indices]

    @property
    def freqs(self):
        return dict(self._freqs)


def load_corpus(max_tokens=-1):
    """整本《时光机器》→ 一条扁平的 token 索引流 corpus (1D LongTensor)。"""
    lines = read_time_machine()
    tokens = tokenize(lines, token="word")
    vocab = Vocab(tokens, min_freq=0, reserved_tokens=["<pad>"])
    flat = [tok for line in tokens for tok in line]
    if max_tokens > 0:
        flat = flat[:max_tokens]
    corpus = torch.tensor([vocab[tok] for tok in flat], dtype=torch.long)
    return corpus, vocab


# ──────────────────────────────────────────────────────
# §8.3 核心: n-gram 与语言模型
# ──────────────────────────────────────────────────────
def ngram_counts(corpus, n):
    """在 token 索引流上数 n-gram 频次, 返回 Counter。"""
    seq = corpus.tolist()
    return collections.Counter(tuple(seq[i:i + n])
                               for i in range(len(seq) - n + 1))


def top_ngrams(counter, vocab, k=10, decode=True):
    """取频次前 k 的 n-gram, 可选地把索引回译成 token。"""
    out = []
    for ng, c in counter.most_common(k):
        if decode:
            ng = " ".join(vocab.to_tokens(list(ng)))
        out.append((ng, c))
    return out


def zipf_products(freqs_sorted):
    """给定降序频次列表, 返回每个 rank 的 f·r, 用于验证 Zipf 律。"""
    return [f * (r + 1) for r, f in enumerate(freqs_sorted)]


# ──────────────────────────────────────────────────────
# §8.3 评估: 用 unigram MLE 给句子算困惑度 (玩具版, 直观理解 LM 目标)
# ──────────────────────────────────────────────────────
def unigram_loglikelihood(corpus, vocab, sentence):
    """
    用经验频率 P(w) = count(w)/N 作 MLE, 给一句话算负对数似然。
    真实 LM 用条件分布 P(w_t|w_<t), 这里用 unigram 近似只为直观演示。
    """
    counter = collections.Counter(corpus.tolist())
    N = len(corpus)
    words = re.sub("[^A-Za-z]+", " ", sentence).strip().lower().split()
    idxs = [vocab[w] for w in words]
    nll = 0.0
    for w, i in zip(words, idxs):
        p = max(counter.get(i, 0) / N, 1e-12)
        nll += -math.log(p)
    ppl = math.exp(nll / max(len(idxs), 1))
    return nll, ppl, list(zip(words, idxs))


# ──────────────────────────────────────────────────────
# §8.3 采样: 把长序列切成 num_steps 的子序列小批
# ──────────────────────────────────────────────────────
def seq_data_iter_random(corpus, batch_size, num_steps):
    """随机采样: 每个小批从随机起点截取 num_steps 长的子序列。"""
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial)
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        idxs = initial[i:i + batch_size]
        X = torch.stack([corpus[j:j + num_steps] for j in idxs])
        Y = torch.stack([corpus[j + 1:j + 1 + num_steps] for j in idxs])
        yield X, Y


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """顺序分区: 起点固定步长, 小批之间首尾相接, 隐状态可跨批传递。"""
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = corpus[offset:offset + num_tokens].reshape(batch_size, -1)
    Ys = corpus[offset + 1:offset + 1 + num_tokens].reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


class SeqDataLoader:
    """d2l 的数据加载器: 把整本小说切成子序列, 支持 random / sequential。"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        self.use_random_iter = use_random_iter
        self.corpus, self.vocab = load_corpus(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        fn = (seq_data_iter_random if self.use_random_iter
              else seq_data_iter_sequential)
        return fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=-1):
    """返回 (data_iter, vocab), 接口对齐 d2l.load_data_time_machine。"""
    data_iter = SeqDataLoader(batch_size, num_steps,
                              use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


# ──────────────────────────────────────────────────────
# main: 把每个概念分段打印, 方便对照 HTML
# ──────────────────────────────────────────────────────
def main():
    random.seed(42); torch.manual_seed(42)
    corpus, vocab = load_corpus()
    print("=" * 60)
    print("Part 1: 语料 → token 索引流 corpus")
    print("=" * 60)
    print(f"词表大小 |V| = {len(vocab)}")
    print(f"corpus 总 token 数 N = {len(corpus)}")
    print(f"corpus 前 20 个索引: {corpus[:20].tolist()}")
    print(f"回译: {' '.join(vocab.to_tokens(corpus[:20].tolist()))}")

    print("\n" + "=" * 60)
    print("Part 2: 链式法则 → 马尔可夫假设 → n-gram")
    print("=" * 60)
    print("P(x_1,...,x_T) = Π P(x_t | x_1,...,x_{t-1})   [链式法则]")
    print("n-gram 假设:    P(x_t | x_<t) ≈ P(x_t | x_{t-n+1},...,x_{t-1})")
    for n in (1, 2, 3):
        cg = ngram_counts(corpus, n)
        print(f"\n{n}-gram: 唯一 {n}-gram 数 = {len(cg)}")
        for ng, c in top_ngrams(cg, vocab, 5):
            print(f"  {c:>5}  {ng!r}")

    print("\n" + "=" * 60)
    print("Part 3: Zipf 律在 n-gram 上的衰减")
    print("=" * 60)
    print(f"{'n':>3} {'rank1':>8} {'rank10':>8} {'rank100':>8} "
          f"{'rank1000':>9} {'高低比':>9}")
    for n in (1, 2, 3):
        cg = ngram_counts(corpus, n)
        freqs = sorted(cg.values(), reverse=True)
        r1 = freqs[0]
        r10 = freqs[min(9, len(freqs) - 1)]
        r100 = freqs[min(99, len(freqs) - 1)]
        r1000 = freqs[min(999, len(freqs) - 1)] if len(freqs) > 999 else freqs[-1]
        print(f"{n:>3} {r1:>8} {r10:>8} {r100:>8} {r1000:>9} "
              f"{r1 / freqs[-1]:>9.0f}×")

    print("\n" + "=" * 60)
    print("Part 4: 语言模型评估 — 负对数似然 + 困惑度 perplexity")
    print("=" * 60)
    print("困惑度 ppl = exp( -(1/N) Σ log P(x_t|...) )  "
          "| ppl=k ≈ '每次在 k 个候选词里犹豫'")
    for s in ["the time machine",
              "the time traveller",
              "the the the the",
              "zzzzq xxxq wkj"]:
        nll, ppl, pairs = unigram_loglikelihood(corpus, vocab, s)
        pairs_str = " ".join(f"{w}→{i}" for w, i in pairs)
        print(f"  {s!r:30} NLL={nll:6.2f}  ppl={ppl:7.2f}   [{pairs_str}]")

    print("\n" + "=" * 60)
    print("Part 5: 序列采样 (num_steps=5, batch_size=2)")
    print("=" * 60)
    sub = corpus[:60]
    print(f"corpus 前 60 个索引 (将用于采样演示):\n  {sub.tolist()}")

    print("\n[随机采样 random_iter] 每个小批起点随机:")
    for b, (X, Y) in enumerate(
            seq_data_iter_random(sub.clone(), 2, 5)):
        print(f"  batch {b}: X={X.tolist()}")
        print(f"           Y={Y.tolist()}  (Y = X 右移一位)")
        if b >= 1:
            break

    print("\n[顺序分区 sequential_iter] 小批首尾相接:")
    for b, (X, Y) in enumerate(
            seq_data_iter_sequential(sub.clone(), 2, 5)):
        print(f"  batch {b}: X={X.tolist()}")
        print(f"           Y={Y.tolist()}")
        if b >= 1:
            break

    print("\n" + "=" * 60)
    print("Part 6: SeqDataLoader — RNN 训练用的标准接口")
    print("=" * 60)
    data_iter, v = load_data_time_machine(
        batch_size=2, num_steps=5, use_random_iter=False, max_tokens=1000)
    print(f"vocab size = {len(v)}, corpus = {len(data_iter.corpus)} tokens "
          f"(截断到 max_tokens=1000)")
    X, Y = next(iter(data_iter))
    print(f"一个小批: X.shape={tuple(X.shape)}  Y.shape={tuple(Y.shape)}")
    print(f"  X (子序列) = {X.tolist()}")
    print(f"  Y (右移一位, 即下一 token) = {Y.tolist()}")
    print("  → RNN 读 X 预测 Y, 这就是语言模型的训练目标。")

    print("\n" + "=" * 60)
    print("Part 7: Zipf 律验证 (unigram, f·r ≈ 常数)")
    print("=" * 60)
    cg1 = ngram_counts(corpus, 1)
    prods = zipf_products(sorted(cg1.values(), reverse=True))
    print(f"  f(1)·1 = {prods[0]:.0f}")
    print(f"  f(10)·10 = {prods[9]:.0f}")
    print(f"  f(100)·100 = {prods[99]:.0f}")
    print(f"  f(1000)·1000 = {prods[999]:.0f}")
    print(f"  → f·r 在宽范围内近似常数, 即 Zipf f(r) ∝ 1/r")
    print("\n结论: 长尾 → 截断 → <unk> → n-gram 越大稀疏越严重 → "
          "这正是 RNN 要解决的根本动机。")


if __name__ == "__main__":
    main()
