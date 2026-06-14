"""
文本预处理 (Text Preprocessing) — 参考 d2l §8.2
https://zh-v2.d2l.ai/chapter_recurrent-neural-networks/text-preprocessing.html

演示内容:
1. 下载《时光机器》语料 (H.G. Wells, The Time Machine)
2. 分词: 文本 → 按行切 → 小写 → 正则拆词 (字母序列) → token 流
3. 构建词表: Counter 统计 → 按频次排序 → <unk>/pad/bos/eos → token↔index 双射
4. 编码: 把任意一行文本 → 用 Vocab 转成整数索引 (→ torch.Tensor)
5. 探索词频分布: Zipf 律 —— 第 r 个高频词的频次 ≈ c / r

全部使用标准库 + requests + torch, 不用 d2l / torchtext。
"""

import collections
import re
import requests
import torch

URL = "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
CACHE = "data/timemachine.txt"


def read_time_machine():
    import os
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
        self._freqs = sorted(
            counter.items(), key=lambda x: x[0])
        self._freqs.sort(key=lambda x: x[1], reverse=True)
        self.unk = 0
        uniq = ["<unk>"] + reserved_tokens
        uniq += [tok for tok, freq in self._freqs if freq >= min_freq and tok not in uniq]
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


def summarize(lines):
    n_lines = len(lines)
    n_nonempty = sum(1 for l in lines if l)
    tokens = tokenize(lines)
    n_tokens = sum(len(l) for l in tokens)
    vocab = Vocab(tokens, min_freq=0, reserved_tokens=["<pad>"])
    return dict(n_lines=n_lines, n_nonempty=n_nonempty,
                n_tokens=n_tokens, vocab=vocab, tokens=tokens)


def main():
    print("=" * 60)
    print("Part 1: 读取语料 → 按行清洗")
    print("=" * 60)
    lines = read_time_machine()
    print(f"总行数: {len(lines)}, 非空行: {sum(1 for l in lines if l)}")
    print(f"前 3 行原文 (清洗后):")
    for i, l in enumerate(lines[:20]):
        if l:
            print(f"  [{i}] {l[:70]}")
            if i >= 3:
                break

    print("\n" + "=" * 60)
    print("Part 2: 分词 (word 级)")
    print("=" * 60)
    tokens = tokenize(lines, token="word")
    print(f"前两行的 token:")
    for i in range(2):
        print(f"  line {i}: {tokens[i][:12]}{' ...' if len(tokens[i]) > 12 else ''}")
    chars = tokenize(lines, token="char")
    print(f"\n对比 char 级 (第 0 行前 20 字符): {chars[0][:20]}")

    print("\n" + "=" * 60)
    print("Part 3: 构建词表")
    print("=" * 60)
    vocab = Vocab(tokens, min_freq=0, reserved_tokens=["<pad>"])
    print(f"词表大小 |V| = {len(vocab)}")
    top10 = list(vocab.freqs.items())[:10]
    print(f"Top-10 高频词 (词, 频次):")
    for w, c in top10:
        print(f"  {w:>10} : {c}")

    print("\n" + "=" * 60)
    print("Part 4: 编码 token → index → tensor")
    print("=" * 60)
    for w in ["the", "time", "machine", "zzzzzunk"]:
        print(f"  '{w}' → index {vocab[w]}  (回译: '{vocab.to_tokens(vocab[w])}')")
    sample = tokens[10][:8]
    idx = vocab[sample]
    t = torch.tensor(idx)
    print(f"\n一句话前 8 个 token: {sample}")
    print(f"  indices: {idx}")
    print(f"  tensor:  {t}  dtype={t.dtype}")
    print(f"  decode:  {vocab.to_tokens(idx)}")

    print("\n" + "=" * 60)
    print("Part 5: 词频分布 (Zipf 律)")
    print("=" * 60)
    freqs = sorted(vocab.freqs.values(), reverse=True)
    print(f"{'rank':>6} {'word':>12} {'freq':>8} {'c=freq*rank':>12}  {'f_r * r^a':>10}")
    alpha = 1.0
    for r in [1, 2, 3, 10, 100, 1000, len(freqs)]:
        if r - 1 >= len(freqs):
            continue
        w = list(vocab.freqs.keys())[r - 1]
        f = freqs[r - 1]
        print(f"{r:>6} {w:>12} {f:>8} {f * r:>12.1f}  {f * r ** alpha:>10.1f}")
    ratio = freqs[0] / freqs[-1]
    print(f"\n最高频/最低频 ≈ {ratio:.0f}×  (Zipf: c/r 衰减, 长尾显著)")

    return dict(lines=lines, tokens=tokens, vocab=vocab)


if __name__ == "__main__":
    main()
