"""字节对编码 (Byte Pair Encoding, BPE) 分词器。

GPT-1 使用 BPE 子词分词。本文件实现一个自包含的最小 BPE，无需任何外部依赖：
  1. 把语料按空格预切分，词内拆成字符并附加结束标记 '</w>'；
  2. 反复合并语料中出现频率最高的相邻符号对，直到达到目标词表大小；
  3. 编码时对每个词用「优先级最高 (合并序号最小) 的相邻对」贪心合并。
特殊 token ([Pad]/[Start]/[Delim]/[Extract]) 追加到词表末尾，占用固定 id。
任何未见过的词都能通过字符级回退编码，天然支持 OOV。
"""
from collections import Counter
from typing import Dict, List

WORD_END = "</w>"                      # 标记词尾，区分词内与词间边界
SPECIALS: List[str] = ["[Pad]", "[Start]", "[Delim]", "[Extract]"]


def _word_freqs(corpus: str) -> Counter:
    """按空格切词并简单预处理 (小写、字母数字)，统计词频。"""
    freqs: Counter = Counter()
    for raw in corpus.split():
        w = "".join(c.lower() if c.isalnum() else " " for c in raw).replace(" ", "")
        if w:
            freqs[w] += 1
    return freqs


class BPETokenizer:
    """可训练的 BPE 分词器。"""

    def __init__(self):
        self.vocab: Dict[str, int] = {}        # 符号 -> id
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[tuple] = []          # 学习到的合并顺序 (按优先级)
        self.bpe_ranks: Dict[tuple, int] = {}
        self.special_to_id: Dict[str, int] = {}
        self._cache: Dict[str, List[str]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_to_id)

    @property
    def pad_id(self) -> int:
        return self.special_to_id["[Pad]"]

    def train(self, corpus: str, target_size: int = 400):
        """在给定语料上训练 BPE 直到词表达到 target_size (含特殊 token)。"""
        word_freqs = _word_freqs(corpus)
        # vocab: word -> (子词符号列表, 词频)
        vocab = {w: (list(w) + [WORD_END], f) for w, f in word_freqs.items()}
        # 基础词表 = 所有出现的字符 + 词尾标记
        base_symbols = set()
        for symbols, _ in vocab.values():
            base_symbols.update(symbols)
        base_symbols = sorted(base_symbols)

        # 迭代合并频率最高的相邻对，直到达到目标数量
        target_merges = max(0, target_size - len(SPECIALS) - len(base_symbols))
        while len(self.merges) < target_merges:
            pairs: Counter = Counter()
            for symbols, freq in vocab.values():
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = _merge_word(best, vocab)
            self.merges.append(best)

        # 构建最终词表：基础字符 + 合并出的子词 + 特殊 token
        tokens: List[str] = list(base_symbols)
        for a, b in self.merges:
            tokens.append(a + b)
        for i, tok in enumerate(SPECIALS):
            self.special_to_id[tok] = len(tokens) + i
        self.vocab = {t: i for i, t in enumerate(tokens)}
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.id_to_token.update({i: t for t, i in self.special_to_id.items()})
        self._cache.clear()

    def _encode_word(self, word: str) -> List[str]:
        """对单个词应用 BPE 合并，返回子词符号列表。"""
        if word in self._cache:
            return self._cache[word]
        symbols = list(word) + [WORD_END]
        while len(symbols) > 1:
            pairs = {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:          # 无可合并对则停止
                break
            symbols = _merge_symbol(best, symbols)
        self._cache[word] = symbols
        return symbols

    def encode(self, text: str) -> List[int]:
        """把文本编码为 id 列表；特殊 token 字符串优先识别。"""
        ids: List[int] = []
        for raw in text.split():
            if raw in self.special_to_id:           # 显式给出的特殊 token
                ids.append(self.special_to_id[raw])
                continue
            w = "".join(c.lower() if c.isalnum() else " " for c in raw).replace(" ", "")
            if not w:
                continue
            for sym in self._encode_word(w):
                ids.append(self.vocab[sym])
        return ids

    def decode(self, ids: List[int]) -> str:
        """把 id 列表还原为文本 (特殊 token 以其字符串形式显示)。"""
        out: List[str] = []
        for i in ids:
            tok = self.id_to_token.get(i, "<unk>")
            if i in self.special_to_id.values():
                out.append(tok)
            elif tok.endswith(WORD_END):
                out.append(tok[: -len(WORD_END)] + " ")
            else:
                out[-1] = out[-1] + tok if out else tok
        return "".join(out).strip()


def _merge_word(pair: tuple, vocab: Dict[str, tuple]) -> Dict[str, tuple]:
    """在全部词内合并某个相邻对，返回新的 vocab。"""
    bigram = pair[0] + pair[1]
    new_vocab = {}
    for word, (symbols, freq) in vocab.items():
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(bigram)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        new_vocab[word] = (merged, freq)
    return new_vocab


def _merge_symbol(pair: tuple, symbols: List[str]) -> List[str]:
    """在单个词的符号列表上合并所有出现的 pair。"""
    bigram, out, i = pair[0] + pair[1], [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(bigram)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out
