"""字节级 BPE (Byte-level Byte Pair Encoding) 分词器。

GPT-2 放弃了 GPT-1 的字符/词级 BPE，改用字节级 BPE，核心好处：
  * 任何文本 (任意 Unicode、任意语言、甚至二进制噪声) 都能编码，永远没有 OOV / <unk>。
  * 词表以 256 个字节为基底，再在其上学习合并，规模可灵活扩展 (GPT-2 实际为 50257)。

实现要点 (对齐 OpenAI gpt-2/src/encoder.py)：
  1. bytes_to_unicode(): 把 0-255 字节映射到可见 Unicode 字符 —— 可见字节映射为自身，
     不可见/控制字节映射到 256 以后 (其中空格 0x20 → 'Ġ' U+0120)，避免与空白/控制符混淆。
  2. 正则预切分 (GPT-2 标准 pattern)：把文本切成 词/数字/标点/空白 等片段，空格附着到后词。
  3. 先把每片段用 UTF-8 编码成字节、再逐字节映射成 unicode 字符，在该表示上做 BPE。
  4. 训练时反复合并语料中出现频率最高的相邻符号对，直到达到目标词表大小。
  5. 编码时对每片段的符号序列按「合并优先级最高 (rank 最小)」贪心合并。
特殊 token <|endoftext|> 作为文档边界 / 序列结束符追加到词表末尾。
"""
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Tuple

# GPT-2 预切分正则。优先用第三方 regex 模块 (精确支持 \p{L}/\p{N} 等 Unicode 类别，
# 与 OpenAI 官方完全一致)；若未安装则退化为 Python 标准库 re 的 Unicode 等价写法
# ([^\W\d_] 近似 \p{L}、\d 近似 \p{N})，足以覆盖教学/一般英文场景。
_GPT2_REGEX = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_FALLBACK_REGEX = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""

try:
    import regex as _re_mod                                    # noqa: F401  (偏好 regex)
    _GPT2_PATTERN = _re_mod.compile(_GPT2_REGEX)
    _HAS_REGEX = True
except ImportError:                                            # 退化为标准库 re
    import re as _re_mod
    _GPT2_PATTERN = _re_mod.compile(_FALLBACK_REGEX)
    _HAS_REGEX = False


@lru_cache()
def bytes_to_unicode() -> Dict[int, str]:
    """把 0-255 字节映射为可见 Unicode 字符 (OpenAI gpt-2 标准实现)。

    可打印 ASCII 与部分拉丁字符保持不变；其余 (含空格、控制符) 映射到 U+0100 之后，
    其中空格 0x20 恰好映射为 'Ġ' (U+0120)，这正是 GPT-2 词表中词前导空格的来源。
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def get_pairs(symbols: List[str]) -> set:
    """返回符号序列中所有相邻对。"""
    return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}


_BYTE_ENCODER = bytes_to_unicode()                             # int 字节 -> unicode 字符
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}       # unicode 字符 -> int 字节

ENDOFTEXT = "<|endoftext|>"                                    # GPT-2 的唯一特殊 token


class ByteBPETokenizer:
    """字节级 BPE 分词器 (可训练 / 自包含 / 无外部数据依赖)。"""

    def __init__(self):
        self.byte_encoder = _BYTE_ENCODER
        self.byte_decoder = _BYTE_DECODER
        self.encoder: Dict[str, int] = {}                      # token 字符串 -> id
        self.decoder: Dict[int, str] = {}
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}        # 合并对 -> 优先级 (越小越先合并)
        self.n_merges: int = 0
        self.endoftext_id: int = -1
        self._cache: Dict[str, str] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    @property
    def has_regex(self) -> bool:
        """是否加载了精确的 regex 模块 (False 表示在用 re 近似切分)。"""
        return _HAS_REGEX

    # ------------------------------------------------------------------ 训练
    def train(self, corpus: str, vocab_size: int = 500):
        """在语料上训练 byte-level BPE，直到词表达到 vocab_size。

        词表构成 = 256 字节基字符 + 学习到的合并 token + 1 个 <|endoftext|>。
        (GPT-2 实际词表 50257 = 256 字节 + 50000 合并 + 1 特殊；本实现可按需缩放。)
        """
        # 1) 正则预切分语料
        chunks = _GPT2_PATTERN.findall(corpus)
        # 2) 每 chunk -> UTF-8 字节 -> unicode 字符串 (作为 BPE 训练的「词」)
        word_freqs: Counter = Counter()
        for chunk in chunks:
            byte_str = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            word_freqs[byte_str] += 1

        base_symbols = list(self.byte_encoder.values())        # 256 个字节字符 (固定基底)
        vocab = {w: (list(w), f) for w, f in word_freqs.items()}   # 词 -> (符号列表, 词频)

        target_merges = max(0, vocab_size - 256 - 1)           # 留出字节基底与特殊 token
        merges: List[Tuple[str, str]] = []
        while len(merges) < target_merges:
            pairs: Counter = Counter()
            for symbols, freq in vocab.values():
                for p in get_pairs(symbols):
                    pairs[p] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)                   # 频率最高的相邻对
            vocab = self._merge_word(best, vocab)
            merges.append(best)
        self.n_merges = len(merges)

        tokens = list(base_symbols) + [a + b for a, b in merges]
        tokens.append(ENDOFTEXT)
        self.encoder = {t: i for i, t in enumerate(tokens)}
        self.decoder = {i: t for t, i in self.encoder.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.endoftext_id = self.encoder[ENDOFTEXT]
        self._cache.clear()

    # --------------------------------------------------------------- BPE 应用
    def _bpe(self, token: str) -> str:
        """对单个字节串应用 BPE 合并，返回以空格分隔的子词 token 串。"""
        if token in self._cache:
            return self._cache[token]
        symbols = list(token)
        while len(symbols) > 1:
            pairs = get_pairs(symbols)
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:                     # 无可合并对则停止
                break
            symbols = self._merge_symbols(best, symbols)
        out = " ".join(symbols)
        self._cache[token] = out
        return out

    # --------------------------------------------------------------- 编码/解码
    def encode(self, text: str) -> List[int]:
        """文本 -> id 列表。正则切分 → 字节表示 → BPE 合并 → 查词表。"""
        ids: List[int] = []
        for chunk in _GPT2_PATTERN.findall(text):
            byte_str = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            for sym in self._bpe(byte_str).split(" "):
                ids.append(self.encoder[sym])
        return ids

    def decode(self, ids: List[int]) -> str:
        """id 列表 -> 文本。token 还原为字节 unicode 字符，再按 UTF-8 解码。"""
        chars: List[str] = []
        for i in ids:
            tok = self.decoder.get(i, "")
            if tok == ENDOFTEXT:
                continue
            chars.append(tok)
        text = "".join(chars)
        byte_array = bytes(self.byte_decoder[c] for c in text if c in self.byte_decoder)
        return byte_array.decode("utf-8", errors="replace")

    # --------------------------------------------------------------- 合并工具
    @staticmethod
    def _merge_word(pair: Tuple[str, str], vocab: Dict[str, tuple]) -> Dict[str, tuple]:
        """在所有词内合并某个相邻对，返回新 vocab (训练用)。"""
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

    @staticmethod
    def _merge_symbols(pair: Tuple[str, str], symbols: List[str]) -> List[str]:
        """在单个符号列表上合并所有出现的 pair (编码用)。"""
        bigram, out, i = pair[0] + pair[1], [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                out.append(bigram)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out
