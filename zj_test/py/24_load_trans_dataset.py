"""
机器翻译与数据集 (Machine Translation and Dataset) — 参考 d2l §9.5
https://zh-v2.d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html

演示内容:
1. 读取"英-中" (Tatoeba cmn-eng) 数据集 —— 每行是 `英文\t中文\t署名`, 需去掉署名
2. 预处理: 不间断空格→普通空格、小写、在英文标点前插空格 (与 §9.5 一致)
3. 词元化:
   - 源语言 (英文): 词级, 按空格切 (Hi. → hi .)
   - 目标语言 (中文): 字级, 按字符切 (你好。 → 你 好 。)
     —— 中文没有词边界, 词级切分会把整句当成一个 token, 字级才是合理选择
     (这正是 §9.5 练习 2 的问题: 对中文/日文, 词级词元化不是好主意)
4. 序列长度分布直方图 (源 vs 目标)
5. 词表: 源 / 目标各一份, 低频词 (<2 次) 归并到 <unk>, 预留 <pad>/<bos>/<eos>
6. 截断与填充 truncate_pad: 统一每个样本到 num_steps 长
7. 构建小批量 build_array_nmt: 末尾加 <eos>, 截断/填充, 记录有效长度
8. load_data_nmt: 返回 (数据迭代器, 源词表, 目标词表), 接口对齐 d2l

数据: data/cmn-eng/cmn.txt (32000+ 句对，来源：http://www.manythings.org/anki/)。
全部使用标准库 + torch + matplotlib, 不用 d2l。
词表实现沿用 20_text_preprocessing.py 的 Vocab, 保证各节衔接。
"""

import collections
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # 非交互后端, 避免弹窗/阻塞
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────
# 数据路径: 优先项目根的相对路径 (从根运行), 否则回退到脚本相对路径
# ──────────────────────────────────────────────────────
def _resolve_data_path():
    rel = Path("data/cmn-eng/cmn.txt")
    if rel.exists():
        return rel
    # 脚本位于 <root>/zj_test/py/, 往上三级回到项目根
    return Path(__file__).resolve().parent.parent.parent / "data" / "cmn-eng" / "cmn.txt"


DATA_PATH = str(_resolve_data_path())


# ──────────────────────────────────────────────────────
# 1. 读取数据集 (去署名)
# ──────────────────────────────────────────────────────
def read_data_nmt(fname=DATA_PATH):
    """
    读取"英-中"数据集, 返回以换行拼接的原始文本。

    cmn.txt 每行形如:  Hi.\t嗨。\tCC-BY 2.0 (France) Attribution: ...
    第三个制表符字段是 tatoeba 的署名信息, 与翻译无关, 这里只保留前两个字段。
    """
    with open(fname, encoding="utf-8") as f:
        raw = f.read()
    lines = []
    for line in raw.split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:  # 过滤空行 / 异常行, 只留 "英文\t中文"
            lines.append("\t".join(parts[:2]))
    return "\n".join(lines)


# ──────────────────────────────────────────────────────
# 2. 预处理
# ──────────────────────────────────────────────────────
def preprocess_nmt(text):
    """
    预处理"英-中"数据集 (与 d2l §9.5 一致):
    - 不间断空格 (\\u202f / \\xa0) → 普通空格
    - 全部转小写
    - 在英文标点 ,.!? 前插入空格 (便于英文按空格分词)
    对中文字符基本是 no-op, 但能让英文源正确分词。
    """
    def no_space(char, prev_char):
        return char in set(",.!?") and prev_char != " "

    text = text.replace(" ", " ").replace("\xa0", " ").lower()
    out = [" " + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return "".join(out)


# ──────────────────────────────────────────────────────
# 3. 词元化
# ──────────────────────────────────────────────────────
def tokenize_nmt(text, num_examples=None):
    """
    词元化"英-中"数据集, 返回 (source, target) 两个 token 列表的列表。

    source[i]: 第 i 句英文的词元 (词级, 按空格切)
    target[i]: 第 i 句中文的词元 (字级, 按字符切)
    中文没有词边界, 故目标端用字级 —— 否则整句会塌成一个 token。
    """
    source, target = [], []
    for i, line in enumerate(text.split("\n")):
        if num_examples and i > num_examples:
            break
        parts = line.split("\t")
        if len(parts) == 2:
            source.append(parts[0].split(" "))   # 英文: 词级
            target.append(list(parts[1]))        # 中文: 字级
    return source, target


# ──────────────────────────────────────────────────────
# 词表 (与 20_text_preprocessing.py 一致, 此处自带以保持本文件自洽)
# ──────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────
# 4. 序列长度分布直方图
# ──────────────────────────────────────────────────────
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist, save_to=None):
    """绘制源 / 目标 token 数分布直方图 (对齐 d2l.show_list_len_pair_hist)。"""
    _, _, patches = plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]], bins=30)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch("/")
    plt.legend(legend)
    if save_to:
        plt.savefig(save_to, dpi=100, bbox_inches="tight")
        print(f"  直方图已保存: {save_to}")
    plt.close()


# ──────────────────────────────────────────────────────
# 5. 截断 / 填充
# ──────────────────────────────────────────────────────
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列到固定长度 num_steps。"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


# ──────────────────────────────────────────────────────
# 6. 文本序列 → 小批量
# ──────────────────────────────────────────────────────
def build_array_nmt(lines, vocab, num_steps):
    """
    把 token 列表转成 (array, valid_len):
    - 逐句转索引, 末尾追加 <eos>
    - 截断/填充到 num_steps
    - valid_len 统计非 <pad> 的有效长度 (后续模型会用到)
    """
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab["<eos>"]] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab["<pad>"]) for l in lines])
    valid_len = (array != vocab["<pad>"]).type(torch.int32).sum(1)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    """构造 PyTorch 数据迭代器 (对齐 d2l.load_array)。"""
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


# ──────────────────────────────────────────────────────
# 7. 汇总入口: 返回迭代器 + 源/目标词表
# ──────────────────────────────────────────────────────
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """
    返回翻译数据集的迭代器和两种词表 (对齐 d2l.load_data_nmt)。

    num_examples: 只取前 N 个句对 (None=全部)。文章默认 600 用于演示。
    """
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2,
                      reserved_tokens=["<pad>", "<bos>", "<eos>"])
    tgt_vocab = Vocab(target, min_freq=2,
                      reserved_tokens=["<pad>", "<bos>", "<eos>"])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


# ──────────────────────────────────────────────────────
# main: 分段测试, 对照 HTML 各小节
# ──────────────────────────────────────────────────────
def main():
    torch.manual_seed(42)
    print("=" * 64)
    print("Part 1: 读取数据集 —— 去掉每行末尾的署名信息")
    print("=" * 64)
    print("原始 cmn.txt 前 2 行 (含署名):")
    with open(DATA_PATH, encoding="utf-8") as f:
        for k, line in enumerate(f):
            if k < 2:
                print(f"  {line.strip()}")
            else:
                break
    raw_text = read_data_nmt()
    print(f"\nread_data_nmt 后前 2 行 (已去署名, 制表符分隔):")
    for line in raw_text.split("\n")[:2]:
        eng, chn = line.split("\t")
        print(f"  src={eng!r:8}  tgt={chn!r}")
    n_pairs = len([l for l in raw_text.split("\n") if l])
    print(f"\n句对总数: {n_pairs}")

    print("\n" + "=" * 64)
    print("Part 2: 预处理 preprocess_nmt")
    print("=" * 64)
    text = preprocess_nmt(raw_text)
    print("预处理后前 2 行 (小写 + 英文标点前插空格):")
    for line in text.split("\n")[:2]:
        print(f"  {line!r}")

    print("\n" + "=" * 64)
    print("Part 3: 词元化 tokenize_nmt")
    print("=" * 64)
    source, target = tokenize_nmt(text)
    print("前 5 句对比 (源=词级, 目标=字级):")
    for i in range(5):
        print(f"  src[{i}]={source[i]}")
        print(f"  tgt[{i}]={target[i]}")
    src_lens = [len(l) for l in source]
    tgt_lens = [len(l) for l in target]
    print(f"\n源 token 数: min={min(src_lens)} max={max(src_lens)} "
          f"mean={sum(src_lens)/len(src_lens):.1f}")
    print(f"目标 token 数: min={min(tgt_lens)} max={max(tgt_lens)} "
          f"mean={sum(tgt_lens)/len(tgt_lens):.1f}")

    print("\n" + "=" * 64)
    print("Part 4: 序列长度分布直方图 (源 vs 目标)")
    print("=" * 64)
    show_list_len_pair_hist(
        ["source", "target"], "# tokens per sequence", "count",
        source, target, save_to="data/cmn-eng/token_len_hist.png")
    for name, lens in [("source", src_lens), ("target", tgt_lens)]:
        lens_sorted = sorted(lens)
        n = len(lens_sorted)
        p50 = lens_sorted[n // 2]
        p95 = lens_sorted[int(n * 0.95)]
        print(f"  {name:7}: 中位数={p50}  p95={p95}  "
              f"≤10 的占比={sum(1 for x in lens if x<=10)/n:.1%}")

    print("\n" + "=" * 64)
    print("Part 5: 构建词表 (min_freq=2, 预留 <pad>/<bos>/<eos>)")
    print("=" * 64)
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"])
    print(f"源词表   |src_V| = {len(src_vocab)}")
    print(f"目标词表 |tgt_V| = {len(tgt_vocab)}")
    print("特殊 token 索引:", {t: src_vocab[t] for t in
          ["<unk>", "<pad>", "<bos>", "<eos>"]})

    print("\n" + "=" * 64)
    print("Part 6: 截断 / 填充 truncate_pad")
    print("=" * 64)
    idx = src_vocab[source[0]]
    print(f"source[0]={source[0]} → 索引 {idx}")
    padded = truncate_pad(idx, 10, src_vocab["<pad>"])
    print(f"truncate_pad(_, 10, <pad>) → {padded}")
    print(f"回译: {src_vocab.to_tokens(padded)}")

    print("\n" + "=" * 64)
    print("Part 7: load_data_nmt —— 第一个小批量 (batch_size=2, num_steps=8)")
    print("=" * 64)
    train_iter, src_v, tgt_v = load_data_nmt(batch_size=2, num_steps=8,
                                             num_examples=600)
    print(f"(用前 600 句对演示; 源词表={len(src_v)}, 目标词表={len(tgt_v)})")
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print("X (源, 索引):")
        print("  ", X.type(torch.int32).tolist())
        print("X 的有效长度:", X_valid_len.tolist())
        print("Y (目标, 索引):")
        print("  ", Y.type(torch.int32).tolist())
        print("Y 的有效长度:", Y_valid_len.tolist())

        print("\n" + "=" * 64)
        print("Part 8: 回译验证 —— 把小批量解码回文本 (跳过 <pad>)")
        print("=" * 64)
        pad = src_v["<pad>"]
        for b in range(X.shape[0]):
            src_tok = src_v.to_tokens(
                [i for i in X[b].tolist() if i != pad])
            tgt_tok = tgt_v.to_tokens(
                [i for i in Y[b].tolist()
                 if i != tgt_v["<pad>"]])
            print(f"  样本{b}: src={' '.join(src_tok)}")
            print(f"        tgt={''.join(tgt_tok)}  (中文字级拼接)")
        break

    print("\n结论: 句对 → 词/字级 token → 双词表 → 截断填充到 num_steps → "
          "小批量 (X, valid_len, Y, valid_len), 可直接喂给 §9.6 的 Encoder-Decoder。")


if __name__ == "__main__":
    main()
