import contextlib
import importlib.util
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

_spec = importlib.util.spec_from_file_location(
    "m24", Path(__file__).resolve().parent / "24_load_trans_dataset.py")
m24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m24)
load_data_nmt, truncate_pad = m24.load_data_nmt, m24.truncate_pad


# ───────────────────────── 日志工具(用于展示 Transformer 处理流程) ─────────────────────────
# 默认静默; 仅在 train 的首个批次 / predict 的首个句子中开启, 避免刷屏干扰核心学习。
# vprint  = 流程级(编码/解码层、子层、KV缓存、预测token), verbose 开启即打印
# vdetail = 数值内部(投影、点积、softmax、FFN、残差), 需 verbose 且 detail 同时开启
_VERBOSE = False
_DETAIL = True
_DEPTH = 0


def vprint(msg=""):
    """流程级日志: 仅 verbose 时打印, 自动按调用层级缩进。"""
    global _DEPTH
    if _VERBOSE:
        print("  " * _DEPTH + str(msg))


def vdetail(msg=""):
    """数值内部日志: verbose 且 detail 同时开启时打印。"""
    if _VERBOSE and _DETAIL:
        print("  " * _DEPTH + str(msg))


@contextlib.contextmanager
def vblock():
    """进入一层缩进。"""
    global _DEPTH
    _DEPTH += 1
    try:
        yield
    finally:
        _DEPTH -= 1


@contextlib.contextmanager
def verbose(on):
    """临时开关详细日志。"""
    global _VERBOSE
    old, _VERBOSE = _VERBOSE, on
    try:
        yield
    finally:
        _VERBOSE = old


@contextlib.contextmanager
def detail(on):
    """临时开关数值内部日志。"""
    global _DETAIL
    old, _DETAIL = _DETAIL, on
    try:
        yield
    finally:
        _DETAIL = old


def _tok(vocab, ids):
    """token id → 可读词; 失败回退为原始 id。"""
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    try:
        return vocab.to_tokens(list(ids))
    except Exception:
        return list(ids)


def sequence_mask(X, valid_len, value=0):
    mask = torch.arange(X.size(1), device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = valid_lens.repeat_interleave(shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, -1e6)
    return F.softmax(X.reshape(shape), dim=-1)


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = sequence_mask(torch.ones_like(label), valid_len)
        self.reduction = "none"
        unweighted = super().forward(pred.permute(0, 2, 1), label)
        return (unweighted * weights).mean(dim=1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        vdetail(f"缩放点积注意力: scores = Q·Kᵀ / √{d}")
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        vdetail(f"  scores 形状 {tuple(scores.shape)}  (每个头一个 [查询序列 × 键序列])")
        self.attention_weights = masked_softmax(scores, valid_lens)
        vdetail(f"  掩蔽非法位置(padding / 未来) + softmax → 注意力权重, 每行和 = 1")
        out = torch.bmm(self.dropout(self.attention_weights), values)
        vdetail(f"  输出 = 注意力权重 · V, 形状 {tuple(out.shape)}")
        return out


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        vdetail(f"多头注意力({self.num_heads} 头): W_q / W_k / W_v 线性投影 → {self.W_q.out_features} 维")
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        vdetail(f"  拆成 {self.num_heads} 头并行 → Q/K/V 形状 {tuple(queries.shape)}  (= batch×头, 序列, 维/头)")
        if valid_lens is not None:
            valid_lens = valid_lens.repeat_interleave(self.num_heads, dim=0)
        with vblock():
            output = self.attention(queries, keys, values, valid_lens)
        vdetail(f"  合并 {self.num_heads} 头 + W_o 输出投影")
        return self.W_o(transpose_output(output, self.num_heads))


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        P = torch.zeros(1, max_len, num_hiddens)
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        P[:, :, 0::2], P[:, :, 1::2] = torch.sin(X), torch.cos(X)
        self.register_buffer("P", P)

    def forward(self, X):
        vdetail(f"  + 位置编码(位置 i 的 sin / cos, 注入顺序信息) + dropout, 形状 {tuple(X.shape)}")
        return self.dropout(X + self.P[:, :X.shape[1], :].to(X.device))


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        vdetail(f"逐位置 FFN: {self.dense1.in_features}→{self.dense1.out_features}→{self.dense2.out_features} (ReLU), 形状不变")
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        vdetail("残差(X + 子层输出) + LayerNorm")
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, i=0, bias=False):
        super().__init__()
        self.i = i
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        vprint(f"── 子层1: 多头自注意力 (Q = K = V = X, 源序列内部交互)")
        with vblock():
            a = self.attention(X, X, X, valid_lens)
        Y = self.addnorm1(X, a)
        vprint(f"── 子层2: 逐位置前馈网络 FFN")
        with vblock():
            f = self.ffn(Y)
        return self.addnorm2(Y, f)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                 dropout, bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.ModuleList([
            EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                         ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, bias)
            for i in range(num_layers)])

    def forward(self, X, valid_lens, *args):
        vprint(f"【编码器】输入 token id 形状 {tuple(X.shape)}")
        vprint(f"词嵌入 → {self.num_hiddens} 维, 再缩放 ×√{self.num_hiddens} (放大后不被位置编码淹没)")
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = []
        for i, blk in enumerate(self.blks):
            vprint(f"════ 编码器层 {i + 1}/{len(self.blks)} ════")
            with vblock():
                X = blk(X, valid_lens)
                self.attention_weights.append(blk.attention.attention.attention_weights)
        vprint(f"【编码器】输出: 编码序列, 形状 {tuple(X.shape)}")
        return X


class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, bias=False):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens, cache = state
        key_values = X if cache[self.i] is None else torch.cat((cache[self.i], X), dim=1)
        cache[self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        vprint(f"── 子层1: 带掩码多头自注意力 (Q=X, K=V=已累积上下文)")
        vprint(f"     KV缓存: 累积 {key_values.shape[1]} 步; "
               f"{'训练用因果掩码(第 t 步只看 1..t, 防偷看未来)' if self.training else '预测无掩码(每步只喂1个新token)'}")
        with vblock():
            X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        vprint(f"── 子层2: 编码器-解码器交叉注意力 (Q=解码端, K=V=编码器输出, 翻译对齐源句)")
        with vblock():
            Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        vprint(f"── 子层3: 逐位置前馈网络 FFN")
        with vblock():
            f = self.ffn(Z)
        return self.addnorm3(Z, f), state


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                 dropout, bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.ModuleList([
            DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                         ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, bias)
            for i in range(num_layers)])
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        vprint(f"【解码器】输入 token id 形状 {tuple(X.shape)}")
        vprint(f"词嵌入 → {self.num_hiddens} 维, 再缩放 ×√{self.num_hiddens}")
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            vprint(f"════ 解码器层 {i + 1}/{len(self.blks)} ════")
            with vblock():
                X, state = blk(X, state)
                self.attention_weights[0][i] = blk.attention1.attention.attention_weights
                self.attention_weights[1][i] = blk.attention2.attention.attention_weights
        vprint(f"线性投影 → 词表大小 {self.dense.out_features} 的 logits")
        logits = self.dense(X)
        vprint(f"输出 logits 形状 {tuple(logits.shape)}  (softmax 后即各词概率)")
        return logits, state


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, enc_X, dec_X, enc_valid_lens):
        vprint("▶ 阶段1: 编码器处理源序列")
        with vblock():
            enc_outputs = self.encoder(enc_X, enc_valid_lens)
        vprint("▶ 阶段2: 解码器处理目标序列 (以编码输出为交叉注意力的 K/V)")
        with vblock():
            return self.decoder(dec_X, self.decoder.init_state(enc_outputs, enc_valid_lens))


def xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def train(net, data_iter, lr, epochs, tgt_vocab, device, src_vocab=None):
    net.apply(xavier).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    for epoch in range(epochs):
        total, ntok = 0.0, 0
        for batch_i, (X, Xvl, Y, Yvl) in enumerate(data_iter):
            X, Xvl, Y, Yvl = (a.to(device) for a in (X, Xvl, Y, Yvl))
            bos = torch.full((Y.shape[0], 1), tgt_vocab["<bos>"], device=device)
            dec_X = torch.cat([bos, Y[:, :-1]], 1)  # teacher forcing: <bos> + 目标右移一位
            opt.zero_grad()
            demo = (epoch == 0 and batch_i == 0)
            with verbose(demo):
                if demo:
                    print("\n" + "═" * 68)
                    print(" 详细演示: 第 1 个 epoch 的第 1 个训练批次 (其余批次 / epoch 静默)")
                    print("═" * 68)
                    vprint(f"源 X id 形状 {tuple(X.shape)}, 有效长度 {Xvl.tolist()}")
                    if src_vocab is not None:
                        vprint(f"     样例源词: {_tok(src_vocab, X[0])}")
                    vprint(f"目标 Y id 形状 {tuple(Y.shape)}, 有效长度 {Yvl.tolist()}")
                    vprint(f"     样例目标词: {_tok(tgt_vocab, Y[0])}")
                    vprint(f"解码器输入 dec_X = [<bos>] + Y[:, :-1] (teacher forcing), 形状 {tuple(dec_X.shape)}")
                    vprint(f"     样例 dec_X 词: {_tok(tgt_vocab, dec_X[0])}")
                    vprint("▼▼▼ 前向传播开始 ▼▼▼")
                Y_hat, _ = net(X, dec_X, Xvl)
                l = loss(Y_hat, Y, Yvl)
                if demo:
                    vprint("▲▲▲ 前向传播结束 ▲▲▲")
                    vprint("▶ 损失: 每个有效位置算交叉熵, 按有效长度加权平均")
                    vprint(f"     批次总 loss = {l.sum().item():.3f}  有效 token 数 = {int(Yvl.sum())}")
                    vprint("▶ 反向传播求梯度 → 梯度裁剪(范数 ≤ 1) → Adam 更新参数")
                l.sum().backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            total += l.sum().item(); ntok += int(Yvl.sum())
            if demo:
                print("═" * 68 + "\n")
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {epoch + 1:3d}, loss {total / ntok:.3f}")


def predict(net, sent, src_vocab, tgt_vocab, num_steps, device, demo=False):
    net.eval()
    tokens = truncate_pad(src_vocab[sent.lower().split()] + [src_vocab["<eos>"]],
                          num_steps, src_vocab["<pad>"])
    enc_X = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    enc_valid_len = torch.tensor([min(len(sent.split()) + 1, num_steps)], device=device)
    # 预测演示只看「自回归流程 + KV缓存增长」, 关闭数值内部细节(训练演示已展示)
    with verbose(demo):
        with detail(not demo):
            if demo:
                print("\n" + "─" * 68)
                print(f' 预测演示: "{sent}"')
                print(f"   分词 {sent.lower().split()} + <eos>, 截断 / 填充至 {num_steps} → id {tokens}")
                print("─" * 68)
                vprint("▼ 阶段1: 一次性编码源序列 (整句, 仅这一次)")
            with vblock():
                enc_outputs = net.encoder(enc_X, enc_valid_len)
            state = net.decoder.init_state(enc_outputs, enc_valid_len)
            if demo:
                vprint(f"编码输出形状 {tuple(enc_outputs.shape)}  (→ 之后每步交叉注意力的 K/V)")
            dec_X = torch.tensor([[tgt_vocab["<bos>"]]], dtype=torch.long, device=device)
            out = []
            if demo:
                vprint("▼ 阶段2: 自回归解码 (每步只喂 1 个 token, 预测下一个, 回填输入)")
            for step in range(num_steps):
                if demo:
                    vprint(f"━━ 步 {step + 1}: 输入 = {_tok(tgt_vocab, dec_X[0])} {dec_X[0].tolist()}")
                with vblock():
                    Y, state = net.decoder(dec_X, state)
                dec_X = Y[:, -1:].argmax(dim=2)
                pred = int(dec_X.squeeze().item())
                if demo:
                    vprint(f"     argmax 预测下一 token = {pred} ({_tok(tgt_vocab, [pred])})")
                if pred == tgt_vocab["<eos>"]:
                    if demo:
                        vprint("     命中 <eos>, 停止解码")
                    break
                out.append(pred)
            if demo:
                print("─" * 68)
    return "".join(tgt_vocab.to_tokens(out))


def bleu(pred_seq, label_seq, k):
    pred, label = list(pred_seq), list(label_seq)
    if not pred:
        return 0
    score = math.exp(min(0, 1 - len(label) / len(pred)))
    for n in range(1, k + 1):
        num, subs = 0, {}
        for i in range(len(label) - n + 1):
            g = tuple(label[i:i + n]); subs[g] = subs.get(g, 0) + 1
        for i in range(len(pred) - n + 1):
            g = tuple(pred[i:i + n])
            if subs.get(g, 0) > 0:
                num += 1; subs[g] -= 1
        score *= math.pow(num / max(len(pred) - n + 1, 1), math.pow(0.5, n))
    return score


def main():
    torch.manual_seed(42)
    has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu")
    batch_size, num_steps, lr, epochs = 64, 10, 0.005, int(os.getenv("EPOCHS", 200))
    num_hiddens, num_layers, dropout, num_heads, ffn_num_hiddens = 32, 2, 0.1, 4, 64
    data_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps, num_examples=600)
    encoder = TransformerEncoder(len(src_vocab), num_hiddens, num_hiddens, num_hiddens,
                                 num_hiddens, [num_hiddens], num_hiddens,
                                 ffn_num_hiddens, num_heads, num_layers, dropout)
    decoder = TransformerDecoder(len(tgt_vocab), num_hiddens, num_hiddens, num_hiddens,
                                 num_hiddens, [num_hiddens], num_hiddens,
                                 ffn_num_hiddens, num_heads, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    print(f"src_vocab={len(src_vocab)}, tgt_vocab={len(tgt_vocab)}, device={device}")
    train(net, data_iter, lr, epochs, tgt_vocab, device, src_vocab)
    print("\n翻译预测 (English => 中文, BLEU):")
    for i, (eng, chn) in enumerate([("hi .", "嗨。"), ("wait !", "等等！"), ("hello !", "你好。"),
                                    ("i try .", "我试试。"), ("i won !", "我赢了。"), ("fire !", "火！")]):
        t = predict(net, eng, src_vocab, tgt_vocab, num_steps, device, demo=(i == 0))
        print(f"  {eng:10} => {t}   (bleu {bleu(t, chn, 2):.3f})")


if __name__ == "__main__":
    main()
