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
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


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
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = valid_lens.repeat_interleave(self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
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
        return self.dropout(X + self.P[:, :X.shape[1], :].to(X.device))


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


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
                         ffn_num_input, ffn_num_hiddens, num_heads, dropout, bias)
            for _ in range(num_layers)])

    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = []
        for blk in self.blks:
            X = blk(X, valid_lens)
            self.attention_weights.append(blk.attention.attention.attention_weights)
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
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


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
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self.attention_weights[0][i] = blk.attention1.attention.attention_weights
            self.attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, enc_X, dec_X, enc_valid_lens):
        enc_outputs = self.encoder(enc_X, enc_valid_lens)
        return self.decoder(dec_X, self.decoder.init_state(enc_outputs, enc_valid_lens))


def xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def train(net, data_iter, lr, epochs, tgt_vocab, device):
    net.apply(xavier).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    for epoch in range(epochs):
        total, ntok = 0.0, 0
        for X, Xvl, Y, Yvl in data_iter:
            X, Xvl, Y, Yvl = (a.to(device) for a in (X, Xvl, Y, Yvl))
            bos = torch.full((Y.shape[0], 1), tgt_vocab["<bos>"], device=device)
            dec_X = torch.cat([bos, Y[:, :-1]], 1)
            opt.zero_grad()
            Y_hat, _ = net(X, dec_X, Xvl)
            l = loss(Y_hat, Y, Yvl)
            l.sum().backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            total += l.sum().item(); ntok += int(Yvl.sum())
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {epoch + 1:3d}, loss {total / ntok:.3f}")


def predict(net, sent, src_vocab, tgt_vocab, num_steps, device):
    net.eval()
    tokens = truncate_pad(src_vocab[sent.lower().split()] + [src_vocab["<eos>"]],
                          num_steps, src_vocab["<pad>"])
    enc_X = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    enc_valid_len = torch.tensor([min(len(sent.split()) + 1, num_steps)], device=device)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.tensor([[tgt_vocab["<bos>"]]], dtype=torch.long, device=device)
    out = []
    for _ in range(num_steps):
        Y, state = net.decoder(dec_X, state)
        dec_X = Y[:, -1:].argmax(dim=2)
        pred = int(dec_X.squeeze().item())
        if pred == tgt_vocab["<eos>"]:
            break
        out.append(pred)
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
    train(net, data_iter, lr, epochs, tgt_vocab, device)
    print("\n翻译预测 (English => 中文, BLEU):")
    for eng, chn in [("hi .", "嗨。"), ("wait !", "等等！"), ("hello !", "你好。"),
                     ("i try .", "我试试。"), ("i won !", "我赢了。"), ("fire !", "火！")]:
        t = predict(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f"  {eng:10} => {t}   (bleu {bleu(t, chn, 2):.3f})")


if __name__ == "__main__":
    main()
