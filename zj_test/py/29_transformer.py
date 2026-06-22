import importlib.util
import math
import os
from pathlib import Path

import torch
from torch import nn

_spec = importlib.util.spec_from_file_location(
    "m24", Path(__file__).resolve().parent / "24_load_trans_dataset.py")
m24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m24)
load_data_nmt, truncate_pad = m24.load_data_nmt, m24.truncate_pad


def sequence_mask(X, valid_len, value=0):
    mask = torch.arange(X.size(1), device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = sequence_mask(torch.ones_like(label), valid_len)
        self.reduction = "none"
        unweighted = super().forward(pred.permute(0, 2, 1), label)
        return (unweighted * weights).mean(dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        P = torch.zeros(1, max_len, d_model)
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        P[:, :, 0::2], P[:, :, 1::2] = torch.sin(X), torch.cos(X)
        self.register_buffer("P", P)

    def forward(self, X):
        return self.dropout(X + self.P[:, :X.shape[1], :])


class TransformerNMT(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=32, ffn=64, heads=4, layers=2, dropout=0.1, pad=0):
        super().__init__()
        self.d_model, self.pad = d_model, pad
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=pad)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=pad)
        self.pos = PositionalEncoding(d_model, dropout)
        self.net = nn.Transformer(d_model, heads, layers, layers, ffn, dropout, batch_first=True)
        self.dense = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_len):
        src_pad = torch.arange(src.shape[1], device=src.device)[None, :] >= src_len[:, None]
        tgt_pad = tgt == self.pad
        tgt_mask = torch.triu(torch.ones(tgt.shape[1], tgt.shape[1], device=tgt.device, dtype=torch.bool), 1)
        src = self.pos(self.src_embed(src) * math.sqrt(self.d_model))
        tgt = self.pos(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        Y = self.net(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad,
                     tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=src_pad)
        return self.dense(Y)


def train(net, data_iter, lr, epochs, tgt_vocab, device):
    net.to(device)
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
            l = loss(net(X, dec_X, Xvl), Y, Yvl)
            l.sum().backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            total += l.sum().item(); ntok += int(Yvl.sum())
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {epoch + 1:3d}, loss {total / ntok:.3f}")


def predict(net, sent, src_vocab, tgt_vocab, num_steps, device):
    net.eval()
    src = truncate_pad(src_vocab[sent.lower().split()] + [src_vocab["<eos>"]], num_steps, src_vocab["<pad>"])
    X = torch.tensor(src, device=device).unsqueeze(0)
    Xvl = torch.tensor([min(len(sent.split()) + 1, num_steps)], device=device)
    dec = torch.tensor([[tgt_vocab["<bos>"]]], device=device)
    out = []
    for _ in range(num_steps):
        Y = net(X, dec, Xvl)
        pred = int(Y[:, -1].argmax(1).item())
        if pred == tgt_vocab["<eos>"]:
            break
        out.append(pred)
        dec = torch.cat([dec, torch.tensor([[pred]], device=device)], 1)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_steps, lr, epochs = 64, 10, 0.005, int(os.getenv("EPOCHS", 200))
    data_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps, num_examples=600)
    net = TransformerNMT(len(src_vocab), len(tgt_vocab), pad=tgt_vocab["<pad>"])
    print(f"src_vocab={len(src_vocab)}, tgt_vocab={len(tgt_vocab)}, device={device}")
    train(net, data_iter, lr, epochs, tgt_vocab, device)
    print("\n翻译预测 (English => 中文, BLEU):")
    for eng, chn in [("hi .", "嗨。"), ("wait !", "等等！"), ("hello !", "你好。"),
                     ("i try .", "我试试。"), ("i won !", "我赢了。"), ("fire !", "火！")]:
        t = predict(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f"  {eng:10} => {t}   (bleu {bleu(t, chn, 2):.3f})")


if __name__ == "__main__":
    main()
