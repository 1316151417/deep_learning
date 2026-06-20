import importlib.util
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

spec = importlib.util.spec_from_file_location("m24", Path(__file__).with_name("24_load_trans_dataset.py"))
m24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m24)
load_data_nmt, truncate_pad = m24.load_data_nmt, m24.truncate_pad


def masked_softmax(X, valid_len):
    if valid_len is None:
        return F.softmax(X, dim=-1)
    shape = X.shape
    X = X.reshape(-1, shape[-1])
    valid_len = valid_len.repeat_interleave(shape[1]) if valid_len.dim() == 1 else valid_len.reshape(-1)
    mask = torch.arange(shape[-1], device=X.device)[None, :] >= valid_len[:, None]
    X = X.masked_fill(mask, -1e6)
    return F.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, q, k, h):
        super().__init__()
        self.W_q, self.W_k, self.W_v = nn.Linear(q, h, bias=False), nn.Linear(k, h, bias=False), nn.Linear(h, 1, bias=False)

    def forward(self, queries, keys, values, valid_len):
        scores = self.W_v(torch.tanh(self.W_q(queries).unsqueeze(2) + self.W_k(keys).unsqueeze(1))).squeeze(-1)
        self.weights = masked_softmax(scores, valid_len)
        return torch.bmm(self.weights, values)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, layers, dropout=dropout)

    def forward(self, X):
        return self.rnn(self.embedding(X).permute(1, 0, 2))


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = AdditiveAttention(hidden_size, hidden_size, hidden_size)
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, layers, dropout=dropout)
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.attention_weights = []

    def init_state(self, enc_outputs, valid_len):
        output, hidden = enc_outputs
        return output.permute(1, 0, 2), hidden, valid_len

    def forward(self, X, state):
        enc_outputs, hidden, valid_len = state
        X = self.embedding(X).permute(1, 0, 2)
        outs, self.attention_weights = [], []
        for x in X:
            query = hidden[-1].unsqueeze(1)
            context = self.attention(query, enc_outputs, enc_outputs, valid_len)
            out, hidden = self.rnn(torch.cat((context, x.unsqueeze(1)), -1).permute(1, 0, 2), hidden)
            outs.append(out)
            self.attention_weights.append(self.attention.weights)
        return self.dense(torch.cat(outs)).permute(1, 0, 2), (enc_outputs, hidden, valid_len)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, X, dec_X, valid_len):
        return self.decoder(dec_X, self.decoder.init_state(self.encoder(X), valid_len))


def loss_fn(pred, y, valid_len):
    w = torch.arange(y.shape[1], device=y.device)[None, :] < valid_len[:, None]
    return (F.cross_entropy(pred.permute(0, 2, 1), y, reduction="none") * w).sum() / w.sum()


def train(net, data_iter, tgt_vocab, device, epochs):
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), 0.005)
    for epoch in range(epochs):
        total, n = 0, 0
        for X, X_len, Y, Y_len in data_iter:
            X, X_len, Y, Y_len = [a.to(device) for a in (X, X_len, Y, Y_len)]
            bos = torch.full((Y.shape[0], 1), tgt_vocab["<bos>"], device=device)
            pred, _ = net(X, torch.cat((bos, Y[:, :-1]), 1), X_len)
            loss = loss_fn(pred, Y, Y_len)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            total += loss.detach().item() * int(Y_len.sum())
            n += int(Y_len.sum())
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {epoch + 1:3d}, loss {total / n:.3f}")


def predict(net, sentence, src_vocab, tgt_vocab, num_steps, device):
    net.eval()
    src = truncate_pad(src_vocab[sentence.lower().split()] + [src_vocab["<eos>"]], num_steps, src_vocab["<pad>"])
    X = torch.tensor(src, device=device).unsqueeze(0)
    valid_len = torch.tensor([min(len(sentence.split()) + 1, num_steps)], device=device)
    state = net.decoder.init_state(net.encoder(X), valid_len)
    dec_X = torch.tensor([[tgt_vocab["<bos>"]]], device=device)
    out, weights = [], []
    for _ in range(num_steps):
        Y, state = net.decoder(dec_X, state)
        weights.append(net.decoder.attention_weights[-1][0, 0].detach().cpu())
        dec_X = Y.argmax(2)
        token = int(dec_X.item())
        if token == tgt_vocab["<eos>"]:
            break
        out.append(token)
    return "".join(tgt_vocab.to_tokens(out)), torch.stack(weights)


def bleu(pred, label, k=2):
    pred, label = list(pred), list(label)
    if not pred:
        return 0
    score = math.exp(min(0, 1 - len(label) / len(pred)))
    for n in range(1, k + 1):
        hit, bag = 0, {}
        for i in range(len(label) - n + 1):
            gram = tuple(label[i:i + n])
            bag[gram] = bag.get(gram, 0) + 1
        for i in range(len(pred) - n + 1):
            gram = tuple(pred[i:i + n])
            if bag.get(gram, 0):
                hit += 1
                bag[gram] -= 1
        score *= (hit / max(len(pred) - n + 1, 1)) ** (0.5 ** n)
    return score


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    data_iter, src_vocab, tgt_vocab = load_data_nmt(64, 10, num_examples=600)
    net = EncoderDecoder(Encoder(len(src_vocab), 32, 32, 2, 0.1), Decoder(len(tgt_vocab), 32, 32, 2, 0.1))
    train(net, data_iter, tgt_vocab, device, int(os.getenv("EPOCHS", 80)))
    for eng, chn in [("go .", "走。"), ("i won !", "我赢了！"), ("i lost .", "我输了。"), ("i am ok .", "我没事。")]:
        pred, weights = predict(net, eng, src_vocab, tgt_vocab, 10, device)
        print(f"{eng:10} => {pred:10} bleu {bleu(pred, chn):.3f}")
        print(weights[:, :len(eng.split()) + 1].round(decimals=2).tolist())


if __name__ == "__main__":
    main()
