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
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_len):
    if valid_len is None:
        return F.softmax(X, dim=-1)
    shape = X.shape
    X = X.reshape(-1, shape[-1])
    valid_len = valid_len.repeat_interleave(shape[1]) if valid_len.dim() == 1 else valid_len.reshape(-1)
    return F.softmax(sequence_mask(X, valid_len, -1e6).reshape(shape), dim=-1)


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = sequence_mask(torch.ones_like(label), valid_len)
        self.reduction = 'none'
        unweighted = super().forward(pred.permute(0, 2, 1), label)
        return (unweighted * weights).mean(dim=1)


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout=0):
        super().__init__()
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_len):
        scores = self.W_v(torch.tanh(self.W_q(queries).unsqueeze(2) + self.W_k(keys).unsqueeze(1))).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_len)
        return torch.bmm(self.dropout(self.attention_weights), values)


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        return self.rnn(self.embedding(X).permute(1, 0, 2))


class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.attention_weights = []

    def init_state(self, enc_outputs, enc_valid_len, *args):
        outputs, hidden_state = enc_outputs
        return outputs.permute(1, 0, 2), hidden_state, enc_valid_len

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_len = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self.attention_weights = [], []
        for x in X:
            query = hidden_state[-1].unsqueeze(1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_len)
            out, hidden_state = self.rnn(torch.cat((context, x.unsqueeze(1)), -1).permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self.attention_weights.append(self.attention.attention_weights)
        return self.dense(torch.cat(outputs)).permute(1, 0, 2), (enc_outputs, hidden_state, enc_valid_len)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, enc_X, dec_X, enc_valid_len):
        enc_outputs = self.encoder(enc_X)
        return self.decoder(dec_X, self.decoder.init_state(enc_outputs, enc_valid_len))


def xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.GRU):
        for name in m._flat_weights_names:
            if 'weight' in name:
                nn.init.xavier_uniform_(m._parameters[name])


def train(net, data_iter, lr, num_epochs, tgt_vocab, device):
    net.apply(xavier).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    for epoch in range(num_epochs):
        total, ntok = 0.0, 0
        for X, Xvl, Y, Yvl in data_iter:
            X, Xvl, Y, Yvl = (x.to(device) for x in (X, Xvl, Y, Yvl))
            opt.zero_grad()
            bos = torch.full((Y.shape[0], 1), tgt_vocab['<bos>'], device=device)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            Y_hat, _ = net(X, dec_input, Xvl)
            l = loss(Y_hat, Y, Yvl)
            l.sum().backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            total += l.sum().item(); ntok += int(Yvl.sum())
        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            print(f'epoch {epoch + 1:3d}, loss {total / ntok:.3f}')


def predict(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    net.eval()
    tokens = truncate_pad(src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']],
                          num_steps, src_vocab['<pad>'])
    enc_X = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    enc_valid_len = torch.tensor([min(len(src_sentence.split(' ')) + 1, num_steps)], device=device)
    state = net.decoder.init_state(net.encoder(enc_X), enc_valid_len)
    dec_X = torch.tensor([[tgt_vocab['<bos>']]], dtype=torch.long, device=device)
    out, weights = [], []
    for _ in range(num_steps):
        Y, state = net.decoder(dec_X, state)
        weights.append(net.decoder.attention_weights[-1][0, 0].detach().cpu())
        dec_X = Y.argmax(dim=2)
        pred = int(dec_X.squeeze(0).item())
        if pred == tgt_vocab['<eos>']:
            break
        out.append(pred)
    return ''.join(tgt_vocab.to_tokens(out)), torch.stack(weights) if weights else torch.empty(0, num_steps)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps, lr = 64, 10, 0.005
    num_epochs = int(os.getenv("EPOCHS", 300))
    data_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps, num_examples=600)
    print(f'src_vocab={len(src_vocab)}, tgt_vocab={len(tgt_vocab)}, device={device}')
    net = EncoderDecoder(
        Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout),
        Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout))
    train(net, data_iter, lr, num_epochs, tgt_vocab, device)
    print('\n翻译预测 (English => 中文, BLEU, attention):')
    for eng, chn in [('hi .', '嗨。'), ('wait !', '等等！'), ('hello !', '你好。'),
                     ('i try .', '我试试。'), ('i won !', '我赢了。'), ('fire !', '火！')]:
        t, w = predict(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'  {eng:10} => {t}   (bleu {bleu(t, chn, 2):.3f})')
        print(' ', w[:, :len(eng.split(" ")) + 1].round(decimals=2).tolist())


if __name__ == '__main__':
    main()
