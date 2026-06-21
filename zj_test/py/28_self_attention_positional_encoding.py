import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout=0, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        P = torch.zeros(1, max_len, num_hiddens)
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        P[:, :, 0::2] = torch.sin(X)
        P[:, :, 1::2] = torch.cos(X)
        self.register_buffer("P", P)

    def forward(self, X):
        return self.dropout(X + self.P[:, :X.shape[1], :])


def self_attention_demo():
    torch.manual_seed(42)
    X = torch.randn(2, 4, 100)
    valid_lens = torch.tensor([3, 2])
    mask = torch.arange(X.shape[1])[None, :] >= valid_lens[:, None]
    attn = nn.MultiheadAttention(100, 5, dropout=0.0, batch_first=True)
    Y, W = attn(X, X, X, key_padding_mask=mask, need_weights=True)
    return X.shape, Y.shape, W.round(decimals=3)


def positional_demo():
    pe = PositionalEncoding(32, 0)
    X = pe(torch.zeros(1, 60, 32))
    P = pe.P[:, :60, :]
    i, delta, j = 7, 5, 3
    w = 1 / 10000 ** (2 * j / 32)
    R = torch.tensor([[math.cos(delta * w), math.sin(delta * w)],
                      [-math.sin(delta * w), math.cos(delta * w)]])
    err = (R @ P[0, i, 2 * j:2 * j + 2] - P[0, i + delta, 2 * j:2 * j + 2]).abs().max()
    return X.shape, P[0, :8, :8].round(decimals=3), float(err)


def main():
    x_shape, y_shape, weights = self_attention_demo()
    pe_shape, pe_block, rel_err = positional_demo()
    print(f"self_attention: X={tuple(x_shape)}, Y={tuple(y_shape)}")
    print("attention weights batch0:")
    print(weights[0])
    print(f"positional_encoding: X+P={tuple(pe_shape)}")
    print("P[:8, :8]:")
    print(pe_block)
    print(f"relative_projection_error={rel_err:.2e}")
    print("complexity: CNN O(knd^2), RNN O(nd^2), self-attention O(n^2d)")


if __name__ == "__main__":
    main()
