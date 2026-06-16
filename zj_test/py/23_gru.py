import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
text = Path(__file__).with_name("data").joinpath("timemachine.txt").read_text().lower()[:20000]
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = dict(enumerate(chars))
data = torch.tensor([stoi[c] for c in text])
steps = 32
X = torch.stack([data[i:i + steps] for i in range(0, len(data) - steps - 1, steps)])
Y = torch.stack([data[i + 1:i + steps + 1] for i in range(0, len(data) - steps - 1, steps)])
loader = DataLoader(TensorDataset(X, Y), 64, shuffle=True)


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(len(chars), 128, batch_first=True)
        self.out = nn.Linear(128, len(chars))

    def forward(self, x, state=None):
        y, state = self.gru(F.one_hot(x, len(chars)).float(), state)
        return self.out(y), state


model = GRU().to(device)
optimizer = torch.optim.Adam(model.parameters(), 0.01)
for epoch in range(int(os.getenv("EPOCHS", 20))):
    total = 0
    for x, y in loader:
        logits, _ = model(x.to(device))
        loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.to(device).reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total += loss.item()
    print(f"{epoch + 1:02d} {total / len(loader):.3f}")

prompt = "time traveller"
state = None
tokens = [stoi[c] for c in prompt]
with torch.no_grad():
    for token in tokens[:-1]:
        _, state = model(torch.tensor([[token]], device=device), state)
    for _ in range(200):
        logits, state = model(torch.tensor([[tokens[-1]]], device=device), state)
        tokens.append(int(torch.multinomial(logits[0, -1].softmax(0), 1)))
print("".join(itos[i] for i in tokens))
