import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import torch
import random
import torch.nn.functional as F

# read names
fullNames = open("./data/names.csv", "r").read().splitlines()
words = [name.split("\t")[0].lower() for name in fullNames]

# names to indexes and back
chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

# training set (x, y)
xs, ys = [], []

# populate training set
for w in words[:]:
    chs = ["."] + list(w) + ["."]
    for c1, c2 in zip(chs, chs[1:]):
        ix1 = stoi[c1]
        ix2 = stoi[c2]
        # print(c1, c2)
        xs.append(ix1)
        ys.append(ix2)

# convert to tensors
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print("number of examples", num)

# random weights
g = torch.Generator().manual_seed(2147483647)
W = torch.rand((27, 27), generator=g, requires_grad=True)

# gradient descent
for i in range(100):
    # one hot encoding as inputs are ints
    xenc = F.one_hot(xs, num_classes=27).float()

    # forward pass
    logits = xenc @ W

    # exp to smooth
    counts = logits.exp()

    # probability distribution
    probs = counts / counts.sum(1, keepdims=True)

    # calculate loss
    loss = -probs[torch.arange(num), ys].log().mean() + 0.1 * (W**2).mean()
    # print(loss.item())

    # backward pass
    W.grad = None
    loss.backward()

    # adjust weights
    W.data += -50 * W.grad

print(loss.item())

for i in range(5):
    ix = 0
    name = ""
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        name += itos[ix]

        if ix == 0:
            break

    print(name)
