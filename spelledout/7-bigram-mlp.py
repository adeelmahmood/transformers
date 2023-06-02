import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import torch
import random
import torch.nn.functional as F


# fullNames = open("./data/names.csv", "r").read().splitlines()
# words = [name.split("\t")[0].lower() for name in fullNames]

words = open("./data/names.txt", "r").read().splitlines()

chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 3
max_steps = 200000
emb_space_size = 10
n_hidden = 200


def build_dataset(words):
    # build training set
    X, Y = [], []

    for w in words:
        # print(w)
        context = [0] * block_size
        for c in w + ".":
            ix = stoi[c]
            X.append(context)
            Y.append(ix)
            # print("".join(itos[i] for i in context), "-->", c)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


g = torch.Generator().manual_seed(2147483647)

# sets
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

C = torch.randn((27, emb_space_size), generator=g)
W1 = torch.randn((block_size * emb_space_size, n_hidden), generator=g) * (
    (5 / 3) / (block_size * emb_space_size) ** 0.5
)
b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, 27), generator=g) * 0.01
b2 = torch.randn(27, generator=g) * 0

bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(f"{sum(p.nelement() for p in parameters)=}")

for p in parameters:
    p.requires_grad = True

lossi = []
stepi = []

for i in range(max_steps):
    # batch
    ix = torch.randint(0, Xtr.shape[0], (32,))

    # forward pass
    emb = C[Xtr[ix]]
    embcat = emb.view(-1, block_size * emb_space_size)
    hpreact = embcat @ W1 + b1

    # batch norm
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < max_steps / 2 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # stats
    lossi.append(loss.item())
    stepi.append(i)
    if i % 10000 == 0:
        print(f"{i}: {loss.item()}")

    # break

with torch.no_grad():
    # total loss
    emb = C[Xtr]
    hpreact = emb.view(-1, block_size * emb_space_size) @ W1 + b1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr)
    print(f"training set loss = {loss.item()}")

    emb = C[Xdev]
    hpreact = emb.view(-1, block_size * emb_space_size) @ W1 + b1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ydev)
    print(f"test set loss = {loss.item()}")


g = torch.Generator().manual_seed(2147483647 + 10)

# show plot for embeddings only when 2 dims are being used
if emb_space_size == 2:
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(
            C[i, 0].item(),
            C[i, 1].item(),
            itos[i],
            va="center",
            ha="center",
            color="white",
        )
    plt.show()


# generate output from model
for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        hpreact = emb.view(1, -1) @ W1  # + b1
        hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)

        if ix == 0:
            break
    print("".join(itos[i] for i in out))

# plt.plot(stepi, lossi)
# plt.show()
