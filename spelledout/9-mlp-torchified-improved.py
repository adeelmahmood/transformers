import math
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import torch
import random
import torch.nn.functional as F


words = open("./data/names.txt", "r").read().splitlines()

chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 8
max_steps = 200000
emb_size = 10
n_hidden = 68
vocab_size = 27
batch_size = 32


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


torch.manual_seed(42)
# sets
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x: torch.Tensor):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1) -> None:
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x: torch.Tensor) -> Any:
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, n_embedding, dim_embedding) -> None:
        self.weight = torch.randn((n_embedding, dim_embedding))

    def __call__(self, ix) -> Any:
        self.out = self.weight[ix]
        return self.out

    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    def __init__(self, n) -> None:
        self.n = n

    def __call__(self, x) -> Any:
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers) -> None:
        self.layers = layers

    def __call__(self, x) -> Any:
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


model = Sequential(
    [
        Embedding(vocab_size, emb_size),
        FlattenConsecutive(2),
        Linear(emb_size * 2, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        FlattenConsecutive(2),
        Linear(n_hidden * 2, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        FlattenConsecutive(2),
        Linear(n_hidden * 2, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        Linear(n_hidden, vocab_size),
    ]
)

with torch.no_grad():
    # make last layer weights low
    model.layers[-1].weight *= 0.1

parameters = model.parameters()
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

# debugging
# ix = torch.randint(0, Xtr.shape[0], (4,))
# logits = model(Xtr[ix])
# for layer in model.layers:
#     print(layer.__class__.__name__, tuple(layer.out.shape))

lossi = []

for i in range(max_steps):
    # batch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    # forward pass
    logits = model(Xtr[ix])
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
    lossi.append(loss.log10().item())
    if i % 10000 == 0:
        print(f"{i}: {loss.item()}")

    # if i == 999:
    #     break

# done with training
for layer in model.layers:
    layer.training = False

with torch.no_grad():
    # total loss
    logits = model(Xtr)
    loss = F.cross_entropy(logits, Ytr)
    print(f"training set loss = {loss.item()}")

    logits = model(Xdev)
    loss = F.cross_entropy(logits, Ydev)
    print(f"test set loss = {loss.item()}")


# generate output from model
for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)

        if ix == 0:
            break
    print("".join(itos[i] for i in out))

plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
plt.show()
