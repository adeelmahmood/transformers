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

block_size = 3
max_steps = 200000
emb_size = 10
n_hidden = 100
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


# sets
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)


class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
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
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
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


C = torch.randn((vocab_size, emb_size), generator=g)
layers = [
    Linear(block_size * emb_size, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size),
    BatchNorm1d(vocab_size),
]

with torch.no_grad():
    # make last layer weights low
    # layers[-1].weight *= 0.1 (not needed with batchnorm)
    layers[-1].gamma *= 0.1
    # all other layers add the tanh kaiming factor to weights
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5 / 3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

for i in range(max_steps):
    # batch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    # forward pass
    emb = C[Xtr[ix]]
    print(emb.shape)
    x = emb.view(emb.shape[0], -1)
    print(x.shape)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Ytr[ix])

    # backward pass
    for layer in layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < max_steps / 2 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # stats
    if i % 10000 == 0:
        print(f"{i}: {loss.item()}")

    # if i > 1000:
    break

# do this with the break in the loop above
# visualize tanh spread (should be stretched out)
# print("tanh spread")
# plt.figure(figsize=(16, 4))
# for i, layer in enumerate(layers[:-1]):
#     if isinstance(layer, Tanh):
#         out = layer.out
#         print("layer %d mean %+.2f std %.2f" % (i, out.mean(), out.std()))
#         hy, hx = torch.histogram(out, density=True)
#         plt.plot(hx[:-1].detach(), hy.detach())
# plt.show()


# visualize grads spread (should be a nice slow curve)
# print("grads spread")
# plt.figure(figsize=(16, 4))
# for i, layer in enumerate(layers[:-1]):
#     if isinstance(layer, Tanh):
#         out = layer.out.grad
#         print("layer %d mean %+.2f std %e" % (i, out.mean(), out.std()))
#         hy, hx = torch.histogram(out, density=True)
#         plt.plot(hx[:-1].detach(), hy.detach())
# plt.show()

# done with training
for layer in layers:
    layer.training = False

with torch.no_grad():
    # total loss
    emb = C[Xtr]
    x = emb.view(-1, block_size * emb_size)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Ytr)
    print(f"training set loss = {loss.item()}")

    emb = C[Xdev]
    x = emb.view(-1, block_size * emb_size)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Ydev)
    print(f"test set loss = {loss.item()}")


g = torch.Generator().manual_seed(2147483647 + 10)

# show plot for embeddings only when 2 dims are being used
if emb_size == 2:
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
        x = emb.view(1, -1)
        for layer in layers:
            x = layer(x)
        probs = F.softmax(x, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)

        if ix == 0:
            break
    print("".join(itos[i] for i in out))

# plt.plot(stepi, lossi)
# plt.show()
