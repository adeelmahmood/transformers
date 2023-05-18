import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import torch
import random

fullNames = open("./data/names.csv", "r").read().splitlines()
words = [name.split("\t")[0].lower() for name in fullNames]

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

for w in words:
    chs = ["."] + list(w) + ["."]
    for c1, c2 in zip(chs, chs[1:]):
        ix1 = stoi[c1]
        ix2 = stoi[c2]
        N[ix1, ix2] += 1
        # print(c1, c2)


def showPlt():
    plt.figure(figsize=(12, 12))
    plt.imshow(N, cmap="Greens")
    for i in range(27):
        for j in range(27):
            chrstr = itos[i] + itos[j]
            plt.text(j, i, chrstr, va="top", ha="center")
            plt.text(j, i, N[i][j].item(), va="bottom", ha="center")
    plt.axis("off")
    plt.show()


# showPlt()

P = (N + 1).float()
P /= P.sum(1, keepdim=True)

g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    ix = 0
    name = ""
    while True:
        p = P[ix]

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        name += itos[ix]

        if ix == 0:
            break

    print(name)


loglikehood = 0
n = 0
for w in words:
    chs = ["."] + list(w) + ["."]
    for c1, c2 in zip(chs, chs[1:]):
        ix1 = stoi[c1]
        ix2 = stoi[c2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        loglikehood += logprob
        n += 1
        # print(f"{c1,c2}: {prob:.4f}: {logprob:.4f}")
print(f"loss={-loglikehood/n}")
