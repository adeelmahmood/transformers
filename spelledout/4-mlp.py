import math
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import torch
import random


def f(x):
    return 3 * x**2 + 5 * x - 2


class Value:
    def __init__(self, data, children=(), op="", label="") -> None:
        self.data = data
        self.prev = set(children)
        self.op = op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value({self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self * other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def tanh(self):
        out = Value(np.tanh(self.data), (self,), op="tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def topoSort(node):
            if node not in visited:
                visited.add(node)
                for child in node.prev:
                    topoSort(child)
                topo.append(node)

        topoSort(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        print(v)
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_graph(root):
    graph = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        graph.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n.op:
            graph.node(name=uid + n.op, label=n.op)
            graph.edge(uid + n.op, uid)

    for n1, n2 in edges:
        graph.edge(str(id(n1)), str(id(n2)) + n2.op)

    return graph


class Neuron:
    def __init__(self, inputs) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x) -> Any:
        # wx + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, inputs, output) -> None:
        self.neurons = [Neuron(inputs) for _ in range(output)]

    def __call__(self, x) -> Any:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, inputs, outputs) -> None:
        sz = [inputs] + outputs
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(sz) - 1)]

    def __call__(self, x) -> Any:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


n = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, 1.5],
    [1.0, -1.0, 0.5],
    [-1.0, 0.0, 3.5],
    [1.0, 1.0, 2.5],
]


ys = [1.0, -1.0, -1.0, 1.0]
print(ys)


ypreds = []
count = 60
batch = 0.25
for i in range(count):
    # forward pass
    ypreds = [n(x) for x in xs]

    # compute loss
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypreds))
    if i % int(count * batch) == 0 or i == count - 1:
        print(f"i[{i}] loss[{loss}]")

    # backprop
    for p in n.parameters():
        p.grad = 0
    loss.backward()

    # update parameters based on grad
    for p in n.parameters():
        p.data += -0.05 * p.grad

print(ypreds)
