import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import torch


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
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

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


# inputs
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
b = Value(6.8813, label="b")

x1 = torch.tensor([2.0]).double()
x1.requires_grad = True

x2 = torch.tensor([0.0]).double()
x2.requires_grad = True

w1 = torch.tensor([-3.0]).double()
w1.requires_grad = True

w2 = torch.tensor([1.0]).double()
w2.requires_grad = True

b = torch.tensor([6.8813]).double()
b.requires_grad = True

n = x1 * w1 + x2 * w2 + b
o = torch.tanh(n)

print(o.data.item())

# set gradients
o.backward()

print(x1.grad.item())
print(w1.grad.item())
print(x2.grad.item())
print(w2.grad.item())

# draw_graph(o).render()
