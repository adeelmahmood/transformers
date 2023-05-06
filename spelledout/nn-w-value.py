import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


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

x1w1 = x1 * w1
x1w1.label = "x1w1"
x2w2 = x2 * w2
x2w2.label = "x2w2"
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1w1x2w2"

n = x1w1x2w2 + b
n.label = "n"

o = n.tanh()
o.label = "o"

# set gradients
o.backward()

draw_graph(o).render()


def lol():
    h = 0.001

    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    d = a * b
    e = d + c
    f = Value(-2.0, label="f")
    L = e * f
    L1 = L.data

    a = Value(2.0, label="a")
    b = Value(-3.0 + h, label="b")
    c = Value(10.0, label="c")
    d = a * b
    e = d + c
    f = Value(-2.0, label="f")
    L = e * f
    L2 = L.data

    print((L2 - L1) / h)


lol()
