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

    def __repr__(self) -> str:
        return f"Value({self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out


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
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
d = a * b
d.label = "d"
e = d + c
e.label = "e"
f = Value(-2.0, label="f")
L = e * f
L.label = "L"

# set gradients
L.grad = 1.0
e.grad = f.data
f.grad = e.data
c.grad = e.grad
d.grad = e.grad
b.grad = a.data * d.grad
a.grad = b.data * d.grad

draw_graph(L).render()


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
