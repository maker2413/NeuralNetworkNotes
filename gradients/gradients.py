#!/usr/bin/env python
# This file was generated from the code blocks in ./README.org.

import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

def f(x):
    return 3*x**2 - 4*x + 5

f(3.0)

xs = np.arange(-5, 5, 0.25)
xs

ys = f(xs)
ys

plt.plot(xs, ys)

h = 0.0001
x = 3.0
f(x + h)

(f(x + h) - f(x))/h

a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d)

h = 0.0001

# inputs
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
a += h
d2 = a*b + c

print('d1:', d1)
print('d2:', d2)
print('slope:', (d2 - d1)/h)

h = 0.0001

# inputs
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
b += h
d2 = a*b + c

print('d1:', d1)
print('d2:', d2)
print('slope:', (d2 - d1)/h)

h = 0.0001

# inputs
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
c += h
d2 = a*b + c

print('d1:', d1)
print('d2:', d2)
print('slope:', (d2 - d1)/h)

# trace pieces together all of the nodes in our math problems
def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

# draw_dot is used to draw a digram of our math problems from a root node
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name = uid, label = "{ %s | data %.4f | grad %.4f }" % (
                n.label,
                n.data,
                n.grad
            ),
            shape='record'
        )
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

# Our Value class implements logic similar to a Tensor class found in PyTorch
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        return out

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

# Manually back propagating the gradients for each node.
# for information on how this is done:
# https://en.wikipedia.org/wiki/Derivative#Rules_of_computation
L.grad = 1.0
f.grad = d.data * L.grad
d.grad = f.data * L.grad
c.grad = d.grad
e.grad = d.grad
a.grad = b.data * e.grad
b.grad = a.data * e.grad

# When backpropagating gradients manually:
# - The root node always has a gradient of 1. This is due to the fact that
#   increasing or decreasing the value of the root node directly effects our
#   answer by the amount increased or decreased.
# - When multiplying two nodes together the gradient of one node is equal to
#   the value of the other node multiplied by the gradient of their product.
# - When adding two nodes together the gradient of each node will be equal to
#   the gradient of their sum. This is because increasing or decreasing the
#   value of either node in the addition will directly effect the sum.
# - When using hyperbolic functions you can reference the Derivatives section
#   of the wikipedia page on hyberbolic functions:
#   https://en.wikipedia.org/wiki/Hyperbolic_functions

# With the gradients dictated, when increasing any number with a positive
# gradient will increase the value of L and increasing any number with a
# negative gradient will decrease the value of L.

# Let's draw our backpropagated problem at this point:
draw_dot(L)
