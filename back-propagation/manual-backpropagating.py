#!/usr/bin/env python
import math
# Importing the graphviv Digraph library to digram our math problems
from graphviz import Digraph

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
draw_dot(L).render(directory='manual-backpropagation-output')

# Now let's add onto our Value class to do more complex examples:

# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
# - This number was chosen to give simpiler numbers to work with during
#   backpropagation
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

o.grad = 1.0
n.grad = 1 - (o.data**2)
x1w1x2w2.grad = n.grad
b.grad = n.grad
x1w1.grad = x1w1x2w2.grad
x2w2.grad = x1w1x2w2.grad
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad

draw_dot(o).render(directory='manual-backpropagation-output2')

# Manual backpropagation is quite tedious and unfeasible though so let's now
# look at automating the backpropagation part in auto-backpropgating.py
