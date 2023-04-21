#!/usr/bin/env python
# Same libraries as before:
import math
import random
from graphviz import Digraph

# Same trace function as before:
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

# Same draw_dot function as before:
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

# Please refer to auto-backpropagation.py for some insight into what this
# class does.
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other): # other + self
        return self + other

    def __rmul__(self, other): # other * self
        return self * other

    def __rtruediv__(self, other): # other / self
        return other / self.data

    def __rpow__(self, other): # other**self
        return other**self.data

    def __rsub__(self, other): # other - self
        return other - self.data

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # build a topologic graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuron:
    # nin = Number of inputs
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

# Let's create a layer of Neurons
class Layer:
    # nout = Number of output Neurons
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

# Let's create an MLP (Multi Layer Perceptron)
class MLP:
    # nouts = list of nout
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])

# We can draw a graph of our MLP:
draw_dot(n(x)).render(directory='micrograd-output')

# Now let's look at a practical example to piece all of this together:
# Let's create the following data set which will contain 4 sets of inputs for
# our neural network
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
# And ys will be the 4 results we would like our Neural network to output given
# each set of inputs
ys = [1.0, -1.0, -1.0, 1.0] # desired targets

# ypred will be what our Neural Network currently outputs for each of these
# examples
ypred = [n(x) for x in xs]
print('ypred:', ypred)

# The loss is a concept in deep learning that represents how far off we are from
# our desired targets.
# ygt = y ground truth
# yout = y outputs
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
print('loss:', loss)

# Now that we have computed our loss we can backpropagate
loss.backward()

# With our backpropagation complete we can output a graph of our new network:
draw_dot(n(x)).render(directory='micrograd-output2')

# Then we can adjust our parameters by their gradient to slowly tweak our
# Neurons to get a smaller loss and thus closer to our desired outputs:
for p in n.parameters():
    p.data += -0.01 * p.grad

# After making these adjustments it is important to recalculate our loss and
# backpropgrate our new gradient values:
ypred = [n(x) for x in xs]
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
print('loss:', loss)

# Don't forget to zero out your gradients before backpropagating!
for p in n.parameters():
    p.grad = 0.0

# After zeroing our gradients we can backpropagate and output a new graph
loss.backward()
draw_dot(n(x)).render(directory='micrograd-output3')

# Now as fun as it is to iterate over these steps over and over until we reach
# our desired outputs let's put this in a loop:
for k in range(100):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.04 * p.grad

    print(k, loss.data)

# Now let's see how close we got to our desired outputs:
print('ypred:', ypred)
draw_dot(n(x)).render(directory='micrograd-output4')
