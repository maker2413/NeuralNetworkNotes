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
