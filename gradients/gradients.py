#!/usr/bin/env python
# This file was generated from the code blocks in ./README.org.

import math
import numpy as np
import matplotlib.pyplot as plt

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
