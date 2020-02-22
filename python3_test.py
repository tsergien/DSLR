#!/usr/bin/env python3

import numpy as np

w = np.zeros((3, 2))
w2 = [[0, 0], [0, 0], [0, 0]]
x = [1, 0]

print(type(w))
print(type(x))

print(np.dot(w, x))

