#!/usr/bin/env python3

import numpy as np

w = np.zeros((3, 2))
w[0][0] = 1
x = [1, 0]
x_ = np.insert(x, 0, 1, axis=0)

print(np.dot(w, x))

