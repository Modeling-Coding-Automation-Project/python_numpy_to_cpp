"""
Demonstrates one-dimensional convolution in full, same, and valid modes.

The inputs and expected results match sample/convolve.cpp, allowing NumPy to
be compared directly with the PythonNumpy C++ functions.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np


a = np.array([1., 2., 3., 4., 5.])
v = np.array([1., 1., 1.])

print("a =", a)
print("v =", v)
print("full =", np.convolve(a, v, mode="full"))
print("same =", np.convolve(a, v, mode="same"))
print("valid =", np.convolve(a, v, mode="valid"))
