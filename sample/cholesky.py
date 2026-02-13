"""
This script demonstrates the use of NumPy to perform Cholesky decomposition on a symmetric positive-definite matrix.
It defines three matrices (A, B, and C), computes the Cholesky factorization of matrix A, and verifies the result by
multiplying the Cholesky factor with its transpose to reconstruct the original matrix. The script prints both the
Cholesky factor and the reconstructed matrix for validation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np


A = np.array([[2., 1., 3.], [1., 4., 2.], [3., 2., 6.]])
B = np.diag([1., 2., 3.])
C = np.array([[1., 0., 0.], [0., 8., 3.], [0., 3., 4.]])

SA = np.linalg.cholesky(A)

print("SA =")
print(SA)

result = SA @ SA.T

print("result =")
print(result)
