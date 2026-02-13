"""
This script demonstrates how to compute the eigenvalues and eigenvectors of a real square matrix using NumPy.
It calculates the eigenvalues and eigenvectors of a predefined 3x3 matrix, prints them, and verifies the result
by checking the equation A * v = v * D, where A is the original matrix, v is the matrix of eigenvectors, and D
is the diagonal matrix of eigenvalues.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

A_c = np.array([[1., 2., 3.], [3., 1., 2.], [2., 3., 1.]])

eigen_values_c, eigen_vectors_c = np.linalg.eig(A_c)

print("eigen_values_c =")
print(eigen_values_c)

print("eigen_vectors_c =")
print(eigen_vectors_c)

result = A_c @ eigen_vectors_c - eigen_vectors_c @ np.diag(eigen_values_c)
print("result =")
print(result)
