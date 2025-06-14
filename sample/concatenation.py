"""
This script demonstrates various NumPy array concatenation and tiling operations.
It creates several 3x3 matrices and performs the following operations:
- Vertical concatenation of two matrices using `np.concatenate` along axis 0.
- Horizontal concatenation of two matrices using `np.concatenate` along axis 1.
- Block matrix construction using `np.block` to combine four matrices into a larger matrix.
- Tiling of a matrix using `np.tile` to repeat the matrix in a grid pattern.
"""
import numpy as np


A = np.array([[1., 2., 3.], [5., 4., 6.], [9., 8., 7.]])
B = np.diag([1., 2., 3.])
C = np.array([[1., 0., 0.], [3., 0., 8.], [0., 2., 4.]])
E = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

vertical = np.concatenate([A, C], 0)

print("vertical =")
print(vertical)
print("\n")

horizontal = np.concatenate([A, B], 1)

print("horizontal =")
print(horizontal)
print("\n")


block = np.block([[A, B], [C, E]])

print("block =")
print(block)
print("\n")


tiled_matrix = np.tile(C, (2, 3))
print("tiled_matrix =")
print(tiled_matrix)
print("\n")
