"""
This script demonstrates basic matrix manipulations using NumPy. It includes:
- Reshaping a 4x4 matrix into an 8x2 matrix using column-major (Fortran-style) order.
- Substituting a 3x3 matrix into a submatrix of a 4x4 zero matrix.
"""
import numpy as np


# reshape
A = np.array([[16, 2, 3, 13],
              [5, 11, 10, 8],
              [9, 7, 6, 12],
              [4, 14, 15, 1]])

B = A.reshape(8, 2, order='F')
print(B)

# substitute
A = np.zeros((4, 4))
C = np.array([[10, 0, 0],
              [50, 0, 60],
              [0, 80, 70]])

A[1:4, 1:4] = C
print(A)
