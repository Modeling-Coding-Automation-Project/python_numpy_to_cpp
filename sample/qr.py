"""
This script demonstrates the use of NumPy to perform QR decomposition on a given matrix.
It defines three matrices (A, B, and C), computes the QR decomposition of matrix A,
and prints the resulting orthogonal matrix Q and upper triangular matrix R.
Finally, it verifies the decomposition by multiplying Q and R to reconstruct the original matrix A.
"""
import numpy as np


A = np.array([[1., 2., 3.], [5., 4., 6.], [9., 8., 7.]])
B = np.diag([1., 2., 3.])
C = np.array([[1., 0., 0.], [3., 0., 8.], [0., 2., 4.]])

Q, R = np.linalg.qr(A)

print("Q =")
print(Q)

print("R =")
print(R)

result = Q @ R
print("result =")
print(result)
