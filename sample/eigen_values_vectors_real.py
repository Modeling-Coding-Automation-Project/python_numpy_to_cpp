"""
This script demonstrates how to compute the eigenvalues and eigenvectors of a real square matrix using NumPy.
It defines a 3x3 real matrix, calculates its eigenvalues and eigenvectors, and prints them.
Additionally, it verifies the eigen decomposition by computing the difference between A @ V and V @ D,
where A is the original matrix, V is the matrix of eigenvectors, and D is the diagonal matrix of eigenvalues.
"""
import numpy as np


A_r = np.array([[6., -3., 5.], [-1., 4., -5.], [-3., 3., -4.]])

eigen_values_r, eigen_vectors_r = np.linalg.eig(A_r)

print("eigen_values_r =")
print(eigen_values_r)

print("eigen_vectors_r =")
print(eigen_vectors_r)

result = A_r @ eigen_vectors_r - eigen_vectors_r @ np.diag(eigen_values_r)
print("result =")
print(result)
