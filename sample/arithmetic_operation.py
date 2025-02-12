import numpy as np

A = np.array([[1., 2., 3.], [5., 4., 6.], [9., 8., 7.]])
B = np.diag([1., 2., 3.])
C = np.array([[1., 0., 0.], [3., 0., 8.], [0., 2., 4.]])

result = A @ C

print("result =")
print(result)
print("\n")

# Create Zero Matrix
E = np.zeros((3, 3))

# Transpose
print("A^T =")
print(A.T)
print("\n")

# Transpose multiply

print("A^T * C =")
print(A.T @ C)
print("\n")

print("A * C^T =")
print(A @ C.T)
print("\n")
