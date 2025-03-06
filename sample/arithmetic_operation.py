import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from python_numpy.numpy_deploy import NumpyDeploy

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

# You can create cpp header which can easily define matrix as C++ code
A_file_name = NumpyDeploy.generate_matrix_cpp_code(A)
print("A file name = " + A_file_name)
B_file_name = NumpyDeploy.generate_matrix_cpp_code(B)
print("B file name = " + B_file_name)
C_file_name = NumpyDeploy.generate_matrix_cpp_code(C)
print("C file name = " + C_file_name)
E_file_name = NumpyDeploy.generate_matrix_cpp_code(E)
print("E file name = " + E_file_name)
