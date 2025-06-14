"""
This script demonstrates solving a system of linear equations using NumPy.
It defines three matrices: A, B, and C. The script then solves the matrix equation A * X = C for X using numpy.linalg.solve,
and prints the resulting matrix. The script also initializes a zero matrix for demonstration purposes.
"""
import numpy as np


Zero = np.zeros((3, 3))

A = np.array([[1., 2., 3.], [5., 4., 6.], [9., 8., 7.]])
B = np.diag([1., 2., 3.])
C = np.array([[1., 0., 0.], [3., 0., 8.], [0., 2., 4.]])

result = np.linalg.solve(A, C)

print("result =")
print(result)
