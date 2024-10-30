import numpy as np

Zero = np.zeros((3, 3))

A = np.array([[1., 2., 3.], [5., 4., 6.], [9., 8., 7.]])
B = np.diag([1., 2., 3.])
C = np.array([[1., 0., 0.], [3., 0., 8.], [0., 2., 4.]])

result = np.linalg.solve(A, C)

print("result =")
print(result)
