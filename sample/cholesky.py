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
