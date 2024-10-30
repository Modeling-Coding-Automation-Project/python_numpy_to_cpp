import numpy as np

A = np.array([[1., 2., 3.], [5., 4., 6.], [9., 8., 7.]])
A_s = np.array([[1., 2., 3.], [5., 4., 6.], [5., 4., 6.]])
B = np.diag([1., 2., 3.])
C = np.array([[1., 0., 0.], [3., 0., 8.], [0., 2., 4.]])

result = np.linalg.inv(A)

print("result =")
print(result)
