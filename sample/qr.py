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
