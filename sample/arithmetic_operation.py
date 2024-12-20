import numpy as np

A = np.array([[1., 2., 3.], [5., 4., 6.], [9., 8., 7.]])
B = np.diag([1., 2., 3.])
C = np.array([[1., 0., 0.], [3., 0., 8.], [0., 2., 4.]])

result = A @ C

print("result =")
print(result)

# Create Zero Matrix
E = np.zeros((3, 3))

# Concatenate Two Matrices
concat_A_B = np.concatenate((A, B), axis=0)

print("concat_A_B =")
print(concat_A_B)

concat_E_C = np.concatenate((E, C), axis=1)

print("concat_E_C =")
print(concat_E_C)
