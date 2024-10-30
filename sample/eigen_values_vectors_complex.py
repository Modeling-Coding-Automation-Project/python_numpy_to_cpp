import numpy as np

A_c = np.array([[1., 2., 3.], [3., 1., 2.], [2., 3., 1.]])

eigen_values_c, eigen_vectors_c = np.linalg.eig(A_c)

print("eigen_values_c =")
print(eigen_values_c)

print("eigen_vectors_c =")
print(eigen_vectors_c)

result = A_c @ eigen_vectors_c - eigen_vectors_c @ np.diag(eigen_values_c)
print("result =")
print(result)
