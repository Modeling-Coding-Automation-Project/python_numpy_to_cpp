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
