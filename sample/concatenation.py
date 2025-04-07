import numpy as np

A = np.array([[1., 2., 3.], [5., 4., 6.], [9., 8., 7.]])
B = np.diag([1., 2., 3.])
C = np.array([[1., 0., 0.], [3., 0., 8.], [0., 2., 4.]])
E = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

vertical = np.concatenate([A, C], 0)

print("vertical =")
print(vertical)
print("\n")

horizontal = np.concatenate([A, B], 1)

print("horizontal =")
print(horizontal)
print("\n")


block = np.block([[A, B], [C, E]])

print("block =")
print(block)
print("\n")


tiled_matrix = np.tile(C, (2, 3))
print("tiled_matrix =")
print(tiled_matrix)
print("\n")
