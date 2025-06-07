import numpy as np

# reshape
A = np.array([[16, 2, 3, 13],
              [5, 11, 10, 8],
              [9, 7, 6, 12],
              [4, 14, 15, 1]])

B = A.reshape(8, 2, order='F')
print(B)

# substitute
A = np.zeros((4, 4))
C = np.array([[10, 0, 0],
              [50, 0, 60],
              [0, 80, 70]])

A[1:4, 1:4] = C
print(A)
