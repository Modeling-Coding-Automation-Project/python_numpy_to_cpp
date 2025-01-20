import numpy as np
import math


def transpose(A):
    # extract elements vector from matrix
    M, N = A.shape

    D = np.zeros((M * N, M * N, M))
    for i in range(M * N):
        D[i, i, math.floor(i / N)] = 1

    D_v = np.zeros((N, N, 1))
    for i in range(N):
        D_v[i, i, 0] = 1

    A_T_vec = np.zeros((M * N, 1))
    k = 0
    for i in range(M):
        for j in range(N):
            A_T_vec += D[k] @ A @ D_v[j]
            k += 1

    # create transposed matrix
    G_v = np.zeros((M, 1, M))
    for i in range(M):
        G_v[i, 0, i] = 1

    G = np.zeros((M * N, N, M * N))
    for i in range(M * N):
        G[i, i % N, i] = 1

    A_T = np.zeros((N, M))
    k = 0
    for i in range(M):
        for j in range(N):
            A_T += G[i * N + j] @ A_T_vec @ G_v[k]
        k += 1

    return A_T


# input: Matrix which you want to transpose
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# A = np.array([[1, 2],
#               [3, 4],
#               [5, 6],
#               [7, 8],
#               [9, 10]])

print("Original Matrix:\n", A)

A_T = transpose(A)
print("Transposed Matrix:\n", A_T)
