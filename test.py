import numpy as np
import math


def transpose(A):
    # extract elements vector from matrix
    M, N = A.shape

    D = np.zeros((M * N, M * N, M))
    for i in range(M * N):
        D[i, i, i % M] = 1

    D_v = np.zeros((N, N, 1))
    for i in range(N):
        D_v[i, i, 0] = 1

    A_vec = np.zeros((M * N, 1))
    k = 0
    for i in range(N):
        for j in range(M):
            A_vec += D[k] @ A @ D_v[i]
            k += 1

    # arrange elements vector
    T = np.zeros((M * N, M * N))
    for i in range(M * N):
        for j in range(M * N):
            q, r = divmod(M * i, M * N)
            if q + r == j:
                T[i, j] = 1

    A_T_vec = T @ A_vec

    # create transposed matrix
    G_v = np.zeros((M, 1, M))
    for i in range(M):
        G_v[i, 0, i] = 1

    A_T_mat = np.zeros((M, M * N, M))
    for i in range(M):
        A_T_mat[i] = A_T_vec @ G_v[i]

    G = np.zeros((M * N, N, M * N))
    for i in range(M * N):
        G[i, i % N, i] = 1

    A_T = np.zeros((N, M))
    k = 0
    for i in range(M):
        for j in range(N):
            A_T += G[i * N + j] @ A_T_mat[k]
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
