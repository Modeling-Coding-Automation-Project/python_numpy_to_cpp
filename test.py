import numpy as np
import math

# input: Matrix which you want to transpose
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Original Matrix:\n", A)

#
D_0 = np.zeros((6, 2))
D_0[0, 0] = 1
D_1 = np.zeros((6, 2))
D_1[1, 1] = 1
D_2 = np.zeros((6, 2))
D_2[2, 0] = 1
D_3 = np.zeros((6, 2))
D_3[3, 1] = 1
D_4 = np.zeros((6, 2))
D_4[4, 0] = 1
D_5 = np.zeros((6, 2))
D_5[5, 1] = 1

D_v_0 = np.array([[1, 0, 0]]).T
D_v_1 = np.array([[0, 1, 0]]).T
D_v_2 = np.array([[0, 0, 1]]).T

A_vec = D_0 @ A @ D_v_0 + D_1 @ A @ D_v_0 + D_2 @ A @ D_v_1 + \
    D_3 @ A @ D_v_1 + D_4 @ A @ D_v_2 + D_5 @ A @ D_v_2

T = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        if i < 3:
            if (2 * i) == j:
                T[i, j] = 1
        else:
            if (2 * (i - 3) + 1) == j:
                T[i, j] = 1

A_T_vec = T @ A_vec

G_v_0 = np.array([[1, 0]])
G_v_1 = np.array([[0, 1]])

A_T_vec_0 = A_T_vec @ G_v_0
A_T_vec_1 = A_T_vec @ G_v_1

G_0 = np.zeros((3, 6))
G_0[0, 0] = 1

G_1 = np.zeros((3, 6))
G_1[1, 1] = 1

G_2 = np.zeros((3, 6))
G_2[2, 2] = 1

G_3 = np.zeros((3, 6))
G_3[0, 3] = 1

G_4 = np.zeros((3, 6))
G_4[1, 4] = 1

G_5 = np.zeros((3, 6))
G_5[2, 5] = 1

A_T = G_0 @ A_T_vec_0 + G_1 @ A_T_vec_0 + G_2 @ A_T_vec_0 + \
    G_3 @ A_T_vec_1 + G_4 @ A_T_vec_1 + G_5 @ A_T_vec_1

print("Transposed Matrix:\n", A_T)
