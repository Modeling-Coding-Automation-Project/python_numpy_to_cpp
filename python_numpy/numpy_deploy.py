import os
import sys
import numpy as np
from enum import Enum


class MatrixType(Enum):
    DENSE = 1
    DIAG = 2
    SPARSE = 3


class NumpyDeploy:
    def __init__(self):
        pass

    @staticmethod
    def generate_matrix_cpp_code(matrix_in):
        matrix = matrix_in

        if len(matrix_in.shape) >= 3:
            raise ValueError("Over 3d arrays are not supported.")
        elif len(matrix_in.shape) == 1:
            matrix = matrix_in.reshape(matrix_in.shape[0], 1)

        return NumpyDeploy.judge_matrix_type(matrix)

    @staticmethod
    def judge_matrix_type(matrix):
        diag_flag = True
        sparse_flag = False
        dense_flag = True

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                zero_flag = False

                if matrix[i][j] == 0.0:
                    zero_flag = True
                    sparse_flag = True
                    dense_flag = False
                else:
                    if i != j:
                        diag_flag = False

                if zero_flag:
                    if i == j:
                        diag_flag = False

        if dense_flag:
            return MatrixType.DENSE
        elif diag_flag:
            return MatrixType.DIAG
        elif sparse_flag:
            return MatrixType.SPARSE
        else:
            return MatrixType.DENSE


a = np.array([1., 2., 3.])
A = np.array([[1., 2., 3.], [5., 4., 6.], [9., 8., 7.]])
B = np.diag([1., 2., 3.])
C = np.array([[1., 0., 0.], [3., 0., 8.], [0., 2., 4.]])


print(NumpyDeploy.generate_matrix_cpp_code(C))
