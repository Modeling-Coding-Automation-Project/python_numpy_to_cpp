import os
import sys
import numpy as np


class NumpyDeploy:
    def __init__(self):
        pass

    @staticmethod
    def generate_ndarray_cpp_code(ndarray):

        if len(ndarray.shape) >= 3:
            raise ValueError("Over 3d arrays are not supported.")

        return ndarray


A =  np.array([[1., 2., 3.], [5., 4., 6.], [9., 8., 7.]])

NumpyDeploy.generate_ndarray_cpp_code(A)

