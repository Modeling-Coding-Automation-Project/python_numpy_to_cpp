import os
import sys
import numpy as np
from enum import Enum
import inspect

python_to_cpp_types = {
    'int8': 'char',
    'int16': 'short',
    'int32': 'long',
    'int64': 'long long',
    'uint8': 'unsigned char',
    'uint16': 'unsigned short',
    'uint32': 'unsigned long',
    'uint64': 'unsigned long long',
    'float32': 'float',
    'float64': 'double'
}


class MatrixType(Enum):
    DENSE = 1
    DIAG = 2
    SPARSE = 3
    SPARSE_EMPTY = 4


class NumpyDeploy:
    def __init__(self):
        pass

    @staticmethod
    def check_dtype(matrix):
        if matrix.dtype.name not in python_to_cpp_types:
            raise ValueError("Unsupported data type: " +
                             str(matrix.dtype.name))

        return python_to_cpp_types[matrix.dtype.name]

    @staticmethod
    def judge_matrix_type(matrix):
        diag_flag = True
        sparse_flag = False
        sparse_empty_flag = True
        dense_flag = True

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                zero_flag = False

                if matrix[i, j] == 0.0:
                    zero_flag = True
                    sparse_flag = True
                    dense_flag = False
                else:
                    sparse_empty_flag = False
                    if i != j:
                        diag_flag = False

                if zero_flag:
                    if i == j:
                        diag_flag = False

        if matrix.shape[0] != matrix.shape[1]:
            diag_flag = False

        if sparse_empty_flag:
            return MatrixType.SPARSE_EMPTY
        elif dense_flag:
            return MatrixType.DENSE
        elif diag_flag:
            return MatrixType.DIAG
        elif sparse_flag:
            return MatrixType.SPARSE
        else:
            return MatrixType.DENSE

    @staticmethod
    def value_to_string_with_type(value, type_name):
        return "static_cast<" + type_name + ">(" + str(value) + ")"

    @staticmethod
    def generate_matrix_cpp_code(matrix_in):
        matrix = matrix_in

        if len(matrix_in.shape) >= 3:
            raise ValueError("Over 3d arrays are not supported.")
        elif len(matrix_in.shape) == 1:
            matrix = matrix_in.reshape(matrix_in.shape[0], 1)

        matrix_type = NumpyDeploy.judge_matrix_type(matrix)

        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is matrix_in:
                variable_name = name
                break

        code_file_name = "python_numpy_gen_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        type_name = NumpyDeploy.check_dtype(matrix)
        sparse_available_name = "SparseAvailable_" + variable_name

        # Write cpp code
        code_text = ""
        code_text += "#ifndef __PYTHON_NUMPY_GEN_" + variable_name.upper() + "_HPP__\n"
        code_text += "#define __PYTHON_NUMPY_GEN_" + \
            variable_name.upper() + "_HPP__\n\n"

        code_text += "#include \"python_numpy.hpp\"\n\n"

        namespace_name = "python_numpy_gen_" + variable_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n\n"

        if matrix_type == MatrixType.DENSE:
            code_text += "using type = " + "DenseMatrix_Type<" + \
                type_name + ", " + \
                str(matrix.shape[0]) + ", " + str(matrix.shape[1]) + ">;\n\n"

        elif matrix_type == MatrixType.DIAG:
            code_text += "using type = " + "DiagMatrix_Type<" + \
                type_name + ", " + \
                str(matrix.shape[0]) + ">;\n\n"

        elif matrix_type == MatrixType.SPARSE or matrix_type == MatrixType.SPARSE_EMPTY:
            code_text += "using " + sparse_available_name + " = SparseAvailable<\n"

            for i in range(matrix.shape[0]):
                code_text += "    ColumnAvailable<"
                for j in range(matrix.shape[1]):
                    if matrix[i, j] != 0:
                        code_text += "true"
                    else:
                        code_text += "false"
                    if j != matrix.shape[1] - 1:
                        code_text += ", "
                if i == matrix.shape[0] - 1:
                    code_text += ">\n"
                else:
                    code_text += ">,\n"

            code_text += ">;\n\n"

            code_text += "using type = " + "SparseMatrix_Type<" + \
                type_name + ", " + \
                sparse_available_name + ">;\n\n"

        code_text += "inline auto make(void) -> type {\n\n"

        if matrix_type == MatrixType.DENSE:
            code_text += "  return make_DenseMatrix<" + \
                str(matrix.shape[0]) + ", " + str(matrix.shape[1]) + ">(\n"

            for i in range(matrix.shape[0]):
                code_text += "    "
                for j in range(matrix.shape[1]):
                    if i == matrix.shape[0] - 1 and j == matrix.shape[1] - 1:
                        code_text += NumpyDeploy.value_to_string_with_type(
                            matrix[i, j], type_name)
                    else:
                        code_text += NumpyDeploy.value_to_string_with_type(
                            matrix[i, j], type_name)

                    if j != matrix.shape[1] - 1:
                        code_text += ", "
                    elif i != matrix.shape[0] - 1:
                        code_text += ","

                code_text += "\n"

            code_text += "  );\n\n"

        elif matrix_type == MatrixType.DIAG:
            code_text += "  return make_DiagMatrix<" + \
                str(matrix.shape[0]) + ">(\n"

            for i in range(matrix.shape[0]):
                code_text += "    "
                if i == matrix.shape[0] - 1:
                    code_text += NumpyDeploy.value_to_string_with_type(
                        matrix[i, i], type_name) + "\n"
                else:
                    code_text += NumpyDeploy.value_to_string_with_type(
                        matrix[i, i], type_name) + ",\n"

            code_text += "  );\n\n"

        elif matrix_type == MatrixType.SPARSE:
            code_text += "  return make_SparseMatrix<" + \
                sparse_available_name + ">(\n"

            sparse_count = 0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if matrix[i, j] != 0:
                        if sparse_count == 0:
                            code_text += "    " + NumpyDeploy.value_to_string_with_type(
                                matrix[i, j], type_name)
                            sparse_count += 1
                        else:
                            code_text += ",\n    " + NumpyDeploy.value_to_string_with_type(
                                matrix[i, j], type_name)

            code_text += "\n  );\n\n"

        elif matrix_type == MatrixType.SPARSE_EMPTY:
            code_text += "  return make_SparseMatrixEmpty<" + \
                type_name + ", " + \
                str(matrix.shape[0]) + ", " + str(matrix.shape[1]) + ">();\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // __PYTHON_NUMPY_GEN_" + variable_name.upper() + \
            "_HPP__\n"

        # write to file
        with open(code_file_name_ext, "w") as f:
            f.write(code_text)

        return code_file_name_ext
