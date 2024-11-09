#ifndef BASE_MATRIX_HPP
#define BASE_MATRIX_HPP

#include "base_matrix_cholesky_decomposition.hpp"
#include "base_matrix_complex.hpp"
#include "base_matrix_concatenate.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_eigen_solver.hpp"
#include "base_matrix_inverse.hpp"
#include "base_matrix_lu_decomposition.hpp"
#include "base_matrix_macros.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_qr_decomposition.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_triangular_sparse.hpp"
#include "base_matrix_utility.hpp"
#include "base_matrix_variable_sparse.hpp"
#include "base_matrix_vector.hpp"

/* Remove compiled matrix operation macros */
#undef BASE_MATRIX_COMPILED_MATRIX_MULTIPLY
#undef BASE_MATRIX_COMPILED_VECTOR_MULTIPLY_SCALAR
#undef BASE_MATRIX_COMPILED_VECTOR_ADD_SCALAR
#undef BASE_MATRIX_COMPILED_VECTOR_SUB_SCALAR
#undef BASE_MATRIX_COMPILED_SCALAR_SUB_VECTOR
#undef BASE_MATRIX_COMPILED_VECTOR_ADD_VECTOR
#undef BASE_MATRIX_COMPILED_VECTOR_MULTIPLY_SCALAR
#undef BASE_MATRIX_COMPILED_COMPLEX_VECTOR_NORM
#undef BASE_MATRIX_COMPILED_GET_REAL_FROM_COMPLEX_VECTOR
#undef BASE_MATRIX_COMPILED_GET_IMAG_FROM_COMPLEX_VECTOR
#undef BASE_MATRIX_COMPILED_MATRIX_ADD_MATRIX
#undef BASE_MATRIX_COMPILED_MATRIX_SUB_MATRIX
#undef BASE_MATRIX_COMPILED_MATRIX_MINUS_MATRIX

#endif // BASE_MATRIX_HPP
