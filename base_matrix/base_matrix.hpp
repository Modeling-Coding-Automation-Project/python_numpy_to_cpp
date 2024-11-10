#ifndef BASE_MATRIX_HPP
#define BASE_MATRIX_HPP

#include "base_matrix_cholesky_decomposition.hpp"
#include "base_matrix_compiled_sparse.hpp"
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
#undef BASE_MATRIX_COMPILED_DIAG_MATRIX_ADDER
#undef BASE_MATRIX_COMPILED_DIAG_MATRIX_ADD_MATRIX
#undef BASE_MATRIX_COMPILED_DIAG_MATRIX_SUBTRACTOR
#undef BASE_MATRIX_COMPILED_DIAG_MATRIX_SUB_MATRIX
#undef BASE_MATRIX_COMPILED_MATRIX_SUB_DIAG_MATRIX
#undef BASE_MATRIX_COMPILED_DIAG_MATRIX_MULTIPLY_SCALAR
#undef BASE_MATRIX_COMPILED_DIAG_MATRIX_MULTIPLY_VECTOR
#undef BASE_MATRIX_COMPILED_DIAG_MATRIX_MULTIPLY_DIAG
#undef BASE_MATRIX_COMPILED_DIAG_MATRIX_MULTIPLY_MATRIX
#undef BASE_MATRIX_COMPILED_MATRIX_MULTIPLY_DIAG_MATRIX
#undef BASE_MATRIX_COMPILED_DIAG_TRACE_CALCULATOR
#undef BASE_MATRIX_COMPILED_DIAG_MATRIX_TO_DENSE
#undef BASE_MATRIX_COMPILED_DIAG_MATRIX_DIVIDER
#undef BASE_MATRIX_COMPILED_DIAG_INV_MULTIPLY_DENSE
#undef BASE_MATRIX_COMPILED_SPARSE_MATRIX_CREATE_DENSE

#endif // BASE_MATRIX_HPP
