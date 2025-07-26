/**
 * @file base_matrix_triangular_sparse.hpp
 * @brief Utilities for handling upper and lower triangular sparse matrices in a
 * compile-time optimized manner.
 *
 * This file provides template metaprogramming utilities and classes for
 * constructing and manipulating upper and lower triangular sparse matrices. The
 * implementation supports both compile-time recursion and runtime for-loop
 * approaches (controlled by the __BASE_MATRIX_USE_FOR_LOOP_OPERATION__ macro).
 * The code is designed for high-performance scenarios where matrix dimensions
 * are known at compile time.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __BASE_MATRIX_TRIANGULAR_SPARSE_HPP__
#define __BASE_MATRIX_TRIANGULAR_SPARSE_HPP__

#include "base_matrix_macros.hpp"

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_templates.hpp"
#include "base_matrix_vector.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace Base {
namespace Matrix {

template <typename T, std::size_t M, std::size_t N>
inline auto create_UpperTriangularSparseMatrix(void)
    -> CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                            UpperTriangularRowPointers<M, N>> {
  // Currently, only support M >= N.
  static_assert(M >= N, "M must be greater than or equal to N");

  CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                       UpperTriangularRowPointers<M, N>>
      Y;

  return Y;
}

template <typename T, std::size_t M, std::size_t N>
inline auto create_UpperTriangularSparseMatrix(const Matrix<T, M, N> &A)
    -> CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                            UpperTriangularRowPointers<M, N>> {
  // Currently, only support M >= N.
  static_assert(M >= N, "M must be greater than or equal to N");

  CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                       UpperTriangularRowPointers<M, N>>
      Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  std::size_t consecutive_index = 0;

  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = i; j < N; j++) {
      Y.values[consecutive_index] = A(i, j);

      consecutive_index++;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  substitute_dense_to_sparse(Y, A);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

template <typename T, std::size_t M, std::size_t N>
inline void set_values_UpperTriangularSparseMatrix(
    CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                         UpperTriangularRowPointers<M, N>> &A,
    const Matrix<T, M, N> &B) {
  // Currently, only support M >= N.
  static_assert(M >= N, "M must be greater than or equal to N");

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  std::size_t consecutive_index = 0;

  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = i; j < N; j++) {
      A.values[consecutive_index] = B(i, j);
      consecutive_index++;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  substitute_dense_to_sparse(A, B);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

/* Lower */

template <typename T, std::size_t M, std::size_t N>
inline auto create_LowerTriangularSparseMatrix(void)
    -> CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                            LowerTriangularRowPointers<M, N>> {
  // Currently, only support M <= N.
  static_assert(M <= N, "M must be smaller than or equal to N");

  CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                       LowerTriangularRowPointers<M, N>>
      Y;

  return Y;
}

template <typename T, std::size_t M, std::size_t N>
inline auto create_LowerTriangularSparseMatrix(const Matrix<T, M, N> &A)
    -> CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                            LowerTriangularRowPointers<M, N>> {
  // Currently, only support M <= N.
  static_assert(M <= N, "M must be smaller than or equal to N");

  CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                       LowerTriangularRowPointers<M, N>>
      Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  std::size_t consecutive_index = 0;

  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < i + 1; j++) {
      Y.values[consecutive_index] = A(i, j);

      consecutive_index++;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  substitute_dense_to_sparse(Y, A);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

template <typename T, std::size_t M, std::size_t N>
inline void set_values_LowerTriangularSparseMatrix(
    CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                         LowerTriangularRowPointers<M, N>> &A,
    const Matrix<T, M, N> &B) {
  // Currently, only support M <= N.
  static_assert(M <= N, "M must be smaller than or equal to N");

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  std::size_t consecutive_index = 0;

  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < i + 1; j++) {
      A.values[consecutive_index] = B(i, j);
      consecutive_index++;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  substitute_dense_to_sparse(A, B);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_TRIANGULAR_SPARSE_HPP__
