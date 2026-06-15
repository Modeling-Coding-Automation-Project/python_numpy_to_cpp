/**
 * @file base_matrix_triangular_sparse.hpp
 * @brief Utilities for handling upper and lower triangular sparse matrices in a
 * compile-time optimized manner.
 *
 * This file provides template metaprogramming utilities and classes for
 * constructing and manipulating upper and lower triangular sparse matrices. The
 * implementation supports both compile-time recursion and runtime for-loop
 * approaches (controlled by the BASE_MATRIX_USE_FOR_LOOP_OPERATION_ macro).
 * The code is designed for high-performance scenarios where matrix dimensions
 * are known at compile time.
 *
 * @note
 * tparam M is the number of rows in the matrix.
 * tparam N is the number of columns in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef BASE_MATRIX_TRIANGULAR_SPARSE_HPP_
#define BASE_MATRIX_TRIANGULAR_SPARSE_HPP_

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

/**
 * * @brief Creates an upper triangular sparse matrix with uninitialized values.
 *
 * This function returns a new instance of a compile-time optimized upper
 * triangular sparse matrix. The values are not initialized and should be set
 * using the set_values_UpperTriangularSparseMatrix function or by directly
 * accessing the values array.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @return CompiledSparseMatrix<T, M, N, UpperTriangularCSRIndices<M, N>,
 * UpperTriangularCSRPointers<M, N>> A new upper triangular sparse matrix with
 * uninitialized values.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto create_UpperTriangularSparseMatrix(void)
    -> CompiledSparseMatrix<T, M, N, UpperTriangularCSRIndices<M, N>,
                            UpperTriangularCSRPointers<M, N>> {
  // Currently, only support M >= N.
  static_assert(M >= N, "M must be greater than or equal to N");

  CompiledSparseMatrix<T, M, N, UpperTriangularCSRIndices<M, N>,
                       UpperTriangularCSRPointers<M, N>>
      Y;

  return Y;
}

/**
 * @brief Creates an upper triangular sparse matrix from a dense matrix.
 *
 * This function constructs an upper triangular sparse matrix by extracting the
 * upper triangular part of a given dense matrix A. The resulting sparse
 * matrix will contain only the non-zero elements from the upper triangular
 * portion of A.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param A The input dense matrix from which to create the upper triangular
 * sparse matrix.
 * @return CompiledSparseMatrix<T, M, N, UpperTriangularCSRIndices<M, N>,
 * UpperTriangularCSRPointers<M, N>> An upper triangular sparse matrix
 * containing the non-zero elements from the upper triangular part of A.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto create_UpperTriangularSparseMatrix(const Matrix<T, M, N> &A)
    -> CompiledSparseMatrix<T, M, N, UpperTriangularCSRIndices<M, N>,
                            UpperTriangularCSRPointers<M, N>> {
  // Currently, only support M >= N.
  static_assert(M >= N, "M must be greater than or equal to N");

  CompiledSparseMatrix<T, M, N, UpperTriangularCSRIndices<M, N>,
                       UpperTriangularCSRPointers<M, N>>
      Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  std::size_t consecutive_index = 0;

  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = i; j < N; j++) {
      Y.values[consecutive_index] = A(i, j);

      consecutive_index++;
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  substitute_dense_to_sparse(Y, A);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return Y;
}

/**
 * @brief Sets the values of an upper triangular sparse matrix from a dense
 * matrix.
 *
 * This function updates the values of an existing upper triangular sparse
 * matrix A by extracting the upper triangular part of a given dense matrix B.
 * The non-zero elements from the upper triangular portion of B are assigned to
 * the corresponding positions in A.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param A The upper triangular sparse matrix to be updated with values from B.
 * @param B The input dense matrix from which to extract values for A.
 */
template <typename T, std::size_t M, std::size_t N>
inline void set_values_UpperTriangularSparseMatrix(
    CompiledSparseMatrix<T, M, N, UpperTriangularCSRIndices<M, N>,
                         UpperTriangularCSRPointers<M, N>> &A,
    const Matrix<T, M, N> &B) {
  // Currently, only support M >= N.
  static_assert(M >= N, "M must be greater than or equal to N");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  std::size_t consecutive_index = 0;

  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = i; j < N; j++) {
      A.values[consecutive_index] = B(i, j);
      consecutive_index++;
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  substitute_dense_to_sparse(A, B);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/* Lower */

/**
 * @brief Creates a lower triangular sparse matrix with uninitialized values.
 *
 * This function returns a new instance of a compile-time optimized lower
 * triangular sparse matrix. The values are not initialized and should be set
 * using the set_values_LowerTriangularSparseMatrix function or by directly
 * accessing the values array.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @return CompiledSparseMatrix<T, M, N, LowerTriangularCSRIndices<M, N>,
 * LowerTriangularCSRPointers<M, N>> A new lower triangular sparse matrix with
 * uninitialized values.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto create_LowerTriangularSparseMatrix(void)
    -> CompiledSparseMatrix<T, M, N, LowerTriangularCSRIndices<M, N>,
                            LowerTriangularCSRPointers<M, N>> {
  // Currently, only support M <= N.
  static_assert(M <= N, "M must be smaller than or equal to N");

  CompiledSparseMatrix<T, M, N, LowerTriangularCSRIndices<M, N>,
                       LowerTriangularCSRPointers<M, N>>
      Y;

  return Y;
}

/**
 * @brief Creates a lower triangular sparse matrix from a dense matrix.
 *
 * This function constructs a lower triangular sparse matrix by extracting the
 * lower triangular part of a given dense matrix A. The resulting sparse
 * matrix will contain only the non-zero elements from the lower triangular
 * portion of A.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param A The input dense matrix from which to create the lower triangular
 * sparse matrix.
 * @return CompiledSparseMatrix<T, M, N, LowerTriangularCSRIndices<M, N>,
 * LowerTriangularCSRPointers<M, N>> A lower triangular sparse matrix containing
 * the non-zero elements from the lower triangular part of A.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto create_LowerTriangularSparseMatrix(const Matrix<T, M, N> &A)
    -> CompiledSparseMatrix<T, M, N, LowerTriangularCSRIndices<M, N>,
                            LowerTriangularCSRPointers<M, N>> {
  // Currently, only support M <= N.
  static_assert(M <= N, "M must be smaller than or equal to N");

  CompiledSparseMatrix<T, M, N, LowerTriangularCSRIndices<M, N>,
                       LowerTriangularCSRPointers<M, N>>
      Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  std::size_t consecutive_index = 0;

  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < i + 1; j++) {
      Y.values[consecutive_index] = A(i, j);

      consecutive_index++;
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  substitute_dense_to_sparse(Y, A);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return Y;
}

/**
 * @brief Sets the values of a lower triangular sparse matrix from a dense
 * matrix.
 *
 * This function updates the values of an existing lower triangular sparse
 * matrix A by extracting the lower triangular part of a given dense matrix B.
 * The non-zero elements from the lower triangular portion of B are assigned to
 * the corresponding positions in A.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param A The lower triangular sparse matrix to be updated with values from B.
 * @param B The input dense matrix from which to extract values for A.
 */
template <typename T, std::size_t M, std::size_t N>
inline void set_values_LowerTriangularSparseMatrix(
    CompiledSparseMatrix<T, M, N, LowerTriangularCSRIndices<M, N>,
                         LowerTriangularCSRPointers<M, N>> &A,
    const Matrix<T, M, N> &B) {
  // Currently, only support M <= N.
  static_assert(M <= N, "M must be smaller than or equal to N");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  std::size_t consecutive_index = 0;

  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < i + 1; j++) {
      A.values[consecutive_index] = B(i, j);
      consecutive_index++;
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  substitute_dense_to_sparse(A, B);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_TRIANGULAR_SPARSE_HPP_
