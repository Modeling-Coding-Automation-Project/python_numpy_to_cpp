/********************************************************************************
@file base_matrix_compiled_sparse_operation.hpp
@brief
This file provides a comprehensive set of template-based operations for sparse
matrix arithmetic in C++. It supports various combinations of sparse, dense, and
diagonal matrices and vectors, including addition, subtraction, multiplication,
and transpose operations. The implementation is highly generic and leverages
template metaprogramming for compile-time recursion and efficient code
generation. Both recursive template and for-loop based implementations are
provided, controlled by macros.

@details
The file defines operations for:
- Sparse matrix negation, addition, subtraction, and scalar multiplication.
- Sparse matrix multiplication with dense matrices, diagonal matrices, and
vectors.
- Dense matrix and diagonal matrix arithmetic with sparse matrices.
- Sparse matrix multiplication with other sparse matrices, including transposed
cases.
- Setting diagonal matrix values into sparse matrices.
- Utility structures for compile-time calculation of result matrix types.

@note
tparam M is the number of columns in the matrix.
tparam N is the number of rows in the matrix.
Somehow programming custom is vice versa,
but in this project, we use the mathematical custom.
********************************************************************************/
#ifndef __BASE_MATRIX_COMPILED_SPARSE_OPERATION_HPP__
#define __BASE_MATRIX_COMPILED_SPARSE_OPERATION_HPP__

#include "base_matrix_macros.hpp"

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_templates.hpp"
#include "base_matrix_vector.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace Base {
namespace Matrix {

/* Sparse Matrix minus */
namespace SparseMatrixMinus {

// Core loop for addition
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K, std::size_t Start,
          std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the negation of all values in a sparse matrix row
   * segment.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the matrix.
   * @tparam N Number of rows in the matrix.
   * @tparam RowIndices Type representing the row indices of nonzero elements.
   * @tparam RowPointers Type representing the row pointers for sparse storage.
   * @tparam J Current row index (compile-time).
   * @tparam K Unused parameter for compatibility.
   * @tparam Start Start index of the current segment.
   * @tparam End End index of the current segment.
   * @param[in]  A Input sparse matrix.
   * @param[out] Y Output sparse matrix, where each value is set to the negation
   * of the corresponding value in A.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &A,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &Y) {
    Y.values[Start] = -A.values[Start];
    Loop<T, M, N, RowIndices, RowPointers, J, K, Start + 1, End>::compute(A, Y);
  }
};

// End of core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K, std::size_t End>
struct Loop<T, M, N, RowIndices, RowPointers, J, K, End, End> {
  /**
   * @brief
   * End condition for the recursive negation loop in sparse matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the matrix.
   * @tparam N Number of rows in the matrix.
   * @tparam RowIndices Type representing the row indices of nonzero elements.
   * @tparam RowPointers Type representing the row pointers for sparse storage.
   * @tparam J Current row index (compile-time).
   * @tparam K Unused parameter for compatibility.
   * @tparam End End index of the current segment.
   * @param[in]  A Input sparse matrix.
   * @param[out] Y Output sparse matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &A,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &Y) {
    static_cast<void>(A);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J>
struct Row {
  /**
   * @brief
   * Computes the negation of all values in a specific row of a sparse matrix.
   *
   * This function recursively processes each row of the sparse matrix, applying
   * the negation operation to all non-zero elements in that row.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the matrix.
   * @tparam N Number of rows in the matrix.
   * @tparam RowIndices Type representing the row indices of nonzero elements.
   * @tparam RowPointers Type representing the row pointers for sparse storage.
   * @tparam J Current row index (compile-time).
   * @param[in]  A Input sparse matrix.
   * @param[out] Y Output sparse matrix, where each value is set to the negation
   * of the corresponding value in A.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &A,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &Y) {
    Loop<T, M, N, RowIndices, RowPointers, J, 0, RowPointers::list[J],
         RowPointers::list[J + 1]>::compute(A, Y);
    Row<T, M, N, RowIndices, RowPointers, J - 1>::compute(A, Y);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct Row<T, M, N, RowIndices, RowPointers, 0> {
  /**
   * @brief
   * Computes the negation of all values in the first row of a sparse matrix.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first row of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the matrix.
   * @tparam N Number of rows in the matrix.
   * @tparam RowIndices Type representing the row indices of nonzero elements.
   * @tparam RowPointers Type representing the row pointers for sparse storage.
   * @param[in]  A Input sparse matrix.
   * @param[out] Y Output sparse matrix, where each value is set to the negation
   * of the corresponding value in A.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &A,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &Y) {
    Loop<T, M, N, RowIndices, RowPointers, 0, 0, RowPointers::list[0],
         RowPointers::list[1]>::compute(A, Y);
  }
};

/**
 * @brief
 * Computes the negation of all values in a sparse matrix.
 *
 * This function serves as the entry point for negating all values in a sparse
 * matrix. It initializes the recursive computation starting from the last row.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the matrix.
 * @tparam N Number of rows in the matrix.
 * @tparam RowIndices Type representing the row indices of nonzero elements.
 * @tparam RowPointers Type representing the row pointers for sparse storage.
 * @param[in]  A Input sparse matrix.
 * @param[out] Y Output sparse matrix, where each value is set to the negation
 * of the corresponding value in A.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &A,
        CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &Y) {
  Row<T, M, N, RowIndices, RowPointers, M - 1>::compute(A, Y);
}

} // namespace SparseMatrixMinus

/**
 * @brief
 * Negates all values in a sparse matrix.
 *
 * This function creates a new sparse matrix where each value is the negation of
 * the corresponding value in the input sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the matrix.
 * @tparam N Number of rows in the matrix.
 * @tparam RowIndices Type representing the row indices of nonzero elements.
 * @tparam RowPointers Type representing the row pointers for sparse storage.
 * @param[in] A Input sparse matrix to be negated.
 * @return A new sparse matrix with negated values.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline CompiledSparseMatrix<T, M, N, RowIndices, RowPointers>
operator-(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &A) {
  CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers::list[j]; k < RowPointers::list[j + 1];
         ++k) {
      Y.values[k] = -A.values[k];
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixMinus::compute<T, M, N, RowIndices, RowPointers>(A, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Sparse Matrix multiply Dense Matrix */
namespace SparseMatrixMultiplyDenseMatrix {

// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the product of a sparse matrix row segment with a
   * dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense
   * matrix.
   * @tparam K Number of rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse
   * storage.
   * @tparam J Current row index (compile-time).
   * @tparam I Current column index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam End End index of the current segment.
   * @param[in]  A Input sparse matrix.
   * @param[in]  B Input dense matrix.
   * @param[out] sum Accumulator for the product result.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, T &sum) {
    sum += A.values[Start] * B.template get<RowIndices_A::list[Start], I>();
    Loop<T, M, N, K, RowIndices_A, RowPointers_A, J, I, Start + 1,
         End>::compute(A, B, sum);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t End>
struct Loop<T, M, N, K, RowIndices_A, RowPointers_A, J, I, End, End> {
  /**
   * @brief
   * End condition for the recursive product computation in sparse matrix
   * operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense
   * matrix.
   * @tparam K Number of rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse
   * storage.
   * @tparam J Current row index (compile-time).
   * @tparam I Current column index (compile-time).
   * @param[in]  A Input sparse matrix.
   * @param[in]  B Input dense matrix.
   * @param[out] sum Accumulator for the product result.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, T &sum) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(sum);
    // End of loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct Core {
  /**
   * @brief
   * Computes the product of a sparse matrix row with a dense matrix.
   *
   * This function serves as the core computation for multiplying a specific row
   * of a sparse matrix with a dense matrix, accumulating the result in a sum.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense
   * matrix.
   * @tparam K Number of rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse
   * storage.
   * @tparam J Current row index (compile-time).
   * @tparam I Current column index (compile-time).
   * @param[in]  A Input sparse matrix.
   * @param[in]  B Input dense matrix.
   * @param[out] Y Output dense matrix where the computed sum is stored.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {

    T sum = static_cast<T>(0);
    Loop<T, M, N, K, RowIndices_A, RowPointers_A, J, I, RowPointers_A::list[J],
         RowPointers_A::list[J + 1]>::compute(A, B, sum);

    Y.template set<J, I>(sum);
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct List {
  /**
   * @brief
   * Computes the product of a sparse matrix with a dense matrix for multiple
   * rows.
   *
   * This function recursively processes each row of the sparse matrix, calling
   * the core computation for each row and accumulating results in the output
   * dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense
   * matrix.
   * @tparam K Number of rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse
   * storage.
   * @tparam J Current row index (compile-time).
   * @tparam I Current column index (compile-time).
   * @param[in]  A Input sparse matrix.
   * @param[in]  B Input dense matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {

    Core<T, M, N, K, RowIndices_A, RowPointers_A, J, I>::compute(A, B, Y);
    List<T, M, N, K, RowIndices_A, RowPointers_A, J - 1, I>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct List<T, M, N, K, RowIndices_A, RowPointers_A, 0, I> {
  /**
   * @brief
   * Computes the product of the first row of a sparse matrix with a dense
   * matrix.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first row of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense
   * matrix.
   * @tparam K Number of rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse
   * storage.
   * @param[in]  A Input sparse matrix.
   * @param[in]  B Input dense matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {

    Core<T, M, N, K, RowIndices_A, RowPointers_A, 0, I>::compute(A, B, Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct Column {
  /**
   * @brief
   * Computes the product of a sparse matrix with a dense matrix for multiple
   * columns.
   *
   * This function recursively processes each column of the dense matrix,
   * calling the list computation for each column and accumulating results in
   * the output dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense
   * matrix.
   * @tparam K Number of rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse
   * storage.
   * @tparam I Current column index (compile-time).
   * @param[in]  A Input sparse matrix.
   * @param[in]  B Input dense matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {

    List<T, M, N, K, RowIndices_A, RowPointers_A, M - 1, I>::compute(A, B, Y);
    Column<T, M, N, K, RowIndices_A, RowPointers_A, I - 1>::compute(A, B, Y);
  }
};

// End of column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A>
struct Column<T, M, N, K, RowIndices_A, RowPointers_A, 0> {
  /**
   * @brief
   * Computes the product of the first column of a sparse matrix with a dense
   * matrix.
   *
   * This function is a specialization for the base case when I is 0, meaning it
   * processes the first column of the dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense
   * matrix.
   * @tparam K Number of rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse
   * storage.
   * @param[in]  A Input sparse matrix.
   * @param[in]  B Input dense matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {

    List<T, M, N, K, RowIndices_A, RowPointers_A, M - 1, 0>::compute(A, B, Y);
  }
};

/**
 * @brief
 * Computes the product of a sparse matrix with a dense matrix.
 *
 * This function serves as the entry point for multiplying a sparse matrix with
 * a dense matrix, producing a resulting dense matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage.
 * @tparam K Number of rows in the dense matrix.
 * @param[in]  A Input sparse matrix.
 * @param[in]  B Input dense matrix.
 * @param[out] Y Output dense matrix where results are accumulated.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {

  Column<T, M, N, K, RowIndices_A, RowPointers_A, K - 1>::compute(A, B, Y);
}

} // namespace SparseMatrixMultiplyDenseMatrix

/**
 * @brief
 * Multiplies a sparse matrix with a dense matrix.
 *
 * This function creates a new dense matrix that is the product of a sparse
 * matrix and a dense matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage.
 * @tparam K Number of rows in the dense matrix.
 * @param[in] A Input sparse matrix to be multiplied.
 * @param[in] B Input dense matrix to be multiplied.
 * @return A new dense matrix that is the product of A and B.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
inline Matrix<T, M, K>
operator*(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B) {
  Matrix<T, M, K> Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < K; i++) {
    for (std::size_t j = 0; j < M; j++) {
      T sum = static_cast<T>(0);
      for (std::size_t k = RowPointers_A::list[j];
           k < RowPointers_A::list[j + 1]; k++) {
        sum += A.values[k] * B(RowIndices_A::list[k], i);
      }
      Y(j, i) = sum;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixMultiplyDenseMatrix::compute<T, M, N, RowIndices_A, RowPointers_A,
                                           K>(A, B, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Dense Matrix multiply Sparse Matrix */
namespace DenseMatrixMultiplySparseMatrix {

// Core loop for multiplication
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t I, std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the product of a dense matrix row with a sparse matrix
   * column segment.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the dense matrix.
   * @tparam N Number of rows in the dense matrix and columns in the sparse
   * matrix.
   * @tparam K Number of rows in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse
   * storage.
   * @tparam J Current row index (compile-time).
   * @tparam I Current column index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam End End index of the current segment.
   * @param[in]  A Input dense matrix.
   * @param[in]  B Input sparse matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {

    Y.template set<I, RowIndices_B::list[Start]>(
        Y.template get<I, RowIndices_B::list[Start]>() +
        B.values[Start] * A.template get<I, J>());
    Loop<T, M, N, K, RowIndices_B, RowPointers_B, J, I, Start + 1,
         End>::compute(A, B, Y);
  }
};

// End of core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t I, std::size_t End>
struct Loop<T, M, N, K, RowIndices_B, RowPointers_B, J, I, End, End> {
  /**
   * @brief
   * End condition for the recursive product computation in dense matrix and
   * sparse matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the dense matrix.
   * @tparam N Number of rows in the dense matrix and columns in the sparse
   * matrix.
   * @tparam K Number of rows in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse
   * storage.
   * @tparam J Current row index (compile-time).
   * @tparam I Current column index (compile-time).
   * @param[in]  A Input dense matrix.
   * @param[in]  B Input sparse matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t I>
struct Core {
  /**
   * @brief
   * Computes the product of a dense matrix row with a sparse matrix column.
   *
   * This function serves as the core computation for multiplying a specific row
   * of a dense matrix with a sparse matrix, accumulating results in the output
   * dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the dense matrix.
   * @tparam N Number of rows in the dense matrix and columns in the sparse
   * matrix.
   * @tparam K Number of rows in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse
   * storage.
   * @tparam J Current row index (compile-time).
   * @tparam I Current column index (compile-time).
   * @param[in]  A Input dense matrix.
   * @param[in]  B Input sparse matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    Loop<T, M, N, K, RowIndices_B, RowPointers_B, J, I, RowPointers_B::list[J],
         RowPointers_B::list[J + 1]>::compute(A, B, Y);
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t I>
struct List {
  /**
   * @brief
   * Computes the product of a dense matrix with a sparse matrix for multiple
   * rows.
   *
   * This function recursively processes each row of the dense matrix, calling
   * the core computation for each row and accumulating results in the output
   * dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the dense matrix.
   * @tparam N Number of rows in the dense matrix and columns in the sparse
   * matrix.
   * @tparam K Number of rows in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse
   * storage.
   * @tparam J Current row index (compile-time).
   * @tparam I Current column index (compile-time).
   * @param[in]  A Input dense matrix.
   * @param[in]  B Input sparse matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {

    Core<T, M, N, K, RowIndices_B, RowPointers_B, J, I>::compute(A, B, Y);
    List<T, M, N, K, RowIndices_B, RowPointers_B, J, I - 1>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J>
struct List<T, M, N, K, RowIndices_B, RowPointers_B, J, 0> {
  /**
   * @brief
   * Computes the product of the first row of a dense matrix with a sparse
   * matrix.
   *
   * This function is a specialization for the base case when I is 0, meaning it
   * processes the first row of the dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the dense matrix.
   * @tparam N Number of rows in the dense matrix and columns in the sparse
   * matrix.
   * @tparam K Number of rows in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse
   * storage.
   * @param[in]  A Input dense matrix.
   * @param[in]  B Input sparse matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    Core<T, M, N, K, RowIndices_B, RowPointers_B, J, 0>::compute(A, B, Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J>
struct Column {
  /**
   * @brief
   * Computes the product of a dense matrix with a sparse matrix for multiple
   * columns.
   *
   * This function recursively processes each column of the sparse matrix,
   * calling the list computation for each column and accumulating results in
   * the output dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the dense matrix.
   * @tparam N Number of rows in the dense matrix and columns in the sparse
   * matrix.
   * @tparam K Number of rows in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse
   * storage.
   * @tparam J Current column index (compile-time).
   * @param[in]  A Input dense matrix.
   * @param[in]  B Input sparse matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {

    List<T, M, N, K, RowIndices_B, RowPointers_B, J, M - 1>::compute(A, B, Y);
    Column<T, M, N, K, RowIndices_B, RowPointers_B, J - 1>::compute(A, B, Y);
  }
};

// End of column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B>
struct Column<T, M, N, K, RowIndices_B, RowPointers_B, 0> {
  /**
   * @brief
   * Computes the product of the first column of a dense matrix with a sparse
   * matrix.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first column of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the dense matrix.
   * @tparam N Number of rows in the dense matrix and columns in the sparse
   * matrix.
   * @tparam K Number of rows in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse
   * storage.
   * @param[in]  A Input dense matrix.
   * @param[in]  B Input sparse matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {

    List<T, M, N, K, RowIndices_B, RowPointers_B, 0, M - 1>::compute(A, B, Y);
  }
};

/**
 * @brief
 * Computes the product of a dense matrix with a sparse matrix.
 *
 * This function serves as the entry point for multiplying a dense matrix with
 * a sparse matrix, producing a resulting dense matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the dense matrix.
 * @tparam N Number of rows in the dense matrix and columns in the sparse
 * matrix.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage.
 * @tparam K Number of rows in the sparse matrix.
 * @param[in]  A Input dense matrix.
 * @param[in]  B Input sparse matrix.
 * @param[out] Y Output dense matrix where results are accumulated.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t K>
inline void
compute(const Matrix<T, M, N> &A,
        const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
        Matrix<T, M, K> &Y) {
  Column<T, M, N, K, RowIndices_B, RowPointers_B, N - 1>::compute(A, B, Y);
}

} // namespace DenseMatrixMultiplySparseMatrix

/**
 * @brief
 * Multiplies a dense matrix with a sparse matrix.
 *
 * This function creates a new dense matrix that is the product of a dense
 * matrix and a sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the dense matrix.
 * @tparam N Number of rows in the dense matrix and columns in the sparse
 * matrix.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage.
 * @tparam K Number of rows in the sparse matrix.
 * @param[in] A Input dense matrix to be multiplied.
 * @param[in] B Input sparse matrix to be multiplied.
 * @return A new dense matrix that is the product of A and B.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t K>
inline Matrix<T, M, K>
operator*(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, M, K> Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < N; j++) {
    for (std::size_t k = RowPointers_B::list[j]; k < RowPointers_B::list[j + 1];
         k++) {
      for (std::size_t i = 0; i < M; i++) {
        Y(i, RowIndices_B::list[k]) += B.values[k] * A(i, j);
      }
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DenseMatrixMultiplySparseMatrix::compute<T, M, N, RowIndices_B, RowPointers_B,
                                           K>(A, B, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Substitute Dense Matrix to Sparse Matrix  */
namespace DenseMatrixSubstituteSparseMatrix {

// Core loop for addition
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K,
          std::size_t Start, std::size_t End>
struct Loop {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, M, N> &B) {

    A.values[Start] = B.template get<J, RowIndices_A::list[Start]>();

    Loop<T, M, N, RowIndices_A, RowPointers_A, J, K, Start + 1, End>::compute(
        A, B);
  }
};

// End of core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K, std::size_t End>
struct Loop<T, M, N, RowIndices_A, RowPointers_A, J, K, End, End> {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, M, N> &B) {
    static_cast<void>(A);
    static_cast<void>(B);
    // End of loop, do nothing
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct Row {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, M, N> &B) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, J, 0, RowPointers_A::list[J],
         RowPointers_A::list[J + 1]>::compute(A, B);
    Row<T, M, N, RowIndices_A, RowPointers_A, J - 1>::compute(A, B);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct Row<T, M, N, RowIndices_A, RowPointers_A, 0> {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, M, N> &B) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, 0, 0, RowPointers_A::list[0],
         RowPointers_A::list[1]>::compute(A, B);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void
compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        const Matrix<T, M, N> &B) {
  Row<T, M, N, RowIndices_A, RowPointers_A, M - 1>::compute(A, B);
}

} // namespace DenseMatrixSubstituteSparseMatrix

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void substitute_dense_to_sparse(
    CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, M, N> &B) {
  DenseMatrixSubstituteSparseMatrix::compute<T, M, N, RowIndices_A,
                                             RowPointers_A>(A, B);
}

/* Sparse Matrix add Dense Matrix */
namespace SparseMatrixAddDenseMatrix {

// Core loop for addition
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K,
          std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the addition of a sparse matrix row with a dense
   * matrix row segment.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse
   * storage.
   * @tparam J Current row index (compile-time).
   * @tparam K Current column index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam End End index of the current segment.
   * @param[in]  A Input sparse matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {

    Y.template set<J, RowIndices_A::list[Start]>(
        Y.template get<J, RowIndices_A::list[Start]>() + A.values[Start]);
    Loop<T, M, N, RowIndices_A, RowPointers_A, J, K, Start + 1, End>::compute(
        A, Y);
  }
};

// End of core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K, std::size_t End>
struct Loop<T, M, N, RowIndices_A, RowPointers_A, J, K, End, End> {
  /**
   * @brief
   * End condition for the recursive addition computation in sparse matrix and
   * dense matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse
   * storage.
   * @tparam J Current row index (compile-time).
   * @tparam K Current column index (compile-time).
   * @param[in]  A Input sparse matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {
    static_cast<void>(A);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct Row {
  /**
   * @brief
   * Computes the addition of a sparse matrix row with a dense matrix row.
   *
   * This function serves as the core computation for adding a specific row of a
   * sparse matrix to a dense matrix, accumulating results in the output dense
   * matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse
   * storage.
   * @tparam J Current row index (compile-time).
   * @param[in]  A Input sparse matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, J, 0, RowPointers_A::list[J],
         RowPointers_A::list[J + 1]>::compute(A, Y);
    Row<T, M, N, RowIndices_A, RowPointers_A, J - 1>::compute(A, Y);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct Row<T, M, N, RowIndices_A, RowPointers_A, 0> {
  /**
   * @brief
   * Computes the addition of the first row of a sparse matrix with a dense
   * matrix row.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first row of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse
   * storage.
   * @param[in]  A Input sparse matrix.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, 0, 0, RowPointers_A::list[0],
         RowPointers_A::list[1]>::compute(A, Y);
  }
};

/**
 * @brief
 * Computes the addition of a sparse matrix with a dense matrix.
 *
 * This function serves as the entry point for adding a sparse matrix to a
 * dense matrix, producing a resulting dense matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage.
 * @param[in]  A Input sparse matrix.
 * @param[out] Y Output dense matrix where results are accumulated.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        Matrix<T, M, N> &Y) {
  Row<T, M, N, RowIndices_A, RowPointers_A, M - 1>::compute(A, Y);
}

} // namespace SparseMatrixAddDenseMatrix

/**
 * @brief
 * Adds a sparse matrix to a dense matrix.
 *
 * This function creates a new dense matrix that is the result of adding a
 * sparse matrix to a dense matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage.
 * @param[in] A Input sparse matrix to be added.
 * @param[in] B Input dense matrix to be added.
 * @return A new dense matrix that is the sum of A and B.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline Matrix<T, M, N>
operator+(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, M, N> &B) {
  Matrix<T, M, N> Y = B;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         ++k) {
      Y(j, RowIndices_A::list[k]) += A.values[k];
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixAddDenseMatrix::compute<T, M, N, RowIndices_A, RowPointers_A>(A,
                                                                            Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Dense Matrix add Sparse Matrix */

/**
 * @brief
 * Adds a dense matrix to a sparse matrix.
 *
 * This function creates a new dense matrix that is the result of adding a
 * dense matrix to a sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage.
 * @param[in] B Input dense matrix to be added.
 * @param[in] A Input sparse matrix to be added.
 * @return A new dense matrix that is the sum of B and A.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline Matrix<T, M, N>
operator+(const Matrix<T, M, N> &B,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
  Matrix<T, M, N> Y = B;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         ++k) {
      Y(j, RowIndices_A::list[k]) += A.values[k];
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixAddDenseMatrix::compute<T, M, N, RowIndices_A, RowPointers_A>(A,
                                                                            Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Sparse Matrix add Sparse Matrix */
namespace SparseMatrixAddSparseMatrix {

// Core loop for addition
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J, std::size_t K, std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the addition of a sparse matrix row with another
   * sparse matrix row segment.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam J Current row index (compile-time).
   * @tparam K Current column index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam End End index of the current segment.
   * @param[in]  A Input sparse matrix to be added.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {

    Base::Matrix::set_sparse_matrix_value<J, RowIndices_A::list[Start]>(
        Y,
        Base::Matrix::get_sparse_matrix_value<J, RowIndices_A::list[Start]>(Y) +
            A.values[Start]);
    Loop<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y, J,
         K, Start + 1, End>::compute(A, Y);
  }
};

// End of core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J, std::size_t K, std::size_t End>
struct Loop<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y,
            J, K, End, End> {
  /**
   * @brief
   * End condition for the recursive addition computation in sparse matrix and
   * sparse matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam J Current row index (compile-time).
   * @tparam K Current column index (compile-time).
   * @param[in]  A Input sparse matrix to be added.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {
    static_cast<void>(A);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J>
struct Row {
  /**
   * @brief
   * Computes the addition of a sparse matrix row with another sparse matrix
   * row.
   *
   * This function serves as the core computation for adding a specific row of a
   * sparse matrix to another sparse matrix, accumulating results in the output
   * sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam J Current row index (compile-time).
   * @param[in]  A Input sparse matrix to be added.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y, J,
         0, RowPointers_A::list[J], RowPointers_A::list[J + 1]>::compute(A, Y);
    Row<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y,
        J - 1>::compute(A, Y);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y>
struct Row<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y,
           0> {
  /**
   * @brief
   * Computes the addition of the first row of a sparse matrix with another
   * sparse matrix row.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first row of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @param[in]  A Input sparse matrix to be added.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y, 0,
         0, RowPointers_A::list[0], RowPointers_A::list[1]>::compute(A, Y);
  }
};

/**
 * @brief
 * Computes the addition of a sparse matrix with another sparse matrix.
 *
 * This function serves as the entry point for adding two sparse matrices,
 * producing a resulting sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the first sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the first sparse matrix.
 * @tparam RowIndices_Y Type representing the row indices of nonzero elements
 * in the second sparse matrix.
 * @tparam RowPointers_Y Type representing the row pointers for sparse storage
 * in the second sparse matrix.
 * @param[in]  A Input sparse matrix to be added.
 * @param[out] Y Output sparse matrix where results are accumulated.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {

  Row<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y,
      M - 1>::compute(A, Y);
}

} // namespace SparseMatrixAddSparseMatrix

/* Set DiagMatrix values to CompiledSparseMatrix */
namespace SetDiagMatrixToValuesSparseMatrix {

template <typename T, std::size_t M, std::size_t N, typename RowIndices_Y,
          typename RowPointers_Y, std::size_t I>
struct Core {
  /**
   * @brief
   * Recursively sets the diagonal values of a sparse matrix from a diagonal
   * matrix.
   *
   * This function processes each diagonal element of the sparse matrix, setting
   * its value based on the corresponding element in the diagonal matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows and columns in the diagonal matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam I Current index (compile-time).
   * @param[in,out] Y Output sparse matrix where diagonal values are set.
   * @param[in] B Input diagonal matrix providing values for the diagonal.
   */
  static void
  apply(CompiledSparseMatrix<T, M, M, RowIndices_Y, RowPointers_Y> &Y,
        const DiagMatrix<T, M> &B) {

    Base::Matrix::set_sparse_matrix_value<I, I>(Y, B[I]);
    Core<T, M, N, RowIndices_Y, RowPointers_Y, I - 1>::apply(Y, B);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_Y,
          typename RowPointers_Y>
struct Core<T, M, N, RowIndices_Y, RowPointers_Y, 0> {
  /**
   * @brief
   * Base case for the recursive setting of diagonal values in a sparse matrix.
   *
   * This specialization is instantiated when I is 0, meaning it processes the
   * first diagonal element. It does not perform any action, serving as the base
   * case to terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows and columns in the diagonal matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @param[in,out] Y Output sparse matrix where diagonal values are set.
   * @param[in] B Input diagonal matrix providing values for the diagonal.
   */
  static void
  apply(CompiledSparseMatrix<T, M, M, RowIndices_Y, RowPointers_Y> &Y,
        const DiagMatrix<T, M> &B) {

    Base::Matrix::set_sparse_matrix_value<0, 0>(Y, B[0]);
  }
};

/**
 * @brief
 * Sets the diagonal values of a sparse matrix from a diagonal matrix.
 *
 * This function serves as the entry point for setting the diagonal values of a
 * sparse matrix using the values from a diagonal matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of rows and columns in the diagonal matrix.
 * @tparam N Number of columns in the sparse matrix.
 * @tparam RowIndices_Y Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_Y Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in,out] Y Output sparse matrix where diagonal values are set.
 * @param[in] B Input diagonal matrix providing values for the diagonal.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_Y,
          typename RowPointers_Y>
inline void
compute(CompiledSparseMatrix<T, M, M, RowIndices_Y, RowPointers_Y> &Y,
        const DiagMatrix<T, M> &B) {
  Core<T, M, N, RowIndices_Y, RowPointers_Y, M - 1>::apply(Y, B);
}

} // namespace SetDiagMatrixToValuesSparseMatrix

namespace CompiledSparseOperation {

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct DiagAddSubSparse {

  using RowIndices_Y = RowIndicesFromSparseAvailable<
      MatrixAddSubSparseAvailable<CreateSparseAvailableFromIndicesAndPointers<
                                      N, RowIndices_A, RowPointers_A>,
                                  DiagAvailable<M>>>;

  using RowPointers_Y = RowPointersFromSparseAvailable<
      MatrixAddSubSparseAvailable<CreateSparseAvailableFromIndicesAndPointers<
                                      N, RowIndices_A, RowPointers_A>,
                                  DiagAvailable<M>>>;

  using Y_Type = CompiledSparseMatrix<T, M, M, RowIndices_Y, RowPointers_Y>;
};

} // namespace CompiledSparseOperation

/**
 * @brief
 * Adds a sparse matrix to a diagonal matrix.
 *
 * This function creates a new sparse matrix that is the result of adding a
 * sparse matrix to a diagonal matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix and rows in the diagonal
 * matrix.
 * @tparam N Number of rows in the sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in] A Input sparse matrix to be added.
 * @param[in] B Input diagonal matrix to be added.
 * @return A new sparse matrix that is the sum of A and B.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline auto
operator+(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, M> &B) ->
    typename CompiledSparseOperation::DiagAddSubSparse<T, M, N, RowIndices_A,
                                                       RowPointers_A>::Y_Type {
  static_assert(M == N, "Argument is not square matrix.");

  using RowIndices_Y = typename CompiledSparseOperation::DiagAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A>::RowIndices_Y;

  using RowPointers_Y = typename CompiledSparseOperation::DiagAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A>::RowPointers_Y;

  using Y_Type =
      typename CompiledSparseOperation::DiagAddSubSparse<T, M, N, RowIndices_A,
                                                         RowPointers_A>::Y_Type;

  Y_Type Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Matrix<T, M, M> Y_temp = Base::Matrix::output_dense_matrix(B);

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         ++k) {
      Y_temp(j, RowIndices_A::list[k]) += A.values[k];
    }
  }

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_Y::list[j]; k < RowPointers_Y::list[j + 1];
         ++k) {
      Y.values[k] = Y_temp(j, RowIndices_Y::list[k]);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SetDiagMatrixToValuesSparseMatrix::compute<T, M, N, RowIndices_Y,
                                             RowPointers_Y>(Y, B);

  SparseMatrixAddSparseMatrix::compute<T, M, N, RowIndices_A, RowPointers_A,
                                       RowIndices_Y, RowPointers_Y>(A, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Diag Matrix add Sparse Matrix */

/**
 * @brief
 * Adds a diagonal matrix to a sparse matrix.
 *
 * This function creates a new sparse matrix that is the result of adding a
 * diagonal matrix to a sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix and rows in the diagonal
 * matrix.
 * @tparam N Number of rows in the sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in] B Input diagonal matrix to be added.
 * @param[in] A Input sparse matrix to be added.
 * @return A new sparse matrix that is the sum of B and A.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline auto
operator+(const DiagMatrix<T, M> &B,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A)
    ->
    typename CompiledSparseOperation::DiagAddSubSparse<T, M, N, RowIndices_A,
                                                       RowPointers_A>::Y_Type {
  static_assert(M == N, "Argument is not square matrix.");

  using RowIndices_Y = typename CompiledSparseOperation::DiagAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A>::RowIndices_Y;

  using RowPointers_Y = typename CompiledSparseOperation::DiagAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A>::RowPointers_Y;

  using Y_Type =
      typename CompiledSparseOperation::DiagAddSubSparse<T, M, N, RowIndices_A,
                                                         RowPointers_A>::Y_Type;

  Y_Type Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Matrix<T, M, M> Y_temp = Base::Matrix::output_dense_matrix(B);

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         ++k) {
      Y_temp(j, RowIndices_A::list[k]) += A.values[k];
    }
  }

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_Y::list[j]; k < RowPointers_Y::list[j + 1];
         ++k) {
      Y.values[k] = Y_temp(j, RowIndices_Y::list[k]);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SetDiagMatrixToValuesSparseMatrix::compute<T, M, N, RowIndices_Y,
                                             RowPointers_Y>(Y, B);

  SparseMatrixAddSparseMatrix::compute<T, M, N, RowIndices_A, RowPointers_A,
                                       RowIndices_Y, RowPointers_Y>(A, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

namespace CompiledSparseOperation {

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_B, typename RowPointers_B>
struct SparseAddSubSparse {

  using RowIndices_Y = RowIndicesFromSparseAvailable<
      MatrixAddSubSparseAvailable<CreateSparseAvailableFromIndicesAndPointers<
                                      N, RowIndices_A, RowPointers_A>,
                                  CreateSparseAvailableFromIndicesAndPointers<
                                      N, RowIndices_B, RowPointers_B>>>;
  using RowPointers_Y = RowPointersFromSparseAvailable<
      MatrixAddSubSparseAvailable<CreateSparseAvailableFromIndicesAndPointers<
                                      N, RowIndices_A, RowPointers_A>,
                                  CreateSparseAvailableFromIndicesAndPointers<
                                      N, RowIndices_B, RowPointers_B>>>;

  using Y_Type = CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y>;
};

} // namespace CompiledSparseOperation

/* Sparse Matrix add Sparse Matrix */

/**
 * @brief
 * Adds two sparse matrices together.
 *
 * This function creates a new sparse matrix that is the result of adding two
 * sparse matrices together.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrices.
 * @tparam N Number of rows in the sparse matrices.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the first sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the first sparse matrix.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the second sparse matrix.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the second sparse matrix.
 * @param[in] A Input first sparse matrix to be added.
 * @param[in] B Input second sparse matrix to be added.
 * @return A new sparse matrix that is the sum of A and B.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_B, typename RowPointers_B>
inline auto
operator+(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B)
    -> typename CompiledSparseOperation::SparseAddSubSparse<
        T, M, N, RowIndices_A, RowPointers_A, RowIndices_B,
        RowPointers_B>::Y_Type {

  using RowIndices_Y = typename CompiledSparseOperation::SparseAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A, RowIndices_B,
      RowPointers_B>::RowIndices_Y;

  using RowPointers_Y = typename CompiledSparseOperation::SparseAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A, RowIndices_B,
      RowPointers_B>::RowPointers_Y;

  using Y_Type = typename CompiledSparseOperation::SparseAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A, RowIndices_B,
      RowPointers_B>::Y_Type;

  Y_Type Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Matrix<T, M, M> Y_temp = Base::Matrix::output_dense_matrix(B);

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         ++k) {
      Y_temp(j, RowIndices_A::list[k]) += A.values[k];
    }
  }

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_Y::list[j]; k < RowPointers_Y::list[j + 1];
         ++k) {
      Y.values[k] = Y_temp(j, RowIndices_Y::list[k]);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixAddSparseMatrix::compute<T, M, N, RowIndices_A, RowPointers_A,
                                       RowIndices_Y, RowPointers_Y>(A, Y);
  SparseMatrixAddSparseMatrix::compute<T, M, N, RowIndices_B, RowPointers_B,
                                       RowIndices_Y, RowPointers_Y>(B, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Sparse Matrix subtract Dense Matrix */

/**
 * @brief
 * Subtracts a dense matrix from a sparse matrix.
 *
 * This function creates a new dense matrix that is the result of subtracting a
 * dense matrix from a sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage.
 * @param[in] A Input sparse matrix to be subtracted from.
 * @param[in] B Input dense matrix to be subtracted.
 * @return A new dense matrix that is the result of A - B.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline Matrix<T, M, N>
operator-(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, M, N> &B) {
  Matrix<T, M, N> Y = -B;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         ++k) {
      Y(j, RowIndices_A::list[k]) += A.values[k];
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixAddDenseMatrix::compute<T, M, N, RowIndices_A, RowPointers_A>(A,
                                                                            Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Dense Matrix subtract Sparse Matrix */
namespace DenseMatrixSubSparseMatrix {

// Core loop for addition
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K,
          std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the subtraction of a sparse matrix row from a dense
   * matrix row segment.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current row index (compile-time).
   * @tparam K Current column index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam End End index of the current segment.
   * @param[in]  A Input sparse matrix to be subtracted.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {

    Y.template set<J, RowIndices_A::list[Start]>(
        Y.template get<J, RowIndices_A::list[Start]>() - A.values[Start]);
    Loop<T, M, N, RowIndices_A, RowPointers_A, J, K, Start + 1, End>::compute(
        A, Y);
  }
};

// End of core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K, std::size_t End>
struct Loop<T, M, N, RowIndices_A, RowPointers_A, J, K, End, End> {
  /**
   * @brief
   * End condition for the recursive subtraction computation in dense matrix and
   * sparse matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current row index (compile-time).
   * @tparam K Current column index (compile-time).
   * @param[in]  A Input sparse matrix to be subtracted.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {
    static_cast<void>(A);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct Row {
  /**
   * @brief
   * Computes the subtraction of a sparse matrix row from a dense matrix row.
   *
   * This function serves as the core computation for subtracting a specific row
   * of a sparse matrix from a dense matrix, accumulating results in the output
   * dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current row index (compile-time).
   * @param[in]  A Input sparse matrix to be subtracted.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, J, 0, RowPointers_A::list[J],
         RowPointers_A::list[J + 1]>::compute(A, Y);
    Row<T, M, N, RowIndices_A, RowPointers_A, J - 1>::compute(A, Y);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct Row<T, M, N, RowIndices_A, RowPointers_A, 0> {
  /**
   * @brief
   * Computes the subtraction of the first row of a sparse matrix from a dense
   * matrix row.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first row of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @param[in]  A Input sparse matrix to be subtracted.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, 0, 0, RowPointers_A::list[0],
         RowPointers_A::list[1]>::compute(A, Y);
  }
};

/**
 * @brief
 * Computes the subtraction of a sparse matrix from a dense matrix.
 *
 * This function serves as the entry point for subtracting a sparse matrix from
 * a dense matrix, producing a resulting dense matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix and rows in the dense
 * matrix.
 * @tparam N Number of rows in the sparse matrix and rows in the dense matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in]  A Input sparse matrix to be subtracted.
 * @param[out] Y Output dense matrix where results are accumulated.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        Matrix<T, M, N> &Y) {
  Row<T, M, N, RowIndices_A, RowPointers_A, M - 1>::compute(A, Y);
}

} // namespace DenseMatrixSubSparseMatrix

/**
 * @brief
 * Subtracts a sparse matrix from a dense matrix.
 *
 * This function creates a new dense matrix that is the result of subtracting a
 * sparse matrix from a dense matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the dense matrix and rows in the sparse
 * matrix.
 * @tparam N Number of rows in the dense matrix and rows in the sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in] B Input dense matrix to be subtracted from.
 * @param[in] A Input sparse matrix to be subtracted.
 * @return A new dense matrix that is the result of B - A.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline Matrix<T, M, N>
operator-(const Matrix<T, M, N> &B,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
  Matrix<T, M, N> Y = B;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         ++k) {
      Y(j, RowIndices_A::list[k]) -= A.values[k];
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DenseMatrixSubSparseMatrix::compute<T, M, N, RowIndices_A, RowPointers_A>(A,
                                                                            Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Sparse Matrix subtract Diag Matrix */

/**
 * @brief
 * Subtracts a diagonal matrix from a sparse matrix.
 *
 * This function creates a new sparse matrix that is the result of subtracting a
 * diagonal matrix from a sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix and rows in the diagonal
 * matrix.
 * @tparam N Number of rows in the sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in] A Input sparse matrix to be subtracted from.
 * @param[in] B Input diagonal matrix to be subtracted.
 * @return A new sparse matrix that is the result of A - B.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline auto
operator-(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, M> &B) ->
    typename CompiledSparseOperation::DiagAddSubSparse<T, M, N, RowIndices_A,
                                                       RowPointers_A>::Y_Type {
  static_assert(M == N, "Argument is not square matrix.");

  using RowIndices_Y = typename CompiledSparseOperation::DiagAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A>::RowIndices_Y;

  using RowPointers_Y = typename CompiledSparseOperation::DiagAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A>::RowPointers_Y;

  using Y_Type =
      typename CompiledSparseOperation::DiagAddSubSparse<T, M, N, RowIndices_A,
                                                         RowPointers_A>::Y_Type;

  Y_Type Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Matrix<T, M, M> Y_temp = -(Base::Matrix::output_dense_matrix(B));

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         ++k) {
      Y_temp(j, RowIndices_A::list[k]) += A.values[k];
    }
  }

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_Y::list[j]; k < RowPointers_Y::list[j + 1];
         ++k) {
      Y.values[k] = Y_temp(j, RowIndices_Y::list[k]);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SetDiagMatrixToValuesSparseMatrix::compute<T, M, N, RowIndices_Y,
                                             RowPointers_Y>(Y, -B);

  SparseMatrixAddSparseMatrix::compute<T, M, N, RowIndices_A, RowPointers_A,
                                       RowIndices_Y, RowPointers_Y>(A, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Diag Matrix subtract Sparse Matrix */
namespace DiagMatrixSubSparseMatrix {

// Core loop for addition
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J, std::size_t K, std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the subtraction of a sparse matrix row from a diagonal
   * matrix row segment.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the diagonal
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the diagonal matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the diagonal matrix.
   * @tparam J Current row index (compile-time).
   * @tparam K Current column index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam End End index of the current segment.
   * @param[in]  A Input sparse matrix to be subtracted.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {

    Base::Matrix::set_sparse_matrix_value<J, RowIndices_A::list[Start]>(
        Y,
        Base::Matrix::get_sparse_matrix_value<J, RowIndices_A::list[Start]>(Y) -
            A.values[Start]);
    Loop<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y, J,
         K, Start + 1, End>::compute(A, Y);
  }
};

// End of core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J, std::size_t K, std::size_t End>
struct Loop<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y,
            J, K, End, End> {
  /**
   * @brief
   * End condition for the recursive subtraction computation in sparse matrix
   * and diagonal matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the diagonal
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the diagonal matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the diagonal matrix.
   * @tparam J Current row index (compile-time).
   * @tparam K Current column index (compile-time).
   * @param[in]  A Input sparse matrix to be subtracted.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {
    static_cast<void>(A);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J>
struct Row {
  /**
   * @brief
   * Computes the subtraction of a sparse matrix row from a diagonal matrix row.
   *
   * This function serves as the core computation for subtracting a specific row
   * of a sparse matrix from a diagonal matrix, accumulating results in the
   * output sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the diagonal
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the diagonal matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the diagonal matrix.
   * @tparam J Current row index (compile-time).
   * @param[in]  A Input sparse matrix to be subtracted.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y, J,
         0, RowPointers_A::list[J], RowPointers_A::list[J + 1]>::compute(A, Y);
    Row<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y,
        J - 1>::compute(A, Y);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y>
struct Row<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y,
           0> {
  /**
   * @brief
   * Computes the subtraction of the first row of a sparse matrix from a
   * diagonal matrix row.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first row of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix and rows in the diagonal
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the diagonal matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the diagonal matrix.
   * @param[in]  A Input sparse matrix to be subtracted.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y, 0,
         0, RowPointers_A::list[0], RowPointers_A::list[1]>::compute(A, Y);
  }
};

/**
 * @brief
 * Computes the subtraction of a sparse matrix from a diagonal matrix.
 *
 * This function serves as the entry point for subtracting a sparse matrix from
 * a diagonal matrix, producing a resulting sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix and rows in the diagonal
 * matrix.
 * @tparam N Number of rows in the sparse matrix and rows in the diagonal
 * matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in]  A Input sparse matrix to be subtracted.
 * @param[out] Y Output sparse matrix where results are accumulated.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {

  Row<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y,
      M - 1>::compute(A, Y);
}

} // namespace DiagMatrixSubSparseMatrix

/**
 * @brief
 * Subtracts a sparse matrix from a diagonal matrix.
 *
 * This function creates a new sparse matrix that is the result of subtracting a
 * sparse matrix from a diagonal matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the diagonal matrix and rows in the sparse
 * matrix.
 * @tparam N Number of rows in the diagonal matrix and rows in the sparse
 * matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in] B Input diagonal matrix to be subtracted from.
 * @param[in] A Input sparse matrix to be subtracted.
 * @return A new sparse matrix that is the result of B - A.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline auto
operator-(const DiagMatrix<T, M> &B,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A)
    ->
    typename CompiledSparseOperation::DiagAddSubSparse<T, M, N, RowIndices_A,
                                                       RowPointers_A>::Y_Type {
  static_assert(M == N, "Argument is not square matrix.");

  using RowIndices_Y = typename CompiledSparseOperation::DiagAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A>::RowIndices_Y;

  using RowPointers_Y = typename CompiledSparseOperation::DiagAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A>::RowPointers_Y;

  using Y_Type =
      typename CompiledSparseOperation::DiagAddSubSparse<T, M, N, RowIndices_A,
                                                         RowPointers_A>::Y_Type;

  Y_Type Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Matrix<T, M, M> Y_temp = Base::Matrix::output_dense_matrix(B);

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         ++k) {
      Y_temp(j, RowIndices_A::list[k]) -= A.values[k];
    }
  }

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_Y::list[j]; k < RowPointers_Y::list[j + 1];
         ++k) {
      Y.values[k] = Y_temp(j, RowIndices_Y::list[k]);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SetDiagMatrixToValuesSparseMatrix::compute<T, M, N, RowIndices_Y,
                                             RowPointers_Y>(Y, B);

  DiagMatrixSubSparseMatrix::compute<T, M, N, RowIndices_A, RowPointers_A>(A,
                                                                           Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Sparse Matrix subtract Sparse Matrix */
namespace SparseMatrixSubSparseMatrix {

// Core loop for addition
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J, std::size_t K, std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the subtraction of a sparse matrix row from another
   * sparse matrix row segment.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam J Current row index (compile-time).
   * @tparam K Current column index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam End End index of the current segment.
   * @param[in]  A Input sparse matrix to be subtracted from.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {

    Base::Matrix::set_sparse_matrix_value<J, RowIndices_A::list[Start]>(
        Y,
        Base::Matrix::get_sparse_matrix_value<J, RowIndices_A::list[Start]>(Y) -
            A.values[Start]);
    Loop<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y, J,
         K, Start + 1, End>::compute(A, Y);
  }
};

// End of core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J, std::size_t K, std::size_t End>
struct Loop<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y,
            J, K, End, End> {
  /**
   * @brief
   * End condition for the recursive subtraction computation in sparse matrix
   * operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam J Current row index (compile-time).
   * @tparam K Current column index (compile-time).
   * @param[in]  A Input sparse matrix to be subtracted from.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {
    static_cast<void>(A);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J>
struct Row {
  /**
   * @brief
   * Computes the subtraction of a sparse matrix row from another sparse matrix
   * row.
   *
   * This function serves as the core computation for subtracting a specific row
   * of a sparse matrix from another sparse matrix, accumulating results in the
   * output sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam J Current row index (compile-time).
   * @param[in]  A Input sparse matrix to be subtracted from.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y, J,
         0, RowPointers_A::list[J], RowPointers_A::list[J + 1]>::compute(A, Y);
    Row<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y,
        J - 1>::compute(A, Y);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y>
struct Row<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y,
           0> {
  /**
   * @brief
   * Computes the subtraction of the first row of a sparse matrix from another
   * sparse matrix row.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first row of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @param[in]  A Input sparse matrix to be subtracted from.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y, 0,
         0, RowPointers_A::list[0], RowPointers_A::list[1]>::compute(A, Y);
  }
};

/**
 * @brief
 * Computes the subtraction of a sparse matrix from another sparse matrix.
 *
 * This function serves as the entry point for subtracting one sparse matrix
 * from another, producing a resulting sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrices.
 * @tparam N Number of rows in the sparse matrices.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the first sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the first sparse matrix.
 * @tparam RowIndices_Y Type representing the row indices of nonzero elements
 * in the second sparse matrix.
 * @tparam RowPointers_Y Type representing the row pointers for sparse storage
 * in the second sparse matrix.
 * @param[in]  A Input sparse matrix to be subtracted from.
 * @param[out] Y Output sparse matrix where results are accumulated.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_Y, typename RowPointers_Y>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        CompiledSparseMatrix<T, M, N, RowIndices_Y, RowPointers_Y> &Y) {

  Row<T, M, N, RowIndices_A, RowPointers_A, RowIndices_Y, RowPointers_Y,
      M - 1>::compute(A, Y);
}

} // namespace SparseMatrixSubSparseMatrix

/**
 * @brief
 * Subtracts one sparse matrix from another sparse matrix.
 *
 * This function creates a new sparse matrix that is the result of subtracting
 * one sparse matrix from another.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrices.
 * @tparam N Number of rows in the sparse matrices.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the first sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the first sparse matrix.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the second sparse matrix.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the second sparse matrix.
 * @param[in] A Input sparse matrix to be subtracted from.
 * @param[in] B Input sparse matrix to be subtracted.
 * @return A new sparse matrix that is the result of A - B.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_B, typename RowPointers_B>
inline auto
operator-(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B)
    -> typename CompiledSparseOperation::SparseAddSubSparse<
        T, M, N, RowIndices_A, RowPointers_A, RowIndices_B,
        RowPointers_B>::Y_Type {

  using RowIndices_Y = typename CompiledSparseOperation::SparseAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A, RowIndices_B,
      RowPointers_B>::RowIndices_Y;

  using RowPointers_Y = typename CompiledSparseOperation::SparseAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A, RowIndices_B,
      RowPointers_B>::RowPointers_Y;

  using Y_Type = typename CompiledSparseOperation::SparseAddSubSparse<
      T, M, N, RowIndices_A, RowPointers_A, RowIndices_B,
      RowPointers_B>::Y_Type;

  Y_Type Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Matrix<T, M, M> Y_temp = Base::Matrix::output_dense_matrix(A);

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_B::list[j]; k < RowPointers_B::list[j + 1];
         ++k) {
      Y_temp(j, RowIndices_B::list[k]) -= B.values[k];
    }
  }

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_Y::list[j]; k < RowPointers_Y::list[j + 1];
         ++k) {
      Y.values[k] = Y_temp(j, RowIndices_Y::list[k]);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixAddSparseMatrix::compute<T, M, N, RowIndices_A, RowPointers_A,
                                       RowIndices_Y, RowPointers_Y>(A, Y);

  SparseMatrixSubSparseMatrix::compute<T, M, N, RowIndices_B, RowPointers_B,
                                       RowIndices_Y, RowPointers_Y>(B, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Sparse Matrix multiply Scalar */
namespace SparseMatrixMultiplyScalar {

// Core loop for scalar multiplication
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the scalar multiplication of a sparse matrix element.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam I Current index (compile-time).
   * @tparam End End index for the loop.
   * @param[in]  A Input sparse matrix to be multiplied by scalar.
   * @param[in]  scalar Scalar value to multiply with.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &scalar,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

    Y.values[I] = scalar * A.values[I];
    Loop<T, M, N, RowIndices_A, RowPointers_A, I + 1, End>::compute(A, scalar,
                                                                    Y);
  }
};

// End of loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t End>
struct Loop<T, M, N, RowIndices_A, RowPointers_A, End, End> {
  /**
   * @brief
   * End condition for the recursive scalar multiplication computation in sparse
   * matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the current
   * index equals the end index, indicating that all elements have been
   * processed. The function does nothing, serving as the base case to terminate
   * recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @param[in]  A Input sparse matrix to be multiplied by scalar.
   * @param[in]  scalar Scalar value to multiply with.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &scalar,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {
    static_cast<void>(A);
    static_cast<void>(scalar);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

/**
 * @brief
 * Computes the scalar multiplication of a sparse matrix.
 *
 * This function serves as the entry point for multiplying a sparse matrix by a
 * scalar, producing a resulting sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in]  A Input sparse matrix to be multiplied by scalar.
 * @param[in]  scalar Scalar value to multiply with.
 * @param[out] Y Output sparse matrix where results are accumulated.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        const T &scalar,
        CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

  Loop<T, M, N, RowIndices_A, RowPointers_A, 0, RowIndices_A::size>::compute(
      A, scalar, Y);
}

} // namespace SparseMatrixMultiplyScalar

/**
 * @brief
 * Multiplies a sparse matrix by a scalar.
 *
 * This function creates a new sparse matrix that is the result of multiplying a
 * sparse matrix by a scalar.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in] A Input sparse matrix to be multiplied by scalar.
 * @param[in] scalar Scalar value to multiply with.
 * @return A new sparse matrix that is the result of A * scalar.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A>
operator*(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &scalar) {
  CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> Y = A;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < RowIndices_A::size; i++) {
    Y.values[i] = scalar * A.values[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixMultiplyScalar::compute<T, M, N, RowIndices_A, RowPointers_A>(
      A, scalar, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Scalar multiply Sparse Matrix */

/**
 * @brief
 * Multiplies a scalar by a sparse matrix.
 *
 * This function creates a new sparse matrix that is the result of multiplying a
 * scalar by a sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in] scalar Scalar value to multiply with.
 * @param[in] A Input sparse matrix to be multiplied by scalar.
 * @return A new sparse matrix that is the result of scalar * A.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A>
operator*(const T &scalar,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
  CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> Y = A;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < RowIndices_A::size; i++) {
    Y.values[i] = scalar * A.values[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixMultiplyScalar::compute<T, M, N, RowIndices_A, RowPointers_A>(
      A, scalar, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Sparse Matrix multiply Vector */
namespace SparseMatrixMultiplyVector {

// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t Start,
          std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix row with a
   * vector segment.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current row index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam End End index of the current segment.
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  b Input vector to multiply with.
   * @param[out] sum Accumulator for the result.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Vector<T, N> &b, T &sum) {

    sum += A.values[Start] * b[RowIndices_A::list[Start]];
    Loop<T, M, N, RowIndices_A, RowPointers_A, J, Start + 1, End>::compute(A, b,
                                                                           sum);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t End>
struct Loop<T, M, N, RowIndices_A, RowPointers_A, J, End, End> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in sparse matrix
   * operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current row index (compile-time).
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  b Input vector to multiply with.
   * @param[out] sum Accumulator for the result.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Vector<T, N> &b, T &sum) {
    static_cast<void>(A);
    static_cast<void>(b);
    static_cast<void>(sum);
    // End of loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct Core {
  /**
   * @brief
   * Computes the multiplication of a sparse matrix row with a vector.
   *
   * This function serves as the core computation for multiplying a specific row
   * of a sparse matrix with a vector, accumulating results in the output
   * vector.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current row index (compile-time).
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  b Input vector to multiply with.
   * @param[out] y Output vector where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Vector<T, N> &b, Vector<T, M> &y) {

    T sum = static_cast<T>(0);
    Loop<T, M, N, RowIndices_A, RowPointers_A, J, RowPointers_A::list[J],
         RowPointers_A::list[J + 1]>::compute(A, b, sum);
    y[J] = sum;
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct List {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix with a vector
   * across multiple rows.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current row index (compile-time).
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  b Input vector to multiply with.
   * @param[out] y Output vector where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Vector<T, N> &b, Vector<T, M> &y) {

    Core<T, M, N, RowIndices_A, RowPointers_A, J>::compute(A, b, y);
    List<T, M, N, RowIndices_A, RowPointers_A, J - 1>::compute(A, b, y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct List<T, M, N, RowIndices_A, RowPointers_A, 0> {
  /**
   * @brief
   * Computes the multiplication of the first row of a sparse matrix with a
   * vector.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first row of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  b Input vector to multiply with.
   * @param[out] y Output vector where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Vector<T, N> &b, Vector<T, M> &y) {

    Core<T, M, N, RowIndices_A, RowPointers_A, 0>::compute(A, b, y);
  }
};

/**
 * @brief
 * Computes the multiplication of a sparse matrix with a vector.
 *
 * This function serves as the entry point for multiplying a sparse matrix by a
 * vector, producing a resulting vector.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in]  A Input sparse matrix to be multiplied.
 * @param[in]  b Input vector to multiply with.
 * @param[out] y Output vector where results are accumulated.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        const Vector<T, N> &b, Vector<T, M> &y) {
  List<T, M, N, RowIndices_A, RowPointers_A, M - 1>::compute(A, b, y);
}

} // namespace SparseMatrixMultiplyVector

/**
 * @brief
 * Multiplies a sparse matrix by a vector.
 *
 * This function creates a new vector that is the result of multiplying a sparse
 * matrix by a vector.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in] A Input sparse matrix to be multiplied.
 * @param[in] b Input vector to multiply with.
 * @return A new vector that is the result of A * b.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline Vector<T, M>
operator*(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Vector<T, N> &b) {
  Vector<T, M> y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; j++) {
    T sum = static_cast<T>(0);
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         k++) {
      sum += A.values[k] * b[RowIndices_A::list[k]];
    }
    y[j] = sum;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixMultiplyVector::compute<T, M, N, RowIndices_A, RowPointers_A>(
      A, b, y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return y;
}

/* ColVector multiply Sparse Matrix */
namespace ColVectorMultiplySparseMatrix {

// Core loop
template <typename T, std::size_t N, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t Start,
          std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the multiplication of a column vector with a sparse
   * matrix segment.
   *
   * @tparam T Value type of the matrix.
   * @tparam N Number of rows in the column vector and columns in the sparse
   * matrix.
   * @tparam K Number of rows in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current column index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam End End index of the current segment.
   * @param[in]  a Input column vector to multiply with.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] y Output column vector where results are accumulated.
   */
  static void
  compute(const ColVector<T, N> &a,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          ColVector<T, K> &y) {

    y[RowIndices_B::list[Start]] += B.values[Start] * a[J];
    Loop<T, N, K, RowIndices_B, RowPointers_B, J, Start + 1, End>::compute(a, B,
                                                                           y);
  }
};

// End of core loop
template <typename T, std::size_t N, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t End>
struct Loop<T, N, K, RowIndices_B, RowPointers_B, J, End, End> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in column vector
   * and sparse matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam N Number of rows in the column vector and columns in the sparse
   * matrix.
   * @tparam K Number of rows in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current column index (compile-time).
   * @param[in]  a Input column vector to multiply with.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] y Output column vector where results are accumulated.
   */
  static void
  compute(const ColVector<T, N> &a,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          ColVector<T, K> &y) {
    static_cast<void>(a);
    static_cast<void>(B);
    static_cast<void>(y);
    // End of loop, do nothing
  }
};

// List loop
template <typename T, std::size_t N, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J>
struct List {
  /**
   * @brief
   * Recursively computes the multiplication of a column vector with a sparse
   * matrix across multiple segments.
   *
   * @tparam T Value type of the matrix.
   * @tparam N Number of rows in the column vector and columns in the sparse
   * matrix.
   * @tparam K Number of rows in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current column index (compile-time).
   * @param[in]  a Input column vector to multiply with.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] y Output column vector where results are accumulated.
   */
  static void
  compute(const ColVector<T, N> &a,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          ColVector<T, K> &y) {

    Loop<T, N, K, RowIndices_B, RowPointers_B, J, RowPointers_B::list[J],
         RowPointers_B::list[J + 1]>::compute(a, B, y);
    List<T, N, K, RowIndices_B, RowPointers_B, J - 1>::compute(a, B, y);
  }
};

// End of list loop
template <typename T, std::size_t N, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
struct List<T, N, K, RowIndices_B, RowPointers_B, 0> {
  /**
   * @brief
   * Computes the multiplication of the first column of a column vector with a
   * sparse matrix.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first column of the column vector.
   *
   * @tparam T Value type of the matrix.
   * @tparam N Number of rows in the column vector and columns in the sparse
   * matrix.
   * @tparam K Number of rows in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @param[in]  a Input column vector to multiply with.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] y Output column vector where results are accumulated.
   */
  static void
  compute(const ColVector<T, N> &a,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          ColVector<T, K> &y) {

    Loop<T, N, K, RowIndices_B, RowPointers_B, 0, RowPointers_B::list[0],
         RowPointers_B::list[1]>::compute(a, B, y);
  }
};

/**
 * @brief
 * Computes the multiplication of a column vector with a sparse matrix.
 *
 * This function serves as the entry point for multiplying a column vector by a
 * sparse matrix, producing a resulting column vector.
 *
 * @tparam T Value type of the matrix.
 * @tparam N Number of rows in the column vector and columns in the sparse
 * matrix.
 * @tparam K Number of rows in the sparse matrix.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in]  a Input column vector to multiply with.
 * @param[in]  B Input sparse matrix to multiply with.
 * @param[out] y Output column vector where results are accumulated.
 */
template <typename T, std::size_t N, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
inline void
compute(const ColVector<T, N> &a,
        const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
        ColVector<T, K> &y) {

  List<T, N, K, RowIndices_B, RowPointers_B, N - 1>::compute(a, B, y);
}

} // namespace ColVectorMultiplySparseMatrix

/**
 * @brief
 * Multiplies a column vector by a sparse matrix.
 *
 * This function creates a new column vector that is the result of multiplying a
 * column vector by a sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam N Number of rows in the column vector and columns in the sparse
 * matrix.
 * @tparam K Number of rows in the sparse matrix.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @param[in] a Input column vector to multiply with.
 * @param[in] B Input sparse matrix to multiply with.
 * @return A new column vector that is the result of a * B.
 */
template <typename T, std::size_t N, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
inline ColVector<T, K> colVector_a_mul_SparseB(
    const ColVector<T, N> &a,
    const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B) {
  ColVector<T, K> y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < N; j++) {
    for (std::size_t k = RowPointers_B::list[j]; k < RowPointers_B::list[j + 1];
         k++) {
      y[RowIndices_B::list[k]] += B.values[k] * a[j];
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  ColVectorMultiplySparseMatrix::compute<T, N, K, RowIndices_B, RowPointers_B>(
      a, B, y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return y;
}

/* Sparse Matrix multiply Dense Matrix Transpose */
namespace SparseMatrixMultiplyDenseTranspose {

// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix row with a dense
   * matrix column segment.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current row index (compile-time).
   * @tparam I Current column index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam End End index of the current segment.
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input dense matrix to multiply with.
   * @param[out] sum Accumulator for the result.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, T &sum) {

    sum += A.values[Start] * B.template get<I, RowIndices_A::list[Start]>();
    Loop<T, M, N, K, RowIndices_A, RowPointers_A, J, I, Start + 1,
         End>::compute(A, B, sum);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t End>
struct Loop<T, M, N, K, RowIndices_A, RowPointers_A, J, I, End, End> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in sparse matrix
   * and dense matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current row index (compile-time).
   * @tparam I Current column index (compile-time).
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input dense matrix to multiply with.
   * @param[out] sum Accumulator for the result.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, T &sum) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(sum);
    // End of loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct Core {
  /**
   * @brief
   * Computes the multiplication of a sparse matrix row with a dense matrix
   * column.
   *
   * This function serves as the core computation for multiplying a specific row
   * of a sparse matrix with a column of a dense matrix, accumulating results in
   * the output matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current row index (compile-time).
   * @tparam I Current column index (compile-time).
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input dense matrix to multiply with.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, Matrix<T, M, K> &Y) {

    T sum = static_cast<T>(0);
    Loop<T, M, N, K, RowIndices_A, RowPointers_A, J, I, RowPointers_A::list[J],
         RowPointers_A::list[J + 1]>::compute(A, B, sum);
    Y(J, I) = sum;
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct List {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix with a dense
   * matrix across multiple rows.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam J Current row index (compile-time).
   * @tparam I Current column index (compile-time).
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input dense matrix to multiply with.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, Matrix<T, M, K> &Y) {

    Core<T, M, N, K, RowIndices_A, RowPointers_A, J, I>::compute(A, B, Y);
    List<T, M, N, K, RowIndices_A, RowPointers_A, J - 1, I>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct List<T, M, N, K, RowIndices_A, RowPointers_A, 0, I> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, Matrix<T, M, K> &Y) {

    Core<T, M, N, K, RowIndices_A, RowPointers_A, 0, I>::compute(A, B, Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct Column {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix with a dense
   * matrix across multiple columns.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam I Current column index (compile-time).
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input dense matrix to multiply with.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, Matrix<T, M, K> &Y) {

    List<T, M, N, K, RowIndices_A, RowPointers_A, M - 1, I>::compute(A, B, Y);
    Column<T, M, N, K, RowIndices_A, RowPointers_A, I - 1>::compute(A, B, Y);
  }
};

// End of column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A>
struct Column<T, M, N, K, RowIndices_A, RowPointers_A, 0> {
  /**
   * @brief
   * Computes the multiplication of the first column of a sparse matrix with a
   * dense matrix.
   *
   * This function is a specialization for the base case when I is 0, meaning it
   * processes the first column of the dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input dense matrix to multiply with.
   * @param[out] Y Output dense matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, Matrix<T, M, K> &Y) {

    List<T, M, N, K, RowIndices_A, RowPointers_A, M - 1, 0>::compute(A, B, Y);
  }
};

/**
 * @brief
 * Computes the multiplication of a sparse matrix with a dense matrix
 * (transposed).
 *
 * This function serves as the entry point for multiplying a sparse matrix by a
 * dense matrix, producing a resulting dense matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @tparam K Number of columns in the dense matrix.
 * @param[in]  A Input sparse matrix to be multiplied.
 * @param[in]  B Input dense matrix to multiply with.
 * @param[out] Y Output dense matrix where results are accumulated.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        const Matrix<T, K, N> &B, Matrix<T, M, K> &Y) {

  Column<T, M, N, K, RowIndices_A, RowPointers_A, K - 1>::compute(A, B, Y);
}

} // namespace SparseMatrixMultiplyDenseTranspose

/**
 * @brief
 * Multiplies a sparse matrix by the transpose of a dense matrix.
 *
 * This function creates a new matrix that is the result of multiplying a sparse
 * matrix by the transpose of a dense matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the sparse matrix.
 * @tparam N Number of rows in the sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix.
 * @tparam K Number of columns in the dense matrix.
 * @param[in] A Input sparse matrix to be multiplied.
 * @param[in] B Input dense matrix to multiply with (transposed).
 * @return A new matrix that is the result of A * B^T.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
inline Matrix<T, M, K> matrix_multiply_SparseA_mul_BTranspose(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, K, N> &B) {
  Matrix<T, M, K> Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < K; i++) {
    for (std::size_t j = 0; j < M; j++) {
      T sum = static_cast<T>(0);
      for (std::size_t k = RowPointers_A::list[j];
           k < RowPointers_A::list[j + 1]; k++) {
        sum += A.values[k] * B(i, RowIndices_A::list[k]);
      }
      Y(j, i) = sum;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixMultiplyDenseTranspose::compute<T, M, N, RowIndices_A,
                                              RowPointers_A, K>(A, B, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Sparse Matrix multiply Sparse Matrix */
namespace SparseMatrixMultiplySparse {

// Inner loop for Sparse Matrix multiply Sparse Matrix
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J, std::size_t Start, std::size_t L, std::size_t LEnd>
struct InnerLoop {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix row with another
   * sparse matrix column segment.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the output sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the output sparse matrix.
   * @tparam J Current row index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam L Current column index (compile-time).
   * @tparam LEnd End index of the current segment.
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          CompiledSparseMatrix<T, M, K, RowIndices_Y, RowPointers_Y> &Y) {

    Base::Matrix::set_sparse_matrix_value<J, RowIndices_B::list[L]>(
        Y, Base::Matrix::get_sparse_matrix_value<J, RowIndices_B::list[L]>(Y) +
               A.values[Start] * B.values[L]);
    InnerLoop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
              RowPointers_B, RowIndices_Y, RowPointers_Y, J, Start, L + 1,
              LEnd>::compute(A, B, Y);
  }
};

// End of Inner loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J, std::size_t Start, std::size_t LEnd>
struct InnerLoop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
                 RowPointers_B, RowIndices_Y, RowPointers_Y, J, Start, LEnd,
                 LEnd> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in sparse matrix
   * and sparse matrix operations.
   *
   * This specialization of the InnerLoop struct is instantiated when the L
   * index equals the LEnd index, indicating that all elements in the current
   * segment have been processed. The function does nothing, serving as the base
   * case to terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the output sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the output sparse matrix.
   * @tparam J Current row index (compile-time).
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          CompiledSparseMatrix<T, M, K, RowIndices_Y, RowPointers_Y> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of inner loop, do nothing
  }
};

// Outer loop for Sparse Matrix multiply Sparse Matrix
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J, std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix row with another
   * sparse matrix column across multiple segments.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the output sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the output sparse matrix.
   * @tparam J Current row index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam End End index of the current segment.
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          CompiledSparseMatrix<T, M, K, RowIndices_Y, RowPointers_Y> &Y) {

    InnerLoop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
              RowPointers_B, RowIndices_Y, RowPointers_Y, J, Start,
              RowPointers_B::list[RowIndices_A::list[Start]],
              RowPointers_B::list[RowIndices_A::list[Start] + 1]>::compute(A, B,
                                                                           Y);
    Loop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         RowIndices_Y, RowPointers_Y, J, Start + 1, End>::compute(A, B, Y);
  }
};

// End of Outer loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J, std::size_t End>
struct Loop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
            RowPointers_B, RowIndices_Y, RowPointers_Y, J, End, End> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in sparse matrix
   * and sparse matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the output sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the output sparse matrix.
   * @tparam J Current row index (compile-time).
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          CompiledSparseMatrix<T, M, K, RowIndices_Y, RowPointers_Y> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of outer loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J>
struct Core {
  /**
   * @brief
   * Computes the multiplication of a sparse matrix row with another sparse
   * matrix column.
   *
   * This function serves as the core computation for multiplying a specific row
   * of a sparse matrix with a column of another sparse matrix, accumulating
   * results in the output sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the output sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the output sparse matrix.
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          CompiledSparseMatrix<T, M, K, RowIndices_Y, RowPointers_Y> &Y) {

    Loop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         RowIndices_Y, RowPointers_Y, J, RowPointers_A::list[J],
         RowPointers_A::list[J + 1]>::compute(A, B, Y);
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename RowIndices_Y, typename RowPointers_Y,
          std::size_t J>
struct List {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix with another
   * sparse matrix across multiple rows.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the output sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the output sparse matrix.
   * @tparam J Current row index (compile-time).
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          CompiledSparseMatrix<T, M, K, RowIndices_Y, RowPointers_Y> &Y) {

    Core<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         RowIndices_Y, RowPointers_Y, J>::compute(A, B, Y);
    List<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         RowIndices_Y, RowPointers_Y, J - 1>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename RowIndices_Y, typename RowPointers_Y>
struct List<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
            RowPointers_B, RowIndices_Y, RowPointers_Y, 0> {
  /**
   * @brief
   * Computes the multiplication of the first row of a sparse matrix with
   * another sparse matrix.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first row of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the output sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the output sparse matrix.
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          CompiledSparseMatrix<T, M, K, RowIndices_Y, RowPointers_Y> &Y) {

    Core<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         RowIndices_Y, RowPointers_Y, 0>::compute(A, B, Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename RowIndices_Y, typename RowPointers_Y>
struct Column {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix with another
   * sparse matrix across multiple columns.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam RowIndices_Y Type representing the row indices of nonzero elements
   * in the output sparse matrix.
   * @tparam RowPointers_Y Type representing the row pointers for sparse storage
   * in the output sparse matrix.
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          CompiledSparseMatrix<T, M, K, RowIndices_Y, RowPointers_Y> &Y) {

    List<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         RowIndices_Y, RowPointers_Y, M - 1>::compute(A, B, Y);
  }
};

/**
 * @brief
 * Computes the multiplication of a sparse matrix with another sparse matrix.
 *
 * This function serves as the entry point for multiplying two sparse matrices,
 * producing a resulting sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the first sparse matrix.
 * @tparam N Number of rows in the first sparse matrix and columns in the
 * second sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the first sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the first sparse matrix.
 * @tparam K Number of rows in the second sparse matrix.
 * @param[in]  A Input sparse matrix to be multiplied.
 * @param[in]  B Input sparse matrix to multiply with.
 * @param[out] Y Output sparse matrix where results are accumulated.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, typename RowIndices_Y, typename RowPointers_Y>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
        CompiledSparseMatrix<T, M, K, RowIndices_Y, RowPointers_Y> &Y) {

  Column<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         RowIndices_Y, RowPointers_Y>::compute(A, B, Y);
}

} // namespace SparseMatrixMultiplySparse

namespace CompiledSparseOperation {

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
struct SparseMulSparse {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                  RowPointers_A>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<K, RowIndices_B,
                                                  RowPointers_B>;

  using SparseAvailable_A_mul_B =
      SparseAvailableMatrixMultiply<SparseAvailable_A, SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_A_mul_B>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_A_mul_B>;

  using Y_Type = CompiledSparseMatrix<T, M, K, RowIndices_Y, RowPointers_Y>;
};

} // namespace CompiledSparseOperation

/**
 * @brief
 * Multiplies a sparse matrix by another sparse matrix.
 *
 * This function creates a new sparse matrix that is the result of multiplying
 * two sparse matrices.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the first sparse matrix.
 * @tparam N Number of rows in the first sparse matrix and columns in the
 * second sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the first sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the first sparse matrix.
 * @tparam K Number of rows in the second sparse matrix.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the second sparse matrix.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the second sparse matrix.
 * @return A new sparse matrix that is the result of A * B.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
inline auto
operator*(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B)
    -> typename CompiledSparseOperation::SparseMulSparse<
        T, M, N, RowIndices_A, RowPointers_A, K, RowIndices_B,
        RowPointers_B>::Y_Type {

  using RowIndices_Y = typename CompiledSparseOperation::SparseMulSparse<
      T, M, N, RowIndices_A, RowPointers_A, K, RowIndices_B,
      RowPointers_B>::RowIndices_Y;

  using RowPointers_Y = typename CompiledSparseOperation::SparseMulSparse<
      T, M, N, RowIndices_A, RowPointers_A, K, RowIndices_B,
      RowPointers_B>::RowPointers_Y;

  using Y_Type = typename CompiledSparseOperation::SparseMulSparse<
      T, M, N, RowIndices_A, RowPointers_A, K, RowIndices_B,
      RowPointers_B>::Y_Type;

  Y_Type Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Matrix<T, M, K> Y_temp;

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         ++k) {
      for (std::size_t l = RowPointers_B::list[RowIndices_A::list[k]];
           l < RowPointers_B::list[RowIndices_A::list[k] + 1]; ++l) {
        Y_temp(j, RowIndices_B::list[l]) += A.values[k] * B.values[l];
      }
    }
  }

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_Y::list[j]; k < RowPointers_Y::list[j + 1];
         ++k) {
      Y.values[k] = Y_temp(j, RowIndices_Y::list[k]);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixMultiplySparse::compute<T, M, N, RowIndices_A, RowPointers_A, K,
                                      RowIndices_B, RowPointers_B, RowIndices_Y,
                                      RowPointers_Y>(A, B, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Sparse Matrix Transpose multiply Sparse Matrix */
namespace SparseMatrixTransposeMultiplySparse {

// Inner loop for Sparse Matrix Transpose multiply Sparse Matrix
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I,
          std::size_t Start, std::size_t L, std::size_t LEnd>
struct InnerLoop {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix row with another
   * sparse matrix column segment.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the dense matrix.
   * @tparam RowPointers_B Type representing the row pointers for dense storage
   * in the dense matrix.
   * @tparam Y_Type Type representing the output matrix.
   * @tparam I Current row index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam L Current column index (compile-time).
   * @tparam LEnd End index of the current segment.
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    Base::Matrix::set_sparse_matrix_value<RowIndices_A::list[Start],
                                          RowIndices_B::list[L]>(
        Y, Base::Matrix::get_sparse_matrix_value<RowIndices_A::list[Start],
                                                 RowIndices_B::list[L]>(Y) +
               A.values[Start] * B.values[L]);

    InnerLoop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
              RowPointers_B, Y_Type, I, Start, L + 1, LEnd>::compute(A, B, Y);
  }
};

// End of Inner loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I,
          std::size_t Start, std::size_t LEnd>
struct InnerLoop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
                 RowPointers_B, Y_Type, I, Start, LEnd, LEnd> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in sparse matrix
   * and sparse matrix operations.
   *
   * This specialization of the InnerLoop struct is instantiated when the L
   * index equals the LEnd index, indicating that all elements in the current
   * segment have been processed. The function does nothing, serving as the base
   * case to terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the dense matrix.
   * @tparam RowPointers_B Type representing the row pointers for dense storage
   * in the dense matrix.
   * @tparam Y_Type Type representing the output matrix.
   * @tparam I Current row index (compile-time).
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of inner loop, do nothing
  }
};

// Outer loop for Sparse Matrix Transpose multiply Sparse Matrix
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I,
          std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix row with another
   * sparse matrix column across multiple segments.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the dense matrix.
   * @tparam RowPointers_B Type representing the row pointers for dense storage
   * in the dense matrix.
   * @tparam Y_Type Type representing the output matrix.
   * @tparam I Current row index (compile-time).
   * @tparam Start Start index of the current segment.
   * @tparam End End index of the current segment.
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    InnerLoop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
              RowPointers_B, Y_Type, I, Start, RowPointers_B::list[I],
              RowPointers_B::list[I + 1]>::compute(A, B, Y);
    Loop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         Y_Type, I, Start + 1, End>::compute(A, B, Y);
  }
};

// End of Outer loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I,
          std::size_t End>
struct Loop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
            RowPointers_B, Y_Type, I, End, End> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in sparse matrix
   * and sparse matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the dense matrix.
   * @tparam RowPointers_B Type representing the row pointers for dense storage
   * in the dense matrix.
   * @tparam Y_Type Type representing the output matrix.
   * @tparam I Current row index (compile-time).
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of outer loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I>
struct Core {
  /**
   * @brief
   * Computes the multiplication of a sparse matrix row with another sparse
   * matrix column.
   *
   * This function serves as the core computation for multiplying a specific row
   * of a sparse matrix with a column of another sparse matrix, accumulating
   * results in the output sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the dense matrix.
   * @tparam RowPointers_B Type representing the row pointers for dense storage
   * in the dense matrix.
   * @tparam Y_Type Type representing the output matrix.
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    Loop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         Y_Type, I, RowPointers_A::list[I],
         RowPointers_A::list[I + 1]>::compute(A, B, Y);
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I>
struct List {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix with another
   * sparse matrix across multiple rows.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the dense matrix.
   * @tparam RowPointers_B Type representing the row pointers for dense storage
   * in the dense matrix.
   * @tparam Y_Type Type representing the output matrix.
   * @tparam I Current row index (compile-time).
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    Core<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         Y_Type, I>::compute(A, B, Y);
    List<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         Y_Type, I - 1>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type>
struct List<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
            RowPointers_B, Y_Type, 0> {
  /**
   * @brief
   * Computes the multiplication of the first row of a sparse matrix with
   * another sparse matrix.
   *
   * This function is a specialization for the base case when I is 0, meaning it
   * processes the first row of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the dense matrix.
   * @tparam RowPointers_B Type representing the row pointers for dense storage
   * in the dense matrix.
   * @tparam Y_Type Type representing the output matrix.
   * @param[in]  A Input sparse matrix to be multiplied.
   * @param[in]  B Input sparse matrix to multiply with.
   * @param[out] Y Output sparse matrix where results are accumulated.
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    Core<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         Y_Type, 0>::compute(A, B, Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type>
struct Column {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix with another
   * sparse matrix across multiple columns.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the sparse matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam K Number of columns in the dense matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the dense matrix.
   * @tparam RowPointers_B Type representing the row pointers for dense storage
   * in the dense matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    List<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         Y_Type, N - 1>::compute(A, B, Y);
  }
};

/**
 * @brief
 * Computes the multiplication of a sparse matrix with another sparse matrix.
 *
 * This function serves as the entry point for multiplying two sparse matrices,
 * producing a resulting sparse matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the first sparse matrix.
 * @tparam N Number of rows in the first sparse matrix and columns in the
 * second sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the first sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the first sparse matrix.
 * @tparam K Number of rows in the second sparse matrix.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the second sparse matrix.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the second sparse matrix.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type>
inline void
compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
        const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
        Y_Type &Y) {
  Column<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         Y_Type>::compute(A, B, Y);
}

} // namespace SparseMatrixTransposeMultiplySparse

namespace CompiledSparseOperation {

template <typename T, std::size_t M, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
struct SparseATransposeMulSparseB {

  using SparseAvailable_AT =
      SparseAvailableTranspose<CreateSparseAvailableFromIndicesAndPointers<
          M, RowIndices_A, RowPointers_A>>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<K, RowIndices_B,
                                                  RowPointers_B>;

  using SparseAvailable_AT_mul_B =
      SparseAvailableMatrixMultiply<SparseAvailable_AT, SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_AT_mul_B>;

  using RowPointers_Y =
      RowPointersFromSparseAvailable<SparseAvailable_AT_mul_B>;

  using Y_Type = CompiledSparseMatrix<T, M, K, RowIndices_Y, RowPointers_Y>;
};

} // namespace CompiledSparseOperation

/**
 * @brief
 * Multiplies the transpose of a sparse matrix by another sparse matrix.
 *
 * This function creates a new sparse matrix that is the result of multiplying
 * the transpose of a sparse matrix A with another sparse matrix B.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of rows in the first sparse matrix (A^T).
 * @tparam N Number of columns in the first sparse matrix (A^T).
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the first sparse matrix (A^T).
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the first sparse matrix (A^T).
 * @tparam K Number of rows in the second sparse matrix (B).
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the second sparse matrix (B).
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the second sparse matrix (B).
 * @return A new sparse matrix that is the result of A^T * B.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
inline auto matrix_multiply_SparseATranspose_mul_SparseB(
    const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B) ->
    typename CompiledSparseOperation::SparseATransposeMulSparseB<
        T, M, RowIndices_A, RowPointers_A, K, RowIndices_B,
        RowPointers_B>::Y_Type {

  /* Logic which realizes (A^T * B) without template */
  // for (std::size_t i = 0; i < N; i++) {
  //   for (std::size_t k = RowPointers_A::list[i]; k < RowPointers_A::list[i +
  //   1];
  //        k++) {
  //     for (std::size_t j = RowPointers_B::list[i];
  //          j < RowPointers_B::list[i + 1]; j++) {
  //       Y(RowIndices_A::list[k], RowIndices_B::list[j]) +=
  //           A.values[k] * B.values[j];
  //     }
  //   }
  // }

  using Y_Type = typename CompiledSparseOperation::SparseATransposeMulSparseB<
      T, M, RowIndices_A, RowPointers_A, K, RowIndices_B,
      RowPointers_B>::Y_Type;

  Y_Type Y;

  SparseMatrixTransposeMultiplySparse::compute<T, M, N, RowIndices_A,
                                               RowPointers_A, K, RowIndices_B,
                                               RowPointers_B, Y_Type>(A, B, Y);

  return Y;
}

/* Sparse Matrix multiply Sparse Matrix Transpose */
namespace SparseMatrixMultiplySparseTranspose {

// Core conditional operation for Sparse Matrix multiply Sparse Matrix
// Transpose
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I, std::size_t J,
          std::size_t L, std::size_t O, std::size_t L_O>
struct CoreConditional {
  /**
   * @brief
   * Computes the multiplication of a sparse matrix row with another sparse
   * matrix column segment, with a conditional operation based on the indices.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of conditional operation, do nothing
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I, std::size_t J,
          std::size_t L, std::size_t O>
struct CoreConditional<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
                       RowPointers_B, Y_Type, I, J, L, O, 0> {
  /**
   * @brief
   * Computes the multiplication of a sparse matrix row with another sparse
   * matrix column segment when the condition is met.
   *
   * This specialization is instantiated when the condition is met, meaning that
   * the indices match and the multiplication can be performed.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    Base::Matrix::set_sparse_matrix_value<I, J>(
        Y, Base::Matrix::get_sparse_matrix_value<I, J>(Y) +
               A.values[L] * B.values[O]);
  }
};

// Core inner loop for Sparse Matrix multiply Sparse Matrix Transpose
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I, std::size_t J,
          std::size_t L, std::size_t O, std::size_t O_End>
struct InnerLoop {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix row with another
   * sparse matrix column segment.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    CoreConditional<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
                    RowPointers_B, Y_Type, I, J, L, O,
                    (RowIndices_A::list[L] -
                     RowIndices_B::list[O])>::compute(A, B, Y);

    InnerLoop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
              RowPointers_B, Y_Type, I, J, L, (O + 1), (O_End - 1)>::compute(A,
                                                                             B,
                                                                             Y);
  }
};

// Core inner loop for Sparse Matrix multiply Sparse Matrix Transpose
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I, std::size_t J,
          std::size_t L, std::size_t O>
struct InnerLoop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
                 RowPointers_B, Y_Type, I, J, L, O, 0> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in sparse matrix
   * and sparse matrix operations.
   *
   * This specialization of the InnerLoop struct is instantiated when the O
   * index equals the O_End index, indicating that all elements in the current
   * segment have been processed. The function does nothing, serving as the base
   * case to terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of inner loop, do nothing
  }
};

// Core outer loop for Sparse Matrix multiply Sparse Matrix Transpose
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I, std::size_t J,
          std::size_t L, std::size_t L_End>
struct OuterLoop {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix row with another
   * sparse matrix column across multiple segments.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    InnerLoop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
              RowPointers_B, Y_Type, I, J, L, RowPointers_B::list[J],
              (RowPointers_B::list[J + 1] -
               RowPointers_B::list[J])>::compute(A, B, Y);

    OuterLoop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
              RowPointers_B, Y_Type, I, J, (L + 1), (L_End - 1)>::compute(A, B,
                                                                          Y);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I, std::size_t J,
          std::size_t L>
struct OuterLoop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
                 RowPointers_B, Y_Type, I, J, L, 0> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in sparse matrix
   * and sparse matrix operations.
   *
   * This specialization of the OuterLoop struct is instantiated when the L
   * index equals the L_End index, indicating that all elements in the current
   * segment have been processed. The function does nothing, serving as the base
   * case to terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of outer loop, do nothing
  }
};

// Core loop for Sparse Matrix multiply Sparse Matrix Transpose
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I, std::size_t J>
struct Core {
  /**
   * @brief
   * Computes the multiplication of a sparse matrix row with another sparse
   * matrix column segment.
   *
   * This function serves as the core computation for multiplying a specific row
   * of a sparse matrix with a column of another sparse matrix, accumulating
   * results in the output sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    OuterLoop<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
              RowPointers_B, Y_Type, I, J, RowPointers_A::list[I],
              (RowPointers_A::list[I + 1] -
               RowPointers_A::list[I])>::compute(A, B, Y);
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I, std::size_t J>
struct List {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix with another
   * sparse matrix across multiple rows.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    Core<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         Y_Type, I, J>::compute(A, B, Y);
    List<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         Y_Type, I - 1, J>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t J>
struct List<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
            RowPointers_B, Y_Type, 0, J> {
  /**
   * @brief
   * Computes the multiplication of the first row of a sparse matrix with
   * another sparse matrix.
   *
   * This function is a specialization for the base case when I is 0, meaning it
   * processes the first row of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    Core<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         Y_Type, 0, J>::compute(A, B, Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t J>
struct Column {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix with another
   * sparse matrix across multiple columns.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    List<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         Y_Type, M - 1, J>::compute(A, B, Y);
    Column<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
           Y_Type, J - 1>::compute(A, B, Y);
  }
};

// End of column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type>
struct Column<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B,
              RowPointers_B, Y_Type, 0> {
  /**
   * @brief
   * Computes the multiplication of the first column of a sparse matrix with
   * another sparse matrix.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first column of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of columns in the first sparse matrix.
   * @tparam N Number of rows in the first sparse matrix and columns in the
   * second sparse matrix.
   * @tparam K Number of rows in the second sparse matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the first sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the first sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the second sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the second sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    List<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         Y_Type, M - 1, 0>::compute(A, B, Y);
  }
};

/**
 * @brief
 * Computes the multiplication of a sparse matrix with another sparse matrix
 * transpose.
 *
 * This function serves as the entry point for multiplying a sparse matrix A
 * with the transpose of another sparse matrix B, producing a resulting sparse
 * matrix.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of columns in the first sparse matrix.
 * @tparam N Number of rows in the first sparse matrix and columns in the
 * second sparse matrix.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the first sparse matrix.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the first sparse matrix.
 * @tparam K Number of rows in the second sparse matrix.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the second sparse matrix.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the second sparse matrix.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
        Y_Type &Y) {
  Column<T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
         Y_Type, K - 1>::compute(A, B, Y);
}
} // namespace SparseMatrixMultiplySparseTranspose

namespace CompiledSparseOperation {

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
struct SparseAMulSparseBTranspose {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                  RowPointers_A>;
  using SparseAvailable_BT =
      SparseAvailableTranspose<CreateSparseAvailableFromIndicesAndPointers<
          N, RowIndices_B, RowPointers_B>>;

  using SparseAvailable_A_mul_BT =
      SparseAvailableMatrixMultiply<SparseAvailable_A, SparseAvailable_BT>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_A_mul_BT>;
  using RowPointers_Y =
      RowPointersFromSparseAvailable<SparseAvailable_A_mul_BT>;

  using Y_Type = CompiledSparseMatrix<T, M, K, RowIndices_Y, RowPointers_Y>;
};

} // namespace CompiledSparseOperation

/**
 * @brief
 * Multiplies a sparse matrix A with the transpose of another sparse matrix B.
 *
 * This function creates a new sparse matrix that is the result of multiplying
 * the sparse matrix A with the transpose of the sparse matrix B.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of rows in the first sparse matrix (A).
 * @tparam N Number of columns in the first sparse matrix (A).
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the first sparse matrix (A).
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the first sparse matrix (A).
 * @tparam K Number of rows in the second sparse matrix (B^T).
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the second sparse matrix (B^T).
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the second sparse matrix (B^T).
 * @return A new sparse matrix that is the result of A * B^T.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
inline auto matrix_multiply_SparseA_mul_SparseBTranspose(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B) ->
    typename CompiledSparseOperation::SparseAMulSparseBTranspose<
        T, M, N, RowIndices_A, RowPointers_A, K, RowIndices_B,
        RowPointers_B>::Y_Type {

  /* Logic which realizes (A * B^T) without template */
  // for (std::size_t i = 0; i < M; i++) {
  //   for (std::size_t j = 0; j < K; j++) {
  //     for (std::size_t l = RowPointers_A::list[i];
  //          l < RowPointers_A::list[i + 1]; l++) {
  //       for (std::size_t o = RowPointers_B::list[j];
  //            o < RowPointers_B::list[j + 1]; o++) {
  //         if (RowIndices_A::list[l] == RowIndices_B::list[o]) {
  //           Y(i, j) += A.values[l] * B.values[o];
  //         }
  //       }
  //     }
  //   }
  // }

  using Y_Type = typename CompiledSparseOperation::SparseAMulSparseBTranspose<
      T, M, N, RowIndices_A, RowPointers_A, K, RowIndices_B,
      RowPointers_B>::Y_Type;

  Y_Type Y;

  SparseMatrixMultiplySparseTranspose::compute<T, M, N, RowIndices_A,
                                               RowPointers_A, K, RowIndices_B,
                                               RowPointers_B, Y_Type>(A, B, Y);

  return Y;
}

/* Sparse Matrix multiply Diag Matrix */
namespace SparseMatrixMultiplyDiagMatrix {

// Core loop for addition
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K,
          std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix row with a
   * diagonal matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the sparse matrix.
   * @tparam N Number of columns in the sparse matrix and size of the diagonal
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, N> &B,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

    Y.values[Start] = A.values[Start] * B[RowIndices_A::list[Start]];
    Loop<T, M, N, RowIndices_A, RowPointers_A, J, K, Start + 1, End>::compute(
        A, B, Y);
  }
};

// End of core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K, std::size_t End>
struct Loop<T, M, N, RowIndices_A, RowPointers_A, J, K, End, End> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in sparse matrix
   * and diagonal matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the sparse matrix.
   * @tparam N Number of columns in the sparse matrix and size of the diagonal
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, N> &B,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct Row {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix row with a
   * diagonal matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the sparse matrix.
   * @tparam N Number of columns in the sparse matrix and size of the diagonal
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, N> &B,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, J, 0, RowPointers_A::list[J],
         RowPointers_A::list[J + 1]>::compute(A, B, Y);
    Row<T, M, N, RowIndices_A, RowPointers_A, J - 1>::compute(A, B, Y);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct Row<T, M, N, RowIndices_A, RowPointers_A, 0> {
  /**
   * @brief
   * Computes the multiplication of the first row of a sparse matrix with a
   * diagonal matrix.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first row of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the sparse matrix.
   * @tparam N Number of columns in the sparse matrix and size of the diagonal
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, N> &B,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

    Loop<T, M, N, RowIndices_A, RowPointers_A, 0, 0, RowPointers_A::list[0],
         RowPointers_A::list[1]>::compute(A, B, Y);
  }
};

/**
 * @brief
 * Computes the multiplication of a sparse matrix with a diagonal matrix.
 *
 * This function serves as the entry point for multiplying a sparse matrix A
 * with a diagonal matrix B, producing a resulting sparse matrix Y.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of rows in the sparse matrix A.
 * @tparam N Number of columns in the sparse matrix A and size of the diagonal
 * matrix B.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix A.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix A.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        const DiagMatrix<T, N> &B,
        CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {
  Row<T, M, N, RowIndices_A, RowPointers_A, M - 1>::compute(A, B, Y);
}
} // namespace SparseMatrixMultiplyDiagMatrix

/**
 * @brief
 * Multiplies a sparse matrix A with a diagonal matrix B.
 *
 * This function creates a new sparse matrix that is the result of multiplying
 * the sparse matrix A with the diagonal matrix B.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of rows in the sparse matrix (A).
 * @tparam N Number of columns in the sparse matrix (A) and size of the diagonal
 * matrix (B).
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix (A).
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix (A).
 * @return A new sparse matrix that is the result of A * B.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline auto
operator*(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, N> &B)
    -> CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> {

  CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; j++) {
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         k++) {
      Y.values[k] = A.values[k] * B[RowIndices_A::list[k]];
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixMultiplyDiagMatrix::compute<T, M, N, RowIndices_A, RowPointers_A>(
      A, B, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Diag Matrix multiply Sparse Matrix */
namespace DiagMatrixMultiplySparseMatrix {

// Core loop for addition
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t K,
          std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the multiplication of a diagonal matrix with a sparse
   * matrix row.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the diagonal matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &Y) {

    Y.values[Start] = B.values[Start] * A[J];
    Loop<T, M, N, RowIndices_B, RowPointers_B, J, K, Start + 1, End>::compute(
        A, B, Y);
  }
};

// End of core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t K, std::size_t End>
struct Loop<T, M, N, RowIndices_B, RowPointers_B, J, K, End, End> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in diagonal
   * matrix and sparse matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the diagonal matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t J>
struct Row {
  /**
   * @brief
   * Recursively computes the multiplication of a diagonal matrix with a sparse
   * matrix row.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the diagonal matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &Y) {

    Loop<T, M, N, RowIndices_B, RowPointers_B, J, 0, RowPointers_B::list[J],
         RowPointers_B::list[J + 1]>::compute(A, B, Y);
    Row<T, M, N, RowIndices_B, RowPointers_B, J - 1>::compute(A, B, Y);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
struct Row<T, M, N, RowIndices_B, RowPointers_B, 0> {
  /**
   * @brief
   * Computes the multiplication of the first row of a diagonal matrix with a
   * sparse matrix.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first row of the diagonal matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the diagonal matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &Y) {

    Loop<T, M, N, RowIndices_B, RowPointers_B, 0, 0, RowPointers_B::list[0],
         RowPointers_B::list[1]>::compute(A, B, Y);
  }
};

/**
 * @brief
 * Computes the multiplication of a diagonal matrix with a sparse matrix.
 *
 * This function serves as the entry point for multiplying a diagonal matrix A
 * with a sparse matrix B, producing a resulting sparse matrix Y.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of rows in the diagonal matrix A.
 * @tparam N Number of columns in the sparse matrix B.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the sparse matrix B.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the sparse matrix B.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
inline void
compute(const DiagMatrix<T, M> &A,
        const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
        CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &Y) {

  Row<T, M, N, RowIndices_B, RowPointers_B, M - 1>::compute(A, B, Y);
}

} // namespace DiagMatrixMultiplySparseMatrix

/**
 * @brief
 * Multiplies a diagonal matrix A with a sparse matrix B.
 *
 * This function creates a new sparse matrix that is the result of multiplying
 * the diagonal matrix A with the sparse matrix B.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of rows in the diagonal matrix (A).
 * @tparam K Number of columns in the sparse matrix (B).
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the sparse matrix (B).
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the sparse matrix (B).
 * @return A new sparse matrix that is the result of A * B.
 */
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
inline auto
operator*(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B)
    -> CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> {

  CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; j++) {
    for (std::size_t k = RowPointers_B::list[j]; k < RowPointers_B::list[j + 1];
         k++) {
      Y.values[k] = B.values[k] * A[j];
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixMultiplySparseMatrix::compute<T, M, K, RowIndices_B, RowPointers_B>(
      A, B, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Transpose (Diag Matrix multiply Sparse Matrix) */
namespace TransposeOfDiagMatrixMultiplySparse {

// Conditional operation False
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I,
          std::size_t KStart_End, std::size_t I_J>
struct ConditionalOperation {
  /**
   * @brief
   * Computes the multiplication of a diagonal matrix with a sparse matrix
   * transpose.
   *
   * This function is a specialization for the case when the condition is false,
   * meaning it does not perform any operation.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the diagonal matrix.
   * @tparam K Number of columns in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // Do nothing
  }
};

// Conditional operation True
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t I,
          std::size_t KStart_End>
struct ConditionalOperation<T, M, K, RowIndices_B, RowPointers_B, Y_Type, I,
                            KStart_End, 0> {
  /**
   * @brief
   * Computes the multiplication of a diagonal matrix with a sparse matrix
   * transpose.
   *
   * This function is a specialization for the case when the condition is true,
   * meaning it performs the multiplication operation.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the diagonal matrix.
   * @tparam K Number of columns in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    set_sparse_matrix_value<RowIndices_B::list[KStart_End], I>(
        Y, get_sparse_matrix_value<RowIndices_B::list[KStart_End], I>(Y) +
               B.values[KStart_End] * A[I]);
  }
};

// Core loop (Inner loop)
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t J, std::size_t I,
          std::size_t KStart, std::size_t KEnd>
struct InnerLoop {
  /**
   * @brief
   * Recursively computes the multiplication of a diagonal matrix with a sparse
   * matrix transpose.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the diagonal matrix.
   * @tparam K Number of columns in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   * @tparam Y_Type Type representing the output matrix.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {
    ConditionalOperation<T, M, K, RowIndices_B, RowPointers_B, Y_Type, I,
                         KStart, (I - J)>::compute(A, B, Y);
    InnerLoop<T, M, K, RowIndices_B, RowPointers_B, Y_Type, J, I, KStart + 1,
              KEnd>::compute(A, B, Y);
  }
};

// End of inner loop
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t J, std::size_t I,
          std::size_t KEnd>
struct InnerLoop<T, M, K, RowIndices_B, RowPointers_B, Y_Type, J, I, KEnd,
                 KEnd> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in diagonal
   * matrix and sparse matrix transpose operations.
   *
   * This specialization of the InnerLoop struct is instantiated when the KStart
   * index equals the KEnd index, indicating that all elements in the current
   * segment have been processed. The function does nothing, serving as the base
   * case to terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the diagonal matrix.
   * @tparam K Number of columns in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    ConditionalOperation<T, M, K, RowIndices_B, RowPointers_B, Y_Type, I, KEnd,
                         (I - J)>::compute(A, B, Y);
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t J, std::size_t I>
struct Core {
  /**
   * @brief
   * Computes the multiplication of a diagonal matrix with a sparse matrix
   * transpose.
   *
   * This function serves as the core computation for multiplying a diagonal
   * matrix A with a sparse matrix B, producing a resulting sparse matrix Y.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the diagonal matrix A.
   * @tparam K Number of columns in the sparse matrix B.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix B.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix B.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    InnerLoop<T, M, K, RowIndices_B, RowPointers_B, Y_Type, J, I,
              RowPointers_B::list[J],
              RowPointers_B::list[J + 1] - 1>::compute(A, B, Y);
  }
};

// List loop
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type, std::size_t J>
struct List {
  /**
   * @brief
   * Recursively computes the multiplication of a diagonal matrix with a sparse
   * matrix transpose.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the diagonal matrix.
   * @tparam K Number of columns in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    Core<T, M, K, RowIndices_B, RowPointers_B, Y_Type, J, J>::compute(A, B, Y);
    List<T, M, K, RowIndices_B, RowPointers_B, Y_Type, J - 1>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type>
struct List<T, M, K, RowIndices_B, RowPointers_B, Y_Type, 0> {
  /**
   * @brief
   * Computes the multiplication of the first row of a diagonal matrix with a
   * sparse matrix transpose.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first row of the diagonal matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the diagonal matrix.
   * @tparam K Number of columns in the sparse matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Y_Type &Y) {

    Core<T, M, K, RowIndices_B, RowPointers_B, Y_Type, 0, 0>::compute(A, B, Y);
  }
};

/**
 * @brief
 * Computes the multiplication of a diagonal matrix with a sparse matrix
 * transpose.
 *
 * This function serves as the entry point for multiplying a diagonal matrix A
 * with a sparse matrix B, producing a resulting sparse matrix Y.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of rows in the diagonal matrix A.
 * @tparam K Number of columns in the sparse matrix B.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the sparse matrix B.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the sparse matrix B.
 */
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, typename Y_Type>
inline void
compute(const DiagMatrix<T, M> &A,
        const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
        Y_Type &Y) {

  List<T, M, K, RowIndices_B, RowPointers_B, Y_Type, M - 1>::compute(A, B, Y);
}

} // namespace TransposeOfDiagMatrixMultiplySparse

namespace CompiledSparseOperation {

template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
struct TransposeDiagAMulSparseB {

  using SparseAvailable_BT =
      SparseAvailableTranspose<CreateSparseAvailableFromIndicesAndPointers<
          K, RowIndices_B, RowPointers_B>>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_BT>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_BT>;

  using Y_Type = CompiledSparseMatrix<T, K, M, RowIndices_Y, RowPointers_Y>;
};

} // namespace CompiledSparseOperation

/**
 * @brief
 * Computes the transpose of the multiplication of a diagonal matrix A with a
 * sparse matrix B.
 *
 * This function serves as the entry point for multiplying a diagonal matrix A
 * with a sparse matrix B, producing a resulting sparse matrix Y that is the
 * transpose of the product.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of rows in the diagonal matrix A.
 * @tparam K Number of columns in the sparse matrix B.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the sparse matrix B.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the sparse matrix B.
 */
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
inline auto matrix_multiply_Transpose_DiagA_mul_SparseB(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B) ->
    typename CompiledSparseOperation::TransposeDiagAMulSparseB<
        T, M, K, RowIndices_B, RowPointers_B>::Y_Type {

  /* Logic which realizes (A * B)^T without template */
  // for (std::size_t j = 0; j < M; j++) {
  //   for (std::size_t k = RowPointers_B::list[j]; k < RowPointers_B::list[j +
  //   1];
  //        k++) {
  //     for (std::size_t i = 0; i < M; i++) {
  //       if (i == j) {
  //         Y(RowIndices_B::list[k], i) += B.values[k] * A[i];
  //       }
  //     }
  //   }
  // }

  using Y_Type = typename CompiledSparseOperation::TransposeDiagAMulSparseB<
      T, M, K, RowIndices_B, RowPointers_B>::Y_Type;

  Y_Type Y;

  TransposeOfDiagMatrixMultiplySparse::compute<T, M, K, RowIndices_B,
                                               RowPointers_B, Y_Type>(A, B, Y);

  return Y;
}

/* Dense Matrix multiply Sparse Matrix Transpose */
namespace DenseMatrixMultiplySparseTranspose {

// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t I,
          std::size_t J, std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the multiplication of a dense matrix with a sparse
   * matrix transpose.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the dense matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam K Number of columns in the dense matrix and rows in the sparse
   * matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const Matrix<T, M, K> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          T &sum) {

    sum += B.values[Start] * A(I, RowIndices_B::list[Start]);
    Loop<T, M, N, K, RowIndices_B, RowPointers_B, I, J, Start + 1,
         End>::compute(A, B, sum);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t I,
          std::size_t J, std::size_t End>
struct Loop<T, M, N, K, RowIndices_B, RowPointers_B, I, J, End, End> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in dense matrix
   * and sparse matrix transpose operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the dense matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam K Number of columns in the dense matrix and rows in the sparse
   * matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const Matrix<T, M, K> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          T &sum) {
    // End of loop, do nothing
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(sum);
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t I,
          std::size_t J>
struct Core {
  /**
   * @brief
   * Computes the multiplication of a dense matrix with a sparse matrix
   * transpose.
   *
   * This function serves as the core computation for multiplying a dense matrix
   * A with a sparse matrix B, producing a resulting dense matrix Y.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the dense matrix A.
   * @tparam N Number of columns in the sparse matrix B.
   * @tparam K Number of columns in the dense matrix A and rows in the sparse
   * matrix B.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix B.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix B.
   */
  static void
  compute(const Matrix<T, M, K> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, N> &Y) {

    T sum = static_cast<T>(0);
    Loop<T, M, N, K, RowIndices_B, RowPointers_B, I, J, RowPointers_B::list[J],
         RowPointers_B::list[J + 1]>::compute(A, B, sum);
    Y(I, J) = sum;
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t I,
          std::size_t J>
struct List {
  /**
   * @brief
   * Recursively computes the multiplication of a dense matrix with a sparse
   * matrix transpose.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the dense matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam K Number of columns in the dense matrix and rows in the sparse
   * matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const Matrix<T, M, K> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, N> &Y) {

    Core<T, M, N, K, RowIndices_B, RowPointers_B, I, J>::compute(A, B, Y);
    List<T, M, N, K, RowIndices_B, RowPointers_B, I - 1, J>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J>
struct List<T, M, N, K, RowIndices_B, RowPointers_B, 0, J> {
  /**
   * @brief
   * Computes the multiplication of the first row of a dense matrix with a
   * sparse matrix transpose.
   *
   * This function is a specialization for the base case when I is 0, meaning it
   * processes the first row of the dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the dense matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam K Number of columns in the dense matrix and rows in the sparse
   * matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const Matrix<T, M, K> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, N> &Y) {

    Core<T, M, N, K, RowIndices_B, RowPointers_B, 0, J>::compute(A, B, Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J>
struct Column {
  /**
   * @brief
   * Recursively computes the multiplication of a dense matrix with a sparse
   * matrix transpose.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the dense matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam K Number of columns in the dense matrix and rows in the sparse
   * matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const Matrix<T, M, K> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, N> &Y) {

    List<T, M, N, K, RowIndices_B, RowPointers_B, M - 1, J>::compute(A, B, Y);
    Column<T, M, N, K, RowIndices_B, RowPointers_B, J - 1>::compute(A, B, Y);
  }
};

// End of column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B>
struct Column<T, M, N, K, RowIndices_B, RowPointers_B, 0> {
  /**
   * @brief
   * Computes the multiplication of the first column of a dense matrix with a
   * sparse matrix transpose.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first column of the dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the dense matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam K Number of columns in the dense matrix and rows in the sparse
   * matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const Matrix<T, M, K> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, N> &Y) {

    List<T, M, N, K, RowIndices_B, RowPointers_B, M - 1, 0>::compute(A, B, Y);
  }
};

/**
 * @brief
 * Computes the multiplication of a dense matrix A with the transpose of a
 * sparse matrix B, resulting in a dense matrix Y.
 *
 * This function serves as the entry point for multiplying a dense matrix A with
 * the transpose of a sparse matrix B, producing a resulting dense matrix Y.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of rows in the dense matrix A.
 * @tparam N Number of columns in the sparse matrix B.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the sparse matrix B.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the sparse matrix B.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t K>
inline void
compute(const Matrix<T, M, K> &A,
        const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
        Matrix<T, M, N> &Y) {

  Column<T, M, N, K, RowIndices_B, RowPointers_B, N - 1>::compute(A, B, Y);
}

} // namespace DenseMatrixMultiplySparseTranspose

/**
 * @brief
 * Multiplies a dense matrix A with the transpose of a sparse matrix B.
 *
 * This function creates a new dense matrix that is the result of multiplying
 * the dense matrix A with the transpose of the sparse matrix B.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of rows in the dense matrix (A).
 * @tparam N Number of columns in the sparse matrix (B).
 * @tparam K Number of columns in the dense matrix (A) and rows in the sparse
 * matrix (B).
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the sparse matrix (B).
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the sparse matrix (B).
 * @return A new dense matrix that is the result of A * B^T.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B>
inline Matrix<T, M, N> matrix_multiply_A_mul_SparseBTranspose(
    const Matrix<T, M, K> &A,
    const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, M, N> Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = static_cast<T>(0);
      for (std::size_t k = RowPointers_B::list[j];
           k < RowPointers_B::list[j + 1]; ++k) {
        sum += B.values[k] * A(i, RowIndices_B::list[k]);
      }
      Y(i, j) = sum;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DenseMatrixMultiplySparseTranspose::compute<T, M, N, RowIndices_B,
                                              RowPointers_B, K>(A, B, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Dense Transpose Matrix multiply Sparse Matrix */
namespace DenseMatrixTransposeMultiplySparse {

// Inner loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t KIndex, std::size_t I, std::size_t Start, std::size_t End>
struct InnerLoop {
  /**
   * @brief
   * Recursively computes the multiplication of a dense matrix with a sparse
   * matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the dense matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam K Number of columns in the dense matrix and rows in the sparse
   * matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const Matrix<T, N, M> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {

    Y.template set<I, RowIndices_B::list[Start]>(
        Y.template get<I, RowIndices_B::list[Start]>() +
        B.values[Start] * A.template get<J, I>());
    InnerLoop<T, M, N, K, RowIndices_B, RowPointers_B, J, KIndex, I, Start + 1,
              End>::compute(A, B, Y);
  }
};

// End of inner loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t KIndex, std::size_t I, std::size_t End>
struct InnerLoop<T, M, N, K, RowIndices_B, RowPointers_B, J, KIndex, I, End,
                 End> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in dense matrix
   * and sparse matrix operations.
   *
   * This specialization of the InnerLoop struct is instantiated when the Start
   * index equals the End index, indicating that all elements in the current
   * segment have been processed. The function does nothing, serving as the base
   * case to terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the dense matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam K Number of columns in the dense matrix and rows in the sparse
   * matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const Matrix<T, N, M> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Middle loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t KIndex, std::size_t I>
struct MiddleLoop {
  /**
   * @brief
   * Recursively computes the multiplication of a dense matrix with a sparse
   * matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the dense matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam K Number of columns in the dense matrix and rows in the sparse
   * matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const Matrix<T, N, M> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {

    InnerLoop<T, M, N, K, RowIndices_B, RowPointers_B, J, KIndex, I,
              RowPointers_B::list[J], RowPointers_B::list[J + 1]>::compute(A, B,
                                                                           Y);
    MiddleLoop<T, M, N, K, RowIndices_B, RowPointers_B, J, KIndex,
               I - 1>::compute(A, B, Y);
  }
};

// End of middle loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t KIndex>
struct MiddleLoop<T, M, N, K, RowIndices_B, RowPointers_B, J, KIndex, 0> {
  /**
   * @brief
   * Computes the multiplication of the first row of a dense matrix with a
   * sparse matrix.
   *
   * This function is a specialization for the base case when I is 0, meaning it
   * processes the first row of the dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the dense matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam K Number of columns in the dense matrix and rows in the sparse
   * matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const Matrix<T, N, M> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {

    InnerLoop<T, M, N, K, RowIndices_B, RowPointers_B, J, KIndex, 0,
              RowPointers_B::list[J], RowPointers_B::list[J + 1]>::compute(A, B,
                                                                           Y);
  }
};

// Outer loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t KIndex>
struct OuterLoop {
  /**
   * @brief
   * Recursively computes the multiplication of a dense matrix with a sparse
   * matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the dense matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam K Number of columns in the dense matrix and rows in the sparse
   * matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const Matrix<T, N, M> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {

    MiddleLoop<T, M, N, K, RowIndices_B, RowPointers_B, J, KIndex,
               M - 1>::compute(A, B, Y);
    OuterLoop<T, M, N, K, RowIndices_B, RowPointers_B, J - 1, KIndex>::compute(
        A, B, Y);
  }
};

// End of outer loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t KIndex>
struct OuterLoop<T, M, N, K, RowIndices_B, RowPointers_B, 0, KIndex> {
  /**
   * @brief
   * Computes the multiplication of the first column of a dense matrix with a
   * sparse matrix.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first column of the dense matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam M Number of rows in the dense matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam K Number of columns in the dense matrix and rows in the sparse
   * matrix.
   * @tparam RowIndices_B Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_B Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const Matrix<T, N, M> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {

    MiddleLoop<T, M, N, K, RowIndices_B, RowPointers_B, 0, KIndex,
               M - 1>::compute(A, B, Y);
  }
};

/**
 * @brief
 * Computes the multiplication of a dense matrix A with a sparse matrix B,
 * resulting in a dense matrix Y.
 *
 * This function serves as the entry point for multiplying a dense matrix A with
 * a sparse matrix B, producing a resulting dense matrix Y.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of rows in the dense matrix A.
 * @tparam N Number of columns in the sparse matrix B.
 * @tparam K Number of columns in the dense matrix A and rows in the sparse
 * matrix B.
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the sparse matrix B.
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the sparse matrix B.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B>
inline void
compute(const Matrix<T, N, M> &A,
        const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
        Matrix<T, M, K> &Y) {

  OuterLoop<T, M, N, K, RowIndices_B, RowPointers_B, N - 1, K - 1>::compute(
      A, B, Y);
}

} // namespace DenseMatrixTransposeMultiplySparse

/**
 * @brief
 * Multiplies a dense matrix A with the transpose of a sparse matrix B.
 *
 * This function creates a new dense matrix that is the result of multiplying
 * the dense matrix A with the transpose of the sparse matrix B.
 *
 * @tparam T Value type of the matrix.
 * @tparam M Number of rows in the dense matrix (A).
 * @tparam N Number of columns in the sparse matrix (B).
 * @tparam K Number of columns in the dense matrix (A) and rows in the sparse
 * matrix (B).
 * @tparam RowIndices_B Type representing the row indices of nonzero elements
 * in the sparse matrix (B).
 * @tparam RowPointers_B Type representing the row pointers for sparse storage
 * in the sparse matrix (B).
 * @return A new dense matrix that is the result of A^T * B.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B>
inline Matrix<T, M, K> matrix_multiply_ATranspose_mul_SparseB(
    const Matrix<T, N, M> &A,
    const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, M, K> Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < N; j++) {
    for (std::size_t k = RowPointers_B::list[j]; k < RowPointers_B::list[j + 1];
         k++) {
      for (std::size_t i = 0; i < M; i++) {
        Y(i, RowIndices_B::list[k]) += B.values[k] * A(j, i);
      }
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DenseMatrixTransposeMultiplySparse::compute<T, M, N, K, RowIndices_B,
                                              RowPointers_B>(A, B, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Sparse Matrix transpose multiply Dense Matrix */
namespace SparseTransposeMatrixMultiplyDenseMatrix {

// Start < End (Core)
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix with a dense
   * matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam M Number of columns in the dense matrix.
   * @tparam K Number of columns in the sparse matrix and rows in the dense
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {

    Y.template set<RowIndices_A::list[Start], I>(
        Y.template get<RowIndices_A::list[Start], I>() +
        A.values[Start] * B.template get<J, I>());
    Loop<T, N, M, K, RowIndices_A, RowPointers_A, J, I, Start + 1,
         End>::compute(A, B, Y);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t End>
struct Loop<T, N, M, K, RowIndices_A, RowPointers_A, J, I, End, End> {
  /**
   * @brief
   * End condition for the recursive multiplication computation in sparse matrix
   * and dense matrix operations.
   *
   * This specialization of the Loop struct is instantiated when the Start index
   * equals the End index, indicating that all elements in the current segment
   * have been processed. The function does nothing, serving as the base case to
   * terminate recursion.
   *
   * @tparam T Value type of the matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam M Number of columns in the dense matrix.
   * @tparam K Number of columns in the sparse matrix and rows in the dense
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct Core {
  /**
   * @brief
   * Computes the multiplication of a sparse matrix with a dense matrix.
   *
   * This function serves as the core computation for multiplying a sparse
   * matrix A with a dense matrix B, producing a resulting dense matrix Y.
   *
   * @tparam T Value type of the matrix.
   * @tparam N Number of rows in the sparse matrix A.
   * @tparam M Number of columns in the dense matrix B.
   * @tparam K Number of columns in the sparse matrix A and rows in the dense
   * matrix B.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix A.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix A.
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {

    Loop<T, N, M, K, RowIndices_A, RowPointers_A, J, I, RowPointers_A::list[J],
         RowPointers_A::list[J + 1]>::compute(A, B, Y);
  }
};

// List loop
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct List {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix with a dense
   * matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam M Number of columns in the dense matrix.
   * @tparam K Number of columns in the sparse matrix and rows in the dense
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {

    Core<T, N, M, K, RowIndices_A, RowPointers_A, J, I>::compute(A, B, Y);
    List<T, N, M, K, RowIndices_A, RowPointers_A, J - 1, I>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct List<T, N, M, K, RowIndices_A, RowPointers_A, 0, I> {
  /**
   * @brief
   * Computes the multiplication of the first column of a sparse matrix with a
   * dense matrix.
   *
   * This function is a specialization for the base case when J is 0, meaning it
   * processes the first column of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam M Number of columns in the dense matrix.
   * @tparam K Number of columns in the sparse matrix and rows in the dense
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {

    Core<T, N, M, K, RowIndices_A, RowPointers_A, 0, I>::compute(A, B, Y);
  }
};

// Column loop
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct Column {
  /**
   * @brief
   * Recursively computes the multiplication of a sparse matrix with a dense
   * matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam M Number of columns in the dense matrix.
   * @tparam K Number of columns in the sparse matrix and rows in the dense
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {

    List<T, N, M, K, RowIndices_A, RowPointers_A, N - 1, I>::compute(A, B, Y);
    Column<T, N, M, K, RowIndices_A, RowPointers_A, I - 1>::compute(A, B, Y);
  }
};

// End of column loop
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A>
struct Column<T, N, M, K, RowIndices_A, RowPointers_A, 0> {
  /**
   * @brief
   * Computes the multiplication of the first column of a sparse matrix with a
   * dense matrix.
   *
   * This function is a specialization for the base case when I is 0, meaning it
   * processes the first column of the sparse matrix.
   *
   * @tparam T Value type of the matrix.
   * @tparam N Number of rows in the sparse matrix.
   * @tparam M Number of columns in the dense matrix.
   * @tparam K Number of columns in the sparse matrix and rows in the dense
   * matrix.
   * @tparam RowIndices_A Type representing the row indices of nonzero elements
   * in the sparse matrix.
   * @tparam RowPointers_A Type representing the row pointers for sparse storage
   * in the sparse matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {

    List<T, N, M, K, RowIndices_A, RowPointers_A, N - 1, 0>::compute(A, B, Y);
  }
};

/**
 * @brief
 * Computes the multiplication of a sparse matrix A with a dense matrix B,
 * resulting in a dense matrix Y.
 *
 * This function serves as the entry point for multiplying a sparse matrix A
 * with a dense matrix B, producing a resulting dense matrix Y.
 *
 * @tparam T Value type of the matrix.
 * @tparam N Number of rows in the sparse matrix A.
 * @tparam M Number of columns in the dense matrix B.
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix A.
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix A.
 */
template <typename T, std::size_t N, std::size_t M, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
inline void
compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
        const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {

  Column<T, N, M, K, RowIndices_A, RowPointers_A, K - 1>::compute(A, B, Y);
}

} // namespace SparseTransposeMatrixMultiplyDenseMatrix

/**
 * @brief
 * Multiplies a sparse matrix A with a dense matrix B.
 *
 * This function creates a new dense matrix that is the result of multiplying
 * the sparse matrix A with the dense matrix B.
 *
 * @tparam T Value type of the matrix.
 * @tparam N Number of rows in the sparse matrix (A).
 * @tparam M Number of columns in the dense matrix (B).
 * @tparam K Number of columns in the sparse matrix (A) and rows in the dense
 * matrix (B).
 * @tparam RowIndices_A Type representing the row indices of nonzero elements
 * in the sparse matrix (A).
 * @tparam RowPointers_A Type representing the row pointers for sparse storage
 * in the sparse matrix (A).
 * @return A new dense matrix that is the result of A * B.
 */
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A>
inline Matrix<T, M, K> matrix_multiply_SparseAT_mul_B(
    const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, N, K> &B) {
  Matrix<T, M, K> Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < N; j++) {
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         k++) {
      for (std::size_t i = 0; i < M; i++) {
        Y(RowIndices_A::list[k], i) += A.values[k] * B(j, i);
      }
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseTransposeMatrixMultiplyDenseMatrix::compute<T, N, M, RowIndices_A,
                                                    RowPointers_A, K>(A, B, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_COMPILED_SPARSE_OPERATION_HPP__
