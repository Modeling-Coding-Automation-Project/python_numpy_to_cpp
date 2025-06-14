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

/* Set values for Upper Triangular Sparse Matrix */
namespace SetValuesForUpperTriangularSparseMatrix {

/**
 * @brief Calculate consecutive index at compile time for upper triangular
 * matrix.
 *
 * This struct computes the index in a flattened array representation of an
 * upper triangular matrix, given the row (I) and column (J) indices.
 *
 * @tparam I Row index.
 * @tparam J Column index.
 * @tparam N Total number of columns in the matrix.
 */
template <std::size_t I, std::size_t J, std::size_t N> struct ConsecutiveIndex {
  static constexpr std::size_t value = (I * (2 * N - I + 1)) / 2 + (J - I);
};

/**
 * @brief Specialization for the base case when both I and J are 0.
 *
 * This specialization provides a base case for the ConsecutiveIndex struct,
 * returning 0 when both indices are 0.
 *
 * @tparam N Total number of columns in the matrix.
 */
template <std::size_t N> struct ConsecutiveIndex<0, 0, N> {
  static constexpr std::size_t value = 0;
};

// Set values in the upper triangular matrix
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J>
struct SetUpperValues {
  /**
   * @brief Computes the index for the upper triangular matrix and sets the
   * value.
   *
   * This function computes the consecutive index for the given row (I) and
   * column (J) in an upper triangular matrix and sets the corresponding value
   * in the CompiledSparseMatrix.
   *
   * @param A The CompiledSparseMatrix where values are set.
   * @param B The source Matrix from which values are taken.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                               UpperTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    constexpr std::size_t index = ConsecutiveIndex<I, J, N>::value;
    A.values[index] = B(I, J);
    SetUpperValues<T, M, N, I, J - 1>::compute(A, B);
  }
};

// Specialization for the end of a row
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct SetUpperValues<T, M, N, I, I> {
  /**
   * @brief Computes the index for the last element in a row of the upper
   * triangular matrix and sets the value.
   *
   * This function computes the consecutive index for the last element in the
   * row (I) of an upper triangular matrix and sets the corresponding value in
   * the CompiledSparseMatrix.
   *
   * @param A The CompiledSparseMatrix where values are set.
   * @param B The source Matrix from which values are taken.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                               UpperTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    constexpr std::size_t index = ConsecutiveIndex<I, I, N>::value;
    A.values[index] = B(I, I);
  }
};

// Set values for each row
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct SetUpperRow {
  /**
   * @brief Computes the values for a specific row (I) in the upper triangular
   * matrix.
   *
   * This function sets the values for all elements in row I of the upper
   * triangular matrix and recursively calls itself for the previous row.
   *
   * @param A The CompiledSparseMatrix where values are set.
   * @param B The source Matrix from which values are taken.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                               UpperTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    SetUpperValues<T, M, N, I, N - 1>::compute(A, B);
    SetUpperRow<T, M, N, I - 1>::compute(A, B);
  }
};

// Specialization for the first row
template <typename T, std::size_t M, std::size_t N>
struct SetUpperRow<T, M, N, 0> {
  /**
   * @brief Computes the values for the first row (0) in the upper triangular
   * matrix.
   *
   * This function sets the values for all elements in the first row of the
   * upper triangular matrix.
   *
   * @param A The CompiledSparseMatrix where values are set.
   * @param B The source Matrix from which values are taken.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                               UpperTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    SetUpperValues<T, M, N, 0, N - 1>::compute(A, B);
  }
};

/**
 * @brief Computes the values for an upper triangular sparse matrix.
 *
 * This function initializes the CompiledSparseMatrix with values from the
 * provided Matrix, specifically for the upper triangular part.
 *
 * @param A The CompiledSparseMatrix where values are set.
 * @param B The source Matrix from which values are taken.
 */
template <typename T, std::size_t M, std::size_t N>
inline void
compute(CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                             UpperTriangularRowPointers<M, N>> &A,
        const Matrix<T, M, N> &B) {
  SetUpperRow<T, M, N, M - 1>::compute(A, B);
}

} // namespace SetValuesForUpperTriangularSparseMatrix

/* Set values for Lower Triangular Sparse Matrix */
namespace SetValuesForLowerTriangularSparseMatrix {

/**
 * @brief Calculate consecutive index at compile time for lower triangular
 * matrix.
 *
 * This struct computes the index in a flattened array representation of a
 * lower triangular matrix, given the row (I) and column (J) indices.
 *
 * @tparam I Column index.
 * @tparam J Row index.
 * @tparam N Total number of columns in the matrix.
 */
template <std::size_t I, std::size_t J, std::size_t N>
struct ConsecutiveIndexLower {
  static constexpr std::size_t value = (I * (I + 1)) / 2 + J;
};

/**
 * @brief Specialization for the base case when both I and J are 0.
 *
 * This specialization provides a base case for the ConsecutiveIndexLower
 * struct, returning 0 when both indices are 0.
 *
 * @tparam N Total number of columns in the matrix.
 */
template <std::size_t N> struct ConsecutiveIndexLower<0, 0, N> {
  static constexpr std::size_t value = 0;
};

// Set values in the lower triangular matrix
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J>
struct SetLowerValues {
  /**
   * @brief Computes the index for the lower triangular matrix and sets the
   * value.
   *
   * This function computes the consecutive index for the given column (I) and
   * row (J) in a lower triangular matrix and sets the corresponding value
   * in the CompiledSparseMatrix.
   *
   * @param A The CompiledSparseMatrix where values are set.
   * @param B The source Matrix from which values are taken.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                               LowerTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    constexpr std::size_t index = ConsecutiveIndexLower<I, J, N>::value;
    A.values[index] = B(I, J);
    SetLowerValues<T, M, N, I, J - 1>::compute(A, B);
  }
};

// Specialization for the end of a row
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct SetLowerValues<T, M, N, I, 0> {
  /**
   * @brief Computes the index for the last element in a row of the lower
   * triangular matrix and sets the value.
   *
   * This function computes the consecutive index for the last element in the
   * column (I) of a lower triangular matrix and sets the corresponding value
   * in the CompiledSparseMatrix.
   *
   * @param A The CompiledSparseMatrix where values are set.
   * @param B The source Matrix from which values are taken.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                               LowerTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    constexpr std::size_t index = ConsecutiveIndexLower<I, 0, N>::value;
    A.values[index] = B(I, 0);
  }
};

// Set values for each row
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct SetLowerRow {
  /**
   * @brief Computes the values for a specific row (I) in the lower triangular
   * matrix.
   *
   * This function sets the values for all elements in row I of the lower
   * triangular matrix and recursively calls itself for the previous row.
   *
   * @param A The CompiledSparseMatrix where values are set.
   * @param B The source Matrix from which values are taken.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                               LowerTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    SetLowerValues<T, M, N, I, I>::compute(A, B);
    SetLowerRow<T, M, N, I - 1>::compute(A, B);
  }
};

// Specialization for the first row
template <typename T, std::size_t M, std::size_t N>
struct SetLowerRow<T, M, N, 0> {
  /**
   * @brief Computes the values for the first row (0) in the lower triangular
   * matrix.
   *
   * This function sets the values for all elements in the first row of the
   * lower triangular matrix.
   *
   * @param A The CompiledSparseMatrix where values are set.
   * @param B The source Matrix from which values are taken.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                               LowerTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    SetLowerValues<T, M, N, 0, 0>::compute(A, B);
  }
};

/**
 * @brief Computes the values for a lower triangular sparse matrix.
 *
 * This function initializes the CompiledSparseMatrix with values from the
 * provided Matrix, specifically for the lower triangular part.
 *
 * @param A The CompiledSparseMatrix where values are set.
 * @param B The source Matrix from which values are taken.
 */
template <typename T, std::size_t M, std::size_t N>
inline void
compute(CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                             LowerTriangularRowPointers<M, N>> &A,
        const Matrix<T, M, N> &B) {
  SetLowerRow<T, M, N, M - 1>::compute(A, B);
}

} // namespace SetValuesForLowerTriangularSparseMatrix

/**
 * @brief Class for handling triangular sparse matrices.
 *
 * This class provides static methods to create and manipulate upper and lower
 * triangular sparse matrices. It supports compile-time optimizations for
 * matrix dimensions and provides methods to set values from a regular matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 */
template <typename T, std::size_t M, std::size_t N> class TriangularSparse {
public:
  /* Constructor */
  TriangularSparse() {}

  /* Upper */

  /**
   * @brief Creates an upper triangular sparse matrix.
   *
   * This function initializes a CompiledSparseMatrix with the appropriate row
   * indices and row pointers for an upper triangular matrix.
   *
   * @return A CompiledSparseMatrix representing an upper triangular matrix.
   */
  static inline auto create_upper(void)
      -> CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                              UpperTriangularRowPointers<M, N>> {
    // Currently, only support M >= N.
    static_assert(M >= N, "M must be greater than or equal to N");

    CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                         UpperTriangularRowPointers<M, N>>
        Y;

    return Y;
  }

  /**
   * @brief Creates an upper triangular sparse matrix from a regular matrix.
   *
   * This function initializes a CompiledSparseMatrix with values from the
   * provided Matrix, specifically for the upper triangular part.
   *
   * @param A The source Matrix from which values are taken.
   * @return A CompiledSparseMatrix representing an upper triangular matrix.
   */
  static inline auto create_upper(const Matrix<T, M, N> &A)
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

    SetValuesForUpperTriangularSparseMatrix::compute<T, M, N>(Y, A);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    return Y;
  }

  /**
   * @brief Sets values in an upper triangular sparse matrix from a regular
   * matrix.
   *
   * This function populates the CompiledSparseMatrix with values from the
   * provided Matrix, specifically for the upper triangular part.
   *
   * @param A The CompiledSparseMatrix where values are set.
   * @param B The source Matrix from which values are taken.
   */
  static inline void set_values_upper(
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

    SetValuesForUpperTriangularSparseMatrix::compute<T, M, N>(A, B);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
  }

  /* Lower */

  /**
   * @brief Creates a lower triangular sparse matrix.
   *
   * This function initializes a CompiledSparseMatrix with the appropriate row
   * indices and row pointers for a lower triangular matrix.
   *
   * @return A CompiledSparseMatrix representing a lower triangular matrix.
   */
  static inline auto create_lower(void)
      -> CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                              LowerTriangularRowPointers<M, N>> {
    // Currently, only support M <= N.
    static_assert(M <= N, "M must be smaller than or equal to N");

    CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                         LowerTriangularRowPointers<M, N>>
        Y;

    return Y;
  }

  /**
   * @brief Creates a lower triangular sparse matrix from a regular matrix.
   *
   * This function initializes a CompiledSparseMatrix with values from the
   * provided Matrix, specifically for the lower triangular part.
   *
   * @param A The source Matrix from which values are taken.
   * @return A CompiledSparseMatrix representing a lower triangular matrix.
   */
  static inline auto create_lower(const Matrix<T, M, N> &A)
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

    SetValuesForLowerTriangularSparseMatrix::compute<T, M, N>(Y, A);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    return Y;
  }

  /**
   * @brief Sets values in a lower triangular sparse matrix from a regular
   * matrix.
   *
   * This function populates the CompiledSparseMatrix with values from the
   * provided Matrix, specifically for the lower triangular part.
   *
   * @param A The CompiledSparseMatrix where values are set.
   * @param B The source Matrix from which values are taken.
   */
  static inline void set_values_lower(
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

    SetValuesForLowerTriangularSparseMatrix::compute<T, M, N>(A, B);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
  }
};

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_TRIANGULAR_SPARSE_HPP__
