#ifndef BASE_MATRIX_TRIANGULAR_SPARSE_HPP
#define BASE_MATRIX_TRIANGULAR_SPARSE_HPP

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
// Calculate consecutive index at compile time
template <std::size_t I, std::size_t J, std::size_t N> struct ConsecutiveIndex {
  static constexpr std::size_t value = (I * (2 * N - I + 1)) / 2 + (J - I);
};

// Specialization for the base case
template <std::size_t N> struct ConsecutiveIndex<0, 0, N> {
  static constexpr std::size_t value = 0;
};

// Set values in the upper triangular matrix
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J>
struct SetUpperValues {
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
  static void
  compute(CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                               UpperTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    SetUpperValues<T, M, N, 0, N - 1>::compute(A, B);
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline void SET_UPPER_TRIANGULAR_VALUES(
    CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                         UpperTriangularRowPointers<M, N>> &A,
    const Matrix<T, M, N> &B) {
  SetUpperRow<T, M, N, M - 1>::compute(A, B);
}

/* Set values for Lower Triangular Sparse Matrix */
// Calculate consecutive index at compile time for lower triangular matrix
template <std::size_t I, std::size_t J, std::size_t N>
struct ConsecutiveIndexLower {
  static constexpr std::size_t value = (I * (I + 1)) / 2 + J;
};

// Specialization for the base case
template <std::size_t N> struct ConsecutiveIndexLower<0, 0, N> {
  static constexpr std::size_t value = 0;
};

// Set values in the lower triangular matrix
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J>
struct SetLowerValues {
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
  static void
  compute(CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                               LowerTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    SetLowerValues<T, M, N, 0, 0>::compute(A, B);
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline void SET_LOWER_TRIANGULAR_VALUES(
    CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                         LowerTriangularRowPointers<M, N>> &A,
    const Matrix<T, M, N> &B) {
  SetLowerRow<T, M, N, M - 1>::compute(A, B);
}

template <typename T, std::size_t M, std::size_t N> class TriangularSparse {
public:
  TriangularSparse() {}

  /* Upper */
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

  static inline auto create_upper(const Matrix<T, M, N> &A)
      -> CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                              UpperTriangularRowPointers<M, N>> {
    // Currently, only support M >= N.
    static_assert(M >= N, "M must be greater than or equal to N");

    CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                         UpperTriangularRowPointers<M, N>>
        Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

    std::size_t consecutive_index = 0;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = i; j < N; j++) {
        Y.values[consecutive_index] = A(i, j);

        consecutive_index++;
      }
    }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION

    Base::Matrix::SET_UPPER_TRIANGULAR_VALUES<T, M, N>(Y, A);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION

    return Y;
  }

  static inline void set_values_upper(
      CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                           UpperTriangularRowPointers<M, N>> &A,
      const Matrix<T, M, N> &B) {
    // Currently, only support M >= N.
    static_assert(M >= N, "M must be greater than or equal to N");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

    std::size_t consecutive_index = 0;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = i; j < N; j++) {
        A.values[consecutive_index] = B(i, j);
        consecutive_index++;
      }
    }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION

    Base::Matrix::SET_UPPER_TRIANGULAR_VALUES<T, M, N>(A, B);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION
  }

  /* Lower */
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

  static inline auto create_lower(const Matrix<T, M, N> &A)
      -> CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                              LowerTriangularRowPointers<M, N>> {
    // Currently, only support M <= N.
    static_assert(M <= N, "M must be smaller than or equal to N");

    CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                         LowerTriangularRowPointers<M, N>>
        Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

    std::size_t consecutive_index = 0;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < i + 1; j++) {
        Y.values[consecutive_index] = A(i, j);

        consecutive_index++;
      }
    }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION

    Base::Matrix::SET_LOWER_TRIANGULAR_VALUES<T, M, N>(Y, A);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION

    return Y;
  }

  static inline void set_values_lower(
      CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                           LowerTriangularRowPointers<M, N>> &A,
      const Matrix<T, M, N> &B) {
    // Currently, only support M <= N.
    static_assert(M <= N, "M must be smaller than or equal to N");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

    std::size_t consecutive_index = 0;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < i + 1; j++) {
        A.values[consecutive_index] = B(i, j);
        consecutive_index++;
      }
    }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION

    Base::Matrix::SET_LOWER_TRIANGULAR_VALUES<T, M, N>(A, B);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION
  }
};

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_TRIANGULAR_SPARSE_HPP
