#ifndef __BASE_MATRIX_COMPILED_SPARSE_HPP__
#define __BASE_MATRIX_COMPILED_SPARSE_HPP__

#include "base_matrix_macros.hpp"

#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_templates.hpp"
#include "base_matrix_vector.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
class CompiledSparseMatrix {
public:
#ifdef __BASE_MATRIX_USE_STD_VECTOR__

  CompiledSparseMatrix() : values(RowPointers::list[M], static_cast<T>(0)) {}

  CompiledSparseMatrix(const std::initializer_list<T> &values)
      : values(values) {}

  CompiledSparseMatrix(const std::vector<T> &values) : values(values) {}

#else // __BASE_MATRIX_USE_STD_VECTOR__

  CompiledSparseMatrix() : values{} {}

  CompiledSparseMatrix(const std::initializer_list<T> &values) : values{} {

    // This may cause runtime error if the size of values is larger than
    // RowIndices::size.
    std::copy(values.begin(), values.end(), this->values.begin());
  }

  CompiledSparseMatrix(const std::array<T, RowIndices::size> &values)
      : values(values) {}

  CompiledSparseMatrix(const std::vector<T> &values) : values{} {

    // This may cause runtime error if the size of values is larger than
    // RowIndices::size.
    std::copy(values.begin(), values.end(), this->values.begin());
  }

#endif // __BASE_MATRIX_USE_STD_VECTOR__

  /* Copy Constructor */
  CompiledSparseMatrix(
      const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &other)
      : values(other.values) {}

  CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &operator=(
      const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &other) {
    if (this != &other) {
      this->values = other.values;
    }
    return *this;
  }

  /* Move Constructor */
  CompiledSparseMatrix(
      CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &&other) noexcept
      : values(std::move(other.values)) {}

  CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &operator=(
      CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &&other) noexcept {
    if (this != &other) {
      this->values = std::move(other.values);
    }
    return *this;
  }

  /* Variable */
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<T> values;
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<T, RowPointers::list[M]> values;
#endif // __BASE_MATRIX_USE_STD_VECTOR__
};

/* Output dense matrix */
// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K, std::size_t Start,
          std::size_t End>
struct OutputDenseMatrixLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {
    result(J, RowIndices::list[Start]) = mat.values[Start];
    OutputDenseMatrixLoop<T, M, N, RowIndices, RowPointers, J, K, Start + 1,
                          End>::compute(mat, result);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K, std::size_t End>
struct OutputDenseMatrixLoop<T, M, N, RowIndices, RowPointers, J, K, End, End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {
    static_cast<void>(mat);
    static_cast<void>(result);
    // End of loop, do nothing
  }
};

// Pointer loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K>
struct OutputDenseMatrixCore {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {
    OutputDenseMatrixLoop<T, M, N, RowIndices, RowPointers, J, K,
                          RowPointers::list[J],
                          RowPointers::list[J + 1]>::compute(mat, result);
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J>
struct OutputDenseMatrixRow {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {
    OutputDenseMatrixCore<T, M, N, RowIndices, RowPointers, J, 0>::compute(
        mat, result);
    OutputDenseMatrixRow<T, M, N, RowIndices, RowPointers, J - 1>::compute(
        mat, result);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct OutputDenseMatrixRow<T, M, N, RowIndices, RowPointers, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {
    OutputDenseMatrixCore<T, M, N, RowIndices, RowPointers, 0, 0>::compute(
        mat, result);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
static inline void COMPILED_SPARSE_OUTPUT_DENSE_MATRIX(
    const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
    Matrix<T, M, N> &result) {
  OutputDenseMatrixRow<T, M, N, RowIndices, RowPointers, M - 1>::compute(
      mat, result);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline Matrix<T, M, N> output_dense_matrix(
    const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; j++) {
    for (std::size_t k = RowPointers::list[j]; k < RowPointers::list[j + 1];
         k++) {
      result(j, RowIndices::list[k]) = mat.values[k];
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_SPARSE_OUTPUT_DENSE_MATRIX<T, M, N, RowIndices,
                                                    RowPointers>(mat, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Substitute Dense Matrix to Sparse Matrix */
// when J_idx < N
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I, std::size_t J_idx>
struct DenseToSparseMatrixSubstituteColumn {
  static void
  compute(const Matrix<T, M, N> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

    Y.values[I * N + J_idx] = A.template get<I, J_idx>();
    DenseToSparseMatrixSubstituteColumn<T, M, N, RowIndices_A, RowPointers_A, I,
                                        J_idx - 1>::compute(A, Y);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I>
struct DenseToSparseMatrixSubstituteColumn<T, M, N, RowIndices_A, RowPointers_A,
                                           I, 0> {
  static void
  compute(const Matrix<T, M, N> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

    Y.values[I * N] = A.template get<I, 0>();
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I_idx>
struct DenseToSparseMatrixSubstituteRow {
  static void
  compute(const Matrix<T, M, N> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

    DenseToSparseMatrixSubstituteColumn<T, M, N, RowIndices_A, RowPointers_A,
                                        I_idx, N - 1>::compute(A, Y);
    DenseToSparseMatrixSubstituteRow<T, M, N, RowIndices_A, RowPointers_A,
                                     I_idx - 1>::compute(A, Y);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct DenseToSparseMatrixSubstituteRow<T, M, N, RowIndices_A, RowPointers_A,
                                        0> {
  static void
  compute(const Matrix<T, M, N> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {
    DenseToSparseMatrixSubstituteColumn<T, M, N, RowIndices_A, RowPointers_A, 0,
                                        N - 1>::compute(A, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
static inline void COMPILED_DENSE_MATRIX_SUBSTITUTE_SPARSE(
    const Matrix<T, M, N> &A,
    CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {
  DenseToSparseMatrixSubstituteRow<T, M, N, RowIndices_A, RowPointers_A,
                                   M - 1>::compute(A, Y);
}

template <typename T, std::size_t M, std::size_t N>
inline auto create_compiled_sparse(const Matrix<T, M, N> &A)
    -> CompiledSparseMatrix<T, M, N, DenseMatrixRowIndices<M, N>,
                            DenseMatrixRowPointers<M, N>> {
  CompiledSparseMatrix<T, M, N, DenseMatrixRowIndices<M, N>,
                       DenseMatrixRowPointers<M, N>>
      Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  std::size_t consecutive_index = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < N; j++) {
      Y.values[consecutive_index] = A(i, j);
      consecutive_index++;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DENSE_MATRIX_SUBSTITUTE_SPARSE<
      T, M, N, DenseMatrixRowIndices<M, N>, DenseMatrixRowPointers<M, N>>(A, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Create Sparse Matrix from Diag Matrix */
template <std::size_t M>
using DiagMatrixRowIndices = typename ToRowIndices<MatrixRowNumbers<M>>::type;

template <std::size_t M>
using DiagMatrixRowPointers =
    typename ToRowIndices<MatrixRowNumbers<(M + 1)>>::type;

template <typename T, std::size_t M>
inline auto create_compiled_sparse(const DiagMatrix<T, M> &A)
    -> CompiledSparseMatrix<T, M, M, DiagMatrixRowIndices<M>,
                            DiagMatrixRowPointers<M>> {
  CompiledSparseMatrix<T, M, M, DiagMatrixRowIndices<M>,
                       DiagMatrixRowPointers<M>>
      Y;

  Y.values = A.data;

  return Y;
}

/* Create Compiled Sparse Matrix from SparseAvailable */
template <typename T, typename SparseAvailable>
inline auto create_compiled_sparse(std::initializer_list<T> values)
    -> CompiledSparseMatrix<T, SparseAvailable::number_of_columns,
                            SparseAvailable::column_size,
                            RowIndicesFromSparseAvailable<SparseAvailable>,
                            RowPointersFromSparseAvailable<SparseAvailable>> {
  CompiledSparseMatrix<T, SparseAvailable::number_of_columns,
                       SparseAvailable::column_size,
                       RowIndicesFromSparseAvailable<SparseAvailable>,
                       RowPointersFromSparseAvailable<SparseAvailable>>
      Y;

  // This may cause runtime error if the size of values is larger than
  // RowIndices::size.
  std::copy(values.begin(),
            values.begin() +
                RowIndicesFromSparseAvailable<SparseAvailable>::size,
            Y.values.begin());

  return Y;
}

/* Set Sparse Matrix Value */
// check if RowToSet == RowIndices_A::list[K]
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, std::size_t RowToGet_I>
struct SetSparseMatrixValueCoreIf {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {
    /* Do nothing */
    static_cast<void>(A);
    static_cast<void>(value);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
struct SetSparseMatrixValueCoreIf<T, M, N, RowIndices_A, RowPointers_A, K, 0> {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    A.values[K] = value;
  }
};

// Core conditional operation for setting sparse matrix value
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K, std::size_t L>
struct SetSparseMatrixValueCoreConditional {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {
    /* Do nothing */
    static_cast<void>(A);
    static_cast<void>(value);
  }
};

template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K>
struct SetSparseMatrixValueCoreConditional<
    ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A, J, K, 0> {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    SetSparseMatrixValueCoreIf<T, M, N, RowIndices_A, RowPointers_A, K,
                               (RowToSet -
                                RowIndices_A::list[K])>::compute(A, value);
  }
};

// Core inner loop for setting sparse matrix value
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K,
          std::size_t K_End>
struct SetSparseMatrixValueInnerLoop {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    SetSparseMatrixValueCoreConditional<
        ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A, J, K,
        (RowToSet - RowIndices_A::list[K])>::compute(A, value);

    SetSparseMatrixValueInnerLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A,
                                  RowPointers_A, J, (K + 1),
                                  (K_End - 1)>::compute(A, value);
  }
};

// End of inner loop
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K>
struct SetSparseMatrixValueInnerLoop<ColumnToSet, RowToSet, T, M, N,
                                     RowIndices_A, RowPointers_A, J, K, 0> {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of inner loop, do nothing
  }
};

// Conditional operation for ColumnSet != J
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t C_J, std::size_t J,
          std::size_t J_End>
struct SetSparseMatrixValueOuterConditional {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of conditional operation, do nothing
  }
};

// Conditional operation for ColumnSet == J
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t J_End>
struct SetSparseMatrixValueOuterConditional<
    ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A, 0, J, J_End> {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    SetSparseMatrixValueInnerLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A,
                                  RowPointers_A, J, RowPointers_A::list[J],
                                  (RowPointers_A::list[J + 1] -
                                   RowPointers_A::list[J])>::compute(A, value);
  }
};

// Core outer loop for setting sparse matrix value
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t J_End>
struct SetSparseMatrixValueOuterLoop {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    SetSparseMatrixValueOuterConditional<
        ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A,
        (ColumnToSet - J), J, J_End>::compute(A, value);

    SetSparseMatrixValueOuterLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A,
                                  RowPointers_A, (J + 1),
                                  (J_End - 1)>::compute(A, value);
  }
};

// End of outer loop
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct SetSparseMatrixValueOuterLoop<ColumnToSet, RowToSet, T, M, N,
                                     RowIndices_A, RowPointers_A, J, 0> {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of outer loop, do nothing
  }
};

template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
static inline void COMPILED_SPARSE_SET_MATRIX_VALUE(
    CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const T &value) {

  SetSparseMatrixValueOuterLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A,
                                RowPointers_A, 0, M>::compute(A, value);
}

template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void set_sparse_matrix_value(
    CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const T &value) {
  static_assert(ColumnToSet < M, "Column number must be less than M");
  static_assert(RowToSet < N, "Row number must be less than N");

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    if (ColumnToSet == j) {

      for (std::size_t k = RowPointers_A::list[j];
           k < RowPointers_A::list[j + 1]; ++k) {
        if (RowToSet == RowIndices_A::list[k]) {

          A.values[k] = value;
        }
      }
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_SPARSE_SET_MATRIX_VALUE<ColumnToSet, RowToSet, T, M, N,
                                                 RowIndices_A, RowPointers_A>(
      A, value);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

/* Set Sparse Matrix each element values */
template <std::size_t ElementToSet, typename T, std::size_t M, std::size_t N,
          typename RowIndices_A, typename RowPointers_A>
inline void set_sparse_matrix_element_value(
    CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const T &value) {

  static_assert(ElementToSet < RowPointers_A::list[M],
                "Element number must be less than RowPointers::list[M]");

  A.values[ElementToSet] = value;
}

/* Get Sparse Matrix Value */
// check if RowToGet == RowIndices_A::list[K]
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, std::size_t RowToGet_I>
struct GetSparseMatrixValueCoreIf {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {
    /* Do nothing */
    static_cast<void>(A);
    static_cast<void>(value);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
struct GetSparseMatrixValueCoreIf<T, M, N, RowIndices_A, RowPointers_A, K, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    value = A.values[K];
  }
};

// Core conditional operation for setting sparse matrix value
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K, std::size_t L>
struct GetSparseMatrixValueCoreConditional {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {
    /* Do nothing */
    static_cast<void>(A);
    static_cast<void>(value);
  }
};

template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K>
struct GetSparseMatrixValueCoreConditional<
    ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A, J, K, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    GetSparseMatrixValueCoreIf<T, M, N, RowIndices_A, RowPointers_A, K,
                               (RowToGet -
                                RowIndices_A::list[K])>::compute(A, value);
  }
};

// Core inner loop for setting sparse matrix value
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K,
          std::size_t K_End>
struct GetSparseMatrixValueInnerLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    GetSparseMatrixValueCoreConditional<
        ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A, J, K,
        (RowToGet - RowIndices_A::list[K])>::compute(A, value);

    GetSparseMatrixValueInnerLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A,
                                  RowPointers_A, J, (K + 1),
                                  (K_End - 1)>::compute(A, value);
  }
};

// End of inner loop
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K>
struct GetSparseMatrixValueInnerLoop<ColumnToGet, RowToGet, T, M, N,
                                     RowIndices_A, RowPointers_A, J, K, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of inner loop, do nothing
  }
};

// Conditional operation for ColumnGet != J
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t C_J, std::size_t J,
          std::size_t J_End>
struct GetSparseMatrixValueOuterConditional {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of conditional operation, do nothing
  }
};

// Conditional operation for ColumnGet == J
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t J_End>
struct GetSparseMatrixValueOuterConditional<
    ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A, 0, J, J_End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    GetSparseMatrixValueInnerLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A,
                                  RowPointers_A, J, RowPointers_A::list[J],
                                  (RowPointers_A::list[J + 1] -
                                   RowPointers_A::list[J])>::compute(A, value);
  }
};

// Core outer loop for setting sparse matrix value
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t J_End>
struct GetSparseMatrixValueOuterLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    GetSparseMatrixValueOuterConditional<
        ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A,
        (ColumnToGet - J), J, J_End>::compute(A, value);

    GetSparseMatrixValueOuterLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A,
                                  RowPointers_A, (J + 1),
                                  (J_End - 1)>::compute(A, value);
  }
};

// End of outer loop
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct GetSparseMatrixValueOuterLoop<ColumnToGet, RowToGet, T, M, N,
                                     RowIndices_A, RowPointers_A, J, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of outer loop, do nothing
  }
};

template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
static inline void COMPILED_SPARSE_GET_MATRIX_VALUE(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    T &value) {
  GetSparseMatrixValueOuterLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A,
                                RowPointers_A, 0, M>::compute(A, value);
}

template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline T get_sparse_matrix_value(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
  static_assert(ColumnToGet < M, "Column number must be less than M");
  static_assert(RowToGet < N, "Row number must be less than N");

  T value = static_cast<T>(0);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    if (ColumnToGet == j) {

      for (std::size_t k = RowPointers_A::list[j];
           k < RowPointers_A::list[j + 1]; ++k) {
        if (RowToGet == RowIndices_A::list[k]) {

          value = A.values[k];
        }
      }
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_SPARSE_GET_MATRIX_VALUE<ColumnToGet, RowToGet, T, M, N,
                                                 RowIndices_A, RowPointers_A>(
      A, value);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return value;
}

/* Get Sparse Matrix each element values */
template <std::size_t ElementToGet, typename T, std::size_t M, std::size_t N,
          typename RowIndices_A, typename RowPointers_A>
inline T get_sparse_matrix_element_value(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
  static_assert(ElementToGet < RowPointers_A::list[M],
                "Element number must be less than RowPointers::list[M]");

  return A.values[ElementToGet];
}

/* Output transpose matrix */
// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, typename Result_Type, std::size_t J,
          std::size_t K, std::size_t Start, std::size_t End>
struct OutputTransposeMatrixLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Result_Type &result) {

    set_sparse_matrix_value<RowIndices::list[Start], J>(result,
                                                        mat.values[Start]);
    OutputTransposeMatrixLoop<T, M, N, RowIndices, RowPointers, Result_Type, J,
                              K, Start + 1, End>::compute(mat, result);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, typename Result_Type, std::size_t J,
          std::size_t K, std::size_t End>
struct OutputTransposeMatrixLoop<T, M, N, RowIndices, RowPointers, Result_Type,
                                 J, K, End, End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Result_Type &result) {
    static_cast<void>(mat);
    static_cast<void>(result);
    // End of loop, do nothing
  }
};

// Pointer loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, typename Result_Type, std::size_t J,
          std::size_t K>
struct OutputTransposeMatrixCore {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Result_Type &result) {

    OutputTransposeMatrixLoop<T, M, N, RowIndices, RowPointers, Result_Type, J,
                              K, RowPointers::list[J],
                              RowPointers::list[J + 1]>::compute(mat, result);
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, typename Result_Type, std::size_t J>
struct OutputTransposeMatrixRow {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Result_Type &result) {

    OutputTransposeMatrixCore<T, M, N, RowIndices, RowPointers, Result_Type, J,
                              0>::compute(mat, result);
    OutputTransposeMatrixRow<T, M, N, RowIndices, RowPointers, Result_Type,
                             J - 1>::compute(mat, result);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, typename Result_Type>
struct OutputTransposeMatrixRow<T, M, N, RowIndices, RowPointers, Result_Type,
                                0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Result_Type &result) {

    OutputTransposeMatrixCore<T, M, N, RowIndices, RowPointers, Result_Type, 0,
                              0>::compute(mat, result);
  }
};

namespace CompiledSparseOperation {

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct Transpose {

  using SparseAvailable_In =
      CreateSparseAvailableFromIndicesAndPointers<N, RowIndices, RowPointers>;

  using SparseAvailable_Out = SparseAvailableTranspose<SparseAvailable_In>;

  using RowIndices_Out = RowIndicesFromSparseAvailable<SparseAvailable_Out>;

  using RowPointers_Out = RowPointersFromSparseAvailable<SparseAvailable_Out>;

  using Result_Type =
      CompiledSparseMatrix<T, N, M, RowIndices_Out, RowPointers_Out>;
};

} // namespace CompiledSparseOperation

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline auto output_matrix_transpose(
    const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat) ->
    typename CompiledSparseOperation::Transpose<T, M, N, RowIndices,
                                                RowPointers>::Result_Type {

  using Result_Type =
      typename CompiledSparseOperation::Transpose<T, M, N, RowIndices,
                                                  RowPointers>::Result_Type;

  Result_Type result;

  OutputTransposeMatrixRow<T, M, N, RowIndices, RowPointers, Result_Type,
                           M - 1>::compute(mat, result);

  return result;
}

/* Convert Real Matrix to Complex Matrix */
/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t I>
struct SparseMatrixRealToComplexLoop {
  static void compute(
      const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &From_matrix,
      CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
          &To_matrix) {
    To_matrix.values[I - 1].real = From_matrix.values[I - 1];
    SparseMatrixRealToComplexLoop<T, M, N, RowIndices, RowPointers,
                                  I - 1>::compute(From_matrix, To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct SparseMatrixRealToComplexLoop<T, M, N, RowIndices, RowPointers, 0> {
  static void compute(
      const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &From_matrix,
      CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
          &To_matrix) {
    /* Do Nothing. */
    static_cast<void>(From_matrix);
    static_cast<void>(To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
convert_matrix_real_to_complex(
    const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &From_matrix) {

  CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < RowPointers::list[M]; ++i) {
    To_matrix.values[i].real = From_matrix.values[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixRealToComplexLoop<T, M, N, RowIndices, RowPointers,
                                RowPointers::list[M]>::compute(From_matrix,
                                                               To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Get Real Matrix from Complex Matrix */
/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t I>
struct SparseMatrixRealFromComplexLoop {
  static void
  compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
              &From_matrix,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

    To_matrix.values[I - 1] = From_matrix.values[I - 1].real;
    SparseMatrixRealFromComplexLoop<T, M, N, RowIndices, RowPointers,
                                    I - 1>::compute(From_matrix, To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct SparseMatrixRealFromComplexLoop<T, M, N, RowIndices, RowPointers, 0> {
  static void
  compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
              &From_matrix,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

    /* Do Nothing. */
    static_cast<void>(From_matrix);
    static_cast<void>(To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline CompiledSparseMatrix<T, M, N, RowIndices, RowPointers>
get_real_matrix_from_complex_matrix(
    const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
        &From_matrix) {

  CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < RowPointers::list[M]; ++i) {
    To_matrix.values[i] = From_matrix.values[i].real;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixRealFromComplexLoop<T, M, N, RowIndices, RowPointers,
                                  RowPointers::list[M]>::compute(From_matrix,
                                                                 To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Get Imag Matrix from Complex Matrix */
/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t I>
struct SparseMatrixImagFromComplexLoop {
  static void
  compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
              &From_matrix,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

    To_matrix.values[I - 1] = From_matrix.values[I - 1].imag;
    SparseMatrixImagFromComplexLoop<T, M, N, RowIndices, RowPointers,
                                    I - 1>::compute(From_matrix, To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct SparseMatrixImagFromComplexLoop<T, M, N, RowIndices, RowPointers, 0> {
  static void
  compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
              &From_matrix,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

    /* Do Nothing. */
    static_cast<void>(From_matrix);
    static_cast<void>(To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline CompiledSparseMatrix<T, M, N, RowIndices, RowPointers>
get_imag_matrix_from_complex_matrix(
    const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
        &From_matrix) {

  CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < RowPointers::list[M]; ++i) {
    To_matrix.values[i] = From_matrix.values[i].imag;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SparseMatrixImagFromComplexLoop<T, M, N, RowIndices, RowPointers,
                                  RowPointers::list[M]>::compute(From_matrix,
                                                                 To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_COMPILED_SPARSE_HPP__
