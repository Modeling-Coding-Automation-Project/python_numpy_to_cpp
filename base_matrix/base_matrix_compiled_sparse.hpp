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

  /* Function */
  T &operator[](std::size_t index) { return this->values[index]; }

  const T &operator[](std::size_t index) const { return this->values[index]; }

  static inline CompiledSparseMatrix<T, M, N, RowIndices, RowPointers>
  full(const T &value) {
    CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> full(
        std::vector<T>(RowPointers::list[M], value));

    return full;
  }

  /* Variable */
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<T> values;
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<T, RowPointers::list[M]> values;
#endif // __BASE_MATRIX_USE_STD_VECTOR__
};

/* Output dense matrix */
namespace OutputDenseMatrix {

// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K, std::size_t Start,
          std::size_t End>
struct Loop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {

    result(J, RowIndices::list[Start]) = mat.values[Start];
    Loop<T, M, N, RowIndices, RowPointers, J, K, Start + 1, End>::compute(
        mat, result);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K, std::size_t End>
struct Loop<T, M, N, RowIndices, RowPointers, J, K, End, End> {
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
struct Core {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {

    Loop<T, M, N, RowIndices, RowPointers, J, K, RowPointers::list[J],
         RowPointers::list[J + 1]>::compute(mat, result);
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J>
struct Row {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {

    Core<T, M, N, RowIndices, RowPointers, J, 0>::compute(mat, result);
    Row<T, M, N, RowIndices, RowPointers, J - 1>::compute(mat, result);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct Row<T, M, N, RowIndices, RowPointers, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {

    Core<T, M, N, RowIndices, RowPointers, 0, 0>::compute(mat, result);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
        Matrix<T, M, N> &result) {

  Row<T, M, N, RowIndices, RowPointers, M - 1>::compute(mat, result);
}

} // namespace OutputDenseMatrix

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

  OutputDenseMatrix::compute<T, M, N, RowIndices, RowPointers>(mat, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Substitute Dense Matrix to Sparse Matrix */
namespace SubstituteDenseMatrixToSparseMatrix {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I, std::size_t J_idx>
struct Column {
  static void
  compute(const Matrix<T, M, N> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

    Y.values[I * N + J_idx] = A.template get<I, J_idx>();
    Column<T, M, N, RowIndices_A, RowPointers_A, I, J_idx - 1>::compute(A, Y);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I>
struct Column<T, M, N, RowIndices_A, RowPointers_A, I, 0> {
  static void
  compute(const Matrix<T, M, N> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

    Y.values[I * N] = A.template get<I, 0>();
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I_idx>
struct Row {
  static void
  compute(const Matrix<T, M, N> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

    Column<T, M, N, RowIndices_A, RowPointers_A, I_idx, N - 1>::compute(A, Y);
    Row<T, M, N, RowIndices_A, RowPointers_A, I_idx - 1>::compute(A, Y);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct Row<T, M, N, RowIndices_A, RowPointers_A, 0> {
  static void
  compute(const Matrix<T, M, N> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {
    Column<T, M, N, RowIndices_A, RowPointers_A, 0, N - 1>::compute(A, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
static inline void
compute(const Matrix<T, M, N> &A,
        CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {
  Row<T, M, N, RowIndices_A, RowPointers_A, M - 1>::compute(A, Y);
}

} // namespace SubstituteDenseMatrixToSparseMatrix

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

  SubstituteDenseMatrixToSparseMatrix::compute<
      T, M, N, DenseMatrixRowIndices<M, N>, DenseMatrixRowPointers<M, N>>(A, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Create Sparse Matrix from Diag Matrix */
template <std::size_t M>
using DiagMatrixRowIndices = typename TemplatesOperation::ToRowIndices<
    TemplatesOperation::MatrixRowNumbers<M>>::type;

template <std::size_t M>
using DiagMatrixRowPointers = typename TemplatesOperation::ToRowIndices<
    TemplatesOperation::MatrixRowNumbers<(M + 1)>>::type;

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
namespace SetSparseMatrixValue {

// check if RowToSet == RowIndices_A::list[K]
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, std::size_t RowToGet_I>
struct CoreIf {
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
struct CoreIf<T, M, N, RowIndices_A, RowPointers_A, K, 0> {
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
struct CoreConditional {
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
struct CoreConditional<ColumnToSet, RowToSet, T, M, N, RowIndices_A,
                       RowPointers_A, J, K, 0> {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    CoreIf<T, M, N, RowIndices_A, RowPointers_A, K,
           (RowToSet - RowIndices_A::list[K])>::compute(A, value);
  }
};

// Core inner loop for setting sparse matrix value
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K,
          std::size_t K_End>
struct InnerLoop {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    CoreConditional<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A,
                    J, K, (RowToSet - RowIndices_A::list[K])>::compute(A,
                                                                       value);

    InnerLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A, J,
              (K + 1), (K_End - 1)>::compute(A, value);
  }
};

// End of inner loop
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K>
struct InnerLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A, J,
                 K, 0> {
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
struct OuterConditional {
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
struct OuterConditional<ColumnToSet, RowToSet, T, M, N, RowIndices_A,
                        RowPointers_A, 0, J, J_End> {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    InnerLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A, J,
              RowPointers_A::list[J],
              (RowPointers_A::list[J + 1] -
               RowPointers_A::list[J])>::compute(A, value);
  }
};

// Core outer loop for setting sparse matrix value
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t J_End>
struct OuterLoop {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    OuterConditional<ColumnToSet, RowToSet, T, M, N, RowIndices_A,
                     RowPointers_A, (ColumnToSet - J), J,
                     J_End>::compute(A, value);

    OuterLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A,
              (J + 1), (J_End - 1)>::compute(A, value);
  }
};

// End of outer loop
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct OuterLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A, J,
                 0> {
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
inline void
compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        const T &value) {

  OuterLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A, 0,
            M>::compute(A, value);
}

} // namespace SetSparseMatrixValue

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

  SetSparseMatrixValue::compute<ColumnToSet, RowToSet, T, M, N, RowIndices_A,
                                RowPointers_A>(A, value);

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
namespace GetSparseMatrixValue {

// check if RowToGet == RowIndices_A::list[K]
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, std::size_t RowToGet_I>
struct CoreIf {
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
struct CoreIf<T, M, N, RowIndices_A, RowPointers_A, K, 0> {
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
struct CoreConditional {
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
struct CoreConditional<ColumnToGet, RowToGet, T, M, N, RowIndices_A,
                       RowPointers_A, J, K, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    CoreIf<T, M, N, RowIndices_A, RowPointers_A, K,
           (RowToGet - RowIndices_A::list[K])>::compute(A, value);
  }
};

// Core inner loop for setting sparse matrix value
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K,
          std::size_t K_End>
struct InnerLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    CoreConditional<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A,
                    J, K, (RowToGet - RowIndices_A::list[K])>::compute(A,
                                                                       value);

    InnerLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A, J,
              (K + 1), (K_End - 1)>::compute(A, value);
  }
};

// End of inner loop
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K>
struct InnerLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A, J,
                 K, 0> {
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
struct OuterConditional {
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
struct OuterConditional<ColumnToGet, RowToGet, T, M, N, RowIndices_A,
                        RowPointers_A, 0, J, J_End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    InnerLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A, J,
              RowPointers_A::list[J],
              (RowPointers_A::list[J + 1] -
               RowPointers_A::list[J])>::compute(A, value);
  }
};

// Core outer loop for setting sparse matrix value
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t J_End>
struct OuterLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    OuterConditional<ColumnToGet, RowToGet, T, M, N, RowIndices_A,
                     RowPointers_A, (ColumnToGet - J), J,
                     J_End>::compute(A, value);

    OuterLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A,
              (J + 1), (J_End - 1)>::compute(A, value);
  }
};

// End of outer loop
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct OuterLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A, J,
                 0> {
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
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        T &value) {
  OuterLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A, 0,
            M>::compute(A, value);
}

} // namespace GetSparseMatrixValue

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

  GetSparseMatrixValue::compute<ColumnToGet, RowToGet, T, M, N, RowIndices_A,
                                RowPointers_A>(A, value);

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
namespace OutputTransposeMatrix {

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

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, typename Result_Type>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
        Result_Type &result) {

  OutputTransposeMatrixRow<T, M, N, RowIndices, RowPointers, Result_Type,
                           M - 1>::compute(mat, result);
}

} // namespace OutputTransposeMatrix

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

  OutputTransposeMatrix::compute<T, M, N, RowIndices, RowPointers, Result_Type>(
      mat, result);

  return result;
}

/* Convert Real Matrix to Complex Matrix */
namespace ConvertRealSparseMatrixToComplex {

/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t I>
struct Loop {
  static void compute(
      const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &From_matrix,
      CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
          &To_matrix) {

    To_matrix.values[I - 1].real = From_matrix.values[I - 1];
    Loop<T, M, N, RowIndices, RowPointers, I - 1>::compute(From_matrix,
                                                           To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct Loop<T, M, N, RowIndices, RowPointers, 0> {
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
inline void compute(
    const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &From_matrix,
    CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
        &To_matrix) {

  Loop<T, M, N, RowIndices, RowPointers, RowPointers::list[M]>::compute(
      From_matrix, To_matrix);
}

} // namespace ConvertRealSparseMatrixToComplex

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

  ConvertRealSparseMatrixToComplex::compute<T, M, N, RowIndices, RowPointers>(
      From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Get Real Matrix from Complex Matrix */
namespace GetRealSparseMatrixFromComplex {

/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t I>
struct Loop {
  static void
  compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
              &From_matrix,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

    To_matrix.values[I - 1] = From_matrix.values[I - 1].real;
    Loop<T, M, N, RowIndices, RowPointers, I - 1>::compute(From_matrix,
                                                           To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct Loop<T, M, N, RowIndices, RowPointers, 0> {
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
inline void
compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
            &From_matrix,
        CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

  Loop<T, M, N, RowIndices, RowPointers, RowPointers::list[M]>::compute(
      From_matrix, To_matrix);
}

} // namespace GetRealSparseMatrixFromComplex

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

  GetRealSparseMatrixFromComplex::compute<T, M, N, RowIndices, RowPointers>(
      From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Get Imag Matrix from Complex Matrix */
namespace GetImagSparseMatrixFromComplex {

/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t I>
struct Loop {
  static void
  compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
              &From_matrix,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

    To_matrix.values[I - 1] = From_matrix.values[I - 1].imag;
    Loop<T, M, N, RowIndices, RowPointers, I - 1>::compute(From_matrix,
                                                           To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct Loop<T, M, N, RowIndices, RowPointers, 0> {
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
inline void
compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
            &From_matrix,
        CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

  Loop<T, M, N, RowIndices, RowPointers, RowPointers::list[M]>::compute(
      From_matrix, To_matrix);
}

} // namespace GetImagSparseMatrixFromComplex

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

  GetImagSparseMatrixFromComplex::compute<T, M, N, RowIndices, RowPointers>(
      From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Diagonal Inverse Multiply Sparse */
namespace DiagonalInverseMultiplySparse {

// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t K,
          std::size_t Start, std::size_t End>
struct Loop {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          const T &division_min, Matrix<T, M, N> &result) {

    result.values[RowIndices_B::list[Start]] =
        B.values[RowIndices_B::list[Start]] /
        Base::Utility::avoid_zero_divide(A[J], division_min);

    Loop<T, M, N, RowIndices_B, RowPointers_B, J, K, Start + 1, End>::compute(
        A, B, division_min, result);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t K, std::size_t End>
struct Loop<T, M, N, RowIndices_B, RowPointers_B, J, K, End, End> {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          const T &division_min, Matrix<T, M, N> &result) {

    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(division_min);
    static_cast<void>(result);
    // End of loop, do nothing
  }
};

// Pointer loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t K>
struct Core {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          const T &division_min, Matrix<T, M, N> &result) {

    Loop<T, M, N, RowIndices_B, RowPointers_B, J, K, RowPointers_B::list[J],
         RowPointers_B::list[J + 1]>::compute(A, B, division_min, result);
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t J>
struct Row {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          const T &division_min, Matrix<T, M, N> &result) {

    Core<T, M, N, RowIndices_B, RowPointers_B, J, 0>::compute(
        A, B, division_min, result);
    Row<T, M, N, RowIndices_B, RowPointers_B, J - 1>::compute(
        A, B, division_min, result);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
struct Row<T, M, N, RowIndices_B, RowPointers_B, 0> {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          const T &division_min, Matrix<T, M, N> &result) {

    Core<T, M, N, RowIndices_B, RowPointers_B, 0, 0>::compute(
        A, B, division_min, result);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
inline void
compute(const DiagMatrix<T, M> &A,
        const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
        const T &division_min, Matrix<T, M, N> &result) {

  Row<T, M, N, RowIndices_B, RowPointers_B, M - 1>::compute(A, B, division_min,
                                                            result);
}

} // namespace DiagonalInverseMultiplySparse

template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
inline CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B>
diag_inv_multiply_sparse(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
    const T &division_min) {

  CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> result = B;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_B::list[j]; k < RowPointers_B::list[j + 1];
         ++k) {

      result.values[k] =
          B.values[k] / Base::Utility::avoid_zero_divide(A[j], division_min);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagonalInverseMultiplySparse::compute<T, M, N, RowIndices_B, RowPointers_B>(
      A, B, division_min, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
inline CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B>
diag_inv_multiply_sparse_partition(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
    const T &division_min, const std::size_t &matrix_size) {

  CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> result = B;

  for (std::size_t j = 0; j < matrix_size; ++j) {
    for (std::size_t k = RowPointers_B::list[j]; k < RowPointers_B::list[j + 1];
         ++k) {

      result.values[k] =
          B.values[k] / Base::Utility::avoid_zero_divide(A[j], division_min);
    }
  }

  return result;
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_COMPILED_SPARSE_HPP__
