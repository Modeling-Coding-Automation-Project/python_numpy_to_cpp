#ifndef BASE_MATRIX_COMPILED_SPARSE_HPP
#define BASE_MATRIX_COMPILED_SPARSE_HPP

#include "base_matrix_diagonal.hpp"
#include "base_matrix_macros.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

const double COMPILED_SPARSE_MATRIX_JUDGE_ZERO_LIMIT_VALUE = 1.0e-20;

template <std::size_t... Sizes> struct list_array {
  static constexpr std::size_t size = sizeof...(Sizes);
  static constexpr std::size_t value[size] = {Sizes...};
};

template <std::size_t... Sizes>
constexpr std::size_t list_array<Sizes...>::value[list_array<Sizes...>::size];

template <typename Array> class CompiledSparseMatrixList {
public:
  typedef const std::size_t *list_type;
  static constexpr list_type list = Array::value;
  static constexpr std::size_t size = Array::size;
};

template <std::size_t... Sizes>
using RowIndices = CompiledSparseMatrixList<list_array<Sizes...>>;

template <std::size_t... Sizes>
using RowPointers = CompiledSparseMatrixList<list_array<Sizes...>>;

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
class CompiledSparseMatrix {
public:
#ifdef BASE_MATRIX_USE_STD_VECTOR

  CompiledSparseMatrix() : values(RowIndices::size, static_cast<T>(0)) {}

  CompiledSparseMatrix(const std::initializer_list<T> &values)
      : values(values) {}

  CompiledSparseMatrix(const std::vector<T> &values) : values(values) {}

#else

  CompiledSparseMatrix() : values{} {}

  CompiledSparseMatrix(const std::initializer_list<T> &values) : values{} {

    std::copy(values.begin(), values.end(), this->values.begin());
  }

  CompiledSparseMatrix(const std::array<T, RowIndices::size> &values)
      : values(values) {}

  CompiledSparseMatrix(const std::vector<T> &values) : values{} {

    std::copy(values.begin(), values.end(), this->values.begin());
  }

#endif

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
  Matrix<T, M, N> create_dense() const { return output_dense_matrix(*this); }

  Matrix<T, N, M> transpose() const { return output_transpose_matrix(*this); }

  /* Variable */
#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values;
#else
  std::array<T, RowIndices::size> values;
#endif
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

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; j++) {
    for (std::size_t k = RowPointers::list[j]; k < RowPointers::list[j + 1];
         k++) {
      result(j, RowIndices::list[k]) = mat.values[k];
    }
  }

#else

  COMPILED_SPARSE_OUTPUT_DENSE_MATRIX<T, M, N, RowIndices, RowPointers>(mat,
                                                                        result);

#endif

  return result;
}

/* Output transpose matrix */
// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K, std::size_t Start,
          std::size_t End>
struct OutputTransposeMatrixLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, N, M> &result) {
    result(RowIndices::list[Start], J) = mat.values[Start];
    OutputTransposeMatrixLoop<T, M, N, RowIndices, RowPointers, J, K, Start + 1,
                              End>::compute(mat, result);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K, std::size_t End>
struct OutputTransposeMatrixLoop<T, M, N, RowIndices, RowPointers, J, K, End,
                                 End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, N, M> &result) {
    static_cast<void>(mat);
    static_cast<void>(result);
    // End of loop, do nothing
  }
};

// Pointer loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K>
struct OutputTransposeMatrixCore {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, N, M> &result) {
    OutputTransposeMatrixLoop<T, M, N, RowIndices, RowPointers, J, K,
                              RowPointers::list[J],
                              RowPointers::list[J + 1]>::compute(mat, result);
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J>
struct OutputTransposeMatrixRow {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, N, M> &result) {
    OutputTransposeMatrixCore<T, M, N, RowIndices, RowPointers, J, 0>::compute(
        mat, result);
    OutputTransposeMatrixRow<T, M, N, RowIndices, RowPointers, J - 1>::compute(
        mat, result);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct OutputTransposeMatrixRow<T, M, N, RowIndices, RowPointers, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, N, M> &result) {
    OutputTransposeMatrixCore<T, M, N, RowIndices, RowPointers, 0, 0>::compute(
        mat, result);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
static inline void COMPILED_SPARSE_TRANSPOSE_DENSE_MATRIX(
    const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
    Matrix<T, N, M> &result) {
  OutputTransposeMatrixRow<T, M, N, RowIndices, RowPointers, M - 1>::compute(
      mat, result);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline Matrix<T, N, M> output_transpose_matrix(
    const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat) {
  Matrix<T, N, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; j++) {
    for (std::size_t k = RowPointers::list[j]; k < RowPointers::list[j + 1];
         k++) {
      result(RowIndices::list[k], j) = mat.values[k];
    }
  }

#else

  COMPILED_SPARSE_TRANSPOSE_DENSE_MATRIX<T, M, N, RowIndices, RowPointers>(
      mat, result);

#endif

  return result;
}

/* Create Sparse Matrix from Dense Matrix */
template <std::size_t... Seq> struct IndexSequence {
  static constexpr std::size_t size = sizeof...(Seq);
  static constexpr std::size_t list[size] = {Seq...};
};

template <std::size_t N, std::size_t... Seq>
struct MakeIndexSequence : MakeIndexSequence<N - 1, N - 1, Seq...> {};

template <std::size_t... Seq> struct MakeIndexSequence<0, Seq...> {
  using type = IndexSequence<Seq...>;
};

template <std::size_t N> struct LogicList {
  using type = typename MakeIndexSequence<N>::type;
};

template <std::size_t N> using MatrixRowNumbers = typename LogicList<N>::type;

template <typename Seq1, typename Seq2> struct Concatenate;

template <std::size_t... Seq1, std::size_t... Seq2>
struct Concatenate<IndexSequence<Seq1...>, IndexSequence<Seq2...>> {
  using type = IndexSequence<Seq1..., Seq2...>;
};

template <typename IndexSequence_1, typename IndexSequence_2>
using ConcatenateMatrixRowNumbers =
    typename Concatenate<IndexSequence_1, IndexSequence_2>::type;

template <std::size_t M, typename MatrixRowNumbers>
struct RepeatConcatenateIndexSequence;

template <typename MatrixRowNumbers>
struct RepeatConcatenateIndexSequence<1, MatrixRowNumbers> {
  using type = MatrixRowNumbers;
};

template <std::size_t M, typename MatrixRowNumbers>
struct RepeatConcatenateIndexSequence {
  using type = ConcatenateMatrixRowNumbers<
      MatrixRowNumbers,
      typename RepeatConcatenateIndexSequence<M - 1, MatrixRowNumbers>::type>;
};

template <typename Seq> struct ToRowIndices;

template <std::size_t... Seq> struct ToRowIndices<IndexSequence<Seq...>> {
  using type = RowIndices<Seq...>;
};

template <std::size_t M, std::size_t N>
using MatrixColumnRowNumbers =
    typename RepeatConcatenateIndexSequence<M, MatrixRowNumbers<N>>::type;

template <std::size_t M, std::size_t N>
using DenseMatrixRowIndices =
    typename ToRowIndices<MatrixColumnRowNumbers<M, N>>::type;

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_COMPILED_SPARSE_HPP
