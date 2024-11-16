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

template <std::size_t... Seq> struct InvalidSequence {
  static constexpr std::size_t size = sizeof...(Seq);
  static constexpr std::size_t list[size] = {Seq...};
};

template <std::size_t N, std::size_t... Seq>
struct MakeIndexSequence : MakeIndexSequence<N - 1, N - 1, Seq...> {};

template <std::size_t... Seq> struct MakeIndexSequence<0, Seq...> {
  using type = IndexSequence<Seq...>;
};

template <std::size_t N> struct IntegerSequenceList {
  using type = typename MakeIndexSequence<N>::type;
};

template <std::size_t N>
using MatrixRowNumbers = typename IntegerSequenceList<N>::type;

template <typename Seq1, typename Seq2> struct Concatenate;

template <std::size_t... Seq1, std::size_t... Seq2>
struct Concatenate<IndexSequence<Seq1...>, IndexSequence<Seq2...>> {
  using type = IndexSequence<Seq1..., Seq2...>;
};

template <std::size_t... Seq1, std::size_t... Seq2>
struct Concatenate<IndexSequence<Seq1...>, InvalidSequence<Seq2...>> {
  using type = IndexSequence<Seq1...>;
};

template <std::size_t... Seq1, std::size_t... Seq2>
struct Concatenate<InvalidSequence<Seq1...>, IndexSequence<Seq2...>> {
  using type = IndexSequence<Seq2...>;
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

template <std::size_t M, std::size_t N, std::size_t... Seq>
struct MakePointerList : MakePointerList<M - 1, N, (M * N), Seq...> {};

template <std::size_t N, std::size_t... Seq>
struct MakePointerList<0, N, Seq...> {
  using type = IndexSequence<0, Seq...>;
};

template <std::size_t M, std::size_t N> struct MatrixDensePointerList {
  using type = typename MakePointerList<M, N>::type;
};

template <std::size_t M, std::size_t N>
using MatrixColumnRowPointers = typename MatrixDensePointerList<M, N>::type;

template <typename Seq> struct ToRowPointers;

template <std::size_t... Seq> struct ToRowPointers<IndexSequence<Seq...>> {
  using type = RowPointers<Seq...>;
};

template <std::size_t M, std::size_t N>
using DenseMatrixRowPointers =
    typename ToRowIndices<MatrixColumnRowPointers<M, N>>::type;

/* Substitute Dense Matrix to Sparse Matrix */
// when J_idx < N
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I, std::size_t J_idx>
struct DenseToSparseMatrixSubstituteColumn {
  static void
  compute(const Matrix<T, M, N> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {
    Y.values[I * N + J_idx] = A(I, J_idx);
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
    Y.values[I * N] = A(I, 0);
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
auto create_compiled_sparse(const Matrix<T, M, N> &A)
    -> CompiledSparseMatrix<T, M, N, DenseMatrixRowIndices<M, N>,
                            DenseMatrixRowPointers<M, N>> {
  CompiledSparseMatrix<T, M, N, DenseMatrixRowIndices<M, N>,
                       DenseMatrixRowPointers<M, N>>
      Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  std::size_t consecutive_index = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < N; j++) {
      Y.values[consecutive_index] = A(i, j);
      consecutive_index++;
    }
  }

#else

  COMPILED_DENSE_MATRIX_SUBSTITUTE_SPARSE<T, M, N, DenseMatrixRowIndices<M, N>,
                                          DenseMatrixRowPointers<M, N>>(A, Y);

#endif

  return Y;
}

/* Create Sparse Matrix from Diag Matrix */
template <std::size_t M>
using DiagMatrixRowIndices = typename ToRowIndices<MatrixRowNumbers<M>>::type;

template <std::size_t M>
using DiagMatrixRowPointers =
    typename ToRowIndices<MatrixRowNumbers<(M + 1)>>::type;

template <typename T, std::size_t M>
auto create_compiled_sparse(const DiagMatrix<T, M> &A)
    -> CompiledSparseMatrix<T, M, M, DiagMatrixRowIndices<M>,
                            DiagMatrixRowPointers<M>> {
  CompiledSparseMatrix<T, M, M, DiagMatrixRowIndices<M>,
                       DiagMatrixRowPointers<M>>
      Y;

  Y.values = A.data;

  return Y;
}

/* Create Sparse Matrix from Matrix Element List */
template <bool... Flags> struct available_list_array {
  static constexpr std::size_t size = sizeof...(Flags);
  static constexpr bool value[size] = {Flags...};
};

template <bool... Flags>
constexpr bool
    available_list_array<Flags...>::value[available_list_array<Flags...>::size];

template <typename Array> class CompiledSparseMatrixElementList {
public:
  typedef const bool *list_type;
  static constexpr list_type list = Array::value;
  static constexpr std::size_t size = Array::size;
};

template <bool... Flags>
using ColumnAvailable =
    CompiledSparseMatrixElementList<available_list_array<Flags...>>;

template <typename... Columns> struct SparseAvailableColumns {
  static constexpr std::size_t number_of_columns = sizeof...(Columns);

  // Helper struct to extract the list from each ColumnAvailable
  template <typename Column> struct ExtractList {
    static constexpr typename Column::list_type value = Column::list;
  };

  // Array of lists from each ColumnAvailable
  static constexpr const bool *lists[number_of_columns] = {
      ExtractList<Columns>::value...};

  // Function to get the size of a specific column
  static constexpr std::size_t column_size =
      std::tuple_element<0, std::tuple<Columns...>>::type::size;
};

template <typename... Columns>
using SparseAvailable = SparseAvailableColumns<Columns...>;

/* Create Row Indices */
template <typename SparseAvailable, std::size_t ColumnElementNumber,
          bool Active, std::size_t RowElementNumber>
struct AssignSparseMatrixRowLoop;

template <typename SparseAvailable, std::size_t ColumnElementNumber,
          std::size_t RowElementNumber>
struct AssignSparseMatrixRowLoop<SparseAvailable, ColumnElementNumber, true,
                                 RowElementNumber> {
  using type = typename Concatenate<
      typename AssignSparseMatrixRowLoop<
          SparseAvailable, ColumnElementNumber,
          SparseAvailable::lists[ColumnElementNumber][RowElementNumber - 1],
          RowElementNumber - 1>::type,
      IndexSequence<RowElementNumber>>::type;
};

template <typename SparseAvailable, std::size_t ColumnElementNumber,
          std::size_t RowElementNumber>
struct AssignSparseMatrixRowLoop<SparseAvailable, ColumnElementNumber, false,
                                 RowElementNumber> {
  using type = typename AssignSparseMatrixRowLoop<
      SparseAvailable, ColumnElementNumber,
      SparseAvailable::lists[ColumnElementNumber][RowElementNumber - 1],
      RowElementNumber - 1>::type;
};

template <typename SparseAvailable, std::size_t ColumnElementNumber>
struct AssignSparseMatrixRowLoop<SparseAvailable, ColumnElementNumber, false,
                                 0> {
  using type = InvalidSequence<0>;
};

template <typename SparseAvailable, std::size_t ColumnElementNumber>
struct AssignSparseMatrixRowLoop<SparseAvailable, ColumnElementNumber, true,
                                 0> {
  using type = IndexSequence<0>;
};

template <typename SparseAvailable, std::size_t ColumnElementNumber>
struct AssignSparseMatrixColumnLoop {
  using type = typename Concatenate<
      typename AssignSparseMatrixColumnLoop<SparseAvailable,
                                            ColumnElementNumber - 1>::type,
      typename AssignSparseMatrixRowLoop<
          SparseAvailable, ColumnElementNumber,
          SparseAvailable::lists[ColumnElementNumber]
                                [SparseAvailable::number_of_columns - 1],
          (SparseAvailable::number_of_columns - 1)>::type>::type;
};

template <typename SparseAvailable>
struct AssignSparseMatrixColumnLoop<SparseAvailable, 0> {
  using type = typename AssignSparseMatrixRowLoop<
      SparseAvailable, 0,
      SparseAvailable::lists[0][SparseAvailable::number_of_columns - 1],
      (SparseAvailable::number_of_columns - 1)>::type;
};

template <typename SparseAvailable>
struct RowIndicesSequenceFromSparseAvailable {
  using type = typename AssignSparseMatrixColumnLoop<
      SparseAvailable, (SparseAvailable::number_of_columns - 1)>::type;
};

template <typename SparseAvailable>
using RowIndicesFromSparseAvailable =
    typename ToRowIndices<typename RowIndicesSequenceFromSparseAvailable<
        SparseAvailable>::type>::type;

/* Create Row Indices */
template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber, bool Active,
          std::size_t RowElementNumber>
struct CountSparseMatrixRowLoop;

template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber, std::size_t RowElementNumber>
struct CountSparseMatrixRowLoop<SparseAvailable, ElementCount,
                                ColumnElementNumber, true, RowElementNumber> {
  using type = typename CountSparseMatrixRowLoop<
      SparseAvailable, (ElementCount + 1), ColumnElementNumber,
      SparseAvailable::lists[ColumnElementNumber][RowElementNumber - 1],
      RowElementNumber - 1>::type;
};

template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber, std::size_t RowElementNumber>
struct CountSparseMatrixRowLoop<SparseAvailable, ElementCount,
                                ColumnElementNumber, false, RowElementNumber> {
  using type = typename CountSparseMatrixRowLoop<
      SparseAvailable, ElementCount, ColumnElementNumber,
      SparseAvailable::lists[ColumnElementNumber][RowElementNumber - 1],
      RowElementNumber - 1>::type;
};

template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber>
struct CountSparseMatrixRowLoop<SparseAvailable, ElementCount,
                                ColumnElementNumber, false, 0> {
  using type = IndexSequence<ElementCount>;
};

template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber>
struct CountSparseMatrixRowLoop<SparseAvailable, ElementCount,
                                ColumnElementNumber, true, 0> {
  using type = IndexSequence<(ElementCount + 1)>;
};

template <typename SparseAvailable, std::size_t ColumnElementNumber>
struct CountSparseMatrixColumnLoop {
  using type = typename Concatenate<
      typename CountSparseMatrixColumnLoop<SparseAvailable,
                                           ColumnElementNumber - 1>::type,
      typename CountSparseMatrixRowLoop<
          SparseAvailable, 0, ColumnElementNumber,
          SparseAvailable::lists[ColumnElementNumber]
                                [SparseAvailable::number_of_columns - 1],
          (SparseAvailable::number_of_columns - 1)>::type>::type;
};

template <typename SparseAvailable>
struct CountSparseMatrixColumnLoop<SparseAvailable, 0> {
  using type = typename Concatenate<
      IndexSequence<0>,
      typename CountSparseMatrixRowLoop<
          SparseAvailable, 0, 0,
          SparseAvailable::lists[0][SparseAvailable::number_of_columns - 1],
          (SparseAvailable::number_of_columns - 1)>::type>::type;
};

template <typename CountSparseMatrixColumnLoop, std::size_t ColumnElementNumber>
struct AccumulateElementNumberLoop {
  static constexpr std::size_t compute() {
    return CountSparseMatrixColumnLoop::list[ColumnElementNumber] +
           AccumulateElementNumberLoop<CountSparseMatrixColumnLoop,
                                       ColumnElementNumber - 1>::compute();
  }
};

template <typename CountSparseMatrixColumnLoop>
struct AccumulateElementNumberLoop<CountSparseMatrixColumnLoop, 0> {
  static constexpr std::size_t compute() { return 0; }
};

template <typename CountSparseMatrixColumnLoop, std::size_t ColumnElementNumber>
struct AccumulateSparseMatrixElementNumberLoop {
  using type = typename Concatenate<
      typename AccumulateSparseMatrixElementNumberLoop<
          CountSparseMatrixColumnLoop, ColumnElementNumber - 1>::type,
      IndexSequence<AccumulateElementNumberLoop<
          CountSparseMatrixColumnLoop, ColumnElementNumber>::compute()>>::type;
};

template <typename CountSparseMatrixColumnLoop>
struct AccumulateSparseMatrixElementNumberLoop<CountSparseMatrixColumnLoop, 0> {
  using type = IndexSequence<CountSparseMatrixColumnLoop::list[0]>;
};

template <typename CountSparseMatrixColumnLoop, typename SparseAvailable>
struct AccumulateSparseMatrixElementNumberStruct {
  using type = typename AccumulateSparseMatrixElementNumberLoop<
      CountSparseMatrixColumnLoop, SparseAvailable::number_of_columns>::type;
};

template <typename SparseAvailable>
using RowPointersFromSparseAvailable =
    typename ToRowPointers<typename AccumulateSparseMatrixElementNumberStruct<
        typename CountSparseMatrixColumnLoop<
            SparseAvailable, (SparseAvailable::number_of_columns - 1)>::type,
        SparseAvailable>::type>::type;

template <typename T, typename SparseAvailable>
auto create_compiled_sparse(std::initializer_list<T> values)
    -> CompiledSparseMatrix<T, SparseAvailable::number_of_columns,
                            SparseAvailable::column_size,
                            RowIndicesFromSparseAvailable<SparseAvailable>,
                            RowPointersFromSparseAvailable<SparseAvailable>> {
  CompiledSparseMatrix<T, SparseAvailable::number_of_columns,
                       SparseAvailable::column_size,
                       RowIndicesFromSparseAvailable<SparseAvailable>,
                       RowPointersFromSparseAvailable<SparseAvailable>>
      Y;

  std::copy(values.begin(),
            values.begin() +
                RowIndicesFromSparseAvailable<SparseAvailable>::size,
            Y.values.begin());

  return Y;
}

/* Set Sparse Matrix Value */
// Core conditional operation for setting sparse matrix value
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K, std::size_t L>
struct SetSparseMatrixValueCoreConditional {
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of conditional operation, do nothing
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
    if (RowToSet == RowIndices_A::list[K]) {
      A.values[K] = value;
    }
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

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

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

#else

  COMPILED_SPARSE_SET_MATRIX_VALUE<ColumnToSet, RowToSet, T, M, N, RowIndices_A,
                                   RowPointers_A>(A, value);

#endif
}

/* Get Sparse Matrix Value */
// Core conditional operation for setting sparse matrix value
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K, std::size_t L>
struct GetSparseMatrixValueCoreConditional {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of conditional operation, do nothing
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
    if (RowToGet == RowIndices_A::list[K]) {
      value = A.values[K];
    }
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

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

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

#else

  COMPILED_SPARSE_GET_MATRIX_VALUE<ColumnToGet, RowToGet, T, M, N, RowIndices_A,
                                   RowPointers_A>(A, value);

#endif

  return value;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_COMPILED_SPARSE_HPP
