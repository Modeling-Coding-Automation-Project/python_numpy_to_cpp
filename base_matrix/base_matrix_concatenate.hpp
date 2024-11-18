#ifndef BASE_MATRIX_CONCATENATE_HPP
#define BASE_MATRIX_CONCATENATE_HPP

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include <cstddef>

namespace Base {
namespace Matrix {

/* Concatenate ColumnAvailable vertically */
template <typename Column1, typename Column2>
struct ConcatenateColumnAvailableLists;

template <bool... Flags1, bool... Flags2>
struct ConcatenateColumnAvailableLists<ColumnAvailable<Flags1...>,
                                       ColumnAvailable<Flags2...>> {
  using type = ColumnAvailable<Flags1..., Flags2...>;
};

template <typename ColumnAvailable_A, typename ColumnAvailable_B>
using ConcatenateColumnAvailableVertically =
    typename ConcatenateColumnAvailableLists<ColumnAvailable_A,
                                             ColumnAvailable_B>::type;

/* Concatenate SparseAvailable vertically */
template <typename SparseAvailable1, typename SparseAvailable2>
struct ConcatenateSparseAvailable;

template <typename... Columns1, typename... Columns2>
struct ConcatenateSparseAvailable<SparseAvailableColumns<Columns1...>,
                                  SparseAvailableColumns<Columns2...>> {
  using type = SparseAvailableColumns<Columns1..., Columns2...>;
};

template <typename SparseAvailable_A, typename SparseAvailable_B>
using ConcatenateSparseAvailableVertically =
    typename ConcatenateSparseAvailable<SparseAvailable_A,
                                        SparseAvailable_B>::type;

/* Create Dense Available */
// Generate false flags
// base case: N = 0
template <std::size_t N, bool... Flags> struct GenerateFalseFlags {
  using type = typename GenerateFalseFlags<N - 1, false, Flags...>::type;
};

// recursive termination case: N = 0
template <bool... Flags> struct GenerateFalseFlags<0, Flags...> {
  using type = ColumnAvailable<Flags...>;
};

template <std::size_t N>
using GenerateFalseColumnAvailable = typename GenerateFalseFlags<N>::type;

// Generate true flags
// base case: N = 0
template <std::size_t N, bool... Flags> struct GenerateTrueFlags {
  using type = typename GenerateTrueFlags<N - 1, true, Flags...>::type;
};

// recursive termination case: N = 0
template <bool... Flags> struct GenerateTrueFlags<0, Flags...> {
  using type = ColumnAvailable<Flags...>;
};

template <std::size_t N>
using GenerateTrueColumnAvailable = typename GenerateTrueFlags<N>::type;

// concatenate true flags vertically
// base case: N = 0
template <std::size_t M, typename ColumnAvailable, typename... Columns>
struct RepeatColumnAvailable {
  using type =
      typename RepeatColumnAvailable<M - 1, ColumnAvailable, ColumnAvailable,
                                     Columns...>::type;
};

// recursive termination case: N = 0
template <typename ColumnAvailable, typename... Columns>
struct RepeatColumnAvailable<0, ColumnAvailable, Columns...> {
  using type = SparseAvailableColumns<Columns...>;
};

// repeat ColumnAvailable M times
template <std::size_t M, std::size_t N>
using DenseAvailable =
    typename RepeatColumnAvailable<M, GenerateTrueColumnAvailable<N>>::type;

/* Create Diag Available */
// base case: N = 0
template <std::size_t N, std::size_t Index, bool... Flags>
struct GenerateIndexedTrueFlags {
  using type = typename GenerateIndexedTrueFlags<
      N - 1, Index, (N - 1 == Index ? true : false), Flags...>::type;
};

// recursive termination case: N = 0
template <std::size_t Index, bool... Flags>
struct GenerateIndexedTrueFlags<0, Index, Flags...> {
  using type = ColumnAvailable<Flags...>;
};

template <std::size_t N, std::size_t Index>
using GenerateIndexedTrueColumnAvailable =
    typename GenerateIndexedTrueFlags<N, Index>::type;

// concatenate indexed true flags vertically
// base case: N = 0
template <std::size_t M, std::size_t N, std::size_t Index,
          typename ColumnAvailable, typename... Columns>
struct IndexedRepeatColumnAvailable {
  using type = typename IndexedRepeatColumnAvailable<
      (M - 1), N, (Index - 1),
      GenerateIndexedTrueColumnAvailable<N, (Index - 1)>, ColumnAvailable,
      Columns...>::type;
};

// recursive termination case: N = 0
template <std::size_t N, std::size_t Index, typename ColumnAvailable,
          typename... Columns>
struct IndexedRepeatColumnAvailable<0, N, Index, ColumnAvailable, Columns...> {
  using type = SparseAvailableColumns<Columns...>;
};

// repeat ColumnAvailable M times
template <std::size_t M>
using DiagAvailable = typename IndexedRepeatColumnAvailable<
    M, M, (M - 1), GenerateIndexedTrueColumnAvailable<M, (M - 1)>>::type;

/* Create Sparse Available from Indices and Pointers */
// base case: N = 0
template <typename ColumnAvailable_A, typename ColumnAvailable_B, std::size_t N,
          bool... Flags>
struct GenerateORTrueFlagsLoop {
  using type = typename GenerateORTrueFlagsLoop<
      ColumnAvailable_A, ColumnAvailable_B, N - 1,
      (ColumnAvailable_A::list[N - 1] | ColumnAvailable_B::list[N - 1]),
      Flags...>::type;
};

// recursive termination case: N = 0
template <typename ColumnAvailable_A, typename ColumnAvailable_B, bool... Flags>
struct GenerateORTrueFlagsLoop<ColumnAvailable_A, ColumnAvailable_B, 0,
                               Flags...> {
  using type = ColumnAvailable<Flags...>;
};

template <typename ColumnAvailable_A, typename ColumnAvailable_B>
using GenerateORTrueFlagsColumnAvailable =
    typename GenerateORTrueFlagsLoop<ColumnAvailable_A, ColumnAvailable_B,
                                     ColumnAvailable_A::size>::type;

template <typename RowIndices, std::size_t N, std::size_t RowIndicesIndex,
          std::size_t RowEndCount>
struct CreateSparsePointersRowLoop {
  using type = typename GenerateORTrueFlagsLoop<
      GenerateIndexedTrueColumnAvailable<N, RowIndices::list[RowIndicesIndex]>,
      typename CreateSparsePointersRowLoop<RowIndices, N, (RowIndicesIndex + 1),
                                           (RowEndCount - 1)>::type,
      N>::type;
};

template <typename RowIndices, std::size_t N, std::size_t RowIndicesIndex>
struct CreateSparsePointersRowLoop<RowIndices, N, RowIndicesIndex, 0> {
  using type = GenerateFalseColumnAvailable<N>;
};

template <std::size_t N, typename RowIndices, typename RowPointers,
          std::size_t EndCount, std::size_t ConsecutiveIndex,
          typename... Columns>
struct CreateSparsePointersLoop {
  using type = typename CreateSparsePointersLoop<
      N, RowIndices, RowPointers, (EndCount - 1),
      RowPointers::list[RowPointers::size - EndCount], Columns...,
      typename CreateSparsePointersRowLoop<
          RowIndices, N, RowPointers::list[RowPointers::size - EndCount - 1],
          (RowPointers::list[RowPointers::size - EndCount] -
           RowPointers::list[RowPointers::size - EndCount - 1])>::type>::type;
};

template <std::size_t N, typename RowIndices, typename RowPointers,
          std::size_t ConsecutiveIndex, typename... Columns>
struct CreateSparsePointersLoop<N, RowIndices, RowPointers, 0, ConsecutiveIndex,
                                Columns...> {
  using type = SparseAvailableColumns<Columns...>;
};

template <std::size_t N, typename RowIndices, typename RowPointers>
using CreateSparseAvailableFromIndicesAndPointers =
    typename CreateSparsePointersLoop<N, RowIndices, RowPointers,
                                      (RowPointers::size - 1), 0>::type;

/* Functions: Concatenate vertically */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
auto concatenate_vertically(const Matrix<T, M, N> &A, const Matrix<T, P, N> &B)
    -> Matrix<T, M + P, N> {
  Matrix<T, M + P, N> result;

  for (std::size_t row = 0; row < N; row++) {
    std::copy(A(row).begin(), A(row).end(), result(row).begin());
    std::copy(B(row).begin(), B(row).end(), result(row).begin() + M);
  }

  return result;
}

template <typename T, std::size_t M, std::size_t N>
auto concatenate_vertically(const Matrix<T, M, N> &A, const DiagMatrix<T, N> &B)
    -> CompiledSparseMatrix<
        T, (M + N), N,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DenseAvailable<M, N>, DiagAvailable<N>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DenseAvailable<M, N>, DiagAvailable<N>>>> {

  CompiledSparseMatrix<
      T, (M + N), N,
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DenseAvailable<M, N>, DiagAvailable<N>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DenseAvailable<M, N>, DiagAvailable<N>>>>
      Y;

  auto sparse_A = create_compiled_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), Y.values.begin());

  auto sparse_B = create_compiled_sparse(B);
  std::copy(sparse_B.values.begin(), sparse_B.values.end(),
            Y.values.begin() + M * N);

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename RowIndices_B, typename RowPointers_B>
auto concatenate_vertically(
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, P, N, RowIndices_B, RowPointers_B> &B)
    -> CompiledSparseMatrix<
        T, (M + P), N,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DenseAvailable<M, N>, CreateSparseAvailableFromIndicesAndPointers<
                                      N, RowIndices_B, RowPointers_B>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DenseAvailable<M, N>, CreateSparseAvailableFromIndicesAndPointers<
                                      N, RowIndices_B, RowPointers_B>>>> {

  CompiledSparseMatrix<
      T, (M + P), N,
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DenseAvailable<M, N>, CreateSparseAvailableFromIndicesAndPointers<
                                    N, RowIndices_B, RowPointers_B>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DenseAvailable<M, N>, CreateSparseAvailableFromIndicesAndPointers<
                                    N, RowIndices_B, RowPointers_B>>>>
      Y;

  auto sparse_A = create_compiled_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), Y.values.begin());

  std::copy(B.values.begin(), B.values.end(), Y.values.begin() + M * N);

  return Y;
}

template <typename T, std::size_t M, std::size_t P>
auto concatenate_vertically(const DiagMatrix<T, M> &A, const Matrix<T, P, M> &B)
    -> CompiledSparseMatrix<
        T, (M + P), M,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DiagAvailable<M>, DenseAvailable<P, M>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DiagAvailable<M>, DenseAvailable<P, M>>>> {

  CompiledSparseMatrix<
      T, (M + P), M,
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DiagAvailable<M>, DenseAvailable<P, M>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DiagAvailable<M>, DenseAvailable<P, M>>>>
      Y;

  auto sparse_A = create_compiled_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), Y.values.begin());

  auto sparse_B = create_compiled_sparse(B);
  std::copy(sparse_B.values.begin(), sparse_B.values.end(),
            Y.values.begin() + M);

  return Y;
}

template <typename T, std::size_t M>
auto concatenate_vertically(const DiagMatrix<T, M> &A,
                            const DiagMatrix<T, M> &B)
    -> CompiledSparseMatrix<
        T, (2 * M), M,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DiagAvailable<M>, DiagAvailable<M>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DiagAvailable<M>, DiagAvailable<M>>>> {

  CompiledSparseMatrix<
      T, (2 * M), M,
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DiagAvailable<M>, DiagAvailable<M>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DiagAvailable<M>, DiagAvailable<M>>>>
      Y;

  auto sparse_A = create_compiled_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), Y.values.begin());

  auto sparse_B = create_compiled_sparse(B);
  std::copy(sparse_B.values.begin(), sparse_B.values.end(),
            Y.values.begin() + M);

  return Y;
}

template <typename T, std::size_t M, std::size_t P, typename RowIndices_B,
          typename RowPointers_B>
auto concatenate_vertically(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, P, M, RowIndices_B, RowPointers_B> &B)
    -> CompiledSparseMatrix<
        T, (M + P), M,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DiagAvailable<M>, CreateSparseAvailableFromIndicesAndPointers<
                                  M, RowIndices_B, RowPointers_B>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DiagAvailable<M>, CreateSparseAvailableFromIndicesAndPointers<
                                  M, RowIndices_B, RowPointers_B>>>> {

  CompiledSparseMatrix<
      T, (M + P), M,
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DiagAvailable<M>, CreateSparseAvailableFromIndicesAndPointers<
                                M, RowIndices_B, RowPointers_B>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DiagAvailable<M>, CreateSparseAvailableFromIndicesAndPointers<
                                M, RowIndices_B, RowPointers_B>>>>
      Y;

  auto sparse_A = create_compiled_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), Y.values.begin());

  std::copy(B.values.begin(), B.values.end(), Y.values.begin() + M);

  return Y;
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t P>
auto concatenate_vertically(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, P, N> &B)
    -> CompiledSparseMatrix<
        T, (M + P), N,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            DenseAvailable<P, N>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            DenseAvailable<P, N>>>> {

  CompiledSparseMatrix<
      T, (M + P), N,
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
          CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                      RowPointers_A>,
          DenseAvailable<P, N>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
          CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                      RowPointers_A>,
          DenseAvailable<P, N>>>>
      Y;

  std::copy(A.values.begin(), A.values.end(), Y.values.begin());

  auto sparse_B = create_compiled_sparse(B);
  std::copy(sparse_B.values.begin(), sparse_B.values.end(),
            Y.values.begin() + RowIndices_A::size);

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
auto concatenate_vertically(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, P, N, RowIndices_B, RowPointers_B> &B)
    -> CompiledSparseMatrix<
        T, (M + P), N,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_B,
                                                        RowPointers_B>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_B,
                                                        RowPointers_B>>>> {

  CompiledSparseMatrix<
      T, (M + P), N,
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
          CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                      RowPointers_A>,
          CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_B,
                                                      RowPointers_B>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
          CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                      RowPointers_A>,
          CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_B,
                                                      RowPointers_B>>>>
      Y;

  std::copy(A.values.begin(), A.values.end(), Y.values.begin());

  std::copy(B.values.begin(), B.values.end(),
            Y.values.begin() + RowIndices_A::size);

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t P>
Matrix<T, M, N + P> concatenate_horizontally(const Matrix<T, M, N> &A,
                                             const Matrix<T, M, P> &B) {
  Matrix<T, M, N + P> result;

  for (std::size_t row = 0; row < N; row++) {
    std::copy(A(row).begin(), A(row).end(), result(row).begin());
  }

  std::size_t B_row = 0;
  for (std::size_t row = N; row < N + P; row++) {
    std::copy(B(B_row).begin(), B(B_row).end(), result(row).begin());
    B_row++;
  }

  return result;
}

template <typename T, std::size_t M, std::size_t N>
SparseMatrix<T, M, (M + N), ((N + 1) * M)>
concatenate_horizontally(const Matrix<T, M, N> &A, const DiagMatrix<T, M> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values((N + 1) * M);
  std::vector<std::size_t> row_indices((N + 1) * M);
  std::vector<std::size_t> row_pointers(M + 1);
#else
  std::array<T, ((N + 1) * M)> values;
  std::array<std::size_t, ((N + 1) * M)> row_indices;
  std::array<std::size_t, (M + 1)> row_pointers;
#endif

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < N) {

        values[value_count] = A(i, j);
        row_indices[value_count] = j;

        value_count++;

      } else if ((j - N) == i) {

        values[value_count] = B[i];
        row_indices[value_count] = j;

        value_count++;
      }
    }
    row_pointers[i + 1] = value_count;
  }

  return SparseMatrix<T, M, (M + N), ((N + 1) * M)>(values, row_indices,
                                                    row_pointers);
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          std::size_t V>
SparseMatrix<T, M, (N + L), ((M * N) + V)>
concatenate_horizontally(const Matrix<T, M, N> &A,
                         const SparseMatrix<T, M, L, V> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values((M * N) + V);
  std::vector<std::size_t> row_indices((M * N) + V);
  std::vector<std::size_t> row_pointers(M + 1);
#else
  std::array<T, ((M * N) + V)> values;
  std::array<std::size_t, ((M * N) + V)> row_indices;
  std::array<std::size_t, (M + 1)> row_pointers;
#endif

  std::size_t value_count = 0;
  std::size_t sparse_value_count = 0;
  std::size_t sparse_col_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + L); j++) {
      if (j < N) {

        values[value_count] = A(i, j);
        row_indices[value_count] = j;

        value_count++;

      } else if ((B.row_pointers[i + 1] - B.row_pointers[i] >
                  sparse_col_count) &&
                 (sparse_value_count < V)) {

        if ((j - N) == B.row_indices[sparse_value_count]) {
          values[value_count] = B.values[sparse_value_count];
          row_indices[value_count] = j;

          value_count++;
          sparse_value_count++;
          sparse_col_count++;
        }
      }
    }

    row_pointers[i + 1] = value_count;
    sparse_col_count = 0;
  }

  return SparseMatrix<T, M, (N + L), ((M * N) + V)>(values, row_indices,
                                                    row_pointers);
}

template <typename T, std::size_t M, std::size_t N>
SparseMatrix<T, M, (M + N), ((N + 1) * M)>
concatenate_horizontally(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values((N + 1) * M);
  std::vector<std::size_t> row_indices((N + 1) * M);
  std::vector<std::size_t> row_pointers(M + 1);
#else
  std::array<T, ((N + 1) * M)> values;
  std::array<std::size_t, ((N + 1) * M)> row_indices;
  std::array<std::size_t, (M + 1)> row_pointers;
#endif

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < N) {
        if (i == j) {
          values[value_count] = A[i];
          row_indices[value_count] = j;

          value_count++;
        }
      } else {

        values[value_count] = B(i, j - N);
        row_indices[value_count] = j;

        value_count++;
      }
    }
    row_pointers[i + 1] = value_count;
  }

  return SparseMatrix<T, M, (M + N), ((N + 1) * M)>(values, row_indices,
                                                    row_pointers);
}

template <typename T, std::size_t M>
SparseMatrix<T, M, (2 * M), (2 * M)>
concatenate_horizontally(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values(2 * M);
  std::vector<std::size_t> row_indices(2 * M);
  std::vector<std::size_t> row_pointers(M + 1);
#else
  std::array<T, (2 * M)> values;
  std::array<std::size_t, (2 * M)> row_indices;
  std::array<std::size_t, (M + 1)> row_pointers;
#endif

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (2 * M); j++) {
      if (j < M) {
        if (i == j) {
          values[value_count] = A[i];
          row_indices[value_count] = j;

          value_count++;
        }
      } else {
        if ((j - M) == i) {
          values[value_count] = B[i];
          row_indices[value_count] = j;

          value_count++;
        }
      }
    }
    row_pointers[i + 1] = value_count;
  }

  return SparseMatrix<T, M, (2 * M), (2 * M)>(values, row_indices,
                                              row_pointers);
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
SparseMatrix<T, M, (M + N), (M + V)>
concatenate_horizontally(const DiagMatrix<T, M> &A,
                         const SparseMatrix<T, M, N, V> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values(M + V);
  std::vector<std::size_t> row_indices(M + V);
  std::vector<std::size_t> row_pointers(M + 1);
#else
  std::array<T, (M + V)> values;
  std::array<std::size_t, (M + V)> row_indices;
  std::array<std::size_t, (M + 1)> row_pointers;
#endif

  std::size_t value_count = 0;
  std::size_t sparse_value_count = 0;
  std::size_t sparse_col_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < N) {
        if (i == j) {
          values[value_count] = A[i];
          row_indices[value_count] = j;

          value_count++;
        }
      } else if ((B.row_pointers[i + 1] - B.row_pointers[i] >
                  sparse_col_count) &&
                 (sparse_value_count < V)) {

        if ((j - N) == B.row_indices[sparse_value_count]) {
          values[value_count] = B.values[sparse_value_count];
          row_indices[value_count] = j;

          value_count++;
          sparse_value_count++;
          sparse_col_count++;
        }
      }
    }

    row_pointers[i + 1] = value_count;
    sparse_col_count = 0;
  }

  return SparseMatrix<T, M, (M + N), (M + V)>(values, row_indices,
                                              row_pointers);
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          std::size_t V, std::size_t W>
SparseMatrix<T, M, (N + L), (V + W)>
concatenate_horizontally(const SparseMatrix<T, M, N, V> &A,
                         const SparseMatrix<T, M, L, W> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values(V + W);
  std::vector<std::size_t> row_indices(V + W);
  std::vector<std::size_t> row_pointers(M + 1);
#else
  std::array<T, (V + W)> values;
  std::array<std::size_t, (V + W)> row_indices;
  std::array<std::size_t, (M + 1)> row_pointers;
#endif

  std::size_t value_count = 0;
  std::size_t sparse_value_count_A = 0;
  std::size_t sparse_col_count_A = 0;
  std::size_t sparse_value_count_B = 0;
  std::size_t sparse_col_count_B = 0;

  row_pointers[0] = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if ((j < N) &&
          (A.row_pointers[i + 1] - A.row_pointers[i] > sparse_col_count_A) &&
          (sparse_value_count_A < V)) {

        if (j == A.row_indices[sparse_value_count_A]) {
          values[value_count] = A.values[sparse_value_count_A];
          row_indices[value_count] = j;

          value_count++;
          sparse_value_count_A++;
          sparse_col_count_A++;
        }
      } else if ((B.row_pointers[i + 1] - B.row_pointers[i] >
                  sparse_col_count_B) &&
                 (sparse_value_count_B < W)) {

        if ((j - N) == B.row_indices[sparse_value_count_B]) {
          values[value_count] = B.values[sparse_value_count_B];
          row_indices[value_count] = j;

          value_count++;
          sparse_value_count_B++;
          sparse_col_count_B++;
        }
      }
    }

    row_pointers[i + 1] = value_count;
    sparse_col_count_A = 0;
    sparse_col_count_B = 0;
  }

  return SparseMatrix<T, M, (N + L), (V + W)>(values, row_indices,
                                              row_pointers);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t K>
Matrix<T, M + K, N + P>
concatenate_square(const Matrix<T, M, N> &A, const Matrix<T, M, P> &B,
                   const Matrix<T, K, N> &C, const Matrix<T, K, P> &D) {
  Matrix<T, M + K, N + P> result;

  for (std::size_t i = 0; i < N; ++i) {
    std::copy(A(i).begin(), A(i).end(), result(i).begin());
    std::copy(C(i).begin(), C(i).end(), result(i).begin() + M);
  }

  std::size_t B_row = 0;
  for (std::size_t i = N; i < N + P; ++i) {
    std::copy(B(B_row).begin(), B(B_row).end(), result(i).begin());
    std::copy(D(B_row).begin(), D(B_row).end(), result(i).begin() + M);
    B_row++;
  }

  return result;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_CONCATENATE_HPP
