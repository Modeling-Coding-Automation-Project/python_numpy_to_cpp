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

/* Concatenate ColumnAvailable */
template <typename Column1, typename Column2>
struct ConcatenateColumnAvailableLists;

template <bool... Flags1, bool... Flags2>
struct ConcatenateColumnAvailableLists<ColumnAvailable<Flags1...>,
                                       ColumnAvailable<Flags2...>> {
  using type = ColumnAvailable<Flags1..., Flags2...>;
};

template <typename ColumnAvailable_A, typename ColumnAvailable_B>
using ConcatenateColumnAvailable =
    typename ConcatenateColumnAvailableLists<ColumnAvailable_A,
                                             ColumnAvailable_B>::type;

/* Get ColumnAvailable from SparseAvailable */
template <std::size_t N, typename SparseAvailable> struct GetColumnAvailable;

template <std::size_t N, typename... Columns>
struct GetColumnAvailable<N, SparseAvailable<Columns...>> {
  using type = typename std::tuple_element<N, std::tuple<Columns...>>::type;
};

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

/* Concatenate SparseAvailable horizontally */

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

/* Concatenate SparseAvailable horizontally */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t ColumnCount>
struct ConcatenateSparseAvailableHorizontallyLoop {
  using type = typename ConcatenateSparseAvailable<
      typename ConcatenateSparseAvailableHorizontallyLoop<
          SparseAvailable_A, SparseAvailable_B, (ColumnCount - 1)>::type,
      SparseAvailableColumns<typename ConcatenateColumnAvailableLists<
          typename GetColumnAvailable<ColumnCount, SparseAvailable_A>::type,
          typename GetColumnAvailable<ColumnCount,
                                      SparseAvailable_B>::type>::type>>::type;
};

template <typename SparseAvailable_A, typename SparseAvailable_B>
struct ConcatenateSparseAvailableHorizontallyLoop<SparseAvailable_A,
                                                  SparseAvailable_B, 0> {
  using type = SparseAvailableColumns<typename ConcatenateColumnAvailableLists<
      typename GetColumnAvailable<0, SparseAvailable_A>::type,
      typename GetColumnAvailable<0, SparseAvailable_B>::type>::type>;
};

template <typename SparseAvailable_A, typename SparseAvailable_B>
using ConcatenateSparseAvailableHorizontally =
    typename ConcatenateSparseAvailableHorizontallyLoop<
        SparseAvailable_A, SparseAvailable_B,
        SparseAvailable_A::number_of_columns - 1>::type;

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

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
auto concatenate_vertically(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const DiagMatrix<T, N> &B)
    -> CompiledSparseMatrix<
        T, (M + N), N,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            DiagAvailable<N>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            DiagAvailable<N>>>> {

  CompiledSparseMatrix<
      T, (M + N), N,
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
          CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                      RowPointers_A>,
          DiagAvailable<N>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
          CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                      RowPointers_A>,
          DiagAvailable<N>>>>
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

/* Functions: Concatenate horizontally */
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
auto concatenate_horizontally(const Matrix<T, M, N> &A,
                              const DiagMatrix<T, M> &B)
    -> CompiledSparseMatrix<
        T, M, (M + N),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DenseAvailable<M, N>, DiagAvailable<M>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DenseAvailable<M, N>, DiagAvailable<M>>>> {

  CompiledSparseMatrix<
      T, M, (M + N),
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          DenseAvailable<M, N>, DiagAvailable<M>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          DenseAvailable<M, N>, DiagAvailable<M>>>>
      Y;

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < N) {

        Y.values[value_count] = A(i, j);
        value_count++;

      } else if ((j - N) == i) {

        Y.values[value_count] = B[i];
        value_count++;
      }
    }
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_B, typename RowPointers_B>
auto concatenate_horizontally(
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, M, L, RowIndices_B, RowPointers_B> &B)
    -> CompiledSparseMatrix<
        T, M, (N + L),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DenseAvailable<M, N>, CreateSparseAvailableFromIndicesAndPointers<
                                      L, RowIndices_B, RowPointers_B>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DenseAvailable<M, N>, CreateSparseAvailableFromIndicesAndPointers<
                                      L, RowIndices_B, RowPointers_B>>>> {

  CompiledSparseMatrix<
      T, M, (N + L),
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          DenseAvailable<M, N>, CreateSparseAvailableFromIndicesAndPointers<
                                    L, RowIndices_B, RowPointers_B>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          DenseAvailable<M, N>, CreateSparseAvailableFromIndicesAndPointers<
                                    L, RowIndices_B, RowPointers_B>>>>
      Y;

  std::size_t value_count = 0;
  std::size_t sparse_value_count = 0;
  std::size_t sparse_col_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + L); j++) {
      if (j < N) {

        Y.values[value_count] = A(i, j);
        value_count++;

      } else if ((RowPointers_B::list[i + 1] - RowPointers_B::list[i] >
                  sparse_col_count) &&
                 (sparse_value_count < RowIndices_B::size)) {

        if ((j - N) == RowIndices_B::list[sparse_value_count]) {
          Y.values[value_count] = B.values[sparse_value_count];

          value_count++;
          sparse_value_count++;
          sparse_col_count++;
        }
      }
    }
    sparse_col_count = 0;
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t N>
auto concatenate_horizontally(const DiagMatrix<T, M> &A,
                              const Matrix<T, M, N> &B)
    -> CompiledSparseMatrix<
        T, M, (M + N),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DiagAvailable<M>, DenseAvailable<M, N>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DiagAvailable<M>, DenseAvailable<M, N>>>> {

  CompiledSparseMatrix<
      T, M, (M + N),
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          DiagAvailable<M>, DenseAvailable<M, N>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          DiagAvailable<M>, DenseAvailable<M, N>>>>
      Y;

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < N) {
        if (i == j) {
          Y.values[value_count] = A[i];

          value_count++;
        }
      } else {

        Y.values[value_count] = B(i, j - N);

        value_count++;
      }
    }
  }

  return Y;
}

template <typename T, std::size_t M>
auto concatenate_horizontally(const DiagMatrix<T, M> &A,
                              const DiagMatrix<T, M> &B)
    -> CompiledSparseMatrix<
        T, M, (2 * M),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DiagAvailable<M>, DiagAvailable<M>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DiagAvailable<M>, DiagAvailable<M>>>> {

  CompiledSparseMatrix<
      T, M, (2 * M),
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          DiagAvailable<M>, DiagAvailable<M>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          DiagAvailable<M>, DiagAvailable<M>>>>
      Y;

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (2 * M); j++) {
      if (j < M) {
        if (i == j) {

          Y.values[value_count] = A[i];
          value_count++;
        }
      } else {
        if ((j - M) == i) {

          Y.values[value_count] = B[i];
          value_count++;
        }
      }
    }
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
auto concatenate_horizontally(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B)
    -> CompiledSparseMatrix<
        T, M, (M + N),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DiagAvailable<M>, CreateSparseAvailableFromIndicesAndPointers<
                                  N, RowIndices_B, RowPointers_B>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DiagAvailable<M>, CreateSparseAvailableFromIndicesAndPointers<
                                  N, RowIndices_B, RowPointers_B>>>> {

  CompiledSparseMatrix<
      T, M, (M + N),
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          DiagAvailable<M>, CreateSparseAvailableFromIndicesAndPointers<
                                N, RowIndices_B, RowPointers_B>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          DiagAvailable<M>, CreateSparseAvailableFromIndicesAndPointers<
                                N, RowIndices_B, RowPointers_B>>>>
      Y;

  std::size_t value_count = 0;
  std::size_t sparse_value_count = 0;
  std::size_t sparse_col_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < N) {
        if (i == j) {

          Y.values[value_count] = A[i];
          value_count++;
        }
      } else if ((RowPointers_B::list[i + 1] - RowPointers_B::list[i] >
                  sparse_col_count) &&
                 (sparse_value_count < RowIndices_B::size)) {

        if ((j - N) == RowIndices_B::list[sparse_value_count]) {
          Y.values[value_count] = B.values[sparse_value_count];

          value_count++;
          sparse_value_count++;
          sparse_col_count++;
        }
      }
    }
    sparse_col_count = 0;
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
auto concatenate_horizontally(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const DiagMatrix<T, M> &B)
    -> CompiledSparseMatrix<
        T, M, (M + N),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            DiagAvailable<M>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            DiagAvailable<M>>>> {

  CompiledSparseMatrix<
      T, M, (M + N),
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                      RowPointers_A>,
          DiagAvailable<M>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                      RowPointers_A>,
          DiagAvailable<M>>>>
      Y;

  std::size_t value_count = 0;
  std::size_t sparse_value_count = 0;
  std::size_t sparse_col_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < N) {

        if ((RowPointers_A::list[i + 1] - RowPointers_A::list[i] >
             sparse_col_count) &&
            (sparse_value_count < RowIndices_A::size)) {

          if (j == RowIndices_A::list[sparse_value_count]) {
            Y.values[value_count] = A.values[sparse_value_count];

            value_count++;
            sparse_value_count++;
            sparse_col_count++;
          }
        }

      } else {
        if (i == (j - N)) {

          Y.values[value_count] = B[i];
          value_count++;
        }
      }
    }
    sparse_col_count = 0;
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
auto concatenate_horizontally(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, M, L, RowIndices_B, RowPointers_B> &B)
    -> CompiledSparseMatrix<
        T, M, (N + L),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_B,
                                                        RowPointers_B>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            CreateSparseAvailableFromIndicesAndPointers<L, RowIndices_A,
                                                        RowPointers_A>,
            CreateSparseAvailableFromIndicesAndPointers<L, RowIndices_B,
                                                        RowPointers_B>>>> {

  CompiledSparseMatrix<
      T, M, (N + L),
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                      RowPointers_A>,
          CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_B,
                                                      RowPointers_B>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          CreateSparseAvailableFromIndicesAndPointers<L, RowIndices_A,
                                                      RowPointers_A>,
          CreateSparseAvailableFromIndicesAndPointers<L, RowIndices_B,
                                                      RowPointers_B>>>>
      Y;

  std::size_t value_count = 0;
  std::size_t sparse_value_count_A = 0;
  std::size_t sparse_col_count_A = 0;
  std::size_t sparse_value_count_B = 0;
  std::size_t sparse_col_count_B = 0;

  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if ((j < N) &&
          (RowPointers_A::list[i + 1] - RowPointers_A::list[i] >
           sparse_col_count_A) &&
          (sparse_value_count_A < RowIndices_A::size)) {

        if (j == RowIndices_A::list[sparse_value_count_A]) {
          Y.values[value_count] = A.values[sparse_value_count_A];

          value_count++;
          sparse_value_count_A++;
          sparse_col_count_A++;
        }
      } else if ((RowPointers_B::list[i + 1] - RowPointers_B::list[i] >
                  sparse_col_count_B) &&
                 (sparse_value_count_B < RowIndices_B::size)) {

        if ((j - N) == RowIndices_B::list[sparse_value_count_B]) {
          Y.values[value_count] = B.values[sparse_value_count_B];

          value_count++;
          sparse_value_count_B++;
          sparse_col_count_B++;
        }
      }
    }
    sparse_col_count_A = 0;
    sparse_col_count_B = 0;
  }

  return Y;
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
