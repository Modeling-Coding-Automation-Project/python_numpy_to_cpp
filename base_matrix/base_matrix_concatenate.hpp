#ifndef BASE_MATRIX_CONCATENATE_HPP
#define BASE_MATRIX_CONCATENATE_HPP

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_templates.hpp"
#include <cstddef>
#include <tuple>

namespace Base {
namespace Matrix {

/* Functions: Concatenate vertically */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline void update_vertically_concatenated_matrix(Matrix<T, M + P, N> &Y,
                                                  const Matrix<T, M, N> &A,
                                                  const Matrix<T, P, N> &B) {
  for (std::size_t row = 0; row < N; row++) {
    std::copy(A(row).begin(), A(row).end(), Y(row).begin());
    std::copy(B(row).begin(), B(row).end(), Y(row).begin() + M);
  }
}

template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline auto concatenate_vertically(const Matrix<T, M, N> &A,
                                   const Matrix<T, P, N> &B)
    -> Matrix<T, M + P, N> {
  Matrix<T, M + P, N> Y;

  update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t N>
inline void update_vertically_concatenated_matrix(
    CompiledSparseMatrix<
        T, (M + N), N,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DenseAvailable<M, N>, DiagAvailable<N>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DenseAvailable<M, N>, DiagAvailable<N>>>> &Y,
    const Matrix<T, M, N> &A, const DiagMatrix<T, N> &B) {

  auto sparse_A = create_compiled_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), Y.values.begin());

  auto sparse_B = create_compiled_sparse(B);
  std::copy(sparse_B.values.begin(), sparse_B.values.end(),
            Y.values.begin() + M * N);
}

template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_vertically(const Matrix<T, M, N> &A,
                                   const DiagMatrix<T, N> &B)
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

  update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename RowIndices_B, typename RowPointers_B>
inline void update_vertically_concatenated_matrix(
    CompiledSparseMatrix<
        T, (M + P), N,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DenseAvailable<M, N>, CreateSparseAvailableFromIndicesAndPointers<
                                      N, RowIndices_B, RowPointers_B>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DenseAvailable<M, N>, CreateSparseAvailableFromIndicesAndPointers<
                                      N, RowIndices_B, RowPointers_B>>>> &Y,
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, P, N, RowIndices_B, RowPointers_B> &B) {

  auto sparse_A = create_compiled_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), Y.values.begin());

  std::copy(B.values.begin(), B.values.end(), Y.values.begin() + M * N);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename RowIndices_B, typename RowPointers_B>
inline auto concatenate_vertically(
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

  update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t P>
inline void update_vertically_concatenated_matrix(
    CompiledSparseMatrix<
        T, (M + P), M,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DiagAvailable<M>, DenseAvailable<P, M>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DiagAvailable<M>, DenseAvailable<P, M>>>> &Y,
    const DiagMatrix<T, M> &A, const Matrix<T, P, M> &B) {

  auto sparse_A = create_compiled_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), Y.values.begin());

  auto sparse_B = create_compiled_sparse(B);
  std::copy(sparse_B.values.begin(), sparse_B.values.end(),
            Y.values.begin() + M);
}

template <typename T, std::size_t M, std::size_t P>
inline auto concatenate_vertically(const DiagMatrix<T, M> &A,
                                   const Matrix<T, P, M> &B)
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

  update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M>
inline void update_vertically_concatenated_matrix(
    CompiledSparseMatrix<
        T, (2 * M), M,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DiagAvailable<M>, DiagAvailable<M>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DiagAvailable<M>, DiagAvailable<M>>>> &Y,
    const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B) {

  auto sparse_A = create_compiled_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), Y.values.begin());

  auto sparse_B = create_compiled_sparse(B);
  std::copy(sparse_B.values.begin(), sparse_B.values.end(),
            Y.values.begin() + M);
}

template <typename T, std::size_t M>
inline auto concatenate_vertically(const DiagMatrix<T, M> &A,
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

  update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t P, typename RowIndices_B,
          typename RowPointers_B>
inline void update_vertically_concatenated_matrix(
    CompiledSparseMatrix<
        T, (M + P), M,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DiagAvailable<M>, CreateSparseAvailableFromIndicesAndPointers<
                                  M, RowIndices_B, RowPointers_B>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            DiagAvailable<M>, CreateSparseAvailableFromIndicesAndPointers<
                                  M, RowIndices_B, RowPointers_B>>>> &Y,
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, P, M, RowIndices_B, RowPointers_B> &B) {

  auto sparse_A = create_compiled_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), Y.values.begin());

  std::copy(B.values.begin(), B.values.end(), Y.values.begin() + M);
}

template <typename T, std::size_t M, std::size_t P, typename RowIndices_B,
          typename RowPointers_B>
inline auto concatenate_vertically(
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

  update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t P>
inline void update_vertically_concatenated_matrix(
    CompiledSparseMatrix<
        T, (M + P), N,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            DenseAvailable<P, N>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            DenseAvailable<P, N>>>> &Y,
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, P, N> &B) {

  std::copy(A.values.begin(), A.values.end(), Y.values.begin());

  auto sparse_B = create_compiled_sparse(B);
  std::copy(sparse_B.values.begin(), sparse_B.values.end(),
            Y.values.begin() + RowIndices_A::size);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t P>
inline auto concatenate_vertically(
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

  update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void update_vertically_concatenated_matrix(
    CompiledSparseMatrix<
        T, (M + N), N,
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            DiagAvailable<N>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            DiagAvailable<N>>>> &Y,
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const DiagMatrix<T, N> &B) {

  std::copy(A.values.begin(), A.values.end(), Y.values.begin());

  auto sparse_B = create_compiled_sparse(B);
  std::copy(sparse_B.values.begin(), sparse_B.values.end(),
            Y.values.begin() + RowIndices_A::size);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline auto concatenate_vertically(
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

  update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
inline void update_vertically_concatenated_matrix(
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
                                                        RowPointers_B>>>> &Y,
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, P, N, RowIndices_B, RowPointers_B> &B) {

  std::copy(A.values.begin(), A.values.end(), Y.values.begin());

  std::copy(B.values.begin(), B.values.end(),
            Y.values.begin() + RowIndices_A::size);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
inline auto concatenate_vertically(
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

  update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

/* Functions: Concatenate horizontally */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline void update_horizontally_concatenated_matrix(Matrix<T, M, N + P> &Y,
                                                    const Matrix<T, M, N> &A,
                                                    const Matrix<T, M, P> &B) {

  for (std::size_t row = 0; row < N; row++) {
    std::copy(A(row).begin(), A(row).end(), Y(row).begin());
  }

  std::size_t B_row = 0;
  for (std::size_t row = N; row < N + P; row++) {
    std::copy(B(B_row).begin(), B(B_row).end(), Y(row).begin());
    B_row++;
  }
}

template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline auto concatenate_horizontally(const Matrix<T, M, N> &A,
                                     const Matrix<T, M, P> &B)
    -> Matrix<T, M, N + P> {
  Matrix<T, M, N + P> Y;

  update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t N>
inline void update_horizontally_concatenated_matrix(
    CompiledSparseMatrix<
        T, M, (M + N),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DenseAvailable<M, N>, DiagAvailable<M>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DenseAvailable<M, N>, DiagAvailable<M>>>> &Y,
    const Matrix<T, M, N> &A, const DiagMatrix<T, M> &B) {

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
}

template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_horizontally(const Matrix<T, M, N> &A,
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

  update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_B, typename RowPointers_B>
inline void update_horizontally_concatenated_matrix(
    CompiledSparseMatrix<
        T, M, (N + L),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DenseAvailable<M, N>, CreateSparseAvailableFromIndicesAndPointers<
                                      L, RowIndices_B, RowPointers_B>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DenseAvailable<M, N>, CreateSparseAvailableFromIndicesAndPointers<
                                      L, RowIndices_B, RowPointers_B>>>> &Y,
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, M, L, RowIndices_B, RowPointers_B> &B) {

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
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_B, typename RowPointers_B>
inline auto concatenate_horizontally(
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

  update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t N>
inline void update_horizontally_concatenated_matrix(
    CompiledSparseMatrix<
        T, M, (M + N),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DiagAvailable<M>, DenseAvailable<M, N>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DiagAvailable<M>, DenseAvailable<M, N>>>> &Y,
    const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B) {

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < M) {
        if (i == j) {
          Y.values[value_count] = A[i];

          value_count++;
        }
      } else {

        Y.values[value_count] = B(i, j - M);

        value_count++;
      }
    }
  }
}

template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_horizontally(const DiagMatrix<T, M> &A,
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

  update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M>
inline void update_horizontally_concatenated_matrix(
    CompiledSparseMatrix<
        T, M, (2 * M),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DiagAvailable<M>, DiagAvailable<M>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DiagAvailable<M>, DiagAvailable<M>>>> &Y,
    const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B) {

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
}

template <typename T, std::size_t M>
inline auto concatenate_horizontally(const DiagMatrix<T, M> &A,
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

  update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
inline void update_horizontally_concatenated_matrix(
    CompiledSparseMatrix<
        T, M, (M + N),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DiagAvailable<M>, CreateSparseAvailableFromIndicesAndPointers<
                                  N, RowIndices_B, RowPointers_B>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            DiagAvailable<M>, CreateSparseAvailableFromIndicesAndPointers<
                                  N, RowIndices_B, RowPointers_B>>>> &Y,
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B) {

  std::size_t value_count = 0;
  std::size_t sparse_value_count = 0;
  std::size_t sparse_col_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < M) {
        if (i == j) {

          Y.values[value_count] = A[i];
          value_count++;
        }
      } else if ((RowPointers_B::list[i + 1] - RowPointers_B::list[i] >
                  sparse_col_count) &&
                 (sparse_value_count < RowIndices_B::size)) {

        if ((j - M) == RowIndices_B::list[sparse_value_count]) {
          Y.values[value_count] = B.values[sparse_value_count];

          value_count++;
          sparse_value_count++;
          sparse_col_count++;
        }
      }
    }
    sparse_col_count = 0;
  }
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
inline auto concatenate_horizontally(
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

  update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_A, typename RowPointers_A>
inline void update_horizontally_concatenated_matrix(
    CompiledSparseMatrix<
        T, M, (N + L),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            CreateSparseAvailableFromIndicesAndPointers<L, RowIndices_A,
                                                        RowPointers_A>,
            DenseAvailable<M, N>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            CreateSparseAvailableFromIndicesAndPointers<L, RowIndices_A,
                                                        RowPointers_A>,
            DenseAvailable<M, N>>>> &Y,
    const CompiledSparseMatrix<T, M, L, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, M, N> &B) {

  std::size_t value_count = 0;
  std::size_t sparse_value_count = 0;
  std::size_t sparse_col_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + L); j++) {
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

        Y.values[value_count] = B(i, j - N);
        value_count++;
      }
    }
    sparse_col_count = 0;
  }
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_A, typename RowPointers_A>
inline auto concatenate_horizontally(
    const CompiledSparseMatrix<T, M, L, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, M, N> &B)
    -> CompiledSparseMatrix<
        T, M, (N + L),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            CreateSparseAvailableFromIndicesAndPointers<L, RowIndices_A,
                                                        RowPointers_A>,
            DenseAvailable<M, N>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            CreateSparseAvailableFromIndicesAndPointers<L, RowIndices_A,
                                                        RowPointers_A>,
            DenseAvailable<M, N>>>> {

  CompiledSparseMatrix<
      T, M, (N + L),
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          CreateSparseAvailableFromIndicesAndPointers<L, RowIndices_A,
                                                      RowPointers_A>,
          DenseAvailable<M, N>>>,
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
          CreateSparseAvailableFromIndicesAndPointers<L, RowIndices_A,
                                                      RowPointers_A>,
          DenseAvailable<M, N>>>>
      Y;

  update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void update_horizontally_concatenated_matrix(
    CompiledSparseMatrix<
        T, M, (M + N),
        RowIndicesFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            DiagAvailable<M>>>,
        RowPointersFromSparseAvailable<ConcatenateSparseAvailableHorizontally<
            CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                        RowPointers_A>,
            DiagAvailable<M>>>> &Y,
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const DiagMatrix<T, M> &B) {

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
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline auto concatenate_horizontally(
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

  update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
inline void update_horizontally_concatenated_matrix(
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
                                                        RowPointers_B>>>> &Y,
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, M, L, RowIndices_B, RowPointers_B> &B) {

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
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
inline auto concatenate_horizontally(
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

  update_horizontally_concatenated_matrix(Y, A, B);

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
