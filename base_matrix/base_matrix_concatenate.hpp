#ifndef __BASE_MATRIX_CONCATENATE_HPP__
#define __BASE_MATRIX_CONCATENATE_HPP__

#include "base_matrix_macros.hpp"

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_templates.hpp"
#include "base_utility.hpp"

#include <cstddef>
#include <iostream>
#include <tuple>

namespace Base {
namespace Matrix {

/* Functions: Concatenate vertically */
template <typename T, std::size_t M, std::size_t P, std::size_t N,
          std::size_t Row>
struct VerticalConcatenateLoop {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, P, N> &B,
                      Matrix<T, M + P, N> &Y) {
    Base::Utility::copy<T, 0, M, 0, M, (M + P)>(A.data[Row], Y.data[Row]);
    Base::Utility::copy<T, 0, P, M, P, (M + P)>(B.data[Row], Y.data[Row]);
    VerticalConcatenateLoop<T, M, P, N, Row - 1>::compute(A, B, Y);
  }
};

// end of recursion
template <typename T, std::size_t M, std::size_t P, std::size_t N>
struct VerticalConcatenateLoop<T, M, P, N, 0> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, P, N> &B,
                      Matrix<T, M + P, N> &Y) {
    Base::Utility::copy<T, 0, M, 0, M, (M + P)>(A.data[0], Y.data[0]);
    Base::Utility::copy<T, 0, P, M, P, (M + P)>(B.data[0], Y.data[0]);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t P>
static inline void
COMPILED_SPARSE_VERTICAL_CONCATENATE(const Matrix<T, M, N> &A,
                                     const Matrix<T, P, N> &B,
                                     Matrix<T, M + P, N> &Y) {
  VerticalConcatenateLoop<T, M, P, N, N - 1>::compute(A, B, Y);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline void update_vertically_concatenated_matrix(Matrix<T, M + P, N> &Y,
                                                  const Matrix<T, M, N> &A,
                                                  const Matrix<T, P, N> &B) {

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t row = 0; row < N; row++) {
    Base::Utility::copy<T, 0, M, 0, M, (M + P)>(A.data[row], Y.data[row]);
    Base::Utility::copy<T, 0, P, M, P, (M + P)>(B.data[row], Y.data[row]);
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_SPARSE_VERTICAL_CONCATENATE<T, M, N, P>(A, B, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline auto concatenate_vertically(const Matrix<T, M, N> &A,
                                   const Matrix<T, P, N> &B)
    -> Matrix<T, M + P, N> {
  Matrix<T, M + P, N> Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t N> struct DenseAndDiag {

  using RowIndices =
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DenseAvailable<M, N>, DiagAvailable<N>>>;

  using RowPointers =
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DenseAvailable<M, N>, DiagAvailable<N>>>;

  using Y_Type = CompiledSparseMatrix<T, (M + N), N, RowIndices, RowPointers>;
};

} // namespace ConcatenateVertically

template <typename T, std::size_t M, std::size_t N>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::DenseAndDiag<T, M, N>::Y_Type &Y,
    const Matrix<T, M, N> &A, const DiagMatrix<T, N> &B) {

  auto sparse_A = create_compiled_sparse(A);
  Base::Utility::copy<T, 0, (M * N), 0, (M * N), ((M * N) + N)>(sparse_A.values,
                                                                Y.values);

  auto sparse_B = create_compiled_sparse(B);
  Base::Utility::copy<T, 0, N, (M * N), N, ((M * N) + N)>(sparse_B.values,
                                                          Y.values);
}

template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_vertically(const Matrix<T, M, N> &A,
                                   const DiagMatrix<T, N> &B) ->
    typename ConcatenateVertically::DenseAndDiag<T, M, N>::Y_Type {

  typename ConcatenateVertically::DenseAndDiag<T, M, N>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename RowIndices_B, typename RowPointers_B>
struct DenseAndSparse {

  using SparseAvailable_A = DenseAvailable<M, N>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_B,
                                                  RowPointers_B>;

  using SparseAvailable_A_v_B =
      ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                           SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_A_v_B>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_A_v_B>;

  using Y_Type =
      CompiledSparseMatrix<T, (M + P), N, RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateVertically

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename RowIndices_B, typename RowPointers_B>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::DenseAndSparse<T, M, N, P, RowIndices_B,
                                                   RowPointers_B>::Y_Type &Y,
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, P, N, RowIndices_B, RowPointers_B> &B) {

  auto sparse_A = Base::Matrix::create_compiled_sparse(A);
  Base::Utility::copy<T, 0, (M * N), 0, (M * N),
                      ((M * N) + RowPointers_B::list[P])>(sparse_A.values,
                                                          Y.values);

  Base::Utility::copy<T, 0, RowPointers_B::list[P], (M * N),
                      RowPointers_B::list[P],
                      ((M * N) + RowPointers_B::list[P])>(B.values, Y.values);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename RowIndices_B, typename RowPointers_B>
inline auto concatenate_vertically(
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, P, N, RowIndices_B, RowPointers_B> &B) ->
    typename ConcatenateVertically::DenseAndSparse<T, M, N, P, RowIndices_B,
                                                   RowPointers_B>::Y_Type {

  typename ConcatenateVertically::DenseAndSparse<T, M, N, P, RowIndices_B,
                                                 RowPointers_B>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t P> struct DiagAndDense {

  using SparseAvailable_A = DiagAvailable<M>;

  using SparseAvailable_B = DenseAvailable<P, M>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                           SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_Y>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, (M + P), M, RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateVertically

template <typename T, std::size_t M, std::size_t P>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::DiagAndDense<T, M, P>::Y_Type &Y,
    const DiagMatrix<T, M> &A, const Matrix<T, P, M> &B) {

  auto sparse_A = Base::Matrix::create_compiled_sparse(A);
  Base::Utility::copy<T, 0, M, 0, M, (M + (M * P))>(sparse_A.values, Y.values);

  auto sparse_B = create_compiled_sparse(B);
  Base::Utility::copy<T, 0, (M * P), M, (M * P), (M + (M * P))>(sparse_B.values,
                                                                Y.values);
}

template <typename T, std::size_t M, std::size_t P>
inline auto concatenate_vertically(const DiagMatrix<T, M> &A,
                                   const Matrix<T, P, M> &B) ->
    typename ConcatenateVertically::DiagAndDense<T, M, P>::Y_Type {

  typename ConcatenateVertically::DiagAndDense<T, M, P>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M> struct DiagAndDiag {

  using SparseAvailable_A = DiagAvailable<M>;

  using SparseAvailable_B = DiagAvailable<M>;

  using RowIndices_Y =
      RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
          SparseAvailable_A, SparseAvailable_B>>;

  using RowPointers_Y =
      RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
          SparseAvailable_A, SparseAvailable_B>>;

  using Y_Type =
      CompiledSparseMatrix<T, (2 * M), M, RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateVertically

template <typename T, std::size_t M>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::DiagAndDiag<T, M>::Y_Type &Y,
    const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B) {

  auto sparse_A = Base::Matrix::create_compiled_sparse(A);
  Base::Utility::copy<T, 0, M, 0, M, (2 * M)>(sparse_A.values, Y.values);

  auto sparse_B = Base::Matrix::create_compiled_sparse(B);
  Base::Utility::copy<T, 0, M, M, M, (2 * M)>(sparse_B.values, Y.values);
}

template <typename T, std::size_t M>
inline auto concatenate_vertically(const DiagMatrix<T, M> &A,
                                   const DiagMatrix<T, M> &B) ->
    typename ConcatenateVertically::DiagAndDiag<T, M>::Y_Type {

  typename ConcatenateVertically::DiagAndDiag<T, M>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t P, typename RowIndices_B,
          typename RowPointers_B>
struct DiagAndSparse {

  using SparseAvailable_A = DiagAvailable<M>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<M, RowIndices_B,
                                                  RowPointers_B>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                           SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_Y>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, (M + P), M, RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateVertically

template <typename T, std::size_t M, std::size_t P, typename RowIndices_B,
          typename RowPointers_B>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::DiagAndSparse<T, M, P, RowIndices_B,
                                                  RowPointers_B>::Y_Type &Y,
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, P, M, RowIndices_B, RowPointers_B> &B) {

  auto sparse_A = Base::Matrix::create_compiled_sparse(A);
  Base::Utility::copy<T, 0, M, 0, M, (M + RowPointers_B::list[P])>(
      sparse_A.values, Y.values);

  Base::Utility::copy<T, 0, RowPointers_B::list[P], M, RowPointers_B::list[P],
                      (M + RowPointers_B::list[P])>(B.values, Y.values);
}

template <typename T, std::size_t M, std::size_t P, typename RowIndices_B,
          typename RowPointers_B>
inline auto concatenate_vertically(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, P, M, RowIndices_B, RowPointers_B> &B) ->
    typename ConcatenateVertically::DiagAndSparse<T, M, P, RowIndices_B,
                                                  RowPointers_B>::Y_Type {

  typename ConcatenateVertically::DiagAndSparse<T, M, P, RowIndices_B,
                                                RowPointers_B>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t P>
struct SparseAndDense {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                  RowPointers_A>;

  using SparseAvailable_B = DenseAvailable<P, N>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                           SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_Y>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, (M + P), N, RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateVertically

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t P>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::SparseAndDense<T, M, N, RowIndices_A,
                                                   RowPointers_A, P>::Y_Type &Y,
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, P, N> &B) {

  Base::Utility::copy<T, 0, RowPointers_A::list[P], 0, RowPointers_A::list[P],
                      (RowPointers_A::list[P] + (P * N))>(A.values, Y.values);

  auto sparse_B = Base::Matrix::create_compiled_sparse(B);
  Base::Utility::copy<T, 0, (P * N), RowPointers_A::list[P], (P * N),
                      (RowPointers_A::list[P] + (P * N))>(sparse_B.values,
                                                          Y.values);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t P>
inline auto concatenate_vertically(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, P, N> &B) ->
    typename ConcatenateVertically::SparseAndDense<T, M, N, RowIndices_A,
                                                   RowPointers_A, P>::Y_Type {

  typename ConcatenateVertically::SparseAndDense<T, M, N, RowIndices_A,
                                                 RowPointers_A, P>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct SparseAndDiag {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                  RowPointers_A>;

  using SparseAvailable_B = DiagAvailable<N>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                           SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_Y>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, (M + N), N, RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateVertically

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::SparseAndDiag<T, M, N, RowIndices_A,
                                                  RowPointers_A>::Y_Type &Y,
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const DiagMatrix<T, N> &B) {

  Base::Utility::copy<T, 0, RowPointers_A::list[M], 0, RowPointers_A::list[M],
                      (RowPointers_A::list[M] + N)>(A.values, Y.values);

  auto sparse_B = Base::Matrix::create_compiled_sparse(B);
  Base::Utility::copy<T, 0, N, RowPointers_A::list[M], N,
                      (RowPointers_A::list[M] + N)>(sparse_B.values, Y.values);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline auto concatenate_vertically(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const DiagMatrix<T, N> &B) ->
    typename ConcatenateVertically::SparseAndDiag<T, M, N, RowIndices_A,
                                                  RowPointers_A>::Y_Type {

  typename ConcatenateVertically::SparseAndDiag<T, M, N, RowIndices_A,
                                                RowPointers_A>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
struct SparseAndSparse {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                  RowPointers_A>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_B,
                                                  RowPointers_B>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                           SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_Y>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, (M + P), N, RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateVertically

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::SparseAndSparse<T, M, N, P, RowIndices_A,
                                                    RowPointers_A, RowIndices_B,
                                                    RowPointers_B>::Y_Type &Y,
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, P, N, RowIndices_B, RowPointers_B> &B) {

  Base::Utility::copy<T, 0, RowPointers_A::list[M], 0, RowPointers_A::list[M],
                      (RowPointers_A::list[M] + RowPointers_B::list[P])>(
      A.values, Y.values);

  Base::Utility::copy<T, 0, RowPointers_B::list[P], RowPointers_A::list[M],
                      RowPointers_B::list[P],
                      (RowPointers_A::list[M] + RowPointers_B::list[P])>(
      B.values, Y.values);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
inline auto concatenate_vertically(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, P, N, RowIndices_B, RowPointers_B> &B) ->
    typename ConcatenateVertically::SparseAndSparse<T, M, N, P, RowIndices_A,
                                                    RowPointers_A, RowIndices_B,
                                                    RowPointers_B>::Y_Type {

  typename ConcatenateVertically::SparseAndSparse<T, M, N, P, RowIndices_A,
                                                  RowPointers_A, RowIndices_B,
                                                  RowPointers_B>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

/* Functions: Concatenate horizontally */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Row>
struct CopyRowsFirstLoop {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N + P> &Y) {
    Base::Utility::copy<T, 0, M, 0, M, M>(A(Row), Y(Row));
    CopyRowsFirstLoop<T, M, N, P, Row - 1>::compute(A, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t P>
struct CopyRowsFirstLoop<T, M, N, P, 0> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N + P> &Y) {
    Base::Utility::copy<T, 0, M, 0, M, M>(A(0), Y(0));
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t P>
static inline void
COMPILED_SPARSE_HORIZONTAL_CONCATENATE_1(const Matrix<T, M, N> &A,
                                         Matrix<T, M, N + P> &Y) {
  CopyRowsFirstLoop<T, M, N, P, N - 1>::compute(A, Y);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Row>
struct CopyRowsSecondLoop {
  static void compute(const Matrix<T, M, P> &B, Matrix<T, M, N + P> &Y) {
    Base::Utility::copy<T, 0, M, 0, M, M>(B(Row), Y(N + Row));
    CopyRowsSecondLoop<T, M, N, P, Row - 1>::compute(B, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t P>
struct CopyRowsSecondLoop<T, M, N, P, 0> {
  static void compute(const Matrix<T, M, P> &B, Matrix<T, M, N + P> &Y) {
    Base::Utility::copy<T, 0, M, 0, M, M>(B(0), Y(N));
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t P>
static inline void
COMPILED_SPARSE_HORIZONTAL_CONCATENATE_2(const Matrix<T, M, P> &B,
                                         Matrix<T, M, N + P> &Y) {
  CopyRowsSecondLoop<T, M, N, P, P - 1>::compute(B, Y);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline void update_horizontally_concatenated_matrix(Matrix<T, M, N + P> &Y,
                                                    const Matrix<T, M, N> &A,
                                                    const Matrix<T, M, P> &B) {

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t row = 0; row < N; row++) {
    std::copy(A(row).begin(), A(row).end(), Y(row).begin());
  }

  std::size_t B_row = 0;
  for (std::size_t row = N; row < N + P; row++) {
    std::copy(B(B_row).begin(), B(B_row).end(), Y(row).begin());
    B_row++;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_SPARSE_HORIZONTAL_CONCATENATE_1<T, M, N, P>(A, Y);
  Base::Matrix::COMPILED_SPARSE_HORIZONTAL_CONCATENATE_2<T, M, N, P>(B, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline auto concatenate_horizontally(const Matrix<T, M, N> &A,
                                     const Matrix<T, M, P> &B)
    -> Matrix<T, M, N + P> {
  Matrix<T, M, N + P> Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

/* Copy DenseMatrix to horizontally concatenated matrix */
// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t Y_Col,
          std::size_t Y_Row, std::size_t Column_Offset, std::size_t Row_Offset,
          typename RowIndices_Y, typename RowPointers_Y, std::size_t I,
          std::size_t J_idx>
struct ConcatMatrixSetFromDenseColumn {
  static void
  compute(CompiledSparseMatrix<T, Y_Col, Y_Row, RowIndices_Y, RowPointers_Y> &Y,
          const Matrix<T, M, N> &A) {

    Base::Matrix::set_sparse_matrix_value<(I + Column_Offset),
                                          (J_idx + Row_Offset)>(Y, A(I, J_idx));

    ConcatMatrixSetFromDenseColumn<T, M, N, Y_Col, Y_Row, Column_Offset,
                                   Row_Offset, RowIndices_Y, RowPointers_Y, I,
                                   J_idx - 1>::compute(Y, A);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t Y_Col,
          std::size_t Y_Row, std::size_t Column_Offset, std::size_t Row_Offset,
          typename RowIndices_Y, typename RowPointers_Y, std::size_t I>
struct ConcatMatrixSetFromDenseColumn<T, M, N, Y_Col, Y_Row, Column_Offset,
                                      Row_Offset, RowIndices_Y, RowPointers_Y,
                                      I, 0> {
  static void
  compute(CompiledSparseMatrix<T, Y_Col, Y_Row, RowIndices_Y, RowPointers_Y> &Y,
          const Matrix<T, M, N> &A) {

    Base::Matrix::set_sparse_matrix_value<(I + Column_Offset), Row_Offset>(
        Y, A(I, 0));
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t Y_Col,
          std::size_t Y_Row, std::size_t Column_Offset, std::size_t Row_Offset,
          typename RowIndices_Y, typename RowPointers_Y, std::size_t I_idx>
struct ConcatMatrixSetFromDenseRow {
  static void
  compute(CompiledSparseMatrix<T, Y_Col, Y_Row, RowIndices_Y, RowPointers_Y> &Y,
          const Matrix<T, M, N> &A) {
    ConcatMatrixSetFromDenseColumn<T, M, N, Y_Col, Y_Row, Column_Offset,
                                   Row_Offset, RowIndices_Y, RowPointers_Y,
                                   I_idx, N - 1>::compute(Y, A);
    ConcatMatrixSetFromDenseRow<T, M, N, Y_Col, Y_Row, Column_Offset,
                                Row_Offset, RowIndices_Y, RowPointers_Y,
                                I_idx - 1>::compute(Y, A);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t Y_Col,
          std::size_t Y_Row, std::size_t Column_Offset, std::size_t Row_Offset,
          typename RowIndices_Y, typename RowPointers_Y>
struct ConcatMatrixSetFromDenseRow<T, M, N, Y_Col, Y_Row, Column_Offset,
                                   Row_Offset, RowIndices_Y, RowPointers_Y, 0> {
  static void
  compute(CompiledSparseMatrix<T, Y_Col, Y_Row, RowIndices_Y, RowPointers_Y> &Y,
          const Matrix<T, M, N> &A) {
    ConcatMatrixSetFromDenseColumn<T, M, N, Y_Col, Y_Row, Column_Offset,
                                   Row_Offset, RowIndices_Y, RowPointers_Y, 0,
                                   N - 1>::compute(Y, A);
  }
};

/* Copy DiagMatrix to horizontally concatenated matrix */
// when I_idx < M
template <typename T, std::size_t M, std::size_t Y_Col, std::size_t Y_Row,
          std::size_t Column_Offset, std::size_t Row_Offset,
          typename RowIndices_Y, typename RowPointers_Y, std::size_t I_idx>
struct ConcatMatrixSetFromDiagRow {
  static void
  compute(CompiledSparseMatrix<T, Y_Col, Y_Row, RowIndices_Y, RowPointers_Y> &Y,
          const DiagMatrix<T, M> &B) {

    Base::Matrix::set_sparse_matrix_value<(I_idx + Column_Offset),
                                          (I_idx + Row_Offset)>(Y, B[I_idx]);

    ConcatMatrixSetFromDiagRow<T, M, Y_Col, Y_Row, Column_Offset, Row_Offset,
                               RowIndices_Y, RowPointers_Y,
                               I_idx - 1>::compute(Y, B);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t Y_Col, std::size_t Y_Row,
          std::size_t Column_Offset, std::size_t Row_Offset,
          typename RowIndices_Y, typename RowPointers_Y>
struct ConcatMatrixSetFromDiagRow<T, M, Y_Col, Y_Row, Column_Offset, Row_Offset,
                                  RowIndices_Y, RowPointers_Y, 0> {
  static void
  compute(CompiledSparseMatrix<T, Y_Col, Y_Row, RowIndices_Y, RowPointers_Y> &Y,
          const DiagMatrix<T, M> &B) {

    Base::Matrix::set_sparse_matrix_value<Column_Offset, Row_Offset>(Y, B[0]);
  }
};

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N> struct DenseAndDiag {

  using SparseAvailable_A = DenseAvailable<M, N>;

  using SparseAvailable_B = DiagAvailable<M>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_Y>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (M + N), RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateHorizontally

template <typename T, std::size_t M, std::size_t N>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::DenseAndDiag<T, M, N>::Y_Type &Y,
    const Matrix<T, M, N> &A, const DiagMatrix<T, M> &B) {

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < N) {

        Y.values[value_count] = A(i, j);
        value_count++;

      } else if ((j - N) == i) {

        Y.values[value_count] = B[i];
        value_count++;

      } else {
        /* Do nothing */
      }
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  using RowIndices_Y =
      typename ConcatenateHorizontally::DenseAndDiag<T, M, N>::RowIndices_Y;

  using RowPointers_Y =
      typename ConcatenateHorizontally::DenseAndDiag<T, M, N>::RowPointers_Y;

  ConcatMatrixSetFromDenseRow<T, M, N, M, (M + N), 0, 0, RowIndices_Y,
                              RowPointers_Y, M - 1>::compute(Y, A);

  ConcatMatrixSetFromDiagRow<T, M, M, (M + N), 0, N, RowIndices_Y,
                             RowPointers_Y, M - 1>::compute(Y, B);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_horizontally(const Matrix<T, M, N> &A,
                                     const DiagMatrix<T, M> &B) ->
    typename ConcatenateHorizontally::DenseAndDiag<T, M, N>::Y_Type {

  typename ConcatenateHorizontally::DenseAndDiag<T, M, N>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

/* Copy SparseMatrix to horizontally concatenated matrix */
// when J_idx < N
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t Y_Col, std::size_t Y_Row,
          std::size_t Column_Offset, std::size_t Row_Offset,
          typename RowIndices_Y, typename RowPointers_Y, std::size_t I,
          std::size_t J_idx>
struct ConcatMatrixSetFromSparseColumn {
  static void
  compute(CompiledSparseMatrix<T, Y_Col, Y_Row, RowIndices_Y, RowPointers_Y> &Y,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {

    Base::Matrix::set_sparse_matrix_value<(I + Column_Offset),
                                          (J_idx + Row_Offset)>(
        Y, Base::Matrix::get_sparse_matrix_value<I, J_idx>(A));

    ConcatMatrixSetFromSparseColumn<
        T, M, N, RowIndices_A, RowPointers_A, Y_Col, Y_Row, Column_Offset,
        Row_Offset, RowIndices_Y, RowPointers_Y, I, J_idx - 1>::compute(Y, A);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t Y_Col, std::size_t Y_Row,
          std::size_t Column_Offset, std::size_t Row_Offset,
          typename RowIndices_Y, typename RowPointers_Y, std::size_t I>
struct ConcatMatrixSetFromSparseColumn<T, M, N, RowIndices_A, RowPointers_A,
                                       Y_Col, Y_Row, Column_Offset, Row_Offset,
                                       RowIndices_Y, RowPointers_Y, I, 0> {
  static void
  compute(CompiledSparseMatrix<T, Y_Col, Y_Row, RowIndices_Y, RowPointers_Y> &Y,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {

    Base::Matrix::set_sparse_matrix_value<(I + Column_Offset), Row_Offset>(
        Y, Base::Matrix::get_sparse_matrix_value<I, 0>(A));
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t Y_Col, std::size_t Y_Row,
          std::size_t Column_Offset, std::size_t Row_Offset,
          typename RowIndices_Y, typename RowPointers_Y, std::size_t I_idx>
struct ConcatMatrixSetFromSparseRow {
  static void
  compute(CompiledSparseMatrix<T, Y_Col, Y_Row, RowIndices_Y, RowPointers_Y> &Y,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
    ConcatMatrixSetFromSparseColumn<
        T, M, N, RowIndices_A, RowPointers_A, Y_Col, Y_Row, Column_Offset,
        Row_Offset, RowIndices_Y, RowPointers_Y, I_idx, N - 1>::compute(Y, A);
    ConcatMatrixSetFromSparseRow<T, M, N, RowIndices_A, RowPointers_A, Y_Col,
                                 Y_Row, Column_Offset, Row_Offset, RowIndices_Y,
                                 RowPointers_Y, I_idx - 1>::compute(Y, A);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t Y_Col, std::size_t Y_Row,
          std::size_t Column_Offset, std::size_t Row_Offset,
          typename RowIndices_Y, typename RowPointers_Y>
struct ConcatMatrixSetFromSparseRow<T, M, N, RowIndices_A, RowPointers_A, Y_Col,
                                    Y_Row, Column_Offset, Row_Offset,
                                    RowIndices_Y, RowPointers_Y, 0> {
  static void
  compute(CompiledSparseMatrix<T, Y_Col, Y_Row, RowIndices_Y, RowPointers_Y> &Y,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
    ConcatMatrixSetFromSparseColumn<
        T, M, N, RowIndices_A, RowPointers_A, Y_Col, Y_Row, Column_Offset,
        Row_Offset, RowIndices_Y, RowPointers_Y, 0, N - 1>::compute(Y, A);
  }
};

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_B, typename RowPointers_B>
struct DenseAndSparse {

  using SparseAvailable_A = DenseAvailable<M, N>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<L, RowIndices_B,
                                                  RowPointers_B>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_Y>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (N + L), RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateHorizontally

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_B, typename RowPointers_B>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::DenseAndSparse<T, M, N, L, RowIndices_B,
                                                     RowPointers_B>::Y_Type &Y,
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, M, L, RowIndices_B, RowPointers_B> &B) {

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  using RowIndices_Y = typename ConcatenateHorizontally::DenseAndSparse<
      T, M, N, L, RowIndices_B, RowPointers_B>::RowIndices_Y;

  using RowPointers_Y = typename ConcatenateHorizontally::DenseAndSparse<
      T, M, N, L, RowIndices_B, RowPointers_B>::RowPointers_Y;

  ConcatMatrixSetFromDenseRow<T, M, N, M, (N + L), 0, 0, RowIndices_Y,
                              RowPointers_Y, M - 1>::compute(Y, A);

  ConcatMatrixSetFromSparseRow<T, M, L, RowIndices_B, RowPointers_B, M, (N + L),
                               0, N, RowIndices_Y, RowPointers_Y,
                               M - 1>::compute(Y, B);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_B, typename RowPointers_B>
inline auto concatenate_horizontally(
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, M, L, RowIndices_B, RowPointers_B> &B) ->
    typename ConcatenateHorizontally::DenseAndSparse<T, M, N, L, RowIndices_B,
                                                     RowPointers_B>::Y_Type {

  typename ConcatenateHorizontally::DenseAndSparse<T, M, N, L, RowIndices_B,
                                                   RowPointers_B>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N> struct DiagAndDense {

  using SparseAvailable_A = DiagAvailable<M>;

  using SparseAvailable_B = DenseAvailable<M, N>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_Y>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (M + N), RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateHorizontally

template <typename T, std::size_t M, std::size_t N>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::DiagAndDense<T, M, N>::Y_Type &Y,
    const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B) {

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  using RowIndices_Y =
      typename ConcatenateHorizontally::DiagAndDense<T, M, N>::RowIndices_Y;

  using RowPointers_Y =
      typename ConcatenateHorizontally::DiagAndDense<T, M, N>::RowPointers_Y;

  ConcatMatrixSetFromDiagRow<T, M, M, (M + N), 0, 0, RowIndices_Y,
                             RowPointers_Y, M - 1>::compute(Y, A);

  ConcatMatrixSetFromDenseRow<T, M, N, M, (M + N), 0, M, RowIndices_Y,
                              RowPointers_Y, M - 1>::compute(Y, B);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_horizontally(const DiagMatrix<T, M> &A,
                                     const Matrix<T, M, N> &B) ->
    typename ConcatenateHorizontally::DiagAndDense<T, M, N>::Y_Type {

  typename ConcatenateHorizontally::DiagAndDense<T, M, N>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateHorizontally {

template <typename T, std::size_t M> struct DiagAndDiag {

  using SparseAvailable_A = DiagAvailable<M>;

  using SparseAvailable_B = DiagAvailable<M>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_Y>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (2 * M), RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateHorizontally

template <typename T, std::size_t M>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::DiagAndDiag<T, M>::Y_Type &Y,
    const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B) {

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  using RowIndices_Y =
      typename ConcatenateHorizontally::DiagAndDiag<T, M>::RowIndices_Y;

  using RowPointers_Y =
      typename ConcatenateHorizontally::DiagAndDiag<T, M>::RowPointers_Y;

  ConcatMatrixSetFromDiagRow<T, M, M, (2 * M), 0, 0, RowIndices_Y,
                             RowPointers_Y, M - 1>::compute(Y, A);

  ConcatMatrixSetFromDiagRow<T, M, M, (2 * M), 0, M, RowIndices_Y,
                             RowPointers_Y, M - 1>::compute(Y, B);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

template <typename T, std::size_t M>
inline auto concatenate_horizontally(const DiagMatrix<T, M> &A,
                                     const DiagMatrix<T, M> &B) ->
    typename ConcatenateHorizontally::DiagAndDiag<T, M>::Y_Type {

  typename ConcatenateHorizontally::DiagAndDiag<T, M>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
struct DiagAndSparse {

  using SparseAvailable_A = DiagAvailable<M>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_B,
                                                  RowPointers_B>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_Y>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (M + N), RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateHorizontally

template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::DiagAndSparse<T, M, N, RowIndices_B,
                                                    RowPointers_B>::Y_Type &Y,
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B) {

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  using RowIndices_Y = typename ConcatenateHorizontally::DiagAndSparse<
      T, M, N, RowIndices_B, RowPointers_B>::RowIndices_Y;

  using RowPointers_Y = typename ConcatenateHorizontally::DiagAndSparse<
      T, M, N, RowIndices_B, RowPointers_B>::RowPointers_Y;

  ConcatMatrixSetFromDiagRow<T, M, M, (M + N), 0, 0, RowIndices_Y,
                             RowPointers_Y, M - 1>::compute(Y, A);

  ConcatMatrixSetFromSparseRow<T, M, N, RowIndices_B, RowPointers_B, M, (M + N),
                               0, M, RowIndices_Y, RowPointers_Y,
                               M - 1>::compute(Y, B);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
inline auto concatenate_horizontally(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B) ->
    typename ConcatenateHorizontally::DiagAndSparse<T, M, N, RowIndices_B,
                                                    RowPointers_B>::Y_Type {

  typename ConcatenateHorizontally::DiagAndSparse<T, M, N, RowIndices_B,
                                                  RowPointers_B>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_A, typename RowPointers_A>
struct SparseAndDense {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<L, RowIndices_A,
                                                  RowPointers_A>;

  using SparseAvailable_B = DenseAvailable<M, N>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_Y>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (N + L), RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateHorizontally

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_A, typename RowPointers_A>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::SparseAndDense<T, M, N, L, RowIndices_A,
                                                     RowPointers_A>::Y_Type &Y,
    const CompiledSparseMatrix<T, M, L, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, M, N> &B) {

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  using RowIndices_Y = typename ConcatenateHorizontally::SparseAndDense<
      T, M, N, L, RowIndices_A, RowPointers_A>::RowIndices_Y;

  using RowPointers_Y = typename ConcatenateHorizontally::SparseAndDense<
      T, M, N, L, RowIndices_A, RowPointers_A>::RowPointers_Y;

  ConcatMatrixSetFromSparseRow<T, M, L, RowIndices_A, RowPointers_A, M, (N + L),
                               0, 0, RowIndices_Y, RowPointers_Y,
                               M - 1>::compute(Y, A);

  ConcatMatrixSetFromDenseRow<T, M, N, M, (N + L), 0, N, RowIndices_Y,
                              RowPointers_Y, M - 1>::compute(Y, B);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_A, typename RowPointers_A>
inline auto concatenate_horizontally(
    const CompiledSparseMatrix<T, M, L, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, M, N> &B) ->
    typename ConcatenateHorizontally::SparseAndDense<T, M, N, L, RowIndices_A,
                                                     RowPointers_A>::Y_Type {

  typename ConcatenateHorizontally::SparseAndDense<T, M, N, L, RowIndices_A,
                                                   RowPointers_A>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct SparseAndDiag {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                  RowPointers_A>;

  using SparseAvailable_B = DiagAvailable<M>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_Y>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (M + N), RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateHorizontally

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::SparseAndDiag<T, M, N, RowIndices_A,
                                                    RowPointers_A>::Y_Type &Y,
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const DiagMatrix<T, M> &B) {

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  using RowIndices_Y = typename ConcatenateHorizontally::SparseAndDiag<
      T, M, N, RowIndices_A, RowPointers_A>::RowIndices_Y;

  using RowPointers_Y = typename ConcatenateHorizontally::SparseAndDiag<
      T, M, N, RowIndices_A, RowPointers_A>::RowPointers_Y;

  ConcatMatrixSetFromSparseRow<T, M, N, RowIndices_A, RowPointers_A, M, (M + N),
                               0, 0, RowIndices_Y, RowPointers_Y,
                               M - 1>::compute(Y, A);

  ConcatMatrixSetFromDiagRow<T, M, M, (M + N), 0, N, RowIndices_Y,
                             RowPointers_Y, M - 1>::compute(Y, B);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline auto concatenate_horizontally(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const DiagMatrix<T, M> &B) ->
    typename ConcatenateHorizontally::SparseAndDiag<T, M, N, RowIndices_A,
                                                    RowPointers_A>::Y_Type {

  typename ConcatenateHorizontally::SparseAndDiag<T, M, N, RowIndices_A,
                                                  RowPointers_A>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
struct SparseAndSparse {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_A,
                                                  RowPointers_A>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<N, RowIndices_B,
                                                  RowPointers_B>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using RowIndices_Y = RowIndicesFromSparseAvailable<SparseAvailable_Y>;

  using RowPointers_Y = RowPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (N + L), RowIndices_Y, RowPointers_Y>;
};

} // namespace ConcatenateHorizontally

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::SparseAndSparse<
        T, M, N, L, RowIndices_A, RowPointers_A, RowIndices_B,
        RowPointers_B>::Y_Type &Y,
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, M, L, RowIndices_B, RowPointers_B> &B) {

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  using RowIndices_Y = typename ConcatenateHorizontally::SparseAndSparse<
      T, M, N, L, RowIndices_A, RowPointers_A, RowIndices_B,
      RowPointers_B>::RowIndices_Y;

  using RowPointers_Y = typename ConcatenateHorizontally::SparseAndSparse<
      T, M, N, L, RowIndices_A, RowPointers_A, RowIndices_B,
      RowPointers_B>::RowPointers_Y;

  ConcatMatrixSetFromSparseRow<T, M, N, RowIndices_A, RowPointers_A, M, (N + L),
                               0, 0, RowIndices_Y, RowPointers_Y,
                               M - 1>::compute(Y, A);

  ConcatMatrixSetFromSparseRow<T, M, L, RowIndices_B, RowPointers_B, M, (N + L),
                               0, N, RowIndices_Y, RowPointers_Y,
                               M - 1>::compute(Y, B);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
inline auto concatenate_horizontally(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, M, L, RowIndices_B, RowPointers_B> &B) ->
    typename ConcatenateHorizontally::SparseAndSparse<
        T, M, N, L, RowIndices_A, RowPointers_A, RowIndices_B,
        RowPointers_B>::Y_Type {

  typename ConcatenateHorizontally::SparseAndSparse<T, M, N, L, RowIndices_A,
                                                    RowPointers_A, RowIndices_B,
                                                    RowPointers_B>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_CONCATENATE_HPP__
