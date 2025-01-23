#ifndef __PYTHON_NUMPY_CONCATENATE_HPP__
#define __PYTHON_NUMPY_CONCATENATE_HPP__

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>

namespace PythonNumpy {

/* Concatenate vertically */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline void
update_vertically_concatenated_matrix(Matrix<DefDense, T, (M + P), N> &Y,
                                      const Matrix<DefDense, T, M, N> &A,
                                      const Matrix<DefDense, T, P, N> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline auto concatenate_vertically(const Matrix<DefDense, T, M, N> &A,
                                   const Matrix<DefDense, T, P, N> &B)
    -> Matrix<DefDense, T, (M + P), N> {

  return Matrix<DefDense, T, (M + P), N>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + N), N,
           ConcatenateSparseAvailableVertically<DenseAvailable<M, N>,
                                                DiagAvailable<N>>> &Y,
    const Matrix<DefDense, T, M, N> &A, const Matrix<DefDiag, T, N> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_vertically(const Matrix<DefDense, T, M, N> &A,
                                   const Matrix<DefDiag, T, N> &B)
    -> Matrix<DefSparse, T, (M + N), N,
              ConcatenateSparseAvailableVertically<DenseAvailable<M, N>,
                                                   DiagAvailable<N>>> {

  return Matrix<DefSparse, T, (M + N), N,
                ConcatenateSparseAvailableVertically<DenseAvailable<M, N>,
                                                     DiagAvailable<N>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename SparseAvailable_B>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + P), N,
           ConcatenateSparseAvailableVertically<DenseAvailable<M, N>,
                                                SparseAvailable_B>> &Y,
    const Matrix<DefDense, T, M, N> &A,
    const Matrix<DefSparse, T, P, N, SparseAvailable_B> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename SparseAvailable_B>
inline auto
concatenate_vertically(const Matrix<DefDense, T, M, N> &A,
                       const Matrix<DefSparse, T, P, N, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, (M + P), N,
              ConcatenateSparseAvailableVertically<DenseAvailable<M, N>,
                                                   SparseAvailable_B>> {

  return Matrix<DefSparse, T, (M + P), N,
                ConcatenateSparseAvailableVertically<DenseAvailable<M, N>,
                                                     SparseAvailable_B>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t P>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + P), M,
           ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                DenseAvailable<P, M>>> &Y,
    const Matrix<DefDiag, T, M> &A, const Matrix<DefDense, T, P, M> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

template <typename T, std::size_t M, std::size_t P>
inline auto concatenate_vertically(const Matrix<DefDiag, T, M> &A,
                                   const Matrix<DefDense, T, P, M> &B)
    -> Matrix<DefSparse, T, (M + P), M,
              ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                   DenseAvailable<P, M>>> {

  return Matrix<DefSparse, T, (M + P), M,
                ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                     DenseAvailable<P, M>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (2 * M), M,
           ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                DiagAvailable<M>>> &Y,
    const Matrix<DefDiag, T, M> &A, const Matrix<DefDiag, T, M> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

template <typename T, std::size_t M>
inline auto concatenate_vertically(const Matrix<DefDiag, T, M> &A,
                                   const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, (2 * M), M,
              ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                   DiagAvailable<M>>> {

  return Matrix<
      DefSparse, T, (2 * M), M,
      ConcatenateSparseAvailableVertically<DiagAvailable<M>, DiagAvailable<M>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t P, typename SparseAvailable_B>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + P), M,
           ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                SparseAvailable_B>> &Y,
    const Matrix<DefDiag, T, M> &A,
    const Matrix<DefSparse, T, P, M, SparseAvailable_B> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

template <typename T, std::size_t M, std::size_t P, typename SparseAvailable_B>
inline auto
concatenate_vertically(const Matrix<DefDiag, T, M> &A,
                       const Matrix<DefSparse, T, P, M, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, (M + P), M,
              ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                   SparseAvailable_B>> {

  return Matrix<DefSparse, T, (M + P), M,
                ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                     SparseAvailable_B>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A,
          std::size_t P>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + P), N,
           ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                DenseAvailable<P, N>>> &Y,
    const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
    const Matrix<DefDense, T, P, N> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A,
          std::size_t P>
inline auto
concatenate_vertically(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                       const Matrix<DefDense, T, P, N> &B)
    -> Matrix<DefSparse, T, (M + P), N,
              ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                   DenseAvailable<P, N>>> {

  return Matrix<DefSparse, T, (M + P), N,
                ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                     DenseAvailable<P, N>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + N), N,
           ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                DiagAvailable<N>>> &Y,
    const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
    const Matrix<DefDiag, T, N> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A>
inline auto
concatenate_vertically(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                       const Matrix<DefDiag, T, N> &B)
    -> Matrix<DefSparse, T, (M + N), N,
              ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                   DiagAvailable<N>>> {

  return Matrix<DefSparse, T, (M + N), N,
                ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                     DiagAvailable<N>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + P), N,
           ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                SparseAvailable_B>> &Y,
    const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
    const Matrix<DefSparse, T, P, N, SparseAvailable_B> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline auto
concatenate_vertically(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                       const Matrix<DefSparse, T, P, N, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, (M + P), N,
              ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                   SparseAvailable_B>> {

  return Matrix<DefSparse, T, (M + P), N,
                ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                     SparseAvailable_B>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

/* Concatenate horizontally */
template <typename T, std::size_t M, std::size_t N, std::size_t L>
inline void
update_horizontally_concatenated_matrix(Matrix<DefDense, T, M, (N + L)> &Y,
                                        const Matrix<DefDense, T, M, N> &A,
                                        const Matrix<DefDense, T, M, L> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

template <typename T, std::size_t M, std::size_t N, std::size_t L>
inline auto concatenate_horizontally(const Matrix<DefDense, T, M, N> &A,
                                     const Matrix<DefDense, T, M, L> &B)
    -> Matrix<DefDense, T, M, (N + L)> {

  return Matrix<DefDense, T, M, (N + L)>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (M + N),
           ConcatenateSparseAvailableHorizontally<DenseAvailable<M, N>,
                                                  DiagAvailable<M>>> &Y,
    const Matrix<DefDense, T, M, N> &A, const Matrix<DefDiag, T, M> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_horizontally(const Matrix<DefDense, T, M, N> &A,
                                     const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, M, (M + N),
              ConcatenateSparseAvailableHorizontally<DenseAvailable<M, N>,
                                                     DiagAvailable<M>>> {

  return Matrix<DefSparse, T, M, (M + N),
                ConcatenateSparseAvailableHorizontally<DenseAvailable<M, N>,
                                                       DiagAvailable<M>>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_B>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (N + L),
           ConcatenateSparseAvailableHorizontally<DenseAvailable<M, N>,
                                                  SparseAvailable_B>> &Y,
    const Matrix<DefDense, T, M, N> &A,
    const Matrix<DefSparse, T, M, L, SparseAvailable_B> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_B>
inline auto
concatenate_horizontally(const Matrix<DefDense, T, M, N> &A,
                         const Matrix<DefSparse, T, M, L, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, M, (N + L),
              ConcatenateSparseAvailableHorizontally<DenseAvailable<M, N>,
                                                     SparseAvailable_B>> {

  return Matrix<DefSparse, T, M, (N + L),
                ConcatenateSparseAvailableHorizontally<DenseAvailable<M, N>,
                                                       SparseAvailable_B>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (M + N),
           ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                  DenseAvailable<M, N>>> &Y,
    const Matrix<DefDiag, T, M> &A, const Matrix<DefDense, T, M, N> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_horizontally(const Matrix<DefDiag, T, M> &A,
                                     const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefSparse, T, M, (M + N),
              ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                     DenseAvailable<M, N>>> {

  return Matrix<DefSparse, T, M, (M + N),
                ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                       DenseAvailable<M, N>>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (2 * M),
           ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                  DiagAvailable<M>>> &Y,
    const Matrix<DefDiag, T, M> &A, const Matrix<DefDiag, T, M> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

template <typename T, std::size_t M>
inline auto concatenate_horizontally(const Matrix<DefDiag, T, M> &A,
                                     const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, M, (2 * M),
              ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                     DiagAvailable<M>>> {

  return Matrix<DefSparse, T, M, (2 * M),
                ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                       DiagAvailable<M>>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_B>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (M + N),
           ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                  SparseAvailable_B>> &Y,
    const Matrix<DefDiag, T, M> &A,
    const Matrix<DefSparse, T, M, N, SparseAvailable_B> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_B>
inline auto
concatenate_horizontally(const Matrix<DefDiag, T, M> &A,
                         const Matrix<DefSparse, T, M, N, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, M, (M + N),
              ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                     SparseAvailable_B>> {

  return Matrix<DefSparse, T, M, (M + N),
                ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                       SparseAvailable_B>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_A>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (N + L),
           ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                  DenseAvailable<M, N>>> &Y,
    const Matrix<DefSparse, T, M, L, SparseAvailable_A> &A,
    const Matrix<DefDense, T, M, N> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_A>
inline auto
concatenate_horizontally(const Matrix<DefSparse, T, M, L, SparseAvailable_A> &A,
                         const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefSparse, T, M, (N + L),
              ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                     DenseAvailable<M, N>>> {

  return Matrix<DefSparse, T, M, (N + L),
                ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                       DenseAvailable<M, N>>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (M + N),
           ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                  DiagAvailable<M>>> &Y,
    const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
    const Matrix<DefDiag, T, M> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A>
inline auto
concatenate_horizontally(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                         const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, M, (M + N),
              ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                     DiagAvailable<M>>> {

  return Matrix<DefSparse, T, M, (M + N),
                ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                       DiagAvailable<M>>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (N + L),
           ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                  SparseAvailable_B>> &Y,
    const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
    const Matrix<DefSparse, T, M, L, SparseAvailable_B> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline auto
concatenate_horizontally(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                         const Matrix<DefSparse, T, M, L, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, M, (N + L),
              ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                     SparseAvailable_B>> {

  return Matrix<DefSparse, T, M, (N + L),
                ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                       SparseAvailable_B>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_CONCATENATE_HPP__
