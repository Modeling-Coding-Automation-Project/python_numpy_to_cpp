#ifndef PYTHON_NUMPY_CONCATENATE_HPP
#define PYTHON_NUMPY_CONCATENATE_HPP

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include <cstddef>

namespace PythonNumpy {

/* Matrix Concatenate */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
auto concatenate_vertically(const Matrix<DefDense, T, M, N> &A,
                            const Matrix<DefDense, T, P, N> &B)
    -> Matrix<DefDense, T, (M + P), N> {

  return Matrix<DefDense, T, (M + P), N>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
auto concatenate_vertically(const Matrix<DefDense, T, M, N> &A,
                            const Matrix<DefDiag, T, N> &B)
    -> Matrix<
        DefSparse, T, (M + N), N,
        CreateSparseAvailableFromIndicesAndPointers<
            N,
            RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
                DenseAvailable<M, N>, DiagAvailable<N>>>,
            RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
                DenseAvailable<M, N>, DiagAvailable<N>>>>> {

  return Matrix<
      DefSparse, T, (M + N), N,
      CreateSparseAvailableFromIndicesAndPointers<
          N,
          RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
              DenseAvailable<M, N>, DiagAvailable<N>>>,
          RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
              DenseAvailable<M, N>, DiagAvailable<N>>>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename SparseAvailable_B>
auto concatenate_vertically(
    const Matrix<DefDense, T, M, N> &A,
    const Matrix<DefSparse, T, P, N, SparseAvailable_B> &B)
    -> Matrix<
        DefSparse, T, (M + P), N,
        CreateSparseAvailableFromIndicesAndPointers<
            N,
            RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
                DenseAvailable<M, N>, SparseAvailable_B>>,
            RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
                DenseAvailable<M, N>, SparseAvailable_B>>>> {

  /* Result */
  return Matrix<
      DefSparse, T, (M + P), N,
      CreateSparseAvailableFromIndicesAndPointers<
          N,
          RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
              DenseAvailable<M, N>, SparseAvailable_B>>,
          RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
              DenseAvailable<M, N>, SparseAvailable_B>>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t P>
auto concatenate_vertically(const Matrix<DefDiag, T, M> &A,
                            const Matrix<DefDense, T, P, M> &B)
    -> Matrix<
        DefSparse, T, (M + P), M,
        CreateSparseAvailableFromIndicesAndPointers<
            M,
            RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
                DiagAvailable<M>, DenseAvailable<P, M>>>,
            RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
                DiagAvailable<M>, DenseAvailable<P, M>>>>> {

  /* Result */
  return Matrix<
      DefSparse, T, (M + P), M,
      CreateSparseAvailableFromIndicesAndPointers<
          M,
          RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
              DiagAvailable<M>, DenseAvailable<P, M>>>,
          RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
              DiagAvailable<M>, DenseAvailable<P, M>>>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M>
auto concatenate_vertically(const Matrix<DefDiag, T, M> &A,
                            const Matrix<DefDiag, T, M> &B)
    -> Matrix<
        DefSparse, T, (2 * M), M,
        CreateSparseAvailableFromIndicesAndPointers<
            M,
            RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
                DiagAvailable<M>, DiagAvailable<M>>>,
            RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
                DiagAvailable<M>, DiagAvailable<M>>>>> {

  /* Result */
  return Matrix<
      DefSparse, T, (2 * M), M,
      CreateSparseAvailableFromIndicesAndPointers<
          M,
          RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
              DiagAvailable<M>, DiagAvailable<M>>>,
          RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
              DiagAvailable<M>, DiagAvailable<M>>>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t P, typename SparseAvailable_B>
auto concatenate_vertically(
    const Matrix<DefDiag, T, M> &A,
    const Matrix<DefSparse, T, P, M, SparseAvailable_B> &B)
    -> Matrix<
        DefSparse, T, (M + P), M,
        CreateSparseAvailableFromIndicesAndPointers<
            M,
            RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
                DiagAvailable<M>, SparseAvailable_B>>,
            RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
                DiagAvailable<M>, SparseAvailable_B>>>> {

  /* Result */
  return Matrix<
      DefSparse, T, (M + P), M,
      CreateSparseAvailableFromIndicesAndPointers<
          M,
          RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
              DiagAvailable<M>, SparseAvailable_B>>,
          RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
              DiagAvailable<M>, SparseAvailable_B>>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename SparseAvailable_A, typename SparseAvailable_B>
auto concatenate_vertically(
    const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
    const Matrix<DefSparse, T, P, N, SparseAvailable_B> &B)
    -> Matrix<
        DefSparse, T, (M + P), N,
        CreateSparseAvailableFromIndicesAndPointers<
            M,
            RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
                SparseAvailable_A, SparseAvailable_B>>,
            RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
                SparseAvailable_A, SparseAvailable_B>>>> {

  /* Result */
  return Matrix<
      DefSparse, T, (M + P), N,
      CreateSparseAvailableFromIndicesAndPointers<
          M,
          RowIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
              SparseAvailable_A, SparseAvailable_B>>,
          RowPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
              SparseAvailable_A, SparseAvailable_B>>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t L>
auto concatenate_horizontally(const Matrix<DefDense, T, M, N> &A,
                              const Matrix<DefDense, T, M, L> &B)
    -> Matrix<DefDense, T, M, (N + L)> {

  return Matrix<DefDense, T, M, (N + L)>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
auto concatenate_horizontally(const Matrix<DefDense, T, M, N> &A,
                              const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, M, (M + N), SparseAvailable_NoUse> {

  return Matrix<DefSparse, T, M, (M + N), SparseAvailable_NoUse>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_B>
auto concatenate_horizontally(
    const Matrix<DefDense, T, M, N> &A,
    const Matrix<DefSparse, T, M, L, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, M, (N + L), SparseAvailable_NoUse> {

  return Matrix<DefSparse, T, M, (N + L), SparseAvailable_NoUse>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
auto concatenate_horizontally(const Matrix<DefDiag, T, M> &A,
                              const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefSparse, T, M, (M + N), SparseAvailable_NoUse> {

  return Matrix<DefSparse, T, M, (M + N), SparseAvailable_NoUse>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M>
auto concatenate_horizontally(const Matrix<DefDiag, T, M> &A,
                              const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, M, (2 * M), SparseAvailable_NoUse> {

  return Matrix<DefSparse, T, M, (2 * M), SparseAvailable_NoUse>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_B>
auto concatenate_horizontally(
    const Matrix<DefDiag, T, M> &A,
    const Matrix<DefSparse, T, M, N, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, M, (M + N), SparseAvailable_NoUse> {

  return Matrix<DefSparse, T, M, (M + N), SparseAvailable_NoUse>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_A, typename SparseAvailable_B>
auto concatenate_horizontally(
    const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
    const Matrix<DefSparse, T, M, L, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, M, (N + L), SparseAvailable_NoUse> {

  return Matrix<DefSparse, T, M, (N + L), SparseAvailable_NoUse>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_CONCATENATE_HPP
