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
    -> Matrix<DefSparse, T, (M + N), N, ((M + 1) * N)> {

  return Matrix<DefSparse, T, (M + N), N, ((M + 1) * N)>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t V>
auto concatenate_vertically(const Matrix<DefDense, T, M, N> &A,
                            const Matrix<DefSparse, T, P, N, V> &B)
    -> Matrix<DefSparse, T, (M + P), N, ((M * N) + V)> {

  /* Result */
  return Matrix<DefSparse, T, (M + P), N, ((M * N) + V)>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t P>
auto concatenate_vertically(const Matrix<DefDiag, T, M> &A,
                            const Matrix<DefDense, T, P, M> &B)
    -> Matrix<DefSparse, T, (M + P), M, ((P + 1) * M)> {

  /* Result */
  return Matrix<DefSparse, T, (M + P), M, ((P + 1) * M)>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M>
auto concatenate_vertically(const Matrix<DefDiag, T, M> &A,
                            const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, (2 * M), M, (2 * M)> {

  /* Result */
  return Matrix<DefSparse, T, (2 * M), M, (2 * M)>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t P, std::size_t V>
auto concatenate_vertically(const Matrix<DefDiag, T, M> &A,
                            const Matrix<DefSparse, T, P, M, V> &B)
    -> Matrix<DefSparse, T, (M + P), M, (M + V)> {

  /* Result */
  return Matrix<DefSparse, T, (M + P), M, (M + V)>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t V, std::size_t W>
auto concatenate_vertically(const Matrix<DefSparse, T, M, N, V> &A,
                            const Matrix<DefSparse, T, P, N, W> &B)
    -> Matrix<DefSparse, T, (M + P), N, (V + W)> {

  /* Result */
  return Matrix<DefSparse, T, (M + P), N, (V + W)>(
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
    -> Matrix<DefSparse, T, M, (M + N), ((N + 1) * M)> {

  return Matrix<DefSparse, T, M, (M + N), ((N + 1) * M)>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          std::size_t V>
auto concatenate_horizontally(const Matrix<DefDense, T, M, N> &A,
                              const Matrix<DefSparse, T, M, L, V> &B)
    -> Matrix<DefSparse, T, M, (N + L), ((M * N) + V)> {

  return Matrix<DefSparse, T, M, (N + L), ((M * N) + V)>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
auto concatenate_horizontally(const Matrix<DefDiag, T, M> &A,
                              const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefSparse, T, M, (M + N), ((N + 1) * M)> {

  return Matrix<DefSparse, T, M, (M + N), ((N + 1) * M)>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M>
auto concatenate_horizontally(const Matrix<DefDiag, T, M> &A,
                              const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, M, (2 * M), (2 * M)> {

  return Matrix<DefSparse, T, M, (2 * M), (2 * M)>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto concatenate_horizontally(const Matrix<DefDiag, T, M> &A,
                              const Matrix<DefSparse, T, M, N, V> &B)
    -> Matrix<DefSparse, T, M, (M + N), (M + V)> {

  return Matrix<DefSparse, T, M, (M + N), (M + V)>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          std::size_t V, std::size_t W>
auto concatenate_horizontally(const Matrix<DefSparse, T, M, N, V> &A,
                              const Matrix<DefSparse, T, M, L, W> &B)
    -> Matrix<DefSparse, T, M, (N + L), (V + W)> {

  return Matrix<DefSparse, T, M, (N + L), (V + W)>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_CONCATENATE_HPP
