/**
 * @file python_numpy_convolve.hpp
 * @brief Provides one-dimensional convolution functions similar to
 * numpy.convolve.
 *
 * This header defines full, same, and valid convolution functions in the
 * PythonNumpy namespace. Each function accepts dense and sparse one-dimensional
 * matrices in any supported combination, delegates computation to the
 * corresponding Base::Matrix function, and returns a dense matrix.
 *
 * @note
 * tparam M is the number of rows in the matrix.
 * tparam N is the number of columns in the matrix.
 */
#ifndef PYTHON_NUMPY_CONVOLVE_HPP_
#define PYTHON_NUMPY_CONVOLVE_HPP_

#include "base_matrix_convolve.hpp"
#include "python_numpy_base.hpp"

namespace PythonNumpy {

/* (Dense Matrix) * (Dense Matrix) */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Q>
inline auto convolve_full(const Matrix<DefDense, T, M, N> &a,
                          const Matrix<DefDense, T, P, Q> &b)
    -> Matrix<DefDense, T, (M == 1 ? 1 : M + P - 1), (M == 1 ? N + Q - 1 : 1)> {
  return Matrix<DefDense, T, (M == 1 ? 1 : M + P - 1),
                (M == 1 ? N + Q - 1 : 1)>(
      Base::Matrix::convolve_full(a.matrix, b.matrix));
}

/* (Dense Matrix) * (Sparse Matrix) */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Q, typename SparseAvailableB>
inline auto convolve_full(const Matrix<DefDense, T, M, N> &a,
                          const Matrix<DefSparse, T, P, Q, SparseAvailableB> &b)
    -> Matrix<DefDense, T, (M == 1 ? 1 : M + P - 1), (M == 1 ? N + Q - 1 : 1)> {
  return Matrix<DefDense, T, (M == 1 ? 1 : M + P - 1),
                (M == 1 ? N + Q - 1 : 1)>(
      Base::Matrix::convolve_full(a.matrix, b.matrix));
}

/* (Sparse Matrix) * (Dense Matrix) */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailableA,
          std::size_t P, std::size_t Q>
inline auto convolve_full(const Matrix<DefSparse, T, M, N, SparseAvailableA> &a,
                          const Matrix<DefDense, T, P, Q> &b)
    -> Matrix<DefDense, T, (M == 1 ? 1 : M + P - 1), (M == 1 ? N + Q - 1 : 1)> {
  return Matrix<DefDense, T, (M == 1 ? 1 : M + P - 1),
                (M == 1 ? N + Q - 1 : 1)>(
      Base::Matrix::convolve_full(a.matrix, b.matrix));
}

/* (Sparse Matrix) * (Sparse Matrix) */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailableA,
          std::size_t P, std::size_t Q, typename SparseAvailableB>
inline auto convolve_full(const Matrix<DefSparse, T, M, N, SparseAvailableA> &a,
                          const Matrix<DefSparse, T, P, Q, SparseAvailableB> &b)
    -> Matrix<DefDense, T, (M == 1 ? 1 : M + P - 1), (M == 1 ? N + Q - 1 : 1)> {
  return Matrix<DefDense, T, (M == 1 ? 1 : M + P - 1),
                (M == 1 ? N + Q - 1 : 1)>(
      Base::Matrix::convolve_full(a.matrix, b.matrix));
}

/* (Dense Matrix) * (Dense Matrix) */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Q>
inline auto convolve_same(const Matrix<DefDense, T, M, N> &a,
                          const Matrix<DefDense, T, P, Q> &b)
    -> Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M : P)),
              (M == 1 ? (N > Q ? N : Q) : 1)> {
  return Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M : P)),
                (M == 1 ? (N > Q ? N : Q) : 1)>(
      Base::Matrix::convolve_same(a.matrix, b.matrix));
}

/* (Dense Matrix) * (Sparse Matrix) */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Q, typename SparseAvailableB>
inline auto convolve_same(const Matrix<DefDense, T, M, N> &a,
                          const Matrix<DefSparse, T, P, Q, SparseAvailableB> &b)
    -> Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M : P)),
              (M == 1 ? (N > Q ? N : Q) : 1)> {
  return Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M : P)),
                (M == 1 ? (N > Q ? N : Q) : 1)>(
      Base::Matrix::convolve_same(a.matrix, b.matrix));
}

/* (Sparse Matrix) * (Dense Matrix) */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailableA,
          std::size_t P, std::size_t Q>
inline auto convolve_same(const Matrix<DefSparse, T, M, N, SparseAvailableA> &a,
                          const Matrix<DefDense, T, P, Q> &b)
    -> Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M : P)),
              (M == 1 ? (N > Q ? N : Q) : 1)> {
  return Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M : P)),
                (M == 1 ? (N > Q ? N : Q) : 1)>(
      Base::Matrix::convolve_same(a.matrix, b.matrix));
}

/* (Sparse Matrix) * (Sparse Matrix) */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailableA,
          std::size_t P, std::size_t Q, typename SparseAvailableB>
inline auto convolve_same(const Matrix<DefSparse, T, M, N, SparseAvailableA> &a,
                          const Matrix<DefSparse, T, P, Q, SparseAvailableB> &b)
    -> Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M : P)),
              (M == 1 ? (N > Q ? N : Q) : 1)> {
  return Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M : P)),
                (M == 1 ? (N > Q ? N : Q) : 1)>(
      Base::Matrix::convolve_same(a.matrix, b.matrix));
}

/* (Dense Matrix) * (Dense Matrix) */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Q>
inline auto convolve_valid(const Matrix<DefDense, T, M, N> &a,
                           const Matrix<DefDense, T, P, Q> &b)
    -> Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M - P + 1 : P - M + 1)),
              (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1) : 1)> {
  return Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M - P + 1 : P - M + 1)),
                (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1) : 1)>(
      Base::Matrix::convolve_valid(a.matrix, b.matrix));
}

/* (Dense Matrix) * (Sparse Matrix) */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Q, typename SparseAvailableB>
inline auto
convolve_valid(const Matrix<DefDense, T, M, N> &a,
               const Matrix<DefSparse, T, P, Q, SparseAvailableB> &b)
    -> Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M - P + 1 : P - M + 1)),
              (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1) : 1)> {
  return Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M - P + 1 : P - M + 1)),
                (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1) : 1)>(
      Base::Matrix::convolve_valid(a.matrix, b.matrix));
}

/* (Sparse Matrix) * (Dense Matrix) */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailableA,
          std::size_t P, std::size_t Q>
inline auto
convolve_valid(const Matrix<DefSparse, T, M, N, SparseAvailableA> &a,
               const Matrix<DefDense, T, P, Q> &b)
    -> Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M - P + 1 : P - M + 1)),
              (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1) : 1)> {
  return Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M - P + 1 : P - M + 1)),
                (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1) : 1)>(
      Base::Matrix::convolve_valid(a.matrix, b.matrix));
}

/* (Sparse Matrix) * (Sparse Matrix) */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailableA,
          std::size_t P, std::size_t Q, typename SparseAvailableB>
inline auto
convolve_valid(const Matrix<DefSparse, T, M, N, SparseAvailableA> &a,
               const Matrix<DefSparse, T, P, Q, SparseAvailableB> &b)
    -> Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M - P + 1 : P - M + 1)),
              (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1) : 1)> {
  return Matrix<DefDense, T, (M == 1 ? 1 : (M > P ? M - P + 1 : P - M + 1)),
                (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1) : 1)>(
      Base::Matrix::convolve_valid(a.matrix, b.matrix));
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_CONVOLVE_HPP_
