#ifndef PYTHON_NUMPY_TRANSPOSE_OPERATION_HPP
#define PYTHON_NUMPY_TRANSPOSE_OPERATION_HPP

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include <cstddef>

namespace PythonNumpy {

/* (matrix) * (transposed matrix) */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
auto A_mul_BT(const Matrix<DefDense, T, M, K> &A,
              const Matrix<DefDense, T, N, K> &B) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      Base::Matrix::matrix_multiply_A_mul_BTranspose(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t K>
auto A_mul_BT(const Matrix<DefDense, T, M, K> &A,
              const Matrix<DefDiag, T, K> &B) -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(A.matrix * B.matrix);
}

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t V>
auto A_mul_BT(const Matrix<DefDense, T, M, K> &A,
              const Matrix<DefSparse, T, N, K, V> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      Base::Matrix::matrix_multiply_A_mul_SparseBT(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
auto A_mul_BT(const Matrix<DefDiag, T, M> &A,
              const Matrix<DefDense, T, N, M> &B) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(A.matrix * B.matrix.transpose());
}

template <typename T, std::size_t M>
auto A_mul_BT(const Matrix<DefDiag, T, M> &A, const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(A.matrix * B.matrix);
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto A_mul_BT(const Matrix<DefDiag, T, M> &A,
              const Matrix<DefSparse, T, N, M, V> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(A.matrix * B.matrix.transpose());
}

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t V>
auto A_mul_BT(const Matrix<DefSparse, T, M, K, V> &A,
              const Matrix<DefDense, T, N, K> &B) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      Base::Matrix::matrix_multiply_SparseA_mul_BTranspose(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t K, std::size_t V>
auto A_mul_BT(const Matrix<DefSparse, T, M, K, V> &A,
              const Matrix<DefDiag, T, K> &B) -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(A.matrix * B.matrix);
}

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t V, std::size_t W>
auto A_mul_BT(const Matrix<DefSparse, T, M, K, V> &A,
              const Matrix<DefSparse, T, N, K, W> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      matrix_multiply_SparseA_mul_SparseBTranspose(A.matrix, B.matrix));
}

/* (transpose matrix) * (matrix) */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
auto AT_mul_B(const Matrix<DefDense, T, N, M> &A,
              const Matrix<DefDense, T, N, K> &B) -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(
      Base::Matrix::matrix_multiply_AT_mul_B(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
auto AT_mul_B(const Matrix<DefDense, T, N, M> &A,
              const Matrix<DefDiag, T, N> &B) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(A.matrix.transpose() * B.matrix);
}

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t V>
auto AT_mul_B(const Matrix<DefDense, T, N, M> &A,
              const Matrix<DefSparse, T, N, K, V> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(
      Base::Matrix::matrix_multiply_AT_mul_SparseB(A.matrix, B.matrix));
}

template <typename T, std::size_t K, std::size_t N>
auto AT_mul_B(const Matrix<DefDiag, T, N> &A,
              const Matrix<DefDense, T, N, K> &B) -> Matrix<DefDense, T, N, K> {

  return Matrix<DefDense, T, N, K>(A.matrix * B.matrix);
}

template <typename T, std::size_t M>
auto AT_mul_B(const Matrix<DefDiag, T, M> &A, const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(A.matrix * B.matrix);
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto AT_mul_B(const Matrix<DefDiag, T, M> &A,
              const Matrix<DefSparse, T, N, M, V> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(A.matrix * B.matrix);
}

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t V>
auto AT_mul_B(const Matrix<DefSparse, T, N, M, V> &A,
              const Matrix<DefDense, T, N, K> &B) -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(
      Base::Matrix::matrix_multiply_SparseAT_mul_B(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto AT_mul_B(const Matrix<DefSparse, T, N, M, V> &A,
              const Matrix<DefDiag, T, N> &B) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      matrix_multiply_T_DiagA_mul_SparseB(B.matrix, A.matrix));
}

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t V, std::size_t W>
auto AT_mul_B(const Matrix<DefSparse, T, N, M, V> &A,
              const Matrix<DefSparse, T, N, K, W> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(
      matrix_multiply_SparseATranspose_mul_SparseB(A.matrix, B.matrix));
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_TRANSPOSE_OPERATION_HPP
