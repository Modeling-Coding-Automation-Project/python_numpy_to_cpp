#ifndef PYTHON_NUMPY_TRANSPOSE_OPERATION_HPP
#define PYTHON_NUMPY_TRANSPOSE_OPERATION_HPP

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>

namespace PythonNumpy {

/* (matrix) * (transposed matrix) */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline auto A_mul_BTranspose(const Matrix<DefDense, T, M, K> &A,
                             const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      Base::Matrix::matrix_multiply_A_mul_BTranspose(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t K>
inline auto A_mul_BTranspose(const Matrix<DefDense, T, M, K> &A,
                             const Matrix<DefDiag, T, K> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(A.matrix * B.matrix);
}

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          typename SparseAvailable>
inline auto
A_mul_BTranspose(const Matrix<DefDense, T, M, K> &A,
                 const Matrix<DefSparse, T, N, K, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      Base::Matrix::matrix_multiply_A_mul_SparseBTranspose(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
inline auto A_mul_BTranspose(const Matrix<DefDiag, T, M> &A,
                             const Matrix<DefDense, T, N, M> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      A.matrix * Base::Matrix::output_matrix_transpose(B.matrix));
}

template <typename T, std::size_t M>
inline auto A_mul_BTranspose(const Matrix<DefDiag, T, M> &A,
                             const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(A.matrix * B.matrix);
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto
A_mul_BTranspose(const Matrix<DefDiag, T, M> &A,
                 const Matrix<DefSparse, T, N, M, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      A.matrix * Base::Matrix::output_matrix_transpose(B.matrix));
}

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          typename SparseAvailable>
inline auto
A_mul_BTranspose(const Matrix<DefSparse, T, M, K, SparseAvailable> &A,
                 const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      Base::Matrix::matrix_multiply_SparseA_mul_BTranspose(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t K, typename SparseAvailable>
inline auto
A_mul_BTranspose(const Matrix<DefSparse, T, M, K, SparseAvailable> &A,
                 const Matrix<DefDiag, T, K> &B)
    -> Matrix<DefSparse, T, M, K, SparseAvailable> {

  return Matrix<DefSparse, T, M, K, SparseAvailable>(A.matrix * B.matrix);
}

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline auto
A_mul_BTranspose(const Matrix<DefSparse, T, M, K, SparseAvailable_A> &A,
                 const Matrix<DefSparse, T, N, K, SparseAvailable_B> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      matrix_multiply_SparseA_mul_SparseBTranspose(A.matrix, B.matrix));
}

/* (transpose matrix) * (matrix) */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline auto ATranspose_mul_B(const Matrix<DefDense, T, N, M> &A,
                             const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(
      Base::Matrix::matrix_multiply_AT_mul_B(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
inline auto ATranspose_mul_B(const Matrix<DefDense, T, N, M> &A,
                             const Matrix<DefDiag, T, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      Base::Matrix::output_matrix_transpose(A.matrix) * B.matrix);
}

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          typename SparseAvailable>
inline auto
ATranspose_mul_B(const Matrix<DefDense, T, N, M> &A,
                 const Matrix<DefSparse, T, N, K, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(
      Base::Matrix::matrix_multiply_ATranspose_mul_SparseB(A.matrix, B.matrix));
}

template <typename T, std::size_t K, std::size_t N>
inline auto ATranspose_mul_B(const Matrix<DefDiag, T, N> &A,
                             const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, N, K> {

  return Matrix<DefDense, T, N, K>(A.matrix * B.matrix);
}

template <typename T, std::size_t M>
inline auto ATranspose_mul_B(const Matrix<DefDiag, T, M> &A,
                             const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(A.matrix * B.matrix);
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto
ATranspose_mul_B(const Matrix<DefDiag, T, M> &A,
                 const Matrix<DefSparse, T, N, M, SparseAvailable> &B)
    -> Matrix<DefSparse, T, N, M, SparseAvailable> {

  return Matrix<DefSparse, T, N, M, SparseAvailable>(A.matrix * B.matrix);
}

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          typename SparseAvailable>
inline auto
ATranspose_mul_B(const Matrix<DefSparse, T, N, M, SparseAvailable> &A,
                 const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(
      Base::Matrix::matrix_multiply_SparseAT_mul_B(A.matrix, B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto
ATranspose_mul_B(const Matrix<DefSparse, T, N, M, SparseAvailable> &A,
                 const Matrix<DefDiag, T, N> &B) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      Base::Matrix::matrix_multiply_Transpose_DiagA_mul_SparseB(B.matrix,
                                                                A.matrix));
}

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline auto
ATranspose_mul_B(const Matrix<DefSparse, T, N, M, SparseAvailable_A> &A,
                 const Matrix<DefSparse, T, N, K, SparseAvailable_B> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(
      Base::Matrix::matrix_multiply_SparseATranspose_mul_SparseB(A.matrix,
                                                                 B.matrix));
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_TRANSPOSE_OPERATION_HPP
