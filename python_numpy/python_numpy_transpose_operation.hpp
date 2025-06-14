/**
 * @file python_numpy_transpose_operation.hpp
 * @brief Provides matrix multiplication operations involving transposed
 * matrices, supporting dense, diagonal, and sparse matrix types.
 *
 * This header defines a set of template functions within the PythonNumpy
 * namespace to perform efficient matrix multiplications where one of the
 * operands is transposed. The operations are designed to work with various
 * matrix representations, including dense, diagonal, and sparse matrices, and
 * their combinations. The functions leverage template specialization to select
 * the optimal multiplication strategy based on the matrix types involved.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __PYTHON_NUMPY_TRANSPOSE_OPERATION_HPP__
#define __PYTHON_NUMPY_TRANSPOSE_OPERATION_HPP__

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>
#include <utility>

namespace PythonNumpy {

/* (matrix) * (transposed matrix) */

/**
 * @brief Multiplies matrix A by the transpose of matrix B.
 *
 * This function computes the matrix product of A and the transpose of B,
 * i.e., it returns the result of A * B^T. Both input matrices must have the
 * same number of columns (K). The resulting matrix will have the same number of
 * rows as A (M) and the same number of rows as B (N).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and B.
 * @tparam N The number of columns in matrix B.
 * @param A The left-hand side matrix of size M x K.
 * @param B The right-hand side matrix of size N x K (to be transposed).
 * @return Matrix<DefDense, T, M, N> The result of A multiplied by B transposed.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline auto A_mul_BTranspose(const Matrix<DefDense, T, M, K> &A,
                             const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      Base::Matrix::matrix_multiply_A_mul_BTranspose(A.matrix, B.matrix));
}

/**
 * @brief Multiplies matrix A by the transpose of diagonal matrix B.
 *
 * This function computes the matrix product of A and the transpose of a
 * diagonal matrix B, i.e., it returns the result of A * B^T. The resulting
 * matrix will have the same number of rows as A (M) and the same number of rows
 * as B (K).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and B.
 * @param A The left-hand side matrix of size M x K.
 * @param B The right-hand side diagonal matrix of size K.
 * @return Matrix<DefDense, T, M, K> The result of
 * A multiplied by B transposed.
 */
template <typename T, std::size_t M, std::size_t K>
inline auto A_mul_BTranspose(const Matrix<DefDense, T, M, K> &A,
                             const Matrix<DefDiag, T, K> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(A.matrix * B.matrix);
}

/**
 * @brief Multiplies matrix A by the transpose of sparse matrix B.
 *
 * This function computes the matrix product of A and the transpose of a sparse
 * matrix B, i.e., it returns the result of A * B^T. The resulting matrix will
 * have the same number of rows as A (M) and the same number of rows as B (N).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and B.
 * @tparam N The number of columns in matrix B.
 * @param A The left-hand side matrix of size M x K.
 * @param B The right-hand side sparse matrix of size N x K (to be transposed).
 * @return Matrix<DefDense, T, M, N> The result of
 * A multiplied by B transposed.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          typename SparseAvailable>
inline auto
A_mul_BTranspose(const Matrix<DefDense, T, M, K> &A,
                 const Matrix<DefSparse, T, N, K, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      Base::Matrix::matrix_multiply_A_mul_SparseBTranspose(A.matrix, B.matrix));
}

/**
 * @brief Multiplies diagonal matrix A by the transpose of matrix B.
 *
 * This function computes the matrix product of a diagonal matrix A and the
 * transpose of matrix B, i.e., it returns the result of A * B^T. The resulting
 * matrix will have the same number of rows as A (M) and the same number of rows
 * as B (N).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A and columns in B.
 * @param A The left-hand side diagonal matrix of size M.
 * @param B The right-hand side matrix of size N x M (to be transposed).
 * @return Matrix<DefDense, T, M, N> The result of A multiplied by B transposed.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto A_mul_BTranspose(const Matrix<DefDiag, T, M> &A,
                             const Matrix<DefDense, T, N, M> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      A.matrix * Base::Matrix::output_matrix_transpose(B.matrix));
}

/**
 * @brief Multiplies diagonal matrix A by the transpose of diagonal matrix B.
 *
 * This function computes the matrix product of two diagonal matrices A and B,
 * i.e., it returns the result of A * B^T. The resulting matrix will be a
 * diagonal matrix of size M.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both diagonal matrices.
 * @param A The left-hand side diagonal matrix of size M.
 * @param B The right-hand side diagonal matrix of size M (to be transposed).
 * @return Matrix<DefDiag, T, M> The result of A multiplied by B transposed.
 */
template <typename T, std::size_t M>
inline auto A_mul_BTranspose(const Matrix<DefDiag, T, M> &A,
                             const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(A.matrix * B.matrix);
}

/**
 * @brief Multiplies diagonal matrix A by the transpose of sparse matrix B.
 *
 * This function computes the matrix product of a diagonal matrix A and the
 * transpose of a sparse matrix B, i.e., it returns the result of A * B^T. The
 * resulting matrix will have the same number of rows as A (M) and the same
 * number of rows as B (N).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A and columns in B.
 * @param A The left-hand side diagonal matrix of size M.
 * @param B The right-hand side sparse matrix of size N x M (to be transposed).
 * @return Matrix<DefSparse, T, N, M, SparseAvailable> The result of
 * A multiplied by B transposed.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto
A_mul_BTranspose(const Matrix<DefDiag, T, M> &A,
                 const Matrix<DefSparse, T, N, M, SparseAvailable> &B)
    -> Matrix<DefSparse, T, M, N, SparseAvailableTranspose<SparseAvailable>> {

  return Matrix<DefSparse, T, M, N, SparseAvailableTranspose<SparseAvailable>>(
      A.matrix * Base::Matrix::output_matrix_transpose(B.matrix));
}

/**
 * @brief Multiplies sparse matrix A by the transpose of matrix B.
 *
 * This function computes the matrix product of a sparse matrix A and the
 * transpose of a dense matrix B, i.e., it returns the result of A * B^T. The
 * resulting matrix will have the same number of rows as A (M) and the same
 * number of rows as B (N).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and B.
 * @tparam N The number of columns in matrix B.
 * @param A The left-hand side sparse matrix of size M x K.
 * @param B The right-hand side dense matrix of size N x K (to be transposed).
 * @return Matrix<DefDense, T, M, N> The result of A multiplied by B transposed.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          typename SparseAvailable>
inline auto
A_mul_BTranspose(const Matrix<DefSparse, T, M, K, SparseAvailable> &A,
                 const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      Base::Matrix::matrix_multiply_SparseA_mul_BTranspose(A.matrix, B.matrix));
}

/**
 * @brief Multiplies sparse matrix A by the transpose of diagonal matrix B.
 *
 * This function computes the matrix product of a sparse matrix A and the
 * transpose of a diagonal matrix B, i.e., it returns the result of A * B^T.
 * The resulting matrix will have the same number of rows as A (M) and the same
 * number of rows as B (K).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and B.
 * @param A The left-hand side sparse matrix of size M x K.
 * @param B The right-hand side diagonal matrix
 * of size K (to be transposed).
 * @return Matrix<DefSparse, T, M, K, SparseAvailable> The result of
 * A multiplied by B transposed.
 */
template <typename T, std::size_t M, std::size_t K, typename SparseAvailable>
inline auto
A_mul_BTranspose(const Matrix<DefSparse, T, M, K, SparseAvailable> &A,
                 const Matrix<DefDiag, T, K> &B)
    -> Matrix<DefSparse, T, M, K, SparseAvailable> {

  return Matrix<DefSparse, T, M, K, SparseAvailable>(A.matrix * B.matrix);
}

/**
 * @brief Multiplies sparse matrix A by the transpose of sparse matrix B.
 *
 * This function computes the matrix product of two sparse matrices A and B,
 * i.e., it returns the result of A * B^T. The resulting matrix will have the
 * same number of rows as A (M) and the same number of rows as B (N).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and B.
 * @tparam N The number of columns in matrix B.
 * @param A The left-hand side sparse matrix of size M x K.
 * @param B The right-hand side sparse matrix of size N x K (to be transposed).
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> The result of
 * A multiplied by B transposed.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline auto
A_mul_BTranspose(const Matrix<DefSparse, T, M, K, SparseAvailable_A> &A,
                 const Matrix<DefSparse, T, N, K, SparseAvailable_B> &B)
    -> Matrix<
        DefSparse, T, M, N,
        SparseAvailableMatrixMultiply<
            SparseAvailable_A, SparseAvailableTranspose<SparseAvailable_B>>> {

  using SparseAvailable_BT = SparseAvailableTranspose<SparseAvailable_B>;
  using SparseAvailable_A_mul_BT =
      SparseAvailableMatrixMultiply<SparseAvailable_A, SparseAvailable_BT>;

  return Matrix<DefSparse, T, M, N, SparseAvailable_A_mul_BT>(
      matrix_multiply_SparseA_mul_SparseBTranspose(A.matrix, B.matrix));
}

/* A_mul_BTranspose Type */
template <typename A_Type, typename B_Type>
using A_mul_BTranspose_Type =
    decltype(A_mul_BTranspose(std::declval<A_Type>(), std::declval<B_Type>()));

/* (transpose matrix) * (matrix) */

/**
 * @brief Multiplies the transpose of matrix A by matrix B.
 *
 * This function computes the matrix product of the transpose of A and B,
 * i.e., it returns the result of A^T * B. The resulting matrix will have the
 * same number of rows as A (M) and the same number of columns as B (K).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and B.
 * @tparam N The number of columns in matrix B.
 * @param A The left-hand side matrix of size N x M (to be transposed).
 * @param B The right-hand side matrix of size N x K.
 * @return Matrix<DefDense, T, M, K> The result of A transposed multiplied by B.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline auto ATranspose_mul_B(const Matrix<DefDense, T, N, M> &A,
                             const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(
      Base::Matrix::matrix_multiply_AT_mul_B(A.matrix, B.matrix));
}

/**
 * @brief Multiplies the transpose of diagonal matrix A by matrix B.
 *
 * This function computes the matrix product of the transpose of a diagonal
 * matrix A and B, i.e., it returns the result of A^T * B. The resulting matrix
 * will have the same number of rows as A (M) and the same number of columns as
 * B (K).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A and columns in B.
 * @param A The left-hand side diagonal matrix
 * of size M (to be transposed).
 * @param B The right-hand side matrix of size N x M.
 * @return Matrix<DefDense, T, M, K> The result of
 * A transposed multiplied by B.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto ATranspose_mul_B(const Matrix<DefDense, T, N, M> &A,
                             const Matrix<DefDiag, T, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(
      Base::Matrix::output_matrix_transpose(A.matrix) * B.matrix);
}

/**
 * @brief Multiplies the transpose of sparse matrix A by matrix B.
 *
 * This function computes the matrix product of the transpose of a sparse
 * matrix A and B, i.e., it returns the result of A^T * B. The resulting matrix
 * will have the same number of rows as A (M) and the same number of columns as
 * B (K).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam K The number of rows in matrix B.
 * @tparam N The number of columns in matrix B.
 * @param A The left-hand side sparse matrix of size N x M (to be transposed).
 * @param B The right-hand side dense matrix of size N x K.
 * @return Matrix<DefDense, T, M, K> The result of
 * A transposed multiplied by B.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          typename SparseAvailable>
inline auto
ATranspose_mul_B(const Matrix<DefDense, T, N, M> &A,
                 const Matrix<DefSparse, T, N, K, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(
      Base::Matrix::matrix_multiply_ATranspose_mul_SparseB(A.matrix, B.matrix));
}

/**
 * @brief Multiplies the transpose of a diagonal matrix A with a dense matrix B.
 *
 * This function computes the product of the transpose of matrix A (assumed to
 * be diagonal) and matrix B (assumed to be dense), returning the result as a
 * dense matrix.
 *
 * @tparam T  The data type of the matrix elements.
 * @tparam K  The number of rows in matrix B and the result.
 * @tparam N  The number of columns in matrix A and the result.
 * @param A   The input diagonal matrix (DefDiag) of size NxN.
 * @param B   The input dense matrix (DefDense) of size NxK.
 * @return    A dense matrix (DefDense) of size NxK representing the product A^T
 * * B.
 */
template <typename T, std::size_t K, std::size_t N>
inline auto ATranspose_mul_B(const Matrix<DefDiag, T, N> &A,
                             const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, N, K> {

  return Matrix<DefDense, T, N, K>(A.matrix * B.matrix);
}

/**
 * @brief Multiplies the transpose of a diagonal matrix A with a diagonal matrix
 * B.
 *
 * This function computes the product of the transpose of matrix A (assumed to
 * be diagonal) and matrix B (also assumed to be diagonal), returning the result
 * as a diagonal matrix.
 *
 * @tparam T  The data type of the matrix elements.
 * @tparam M  The number of rows in both diagonal matrices.
 * @param A   The input diagonal matrix (DefDiag) of size M.
 * @param B   The input diagonal matrix (DefDiag) of size M.
 * @return    A diagonal matrix (DefDiag) of size M representing the product A^T
 * * B.
 */
template <typename T, std::size_t M>
inline auto ATranspose_mul_B(const Matrix<DefDiag, T, M> &A,
                             const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(A.matrix * B.matrix);
}

/**
 * @brief Multiplies the transpose of a diagonal matrix A with a sparse matrix
 * B.
 *
 * This function computes the product of the transpose of matrix A (assumed to
 * be diagonal) and matrix B (assumed to be sparse), returning the result as a
 * sparse matrix.
 *
 * @tparam T  The data type of the matrix elements.
 * @tparam M  The number of rows in matrix A and columns in B.
 * @tparam N  The number of columns in matrix B.
 * @param A   The input diagonal matrix (DefDiag) of size M.
 * @param B   The input sparse matrix (DefSparse) of size N x M.
 * @return    A sparse matrix (DefSparse) of size M x N representing the product
 * A^T * B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto
ATranspose_mul_B(const Matrix<DefDiag, T, M> &A,
                 const Matrix<DefSparse, T, N, M, SparseAvailable> &B)
    -> Matrix<DefSparse, T, N, M, SparseAvailable> {

  return Matrix<DefSparse, T, N, M, SparseAvailable>(A.matrix * B.matrix);
}

/**
 * @brief Multiplies the transpose of a sparse matrix A with a dense matrix B.
 *
 * This function computes the product of the transpose of a sparse matrix A and
 * a dense matrix B, returning the result as a dense matrix.
 *
 * @tparam T  The data type of the matrix elements.
 * @tparam M  The number of rows in matrix A and columns in B.
 * @tparam K  The number of rows in matrix B.
 * @param A   The input sparse matrix (DefSparse) of
 * size N x M (to be transposed).
 * @param B   The input dense matrix (DefDense) of size N x K.
 * @return    A dense matrix (DefDense) of size M x K representing the product
 * A^T * B.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          typename SparseAvailable>
inline auto
ATranspose_mul_B(const Matrix<DefSparse, T, N, M, SparseAvailable> &A,
                 const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(
      Base::Matrix::matrix_multiply_SparseAT_mul_B(A.matrix, B.matrix));
}

/**
 * @brief Multiplies the transpose of a sparse matrix A by a diagonal matrix B.
 *
 * This function computes the matrix multiplication of the transpose of matrix A
 * (which is a sparse matrix of size N x M) with matrix B (a diagonal matrix of
 * size N x N). The result is a sparse matrix of size M x N, with the sparsity
 * pattern determined by SparseAvailableTranspose<SparseAvailable>.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in the original matrix A.
 * @tparam N The number of rows in the original matrix A (and the size of the
 * diagonal matrix B).
 * @tparam SparseAvailable The sparsity pattern or trait for the input matrix A.
 *
 * @param A The input sparse matrix of type Matrix<DefSparse, T, N, M,
 * SparseAvailable>.
 * @param B The input diagonal matrix of type Matrix<DefDiag, T, N>.
 * @return Matrix<DefSparse, T, M, N, SparseAvailableTranspose<SparseAvailable>>
 *         The result of (A^T) * B as a sparse matrix.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto
ATranspose_mul_B(const Matrix<DefSparse, T, N, M, SparseAvailable> &A,
                 const Matrix<DefDiag, T, N> &B)
    -> Matrix<DefSparse, T, M, N, SparseAvailableTranspose<SparseAvailable>> {

  return Matrix<DefSparse, T, M, N, SparseAvailableTranspose<SparseAvailable>>(
      Base::Matrix::matrix_multiply_Transpose_DiagA_mul_SparseB(B.matrix,
                                                                A.matrix));
}

/**
 * @brief Multiplies the transpose of a sparse matrix A with another sparse
 * matrix B.
 *
 * This function computes the matrix multiplication of the transpose of a sparse
 * matrix A (of size N x M) with another sparse matrix B (of size N x K). The
 * result is a sparse matrix of size M x K, with the sparsity pattern determined
 * by SparseAvailableMatrixMultiply<SparseAvailableTranspose<SparseAvailable_A>,
 * SparseAvailable_B>.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the original matrix A.
 * @tparam K The number of rows in the original matrix B.
 * @tparam N The number of columns in both matrices A and B.
 * @tparam SparseAvailable_A The sparsity pattern or trait for the input matrix
 * A.
 * @tparam SparseAvailable_B The sparsity pattern or trait for the input matrix
 * B.
 *
 * @param A The input sparse matrix of type Matrix<DefSparse, T, N, M,
 * SparseAvailable_A>.
 * @param B The input sparse matrix of type Matrix<DefSparse, T, N, K,
 * SparseAvailable_B>.
 * @return Matrix<DefSparse, T, M, K,
 *         SparseAvailableMatrixMultiply<SparseAvailableTranspose<SparseAvailable_A>,
 *         SparseAvailable_B>>
 *         The result of (A^T) * B as a sparse matrix.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline auto
ATranspose_mul_B(const Matrix<DefSparse, T, N, M, SparseAvailable_A> &A,
                 const Matrix<DefSparse, T, N, K, SparseAvailable_B> &B)
    -> Matrix<
        DefSparse, T, M, K,
        SparseAvailableMatrixMultiply<
            SparseAvailableTranspose<SparseAvailable_A>, SparseAvailable_B>> {

  using SparseAvailable_AT = SparseAvailableTranspose<SparseAvailable_A>;
  using SparseAvailable_AT_mul_B =
      SparseAvailableMatrixMultiply<SparseAvailable_AT, SparseAvailable_B>;

  return Matrix<DefSparse, T, M, K, SparseAvailable_AT_mul_B>(
      Base::Matrix::matrix_multiply_SparseATranspose_mul_SparseB(A.matrix,
                                                                 B.matrix));
}

/* ATranspose_mul_B Type */
template <typename A_Type, typename B_Type>
using ATranspose_mul_B_Type =
    decltype(ATranspose_mul_B(std::declval<A_Type>(), std::declval<B_Type>()));

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_TRANSPOSE_OPERATION_HPP__
