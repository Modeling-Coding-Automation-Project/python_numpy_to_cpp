/**
 * @file base_matrix_cholesky_decomposition.hpp
 * @brief Provides Cholesky decomposition implementations for dense, diagonal,
 * and sparse matrices.
 *
 * This header defines a set of template functions within the Base::Matrix
 * namespace for performing Cholesky decomposition on different matrix types,
 * including dense matrices, diagonal matrices, and compiled sparse matrices.
 * The decomposition is implemented for fixed-size matrices and supports error
 * handling for non-positive-definite matrices.
 *
 * All functions include error handling for non-positive-definite matrices and
 * return a fallback matrix if decomposition fails.
 */
#ifndef __BASE_MATRIX_CHOLESKY_DECOMPOSITION_HPP__
#define __BASE_MATRIX_CHOLESKY_DECOMPOSITION_HPP__

#include "base_matrix_macros.hpp"

#include "base_math.hpp"
#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"

#include <cstddef>

namespace Base {
namespace Matrix {

/**
 * @brief Computes the Cholesky decomposition of a symmetric, positive-definite
 * matrix.
 *
 * This function performs the Cholesky decomposition on the input matrix U,
 * producing an upper-triangular matrix Y such that U = Y^T * Y. If the
 * decomposition fails due to non-positive-definite input or near-zero division,
 * the function sets the zero_div_flag to true and returns the fallback matrix
 * Y_b.
 *
 * @tparam T Numeric type of the matrix elements (e.g., float, double).
 * @tparam M The dimension of the square matrix.
 * @param U The input symmetric, positive-definite matrix to decompose (size M x
 * M).
 * @param Y_b The fallback matrix to return if decomposition fails.
 * @param division_min Minimum value to avoid division by zero or very small
 * numbers.
 * @param[out] zero_div_flag Set to true if a zero or negative value is
 * encountered during decomposition.
 * @return Matrix<T, M, M> The upper-triangular matrix resulting from the
 * Cholesky decomposition, or Y_b on failure.
 */
template <typename T, std::size_t M>
inline Matrix<T, M, M>
cholesky_decomposition(const Matrix<T, M, M> &U, const Matrix<T, M, M> &Y_b,
                       const T &division_min, bool &zero_div_flag) {
  Matrix<T, M, M> Y;

  for (std::size_t i = 0; i < M; ++i) {
    if (i == 0) {
      T temp = U(i, i);
      if (temp <= static_cast<T>(0)) {
        zero_div_flag = true;
        return Y_b;
      }

      T temp_inv = Base::Math::rsqrt<T>(temp, division_min);
      Y(i, i) = static_cast<T>(1) / temp_inv;

      for (std::size_t j = 1; j < M; ++j) {
        Y(0, j) = U(j, 0) * temp_inv;
      }
    } else if (i < M - 1) {
      T temp = static_cast<T>(0);
      for (std::size_t j = 0; j < i; ++j) {
        temp += Y(j, i) * Y(j, i);
      }
      temp = U(i, i) - temp;
      if (temp <= static_cast<T>(0)) {
        zero_div_flag = true;
        return Y_b;
      }

      T temp_inv = Base::Math::rsqrt<T>(temp, division_min);
      Y(i, i) = static_cast<T>(1) / temp_inv;

      for (std::size_t j = i + 1; j < M; ++j) {
        T sum = static_cast<T>(0);
        for (std::size_t k = 0; k < i; ++k) {
          sum += Y(k, j) * Y(k, i);
        }
        Y(i, j) = (U(j, i) - sum) * temp_inv;
      }
    } else {
      T temp = static_cast<T>(0);
      for (std::size_t j = 0; j < i; ++j) {
        temp += Y(j, i) * Y(j, i);
      }
      temp = U(i, i) - temp;
      if (temp <= static_cast<T>(0)) {
        zero_div_flag = true;
        return Y_b;
      }
      Y(i, i) = Base::Math::sqrt<T>(temp);
    }
  }

  return Y;
}

/**
 * @brief Computes the Cholesky decomposition for a diagonal matrix.
 *
 * This function calculates the Cholesky decomposition of a diagonal matrix `U`.
 * The result is a diagonal matrix `Y` such that `Y * Y^T = U`, where each
 * diagonal element of `Y` is the square root of the corresponding element in
 * `U`.
 *
 * If any diagonal element of `U` is negative, the function sets the
 * `zero_div_flag` to true, assigns the fallback matrix `Y_b` to `Y`, and
 * terminates the computation.
 *
 * @tparam T The numeric type of the matrix elements.
 * @tparam M The size of the diagonal matrix.
 * @param U The input diagonal matrix to decompose.
 * @param Y_b The fallback diagonal matrix to use if a negative value is
 * encountered.
 * @param zero_div_flag Reference to a boolean flag that is set to true if a
 * negative diagonal element is found in `U`.
 * @return The resulting diagonal matrix from the Cholesky decomposition, or
 * `Y_b` if a negative value is encountered.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> cholesky_decomposition_diag(const DiagMatrix<T, M> &U,
                                                    const DiagMatrix<T, M> &Y_b,
                                                    bool &zero_div_flag) {
  DiagMatrix<T, M> Y;

  for (std::size_t i = 0; i < M; ++i) {
    if (U[i] >= static_cast<T>(0)) {
      Y[i] = Base::Math::sqrt<T>(U[i]);
    } else {
      zero_div_flag = true;
      Y = Y_b;
      break;
    }
  }

  return Y;
}

/**
 * @brief Computes the Cholesky decomposition of a sparse symmetric
 * positive-definite matrix.
 *
 * This function takes a sparse matrix in compiled format and computes its
 * Cholesky decomposition, returning the upper-triangular matrix Y such that U =
 * Y^T * Y. The function handles numerical stability by checking for
 * non-positive pivots and sets a flag if a zero or negative division is
 * detected.
 *
 * @tparam T             The numeric type of the matrix elements (e.g., float,
 * double).
 * @tparam M             The dimension of the square matrix.
 * @tparam RowIndices_U  The type used for row indices in the sparse matrix
 * representation.
 * @tparam RowPointers_U The type used for row pointers in the sparse matrix
 * representation.
 * @param U              The input sparse matrix in compiled format.
 * @param Y_b            The fallback matrix to return if decomposition fails.
 * @param division_min   The minimum value used to avoid division by zero in
 * reciprocal square root.
 * @param zero_div_flag  Reference to a boolean flag that is set to true if a
 * zero or negative pivot is encountered.
 * @return Matrix<T, M, M> The upper-triangular matrix resulting from the
 * Cholesky decomposition, or Y_b if failed.
 */
template <typename T, std::size_t M, typename RowIndices_U,
          typename RowPointers_U>
inline Matrix<T, M, M> cholesky_decomposition_sparse(
    const CompiledSparseMatrix<T, M, M, RowIndices_U, RowPointers_U> &U,
    const Matrix<T, M, M> &Y_b, const T &division_min, bool &zero_div_flag) {
  Matrix<T, M, M> U_dense = Base::Matrix::output_dense_matrix(U);
  Matrix<T, M, M> Y;

  for (std::size_t i = 0; i < M; ++i) {
    if (i == 0) {
      T temp = U_dense(i, i);
      if (temp <= static_cast<T>(0)) {
        zero_div_flag = true;
        return Y_b;
      }

      T temp_inv = Base::Math::rsqrt<T>(temp, division_min);
      Y(i, i) = static_cast<T>(1) / temp_inv;

      for (std::size_t j = 1; j < M; ++j) {
        Y(0, j) = U_dense(j, 0) * temp_inv;
      }
    } else if (i < M - 1) {
      T temp = static_cast<T>(0);
      for (std::size_t j = 0; j < i; ++j) {
        temp += Y(j, i) * Y(j, i);
      }
      temp = U_dense(i, i) - temp;
      if (temp <= static_cast<T>(0)) {
        zero_div_flag = true;
        return Y_b;
      }

      T temp_inv = Base::Math::rsqrt<T>(temp, division_min);
      Y(i, i) = static_cast<T>(1) / temp_inv;

      for (std::size_t j = i + 1; j < M; ++j) {
        T sum = static_cast<T>(0);
        for (std::size_t k = 0; k < i; ++k) {
          sum += Y(k, j) * Y(k, i);
        }
        Y(i, j) = (U_dense(j, i) - sum) * temp_inv;
      }
    } else {
      T temp = static_cast<T>(0);
      for (std::size_t j = 0; j < i; ++j) {
        temp += Y(j, i) * Y(j, i);
      }
      temp = U_dense(i, i) - temp;
      if (temp <= static_cast<T>(0)) {
        zero_div_flag = true;
        return Y_b;
      }
      Y(i, i) = Base::Math::sqrt<T>(temp);
    }
  }

  return Y;
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_CHOLESKY_DECOMPOSITION_HPP__
