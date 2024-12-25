#ifndef BASE_MATRIX_CHOLESKY_DECOMPOSITION_HPP
#define BASE_MATRIX_CHOLESKY_DECOMPOSITION_HPP

#include "base_math.hpp"
#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"

#include <cstddef>

namespace Base {
namespace Matrix {

constexpr double CHOLESKY_DECOMPOSITION_DIVISION_MIN_DEFAULT = 1.0e-20;

template <typename T, std::size_t M>
inline Matrix<T, M, M> cholesky_decomposition(const Matrix<T, M, M> &U,
                                              const Matrix<T, M, M> &Y_b,
                                              bool &zero_div_flag) {
  Matrix<T, M, M> Y;

  for (std::size_t i = 0; i < M; ++i) {
    if (i == 0) {
      T temp = U(i, i);
      if (temp <= static_cast<T>(0)) {
        zero_div_flag = true;
        return Y_b;
      }

      T temp_inv = Base::Math::rsqrt<T>(
          temp, static_cast<T>(
                    Base::Matrix::CHOLESKY_DECOMPOSITION_DIVISION_MIN_DEFAULT));
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

      T temp_inv = Base::Math::rsqrt<T>(
          temp, static_cast<T>(
                    Base::Matrix::CHOLESKY_DECOMPOSITION_DIVISION_MIN_DEFAULT));
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

template <typename T, std::size_t M, typename RowIndices_U,
          typename RowPointers_U>
inline Matrix<T, M, M> cholesky_decomposition_sparse(
    const CompiledSparseMatrix<T, M, M, RowIndices_U, RowPointers_U> &U,
    const Matrix<T, M, M> &Y_b, bool &zero_div_flag) {
  Matrix<T, M, M> U_dense = Base::Matrix::output_dense_matrix(U);
  Matrix<T, M, M> Y;

  for (std::size_t i = 0; i < M; ++i) {
    if (i == 0) {
      T temp = U_dense(i, i);
      if (temp <= static_cast<T>(0)) {
        zero_div_flag = true;
        return Y_b;
      }

      T temp_inv = Base::Math::rsqrt<T>(
          temp, static_cast<T>(
                    Base::Matrix::CHOLESKY_DECOMPOSITION_DIVISION_MIN_DEFAULT));
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

      T temp_inv = Base::Math::rsqrt<T>(
          temp, static_cast<T>(
                    Base::Matrix::CHOLESKY_DECOMPOSITION_DIVISION_MIN_DEFAULT));
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

#endif // BASE_MATRIX_CHOLESKY_DECOMPOSITION_HPP
