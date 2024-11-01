#ifndef BASE_MATRIX_CHOLESKY_DECOMPOSITION_HPP
#define BASE_MATRIX_CHOLESKY_DECOMPOSITION_HPP

#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include <cmath>
#include <cstddef>

namespace Base {
namespace Matrix {

template <typename T, std::size_t M>
Matrix<T, M, M> cholesky_decomposition(const Matrix<T, M, M> &U,
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
      Y(i, i) = std::sqrt(temp);
      T temp_inv = static_cast<T>(1) / Y(i, i);

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
      Y(i, i) = std::sqrt(temp);
      T temp_inv = static_cast<T>(1) / Y(i, i);

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
      Y(i, i) = std::sqrt(temp);
    }
  }

  return Y;
}

template <typename T, std::size_t M>
DiagMatrix<T, M> cholesky_decomposition_diag(const DiagMatrix<T, M> &U,
                                             const DiagMatrix<T, M> &Y_b,
                                             bool &zero_div_flag) {
  DiagMatrix<T, M> Y;

  for (std::size_t i = 0; i < M; ++i) {
    if (U[i] >= static_cast<T>(0)) {
      Y[i] = std::sqrt(U[i]);
    } else {
      zero_div_flag = true;
      Y = Y_b;
      break;
    }
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t V>
Matrix<T, M, M> cholesky_decomposition_sparse(const SparseMatrix<T, M, M, V> &U,
                                              const Matrix<T, M, M> &Y_b,
                                              bool &zero_div_flag) {
  Matrix<T, M, M> U_dense = U.create_dense();
  Matrix<T, M, M> Y;

  for (std::size_t i = 0; i < M; ++i) {
    if (i == 0) {
      T temp = U_dense(i, i);
      if (temp <= static_cast<T>(0)) {
        zero_div_flag = true;
        return Y_b;
      }
      Y(i, i) = std::sqrt(temp);
      T temp_inv = static_cast<T>(1) / Y(i, i);

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
      Y(i, i) = std::sqrt(temp);
      T temp_inv = static_cast<T>(1) / Y(i, i);

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
      Y(i, i) = std::sqrt(temp);
    }
  }

  return Y;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_CHOLESKY_DECOMPOSITION_HPP
