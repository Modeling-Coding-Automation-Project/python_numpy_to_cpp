/**
 * @file base_matrix_inverse.hpp
 * @brief GMRES-based matrix inverse and linear solver utilities for dense,
 * sparse, and complex matrices.
 *
 * This header provides a collection of template functions for solving linear
 * systems and computing matrix inverses using the GMRES (Generalized Minimal
 * Residual) iterative method. It supports dense matrices, sparse matrices
 * (CSR-like), and complex-valued matrices, as well as diagonal matrix
 * inversion. The implementation is generic and works with fixed-size matrices
 * and vectors, supporting both real and complex types.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __BASE_MATRIX_INVERSE_HPP__
#define __BASE_MATRIX_INVERSE_HPP__

#include "base_matrix_macros.hpp"

#include "base_math.hpp"
#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_complex.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"

#include <cstddef>

namespace Base {
namespace Matrix {

/* GMRES K */

namespace InverseOperation {

/**
 * @brief Core function for GMRES K method for dense matrices.
 *
 * This function implements the GMRES K algorithm to solve the linear system
 * Ax = b, where A is a square matrix, b is a vector, and x_1 is an initial
 * guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The input matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @param matrix_size The size of the matrix (M).
 * @return The computed solution vector x.
 */
template <typename T, std::size_t M>
inline typename std::enable_if<(M > 1), Vector<T, M>>::type
gmres_k_core(const Matrix<T, M, M> &A, const Vector<T, M> &b,
             const Vector<T, M> &x_1, const T &decay_rate,
             const T &division_min, T &rho, std::size_t &rep_num,
             const std::size_t &matrix_size) {
  static_assert(M > 1, "Matrix size must be equal or larger than 2x2.");

  Matrix<T, M, M> r;
  Vector<T, M + 1> b_hat;
  b_hat[0] = static_cast<T>(1);
  Matrix<T, M, M + 1> q;
  Matrix<T, M + 1, M> h;
  Vector<T, M> c;
  Vector<T, M> s;
  Vector<T, M> y;
  Vector<T, M> x_dif;
  T ZERO = static_cast<T>(0);

  // b - Ax
  Vector<T, M> b_ax;
  for (std::size_t i = 0; i < matrix_size; ++i) {
    T sum = static_cast<T>(0);
    for (std::size_t j = 0; j < matrix_size; ++j) {
      sum += A(i, j) * x_1[j];
    }
    b_ax[i] = b[i] - sum;
  }

  // Normalize b_Ax
  T b_norm = b_ax.norm(division_min);
  for (std::size_t i = 0; i < matrix_size; ++i) {
    q(i, 0) = b_ax[i] / Base::Utility::avoid_zero_divide(b_norm, division_min);
  }
  b_hat[0] = b_norm;

  for (std::size_t n = 1; n <= matrix_size; n++) {
    // Generate orthogonal basis
    Vector<T, M> v;
    for (std::size_t i = 0; i < matrix_size; ++i) {
      for (std::size_t j = 0; j < matrix_size; ++j) {
        v[i] += A(i, j) * q(j, n - 1);
      }
    }

    for (std::size_t j = 0; j < n; ++j) {
      h(j, n - 1) = 0;
      for (std::size_t i = 0; i < matrix_size; ++i) {
        h(j, n - 1) += q(i, j) * v[i];
      }
      for (std::size_t i = 0; i < matrix_size; ++i) {
        v[i] -= h(j, n - 1) * q(i, j);
      }
    }

    if (n < matrix_size) {
      h(n, n - 1) = v.norm(division_min);
      for (std::size_t i = 0; i < matrix_size; ++i) {
        q(i, n) =
            v[i] / Base::Utility::avoid_zero_divide(h(n, n - 1), division_min);
      }
    }

    // Givens rotation for QR decomposition
    r(0, n - 1) = h(0, n - 1);
    if (n >= 2) {
      for (std::size_t j = 1; j < n; ++j) {
        T gamma = c[j - 1] * r(j - 1, n - 1) + s[j - 1] * h(j, n - 1);
        r(j, n - 1) = -s[j - 1] * r(j - 1, n - 1) + c[j - 1] * h(j, n - 1);
        r(j - 1, n - 1) = gamma;
      }
    }

    T delta_inv = Base::Math::rsqrt<T>(r(n - 1, n - 1) * r(n - 1, n - 1) +
                                           h(n, n - 1) * h(n, n - 1),
                                       division_min);

    c[n - 1] = r(n - 1, n - 1) * delta_inv;
    s[n - 1] = h(n, n - 1) * delta_inv;

    // Update b_hat
    r(n - 1, n - 1) = c[n - 1] * r(n - 1, n - 1) + s[n - 1] * h(n, n - 1);
    b_hat[n] = -s[n - 1] * b_hat[n - 1];
    b_hat[n - 1] *= c[n - 1];
    rho = Base::Math::abs(b_hat[n]);

    rep_num = n;

    // Check for convergence
    if (rho / Base::Utility::avoid_zero_divide(b_norm, division_min) <
        decay_rate) {
      break;
    }
  }

  // Back substitution to solve
  for (std::size_t j = rep_num; j-- > 0;) {
    T temp = ZERO;
    for (std::size_t m = j + 1; m < rep_num; ++m) {
      temp += r(j, m) * y[m];
    }
    y[j] = (b_hat[j] - temp) /
           Base::Utility::avoid_zero_divide(r(j, j), division_min);
  }

  for (std::size_t i = 0; i < rep_num; ++i) {
    for (std::size_t j = 0; j < matrix_size; ++j) {
      x_dif[j] += y[i] * q(j, i);
    }
  }

  Vector<T, M> x;
  for (std::size_t i = 0; i < matrix_size; ++i) {
    x[i] = x_1[i] + x_dif[i];
  }

  return x;
}

/**
 * @brief Core function for GMRES K method for 1x1 matrices.
 *
 * This specialization handles the case where the matrix size is 1x1, which is a
 * trivial case. It simply computes the inverse of the single element in the
 * matrix.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M), must be 1 in this case.
 * @param A The input matrix A, which is expected to be 1x1.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @param matrix_size The size of the matrix (M), must be 1 in this case.
 * @return The computed solution vector x, which will also be a 1x1 vector.
 */
template <typename T, std::size_t M>
inline typename std::enable_if<(M <= 1), Vector<T, M>>::type
gmres_k_core(const Matrix<T, M, M> &A, const Vector<T, M> &b,
             const Vector<T, M> &x_1, const T &decay_rate,
             const T &division_min, T &rho, std::size_t &rep_num,
             const std::size_t &matrix_size) {
  static_assert(M == 1,
                "Matrix size must be exactly 1x1 for this specialization.");
  static_cast<void>(decay_rate);
  static_cast<void>(division_min);
  static_cast<void>(rep_num);
  static_cast<void>(x_1);
  static_cast<void>(rho);
  static_cast<void>(matrix_size);

  Vector<T, M> x;

  x[0] = b[0] /
         Base::Utility::avoid_zero_divide(A.template get<0, 0>(), division_min);

  return x;
}

} // namespace InverseOperation

/**
 * @brief GMRES K method for dense matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a square matrix, b is a vector, and x_1 is an initial guess.
 * It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The input matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @return The computed solution vector x.
 */
template <typename T, std::size_t M>
inline Vector<T, M> gmres_k(const Matrix<T, M, M> &A, const Vector<T, M> &b,
                            const Vector<T, M> &x_1, const T &decay_rate,
                            const T &division_min, T &rho,
                            std::size_t &rep_num) {

  return InverseOperation::gmres_k_core<T, M>(A, b, x_1, decay_rate,
                                              division_min, rho, rep_num, M);
}

/**
 * @brief GMRES K method for partitioned matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a square matrix, b is a vector, and x_1 is an initial guess.
 * It supports partitioning of the matrix and returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The input matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @param matrix_size The size of the matrix (M).
 * @return The computed solution vector x.
 */
template <typename T, std::size_t M>
inline Vector<T, M>
gmres_k_partition(const Matrix<T, M, M> &A, const Vector<T, M> &b,
                  const Vector<T, M> &x_1, const T &decay_rate,
                  const T &division_min, T &rho, std::size_t &rep_num,
                  const std::size_t &matrix_size) {

  return InverseOperation::gmres_k_core<T, M>(
      A, b, x_1, decay_rate, division_min, rho, rep_num, matrix_size);
}

/* GMRES K for Matrix */

/**
 * @brief GMRES K method for matrices.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a square matrix, B is a matrix with multiple right-hand side
 * vectors, and X_1 is an initial guess. It returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam K The number of rows of matrix B.
 * @param A The input matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each column.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each column.
 */
template <typename T, std::size_t M, std::size_t K>
inline void gmres_k_matrix(const Matrix<T, M, M> &A, const Matrix<T, M, K> &B,
                           Matrix<T, M, K> &X_1, const T &decay_rate,
                           const T &division_min, std::array<T, K> &rho,
                           std::array<std::size_t, K> &rep_num) {

  for (std::size_t i = 0; i < K; i++) {
    Vector<T, M> x =
        Base::Matrix::gmres_k(A, B.get_row(i), X_1.get_row(i), decay_rate,
                              division_min, rho[i], rep_num[i]);
    X_1.set_row(i, x);
  }
}

/**
 * @brief GMRES K method for matrices with diagonal matrix B.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a square matrix, B is a diagonal matrix, and X_1 is an initial
 * guess. It returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The input matrix A.
 * @param B The diagonal matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each row.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each row.
 */
template <typename T, std::size_t M>
inline void gmres_k_matrix(const Matrix<T, M, M> &A, const DiagMatrix<T, M> &B,
                           Matrix<T, M, M> &X_1, const T &decay_rate,
                           const T &division_min, std::array<T, M> &rho,
                           std::array<std::size_t, M> &rep_num) {

  for (std::size_t i = 0; i < M; i++) {
    Vector<T, M> x =
        Base::Matrix::gmres_k(A, B.get_row(i), X_1.get_row(i), decay_rate,
                              division_min, rho[i], rep_num[i]);
    X_1.set_row(i, x);
  }
}

/**
 * @brief GMRES K method for partitioned matrices.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a square matrix, B is a matrix with multiple right-hand side
 * vectors, and X_1 is an initial guess. It supports partitioning of the matrix
 * and returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam K The number of rows of matrix B.
 * @param A The input matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each column.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each column.
 */
template <typename T, std::size_t M, std::size_t K>
inline void gmres_k_partition_matrix(
    const Matrix<T, M, M> &A, const Matrix<T, M, K> &B, Matrix<T, M, K> &X_1,
    const T &decay_rate, const T &division_min, std::array<T, K> &rho,
    std::array<std::size_t, K> &rep_num, const std::size_t &matrix_size) {

  std::size_t repeat_number;
  if (matrix_size < K) {
    repeat_number = matrix_size;
  } else {
    repeat_number = K;
  }

  for (std::size_t i = 0; i < repeat_number; i++) {
    Vector<T, M> x = Base::Matrix::gmres_k_partition(
        A, B.get_row(i), X_1.get_row(i), decay_rate, division_min, rho[i],
        rep_num[i], matrix_size);
    X_1.set_row(i, x);
  }
}

/**
 * @brief GMRES K method for partitioned matrices with diagonal matrix B.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a square matrix, B is a diagonal matrix, and X_1 is an initial
 * guess. It supports partitioning of the matrix and returns the solution
 * matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The input matrix A.
 * @param B The diagonal matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each row.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each row.
 */
template <typename T, std::size_t M>
inline void gmres_k_partition_matrix(
    const Matrix<T, M, M> &A, const DiagMatrix<T, M> &B, Matrix<T, M, M> &X_1,
    const T &decay_rate, const T &division_min, std::array<T, M> &rho,
    std::array<std::size_t, M> &rep_num, const std::size_t &matrix_size) {

  for (std::size_t i = 0; i < matrix_size; i++) {
    Vector<T, M> x = Base::Matrix::gmres_k_partition(
        A, B.get_row(i), X_1.get_row(i), decay_rate, division_min, rho[i],
        rep_num[i], matrix_size);
    X_1.set_row(i, x);
  }
}

/* GMRES K for rectangular matrix */

/**
 * @brief GMRES K method for rectangular matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a rectangular matrix (M x N), b is a vector, and x_1 is an
 * initial guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The number of rows in the matrix A.
 * @tparam N The number of columns in the matrix A.
 * @param In_A The input matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @return The computed solution vector x.
 */
template <typename T, std::size_t M, std::size_t N>
inline Vector<T, N> gmres_k_rect(const Matrix<T, M, N> &In_A,
                                 const Vector<T, M> &b, const Vector<T, N> &x_1,
                                 T decay_rate, const T &division_min, T &rho,
                                 std::size_t &rep_num) {
  static_assert(M > 1, "Matrix size must be equal or larger than 2x2.");
  static_assert(M > N, "Column number must be larger than row number.");

  Matrix<T, N, N> r;
  Vector<T, N + 1> b_hat;
  b_hat[0] = static_cast<T>(1);
  Matrix<T, N, N + 1> q;
  Matrix<T, N + 1, N> h;
  Vector<T, N> c;
  Vector<T, N> s;
  Vector<T, N> y;
  Vector<T, N> x_dif;
  T ZERO = static_cast<T>(0);

  // b - Ax
  Vector<T, M> b_ax_temp;
  for (std::size_t i = 0; i < M; ++i) {
    T sum = static_cast<T>(0);
    for (std::size_t j = 0; j < N; ++j) {
      sum += In_A(i, j) * x_1[j];
    }
    b_ax_temp[i] = b[i] - sum;
  }

  Matrix<T, N, N> A = Base::Matrix::matrix_multiply_AT_mul_B(In_A, In_A);
  Vector<T, N> b_ax = Base::Matrix::matrix_multiply_AT_mul_b(In_A, b_ax_temp);

  // Normalize b_Ax
  T b_norm = b_ax.norm(division_min);
  for (std::size_t i = 0; i < N; ++i) {
    q(i, 0) = b_ax[i] / Base::Utility::avoid_zero_divide(b_norm, division_min);
  }
  b_hat[0] = b_norm;

  for (std::size_t n = 1; n <= N; n++) {
    // Generate orthogonal basis
    Vector<T, N> v;
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        v[i] += A(i, j) * q(j, n - 1);
      }
    }

    for (std::size_t j = 0; j < n; ++j) {
      h(j, n - 1) = 0;
      for (std::size_t i = 0; i < N; ++i) {
        h(j, n - 1) += q(i, j) * v[i];
      }
      for (std::size_t i = 0; i < N; ++i) {
        v[i] -= h(j, n - 1) * q(i, j);
      }
    }

    if (n < N) {
      h(n, n - 1) = v.norm(division_min);
      for (std::size_t i = 0; i < N; ++i) {
        q(i, n) =
            v[i] / Base::Utility::avoid_zero_divide(h(n, n - 1), division_min);
      }
    }

    // Givens rotation for QR decomposition
    r(0, n - 1) = h(0, n - 1);
    if (n >= 2) {
      for (std::size_t j = 1; j < n; ++j) {
        T gamma = c[j - 1] * r(j - 1, n - 1) + s[j - 1] * h(j, n - 1);
        r(j, n - 1) = -s[j - 1] * r(j - 1, n - 1) + c[j - 1] * h(j, n - 1);
        r(j - 1, n - 1) = gamma;
      }
    }

    T delta_inv = Base::Math::rsqrt<T>(r(n - 1, n - 1) * r(n - 1, n - 1) +
                                           h(n, n - 1) * h(n, n - 1),
                                       division_min);

    c[n - 1] = r(n - 1, n - 1) * delta_inv;
    s[n - 1] = h(n, n - 1) * delta_inv;

    // Update b_hat
    r(n - 1, n - 1) = c[n - 1] * r(n - 1, n - 1) + s[n - 1] * h(n, n - 1);
    b_hat[n] = -s[n - 1] * b_hat[n - 1];
    b_hat[n - 1] *= c[n - 1];
    rho = Base::Math::abs(b_hat[n]);

    rep_num = n;

    // Check for convergence
    if (rho / Base::Utility::avoid_zero_divide(b_norm, division_min) <
        decay_rate) {
      break;
    }
  }

  // Back substitution to solve
  for (std::size_t j = rep_num; j-- > 0;) {
    T temp = ZERO;
    for (std::size_t m = j + 1; m < rep_num; ++m) {
      temp += r(j, m) * y[m];
    }
    y[j] = (b_hat[j] - temp) /
           Base::Utility::avoid_zero_divide(r(j, j), division_min);
  }

  for (std::size_t i = 0; i < rep_num; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      x_dif[j] += y[i] * q(j, i);
    }
  }

  Vector<T, N> x;
  for (std::size_t i = 0; i < N; ++i) {
    x[i] = x_1[i] + x_dif[i];
  }

  return x;
}

/**
 * @brief GMRES K method for rectangular matrices with multiple right-hand side
 * vectors.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a rectangular matrix (M x N), B is a matrix with multiple right-
 * hand side vectors, and X_1 is an initial guess. It returns the solution
 * matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The number of columns in the matrix A.
 * @tparam N The number of rows in the matrix A.
 * @tparam K The number of rows of matrix B.
 * @param A The input matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each row.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each row.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K>
inline void gmres_k_rect_matrix(const Matrix<T, M, N> &A,
                                const Matrix<T, M, K> &B, Matrix<T, N, K> &X_1,
                                T decay_rate, const T &division_min,
                                std::array<T, K> &rho,
                                std::array<std::size_t, K> &rep_num) {

  for (std::size_t i = 0; i < K; i++) {
    Vector<T, N> x =
        Base::Matrix::gmres_k_rect(A, B.get_row(i), X_1.get_row(i), decay_rate,
                                   division_min, rho[i], rep_num[i]);
    X_1.set_row(i, x);
  }
}

template <typename T, std::size_t M, std::size_t N>
inline void gmres_k_rect_matrix(const Matrix<T, M, N> &A,
                                const DiagMatrix<T, M> &B, Matrix<T, N, M> &X_1,
                                T decay_rate, const T &division_min,
                                std::array<T, M> &rho,
                                std::array<std::size_t, M> &rep_num) {

  for (std::size_t i = 0; i < M; i++) {
    Vector<T, N> x =
        Base::Matrix::gmres_k_rect(A, B.get_row(i), X_1.get_row(i), decay_rate,
                                   division_min, rho[i], rep_num[i]);
    X_1.set_row(i, x);
  }
}

/* GMRES K for matrix inverse */

/**
 * @brief GMRES K method for matrix inverse.
 *
 * This function computes the inverse of a square matrix A using the GMRES K
 * method, where A is a square matrix, and X_1 is an initial guess. It returns
 * the inverse matrix X.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The size of the square matrix (M x M).
 * @param In_A The input matrix A.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each row.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each row.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @return The computed inverse matrix X.
 */
template <typename T, std::size_t M>
inline Matrix<T, M, M>
gmres_k_matrix_inv(const Matrix<T, M, M> In_A, const T &decay_rate,
                   const T &division_min, std::array<T, M> &rho,
                   std::array<std::size_t, M> &rep_num,
                   const Matrix<T, M, M> X_1) {
  Matrix<T, M, M> B = Matrix<T, M, M>::identity();
  Matrix<T, M, M> X;

  for (std::size_t i = 0; i < M; i++) {
    Vector<T, M> x;

    x = Base::Matrix::gmres_k(In_A, B.get_row(i), X_1.get_row(i), decay_rate,
                              division_min, rho[i], rep_num[i]);
    X.set_row(i, x);
  }

  return X;
}

/* Sparse GMRES K */
namespace InverseOperation {

/**
 * @brief Core function for GMRES K method for sparse matrices.
 *
 * This function implements the core logic of the GMRES K method for sparse
 * matrices, handling the case where the matrix size is larger than 1x1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam RowIndices_A The type representing row indices of the sparse matrix.
 * @tparam RowPointers_A The type representing row pointers of the sparse
 * matrix.
 * @param SA The compiled sparse matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @param matrix_size The size of the matrix (M).
 * @return The computed solution vector x.
 */
template <typename T, std::size_t M, typename RowIndices_A,
          typename RowPointers_A>
inline typename std::enable_if<(M > 1), Vector<T, M>>::type sparse_gmres_k_core(
    const CompiledSparseMatrix<T, M, M, RowIndices_A, RowPointers_A> &SA,
    const Vector<T, M> &b, const Vector<T, M> &x_1, const T &decay_rate,
    const T &division_min, T &rho, std::size_t &rep_num,
    const std::size_t &matrix_size) {
  static_assert(M > 1, "Matrix size must be equal or larger than 2x2.");

  Matrix<T, M, M> r;
  Vector<T, M + 1> b_hat;
  b_hat[0] = static_cast<T>(1);
  Matrix<T, M, M + 1> q;
  Matrix<T, M + 1, M> h;
  Vector<T, M> c;
  Vector<T, M> s;
  Vector<T, M> y;
  Vector<T, M> x_dif;
  T ZERO = static_cast<T>(0);

  // b - Ax
  Vector<T, M> b_ax;
  for (std::size_t j = 0; j < matrix_size; j++) {
    T sum = static_cast<T>(0);
    for (std::size_t k = RowPointers_A::list[j]; k < RowPointers_A::list[j + 1];
         k++) {

      if (RowIndices_A::list[k] < matrix_size) {

        sum += SA.values[k] * x_1[RowIndices_A::list[k]];
      }
    }
    b_ax[j] = b[j] - sum;
  }

  // Normalize b_Ax
  T b_norm = b_ax.norm(division_min);
  for (std::size_t i = 0; i < matrix_size; ++i) {
    q(i, 0) = b_ax[i] / Base::Utility::avoid_zero_divide(b_norm, division_min);
  }
  b_hat[0] = b_norm;

  for (std::size_t n = 1; n <= matrix_size; n++) {
    // Generate orthogonal basis
    Vector<T, M> v;
    Vector<T, M> q_row_vector = q.create_row_vector(n - 1);
    for (std::size_t j = 0; j < matrix_size; j++) {
      T sum = static_cast<T>(0);
      for (std::size_t k = RowPointers_A::list[j];
           k < RowPointers_A::list[j + 1]; k++) {

        if (RowIndices_A::list[k] < matrix_size) {

          sum += SA.values[k] * q_row_vector[RowIndices_A::list[k]];
        }
      }
      v[j] = sum;
    }

    for (std::size_t j = 0; j < n; ++j) {
      h(j, n - 1) = 0;
      for (std::size_t i = 0; i < matrix_size; ++i) {
        h(j, n - 1) += q(i, j) * v[i];
      }
      for (std::size_t i = 0; i < matrix_size; ++i) {
        v[i] -= h(j, n - 1) * q(i, j);
      }
    }

    if (n < matrix_size) {
      h(n, n - 1) = v.norm(division_min);
      for (std::size_t i = 0; i < matrix_size; ++i) {
        q(i, n) =
            v[i] / Base::Utility::avoid_zero_divide(h(n, n - 1), division_min);
      }
    }

    // Givens rotation for QR decomposition
    r(0, n - 1) = h(0, n - 1);
    if (n >= 2) {
      for (std::size_t j = 1; j < n; ++j) {
        T gamma = c[j - 1] * r(j - 1, n - 1) + s[j - 1] * h(j, n - 1);
        r(j, n - 1) = -s[j - 1] * r(j - 1, n - 1) + c[j - 1] * h(j, n - 1);
        r(j - 1, n - 1) = gamma;
      }
    }

    T delta_inv = Base::Math::rsqrt<T>(r(n - 1, n - 1) * r(n - 1, n - 1) +
                                           h(n, n - 1) * h(n, n - 1),
                                       division_min);

    c[n - 1] = r(n - 1, n - 1) * delta_inv;
    s[n - 1] = h(n, n - 1) * delta_inv;

    // Update b_hat
    r(n - 1, n - 1) = c[n - 1] * r(n - 1, n - 1) + s[n - 1] * h(n, n - 1);
    b_hat[n] = -s[n - 1] * b_hat[n - 1];
    b_hat[n - 1] *= c[n - 1];
    rho = Base::Math::abs(b_hat[n]);

    rep_num = n;

    // Check for convergence
    if (rho / Base::Utility::avoid_zero_divide(b_norm, division_min) <
        decay_rate) {
      break;
    }
  }

  // Back substitution to solve
  for (std::size_t j = rep_num; j-- > 0;) {
    T temp = ZERO;
    for (std::size_t m = j + 1; m < rep_num; ++m) {
      temp += r(j, m) * y[m];
    }
    y[j] = (b_hat[j] - temp) /
           Base::Utility::avoid_zero_divide(r(j, j), division_min);
  }

  for (std::size_t i = 0; i < rep_num; ++i) {
    for (std::size_t j = 0; j < matrix_size; ++j) {
      x_dif[j] += y[i] * q(j, i);
    }
  }

  Vector<T, M> x;
  for (std::size_t i = 0; i < matrix_size; ++i) {
    x[i] = x_1[i] + x_dif[i];
  }

  return x;
}

/**
 * @brief Core function for GMRES K method for sparse matrices with size 1x1.
 *
 * This function implements the core logic of the GMRES K method for sparse
 * matrices, handling the case where the matrix size is exactly 1x1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param SA The compiled sparse matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @param matrix_size The size of the matrix (M).
 * @return The computed solution vector x.
 */
template <typename T, std::size_t M, typename RowIndices_A,
          typename RowPointers_A>
inline typename std::enable_if<(M <= 1), Vector<T, M>>::type
sparse_gmres_k_core(
    const CompiledSparseMatrix<T, M, M, RowIndices_A, RowPointers_A> &SA,
    const Vector<T, M> &b, const Vector<T, M> &x_1, const T &decay_rate,
    const T &division_min, T &rho, std::size_t &rep_num,
    const std::size_t &matrix_size) {
  static_assert(M == 1,
                "Matrix size must be exactly 1x1 for this specialization.");
  static_cast<void>(decay_rate);
  static_cast<void>(division_min);
  static_cast<void>(rep_num);
  static_cast<void>(x_1);
  static_cast<void>(rho);
  static_cast<void>(matrix_size);

  Vector<T, M> x;

  x[0] = b[0] / Base::Utility::avoid_zero_divide(
                    get_sparse_matrix_value<0, 0>(SA), division_min);

  return x;
}

} // namespace InverseOperation

/**
 * @brief GMRES K method for sparse matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a sparse matrix, b is a vector, and x_1 is an initial guess.
 * It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param SA The compiled sparse matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @return The computed solution vector x.
 */
template <typename T, std::size_t M, typename RowIndices_A,
          typename RowPointers_A>
inline Vector<T, M> sparse_gmres_k(
    const CompiledSparseMatrix<T, M, M, RowIndices_A, RowPointers_A> &SA,
    const Vector<T, M> &b, const Vector<T, M> &x_1, const T &decay_rate,
    const T &division_min, T &rho, std::size_t &rep_num) {

  return InverseOperation::sparse_gmres_k_core<T, M>(
      SA, b, x_1, decay_rate, division_min, rho, rep_num, M);
}

/**
 * @brief GMRES K method for sparse matrices with partitioning.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a sparse matrix, b is a vector, and x_1 is an initial guess.
 * It supports partitioning of the matrix and returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param SA The compiled sparse matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @param matrix_size The size of the matrix (M).
 * @return The computed solution vector x.
 */
template <typename T, std::size_t M, typename RowIndices_A,
          typename RowPointers_A>
inline Vector<T, M> sparse_gmres_k_partition(
    const CompiledSparseMatrix<T, M, M, RowIndices_A, RowPointers_A> &SA,
    const Vector<T, M> &b, const Vector<T, M> &x_1, const T &decay_rate,
    const T &division_min, T &rho, std::size_t &rep_num,
    const std::size_t &matrix_size) {

  return InverseOperation::sparse_gmres_k_core<T, M>(
      SA, b, x_1, decay_rate, division_min, rho, rep_num, matrix_size);
}

/**
 * @brief GMRES K method for sparse matrices with multiple right-hand side
 * vectors.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a sparse matrix, B is a matrix with multiple right-hand side
 * vectors, and X_1 is an initial guess. It returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam K The number of rows of matrix B.
 * @param SA The compiled sparse matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each row.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each row.
 */
template <typename T, std::size_t M, std::size_t K, typename RowIndices_A,
          typename RowPointers_A>
inline void sparse_gmres_k_matrix(
    const CompiledSparseMatrix<T, M, M, RowIndices_A, RowPointers_A> &SA,
    const Matrix<T, M, K> &B, Matrix<T, M, K> &X_1, const T &decay_rate,
    const T &division_min, std::array<T, K> &rho,
    std::array<std::size_t, K> &rep_num) {

  for (std::size_t i = 0; i < K; i++) {
    Vector<T, M> x = Base::Matrix::sparse_gmres_k(
        SA, B.get_row(i), X_1.get_row(i), decay_rate, division_min, rho[i],
        rep_num[i]);
    X_1.set_row(i, x);
  }
}

/**
 * @brief GMRES K method for sparse matrices with multiple right-hand side
 * vectors and partitioning.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a sparse matrix, B is a matrix with multiple right-hand side
 * vectors, and X_1 is an initial guess. It supports partitioning of the matrix
 * and returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam K The number of rows of matrix B.
 * @param SA The compiled sparse matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each row.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each row.
 * @param matrix_size The size of the matrix (M).
 */
template <typename T, std::size_t M, std::size_t K, typename RowIndices_A,
          typename RowPointers_A>
inline void sparse_gmres_k_partition_matrix(
    const CompiledSparseMatrix<T, M, M, RowIndices_A, RowPointers_A> &SA,
    const Matrix<T, M, K> &B, Matrix<T, M, K> &X_1, const T &decay_rate,
    const T &division_min, std::array<T, K> &rho,
    std::array<std::size_t, K> &rep_num, const std::size_t &matrix_size) {

  std::size_t repeat_number;
  if (matrix_size < K) {
    repeat_number = matrix_size;
  } else {
    repeat_number = K;
  }

  for (std::size_t i = 0; i < repeat_number; i++) {
    Vector<T, M> x = Base::Matrix::sparse_gmres_k_partition(
        SA, B.get_row(i), X_1.get_row(i), decay_rate, division_min, rho[i],
        rep_num[i], matrix_size);
    X_1.set_row(i, x);
  }
}

/* Sparse GMRES K for rectangular matrix */

/**
 * @brief GMRES K method for rectangular matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a rectangular matrix (M x N), b is a vector, and x_1 is an
 * initial guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The number of rows in the matrix A.
 * @tparam N The number of columns in the matrix A.
 * @param In_SA The compiled sparse matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @return The computed solution vector x.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline Vector<T, N> sparse_gmres_k_rect(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &In_SA,
    const Vector<T, M> &b, const Vector<T, N> &x_1, const T &decay_rate,
    const T &division_min, T &rho, std::size_t &rep_num) {
  static_assert(M > 1, "Matrix size must be equal or larger than 2x2.");
  static_assert(M > N, "Column number must be larger than row number.");

  Matrix<T, N, N> r;
  Vector<T, N + 1> b_hat;
  b_hat[0] = static_cast<T>(1);
  Matrix<T, N, N + 1> q;
  Matrix<T, N + 1, N> h;
  Vector<T, N> c;
  Vector<T, N> s;
  Vector<T, N> y;
  Vector<T, N> x_dif;
  T ZERO = static_cast<T>(0);

  // b - Ax
  Vector<T, M> b_ax_temp = b - (In_SA * x_1);

  Matrix<T, N, N> A = Base::Matrix::matrix_multiply_ATranspose_mul_SparseB(
      Base::Matrix::output_dense_matrix(In_SA), In_SA);

  ColVector<T, M> b_ax_temp_col(b_ax_temp);
  ColVector<T, N> b_SA =
      Base::Matrix::colVector_a_mul_SparseB(b_ax_temp_col, In_SA);
  Vector<T, N> b_ax = b_SA.transpose();

  // Normalize b_Ax
  T b_norm = b_ax.norm(division_min);
  for (std::size_t i = 0; i < N; ++i) {
    q(i, 0) = b_ax[i] / Base::Utility::avoid_zero_divide(b_norm, division_min);
  }
  b_hat[0] = b_norm;

  for (std::size_t n = 1; n <= N; n++) {
    // Generate orthogonal basis
    Vector<T, N> v;
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        v[i] += A(i, j) * q(j, n - 1);
      }
    }

    for (std::size_t j = 0; j < n; ++j) {
      h(j, n - 1) = 0;
      for (std::size_t i = 0; i < N; ++i) {
        h(j, n - 1) += q(i, j) * v[i];
      }
      for (std::size_t i = 0; i < N; ++i) {
        v[i] -= h(j, n - 1) * q(i, j);
      }
    }

    if (n < N) {
      h(n, n - 1) = v.norm(division_min);
      for (std::size_t i = 0; i < N; ++i) {
        q(i, n) =
            v[i] / Base::Utility::avoid_zero_divide(h(n, n - 1), division_min);
      }
    }

    // Givens rotation for QR decomposition
    r(0, n - 1) = h(0, n - 1);
    if (n >= 2) {
      for (std::size_t j = 1; j < n; ++j) {
        T gamma = c[j - 1] * r(j - 1, n - 1) + s[j - 1] * h(j, n - 1);
        r(j, n - 1) = -s[j - 1] * r(j - 1, n - 1) + c[j - 1] * h(j, n - 1);
        r(j - 1, n - 1) = gamma;
      }
    }

    T delta_inv = Base::Math::rsqrt<T>(r(n - 1, n - 1) * r(n - 1, n - 1) +
                                           h(n, n - 1) * h(n, n - 1),
                                       division_min);

    c[n - 1] = r(n - 1, n - 1) * delta_inv;
    s[n - 1] = h(n, n - 1) * delta_inv;

    // Update b_hat
    r(n - 1, n - 1) = c[n - 1] * r(n - 1, n - 1) + s[n - 1] * h(n, n - 1);
    b_hat[n] = -s[n - 1] * b_hat[n - 1];
    b_hat[n - 1] *= c[n - 1];
    rho = Base::Math::abs(b_hat[n]);

    rep_num = n;

    // Check for convergence
    if (rho / Base::Utility::avoid_zero_divide(b_norm, division_min) <
        decay_rate) {
      break;
    }
  }

  // Back substitution to solve
  for (std::size_t j = rep_num; j-- > 0;) {
    T temp = ZERO;
    for (std::size_t m = j + 1; m < rep_num; ++m) {
      temp += r(j, m) * y[m];
    }
    y[j] = (b_hat[j] - temp) /
           Base::Utility::avoid_zero_divide(r(j, j), division_min);
  }

  for (std::size_t i = 0; i < rep_num; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      x_dif[j] += y[i] * q(j, i);
    }
  }

  Vector<T, N> x;
  for (std::size_t i = 0; i < N; ++i) {
    x[i] = x_1[i] + x_dif[i];
  }

  return x;
}

/**
 * @brief GMRES K method for rectangular matrices with multiple right-hand side
 * vectors.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a rectangular matrix (M x N), B is a matrix with multiple right-
 * hand side vectors, and X_1 is an initial guess. It returns the solution
 * matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The number of columns in the matrix A.
 * @tparam N The number of rows in the matrix A.
 * @tparam K The number of rows of matrix B.
 * @param In_SA The compiled sparse matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each row.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each row.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A>
inline void sparse_gmres_k_rect_matrix(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &In_SA,
    const Matrix<T, M, K> &B, Matrix<T, N, K> &X_1, const T &decay_rate,
    const T &division_min, std::array<T, K> &rho,
    std::array<std::size_t, K> &rep_num) {

  for (std::size_t i = 0; i < K; i++) {
    Vector<T, N> x = Base::Matrix::sparse_gmres_k_rect(
        In_SA, B.get_row(i), X_1.get_row(i), decay_rate, division_min, rho[i],
        rep_num[i]);
    X_1.set_row(i, x);
  }
}

/**
 * @brief GMRES K method for rectangular matrices with multiple right-hand side
 * vectors and partitioning.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a rectangular matrix (M x N), B is a matrix with multiple right-
 * hand side vectors, and X_1 is an initial guess. It supports partitioning of
 * the matrix and returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The number of columns in the matrix A.
 * @tparam N The number of rows in the matrix A.
 * @tparam RowIndices_A The type representing row indices of the sparse matrix.
 * @tparam RowPointers_A The type representing row pointers of the sparse
 * matrix.
 * @param In_SA The compiled sparse matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each row.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each row.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void sparse_gmres_k_rect_matrix(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &In_SA,
    const DiagMatrix<T, M> &B, Matrix<T, N, M> &X_1, const T &decay_rate,
    const T &division_min, std::array<T, M> &rho,
    std::array<std::size_t, M> &rep_num) {

  for (std::size_t i = 0; i < M; i++) {
    Vector<T, N> x = Base::Matrix::sparse_gmres_k_rect(
        In_SA, B.get_row(i), X_1.get_row(i), decay_rate, division_min, rho[i],
        rep_num[i]);
    X_1.set_row(i, x);
  }
}

/* Sparse GMRES K for matrix inverse */

/**
 * @brief GMRES K method for sparse matrix inverse.
 *
 * This function computes the inverse of a sparse matrix using the GMRES K
 * method, where A is a sparse matrix, decay_rate is the convergence criterion,
 * division_min is a small value to avoid division by zero, rho stores the
 * residual norms, rep_num stores the number of iterations performed, and X_1 is
 * an initial guess for the solution matrix.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam RowIndices_A The type representing row indices of the sparse matrix.
 * @tparam RowPointers_A The type representing row pointers of the sparse
 * matrix.
 * @param In_A The compiled sparse matrix A.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each row.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each row.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @return The computed inverse matrix X.
 */
template <typename T, std::size_t M, typename RowIndices_A,
          typename RowPointers_A>
inline Matrix<T, M, M> sparse_gmres_k_matrix_inv(
    const CompiledSparseMatrix<T, M, M, RowIndices_A, RowPointers_A> In_A,
    T decay_rate, const T &division_min, std::array<T, M> &rho,
    std::array<std::size_t, M> &rep_num, const Matrix<T, M, M> X_1) {
  Matrix<T, M, M> B = Matrix<T, M, M>::identity();
  Matrix<T, M, M> X;

  for (std::size_t i = 0; i < M; i++) {
    Vector<T, M> x;

    x = Base::Matrix::sparse_gmres_k(In_A, B.get_row(i), X_1.get_row(i),
                                     decay_rate, division_min, rho[i],
                                     rep_num[i]);
    X.set_row(i, x);
  }

  return X;
}

/* Complex GMRES K */

/**
 * @brief GMRES K method for complex matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a complex matrix, b is a complex vector, and x_1 is an initial
 * guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The complex matrix A.
 * @param b The right-hand side complex vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @return The computed solution vector x.
 */
template <typename T, std::size_t M>
inline typename std::enable_if<(M > 1), Vector<Complex<T>, M>>::type
complex_gmres_k(const Matrix<Complex<T>, M, M> &A,
                const Vector<Complex<T>, M> &b,
                const Vector<Complex<T>, M> &x_1, T decay_rate,
                const T &division_min, T &rho, std::size_t &rep_num) {
  static_assert(M > 1, "Matrix size must be equal or larger than 2x2.");

  Matrix<Complex<T>, M, M> r;
  Vector<Complex<T>, M + 1> b_hat;
  b_hat[0] = static_cast<T>(1);
  Matrix<Complex<T>, M, M + 1> q;
  Matrix<Complex<T>, M + 1, M> h;
  Vector<Complex<T>, M> c;
  Vector<Complex<T>, M> s;
  Vector<Complex<T>, M> y;
  Vector<Complex<T>, M> x_dif;
  Complex<T> ZERO = static_cast<T>(0);

  // b - Ax
  Vector<Complex<T>, M> b_ax;
  for (std::size_t i = 0; i < M; ++i) {
    Complex<T> sum(static_cast<T>(0));
    for (std::size_t j = 0; j < M; ++j) {
      sum += A(i, j) * x_1[j];
    }
    b_ax[i] = b[i] - sum;
  }

  // Normalize b_Ax
  T b_norm = Base::Matrix::complex_vector_norm(b_ax, division_min);
  for (std::size_t i = 0; i < M; ++i) {
    q(i, 0) = b_ax[i] / Base::Utility::avoid_zero_divide(b_norm, division_min);
  }
  b_hat[0] = b_norm;

  for (std::size_t n = 1; n <= M; n++) {
    // Generate orthogonal basis
    Vector<Complex<T>, M> v;
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < M; ++j) {
        v[i] += A(i, j) * q(j, n - 1);
      }
    }

    for (std::size_t j = 0; j < n; ++j) {
      h(j, n - 1) = 0;
      for (std::size_t i = 0; i < M; ++i) {
        h(j, n - 1) += Base::Matrix::complex_conjugate(q(i, j)) * v[i];
      }
      for (std::size_t i = 0; i < M; ++i) {
        v[i] -= h(j, n - 1) * q(i, j);
      }
    }

    if (n < M) {
      h(n, n - 1) = Base::Matrix::complex_vector_norm(v, division_min);
      for (std::size_t i = 0; i < M; ++i) {
        q(i, n) = Base::Matrix::complex_divide(v[i], h(n, n - 1), division_min);
      }
    }

    // Givens rotation for QR decomposition
    r(0, n - 1) = h(0, n - 1);
    if (n >= 2) {
      for (std::size_t j = 1; j < n; ++j) {
        Complex<T> gamma = c[j - 1] * r(j - 1, n - 1) + s[j - 1] * h(j, n - 1);
        r(j, n - 1) =
            -Base::Matrix::complex_conjugate(s[j - 1]) * r(j - 1, n - 1) +
            c[j - 1] * h(j, n - 1);
        r(j - 1, n - 1) = gamma;
      }
    }

    T delta_inv =
        Base::Math::rsqrt<T>(Base::Matrix::complex_abs_sq(r(n - 1, n - 1)) +
                                 Base::Matrix::complex_abs_sq(h(n, n - 1)),
                             division_min);

    c[n - 1] = r(n - 1, n - 1) * delta_inv;
    s[n - 1] = h(n, n - 1) * delta_inv;

    // Update b_hat
    r(n - 1, n - 1) = c[n - 1] * r(n - 1, n - 1) + s[n - 1] * h(n, n - 1);
    b_hat[n] = -Base::Matrix::complex_conjugate(s[n - 1]) * b_hat[n - 1];
    b_hat[n - 1] = b_hat[n - 1] * c[n - 1];
    rho = Base::Matrix::complex_abs_sq(b_hat[n]);

    rep_num = n;

    // Check for convergence
    if (rho / Base::Utility::avoid_zero_divide(b_norm * b_norm, division_min) <
        decay_rate) {
      break;
    }
  }

  // Back substitution to solve
  for (std::size_t j = rep_num; j-- > 0;) {
    Complex<T> temp(ZERO);
    for (std::size_t m = j + 1; m < rep_num; ++m) {
      temp += r(j, m) * y[m];
    }
    y[j] = Base::Matrix::complex_divide(b_hat[j] - temp, r(j, j), division_min);
  }

  for (std::size_t i = 0; i < rep_num; ++i) {
    for (std::size_t j = 0; j < M; ++j) {
      x_dif[j] += y[i] * q(j, i);
    }
  }

  Vector<Complex<T>, M> x;
  for (std::size_t i = 0; i < M; ++i) {
    x[i] = x_1[i] + x_dif[i];
  }

  return x;
}

/**
 * @brief GMRES K method for complex matrices with size 1x1.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a complex matrix of size 1x1, b is a complex vector, and x_1 is
 * an initial guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The complex matrix A.
 * @param b The right-hand side complex vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @return The computed solution vector x.
 */
template <typename T, std::size_t M>
inline typename std::enable_if<(M <= 1), Vector<Complex<T>, M>>::type
complex_gmres_k(const Matrix<Complex<T>, M, M> &A,
                const Vector<Complex<T>, M> &b,
                const Vector<Complex<T>, M> &x_1, T decay_rate,
                const T &division_min, T &rho, std::size_t &rep_num) {
  static_assert(M == 1,
                "Matrix size must be exactly 1x1 for this specialization.");
  static_cast<void>(decay_rate);
  static_cast<void>(division_min);
  static_cast<void>(rep_num);
  static_cast<void>(x_1);

  Vector<Complex<T>, M> x;

  x[0] =
      Base::Matrix::complex_divide(b[0], A.template get<0, 0>(), division_min);

  return x;
}

/**
 * @brief GMRES K method for complex matrices with multiple right-hand side
 * vectors.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a complex matrix, B is a matrix with multiple right-hand side
 * vectors, and X_1 is an initial guess. It returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam K The number of rows of matrix B.
 * @param A The complex matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each row.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each row.
 */
template <typename T, std::size_t M, std::size_t K>
inline void complex_gmres_k_matrix(const Matrix<Complex<T>, M, M> &A,
                                   const Matrix<Complex<T>, M, K> &B,
                                   Matrix<Complex<T>, M, K> &X_1,
                                   const T &decay_rate, const T &division_min,
                                   std::array<T, K> &rho,
                                   std::array<std::size_t, K> &rep_num) {

  for (std::size_t i = 0; i < K; i++) {
    Vector<Complex<T>, M> x = Base::Matrix::complex_gmres_k(
        A, B.get_row(i), X_1.get_row(i), decay_rate, division_min, rho[i],
        rep_num[i]);
    X_1.set_row(i, x);
  }
}

/**
 * @brief GMRES K method for complex matrices with multiple right-hand side
 * vectors and partitioning.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a complex matrix, B is a matrix with multiple right-hand side
 * vectors, and X_1 is an initial guess. It supports partitioning of the matrix
 * and returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The complex matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each row.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each row.
 */
template <typename T, std::size_t M>
inline void complex_gmres_k_matrix(const Matrix<Complex<T>, M, M> &A,
                                   const DiagMatrix<Complex<T>, M> &B,
                                   Matrix<Complex<T>, M, M> &X_1,
                                   const T &decay_rate, const T &division_min,
                                   std::array<T, M> &rho,
                                   std::array<std::size_t, M> &rep_num) {

  for (std::size_t i = 0; i < M; i++) {
    Vector<Complex<T>, M> x = Base::Matrix::complex_gmres_k(
        A, B.get_row(i), X_1.get_row(i), decay_rate, division_min, rho[i],
        rep_num[i]);
    X_1.set_row(i, x);
  }
}

/**
 * @brief GMRES K method for complex sparse matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a complex sparse matrix, b is a complex vector, and x_1 is an
 * initial guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param SA The compiled sparse matrix A.
 * @param b The right-hand side complex vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @return The computed solution vector x.
 */
template <typename T, std::size_t M, typename RowIndices_A,
          typename RowPointers_A>
inline typename std::enable_if<(M > 1), Vector<Complex<T>, M>>::type
complex_sparse_gmres_k(
    const CompiledSparseMatrix<Complex<T>, M, M, RowIndices_A, RowPointers_A>
        &SA,
    const Vector<Complex<T>, M> &b, const Vector<Complex<T>, M> &x_1,
    T decay_rate, const T &division_min, T &rho, std::size_t &rep_num) {
  static_assert(M > 1, "Matrix size must be equal or larger than 2x2.");

  Matrix<Complex<T>, M, M> r;
  Vector<Complex<T>, M + 1> b_hat;
  b_hat[0] = static_cast<T>(1);
  Matrix<Complex<T>, M, M + 1> q;
  Matrix<Complex<T>, M + 1, M> h;
  Vector<Complex<T>, M> c;
  Vector<Complex<T>, M> s;
  Vector<Complex<T>, M> y;
  Vector<Complex<T>, M> x_dif;
  Complex<T> ZERO;

  // b - Ax
  Vector<Complex<T>, M> b_ax = b - (SA * x_1);

  // Normalize b_Ax
  T b_norm = Base::Matrix::complex_vector_norm(b_ax, division_min);
  for (std::size_t i = 0; i < M; ++i) {
    q(i, 0) = b_ax[i] / Base::Utility::avoid_zero_divide(b_norm, division_min);
  }
  b_hat[0] = b_norm;

  for (std::size_t n = 1; n <= M; n++) {
    // Generate orthogonal basis
    Vector<Complex<T>, M> v = SA * q.create_row_vector(n - 1);

    for (std::size_t j = 0; j < n; ++j) {
      h(j, n - 1) = 0;
      for (std::size_t i = 0; i < M; ++i) {
        h(j, n - 1) += Base::Matrix::complex_conjugate(q(i, j)) * v[i];
      }
      for (std::size_t i = 0; i < M; ++i) {
        v[i] -= h(j, n - 1) * q(i, j);
      }
    }

    if (n < M) {
      h(n, n - 1) = Base::Matrix::complex_vector_norm(v, division_min);
      for (std::size_t i = 0; i < M; ++i) {
        q(i, n) = Base::Matrix::complex_divide(v[i], h(n, n - 1), division_min);
      }
    }

    // Givens rotation for QR decomposition
    r(0, n - 1) = h(0, n - 1);
    if (n >= 2) {
      for (std::size_t j = 1; j < n; ++j) {
        Complex<T> gamma = c[j - 1] * r(j - 1, n - 1) + s[j - 1] * h(j, n - 1);
        r(j, n - 1) =
            -Base::Matrix::complex_conjugate(s[j - 1]) * r(j - 1, n - 1) +
            c[j - 1] * h(j, n - 1);
        r(j - 1, n - 1) = gamma;
      }
    }

    T delta_inv =
        Base::Math::rsqrt<T>(Base::Matrix::complex_abs_sq(r(n - 1, n - 1)) +
                                 Base::Matrix::complex_abs_sq(h(n, n - 1)),
                             division_min);

    c[n - 1] = r(n - 1, n - 1) * delta_inv;
    s[n - 1] = h(n, n - 1) * delta_inv;

    // Update b_hat
    r(n - 1, n - 1) = c[n - 1] * r(n - 1, n - 1) + s[n - 1] * h(n, n - 1);
    b_hat[n] = -Base::Matrix::complex_conjugate(s[n - 1]) * b_hat[n - 1];
    b_hat[n - 1] *= c[n - 1];
    rho = Base::Matrix::complex_abs_sq(b_hat[n]);

    rep_num = n;

    // Check for convergence
    if (rho / Base::Utility::avoid_zero_divide(b_norm, division_min) <
        decay_rate) {
      break;
    }
  }

  // Back substitution to solve
  for (std::size_t j = rep_num; j-- > 0;) {
    Complex<T> temp = ZERO;
    for (std::size_t m = j + 1; m < rep_num; ++m) {
      temp += r(j, m) * y[m];
    }
    y[j] = Base::Matrix::complex_divide(b_hat[j] - temp, r(j, j), division_min);
  }

  for (std::size_t i = 0; i < rep_num; ++i) {
    for (std::size_t j = 0; j < M; ++j) {
      x_dif[j] += y[i] * q(j, i);
    }
  }

  Vector<Complex<T>, M> x;
  for (std::size_t i = 0; i < M; ++i) {
    x[i] = x_1[i] + x_dif[i];
  }

  return x;
}

/**
 * @brief GMRES K method for complex sparse matrices with size 1x1.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a complex sparse matrix of size 1x1, b is a complex vector, and
 * x_1 is an initial guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param SA The compiled sparse matrix A.
 * @param b The right-hand side complex vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norm.
 * @param rep_num Output parameter to store the number of iterations performed.
 * @return The computed solution vector x.
 */
template <typename T, std::size_t M, typename RowIndices_A,
          typename RowPointers_A>
inline typename std::enable_if<(M <= 1), Vector<Complex<T>, M>>::type
complex_sparse_gmres_k(
    const CompiledSparseMatrix<Complex<T>, M, M, RowIndices_A, RowPointers_A>
        &SA,
    const Vector<Complex<T>, M> &b, const Vector<Complex<T>, M> &x_1,
    T decay_rate, const T &division_min, T &rho, std::size_t &rep_num) {
  static_assert(M == 1,
                "Matrix size must be exactly 1x1 for this specialization.");
  static_cast<void>(decay_rate);
  static_cast<void>(division_min);
  static_cast<void>(rep_num);
  static_cast<void>(x_1);

  Vector<Complex<T>, M> x;

  x[0] =
      Base::Matrix::complex_divide(b[0], SA.template get<0, 0>(), division_min);

  return x;
}

/**
 * @brief GMRES K method for complex sparse matrices with multiple right-hand
 * side vectors.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a complex sparse matrix, B is a matrix with multiple right-hand
 * side vectors, and X_1 is an initial guess. It returns the solution matrix
 * X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam K The number of rows of matrix B.
 * @param In_A The compiled sparse matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 */
template <typename T, std::size_t M, typename RowIndices_A,
          typename RowPointers_A>
inline Matrix<Complex<T>, M, M> complex_sparse_gmres_k_matrix(
    const CompiledSparseMatrix<Complex<T>, M, M, RowIndices_A, RowPointers_A>
        In_A,
    T decay_rate, const T &division_min, const Matrix<Complex<T>, M, M> X_1) {
  Matrix<Complex<T>, M, M> B = Matrix<Complex<T>, M, M>::identity();
  Matrix<Complex<T>, M, M> X;
  Vector<T, M> rho_vec;
  Vector<std::size_t, M> rep_num_vec;

  for (std::size_t i = 0; i < M; i++) {
    Vector<Complex<T>, M> x;

    x = Base::Matrix::complex_sparse_gmres_k(In_A, B.get_row(i), X_1.get_row(i),
                                             decay_rate, division_min,
                                             rho_vec[i], rep_num_vec[i]);

    X.set_row(i, x);
  }

  return X;
}

/**
 * @brief GMRES K method for complex matrix inverse.
 *
 * This function computes the inverse of a complex matrix using the GMRES K
 * method, where A is a complex matrix, decay_rate is the convergence criterion,
 * division_min is a small value to avoid division by zero, rho stores the
 * residual norms, rep_num stores the number of iterations performed, and X_1 is
 * an initial guess for the solution matrix.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param In_A The complex matrix A.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each row.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each row.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @return The computed inverse matrix X.
 */
template <typename T, std::size_t M>
inline Matrix<Complex<T>, M, M> complex_gmres_k_matrix_inv(
    const Matrix<Complex<T>, M, M> In_A, const T &decay_rate,
    const T &division_min, std::array<T, M> &rho,
    std::array<std::size_t, M> &rep_num, const Matrix<Complex<T>, M, M> X_1) {
  Matrix<Complex<T>, M, M> B = Matrix<Complex<T>, M, M>::identity();
  Matrix<Complex<T>, M, M> X;

  for (std::size_t i = 0; i < M; i++) {
    Vector<Complex<T>, M> x;

    x = Base::Matrix::complex_gmres_k(In_A, B.get_row(i), X_1.get_row(i),
                                      decay_rate, division_min, rho[i],
                                      rep_num[i]);
    X.set_row(i, x);
  }

  return X;
}

/**
 * @brief GMRES K method for complex sparse matrix inverse.
 *
 * This function computes the inverse of a complex sparse matrix using the GMRES
 * K method, where A is a complex sparse matrix, decay_rate is the convergence
 * criterion, division_min is a small value to avoid division by zero, rho
 * stores the residual norms, rep_num stores the number of iterations performed,
 * and X_1 is an initial guess for the solution matrix.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param In_A The compiled sparse matrix A.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param rho Output parameter to store the residual norms for each row.
 * @param rep_num Output parameter to store the number of iterations performed
 * for each row.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @return The computed inverse matrix X.
 */
template <typename T, std::size_t M, typename RowIndices_A,
          typename RowPointers_A>
inline Matrix<Complex<T>, M, M> complex_sparse_gmres_k_matrix_inv(
    const CompiledSparseMatrix<Complex<T>, M, M, RowIndices_A, RowPointers_A>
        In_A,
    T decay_rate, const T &division_min, std::array<T, M> &rho,
    std::array<std::size_t, M> &rep_num, const Matrix<Complex<T>, M, M> X_1) {
  Matrix<Complex<T>, M, M> B = Matrix<Complex<T>, M, M>::identity();
  Matrix<Complex<T>, M, M> X;

  for (std::size_t i = 0; i < M; i++) {
    Vector<Complex<T>, M> x;

    x = Base::Matrix::complex_sparse_gmres_k(In_A, B.get_row(i), X_1.get_row(i),
                                             decay_rate, division_min, rho[i],
                                             rep_num[i]);
    X.set_row(i, x);
  }

  return X;
}

/* Diag Matrix inverse */

/**
 * @brief Inverse of a diagonal matrix.
 *
 * This function computes the inverse of a diagonal matrix, where each element
 * is divided by a small value to avoid division by zero.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The size of the diagonal matrix (M x M).
 * @param input The input diagonal matrix to be inverted.
 * @param division_min A small value to avoid division by zero.
 * @return The inverted diagonal matrix.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> inverse_diag_matrix(const DiagMatrix<T, M> &input,
                                            const T &division_min) {
  DiagMatrix<T, M> result;

  for (std::size_t i = 0; i < M; i++) {
    result[i] = static_cast<T>(1) /
                Base::Utility::avoid_zero_divide(input.data[i], division_min);
  }

  return result;
}

/**
 * @brief Inverse of a complex diagonal matrix.
 *
 * This function computes the inverse of a complex diagonal matrix, where each
 * element is divided by a small value to avoid division by zero.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The size of the diagonal matrix (M x M).
 * @param input The input complex diagonal matrix to be inverted.
 * @param division_min A small value to avoid division by zero.
 * @return The inverted complex diagonal matrix.
 */
template <typename T, std::size_t M>
inline DiagMatrix<Complex<T>, M>
inverse_complex_diag_matrix(const DiagMatrix<Complex<T>, M> &input,
                            const T &division_min) {
  DiagMatrix<Complex<T>, M> result;

  for (std::size_t i = 0; i < M; i++) {
    result[i] = Base::Matrix::complex_divide(static_cast<T>(1), input.data[i],
                                             division_min);
  }

  return result;
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_INVERSE_HPP__
