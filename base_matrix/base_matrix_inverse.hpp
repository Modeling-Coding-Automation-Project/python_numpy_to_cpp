#ifndef BASE_MATRIX_INVERSE_HPP
#define BASE_MATRIX_INVERSE_HPP

#include "base_matrix_complex.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_vector.hpp"
#include <cmath>
#include <cstddef>

namespace Base {
namespace Matrix {

/* GMRES K */
template <typename T, std::size_t M>
Vector<T, M> gmres_k(const Matrix<T, M, M> &A, const Vector<T, M> &b,
                     const Vector<T, M> &x_1, T decay_rate, T division_min,
                     T &rho, std::size_t &rep_num) {
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
  for (std::size_t i = 0; i < M; ++i) {
    T sum = static_cast<T>(0);
    for (std::size_t j = 0; j < M; ++j) {
      sum += A(i, j) * x_1[j];
    }
    b_ax[i] = b[i] - sum;
  }

  // Normalize b_Ax
  T b_norm = b_ax.norm();
  for (std::size_t i = 0; i < M; ++i) {
    q(i, 0) = b_ax[i] / avoid_zero_divide(b_norm, division_min);
  }
  b_hat[0] = b_norm;

  for (std::size_t n = 1; n <= M; n++) {
    // Generate orthogonal basis
    Vector<T, M> v;
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < M; ++j) {
        v[i] += A(i, j) * q(j, n - 1);
      }
    }

    for (std::size_t j = 0; j < n; ++j) {
      h(j, n - 1) = 0;
      for (std::size_t i = 0; i < M; ++i) {
        h(j, n - 1) += q(i, j) * v[i];
      }
      for (std::size_t i = 0; i < M; ++i) {
        v[i] -= h(j, n - 1) * q(i, j);
      }
    }

    if (n < M) {
      h(n, n - 1) = v.norm();
      for (std::size_t i = 0; i < M; ++i) {
        q(i, n) = v[i] / avoid_zero_divide(h(n, n - 1), division_min);
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

    T delta = avoid_zero_divide(std::sqrt(r(n - 1, n - 1) * r(n - 1, n - 1) +
                                          h(n, n - 1) * h(n, n - 1)),
                                division_min);

    c[n - 1] = r(n - 1, n - 1) / delta;
    s[n - 1] = h(n, n - 1) / delta;

    // Update b_hat
    r(n - 1, n - 1) = c[n - 1] * r(n - 1, n - 1) + s[n - 1] * h(n, n - 1);
    b_hat[n] = -s[n - 1] * b_hat[n - 1];
    b_hat[n - 1] *= c[n - 1];
    rho = std::abs(b_hat[n]);

    rep_num = n;

    // Check for convergence
    if (rho / avoid_zero_divide(b_norm, division_min) < decay_rate) {
      break;
    }
  }

  // Back substitution to solve
  for (std::size_t j = rep_num; j-- > 0;) {
    T temp = ZERO;
    for (std::size_t m = j + 1; m < rep_num; ++m) {
      temp += r(j, m) * y[m];
    }
    y[j] = (b_hat[j] - temp) / avoid_zero_divide(r(j, j), division_min);
  }

  for (std::size_t i = 0; i < rep_num; ++i) {
    for (std::size_t j = 0; j < M; ++j) {
      x_dif[j] += y[i] * q(j, i);
    }
  }

  Vector<T, M> x;
  for (std::size_t i = 0; i < M; ++i) {
    x[i] = x_1[i] + x_dif[i];
  }

  return x;
}

/* GMRES K for rectangular matrix */
template <typename T, std::size_t M, std::size_t N>
Vector<T, N> gmres_k_rect(const Matrix<T, M, N> &In_A, const Vector<T, M> &b,
                          const Vector<T, N> &x_1, T decay_rate, T division_min,
                          T &rho, std::size_t &rep_num) {
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

  Matrix<T, N, N> A = matrix_multiply_AT_mul_B(In_A, In_A);
  Vector<T, N> b_ax = matrix_multiply_AT_mul_b(In_A, b_ax_temp);

  // Normalize b_Ax
  T b_norm = b_ax.norm();
  for (std::size_t i = 0; i < N; ++i) {
    q(i, 0) = b_ax[i] / avoid_zero_divide(b_norm, division_min);
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
      h(n, n - 1) = v.norm();
      for (std::size_t i = 0; i < N; ++i) {
        q(i, n) = v[i] / avoid_zero_divide(h(n, n - 1), division_min);
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

    T delta = avoid_zero_divide(std::sqrt(r(n - 1, n - 1) * r(n - 1, n - 1) +
                                          h(n, n - 1) * h(n, n - 1)),
                                division_min);

    c[n - 1] = r(n - 1, n - 1) / delta;
    s[n - 1] = h(n, n - 1) / delta;

    // Update b_hat
    r(n - 1, n - 1) = c[n - 1] * r(n - 1, n - 1) + s[n - 1] * h(n, n - 1);
    b_hat[n] = -s[n - 1] * b_hat[n - 1];
    b_hat[n - 1] *= c[n - 1];
    rho = std::abs(b_hat[n]);

    rep_num = n;

    // Check for convergence
    if (rho / avoid_zero_divide(b_norm, division_min) < decay_rate) {
      break;
    }
  }

  // Back substitution to solve
  for (std::size_t j = rep_num; j-- > 0;) {
    T temp = ZERO;
    for (std::size_t m = j + 1; m < rep_num; ++m) {
      temp += r(j, m) * y[m];
    }
    y[j] = (b_hat[j] - temp) / avoid_zero_divide(r(j, j), division_min);
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

/* GMRES K for matrix inverse */
template <typename T, std::size_t M>
Matrix<T, M, M> gmres_k_matrix_inv(const Matrix<T, M, M> In_A, T decay_rate,
                                   T division_min, const Matrix<T, M, M> X_1) {
  Matrix<T, M, M> B = Matrix<T, M, M>::identity();
  Matrix<T, M, M> X;
  Vector<T, M> rho_vec;
  Vector<std::size_t, M> rep_num_vec;

  for (std::size_t i = 0; i < M; i++) {
    Vector<T, M> x;

    x = gmres_k(In_A, B.get_row(i), X_1.get_row(i), decay_rate, division_min,
                rho_vec[i], rep_num_vec[i]);
    X.set_row(i, x);
  }

  return X;
}

/* Sparse GMRES K */
template <typename T, std::size_t M, std::size_t V>
Vector<T, M> sparse_gmres_k(const SparseMatrix<T, M, M, V> &SA,
                            const Vector<T, M> &b, const Vector<T, M> &x_1,
                            T decay_rate, T division_min, T &rho,
                            std::size_t &rep_num) {
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
  Vector<T, M> b_ax = b - (SA * x_1);

  // Normalize b_Ax
  T b_norm = b_ax.norm();
  for (std::size_t i = 0; i < M; ++i) {
    q(i, 0) = b_ax[i] / avoid_zero_divide(b_norm, division_min);
  }
  b_hat[0] = b_norm;

  for (std::size_t n = 1; n <= M; n++) {
    // Generate orthogonal basis
    Vector<T, M> v = SA * q.create_row_vector(n - 1);

    for (std::size_t j = 0; j < n; ++j) {
      h(j, n - 1) = 0;
      for (std::size_t i = 0; i < M; ++i) {
        h(j, n - 1) += q(i, j) * v[i];
      }
      for (std::size_t i = 0; i < M; ++i) {
        v[i] -= h(j, n - 1) * q(i, j);
      }
    }

    if (n < M) {
      h(n, n - 1) = v.norm();
      for (std::size_t i = 0; i < M; ++i) {
        q(i, n) = v[i] / avoid_zero_divide(h(n, n - 1), division_min);
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

    T delta = avoid_zero_divide(std::sqrt(r(n - 1, n - 1) * r(n - 1, n - 1) +
                                          h(n, n - 1) * h(n, n - 1)),
                                division_min);

    c[n - 1] = r(n - 1, n - 1) / delta;
    s[n - 1] = h(n, n - 1) / delta;

    // Update b_hat
    r(n - 1, n - 1) = c[n - 1] * r(n - 1, n - 1) + s[n - 1] * h(n, n - 1);
    b_hat[n] = -s[n - 1] * b_hat[n - 1];
    b_hat[n - 1] *= c[n - 1];
    rho = std::abs(b_hat[n]);

    rep_num = n;

    // Check for convergence
    if (rho / avoid_zero_divide(b_norm, division_min) < decay_rate) {
      break;
    }
  }

  // Back substitution to solve
  for (std::size_t j = rep_num; j-- > 0;) {
    T temp = ZERO;
    for (std::size_t m = j + 1; m < rep_num; ++m) {
      temp += r(j, m) * y[m];
    }
    y[j] = (b_hat[j] - temp) / avoid_zero_divide(r(j, j), division_min);
  }

  for (std::size_t i = 0; i < rep_num; ++i) {
    for (std::size_t j = 0; j < M; ++j) {
      x_dif[j] += y[i] * q(j, i);
    }
  }

  Vector<T, M> x;
  for (std::size_t i = 0; i < M; ++i) {
    x[i] = x_1[i] + x_dif[i];
  }

  return x;
}

/* Sparse GMRES K for rectangular matrix */
template <typename T, std::size_t M, std::size_t N, std::size_t V>
Vector<T, N> sparse_gmres_k_rect(const SparseMatrix<T, M, N, V> &In_SA,
                                 const Vector<T, M> &b, const Vector<T, N> &x_1,
                                 T decay_rate, T division_min, T &rho,
                                 std::size_t &rep_num) {
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

  Matrix<T, N, N> A =
      matrix_multiply_ATranspose_mul_SparseB(In_SA.create_dense(), In_SA);

  ColVector<T, M> b_ax_temp_col(b_ax_temp);
  ColVector<T, N> b_SA = colVector_a_mul_SparseB(b_ax_temp_col, In_SA);
  Vector<T, N> b_ax = b_SA.transpose();

  // Normalize b_Ax
  T b_norm = b_ax.norm();
  for (std::size_t i = 0; i < N; ++i) {
    q(i, 0) = b_ax[i] / avoid_zero_divide(b_norm, division_min);
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
      h(n, n - 1) = v.norm();
      for (std::size_t i = 0; i < N; ++i) {
        q(i, n) = v[i] / avoid_zero_divide(h(n, n - 1), division_min);
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

    T delta = avoid_zero_divide(std::sqrt(r(n - 1, n - 1) * r(n - 1, n - 1) +
                                          h(n, n - 1) * h(n, n - 1)),
                                division_min);

    c[n - 1] = r(n - 1, n - 1) / delta;
    s[n - 1] = h(n, n - 1) / delta;

    // Update b_hat
    r(n - 1, n - 1) = c[n - 1] * r(n - 1, n - 1) + s[n - 1] * h(n, n - 1);
    b_hat[n] = -s[n - 1] * b_hat[n - 1];
    b_hat[n - 1] *= c[n - 1];
    rho = std::abs(b_hat[n]);

    rep_num = n;

    // Check for convergence
    if (rho / avoid_zero_divide(b_norm, division_min) < decay_rate) {
      break;
    }
  }

  // Back substitution to solve
  for (std::size_t j = rep_num; j-- > 0;) {
    T temp = ZERO;
    for (std::size_t m = j + 1; m < rep_num; ++m) {
      temp += r(j, m) * y[m];
    }
    y[j] = (b_hat[j] - temp) / avoid_zero_divide(r(j, j), division_min);
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

/* Sparse GMRES K for matrix inverse */
template <typename T, std::size_t M, std::size_t V>
Matrix<T, M, M> sparse_gmres_k_matrix_inv(const SparseMatrix<T, M, M, V> In_A,
                                          T decay_rate, T division_min,
                                          const Matrix<T, M, M> X_1) {
  Matrix<T, M, M> B = Matrix<T, M, M>::identity();
  Matrix<T, M, M> X;
  Vector<T, M> rho_vec;
  Vector<std::size_t, M> rep_num_vec;

  for (std::size_t i = 0; i < M; i++) {
    Vector<T, M> x;

    x = sparse_gmres_k(In_A, B.get_row(i), X_1.get_row(i), decay_rate,
                       division_min, rho_vec[i], rep_num_vec[i]);
    X.set_row(i, x);
  }

  return X;
}

/* Complex GMRES K */
template <typename T, std::size_t M>
Vector<Complex<T>, M> complex_gmres_k(const Matrix<Complex<T>, M, M> &A,
                                      const Vector<Complex<T>, M> &b,
                                      const Vector<Complex<T>, M> &x_1,
                                      T decay_rate, T division_min, T &rho,
                                      std::size_t &rep_num) {
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
  T b_norm = complex_vector_norm(b_ax);
  for (std::size_t i = 0; i < M; ++i) {
    q(i, 0) = b_ax[i] / avoid_zero_divide(b_norm, division_min);
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
        h(j, n - 1) += complex_conjugate(q(i, j)) * v[i];
      }
      for (std::size_t i = 0; i < M; ++i) {
        v[i] -= h(j, n - 1) * q(i, j);
      }
    }

    if (n < M) {
      h(n, n - 1) = complex_vector_norm(v);
      for (std::size_t i = 0; i < M; ++i) {
        q(i, n) = complex_divide(v[i], h(n, n - 1), division_min);
      }
    }

    // Givens rotation for QR decomposition
    r(0, n - 1) = h(0, n - 1);
    if (n >= 2) {
      for (std::size_t j = 1; j < n; ++j) {
        Complex<T> gamma = c[j - 1] * r(j - 1, n - 1) + s[j - 1] * h(j, n - 1);
        r(j, n - 1) = -complex_conjugate(s[j - 1]) * r(j - 1, n - 1) +
                      c[j - 1] * h(j, n - 1);
        r(j - 1, n - 1) = gamma;
      }
    }

    T delta = avoid_zero_divide(std::sqrt(complex_abs_sq(r(n - 1, n - 1)) +
                                          complex_abs_sq(h(n, n - 1))),
                                division_min);

    c[n - 1] = r(n - 1, n - 1) / delta;
    s[n - 1] = h(n, n - 1) / delta;

    // Update b_hat
    r(n - 1, n - 1) = c[n - 1] * r(n - 1, n - 1) + s[n - 1] * h(n, n - 1);
    b_hat[n] = -complex_conjugate(s[n - 1]) * b_hat[n - 1];
    b_hat[n - 1] = b_hat[n - 1] * c[n - 1];
    rho = complex_abs_sq(b_hat[n]);

    rep_num = n;

    // Check for convergence
    if (rho / avoid_zero_divide(b_norm * b_norm, division_min) < decay_rate) {
      break;
    }
  }

  // Back substitution to solve
  for (std::size_t j = rep_num; j-- > 0;) {
    Complex<T> temp(ZERO);
    for (std::size_t m = j + 1; m < rep_num; ++m) {
      temp += r(j, m) * y[m];
    }
    y[j] = complex_divide(b_hat[j] - temp, r(j, j), division_min);
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

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_INVERSE_HPP
