#ifndef BASE_MATRIX_MATRIX_HPP
#define BASE_MATRIX_MATRIX_HPP

#include "base_matrix_complex.hpp"
#include "base_matrix_macros.hpp"
#include "base_matrix_vector.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

template <typename T, std::size_t M, std::size_t N> class Matrix {
public:
#ifdef USE_STD_VECTOR

  Matrix() : data(N, std::vector<T>(M, static_cast<T>(0))) {}

  Matrix(const std::initializer_list<std::initializer_list<T>> &input)
      : data(N, std::vector<T>(M, static_cast<T>(0))) {

    auto outer_it = input.begin();
    for (std::size_t i = 0; i < M; i++) {
      auto inner_it = outer_it->begin();
      for (std::size_t j = 0; j < N; j++) {
        this->data[j][i] = *inner_it;
        ++inner_it;
      }
      ++outer_it;
    }
  }

  Matrix(const std::vector<T> &input)
      : data(N, std::vector<T>(M, static_cast<T>(0))) {
    std::memcpy(&this->data[0][0], &input[0], M * sizeof(this->data[0][0]));
  }

  Matrix(T input[][N]) : data(N, std::vector<T>(M, static_cast<T>(0))) {

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        this->data[j][i] = input[i][j];
      }
    }
  }

#else

  Matrix() : data{} {}

  Matrix(const std::initializer_list<std::initializer_list<T>> &input)
      : data{} {

    auto outer_it = input.begin();
    for (std::size_t i = 0; i < M; i++) {
      auto inner_it = outer_it->begin();
      for (std::size_t j = 0; j < N; j++) {
        this->data[j][i] = *inner_it;
        ++inner_it;
      }
      ++outer_it;
    }
  }

  Matrix(const std::array<std::array<T, N>, M> &input) : data(input) {}

  Matrix(const std::vector<T> &input) : data{} {
    std::copy(input.begin(), input.end(), this->data[0].begin());
  }

  Matrix(const std::array<T, M> &input) : data{} {
    std::copy(input.begin(), input.end(), this->data[0].begin());
  }

  Matrix(T input[][N]) : data{} {

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        this->data[j][i] = input[i][j];
      }
    }
  }

#endif

  /* Copy Constructor */
  Matrix(const Matrix<T, M, N> &other) : data(other.data) {}

  Matrix<T, M, N> &operator=(const Matrix<T, M, N> &other) {
    if (this != &other) {
      this->data = other.data;
    }
    return *this;
  }

  /* Move Constructor */
  Matrix(Matrix<T, M, N> &&other) noexcept : data(std::move(other.data)) {}

  Matrix<T, M, N> &operator=(Matrix<T, M, N> &&other) noexcept {
    if (this != &other) {
      this->data = std::move(other.data);
    }
    return *this;
  }

  /* Function */
  static Matrix<T, M, M> identity() {
    Matrix<T, M, M> identity;
    for (std::size_t i = 0; i < M; i++) {
      identity(i, i) = static_cast<T>(1);
    }
    return identity;
  }

  static Matrix<T, M, N> ones() {
    Matrix<T, M, N> Y;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        Y(i, j) = static_cast<T>(1);
      }
    }

    return Y;
  }

  Vector<T, M> create_row_vector(std::size_t row) const {
    Vector<T, M> result;

    if (row >= M) {
      row = M - 1;
    }

    std::memcpy(&result[0], &this->data[row][0], M * sizeof(result[0]));

    return result;
  }

  T &operator()(std::size_t col, std::size_t row) {
    if (col >= M) {
      col = M - 1;
    }
    if (row >= N) {
      row = N - 1;
    }

    return this->data[row][col];
  }

  const T &operator()(std::size_t col, std::size_t row) const {
    if (col >= M) {
      col = M - 1;
    }
    if (row >= N) {
      row = N - 1;
    }

    return this->data[row][col];
  }

#ifdef USE_STD_VECTOR

  std::vector<T> &operator()(std::size_t row) {
    if (row >= N) {
      row = N - 1;
    }

    return this->data[row];
  }

  const std::vector<T> &operator()(std::size_t row) const {
    if (row >= N) {
      row = N - 1;
    }

    return this->data[row];
  }

#else

  std::array<T, M> &operator()(std::size_t row) {
    if (row >= N) {
      row = N - 1;
    }

    return this->data[row];
  }

  const std::array<T, M> &operator()(std::size_t row) const {
    if (row >= N) {
      row = N - 1;
    }

    return this->data[row];
  }

#endif

  Matrix<T, M, N> operator+(const Matrix<T, M, N> &mat) const {
    Matrix<T, M, N> result;

    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        result(i, j) = this->data[j][i] + mat(i, j);
      }
    }

    return result;
  }

  Matrix<T, M, N> operator-(const Matrix<T, M, N> &mat) const {
    Matrix<T, M, N> result;

    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        result(i, j) = this->data[j][i] - mat(i, j);
      }
    }

    return result;
  }

  Matrix<T, M, N> operator-() const {
    Matrix<T, M, N> result;

    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        result(i, j) = -this->data[j][i];
      }
    }

    return result;
  }

  Matrix<T, M, N> operator*(const T &scalar) const {
    Matrix<T, M, N> result;
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        result(i, j) = this->data[j][i] * scalar;
      }
    }
    return result;
  }

  Vector<T, M> operator*(const Vector<T, N> &vec) const {
    Vector<T, M> result;
    for (std::size_t i = 0; i < M; ++i) {
      T sum = 0;
      for (std::size_t j = 0; j < N; ++j) {
        sum += this->data[j][i] * vec[j];
      }
      result[i] = sum;
    }
    return result;
  }

  Matrix<T, N, M> transpose() const {
    Matrix<T, N, M> result;
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        result(j, i) = this->data[j][i];
      }
    }
    return result;
  }

  std::size_t rows() const { return N; }

  std::size_t cols() const { return M; }

  Vector<T, M> get_row(std::size_t row) const {
    if (row >= N) {
      row = N - 1;
    }

    Vector<T, M> result;
    std::memcpy(&result[0], &this->data[row][0], M * sizeof(result[0]));

    return result;
  }

  void set_row(std::size_t row, const Vector<T, M> &row_vector) {
    if (row >= N) {
      row = N - 1;
    }

    std::memcpy(&this->data[row][0], &row_vector[0],
                M * sizeof(this->data[row][0]));
  }

  Matrix<T, M, M> inv() const {
    Matrix<T, M, M> X_temp = Matrix<T, M, M>::identity();
    Matrix<T, M, M> Inv = gmres_k_matrix_inv(*this, static_cast<T>(0.0),
                                             static_cast<T>(1.0e-10), X_temp);

    return Inv;
  }

  T get_trace() const {
    T trace = static_cast<T>(0);
    for (std::size_t i = 0; i < M; i++) {
      trace += this->data[i][i];
    }
    return trace;
  }

/* Variable */
#ifdef USE_STD_VECTOR
  std::vector<std::vector<T>> data;
#else
  std::array<std::array<T, M>, N> data;
#endif
};

/* swap columns */
template <typename T, std::size_t M, std::size_t N>
void matrix_col_swap(std::size_t col_1, std::size_t col_2,
                     Matrix<T, M, N> &mat) {
  T temp;

  if (col_1 >= M) {
    col_1 = M - 1;
  }
  if (col_2 >= M) {
    col_2 = M - 1;
  }

  for (std::size_t i = 0; i < N; i++) {
    temp = mat(col_1, i);
    mat(col_1, i) = mat(col_2, i);
    mat(col_2, i) = temp;
  }
}

/* swap rows */
template <typename T, std::size_t M, std::size_t N>
void matrix_row_swap(std::size_t row_1, std::size_t row_2,
                     Matrix<T, M, N> &mat) {
  Vector<T, M> temp_vec;

  if (row_1 >= N) {
    row_1 = N - 1;
  }
  if (row_2 >= N) {
    row_2 = N - 1;
  }

  std::memcpy(&temp_vec[0], &mat(row_1)[0], M * sizeof(temp_vec[0]));
  std::memcpy(&mat(row_1)[0], &mat(row_2)[0], M * sizeof(mat(row_1)[0]));
  std::memcpy(&mat(row_2)[0], &temp_vec[0], M * sizeof(mat(row_2)[0]));
}

/* (Scalar) * (Matrix) */
template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> operator*(const T &scalar, const Matrix<T, M, N> &mat) {
  Matrix<T, M, N> result;

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = scalar * mat(j, k);
    }
  }
  return result;
}

/* (Row Vector) * (Matrix) */
template <typename T, std::size_t L, std::size_t M, std::size_t N>
Matrix<T, L, N> operator*(const Vector<T, L> &vec, const Matrix<T, M, N> &mat) {
  static_assert(M == 1, "Invalid size.");

  Matrix<T, L, N> result;
  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t k = 0; k < L; ++k) {
      result(k, j) = vec[k] * mat(0, j);
    }
  }
  return result;
}

/* (Column Vector) * (Matrix) */
template <typename T, std::size_t M, std::size_t N>
ColVector<T, N> operator*(const ColVector<T, M> &vec,
                          const Matrix<T, M, N> &mat) {
  ColVector<T, N> result;
  for (std::size_t j = 0; j < N; ++j) {
    T sum = 0;
    for (std::size_t i = 0; i < M; ++i) {
      sum += vec[i] * mat(i, j);
    }
    result[j] = sum;
  }
  return result;
}

/* Matrix Multiplication */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
Matrix<T, M, N> operator*(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }
  return result;
}

template <typename T, std::size_t M, std::size_t K, std::size_t N>
Matrix<T, M, N>
matrix_multiply_Upper_triangular_A_mul_B(const Matrix<T, M, K> &A,
                                         const Matrix<T, K, N> &B) {

  Matrix<T, M, N> result;
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = i; k < K; ++k) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }
  return result;
}

template <typename T, std::size_t M, std::size_t K, std::size_t N>
Matrix<T, M, N> matrix_multiply_AT_mul_B(const Matrix<T, K, M> &A,
                                         const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(k, i) * B(k, j);
      }
      result(i, j) = sum;
    }
  }
  return result;
}

template <typename T, std::size_t M, std::size_t N>
Vector<T, N> matrix_multiply_AT_mul_b(const Matrix<T, M, N> &A,
                                      const Vector<T, M> &b) {
  Vector<T, N> result;
  for (std::size_t n = 0; n < N; ++n) {
    T sum = 0;
    for (std::size_t m = 0; m < M; ++m) {
      sum += A(m, n) * b[m];
    }
    result[n] = sum;
  }
  return result;
}

template <typename T, std::size_t M, std::size_t K, std::size_t N>
Matrix<T, M, N> matrix_multiply_A_mul_BT(const Matrix<T, M, K> &A,
                                         const Matrix<T, N, K> &B) {
  Matrix<T, M, N> result;
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(i, k) * B(j, k);
      }
      result(i, j) = sum;
    }
  }
  return result;
}

template <typename T, std::size_t M, std::size_t N>
Matrix<Complex<T>, M, N>
convert_matrix_real_to_complex(const Matrix<T, M, N> &From_matrix) {

  Matrix<Complex<T>, M, N> To_matrix;

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j) = From_matrix(i, j);
    }
  }

  return To_matrix;
}

template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> get_real_matrix_from_complex_matrix(
    const Matrix<Complex<T>, M, N> &From_matrix) {

  Matrix<T, M, N> To_matrix;

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j) = From_matrix(i, j).real;
    }
  }

  return To_matrix;
}

template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> get_imag_matrix_from_complex_matrix(
    const Matrix<Complex<T>, M, N> &From_matrix) {

  Matrix<T, M, N> To_matrix;

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j) = From_matrix(i, j).imag;
    }
  }

  return To_matrix;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_MATRIX_HPP
