#ifndef BASE_MATRIX_MATRIX_HPP
#define BASE_MATRIX_MATRIX_HPP

#include "base_matrix_complex.hpp"
#include "base_matrix_macros.hpp"
#include "base_matrix_vector.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

template <typename T, std::size_t M, std::size_t N> class Matrix {
public:
#ifdef BASE_MATRIX_USE_STD_VECTOR

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
    std::copy(input.begin(), input.end(), this->data[0].begin());
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

    std::copy(this->data[row].begin(), this->data[row].end(),
              result.data.begin());

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

#ifdef BASE_MATRIX_USE_STD_VECTOR

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

  Matrix<T, M, N> operator-() const { return output_minus_matrix(*this); }

  Matrix<T, N, M> transpose() const {
    Matrix<T, N, M> result;
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        result(j, i) = this->data[j][i];
      }
    }
    return result;
  }

  constexpr std::size_t rows() const { return N; }

  constexpr std::size_t cols() const { return M; }

  Vector<T, M> get_row(std::size_t row) const {
    if (row >= N) {
      row = N - 1;
    }

    Vector<T, M> result;
    std::copy(this->data[row].begin(), this->data[row].end(),
              result.data.begin());

    return result;
  }

  void set_row(std::size_t row, const Vector<T, M> &row_vector) {
    if (row >= N) {
      row = N - 1;
    }

    std::copy(row_vector.data.begin(), row_vector.data.end(),
              this->data[row].begin());
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
#ifdef BASE_MATRIX_USE_STD_VECTOR
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

  std::copy(mat(row_1).begin(), mat(row_1).end(), temp_vec.data.begin());
  std::copy(mat(row_2).begin(), mat(row_2).end(), mat(row_1).begin());
  std::copy(temp_vec.data.begin(), temp_vec.data.end(), mat(row_2).begin());
}

/* Matrix Addition */
// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct MatrixAdderColumn {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    result(I, J_idx) = A(I, J_idx) + B(I, J_idx);
    MatrixAdderColumn<T, M, N, I, J_idx - 1>::compute(A, B, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct MatrixAdderColumn<T, M, N, I, 0> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    result(I, 0) = A(I, 0) + B(I, 0);
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct MatrixAdderRow {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    MatrixAdderColumn<T, M, N, I_idx, N - 1>::compute(A, B, result);
    MatrixAdderRow<T, M, N, I_idx - 1>::compute(A, B, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N>
struct MatrixAdderRow<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    MatrixAdderColumn<T, M, N, 0, N - 1>::compute(A, B, result);
  }
};

#define BASE_MATRIX_COMPILED_MATRIX_ADD_MATRIX(T, M, N, A, B, result)          \
  MatrixAdderRow<T, M, N, M - 1>::compute(A, B, result);

template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> operator+(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(i, j) = A(i, j) + B(i, j);
    }
  }

#else

  BASE_MATRIX_COMPILED_MATRIX_ADD_MATRIX(T, M, N, A, B, result);

#endif

  return result;
}

/* Matrix Subtraction */
// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct MatrixSubtractorColumn {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    result(I, J_idx) = A(I, J_idx) - B(I, J_idx);
    MatrixSubtractorColumn<T, M, N, I, J_idx - 1>::compute(A, B, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct MatrixSubtractorColumn<T, M, N, I, 0> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    result(I, 0) = A(I, 0) - B(I, 0);
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct MatrixSubtractorRow {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    MatrixSubtractorColumn<T, M, N, I_idx, N - 1>::compute(A, B, result);
    MatrixSubtractorRow<T, M, N, I_idx - 1>::compute(A, B, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N>
struct MatrixSubtractorRow<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    MatrixSubtractorColumn<T, M, N, 0, N - 1>::compute(A, B, result);
  }
};

#define BASE_MATRIX_COMPILED_MATRIX_SUB_MATRIX(T, M, N, A, B, result)          \
  MatrixSubtractorRow<T, M, N, M - 1>::compute(A, B, result);

template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> operator-(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(i, j) = A(i, j) - B(i, j);
    }
  }

#else

  BASE_MATRIX_COMPILED_MATRIX_SUB_MATRIX(T, M, N, A, B, result);

#endif

  return result;
}

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct MatrixMinusColumn {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
    result(I, J_idx) = -A(I, J_idx);
    MatrixMinusColumn<T, M, N, I, J_idx - 1>::compute(A, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct MatrixMinusColumn<T, M, N, I, 0> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
    result(I, 0) = -A(I, 0);
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct MatrixMinusRow {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
    MatrixMinusColumn<T, M, N, I_idx, N - 1>::compute(A, result);
    MatrixMinusRow<T, M, N, I_idx - 1>::compute(A, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N>
struct MatrixMinusRow<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
    MatrixMinusColumn<T, M, N, 0, N - 1>::compute(A, result);
  }
};

#define BASE_MATRIX_COMPILED_MATRIX_MINUS_MATRIX(T, M, N, A, result)           \
  MatrixMinusRow<T, M, N, M - 1>::compute(A, result);

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> output_minus_matrix(const Matrix<T, M, N> &A) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(i, j) = -A(i, j);
    }
  }

#else

  BASE_MATRIX_COMPILED_MATRIX_MINUS_MATRIX(T, M, N, A, result);

#endif

  return result;
}

/* (Scalar) * (Matrix) */
// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct MatrixMultiplyScalarColumn {
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {
    result(I, J_idx) = scalar * mat(I, J_idx);
    MatrixMultiplyScalarColumn<T, M, N, I, J_idx - 1>::compute(scalar, mat,
                                                               result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct MatrixMultiplyScalarColumn<T, M, N, I, 0> {
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {
    result(I, 0) = scalar * mat(I, 0);
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct MatrixMultiplyScalarRow {
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {
    MatrixMultiplyScalarColumn<T, M, N, I_idx, N - 1>::compute(scalar, mat,
                                                               result);
    MatrixMultiplyScalarRow<T, M, N, I_idx - 1>::compute(scalar, mat, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N>
struct MatrixMultiplyScalarRow<T, M, N, 0> {
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {
    MatrixMultiplyScalarColumn<T, M, N, 0, N - 1>::compute(scalar, mat, result);
  }
};

#define BASE_MATRIX_COMPILED_SCALAR_MULTIPLY_MATRIX(T, M, N, scalar, mat,      \
                                                    result)                    \
  MatrixMultiplyScalarRow<T, M, N, M - 1>::compute(scalar, mat, result);

template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> operator*(const T &scalar, const Matrix<T, M, N> &mat) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = scalar * mat(j, k);
    }
  }

#else

  BASE_MATRIX_COMPILED_SCALAR_MULTIPLY_MATRIX(T, M, N, scalar, mat, result);

#endif

  return result;
}

template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> operator*(const Matrix<T, M, N> &mat, const T &scalar) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = scalar * mat(j, k);
    }
  }

#else

  BASE_MATRIX_COMPILED_SCALAR_MULTIPLY_MATRIX(T, M, N, scalar, mat, result);

#endif

  return result;
}

/* Matrix multiply Vector */
// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct MatrixVectorMultiplierCore {
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec,
                   std::size_t i) {
    return mat(i, J) * vec[J] +
           MatrixVectorMultiplierCore<T, M, N, J - 1>::compute(mat, vec, i);
  }
};

// if J == 0
template <typename T, std::size_t M, std::size_t N>
struct MatrixVectorMultiplierCore<T, M, N, 0> {
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec,
                   std::size_t i) {
    return mat(i, 0) * vec[0];
  }
};

// column recursion
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct MatrixVectorMultiplierRow {
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec,
                      Vector<T, M> &result) {
    result[I] =
        MatrixVectorMultiplierCore<T, M, N, N - 1>::compute(mat, vec, I);
    MatrixVectorMultiplierRow<T, M, N, I - 1>::compute(mat, vec, result);
  }
};

// if I == 0
template <typename T, std::size_t M, std::size_t N>
struct MatrixVectorMultiplierRow<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec,
                      Vector<T, M> &result) {
    result[0] =
        MatrixVectorMultiplierCore<T, M, N, N - 1>::compute(mat, vec, 0);
  }
};

#define BASE_MATRIX_MATRIX_MULTIPLY_VECTOR(T, M, N, mat, vec, result)          \
  MatrixVectorMultiplierRow<T, M, N, M - 1>::compute(mat, vec, result);

template <typename T, std::size_t M, std::size_t N>
Vector<T, M> operator*(const Matrix<T, M, N> &mat, const Vector<T, N> &vec) {
  Vector<T, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    T sum = 0;
    for (std::size_t j = 0; j < N; ++j) {
      sum += mat(i, j) * vec[j];
    }
    result[i] = sum;
  }

#else

  BASE_MATRIX_MATRIX_MULTIPLY_VECTOR(T, M, N, mat, vec, result);

#endif

  return result;
}

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

/* Compiled Matrix Multiplier Classes */
// when K_idx < K
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t K_idx>
struct MatrixMultiplierCore {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {
    return A(I, K_idx) * B(K_idx, J) +
           MatrixMultiplierCore<T, M, K, N, I, J, K_idx - 1>::compute(A, B);
  }
};

// when K_idx reached 0
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct MatrixMultiplierCore<T, M, K, N, I, J, 0> {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {
    return A(I, 0) * B(0, J);
  }
};

// After completing the J column, go to the next row I
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct MatrixMultiplierColumn {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    result(I, J) = MatrixMultiplierCore<T, M, K, N, I, J, K - 1>::compute(A, B);
    MatrixMultiplierColumn<T, M, K, N, I, J - 1>::compute(A, B, result);
  }
};

// Row recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct MatrixMultiplierColumn<T, M, K, N, I, 0> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    result(I, 0) = MatrixMultiplierCore<T, M, K, N, I, 0, K - 1>::compute(A, B);
  }
};

// proceed to the next row after completing the I row
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct MatrixMultiplierRow {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    MatrixMultiplierColumn<T, M, K, N, I, N - 1>::compute(A, B, result);
    MatrixMultiplierRow<T, M, K, N, I - 1>::compute(A, B, result);
  }
};

// Column recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N>
struct MatrixMultiplierRow<T, M, K, N, 0> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    MatrixMultiplierColumn<T, M, K, N, 0, N - 1>::compute(A, B, result);
  }
};

#define BASE_MATRIX_COMPILED_MATRIX_MULTIPLY(T, M, K, N, A, B, result)         \
  MatrixMultiplierRow<T, M, K, N, M - 1>::compute(A, B, result);

/* Matrix Multiplication */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
Matrix<T, M, N> operator*(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }

#else

  BASE_MATRIX_COMPILED_MATRIX_MULTIPLY(T, M, K, N, A, B, result);

#endif

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
