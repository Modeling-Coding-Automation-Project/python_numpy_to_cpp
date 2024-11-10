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
  /* Identity */
  // P_idx < P
  template <typename U, std::size_t P, std::size_t P_idx>
  struct CreateIdentityCore {
    static void compute(Matrix<U, M, M> &identity) {
      identity(P_idx, P_idx) = static_cast<U>(1);
      CreateIdentityCore<U, P, P_idx - 1>::compute(identity);
    }
  };

  // Termination condition: P_idx == 0
  template <typename U, std::size_t P> struct CreateIdentityCore<U, P, 0> {
    static void compute(Matrix<U, P, P> &identity) {
      identity(0, 0) = static_cast<U>(1);
    }
  };

  template <typename U, std::size_t P>
  static inline void
  BASE_MATRIX_COMPILED_MATRIX_IDENTITY(Matrix<U, P, P> &identity) {
    CreateIdentityCore<U, P, P - 1>::compute(identity);
  }

  static Matrix<T, M, M> identity() {
    Matrix<T, M, M> identity;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

    for (std::size_t i = 0; i < M; i++) {
      identity(i, i) = static_cast<T>(1);
    }

#else

    BASE_MATRIX_COMPILED_MATRIX_IDENTITY<T, M>(identity);

#endif

    return identity;
  }

  /* Ones */
  // when J_idx < P
  template <typename U, std::size_t O, std::size_t P, std::size_t I,
            std::size_t J_idx>
  struct MatrixOnesColumn {
    static void compute(Matrix<U, O, P> &Ones) {
      Ones(I, J_idx) = static_cast<U>(1);
      MatrixOnesColumn<U, O, P, I, J_idx - 1>::compute(Ones);
    }
  };

  // column recursion termination
  template <typename U, std::size_t O, std::size_t P, std::size_t I>
  struct MatrixOnesColumn<U, O, P, I, 0> {
    static void compute(Matrix<U, O, P> &Ones) {
      Ones(I, 0) = static_cast<U>(1);
    }
  };

  // when I_idx < M
  template <typename U, std::size_t O, std::size_t P, std::size_t I_idx>
  struct MatrixOnesRow {
    static void compute(Matrix<U, O, P> &Ones) {
      MatrixOnesColumn<U, O, P, I_idx, P - 1>::compute(Ones);
      MatrixOnesRow<U, O, P, I_idx - 1>::compute(Ones);
    }
  };

  // row recursion termination
  template <typename U, std::size_t O, std::size_t P>
  struct MatrixOnesRow<U, O, P, 0> {
    static void compute(Matrix<U, O, P> &Ones) {
      MatrixOnesColumn<U, O, P, 0, P - 1>::compute(Ones);
    }
  };

  template <typename U, std::size_t O, std::size_t P>
  static inline void BASE_MATRIX_COMPILED_MATRIX_ONES(Matrix<U, O, P> &Ones) {
    MatrixOnesRow<U, O, P, O - 1>::compute(Ones);
  }

  static Matrix<T, M, N> ones() {
    Matrix<T, M, N> Ones;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        Ones(i, j) = static_cast<T>(1);
      }
    }

#else

    BASE_MATRIX_COMPILED_MATRIX_ONES<T, M, N>(Ones);

#endif

    return Ones;
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

  Matrix<T, N, M> transpose() const { return output_matrix_transpose(*this); }

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

  T get_trace() const { return output_matrix_trace(*this); }

/* Variable */
#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<std::vector<T>> data;
#else
  std::array<std::array<T, M>, N> data;
#endif
};

/* swap columns */
// Swap N_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t N_idx>
struct MatrixSwapColumnsCore {
  static void compute(std::size_t col_1, std::size_t col_2,
                      Matrix<T, M, N> &mat, T temp) {
    temp = mat(col_1, N_idx);
    mat(col_1, N_idx) = mat(col_2, N_idx);
    mat(col_2, N_idx) = temp;
    MatrixSwapColumnsCore<T, M, N, N_idx - 1>::compute(col_1, col_2, mat, temp);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t M, std::size_t N>
struct MatrixSwapColumnsCore<T, M, N, 0> {
  static void compute(std::size_t col_1, std::size_t col_2,
                      Matrix<T, M, N> &mat, T temp) {
    temp = mat(col_1, 0);
    mat(col_1, 0) = mat(col_2, 0);
    mat(col_2, 0) = temp;
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline void
BASE_MATRIX_COMPILED_MATRIX_COLUMN_SWAP(std::size_t col_1, std::size_t col_2,
                                        Matrix<T, M, N> &mat, T &temp) {
  MatrixSwapColumnsCore<T, M, N, N - 1>::compute(col_1, col_2, mat, temp);
}

template <typename T, std::size_t M, std::size_t N>
void matrix_col_swap(std::size_t col_1, std::size_t col_2,
                     Matrix<T, M, N> &mat) {
  T temp = static_cast<T>(0);

  if (col_1 >= M) {
    col_1 = M - 1;
  }
  if (col_2 >= M) {
    col_2 = M - 1;
  }

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; i++) {
    temp = mat(col_1, i);
    mat(col_1, i) = mat(col_2, i);
    mat(col_2, i) = temp;
  }

#else

  BASE_MATRIX_COMPILED_MATRIX_COLUMN_SWAP<T, M, N>(col_1, col_2, mat, temp);

#endif
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

/* Trace */
// calculate trace of matrix
template <typename T, std::size_t N, std::size_t I> struct MatrixTraceCore {
  static T compute(const Matrix<T, N, N> &mat) {
    return mat(I, I) + MatrixTraceCore<T, N, I - 1>::compute(mat);
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline T BASE_MATRIX_COMPILED_MATRIX_TRACE(const Matrix<T, M, N> &mat) {
  return MatrixTraceCore<T, N, N - 1>::compute(mat);
}

// if I == 0
template <typename T, std::size_t N> struct MatrixTraceCore<T, N, 0> {
  static T compute(const Matrix<T, N, N> &mat) { return mat(0, 0); }
};

template <typename T, std::size_t M, std::size_t N>
inline T output_matrix_trace(const Matrix<T, M, N> &mat) {
  static_assert(M == N, "Matrix must be square matrix");
  T trace = static_cast<T>(0);

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; i++) {
    trace += mat(i, i);
  }

#else

  trace = BASE_MATRIX_COMPILED_MATRIX_TRACE<T, M, N>(mat);

#endif

  return trace;
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

template <typename T, std::size_t M, std::size_t N>
static inline void
BASE_MATRIX_COMPILED_MATRIX_ADD_MATRIX(const Matrix<T, M, N> &A,
                                       const Matrix<T, M, N> &B,
                                       Matrix<T, M, N> &result) {
  MatrixAdderRow<T, M, N, M - 1>::compute(A, B, result);
}

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

  BASE_MATRIX_COMPILED_MATRIX_ADD_MATRIX<T, M, N>(A, B, result);

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

template <typename T, std::size_t M, std::size_t N>
static inline void
BASE_MATRIX_COMPILED_MATRIX_SUB_MATRIX(const Matrix<T, M, N> &A,
                                       const Matrix<T, M, N> &B,
                                       Matrix<T, M, N> &result) {
  MatrixSubtractorRow<T, M, N, M - 1>::compute(A, B, result);
}

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

  BASE_MATRIX_COMPILED_MATRIX_SUB_MATRIX<T, M, N>(A, B, result);

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

template <typename T, std::size_t M, std::size_t N>
static inline void
BASE_MATRIX_COMPILED_MATRIX_MINUS_MATRIX(const Matrix<T, M, N> &A,
                                         Matrix<T, M, N> &result) {
  MatrixMinusRow<T, M, N, M - 1>::compute(A, result);
}

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

  BASE_MATRIX_COMPILED_MATRIX_MINUS_MATRIX<T, M, N>(A, result);

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

template <typename T, std::size_t M, std::size_t N>
static inline void BASE_MATRIX_COMPILED_SCALAR_MULTIPLY_MATRIX(
    const T &scalar, const Matrix<T, M, N> &mat, Matrix<T, M, N> &result) {
  MatrixMultiplyScalarRow<T, M, N, M - 1>::compute(scalar, mat, result);
}

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

  BASE_MATRIX_COMPILED_SCALAR_MULTIPLY_MATRIX<T, M, N>(scalar, mat, result);

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

// calculate if K_idx > 0
template <typename T, std::size_t L, std::size_t N, std::size_t K_idx>
struct VectorMatrixMultiplierCore {
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result, std::size_t j) {
    result(K_idx, j) = vec[K_idx] * mat(0, j);
    VectorMatrixMultiplierCore<T, L, N, K_idx - 1>::compute(vec, mat, result,
                                                            j);
  }
};

// if K_idx = 0
template <typename T, std::size_t L, std::size_t N>
struct VectorMatrixMultiplierCore<T, L, N, 0> {
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result, std::size_t j) {
    result(0, j) = vec[0] * mat(0, j);
  }
};

// row recursion
template <typename T, std::size_t L, std::size_t N, std::size_t J>
struct VectorMatrixMultiplierColumn {
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {
    VectorMatrixMultiplierCore<T, L, N, L - 1>::compute(vec, mat, result, J);
    VectorMatrixMultiplierColumn<T, L, N, J - 1>::compute(vec, mat, result);
  }
};

// if J = 0
template <typename T, std::size_t L, std::size_t N>
struct VectorMatrixMultiplierColumn<T, L, N, 0> {
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {
    VectorMatrixMultiplierCore<T, L, N, L - 1>::compute(vec, mat, result, 0);
  }
};

template <typename T, std::size_t L, std::size_t M, std::size_t N>
static inline void
BASE_MATRIX_COMPILED_VECTOR_MULTIPLY_MATRIX(const Vector<T, L> &vec,
                                            const Matrix<T, M, N> &mat,
                                            Matrix<T, L, N> &result) {
  VectorMatrixMultiplierColumn<T, L, N, N - 1>::compute(vec, mat, result);
}

template <typename T, std::size_t L, std::size_t M, std::size_t N>
Matrix<T, L, N> operator*(const Vector<T, L> &vec, const Matrix<T, M, N> &mat) {
  static_assert(M == 1, "Invalid size.");
  Matrix<T, L, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t k = 0; k < L; ++k) {
      result(k, j) = vec[k] * mat(0, j);
    }
  }

#else

  BASE_MATRIX_COMPILED_VECTOR_MULTIPLY_MATRIX<T, L, M, N>(vec, mat, result);

#endif

  return result;
}

/* (Column Vector) * (Matrix) */
// calculation when I > 0
template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t I>
struct ColVectorMatrixMultiplierCore {
  static T compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat) {
    return vec[I] * mat(I, J) +
           ColVectorMatrixMultiplierCore<T, M, N, J, I - 1>::compute(vec, mat);
  }
};

// if I = 0
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct ColVectorMatrixMultiplierCore<T, M, N, J, 0> {
  static T compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat) {
    return vec[0] * mat(0, J);
  }
};

// row recursion
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct ColVectorMatrixMultiplierColumn {
  static void compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat,
                      ColVector<T, N> &result) {
    result[J] =
        ColVectorMatrixMultiplierCore<T, M, N, J, M - 1>::compute(vec, mat);
    ColVectorMatrixMultiplierColumn<T, M, N, J - 1>::compute(vec, mat, result);
  }
};

// if J = 0
template <typename T, std::size_t M, std::size_t N>
struct ColVectorMatrixMultiplierColumn<T, M, N, 0> {
  static void compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat,
                      ColVector<T, N> &result) {
    result[0] =
        ColVectorMatrixMultiplierCore<T, M, N, 0, M - 1>::compute(vec, mat);
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline void
BASE_MATRIX_COMPILED_COLUMN_VECTOR_MULTIPLY_MATRIX(const ColVector<T, M> &vec,
                                                   const Matrix<T, M, N> &mat,
                                                   ColVector<T, N> &result) {
  ColVectorMatrixMultiplierColumn<T, M, N, N - 1>::compute(vec, mat, result);
}

template <typename T, std::size_t M, std::size_t N>
ColVector<T, N> operator*(const ColVector<T, M> &vec,
                          const Matrix<T, M, N> &mat) {
  ColVector<T, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < N; ++j) {
    T sum = 0;
    for (std::size_t i = 0; i < M; ++i) {
      sum += vec[i] * mat(i, j);
    }
    result[j] = sum;
  }

#else

  BASE_MATRIX_COMPILED_COLUMN_VECTOR_MULTIPLY_MATRIX<T, M, N>(vec, mat, result);

#endif

  return result;
}

/* Matrix Multiply Matrix */
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

template <typename T, std::size_t M, std::size_t K, std::size_t N>
static inline void
BASE_MATRIX_COMPILED_MATRIX_MULTIPLY(const Matrix<T, M, K> &A,
                                     const Matrix<T, K, N> &B,
                                     Matrix<T, M, N> &result) {
  MatrixMultiplierRow<T, M, K, N, M - 1>::compute(A, B, result);
}

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

  BASE_MATRIX_COMPILED_MATRIX_MULTIPLY<T, M, K, N>(A, B, result);

#endif

  return result;
}

/* Transpose */
// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct MatrixTransposeColumn {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
    result(J_idx, I) = A(I, J_idx);
    MatrixTransposeColumn<T, M, N, I, J_idx - 1>::compute(A, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct MatrixTransposeColumn<T, M, N, I, 0> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
    result(0, I) = A(I, 0);
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct MatrixTransposeRow {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
    MatrixTransposeColumn<T, M, N, I_idx, N - 1>::compute(A, result);
    MatrixTransposeRow<T, M, N, I_idx - 1>::compute(A, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N>
struct MatrixTransposeRow<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
    MatrixTransposeColumn<T, M, N, 0, N - 1>::compute(A, result);
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline void
BASE_MATRIX_COMPILED_MATRIX_TRANSPOSE(const Matrix<T, M, N> &A,
                                      Matrix<T, N, M> &result) {
  MatrixTransposeRow<T, M, N, M - 1>::compute(A, result);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, N, M> output_matrix_transpose(const Matrix<T, M, N> &mat) {
  Matrix<T, N, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(j, i) = mat(i, j);
    }
  }

#else

  BASE_MATRIX_COMPILED_MATRIX_TRANSPOSE<T, M, N>(mat, result);

#endif

  return result;
}

/* Upper Triangular Matrix Multiply Matrix */
// when K_idx >= I
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t K_idx>
struct UpperTriangularMatrixMultiplierCore {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {
    return (K_idx >= I)
               ? (A(I, K_idx) * B(K_idx, J) +
                  UpperTriangularMatrixMultiplierCore<T, M, K, N, I, J,
                                                      K_idx - 1>::compute(A, B))
               : static_cast<T>(0);
  }
};

// recursion termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct UpperTriangularMatrixMultiplierCore<T, M, K, N, I, J,
                                           static_cast<std::size_t>(-1)> {
  static T compute(const Matrix<T, M, K> &, const Matrix<T, K, N> &) {
    return static_cast<T>(0);
  }
};

// when K_idx reaches I (base case)
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct UpperTriangularMatrixMultiplierCore<T, M, K, N, I, J, I> {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {
    return A(I, I) * B(I, J);
  }
};

// Column-wise computation
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct UpperTriangularMatrixMultiplierColumn {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    result(I, J) =
        UpperTriangularMatrixMultiplierCore<T, M, K, N, I, J, K - 1>::compute(
            A, B);
    UpperTriangularMatrixMultiplierColumn<T, M, K, N, I, J - 1>::compute(
        A, B, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct UpperTriangularMatrixMultiplierColumn<T, M, K, N, I, 0> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    result(I, 0) =
        UpperTriangularMatrixMultiplierCore<T, M, K, N, I, 0, K - 1>::compute(
            A, B);
  }
};

// Row-wise computation
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct UpperTriangularMatrixMultiplierRow {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    UpperTriangularMatrixMultiplierColumn<T, M, K, N, I, N - 1>::compute(
        A, B, result);
    UpperTriangularMatrixMultiplierRow<T, M, K, N, I - 1>::compute(A, B,
                                                                   result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t K, std::size_t N>
struct UpperTriangularMatrixMultiplierRow<T, M, K, N, 0> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    UpperTriangularMatrixMultiplierColumn<T, M, K, N, 0, N - 1>::compute(
        A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N>
static inline void
BASE_MATRIX_COMPILED_UPPER_TRIANGULAR_MATRIX_MULTIPLY(const Matrix<T, M, K> &A,
                                                      const Matrix<T, K, N> &B,
                                                      Matrix<T, M, N> &result) {
  UpperTriangularMatrixMultiplierRow<T, M, K, N, M - 1>::compute(A, B, result);
}

template <typename T, std::size_t M, std::size_t K, std::size_t N>
Matrix<T, M, N>
matrix_multiply_Upper_triangular_A_mul_B(const Matrix<T, M, K> &A,
                                         const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = i; k < K; ++k) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }

#else

  BASE_MATRIX_COMPILED_UPPER_TRIANGULAR_MATRIX_MULTIPLY<T, M, K, N>(A, B,
                                                                    result);

#endif

  return result;
}

/* Matrix Transpose Multiply Matrix */
// when K_idx < K
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t K_idx>
struct MatrixTransposeMultiplyMatrixCore {
  static T compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B) {
    return A(K_idx, I) * B(K_idx, J) +
           MatrixTransposeMultiplyMatrixCore<T, M, K, N, I, J,
                                             K_idx - 1>::compute(A, B);
  }
};

// when K_idx reached 0
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct MatrixTransposeMultiplyMatrixCore<T, M, K, N, I, J, 0> {
  static T compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B) {
    return A(0, I) * B(0, J);
  }
};

// After completing the J column, go to the next row I
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct MatrixTransposeMultiplyMatrixColumn {
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    result(I, J) =
        MatrixTransposeMultiplyMatrixCore<T, M, K, N, I, J, K - 1>::compute(A,
                                                                            B);
    MatrixTransposeMultiplyMatrixColumn<T, M, K, N, I, J - 1>::compute(A, B,
                                                                       result);
  }
};

// Row recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct MatrixTransposeMultiplyMatrixColumn<T, M, K, N, I, 0> {
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    result(I, 0) =
        MatrixTransposeMultiplyMatrixCore<T, M, K, N, I, 0, K - 1>::compute(A,
                                                                            B);
  }
};

// proceed to the next row after completing the I row
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct MatrixTransposeMultiplyMatrixRow {
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    MatrixTransposeMultiplyMatrixColumn<T, M, K, N, I, N - 1>::compute(A, B,
                                                                       result);
    MatrixTransposeMultiplyMatrixRow<T, M, K, N, I - 1>::compute(A, B, result);
  }
};

// Column recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N>
struct MatrixTransposeMultiplyMatrixRow<T, M, K, N, 0> {
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    MatrixTransposeMultiplyMatrixColumn<T, M, K, N, 0, N - 1>::compute(A, B,
                                                                       result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N>
static inline void
BASE_MATRIX_COMPILED_MATRIX_T_MULTIPLY_MATRIX(const Matrix<T, K, M> &A,
                                              const Matrix<T, K, N> &B,
                                              Matrix<T, M, N> &result) {
  MatrixTransposeMultiplyMatrixRow<T, M, K, N, M - 1>::compute(A, B, result);
}

template <typename T, std::size_t M, std::size_t K, std::size_t N>
Matrix<T, M, N> matrix_multiply_AT_mul_B(const Matrix<T, K, M> &A,
                                         const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(k, i) * B(k, j);
      }
      result(i, j) = sum;
    }
  }

#else

  BASE_MATRIX_COMPILED_MATRIX_T_MULTIPLY_MATRIX<T, M, K, N>(A, B, result);

#endif

  return result;
}

/* Transpose Matrix multiply Vector  */
// when M_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t M_idx>
struct MatrixTransposeVectorMultiplierCore {
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec,
                   std::size_t n) {
    return mat(M_idx, n) * vec[M_idx] +
           MatrixTransposeVectorMultiplierCore<T, M, N, M_idx - 1>::compute(
               mat, vec, n);
  }
};

// if M_idx == 0
template <typename T, std::size_t M, std::size_t N>
struct MatrixTransposeVectorMultiplierCore<T, M, N, 0> {
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec,
                   std::size_t n) {
    return mat(0, n) * vec[0];
  }
};

// column recursion
template <typename T, std::size_t M, std::size_t N, std::size_t N_idx>
struct MatrixTransposeVectorMultiplierColumn {
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec,
                      Vector<T, N> &result) {
    result[N_idx] =
        MatrixTransposeVectorMultiplierCore<T, M, N, M - 1>::compute(mat, vec,
                                                                     N_idx);
    MatrixTransposeVectorMultiplierColumn<T, M, N, N_idx - 1>::compute(mat, vec,
                                                                       result);
  }
};

// if N_idx == 0
template <typename T, std::size_t M, std::size_t N>
struct MatrixTransposeVectorMultiplierColumn<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec,
                      Vector<T, N> &result) {
    result[0] = MatrixTransposeVectorMultiplierCore<T, M, N, M - 1>::compute(
        mat, vec, 0);
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline void BASE_MATRIX_MATRIX_TRANSPOSE_MULTIPLY_VECTOR(
    const Matrix<T, M, N> &mat, const Vector<T, M> &vec, Vector<T, N> &result) {
  MatrixTransposeVectorMultiplierColumn<T, M, N, N - 1>::compute(mat, vec,
                                                                 result);
}

template <typename T, std::size_t M, std::size_t N>
Vector<T, N> matrix_multiply_AT_mul_b(const Matrix<T, M, N> &A,
                                      const Vector<T, M> &b) {
  Vector<T, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t n = 0; n < N; ++n) {
    T sum = 0;
    for (std::size_t m = 0; m < M; ++m) {
      sum += A(m, n) * b[m];
    }
    result[n] = sum;
  }

#else

  BASE_MATRIX_MATRIX_TRANSPOSE_MULTIPLY_VECTOR<T, M, N>(A, b, result);

#endif

  return result;
}

/* Matrix multiply Transpose Matrix */
// when K_idx < K
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t K_idx>
struct MatrixMultiplyTransposeMatrixCore {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B) {
    return A(I, K_idx) * B(J, K_idx) +
           MatrixMultiplyTransposeMatrixCore<T, M, K, N, I, J,
                                             K_idx - 1>::compute(A, B);
  }
};

// when K_idx reached 0
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct MatrixMultiplyTransposeMatrixCore<T, M, K, N, I, J, 0> {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B) {
    return A(I, 0) * B(J, 0);
  }
};

// After completing the J column, go to the next row I
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct MatrixMultiplyTransposeMatrixColumn {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {
    result(I, J) =
        MatrixMultiplyTransposeMatrixCore<T, M, K, N, I, J, K - 1>::compute(A,
                                                                            B);
    MatrixMultiplyTransposeMatrixColumn<T, M, K, N, I, J - 1>::compute(A, B,
                                                                       result);
  }
};

// Row recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct MatrixMultiplyTransposeMatrixColumn<T, M, K, N, I, 0> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {
    result(I, 0) =
        MatrixMultiplyTransposeMatrixCore<T, M, K, N, I, 0, K - 1>::compute(A,
                                                                            B);
  }
};

// proceed to the next row after completing the I row
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct MatrixMultiplyTransposeMatrixRow {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {
    MatrixMultiplyTransposeMatrixColumn<T, M, K, N, I, N - 1>::compute(A, B,
                                                                       result);
    MatrixMultiplyTransposeMatrixRow<T, M, K, N, I - 1>::compute(A, B, result);
  }
};

// Column recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N>
struct MatrixMultiplyTransposeMatrixRow<T, M, K, N, 0> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {
    MatrixMultiplyTransposeMatrixColumn<T, M, K, N, 0, N - 1>::compute(A, B,
                                                                       result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N>
static inline void
BASE_MATRIX_COMPILED_MATRIX_MULTIPLY_TRANSPOSE_MATRIX(const Matrix<T, M, K> &A,
                                                      const Matrix<T, N, K> &B,
                                                      Matrix<T, M, N> &result) {
  MatrixMultiplyTransposeMatrixRow<T, M, K, N, M - 1>::compute(A, B, result);
}

template <typename T, std::size_t M, std::size_t K, std::size_t N>
Matrix<T, M, N> matrix_multiply_A_mul_BT(const Matrix<T, M, K> &A,
                                         const Matrix<T, N, K> &B) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(i, k) * B(j, k);
      }
      result(i, j) = sum;
    }
  }

#else

  BASE_MATRIX_COMPILED_MATRIX_MULTIPLY_TRANSPOSE_MATRIX<T, M, K, N>(A, B,
                                                                    result);

#endif

  return result;
}

/* Matrix real from complex */
// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct MatrixRealToComplexColumn {
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {
    To_matrix(I, J_idx).real = From_matrix(I, J_idx);
    MatrixRealToComplexColumn<T, M, N, I, J_idx - 1>::compute(From_matrix,
                                                              To_matrix);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct MatrixRealToComplexColumn<T, M, N, I, 0> {
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {
    To_matrix(I, 0).real = From_matrix(I, 0);
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct MatrixRealToComplexRow {
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {
    MatrixRealToComplexColumn<T, M, N, I_idx, N - 1>::compute(From_matrix,
                                                              To_matrix);
    MatrixRealToComplexRow<T, M, N, I_idx - 1>::compute(From_matrix, To_matrix);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N>
struct MatrixRealToComplexRow<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {
    MatrixRealToComplexColumn<T, M, N, 0, N - 1>::compute(From_matrix,
                                                          To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline void BASE_MATRIX_COMPILED_MATRIX_REAL_TO_COMPLEX(
    const Matrix<T, M, N> &From_matrix, Matrix<Complex<T>, M, N> &To_matrix) {
  MatrixRealToComplexRow<T, M, N, M - 1>::compute(From_matrix, To_matrix);
}

template <typename T, std::size_t M, std::size_t N>
Matrix<Complex<T>, M, N>
convert_matrix_real_to_complex(const Matrix<T, M, N> &From_matrix) {

  Matrix<Complex<T>, M, N> To_matrix;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j).real = From_matrix(i, j);
    }
  }

#else

  BASE_MATRIX_COMPILED_MATRIX_REAL_TO_COMPLEX<T, M, N>(From_matrix, To_matrix);

#endif

  return To_matrix;
}

/* Matrix real from complex */
// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct MatrixRealFromComplexColumn {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    To_matrix(I, J_idx) = From_matrix(I, J_idx).real;
    MatrixRealFromComplexColumn<T, M, N, I, J_idx - 1>::compute(From_matrix,
                                                                To_matrix);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct MatrixRealFromComplexColumn<T, M, N, I, 0> {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    To_matrix(I, 0) = From_matrix(I, 0).real;
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct MatrixRealFromComplexRow {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    MatrixRealFromComplexColumn<T, M, N, I_idx, N - 1>::compute(From_matrix,
                                                                To_matrix);
    MatrixRealFromComplexRow<T, M, N, I_idx - 1>::compute(From_matrix,
                                                          To_matrix);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N>
struct MatrixRealFromComplexRow<T, M, N, 0> {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    MatrixRealFromComplexColumn<T, M, N, 0, N - 1>::compute(From_matrix,
                                                            To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline void BASE_MATRIX_COMPILED_MATRIX_REAL_FROM_COMPLEX(
    const Matrix<Complex<T>, M, N> &From_matrix, Matrix<T, M, N> &To_matrix) {
  MatrixRealFromComplexRow<T, M, N, M - 1>::compute(From_matrix, To_matrix);
}

template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> get_real_matrix_from_complex_matrix(
    const Matrix<Complex<T>, M, N> &From_matrix) {

  Matrix<T, M, N> To_matrix;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j) = From_matrix(i, j).real;
    }
  }

#else

  BASE_MATRIX_COMPILED_MATRIX_REAL_FROM_COMPLEX<T, M, N>(From_matrix,
                                                         To_matrix);

#endif

  return To_matrix;
}

/* Matrix imag from complex */
// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct MatrixImagFromComplexColumn {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    To_matrix(I, J_idx) = From_matrix(I, J_idx).imag;
    MatrixImagFromComplexColumn<T, M, N, I, J_idx - 1>::compute(From_matrix,
                                                                To_matrix);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct MatrixImagFromComplexColumn<T, M, N, I, 0> {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    To_matrix(I, 0) = From_matrix(I, 0).imag;
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct MatrixImagFromComplexRow {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    MatrixImagFromComplexColumn<T, M, N, I_idx, N - 1>::compute(From_matrix,
                                                                To_matrix);
    MatrixImagFromComplexRow<T, M, N, I_idx - 1>::compute(From_matrix,
                                                          To_matrix);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N>
struct MatrixImagFromComplexRow<T, M, N, 0> {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    MatrixImagFromComplexColumn<T, M, N, 0, N - 1>::compute(From_matrix,
                                                            To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline void BASE_MATRIX_COMPILED_MATRIX_IMAG_FROM_COMPLEX(
    const Matrix<Complex<T>, M, N> &From_matrix, Matrix<T, M, N> &To_matrix) {
  MatrixImagFromComplexRow<T, M, N, M - 1>::compute(From_matrix, To_matrix);
}

template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> get_imag_matrix_from_complex_matrix(
    const Matrix<Complex<T>, M, N> &From_matrix) {

  Matrix<T, M, N> To_matrix;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j) = From_matrix(i, j).imag;
    }
  }

#else

  BASE_MATRIX_COMPILED_MATRIX_IMAG_FROM_COMPLEX<T, M, N>(From_matrix,
                                                         To_matrix);

#endif

  return To_matrix;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_MATRIX_HPP
