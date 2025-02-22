#ifndef __BASE_MATRIX_MATRIX_HPP__
#define __BASE_MATRIX_MATRIX_HPP__

#include "base_matrix_macros.hpp"

#include "base_matrix_complex.hpp"
#include "base_matrix_vector.hpp"
#include "base_utility.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

template <typename T, std::size_t M, std::size_t N> class Matrix {
public:
#ifdef __BASE_MATRIX_USE_STD_VECTOR__

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
    // This may cause runtime error if the size of values is larger than M.
    std::copy(input.begin(), input.end(), this->data[0].begin());
  }

  Matrix(T input[][N]) : data(N, std::vector<T>(M, static_cast<T>(0))) {

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        this->data[j][i] = input[i][j];
      }
    }
  }

#else // __BASE_MATRIX_USE_STD_VECTOR__

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
    // This may cause runtime error if the size of values is larger than M.
    std::copy(input.begin(), input.end(), this->data[0].begin());
  }

  Matrix(const std::array<T, M> &input) : data{} {
    Base::Utility::copy<T, 0, M, 0, M, M>(input, this->data[0]);
  }

  Matrix(T input[][N]) : data{} {

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        this->data[j][i] = input[i][j];
      }
    }
  }

#endif // __BASE_MATRIX_USE_STD_VECTOR__

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

      identity.template set<P_idx, P_idx>(static_cast<U>(1));
      CreateIdentityCore<U, P, P_idx - 1>::compute(identity);
    }
  };

  // Termination condition: P_idx == 0
  template <typename U, std::size_t P> struct CreateIdentityCore<U, P, 0> {
    static void compute(Matrix<U, P, P> &identity) {

      identity.template set<0, 0>(static_cast<U>(1));
    }
  };

  template <typename U, std::size_t P>
  static inline void COMPILED_MATRIX_IDENTITY(Matrix<U, P, P> &identity) {
    CreateIdentityCore<U, P, P - 1>::compute(identity);
  }

  static inline Matrix<T, M, M> identity() {
    Matrix<T, M, M> identity;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    for (std::size_t i = 0; i < M; i++) {
      identity(i, i) = static_cast<T>(1);
    }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    COMPILED_MATRIX_IDENTITY<T, M>(identity);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    return identity;
  }

  /* Full */
  // when J_idx < P
  template <typename U, std::size_t O, std::size_t P, std::size_t I,
            std::size_t J_idx>
  struct MatrixFullColumn {
    static void compute(Matrix<U, O, P> &Full, const U &value) {

      Full.template set<I, J_idx>(value);
      MatrixFullColumn<U, O, P, I, J_idx - 1>::compute(Full, value);
    }
  };

  // column recursion termination
  template <typename U, std::size_t O, std::size_t P, std::size_t I>
  struct MatrixFullColumn<U, O, P, I, 0> {
    static void compute(Matrix<U, O, P> &Full, const U &value) {

      Full.template set<I, 0>(value);
    }
  };

  // when I_idx < M
  template <typename U, std::size_t O, std::size_t P, std::size_t I_idx>
  struct MatrixFullRow {
    static void compute(Matrix<U, O, P> &Full, const U &value) {
      MatrixFullColumn<U, O, P, I_idx, P - 1>::compute(Full, value);
      MatrixFullRow<U, O, P, I_idx - 1>::compute(Full, value);
    }
  };

  // row recursion termination
  template <typename U, std::size_t O, std::size_t P>
  struct MatrixFullRow<U, O, P, 0> {
    static void compute(Matrix<U, O, P> &Full, const U &value) {
      MatrixFullColumn<U, O, P, 0, P - 1>::compute(Full, value);
    }
  };

  template <typename U, std::size_t O, std::size_t P>
  static inline void COMPILED_MATRIX_FULL(Matrix<U, O, P> &Full,
                                          const U &value) {
    MatrixFullRow<U, O, P, O - 1>::compute(Full, value);
  }

  static inline Matrix<T, M, N> ones() {
    Matrix<T, M, N> Ones;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        Ones(i, j) = static_cast<T>(1);
      }
    }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    COMPILED_MATRIX_FULL<T, M, N>(Ones, static_cast<T>(1));

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    return Ones;
  }

  static inline Matrix<T, M, N> full(const T &value) {
    Matrix<T, M, N> Full;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        Full(i, j) = value;
      }
    }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    COMPILED_MATRIX_FULL<T, M, N>(Full, value);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    return Full;
  }

  inline Vector<T, M> create_row_vector(std::size_t row) const {
    Vector<T, M> result;

    if (row >= M) {
      row = M - 1;
    }

    Base::Utility::copy<T, 0, M, 0, M, M>(this->data[row], result.data);

    return result;
  }

  T &operator()(std::size_t col, std::size_t row) {

    return this->data[row][col];
  }

  const T &operator()(std::size_t col, std::size_t row) const {

    return this->data[row][col];
  }

#ifdef __BASE_MATRIX_USE_STD_VECTOR__

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

#else // __BASE_MATRIX_USE_STD_VECTOR__

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

#endif // __BASE_MATRIX_USE_STD_VECTOR__

  constexpr std::size_t rows() const { return N; }

  constexpr std::size_t cols() const { return M; }

  inline Vector<T, M> get_row(std::size_t row) const {
    if (row >= N) {
      row = N - 1;
    }

    Vector<T, M> result;

    Base::Utility::copy<T, 0, M, 0, M, M>(this->data[row], result.data);

    return result;
  }

  inline void set_row(std::size_t row, const Vector<T, M> &row_vector) {
    if (row >= N) {
      row = N - 1;
    }

    Base::Utility::copy<T, 0, M, 0, M, M>(row_vector.data, this->data[row]);
  }

  inline Matrix<T, M, M> inv() const {
    Matrix<T, M, M> X_temp = Matrix<T, M, M>::identity();
    std::array<T, M> rho;
    std::array<std::size_t, M> rep_num;

    Matrix<T, M, M> Inv =
        gmres_k_matrix_inv(*this, static_cast<T>(0.0), static_cast<T>(1.0e-10),
                           rho, rep_num, X_temp);

    return Inv;
  }

  /* Get Dense Matrix value */
  template <std::size_t COL, std::size_t ROW> inline T get() const {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    return data[ROW][COL];
  }

  /* Set Dense Matrix value */
  template <std::size_t COL, std::size_t ROW> inline void set(const T &value) {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    data[ROW][COL] = value;
  }

  inline T get_trace() const { return output_matrix_trace(*this); }

/* Variable */
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<std::vector<T>> data;
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<std::array<T, M>, N> data;
#endif // __BASE_MATRIX_USE_STD_VECTOR__
};

/* swap columns */
namespace MatrixSwapColumns {

// Swap N_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t N_idx>
struct Core {
  static void compute(std::size_t col_1, std::size_t col_2,
                      Matrix<T, M, N> &mat, T temp) {

    temp = mat.data[N_idx][col_1];
    mat.data[N_idx][col_1] = mat.data[N_idx][col_2];
    mat.data[N_idx][col_2] = temp;
    Core<T, M, N, N_idx - 1>::compute(col_1, col_2, mat, temp);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t M, std::size_t N> struct Core<T, M, N, 0> {
  static void compute(std::size_t col_1, std::size_t col_2,
                      Matrix<T, M, N> &mat, T temp) {

    temp = mat.data[0][col_1];
    mat.data[0][col_1] = mat.data[0][col_2];
    mat.data[0][col_2] = temp;
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(std::size_t col_1, std::size_t col_2, Matrix<T, M, N> &mat,
                    T &temp) {
  Core<T, M, N, N - 1>::compute(col_1, col_2, mat, temp);
}

} // namespace MatrixSwapColumns

template <typename T, std::size_t M, std::size_t N>
inline void matrix_col_swap(std::size_t col_1, std::size_t col_2,
                            Matrix<T, M, N> &mat) {
  T temp = static_cast<T>(0);

  if (col_1 >= M) {
    col_1 = M - 1;
  }
  if (col_2 >= M) {
    col_2 = M - 1;
  }

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; i++) {

    temp = mat.data[i][col_1];
    mat.data[i][col_1] = mat.data[i][col_2];
    mat.data[i][col_2] = temp;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixSwapColumns::compute<T, M, N>(col_1, col_2, mat, temp);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

/* swap rows */
template <typename T, std::size_t M, std::size_t N>
inline void matrix_row_swap(std::size_t row_1, std::size_t row_2,
                            Matrix<T, M, N> &mat) {
  Vector<T, M> temp_vec;

  if (row_1 >= N) {
    row_1 = N - 1;
  }
  if (row_2 >= N) {
    row_2 = N - 1;
  }

  Base::Utility::copy<T, 0, M, 0, M, M>(mat(row_1), temp_vec.data);
  Base::Utility::copy<T, 0, M, 0, M, M>(mat(row_2), mat(row_1));
  Base::Utility::copy<T, 0, M, 0, M, M>(temp_vec.data, mat(row_2));
}

/* Trace */
namespace MatrixTrace {

// calculate trace of matrix
template <typename T, std::size_t N, std::size_t I> struct Core {
  static T compute(const Matrix<T, N, N> &mat) {
    return mat.template get<I, I>() + Core<T, N, I - 1>::compute(mat);
  }
};

// if I == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  static T compute(const Matrix<T, N, N> &mat) {
    return mat.template get<0, 0>();
  }
};

template <typename T, std::size_t M, std::size_t N>
inline T compute(const Matrix<T, M, N> &mat) {
  return Core<T, N, N - 1>::compute(mat);
}

} // namespace MatrixTrace

template <typename T, std::size_t M, std::size_t N>
inline T output_matrix_trace(const Matrix<T, M, N> &mat) {
  static_assert(M == N, "Matrix must be square matrix");
  T trace = static_cast<T>(0);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; i++) {
    trace += mat(i, i);
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  trace = MatrixTrace::compute<T, M, N>(mat);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return trace;
}

/* Matrix Addition */
namespace MatrixAddMatrix {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, J_idx>(A.template get<I, J_idx>() +
                                  B.template get<I, J_idx>());
    Column<T, M, N, I, J_idx - 1>::compute(A, B, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(A.template get<I, 0>() + B.template get<I, 0>());
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, I_idx, N - 1>::compute(A, B, result);
    Row<T, M, N, I_idx - 1>::compute(A, B, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, 0, N - 1>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                    Matrix<T, M, N> &result) {
  Row<T, M, N, M - 1>::compute(A, B, result);
}

} // namespace MatrixAddMatrix

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator+(const Matrix<T, M, N> &A,
                                 const Matrix<T, M, N> &B) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(i, j) = A(i, j) + B(i, j);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixAddMatrix::compute<T, M, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix Subtraction */
namespace MatrixSubMatrix {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, J_idx>(A.template get<I, J_idx>() -
                                  B.template get<I, J_idx>());
    Column<T, M, N, I, J_idx - 1>::compute(A, B, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(A.template get<I, 0>() - B.template get<I, 0>());
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, I_idx, N - 1>::compute(A, B, result);
    Row<T, M, N, I_idx - 1>::compute(A, B, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, 0, N - 1>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                    Matrix<T, M, N> &result) {
  Row<T, M, N, M - 1>::compute(A, B, result);
}

} // namespace MatrixSubMatrix

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator-(const Matrix<T, M, N> &A,
                                 const Matrix<T, M, N> &B) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(i, j) = A(i, j) - B(i, j);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixSubMatrix::compute<T, M, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

namespace MatrixMinus {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {

    result.template set<I, J_idx>(-A.template get<I, J_idx>());
    Column<T, M, N, I, J_idx - 1>::compute(A, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {

    result.template set<I, 0>(-A.template get<I, 0>());
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
    Column<T, M, N, I_idx, N - 1>::compute(A, result);
    Row<T, M, N, I_idx - 1>::compute(A, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
    Column<T, M, N, 0, N - 1>::compute(A, result);
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
  Row<T, M, N, M - 1>::compute(A, result);
}

} // namespace MatrixMinus

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator-(const Matrix<T, M, N> &A) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(i, j) = -A(i, j);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixMinus::compute<T, M, N>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* (Scalar) * (Matrix) */
namespace MatrixMultiplyScalar {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {

    result.template set<I, J_idx>(scalar * mat.template get<I, J_idx>());
    Column<T, M, N, I, J_idx - 1>::compute(scalar, mat, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(scalar * mat.template get<I, 0>());
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, I_idx, N - 1>::compute(scalar, mat, result);
    Row<T, M, N, I_idx - 1>::compute(scalar, mat, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, 0, N - 1>::compute(scalar, mat, result);
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(const T &scalar, const Matrix<T, M, N> &mat,
                    Matrix<T, M, N> &result) {
  Row<T, M, N, M - 1>::compute(scalar, mat, result);
}

} // namespace MatrixMultiplyScalar

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator*(const T &scalar, const Matrix<T, M, N> &mat) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = scalar * mat(j, k);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixMultiplyScalar::compute<T, M, N>(scalar, mat, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator*(const Matrix<T, M, N> &mat, const T &scalar) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = scalar * mat(j, k);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixMultiplyScalar::compute<T, M, N>(scalar, mat, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix multiply Vector */
namespace MatrixMultiplyVector {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J>
struct Core {
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec) {

    return mat.template get<I, J>() * vec[J] +
           Core<T, M, N, I, J - 1>::compute(mat, vec);
  }
};

// if J == 0
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Core<T, M, N, I, 0> {
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec) {
    return mat.template get<I, 0>() * vec[0];
  }
};

// column recursion
template <typename T, std::size_t M, std::size_t N, std::size_t I> struct Row {
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec,
                      Vector<T, M> &result) {
    result[I] = Core<T, M, N, I, N - 1>::compute(mat, vec);
    Row<T, M, N, I - 1>::compute(mat, vec, result);
  }
};

// if I == 0
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec,
                      Vector<T, M> &result) {
    result[0] = Core<T, M, N, 0, N - 1>::compute(mat, vec);
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec,
                    Vector<T, M> &result) {
  Row<T, M, N, M - 1>::compute(mat, vec, result);
}

} // namespace MatrixMultiplyVector

template <typename T, std::size_t M, std::size_t N>
inline Vector<T, M> operator*(const Matrix<T, M, N> &mat,
                              const Vector<T, N> &vec) {
  Vector<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    T sum = 0;
    for (std::size_t j = 0; j < N; ++j) {
      sum += mat(i, j) * vec[j];
    }
    result[i] = sum;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixMultiplyVector::compute<T, M, N>(mat, vec, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

namespace VectorMultiplyMatrix {

// calculate if K_idx > 0
template <typename T, std::size_t L, std::size_t N, std::size_t J,
          std::size_t K_idx>
struct Core {
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {

    result.template set<K_idx, J>(vec[K_idx] * mat.template get<0, J>());
    Core<T, L, N, J, K_idx - 1>::compute(vec, mat, result);
  }
};

// if K_idx = 0
template <typename T, std::size_t L, std::size_t N, std::size_t J>
struct Core<T, L, N, J, 0> {
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {
    result.template set<0, J>(vec[0] * mat.template get<0, J>());
  }
};

// row recursion
template <typename T, std::size_t L, std::size_t N, std::size_t J>
struct Column {
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {
    Core<T, L, N, J, L - 1>::compute(vec, mat, result);
    Column<T, L, N, J - 1>::compute(vec, mat, result);
  }
};

// if J = 0
template <typename T, std::size_t L, std::size_t N> struct Column<T, L, N, 0> {
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {
    Core<T, L, N, 0, L - 1>::compute(vec, mat, result);
  }
};

template <typename T, std::size_t L, std::size_t M, std::size_t N>
inline void compute(const Vector<T, L> &vec, const Matrix<T, M, N> &mat,
                    Matrix<T, L, N> &result) {
  Column<T, L, N, N - 1>::compute(vec, mat, result);
}

} // namespace VectorMultiplyMatrix

template <typename T, std::size_t L, std::size_t M, std::size_t N>
inline Matrix<T, L, N> operator*(const Vector<T, L> &vec,
                                 const Matrix<T, M, N> &mat) {
  static_assert(M == 1, "Invalid size.");
  Matrix<T, L, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t k = 0; k < L; ++k) {
      result(k, j) = vec[k] * mat(0, j);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  VectorMultiplyMatrix::compute<T, L, M, N>(vec, mat, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* (Column Vector) * (Matrix) */
namespace ColumnVectorMultiplyMatrix {

// calculation when I > 0
template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t I>
struct Core {
  static T compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat) {
    return vec[I] * mat.template get<I, J>() +
           Core<T, M, N, J, I - 1>::compute(vec, mat);
  }
};

// if I = 0
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct Core<T, M, N, J, 0> {
  static T compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat) {
    return vec[0] * mat.template get<0, J>();
  }
};

// row recursion
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct Column {
  static void compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat,
                      ColVector<T, N> &result) {
    result[J] = Core<T, M, N, J, M - 1>::compute(vec, mat);
    Column<T, M, N, J - 1>::compute(vec, mat, result);
  }
};

// if J = 0
template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, 0> {
  static void compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat,
                      ColVector<T, N> &result) {
    result[0] = Core<T, M, N, 0, M - 1>::compute(vec, mat);
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat,
                    ColVector<T, N> &result) {
  Column<T, M, N, N - 1>::compute(vec, mat, result);
}

} // namespace ColumnVectorMultiplyMatrix

template <typename T, std::size_t M, std::size_t N>
inline ColVector<T, N> operator*(const ColVector<T, M> &vec,
                                 const Matrix<T, M, N> &mat) {
  ColVector<T, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < N; ++j) {
    T sum = 0;
    for (std::size_t i = 0; i < M; ++i) {
      sum += vec[i] * mat(i, j);
    }
    result[j] = sum;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  ColumnVectorMultiplyMatrix::compute<T, M, N>(vec, mat, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix Multiply Matrix */
namespace MatrixMultiplyMatrix {

// when K_idx < K
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t K_idx>
struct Core {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {

    return A.template get<I, K_idx>() * B.template get<K_idx, J>() +
           Core<T, M, K, N, I, J, K_idx - 1>::compute(A, B);
  }
};

// when K_idx reached 0
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Core<T, M, K, N, I, J, 0> {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {

    return A.template get<I, 0>() * B.template get<0, J>();
  }
};

// After completing the J column, go to the next row I
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Column {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, J>(Core<T, M, K, N, I, J, K - 1>::compute(A, B));
    Column<T, M, K, N, I, J - 1>::compute(A, B, result);
  }
};

// Row recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Column<T, M, K, N, I, 0> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(Core<T, M, K, N, I, 0, K - 1>::compute(A, B));
  }
};

// proceed to the next row after completing the I row
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Row {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, I, N - 1>::compute(A, B, result);
    Row<T, M, K, N, I - 1>::compute(A, B, result);
  }
};

// Column recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N>
struct Row<T, M, K, N, 0> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, 0, N - 1>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                    Matrix<T, M, N> &result) {
  Row<T, M, K, N, M - 1>::compute(A, B, result);
}

} // namespace MatrixMultiplyMatrix

template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline Matrix<T, M, N> operator*(const Matrix<T, M, K> &A,
                                 const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixMultiplyMatrix::compute<T, M, K, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Transpose */
namespace MatrixTranspose {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {

    result.template set<J_idx, I>(A.template get<I, J_idx>());
    Column<T, M, N, I, J_idx - 1>::compute(A, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {

    result.template set<0, I>(A.template get<I, 0>());
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
    Column<T, M, N, I_idx, N - 1>::compute(A, result);
    Row<T, M, N, I_idx - 1>::compute(A, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
    Column<T, M, N, 0, N - 1>::compute(A, result);
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
  Row<T, M, N, M - 1>::compute(A, result);
}

} // namespace MatrixTranspose

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, N, M> output_matrix_transpose(const Matrix<T, M, N> &mat) {
  Matrix<T, N, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(j, i) = mat(i, j);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixTranspose::compute<T, M, N>(mat, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Upper Triangular Matrix Multiply Matrix */
namespace UpperTriangularMultiplyMatrix {

// when K_idx >= I
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t K_idx>
struct Core {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {

    return (K_idx >= I)
               ? (A.template get<I, K_idx>() * B.template get<K_idx, J>() +
                  Core<T, M, K, N, I, J, K_idx - 1>::compute(A, B))
               : static_cast<T>(0);
  }
};

// recursion termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Core<T, M, K, N, I, J, static_cast<std::size_t>(-1)> {
  static T compute(const Matrix<T, M, K> &, const Matrix<T, K, N> &) {

    return static_cast<T>(0);
  }
};

// when K_idx reaches I (base case)
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Core<T, M, K, N, I, J, I> {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {

    return A.template get<I, I>() * B.template get<I, J>();
  }
};

// Column-wise computation
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Column {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, J>(Core<T, M, K, N, I, J, K - 1>::compute(A, B));
    Column<T, M, K, N, I, J - 1>::compute(A, B, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Column<T, M, K, N, I, 0> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(Core<T, M, K, N, I, 0, K - 1>::compute(A, B));
  }
};

// Row-wise computation
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Row {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, I, N - 1>::compute(A, B, result);
    Row<T, M, K, N, I - 1>::compute(A, B, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t K, std::size_t N>
struct Row<T, M, K, N, 0> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, 0, N - 1>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                    Matrix<T, M, N> &result) {
  Row<T, M, K, N, M - 1>::compute(A, B, result);
}

} // namespace UpperTriangularMultiplyMatrix

template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline Matrix<T, M, N>
matrix_multiply_Upper_triangular_A_mul_B(const Matrix<T, M, K> &A,
                                         const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = i; k < K; ++k) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  UpperTriangularMultiplyMatrix::compute<T, M, K, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix Transpose Multiply Matrix */
namespace MatrixTransposeMultiplyMatrix {

// when K_idx < K
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t K_idx>
struct Core {
  static T compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B) {

    return A.template get<K_idx, I>() * B.template get<K_idx, J>() +
           Core<T, M, K, N, I, J, K_idx - 1>::compute(A, B);
  }
};

// when K_idx reached 0
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Core<T, M, K, N, I, J, 0> {
  static T compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B) {

    return A.template get<0, I>() * B.template get<0, J>();
  }
};

// After completing the J column, go to the next row I
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Column {
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, J>(Core<T, M, K, N, I, J, K - 1>::compute(A, B));
    Column<T, M, K, N, I, J - 1>::compute(A, B, result);
  }
};

// Row recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Column<T, M, K, N, I, 0> {
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(Core<T, M, K, N, I, 0, K - 1>::compute(A, B));
  }
};

// proceed to the next row after completing the I row
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Row {
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, I, N - 1>::compute(A, B, result);
    Row<T, M, K, N, I - 1>::compute(A, B, result);
  }
};

// Column recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N>
struct Row<T, M, K, N, 0> {
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, 0, N - 1>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                    Matrix<T, M, N> &result) {
  Row<T, M, K, N, M - 1>::compute(A, B, result);
}

} // namespace MatrixTransposeMultiplyMatrix

template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline Matrix<T, M, N> matrix_multiply_AT_mul_B(const Matrix<T, K, M> &A,
                                                const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(k, i) * B(k, j);
      }
      result(i, j) = sum;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixTransposeMultiplyMatrix::compute<T, M, K, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Transpose Matrix multiply Vector  */
namespace MatrixTransposeMultiplyVector {

// when M_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t N_idx,
          std::size_t M_idx>
struct Core {
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec) {

    return mat.template get<M_idx, N_idx>() * vec[M_idx] +
           Core<T, M, N, N_idx, M_idx - 1>::compute(mat, vec);
  }
};

// if M_idx == 0
template <typename T, std::size_t M, std::size_t N, std::size_t N_idx>
struct Core<T, M, N, N_idx, 0> {
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec) {
    return mat.template get<0, N_idx>() * vec[0];
  }
};

// column recursion
template <typename T, std::size_t M, std::size_t N, std::size_t N_idx>
struct Column {
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec,
                      Vector<T, N> &result) {
    result[N_idx] = Core<T, M, N, N_idx, M - 1>::compute(mat, vec);
    Column<T, M, N, N_idx - 1>::compute(mat, vec, result);
  }
};

// if N_idx == 0
template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec,
                      Vector<T, N> &result) {
    result[0] = Core<T, M, N, 0, M - 1>::compute(mat, vec);
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec,
                    Vector<T, N> &result) {
  Column<T, M, N, N - 1>::compute(mat, vec, result);
}

} // namespace MatrixTransposeMultiplyVector

template <typename T, std::size_t M, std::size_t N>
inline Vector<T, N> matrix_multiply_AT_mul_b(const Matrix<T, M, N> &A,
                                             const Vector<T, M> &b) {
  Vector<T, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t n = 0; n < N; ++n) {
    T sum = 0;
    for (std::size_t m = 0; m < M; ++m) {
      sum += A(m, n) * b[m];
    }
    result[n] = sum;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixTransposeMultiplyVector::compute<T, M, N>(A, b, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix multiply Transpose Matrix */
namespace MatrixMultiplyTransposeMatrix {

// when K_idx < K
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t K_idx>
struct Core {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B) {

    return A.template get<I, K_idx>() * B.template get<J, K_idx>() +
           Core<T, M, K, N, I, J, K_idx - 1>::compute(A, B);
  }
};

// when K_idx reached 0
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Core<T, M, K, N, I, J, 0> {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B) {

    return A.template get<I, 0>() * B.template get<J, 0>();
  }
};

// After completing the J column, go to the next row I
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Column {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, J>(Core<T, M, K, N, I, J, K - 1>::compute(A, B));
    Column<T, M, K, N, I, J - 1>::compute(A, B, result);
  }
};

// Row recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Column<T, M, K, N, I, 0> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(Core<T, M, K, N, I, 0, K - 1>::compute(A, B));
  }
};

// proceed to the next row after completing the I row
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Row {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, I, N - 1>::compute(A, B, result);
    Row<T, M, K, N, I - 1>::compute(A, B, result);
  }
};

// Column recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N>
struct Row<T, M, K, N, 0> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, 0, N - 1>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                    Matrix<T, M, N> &result) {
  Row<T, M, K, N, M - 1>::compute(A, B, result);
}

} // namespace MatrixMultiplyTransposeMatrix

template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline Matrix<T, M, N>
matrix_multiply_A_mul_BTranspose(const Matrix<T, M, K> &A,
                                 const Matrix<T, N, K> &B) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(i, k) * B(j, k);
      }
      result(i, j) = sum;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixMultiplyTransposeMatrix::compute<T, M, K, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix real from complex */
namespace MatrixRealToComplex {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {

    To_matrix(I, J_idx).real = From_matrix.template get<I, J_idx>();
    Column<T, M, N, I, J_idx - 1>::compute(From_matrix, To_matrix);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {

    To_matrix(I, 0).real = From_matrix.template get<I, 0>();
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {
    Column<T, M, N, I_idx, N - 1>::compute(From_matrix, To_matrix);
    Row<T, M, N, I_idx - 1>::compute(From_matrix, To_matrix);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {
    Column<T, M, N, 0, N - 1>::compute(From_matrix, To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &From_matrix,
                    Matrix<Complex<T>, M, N> &To_matrix) {
  Row<T, M, N, M - 1>::compute(From_matrix, To_matrix);
}

} // namespace MatrixRealToComplex

template <typename T, std::size_t M, std::size_t N>
inline Matrix<Complex<T>, M, N>
convert_matrix_real_to_complex(const Matrix<T, M, N> &From_matrix) {

  Matrix<Complex<T>, M, N> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j).real = From_matrix(i, j);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixRealToComplex::compute<T, M, N>(From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Matrix real from complex */
namespace MatrixRealFromComplex {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {

    To_matrix.template set<I, J_idx>(From_matrix(I, J_idx).real);
    Column<T, M, N, I, J_idx - 1>::compute(From_matrix, To_matrix);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {

    To_matrix.template set<I, 0>(From_matrix(I, 0).real);
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Column<T, M, N, I_idx, N - 1>::compute(From_matrix, To_matrix);
    Row<T, M, N, I_idx - 1>::compute(From_matrix, To_matrix);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Column<T, M, N, 0, N - 1>::compute(From_matrix, To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                    Matrix<T, M, N> &To_matrix) {
  Row<T, M, N, M - 1>::compute(From_matrix, To_matrix);
}

} // namespace MatrixRealFromComplex

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> get_real_matrix_from_complex_matrix(
    const Matrix<Complex<T>, M, N> &From_matrix) {

  Matrix<T, M, N> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j) = From_matrix(i, j).real;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixRealFromComplex::compute<T, M, N>(From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Matrix imag from complex */
namespace MatrixImagFromComplex {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {

    To_matrix.template set<I, J_idx>(From_matrix(I, J_idx).imag);
    Column<T, M, N, I, J_idx - 1>::compute(From_matrix, To_matrix);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {

    To_matrix.template set<I, 0>(From_matrix(I, 0).imag);
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Column<T, M, N, I_idx, N - 1>::compute(From_matrix, To_matrix);
    Row<T, M, N, I_idx - 1>::compute(From_matrix, To_matrix);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Column<T, M, N, 0, N - 1>::compute(From_matrix, To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                    Matrix<T, M, N> &To_matrix) {
  Row<T, M, N, M - 1>::compute(From_matrix, To_matrix);
}

} // namespace MatrixImagFromComplex

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> get_imag_matrix_from_complex_matrix(
    const Matrix<Complex<T>, M, N> &From_matrix) {

  Matrix<T, M, N> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j) = From_matrix(i, j).imag;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixImagFromComplex::compute<T, M, N>(From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_MATRIX_HPP__
