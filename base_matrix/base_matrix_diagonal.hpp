#ifndef BASE_MATRIX_DIAGONAL_HPP
#define BASE_MATRIX_DIAGONAL_HPP

#include "base_matrix_macros.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"
#include <array>
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

template <typename T, std::size_t M> class DiagMatrix {
public:
#ifdef BASE_MATRIX_USE_STD_VECTOR

  DiagMatrix() : data(M, static_cast<T>(0)) {}

  DiagMatrix(const std::vector<T> &input) : data(input) {}

  DiagMatrix(const std::initializer_list<T> &input) : data(input) {}

  DiagMatrix(T input[M]) : data(M, static_cast<T>(0)) {

    for (std::size_t i = 0; i < M; i++) {
      this->data[i] = input[i];
    }
  }

#else

  DiagMatrix() : data{} {}

  DiagMatrix(const std::initializer_list<T> &input) : data{} {

    std::copy(input.begin(), input.end(), this->data.begin());
  }

  DiagMatrix(const std::array<T, M> &input) : data(input) {}

  DiagMatrix(const std::vector<T> &input) : data{} {

    std::copy(input.begin(), input.end(), this->data.begin());
  }

  DiagMatrix(T input[M]) : data{} {

    for (std::size_t i = 0; i < M; i++) {
      this->data[i] = input[i];
    }
  }

#endif

  /* Copy Constructor */
  DiagMatrix(const DiagMatrix<T, M> &other) : data(other.data) {}

  DiagMatrix<T, M> &operator=(const DiagMatrix<T, M> &other) {
    if (this != &other) {
      this->data = other.data;
    }
    return *this;
  }

  /* Move Constructor */
  DiagMatrix(DiagMatrix<T, M> &&other) noexcept : data(std::move(other.data)) {}

  DiagMatrix<T, M> &operator=(DiagMatrix<T, M> &&other) noexcept {
    if (this != &other) {
      this->data = std::move(other.data);
    }
    return *this;
  }

  /* Function */
  static DiagMatrix<T, M> identity() {
    DiagMatrix<T, M> identity(std::vector<T>(M, static_cast<T>(1)));

    return identity;
  }

  T &operator[](std::size_t index) { return this->data[index]; }

  const T &operator[](std::size_t index) const { return this->data[index]; }

  constexpr std::size_t rows() const { return M; }

  constexpr std::size_t cols() const { return M; }

  Vector<T, M> get_row(std::size_t row) const {
    if (row >= M) {
      row = M - 1;
    }

    Vector<T, M> result;
    result[row] = this->data[row];

    return result;
  }

  T get_trace() const { return output_trace(*this); }

  Matrix<T, M, M> create_dense() const { return output_dense(*this); }

  DiagMatrix<T, M> inv(T division_min) const {
    DiagMatrix<T, M> result;

    for (std::size_t i = 0; i < M; i++) {
      result[i] =
          static_cast<T>(1) / avoid_zero_divide(this->data[i], division_min);
    }

    return result;
  }

/* Variable */
#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> data;
#else
  std::array<T, M> data;
#endif
};

/* Matrix Addition */
// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx>
struct DiagMatrixAdderCore {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[M_idx] = A[M_idx] + B[M_idx];
    DiagMatrixAdderCore<T, M, M_idx - 1>::compute(A, B, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct DiagMatrixAdderCore<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[0] = A[0] + B[0];
  }
};

#define BASE_MATRIX_COMPILED_DIAG_MATRIX_ADDER(T, M, A, B, result)             \
  DiagMatrixAdderCore<T, M, M - 1>::compute(A, B, result);

template <typename T, std::size_t M>
DiagMatrix<T, M> operator+(const DiagMatrix<T, M> &A,
                           const DiagMatrix<T, M> &B) {
  DiagMatrix<T, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] + B[j];
  }

#else

  BASE_MATRIX_COMPILED_DIAG_MATRIX_ADDER(T, M, A, B, result);

#endif

  return result;
}

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx>
struct DiagMatrixAddMatrixCore {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
    result(M_idx, M_idx) += A[M_idx];
    DiagMatrixAddMatrixCore<T, M, M_idx - 1>::compute(A, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct DiagMatrixAddMatrixCore<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
    result(0, 0) += A[0];
  }
};

#define BASE_MATRIX_COMPILED_DIAG_MATRIX_ADD_MATRIX(T, M, A, result)           \
  DiagMatrixAddMatrixCore<T, M, M - 1>::compute(A, result);

template <typename T, std::size_t M>
Matrix<T, M, M> operator+(const DiagMatrix<T, M> &A, const Matrix<T, M, M> &B) {
  Matrix<T, M, M> result = B;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += A[i];
  }

#else

  BASE_MATRIX_COMPILED_DIAG_MATRIX_ADD_MATRIX(T, M, A, result);

#endif

  return result;
}

template <typename T, std::size_t M>
Matrix<T, M, M> operator+(const Matrix<T, M, M> &A, const DiagMatrix<T, M> &B) {
  Matrix<T, M, M> result = A;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += B[i];
  }

#else

  BASE_MATRIX_COMPILED_DIAG_MATRIX_ADD_MATRIX(T, M, B, result);

#endif

  return result;
}

/* Matrix Subtraction */
// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx>
struct DiagMatrixSubtractorCore {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[M_idx] = A[M_idx] - B[M_idx];
    DiagMatrixSubtractorCore<T, M, M_idx - 1>::compute(A, B, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct DiagMatrixSubtractorCore<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[0] = A[0] - B[0];
  }
};

#define BASE_MATRIX_COMPILED_DIAG_MATRIX_SUBTRACTOR(T, M, A, B, result)        \
  DiagMatrixSubtractorCore<T, M, M - 1>::compute(A, B, result);

template <typename T, std::size_t M>
DiagMatrix<T, M> operator-(const DiagMatrix<T, M> &A,
                           const DiagMatrix<T, M> &B) {
  DiagMatrix<T, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] - B[j];
  }

#else

  BASE_MATRIX_COMPILED_DIAG_MATRIX_SUBTRACTOR(T, M, A, B, result);

#endif

  return result;
}

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx>
struct DiagMatrixSubMatrixCore {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
    result(M_idx, M_idx) += A[M_idx];
    DiagMatrixSubMatrixCore<T, M, M_idx - 1>::compute(A, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct DiagMatrixSubMatrixCore<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
    result(0, 0) += A[0];
  }
};

#define BASE_MATRIX_COMPILED_DIAG_MATRIX_SUB_MATRIX(T, M, A, result)           \
  DiagMatrixSubMatrixCore<T, M, M - 1>::compute(A, result);

template <typename T, std::size_t M>
Matrix<T, M, M> operator-(const DiagMatrix<T, M> &A, const Matrix<T, M, M> &B) {
  Matrix<T, M, M> result = -B;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += A[i];
  }

#else

  BASE_MATRIX_COMPILED_DIAG_MATRIX_SUB_MATRIX(T, M, A, result);

#endif

  return result;
}

template <typename T, std::size_t M>
Matrix<T, M, M> operator-(const Matrix<T, M, M> &A, const DiagMatrix<T, M> &B) {
  Matrix<T, M, M> result = A;
  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) -= B[i];
  }

  return result;
}

/* Matrix multiply Scalar */
template <typename T, std::size_t M>
DiagMatrix<T, M> operator*(const DiagMatrix<T, M> &A, const T &scalar) {
  DiagMatrix<T, M> result;
  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * scalar;
  }

  return result;
}

/* Matrix multiply Vector */
template <typename T, std::size_t M>
Vector<T, M> operator*(const DiagMatrix<T, M> &A, const Vector<T, M> &vec) {
  Vector<T, M> result;
  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * vec[i];
  }

  return result;
}

/* Matrix Multiplication */
template <typename T, std::size_t M>
DiagMatrix<T, M> operator*(const DiagMatrix<T, M> &A,
                           const DiagMatrix<T, M> &B) {
  DiagMatrix<T, M> result;
  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] * B[j];
  }

  return result;
}

template <typename T, std::size_t M>
DiagMatrix<T, M> operator*(const T &scalar, const DiagMatrix<T, M> &A) {
  DiagMatrix<T, M> result;
  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * scalar;
  }

  return result;
}

template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> operator*(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B) {
  Matrix<T, M, N> result;
  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = A[j] * B(j, k);
    }
  }

  return result;
}

template <typename T, std::size_t L, std::size_t M>
Matrix<T, L, M> operator*(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B) {
  Matrix<T, L, M> result;
  for (std::size_t j = 0; j < L; ++j) {
    for (std::size_t k = 0; k < M; ++k) {
      result(j, k) = A(j, k) * B[k];
    }
  }

  return result;
}

/* Trace */
template <typename T, std::size_t M>
inline T output_trace(const DiagMatrix<T, M> &A) {
  T trace = static_cast<T>(0);
  for (std::size_t i = 0; i < M; i++) {
    trace += A[i];
  }
  return trace;
}

/* Create dense */
template <typename T, std::size_t M>
inline Matrix<T, M, M> output_dense(const DiagMatrix<T, M> &A) {
  Matrix<T, M, M> result;

  for (std::size_t i = 0; i < M; i++) {
    result(i, i) = A[i];
  }

  return result;
}

template <typename T, std::size_t M>
DiagMatrix<T, M> diag_divide_diag(const DiagMatrix<T, M> &A,
                                  const DiagMatrix<T, M> &B, T division_min) {
  DiagMatrix<T, M> result;
  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] / avoid_zero_divide(B[j], division_min);
  }

  return result;
}

template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> diag_inv_multiply_dense(const DiagMatrix<T, M> &A,
                                        const Matrix<T, M, N> &B) {
  Matrix<T, M, N> result;
  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = B(j, k) / A[j];
    }
  }

  return result;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_DIAGONAL_HPP
