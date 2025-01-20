#ifndef __BASE_MATRIX_DIAGONAL_HPP__
#define __BASE_MATRIX_DIAGONAL_HPP__

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
#ifdef __BASE_MATRIX_USE_STD_VECTOR__

  DiagMatrix() : data(M, static_cast<T>(0)) {}

  DiagMatrix(const std::vector<T> &input) : data(input) {}

  DiagMatrix(const std::initializer_list<T> &input) : data(input) {}

  DiagMatrix(T input[M]) : data(M, static_cast<T>(0)) {

    for (std::size_t i = 0; i < M; i++) {
      this->data[i] = input[i];
    }
  }

#else // __BASE_MATRIX_USE_STD_VECTOR__

  DiagMatrix() : data{} {}

  DiagMatrix(const std::initializer_list<T> &input) : data{} {

    // This may cause runtime error if the size of values is larger than M.
    std::copy(input.begin(), input.end(), this->data.begin());
  }

  DiagMatrix(const std::array<T, M> &input) : data(input) {}

  DiagMatrix(const std::vector<T> &input) : data{} {

    // This may cause runtime error if the size of values is larger than M.
    std::copy(input.begin(), input.end(), this->data.begin());
  }

  DiagMatrix(T input[M]) : data{} {

    for (std::size_t i = 0; i < M; i++) {
      this->data[i] = input[i];
    }
  }

#endif // __BASE_MATRIX_USE_STD_VECTOR__

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
  static inline DiagMatrix<T, M> identity() {
    DiagMatrix<T, M> identity(std::vector<T>(M, static_cast<T>(1)));

    return identity;
  }

  T &operator[](std::size_t index) { return this->data[index]; }

  const T &operator[](std::size_t index) const { return this->data[index]; }

  constexpr std::size_t rows() const { return M; }

  constexpr std::size_t cols() const { return M; }

  inline Vector<T, M> get_row(std::size_t row) const {
    if (row >= M) {
      row = M - 1;
    }

    Vector<T, M> result;
    result[row] = this->data[row];

    return result;
  }

  inline DiagMatrix<T, M> inv(T division_min) const {
    DiagMatrix<T, M> result;

    for (std::size_t i = 0; i < M; i++) {
      result[i] = static_cast<T>(1) /
                  Base::Utility::avoid_zero_divide(this->data[i], division_min);
    }

    return result;
  }

/* Variable */
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<T> data;
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<T, M> data;
#endif // __BASE_MATRIX_USE_STD_VECTOR__
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

template <typename T, std::size_t M>
static inline void COMPILED_DIAG_MATRIX_ADDER(const DiagMatrix<T, M> &A,
                                              const DiagMatrix<T, M> &B,
                                              DiagMatrix<T, M> &result) {
  DiagMatrixAdderCore<T, M, M - 1>::compute(A, B, result);
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator+(const DiagMatrix<T, M> &A,
                                  const DiagMatrix<T, M> &B) {
  DiagMatrix<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] + B[j];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DIAG_MATRIX_ADDER<T, M>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

template <typename T, std::size_t M>
static inline void COMPILED_DIAG_MATRIX_ADD_MATRIX(const DiagMatrix<T, M> &A,
                                                   Matrix<T, M, M> &result) {
  DiagMatrixAddMatrixCore<T, M, M - 1>::compute(A, result);
}

template <typename T, std::size_t M>
inline Matrix<T, M, M> operator+(const DiagMatrix<T, M> &A,
                                 const Matrix<T, M, M> &B) {
  Matrix<T, M, M> result = B;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DIAG_MATRIX_ADD_MATRIX<T, M>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

template <typename T, std::size_t M>
inline Matrix<T, M, M> operator+(const Matrix<T, M, M> &A,
                                 const DiagMatrix<T, M> &B) {
  Matrix<T, M, M> result = A;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += B[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DIAG_MATRIX_ADD_MATRIX<T, M>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Minus */
// when I_idx < M
template <typename T, std::size_t M, std::size_t I_idx>
struct DiagMatrixMinusLoop {
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> &result) {
    result[I_idx] = -A[I_idx];
    DiagMatrixMinusLoop<T, M, I_idx - 1>::compute(A, result);
  }
};

// row recursion termination
template <typename T, std::size_t M> struct DiagMatrixMinusLoop<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> &result) {
    result[0] = -A[0];
  }
};

template <typename T, std::size_t M>
static inline void COMPILED_MATRIX_MINUS_DIAG_MATRIX(const DiagMatrix<T, M> &A,
                                                     DiagMatrix<T, M> &result) {
  DiagMatrixMinusLoop<T, M, M - 1>::compute(A, result);
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator-(const DiagMatrix<T, M> &A) {
  DiagMatrix<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = -A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_MATRIX_MINUS_DIAG_MATRIX<T, M>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

template <typename T, std::size_t M>
static inline void COMPILED_DIAG_MATRIX_SUBTRACTOR(const DiagMatrix<T, M> &A,
                                                   const DiagMatrix<T, M> &B,
                                                   DiagMatrix<T, M> &result) {
  DiagMatrixSubtractorCore<T, M, M - 1>::compute(A, B, result);
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator-(const DiagMatrix<T, M> &A,
                                  const DiagMatrix<T, M> &B) {
  DiagMatrix<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] - B[j];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DIAG_MATRIX_SUBTRACTOR<T, M>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

template <typename T, std::size_t M>
static inline void COMPILED_DIAG_MATRIX_SUB_MATRIX(const Matrix<T, M, M> &A,
                                                   Matrix<T, M, M> &result) {
  DiagMatrixSubMatrixCore<T, M, M - 1>::compute(A, result);
}

template <typename T, std::size_t M>
inline Matrix<T, M, M> operator-(const DiagMatrix<T, M> &A,
                                 const Matrix<T, M, M> &B) {
  Matrix<T, M, M> result = -B;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DIAG_MATRIX_SUB_MATRIX<T, M>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx>
struct MatrixSubDiagMatrixCore {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
    result(M_idx, M_idx) -= A[M_idx];
    MatrixSubDiagMatrixCore<T, M, M_idx - 1>::compute(A, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct MatrixSubDiagMatrixCore<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
    result(0, 0) -= A[0];
  }
};

template <typename T, std::size_t M>
static inline void COMPILED_MATRIX_SUB_DIAG_MATRIX(const DiagMatrix<T, M> &A,
                                                   Matrix<T, M, M> &result) {
  MatrixSubDiagMatrixCore<T, M, M - 1>::compute(A, result);
}

template <typename T, std::size_t M>
inline Matrix<T, M, M> operator-(const Matrix<T, M, M> &A,
                                 const DiagMatrix<T, M> &B) {
  Matrix<T, M, M> result = A;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) -= B[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_MATRIX_SUB_DIAG_MATRIX<T, M>(B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix multiply Scalar */
// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx>
struct DiagMatrixMultiplyScalarCore {
  static void compute(const DiagMatrix<T, M> &mat, const T scalar,
                      DiagMatrix<T, M> &result) {
    result[M_idx] = mat[M_idx] * scalar;
    DiagMatrixMultiplyScalarCore<T, M, M_idx - 1>::compute(mat, scalar, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M>
struct DiagMatrixMultiplyScalarCore<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &mat, const T scalar,
                      DiagMatrix<T, M> &result) {
    result[0] = mat[0] * scalar;
  }
};

template <typename T, std::size_t M>
static inline void COMPILED_DIAG_MATRIX_MULTIPLY_SCALAR(
    const DiagMatrix<T, M> &mat, const T &scalar, DiagMatrix<T, M> &result) {
  DiagMatrixMultiplyScalarCore<T, M, M - 1>::compute(mat, scalar, result);
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator*(const DiagMatrix<T, M> &A, const T &scalar) {
  DiagMatrix<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * scalar;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DIAG_MATRIX_MULTIPLY_SCALAR<T, M>(A, scalar, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator*(const T &scalar, const DiagMatrix<T, M> &A) {
  DiagMatrix<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * scalar;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DIAG_MATRIX_MULTIPLY_SCALAR<T, M>(A, scalar, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix multiply Vector */
// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx>
struct DiagMatrixMultiplyVectorCore {
  static void compute(const DiagMatrix<T, M> &A, Vector<T, M> vec,
                      Vector<T, M> &result) {
    result[M_idx] = A[M_idx] * vec[M_idx];
    DiagMatrixMultiplyVectorCore<T, M, M_idx - 1>::compute(A, vec, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M>
struct DiagMatrixMultiplyVectorCore<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, Vector<T, M> vec,
                      Vector<T, M> &result) {
    result[0] = A[0] * vec[0];
  }
};

template <typename T, std::size_t M>
static inline void COMPILED_DIAG_MATRIX_MULTIPLY_VECTOR(
    const DiagMatrix<T, M> &A, const Vector<T, M> &vec, Vector<T, M> &result) {
  DiagMatrixMultiplyVectorCore<T, M, M - 1>::compute(A, vec, result);
}

template <typename T, std::size_t M>
inline Vector<T, M> operator*(const DiagMatrix<T, M> &A,
                              const Vector<T, M> &vec) {
  Vector<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * vec[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DIAG_MATRIX_MULTIPLY_VECTOR<T, M>(A, vec, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix Multiplication */
// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx>
struct DiagMatrixMultiplyDiagCore {
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[M_idx] = A[M_idx] * B[M_idx];
    DiagMatrixMultiplyDiagCore<T, M, M_idx - 1>::compute(A, B, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M>
struct DiagMatrixMultiplyDiagCore<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[0] = A[0] * B[0];
  }
};

template <typename T, std::size_t M>
static inline void
COMPILED_DIAG_MATRIX_MULTIPLY_DIAG(const DiagMatrix<T, M> &A,
                                   const DiagMatrix<T, M> &B,
                                   DiagMatrix<T, M> &result) {
  DiagMatrixMultiplyDiagCore<T, M, M - 1>::compute(A, B, result);
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator*(const DiagMatrix<T, M> &A,
                                  const DiagMatrix<T, M> &B) {
  DiagMatrix<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] * B[j];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DIAG_MATRIX_MULTIPLY_DIAG<T, M>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Diag Matrix Multiply Matrix */
// Core multiplication for DiagMatrix and Matrix
template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t K>
struct DiagMatrixMultiplyMatrixCore {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    result(J, K) = A[J] * B(J, K);
    DiagMatrixMultiplyMatrixCore<T, M, N, J, K - 1>::compute(A, B, result);
  }
};

// Specialization for K == 0
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct DiagMatrixMultiplyMatrixCore<T, M, N, J, 0> {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    result(J, 0) = A[J] * B(J, 0);
  }
};

// Column-wise multiplication
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct DiagMatrixMultiplyMatrixColumn {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    DiagMatrixMultiplyMatrixCore<T, M, N, J, N - 1>::compute(A, B, result);
    DiagMatrixMultiplyMatrixColumn<T, M, N, J - 1>::compute(A, B, result);
  }
};

// Specialization for J == 0
template <typename T, std::size_t M, std::size_t N>
struct DiagMatrixMultiplyMatrixColumn<T, M, N, 0> {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    DiagMatrixMultiplyMatrixCore<T, M, N, 0, N - 1>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline void
COMPILED_DIAG_MATRIX_MULTIPLY_MATRIX(const DiagMatrix<T, M> &A,
                                     const Matrix<T, M, N> &B,
                                     Matrix<T, M, N> &result) {
  DiagMatrixMultiplyMatrixColumn<T, M, N, M - 1>::compute(A, B, result);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator*(const DiagMatrix<T, M> &A,
                                 const Matrix<T, M, N> &B) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = A[j] * B(j, k);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DIAG_MATRIX_MULTIPLY_MATRIX<T, M, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

// Core multiplication for each element
template <typename T, std::size_t L, std::size_t M, std::size_t I,
          std::size_t J>
struct DiagMatrixMultiplierCore {
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {
    result(I, J) = A(I, J) * B[J];
    DiagMatrixMultiplierCore<T, L, M, I, J - 1>::compute(A, B, result);
  }
};

// Specialization for J = 0
template <typename T, std::size_t L, std::size_t M, std::size_t I>
struct DiagMatrixMultiplierCore<T, L, M, I, 0> {
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {
    result(I, 0) = A(I, 0) * B[0];
  }
};

// Column-wise multiplication
template <typename T, std::size_t L, std::size_t M, std::size_t I>
struct DiagMatrixMultiplierColumn {
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {
    DiagMatrixMultiplierCore<T, L, M, I, M - 1>::compute(A, B, result);
    DiagMatrixMultiplierColumn<T, L, M, I - 1>::compute(A, B, result);
  }
};

// Specialization for I = 0
template <typename T, std::size_t L, std::size_t M>
struct DiagMatrixMultiplierColumn<T, L, M, 0> {
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {
    DiagMatrixMultiplierCore<T, L, M, 0, M - 1>::compute(A, B, result);
  }
};

template <typename T, std::size_t L, std::size_t M>
static inline void
COMPILED_MATRIX_MULTIPLY_DIAG_MATRIX(const Matrix<T, L, M> &A,
                                     const DiagMatrix<T, M> &B,
                                     Matrix<T, L, M> &result) {
  DiagMatrixMultiplierColumn<T, L, M, L - 1>::compute(A, B, result);
}

template <typename T, std::size_t L, std::size_t M>
inline Matrix<T, L, M> operator*(const Matrix<T, L, M> &A,
                                 const DiagMatrix<T, M> &B) {
  Matrix<T, L, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < L; ++j) {
    for (std::size_t k = 0; k < M; ++k) {
      result(j, k) = A(j, k) * B[k];
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_MATRIX_MULTIPLY_DIAG_MATRIX<T, L, M>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Trace */
// Base case: when index reaches 0
template <typename T, std::size_t M, std::size_t Index>
struct DiagMatrixTraceCalculator {
  static T compute(const DiagMatrix<T, M> &A) {
    return A[Index] + DiagMatrixTraceCalculator<T, M, Index - 1>::compute(A);
  }
};

// Specialization for the base case when Index is 0
template <typename T, std::size_t M> struct DiagMatrixTraceCalculator<T, M, 0> {
  static T compute(const DiagMatrix<T, M> &A) { return A[0]; }
};

template <typename T, std::size_t M>
static inline T COMPILED_DIAG_TRACE_CALCULATOR(const DiagMatrix<T, M> &A) {
  return DiagMatrixTraceCalculator<T, M, M - 1>::compute(A);
}

template <typename T, std::size_t M>
inline T output_trace(const DiagMatrix<T, M> &A) {
  T trace = static_cast<T>(0);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; i++) {
    trace += A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  trace = Base::Matrix::COMPILED_DIAG_TRACE_CALCULATOR<T, M>(A);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return trace;
}

/* Create dense */
// Diagonal element assignment core
template <typename T, std::size_t M, std::size_t I>
struct DiagMatrixToDenseCore {
  static void assign(Matrix<T, M, M> &result, const DiagMatrix<T, M> &A) {
    result(I, I) = A[I];
    DiagMatrixToDenseCore<T, M, I - 1>::assign(result, A);
  }
};

// Base case for recursion termination
template <typename T, std::size_t M> struct DiagMatrixToDenseCore<T, M, 0> {
  static void assign(Matrix<T, M, M> &result, const DiagMatrix<T, M> &A) {
    result(0, 0) = A[0];
  }
};

template <typename T, std::size_t M>
static inline void COMPILED_DIAG_MATRIX_TO_DENSE(const DiagMatrix<T, M> &A,
                                                 Matrix<T, M, M> &result) {
  DiagMatrixToDenseCore<T, M, M - 1>::assign(result, A);
}

template <typename T, std::size_t M>
inline Matrix<T, M, M> output_dense_matrix(const DiagMatrix<T, M> &A) {
  Matrix<T, M, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; i++) {
    result(i, i) = A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DIAG_MATRIX_TO_DENSE<T, M>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Diag Matrix divide Diag Matrix */
// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx>
struct DiagMatrixDividerCore {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result, const T division_min) {
    result[M_idx] =
        A[M_idx] / Base::Utility::avoid_zero_divide(B[M_idx], division_min);
    DiagMatrixDividerCore<T, M, M_idx - 1>::compute(A, B, result, division_min);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct DiagMatrixDividerCore<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result, const T division_min) {
    result[0] = A[0] / Base::Utility::avoid_zero_divide(B[0], division_min);
  }
};

template <typename T, std::size_t M>
static inline void COMPILED_DIAG_MATRIX_DIVIDER(const DiagMatrix<T, M> &A,
                                                const DiagMatrix<T, M> &B,
                                                DiagMatrix<T, M> &result,
                                                const T division_min) {
  DiagMatrixDividerCore<T, M, M - 1>::compute(A, B, result, division_min);
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> diag_divide_diag(const DiagMatrix<T, M> &A,
                                         const DiagMatrix<T, M> &B,
                                         const T division_min) {
  DiagMatrix<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] / Base::Utility::avoid_zero_divide(B[j], division_min);
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DIAG_MATRIX_DIVIDER<T, M>(A, B, result, division_min);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Diag Matrix Inverse multiply Matrix */
// core multiplication for each element
template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t K>
struct DiagInvMultiplyDenseColumn {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    result(J, K) =
        B(J, K) / Base::Utility::avoid_zero_divide(A[J], division_min);
    DiagInvMultiplyDenseColumn<T, M, N, J, K - 1>::compute(A, B, result,
                                                           division_min);
  }
};

// if K == 0
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct DiagInvMultiplyDenseColumn<T, M, N, J, 0> {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    result(J, 0) =
        B(J, 0) / Base::Utility::avoid_zero_divide(A[J], division_min);
  }
};

// Column-wise multiplication
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct DiagInvMultiplyDenseRow {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    DiagInvMultiplyDenseColumn<T, M, N, J, N - 1>::compute(A, B, result,
                                                           division_min);
    DiagInvMultiplyDenseRow<T, M, N, J - 1>::compute(A, B, result,
                                                     division_min);
  }
};

// if J == 0
template <typename T, std::size_t M, std::size_t N>
struct DiagInvMultiplyDenseRow<T, M, N, 0> {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    DiagInvMultiplyDenseColumn<T, M, N, 0, N - 1>::compute(A, B, result,
                                                           division_min);
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline void COMPILED_DIAG_INV_MULTIPLY_DENSE(const DiagMatrix<T, M> &A,
                                                    const Matrix<T, M, N> &B,
                                                    Matrix<T, M, N> &result,
                                                    T division_min) {
  DiagInvMultiplyDenseRow<T, M, N, M - 1>::compute(A, B, result, division_min);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> diag_inv_multiply_dense(const DiagMatrix<T, M> &A,
                                               const Matrix<T, M, N> &B,
                                               const T division_min) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) =
          B(j, k) / Base::Utility::avoid_zero_divide(A[j], division_min);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  Base::Matrix::COMPILED_DIAG_INV_MULTIPLY_DENSE<T, M, N>(A, B, result,
                                                          division_min);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix real from complex */
template <typename T, std::size_t M>
inline DiagMatrix<Complex<T>, M>
convert_matrix_real_to_complex(const DiagMatrix<T, M> &From_matrix) {

  DiagMatrix<Complex<T>, M> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    To_matrix[i].real = From_matrix[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    To_matrix[i].real = From_matrix[i];
  }

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_DIAGONAL_HPP__
