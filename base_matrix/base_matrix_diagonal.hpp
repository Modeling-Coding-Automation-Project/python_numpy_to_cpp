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

  static inline DiagMatrix<T, M> full(const T &value) {
    DiagMatrix<T, M> full(std::vector<T>(M, value));

    return full;
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
namespace DiagMatrixAddDiagMatrix {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[M_idx] = A[M_idx] + B[M_idx];
    Core<T, M, M_idx - 1>::compute(A, B, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[0] = A[0] + B[0];
  }
};

template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                    DiagMatrix<T, M> &result) {
  Core<T, M, M - 1>::compute(A, B, result);
}

} // namespace DiagMatrixAddDiagMatrix

template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator+(const DiagMatrix<T, M> &A,
                                  const DiagMatrix<T, M> &B) {
  DiagMatrix<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] + B[j];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixAddDiagMatrix::compute<T, M>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

namespace DiagMatrixAddMatrix {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {

    result.template set<M_idx, M_idx>(result.template get<M_idx, M_idx>() +
                                      A[M_idx]);
    Core<T, M, M_idx - 1>::compute(A, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {

    result.template set<0, 0>(result.template get<0, 0>() + A[0]);
  }
};

template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
  Core<T, M, M - 1>::compute(A, result);
}

} // namespace DiagMatrixAddMatrix

template <typename T, std::size_t M>
inline Matrix<T, M, M> operator+(const DiagMatrix<T, M> &A,
                                 const Matrix<T, M, M> &B) {
  Matrix<T, M, M> result = B;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixAddMatrix::compute<T, M>(A, result);

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

  DiagMatrixAddMatrix::compute<T, M>(B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Minus */
namespace DiagMatrixMinus {

// when I_idx < M
template <typename T, std::size_t M, std::size_t I_idx> struct Loop {
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> &result) {
    result[I_idx] = -A[I_idx];
    Loop<T, M, I_idx - 1>::compute(A, result);
  }
};

// row recursion termination
template <typename T, std::size_t M> struct Loop<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> &result) {
    result[0] = -A[0];
  }
};

template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> &result) {
  Loop<T, M, M - 1>::compute(A, result);
}

} // namespace DiagMatrixMinus

template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator-(const DiagMatrix<T, M> &A) {
  DiagMatrix<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = -A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixMinus::compute<T, M>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix Subtraction */
namespace DiagMatrixSubDiagMatrix {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[M_idx] = A[M_idx] - B[M_idx];
    Core<T, M, M_idx - 1>::compute(A, B, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[0] = A[0] - B[0];
  }
};

template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                    DiagMatrix<T, M> &result) {
  Core<T, M, M - 1>::compute(A, B, result);
}

} // namespace DiagMatrixSubDiagMatrix

template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator-(const DiagMatrix<T, M> &A,
                                  const DiagMatrix<T, M> &B) {
  DiagMatrix<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] - B[j];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixSubDiagMatrix::compute<T, M>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

namespace DiagMatrixSubMatrix {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {

    result.template set<M_idx, M_idx>(result.template get<M_idx, M_idx>() +
                                      A[M_idx]);
    Core<T, M, M_idx - 1>::compute(A, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {

    result.template set<0, 0>(result.template get<0, 0>() + A[0]);
  }
};

template <typename T, std::size_t M>
inline void compute(const Matrix<T, M, M> &A, Matrix<T, M, M> &result) {
  Core<T, M, M - 1>::compute(A, result);
}

} // namespace DiagMatrixSubMatrix

template <typename T, std::size_t M>
inline Matrix<T, M, M> operator-(const DiagMatrix<T, M> &A,
                                 const Matrix<T, M, M> &B) {
  Matrix<T, M, M> result = -B;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixAddMatrix::compute<T, M>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

namespace MatrixSubDiagMatrix {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {

    result.template set<M_idx, M_idx>(result.template get<M_idx, M_idx>() -
                                      A[M_idx]);
    Core<T, M, M_idx - 1>::compute(A, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
    result.template set<0, 0>(result.template get<0, 0>() - A[0]);
  }
};

template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
  Core<T, M, M - 1>::compute(A, result);
}

} // namespace MatrixSubDiagMatrix

template <typename T, std::size_t M>
inline Matrix<T, M, M> operator-(const Matrix<T, M, M> &A,
                                 const DiagMatrix<T, M> &B) {
  Matrix<T, M, M> result = A;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) -= B[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixSubDiagMatrix::compute<T, M>(B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Diag Matrix multiply Scalar */
namespace DiagMatrixMultiplyScalar {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  static void compute(const DiagMatrix<T, M> &mat, const T scalar,
                      DiagMatrix<T, M> &result) {
    result[M_idx] = mat[M_idx] * scalar;
    Core<T, M, M_idx - 1>::compute(mat, scalar, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &mat, const T scalar,
                      DiagMatrix<T, M> &result) {
    result[0] = mat[0] * scalar;
  }
};

template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &mat, const T &scalar,
                    DiagMatrix<T, M> &result) {
  Core<T, M, M - 1>::compute(mat, scalar, result);
}

} // namespace DiagMatrixMultiplyScalar

template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator*(const DiagMatrix<T, M> &A, const T &scalar) {
  DiagMatrix<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * scalar;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixMultiplyScalar::compute<T, M>(A, scalar, result);

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

  DiagMatrixMultiplyScalar::compute<T, M>(A, scalar, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Diag Matrix multiply Vector */
namespace DiagMatrixMultiplyVector {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  static void compute(const DiagMatrix<T, M> &A, Vector<T, M> vec,
                      Vector<T, M> &result) {
    result[M_idx] = A[M_idx] * vec[M_idx];
    Core<T, M, M_idx - 1>::compute(A, vec, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, Vector<T, M> vec,
                      Vector<T, M> &result) {
    result[0] = A[0] * vec[0];
  }
};

template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, const Vector<T, M> &vec,
                    Vector<T, M> &result) {
  Core<T, M, M - 1>::compute(A, vec, result);
}

} // namespace DiagMatrixMultiplyVector

template <typename T, std::size_t M>
inline Vector<T, M> operator*(const DiagMatrix<T, M> &A,
                              const Vector<T, M> &vec) {
  Vector<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * vec[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixMultiplyVector::compute<T, M>(A, vec, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix Multiplication */
namespace DiagMatrixMultiplyDiagMatrix {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[M_idx] = A[M_idx] * B[M_idx];
    Core<T, M, M_idx - 1>::compute(A, B, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[0] = A[0] * B[0];
  }
};

template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                    DiagMatrix<T, M> &result) {
  Core<T, M, M - 1>::compute(A, B, result);
}

} // namespace DiagMatrixMultiplyDiagMatrix

template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator*(const DiagMatrix<T, M> &A,
                                  const DiagMatrix<T, M> &B) {
  DiagMatrix<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] * B[j];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixMultiplyDiagMatrix::compute<T, M>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Diag Matrix Multiply Matrix */
namespace DiagMatrixMultiplyMatrix {

// Core multiplication for DiagMatrix and Matrix
template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t K>
struct Core {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<J, K>(A[J] * B.template get<J, K>());
    Core<T, M, N, J, K - 1>::compute(A, B, result);
  }
};

// Specialization for K == 0
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct Core<T, M, N, J, 0> {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<J, 0>(A[J] * B.template get<J, 0>());
  }
};

// Column-wise multiplication
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct Column {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Core<T, M, N, J, N - 1>::compute(A, B, result);
    Column<T, M, N, J - 1>::compute(A, B, result);
  }
};

// Specialization for J == 0
template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, 0> {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Core<T, M, N, 0, N - 1>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                    Matrix<T, M, N> &result) {
  Column<T, M, N, M - 1>::compute(A, B, result);
}

} // namespace DiagMatrixMultiplyMatrix

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

  DiagMatrixMultiplyMatrix::compute<T, M, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

namespace MatrixMultiplyDiagMatrix {

// Core multiplication for each element
template <typename T, std::size_t L, std::size_t M, std::size_t I,
          std::size_t J>
struct Core {
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {

    result.template set<I, J>(A.template get<I, J>() * B[J]);
    Core<T, L, M, I, J - 1>::compute(A, B, result);
  }
};

// Specialization for J = 0
template <typename T, std::size_t L, std::size_t M, std::size_t I>
struct Core<T, L, M, I, 0> {
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {

    result.template set<I, 0>(A.template get<I, 0>() * B[0]);
  }
};

// Column-wise multiplication
template <typename T, std::size_t L, std::size_t M, std::size_t I>
struct Column {
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {
    Core<T, L, M, I, M - 1>::compute(A, B, result);
    Column<T, L, M, I - 1>::compute(A, B, result);
  }
};

// Specialization for I = 0
template <typename T, std::size_t L, std::size_t M> struct Column<T, L, M, 0> {
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {
    Core<T, L, M, 0, M - 1>::compute(A, B, result);
  }
};

template <typename T, std::size_t L, std::size_t M>
inline void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                    Matrix<T, L, M> &result) {
  Column<T, L, M, L - 1>::compute(A, B, result);
}

} // namespace MatrixMultiplyDiagMatrix

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

  MatrixMultiplyDiagMatrix::compute<T, L, M>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Trace */
namespace DiagMatrixTrace {

// Base case: when index reaches 0
template <typename T, std::size_t M, std::size_t Index> struct Core {
  static T compute(const DiagMatrix<T, M> &A) {
    return A[Index] + Core<T, M, Index - 1>::compute(A);
  }
};

// Specialization for the base case when Index is 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  static T compute(const DiagMatrix<T, M> &A) { return A[0]; }
};

template <typename T, std::size_t M>
inline T compute(const DiagMatrix<T, M> &A) {
  return Core<T, M, M - 1>::compute(A);
}

} // namespace DiagMatrixTrace

template <typename T, std::size_t M>
inline T output_trace(const DiagMatrix<T, M> &A) {
  T trace = static_cast<T>(0);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; i++) {
    trace += A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  trace = DiagMatrixTrace::compute<T, M>(A);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return trace;
}

/* Create dense */
namespace DiagMatrixToDense {

// Diagonal element assignment core
template <typename T, std::size_t M, std::size_t I> struct Core {
  static void assign(Matrix<T, M, M> &result, const DiagMatrix<T, M> &A) {

    result.template set<I, I>(A[I]);
    Core<T, M, I - 1>::assign(result, A);
  }
};

// Base case for recursion termination
template <typename T, std::size_t M> struct Core<T, M, 0> {
  static void assign(Matrix<T, M, M> &result, const DiagMatrix<T, M> &A) {
    result.template set<0, 0>(A[0]);
  }
};

template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
  Core<T, M, M - 1>::assign(result, A);
}

} // namespace DiagMatrixToDense

template <typename T, std::size_t M>
inline Matrix<T, M, M> output_dense_matrix(const DiagMatrix<T, M> &A) {
  Matrix<T, M, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; i++) {
    result(i, i) = A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixToDense::compute<T, M>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Diag Matrix divide Diag Matrix */
namespace DiagMatrixDivideDiagMatrix {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result, const T division_min) {
    result[M_idx] =
        A[M_idx] / Base::Utility::avoid_zero_divide(B[M_idx], division_min);
    Core<T, M, M_idx - 1>::compute(A, B, result, division_min);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result, const T division_min) {
    result[0] = A[0] / Base::Utility::avoid_zero_divide(B[0], division_min);
  }
};

template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                    DiagMatrix<T, M> &result, const T division_min) {
  Core<T, M, M - 1>::compute(A, B, result, division_min);
}

} // namespace DiagMatrixDivideDiagMatrix

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

  DiagMatrixDivideDiagMatrix::compute<T, M>(A, B, result, division_min);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Diag Matrix Inverse multiply Matrix */
namespace DiagMatrixInverseMultiplyMatrix {

// core multiplication for each element
template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t K>
struct Column {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    result(J, K) =
        B(J, K) / Base::Utility::avoid_zero_divide(A[J], division_min);
    Column<T, M, N, J, K - 1>::compute(A, B, result, division_min);
  }
};

// if K == 0
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct Column<T, M, N, J, 0> {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    result(J, 0) =
        B(J, 0) / Base::Utility::avoid_zero_divide(A[J], division_min);
  }
};

// Column-wise multiplication
template <typename T, std::size_t M, std::size_t N, std::size_t J> struct Row {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    Column<T, M, N, J, N - 1>::compute(A, B, result, division_min);
    Row<T, M, N, J - 1>::compute(A, B, result, division_min);
  }
};

// if J == 0
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    Column<T, M, N, 0, N - 1>::compute(A, B, result, division_min);
  }
};

template <typename T, std::size_t M, std::size_t N>
inline void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                    Matrix<T, M, N> &result, T division_min) {
  Row<T, M, N, M - 1>::compute(A, B, result, division_min);
}

} // namespace DiagMatrixInverseMultiplyMatrix

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

  DiagMatrixInverseMultiplyMatrix::compute<T, M, N>(A, B, result, division_min);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> diag_inv_multiply_dense_partition(
    const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B, const T &division_min,
    const std::size_t &matrix_size) {
  Matrix<T, M, N> result;

  for (std::size_t j = 0; j < matrix_size; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result.access(j, k) =
          B.access(j, k) / Base::Utility::avoid_zero_divide(A[j], division_min);
    }
  }

  return result;
}

/* Convert Real Matrix to Complex */
namespace DiagMatrixRealToComplex {

/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t I>
struct DiagMatrixRealToComplexLoop {
  static void compute(const DiagMatrix<T, M> &From_matrix,
                      DiagMatrix<Complex<T>, M> &To_matrix) {
    To_matrix[I].real = From_matrix[I];
    DiagMatrixRealToComplexLoop<T, M, I - 1>::compute(From_matrix, To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M>
struct DiagMatrixRealToComplexLoop<T, M, 0> {
  static void compute(const DiagMatrix<T, M> &From_matrix,
                      DiagMatrix<Complex<T>, M> &To_matrix) {
    To_matrix[0].real = From_matrix[0];
  }
};

template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &From_matrix,
                    DiagMatrix<Complex<T>, M> &To_matrix) {
  DiagMatrixRealToComplexLoop<T, M, M - 1>::compute(From_matrix, To_matrix);
}

} // namespace DiagMatrixRealToComplex

template <typename T, std::size_t M>
inline DiagMatrix<Complex<T>, M>
convert_matrix_real_to_complex(const DiagMatrix<T, M> &From_matrix) {

  DiagMatrix<Complex<T>, M> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    To_matrix[i].real = From_matrix[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixRealToComplex::compute<T, M>(From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Get Real Matrix from Complex */
namespace GetRealDiagMatrixFromComplex {

/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t I> struct Loop {
  static void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                      DiagMatrix<T, M> &To_matrix) {
    To_matrix[I] = From_matrix[I].real;
    Loop<T, M, I - 1>::compute(From_matrix, To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M> struct Loop<T, M, 0> {
  static void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                      DiagMatrix<T, M> &To_matrix) {
    To_matrix[0] = From_matrix[0].real;
  }
};

template <typename T, std::size_t M>
inline void compute(const DiagMatrix<Complex<T>, 3> &From_matrix,
                    DiagMatrix<T, 3> &To_matrix) {
  Loop<T, M, M - 1>::compute(From_matrix, To_matrix);
}

} // namespace GetRealDiagMatrixFromComplex

template <typename T, std::size_t M>
inline auto get_real_matrix_from_complex_matrix(
    const DiagMatrix<Complex<T>, M> &From_matrix) -> DiagMatrix<T, M> {

  DiagMatrix<T, M> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    To_matrix[i] = From_matrix[i].real;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  GetRealDiagMatrixFromComplex::compute<T, M>(From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Get Imag Matrix from Complex */
namespace GetImagDiagMatrixFromComplex {

/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t I> struct Loop {
  static void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                      DiagMatrix<T, M> &To_matrix) {
    To_matrix[I] = From_matrix[I].imag;
    Loop<T, M, I - 1>::compute(From_matrix, To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M> struct Loop<T, M, 0> {
  static void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                      DiagMatrix<T, M> &To_matrix) {
    To_matrix[0] = From_matrix[0].imag;
  }
};

template <typename T, std::size_t M>
inline void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                    DiagMatrix<T, M> &To_matrix) {
  Loop<T, M, M - 1>::compute(From_matrix, To_matrix);
}

} // namespace GetImagDiagMatrixFromComplex

template <typename T, std::size_t M>
inline auto get_imag_matrix_from_complex_matrix(
    const DiagMatrix<Complex<T>, M> &From_matrix) -> DiagMatrix<T, M> {

  DiagMatrix<T, M> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    To_matrix[i] = From_matrix[i].imag;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  GetImagDiagMatrixFromComplex::compute<T, M>(From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_DIAGONAL_HPP__
