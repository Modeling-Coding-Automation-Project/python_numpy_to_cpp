#ifndef BASE_MATRIX_DIAGONAL_HPP
#define BASE_MATRIX_DIAGONAL_HPP

#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

template <typename T, std::size_t M> class DiagMatrix {
public:
  DiagMatrix() : data(M, static_cast<T>(0)) {}

  DiagMatrix(const std::vector<T> &input) : data(input) {}

  DiagMatrix(const std::initializer_list<T> &input) : data(input) {}

  DiagMatrix(T input[M]) : data(M, static_cast<T>(0)) {

    for (std::size_t i = 0; i < M; i++) {
      this->data[i] = input[i];
    }
  }

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

  Vector<T, M> operator*(const Vector<T, M> &vec) const {
    Vector<T, M> result;
    for (std::size_t i = 0; i < M; ++i) {
      result[i] = this->data[i] * vec[i];
    }

    return result;
  }

  DiagMatrix<T, M> operator*(const T &scalar) const {
    DiagMatrix<T, M> result;
    for (std::size_t i = 0; i < M; ++i) {
      result[i] = this->data[i] * scalar;
    }

    return result;
  }

  DiagMatrix<T, M> operator+(const DiagMatrix<T, M> &B) const {
    DiagMatrix<T, M> result;
    for (std::size_t j = 0; j < M; ++j) {
      result[j] = this->data[j] + B[j];
    }

    return result;
  }

  DiagMatrix<T, M> operator-(const DiagMatrix<T, M> &B) const {
    DiagMatrix<T, M> result;
    for (std::size_t j = 0; j < M; ++j) {
      result[j] = this->data[j] - B[j];
    }

    return result;
  }

  DiagMatrix<T, M> operator*(const DiagMatrix<T, M> &B) const {
    DiagMatrix<T, M> result;
    for (std::size_t j = 0; j < M; ++j) {
      result[j] = this->data[j] * B[j];
    }

    return result;
  }

  Matrix<T, M, M> operator+(const Matrix<T, M, M> &mat) const {
    Matrix<T, M, M> result = mat;
    for (std::size_t i = 0; i < M; ++i) {
      result(i, i) += this->data[i];
    }

    return result;
  }

  Matrix<T, M, M> operator-(const Matrix<T, M, M> &mat) const {
    Matrix<T, M, M> result = -mat;
    for (std::size_t i = 0; i < M; ++i) {
      result(i, i) += this->data[i];
    }

    return result;
  }

  std::size_t rows() const { return M; }

  std::size_t cols() const { return M; }

  Vector<T, M> get_row(std::size_t row) const {
    if (row >= M) {
      row = M - 1;
    }

    Vector<T, M> result;
    result[row] = this->data[row];

    return result;
  }

  T get_trace() const {
    T trace = static_cast<T>(0);
    for (std::size_t i = 0; i < M; i++) {
      trace += this->data[i];
    }
    return trace;
  }

  Matrix<T, M, M> create_dense() const {
    Matrix<T, M, M> result;

    for (std::size_t i = 0; i < M; i++) {
      result(i, i) = this->data[i];
    }

    return result;
  }

  DiagMatrix<T, M> inv(T division_min) const {
    DiagMatrix<T, M> result;

    for (std::size_t i = 0; i < M; i++) {
      result[i] =
          static_cast<T>(1) / avoid_zero_divide(this->data[i], division_min);
    }

    return result;
  }

  /* Variable */
  std::vector<T> data;
};

/* Matrix Addition */
template <typename T, std::size_t M>
Matrix<T, M, M> operator+(const Matrix<T, M, M> &A, const DiagMatrix<T, M> &B) {
  Matrix<T, M, M> result = A;
  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += B[i];
  }

  return result;
}

/* Matrix Subtraction */
template <typename T, std::size_t M>
Matrix<T, M, M> operator-(const Matrix<T, M, M> &A, const DiagMatrix<T, M> &B) {
  Matrix<T, M, M> result = A;
  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) -= B[i];
  }

  return result;
}

/* Matrix Multiplication */
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
