#ifndef BASE_MATRIX_VARIABLE_SPARSE_HPP
#define BASE_MATRIX_VARIABLE_SPARSE_HPP

#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_vector.hpp"
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

template <typename T, std::size_t M, std::size_t N> class VariableSparseMatrix {
public:
#ifdef BASE_MATRIX_USE_STD_VECTOR
  VariableSparseMatrix()
      : values(M * N, static_cast<T>(0)),
        row_indices(M * N, static_cast<std::size_t>(0)),
        row_pointers(M + 1, static_cast<std::size_t>(0)) {}
#else
  VariableSparseMatrix() : values{}, row_indices{}, row_pointers{} {}
#endif

  /* Copy Constructor */
  VariableSparseMatrix(const VariableSparseMatrix<T, M, N> &matrix)
      : values(matrix.values), row_indices(matrix.row_indices),
        row_pointers(matrix.row_pointers) {}

  VariableSparseMatrix<T, M, N> &
  operator=(const VariableSparseMatrix<T, M, N> &matrix) {
    if (this != &matrix) {
      this->values = matrix.values;
      this->row_indices = matrix.row_indices;
      this->row_pointers = matrix.row_pointers;
    }
    return *this;
  }

  /* Move Constructor */
  VariableSparseMatrix(VariableSparseMatrix<T, M, N> &&matrix) noexcept
      : values(std::move(matrix.values)),
        row_indices(std::move(matrix.row_indices)),
        row_pointers(std::move(matrix.row_pointers)) {}

  VariableSparseMatrix<T, M, N> &
  operator=(VariableSparseMatrix<T, M, N> &&matrix) noexcept {
    if (this != &matrix) {
      this->values = std::move(matrix.values);
      this->row_indices = std::move(matrix.row_indices);
      this->row_pointers = std::move(matrix.row_pointers);
    }
    return *this;
  }

  /* Function */
  T value(std::size_t i) { return this->values[i]; }

  const T value(std::size_t i) const { return this->values[i]; }

  std::size_t row_index(std::size_t i) { return this->row_indices[i]; }

  const std::size_t row_index(std::size_t i) const {
    return this->row_indices[i];
  }

  std::size_t row_pointer(std::size_t i) { return this->row_pointers[i]; }

  const std::size_t row_pointer(std::size_t i) const {
    return this->row_pointers[i];
  }

/* Variable */
#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values;
  std::vector<std::size_t> row_indices;
  std::vector<std::size_t> row_pointers;
#else
  std::array<T, M * N> values;
  std::array<std::size_t, M * N> row_indices;
  std::array<std::size_t, M + 1> row_pointers;
#endif
};

/* SparseMatrix * Matrix */
template <typename T, std::size_t M, std::size_t N, std::size_t K>
Matrix<T, M, K> operator*(const VariableSparseMatrix<T, M, N> &A,
                          const Matrix<T, N, K> &B) {
  Matrix<T, M, K> Y;

  for (std::size_t i = 0; i < K; i++) {
    for (std::size_t j = 0; j < M; j++) {
      T sum = static_cast<T>(0);
      for (std::size_t k = A.row_pointer(j); k < A.row_pointer(j + 1); k++) {
        sum += A.value(k) * B(A.row_index(k), i);
      }
      Y(j, i) = sum;
    }
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t K>
Matrix<T, M, K> operator*(const Matrix<T, M, N> &A,
                          const VariableSparseMatrix<T, N, K> &B) {
  Matrix<T, M, K> Y;

  for (std::size_t j = 0; j < N; j++) {
    for (std::size_t k = B.row_pointer(j); k < B.row_pointer(j + 1); k++) {
      for (std::size_t i = 0; i < M; i++) {
        Y(i, B.row_index(k)) += B.value(k) * A(i, j);
      }
    }
  }

  return Y;
}

/* SparseMatrix * SparseMatrix */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V>
Matrix<T, M, K> operator*(const VariableSparseMatrix<T, M, N> &A,
                          const SparseMatrix<T, N, K, V> &B) {
  Matrix<T, M, K> Y;

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = A.row_pointer(j); k < A.row_pointer(j + 1); ++k) {
      std::size_t a_col = A.row_index(k);
      T a_val = A.value(k);

      for (std::size_t l = B.row_pointer(a_col); l < B.row_pointer(a_col + 1);
           ++l) {
        std::size_t b_col = B.row_index(l);
        T b_val = B.value(l);

        Y(j, b_col) += a_val * b_val;
      }
    }
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V>
Matrix<T, M, K> operator*(const SparseMatrix<T, M, N, V> &A,
                          const VariableSparseMatrix<T, N, K> &B) {
  Matrix<T, M, K> Y;

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = A.row_pointer(j); k < A.row_pointer(j + 1); ++k) {
      std::size_t a_col = A.row_index(k);
      T a_val = A.value(k);

      for (std::size_t l = B.row_pointer(a_col); l < B.row_pointer(a_col + 1);
           ++l) {
        std::size_t b_col = B.row_index(l);
        T b_val = B.value(l);

        Y(j, b_col) += a_val * b_val;
      }
    }
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t K>
Matrix<T, M, K> operator*(const VariableSparseMatrix<T, M, N> &A,
                          const VariableSparseMatrix<T, N, K> &B) {
  Matrix<T, M, K> Y;

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = A.row_pointer(j); k < A.row_pointer(j + 1); ++k) {
      std::size_t a_col = A.row_index(k);
      T a_val = A.value(k);

      for (std::size_t l = B.row_pointer(a_col); l < B.row_pointer(a_col + 1);
           ++l) {
        std::size_t b_col = B.row_index(l);
        T b_val = B.value(l);

        Y(j, b_col) += a_val * b_val;
      }
    }
  }

  return Y;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_VARIABLE_SPARSE_HPP
