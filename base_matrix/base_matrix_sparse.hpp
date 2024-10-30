#ifndef BASE_MATRIX_SPARSE_HPP
#define BASE_MATRIX_SPARSE_HPP

#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

const double SPARSE_MATRIX_JUDGE_ZERO_LIMIT_VALUE = 1.0e-20;

template <typename T, std::size_t M, std::size_t N, std::size_t V>
class SparseMatrix {
public:
  SparseMatrix()
      : values(V, static_cast<T>(0)),
        row_indices(V, static_cast<std::size_t>(0)),
        row_pointers(M + 1, static_cast<std::size_t>(0)) {}

  SparseMatrix(const std::initializer_list<T> &values,
               const std::initializer_list<std::size_t> &row_indices,
               const std::initializer_list<std::size_t> &row_pointers)
      : values(values), row_indices(row_indices), row_pointers(row_pointers) {}

  SparseMatrix(const std::vector<T> &values,
               const std::vector<std::size_t> &row_indices,
               const std::vector<std::size_t> &row_pointers)
      : values(values), row_indices(row_indices), row_pointers(row_pointers) {}

  SparseMatrix(const Matrix<T, M, N> &input)
      : values(V, static_cast<T>(0)),
        row_indices(V, static_cast<std::size_t>(0)),
        row_pointers(M + 1, static_cast<std::size_t>(0)) {

    std::size_t row_index_index = 0;
    std::size_t row_pointer_count = 0;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        if (std::abs(input(i, j)) >
            static_cast<T>(SPARSE_MATRIX_JUDGE_ZERO_LIMIT_VALUE)) {
          this->values[row_index_index] = input(i, j);
          this->row_indices[row_index_index] = j;

          row_pointer_count++;
          row_index_index++;

          if (row_index_index >= V) {
            this->row_pointers[i + 1] = row_pointer_count;
            return;
          }
        }
      }

      this->row_pointers[i + 1] = row_pointer_count;
    }
  }

  /* Copy Constructor */
  SparseMatrix(const SparseMatrix<T, M, N, V> &other)
      : values(other.values), row_indices(other.row_indices),
        row_pointers(other.row_pointers) {}

  SparseMatrix<T, M, N, V> &operator=(const SparseMatrix<T, M, N, V> &other) {
    if (this != &other) {
      this->values = other.values;
      this->row_indices = other.row_indices;
      this->row_pointers = other.row_pointers;
    }
    return *this;
  }

  /* Move Constructor */
  SparseMatrix(SparseMatrix<T, M, N, V> &&other) noexcept
      : values(std::move(other.values)),
        row_indices(std::move(other.row_indices)),
        row_pointers(std::move(other.row_pointers)) {}

  SparseMatrix<T, M, N, V> &
  operator=(SparseMatrix<T, M, N, V> &&other) noexcept {
    if (this != &other) {
      this->values = std::move(other.values);
      this->row_indices = std::move(other.row_indices);
      this->row_indices = std::move(other.row_indices);
    }
    return *this;
  }

  /* Function */
  Matrix<T, M, N> create_dense() const {
    Matrix<T, M, N> result;

    for (std::size_t j = 0; j < M; j++) {
      for (std::size_t k = this->row_pointers[j]; k < this->row_pointers[j + 1];
           k++) {
        result(j, this->row_indices[k]) = this->values[k];
      }
    }

    return result;
  }

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

  Matrix<T, N, M> transpose(void) const {
    Matrix<T, N, M> Y;

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = this->row_pointers[j]; k < this->row_pointers[j + 1];
           ++k) {
        Y(this->row_indices[k], j) = this->values[k];
      }
    }

    return Y;
  }

  Matrix<T, M, N> operator+(const Matrix<T, M, N> &B) const {
    Matrix<T, M, N> Y = B;

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = this->row_pointers[j]; k < this->row_pointers[j + 1];
           ++k) {
        Y(j, this->row_indices[k]) += this->values[k];
      }
    }

    return Y;
  }

  Matrix<T, M, M> operator+(const DiagMatrix<T, M> &B) const {
    Matrix<T, M, M> Y = B.create_dense();

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = this->row_pointers[j]; k < this->row_pointers[j + 1];
           ++k) {
        Y(j, this->row_indices[k]) += this->values[k];
      }
    }

    return Y;
  }

  Matrix<T, M, N> operator+(const SparseMatrix<T, M, N, V> &mat) const {
    Matrix<T, M, N> Y = this->create_dense();

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = mat.row_pointer(j); k < mat.row_pointer(j + 1);
           ++k) {
        Y(j, mat.row_index(k)) += mat.value(k);
      }
    }

    return Y;
  }

  Matrix<T, M, N> operator-(const Matrix<T, M, N> &B) const {
    Matrix<T, M, N> Y = -B;

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = this->row_pointers[j]; k < this->row_pointers[j + 1];
           ++k) {
        Y(j, this->row_indices[k]) += this->values[k];
      }
    }

    return Y;
  }

  Matrix<T, M, M> operator-(const DiagMatrix<T, M> &B) const {
    Matrix<T, M, M> Y = -(B.create_dense());

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = this->row_pointers[j]; k < this->row_pointers[j + 1];
           ++k) {
        Y(j, this->row_indices[k]) += this->values[k];
      }
    }

    return Y;
  }

  Matrix<T, M, N> operator-(const SparseMatrix<T, M, N, V> &mat) const {
    Matrix<T, M, N> Y = this->create_dense();

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = mat.row_pointer(j); k < mat.row_pointer(j + 1);
           ++k) {
        Y(j, mat.row_index(k)) -= mat.value(k);
      }
    }

    return Y;
  }

  SparseMatrix<T, M, N, V> operator*(const T &scalar) const {
    SparseMatrix<T, M, N, V> Y = *this;

    for (std::size_t i = 0; i < V; i++) {
      Y.values[i] = scalar * this->values[i];
    }

    return Y;
  }

  std::size_t rows() const { return N; }

  std::size_t cols() const { return M; }

  /* Variable */
  std::vector<T> values;
  std::vector<std::size_t> row_indices;
  std::vector<std::size_t> row_pointers;
};

template <typename T, std::size_t M, std::size_t N>
SparseMatrix<T, M, N, (M * N)> create_sparse(const Matrix<T, M, N> &A) {
  std::size_t consecutive_index = 0;
  std::vector<T> values(M * N);
  std::vector<std::size_t> row_indices(M * N);
  std::vector<std::size_t> row_pointers(M + 1);

  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < N; j++) {
      values[consecutive_index] = A(i, j);
      row_indices[consecutive_index] = j;

      consecutive_index++;
    }
    row_pointers[i + 1] = consecutive_index;
  }

  return SparseMatrix<T, M, N, (M * N)>(values, row_indices, row_pointers);
}

/* Create */
template <typename T, std::size_t M>
SparseMatrix<T, M, M, M> create_sparse(const DiagMatrix<T, M> &A) {
  std::size_t consecutive_index = 0;
  std::vector<T> values(M);
  std::vector<std::size_t> row_indices(M);
  std::vector<std::size_t> row_pointers(M + 1);

  for (std::size_t i = 0; i < M; i++) {
    values[consecutive_index] = A[i];
    row_indices[consecutive_index] = i;

    consecutive_index++;
    row_pointers[i + 1] = consecutive_index;
  }

  return SparseMatrix<T, M, M, M>(values, row_indices, row_pointers);
}

/* Operator */
template <typename T, std::size_t M, std::size_t V>
Matrix<T, M, M> operator+(const DiagMatrix<T, M> &B,
                          const SparseMatrix<T, M, M, V> &A) {
  Matrix<T, M, M> Y = B.create_dense();

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = A.row_pointer(j); k < A.row_pointer(j + 1); ++k) {
      Y(j, A.row_index(k)) += A.value(k);
    }
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t V>
Matrix<T, M, M> operator-(const DiagMatrix<T, M> &B,
                          const SparseMatrix<T, M, M, V> &A) {
  Matrix<T, M, M> Y = B.create_dense();

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = A.row_pointer(j); k < A.row_pointer(j + 1); ++k) {
      Y(j, A.row_index(k)) -= A.value(k);
    }
  }

  return Y;
}

/* Scalar * SparseMatrix */
template <typename T, std::size_t M, std::size_t N, std::size_t V>
SparseMatrix<T, M, N, V> operator*(const T &scalar,
                                   const SparseMatrix<T, M, N, V> &A) {
  SparseMatrix<T, M, N, V> Y = A;

  for (std::size_t i = 0; i < V; i++) {
    Y.values[i] = scalar * A.value(i);
  }

  return Y;
}

/* SparseMatrix * Vector */
template <typename T, std::size_t M, std::size_t N, std::size_t V>
Vector<T, M> operator*(const SparseMatrix<T, M, N, V> &A,
                       const Vector<T, N> &b) {
  Vector<T, M> y;

  for (std::size_t j = 0; j < M; j++) {
    T sum = static_cast<T>(0);
    for (std::size_t k = A.row_pointer(j); k < A.row_pointer(j + 1); k++) {
      sum += A.value(k) * b[A.row_index(k)];
    }
    y[j] = sum;
  }

  return y;
}

template <typename T, std::size_t N, std::size_t K, std::size_t V>
ColVector<T, K> colV_mul_SB(const ColVector<T, N> &a,
                            const SparseMatrix<T, N, K, V> &B) {
  ColVector<T, K> y;

  for (std::size_t j = 0; j < N; j++) {
    for (std::size_t k = B.row_pointer(j); k < B.row_pointer(j + 1); k++) {
      y[B.row_index(k)] += B.value(k) * a[j];
    }
  }

  return y;
}

/* SparseMatrix * Matrix */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V>
Matrix<T, M, K> operator*(const SparseMatrix<T, M, N, V> &A,
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

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V>
Matrix<T, M, K> operator*(const Matrix<T, M, N> &A,
                          const SparseMatrix<T, N, K, V> &B) {
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

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V>
Matrix<T, M, K>
matrix_multiply_SparseA_mul_BT(const SparseMatrix<T, M, N, V> &A,
                               const Matrix<T, K, N> &B) {
  Matrix<T, M, K> Y;

  for (std::size_t i = 0; i < K; i++) {
    for (std::size_t j = 0; j < M; j++) {
      T sum = static_cast<T>(0);
      for (std::size_t k = A.row_pointer(j); k < A.row_pointer(j + 1); k++) {
        sum += A.value(k) * B(i, A.row_index(k));
      }
      Y(j, i) = sum;
    }
  }

  return Y;
}

/* SparseMatrix * SparseMatrix */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V, std::size_t W>
Matrix<T, M, K> operator*(const SparseMatrix<T, M, N, V> &A,
                          const SparseMatrix<T, N, K, W> &B) {
  Matrix<T, M, K> Y;

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = A.row_pointer(j); k < A.row_pointer(j + 1); ++k) {
      for (std::size_t l = B.row_pointer(A.row_index(k));
           l < B.row_pointer(A.row_index(k) + 1); ++l) {
        Y(j, B.row_index(l)) += A.value(k) * B.value(l);
      }
    }
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V, std::size_t W>
Matrix<T, M, K>
matrix_multiply_SparseA_mul_SparseBT(const SparseMatrix<T, M, N, V> &A,
                                     const SparseMatrix<T, K, N, W> &B) {
  Matrix<T, M, K> Y;

  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < K; j++) {
      for (std::size_t l = A.row_pointer(i); l < A.row_pointer(i + 1); l++) {
        for (std::size_t o = B.row_pointer(j); o < B.row_pointer(j + 1); o++) {
          if (A.row_index(l) == B.row_index(o)) {
            Y(i, j) += A.value(l) * B.value(o);
          }
        }
      }
    }
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V, std::size_t W>
Matrix<T, M, K>
matrix_multiply_SparseAT_mul_SparseB(const SparseMatrix<T, N, M, V> &A,
                                     const SparseMatrix<T, N, K, W> &B) {
  Matrix<T, M, K> Y;

  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t k = A.row_pointer(i); k < A.row_pointer(i + 1); k++) {
      for (std::size_t j = B.row_pointer(i); j < B.row_pointer(i + 1); j++) {
        Y(A.row_index(k), B.row_index(j)) += A.value(k) * B.value(j);
      }
    }
  }

  return Y;
}

/* DiagMatrix */
template <typename T, std::size_t M, std::size_t N, std::size_t V>
Matrix<T, M, N> operator*(const SparseMatrix<T, M, N, V> &A,
                          const DiagMatrix<T, N> &B) {
  Matrix<T, M, N> Y;

  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < M; j++) {
      T sum = static_cast<T>(0);
      for (std::size_t k = A.row_pointer(j); k < A.row_pointer(j + 1); k++) {
        if (A.row_index(k) == i) {
          sum += A.value(k) * B[i];
        }
        Y(j, i) = sum;
      }
    }
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t K, std::size_t V>
Matrix<T, M, K> operator*(const DiagMatrix<T, M> &A,
                          const SparseMatrix<T, M, K, V> &B) {
  Matrix<T, M, K> Y;

  for (std::size_t j = 0; j < M; j++) {
    for (std::size_t k = B.row_pointer(j); k < B.row_pointer(j + 1); k++) {
      for (std::size_t i = 0; i < M; i++) {
        if (i == j) {
          Y(i, B.row_index(k)) += B.value(k) * A[i];
        }
      }
    }
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t K, std::size_t V>
Matrix<T, K, M>
matrix_multiply_T_DiagA_mul_SparseB(const DiagMatrix<T, M> &A,
                                    const SparseMatrix<T, M, K, V> &B) {
  Matrix<T, K, M> Y;

  for (std::size_t j = 0; j < M; j++) {
    for (std::size_t k = B.row_pointer(j); k < B.row_pointer(j + 1); k++) {
      for (std::size_t i = 0; i < M; i++) {
        if (i == j) {
          Y(B.row_index(k), i) += B.value(k) * A[i];
        }
      }
    }
  }

  return Y;
}

/* Plus, Minus */
template <typename T, std::size_t M, std::size_t N, std::size_t V>
Matrix<T, M, N> operator+(const Matrix<T, M, N> &B,
                          const SparseMatrix<T, M, N, V> SA) {
  Matrix<T, M, N> Y = B;

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = SA.row_pointer(j); k < SA.row_pointer(j + 1); ++k) {
      Y(j, SA.row_index(k)) += SA.value(k);
    }
  }

  return Y;
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
Matrix<T, M, N> operator-(const Matrix<T, M, N> &B,
                          const SparseMatrix<T, M, N, V> SA) {
  Matrix<T, M, N> Y = B;

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = SA.row_pointer(j); k < SA.row_pointer(j + 1); ++k) {
      Y(j, SA.row_index(k)) -= SA.value(k);
    }
  }

  return Y;
}

/* Transpose */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V>
Matrix<T, N, K>
matrix_multiply_A_mul_SparseBT(const Matrix<T, M, N> &A,
                               const SparseMatrix<T, K, N, V> &SB) {
  Matrix<T, N, K> Y;

  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < K; ++j) {
      T sum = static_cast<T>(0);
      for (std::size_t k = SB.row_pointer(j); k < SB.row_pointer(j + 1); ++k) {
        sum += SB.value(k) * A(i, SB.row_index(k));
      }
      Y(i, j) = sum;
    }
  }

  return Y;
}

template <typename T, std::size_t N, std::size_t M, std::size_t K,
          std::size_t V>
Matrix<T, M, K>
matrix_multiply_AT_mul_SparseB(const Matrix<T, N, M> &A,
                               const SparseMatrix<T, N, K, V> &B) {
  Matrix<T, M, K> Y;

  for (std::size_t j = 0; j < N; j++) {
    for (std::size_t k = B.row_pointer(j); k < B.row_pointer(j + 1); k++) {
      for (std::size_t i = 0; i < M; i++) {
        Y(i, B.row_index(k)) += B.value(k) * A(j, i);
      }
    }
  }

  return Y;
}

template <typename T, std::size_t N, std::size_t M, std::size_t K,
          std::size_t V>
Matrix<T, M, K>
matrix_multiply_SparseAT_mul_B(const SparseMatrix<T, N, M, V> &A,
                               const Matrix<T, N, K> &B) {
  Matrix<T, M, K> Y;

  for (std::size_t j = 0; j < N; j++) {
    for (std::size_t k = A.row_pointer(j); k < A.row_pointer(j + 1); k++) {
      for (std::size_t i = 0; i < M; i++) {
        Y(A.row_index(k), i) += A.value(k) * B(j, i);
      }
    }
  }

  return Y;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_SPARSE_HPP
