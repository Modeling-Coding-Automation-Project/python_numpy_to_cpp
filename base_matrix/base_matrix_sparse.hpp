/**
 * @file base_matrix_sparse.hpp
 * @brief Sparse matrix implementation and related operations for fixed-size and
 * vector-based storage.
 *
 * This header provides a templated SparseMatrix class supporting both
 * std::array and std::vector storage for efficient representation of sparse
 * matrices. It includes constructors for various initialization methods,
 * conversion to dense matrices, and arithmetic operations with dense matrices,
 * diagonal matrices, vectors, and other sparse matrices. The file also provides
 * a set of free functions for creating sparse matrices from dense or diagonal
 * matrices, and for performing matrix-matrix and matrix-vector multiplications
 * involving sparse matrices.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __BASE_MATRIX_SPARSE_HPP__
#define __BASE_MATRIX_SPARSE_HPP__

#include "base_matrix_macros.hpp"

#include "base_math.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

const double SPARSE_MATRIX_JUDGE_ZERO_LIMIT_VALUE = 1.0e-20;

/**
 * @brief SparseMatrix class for fixed-size and vector-based storage.
 *
 * This class implements a sparse matrix using either std::array or std::vector
 * for storage. It supports various operations such as addition, subtraction,
 * multiplication with dense matrices, diagonal matrices, and other sparse
 * matrices, as well as conversion to dense format.
 *
 * @tparam T The type of elements in the matrix.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t V>
class SparseMatrix {
public:
#ifdef __BASE_MATRIX_USE_STD_VECTOR__

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
        if (Base::Math::abs(input(i, j)) >
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

#else // __BASE_MATRIX_USE_STD_VECTOR__

  SparseMatrix() : values{}, row_indices{}, row_pointers{} {}

  SparseMatrix(const std::initializer_list<T> &values,
               const std::initializer_list<std::size_t> &row_indices,
               const std::initializer_list<std::size_t> &row_pointers)
      : values{}, row_indices{}, row_pointers{} {

    // This may cause runtime error if the size of values is larger than V.
    std::copy(values.begin(), values.end(), this->values.begin());
    std::copy(row_indices.begin(), row_indices.end(),
              this->row_indices.begin());
    std::copy(row_pointers.begin(), row_pointers.end(),
              this->row_pointers.begin());
  }

  SparseMatrix(const std::array<T, V> &values,
               const std::array<std::size_t, V> &row_indices,
               const std::array<std::size_t, (M + 1)> &row_pointers)
      : values(values), row_indices(row_indices), row_pointers(row_pointers) {}

  SparseMatrix(const std::vector<T> &values,
               const std::vector<std::size_t> &row_indices,
               const std::vector<std::size_t> &row_pointers)
      : values{}, row_indices{}, row_pointers{} {

    // This may cause runtime error if the size of values is larger than V.
    std::copy(values.begin(), values.end(), this->values.begin());
    std::copy(row_indices.begin(), row_indices.end(),
              this->row_indices.begin());
    std::copy(row_pointers.begin(), row_pointers.end(),
              this->row_pointers.begin());
  }

  SparseMatrix(const Matrix<T, M, N> &input)
      : values{}, row_indices{}, row_pointers{} {

    std::size_t row_index_index = 0;
    std::size_t row_pointer_count = 0;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        if (Base::Math::abs(input(i, j)) >
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

#endif // __BASE_MATRIX_USE_STD_VECTOR__

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

public:
  /* Function */

  /**
   * @brief Creates a dense matrix from the sparse matrix.
   *
   * This function iterates through the non-zero elements of the sparse matrix
   * and fills a dense matrix with the corresponding values.
   *
   * @return A dense matrix representation of the sparse matrix.
   */
  inline Matrix<T, M, N> create_dense() const {
    Matrix<T, M, N> result;

    for (std::size_t j = 0; j < M; j++) {
      for (std::size_t k = this->row_pointers[j]; k < this->row_pointers[j + 1];
           k++) {
        result(j, this->row_indices[k]) = this->values[k];
      }
    }

    return result;
  }

  /**
   * @brief Creates a diagonal matrix from the sparse matrix.
   *
   * This function extracts the diagonal elements from the sparse matrix and
   * creates a diagonal matrix with those values.
   *
   * @return A diagonal matrix representation of the sparse matrix.
   */
  T value(std::size_t i) { return this->values[i]; }

  /**
   * @brief Returns the value at the specified index in the sparse matrix.
   *
   * This function retrieves the value at the given index in the sparse matrix.
   *
   * @param i The index of the value to retrieve.
   * @return The value at the specified index.
   */
  const T value(std::size_t i) const { return this->values[i]; }

  /**
   * @brief Returns the index of the row corresponding to the specified index.
   *
   * This function retrieves the row index for the given index in the sparse
   * matrix.
   *
   * @param i The index of the value to retrieve the row index for.
   * @return The row index corresponding to the specified index.
   */
  std::size_t row_index(std::size_t i) { return this->row_indices[i]; }

  const std::size_t row_index(std::size_t i) const {
    return this->row_indices[i];
  }

  /**
   * @brief Returns the pointer to the start of the specified row.
   *
   * This function retrieves the starting index of the specified row in the
   * sparse matrix.
   *
   * @param i The index of the row to retrieve the pointer for.
   * @return The starting index of the specified row.
   */
  std::size_t row_pointer(std::size_t i) { return this->row_pointers[i]; }

  /**
   * @brief Returns the pointer to the start of the specified row.
   *
   * This function retrieves the starting index of the specified row in the
   * sparse matrix.
   *
   * @param i The index of the row to retrieve the pointer for.
   * @return The starting index of the specified row.
   */
  const std::size_t row_pointer(std::size_t i) const {
    return this->row_pointers[i];
  }

  /**
   * @brief Returns the number of non-zero values in the sparse matrix.
   *
   * This function returns the total number of non-zero values stored in the
   * sparse matrix.
   *
   * @return The number of non-zero values in the sparse matrix.
   */
  inline Matrix<T, N, M> transpose(void) const {
    Matrix<T, N, M> Y;

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = this->row_pointers[j]; k < this->row_pointers[j + 1];
           ++k) {
        Y(this->row_indices[k], j) = this->values[k];
      }
    }

    return Y;
  }

  /**
   * @brief Returns the number of non-zero values in the sparse matrix.
   *
   * This function returns the total number of non-zero values stored in the
   * sparse matrix.
   *
   * @return The number of non-zero values in the sparse matrix.
   */
  inline Matrix<T, M, N> operator+(const Matrix<T, M, N> &B) const {
    Matrix<T, M, N> Y = B;

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = this->row_pointers[j]; k < this->row_pointers[j + 1];
           ++k) {
        Y(j, this->row_indices[k]) += this->values[k];
      }
    }

    return Y;
  }

  /**
   * @brief Adds a diagonal matrix to the sparse matrix.
   *
   * This function creates a dense matrix from the diagonal matrix and adds it
   * to the sparse matrix.
   *
   * @param B The diagonal matrix to add.
   * @return A dense matrix resulting from the addition.
   */
  inline Matrix<T, M, M> operator+(const DiagMatrix<T, M> &B) const {
    Matrix<T, M, M> Y = B.create_dense();

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = this->row_pointers[j]; k < this->row_pointers[j + 1];
           ++k) {
        Y(j, this->row_indices[k]) += this->values[k];
      }
    }

    return Y;
  }

  /**
   * @brief Adds a sparse matrix to the sparse matrix.
   *
   * This function creates a dense matrix from the sparse matrix and adds it to
   * the current sparse matrix.
   *
   * @param mat The sparse matrix to add.
   * @return A dense matrix resulting from the addition.
   */
  inline Matrix<T, M, N> operator+(const SparseMatrix<T, M, N, V> &mat) const {
    Matrix<T, M, N> Y = this->create_dense();

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = mat.row_pointer(j); k < mat.row_pointer(j + 1);
           ++k) {
        Y(j, mat.row_index(k)) += mat.value(k);
      }
    }

    return Y;
  }

  /**
   * @brief Subtracts a dense matrix from the sparse matrix.
   *
   * This function creates a dense matrix from the sparse matrix and subtracts
   * the given dense matrix from it.
   *
   * @param B The dense matrix to subtract.
   * @return A dense matrix resulting from the subtraction.
   */
  inline Matrix<T, M, N> operator-(const Matrix<T, M, N> &B) const {
    Matrix<T, M, N> Y = -B;

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = this->row_pointers[j]; k < this->row_pointers[j + 1];
           ++k) {
        Y(j, this->row_indices[k]) += this->values[k];
      }
    }

    return Y;
  }

  /**
   * @brief Subtracts a diagonal matrix from the sparse matrix.
   *
   * This function creates a dense matrix from the diagonal matrix and subtracts
   * it from the sparse matrix.
   *
   * @param B The diagonal matrix to subtract.
   * @return A dense matrix resulting from the subtraction.
   */
  inline Matrix<T, M, M> operator-(const DiagMatrix<T, M> &B) const {
    Matrix<T, M, M> Y = -(B.create_dense());

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = this->row_pointers[j]; k < this->row_pointers[j + 1];
           ++k) {
        Y(j, this->row_indices[k]) += this->values[k];
      }
    }

    return Y;
  }

  /**
   * @brief Subtracts a sparse matrix from the sparse matrix.
   *
   * This function creates a dense matrix from the sparse matrix and subtracts
   * the given sparse matrix from it.
   *
   * @param mat The sparse matrix to subtract.
   * @return A dense matrix resulting from the subtraction.
   */
  inline Matrix<T, M, N> operator-(const SparseMatrix<T, M, N, V> &mat) const {
    Matrix<T, M, N> Y = this->create_dense();

    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t k = mat.row_pointer(j); k < mat.row_pointer(j + 1);
           ++k) {
        Y(j, mat.row_index(k)) -= mat.value(k);
      }
    }

    return Y;
  }

  /**
   * @brief Multiplies the sparse matrix by a dense matrix.
   *
   * This function performs matrix multiplication between the sparse matrix and
   * the given dense matrix.
   *
   * @param B The dense matrix to multiply with.
   * @return A dense matrix resulting from the multiplication.
   */
  inline SparseMatrix<T, M, N, V> operator*(const T &scalar) const {
    SparseMatrix<T, M, N, V> Y = *this;

    for (std::size_t i = 0; i < V; i++) {
      Y.values[i] = scalar * this->values[i];
    }

    return Y;
  }

  /**
   * @brief Returns the number of rows in the sparse matrix.
   *
   * This function returns the number of rows in the sparse matrix.
   *
   * @return The number of rows in the sparse matrix.
   */
  constexpr std::size_t rows() const { return N; }

  /**
   * @brief Returns the number of columns in the sparse matrix.
   *
   * This function returns the number of columns in the sparse matrix.
   *
   * @return The number of columns in the sparse matrix.
   */
  constexpr std::size_t cols() const { return M; }

/* Variable */
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<T> values;
  std::vector<std::size_t> row_indices;
  std::vector<std::size_t> row_pointers;
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<T, V> values;
  std::array<std::size_t, V> row_indices;
  std::array<std::size_t, M + 1> row_pointers;
#endif // __BASE_MATRIX_USE_STD_VECTOR__
};

/** * @brief Creates a sparse matrix from a dense matrix.
 *
 * This function converts a dense matrix into a sparse matrix by extracting
 * non-zero elements and their corresponding row indices.
 *
 * @tparam T The type of elements in the matrix.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param A The dense matrix to convert.
 * @return A sparse matrix representation of the dense matrix.
 */
template <typename T, std::size_t M, std::size_t N>
inline SparseMatrix<T, M, N, (M * N)> create_sparse(const Matrix<T, M, N> &A) {
  std::size_t consecutive_index = 0;

#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<T> values(M * N);
  std::vector<std::size_t> row_indices(M * N);
  std::vector<std::size_t> row_pointers(M + 1);
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<T, (M * N)> values;
  std::array<std::size_t, (M * N)> row_indices;
  std::array<std::size_t, (M + 1)> row_pointers;
#endif // __BASE_MATRIX_USE_STD_VECTOR__

  row_pointers[0] = 0;
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

/**
 * @brief Creates a sparse matrix from a diagonal matrix.
 *
 * This function converts a diagonal matrix into a sparse matrix by extracting
 * the diagonal elements and their corresponding row indices.
 *
 * @tparam T The type of elements in the matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param A The diagonal matrix to convert.
 * @return A sparse matrix representation of the diagonal matrix.
 */
template <typename T, std::size_t M>
inline SparseMatrix<T, M, M, M> create_sparse(const DiagMatrix<T, M> &A) {
  std::size_t consecutive_index = 0;

#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<T> values(M);
  std::vector<std::size_t> row_indices(M);
  std::vector<std::size_t> row_pointers(M + 1);
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<T, M> values;
  std::array<std::size_t, M> row_indices;
  std::array<std::size_t, M + 1> row_pointers;
#endif // __BASE_MATRIX_USE_STD_VECTOR__

  row_pointers[0] = 0;
  for (std::size_t i = 0; i < M; i++) {
    values[consecutive_index] = A[i];
    row_indices[consecutive_index] = i;

    consecutive_index++;
    row_pointers[i + 1] = consecutive_index;
  }

  return SparseMatrix<T, M, M, M>(values, row_indices, row_pointers);
}

/* Operator */

/**
 * @brief Adds a dense matrix to a sparse matrix.
 *
 * This function creates a dense matrix from the sparse matrix and adds the
 * given dense matrix to it.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param B The dense matrix to add.
 * @param A The sparse matrix to add to.
 * @return A dense matrix resulting from the addition.
 */
template <typename T, std::size_t M, std::size_t V>
inline Matrix<T, M, M> operator+(const DiagMatrix<T, M> &B,
                                 const SparseMatrix<T, M, M, V> &A) {
  Matrix<T, M, M> Y = B.create_dense();

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = A.row_pointer(j); k < A.row_pointer(j + 1); ++k) {
      Y(j, A.row_index(k)) += A.value(k);
    }
  }

  return Y;
}

/**
 * @brief Subtracts a sparse matrix from a diagonal matrix.
 *
 * This function creates a dense matrix from the diagonal matrix and subtracts
 * the given sparse matrix from it.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param B The diagonal matrix to subtract from.
 * @param A The sparse matrix to subtract.
 * @return A dense matrix resulting from the subtraction.
 */
template <typename T, std::size_t M, std::size_t V>
inline Matrix<T, M, M> operator-(const DiagMatrix<T, M> &B,
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

/**
 * @brief Multiplies a scalar with a sparse matrix.
 *
 * This function scales all values in the sparse matrix by the given scalar.
 *
 * @tparam T The type of elements in the matrix.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param scalar The scalar value to multiply with.
 * @param A The sparse matrix to scale.
 * @return A new sparse matrix with scaled values.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t V>
inline SparseMatrix<T, M, N, V> operator*(const T &scalar,
                                          const SparseMatrix<T, M, N, V> &A) {
  SparseMatrix<T, M, N, V> Y = A;

  for (std::size_t i = 0; i < V; i++) {
    Y.values[i] = scalar * A.value(i);
  }

  return Y;
}

/* SparseMatrix * Vector */

/**
 * @brief Multiplies a sparse matrix with a vector.
 *
 * This function performs matrix-vector multiplication between the sparse matrix
 * and the given vector, returning a new vector as the result.
 *
 * @tparam T The type of elements in the matrix and vector.
 * @tparam M The number of columns in the sparse matrix.
 * @tparam N The number of rows in the sparse matrix (and size of the
 * vector).
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param A The sparse matrix to multiply.
 * @param b The vector to multiply with.
 * @return A new vector resulting from the multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t V>
inline Vector<T, M> operator*(const SparseMatrix<T, M, N, V> &A,
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

/**
 * @brief Multiplies a vector with a sparse matrix.
 *
 * This function performs vector-matrix multiplication between the given vector
 * and the sparse matrix, returning a new vector as the result.
 *
 * @tparam T The type of elements in the matrix and vector.
 * @tparam N The number of columns in the sparse matrix (and size of the
 * vector).
 * @tparam K The number of rows in the sparse matrix.
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param a The vector to multiply with.
 * @param B The sparse matrix to multiply.
 * @return A new vector resulting from the multiplication.
 */
template <typename T, std::size_t N, std::size_t K, std::size_t V>
inline ColVector<T, K>
colVector_a_mul_SparseB(const ColVector<T, N> &a,
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

/**
 * @brief Multiplies a sparse matrix with a dense matrix.
 *
 * This function performs matrix multiplication between the sparse matrix and
 * the given dense matrix, returning a new dense matrix as the result.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The number of columns in the sparse matrix.
 * @tparam N The number of rows in the sparse matrix (and columns in the dense
 * matrix).
 * @tparam K The number of columns in the dense matrix.
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param A The sparse matrix to multiply.
 * @param B The dense matrix to multiply with.
 * @return A new dense matrix resulting from the multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V>
inline Matrix<T, M, K> operator*(const SparseMatrix<T, M, N, V> &A,
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

/**
 * @brief Multiplies a dense matrix with a sparse matrix.
 *
 * This function performs matrix multiplication between the given dense matrix
 * and the sparse matrix, returning a new dense matrix as the result.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The number of columns in the dense matrix (and columns in the
 * sparse matrix).
 * @tparam N The number of rows in the dense matrix (and rows in the sparse
 * matrix).
 * @tparam K The number of rows in the sparse matrix.
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param A The dense matrix to multiply.
 * @param B The sparse matrix to multiply with.
 * @return A new dense matrix resulting from the multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V>
inline Matrix<T, M, K> operator*(const Matrix<T, M, N> &A,
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

/**
 * @brief Multiplies a sparse matrix with the transpose of a dense matrix.
 *
 * This function performs matrix multiplication between the sparse matrix and
 * the transpose of the given dense matrix, returning a new dense matrix as the
 * result.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The number of columns in the sparse matrix.
 * @tparam N The number of rows in the sparse matrix (and columns in the dense
 * matrix).
 * @tparam K The number of rows in the dense matrix.
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param A The sparse matrix to multiply.
 * @param B The dense matrix to multiply with (transposed).
 * @return A new dense matrix resulting from the multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V>
inline Matrix<T, M, K>
matrix_multiply_SparseA_mul_BTranspose(const SparseMatrix<T, M, N, V> &A,
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

/**
 * @brief Multiplies two sparse matrices.
 *
 * This function performs matrix multiplication between two sparse matrices,
 * returning a new dense matrix as the result.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The number of columns in the first sparse matrix.
 * @tparam N The number of rows in the first sparse matrix (and columns in the
 * second sparse matrix).
 * @tparam K The number of rows in the second sparse matrix.
 * @tparam V The maximum number of non-zero values in the first sparse matrix.
 * @param A The first sparse matrix to multiply.
 * @param B The second sparse matrix to multiply with.
 * @return A new dense matrix resulting from the multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V, std::size_t W>
inline Matrix<T, M, K> operator*(const SparseMatrix<T, M, N, V> &A,
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

/**
 * @brief Multiplies a sparse matrix with the transpose of another sparse
 * matrix.
 *
 * This function performs matrix multiplication between a sparse matrix and the
 * transpose of another sparse matrix, returning a new dense matrix as the
 * result.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The number of columns in the first sparse matrix.
 * @tparam N The number of rows in the first sparse matrix (and rows in the
 * second sparse matrix).
 * @tparam K The number of columns in the second sparse matrix.
 * @tparam V The maximum number of non-zero values in the first sparse matrix.
 * @param A The first sparse matrix to multiply.
 * @param B The second sparse matrix to multiply with (transposed).
 * @return A new dense matrix resulting from the multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V, std::size_t W>
inline Matrix<T, M, K> matrix_multiply_SparseA_mul_SparseBTranspose(
    const SparseMatrix<T, M, N, V> &A, const SparseMatrix<T, K, N, W> &B) {
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

/**
 * @brief Multiplies the transpose of a sparse matrix with another sparse
 * matrix.
 *
 * This function performs matrix multiplication between the transpose of a
 * sparse matrix and another sparse matrix, returning a new dense matrix as the
 * result.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The number of columns in the first sparse matrix (and columns in
 * the second sparse matrix).
 * @tparam N The number of rows in the first sparse matrix (and rows in the
 * second sparse matrix).
 * @tparam K The number of columns in the second sparse matrix.
 * @tparam V The maximum number of non-zero values in the first sparse matrix.
 * @param A The first sparse matrix to multiply (transposed).
 * @param B The second sparse matrix to multiply with.
 * @return A new dense matrix resulting from the multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V, std::size_t W>
inline Matrix<T, M, K> matrix_multiply_SparseATranspose_mul_SparseB(
    const SparseMatrix<T, N, M, V> &A, const SparseMatrix<T, N, K, W> &B) {
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

/**
 * @brief Multiplies a sparse matrix with a diagonal matrix.
 *
 * This function performs matrix multiplication between the sparse matrix and
 * the diagonal matrix, returning a new dense matrix as the result.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The number of columns in the sparse matrix.
 * @tparam N The number of rows in the sparse matrix (and size of the diagonal
 * matrix).
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param A The sparse matrix to multiply.
 * @param B The diagonal matrix to multiply with.
 * @return A new dense matrix resulting from the multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t V>
inline Matrix<T, M, N> operator*(const SparseMatrix<T, M, N, V> &A,
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

/**
 * @brief Multiplies a diagonal matrix with a sparse matrix.
 *
 * This function performs matrix multiplication between the diagonal matrix and
 * the sparse matrix, returning a new dense matrix as the result.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @tparam K The number of columns in the sparse matrix.
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param A The diagonal matrix to multiply.
 * @param B The sparse matrix to multiply with.
 * @return A new dense matrix resulting from the multiplication.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t V>
inline Matrix<T, M, K> operator*(const DiagMatrix<T, M> &A,
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

/**
 * @brief Multiplies a diagonal matrix with the transpose of a sparse matrix.
 *
 * This function performs matrix multiplication between the diagonal matrix and
 * the transpose of the sparse matrix, returning a new dense matrix as the
 * result.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @tparam K The number of rows in the sparse matrix.
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param A The diagonal matrix to multiply.
 * @param B The sparse matrix to multiply with (transposed).
 * @return A new dense matrix resulting from the multiplication.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t V>
inline Matrix<T, K, M>
matrix_multiply_Transpose_DiagA_mul_SparseB(const DiagMatrix<T, M> &A,
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

/**
 * @brief Adds a sparse matrix to a dense matrix.
 *
 * This function creates a dense matrix from the sparse matrix and adds the
 * given dense matrix to it.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param B The dense matrix to add.
 * @param SA The sparse matrix to add to.
 * @return A dense matrix resulting from the addition.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t V>
inline Matrix<T, M, N> operator+(const Matrix<T, M, N> &B,
                                 const SparseMatrix<T, M, N, V> SA) {
  Matrix<T, M, N> Y = B;

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = SA.row_pointer(j); k < SA.row_pointer(j + 1); ++k) {
      Y(j, SA.row_index(k)) += SA.value(k);
    }
  }

  return Y;
}

/**
 * @brief Subtracts a sparse matrix from a dense matrix.
 *
 * This function creates a dense matrix from the sparse matrix and subtracts
 * the given dense matrix from it.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param B The dense matrix to subtract.
 * @param SA The sparse matrix to subtract from.
 * @return A dense matrix resulting from the subtraction.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t V>
inline Matrix<T, M, N> operator-(const Matrix<T, M, N> &B,
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

/**
 * @brief Multiplies a dense matrix with the transpose of a sparse matrix.
 *
 * This function performs matrix multiplication between the dense matrix and
 * the transpose of the sparse matrix, returning a new dense matrix as the
 * result.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The number of columns in the dense matrix (and rows in the sparse
 * matrix).
 * @tparam N The number of rows in the dense matrix (and columns in the sparse
 * matrix).
 * @tparam K The number of rows in the sparse matrix.
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param A The dense matrix to multiply.
 * @param SB The sparse matrix to multiply with (transposed).
 * @return A new dense matrix resulting from the multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V>
inline Matrix<T, N, K>
matrix_multiply_A_mul_SparseBTranspose(const Matrix<T, M, N> &A,
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

/**
 * @brief Multiplies the transpose of a sparse matrix with a dense matrix.
 *
 * This function performs matrix multiplication between the transpose of the
 * sparse matrix and the dense matrix, returning a new dense matrix as the
 * result.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam N The number of rows in the sparse matrix (and columns in the dense
 * matrix).
 * @tparam M The number of columns in the sparse matrix (and rows in the dense
 * matrix).
 * @tparam K The number of columns in the dense matrix.
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param A The sparse matrix to multiply (transposed).
 * @param B The dense matrix to multiply with.
 * @return A new dense matrix resulting from the multiplication.
 */
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          std::size_t V>
inline Matrix<T, M, K>
matrix_multiply_ATranspose_mul_SparseB(const Matrix<T, N, M> &A,
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

/**
 * @brief Multiplies the transpose of a sparse matrix with a dense matrix.
 *
 * This function performs matrix multiplication between the transpose of the
 * sparse matrix and the dense matrix, returning a new dense matrix as the
 * result.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam N The number of rows in the sparse matrix (and columns in the dense
 * matrix).
 * @tparam M The number of columns in the sparse matrix (and rows in the dense
 * matrix).
 * @tparam K The number of columns in the dense matrix.
 * @tparam V The maximum number of non-zero values in the sparse matrix.
 * @param A The sparse matrix to multiply (transposed).
 * @param B The dense matrix to multiply with.
 * @return A new dense matrix resulting from the multiplication.
 */
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          std::size_t V>
inline Matrix<T, M, K>
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

#endif // __BASE_MATRIX_SPARSE_HPP__
