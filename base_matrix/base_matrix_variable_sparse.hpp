/**
 * @file base_matrix_variable_sparse.hpp
 * @brief Defines the VariableSparseMatrix class template and related matrix
 * multiplication operators for sparse matrix computations.
 *
 * This file provides the implementation of a variable-size sparse matrix class,
 * `VariableSparseMatrix`, which supports both standard vector and fixed-size
 * array storage depending on compile-time flags. The class is designed for
 * efficient storage and manipulation of sparse matrices, and includes copy/move
 * constructors and assignment operators.
 *
 * The file also defines several overloaded operator* functions to enable
 * multiplication between VariableSparseMatrix, dense Matrix, and SparseMatrix
 * types, supporting various combinations of sparse and dense matrix
 * multiplication.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __BASE_MATRIX_VARIABLE_SPARSE_HPP__
#define __BASE_MATRIX_VARIABLE_SPARSE_HPP__

#include "base_matrix_macros.hpp"

#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_vector.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

/**
 * @brief VariableSparseMatrix class template for sparse matrix representation.
 *
 * This class template provides a sparse matrix representation that can be
 * configured to use either `std::vector` or fixed-size `std::array` for
 * storage, depending on the compile-time flag `__BASE_MATRIX_USE_STD_VECTOR__`.
 *
 * It supports basic operations such as copy and move semantics, and provides
 * access to matrix values, row indices, and row pointers.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 */
template <typename T, std::size_t M, std::size_t N> class VariableSparseMatrix {
public:
/* Constructor */
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  VariableSparseMatrix()
      : values(M * N, static_cast<T>(0)),
        row_indices(M * N, static_cast<std::size_t>(0)),
        row_pointers(M + 1, static_cast<std::size_t>(0)) {}
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  VariableSparseMatrix() : values{}, row_indices{}, row_pointers{} {}
#endif // __BASE_MATRIX_USE_STD_VECTOR__

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

public:
  /* Function */

  /**
   * @brief Accessor for the value at index i.
   *
   * This function returns the value at the specified index i in the sparse
   * matrix.
   *
   * @param i The index of the value to access.
   * @return The value at index i.
   */
  T value(std::size_t i) { return this->values[i]; }

  /**
   * @brief Const accessor for the value at index i.
   *
   * This function returns the value at the specified index i in the sparse
   * matrix, without allowing modification.
   *
   * @param i The index of the value to access.
   * @return The value at index i.
   */
  const T value(std::size_t i) const { return this->values[i]; }

  /**
   * @brief Accessor for the row index at index i.
   *
   * This function returns the row index at the specified index i in the sparse
   * matrix.
   *
   * @param i The index of the row index to access.
   * @return The row index at index i.
   */
  std::size_t row_index(std::size_t i) { return this->row_indices[i]; }

  /**
   * @brief Const accessor for the row index at index i.
   *
   * This function returns the row index at the specified index i in the sparse
   * matrix, without allowing modification.
   *
   * @param i The index of the row index to access.
   * @return The row index at index i.
   */
  const std::size_t row_index(std::size_t i) const {
    return this->row_indices[i];
  }

  /**
   * @brief Accessor for the row pointer at index i.
   *
   * This function returns the row pointer at the specified index i in the
   * sparse matrix.
   *
   * @param i The index of the row pointer to access.
   * @return The row pointer at index i.
   */
  std::size_t row_pointer(std::size_t i) { return this->row_pointers[i]; }

  /**
   * @brief Const accessor for the row pointer at index i.
   *
   * This function returns the row pointer at the specified index i in the
   * sparse matrix, without allowing modification.
   *
   * @param i The index of the row pointer to access.
   * @return The row pointer at index i.
   */
  const std::size_t row_pointer(std::size_t i) const {
    return this->row_pointers[i];
  }

/* Variable */
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<T> values;
  std::vector<std::size_t> row_indices;
  std::vector<std::size_t> row_pointers;
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<T, M * N> values;
  std::array<std::size_t, M * N> row_indices;
  std::array<std::size_t, M + 1> row_pointers;
#endif // __BASE_MATRIX_USE_STD_VECTOR__
};

/* SparseMatrix * Matrix */

/**
 * @brief Overloaded operator* for multiplying a VariableSparseMatrix with a
 * Matrix.
 *
 * This function performs matrix multiplication between a VariableSparseMatrix
 * and a Matrix, returning the resulting Matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the VariableSparseMatrix.
 * @tparam N The number of rows in the VariableSparseMatrix and rows in the
 * Matrix.
 * @tparam K The number of columns in the Matrix.
 * @param A The VariableSparseMatrix to multiply.
 * @param B The Matrix to multiply with.
 * @return The resulting Matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K>
inline Matrix<T, M, K> operator*(const VariableSparseMatrix<T, M, N> &A,
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
 * @brief Overloaded operator* for multiplying a SparseMatrix with a
 * VariableSparseMatrix.
 *
 * This function performs matrix multiplication between a SparseMatrix and a
 * VariableSparseMatrix, returning the resulting Matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the SparseMatrix.
 * @tparam N The number of rows in the SparseMatrix and rows in the
 * VariableSparseMatrix.
 * @tparam K The number of columns in the VariableSparseMatrix.
 * @param A The SparseMatrix to multiply.
 * @param B The VariableSparseMatrix to multiply with.
 * @return The resulting Matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K>
inline Matrix<T, M, K> operator*(const Matrix<T, M, N> &A,
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

/**
 * @brief Overloaded operator* for multiplying two SparseMatrix objects.
 *
 * This function performs matrix multiplication between two SparseMatrix
 * objects, returning the resulting Matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the first SparseMatrix.
 * @tparam N The number of rows in the first SparseMatrix and rows in the
 * second SparseMatrix.
 * @tparam K The number of columns in the second SparseMatrix.
 * @tparam V The maximum number of non-zero values in the SparseMatrix.
 * @param A The first SparseMatrix to multiply.
 * @param B The second SparseMatrix to multiply with.
 * @return The resulting Matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V>
inline Matrix<T, M, K> operator*(const VariableSparseMatrix<T, M, N> &A,
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

/**
 * @brief Overloaded operator* for multiplying a SparseMatrix with a
 * VariableSparseMatrix.
 *
 * This function performs matrix multiplication between a SparseMatrix and a
 * VariableSparseMatrix, returning the resulting Matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the SparseMatrix.
 * @tparam N The number of rows in the SparseMatrix and rows in the
 * VariableSparseMatrix.
 * @tparam K The number of columns in the VariableSparseMatrix.
 * @tparam V The maximum number of non-zero values in the SparseMatrix.
 * @param A The SparseMatrix to multiply.
 * @param B The VariableSparseMatrix to multiply with.
 * @return The resulting Matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V>
inline Matrix<T, M, K> operator*(const SparseMatrix<T, M, N, V> &A,
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

/**
 * @brief Overloaded operator* for multiplying two VariableSparseMatrix
 * objects.
 *
 * This function performs matrix multiplication between two VariableSparseMatrix
 * objects, returning the resulting Matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the first VariableSparseMatrix.
 * @tparam N The number of rows in the first VariableSparseMatrix and rows in
 * the second VariableSparseMatrix.
 * @tparam K The number of columns in the second VariableSparseMatrix.
 * @param A The first VariableSparseMatrix to multiply.
 * @param B The second VariableSparseMatrix to multiply with.
 * @return The resulting Matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K>
inline Matrix<T, M, K> operator*(const VariableSparseMatrix<T, M, N> &A,
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

#endif // __BASE_MATRIX_VARIABLE_SPARSE_HPP__
