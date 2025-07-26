/**
 * @file base_matrix_qr_decomposition.hpp
 * @brief QR decomposition utilities for dense, diagonal, and sparse matrices.
 *
 * This header provides template classes and functions for performing QR
 * decomposition using Givens rotations on various matrix types, including
 * dense, diagonal, and compiled sparse matrices. The QR decomposition is a
 * fundamental matrix factorization technique used in numerical linear algebra.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __BASE_MATRIX_QR_DECOMPOSITION_HPP__
#define __BASE_MATRIX_QR_DECOMPOSITION_HPP__

#include "base_matrix_macros.hpp"

#include "base_math.hpp"
#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_vector.hpp"

#include <cstddef>

namespace Base {
namespace Matrix {

constexpr double DEFAULT_DIVISION_MIN_QR = 1.0e-10;

/* Given's Rotation */

/**
 * @brief Performs a Givens rotation on the specified rows of the matrix.
 *
 * This function applies a Givens rotation to the rows i and j of the R_matrix
 * and updates the Q_matrix accordingly. The rotation is used to zero out the
 * element at (i, j) in the R_matrix.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The number of columns in the Q_matrix.
 * @tparam N The number of rows in the R_matrix.
 * @param i The index of the first row to rotate.
 * @param j The index of the second row to rotate.
 * @param Q_matrix The orthogonal matrix being built.
 * @param R_matrix The matrix being decomposed.
 * @param division_min A small value to avoid division by zero.
 */
template <typename T, std::size_t M, std::size_t N>
static inline void
qr_givensRotation(std::size_t i, std::size_t j, Matrix<T, M, M> &Q_matrix,
                  Matrix<T, M, N> &R_matrix, T division_min) {
  T c;
  T s;

  if (Base::Math::abs(R_matrix(i, j)) > Base::Math::abs(R_matrix(j, j))) {
    T t = -R_matrix(j, j) /
          Base::Utility::avoid_zero_divide(R_matrix(i, j), division_min);
    s = Base::Math::rsqrt<T>(static_cast<T>(1) + t * t, division_min);
    c = s * t;

    for (std::size_t k = j; k < N; ++k) {
      T x = R_matrix(j, k);
      T y = R_matrix(i, k);
      R_matrix(j, k) = c * x - s * y;
      R_matrix(i, k) = s * x + c * y;
    }

    for (std::size_t k = 0; k < M; ++k) {
      T u = Q_matrix(k, j);
      T v = Q_matrix(k, i);
      Q_matrix(k, j) = c * u - s * v;
      Q_matrix(k, i) = s * u + c * v;
    }
  } else {
    T t = -R_matrix(i, j) /
          Base::Utility::avoid_zero_divide(R_matrix(j, j), division_min);
    c = Base::Math::rsqrt<T>(static_cast<T>(1) + t * t, division_min);
    s = c * t;

    for (std::size_t k = j; k < N; ++k) {
      T x = R_matrix(j, k);
      T y = R_matrix(i, k);
      R_matrix(j, k) = c * x - s * y;
      R_matrix(i, k) = s * x + c * y;
    }

    for (std::size_t k = 0; k < M; ++k) {
      T u = Q_matrix(k, j);
      T v = Q_matrix(k, i);
      Q_matrix(k, j) = c * u - s * v;
      Q_matrix(k, i) = s * u + c * v;
    }
  }
}

/**
 * @brief QR Decomposition using Givens rotations.
 *
 * This class performs QR decomposition on a matrix using Givens rotations.
 * It decomposes the input matrix A into an orthogonal matrix Q and an upper
 * triangular matrix R such that A = Q * R.
 *
 * @tparam T The type of elements in the matrices.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 */
template <typename T, std::size_t M, std::size_t N> class QRDecomposition {
public:
  /* Check Compatibility */
  static_assert(M >= N, "Incompatible matrix dimensions");

public:
  /* Constructor */
  QRDecomposition()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_QR)),
        _Q_matrix(Matrix<T, M, M>::identity()), _R_matrix() {}

  /* Copy Constructor */
  QRDecomposition(const QRDecomposition<T, M, N> &other)
      : division_min(other.division_min), _Q_matrix(other._Q_matrix),
        _R_matrix(other._R_matrix) {}

  QRDecomposition<T, M, N> &operator=(const QRDecomposition<T, M, N> &other) {
    if (this != &other) {
      this->division_min = other.division_min;
      this->_Q_matrix = other._Q_matrix;
      this->_R_matrix = other._R_matrix;
    }

    return *this;
  }

  /* Move Constructor */
  QRDecomposition(QRDecomposition<T, M, N> &&other) noexcept
      : division_min(other.division_min), _Q_matrix(std::move(other._Q_matrix)),
        _R_matrix(std::move(other._R_matrix)) {}

  QRDecomposition<T, M, N> &
  operator=(QRDecomposition<T, M, N> &&other) noexcept {
    if (this != &other) {
      this->division_min = other.division_min;
      this->_Q_matrix = std::move(other._Q_matrix);
      this->_R_matrix = std::move(other._R_matrix);
    }

    return *this;
  }

public:
  /* Function */

  /**
   * @brief Solves the QR decomposition for the given matrix A.
   *
   * This function sets the internal R matrix to the provided matrix A
   * and performs QR decomposition by calling the internal _decompose method.
   *
   * @param A The input matrix to decompose.
   */
  inline void solve(const Matrix<T, M, N> &A) {
    this->_R_matrix = A;
    this->_decompose();
  }

  /**
   * @brief Returns the orthogonal matrix Q from the QR decomposition.
   *
   * This function returns the orthogonal matrix Q that was computed during
   * the QR decomposition process.
   *
   * @return The orthogonal matrix Q.
   */
  inline Matrix<T, M, M> get_Q() const { return this->_Q_matrix; }

  /**
   * @brief Returns the upper triangular matrix R from the QR decomposition.
   *
   * This function returns the upper triangular matrix R that was computed
   * during the QR decomposition process.
   *
   * @return The upper triangular matrix R.
   */
  inline Matrix<T, M, N> get_R() const { return this->_R_matrix; }

public:
  /* Variable */
  T division_min;

protected:
  /* Variable */
  Matrix<T, M, M> _Q_matrix;
  Matrix<T, M, N> _R_matrix;

protected:
  /* Function */

  /**
   * @brief Performs QR decomposition of the matrix using Givens rotations.
   *
   * Iterates over the lower triangular part of the matrix and applies Givens
   * rotations to zero out sub-diagonal elements, transforming the matrix into
   * an upper triangular form. The method checks if each sub-diagonal element is
   * not near zero (within a specified threshold) before applying the rotation.
   */
  inline void _decompose() {
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t i = j + 1; i < M; ++i) {

        if (!Base::Utility::near_zero(this->_R_matrix(i, j),
                                      this->division_min)) {
          this->_givensRotation(i, j);
        }
      }
    }
  }

  /**
   * @brief Applies a Givens rotation to the specified rows of the matrix.
   *
   * This function performs a Givens rotation on the rows i and j of the
   * R_matrix and updates the Q_matrix accordingly. The rotation is used to zero
   * out the element at (i, j) in the R_matrix.
   *
   * @param i The index of the first row to rotate.
   * @param j The index of the second row to rotate.
   */
  inline void _givensRotation(std::size_t i, std::size_t j) {
    Base::Matrix::qr_givensRotation(i, j, this->_Q_matrix, this->_R_matrix,
                                    this->division_min);
  }
};

/**
 * @brief QR Decomposition for Diagonal Matrices.
 *
 * This class provides a specialized implementation of QR decomposition for
 * diagonal matrices. It assumes that the input matrix is diagonal and performs
 * the decomposition accordingly.
 *
 * @tparam T The type of elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows/columns).
 */
template <typename T, std::size_t M> class QRDecompositionDiag {
public:
  /* Constructor */
  QRDecompositionDiag()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_QR)),
        _Q_matrix(DiagMatrix<T, M>::identity()), _R_matrix() {}

  /* Copy Constructor */
  QRDecompositionDiag(const QRDecompositionDiag<T, M> &other)
      : division_min(other.division_min), _Q_matrix(other._Q_matrix),
        _R_matrix(other._R_matrix) {}
  QRDecompositionDiag<T, M> &operator=(const QRDecompositionDiag<T, M> &other) {
    if (this != &other) {
      this->division_min = other.division_min;
      this->_Q_matrix = other._Q_matrix;
      this->_R_matrix = other._R_matrix;
    }
    return *this;
  }

  /* Move Constructor */
  QRDecompositionDiag(QRDecompositionDiag<T, M> &&other) noexcept
      : division_min(other.division_min), _Q_matrix(std::move(other._Q_matrix)),
        _R_matrix(std::move(other._R_matrix)) {}
  QRDecompositionDiag<T, M> &
  operator=(QRDecompositionDiag<T, M> &&other) noexcept {
    if (this != &other) {
      this->division_min = other.division_min;
      this->_Q_matrix = std::move(other._Q_matrix);
      this->_R_matrix = std::move(other._R_matrix);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Solves the QR decomposition for the given diagonal matrix A.
   *
   * This function sets the internal R matrix to the provided diagonal matrix A
   * and performs QR decomposition by calling the internal solve method.
   *
   * @param A The input diagonal matrix to decompose.
   */
  inline void solve(const DiagMatrix<T, M> &A) { this->_R_matrix = A; }

  /**
   * @brief Returns the orthogonal matrix Q from the QR decomposition.
   *
   * This function returns the orthogonal matrix Q that was computed during
   * the QR decomposition process.
   *
   * @return The orthogonal matrix Q.
   */
  inline DiagMatrix<T, M> get_Q() const { return this->_Q_matrix; }

  /**
   * @brief Returns the upper triangular matrix R from the QR decomposition.
   *
   * This function returns the upper triangular matrix R that was computed
   * during the QR decomposition process.
   *
   * @return The upper triangular matrix R.
   */
  inline DiagMatrix<T, M> get_R() const { return this->_R_matrix; }

public:
  /* Variable */
  T division_min;

protected:
  /* Variable */
  DiagMatrix<T, M> _Q_matrix;
  DiagMatrix<T, M> _R_matrix;
};

/**
 * @brief QR Decomposition for Sparse Matrices.
 *
 * This class provides a specialized implementation of QR decomposition for
 * sparse matrices. It uses Givens rotations to decompose the input matrix into
 * an orthogonal matrix Q and an upper triangular matrix R.
 *
 * @tparam T The type of elements in the sparse matrix.
 * @tparam M The number of columns in the sparse matrix.
 * @tparam N The number of rows in the sparse matrix.
 * @tparam RowIndices_A Type representing row indices of the sparse matrix.
 * @tparam RowPointers_A Type representing row pointers of the sparse matrix.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
class QRDecompositionSparse {
public:
  /* Check Compatibility */
  static_assert(M >= N, "Incompatible matrix dimensions");

public:
  /* Constructor */
  QRDecompositionSparse()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_QR)),
        _Q_matrix(Matrix<T, M, M>::identity()), _R_matrix() {}

  /* Copy Constructor */
  QRDecompositionSparse(
      const QRDecompositionSparse<T, M, N, RowIndices_A, RowPointers_A> &other)
      : division_min(other.division_min), _Q_matrix(other._Q_matrix),
        _R_matrix(other._R_matrix) {}
  QRDecompositionSparse<T, M, N, RowIndices_A, RowPointers_A> &
  operator=(const QRDecompositionSparse<T, M, N, RowIndices_A, RowPointers_A>
                &other) {
    if (this != &other) {
      this->division_min = other.division_min;
      this->_Q_matrix = other._Q_matrix;
      this->_R_matrix = other._R_matrix;
    }
    return *this;
  }

  /* Move Constructor */
  QRDecompositionSparse(QRDecompositionSparse<T, M, N, RowIndices_A,
                                              RowPointers_A> &&other) noexcept
      : division_min(other.division_min), _Q_matrix(std::move(other._Q_matrix)),
        _R_matrix(std::move(other._R_matrix)) {}
  QRDecompositionSparse<T, M, N, RowIndices_A, RowPointers_A> &
  operator=(QRDecompositionSparse<T, M, N, RowIndices_A, RowPointers_A>
                &&other) noexcept {
    if (this != &other) {
      this->division_min = other.division_min;
      this->_Q_matrix = std::move(other._Q_matrix);
      this->_R_matrix = std::move(other._R_matrix);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Solves the QR decomposition for the given sparse matrix A.
   *
   * This function sets the internal R matrix to the provided sparse matrix A
   * and performs QR decomposition by calling the internal _decompose method.
   *
   * @param A The input sparse matrix to decompose.
   */
  inline void
  solve(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
    this->_R_matrix = Base::Matrix::output_dense_matrix(A);
    this->_decompose(A);
  }

  /**
   * @brief Returns the orthogonal matrix Q from the QR decomposition.
   *
   * This function returns the orthogonal matrix Q that was computed during
   * the QR decomposition process.
   *
   * @return The orthogonal matrix Q.
   */
  inline Matrix<T, M, M> get_Q() const { return this->_Q_matrix; }

  /**
   * @brief Returns the upper triangular matrix R from the QR decomposition.
   *
   * This function returns the upper triangular matrix R that was computed
   * during the QR decomposition process.
   *
   * @return The upper triangular matrix R.
   */
  inline Matrix<T, M, N> get_R() const { return this->_R_matrix; }

public:
  /* Variable */
  T division_min;

protected:
  /* Variable */
  Matrix<T, M, M> _Q_matrix;
  Matrix<T, M, N> _R_matrix;

protected:
  /* Function */

  /**
   * @brief Performs QR decomposition of the sparse matrix using Givens
   * rotations.
   *
   * Iterates over the non-zero elements of the sparse matrix and applies Givens
   * rotations to zero out sub-diagonal elements, transforming the matrix into
   * an upper triangular form. The method checks if each sub-diagonal element is
   * not near zero (within a specified threshold) before applying the rotation.
   *
   * @param A The input sparse matrix to decompose.
   */
  inline void _decompose(
      const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t k = RowPointers_A::list[i];
           k < RowPointers_A::list[i + 1]; k++) {
        if ((i >= RowIndices_A::list[k] + 1) &&
            (!Base::Utility::near_zero(A.values[k], this->division_min))) {
          this->_givensRotation(i, RowIndices_A::list[k]);
        }
      }
    }
  }

  /**
   * @brief Applies a Givens rotation to the specified rows of the matrix.
   *
   * This function performs a Givens rotation on the rows i and j of the
   * R_matrix and updates the Q_matrix accordingly. The rotation is used to zero
   * out the element at (i, j) in the R_matrix.
   *
   * @param i The index of the first row to rotate.
   * @param j The index of the second row to rotate.
   */
  inline void _givensRotation(std::size_t i, std::size_t j) {
    Base::Matrix::qr_givensRotation(i, j, this->_Q_matrix, this->_R_matrix,
                                    this->division_min);
  }
};

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_QR_DECOMPOSITION_HPP__
