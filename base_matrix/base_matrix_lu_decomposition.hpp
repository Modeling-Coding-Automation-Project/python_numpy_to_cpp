/**
 * @file base_matrix_lu_decomposition.hpp
 * @brief Provides LU decomposition functionality for square matrices.
 *
 * This file defines the LUDecomposition class template within the Base::Matrix
 * namespace. The LUDecomposition class implements LU decomposition for square
 * matrices of fixed size, supporting operations such as matrix decomposition,
 * solving linear systems, and computing determinants. The class is templated on
 * the matrix element type and matrix size, and supports copy and move
 * semantics.
 */
#ifndef BASE_MATRIX_LU_DECOMPOSITION_HPP_
#define BASE_MATRIX_LU_DECOMPOSITION_HPP_

#include "base_matrix_macros.hpp"

#include "base_math.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"

#include <cstddef>

namespace Base {
namespace Matrix {

constexpr double DEFAULT_DIVISION_MIN_LU_DECOMPOSITION = 1.0e-10;

/**
 * @brief LU decomposition class for square matrices.
 *
 * This class implements LU decomposition for square matrices of fixed size M.
 * It provides methods to decompose a matrix into lower and upper triangular
 * matrices, solve linear systems, and compute determinants.
 *
 * @tparam T The type of the matrix elements (e.g., float, double).
 * @tparam M The size of the square matrix (M x M).
 */
template <typename T, std::size_t M> class LUDecomposition {
public:
  LUDecomposition()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_LU_DECOMPOSITION)) {}

  LUDecomposition(const DiagMatrix<T, M> &matrix)
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_LU_DECOMPOSITION)) {
    this->Lower_ = Matrix<T, M, M>::identity();
    this->Upper_ = Base::Matrix::output_dense_matrix(matrix);
  }

  /* Copy Constructor */
  LUDecomposition(const LUDecomposition<T, M> &other)
      : division_min(other.division_min), Lower_(other.Lower_),
        Upper_(other.Upper_), _pivot_index_vec(other._pivot_index_vec) {}

  LUDecomposition<T, M> &operator=(const LUDecomposition<T, M> &other) {
    if (this != &other) {
      this->division_min = other.division_min;
      this->Lower_ = other.Lower_;
      this->Upper_ = other.Upper_;
      this->_pivot_index_vec = other._pivot_index_vec;
    }
    return *this;
  }

  /* Move Constructor */
  LUDecomposition(LUDecomposition<T, M> &&other) noexcept
      : division_min(std::move(other.division_min)),
        Lower_(std::move(other.Lower_)), Upper_(std::move(other.Upper_)),
        _pivot_index_vec(std::move(other._pivot_index_vec)) {}

  LUDecomposition<T, M> &operator=(LUDecomposition<T, M> &&other) noexcept {
    if (this != &other) {
      this->division_min = std::move(other.division_min);
      this->Lower_ = std::move(other.Lower_);
      this->Upper_ = std::move(other.Upper_);
      this->_pivot_index_vec = std::move(other._pivot_index_vec);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Returns the lower triangular matrix (L) from the LU decomposition.
   *
   * @return Matrix<T, M, M> The lower triangular matrix of the decomposition.
   */
  inline Matrix<T, M, M> get_L() const { return Lower_; }

  /**
   * @brief Returns the upper triangular matrix (U) from the LU decomposition.
   *
   * @return Matrix<T, M, M> The upper triangular matrix of the decomposition.
   */
  inline Matrix<T, M, M> get_U() const { return Upper_; }

  /**
   * @brief Solves the linear system Ax = b using LU decomposition.
   *
   * This function performs LU decomposition of the input matrix A, applies the
   * pivoting to the right-hand side vector b, and then solves the resulting
   * lower and upper triangular systems using forward and backward substitution.
   *
   * @tparam T The type of the matrix and vector elements.
   * @tparam M The dimension of the square matrix and vectors.
   * @param A The coefficient matrix (assumed to be square of size MxM).
   * @param b The right-hand side vector.
   * @return Vector<T, M> The solution vector x such that Ax = b.
   */
  inline Vector<T, M> solve(const Matrix<T, M, M> &A, const Vector<T, M> &b) {
    this->_decompose(A);

    Vector<T, M> b_p;
    for (std::size_t i = 0; i < M; i++) {
      b_p[i] = b[this->_pivot_index_vec[i]];
    }

    Vector<T, M> y = this->_forward_substitution(b_p);
    return this->_backward_substitution(y);
  }

  /**
   * @brief Solves the linear system Ax = b using LU decomposition.
   *
   * This function performs LU decomposition of the input matrix A and solves
   * the linear system Ax = b using the decomposed matrices.
   *
   * @param A The coefficient matrix (assumed to be square of size MxM).
   * @return Vector<T, M> The solution vector x such that Ax = b.
   */
  inline void solve(const Matrix<T, M, M> &A) { this->_decompose(A); }

  /**
   * @brief Computes the determinant of the matrix from the LU decomposition.
   *
   * This function calculates the determinant of the matrix by multiplying the
   * diagonal elements of the lower and upper triangular matrices obtained from
   * LU decomposition.
   *
   * @return T The determinant of the matrix.
   */
  inline T get_determinant() const {
    T det = static_cast<T>(1);

    for (std::size_t i = 0; i < M; i++) {
      det *= this->Lower_(i, i) * this->Upper_(i, i);
    }

    return det;
  }

public:
  /* Variable */
  T division_min;

protected:
  /* Variable */
  Matrix<T, M, M> Lower_;
  Matrix<T, M, M> Upper_;
  Vector<std::size_t, M> _pivot_index_vec;

protected:
  /* Function */

  /**
   * @brief Decomposes the input matrix into lower and upper triangular
   * matrices.
   *
   * This function performs LU decomposition of the input square matrix, storing
   * the results in the Lower_ and Upper_ member variables. It also handles
   * pivoting to ensure numerical stability.
   *
   * @param matrix The input square matrix to be decomposed.
   */
  inline void _decompose(const Matrix<T, M, M> &matrix) {
    this->Lower_ = Matrix<T, M, M>();
    this->Upper_ = matrix;

    for (std::size_t i = 0; i < M; ++i) {
      this->_pivot_index_vec[i] = i;
    }

    for (std::size_t i = 0; i < M; ++i) {
      this->Lower_(i, i) = 1;

      // Pivoting
      if (Base::Utility::near_zero(this->Upper_(i, i), this->division_min)) {
        std::size_t maxCol = i;
        T maxVal = Base::Math::abs(this->Upper_(i, i));
        for (std::size_t k = i + 1; k < M; ++k) {
          T absVal = Base::Math::abs(this->Upper_(k, i));
          if (absVal > maxVal) {
            maxVal = absVal;
            maxCol = k;
          }
        }
        if (maxCol != i) {
          Base::Utility::swap_value(this->_pivot_index_vec[i],
                                    this->_pivot_index_vec[maxCol]);
          Base::Matrix::matrix_row_swap(i, maxCol, this->Upper_);
          Base::Matrix::matrix_row_swap(i, maxCol, this->Lower_);
        }
      }

      for (std::size_t j = i + 1; j < M; ++j) {
        T factor = this->Upper_(j, i) /
                   Base::Utility::avoid_zero_divide(this->Upper_(i, i),
                                                    this->division_min);
        this->Lower_(j, i) = factor;
        for (std::size_t k = i; k < M; ++k) {
          this->Upper_(j, k) -= factor * this->Upper_(i, k);
        }
      }
    }
  }

  /**
   * @brief Performs forward substitution to solve the lower triangular system.
   *
   * This function computes the intermediate vector y by solving the lower
   * triangular system Ly = b, where L is the lower triangular matrix from LU
   * decomposition and b is the right-hand side vector.
   *
   * @param b The right-hand side vector.
   * @return Vector<T, M> The intermediate vector y.
   */
  inline Vector<T, M> _forward_substitution(const Vector<T, M> &b) const {
    Vector<T, M> y;
    for (std::size_t i = 0; i < M; ++i) {
      T sum = b[i];
      for (std::size_t j = 0; j < i; ++j) {
        sum -= this->Lower_(i, j) * y[j];
      }
      y[i] = sum;
    }
    return y;
  }

  /**
   * @brief Performs backward substitution to solve the upper triangular system.
   *
   * This function computes the solution vector x by solving the upper
   * triangular system Ux = y, where U is the upper triangular matrix from LU
   * decomposition and y is the intermediate vector obtained from forward
   * substitution.
   *
   * @param y The intermediate vector obtained from forward substitution.
   * @return Vector<T, M> The solution vector x.
   */
  inline Vector<T, M> _backward_substitution(const Vector<T, M> &y) const {
    Vector<T, M> x;
    for (std::size_t i = M; i-- > 0;) {
      T sum = y[i];
      for (std::size_t j = i + 1; j < M; ++j) {
        sum -= this->Upper_(i, j) * x[j];
      }
      x[i] = sum / Base::Utility::avoid_zero_divide(this->Upper_(i, i),
                                                    this->division_min);
    }
    return x;
  }
};

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_LU_DECOMPOSITION_HPP_
