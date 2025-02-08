#ifndef __BASE_MATRIX_LU_DECOMPOSITION_HPP__
#define __BASE_MATRIX_LU_DECOMPOSITION_HPP__

#include "base_matrix_macros.hpp"

#include "base_math.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"

#include <cstddef>

namespace Base {
namespace Matrix {

constexpr double DEFAULT_DIVISION_MIN_LU_DECOMPOSITION = 1.0e-10;

template <typename T, std::size_t M> class LUDecomposition {
public:
  LUDecomposition()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_LU_DECOMPOSITION)) {}

  LUDecomposition(const DiagMatrix<T, M> &matrix)
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_LU_DECOMPOSITION)) {
    this->_Lower = Matrix<T, M, M>::identity();
    this->_Upper = Base::Matrix::output_dense_matrix(matrix);
  }

  /* Copy Constructor */
  LUDecomposition(const LUDecomposition<T, M> &other)
      : _Lower(other._Lower), _Upper(other._Upper),
        _pivot_index_vec(other._pivot_index_vec),
        division_min(other.division_min) {}

  LUDecomposition<T, M> &operator=(const LUDecomposition<T, M> &other) {
    if (this != &other) {
      this->division_min = other.division_min;
      this->_Lower = other._Lower;
      this->_Upper = other._Upper;
      this->_pivot_index_vec = other._pivot_index_vec;
    }
    return *this;
  }

  /* Move Constructor */
  LUDecomposition(LUDecomposition<T, M> &&other) noexcept
      : _Lower(std::move(other._Lower)), _Upper(std::move(other._Upper)),
        _pivot_index_vec(std::move(other._pivot_index_vec)),
        division_min(std::move(other.division_min)) {}

  LUDecomposition<T, M> &operator=(LUDecomposition<T, M> &&other) noexcept {
    if (this != &other) {
      this->division_min = std::move(other.division_min);
      this->_Lower = std::move(other._Lower);
      this->_Upper = std::move(other._Upper);
      this->_pivot_index_vec = std::move(other._pivot_index_vec);
    }
    return *this;
  }

  /* Function */
  inline Matrix<T, M, M> get_L() const { return _Lower; }

  inline Matrix<T, M, M> get_U() const { return _Upper; }

  inline Vector<T, M> solve(const Matrix<T, M, M> &A, const Vector<T, M> &b) {
    this->_decompose(A);

    Vector<T, M> b_p;
    for (std::size_t i = 0; i < M; i++) {
      b_p[i] = b[this->_pivot_index_vec[i]];
    }

    Vector<T, M> y = this->_forward_substitution(b_p);
    return this->_backward_substitution(y);
  }

  inline void solve(const Matrix<T, M, M> &A) { this->_decompose(A); }

  inline T get_determinant() const {
    T det = static_cast<T>(1);

    for (std::size_t i = 0; i < M; i++) {
      det *= this->_Lower(i, i) * this->_Upper(i, i);
    }

    return det;
  }

public:
  /* Variable */
  T division_min;

private:
  /* Variable */
  Matrix<T, M, M> _Lower;
  Matrix<T, M, M> _Upper;
  Vector<std::size_t, M> _pivot_index_vec;

  /* Function */
  inline void _decompose(const Matrix<T, M, M> &matrix) {
    this->_Lower = Matrix<T, M, M>();
    this->_Upper = matrix;

    for (std::size_t i = 0; i < M; ++i) {
      this->_pivot_index_vec[i] = i;
    }

    for (std::size_t i = 0; i < M; ++i) {
      this->_Lower(i, i) = 1;

      // Pivoting
      if (Base::Utility::near_zero(this->_Upper(i, i), this->division_min)) {
        std::size_t maxRow = i;
        T maxVal = Base::Math::abs(this->_Upper(i, i));
        for (std::size_t k = i + 1; k < M; ++k) {
          T absVal = Base::Math::abs(this->_Upper(k, i));
          if (absVal > maxVal) {
            maxVal = absVal;
            maxRow = k;
          }
        }
        if (maxRow != i) {
          Base::Utility::swap_value(this->_pivot_index_vec[i],
                                    this->_pivot_index_vec[maxRow]);
          Base::Matrix::matrix_col_swap(i, maxRow, this->_Upper);
          Base::Matrix::matrix_col_swap(i, maxRow, this->_Lower);
        }
      }

      for (std::size_t j = i + 1; j < M; ++j) {
        T factor = this->_Upper(j, i) /
                   Base::Utility::avoid_zero_divide(this->_Upper(i, i),
                                                    this->division_min);
        this->_Lower(j, i) = factor;
        for (std::size_t k = i; k < M; ++k) {
          this->_Upper(j, k) -= factor * this->_Upper(i, k);
        }
      }
    }
  }

  inline Vector<T, M> _forward_substitution(const Vector<T, M> &b) const {
    Vector<T, M> y;
    for (std::size_t i = 0; i < M; ++i) {
      T sum = b[i];
      for (std::size_t j = 0; j < i; ++j) {
        sum -= this->_Lower(i, j) * y[j];
      }
      y[i] = sum;
    }
    return y;
  }

  inline Vector<T, M> _backward_substitution(const Vector<T, M> &y) const {
    Vector<T, M> x;
    for (std::size_t i = M; i-- > 0;) {
      T sum = y[i];
      for (std::size_t j = i + 1; j < M; ++j) {
        sum -= this->_Upper(i, j) * x[j];
      }
      x[i] = sum / Base::Utility::avoid_zero_divide(this->_Upper(i, i),
                                                    this->division_min);
    }
    return x;
  }
};

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_LU_DECOMPOSITION_HPP__
