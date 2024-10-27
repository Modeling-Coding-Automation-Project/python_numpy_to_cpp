#ifndef BASE_MATRIX_LU_DECOMPOSITION_HPP
#define BASE_MATRIX_LU_DECOMPOSITION_HPP

#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"
#include <cmath>
#include <cstddef>

namespace Base {
namespace Matrix {

template <typename T, std::size_t M> class LUDecomposition {
public:
  LUDecomposition() {}

  LUDecomposition(const Matrix<T, M, M> &matrix, T division_min)
      : _division_min(division_min) {
    this->_decompose(matrix);
  }

  LUDecomposition(const DiagMatrix<T, M> &matrix) {
    this->_L = Matrix<T, M, M>::identity();
    this->_U = matrix.create_dense();
  }

  /* Copy Constructor */
  LUDecomposition(const LUDecomposition<T, M> &other)
      : _L(other._L), _U(other._U), _pivot_index_vec(other._pivot_index_vec),
        _division_min(other._division_min) {}

  LUDecomposition<T, M> &operator=(const LUDecomposition<T, M> &other) {
    if (this != &other) {
      this->_L = other._L;
      this->_U = other._U;
      this->_pivot_index_vec = other._pivot_index_vec;
      this->_division_min = other._division_min;
    }
    return *this;
  }

  /* Move Constructor */
  LUDecomposition(LUDecomposition<T, M> &&other) noexcept
      : _L(std::move(other._L)), _U(std::move(other._U)),
        _pivot_index_vec(std::move(other._pivot_index_vec)),
        _division_min(std::move(other._division_min)) {}

  LUDecomposition<T, M> &operator=(LUDecomposition<T, M> &&other) noexcept {
    if (this != &other) {
      this->_L = std::move(other._L);
      this->_U = std::move(other._U);
      this->_pivot_index_vec = std::move(other._pivot_index_vec);
      this->_division_min = std::move(other._division_min);
    }
    return *this;
  }

  /* Function */
  Matrix<T, M, M> get_L() const { return _L; }

  Matrix<T, M, M> get_U() const { return _U; }

  Vector<T, M> solve(const Vector<T, M> &b) const {
    Vector<T, M> b_p;
    for (std::size_t i = 0; i < M; i++) {
      b_p[i] = b[this->_pivot_index_vec[i]];
    }

    Vector<T, M> y = _forward_substitution(b_p);
    return _backward_substitution(y);
  }

  T get_determinant() const {
    T det = static_cast<T>(1);

    for (std::size_t i = 0; i < M; i++) {
      det *= this->_L(i, i) * this->_U(i, i);
    }

    return det;
  }

private:
  /* Variable */
  Matrix<T, M, M> _L;
  Matrix<T, M, M> _U;
  Vector<std::size_t, M> _pivot_index_vec;
  T _division_min;

  /* Function */
  void _decompose(const Matrix<T, M, M> &matrix) {
    this->_L = Matrix<T, M, M>();
    this->_U = matrix;

    for (std::size_t i = 0; i < M; ++i) {
      this->_pivot_index_vec[i] = i;
    }

    for (std::size_t i = 0; i < M; ++i) {
      this->_L(i, i) = 1;

      // Pivoting
      if (near_zero(this->_U(i, i), this->_division_min)) {
        std::size_t maxRow = i;
        T maxVal = std::abs(this->_U(i, i));
        for (std::size_t k = i + 1; k < M; ++k) {
          T absVal = std::abs(this->_U(k, i));
          if (absVal > maxVal) {
            maxVal = absVal;
            maxRow = k;
          }
        }
        if (maxRow != i) {
          swap_value(this->_pivot_index_vec[i], this->_pivot_index_vec[maxRow]);
          matrix_col_swap(i, maxRow, this->_U);
          matrix_col_swap(i, maxRow, this->_L);
        }
      }

      for (std::size_t j = i + 1; j < M; ++j) {
        T factor = this->_U(j, i) /
                   avoid_zero_divide(this->_U(i, i), this->_division_min);
        this->_L(j, i) = factor;
        for (std::size_t k = i; k < M; ++k) {
          this->_U(j, k) -= factor * this->_U(i, k);
        }
      }
    }
  }

  Vector<T, M> _forward_substitution(const Vector<T, M> &b) const {
    Vector<T, M> y;
    for (std::size_t i = 0; i < M; ++i) {
      T sum = b[i];
      for (std::size_t j = 0; j < i; ++j) {
        sum -= this->_L(i, j) * y[j];
      }
      y[i] = sum;
    }
    return y;
  }

  Vector<T, M> _backward_substitution(const Vector<T, M> &y) const {
    Vector<T, M> x;
    for (std::size_t i = M; i-- > 0;) {
      T sum = y[i];
      for (std::size_t j = i + 1; j < M; ++j) {
        sum -= this->_U(i, j) * x[j];
      }
      x[i] = sum / avoid_zero_divide(this->_U(i, i), this->_division_min);
    }
    return x;
  }
};

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_LU_DECOMPOSITION_HPP
