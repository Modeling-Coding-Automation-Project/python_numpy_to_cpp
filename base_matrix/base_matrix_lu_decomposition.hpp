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
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace Base {
namespace Matrix {

constexpr double DEFAULT_DIVISION_MIN_LU_DECOMPOSITION = 1.0e-10;

/**
 * @brief Compact partial-pivoted LU, with the convention P A = L U.
 *
 * L has an implicit unit diagonal. A successful solve(A) keeps its factors for
 * solve_rhs(b). Modifying A does not update those factors. CACHE_INVERSE adds M
 * reciprocals for applications that solve many right-hand sides.
 * Inputs and arithmetic results must be finite and representable in T.
 */
template <typename T, std::size_t M, bool CACHE_INVERSE = false>
class LUDecomposition {
public:
  LUDecomposition()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_LU_DECOMPOSITION)),
        factor_valid_(false), singular_(false), threshold_adjusted_(false),
        odd_permutation_(false) {}

  LUDecomposition(const DiagMatrix<T, M> &matrix) : LUDecomposition() {
    this->factor_valid_ = true;
    for (std::size_t i = 0; i < M; ++i) {
      this->pivot_index_[i] = i;
      this->LU_(i, i) = matrix[i];
      T magnitude = Base::Math::abs(matrix[i]);
      if (magnitude < this->division_min) {
        this->factor_valid_ = false;
        this->singular_ = true;
      }
      if (CACHE_INVERSE) {
        bool reciprocal_is_safe = magnitude >= this->division_min &&
                                  magnitude >= std::numeric_limits<T>::min();
        this->inverse_diagonal_[i] = reciprocal_is_safe
                                         ? static_cast<T>(1) / matrix[i]
                                         : static_cast<T>(0);
      }
    }
  }

  /* Copy and move preserve the compact factors and their validity. */
  LUDecomposition(const LUDecomposition &) = default;
  LUDecomposition &operator=(const LUDecomposition &) = default;
  LUDecomposition(LUDecomposition &&) noexcept = default;
  LUDecomposition &operator=(LUDecomposition &&) noexcept = default;

  /* Get */
  inline Matrix<T, M, M> get_L() const {
    Matrix<T, M, M> result = Matrix<T, M, M>::identity();
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = j + 1; i < M; ++i) {
        result(i, j) = this->LU_(i, j);
      }
    }
    return result;
  }

  inline Matrix<T, M, M> get_U() const {
    Matrix<T, M, M> result;
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        result(i, j) = this->LU_(i, j);
      }
    }
    return result;
  }

  inline Vector<std::size_t, M> get_pivot_indices() const {
    return this->pivot_index_;
  }

  inline Matrix<T, M, M> get_P() const {
    Matrix<T, M, M> result;
    for (std::size_t i = 0; i < M; ++i) {
      result(i, this->pivot_index_[i]) = static_cast<T>(1);
    }
    return result;
  }

  inline bool is_valid() const { return this->factor_valid_; }
  inline bool is_singular() const { return this->singular_; }
  inline bool threshold_was_adjusted() const {
    return this->threshold_adjusted_;
  }

  inline T get_determinant() const {
    T result = this->odd_permutation_ ? static_cast<T>(-1) : static_cast<T>(1);
    for (std::size_t i = 0; i < M; ++i) {
      result *= this->LU_(i, i);
    }
    return this->factor_valid_ ? result : static_cast<T>(0);
  }

  /* Solve */
  inline void solve(const Matrix<T, M, M> &A) { this->_decompose<false>(A); }

  // Row scales affect pivot comparison only; neither A nor b is rescaled.
  inline void solve_scaled(const Matrix<T, M, M> &A) {
    this->_decompose<true>(A);
  }

  inline Vector<T, M> solve(const Matrix<T, M, M> &A, const Vector<T, M> &b) {
    this->solve(A);
    return this->solve_rhs(b);
  }

  inline Vector<T, M> solve_rhs(const Vector<T, M> &b) const {
    Vector<T, M> result;
    if (this->factor_valid_) {
      for (std::size_t i = 0; i < M; ++i) {
        T value = b[this->pivot_index_[i]];
        for (std::size_t j = 0; j < i; ++j) {
          value -= this->LU_(i, j) * result[j];
        }
        result[i] = value;
      }
      for (std::size_t offset = 0; offset < M; ++offset) {
        std::size_t i = M - 1 - offset;
        T value = result[i];
        for (std::size_t j = i + 1; j < M; ++j) {
          value -= this->LU_(i, j) * result[j];
        }
        // Successful factorization guarantees a nonzero diagonal. Reapplying
        // a user-modified threshold here would change the stored factorization.
        if (CACHE_INVERSE && this->inverse_diagonal_[i] != static_cast<T>(0)) {
          result[i] = value * this->inverse_diagonal_[i];
        } else {
          result[i] = value / this->LU_(i, i);
        }
      }
    }
    return result;
  }

  /** Fixed correction budget; original A is borrowed, never copied. */
  template <typename ResidualT = T>
  inline Vector<T, M> refine(const Matrix<T, M, M> &A, const Vector<T, M> &b,
                             Vector<T, M> x, std::size_t corrections) const {
    for (std::size_t pass = 0; pass < corrections; ++pass) {
      if (this->factor_valid_) {
        Vector<T, M> residual;
        for (std::size_t i = 0; i < M; ++i) {
          ResidualT value = static_cast<ResidualT>(b[i]);
          for (std::size_t j = 0; j < M; ++j) {
            value -=
                static_cast<ResidualT>(A(i, j)) * static_cast<ResidualT>(x[j]);
          }
          residual[i] = static_cast<T>(value);
        }
        Vector<T, M> correction = this->solve_rhs(residual);
        for (std::size_t i = 0; i < M; ++i) {
          x[i] += correction[i];
        }
      }
    }
    return x;
  }

  template <typename ResidualT = T>
  inline Vector<T, M> solve_refined(const Matrix<T, M, M> &A,
                                    const Vector<T, M> &b,
                                    std::size_t corrections) {
    Vector<T, M> x = this->solve(A, b);
    return this->template refine<ResidualT>(A, b, x, corrections);
  }

public:
  T division_min;

protected:
  Matrix<T, M, M> LU_;
  Vector<std::size_t, M> pivot_index_;
  std::array<T, CACHE_INVERSE ? M : 0> inverse_diagonal_{};
  bool factor_valid_;
  bool singular_;
  bool threshold_adjusted_;
  bool odd_permutation_;

  template <bool SCALED> inline void _decompose(const Matrix<T, M, M> &A) {
    this->LU_ = A;
    this->threshold_adjusted_ = !(this->division_min > static_cast<T>(0));
    if (this->threshold_adjusted_) {
      this->division_min =
          static_cast<T>(DEFAULT_DIVISION_MIN_LU_DECOMPOSITION);
    }
    this->factor_valid_ = true;
    this->singular_ = false;
    this->odd_permutation_ = false;
    std::array<T, SCALED ? M : 0> row_scale{};
    for (std::size_t i = 0; i < M; ++i) {
      this->pivot_index_[i] = i;
      if (SCALED) {
        for (std::size_t j = 0; j < M; ++j) {
          T value = Base::Math::abs(A(i, j));
          if (value > row_scale[i]) {
            row_scale[i] = value;
          }
        }
      }
    }

    for (std::size_t k = 0; k < M; ++k) {
      if (this->factor_valid_) {
        std::size_t pivot = k;
        T best = static_cast<T>(-1);
        for (std::size_t i = k; i < M; ++i) {
          T candidate = Base::Math::abs(this->LU_(i, k));
          if (SCALED) {
            // A zero row has comparison score zero, without division.
            candidate = row_scale[i] > static_cast<T>(0)
                            ? candidate / row_scale[i]
                            : static_cast<T>(0);
          }
          if (candidate > best) {
            best = candidate;
            pivot = i;
          }
        }
        if (Base::Math::abs(this->LU_(pivot, k)) < this->division_min) {
          this->factor_valid_ = false;
          this->singular_ = true;
        } else {
          if (pivot != k) {
            // Swapping the entire compact row also moves established L entries.
            for (std::size_t j = 0; j < M; ++j) {
              Base::Utility::swap_value(this->LU_(k, j), this->LU_(pivot, j));
            }
            Base::Utility::swap_value(this->pivot_index_[k],
                                      this->pivot_index_[pivot]);
            if (SCALED) {
              Base::Utility::swap_value(row_scale[k], row_scale[pivot]);
            }
            this->odd_permutation_ = !this->odd_permutation_;
          }
          // Subnormal pivots can have finite ratios although their reciprocal
          // would overflow. In that range retain direct division instead.
          bool use_reciprocal =
              Base::Math::abs(this->LU_(k, k)) >= std::numeric_limits<T>::min();
          T inverse_pivot = use_reciprocal ? static_cast<T>(1) / this->LU_(k, k)
                                           : static_cast<T>(0);
          if (CACHE_INVERSE) {
            this->inverse_diagonal_[k] = inverse_pivot;
          }
          for (std::size_t i = k + 1; i < M; ++i) {
            if (use_reciprocal) {
              this->LU_(i, k) *= inverse_pivot;
            } else {
              this->LU_(i, k) /= this->LU_(k, k);
            }
          }
          // Column-major trailing update: the inner index is contiguous.
          for (std::size_t j = k + 1; j < M; ++j) {
            T upper = this->LU_(k, j);
            for (std::size_t i = k + 1; i < M; ++i) {
              this->LU_(i, j) -= this->LU_(i, k) * upper;
            }
          }
        }
      }
    }
  }
};

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_LU_DECOMPOSITION_HPP_
