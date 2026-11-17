/**
 * @file base_matrix_lu_decomposition.hpp
 * @brief Provides LU decomposition functionality for square matrices.
 *
 * This file defines the LUDecomposition class template within the Base::Matrix
 * namespace. The LUDecomposition class implements LU decomposition for square
 * matrices of fixed size using compact partial-pivoted factors. It supports
 * repeated right-hand sides, determinant calculation, optional reciprocal
 * caching, scaled pivot selection, and a fixed number of refinement steps.
 * Factorization and solve status are reported without exceptions.
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

namespace Base {
namespace Matrix {

constexpr double DEFAULT_DIVISION_MIN_LU_DECOMPOSITION = 1.0e-10;
// Below this magnitude, divide directly instead of caching a reciprocal.
// This does not change the pivot acceptance threshold (division_min).
constexpr double RECIPROCAL_MIN_LU_DECOMPOSITION = 1.0e-30;
// Validate the shared float/double setting using the narrower type.
static_assert(static_cast<float>(RECIPROCAL_MIN_LU_DECOMPOSITION) > 0.0f &&
                  1.0f / static_cast<float>(RECIPROCAL_MIN_LU_DECOMPOSITION) >
                      0.0f,
              "The LU reciprocal threshold must have a positive, "
              "representable float reciprocal.");

/**
 * @brief Compact partial-pivoted LU, with the convention P A = L U.
 *
 * L has an implicit unit diagonal. A successful solve(A) keeps its factors for
 * solve_rhs(b). Modifying A does not update those factors. CACHE_INVERSE adds M
 * reciprocals for applications that solve many right-hand sides.
 * Inputs and arithmetic results must be finite and representable in T.
 * A pivot below division_min invalidates the factorization, and solve_rhs()
 * returns a zero vector for invalid factors.
 *
 * @tparam T The real scalar type of the matrix elements and stored factors.
 * @tparam M The number of rows and columns in the square matrix.
 * @tparam CACHE_INVERSE Whether to retain usable U-diagonal reciprocals for
 * repeated solves; very small pivots continue to use direct division.
 */
template <typename T, std::size_t M, bool CACHE_INVERSE = false>
class LUDecomposition {
public:
  /**
   * @brief Initializes the default threshold with no valid factorization.
   */
  LUDecomposition()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_LU_DECOMPOSITION)),
        factor_valid_(false), singular_(false), threshold_adjusted_(false),
        odd_permutation_(false) {}

  /**
   * @brief Initializes compact factors directly from a diagonal matrix.
   *
   * The permutation and L are identity. Diagonal magnitudes below the default
   * division_min mark the factors invalid. When caching is enabled, usable
   * diagonal reciprocals are retained for subsequent right-hand sides.
   *
   * @param matrix The finite diagonal matrix supplying U.
   */
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
        bool reciprocal_is_safe =
            magnitude >= this->division_min &&
            magnitude >= static_cast<T>(RECIPROCAL_MIN_LU_DECOMPOSITION);
        this->inverse_diagonal_[i] = reciprocal_is_safe
                                         ? static_cast<T>(1) / matrix[i]
                                         : static_cast<T>(0);
      }
    }
  }

  /* Copy and move preserve the compact factors and their validity. */
  /** @brief Copies the factors, permutation, threshold, cache, and status. */
  LUDecomposition(const LUDecomposition &) = default;
  /**
   * @brief Copies the factors, permutation, threshold, cache, and status.
   * @return A reference to this solver.
   */
  LUDecomposition &operator=(const LUDecomposition &) = default;
  /** @brief Moves the factors and preserves their permutation and status. */
  LUDecomposition(LUDecomposition &&) noexcept = default;
  /**
   * @brief Moves the factors and preserves their permutation and status.
   * @return A reference to this solver.
   */
  LUDecomposition &operator=(LUDecomposition &&) noexcept = default;

  /* Get */
  /**
   * @brief Expands the unit lower triangular factor from compact storage.
   * @return L with an explicit unit diagonal and zero upper triangle.
   */
  inline Matrix<T, M, M> get_L() const {
    Matrix<T, M, M> result = Matrix<T, M, M>::identity();
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = j + 1; i < M; ++i) {
        result(i, j) = this->LU_(i, j);
      }
    }
    return result;
  }

  /**
   * @brief Expands the upper triangular factor from compact storage.
   * @return U with zero entries below its diagonal.
   */
  inline Matrix<T, M, M> get_U() const {
    Matrix<T, M, M> result;
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        result(i, j) = this->LU_(i, j);
      }
    }
    return result;
  }

  /**
   * @brief Returns the row permutation accumulated during factorization.
   * @return Indices mapping each factor row to its original input row.
   */
  inline Vector<std::size_t, M> get_pivot_indices() const {
    return this->pivot_index_;
  }

  /**
   * @brief Expands the stored row permutation as a matrix.
   * @return P with P(i, pivot_index[i]) = 1, so valid factors satisfy P A = L
   * U.
   */
  inline Matrix<T, M, M> get_P() const {
    Matrix<T, M, M> result;
    for (std::size_t i = 0; i < M; ++i) {
      result(i, this->pivot_index_[i]) = static_cast<T>(1);
    }
    return result;
  }

  /**
   * @brief Reports whether the retained factors can be used by solve_rhs().
   * @return True after a factorization that accepted every pivot.
   */
  inline bool is_valid() const { return this->factor_valid_; }
  /**
   * @brief Reports whether a pivot was rejected by the absolute threshold.
   * @return True if a pivot magnitude was below division_min.
   */
  inline bool is_singular() const { return this->singular_; }
  /**
   * @brief Reports whether dense factorization corrected division_min.
   * @return True if a nonpositive requested threshold was replaced.
   */
  inline bool threshold_was_adjusted() const {
    return this->threshold_adjusted_;
  }

  /**
   * @brief Computes the determinant from U and the row-swap parity.
   * @return The signed product of U's diagonal, or zero for invalid factors.
   */
  inline T get_determinant() const {
    T result = this->odd_permutation_ ? static_cast<T>(-1) : static_cast<T>(1);
    for (std::size_t i = 0; i < M; ++i) {
      result *= this->LU_(i, i);
    }
    return this->factor_valid_ ? result : static_cast<T>(0);
  }

  /* Solve */
  /**
   * @brief Factors a dense matrix using partial pivoting.
   *
   * New factors, permutation, and status replace the previous factorization.
   * A nonpositive division_min is replaced by the default before pivot checks.
   *
   * @param A The finite square input matrix to factor without modifying it.
   */
  inline void solve(const Matrix<T, M, M> &A) { this->_decompose<false>(A); }

  // Row scales affect pivot comparison only; neither A nor b is rescaled.
  /**
   * @brief Factors a dense matrix using row-scaled partial pivot selection.
   *
   * Each pivot candidate is scored relative to its input row's largest
   * magnitude. Acceptance still uses the absolute division_min threshold.
   *
   * @param A The finite square input matrix to factor without modifying it.
   */
  inline void solve_scaled(const Matrix<T, M, M> &A) {
    this->_decompose<true>(A);
  }

  /**
   * @brief Factors A and solves one right-hand side with the retained factors.
   * @param A The finite square coefficient matrix.
   * @param b The right-hand side in the original row order.
   * @return The solution of A x = b, or a zero vector if factorization failed.
   */
  inline Vector<T, M> solve(const Matrix<T, M, M> &A, const Vector<T, M> &b) {
    this->solve(A);
    return this->solve_rhs(b);
  }

  /**
   * @brief Solves a right-hand side using the current compact LU factors.
   *
   * The stored permutation is applied before forward and backward
   * substitution. Cached reciprocals are used when available; otherwise each
   * accepted diagonal is divided directly. Changing division_min after
   * factorization does not change this solve's denominators.
   *
   * @param b The right-hand side in the original row order.
   * @return The solution from retained factors, or zero if they are invalid.
   */
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

  /**
   * @brief Applies a fixed number of residual-based solution corrections.
   *
   * The original A is borrowed, never copied. Residuals are accumulated in
   * ResidualT and converted to T before solving with the retained factors.
   * The factors must correspond to A; invalid factors leave x unchanged.
   *
   * @tparam ResidualT The scalar type used to accumulate each residual.
   * @param A The original coefficient matrix matching the retained factors.
   * @param b The original right-hand side.
   * @param x The initial solution estimate, updated in a local copy.
   * @param corrections The fixed number of correction passes to perform.
   * @return The estimate after the requested correction passes.
   */
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

  /**
   * @brief Factors A, solves for b, and performs fixed-budget refinement.
   *
   * This uses partial pivoting for the initial factorization and retains the
   * same factors for every correction pass.
   *
   * @tparam ResidualT The scalar type used to accumulate refinement residuals.
   * @param A The finite square coefficient matrix.
   * @param b The right-hand side in the original row order.
   * @param corrections The fixed number of correction passes to perform.
   * @return The refined solution, or a zero vector if factorization failed.
   */
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

  /**
   * @brief Builds compact LU factors and records row permutations and status.
   *
   * Each pivot step selects and swaps a row before eliminating the trailing
   * submatrix. A rejected pivot disables the remaining fixed pivot slots.
   * Reciprocal sharing is used only above its configured magnitude limit;
   * smaller accepted pivots retain direct division.
   *
   * @tparam SCALED Whether row maxima scale the pivot-comparison scores.
   * @param A The finite square matrix copied into compact factor storage.
   */
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
          // Small pivots can have finite ratios although their reciprocal
          // would overflow. Below the configured limit, divide directly.
          bool use_reciprocal = Base::Math::abs(this->LU_(k, k)) >=
                                static_cast<T>(RECIPROCAL_MIN_LU_DECOMPOSITION);
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
