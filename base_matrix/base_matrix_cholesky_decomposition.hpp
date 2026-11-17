/**
 * @file base_matrix_cholesky_decomposition.hpp
 * @brief Provides Cholesky and SPD LDLT factorizations with fixed storage.
 *
 * This header defines a set of template functions within the Base::Matrix
 * namespace for performing Cholesky decomposition on different matrix types,
 * including dense matrices, diagonal matrices, and compiled sparse matrices.
 * The decomposition is implemented for fixed-size matrices and supports error
 * handling for non-positive-definite matrices.
 *
 * Compatibility functions return a supplied fallback on failure. Output-
 * reference functions instead report failure and may retain a partial factor.
 * PackedCholesky and SPDLDLT retain compact factors for repeated right-hand
 * sides and expose validity separately. Symmetry and finite inputs are caller
 * preconditions; square roots use the configured Base::Math implementation.
 */
#ifndef BASE_MATRIX_CHOLESKY_DECOMPOSITION_HPP_
#define BASE_MATRIX_CHOLESKY_DECOMPOSITION_HPP_

#include "base_matrix_macros.hpp"

#include "base_math.hpp"
#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"

#include <array>
#include <cstddef>
#include <tuple>

namespace Base {
namespace Matrix {

constexpr double DEFAULT_DIVISION_MIN_CHOLESKY = 1.0e-10;
// Zero disables the scale-relative contribution to the pivot floor.
constexpr double DEFAULT_RELATIVE_PIVOT_MIN_CHOLESKY = 0.0;
// SPD LDLT divides directly below this magnitude, independently of pivot
// limits.
constexpr double RECIPROCAL_MIN_CHOLESKY = 1.0e-30;
// Validate the shared float/double setting using the narrower type.
static_assert(static_cast<float>(RECIPROCAL_MIN_CHOLESKY) > 0.0f &&
                  1.0f / static_cast<float>(RECIPROCAL_MIN_CHOLESKY) > 0.0f,
              "The Cholesky reciprocal threshold must have a positive, "
              "representable float reciprocal.");

namespace CholeskyOperation {

/**
 * @brief Stores an upper triangle in contiguous packed column order.
 *
 * Only M*(M+1)/2 elements are allocated. Access is unchecked, and callers must
 * provide indices satisfying row <= column < M.
 *
 * @tparam T The stored element type.
 * @tparam M The number of rows and columns in the represented matrix.
 */
template <typename T, std::size_t M> class PackedUpper {
public:
  /**
   * @brief Accesses a mutable element in the packed upper triangle.
   * @param row The row index, no greater than column.
   * @param column The column index, less than M.
   * @return A reference to the stored element.
   */
  inline T &operator()(std::size_t row, std::size_t column) {
    return this->data_[column * (column + 1) / 2 + row];
  }
  /**
   * @brief Accesses an element in the packed upper triangle without
   * modification.
   * @param row The row index, no greater than column.
   * @param column The column index, less than M.
   * @return A constant reference to the stored element.
   */
  inline const T &operator()(std::size_t row, std::size_t column) const {
    return this->data_[column * (column + 1) / 2 + row];
  }

protected:
  std::array<T, M *(M + 1) / 2> data_{};
};

/**
 * @brief Computes an upper Cholesky factor by right-looking updates in place.
 *
 * The workspace already contains the upper triangle of the symmetric input.
 * A failed pivot masks the remaining fixed pivot slots.
 * Reciprocal square roots follow the configured Base::Math implementation.
 * Its default float-based seed requires positive normal float pivots, even
 * when T is double. No nonfinite checks run.
 *
 * @tparam T The real floating-point element type.
 * @tparam M The number of rows and columns.
 * @tparam Work The workspace type with mutable upper-triangle element access.
 * @param[in,out] Y The initialized triangle, replaced by the factor or a
 * partial factor on failure.
 * @param division_min The positive absolute pivot floor and rsqrt protection.
 * @param relative_pivot_min The nonnegative multiplier of the largest initial
 * diagonal magnitude, added to the pivot floor.
 * @return true if a pivot is nonpositive or below the combined floor.
 */
template <typename T, std::size_t M, typename Work>
inline bool factor(Work &Y, T division_min, T relative_pivot_min) {
  T scale = static_cast<T>(0);
  for (std::size_t i = 0; i < M; ++i) {
    T diagonal = Base::Math::abs(Y(i, i));
    if (diagonal > scale) {
      scale = diagonal;
    }
  }
  T pivot_floor = division_min + relative_pivot_min * scale;
  bool failed = false;
  for (std::size_t k = 0; k < M; ++k) {
    if (!failed) {
      T pivot = Y(k, k);
      if (pivot < pivot_floor || pivot <= static_cast<T>(0)) {
        failed = true;
      } else {
        T inverse_root = Base::Math::rsqrt<T>(pivot, division_min);
        // inverse_root is positive and nonzero in the documented input range.
        // Do not clamp it by the input-domain floor (its units are different).
        Y(k, k) = static_cast<T>(1) / inverse_root;
        for (std::size_t j = k + 1; j < M; ++j) {
          Y(k, j) *= inverse_root;
        }
        for (std::size_t j = k + 1; j < M; ++j) {
          T row_value = Y(k, j);
          for (std::size_t i = k + 1; i <= j; ++i) {
            Y(i, j) -= Y(k, i) * row_value;
          }
        }
      }
    }
  }
  return failed;
}

/**
 * @brief Computes an upper Cholesky factor using in-place inner products.
 *
 * The unprocessed upper triangle still stores
 * A; processed rows store the factor. This avoids the dense input copy while
 * retaining the faster small-matrix traversal found in host measurements.
 * The input must satisfy the same rsqrt range requirements as factor().
 *
 * @tparam T The real floating-point element type.
 * @tparam M The number of rows and columns.
 * @tparam Work The workspace type with mutable upper-triangle element access.
 * @param[in,out] Y The input upper triangle, overwritten by a full or partial
 * factor.
 * @param division_min The positive absolute pivot floor and rsqrt protection.
 * @return true if a pivot is nonpositive or below division_min.
 */
template <typename T, std::size_t M, typename Work>
inline bool factor_inner(Work &Y, T division_min) {
  bool failed = false;
  for (std::size_t k = 0; k < M; ++k) {
    if (!failed) {
      T pivot = Y(k, k);
      for (std::size_t i = 0; i < k; ++i) {
        pivot -= Y(i, k) * Y(i, k);
      }
      if (pivot < division_min || pivot <= static_cast<T>(0)) {
        failed = true;
      } else {
        T inverse_root = Base::Math::rsqrt<T>(pivot, division_min);
        Y(k, k) = static_cast<T>(1) / inverse_root;
        for (std::size_t j = k + 1; j < M; ++j) {
          T value = Y(k, j);
          for (std::size_t i = 0; i < k; ++i) {
            value -= Y(i, k) * Y(i, j);
          }
          Y(k, j) = value * inverse_root;
        }
      }
    }
  }
  return failed;
}

/**
 * @brief Replaces a nonpositive Cholesky threshold with its default value.
 * @tparam T The real floating-point type of the threshold.
 * @param value The requested absolute pivot threshold.
 * @return value when positive, or DEFAULT_DIVISION_MIN_CHOLESKY otherwise.
 */
template <typename T> inline T positive_threshold(T value) {
  return value > static_cast<T>(0)
             ? value
             : static_cast<T>(DEFAULT_DIVISION_MIN_CHOLESKY);
}

} // namespace CholeskyOperation

/**
 * @brief Writes the upper Cholesky factor into caller-owned dense storage.
 *
 * The caller owns the only dense factor workspace.
 * Returns true on failure without substituting a fallback. The output may
 * contain a partial factor; a corrected threshold can also leave a full factor.
 * Dense input and output may alias because all lower inputs are read before
 * clearing the lower output. With the default float-seeded reciprocal-square-
 * root implementation, positive pivots must fit the normal float range.
 *
 * @tparam T The real floating-point element type.
 * @tparam M The number of rows and columns.
 * @param U The symmetric input matrix; only its lower triangle is read.
 * @param[out] Y The upper factor satisfying U = Y^T Y on success, with zeroed
 * lower entries. On failure it may contain a partial factor.
 * @param division_min The positive absolute pivot threshold. A nonpositive
 * value is replaced by the default but still reported as failure.
 * @return true on an invalid threshold or a failed pivot, false on success.
 */
template <typename T, std::size_t M>
inline bool cholesky_decomposition_to(const Matrix<T, M, M> &U,
                                      Matrix<T, M, M> &Y, T division_min) {
  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      Y(i, j) = U(j, i);
    }
  }
  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t i = j + 1; i < M; ++i) {
      Y(i, j) = static_cast<T>(0);
    }
  }
  T floor = CholeskyOperation::positive_threshold(division_min);
  bool failed = CholeskyOperation::factor_inner<T, M>(Y, floor);
  return failed || !(division_min > static_cast<T>(0));
}

/**
 * @brief Factors a lower-triangle CSR input into caller-owned dense storage.
 *
 * Stored upper-triangle entries are ignored. Missing lower-triangle entries
 * are treated as zero, and the input is assumed symmetric. The square-root
 * range requirements are the same as for cholesky_decomposition_to().
 *
 * @tparam T The real floating-point element type.
 * @tparam M The number of rows and columns.
 * @tparam Indices The compile-time CSR column-index sequence.
 * @tparam Pointers The compile-time CSR row-pointer sequence.
 * @param U The compiled sparse input containing the lower triangle.
 * @param[out] Y The upper factor, which may be partial on failure. Correcting
 * an invalid threshold can leave a full factor while still reporting failure.
 * @param division_min The positive absolute pivot threshold. Nonpositive
 * values are corrected and reported as failure.
 * @return true on an invalid threshold or a failed pivot, false on success.
 */
template <typename T, std::size_t M, typename Indices, typename Pointers>
inline bool cholesky_decomposition_sparse_to(
    const CompiledSparseMatrix<T, M, M, Indices, Pointers> &U,
    Matrix<T, M, M> &Y, T division_min) {
  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t i = 0; i < M; ++i) {
      Y(i, j) = static_cast<T>(0);
    }
  }
  for (std::size_t row = 0; row < M; ++row) {
    for (std::size_t entry = Pointers::list[row];
         entry < Pointers::list[row + 1]; ++entry) {
      std::size_t column = Indices::list[entry];
      if (column <= row) {
        Y(column, row) = U.values[entry];
      }
    }
  }
  T floor = CholeskyOperation::positive_threshold(division_min);
  bool failed = CholeskyOperation::factor_inner<T, M>(Y, floor);
  return failed || !(division_min > static_cast<T>(0));
}

/**
 * @brief Returns a dense upper Cholesky factor and a failure flag.
 *
 * The factor satisfies U = Y^T Y, with Y_b returned on any invalid pivot.
 * Symmetry is a caller precondition. Positive pivots below division_min fail.
 * Invalid division_min is corrected to the default and reported as failure.
 *
 * @tparam T The real floating-point element type.
 * @tparam M The number of rows and columns.
 * @param U The symmetric input matrix; only its lower triangle is read.
 * @param Y_b The fallback matrix returned on failure.
 * @param division_min The requested positive absolute pivot threshold.
 * @return A tuple containing the upper factor or Y_b, followed by a flag that
 * is true on failure.
 */
template <typename T, std::size_t M>
inline std::tuple<Matrix<T, M, M>, bool>
cholesky_decomposition(const Matrix<T, M, M> &U, const Matrix<T, M, M> &Y_b,
                       const T &division_min) {
  std::tuple<Matrix<T, M, M>, bool> result;
  bool failed = cholesky_decomposition_to(U, std::get<0>(result), division_min);
  std::get<1>(result) = failed;
  if (failed) {
    std::get<0>(result) = Y_b;
  }
  return result;
}

/**
 * @brief Computes square roots of a nonnegative diagonal matrix.
 *
 * Exact zeros remain zero, preserving the positive-semidefinite allowance
 * of the diagonal API. Negative entries cause the entire fallback to be used.
 * Positive entries use Base::Math::sqrt with its configured accuracy and range.
 *
 * @tparam T The real floating-point element type.
 * @tparam M The number of diagonal elements.
 * @param U The input diagonal matrix.
 * @param Y_b The fallback returned if any entry is negative.
 * @return A tuple containing the diagonal square root or Y_b, followed by a
 * flag that is true on failure.
 */
template <typename T, std::size_t M>
inline std::tuple<DiagMatrix<T, M>, bool>
cholesky_decomposition_diag(const DiagMatrix<T, M> &U,
                            const DiagMatrix<T, M> &Y_b) {
  DiagMatrix<T, M> Y;
  bool failed = false;
  for (std::size_t i = 0; i < M; ++i) {
    if (!failed) {
      if (U[i] < static_cast<T>(0)) {
        failed = true;
      } else {
        Y[i] = U[i] == static_cast<T>(0) ? static_cast<T>(0)
                                         : Base::Math::sqrt<T>(U[i], U[i]);
      }
    }
  }
  return std::make_tuple(failed ? Y_b : Y, failed);
}

/**
 * @brief Returns a dense Cholesky factor of a lower-triangle CSR input.
 *
 * The lower sparse triangle is scattered directly into the dense workspace.
 * Symmetry and square-root range requirements match the dense API.
 *
 * @tparam T The real floating-point element type.
 * @tparam M The number of rows and columns.
 * @tparam CSRIndices_U The compile-time CSR column-index sequence.
 * @tparam CSRPointers_U The compile-time CSR row-pointer sequence.
 * @param U The sparse input containing the symmetric matrix's lower triangle.
 * @param Y_b The fallback matrix returned on failure.
 * @param division_min The requested positive absolute pivot threshold.
 * @return A tuple containing the upper factor or Y_b, followed by a flag that
 * is true on failure.
 */
template <typename T, std::size_t M, typename CSRIndices_U,
          typename CSRPointers_U>
inline std::tuple<Matrix<T, M, M>, bool> cholesky_decomposition_sparse(
    const CompiledSparseMatrix<T, M, M, CSRIndices_U, CSRPointers_U> &U,
    const Matrix<T, M, M> &Y_b, const T &division_min) {
  std::tuple<Matrix<T, M, M>, bool> result;
  bool failed =
      cholesky_decomposition_sparse_to(U, std::get<0>(result), division_min);
  std::get<1>(result) = failed;
  if (failed) {
    std::get<0>(result) = Y_b;
  }
  return result;
}

/**
 * @brief Retains a packed upper Cholesky factor for repeated linear solves.
 *
 * Dense get_U() is optional and its output costs M*M elements of type T.
 * Reciprocal square roots use Base::Math without class-specific settings.
 * Inputs must be symmetric positive definite, with pivots in the supported
 * rsqrt range. Failed factorizations are reported by is_valid().
 *
 * @tparam T The real floating-point element type.
 * @tparam M The number of rows and columns.
 */
template <typename T, std::size_t M> class PackedCholesky {
public:
  /**
   * @brief Initializes default pivot thresholds and an invalid factor state.
   */
  PackedCholesky()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_CHOLESKY)),
        relative_pivot_min(static_cast<T>(DEFAULT_RELATIVE_PIVOT_MIN_CHOLESKY)),
        valid_(false), threshold_adjusted_(false) {}

  /**
   * @brief Factors the lower triangle of a symmetric dense input.
   *
   * Invalid thresholds are corrected and recorded separately from factor
   * validity. Use is_valid() before consuming the resulting factor.
   *
   * @param A The symmetric positive-definite input; its upper triangle is
   * unused.
   */
  inline void solve(const Matrix<T, M, M> &A) {
    this->_check_thresholds();
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        this->factor_(i, j) = A(j, i);
      }
    }
    this->_factor();
  }

  /**
   * @brief Factors a symmetric sparse input using its stored lower triangle.
   *
   * The packed workspace is cleared before scattering, so a previous sparse
   * pattern cannot leave entries in the new factorization.
   *
   * @tparam Indices The compile-time CSR column-index sequence.
   * @tparam Pointers The compile-time CSR row-pointer sequence.
   * @param A The compiled sparse input containing the lower triangle.
   */
  template <typename Indices, typename Pointers>
  inline void solve(const CompiledSparseMatrix<T, M, M, Indices, Pointers> &A) {
    this->_check_thresholds();
    // A different sparse pattern on the next call must not retain old values.
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        this->factor_(i, j) = static_cast<T>(0);
      }
    }
    for (std::size_t row = 0; row < M; ++row) {
      for (std::size_t entry = Pointers::list[row];
           entry < Pointers::list[row + 1]; ++entry) {
        std::size_t column = Indices::list[entry];
        if (column <= row) {
          this->factor_(column, row) = A.values[entry];
        }
      }
    }
    this->_factor();
  }

  /**
   * @brief Expands the packed upper triangle into a dense matrix.
   * @return The stored upper factor with zero lower entries. A complete
   * factorization is available only when is_valid() is true.
   */
  inline Matrix<T, M, M> get_U() const {
    Matrix<T, M, M> result;
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        result(i, j) = this->factor_(i, j);
      }
    }
    return result;
  }

  /**
   * @brief Solves a new right-hand side using the retained Cholesky factor.
   *
   * Forward and backward substitutions use diagonals established by the last
   * successful solve(). Changing thresholds does not refactor the matrix.
   *
   * @param b The right-hand side vector.
   * @return The solution, or a zero vector if no valid factor is available.
   */
  inline Vector<T, M> solve_rhs(const Vector<T, M> &b) const {
    Vector<T, M> result;
    if (this->valid_) {
      for (std::size_t i = 0; i < M; ++i) {
        T value = b[i];
        for (std::size_t j = 0; j < i; ++j) {
          value -= this->factor_(j, i) * result[j];
        }
        result[i] = value / this->factor_(i, i);
      }
      for (std::size_t offset = 0; offset < M; ++offset) {
        std::size_t i = M - 1 - offset;
        T value = result[i];
        for (std::size_t j = i + 1; j < M; ++j) {
          value -= this->factor_(i, j) * result[j];
        }
        result[i] = value / this->factor_(i, i);
      }
    }
    return result;
  }

  /**
   * @brief Reports whether the retained factorization completed successfully.
   * @return true after a successful solve(), false initially or after failure.
   */
  inline bool is_valid() const { return this->valid_; }
  /**
   * @brief Reports threshold correction during the last factorization.
   * @return true if a nonpositive absolute or negative relative threshold was
   * replaced by its default value.
   */
  inline bool threshold_was_adjusted() const {
    return this->threshold_adjusted_;
  }

public:
  T division_min;
  T relative_pivot_min;

protected:
  CholeskyOperation::PackedUpper<T, M> factor_;
  bool valid_;
  bool threshold_adjusted_;

  /**
   * @brief Corrects invalid pivot thresholds and records whether they changed.
   */
  inline void _check_thresholds() {
    this->threshold_adjusted_ = !(this->division_min > static_cast<T>(0)) ||
                                this->relative_pivot_min < static_cast<T>(0);
    this->division_min =
        CholeskyOperation::positive_threshold(this->division_min);
    if (this->relative_pivot_min < static_cast<T>(0)) {
      this->relative_pivot_min =
          static_cast<T>(DEFAULT_RELATIVE_PIVOT_MIN_CHOLESKY);
    }
  }

  /**
   * @brief Factors the initialized packed workspace and stores its validity.
   */
  inline void _factor() {
    this->valid_ = !CholeskyOperation::factor<T, M>(
        this->factor_, this->division_min, this->relative_pivot_min);
  }
};

/**
 * @brief Retains a square-root-free SPD factorization A = L D L^T.
 *
 * The packed diagonal is D;
 * off-diagonal entries store L transposed. This is not an indefinite solver.
 * Only the input lower triangle is loaded, and symmetry is a caller
 * precondition. Small accepted pivots use direct division when their
 * reciprocals fall outside the configured reciprocal-use policy.
 *
 * @tparam T The real floating-point element type.
 * @tparam M The number of rows and columns.
 */
template <typename T, std::size_t M> class SPDLDLT {
public:
  /**
   * @brief Initializes default pivot thresholds and an invalid factor state.
   */
  SPDLDLT()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_CHOLESKY)),
        relative_pivot_min(static_cast<T>(DEFAULT_RELATIVE_PIVOT_MIN_CHOLESKY)),
        valid_(false), threshold_adjusted_(false) {}

  /**
   * @brief Computes packed LDLT factors from a symmetric dense input.
   *
   * Nonpositive pivots and pivots below the combined absolute/relative floor
   * invalidate the factor. Invalid threshold settings are corrected and
   * reported separately by threshold_was_adjusted().
   *
   * @param A The symmetric positive-definite matrix; only its lower triangle
   * is used for the factorization.
   */
  inline void solve(const Matrix<T, M, M> &A) {
    this->threshold_adjusted_ = !(this->division_min > static_cast<T>(0)) ||
                                this->relative_pivot_min < static_cast<T>(0);
    this->division_min =
        CholeskyOperation::positive_threshold(this->division_min);
    if (this->relative_pivot_min < static_cast<T>(0)) {
      this->relative_pivot_min =
          static_cast<T>(DEFAULT_RELATIVE_PIVOT_MIN_CHOLESKY);
    }
    T scale = static_cast<T>(0);
    for (std::size_t j = 0; j < M; ++j) {
      T diagonal = Base::Math::abs(A(j, j));
      if (diagonal > scale) {
        scale = diagonal;
      }
      for (std::size_t i = 0; i <= j; ++i) {
        this->factor_(i, j) = A(j, i);
      }
    }
    T floor = this->division_min + this->relative_pivot_min * scale;
    this->valid_ = true;
    for (std::size_t k = 0; k < M; ++k) {
      if (this->valid_) {
        T diagonal = this->factor_(k, k);
        if (diagonal < floor || diagonal <= static_cast<T>(0)) {
          this->valid_ = false;
        } else {
          bool use_reciprocal =
              diagonal >= static_cast<T>(RECIPROCAL_MIN_CHOLESKY);
          T inverse_diagonal =
              use_reciprocal ? static_cast<T>(1) / diagonal : static_cast<T>(0);
          for (std::size_t j = k + 1; j < M; ++j) {
            if (use_reciprocal) {
              this->factor_(k, j) *= inverse_diagonal;
            } else {
              this->factor_(k, j) /= diagonal;
            }
          }
          for (std::size_t j = k + 1; j < M; ++j) {
            T weighted = diagonal * this->factor_(k, j);
            for (std::size_t i = k + 1; i <= j; ++i) {
              this->factor_(i, j) -= this->factor_(k, i) * weighted;
            }
          }
        }
      }
    }
  }

  /**
   * @brief Expands the stored multipliers into a unit lower-triangular matrix.
   * @return The stored L factor. Use is_valid() to check its completeness.
   */
  inline Matrix<T, M, M> get_L() const {
    Matrix<T, M, M> result = Matrix<T, M, M>::identity();
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = j + 1; i < M; ++i) {
        result(i, j) = this->factor_(j, i);
      }
    }
    return result;
  }

  /**
   * @brief Extracts the diagonal factor from packed storage.
   * @return The stored D factor. Use is_valid() to check its completeness.
   */
  inline DiagMatrix<T, M> get_D() const {
    DiagMatrix<T, M> result;
    for (std::size_t i = 0; i < M; ++i) {
      result[i] = this->factor_(i, i);
    }
    return result;
  }

  /**
   * @brief Solves a right-hand side using the retained L, D, and L^T factors.
   * @param b The right-hand side vector.
   * @return The solution, or a zero vector if no valid factor is available.
   */
  inline Vector<T, M> solve_rhs(const Vector<T, M> &b) const {
    Vector<T, M> result;
    if (this->valid_) {
      for (std::size_t i = 0; i < M; ++i) {
        T value = b[i];
        for (std::size_t j = 0; j < i; ++j) {
          value -= this->factor_(j, i) * result[j];
        }
        result[i] = value;
      }
      for (std::size_t i = 0; i < M; ++i) {
        result[i] /= this->factor_(i, i);
      }
      for (std::size_t offset = 0; offset < M; ++offset) {
        std::size_t i = M - 1 - offset;
        for (std::size_t j = i + 1; j < M; ++j) {
          result[i] -= this->factor_(i, j) * result[j];
        }
      }
    }
    return result;
  }

  /**
   * @brief Reports whether the retained LDLT factorization is valid.
   * @return true after a successful solve(), false initially or after failure.
   */
  inline bool is_valid() const { return this->valid_; }
  /**
   * @brief Reports threshold correction during the last factorization.
   * @return true if a nonpositive absolute or negative relative threshold was
   * replaced by its default value.
   */
  inline bool threshold_was_adjusted() const {
    return this->threshold_adjusted_;
  }

public:
  T division_min;
  T relative_pivot_min;

protected:
  CholeskyOperation::PackedUpper<T, M> factor_;
  bool valid_;
  bool threshold_adjusted_;
};

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_CHOLESKY_DECOMPOSITION_HPP_
