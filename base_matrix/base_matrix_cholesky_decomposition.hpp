/**
 * @file base_matrix_cholesky_decomposition.hpp
 * @brief Provides Cholesky decomposition implementations for dense, diagonal,
 * and sparse matrices.
 *
 * This header defines a set of template functions within the Base::Matrix
 * namespace for performing Cholesky decomposition on different matrix types,
 * including dense matrices, diagonal matrices, and compiled sparse matrices.
 * The decomposition is implemented for fixed-size matrices and supports error
 * handling for non-positive-definite matrices.
 *
 * All functions include error handling for non-positive-definite matrices and
 * return a fallback matrix if decomposition fails.
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
#include <limits>
#include <tuple>

namespace Base {
namespace Matrix {

constexpr double DEFAULT_DIVISION_MIN_CHOLESKY = 1.0e-10;

namespace CholeskyOperation {

/** Upper triangle, in contiguous column order. */
template <typename T, std::size_t M> class PackedUpper {
public:
  inline T &operator()(std::size_t row, std::size_t column) {
    return this->data_[column * (column + 1) / 2 + row];
  }
  inline const T &operator()(std::size_t row, std::size_t column) const {
    return this->data_[column * (column + 1) / 2 + row];
  }

protected:
  std::array<T, M *(M + 1) / 2> data_{};
};

/**
 * Right-looking factorization of an already initialized upper triangle.
 * The caller guarantees symmetry; only the input lower triangle is loaded.
 * A failed pivot masks the remaining fixed pivot slots.
 * Reciprocal square roots follow the configured Base::Math implementation.
 * Its default float-based seed requires positive normal float pivots, even
 * when T is double. No nonfinite checks run.
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
 * In-place inner-product variant. The unprocessed upper triangle still stores
 * A; processed rows store the factor. This avoids the dense input copy while
 * retaining the faster small-matrix traversal found in host measurements.
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

template <typename T> inline T positive_threshold(T value) {
  return value > static_cast<T>(0)
             ? value
             : static_cast<T>(DEFAULT_DIVISION_MIN_CHOLESKY);
}

} // namespace CholeskyOperation

/**
 * Output-reference API: the caller owns the only dense factor workspace.
 * Returns true on failure; output is then a partial factor, not a fallback.
 * Dense input and output may alias because all lower inputs are read before
 * clearing the lower output. Positive pivots must fit normal float range for
 * the configured reciprocal-square-root implementation.
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
 * Dense compatibility API. A = Y^T Y, with Y_b returned on any invalid pivot.
 * Symmetry is a caller precondition. Positive pivots below division_min fail.
 * Invalid division_min is corrected to the default and reported as failure.
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

/** Diagonal input retains its historical positive-semidefinite allowance. */
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

/** Scatter lower CSR input directly into the only dense factor workspace. */
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
 * Compact SPD factor; dense get_U() is optional and its output costs M*M T.
 * Reciprocal square roots use Base::Math without class-specific settings.
 */
template <typename T, std::size_t M> class PackedCholesky {
public:
  PackedCholesky()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_CHOLESKY)),
        relative_pivot_min(static_cast<T>(0)), valid_(false),
        threshold_adjusted_(false) {}

  inline void solve(const Matrix<T, M, M> &A) {
    this->_check_thresholds();
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        this->factor_(i, j) = A(j, i);
      }
    }
    this->_factor();
  }

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

  inline Matrix<T, M, M> get_U() const {
    Matrix<T, M, M> result;
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        result(i, j) = this->factor_(i, j);
      }
    }
    return result;
  }

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

  inline bool is_valid() const { return this->valid_; }
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

  inline void _check_thresholds() {
    this->threshold_adjusted_ = !(this->division_min > static_cast<T>(0)) ||
                                this->relative_pivot_min < static_cast<T>(0);
    this->division_min =
        CholeskyOperation::positive_threshold(this->division_min);
    if (this->relative_pivot_min < static_cast<T>(0)) {
      this->relative_pivot_min = static_cast<T>(0);
    }
  }

  inline void _factor() {
    this->valid_ = !CholeskyOperation::factor<T, M>(
        this->factor_, this->division_min, this->relative_pivot_min);
  }
};

/**
 * Square-root-free SPD factorization A = L D L^T. The packed diagonal is D;
 * off-diagonal entries store L transposed. This is not an indefinite solver.
 */
template <typename T, std::size_t M> class SPDLDLT {
public:
  SPDLDLT()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_CHOLESKY)),
        relative_pivot_min(static_cast<T>(0)), valid_(false),
        threshold_adjusted_(false) {}

  inline void solve(const Matrix<T, M, M> &A) {
    this->threshold_adjusted_ = !(this->division_min > static_cast<T>(0)) ||
                                this->relative_pivot_min < static_cast<T>(0);
    this->division_min =
        CholeskyOperation::positive_threshold(this->division_min);
    if (this->relative_pivot_min < static_cast<T>(0)) {
      this->relative_pivot_min = static_cast<T>(0);
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
          bool use_reciprocal = diagonal >= std::numeric_limits<T>::min();
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

  inline Matrix<T, M, M> get_L() const {
    Matrix<T, M, M> result = Matrix<T, M, M>::identity();
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = j + 1; i < M; ++i) {
        result(i, j) = this->factor_(j, i);
      }
    }
    return result;
  }

  inline DiagMatrix<T, M> get_D() const {
    DiagMatrix<T, M> result;
    for (std::size_t i = 0; i < M; ++i) {
      result[i] = this->factor_(i, i);
    }
    return result;
  }

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

  inline bool is_valid() const { return this->valid_; }
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
