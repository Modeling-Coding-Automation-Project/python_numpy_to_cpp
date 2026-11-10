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
 * tparam M is the number of rows in the matrix.
 * tparam N is the number of columns in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef BASE_MATRIX_QR_DECOMPOSITION_HPP_
#define BASE_MATRIX_QR_DECOMPOSITION_HPP_

#include "base_matrix_macros.hpp"

#include "base_math.hpp"
#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_vector.hpp"

#include <cstddef>
#include <tuple>

namespace Base {
namespace Matrix {

constexpr double DEFAULT_DIVISION_MIN_QR = 1.0e-10;

/* Givens Rotation */

namespace QROperation {

/** @brief Correct an invalid finite denominator threshold at an API boundary.
 */
template <typename T> inline T division_threshold(T value) {
  return value > static_cast<T>(0) ? value
                                   : static_cast<T>(DEFAULT_DIVISION_MIN_QR);
}

/**
 * @brief Apply a plane rotation without copying either factor.
 *
 * The ratio is formed with the larger nonzero component as denominator.
 * Thus the reciprocal square root receives a value in [1, 2], independent
 * of the magnitude of the original column. The signs match the legacy API.
 */
template <typename T, std::size_t M, std::size_t N>
inline void givens_rotation(std::size_t i, std::size_t j, Matrix<T, M, M> &Q,
                            Matrix<T, M, N> &R) {
  const T f = R(j, j);
  const T g = R(i, j);
  if (g == static_cast<T>(0)) {
    return;
  }

  T c;
  T s;
  if (Base::Math::abs(g) > Base::Math::abs(f)) {
    const T ratio = -f / g; // g is nonzero and has the larger magnitude.
    s = Base::Math::rsqrt_newton_method<T, 2>(static_cast<T>(1) + ratio * ratio,
                                              static_cast<T>(1));
    c = s * ratio;
  } else {
    const T ratio = -g / f; // |f| >= |g| > 0.
    c = Base::Math::rsqrt_newton_method<T, 2>(static_cast<T>(1) + ratio * ratio,
                                              static_cast<T>(1));
    s = c * ratio;
  }

  for (std::size_t column = j; column < N; ++column) {
    const T x = R(j, column);
    const T y = R(i, column);
    R(j, column) = c * x - s * y;
    R(i, column) = s * x + c * y;
  }
  R(i, j) = static_cast<T>(0);

  for (std::size_t row = 0; row < M; ++row) {
    const T x = Q(row, j);
    const T y = Q(row, i);
    Q(row, j) = c * x - s * y;
    Q(row, i) = s * x + c * y;
  }
}

} // namespace QROperation

/** @brief Compatibility value-returning wrapper around the in-place core. */
template <typename T, std::size_t M, std::size_t N>
static inline std::tuple<Matrix<T, M, M>, Matrix<T, M, N>>
qr_givensRotation(std::size_t i, std::size_t j, Matrix<T, M, M> Q_matrix,
                  Matrix<T, M, N> R_matrix, T division_min) {
  static_cast<void>(division_min);
  QROperation::givens_rotation(i, j, Q_matrix, R_matrix);
  return std::make_tuple(std::move(Q_matrix), std::move(R_matrix));
}

/**
 * @brief QR decomposition by in-place Givens rotations, A = Q R.
 *
 * Full Q and R are retained for compatibility. Use QRDecompositionHouseholder
 * when only the factorization or Q-transpose application is needed.
 */
template <typename T, std::size_t M, std::size_t N> class QRDecomposition {
public:
  static_assert(M >= N && N > 0, "Incompatible matrix dimensions");

  QRDecomposition()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_QR)), Q_matrix_(),
        R_matrix_(), rank_deficient_(false), threshold_corrected_(false) {
    for (std::size_t diagonal = 0; diagonal < M; ++diagonal) {
      this->Q_matrix_(diagonal, diagonal) = static_cast<T>(1);
    }
  }

  QRDecomposition(const QRDecomposition &) = default;
  QRDecomposition &operator=(const QRDecomposition &) = default;
  QRDecomposition(QRDecomposition &&) noexcept = default;
  QRDecomposition &operator=(QRDecomposition &&) noexcept = default;

  /** @brief Reset both factors and decompose a general dense matrix. */
  inline void solve(const Matrix<T, M, N> &A) {
    this->_initialize(A);
    this->_decompose();
  }

  /**
   * @brief Decompose an upper Hessenberg matrix using adjacent rotations.
   *
   * The caller must guarantee A(row, column) == 0 for row > column + 1.
   * Use solve() for matrices without this structure. This specialization
   * omits the general lower-triangle scan; the factor formats are unchanged.
   */
  inline void solve_hessenberg(const Matrix<T, M, N> &A) {
    this->_initialize(A);
    for (std::size_t column = 0; column < N; ++column) {
      if (column + 1 < M) {
        this->_eliminate(column + 1, column);
      }
    }
    this->_check_rank();
  }

  inline Matrix<T, M, M> get_Q() const { return this->Q_matrix_; }
  inline Matrix<T, M, N> get_R() const { return this->R_matrix_; }
  inline void get_Q(Matrix<T, M, M> &Q) const { Q = this->Q_matrix_; }
  inline void get_R(Matrix<T, M, N> &R) const { R = this->R_matrix_; }
  inline bool get_rank_deficient() const { return this->rank_deficient_; }
  inline bool get_threshold_corrected() const {
    return this->threshold_corrected_;
  }

  T division_min;

protected:
  Matrix<T, M, M> Q_matrix_;
  Matrix<T, M, N> R_matrix_;
  bool rank_deficient_;
  bool threshold_corrected_;

  inline void _initialize(const Matrix<T, M, N> &A) {
    this->threshold_corrected_ = !(this->division_min > static_cast<T>(0));
    this->division_min = QROperation::division_threshold(this->division_min);
    for (std::size_t column = 0; column < M; ++column) {
      for (std::size_t row = 0; row < M; ++row) {
        this->Q_matrix_(row, column) =
            row == column ? static_cast<T>(1) : static_cast<T>(0);
      }
    }
    this->R_matrix_ = A;
    this->rank_deficient_ = false;
  }

  inline void _eliminate(std::size_t row, std::size_t column) {
    // Tiny entries retain the legacy user-controlled approximation policy.
    if (Base::Math::abs(this->R_matrix_(row, column)) > this->division_min) {
      this->_givensRotation(row, column);
    } else {
      this->R_matrix_(row, column) = static_cast<T>(0);
    }
  }

  inline void _check_rank() {
    for (std::size_t column = 0; column < N; ++column) {
      if (Base::Math::abs(this->R_matrix_(column, column)) <=
          this->division_min) {
        this->rank_deficient_ = true;
      }
    }
  }

  inline void _decompose() {
    for (std::size_t column = 0; column < N; ++column) {
      for (std::size_t row = column + 1; row < M; ++row) {
        this->_eliminate(row, column);
      }
    }
    this->_check_rank();
  }

  inline void _givensRotation(std::size_t i, std::size_t j) {
    QROperation::givens_rotation(i, j, this->Q_matrix_, this->R_matrix_);
  }
};

/**
 * @brief Compact real Householder QR, with optional explicit Q generation.
 *
 * The upper triangle stores R; the lower triangle stores reflector tails,
 * with an implicit leading one and a separate tau vector. No full Q is kept.
 * SqrtIterations selects the existing math library's fixed Newton budget.
 * Inputs must be finite and scaled so the resulting factors are representable.
 */
template <typename T, std::size_t M, std::size_t N,
          std::size_t SqrtIterations = 2>
class QRDecompositionHouseholder {
public:
  static_assert(M >= N && N > 0, "Incompatible matrix dimensions");

  QRDecompositionHouseholder()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_QR)), factors_(),
        tau_(), rank_deficient_(false), threshold_corrected_(false) {}

  /** @brief Factor A without materializing any reflection matrix or Q. */
  inline void solve(const Matrix<T, M, N> &A) {
    this->threshold_corrected_ = !(this->division_min > static_cast<T>(0));
    this->division_min = QROperation::division_threshold(this->division_min);
    this->factors_ = A;
    this->rank_deficient_ = false;

    for (std::size_t column = 0; column < N; ++column) {
      T scale = static_cast<T>(0);
      for (std::size_t row = column; row < M; ++row) {
        const T magnitude = Base::Math::abs(this->factors_(row, column));
        if (magnitude > scale) {
          scale = magnitude;
        }
      }
      this->tau_[column] = static_cast<T>(0);
      if (scale <= this->division_min) {
        this->rank_deficient_ = true;
        for (std::size_t row = column + 1; row < M; ++row) {
          this->factors_(row, column) = static_cast<T>(0);
        }
      } else {
        T sum = static_cast<T>(0);
        for (std::size_t row = column; row < M; ++row) {
          const T value = this->factors_(row, column) / scale;
          sum += value * value;
        }
        // 1 <= sum <= M: the float-based initial estimate stays in range.
        const T norm = Base::Math::sqrt_newton_method<T, SqrtIterations>(
            sum, static_cast<T>(1));
        const T alpha = this->factors_(column, column) / scale;
        const T beta = alpha >= static_cast<T>(0) ? -norm : norm;
        const T denominator = alpha - beta;
        // beta != 0, and choosing the opposite sign avoids cancellation.
        this->tau_[column] = (beta - alpha) / beta;
        for (std::size_t row = column + 1; row < M; ++row) {
          this->factors_(row, column) =
              (this->factors_(row, column) / scale) / denominator;
        }
        this->factors_(column, column) = beta * scale;

        for (std::size_t target = column + 1; target < N; ++target) {
          T dot = this->factors_(column, target);
          for (std::size_t row = column + 1; row < M; ++row) {
            dot += this->factors_(row, column) * this->factors_(row, target);
          }
          dot *= this->tau_[column];
          this->factors_(column, target) -= dot;
          for (std::size_t row = column + 1; row < M; ++row) {
            this->factors_(row, target) -= this->factors_(row, column) * dot;
          }
        }
      }
    }
  }

  /** @brief Expand R, omitting the stored reflector tails. */
  inline void get_R(Matrix<T, M, N> &R) const {
    for (std::size_t column = 0; column < N; ++column) {
      for (std::size_t row = 0; row < M; ++row) {
        R(row, column) =
            row <= column ? this->factors_(row, column) : static_cast<T>(0);
      }
    }
  }

  inline Matrix<T, M, N> get_R() const {
    Matrix<T, M, N> R;
    this->get_R(R);
    return R;
  }

  /** @brief Generate Q on demand by applying reflectors in reverse order. */
  inline void get_Q(Matrix<T, M, M> &Q) const {
    for (std::size_t column = 0; column < M; ++column) {
      for (std::size_t row = 0; row < M; ++row) {
        Q(row, column) = row == column ? static_cast<T>(1) : static_cast<T>(0);
      }
    }
    for (std::size_t step = 0; step < N; ++step) {
      const std::size_t column = N - 1 - step;
      for (std::size_t target = column; target < M; ++target) {
        T dot = Q(column, target);
        for (std::size_t row = column + 1; row < M; ++row) {
          dot += this->factors_(row, column) * Q(row, target);
        }
        dot *= this->tau_[column];
        Q(column, target) -= dot;
        for (std::size_t row = column + 1; row < M; ++row) {
          Q(row, target) -= this->factors_(row, column) * dot;
        }
      }
    }
  }

  inline Matrix<T, M, M> get_Q() const {
    Matrix<T, M, M> Q;
    this->get_Q(Q);
    return Q;
  }

  /** @brief Apply Q transpose in place, using no dense Q or extra vector. */
  inline void apply_Q_transpose_in_place(Vector<T, M> &b) const {
    for (std::size_t column = 0; column < N; ++column) {
      T dot = b[column];
      for (std::size_t row = column + 1; row < M; ++row) {
        dot += this->factors_(row, column) * b[row];
      }
      dot *= this->tau_[column];
      b[column] -= dot;
      for (std::size_t row = column + 1; row < M; ++row) {
        b[row] -= this->factors_(row, column) * dot;
      }
    }
  }

  inline Vector<T, M> apply_Q_transpose(const Vector<T, M> &b) const {
    Vector<T, M> result = b;
    this->apply_Q_transpose_in_place(result);
    return result;
  }

  inline bool get_rank_deficient() const { return this->rank_deficient_; }
  inline bool get_threshold_corrected() const {
    return this->threshold_corrected_;
  }

  T division_min;

protected:
  Matrix<T, M, N> factors_;
  Vector<T, N> tau_;
  bool rank_deficient_;
  bool threshold_corrected_;
};

/**
 * @brief QR Decomposition for Diagonal Matrices.
 *
 * This class provides a specialized implementation of QR decomposition for
 * diagonal matrices. It assumes that the input matrix is diagonal and performs
 * the decomposition accordingly.
 *
 * @tparam T The type of elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of columns/rows).
 */
template <typename T, std::size_t M> class QRDecompositionDiag {
public:
  /* Constructor */
  QRDecompositionDiag()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_QR)),
        Q_matrix_(DiagMatrix<T, M>::identity()), R_matrix_() {}

  /* Copy Constructor */
  QRDecompositionDiag(const QRDecompositionDiag<T, M> &other)
      : division_min(other.division_min), Q_matrix_(other.Q_matrix_),
        R_matrix_(other.R_matrix_) {}
  QRDecompositionDiag<T, M> &operator=(const QRDecompositionDiag<T, M> &other) {
    if (this != &other) {
      this->division_min = other.division_min;
      this->Q_matrix_ = other.Q_matrix_;
      this->R_matrix_ = other.R_matrix_;
    }
    return *this;
  }

  /* Move Constructor */
  QRDecompositionDiag(QRDecompositionDiag<T, M> &&other) noexcept
      : division_min(other.division_min), Q_matrix_(std::move(other.Q_matrix_)),
        R_matrix_(std::move(other.R_matrix_)) {}
  QRDecompositionDiag<T, M> &
  operator=(QRDecompositionDiag<T, M> &&other) noexcept {
    if (this != &other) {
      this->division_min = other.division_min;
      this->Q_matrix_ = std::move(other.Q_matrix_);
      this->R_matrix_ = std::move(other.R_matrix_);
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
  inline void solve(const DiagMatrix<T, M> &A) { this->R_matrix_ = A; }

  /**
   * @brief Returns the orthogonal matrix Q from the QR decomposition.
   *
   * This function returns the orthogonal matrix Q that was computed during
   * the QR decomposition process.
   *
   * @return The orthogonal matrix Q.
   */
  inline DiagMatrix<T, M> get_Q() const { return this->Q_matrix_; }

  /**
   * @brief Returns the upper triangular matrix R from the QR decomposition.
   *
   * This function returns the upper triangular matrix R that was computed
   * during the QR decomposition process.
   *
   * @return The upper triangular matrix R.
   */
  inline DiagMatrix<T, M> get_R() const { return this->R_matrix_; }

public:
  /* Variable */
  T division_min;

protected:
  /* Variable */
  DiagMatrix<T, M> Q_matrix_;
  DiagMatrix<T, M> R_matrix_;
};

/**
 * @brief Sparse-input QR with dense factors and fill-in-aware elimination.
 *
 * The original CSR pattern is used only to scatter the input. Every current
 * subdiagonal entry is considered during factorization, including fill-in.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A>
class QRDecompositionSparse : public QRDecomposition<T, M, N> {
public:
  QRDecompositionSparse() : QRDecomposition<T, M, N>() {}

  inline void
  solve(const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A) {
    this->threshold_corrected_ = !(this->division_min > static_cast<T>(0));
    this->division_min = QROperation::division_threshold(this->division_min);
    for (std::size_t column = 0; column < M; ++column) {
      for (std::size_t row = 0; row < M; ++row) {
        this->Q_matrix_(row, column) =
            row == column ? static_cast<T>(1) : static_cast<T>(0);
      }
    }
    this->rank_deficient_ = false;
    for (std::size_t column = 0; column < N; ++column) {
      for (std::size_t row = 0; row < M; ++row) {
        this->R_matrix_(row, column) = static_cast<T>(0);
      }
    }
    for (std::size_t row = 0; row < M; ++row) {
      for (std::size_t entry = CSRPointers_A::list[row];
           entry < CSRPointers_A::list[row + 1]; ++entry) {
        this->R_matrix_(row, CSRIndices_A::list[entry]) = A.values[entry];
      }
    }
    this->_decompose();
  }
};

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_QR_DECOMPOSITION_HPP_
