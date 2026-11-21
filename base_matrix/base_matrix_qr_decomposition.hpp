/**
 * @file base_matrix_qr_decomposition.hpp
 * @brief QR decomposition utilities for dense, diagonal, and sparse matrices.
 *
 * This header provides Givens QR with explicit Q and R factors, compact real
 * Householder QR with on-demand Q generation, and diagonal and compiled-sparse
 * input adapters. The Householder representation can apply Q transpose without
 * constructing a dense Q. All factor storage has compile-time dimensions.
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
#include "base_matrix_reciprocal.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_vector.hpp"

#include <cstddef>
#include <tuple>

namespace Base {
namespace Matrix {

constexpr double DEFAULT_DIVISION_MIN_QR = 1.0e-10;
// Lower bound of the normalized squared inputs passed to sqrt/rsqrt.
constexpr double QR_NORMALIZED_ROOT_DIVISION_MIN = 1.0;

/* Givens Rotation */

namespace QROperation {

/**
 * @brief Returns a positive threshold for QR elimination and rank checks.
 *
 * A nonpositive finite input is replaced by the QR default. Nonfinite values
 * must be checked by the caller.
 *
 * @tparam T The real scalar type used for the threshold.
 * @param value The requested threshold.
 * @return The requested positive value, or DEFAULT_DIVISION_MIN_QR.
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
 * An already-zero R(i, j) leaves both factors unchanged.
 *
 * @tparam T The real scalar type of the factors.
 * @tparam M The number of rows in R and the size of Q.
 * @tparam N The number of columns in R.
 * @param i The row to eliminate, with j < i < M.
 * @param j The pivot row and column, with j < N and j < M.
 * @param Q The accumulated orthogonal factor, updated in place.
 * @param R The working factor whose entry at (i, j) is eliminated.
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
    s = Base::Math::rsqrt<T>(static_cast<T>(1) + ratio * ratio,
                             static_cast<T>(QR_NORMALIZED_ROOT_DIVISION_MIN));
    c = s * ratio;
  } else {
    const T ratio = -g / f; // |f| >= |g| > 0.
    c = Base::Math::rsqrt<T>(static_cast<T>(1) + ratio * ratio,
                             static_cast<T>(QR_NORMALIZED_ROOT_DIVISION_MIN));
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

/**
 * @brief Returns factors updated by one Givens rotation.
 *
 * This compatibility wrapper takes copies of the factors and delegates to the
 * in-place core. Its division_min argument is retained for API compatibility;
 * the core uses a nonzero component ratio and a normalized root threshold.
 *
 * @tparam T The real scalar type of the factors.
 * @tparam M The number of rows in R_matrix and the size of Q_matrix.
 * @tparam N The number of columns in R_matrix.
 * @param i The row to eliminate, with j < i < M.
 * @param j The pivot row and column, with j < N and j < M.
 * @param Q_matrix The accumulated orthogonal factor to update.
 * @param R_matrix The working factor to update.
 * @param division_min The compatibility argument, unused by the scaled core.
 * @return The updated Q_matrix and R_matrix, in that order.
 */
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
 * The configured division_min controls small-entry elimination and the
 * diagonal-based rank-deficiency flag; inputs must be finite.
 *
 * @tparam T The real scalar type of the matrix elements.
 * @tparam M The number of matrix rows, with M >= N.
 * @tparam N The positive number of matrix columns.
 */
template <typename T, std::size_t M, std::size_t N> class QRDecomposition {
public:
  static_assert(M >= N && N > 0, "Incompatible matrix dimensions");

  /**
   * @brief Initializes identity Q, zero R, and the default QR threshold.
   */
  QRDecomposition()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_QR)), Q_matrix_(),
        R_matrix_(), rank_deficient_(false), threshold_corrected_(false) {
    for (std::size_t diagonal = 0; diagonal < M; ++diagonal) {
      this->Q_matrix_(diagonal, diagonal) = static_cast<T>(1);
    }
  }

  /* Copy Constructor */
  /** @brief Copies the factors, threshold, and status from another solver. */
  QRDecomposition(const QRDecomposition &) = default;
  /**
   * @brief Copies the factors, threshold, and status from another solver.
   * @return A reference to this solver.
   */
  QRDecomposition &operator=(const QRDecomposition &) = default;
  /* Move Constructor */
  /** @brief Moves the factors and preserves their threshold and status. */
  QRDecomposition(QRDecomposition &&) noexcept = default;
  /**
   * @brief Moves the factors and preserves their threshold and status.
   * @return A reference to this solver.
   */
  QRDecomposition &operator=(QRDecomposition &&) noexcept = default;

  /**
   * @brief Resets both factors and decomposes a general dense matrix.
   *
   * Nonpositive division_min is replaced by the default before elimination.
   * The rank-deficiency flag is updated from the final R diagonal.
   *
   * @param A The finite M-by-N input matrix to factor.
   */
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
   *
   * @param A The finite upper Hessenberg input matrix to factor.
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

  /**
   * @brief Returns a copy of the accumulated orthogonal factor.
   * @return The stored M-by-M Q factor.
   */
  inline Matrix<T, M, M> get_Q() const { return this->Q_matrix_; }
  /**
   * @brief Returns a copy of the upper trapezoidal factor.
   * @return The stored M-by-N R factor.
   */
  inline Matrix<T, M, N> get_R() const { return this->R_matrix_; }
  /**
   * @brief Copies the accumulated orthogonal factor into caller storage.
   * @param Q The destination for the stored M-by-M Q factor.
   */
  inline void get_Q(Matrix<T, M, M> &Q) const { Q = this->Q_matrix_; }
  /**
   * @brief Copies the upper trapezoidal factor into caller storage.
   * @param R The destination for the stored M-by-N R factor.
   */
  inline void get_R(Matrix<T, M, N> &R) const { R = this->R_matrix_; }
  /**
   * @brief Reports the diagonal-based rank-deficiency check.
   * @return True if a final R diagonal magnitude was at most division_min.
   */
  inline bool get_rank_deficient() const { return this->rank_deficient_; }
  /**
   * @brief Reports whether the last factorization corrected division_min.
   * @return True if a nonpositive requested threshold was replaced.
   */
  inline bool get_threshold_corrected() const {
    return this->threshold_corrected_;
  }

  T division_min;

protected:
  Matrix<T, M, M> Q_matrix_;
  Matrix<T, M, N> R_matrix_;
  bool rank_deficient_;
  bool threshold_corrected_;

  /**
   * @brief Prepares identity Q and a fresh copy of A as the working R.
   *
   * This also corrects a nonpositive threshold and resets the rank flag.
   *
   * @param A The input matrix copied into the working factor.
   */
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

  /**
   * @brief Eliminates one subdiagonal entry using the configured threshold.
   *
   * Entries above the threshold use a Givens rotation; smaller entries are
   * set to zero without rotating Q.
   *
   * @param row The row of the subdiagonal entry, with column < row < M.
   * @param column The pivot column, with column < N.
   */
  inline void _eliminate(std::size_t row, std::size_t column) {
    // Tiny entries retain the legacy user-controlled approximation policy.
    if (Base::Math::abs(this->R_matrix_(row, column)) > this->division_min) {
      this->_givensRotation(row, column);
    } else {
      this->R_matrix_(row, column) = static_cast<T>(0);
    }
  }

  /**
   * @brief Flags any R diagonal whose magnitude is at most division_min.
   *
   * This threshold check does not compute a rank-revealing pivoted QR.
   */
  inline void _check_rank() {
    for (std::size_t column = 0; column < N; ++column) {
      if (Base::Math::abs(this->R_matrix_(column, column)) <=
          this->division_min) {
        this->rank_deficient_ = true;
      }
    }
  }

  /**
   * @brief Eliminates the full subdiagonal region and checks the R diagonal.
   *
   * Q and R must already have been initialized for the current input.
   */
  inline void _decompose() {
    for (std::size_t column = 0; column < N; ++column) {
      for (std::size_t row = column + 1; row < M; ++row) {
        this->_eliminate(row, column);
      }
    }
    this->_check_rank();
  }

  /**
   * @brief Applies one plane rotation to the stored Q and R factors.
   * @param i The row to eliminate, with j < i < M.
   * @param j The pivot row and column, with j < N.
   */
  inline void _givensRotation(std::size_t i, std::size_t j) {
    QROperation::givens_rotation(i, j, this->Q_matrix_, this->R_matrix_);
  }
};

/**
 * @brief Compact real Householder QR, with optional explicit Q generation.
 *
 * The upper triangle stores R; the lower triangle stores reflector tails,
 * with an implicit leading one and a separate tau vector. No full Q is kept.
 * Square-root calculation follows the configured Base::Math implementation.
 * Inputs must be finite and scaled so the resulting factors are representable.
 *
 * @tparam T The real scalar type of the matrix elements.
 * @tparam M The number of matrix rows, with M >= N.
 * @tparam N The positive number of matrix columns and stored reflectors.
 */
template <typename T, std::size_t M, std::size_t N>
class QRDecompositionHouseholder {
public:
  static_assert(M >= N && N > 0, "Incompatible matrix dimensions");

  /**
   * @brief Initializes empty compact factors and the default QR threshold.
   */
  QRDecompositionHouseholder()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_QR)), factors_(),
        tau_(), rank_deficient_(false), threshold_corrected_(false) {}

  /* Copy Constructor */
  QRDecompositionHouseholder(const QRDecompositionHouseholder &) = default;
  QRDecompositionHouseholder &
  operator=(const QRDecompositionHouseholder &) = default;

  /* Move Constructor */
  QRDecompositionHouseholder(QRDecompositionHouseholder &&) = default;
  QRDecompositionHouseholder &
  operator=(QRDecompositionHouseholder &&) = default;

  /**
   * @brief Factors A without materializing any reflection matrix or Q.
   *
   * Each active column is scaled before its norm is formed. A column scale
   * at most division_min skips its reflector and sets the rank-deficiency
   * flag. A nonpositive threshold is replaced by the default.
   *
   * @param A The finite M-by-N input matrix copied into compact factor storage.
   */
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
        const Detail::RepeatedDivision<T> divide_by_scale(scale);
        T sum = static_cast<T>(0);
        for (std::size_t row = column; row < M; ++row) {
          const T value = divide_by_scale.apply(this->factors_(row, column));
          sum += value * value;
        }
        // 1 <= sum <= M: the float-based initial estimate stays in range.
        const T norm = Base::Math::sqrt<T>(
            sum, static_cast<T>(QR_NORMALIZED_ROOT_DIVISION_MIN));
        const T alpha = divide_by_scale.apply(this->factors_(column, column));
        const T beta = alpha >= static_cast<T>(0) ? -norm : norm;
        const Detail::RepeatedDivision<T> divide_by_denominator(alpha - beta);
        // beta != 0, and choosing the opposite sign avoids cancellation.
        this->tau_[column] = (beta - alpha) / beta;
        for (std::size_t row = column + 1; row < M; ++row) {
          this->factors_(row, column) = divide_by_denominator.apply(
              divide_by_scale.apply(this->factors_(row, column)));
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

  /**
   * @brief Expands R while omitting the stored reflector tails.
   * @param R The destination for the upper trapezoidal M-by-N factor.
   */
  inline void get_R(Matrix<T, M, N> &R) const {
    for (std::size_t column = 0; column < N; ++column) {
      for (std::size_t row = 0; row < M; ++row) {
        R(row, column) =
            row <= column ? this->factors_(row, column) : static_cast<T>(0);
      }
    }
  }

  /**
   * @brief Returns the explicit upper trapezoidal factor.
   * @return An M-by-N R matrix with zeros below the diagonal.
   */
  inline Matrix<T, M, N> get_R() const {
    Matrix<T, M, N> R;
    this->get_R(R);
    return R;
  }

  /**
   * @brief Generates Q on demand by applying reflectors in reverse order.
   * @param Q The destination, initialized to identity before applying
   * reflectors.
   */
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

  /**
   * @brief Returns an explicit Q generated from the compact reflectors.
   * @return The M-by-M orthogonal factor.
   */
  inline Matrix<T, M, M> get_Q() const {
    Matrix<T, M, M> Q;
    this->get_Q(Q);
    return Q;
  }

  /**
   * @brief Applies Q transpose in place without a dense Q or extra vector.
   *
   * Reflectors are applied in factorization order using the retained factors.
   *
   * @param b The input vector, overwritten by Q transpose times its old value.
   */
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

  /**
   * @brief Applies Q transpose to a copy of the input vector.
   * @param b The vector to transform without modifying its storage.
   * @return Q transpose multiplied by b, computed from compact reflectors.
   */
  inline Vector<T, M> apply_Q_transpose(const Vector<T, M> &b) const {
    Vector<T, M> result = b;
    this->apply_Q_transpose_in_place(result);
    return result;
  }

  /**
   * @brief Reports whether a reflector was skipped for a small column scale.
   * @return True if an active column scale was at most division_min.
   */
  inline bool get_rank_deficient() const { return this->rank_deficient_; }
  /**
   * @brief Reports whether the last factorization corrected division_min.
   * @return True if a nonpositive requested threshold was replaced.
   */
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
  /** @brief Initializes identity Q, zero R, and the compatibility threshold. */
  QRDecompositionDiag()
      : division_min(static_cast<T>(DEFAULT_DIVISION_MIN_QR)),
        Q_matrix_(DiagMatrix<T, M>::identity()), R_matrix_() {}

  /* Copy Constructor */
  /**
   * @brief Copies the diagonal factors and threshold from another solver.
   * @param other The solver whose stored state is copied.
   */
  QRDecompositionDiag(const QRDecompositionDiag<T, M> &other)
      : division_min(other.division_min), Q_matrix_(other.Q_matrix_),
        R_matrix_(other.R_matrix_) {}
  /**
   * @brief Copies the diagonal factors and threshold from another solver.
   * @param other The source solver; self-assignment preserves the current
   * state.
   * @return A reference to this solver.
   */
  QRDecompositionDiag<T, M> &operator=(const QRDecompositionDiag<T, M> &other) {
    if (this != &other) {
      this->division_min = other.division_min;
      this->Q_matrix_ = other.Q_matrix_;
      this->R_matrix_ = other.R_matrix_;
    }
    return *this;
  }

  /* Move Constructor */
  /**
   * @brief Moves the diagonal factors and copies their threshold.
   * @param other The source solver supplying the factors.
   */
  QRDecompositionDiag(QRDecompositionDiag<T, M> &&other) noexcept
      : division_min(other.division_min), Q_matrix_(std::move(other.Q_matrix_)),
        R_matrix_(std::move(other.R_matrix_)) {}
  /**
   * @brief Moves the diagonal factors and copies their threshold.
   * @param other The source solver; self-assignment preserves the current
   * state.
   * @return A reference to this solver.
   */
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
   * This function sets R to A and retains identity Q, since a diagonal input
   * is already upper triangular. No rotations or threshold checks are needed.
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
 *
 * @tparam T The real scalar type of the matrix elements.
 * @tparam M The number of matrix rows, with M >= N.
 * @tparam N The positive number of matrix columns.
 * @tparam CSRIndices_A The compile-time CSR column-index list of the input.
 * @tparam CSRPointers_A The compile-time CSR row-pointer list of the input.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A>
class QRDecompositionSparse : public QRDecomposition<T, M, N> {
public:
  /** @brief Initializes the dense factor storage through the base solver. */
  QRDecompositionSparse() : QRDecomposition<T, M, N>() {}

  /* Copy Constructor */
  QRDecompositionSparse(const QRDecompositionSparse &) = default;
  QRDecompositionSparse &operator=(const QRDecompositionSparse &) = default;

  /* Move Constructor */
  QRDecompositionSparse(QRDecompositionSparse &&) = default;
  QRDecompositionSparse &operator=(QRDecompositionSparse &&) = default;

  /**
   * @brief Scatters a sparse input into R and performs full Givens elimination.
   *
   * Q is reset to identity and all of R is cleared before scattering, so
   * absent CSR entries cannot retain values from an earlier factorization.
   * Fill-in is handled by the dense base-class elimination.
   *
   * @param A The finite sparse input with the declared compile-time CSR layout.
   */
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
