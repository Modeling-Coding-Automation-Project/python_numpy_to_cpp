/**
 * @file base_matrix_eigen_solver.hpp
 * @brief Fixed-budget Hessenberg QR and symmetric Jacobi eigensolvers.
 *
 * The general solvers accept real input. EigenSolverComplex also represents
 * complex conjugate eigenvalue pairs. Inputs must be finite; the caller checks
 * non-finite outputs. iteration_max is the total number of QR sweep slots per
 * call, including continue_solving_eigen_values().
 */
#ifndef BASE_MATRIX_EIGEN_SOLVER_HPP_
#define BASE_MATRIX_EIGEN_SOLVER_HPP_

#include "base_matrix_diagonal.hpp"
#include "base_matrix_inverse.hpp"
#include "base_matrix_macros.hpp"
#include "base_matrix_numerical.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace Base {
namespace Matrix {

const std::size_t DEFAULT_ITERATION_MAX_EIGEN_SOLVER = 10;
const double DEFAULT_DIVISION_MIN_EIGEN_SOLVER = 1.0e-20;

// Default inverse-iteration budget and its inner GMRES settings.
constexpr std::size_t DEFAULT_ITERATION_MULTIPLIER_EIGEN_VECTOR = 3;
constexpr std::size_t DEFAULT_GMRES_CYCLES_EIGEN_SOLVER = 1;
constexpr std::size_t DEFAULT_GMRES_ORTHOGONALIZATION_PASSES_EIGEN_SOLVER = 1;
// A nonpositive decay rate selects the fixed inner GMRES tolerance below.
constexpr double DEFAULT_GMRES_DECAY_RATE_EIGEN_SOLVER = 0.0;
constexpr double DEFAULT_ABSOLUTE_TOLERANCE_EIGEN_SOLVER = 0.0;

// Independent relative thresholds shared by float and double.
constexpr double DEFAULT_RELATIVE_TOLERANCE_EIGEN_SOLVER = 1.0e-6;
constexpr double DEFAULT_EIGENVECTOR_RELATIVE_TOLERANCE = 4.0e-6;
constexpr double DEFAULT_EIGENVECTOR_RELATIVE_SHIFT = 3.0e-4;
constexpr double DEFAULT_INNER_GMRES_RELATIVE_TOLERANCE = 1.0e-3;
// Preserve the existing name for the default inverse-iteration shift.
const double EIGEN_SMALL_VALUE = DEFAULT_EIGENVECTOR_RELATIVE_SHIFT;
// Number of QR sweeps between deterministic exceptional shifts.
constexpr std::size_t EIGEN_EXCEPTIONAL_SHIFT_PERIOD = 8;

// Keep tiny inputs representable by the dependency's float-based sqrt seed.
// Input scaling and output compensation satisfy factor * compensation^2 = 1.
constexpr double EIGEN_SQRT_INPUT_MIN = 1.0e-30;
constexpr double EIGEN_SQRT_INPUT_SCALE_FACTOR = 1.0e30;
constexpr double EIGEN_SQRT_OUTPUT_SCALE_FACTOR = 1.0e-15;
// Eleven scaling slots cover IEEE float/double subnormals.
constexpr std::size_t EIGEN_SQRT_SCALE_MAX_PASSES = 11;

/**
 * @brief Describes the latest eigenvalue or eigenvector solve outcome.
 *
 * A converged status applies to the most recently completed solve stage. The
 * separate eigenvalue and eigenvector flags retain their respective results.
 */
enum class EigenSolverStatus {
  not_started,
  not_converged,
  converged,
  complex_pair,
  inner_failure,
  zero_vector
};

namespace EigenOperation {

/**
 * @brief Evaluates a square root after scaling very small arguments.
 *
 * Positive inputs are scaled through a fixed number of slots before delegating
 * to Base::Math::sqrt, and the result is compensated for that scaling. This
 * helper does not select the dependency's square-root iteration count.
 *
 * @tparam T The real floating-point type.
 * @param value The nonnegative value from a scaled eigenvalue calculation.
 * @return The approximate square root, or zero for a nonpositive input.
 */
template <typename T> inline T square_root(T value) {
  if (value <= static_cast<T>(0)) {
    return static_cast<T>(0);
  }
  T root_scale = static_cast<T>(1);
  if (value < static_cast<T>(EIGEN_SQRT_INPUT_MIN)) {
    // Multiplication starts below the scaling threshold and cannot overflow.
    for (std::size_t pass = 0; pass < EIGEN_SQRT_SCALE_MAX_PASSES; ++pass) {
      if (value < static_cast<T>(EIGEN_SQRT_INPUT_MIN)) {
        value *= static_cast<T>(EIGEN_SQRT_INPUT_SCALE_FACTOR);
        root_scale *= static_cast<T>(EIGEN_SQRT_OUTPUT_SCALE_FACTOR);
      }
    }
  }
  return root_scale *
         Base::Math::sqrt<T>(value, static_cast<T>(EIGEN_SQRT_INPUT_MIN));
}

/**
 * @brief Supplies real-valued operations for a trailing-block QR shift.
 * @tparam S The real floating-point type used for the shift.
 */
template <typename S> struct Shift {
  using T = S;
  /**
   * @brief Returns a real scalar without conversion.
   * @param value The real scalar.
   * @return The input value.
   */
  static inline T real_part(T value) { return value; }
  /**
   * @brief Evaluates a real discriminant root and identifies a complex pair.
   * @param value The real discriminant of the scaled trailing block.
   * @param complex_pair Set to true when the discriminant is negative.
   * @return The square root for a nonnegative discriminant, otherwise zero.
   */
  static inline T square_root_scalar(T value, bool &complex_pair) {
    complex_pair = value < static_cast<T>(0);
    return square_root(value);
  }
};

/**
 * @brief Supplies complex-valued operations for a trailing-block QR shift.
 * @tparam T The real floating-point component type.
 */
template <typename T> struct Shift<Complex<T>> {
  /**
   * @brief Extracts the real component of a complex scalar.
   * @param value The complex scalar.
   * @return The real component.
   */
  static inline T real_part(const Complex<T> &value) { return value.real; }
  /**
   * @brief Computes a scaled complex square root for shift selection.
   *
   * Component scaling avoids directly squaring a large or tiny input. The
   * returned root has a nonnegative real component; a negative real input
   * with zero imaginary part selects the root with positive imaginary part.
   *
   * @param value The complex discriminant of the scaled trailing block.
   * @param complex_pair Set to false because this scalar type supports the
   * complex root.
   * @return The approximate complex square root, or zero for a zero input.
   */
  static inline Complex<T> square_root_scalar(const Complex<T> &value,
                                              bool &complex_pair) {
    complex_pair = false;
    T scale = Numerical::Scalar<Complex<T>>::component_max(value);
    if (scale == static_cast<T>(0)) {
      return Complex<T>();
    }
    Complex<T> normalized = value / scale;
    T magnitude = Numerical::Scalar<Complex<T>>::magnitude(normalized);
    T dominant = square_root((magnitude + Base::Math::abs(normalized.real)) *
                             static_cast<T>(0.5));
    T root_scale = square_root(scale);
    // dominant is nonzero because normalized has a component of magnitude 1.
    T minor = normalized.imag / (static_cast<T>(2) * dominant);
    if (normalized.real >= static_cast<T>(0)) {
      return Complex<T>(root_scale * dominant, root_scale * minor);
    }
    T sign = normalized.imag < static_cast<T>(0) ? static_cast<T>(-1)
                                                 : static_cast<T>(1);
    return Complex<T>(root_scale * Base::Math::abs(minor),
                      root_scale * sign * dominant);
  }
};

/**
 * @brief Reduces a real matrix to upper Hessenberg form by similarity updates.
 *
 * Householder reflectors are applied directly from both sides without forming
 * dense reflector matrices. The result may use real or complex storage, but
 * the input and the reflectors are real.
 *
 * @tparam S The scalar type of the Hessenberg matrix.
 * @tparam T The real floating-point input type.
 * @tparam M The matrix dimension.
 * @param input The real square matrix to reduce.
 * @param result The destination for the Hessenberg approximation.
 * @param division_min The positive denominator floor for reflector scaling.
 */
template <typename S, typename T, std::size_t M>
inline void hessenberg(const Matrix<T, M, M> &input, Matrix<S, M, M> &result,
                       T division_min) {
  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t i = 0; i < M; ++i) {
      result(i, j) = S(input(i, j));
    }
  }
  Vector<T, M> reflector;
  for (std::size_t k = 0; k < M; ++k) {
    if (k + 2 < M) {
      T scale = static_cast<T>(0);
      for (std::size_t i = 0; i < M; ++i) {
        if (i > k) {
          T candidate = Base::Math::abs(Shift<S>::real_part(result(i, k)));
          if (candidate > scale) {
            scale = candidate;
          }
        }
      }
      if (scale > static_cast<T>(0)) {
        T sum = static_cast<T>(0);
        for (std::size_t i = 0; i < M; ++i) {
          reflector[i] = static_cast<T>(0);
          if (i > k) {
            reflector[i] = Shift<S>::real_part(result(i, k)) / scale;
            sum += reflector[i] * reflector[i];
          }
        }
        T length = Numerical::normalized_sqrt(sum);
        T sign = reflector[k + 1] < static_cast<T>(0) ? static_cast<T>(-1)
                                                      : static_cast<T>(1);
        reflector[k + 1] += sign * length;
        T square = static_cast<T>(0);
        for (std::size_t i = 0; i < M; ++i) {
          square += reflector[i] * reflector[i];
        }
        // The scaled reflector has a nonzero leading component.
        T tau = static_cast<T>(2) /
                Base::Utility::avoid_zero_divide(square, division_min);
        for (std::size_t j = 0; j < M; ++j) {
          S product = S();
          for (std::size_t i = 0; i < M; ++i) {
            product += result(i, j) * reflector[i];
          }
          for (std::size_t i = 0; i < M; ++i) {
            if (i > k) {
              result(i, j) -= product * (tau * reflector[i]);
            }
          }
        }
        for (std::size_t i = 0; i < M; ++i) {
          S product = S();
          for (std::size_t j = 0; j < M; ++j) {
            product += result(i, j) * reflector[j];
          }
          for (std::size_t j = 0; j < M; ++j) {
            if (j > k) {
              result(i, j) -= product * (tau * reflector[j]);
            }
          }
        }
        for (std::size_t i = 0; i < M; ++i) {
          if (i > k + 1) {
            result(i, k) = S();
          }
        }
      }
    }
  }
}

/**
 * @brief Selects a shift from the trailing two-by-two block.
 *
 * The block is scaled before evaluating its two eigenvalue candidates. The
 * candidate nearest its final diagonal entry is selected. With real storage,
 * a negative discriminant instead yields the pair's real center and sets the
 * complex-pair flag; that center is not itself an eigenvalue of the block.
 *
 * @tparam S The real or complex scalar type.
 * @tparam M The matrix dimension.
 * @param matrix The current Hessenberg matrix.
 * @param end The final block index, which must satisfy 1 <= end < M.
 * @param complex_pair Receives the unsupported-complex-pair flag for real S.
 * @return The selected shift, or zero when the trailing block is zero.
 */
template <typename S, std::size_t M>
inline S trailing_shift(const Matrix<S, M, M> &matrix, std::size_t end,
                        bool &complex_pair) {
  using T = typename Numerical::Scalar<S>::Real;
  T scale = static_cast<T>(0);
  for (std::size_t i = end - 1; i <= end; ++i) {
    for (std::size_t j = end - 1; j <= end; ++j) {
      T candidate = Numerical::Scalar<S>::component_max(matrix(i, j));
      if (candidate > scale) {
        scale = candidate;
      }
    }
  }
  if (scale == static_cast<T>(0)) {
    complex_pair = false;
    return S();
  }

  T scale_inverse = static_cast<T>(1) / scale;

  S a = matrix(end - 1, end - 1) * scale_inverse;
  S b = matrix(end - 1, end) * scale_inverse;
  S c = matrix(end, end - 1) * scale_inverse;
  S d = matrix(end, end) * scale_inverse;

  S half_difference = (a - d) * static_cast<T>(0.5);
  S root = Shift<S>::square_root_scalar(
      half_difference * half_difference + b * c, complex_pair);
  S center = (a + d) * static_cast<T>(0.5);
  S first = center + root;
  S second = center - root;
  return (Numerical::Scalar<S>::magnitude(first - d) <
                  Numerical::Scalar<S>::magnitude(second - d)
              ? first
              : second) *
         scale;
}

/**
 * @brief Applies one implicit shifted QR sweep to an active Hessenberg block.
 *
 * Givens similarities chase the bulge through the selected block. Each row
 * and column update includes its coupling to the rest of the matrix.
 *
 * @tparam S The real or complex scalar type.
 * @tparam M The matrix dimension.
 * @param matrix The Hessenberg matrix updated in place.
 * @param begin The first active index, with begin < end.
 * @param end The last active index, which must be less than M.
 * @param shift The scalar shift used to introduce the first bulge.
 */
template <typename S, std::size_t M>
inline void implicit_qr_sweep(Matrix<S, M, M> &matrix, std::size_t begin,
                              std::size_t end, const S &shift) {
  using T = typename Numerical::Scalar<S>::Real;
  for (std::size_t k = 0; k + 1 < M; ++k) {
    if (k >= begin && k < end) {
      S first = k == begin ? matrix(k, k) - shift : matrix(k, k - 1);
      S second = k == begin ? matrix(k + 1, k) : matrix(k + 1, k - 1);
      T cosine;
      S sine;
      Numerical::givens(first, second, cosine, sine);
      S sine_conjugate = Numerical::Scalar<S>::conjugate(sine);
      for (std::size_t j = 0; j < M; ++j) {
        S upper = matrix(k, j);
        S lower = matrix(k + 1, j);
        matrix(k, j) = upper * cosine + sine * lower;
        matrix(k + 1, j) = -sine_conjugate * upper + lower * cosine;
      }
      for (std::size_t i = 0; i < M; ++i) {
        S left = matrix(i, k);
        S right = matrix(i, k + 1);
        matrix(i, k) = left * cosine + right * sine_conjugate;
        matrix(i, k + 1) = -left * sine + right * cosine;
      }
      if (k > begin) {
        matrix(k + 1, k - 1) = S();
      }
    }
  }
}

/**
 * @brief Shares bounded QR and inverse-iteration logic between general solvers.
 *
 * Real input is reduced to Hessenberg form, followed by shifted QR sweeps.
 * Eigenvectors use the stored eigenvalue estimates and bounded inner GMRES
 * solves. A completed call may remain unconverged; status and residual getters
 * expose that outcome. Eigenvalues and their vector columns are not sorted.
 *
 * @tparam S The real or complex scalar type used for the results.
 * @tparam T The real floating-point input and tolerance type.
 * @tparam M The positive matrix dimension.
 * @tparam InnerK The fixed inner GMRES capacity, satisfying 1 <= InnerK <= M.
 */
template <typename S, typename T, std::size_t M, std::size_t InnerK = M>
class Solver {
public:
  static_assert(M > 0, "An eigensolver requires a nonempty matrix.");
  static_assert(InnerK > 0 && InnerK <= M, "Inner GMRES requires 1 <= K <= M.");
#ifdef BASE_MATRIX_USE_STD_VECTOR_
  using EigenValues_Type = std::vector<S>;
#else
  using EigenValues_Type = std::array<S, M>;
#endif

  /**
   * @brief Initializes iteration budgets, tolerances, and unsolved state.
   *
   * Eigenvalue storage is fixed unless BASE_MATRIX_USE_STD_VECTOR_ is enabled.
   */
  inline Solver()
      : iteration_max(M * DEFAULT_ITERATION_MAX_EIGEN_SOLVER),
        iteration_max_for_eigen_vector(
            DEFAULT_ITERATION_MULTIPLIER_EIGEN_VECTOR *
            DEFAULT_ITERATION_MAX_EIGEN_SOLVER),
        gmres_k_dimension(InnerK),
        gmres_cycles(DEFAULT_GMRES_CYCLES_EIGEN_SOLVER),
        gmres_orthogonalization_passes(
            DEFAULT_GMRES_ORTHOGONALIZATION_PASSES_EIGEN_SOLVER),
        division_min(static_cast<T>(DEFAULT_DIVISION_MIN_EIGEN_SOLVER)),
        small_value(static_cast<T>(DEFAULT_EIGENVECTOR_RELATIVE_SHIFT)),
        gmres_k_decay_rate(
            static_cast<T>(DEFAULT_GMRES_DECAY_RATE_EIGEN_SOLVER)),
        relative_tolerance(
            static_cast<T>(DEFAULT_RELATIVE_TOLERANCE_EIGEN_SOLVER)),
        absolute_tolerance(
            static_cast<T>(DEFAULT_ABSOLUTE_TOLERANCE_EIGEN_SOLVER)),
        eigenvector_tolerance(
            static_cast<T>(DEFAULT_EIGENVECTOR_RELATIVE_TOLERANCE)),
        Hessen_(), _eigen_values(), _eigen_vectors(),
        _status(EigenSolverStatus::not_started), _values_converged(false),
        _vectors_converged(false), _threshold_adjusted(false),
        _inner_failed(false), _initialized(false), _vectors_initialized(false),
        _scale(0), _eigenvector_residual(0), _sweeps(0) {
#ifdef BASE_MATRIX_USE_STD_VECTOR_
    this->_eigen_values.resize(M);
#endif
  }

  /* Copy Constructor */
  Solver(const Solver &) = default;
  Solver &operator=(const Solver &) = default;

  /* Move Constructor */
  Solver(Solver &&) = default;
  Solver &operator=(Solver &&) = default;

  /**
   * @brief Starts an eigenvalue solve from a real square matrix.
   *
   * Rebuilds the Hessenberg matrix, resets the accumulated sweep count and
   * eigenvector initialization, and runs one iteration_max budget. Use the
   * convergence flag before interpreting the diagonal estimates as converged
   * eigenvalues.
   *
   * @param matrix The finite real input matrix.
   */
  inline void solve_eigen_values(const Matrix<T, M, M> &matrix) {
    this->_threshold_adjusted = false;
    this->_validate_threshold();
    this->_scale = static_cast<T>(0);
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = 0; i < M; ++i) {
        T candidate = Base::Math::abs(matrix(i, j));
        if (candidate > this->_scale) {
          this->_scale = candidate;
        }
      }
    }
    hessenberg(matrix, this->Hessen_, this->division_min);
    this->_initialized = true;
    this->_values_converged = false;
    this->_vectors_converged = false;
    this->_vectors_initialized = false;
    this->_status = EigenSolverStatus::not_converged;
    this->_sweeps = 0;
    this->continue_solving_eigen_values();
  }

  /**
   * @brief Spends another QR budget on the stored Hessenberg matrix.
   *
   * Processes iteration_max sweep slots, skipping work after deflation or an
   * unsupported real complex pair. The current diagonal is then copied to the
   * eigenvalue estimates. Does nothing before solve_eigen_values initializes
   * the matrix.
   */
  inline void continue_solving_eigen_values() {
    this->_validate_threshold();
    if (this->_initialized) {
      for (std::size_t sweep = 0; sweep < this->iteration_max; ++sweep) {
        std::size_t begin = 0;
        std::size_t end = 0;
        bool active = this->_find_active_block(begin, end);
        if (active && this->_status != EigenSolverStatus::complex_pair) {
          bool complex_pair = false;
          S shift = trailing_shift(this->Hessen_, end, complex_pair);
          if (complex_pair && end == begin + 1) {
            this->_status = EigenSolverStatus::complex_pair;
          } else {
            // A real trailing pair in a larger real block can be complex;
            // its real part is a valid real shift, but is not an eigenvalue.
            // A deterministic exceptional shift breaks zero-shift cycles
            // (for example, a cyclic permutation / companion matrix).
            if ((this->_sweeps + 1) % EIGEN_EXCEPTIONAL_SHIFT_PERIOD == 0) {
              T offset =
                  Numerical::Scalar<S>::magnitude(this->Hessen_(end, end - 1));
              shift = this->Hessen_(end, end) + offset;
            }
            implicit_qr_sweep(this->Hessen_, begin, end, shift);
            ++this->_sweeps;
          }
        }
      }
      std::size_t begin = 0;
      std::size_t end = 0;
      this->_values_converged = !this->_find_active_block(begin, end);
      if (this->_values_converged) {
        this->_status = EigenSolverStatus::converged;
      }
      for (std::size_t i = 0; i < M; ++i) {
        this->_eigen_values[i] = this->Hessen_(i, i);
      }
    }
  }

  /**
   * @brief Refines eigenvector columns by bounded shifted inverse iteration.
   *
   * Call solve_eigen_values first with the same matrix. Inverse iterations run
   * only when the eigenvalues have converged. Repeated calls reuse the stored
   * vector directions. Zero candidates and inner GMRES breakdown are reported
   * through status flags; a returned direction does not guarantee convergence.
   *
   * @param matrix The real matrix corresponding to the stored eigenvalues.
   */
  inline void solve_eigen_vectors(const Matrix<T, M, M> &matrix) {
    this->_validate_threshold();
    this->_vectors_converged = this->_values_converged;
    if (this->_values_converged) {
      this->_status = EigenSolverStatus::not_converged;
    }
    this->_inner_failed = false;
    this->_eigenvector_residual = static_cast<T>(0);
    // Residuals are divided by the input scale, not by the eigenvalue alone.
    T scale =
        this->_scale > static_cast<T>(0) ? this->_scale : static_cast<T>(1);
    T perturbation = Base::Math::abs(this->small_value) * scale;
    if (perturbation < this->division_min) {
      perturbation = this->division_min;
    }
    for (std::size_t k = 0; k < M; ++k) {
      Matrix<S, M, M> shifted;
      for (std::size_t j = 0; j < M; ++j) {
        for (std::size_t i = 0; i < M; ++i) {
          shifted(i, j) = S(matrix(i, j));
        }
        shifted(j, j) -= this->_eigen_values[k] + perturbation;
      }
      Vector<S, M> vector;
      // Distinct deterministic starting vectors avoid a shared zero projection.
      for (std::size_t i = 0; i < M; ++i) {
        vector[i] = this->_vectors_initialized ? this->_eigen_vectors(i, k)
                                               : S(static_cast<T>(1));
      }
      if (!this->_vectors_initialized) {
        vector[k] += static_cast<T>(1);
      }
      T length_inverse = Numerical::norm_inv(vector);
      for (std::size_t i = 0; i < M; ++i) {
        vector[i] = vector[i] * length_inverse;
      }
      GmresSolver<S, M, InnerK> inner;
      inner.division_min = this->division_min;
      inner.cycles = this->gmres_cycles;
      inner.krylov_dimension = this->gmres_k_dimension;
      inner.orthogonalization_passes = this->gmres_orthogonalization_passes;
      inner.relative_tolerance =
          this->gmres_k_decay_rate > static_cast<T>(0)
              ? this->gmres_k_decay_rate
              : static_cast<T>(DEFAULT_INNER_GMRES_RELATIVE_TOLERANCE);
      bool active = this->_values_converged;
      bool converged = false;
      for (std::size_t iteration = 0;
           iteration < this->iteration_max_for_eigen_vector; ++iteration) {
        if (active) {
          Vector<S, M> initial;
          Vector<S, M> candidate = inner.solve(shifted, vector, initial);
          T candidate_length = Numerical::norm(candidate);
          if (candidate_length <= this->division_min) {
            active = false;
            this->_status = EigenSolverStatus::zero_vector;
          } else {
            for (std::size_t i = 0; i < M; ++i) {
              vector[i] = candidate[i] / candidate_length;
            }
            T residual =
                this->_vector_residual(matrix, vector, this->_eigen_values[k]) /
                scale;
            converged = residual <= this->eigenvector_tolerance;
            if (converged) {
              active = false;
            }
            if (inner.has_breakdown() && !inner.is_converged()) {
              this->_inner_failed = true;
              // Preserve the computed direction but never report inner success.
            }
          }
        }
      }
      T residual =
          this->_vector_residual(matrix, vector, this->_eigen_values[k]) /
          scale;
      if (residual > this->_eigenvector_residual) {
        this->_eigenvector_residual = residual;
      }
      this->_vectors_converged = this->_vectors_converged && converged;
      for (std::size_t i = 0; i < M; ++i) {
        this->_eigen_vectors(i, k) = vector[i];
      }
    }
    this->_vectors_initialized = true;
    if (this->_inner_failed) {
      this->_vectors_converged = false;
      this->_status = EigenSolverStatus::inner_failure;
    } else if (this->_values_converged &&
               this->_status != EigenSolverStatus::zero_vector) {
      this->_status = this->_vectors_converged
                          ? EigenSolverStatus::converged
                          : EigenSolverStatus::not_converged;
    }
  }

  /**
   * @brief Starts the eigenvalue solve and then computes eigenvector estimates.
   * @param matrix The finite real input matrix for both solve stages.
   */
  inline void solve_eigen_values_and_vectors(const Matrix<T, M, M> &matrix) {
    this->solve_eigen_values(matrix);
    this->solve_eigen_vectors(matrix);
  }

  /**
   * @brief Copies the current eigenvalue estimates in their stored order.
   * @return The diagonal estimates, which may still be unconverged.
   */
  inline EigenValues_Type get_eigen_values() const {
    return this->_eigen_values;
  }
  /**
   * @brief Copies the current eigenvector estimates.
   * @return A matrix whose columns correspond to the stored eigenvalues.
   */
  inline Matrix<S, M, M> get_eigen_vectors() const {
    return this->_eigen_vectors;
  }
  /**
   * @brief Copies the Hessenberg matrix at the current QR stage.
   * @return The stored matrix after reduction and any completed sweeps.
   */
  inline Matrix<S, M, M> get_hessenberg() const { return this->Hessen_; }
  /**
   * @brief Retrieves the status of the latest solve stage.
   * @return The stored eigenvalue or eigenvector status.
   */
  inline EigenSolverStatus get_status() const { return this->_status; }
  /**
   * @brief Reports whether all Hessenberg subdiagonals were deflated.
   * @return True when the latest eigenvalue stage met its deflation criteria.
   */
  inline bool eigen_values_converged() const { return this->_values_converged; }
  /**
   * @brief Reports whether all eigenvector columns met the solve criteria.
   * @return True when the eigenvalues and vector residuals passed and no inner
   * failure was recorded.
   */
  inline bool eigen_vectors_converged() const {
    return this->_vectors_converged;
  }
  /**
   * @brief Reports correction of a nonpositive denominator threshold.
   * @return True if a correction occurred since the latest eigenvalue restart.
   */
  inline bool threshold_was_adjusted() const {
    return this->_threshold_adjusted;
  }
  /**
   * @brief Reports an unconverged inner GMRES breakdown during vector solving.
   * @return The failure flag from the latest eigenvector solve.
   */
  inline bool inner_solve_failed() const { return this->_inner_failed; }
  /**
   * @brief Retrieves the largest scaled residual of the stored vector columns.
   * @return The largest norm of A*v-lambda*v divided by the input maximum
   * absolute entry, using scale one for a zero input matrix.
   */
  inline T get_eigenvector_residual() const {
    return this->_eigenvector_residual;
  }
  /**
   * @brief Retrieves the number of QR sweeps actually applied.
   * @return The count accumulated since solve_eigen_values last restarted.
   */
  inline std::size_t get_sweep_count() const { return this->_sweeps; }

  /**
   * @brief Sets the QR sweep-slot budget for each start or continuation call.
   * @param value The maximum number of sweep slots per call, including zero.
   */
  inline void set_iteration_max(const std::size_t &value) {
    this->iteration_max = value;
  }
  /**
   * @brief Sets the inverse-iteration budget for each eigenvector column.
   * @param value The number of iteration slots per column and call.
   */
  inline void set_iteration_max_for_eigen_vector(const std::size_t &value) {
    this->iteration_max_for_eigen_vector = value;
  }
  /**
   * @brief Stores the denominator-protection and minimum-shift threshold.
   * @param value The requested threshold; a nonpositive value is replaced by
   * the default when a solve stage validates its configuration.
   */
  inline void set_division_min(const T &value) { this->division_min = value; }
  /**
   * @brief Sets the relative perturbation used by shifted inverse iteration.
   * @param value The shift coefficient; its absolute value is scaled by the
   * input magnitude and the resulting perturbation is floored by division_min.
   */
  inline void set_small_value(const T &value) { this->small_value = value; }
  /**
   * @brief Sets the relative residual tolerance for inner GMRES solves.
   * @param value A positive tolerance, or a nonpositive value to select
   * DEFAULT_INNER_GMRES_RELATIVE_TOLERANCE.
   */
  inline void set_gmres_k_decay_rate(const T &value) {
    this->gmres_k_decay_rate = value;
  }
  /**
   * @brief Stores the work budget of each inner GMRES solve.
   * @param dimension The requested Krylov size, clamped by GMRES to
   * [1, InnerK].
   * @param cycles The maximum cycle count, including the first cycle.
   * @param passes The orthogonalization setting: at most one selects one pass,
   * and larger values select two passes.
   */
  inline void set_gmres_budget(std::size_t dimension, std::size_t cycles,
                               std::size_t passes) {
    this->gmres_k_dimension = dimension;
    this->gmres_cycles = cycles;
    this->gmres_orthogonalization_passes = passes;
  }
  /**
   * @brief Forms the residual matrix for the stored eigenpairs.
   *
   * This diagnostic returns A*V-V*D without modifying convergence flags.
   *
   * @param matrix The real input matrix associated with the stored estimates.
   * @return A matrix whose j-th column is A*v_j-lambda_j*v_j.
   */
  inline Matrix<S, M, M> check_validity(const Matrix<T, M, M> &matrix) const {
    Matrix<S, M, M> result;
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < M; ++k) {
          result(i, j) += this->_eigen_vectors(k, j) * matrix(i, k);
        }
        result(i, j) -= this->_eigen_vectors(i, j) * this->_eigen_values[j];
      }
    }
    return result;
  }

  std::size_t iteration_max;
  std::size_t iteration_max_for_eigen_vector;
  std::size_t gmres_k_dimension;
  std::size_t gmres_cycles;
  std::size_t gmres_orthogonalization_passes;
  T division_min;
  T small_value;
  T gmres_k_decay_rate;
  T relative_tolerance;
  T absolute_tolerance;
  T eigenvector_tolerance;

protected:
  Matrix<S, M, M> Hessen_;
  EigenValues_Type _eigen_values;
  Matrix<S, M, M> _eigen_vectors;
  EigenSolverStatus _status;
  bool _values_converged;
  bool _vectors_converged;
  bool _threshold_adjusted;
  bool _inner_failed;
  bool _initialized;
  bool _vectors_initialized;
  T _scale;
  T _eigenvector_residual;
  std::size_t _sweeps;

  /**
   * @brief Replaces a nonpositive denominator threshold and records correction.
   */
  inline void _validate_threshold() {
    if (this->division_min <= static_cast<T>(0)) {
      this->division_min = static_cast<T>(DEFAULT_DIVISION_MIN_EIGEN_SOLVER);
      this->_threshold_adjusted = true;
    }
  }

  /**
   * @brief Deflates small subdiagonals and locates the bottommost active block.
   *
   * Deflation uses the configured absolute and relative tolerances. The input
   * scale is used when the adjacent diagonal magnitudes both vanish.
   *
   * @param begin Receives the first active index when a block is found.
   * @param end Receives the last active index when a block is found.
   * @return True if a nonzero subdiagonal remains; output indices should only
   * be used in that case.
   */
  inline bool _find_active_block(std::size_t &begin, std::size_t &end) {
    T absolute = this->absolute_tolerance > static_cast<T>(0)
                     ? this->absolute_tolerance
                     : static_cast<T>(0);
    T relative = this->relative_tolerance > static_cast<T>(0)
                     ? this->relative_tolerance
                     : static_cast<T>(0);
    for (std::size_t i = 1; i < M; ++i) {
      T local_scale =
          Numerical::Scalar<S>::magnitude(this->Hessen_(i - 1, i - 1)) +
          Numerical::Scalar<S>::magnitude(this->Hessen_(i, i));
      if (local_scale == static_cast<T>(0)) {
        local_scale = this->_scale;
      }
      if (Numerical::Scalar<S>::magnitude(this->Hessen_(i, i - 1)) <=
          absolute + relative * local_scale) {
        this->Hessen_(i, i - 1) = S();
      }
    }
    bool found = false;
    // The last nonzero subdiagonal identifies the bottommost active block.
    for (std::size_t i = 1; i < M; ++i) {
      if (Numerical::Scalar<S>::component_max(this->Hessen_(i, i - 1)) >
          static_cast<T>(0)) {
        end = i;
        found = true;
      }
    }
    begin = 0;
    for (std::size_t i = 1; i < M; ++i) {
      if (i <= end && Numerical::Scalar<S>::component_max(
                          this->Hessen_(i, i - 1)) == static_cast<T>(0)) {
        begin = i;
      }
    }
    return found;
  }

  /**
   * @brief Evaluates an eigenpair residual before input-scale normalization.
   * @param matrix The real matrix whose eigenpair is being checked.
   * @param vector The eigenvector estimate.
   * @param value The corresponding eigenvalue estimate.
   * @return The scaled-norm implementation's estimate of
   * ||A*vector-value*vector||.
   */
  inline T _vector_residual(const Matrix<T, M, M> &matrix,
                            const Vector<S, M> &vector, const S &value) const {
    Vector<S, M> residual;
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < M; ++j) {
        residual[i] += vector[j] * matrix(i, j);
      }
      residual[i] -= vector[i] * value;
    }
    return Numerical::norm(residual);
  }
};

} // namespace EigenOperation

/**
 * @brief Computes real eigenpairs of a real matrix with bounded QR iteration.
 *
 * An isolated complex conjugate pair is reported as complex_pair rather than
 * represented by real eigenvalues. Use EigenSolverComplex for that spectrum.
 *
 * @tparam T The real floating-point input and result type.
 * @tparam M The positive matrix dimension.
 * @tparam InnerK The fixed inner GMRES capacity, satisfying 1 <= InnerK <= M.
 */
template <typename T, std::size_t M, std::size_t InnerK = M>
class EigenSolverReal : public EigenOperation::Solver<T, T, M, InnerK> {
public:
  EigenSolverReal() = default;

  /* Copy Constructor */
  EigenSolverReal(const EigenSolverReal &) = default;
  EigenSolverReal &operator=(const EigenSolverReal &) = default;

  /* Move Constructor */
  EigenSolverReal(EigenSolverReal &&) = default;
  EigenSolverReal &operator=(EigenSolverReal &&) = default;
};

/**
 * @brief Computes complex eigenpairs of a real matrix with bounded iteration.
 *
 * Complex storage represents conjugate eigenvalue pairs and the corresponding
 * inverse-iteration vector estimates. The input matrix remains real.
 *
 * @tparam T The real floating-point input and component type.
 * @tparam M The positive matrix dimension.
 * @tparam InnerK The fixed inner GMRES capacity, satisfying 1 <= InnerK <= M.
 */
template <typename T, std::size_t M, std::size_t InnerK = M>
class EigenSolverComplex
    : public EigenOperation::Solver<Complex<T>, T, M, InnerK> {
public:
  EigenSolverComplex() = default;

  /* Copy Constructor */
  EigenSolverComplex(const EigenSolverComplex &) = default;
  EigenSolverComplex &operator=(const EigenSolverComplex &) = default;

  /* Move Constructor */
  EigenSolverComplex(EigenSolverComplex &&) = default;
  EigenSolverComplex &operator=(EigenSolverComplex &&) = default;
};

/**
 * @brief Cyclic Jacobi for caller-guaranteed symmetric input.
 *
 * The upper triangle is packed. ComputeVectors=false removes the dense
 * eigenvector storage, and get_eigen_vectors is only available in the true
 * mode. Symmetry is a caller precondition and is not checked internally.
 *
 * @tparam T The real floating-point element and tolerance type.
 * @tparam M The positive matrix dimension.
 * @tparam ComputeVectors Whether to accumulate the Jacobi eigenvector matrix.
 */
template <typename T, std::size_t M, bool ComputeVectors = true>
class EigenSolverSymmetric {
public:
  static_assert(M > 0, "An eigensolver requires a nonempty matrix.");
  /**
   * @brief Initializes the Jacobi budget, tolerances, and packed workspaces.
   */
  inline EigenSolverSymmetric()
      : iteration_max(DEFAULT_ITERATION_MAX_EIGEN_SOLVER),
        division_min(static_cast<T>(DEFAULT_DIVISION_MIN_EIGEN_SOLVER)),
        relative_tolerance(
            static_cast<T>(DEFAULT_RELATIVE_TOLERANCE_EIGEN_SOLVER)),
        absolute_tolerance(
            static_cast<T>(DEFAULT_ABSOLUTE_TOLERANCE_EIGEN_SOLVER)),
        _triangle(), _vectors(), _converged(false), _threshold_adjusted(false),
        _scale(0) {}

  /* Copy Constructor */
  EigenSolverSymmetric(const EigenSolverSymmetric &) = default;
  EigenSolverSymmetric &operator=(const EigenSolverSymmetric &) = default;

  /* Move Constructor */
  EigenSolverSymmetric(EigenSolverSymmetric &&) = default;
  EigenSolverSymmetric &operator=(EigenSolverSymmetric &&) = default;

  /**
   * @brief Starts cyclic Jacobi iteration from a symmetric real matrix.
   *
   * Loads only the upper triangle, initializes the optional eigenvectors to
   * identity, and spends one iteration_max sweep budget. Symmetry is assumed;
   * the lower triangle is not inspected.
   *
   * @param matrix The finite symmetric real input matrix.
   */
  inline void solve_eigen_values(const Matrix<T, M, M> &matrix) {
    this->_threshold_adjusted = false;
    if (this->division_min <= static_cast<T>(0)) {
      this->division_min = static_cast<T>(DEFAULT_DIVISION_MIN_EIGEN_SOLVER);
      this->_threshold_adjusted = true;
    }
    this->_scale = static_cast<T>(0);
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        this->_at(i, j) = matrix(i, j);
        T candidate = Base::Math::abs(matrix(i, j));
        if (candidate > this->_scale) {
          this->_scale = candidate;
        }
      }
    }
    for (std::size_t i = 0; i < this->_vectors.size(); ++i) {
      this->_vectors[i] = static_cast<T>(0);
    }
    if (ComputeVectors) {
      for (std::size_t i = 0; i < M; ++i) {
        this->_vectors[i * M + i] = static_cast<T>(1);
      }
    }
    this->continue_solving_eigen_values();
  }

  /**
   * @brief Applies another cyclic Jacobi budget to the stored upper triangle.
   *
   * Call solve_eigen_values first to load the matrix. Each sweep visits all
   * index pairs, and rotations already within tolerance do no work. The final
   * off-diagonal entries determine the convergence flag.
   */
  inline void continue_solving_eigen_values() {
    for (std::size_t sweep = 0; sweep < this->iteration_max; ++sweep) {
      for (std::size_t p = 0; p < M; ++p) {
        for (std::size_t q = p + 1; q < M; ++q) {
          this->_rotate(p, q);
        }
      }
    }
    this->_converged = true;
    T tolerance = this->_tolerance();
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = 0; i < j; ++i) {
        if (Base::Math::abs(this->_at(i, j)) > tolerance) {
          this->_converged = false;
        }
      }
    }
  }

  /**
   * @brief Starts the symmetric solve, accumulating vectors when enabled.
   *
   * This shares the eigenvalue solve path. ComputeVectors=false still omits
   * eigenvector storage and updates.
   *
   * @param matrix The finite symmetric real input matrix.
   */
  inline void solve_eigen_values_and_vectors(const Matrix<T, M, M> &matrix) {
    this->solve_eigen_values(matrix);
  }
  /**
   * @brief Copies the current unsorted Jacobi eigenvalue estimates.
   * @return The packed diagonal entries, which may still be unconverged.
   */
  inline std::array<T, M> get_eigen_values() const {
    std::array<T, M> values = {};
    for (std::size_t i = 0; i < M; ++i) {
      values[i] = this->_at(i, i);
    }
    return values;
  }
  /**
   * @brief Copies the accumulated Jacobi eigenvector estimates.
   * @tparam Enabled Enables this getter only when vector storage is present.
   * @return A matrix whose columns correspond to get_eigen_values().
   */
  template <bool Enabled = ComputeVectors>
  inline
      typename std::enable_if<Enabled && ComputeVectors, Matrix<T, M, M>>::type
      get_eigen_vectors() const {
    Matrix<T, M, M> result;
    for (std::size_t j = 0; j < M; ++j) {
      for (std::size_t i = 0; i < M; ++i) {
        result(i, j) = this->_vectors[j * M + i];
      }
    }
    return result;
  }
  /**
   * @brief Reports whether every stored off-diagonal entry is within tolerance.
   * @return The convergence flag from the latest Jacobi budget.
   */
  inline bool eigen_values_converged() const { return this->_converged; }
  /**
   * @brief Reports correction of a nonpositive denominator threshold.
   * @return True if solve_eigen_values replaced the requested threshold.
   */
  inline bool threshold_was_adjusted() const {
    return this->_threshold_adjusted;
  }
  /**
   * @brief Sets the number of cyclic sweep slots per solve or continuation.
   * @param value The nonnegative sweep budget.
   */
  inline void set_iteration_max(const std::size_t &value) {
    this->iteration_max = value;
  }
  /**
   * @brief Stores the solver's denominator-threshold configuration.
   *
   * A nonpositive value is corrected on the next solve_eigen_values call.
   * Off-diagonal convergence is controlled by the separate absolute and
   * relative tolerances.
   *
   * @param value The requested denominator threshold.
   */
  inline void set_division_min(const T &value) { this->division_min = value; }

  std::size_t iteration_max;
  T division_min;
  T relative_tolerance;
  T absolute_tolerance;

protected:
  std::array<T, M *(M + 1) / 2> _triangle;
  std::array<T, ComputeVectors ? M * M : 0> _vectors;
  bool _converged;
  bool _threshold_adjusted;
  T _scale;

  /**
   * @brief Accesses a symmetric entry through its packed upper-triangle slot.
   * @param i The row index, less than M.
   * @param j The column index, less than M.
   * @return A writable reference shared by entries (i, j) and (j, i).
   */
  inline T &_at(std::size_t i, std::size_t j) {
    return i <= j ? this->_triangle[j * (j + 1) / 2 + i]
                  : this->_triangle[i * (i + 1) / 2 + j];
  }
  /**
   * @brief Reads a symmetric entry from packed upper-triangle storage.
   * @param i The row index, less than M.
   * @param j The column index, less than M.
   * @return A constant reference shared by entries (i, j) and (j, i).
   */
  inline const T &_at(std::size_t i, std::size_t j) const {
    return i <= j ? this->_triangle[j * (j + 1) / 2 + i]
                  : this->_triangle[i * (i + 1) / 2 + j];
  }

  /**
   * @brief Combines the absolute tolerance and input-scaled relative tolerance.
   * @return The nonnegative off-diagonal threshold; negative settings
   * contribute zero rather than reducing the other contribution.
   */
  inline T _tolerance() const {
    T absolute = this->absolute_tolerance > static_cast<T>(0)
                     ? this->absolute_tolerance
                     : static_cast<T>(0);
    T relative = this->relative_tolerance > static_cast<T>(0)
                     ? this->relative_tolerance
                     : static_cast<T>(0);
    return absolute + relative * this->_scale;
  }

  /**
   * @brief Eliminates one significant off-diagonal entry by a Jacobi rotation.
   *
   * Entries already within tolerance are left unchanged. The rotation updates
   * the packed matrix and, when enabled, the two corresponding vector columns.
   *
   * @param p The first index, satisfying p < q.
   * @param q The second index, less than M.
   */
  inline void _rotate(std::size_t p, std::size_t q) {
    T off_diagonal = this->_at(p, q);
    T tolerance = this->_tolerance();
    if (Base::Math::abs(off_diagonal) > tolerance) {
      T scale = Base::Math::abs(this->_at(p, p));
      T candidate = Base::Math::abs(this->_at(q, q));
      if (candidate > scale) {
        scale = candidate;
      }
      candidate = Base::Math::abs(off_diagonal);
      if (candidate > scale) {
        scale = candidate;
      }
      T delta = (this->_at(q, q) / scale - this->_at(p, p) / scale) *
                static_cast<T>(0.5);
      T off_scaled = off_diagonal / scale;
      T denominator =
          Base::Math::abs(delta) +
          EigenOperation::square_root(delta * delta + off_scaled * off_scaled);
      T tangent = denominator == static_cast<T>(0) ? static_cast<T>(0)
                                                   : off_scaled / denominator;
      if (delta < static_cast<T>(0)) {
        tangent = -tangent;
      }
      T cosine = static_cast<T>(1) / Numerical::normalized_sqrt(
                                         static_cast<T>(1) + tangent * tangent);
      T sine = tangent * cosine;
      T first_diagonal = this->_at(p, p);
      T second_diagonal = this->_at(q, q);
      this->_at(p, p) = first_diagonal - tangent * off_diagonal;
      this->_at(q, q) = second_diagonal + tangent * off_diagonal;
      this->_at(p, q) = static_cast<T>(0);
      for (std::size_t i = 0; i < M; ++i) {
        if (i != p && i != q) {
          T first = this->_at(i, p);
          T second = this->_at(i, q);
          this->_at(i, p) = cosine * first - sine * second;
          this->_at(i, q) = sine * first + cosine * second;
        }
      }
      if (ComputeVectors) {
        for (std::size_t i = 0; i < M; ++i) {
          T first = this->_vectors[p * M + i];
          T second = this->_vectors[q * M + i];
          this->_vectors[p * M + i] = cosine * first - sine * second;
          this->_vectors[q * M + i] = sine * first + cosine * second;
        }
      }
    }
  }
};

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_EIGEN_SOLVER_HPP_
