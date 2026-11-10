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
#include <limits>
#include <vector>

namespace Base {
namespace Matrix {

const std::size_t DEFAULT_ITERATION_MAX_EIGEN_SOLVER = 10;
const double DEFAULT_DIVISION_MIN_EIGEN_SOLVER = 1.0e-20;
const double EIGEN_SMALL_VALUE = 1.0e-6;

enum class EigenSolverStatus {
  not_started,
  not_converged,
  converged,
  complex_pair,
  inner_failure,
  zero_vector
};

namespace EigenOperation {

// Inputs are nonnegative, dimensionless values from a scaled 2x2 block.
// Normalize very small values before the dependency's float-based seed.
template <typename T> inline T square_root(T value) {
  if (value <= static_cast<T>(0)) {
    return static_cast<T>(0);
  }
  T root_scale = static_cast<T>(1);
  if (value < static_cast<T>(1e-30)) {
    // Eleven slots cover the IEEE float/double exponent ranges, including
    // subnormals. Each multiplication starts below 1e-30 and cannot overflow.
    for (std::size_t pass = 0; pass < 11; ++pass) {
      if (value < static_cast<T>(1e-30)) {
        value *= static_cast<T>(1e30);
        root_scale *= static_cast<T>(1e-15);
      }
    }
  }
  return root_scale *
         Base::Math::sqrt_newton_method<T, 3>(value, static_cast<T>(1e-30));
}

template <typename S> struct Shift {
  using T = S;
  static inline T real_part(T value) { return value; }
  static inline T square_root_scalar(T value, bool &complex_pair) {
    complex_pair = value < static_cast<T>(0);
    return square_root(value);
  }
};

template <typename T> struct Shift<Complex<T>> {
  static inline T real_part(const Complex<T> &value) { return value.real; }
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

/** Direct real Householder similarity reduction; no reflector matrix. */
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
  S a = matrix(end - 1, end - 1) / scale;
  S b = matrix(end - 1, end) / scale;
  S c = matrix(end, end - 1) / scale;
  S d = matrix(end, end) / scale;
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

/** G H G^H, with a moving bulge and full coupling to the other blocks. */
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

/** Shared implementation; the public real/complex classes preserve names. */
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

  Solver()
      : iteration_max(M * DEFAULT_ITERATION_MAX_EIGEN_SOLVER),
        iteration_max_for_eigen_vector(3 * DEFAULT_ITERATION_MAX_EIGEN_SOLVER),
        gmres_k_dimension(InnerK), gmres_cycles(1),
        gmres_orthogonalization_passes(1),
        division_min(static_cast<T>(DEFAULT_DIVISION_MIN_EIGEN_SOLVER)),
        small_value(static_cast<T>(EIGEN_SMALL_VALUE)), gmres_k_decay_rate(0),
        relative_tolerance(static_cast<T>(8) *
                           std::numeric_limits<T>::epsilon()),
        absolute_tolerance(0),
        eigenvector_tolerance(static_cast<T>(32) *
                              std::numeric_limits<T>::epsilon()),
        Hessen_(), _eigen_values(), _eigen_vectors(),
        _status(EigenSolverStatus::not_started), _values_converged(false),
        _vectors_converged(false), _threshold_adjusted(false),
        _inner_failed(false), _initialized(false), _vectors_initialized(false),
        _scale(0), _eigenvector_residual(0), _sweeps(0) {
#ifdef BASE_MATRIX_USE_STD_VECTOR_
    this->_eigen_values.resize(M);
#endif
    T precision_shift = square_root(std::numeric_limits<T>::epsilon());
    if (this->small_value < precision_shift) {
      this->small_value = precision_shift;
    }
  }

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
            if ((this->_sweeps + 1) % 8 == 0) {
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
      T length = Numerical::norm(vector);
      for (std::size_t i = 0; i < M; ++i) {
        vector[i] = vector[i] / length;
      }
      GmresSolver<S, M, InnerK> inner;
      inner.division_min = this->division_min;
      inner.cycles = this->gmres_cycles;
      inner.krylov_dimension = this->gmres_k_dimension;
      inner.orthogonalization_passes = this->gmres_orthogonalization_passes;
      inner.relative_tolerance =
          this->gmres_k_decay_rate > static_cast<T>(0)
              ? this->gmres_k_decay_rate
              : static_cast<T>(16) * EigenOperation::square_root(
                                         std::numeric_limits<T>::epsilon());
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

  inline void solve_eigen_values_and_vectors(const Matrix<T, M, M> &matrix) {
    this->solve_eigen_values(matrix);
    this->solve_eigen_vectors(matrix);
  }

  inline EigenValues_Type get_eigen_values() const {
    return this->_eigen_values;
  }
  inline Matrix<S, M, M> get_eigen_vectors() const {
    return this->_eigen_vectors;
  }
  inline Matrix<S, M, M> get_hessenberg() const { return this->Hessen_; }
  inline EigenSolverStatus get_status() const { return this->_status; }
  inline bool eigen_values_converged() const { return this->_values_converged; }
  inline bool eigen_vectors_converged() const {
    return this->_vectors_converged;
  }
  inline bool threshold_was_adjusted() const {
    return this->_threshold_adjusted;
  }
  inline bool inner_solve_failed() const { return this->_inner_failed; }
  inline T get_eigenvector_residual() const {
    return this->_eigenvector_residual;
  }
  inline std::size_t get_sweep_count() const { return this->_sweeps; }

  inline void set_iteration_max(const std::size_t &value) {
    this->iteration_max = value;
  }
  inline void set_iteration_max_for_eigen_vector(const std::size_t &value) {
    this->iteration_max_for_eigen_vector = value;
  }
  inline void set_division_min(const T &value) { this->division_min = value; }
  inline void set_small_value(const T &value) { this->small_value = value; }
  inline void set_gmres_k_decay_rate(const T &value) {
    this->gmres_k_decay_rate = value;
  }
  inline void set_gmres_budget(std::size_t dimension, std::size_t cycles,
                               std::size_t passes) {
    this->gmres_k_dimension = dimension;
    this->gmres_cycles = cycles;
    this->gmres_orthogonalization_passes = passes;
  }
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

  inline void _validate_threshold() {
    if (this->division_min <= static_cast<T>(0)) {
      this->division_min = static_cast<T>(DEFAULT_DIVISION_MIN_EIGEN_SOLVER);
      this->_threshold_adjusted = true;
    }
  }

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

/** Real spectrum only; complex_pair status requires EigenSolverComplex. */
template <typename T, std::size_t M, std::size_t InnerK = M>
class EigenSolverReal : public EigenOperation::Solver<T, T, M, InnerK> {};

/** Real input, complex spectrum and complex inverse-iteration eigenvectors. */
template <typename T, std::size_t M, std::size_t InnerK = M>
class EigenSolverComplex
    : public EigenOperation::Solver<Complex<T>, T, M, InnerK> {};

/**
 * @brief Cyclic Jacobi for caller-guaranteed symmetric input.
 *
 * The upper triangle is packed. ComputeVectors=false removes the dense
 * eigenvector storage, and get_eigen_vectors is only available in the true
 * mode.
 */
template <typename T, std::size_t M, bool ComputeVectors = true>
class EigenSolverSymmetric {
public:
  static_assert(M > 0, "An eigensolver requires a nonempty matrix.");
  EigenSolverSymmetric()
      : iteration_max(DEFAULT_ITERATION_MAX_EIGEN_SOLVER),
        division_min(static_cast<T>(DEFAULT_DIVISION_MIN_EIGEN_SOLVER)),
        relative_tolerance(static_cast<T>(8) *
                           std::numeric_limits<T>::epsilon()),
        absolute_tolerance(0), _triangle(), _vectors(), _converged(false),
        _threshold_adjusted(false), _scale(0) {}

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

  inline void solve_eigen_values_and_vectors(const Matrix<T, M, M> &matrix) {
    this->solve_eigen_values(matrix);
  }
  inline std::array<T, M> get_eigen_values() const {
    std::array<T, M> values = {};
    for (std::size_t i = 0; i < M; ++i) {
      values[i] = this->_at(i, i);
    }
    return values;
  }
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
  inline bool eigen_values_converged() const { return this->_converged; }
  inline bool threshold_was_adjusted() const {
    return this->_threshold_adjusted;
  }
  inline void set_iteration_max(const std::size_t &value) {
    this->iteration_max = value;
  }
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

  inline T &_at(std::size_t i, std::size_t j) {
    return i <= j ? this->_triangle[j * (j + 1) / 2 + i]
                  : this->_triangle[i * (i + 1) / 2 + j];
  }
  inline const T &_at(std::size_t i, std::size_t j) const {
    return i <= j ? this->_triangle[j * (j + 1) / 2 + i]
                  : this->_triangle[i * (i + 1) / 2 + j];
  }

  inline T _tolerance() const {
    T absolute = this->absolute_tolerance > static_cast<T>(0)
                     ? this->absolute_tolerance
                     : static_cast<T>(0);
    T relative = this->relative_tolerance > static_cast<T>(0)
                     ? this->relative_tolerance
                     : static_cast<T>(0);
    return absolute + relative * this->_scale;
  }

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
