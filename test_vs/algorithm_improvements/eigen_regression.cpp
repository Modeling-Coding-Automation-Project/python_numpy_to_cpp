#include "base_matrix_eigen_solver.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>

using namespace Base::Matrix;

namespace {
int failures = 0;
void require(bool condition, const char *message) {
  if (!condition) {
    ++failures;
    std::printf("FAIL: %s\n", message);
  }
}

template <typename T, std::size_t N>
Matrix<T, N, N> rotated_diagonal(bool nonnormal = false) {
  Matrix<T, N, N> orthogonal = Matrix<T, N, N>::identity();
  for (std::size_t p = 0; p < N; ++p) {
    for (std::size_t q = p + 1; q < N; ++q) {
      for (std::size_t i = 0; i < N; ++i) {
        T first = orthogonal(i, p);
        T second = orthogonal(i, q);
        orthogonal(i, p) = T(0.8) * first - T(0.6) * second;
        orthogonal(i, q) = T(0.6) * first + T(0.8) * second;
      }
    }
  }
  Matrix<T, N, N> result;
  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t k = 0; k < N; ++k) {
        result(i, j) += orthogonal(i, k) * T(k + 1) * orthogonal(j, k);
        if (nonnormal && k + 1 < N) {
          result(i, j) += T(0.5) * orthogonal(i, k) * orthogonal(j, k + 1);
        }
      }
    }
  }
  return result;
}

template <typename T> T tolerance() {
  if (std::is_same<T, float>::value) { return T(2e-4); }
#ifdef BASE_MATH_USE_STD_MATH_
  return T(2e-11);
#else
  // External default sqrt error is about 9e-8; repeated transformations
  // accumulate this error. Keep strict double checks in the std-math build.
  return T(2e-6);
#endif
}

constexpr double HIGH_ACCURACY_VALUE_TOLERANCE = 8e-12;
constexpr double HIGH_ACCURACY_VECTOR_TOLERANCE = 3.2e-10;
constexpr double HIGH_ACCURACY_RELATIVE_SHIFT = 1e-11;
constexpr double DEFAULT_SOLUTION_QUALITY_LIMIT = 2e-6;

template <typename T, typename Solver>
inline void configure_value_accuracy(Solver &solver) {
#ifdef BASE_MATH_USE_STD_MATH_
  if (std::is_same<T, double>::value) {
    solver.relative_tolerance = T(HIGH_ACCURACY_VALUE_TOLERANCE);
  }
#else
  static_cast<void>(solver);
#endif
}

template <typename T, typename Solver>
inline void configure_general_accuracy(Solver &solver) {
  configure_value_accuracy<T>(solver);
#ifdef BASE_MATH_USE_STD_MATH_
  if (std::is_same<T, double>::value) {
    // Preserve the strict double assertions with an explicit precision target.
    solver.eigenvector_tolerance = T(HIGH_ACCURACY_VECTOR_TOLERANCE);
    solver.set_small_value(T(HIGH_ACCURACY_RELATIVE_SHIFT));
    solver.set_gmres_k_decay_rate(T(DEFAULT_INNER_GMRES_RELATIVE_TOLERANCE));
  }
#endif
}

template <typename T, std::size_t N, typename Values>
void check_spectrum(const Values &values, const char *message) {
  bool used[N] = {};
  for (std::size_t k = 0; k < N; ++k) {
    T best = std::numeric_limits<T>::max();
    std::size_t index = 0;
    for (std::size_t i = 0; i < N; ++i) {
      T error = std::abs(values[i] - T(k + 1));
      if (!used[i] && error < best) {
        best = error;
        index = i;
      }
    }
    used[index] = true;
    require(best < tolerance<T>() * T(N), message);
  }
}

template <typename T> void run() {
  const T tol = tolerance<T>();
  Matrix<T, 1, 1> single;
  single(0, 0) = T(7);
  EigenSolverReal<T, 1> one;
  configure_general_accuracy<T>(one);
  one.iteration_max = 0;
  one.solve_eigen_values_and_vectors(single);
  require(one.eigen_values_converged() && one.get_eigen_values()[0] == T(7),
          "1x1, zero QR budget");
  require(one.eigen_vectors_converged(), "1x1 inverse iteration");
  EigenSolverComplex<T, 1> complex_one;
  configure_general_accuracy<T>(complex_one);
  complex_one.solve_eigen_values_and_vectors(single);
  require(complex_one.eigen_values_converged() &&
              complex_one.get_eigen_values()[0].real == T(7),
          "complex 1x1 input");

  Matrix<T, 3, 3> zero;
  EigenSolverReal<T, 3> zeros;
  configure_general_accuracy<T>(zeros);
  zeros.solve_eigen_values(zero);
  require(zeros.eigen_values_converged(), "zero spectrum");
  require(zeros.get_eigen_values()[0] == T(0), "exact zero eigenvalue");
  zeros.division_min = T(-1);
  zeros.solve_eigen_values(zero);
  require(zeros.threshold_was_adjusted(), "negative threshold corrected");
  zeros.solve_eigen_values(zero);
  require(!zeros.threshold_was_adjusted(), "threshold adjustment reset");

  Matrix<T, 3, 3> active({{2, 1, 0}, {1, 2, 0}, {0, 0, 100}});
  EigenSolverReal<T, 3> active_solver;
  configure_general_accuracy<T>(active_solver);
  active_solver.solve_eigen_values(active);
  auto active_values = active_solver.get_eigen_values();
  require(std::abs(active_values[2] - T(100)) < tol, "deflated trailing value");
  require(std::abs(active_values[0] + active_values[1] - T(4)) < tol,
          "active block trace");
  require(std::abs(active_values[0] * active_values[1] - T(3)) < tol,
          "active block correct shift");

  Matrix<T, 6, 6> dense = rotated_diagonal<T, 6>(true);
  EigenSolverReal<T, 6> real;
  configure_general_accuracy<T>(real);
  real.iteration_max = 0;
  real.solve_eigen_values(dense);
  require(!real.eigen_values_converged(), "zero budget retains unreduced block");
  auto reduced = real.get_hessenberg();
  T trace_before = 0;
  T trace_after = 0;
  T norm_before = 0;
  T norm_after = 0;
  for (std::size_t j = 0; j < 6; ++j) {
    trace_before += dense(j, j);
    trace_after += reduced(j, j);
    for (std::size_t i = 0; i < 6; ++i) {
      norm_before += dense(i, j) * dense(i, j);
      norm_after += reduced(i, j) * reduced(i, j);
      if (i > j + 1) {
        require(reduced(i, j) == T(0), "Hessenberg exact structural zeros");
      }
    }
  }
  require(std::abs(trace_before - trace_after) < T(50) * tol,
          "direct Householder trace invariant");
  require(std::abs(norm_before - norm_after) < T(200) * tol,
          "direct Householder Frobenius invariant");
  real.iteration_max = 1;
  for (std::size_t i = 0; i < 100; ++i) {
    real.continue_solving_eigen_values();
  }
  require(real.eigen_values_converged(), "continued real QR converges");
  check_spectrum<T, 6>(real.get_eigen_values(), "real nonnormal known spectrum");
  real.solve_eigen_vectors(dense);
  require(real.get_eigenvector_residual() < T(30) * tol,
          "GMRES inverse iteration true residual");

  Matrix<T, 2, 2> imaginary({{0, -1}, {1, 0}});
  EigenSolverReal<T, 2> cannot_represent;
  configure_general_accuracy<T>(cannot_represent);
  cannot_represent.solve_eigen_values(imaginary);
  require(cannot_represent.get_status() == EigenSolverStatus::complex_pair &&
              !cannot_represent.eigen_values_converged(),
          "real API reports complex spectrum");
  EigenSolverComplex<T, 2> complex;
  configure_general_accuracy<T>(complex);
  complex.iteration_max = 32;
  complex.solve_eigen_values_and_vectors(imaginary);
  auto complex_values = complex.get_eigen_values();
  require(std::abs(complex_values[0].real) < tol &&
              std::abs(std::abs(complex_values[0].imag) - T(1)) < tol &&
              std::abs(complex_values[0].imag + complex_values[1].imag) < tol,
          "pure imaginary eigenvalue pair");
  require(complex.get_eigenvector_residual() < T(30) * tol,
          "complex inverse iteration residual");

  Matrix<T, 3, 3> cyclic({{0, 0, 1}, {1, 0, 0}, {0, 1, 0}});
  EigenSolverComplex<T, 3> cycle;
  configure_general_accuracy<T>(cycle);
  cycle.iteration_max = 96;
  cycle.solve_eigen_values(cyclic);
  require(cycle.eigen_values_converged(), "exceptional shift escapes zero-shift cycle");
  for (auto value : cycle.get_eigen_values()) {
    Complex<T> polynomial = value * value * value - T(1);
    require(Numerical::Scalar<Complex<T>>::magnitude(polynomial) < T(20) * tol,
            "cyclic permutation roots satisfy cubic");
  }

  Matrix<T, 6, 6> symmetric = rotated_diagonal<T, 6>();
  EigenSolverSymmetric<T, 6> jacobi;
  configure_value_accuracy<T>(jacobi);
  jacobi.iteration_max = 10;
  jacobi.solve_eigen_values_and_vectors(symmetric);
  require(jacobi.eigen_values_converged(), "Jacobi convergence");
  check_spectrum<T, 6>(jacobi.get_eigen_values(), "Jacobi known spectrum");
  auto vectors = jacobi.get_eigen_vectors();
  auto values = jacobi.get_eigen_values();
  for (std::size_t j = 0; j < 6; ++j) {
    for (std::size_t i = 0; i < 6; ++i) {
      T dot = 0;
      T residual = -vectors(i, j) * values[j];
      for (std::size_t k = 0; k < 6; ++k) {
        dot += vectors(k, i) * vectors(k, j);
        residual += symmetric(i, k) * vectors(k, j);
      }
      require(std::abs(dot - T(i == j)) < T(20) * tol,
              "Jacobi orthogonal eigenbasis");
      require(std::abs(residual) < T(20) * tol, "Jacobi eigenvector residual");
    }
  }
  EigenSolverSymmetric<T, 6, false> packed;
  configure_value_accuracy<T>(packed);
  packed.solve_eigen_values(symmetric);
  check_spectrum<T, 6>(packed.get_eigen_values(), "Jacobi packed values-only spectrum");
  require(sizeof(packed) < sizeof(jacobi), "values-only omits dense vectors");

  Matrix<T, 3, 3> repeated = Matrix<T, 3, 3>::identity();
  EigenSolverSymmetric<T, 3> repeated_solver;
  configure_value_accuracy<T>(repeated_solver);
  repeated_solver.solve_eigen_values_and_vectors(repeated);
  require(repeated_solver.eigen_values_converged(), "repeated eigenvalues");
  repeated_solver.relative_tolerance = T(-1);
  repeated_solver.absolute_tolerance = T(-1);
  repeated_solver.solve_eigen_values(zero);
  require(repeated_solver.get_eigen_values()[0] == T(0),
          "negative tolerance does not divide zero pivot");

  EigenSolverReal<T, 6, 3> restarted;
  configure_general_accuracy<T>(restarted);
  restarted.iteration_max = 100;
  restarted.gmres_cycles = 4;
  restarted.solve_eigen_values_and_vectors(dense);
  require(restarted.eigen_values_converged(), "reduced inner capacity retains QR spectrum");
  require(std::isfinite(restarted.get_eigenvector_residual()),
          "reduced inner capacity has finite diagnostic");

  Matrix<T, 2, 2> jordan({{1, 1}, {0, 1}});
  EigenSolverReal<T, 2> defective;
  configure_general_accuracy<T>(defective);
  defective.solve_eigen_values_and_vectors(jordan);
  require(defective.eigen_values_converged() &&
              defective.get_eigen_values()[0] == T(1) &&
              defective.get_eigen_values()[1] == T(1),
          "defective repeated spectrum does not require independent vectors");

  real.gmres_cycles = 0;
  real.solve_eigen_vectors(dense);
  require(!real.eigen_vectors_converged(), "zero inner budget cannot report success");
}

template <typename T> inline std::complex<double> host_value(const T &value) {
  return std::complex<double>(double(value), 0);
}

template <typename T>
inline std::complex<double> host_value(const Complex<T> &value) {
  return std::complex<double>(double(value.real), double(value.imag));
}

template <typename T, typename S, std::size_t N, typename Values>
inline double independent_residual(const Matrix<T, N, N> &matrix,
                                   const Values &values,
                                   const Matrix<S, N, N> &vectors) {
  double matrix_squared_norm = 0;
  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i < N; ++i) {
      matrix_squared_norm += double(matrix(i, j)) * double(matrix(i, j));
    }
  }
  double maximum = 0;
  for (std::size_t j = 0; j < N; ++j) {
    double vector_squared_norm = 0;
    double residual_squared_norm = 0;
    for (std::size_t i = 0; i < N; ++i) {
      std::complex<double> residual = -host_value(values[j]) * host_value(vectors(i, j));
      for (std::size_t k = 0; k < N; ++k) {
        residual += double(matrix(i, k)) * host_value(vectors(k, j));
      }
      vector_squared_norm += std::norm(host_value(vectors(i, j)));
      residual_squared_norm += std::norm(residual);
    }
    if (vector_squared_norm == 0 || matrix_squared_norm == 0) {
      return std::numeric_limits<double>::infinity();
    }
    double residual = std::sqrt(residual_squared_norm /
                                (matrix_squared_norm * vector_squared_norm));
    if (!std::isfinite(residual)) { return residual; }
    if (residual > maximum) { maximum = residual; }
  }
  return maximum;
}

template <typename T, typename S, std::size_t N, typename Values>
inline void check_default_eigenpairs(const Matrix<T, N, N> &matrix, T scale,
                                    const Values &values,
                                    const Matrix<S, N, N> &vectors) {
  // Check returned values against the known spectrum independently of solver flags.
  bool used[N] = {};
  for (std::size_t k = 0; k < N; ++k) {
    double best = std::numeric_limits<double>::max();
    std::size_t index = 0;
    for (std::size_t i = 0; i < N; ++i) {
      double error = std::abs(host_value(values[i]) / double(scale) - double(k + 1));
      if (!used[i] && error < best) {
        best = error;
        index = i;
      }
    }
    used[index] = true;
    require(best < DEFAULT_SOLUTION_QUALITY_LIMIT * double(N),
            "fixed defaults retain known scaled spectrum");
  }
  require(independent_residual(matrix, values, vectors) < DEFAULT_SOLUTION_QUALITY_LIMIT,
          "fixed defaults satisfy independent normalized eigenpair residual");
}

template <typename T, typename Solver>
inline void check_default_general_solver(const Matrix<T, 3, 3> &matrix, T scale) {
  Solver automatic;
  automatic.solve_eigen_values_and_vectors(matrix);
  require(automatic.eigen_values_converged() && automatic.eigen_vectors_converged() &&
              !automatic.inner_solve_failed(),
          "fixed general defaults report converged eigenpairs");
  check_default_eigenpairs(matrix, scale, automatic.get_eigen_values(),
                          automatic.get_eigen_vectors());

  // A nonpositive rate must select the same tolerance as its explicit value.
  Solver explicit_inner;
  explicit_inner.set_gmres_k_decay_rate(T(DEFAULT_INNER_GMRES_RELATIVE_TOLERANCE));
  explicit_inner.solve_eigen_values_and_vectors(matrix);
  require(explicit_inner.eigen_vectors_converged() && !explicit_inner.inner_solve_failed(),
          "explicit fixed inner tolerance converges");
  auto automatic_vectors = automatic.get_eigen_vectors();
  auto explicit_vectors = explicit_inner.get_eigen_vectors();
  for (std::size_t j = 0; j < 3; ++j) {
    for (std::size_t i = 0; i < 3; ++i) {
      require(host_value(automatic_vectors(i, j)) == host_value(explicit_vectors(i, j)),
              "automatic inner tolerance matches the explicit fixed tolerance");
    }
  }
}

template <typename T, typename Solver>
inline void check_default_tridiagonal_solver(const Matrix<T, 8, 8> &matrix) {
  // This input exposed a large eigenvector residual with the previous defaults.
  // Use the actual constructor defaults, including both iteration budgets.
  Solver solver;
  solver.solve_eigen_values_and_vectors(matrix);
  require(solver.eigen_values_converged() && solver.eigen_vectors_converged() &&
              !solver.inner_solve_failed(),
          "default 8x8 tridiagonal eigenpairs converge");
  auto values = solver.get_eigen_values();
  require(independent_residual(matrix, values, solver.get_eigen_vectors()) <
              DEFAULT_SOLUTION_QUALITY_LIMIT,
          "default 8x8 tridiagonal eigenvectors meet independent residual target");

  const double pi = std::acos(-1.0);
  const double matrix_norm = std::sqrt(46.0);
  bool used[8] = {};
  for (std::size_t mode = 0; mode < 8; ++mode) {
    double reference = 2.0 - 2.0 * std::cos(double(mode + 1) * pi / 9.0);
    double best = std::numeric_limits<double>::max();
    std::size_t index = 0;
    for (std::size_t slot = 0; slot < 8; ++slot) {
      double error = std::abs(host_value(values[slot]) - reference);
      if (!used[slot] && error < best) {
        best = error;
        index = slot;
      }
    }
    used[index] = true;
    require(best / matrix_norm < DEFAULT_SOLUTION_QUALITY_LIMIT,
            "default tridiagonal eigenvalues retain the analytic spectrum");
  }
}

template <typename T> inline void run_default_parameters() {
  Matrix<T, 8, 8> tridiagonal;
  for (std::size_t i = 0; i < 8; ++i) {
    tridiagonal(i, i) = T(2);
    if (i + 1 < 8) {
      tridiagonal(i, i + 1) = T(-1);
      tridiagonal(i + 1, i) = T(-1);
    }
  }
  check_default_tridiagonal_solver<T, EigenSolverReal<T, 8>>(tridiagonal);
  check_default_tridiagonal_solver<T, EigenSolverComplex<T, 8>>(tridiagonal);
  for (std::size_t kind = 0; kind < 3; ++kind) {
    T scale = kind == 1 ? T(1e-12) : (kind == 2 ? T(1e12) : T(1));
    Matrix<T, 3, 3> general = rotated_diagonal<T, 3>(true);
    Matrix<T, 3, 3> symmetric = rotated_diagonal<T, 3>();
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t i = 0; i < 3; ++i) {
        general(i, j) *= scale;
        symmetric(i, j) *= scale;
      }
    }
    check_default_general_solver<T, EigenSolverReal<T, 3>>(general, scale);
    check_default_general_solver<T, EigenSolverComplex<T, 3>>(general, scale);
    EigenSolverSymmetric<T, 3> jacobi;
    jacobi.solve_eigen_values_and_vectors(symmetric);
    require(jacobi.eigen_values_converged(), "fixed Jacobi defaults converge");
    check_default_eigenpairs(symmetric, scale, jacobi.get_eigen_values(),
                            jacobi.get_eigen_vectors());
  }
}

inline void check_explicit_small_shift() {
#ifdef BASE_MATH_USE_STD_MATH_
  Matrix<double, 3, 3> matrix = rotated_diagonal<double, 3>(true);
  EigenSolverReal<double, 3> solver;
  configure_general_accuracy<double>(solver);
  solver.solve_eigen_values_and_vectors(matrix);
  require(solver.small_value == HIGH_ACCURACY_RELATIVE_SHIFT,
          "explicit small shift is not replaced by the default shift");
  require(solver.eigen_values_converged() && solver.eigen_vectors_converged() &&
              !solver.inner_solve_failed(),
          "explicit precision parameters converge");
  require(independent_residual(matrix, solver.get_eigen_values(), solver.get_eigen_vectors()) <
              HIGH_ACCURACY_VECTOR_TOLERANCE,
          "explicit small shift achieves the independent high accuracy target");
#endif
}
} // namespace

int main() {
  run<float>();
  run<double>();
  run_default_parameters<float>();
  run_default_parameters<double>();
  check_explicit_small_shift();
  double tiny = EigenOperation::square_root(1e-300);
  require(std::abs(tiny / 1e-150 - 1.0) < 1e-12, "tiny normalized discriminant sqrt");
  std::printf("eigen regression: %d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
