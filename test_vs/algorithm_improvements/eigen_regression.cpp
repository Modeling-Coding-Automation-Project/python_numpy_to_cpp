#include "base_matrix_eigen_solver.hpp"

#include <cmath>
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
  one.iteration_max = 0;
  one.solve_eigen_values_and_vectors(single);
  require(one.eigen_values_converged() && one.get_eigen_values()[0] == T(7),
          "1x1, zero QR budget");
  require(one.eigen_vectors_converged(), "1x1 inverse iteration");
  EigenSolverComplex<T, 1> complex_one;
  complex_one.solve_eigen_values_and_vectors(single);
  require(complex_one.eigen_values_converged() &&
              complex_one.get_eigen_values()[0].real == T(7),
          "complex 1x1 input");

  Matrix<T, 3, 3> zero;
  EigenSolverReal<T, 3> zeros;
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
  active_solver.solve_eigen_values(active);
  auto active_values = active_solver.get_eigen_values();
  require(std::abs(active_values[2] - T(100)) < tol, "deflated trailing value");
  require(std::abs(active_values[0] + active_values[1] - T(4)) < tol,
          "active block trace");
  require(std::abs(active_values[0] * active_values[1] - T(3)) < tol,
          "active block correct shift");

  Matrix<T, 6, 6> dense = rotated_diagonal<T, 6>(true);
  EigenSolverReal<T, 6> real;
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
  cannot_represent.solve_eigen_values(imaginary);
  require(cannot_represent.get_status() == EigenSolverStatus::complex_pair &&
              !cannot_represent.eigen_values_converged(),
          "real API reports complex spectrum");
  EigenSolverComplex<T, 2> complex;
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
  packed.solve_eigen_values(symmetric);
  check_spectrum<T, 6>(packed.get_eigen_values(), "Jacobi packed values-only spectrum");
  require(sizeof(packed) < sizeof(jacobi), "values-only omits dense vectors");

  Matrix<T, 3, 3> repeated = Matrix<T, 3, 3>::identity();
  EigenSolverSymmetric<T, 3> repeated_solver;
  repeated_solver.solve_eigen_values_and_vectors(repeated);
  require(repeated_solver.eigen_values_converged(), "repeated eigenvalues");
  repeated_solver.relative_tolerance = T(-1);
  repeated_solver.absolute_tolerance = T(-1);
  repeated_solver.solve_eigen_values(zero);
  require(repeated_solver.get_eigen_values()[0] == T(0),
          "negative tolerance does not divide zero pivot");

  EigenSolverReal<T, 6, 3> restarted;
  restarted.iteration_max = 100;
  restarted.gmres_cycles = 4;
  restarted.solve_eigen_values_and_vectors(dense);
  require(restarted.eigen_values_converged(), "reduced inner capacity retains QR spectrum");
  require(std::isfinite(restarted.get_eigenvector_residual()),
          "reduced inner capacity has finite diagnostic");

  Matrix<T, 2, 2> jordan({{1, 1}, {0, 1}});
  EigenSolverReal<T, 2> defective;
  defective.solve_eigen_values_and_vectors(jordan);
  require(defective.eigen_values_converged() &&
              defective.get_eigen_values()[0] == T(1) &&
              defective.get_eigen_values()[1] == T(1),
          "defective repeated spectrum does not require independent vectors");

  real.gmres_cycles = 0;
  real.solve_eigen_vectors(dense);
  require(!real.eigen_vectors_converged(), "zero inner budget cannot report success");
}
} // namespace

int main() {
  run<float>();
  run<double>();
  double tiny = EigenOperation::square_root(1e-300);
  require(std::abs(tiny / 1e-150 - 1.0) < 1e-12, "tiny normalized discriminant sqrt");
  std::printf("eigen regression: %d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
