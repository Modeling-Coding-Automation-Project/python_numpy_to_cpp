// Host-only diagnostics for the 108 configured rows of gmres_benchmark.cpp.
// Keep the fixtures, RHS rounding, 1000 solves, and six configurations aligned
// with that benchmark. This program measures solver work, not elapsed time.
// Build the same source against each revision without changing its defaults.
#include "base_matrix_inverse.hpp"

#include <cmath>
#include <cstdio>

using namespace Base::Matrix;

namespace {
constexpr std::size_t DIAGNOSTIC_REPEATS = 1000;

template <typename T, std::size_t N>
inline Matrix<T, N, N> diagnostic_fixture(int kind) {
  Matrix<T, N, N> matrix;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      if (kind == 1) {
        matrix(i, j) = T(1) / T(i + j + 1);
      } else if (kind == 2) {
        T scale = T(1);
        for (std::size_t k = 0; k < j; ++k) {
          scale *= T(2);
        }
        matrix(i, j) =
            (i == j ? T(4) : T((i + j) % 3) * T(0.01)) * scale;
      } else {
        matrix(i, j) =
            i == j ? T(4) + T(i) / T(N) : T((i * 7 + j * 3) % 5) * T(0.03);
      }
    }
  }
  return matrix;
}

template <typename T, std::size_t N>
inline double target_error(const Vector<T, N> &solution,
                           const Vector<T, N> &target) {
  double sum = 0;
  for (std::size_t i = 0; i < N; ++i) {
    const double delta = double(solution[i]) - double(target[i]);
    sum += delta * delta;
  }
  return std::sqrt(sum / double(N));
}

// Compute the final residual from the actual stored inputs and solution.
// This deliberately does not call the library's product or norm routines.
// Neither residual nor distance to the manufactured target is a forward-error
// oracle for ill-conditioned rounded systems; use the Decimal oracle for that.
template <typename Precision, typename T, std::size_t N>
inline void residual_metrics(const Matrix<T, N, N> &matrix,
                             const Vector<T, N> &rhs,
                             const Vector<T, N> &solution,
                             Precision &absolute, Precision &relative) {
  Precision residual_square = Precision(0);
  Precision rhs_square = Precision(0);
  for (std::size_t i = 0; i < N; ++i) {
    Precision product = Precision(0);
    for (std::size_t j = 0; j < N; ++j) {
      product += static_cast<Precision>(matrix(i, j)) *
                 static_cast<Precision>(solution[j]);
    }
    const Precision input = static_cast<Precision>(rhs[i]);
    const Precision residual = input - product;
    residual_square += residual * residual;
    rhs_square += input * input;
  }
  absolute = std::sqrt(residual_square);
  const Precision rhs_norm = std::sqrt(rhs_square);
  relative = rhs_norm > Precision(0) ? absolute / rhs_norm : absolute;
}

template <typename T, std::size_t N, std::size_t K>
inline void configured_diagnostics(int kind, std::size_t cycles,
                                   std::size_t passes, bool precondition) {
  static_assert(N > 0 && K > 0, "Diagnostics require nonempty dimensions.");
  const auto matrix = diagnostic_fixture<T, N>(kind);
  Vector<T, N> target;
  for (std::size_t i = 0; i < N; ++i) {
    target[i] = T(1);
  }
  auto rhs = matrix * target;
  Vector<T, N> zero, solution;
  const auto diagonal = gmres_inverse_diagonal(matrix, T(1e-25));
  GmresSolver<T, N, K> solver;
  solver.cycles = cycles;
  solver.orthogonalization_passes = passes;
  solver.division_min = T(1e-25);

  const std::size_t iteration_cap = K * cycles;
  std::size_t total_iterations = 0;
  std::size_t minimum_iterations = 0;
  std::size_t maximum_iterations = 0;
  std::size_t converged_count = 0;
  std::size_t breakdown_count = 0;
  std::size_t budget_violation_count = 0;
  for (std::size_t repeat = 0; repeat < DIAGNOSTIC_REPEATS; ++repeat) {
    rhs[0] += T((repeat & 1) ? -1e-6 : 1e-6);
    solution = precondition
                   ? solver.solve_preconditioned(matrix, rhs, zero, diagonal)
                   : solver.solve(matrix, rhs, zero);
    const std::size_t iterations = solver.get_iterations();
    total_iterations += iterations;
    if (repeat == 0 || iterations < minimum_iterations) {
      minimum_iterations = iterations;
    }
    if (iterations > maximum_iterations) {
      maximum_iterations = iterations;
    }
    if (iterations > iteration_cap) {
      ++budget_violation_count;
    }
    converged_count += solver.is_converged() ? 1 : 0;
    breakdown_count += solver.has_breakdown() ? 1 : 0;
  }

  double absolute_double = 0;
  double relative_double = 0;
  long double absolute_long_double = 0;
  long double relative_long_double = 0;
  residual_metrics(matrix, rhs, solution, absolute_double, relative_double);
  residual_metrics(matrix, rhs, solution,
                   absolute_long_double, relative_long_double);
  const std::size_t preconditioner_bytes = precondition ? sizeof(diagonal) : 0;

  std::printf("configured,%s,%zu,%d,%zu,%zu,%zu,%d,%zu,%zu,%.9g,%zu,%zu,"
              "%zu,%zu,%zu,%zu,%zu,%d,%d,%.17g,%.17g,%.17g,%.17g,"
              "%.21Lg,%.21Lg,%zu,%zu,%zu,%.17g,%.17g,%.17g\n",
              sizeof(T) == sizeof(float) ? "float" : "double",
              N, kind, K, cycles, passes, int(precondition),
              DIAGNOSTIC_REPEATS, total_iterations,
              double(total_iterations) / double(DIAGNOSTIC_REPEATS),
              minimum_iterations, maximum_iterations, iteration_cap,
              budget_violation_count, converged_count, breakdown_count,
              solver.get_iterations(), int(solver.is_converged()),
              int(solver.has_breakdown()), target_error(solution, target),
              double(solver.get_residual_norm()), absolute_double,
              relative_double, absolute_long_double, relative_long_double,
              sizeof(solver), preconditioner_bytes,
              sizeof(solver) + preconditioner_bytes,
              double(solver.relative_tolerance), double(solver.absolute_tolerance),
              double(solver.division_min));
}

template <typename T, std::size_t N> inline void diagnostic_rows() {
  for (int kind = 0; kind < 3; ++kind) {
    configured_diagnostics<T, N, N>(kind, 1, 1, false);
    configured_diagnostics<T, N, N>(kind, 1, 2, false);
    configured_diagnostics<T, N, N / 2>(kind, 1, 1, false);
    configured_diagnostics<T, N, N / 2>(kind, 4, 1, false);
    configured_diagnostics<T, N, N / 2>(kind, 4, 2, false);
    configured_diagnostics<T, N, N / 2>(kind, 4, 1, true);
  }
}
} // namespace

int main() {
  std::puts("mode,type,n,fixture,k,cycles,passes,precondition,repeats,"
            "total_iterations,mean_iterations,min_iterations,max_iterations,"
            "iteration_cap,budget_violation_count,converged_count,breakdown_count,"
            "last_iterations,last_converged,last_breakdown,rms_target_error,"
            "reported_residual,true_residual_double,relative_true_residual_double,"
            "true_residual_long_double,relative_true_residual_long_double,"
            "solver_bytes,preconditioner_bytes,object_bytes,relative_tolerance,"
            "absolute_tolerance,division_min");
  diagnostic_rows<float, 4>();
  diagnostic_rows<double, 4>();
  diagnostic_rows<float, 8>();
  diagnostic_rows<double, 8>();
  diagnostic_rows<float, 16>();
  diagnostic_rows<double, 16>();
}
