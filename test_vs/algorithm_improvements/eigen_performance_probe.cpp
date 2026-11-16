// Host-only comparison of Eigen defaults and explicit accuracy settings.
// Compile this identical source against each revision; GCC/Clang barriers keep
// the input opaque and retain the entire solver output in every timed solve.
#include "base_matrix_eigen_solver.hpp"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>

using namespace Base::Matrix;

namespace {
constexpr double PROFILE_A_VALUE_TOLERANCE = 8e-9;
constexpr double PROFILE_A_VECTOR_TOLERANCE = 3.2e-6;
constexpr double PROFILE_A_RELATIVE_SHIFT = 3e-4;
constexpr double PROFILE_B_VALUE_TOLERANCE = 8e-12;
constexpr double PROFILE_B_VECTOR_TOLERANCE = 3.2e-10;
constexpr double PROFILE_B_RELATIVE_SHIFT = 1e-11;
constexpr double EXPLICIT_INNER_TOLERANCE = 5e-3;
constexpr double DIVISION_MINIMUM = 1e-20;
constexpr std::size_t QR_BUDGET = 64;
constexpr std::size_t INVERSE_BUDGET = 30;
std::size_t benchmark_repetitions = 0;

template <typename T, std::size_t N>
inline Matrix<T, N, N> input_matrix(bool similarity) {
  // L has exact eigenvalues 2 - 2*cos(k*pi/(N+1)). D*L*D^-1 has the same
  // spectrum, with D alternating between 1 and 2. Every input is exact in T.
  Matrix<T, N, N> matrix;
  for (std::size_t i = 0; i < N; ++i) {
    matrix(i, i) = T(2);
    if (i + 1 < N) {
      T first = similarity && i % 2 ? T(2) : T(1);
      T second = similarity && (i + 1) % 2 ? T(2) : T(1);
      matrix(i, i + 1) = -first / second;
      matrix(i + 1, i) = -second / first;
    }
  }
  return matrix;
}

template <typename T, typename Solver>
inline void configure(Solver &solver, int profile) {
  solver.iteration_max = QR_BUDGET;
  solver.iteration_max_for_eigen_vector = INVERSE_BUDGET;
  solver.gmres_cycles = 1;
  solver.gmres_orthogonalization_passes = 1;
  solver.division_min = T(DIVISION_MINIMUM);
  solver.absolute_tolerance = T(0);
  if (profile != 0) {
    solver.relative_tolerance = T(profile == 1 ? PROFILE_A_VALUE_TOLERANCE
                                             : PROFILE_B_VALUE_TOLERANCE);
    solver.eigenvector_tolerance = T(profile == 1 ? PROFILE_A_VECTOR_TOLERANCE
                                                : PROFILE_B_VECTOR_TOLERANCE);
    solver.set_small_value(T(profile == 1 ? PROFILE_A_RELATIVE_SHIFT
                                         : PROFILE_B_RELATIVE_SHIFT));
    solver.set_gmres_k_decay_rate(T(EXPLICIT_INNER_TOLERANCE));
  }
}

template <typename T>
inline std::complex<long double> host_value(const T &value) {
  return std::complex<long double>(static_cast<long double>(value), 0);
}

template <typename T>
inline std::complex<long double> host_value(const Complex<T> &value) {
  return std::complex<long double>(static_cast<long double>(value.real),
                                   static_cast<long double>(value.imag));
}

template <typename T, std::size_t N>
inline long double squared_norm(const Matrix<T, N, N> &matrix) {
  long double sum = 0;
  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i < N; ++i) {
      long double value = matrix(i, j);
      sum += value * value;
    }
  }
  return sum;
}

template <std::size_t N, typename Values>
inline double spectrum_error(const Values &values, long double matrix_norm) {
  bool used[N] = {};
  long double maximum = 0;
  const long double pi = std::acos(-1.0L);
  for (std::size_t k = 0; k < N; ++k) {
    const long double truth = 2 - 2 * std::cos((k + 1) * pi / (N + 1));
    long double best = std::numeric_limits<long double>::infinity();
    std::size_t selected = 0;
    for (std::size_t i = 0; i < N; ++i) {
      long double error = std::abs(host_value(values[i]) - truth);
      if (!std::isfinite(error)) { return std::numeric_limits<double>::infinity(); }
      if (!used[i] && error < best) {
        best = error;
        selected = i;
      }
    }
    used[selected] = true;
    if (best > maximum) { maximum = best; }
  }
  return double(maximum / matrix_norm);
}

template <typename T, typename S, std::size_t N, typename Values>
inline double vector_error(const Matrix<T, N, N> &matrix, const Values &values,
                           const Matrix<S, N, N> &vectors) {
  const long double matrix_square = squared_norm(matrix);
  long double maximum = 0;
  for (std::size_t j = 0; j < N; ++j) {
    long double vector_square = 0;
    long double residual_square = 0;
    for (std::size_t i = 0; i < N; ++i) {
      auto residual = -host_value(values[j]) * host_value(vectors(i, j));
      for (std::size_t k = 0; k < N; ++k) {
        residual += static_cast<long double>(matrix(i, k)) * host_value(vectors(k, j));
      }
      vector_square += std::norm(host_value(vectors(i, j)));
      residual_square += std::norm(residual);
    }
    if (vector_square == 0) { return std::numeric_limits<double>::infinity(); }
    long double error = std::sqrt(residual_square / (matrix_square * vector_square));
    if (!std::isfinite(error)) { return std::numeric_limits<double>::infinity(); }
    if (error > maximum) { maximum = error; }
  }
  return double(maximum);
}

template <typename Function>
inline double time_call(Function function, std::size_t repetitions) {
  function();
  double samples[3];
  for (std::size_t trial = 0; trial < 3; ++trial) {
    const auto begin = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
      function();
    }
    const auto end = std::chrono::steady_clock::now();
    samples[trial] = std::chrono::duration<double, std::nano>(end - begin).count() /
                     double(repetitions);
  }
  for (std::size_t i = 1; i < 3; ++i) {
    for (std::size_t j = i; j > 0; --j) {
      if (samples[j] < samples[j - 1]) {
        double temporary = samples[j];
        samples[j] = samples[j - 1];
        samples[j - 1] = temporary;
      }
    }
  }
  return samples[1];
}

template <typename T, std::size_t N, typename Solver>
inline void run_solver(const char *algorithm, bool similarity, int profile,
                       bool vectors) {
  const auto matrix = input_matrix<T, N>(similarity);
  Solver reusable;
  configure<T>(reusable, profile);
  const std::size_t repetitions = benchmark_repetitions != 0
      ? benchmark_repetitions : (N <= 4 ? 80 : (N <= 8 ? 30 : 8));
  double ns = time_call([&]() {
    // The barrier prevents an invariant input from enabling solve hoisting.
    asm volatile("" : : "g"(&matrix) : "memory");
    if (vectors) {
      // Full solves include constructor/configuration and do not warm-start.
      Solver fresh;
      configure<T>(fresh, profile);
      fresh.solve_eigen_values_and_vectors(matrix);
      asm volatile("" : : "g"(&fresh) : "memory");
    } else {
      reusable.solve_eigen_values(matrix);
      asm volatile("" : : "g"(&reusable) : "memory");
    }
  }, repetitions);

  // Quality and status are measured separately from the timed region.
  Solver result;
  configure<T>(result, profile);
  if (vectors) {
    result.solve_eigen_values_and_vectors(matrix);
  } else {
    result.solve_eigen_values(matrix);
  }
  const double value_error = spectrum_error<N>(result.get_eigen_values(),
                                               std::sqrt(squared_norm(matrix)));
  const double residual = vectors ? vector_error(matrix, result.get_eigen_values(),
                                                 result.get_eigen_vectors()) : -1;
  std::printf("%s,%s,%zu,%s,%s,%s,%.3f,%.12g,%.12g,%d,%d,%d,%d,%zu,%zu,"
              "%zu,%zu,%zu,%zu,%zu,%zu,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%d\n",
              algorithm, std::is_same<T, float>::value ? "float" : "double", N,
              similarity ? "similarity" : "symmetric",
              profile == 0 ? "defaults" : (profile == 1 ? "explicit_a" : "explicit_b"),
              vectors ? "values_vectors_fresh" : "values_reuse", ns,
              value_error, residual, int(result.eigen_values_converged()),
              int(result.eigen_vectors_converged()), int(result.inner_solve_failed()),
              int(result.get_status()), result.get_sweep_count(), sizeof(result),
              repetitions, result.iteration_max, result.iteration_max_for_eigen_vector,
              result.gmres_k_dimension, result.gmres_cycles,
              result.gmres_orthogonalization_passes, double(result.relative_tolerance),
              double(result.eigenvector_tolerance), double(result.small_value),
              double(result.gmres_k_decay_rate), double(result.division_min),
              double(result.absolute_tolerance),
              int(std::isfinite(value_error) && (!vectors || std::isfinite(residual))));
}

template <typename T, std::size_t N> inline void run() {
  for (int fixture = 0; fixture < 2; ++fixture) {
    for (int profile = 0; profile < 3; ++profile) {
      for (int vectors = 0; vectors < 2; ++vectors) {
        run_solver<T, N, EigenSolverReal<T, N>>("real", fixture != 0, profile, vectors != 0);
        run_solver<T, N, EigenSolverComplex<T, N>>("complex", fixture != 0, profile, vectors != 0);
      }
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc > 1) { benchmark_repetitions = std::strtoul(argv[1], nullptr, 10); }
  std::puts("algorithm,type,n,fixture,profile,stage,ns,value_error,vector_residual,"
            "values_converged,vectors_converged,inner_failed,status,qr_sweeps,object_bytes,"
            "repetitions,qr_budget,vector_budget,inner_k,inner_cycles,inner_passes,rtol,vtol,shift,"
            "inner_requested,division_min,absolute_tolerance,finite_output");
  run<float, 4>();
  run<double, 4>();
  run<float, 8>();
  run<double, 8>();
  run<float, 16>();
  run<double, 16>();
}
