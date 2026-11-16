// Host-only single-factor diagnosis derived from eigen_performance_probe.cpp.
// No timing. Only this probe changes the four public runtime parameters.
// Machine epsilon below reconstructs the OLD float defaults for comparison.
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


template <typename Solver>
inline void diagnose(const char *algorithm, int profile) {
  typedef float T;
  const auto matrix = input_matrix<T, 8>(false);
  Solver solver;
  configure<T>(solver, 1);
  const T old_relative = T(8) * std::numeric_limits<T>::epsilon();
  const T old_vector = T(32) * std::numeric_limits<T>::epsilon();
  const T old_shift = EigenOperation::square_root(std::numeric_limits<T>::epsilon());
  const T old_inner = T(16) * old_shift;
  if (profile == 1 || profile == 5) { solver.relative_tolerance = old_relative; }
  if (profile == 2 || profile == 5) { solver.eigenvector_tolerance = old_vector; }
  if (profile == 3 || profile == 5) { solver.small_value = old_shift; }
  if (profile == 4 || profile == 5) { solver.gmres_k_decay_rate = old_inner; }
  solver.solve_eigen_values_and_vectors(matrix);
  auto values = solver.get_eigen_values();
  auto vectors = solver.get_eigen_vectors();
  const char *names[] = {"current_a", "old_rtol", "old_vtol", "old_shift", "old_inner", "all_old_runtime"};
  std::printf("%s,%s,%.17g,%.17g,%.17g,%.17g,%.12g,%.12g,%d,%d,%d,%d,%zu,",
              algorithm, names[profile], double(solver.relative_tolerance),
              double(solver.eigenvector_tolerance), double(solver.small_value),
              double(solver.gmres_k_decay_rate),
              spectrum_error<8>(values, std::sqrt(squared_norm(matrix))),
              vector_error(matrix, values, vectors),
              int(solver.eigen_values_converged()), int(solver.eigen_vectors_converged()),
              int(solver.inner_solve_failed()), int(solver.get_status()), solver.get_sweep_count());
  // Determine which exact sine mode is assigned to each eigenvalue slot.
  // Compare its projection on the actual deterministic seed: ones + e_slot.
  const long double pi = std::acos(-1.0L);
  constexpr long double PROJECTION_ZERO_LIMIT = 1e-12L;
  std::size_t zero_projection_count = 0;
  for (std::size_t slot = 0; slot < 8; ++slot) {
    std::size_t mode = 1;
    long double minimum = std::numeric_limits<long double>::infinity();
    for (std::size_t candidate = 1; candidate <= 8; ++candidate) {
      long double truth = 2 - 2 * std::cos(candidate * pi / 9);
      long double error = std::abs(host_value(values[slot]) - truth);
      if (error < minimum) { minimum = error; mode = candidate; }
    }
    long double projection = 0;
    long double mode_square = 0;
    for (std::size_t i = 0; i < 8; ++i) {
      long double component = std::sin((i + 1) * mode * pi / 9);
      projection += component * (i == slot ? 2 : 1);
      mode_square += component * component;
    }
    projection = std::abs(projection) / std::sqrt(mode_square * 11);
    if (projection < PROJECTION_ZERO_LIMIT) { ++zero_projection_count; }
    std::printf(slot == 0 ? "%zu" : ":%zu", mode);
  }
  std::printf(",%zu\n", zero_projection_count);
}
} // namespace

int main() {
  std::puts("algorithm,profile,rtol,vtol,shift,inner,value_error,vector_residual,values_converged,vectors_converged,inner_failed,status,qr_sweeps,mode_order,zero_seed_projection_count");
  for (int profile = 0; profile < 6; ++profile) {
    diagnose<EigenSolverReal<float, 8>>("real", profile);
    diagnose<EigenSolverComplex<float, 8>>("complex", profile);
  }
}
