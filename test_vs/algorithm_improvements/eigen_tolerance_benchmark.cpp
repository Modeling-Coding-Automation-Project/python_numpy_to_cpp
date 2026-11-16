// Host-only timing of the frozen A and common-candidate parameter profiles.
// Fixtures are copied unchanged from eigen_tolerance_probe.cpp.
// argv: profile (0=A, 1=candidate), optional repetitions override.
// One binary runs both profiles, preserving the same object and stack layout.
#include "base_matrix_eigen_solver.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_
#error "This study must preserve template operation expansion."
#endif

using namespace Base::Matrix;

template <typename T> inline double real_part(const T &value) {
  return static_cast<double>(value);
}
template <typename T> inline double imaginary_part(const T &) { return 0.0; }
template <typename T> inline double real_part(const Complex<T> &value) {
  return static_cast<double>(value.real);
}
template <typename T> inline double imaginary_part(const Complex<T> &value) {
  return static_cast<double>(value.imag);
}

template <typename T> inline Matrix<T, 3, 3> small_fixture(int kind) {
  Matrix<T, 3, 3> result;
  if (kind == 1) {
    // Similarity of an upper triangular matrix with spectrum 1, 2, 3.
    Matrix<T, 3, 3> q({{T(.8), T(-.48), T(.36)},
                        {T(.6), T(.64), T(-.48)},
                        {T(0), T(.6), T(.8)}});
    Matrix<T, 3, 3> triangular({{T(1), T(5), T(0)},
                                 {T(0), T(2), T(5)},
                                 {T(0), T(0), T(3)}});
    for (std::size_t i = 0; i < 3; ++i) {
      for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t k = 0; k < 3; ++k) {
          for (std::size_t l = 0; l < 3; ++l) {
            result(i, j) += q(i, k) * triangular(k, l) * q(j, l);
          }
        }
      }
    }
  } else if (kind == 2 || kind == 3) {
    const T delta = kind == 2 ? T(1e-6) : T(1e-10);
    result = Matrix<T, 3, 3>({{T(1), delta, T(0)},
                               {delta, T(1), T(0)},
                               {T(0), T(0), T(2)}});
  } else if (kind == 4) {
    result = Matrix<T, 3, 3>::identity();
  } else if (kind == 5) {
    result = Matrix<T, 3, 3>({{T(0), T(-1), T(0)},
                               {T(1), T(0), T(0)},
                               {T(0), T(0), T(2)}});
  } else if (kind == 6) {
    result = Matrix<T, 3, 3>({{T(0), T(0), T(1)},
                               {T(1), T(0), T(0)},
                               {T(0), T(1), T(0)}});
  } else {
    result = Matrix<T, 3, 3>({{T(4), T(1), T(.2)},
                               {T(1), T(3), T(.4)},
                               {T(.2), T(.4), T(2)}});
    const T scale = kind == 7 ? T(1e-12) : (kind == 8 ? T(1e12) : T(1));
    for (std::size_t i = 0; i < 3; ++i) {
      for (std::size_t j = 0; j < 3; ++j) { result(i, j) *= scale; }
    }
  }
  return result;
}

constexpr double FIXED_RELATIVE_SHIFT = 3e-4;
constexpr double FIXED_DIVISION_MIN = 1e-20;
constexpr double DEFAULT_VALUE_TOLERANCE = 8e-9;
constexpr double DEFAULT_VECTOR_TOLERANCE = 3.2e-6;
constexpr double DEFAULT_INNER_TOLERANCE = 5e-3;
static_assert(GMRES_BREAKDOWN_RELATIVE_TOLERANCE == 8e-12,
              "The breakdown coefficient must remain fixed during this study.");

template <typename T, std::size_t N>
inline Matrix<T, N, N> large_fixture(int kind) {
  Matrix<T, N, N> matrix;
  if (kind == 2) {
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
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t k = 0; k < N; ++k) {
          matrix(i, j) += orthogonal(i, k) * T(k + 1) * orthogonal(j, k);
          if (k + 1 < N) {
            matrix(i, j) += T(0.5) * orthogonal(i, k) * orthogonal(j, k + 1);
          }
        }
      }
    }
  } else if (kind == 3) {
    for (std::size_t i = 0; i < N; ++i) {
      matrix(i, i) = T(i / 2) / T(4);
      if (i % 2 == 0 && i + 1 < N) {
        matrix(i, i + 1) = T(-1);
        matrix(i + 1, i) = T(1);
      }
    }
    if (N % 2 != 0) { matrix(N - 1, N - 1) = T(N); }
  } else {
    for (std::size_t i = 0; i < N; ++i) {
      matrix(i, i) = T(2);
      if (i + 1 < N) {
        T first = kind == 1 && i % 2 ? T(2) : T(1);
        T second = kind == 1 && (i + 1) % 2 ? T(2) : T(1);
        matrix(i, i + 1) = -first / second;
        matrix(i + 1, i) = -second / first;
      }
    }
  }
  return matrix;
}


std::size_t repetitions_override = 0;
constexpr double CANDIDATE_VALUE_TOLERANCE = 1e-6;
constexpr double CANDIDATE_VECTOR_TOLERANCE = 4.8e-6;
constexpr double CANDIDATE_INNER_TOLERANCE = 1e-3;

inline std::size_t repetitions(std::size_t dimension) {
  return repetitions_override != 0 ? repetitions_override
      : (dimension <= 4 ? 100 : (dimension <= 8 ? 30 : 8));
}

template <typename Function>
inline double time_call(Function function, std::size_t repeat_count) {
  function();
  double samples[3];
  for (std::size_t trial = 0; trial < 3; ++trial) {
    auto begin = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < repeat_count; ++repeat) { function(); }
    auto end = std::chrono::steady_clock::now();
    samples[trial] = std::chrono::duration<double, std::nano>(end - begin).count() /
                     double(repeat_count);
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
inline void configure(Solver &solver, double relative, double vector, double inner) {
  solver.iteration_max = N == 3 ? 128 : 64;
  solver.iteration_max_for_eigen_vector = 30;
  solver.gmres_cycles = 1;
  solver.gmres_orthogonalization_passes = 1;
  solver.division_min = T(FIXED_DIVISION_MIN);
  solver.absolute_tolerance = T(0);
  solver.relative_tolerance = T(relative);
  solver.eigenvector_tolerance = T(vector);
  solver.set_small_value(T(FIXED_RELATIVE_SHIFT));
  solver.set_gmres_k_decay_rate(T(inner));
}

template <typename T, std::size_t N, typename Solver>
inline void general_case(const Matrix<T, N, N> &matrix, const char *fixture,
                         const char *algorithm, const char *profile,
                         double relative, double vector, double inner) {
  const std::size_t count = repetitions(N);
  double elapsed = time_call([&]() {
    asm volatile("" : : "g"(&matrix) : "memory");
    Solver fresh;
    configure<T, N>(fresh, relative, vector, inner);
    fresh.solve_eigen_values_and_vectors(matrix);
    asm volatile("" : : "g"(&fresh) : "memory");
  }, count);
  Solver result;
  configure<T, N>(result, relative, vector, inner);
  result.solve_eigen_values_and_vectors(matrix);
  std::printf("%s,%s,%zu,%s,%s,%s,%.3f,%d,%d,%d,%d,%zu,%zu,%zu,%.17g,%.17g,%.17g,%.17g\n",
              algorithm, sizeof(T) == sizeof(float) ? "float" : "double", N,
              fixture, N == 5 || N == 7 || N == 12 ? "holdout" : "search", profile,
              elapsed, int(result.eigen_values_converged()), int(result.eigen_vectors_converged()),
              int(result.inner_solve_failed()), int(result.get_status()), result.get_sweep_count(),
              sizeof(result), count, double(result.relative_tolerance),
              double(result.eigenvector_tolerance), double(result.gmres_k_decay_rate),
              double(result.small_value));
}

template <typename T, std::size_t N>
inline void symmetric_case(const Matrix<T, N, N> &matrix, const char *fixture,
                           const char *profile, double relative) {
  const std::size_t count = repetitions(N);
  double elapsed = time_call([&]() {
    asm volatile("" : : "g"(&matrix) : "memory");
    EigenSolverSymmetric<T, N> fresh;
    fresh.iteration_max = 10;
    fresh.division_min = T(FIXED_DIVISION_MIN);
    fresh.absolute_tolerance = T(0);
    fresh.relative_tolerance = T(relative);
    fresh.solve_eigen_values_and_vectors(matrix);
    asm volatile("" : : "g"(&fresh) : "memory");
  }, count);
  EigenSolverSymmetric<T, N> result;
  result.iteration_max = 10;
  result.division_min = T(FIXED_DIVISION_MIN);
  result.absolute_tolerance = T(0);
  result.relative_tolerance = T(relative);
  result.solve_eigen_values_and_vectors(matrix);
  std::printf("jacobi,%s,%zu,%s,%s,%s,%.3f,%d,-1,-1,-1,-1,%zu,%zu,%.17g,0,0,0\n",
              sizeof(T) == sizeof(float) ? "float" : "double", N, fixture,
              N == 5 || N == 7 || N == 12 ? "holdout" : "search", profile,
              elapsed, int(result.eigen_values_converged()), sizeof(result), count,
              double(result.relative_tolerance));
}

template <typename T>
inline void small_cases(const char *profile, double relative, double vector, double inner) {
  const char *names[] = {"ordinary", "nonnormal", "clustered_1e-6", "clustered_1e-10",
                         "repeated", "complex_pair", "cyclic", "scaled_1e-12", "scaled_1e12"};
  for (int kind = 0; kind < 9; ++kind) {
    const auto matrix = small_fixture<T>(kind);
    if (kind != 5 && kind != 6) {
      general_case<T, 3, EigenSolverReal<T, 3>>(matrix, names[kind], "real", profile, relative, vector, inner);
    }
    general_case<T, 3, EigenSolverComplex<T, 3>>(matrix, names[kind], "complex", profile, relative, vector, inner);
    if (kind != 1 && kind != 5 && kind != 6) { symmetric_case(matrix, names[kind], profile, relative); }
  }
}

template <typename T, std::size_t N>
inline void large_cases(const char *profile, double relative, double vector, double inner) {
  const char *names[] = {"tridiagonal", "similarity", "rotated_nonnormal", "complex_blocks"};
  for (int kind = 0; kind < 4; ++kind) {
    const auto matrix = large_fixture<T, N>(kind);
    if (kind != 3) {
      general_case<T, N, EigenSolverReal<T, N>>(matrix, names[kind], "real", profile, relative, vector, inner);
    }
    general_case<T, N, EigenSolverComplex<T, N>>(matrix, names[kind], "complex", profile, relative, vector, inner);
    if (kind == 0) { symmetric_case(matrix, names[kind], profile, relative); }
  }
}

template <typename T>
inline void run(const char *profile, double relative, double vector, double inner) {
  small_cases<T>(profile, relative, vector, inner);
  large_cases<T, 4>(profile, relative, vector, inner);
  large_cases<T, 8>(profile, relative, vector, inner);
  large_cases<T, 16>(profile, relative, vector, inner);
  large_cases<T, 5>(profile, relative, vector, inner);
  large_cases<T, 7>(profile, relative, vector, inner);
  large_cases<T, 12>(profile, relative, vector, inner);
}

int main(int argc, char **argv) {
  const bool candidate = argc > 1 && std::atoi(argv[1]) == 1;
  if (argc > 2) { repetitions_override = std::strtoul(argv[2], nullptr, 10); }
  const char *profile = candidate ? "candidate_common" : "current_a";
  const double relative = candidate ? CANDIDATE_VALUE_TOLERANCE : DEFAULT_VALUE_TOLERANCE;
  const double vector = candidate ? CANDIDATE_VECTOR_TOLERANCE : DEFAULT_VECTOR_TOLERANCE;
  const double inner = candidate ? CANDIDATE_INNER_TOLERANCE : DEFAULT_INNER_TOLERANCE;
  std::puts("algorithm,type,n,fixture,corpus,profile,ns,values_converged,vectors_converged,"
            "inner_failed,status,qr_sweeps,object_bytes,repetitions,rtol,vtol,inner,shift");
  run<float>(profile, relative, vector, inner);
  run<double>(profile, relative, vector, inner);
}
