// Host-only balanced-default selection on all 140 formerly measured cases.
// argv: rtol vtol inner reference_mode (0=fixed, 1=old tolerances, 2=old full profile).
// BALANCED_EIGEN_VALIDATION selects new N6/10/14 after candidate selection.
#include "base_matrix_eigen_solver.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>

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

int reference_mode = 0;
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

template <typename T, typename S, std::size_t N, typename Values>
inline void output(const char *algorithm, const char *fixture,
                   const Matrix<T, N, N> &matrix, const Values &values,
                   const Matrix<S, N, N> &vectors, bool values_converged,
                   int vectors_converged, int inner_failed, int status,
                   long sweeps, std::size_t bytes, T relative, T vector,
                   T shift, T inner, T internal_residual) {
  std::printf("%s,%s,%zu,%s,%d,%d,%d,%d,%ld,%zu,%.17g,%.17g,%.17g,%.17g,%.17g",
              sizeof(T) == sizeof(float) ? "float" : "double", algorithm, N,
              fixture, int(values_converged), vectors_converged, inner_failed,
              status, sweeps, bytes, double(relative), double(vector),
              double(shift), double(inner), double(internal_residual));
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) { std::printf(",%a", double(matrix(i, j))); }
  }
  for (std::size_t i = 0; i < N; ++i) {
    std::printf(",%a,%a", real_part(values[i]), imaginary_part(values[i]));
  }
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      std::printf(",%a,%a", real_part(vectors(i, j)), imaginary_part(vectors(i, j)));
    }
  }
  std::putchar('\n');
}

template <typename T, std::size_t N, typename Solver>
inline void general_case(const Matrix<T, N, N> &matrix, const char *fixture,
                         const char *algorithm, double relative, double vector,
                         double inner) {
  Solver solver;
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
  if (reference_mode != 0) {
    solver.relative_tolerance = T(8) * std::numeric_limits<T>::epsilon();
    solver.eigenvector_tolerance = T(32) * std::numeric_limits<T>::epsilon();
    const T precision_shift = EigenOperation::square_root(std::numeric_limits<T>::epsilon());
    solver.set_gmres_k_decay_rate(T(16) * precision_shift);
    if (reference_mode == 2) {
      solver.set_small_value(precision_shift > T(1e-6) ? precision_shift : T(1e-6));
    }
  }
  solver.solve_eigen_values_and_vectors(matrix);
  output(algorithm, fixture, matrix, solver.get_eigen_values(), solver.get_eigen_vectors(),
         solver.eigen_values_converged(), int(solver.eigen_vectors_converged()),
         int(solver.inner_solve_failed()), int(solver.get_status()),
         long(solver.get_sweep_count()), sizeof(solver), solver.relative_tolerance,
         solver.eigenvector_tolerance, solver.small_value, solver.gmres_k_decay_rate,
         solver.get_eigenvector_residual());
}

template <typename T, std::size_t N>
inline void symmetric_case(const Matrix<T, N, N> &matrix, const char *fixture,
                           double relative) {
  EigenSolverSymmetric<T, N> solver;
  solver.iteration_max = 10;
  solver.division_min = T(FIXED_DIVISION_MIN);
  solver.absolute_tolerance = T(0);
  solver.relative_tolerance = reference_mode != 0
      ? T(8) * std::numeric_limits<T>::epsilon() : T(relative);
  solver.solve_eigen_values_and_vectors(matrix);
  output("jacobi", fixture, matrix, solver.get_eigen_values(), solver.get_eigen_vectors(),
         solver.eigen_values_converged(), -1, -1, -1, -1, sizeof(solver),
         solver.relative_tolerance, T(0), T(0), T(0), T(-1));
}

template <typename T>
inline void small_cases(double relative, double vector, double inner) {
  const char *names[] = {"ordinary", "nonnormal", "clustered_1e-6", "clustered_1e-10",
                         "repeated", "complex_pair", "cyclic", "scaled_1e-12", "scaled_1e12"};
  for (int kind = 0; kind < 9; ++kind) {
    const auto matrix = small_fixture<T>(kind);
    if (kind != 5 && kind != 6) {
      general_case<T, 3, EigenSolverReal<T, 3>>(matrix, names[kind], "real", relative, vector, inner);
    }
    general_case<T, 3, EigenSolverComplex<T, 3>>(matrix, names[kind], "complex", relative, vector, inner);
    if (kind != 1 && kind != 5 && kind != 6) { symmetric_case(matrix, names[kind], relative); }
  }
}

template <typename T, std::size_t N>
inline void large_cases(double relative, double vector, double inner) {
  const char *names[] = {"tridiagonal", "similarity", "rotated_nonnormal", "complex_blocks"};
  for (int kind = 0; kind < 4; ++kind) {
    const auto matrix = large_fixture<T, N>(kind);
    if (kind != 3) {
      general_case<T, N, EigenSolverReal<T, N>>(matrix, names[kind], "real", relative, vector, inner);
    }
    general_case<T, N, EigenSolverComplex<T, N>>(matrix, names[kind], "complex", relative, vector, inner);
    if (kind == 0) { symmetric_case(matrix, names[kind], relative); }
  }
}

template <typename T> inline void run(double relative, double vector, double inner) {
#ifdef BALANCED_EIGEN_VALIDATION
  large_cases<T, 6>(relative, vector, inner);
  large_cases<T, 10>(relative, vector, inner);
  large_cases<T, 14>(relative, vector, inner);
#else
  small_cases<T>(relative, vector, inner);
  large_cases<T, 4>(relative, vector, inner);
  large_cases<T, 8>(relative, vector, inner);
  large_cases<T, 16>(relative, vector, inner);
  large_cases<T, 5>(relative, vector, inner);
  large_cases<T, 7>(relative, vector, inner);
  large_cases<T, 12>(relative, vector, inner);
#endif
}

int main(int argc, char **argv) {
  const double relative = argc > 1 ? std::strtod(argv[1], nullptr) : DEFAULT_VALUE_TOLERANCE;
  const double vector = argc > 2 ? std::strtod(argv[2], nullptr) : DEFAULT_VECTOR_TOLERANCE;
  const double inner = argc > 3 ? std::strtod(argv[3], nullptr) : DEFAULT_INNER_TOLERANCE;
  reference_mode = argc > 4 ? std::atoi(argv[4]) : 0;
  run<float>(relative, vector, inner);
  run<double>(relative, vector, inner);
}
