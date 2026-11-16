// Host-only comparison of independent final tolerances, shift and breakdown.
// TEST_BREAKDOWN is supplied only to an isolated copy of the GMRES header.
#include "base_matrix_eigen_solver.hpp"
#include <cstdio>

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

template <typename T> inline Matrix<T, 3, 3> fixture(int kind) {
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

template <typename T, typename Values, typename S>
inline void output(std::size_t profile, double inner_tolerance,
                   const char *algorithm, int kind,
                   const Matrix<T, 3, 3> &matrix, const Values &values,
                   const Matrix<S, 3, 3> &vectors, bool values_converged,
                   int vectors_converged, int inner_failed, int status,
                   long sweeps, std::size_t object_bytes,
                   T relative_tolerance, T vector_tolerance, T small_value) {
  std::printf("%zu,%.17g,%.17g,", profile, double(TEST_BREAKDOWN), inner_tolerance);
  std::printf("%s,%s,%d,%d,%d,%d,%d,%ld,%zu,%.17g,%.17g,%.17g",
              sizeof(T) == sizeof(float) ? "float" : "double", algorithm,
              kind, int(values_converged), vectors_converged, inner_failed,
              status, sweeps, object_bytes, double(relative_tolerance),
              double(vector_tolerance), double(small_value));
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      std::printf(",%a", double(matrix(i, j)));
    }
  }
  for (std::size_t i = 0; i < 3; ++i) {
    std::printf(",%a,%a", real_part(values[i]), imaginary_part(values[i]));
  }
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      std::printf(",%a,%a", real_part(vectors(i, j)), imaginary_part(vectors(i, j)));
    }
  }
  std::putchar('\n');
}

template <typename T, typename Solver>
inline void configure(Solver &solver, double relative_tolerance,
                      double vector_tolerance, double shift,
                      double inner_tolerance) {
  // Assign after construction, replacing the constructor's epsilon-based defaults.
  solver.iteration_max = 128;
  solver.relative_tolerance = static_cast<T>(relative_tolerance);
  solver.eigenvector_tolerance = static_cast<T>(vector_tolerance);
  solver.small_value = static_cast<T>(shift);
  solver.gmres_k_decay_rate = static_cast<T>(inner_tolerance);
}

template <typename T>
inline void run(std::size_t profile, double relative_tolerance,
                double vector_tolerance, double shift, double inner_tolerance) {
  for (int kind = 0; kind < 9; ++kind) {
    const auto matrix = fixture<T>(kind);
    if (kind != 5 && kind != 6) {
      EigenSolverReal<T, 3> solver;
      configure<T>(solver, relative_tolerance, vector_tolerance, shift, inner_tolerance);
      solver.solve_eigen_values_and_vectors(matrix);
      output(profile, inner_tolerance, "real", kind, matrix, solver.get_eigen_values(),
             solver.get_eigen_vectors(), solver.eigen_values_converged(),
             int(solver.eigen_vectors_converged()), int(solver.inner_solve_failed()),
             int(solver.get_status()), long(solver.get_sweep_count()), sizeof(solver),
             solver.relative_tolerance, solver.eigenvector_tolerance, solver.small_value);
    }
    EigenSolverComplex<T, 3> complex;
    configure<T>(complex, relative_tolerance, vector_tolerance, shift, inner_tolerance);
    complex.solve_eigen_values_and_vectors(matrix);
    output(profile, inner_tolerance, "complex", kind, matrix, complex.get_eigen_values(),
           complex.get_eigen_vectors(), complex.eigen_values_converged(),
           int(complex.eigen_vectors_converged()), int(complex.inner_solve_failed()),
           int(complex.get_status()), long(complex.get_sweep_count()), sizeof(complex),
           complex.relative_tolerance, complex.eigenvector_tolerance, complex.small_value);
    if (kind != 1 && kind != 5 && kind != 6) {
      EigenSolverSymmetric<T, 3> jacobi;
      jacobi.relative_tolerance = static_cast<T>(relative_tolerance);
      jacobi.solve_eigen_values_and_vectors(matrix);
      // Jacobi reports value convergence; it has no separate vector status or sweep counter.
      output(profile, inner_tolerance, "jacobi", kind, matrix, jacobi.get_eigen_values(),
             jacobi.get_eigen_vectors(), jacobi.eigen_values_converged(),
             -1, -1, -1, -1, sizeof(jacobi), jacobi.relative_tolerance, T(0), T(0));
    }
  }
}

int main() {
#ifndef TEST_EXTRA_ONLY
  const double relative_tolerances[] = {8e-7, 8e-12};
  const double vector_tolerances[] = {3.2e-6, 3.2e-12};
  const double shifts[] = {1e-3, 1e-6, 1e-9};
  const double inner_tolerances[] = {5e-3, 1e-6, 1e-12};
  std::size_t profile = 0;
  for (std::size_t value = 0; value < 2; ++value) {
    for (std::size_t vector = 0; vector < 2; ++vector) {
      for (std::size_t shift = 0; shift < 3; ++shift) {
        for (std::size_t inner = 0; inner < 3; ++inner) {
          run<float>(profile, relative_tolerances[value], vector_tolerances[vector],
                      shifts[shift], inner_tolerances[inner]);
          run<double>(profile, relative_tolerances[value], vector_tolerances[vector],
                       shifts[shift], inner_tolerances[inner]);
          ++profile;
        }
      }
    }
  }
#else
  std::size_t profile = 36;
#endif
  // Follow-up profiles isolate the middle shift and matching residual target.
  const double additional_profiles[][4] = {
      {8e-7, 3.2e-6, 3e-4, 5e-3}, {8e-7, 3.2e-6, 3e-4, 1e-6},
      {8e-9, 3.2e-6, 3e-4, 5e-3}, {8e-9, 3.2e-6, 1e-6, 5e-3},
      {8e-12, 3.2e-10, 1e-9, 5e-3}, {8e-12, 3.2e-10, 1e-9, 1e-6},
      {8e-12, 3.2e-10, 1e-11, 5e-3}, {8e-9, 3.2e-6, 1e-4, 5e-3}};
  for (std::size_t index = 0; index < 8; ++index) {
    const double *setting = additional_profiles[index];
    run<float>(profile, setting[0], setting[1], setting[2], setting[3]);
    run<double>(profile, setting[0], setting[1], setting[2], setting[3]);
    ++profile;
  }
}
