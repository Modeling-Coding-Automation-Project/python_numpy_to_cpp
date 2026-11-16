// Host-only comparison of independent breakdown and residual tolerances.
#include "base_matrix_gmres.hpp"
#include <cstdio>
#include <type_traits>

using namespace Base::Matrix;

template <typename T> struct IndependentExportScalar {
  using Real = T;
  static inline T make(double real, double) { return static_cast<T>(real); }
  static inline void print(const T &value) {
    std::printf(",%a,%a", static_cast<double>(value), 0.0);
  }
};

template <typename T> struct IndependentExportScalar<Complex<T>> {
  using Real = T;
  static inline Complex<T> make(double real, double imag) {
    return Complex<T>(static_cast<T>(real), static_cast<T>(imag));
  }
  static inline void print(const Complex<T> &value) {
    std::printf(",%a,%a", static_cast<double>(value.real),
                static_cast<double>(value.imag));
  }
};

template <typename T, std::size_t N>
inline void measure_independent(const char *fixture,
                                const Matrix<T, N, N> &matrix,
                                const Vector<T, N> &rhs) {
  using Real = typename IndependentExportScalar<T>::Real;
  const double tolerances[] = {0, 1e-3, 1e-6, 1e-12};
  const char *labels[] = {"0", "1e-3", "1e-6", "1e-12"};
  for (std::size_t tolerance_index = 0; tolerance_index < 4; ++tolerance_index) {
    for (std::size_t passes = 1; passes <= 2; ++passes) {
      GmresSolver<T, N> solver;
      solver.division_min = static_cast<Real>(1e-25);
      solver.relative_tolerance = static_cast<Real>(tolerances[tolerance_index]);
      solver.orthogonalization_passes = passes;
      Vector<T, N> result = solver.solve(matrix, rhs, Vector<T, N>());
      std::printf("%s,%s,%s,p%zu_tau%s,%zu,%zu,%zu,1,%zu,%d,%d,%zu,%a,%a",
          sizeof(Real) == sizeof(float) ? "float" : "double",
          std::is_same<T, Real>::value ? "real" : "complex", fixture,
          passes, labels[tolerance_index], N, N, passes,
          solver.get_iterations(), int(solver.is_converged()),
          int(solver.has_breakdown()), sizeof(solver),
          static_cast<double>(solver.relative_tolerance),
          static_cast<double>(solver.get_residual_norm()));
      for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
          IndependentExportScalar<T>::print(matrix(i, j));
        }
      }
      for (std::size_t i = 0; i < N; ++i) {
        IndependentExportScalar<T>::print(rhs[i]);
      }
      for (std::size_t i = 0; i < N; ++i) {
        IndependentExportScalar<T>::print(result[i]);
      }
      std::putchar('\n');
    }
  }
}

template <typename T, std::size_t N>
inline void independent_family(const char *name, bool nonnormal) {
  Matrix<T, N, N> matrix;
  Vector<T, N> truth;
  for (std::size_t i = 0; i < N; ++i) {
    truth[i] = IndependentExportScalar<T>::make(
        double(i + 1) / double(N), double(N - i) / double(2 * N));
    for (std::size_t j = 0; j < N; ++j) {
      if (nonnormal) {
        matrix(i, j) = IndependentExportScalar<T>::make(
            i == j ? 1.0 : (j == i + 1 ? 1.5 : 0.0),
            j == i + 1 ? 0.25 : 0.0);
      } else {
        const int real_code = int((i * 7 + j * 3) % 5) - 2;
        const int imag_code = int((i * 3 + j * 7) % 5) - 2;
        matrix(i, j) = IndependentExportScalar<T>::make(
            i == j ? 4.0 + double(i) / double(N) : double(real_code) * 0.03,
            i == j ? 0.0 : double(imag_code) * 0.02);
      }
    }
  }
  Vector<T, N> rhs;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      rhs[i] += matrix(i, j) * truth[j];
    }
  }
  measure_independent(name, matrix, rhs);
}

template <typename T> inline void independent_near_breakdown() {
  const double deltas[] = {2e-6, 2e-10, 2e-14};
  const char *names[] = {"near_2e-6", "near_2e-10", "near_2e-14"};
  for (std::size_t index = 0; index < 3; ++index) {
    Matrix<T, 2, 2> matrix;
    matrix(0, 0) = T(1);
    matrix(1, 0) = T(deltas[index]);
    matrix(1, 1) = T(deltas[index]);
    measure_independent(names[index], matrix, Vector<T, 2>({T(1), T(0)}));
  }
}

int main() {
  independent_family<float, 6>("well", false);
  independent_family<double, 6>("well", false);
  independent_family<float, 6>("nonnormal", true);
  independent_family<double, 6>("nonnormal", true);
  independent_family<Complex<float>, 4>("well", false);
  independent_family<Complex<double>, 4>("well", false);
  independent_family<Complex<float>, 4>("nonnormal", true);
  independent_family<Complex<double>, 4>("nonnormal", true);
  independent_near_breakdown<float>();
  independent_near_breakdown<double>();
}
