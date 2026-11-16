// Host-only epsilon sweep. The driver selects baseline or isolated fixed-value headers.
#include "base_matrix_gmres.hpp"
#include <cstdio>
#include <type_traits>

using namespace Base::Matrix;

template <typename T> struct ExportScalar {
  using Real = T;
  static inline T make(double real, double) { return static_cast<T>(real); }
  static inline void print(const T &value) {
    std::printf(",%a,%a", static_cast<double>(value), 0.0);
  }
};

template <typename T> struct ExportScalar<Complex<T>> {
  using Real = T;
  static inline Complex<T> make(double real, double imag) {
    return Complex<T>(static_cast<T>(real), static_cast<T>(imag));
  }
  static inline void print(const Complex<T> &value) {
    std::printf(",%a,%a", static_cast<double>(value.real),
                static_cast<double>(value.imag));
  }
};

template <typename T, std::size_t N, std::size_t K>
inline void measure(const char *fixture, const char *setting,
                    const Matrix<T, N, N> &matrix, const Vector<T, N> &rhs,
                    std::size_t passes, std::size_t cycles, bool exact_stop) {
  using Real = typename ExportScalar<T>::Real;
  GmresSolver<T, N, K> solver;
  solver.division_min = static_cast<Real>(1e-25);
  solver.relative_tolerance = exact_stop ? static_cast<Real>(0)
      : sizeof(Real) == sizeof(float) ? static_cast<Real>(2e-5)
                                     : static_cast<Real>(1e-12);
  solver.orthogonalization_passes = passes;
  solver.cycles = cycles;
  Vector<T, N> result = solver.solve(matrix, rhs, Vector<T, N>());
  std::printf("%s,%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%d,%d,%zu,%a,%a",
      sizeof(Real) == sizeof(float) ? "float" : "double",
      std::is_same<T, Real>::value ? "real" : "complex", fixture, setting,
      N, K, passes, cycles, solver.get_iterations(), int(solver.is_converged()),
      int(solver.has_breakdown()), sizeof(solver),
      static_cast<double>(solver.relative_tolerance),
      static_cast<double>(solver.get_residual_norm()));
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      ExportScalar<T>::print(matrix(i, j));
    }
  }
  for (std::size_t i = 0; i < N; ++i) { ExportScalar<T>::print(rhs[i]); }
  for (std::size_t i = 0; i < N; ++i) { ExportScalar<T>::print(result[i]); }
  std::putchar('\n');
}

template <typename T, std::size_t N>
inline void family(const char *name, int kind) {
  Matrix<T, N, N> matrix;
  Vector<T, N> truth;
  for (std::size_t i = 0; i < N; ++i) {
    truth[i] = ExportScalar<T>::make(double(i + 1) / double(N),
                                    double(N - i) / double(2 * N));
    for (std::size_t j = 0; j < N; ++j) {
      if (kind == 1) {
        matrix(i, j) = ExportScalar<T>::make(1.0 / double(i + j + 1), 0.0);
      } else if (kind == 2) {
        matrix(i, j) = ExportScalar<T>::make(i == j ? 1.0 : (j == i + 1 ? 1.5 : 0.0),
                                            j == i + 1 ? 0.25 : 0.0);
      } else {
        const int real_code = int((i * 7 + j * 3) % 5) - 2;
        const int imag_code = int((i * 3 + j * 7) % 5) - 2;
        matrix(i, j) = ExportScalar<T>::make(
            i == j ? 4.0 + double(i) / double(N) : double(real_code) * 0.03,
            i == j ? 0.0 : double(imag_code) * 0.02);
      }
    }
  }
  Vector<T, N> rhs;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) { rhs[i] += matrix(i, j) * truth[j]; }
  }
  measure<T, N, N>(name, "p1", matrix, rhs, 1, 1, false);
  measure<T, N, N>(name, "p2", matrix, rhs, 2, 1, false);
  measure<T, N, N>(name, "p1_zero", matrix, rhs, 1, 1, true);
  measure<T, N, N / 2>(name, "restart_p2", matrix, rhs, 2, 4, false);
}

template <typename T> inline void near_breakdown() {
  const double deltas[] = {2e-6, 2e-8, 2e-10, 2e-12, 2e-14, 2e-16};
  const char *names[] = {"near_2e-6", "near_2e-8", "near_2e-10",
                         "near_2e-12", "near_2e-14", "near_2e-16"};
  for (std::size_t index = 0; index < 6; ++index) {
    Matrix<T, 2, 2> matrix;
    matrix(0, 0) = T(1);
    matrix(1, 0) = T(deltas[index]);
    matrix(1, 1) = T(deltas[index]);
    Vector<T, 2> rhs({T(1), T(0)});
    measure<T, 2, 2>(names[index], "p1_zero", matrix, rhs, 1, 1, true);
    measure<T, 2, 2>(names[index], "p2_zero", matrix, rhs, 2, 1, true);
  }
}

int main() {
  family<float, 6>("well", 0);
  family<double, 6>("well", 0);
  family<float, 6>("hilbert", 1);
  family<double, 6>("hilbert", 1);
  family<float, 6>("nonnormal", 2);
  family<double, 6>("nonnormal", 2);
  family<Complex<float>, 4>("well", 0);
  family<Complex<double>, 4>("well", 0);
  family<Complex<float>, 4>("nonnormal", 2);
  family<Complex<double>, 4>("nonnormal", 2);
  near_breakdown<float>();
  near_breakdown<double>();
}
