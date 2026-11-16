// Host-only tolerance sweep and benchmark; the production headers are unchanged.
#include "base_matrix_inverse.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace Base::Matrix;
static volatile double tolerance_sink = 0;
constexpr std::size_t TOLERANCE_REPEATS = 1000;

template <typename T, std::size_t N>
inline Matrix<T, N, N> tolerance_fixture(int kind) {
  Matrix<T, N, N> matrix;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      if (kind == 1) {
        matrix(i, j) = T(1) / T(i + j + 1);
      } else if (kind == 2) {
        T scale = T(1);
        for (std::size_t k = 0; k < j; ++k) { scale *= T(2); }
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

template <typename T, std::size_t N, std::size_t K>
inline void export_quality(const char *fixture, std::size_t phase,
                           const Matrix<T, N, N> &matrix,
                           const Vector<T, N> &rhs,
                           const Vector<T, N> &solution,
                           const GmresSolver<T, N, K> &solver,
                           bool precondition) {
  std::printf("%s,%zu,%s,%zu,%zu,%zu,%d,%zu,%zu,%d,%d,%zu,%a,%a",
      sizeof(T) == sizeof(float) ? "float" : "double", N, fixture, K,
      solver.cycles, solver.orthogonalization_passes, int(precondition), phase,
      solver.get_iterations(), int(solver.is_converged()),
      int(solver.has_breakdown()),
      sizeof(solver) + (precondition ? sizeof(Vector<T, N>) : 0),
      double(solver.relative_tolerance), double(solver.get_residual_norm()));
  for (std::size_t row = 0; row < N; ++row) {
    for (std::size_t column = 0; column < N; ++column) {
      std::printf(",%a", double(matrix(row, column)));
    }
  }
  for (std::size_t row = 0; row < N; ++row) { std::printf(",%a", double(rhs[row])); }
  for (std::size_t row = 0; row < N; ++row) { std::printf(",%a", double(solution[row])); }
  std::putchar('\n');
}

template <typename T, std::size_t N, std::size_t K>
inline void tolerance_case(int kind, std::size_t cycles, std::size_t passes,
                           bool precondition, double tolerance, bool timing) {
  const auto matrix = tolerance_fixture<T, N>(kind);
  Vector<T, N> target;
  for (std::size_t i = 0; i < N; ++i) { target[i] = T(1); }
  auto rhs = matrix * target;
  Vector<T, N> zero, solution;
  const auto diagonal = gmres_inverse_diagonal(matrix, T(1e-25));
  GmresSolver<T, N, K> solver;
  solver.cycles = cycles;
  solver.orthogonalization_passes = passes;
  solver.division_min = T(1e-25);
  solver.relative_tolerance = T(tolerance);
  if (timing) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0; repeat < TOLERANCE_REPEATS; ++repeat) {
      rhs[0] += T((repeat & 1) ? -1e-6 : 1e-6);
      solution = precondition
          ? solver.solve_preconditioned(matrix, rhs, zero, diagonal)
          : solver.solve(matrix, rhs, zero);
      tolerance_sink += double(solution[repeat % N]);
    }
    const auto end = std::chrono::steady_clock::now();
    const double nanoseconds =
        std::chrono::duration<double, std::nano>(end - start).count() /
        double(TOLERANCE_REPEATS);
    std::printf("%s,%zu,%d,%zu,%zu,%zu,%d,%.9g,%zu,%d,%d,%a\n",
        sizeof(T) == sizeof(float) ? "float" : "double", N, kind, K,
        cycles, passes, int(precondition), nanoseconds, solver.get_iterations(),
        int(solver.is_converged()), int(solver.has_breakdown()), double(solution[0]));
    return;
  }

  // Only two stored RHS values recur in the benchmark. Verify all 1000 updates,
  // and solve each state once; the same zero initial solution is used each time.
  T states[2] = {};
  const char *names[] = {"well", "hilbert", "scaled"};
  for (std::size_t repeat = 0; repeat < TOLERANCE_REPEATS; ++repeat) {
    rhs[0] += T((repeat & 1) ? -1e-6 : 1e-6);
    if (repeat < 2) {
      states[repeat] = rhs[0];
      solution = precondition
          ? solver.solve_preconditioned(matrix, rhs, zero, diagonal)
          : solver.solve(matrix, rhs, zero);
      export_quality(names[kind], repeat, matrix, rhs, solution, solver, precondition);
    } else if (rhs[0] != states[repeat % 2]) {
      std::fprintf(stderr, "The RHS sequence has more than two states.\n");
      std::exit(EXIT_FAILURE);
    }
  }
}

template <typename T, std::size_t N>
inline void tolerance_rows(double tolerance, bool timing) {
  for (int kind = 0; kind < 3; ++kind) {
    tolerance_case<T, N, N>(kind, 1, 1, false, tolerance, timing);
    tolerance_case<T, N, N>(kind, 1, 2, false, tolerance, timing);
    tolerance_case<T, N, N / 2>(kind, 1, 1, false, tolerance, timing);
    tolerance_case<T, N, N / 2>(kind, 4, 1, false, tolerance, timing);
    tolerance_case<T, N, N / 2>(kind, 4, 2, false, tolerance, timing);
    tolerance_case<T, N, N / 2>(kind, 4, 1, true, tolerance, timing);
  }
}

template <typename T> inline void tolerance_near(double tolerance) {
  const double deltas[] = {2e-6, 2e-10, 2e-14};
  const char *names[] = {"near_2e-6", "near_2e-10", "near_2e-14"};
  for (std::size_t index = 0; index < 3; ++index) {
    Matrix<T, 2, 2> matrix({{T(1), T(0)}, {T(deltas[index]), T(deltas[index])}});
    Vector<T, 2> rhs({T(1), T(0)});
    GmresSolver<T, 2> solver;
    solver.division_min = T(1e-25);
    solver.relative_tolerance = T(tolerance);
    const auto solution = solver.solve(matrix, rhs, Vector<T, 2>());
    export_quality(names[index], 0, matrix, rhs, solution, solver, false);
  }
}

int main(int argc, char **argv) {
  if (argc != 3) { return EXIT_FAILURE; }
  const bool timing = std::strcmp(argv[1], "time") == 0;
  const double tolerance = std::strtod(argv[2], nullptr);
  tolerance_rows<float, 4>(tolerance, timing);
  tolerance_rows<double, 4>(tolerance, timing);
  tolerance_rows<float, 8>(tolerance, timing);
  tolerance_rows<double, 8>(tolerance, timing);
  tolerance_rows<float, 16>(tolerance, timing);
  tolerance_rows<double, 16>(tolerance, timing);
  if (!timing) {
    tolerance_near<float>(tolerance);
    tolerance_near<double>(tolerance);
  }
  return EXIT_SUCCESS;
}
