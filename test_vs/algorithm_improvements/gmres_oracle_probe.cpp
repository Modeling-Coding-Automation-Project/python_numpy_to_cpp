// Host-only data export for the independent Decimal oracle.
// Keep fixture/RHS construction aligned with gmres_benchmark.cpp.
#include "base_matrix_inverse.hpp"
#include <cstdio>

using namespace Base::Matrix;

template <typename T, std::size_t N> Matrix<T, N, N> oracle_fixture(int kind) {
  Matrix<T, N, N> matrix;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      if (kind == 1) {
        matrix(i, j) = T(1) / T(i + j + 1);
      } else if (kind == 2) {
        T scale = T(1);
        for (std::size_t k = 0; k < j; ++k) { scale *= T(2); }
        matrix(i, j) = (i == j ? T(4) : T((i + j) % 3) * T(0.01)) * scale;
      } else {
        matrix(i, j) =
            i == j ? T(4) + T(i) / T(N) : T((i * 7 + j * 3) % 5) * T(0.03);
      }
    }
  }
  return matrix;
}

template <typename T, std::size_t N>
Vector<T, N> final_rhs(const Matrix<T, N, N> &matrix) {
  Vector<T, N> ones;
  for (std::size_t i = 0; i < N; ++i) { ones[i] = T(1); }
  Vector<T, N> rhs = matrix * ones;
  // Each timed solve starts at zero, so only the final rounded RHS is needed.
  for (std::size_t repeat = 0; repeat < 1000; ++repeat) {
    rhs[0] += T((repeat & 1) ? -1e-6 : 1e-6);
  }
  return rhs;
}

template <typename T, std::size_t N>
void output(const char *mode, int fixture, std::size_t k, std::size_t cycles,
            std::size_t passes, bool precondition, const Matrix<T, N, N> &matrix,
            const Vector<T, N> &rhs, const Vector<T, N> &solution, T reported_rho,
            int converged = -1, int breakdown = -1) {
  std::printf("%s,%s,%zu,%d,%zu,%zu,%zu,%d,%d,%d,%a", mode,
              sizeof(T) == 4 ? "float" : "double", N, fixture, k, cycles,
              passes, int(precondition), converged, breakdown, double(reported_rho));
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      std::printf(",%a", double(matrix(i, j)));
    }
  }
  for (std::size_t i = 0; i < N; ++i) { std::printf(",%a", double(rhs[i])); }
  for (std::size_t i = 0; i < N; ++i) { std::printf(",%a", double(solution[i])); }
  std::putchar('\n');
}

#ifdef ALGORITHM_IMPROVEMENTS
template <typename T, std::size_t N, std::size_t K>
void configured(int kind, std::size_t cycles, std::size_t passes,
                 bool precondition) {
  auto matrix = oracle_fixture<T, N>(kind);
  auto rhs = final_rhs(matrix);
  Vector<T, N> zero;
  GmresSolver<T, N, K> solver;
  solver.cycles = cycles;
  solver.orthogonalization_passes = passes;
  solver.division_min = T(1e-25);
  auto diagonal = gmres_inverse_diagonal(matrix, T(1e-25));
  auto solution = precondition
                      ? solver.solve_preconditioned(matrix, rhs, zero, diagonal)
                      : solver.solve(matrix, rhs, zero);
  output("configured", kind, K, cycles, passes, precondition, matrix, rhs,
         solution, solver.get_residual_norm(), int(solver.is_converged()),
         int(solver.has_breakdown()));
}
#endif

template <typename T, std::size_t N> void rows() {
  for (int kind = 0; kind < 3; ++kind) {
    auto matrix = oracle_fixture<T, N>(kind);
    auto rhs = final_rhs(matrix);
    Vector<T, N> zero;
    auto result = gmres_k(matrix, rhs, zero, T(0), T(1e-25));
    output("legacy", kind, N, 1, 1, false, matrix, rhs,
           std::get<0>(result), std::get<1>(result));
#ifdef ALGORITHM_IMPROVEMENTS
    configured<T, N, N>(kind, 1, 1, false);
    configured<T, N, N>(kind, 1, 2, false);
    configured<T, N, N / 2>(kind, 1, 1, false);
    configured<T, N, N / 2>(kind, 4, 1, false);
    configured<T, N, N / 2>(kind, 4, 2, false);
    configured<T, N, N / 2>(kind, 4, 1, true);
    // Additional full/half K diagonal and P2 trials use the same fixed RHS.
    configured<T, N, N>(kind, 1, 1, true);
    configured<T, N, N>(kind, 1, 2, true);
    configured<T, N, N / 2>(kind, 1, 1, true);
    configured<T, N, N / 2>(kind, 1, 2, false);
    configured<T, N, N / 2>(kind, 1, 2, true);
    configured<T, N, N / 2>(kind, 4, 2, true);
#endif
  }
}

int main() {
  rows<float, 4>(); rows<double, 4>();
  rows<float, 8>(); rows<double, 8>();
  rows<float, 16>(); rows<double, 16>();
}
