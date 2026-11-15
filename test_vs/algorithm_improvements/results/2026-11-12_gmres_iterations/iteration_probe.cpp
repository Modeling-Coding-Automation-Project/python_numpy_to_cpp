// Host benchmark. Build the same source against baseline and current headers.
// Error here is to a manufactured target. For ill-conditioned rounded inputs,
// use gmres_oracle.py for the mathematical solution of the stored A and b.
#include "base_matrix_inverse.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace Base::Matrix;
static volatile double observed = 0;

template <typename T, std::size_t N>
double error(const Vector<T, N> &x, const Vector<T, N> &target) {
  double sum = 0;
  for (std::size_t i = 0; i < N; ++i) {
    double delta = double(x[i]) - double(target[i]);
    sum += delta * delta;
  }
  return std::sqrt(sum / double(N));
}

template <typename T, std::size_t N> Matrix<T, N, N> fixture(int kind) {
  Matrix<T, N, N> matrix;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      if (kind == 1) {
        matrix(i, j) = T(1) / T(i + j + 1); // Hilbert; target is manufactured.
      } else if (kind == 2) {
        T scale = T(1);
        for (std::size_t k = 0; k < j; ++k) {
          scale *= T(2);
        }
        matrix(i, j) = (i == j ? T(4) : T((i + j) % 3) * T(0.01)) * scale;
      } else {
        matrix(i, j) =
            i == j ? T(4) + T(i) / T(N) : T((i * 7 + j * 3) % 5) * T(0.03);
      }
    }
  }
  return matrix;
}

template <typename T, std::size_t N> void legacy_row(int kind) {
  auto matrix = fixture<T, N>(kind);
  Vector<T, N> target;
  for (std::size_t i = 0; i < N; ++i) {
    target[i] = T(1);
  }
  auto rhs = matrix * target;
  Vector<T, N> zero;
  Vector<T, N> solution;
  T rho = T(0);
  const std::size_t repeats = 1000;
  auto start = std::chrono::steady_clock::now();
  for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
    rhs[0] += T((repeat & 1) ? -1e-6 : 1e-6);
    auto result = gmres_k(matrix, rhs, zero, T(0), T(1e-25));
    solution = std::get<0>(result);
    rho = std::get<1>(result);
    observed += double(solution[repeat % N]);
  }
  auto end = std::chrono::steady_clock::now();
  double ns =
      std::chrono::duration<double, std::nano>(end - start).count() / repeats;
  std::printf("legacy,%s,%zu,%d,%zu,1,1,0,%.3f,%.9g,%.9g,0\n",
              sizeof(T) == 4 ? "float" : "double", N, kind, N, ns,
              error(solution, target), double(rho));
}

#ifdef ALGORITHM_IMPROVEMENTS
template <typename T, std::size_t N, std::size_t K>
void configured_row(int kind, std::size_t cycles, std::size_t passes,
                    bool precondition) {
  auto matrix = fixture<T, N>(kind);
  Vector<T, N> target;
  for (std::size_t i = 0; i < N; ++i) {
    target[i] = T(1);
  }
  auto rhs = matrix * target;
  Vector<T, N> zero, solution;
  auto diagonal = gmres_inverse_diagonal(matrix, T(1e-25));
  GmresSolver<T, N, K> solver;
  solver.cycles = cycles;
  solver.orthogonalization_passes = passes;
  solver.division_min = T(1e-25);
  const std::size_t repeats = 1000;
  auto start = std::chrono::steady_clock::now();
  for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
    rhs[0] += T((repeat & 1) ? -1e-6 : 1e-6);
    solution = precondition
                   ? solver.solve_preconditioned(matrix, rhs, zero, diagonal)
                   : solver.solve(matrix, rhs, zero);
    observed += double(solution[repeat % N]);
  }
  auto end = std::chrono::steady_clock::now();
  double ns =
      std::chrono::duration<double, std::nano>(end - start).count() / repeats;
  std::printf("configured,%s,%zu,%d,%zu,%zu,%zu,%d,%.3f,%.9g,%.9g,%zu\n",
              sizeof(T) == 4 ? "float" : "double", N, kind, K, cycles, passes,
              int(precondition), ns, error(solution, target),
              double(solver.get_residual_norm()),
              sizeof(solver) + (precondition ? sizeof(diagonal) : 0));
}
#endif

template <typename T, std::size_t N> void rows() {
  for (int kind = 0; kind < 3; ++kind) {
    legacy_row<T, N>(kind);
#ifdef ALGORITHM_IMPROVEMENTS
    configured_row<T, N, N>(kind, 1, 1, false);
    configured_row<T, N, N>(kind, 1, 2, false);
    configured_row<T, N, N / 2>(kind, 1, 1, false);
    configured_row<T, N, N / 2>(kind, 4, 1, false);
    configured_row<T, N, N / 2>(kind, 4, 2, false);
    configured_row<T, N, N / 2>(kind, 4, 1, true);
#endif
  }
}

template <typename T, std::size_t N, std::size_t K>
void count_updates(int kind, std::size_t cycles, bool precondition) {
  auto matrix = fixture<T, N>(kind);
  Vector<T, N> target;
  for (std::size_t i = 0; i < N; ++i) { target[i] = T(1); }
  auto rhs = matrix * target;
  Vector<T, N> zero, solution;
  auto diagonal = gmres_inverse_diagonal(matrix, T(1e-25));
  GmresSolver<T, N, K> solver;
  solver.cycles = cycles;
  solver.orthogonalization_passes = 1;
  solver.division_min = T(1e-25);
  std::size_t iterations = 0, minimum = 100000, maximum = 0;
  std::size_t converged = 0, breakdown = 0;
  for (std::size_t repeat = 0; repeat < 1000; ++repeat) {
    rhs[0] += T((repeat & 1) ? -1e-6 : 1e-6);
    solution = precondition
        ? solver.solve_preconditioned(matrix, rhs, zero, diagonal)
        : solver.solve(matrix, rhs, zero);
    std::size_t count = solver.get_iterations();
    iterations += count;
    if (count < minimum) { minimum = count; }
    if (count > maximum) { maximum = count; }
    converged += solver.is_converged();
    breakdown += solver.has_breakdown();
  }
  std::printf("%s N=%zu K=%zu C=%zu kind=%d pre=%d mean_updates=%.3f min=%zu max=%zu converged=%zu breakdown=%zu rho=%.9g rms_target_error=%.9g\n",
      sizeof(T)==4 ? "float" : "double", N, K, cycles, kind, int(precondition),
      double(iterations)/1000, minimum, maximum, converged, breakdown,
      double(solver.get_residual_norm()), error(solution, target));
}
int main() {
  count_updates<float,16,8>(0,4,true);
  count_updates<float,16,8>(1,4,true);
  count_updates<double,16,16>(0,1,false);
  count_updates<double,8,4>(2,4,true);
}
