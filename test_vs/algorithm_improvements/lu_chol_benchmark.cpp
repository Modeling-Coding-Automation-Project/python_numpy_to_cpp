// Host benchmark. Build both revisions with identical flags; see run_validation.py.
#include "base_matrix_cholesky_decomposition.hpp"
#include "base_matrix_lu_decomposition.hpp"
#include "base_matrix_triangular_sparse.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <type_traits>

using namespace Base::Matrix;
volatile double benchmark_sink = 0;

template <typename Function> double timing(Function function, int repetitions) {
  double samples[5];
  function();
  for (int sample = 0; sample < 5; ++sample) {
    auto begin = std::chrono::steady_clock::now();
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      function();
    }
    auto end = std::chrono::steady_clock::now();
    samples[sample] = std::chrono::duration<double, std::nano>(end - begin).count() /
                      double(repetitions);
  }
  for (int i = 1; i < 5; ++i) {
    for (int j = i; j > 0; --j) {
      if (samples[j] < samples[j - 1]) {
        double temporary = samples[j];
        samples[j] = samples[j - 1];
        samples[j - 1] = temporary;
      }
    }
  }
  return samples[2];
}

template <typename T, std::size_t N>
void reference(const Matrix<T, N, N> &a, const Vector<T, N> &b,
               long double (&x)[N]) {
  long double work[N][N];
  for (std::size_t i = 0; i < N; ++i) {
    x[i] = b[i];
    for (std::size_t j = 0; j < N; ++j) {
      work[i][j] = a(i, j);
    }
  }
  for (std::size_t k = 0; k < N; ++k) {
    std::size_t pivot = k;
    for (std::size_t i = k + 1; i < N; ++i) {
      if (std::abs(work[i][k]) > std::abs(work[pivot][k])) { pivot = i; }
    }
    for (std::size_t j = k; j < N; ++j) {
      long double temporary = work[k][j];
      work[k][j] = work[pivot][j];
      work[pivot][j] = temporary;
    }
    long double temporary = x[k]; x[k] = x[pivot]; x[pivot] = temporary;
    for (std::size_t i = k + 1; i < N; ++i) {
      long double multiplier = work[i][k] / work[k][k];
      for (std::size_t j = k + 1; j < N; ++j) {
        work[i][j] -= multiplier * work[k][j];
      }
      x[i] -= multiplier * x[k];
    }
  }
  for (std::size_t offset = 0; offset < N; ++offset) {
    std::size_t i = N - 1 - offset;
    for (std::size_t j = i + 1; j < N; ++j) { x[i] -= work[i][j] * x[j]; }
    x[i] /= work[i][i];
  }
}

template <typename T, std::size_t N>
double solution_error(const Vector<T, N> &x, const long double (&truth)[N]) {
  long double error = 0, scale = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (std::abs(truth[i]) > scale) { scale = std::abs(truth[i]); }
    if (std::abs(x[i] - truth[i]) > error) { error = std::abs(x[i] - truth[i]); }
  }
  return double(error / (scale == 0 ? 1 : scale));
}

template <typename T, std::size_t N>
double factor_error(const Matrix<T, N, N> &a, const Matrix<T, N, N> &upper) {
  long double error = 0, scale = 0;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      long double value = 0;
      for (std::size_t k = 0; k < N; ++k) {
        value += static_cast<long double>(upper(k, i)) * upper(k, j);
      }
      long double difference = std::abs(value - a(i, j));
      if (difference > error) { error = difference; }
      if (std::abs(static_cast<long double>(a(i, j))) > scale) {
        scale = std::abs(static_cast<long double>(a(i, j)));
      }
    }
  }
  return double(error / (scale == 0 ? 1 : scale));
}

template <typename T, std::size_t N>
__attribute__((noinline)) Vector<T, N> lu_solve(const Matrix<T, N, N> &a,
                                               const Vector<T, N> &b,
                                               LUDecomposition<T, N> &solver) {
  return solver.solve(a, b);
}

template <typename T, std::size_t N>
__attribute__((noinline)) Matrix<T, N, N> chol(const Matrix<T, N, N> &a) {
  return std::get<0>(cholesky_decomposition(a, Matrix<T, N, N>(), T(1e-20)));
}

template <typename T, std::size_t N, typename Indices, typename Pointers>
__attribute__((noinline)) Matrix<T, N, N> sparse_chol(
    const CompiledSparseMatrix<T, N, N, Indices, Pointers> &a) {
  return std::get<0>(cholesky_decomposition_sparse(
      a, Matrix<T, N, N>(), T(1e-20)));
}

#ifdef ALGORITHM_IMPROVEMENTS
template <typename T, std::size_t N>
__attribute__((noinline)) bool chol_to(const Matrix<T, N, N> &a,
                                      Matrix<T, N, N> &upper) {
  return cholesky_decomposition_to(a, upper, T(1e-20));
}

template <typename T, std::size_t N, typename Indices, typename Pointers>
__attribute__((noinline)) bool sparse_chol_to(
    const CompiledSparseMatrix<T, N, N, Indices, Pointers> &a,
    Matrix<T, N, N> &upper) {
  return cholesky_decomposition_sparse_to(a, upper, T(1e-20));
}

// Rejected dense right-looking default is retained here as a reproducible trial.
template <typename T, std::size_t N>
__attribute__((noinline)) Matrix<T, N, N> chol_right_trial(
    const Matrix<T, N, N> &a) {
  Matrix<T, N, N> upper;
  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i <= j; ++i) { upper(i, j) = a(j, i); }
  }
  CholeskyOperation::factor<T, N>(
      upper, T(1e-20), T(0));
  return upper;
}

template <typename T, std::size_t N, bool CACHE>
__attribute__((noinline)) Vector<T, N> reused(
    const LUDecomposition<T, N, CACHE> &solver, const Vector<T, N> &b) {
  return solver.solve_rhs(b);
}

template <typename T, std::size_t N>
__attribute__((noinline)) Vector<T, N> scaled_lu(
    LUDecomposition<T, N> &solver, const Matrix<T, N, N> &a, const Vector<T, N> &b) {
  solver.solve_scaled(a);
  return solver.solve_rhs(b);
}

template <typename Accumulator, typename T, std::size_t N>
__attribute__((noinline)) Vector<T, N> refined_lu(
    LUDecomposition<T, N> &solver, const Matrix<T, N, N> &a,
    const Vector<T, N> &b, std::size_t passes) {
  return solver.template solve_refined<Accumulator>(a, b, passes);
}

template <typename Solver, typename T, std::size_t N>
__attribute__((noinline)) void packed_factor(Solver &solver,
                                            const Matrix<T, N, N> &a) {
  solver.solve(a);
}
#endif

template <typename T, std::size_t N>
void print(const char *method, const char *kind, double ns, double error,
           std::size_t bytes) {
  std::printf("%s,%zu,%s,%s,%.3f,%.9g,%zu\n",
              std::is_same<T, float>::value ? "float" : "double", N,
              method, kind, ns, error, bytes);
}

template <typename T, std::size_t N> void run_lu(const char *kind, int example) {
  Matrix<T, N, N> a;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      a(i, j) = T(int((i * 13 + j * 7) % 19) - 9) / T(9);
      if (i == j) { a(i, j) += T(N); }
      if (example == 2) { a(i, j) = T(1) / T(i + j + 1); }
      if (example == 3) {
        T scale = i % 3 == 0 ? T(1e-4) : (i % 3 == 1 ? T(1e4) : T(1));
        a(i, j) *= scale;
      }
    }
  }
  if (example == 1) { a(0, 0) = T(1e-7); }
  Vector<T, N> b;
  for (std::size_t i = 0; i < N; ++i) { b[i] = T(i + 1); }
  long double truth[N]; reference(a, b, truth);
  LUDecomposition<T, N> lu; lu.division_min = T(1e-20);
  Vector<T, N> x;
  double ns = timing([&]() { x = lu_solve(a, b, lu); benchmark_sink += x[0]; }, 500);
  print<T, N>("lu_default", kind, ns, solution_error(x, truth), sizeof(lu));
#ifdef ALGORITHM_IMPROVEMENTS
  ns = timing([&]() { x = scaled_lu(lu, a, b); benchmark_sink += x[0]; }, 500);
  print<T, N>("lu_scaled", kind, ns, solution_error(x, truth), sizeof(lu));
  lu.solve(a);
  ns = timing([&]() { x = reused(lu, b); benchmark_sink += x[0]; }, 500);
  print<T, N>("lu_reused", kind, ns, solution_error(x, truth), sizeof(lu));
  LUDecomposition<T, N, true> cached; cached.division_min = T(1e-20); cached.solve(a);
  ns = timing([&]() { x = reused(cached, b); benchmark_sink += x[0]; }, 500);
  print<T, N>("lu_cached", kind, ns, solution_error(x, truth), sizeof(cached));
  for (std::size_t passes = 1; passes <= 2; ++passes) {
    ns = timing([&]() { x = refined_lu<T>(lu, a, b, passes); benchmark_sink += x[0]; }, 500);
    print<T, N>(passes == 1 ? "lu_refine1" : "lu_refine2", kind, ns,
                solution_error(x, truth), sizeof(lu));
    ns = timing([&]() { x = refined_lu<double>(lu, a, b, passes); benchmark_sink += x[0]; }, 500);
    print<T, N>(passes == 1 ? "lu_mixed1" : "lu_mixed2", kind, ns,
                solution_error(x, truth), sizeof(lu));
  }
#endif
}

#ifdef ALGORITHM_IMPROVEMENTS
template <typename T, std::size_t N>
void run_packed(const Matrix<T, N, N> &a, const char *label) {
  PackedCholesky<T, N> solver;
  solver.division_min = T(1e-20);
  double ns = timing([&]() { packed_factor(solver, a);
                             benchmark_sink += solver.is_valid(); }, 500);
  print<T, N>(label, "spd", ns, factor_error(a, solver.get_U()), sizeof(solver));
}
#endif

template <typename T, std::size_t N> void run_chol() {
  Matrix<T, N, N> a;
  Matrix<T, N, N> lower;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      lower(i, j) = i == j ? T(2) + T(i % 5 + 1) / T(7) : T(int((i * 5 + j * 3) % 7) - 3) / T(11);
    }
  }
  a = lower * output_matrix_transpose(lower);
  Matrix<T, N, N> upper;
  double ns = timing([&]() { upper = chol(a); benchmark_sink += upper(0, 0); }, 500);
  print<T, N>("chol_default", "spd", ns, factor_error(a, upper), sizeof(upper));
  auto sparse = create_LowerTriangularSparseMatrix(a);
  ns = timing([&]() { upper = sparse_chol(sparse);
                     benchmark_sink += upper(0, 0); }, 500);
  print<T, N>("chol_sparse", "spd", ns, factor_error(a, upper), sizeof(upper));
#ifdef ALGORITHM_IMPROVEMENTS
  ns = timing([&]() { benchmark_sink += chol_to(a, upper); }, 500);
  print<T, N>("chol_to", "spd", ns, factor_error(a, upper), sizeof(upper));
  ns = timing([&]() { benchmark_sink += sparse_chol_to(sparse, upper); }, 500);
  print<T, N>("chol_sparse_to", "spd", ns, factor_error(a, upper), sizeof(upper));
  ns = timing([&]() { upper = chol_right_trial(a);
                     benchmark_sink += upper(0, 0); }, 500);
  print<T, N>("chol_right_trial", "spd", ns, factor_error(a, upper), sizeof(upper));
  run_packed<T, N>(a, "chol_packed");
  SPDLDLT<T, N> solver;
  ns = timing([&]() { packed_factor(solver, a);
                     benchmark_sink += solver.is_valid(); }, 500);
  auto l = solver.get_L(); auto d = solver.get_D();
  long double maximum = 0, scale = 0;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      long double value = 0;
      for (std::size_t k = 0; k < N; ++k) {
        value += static_cast<long double>(l(i, k)) * d[k] * l(j, k);
      }
      if (std::abs(value - a(i, j)) > maximum) { maximum = std::abs(value - a(i, j)); }
      if (std::abs(static_cast<long double>(a(i, j))) > scale) {
        scale = std::abs(static_cast<long double>(a(i, j)));
      }
    }
  }
  print<T, N>("spd_ldlt", "spd", ns, double(maximum / scale), sizeof(solver));
#endif
}

int main() {
  std::printf("type,n,method,case,median_ns,relative_error,object_bytes\n");
  run_lu<float, 4>("regular", 0); run_lu<double, 4>("regular", 0);
  run_lu<float, 16>("regular", 0); run_lu<double, 16>("regular", 0);
  run_lu<float, 6>("tiny_pivot", 1); run_lu<double, 6>("tiny_pivot", 1);
  run_lu<float, 6>("hilbert", 2); run_lu<double, 6>("hilbert", 2);
  run_lu<float, 6>("row_scaled", 3); run_lu<double, 6>("row_scaled", 3);
  run_chol<float, 4>(); run_chol<double, 4>();
  run_chol<float, 16>(); run_chol<double, 16>();
  return benchmark_sink == -1234567;
}
