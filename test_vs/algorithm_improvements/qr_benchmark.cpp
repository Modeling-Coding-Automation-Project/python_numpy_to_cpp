// Host-only benchmark. Define ALGORITHM_IMPROVEMENTS for the new optional APIs.
#include "base_matrix_qr_decomposition.hpp"
#include <chrono>
#include <cstdio>
#include <type_traits>

using namespace Base::Matrix;

namespace {
template <typename T> T magnitude(T value) { return value < T(0) ? -value : value; }

template <typename T, std::size_t M, std::size_t N>
Matrix<T, M, N> input(bool hessenberg, bool near_diagonal) {
  Matrix<T, M, N> A;
  for (std::size_t column = 0; column < N; ++column) {
    for (std::size_t row = 0; row < M; ++row) {
      T value = T(int((row * 13 + column * 7) % 19) - 9) / T(7);
      if (near_diagonal) { value *= T(1e-3); }
      if (row == column) { value += T(3); }
      if (hessenberg && row > column + 1) { value = T(0); }
      A(row, column) = value;
    }
  }
  return A;
}

template <typename T, std::size_t M, std::size_t N>
void errors(const Matrix<T, M, N> &A, const Matrix<T, M, M> &Q,
            const Matrix<T, M, N> &R, double &reconstruction, double &orthogonality, double &lower_triangle) {
  T max_error = T(0);
  T scale = T(0);
  T max_orthogonal = T(0);
  T max_lower = T(0);
  for (std::size_t column = 0; column < N; ++column) {
    for (std::size_t row = 0; row < M; ++row) {
      if (row > column && magnitude(R(row, column)) > max_lower) {
        max_lower = magnitude(R(row, column));
      }
      T value = T(0);
      for (std::size_t k = 0; k < M; ++k) { value += Q(row, k) * R(k, column); }
      if (magnitude(value - A(row, column)) > max_error) {
        max_error = magnitude(value - A(row, column));
      }
      if (magnitude(A(row, column)) > scale) { scale = magnitude(A(row, column)); }
    }
  }
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < M; ++j) {
      T value = i == j ? T(-1) : T(0);
      for (std::size_t k = 0; k < M; ++k) { value += Q(k, i) * Q(k, j); }
      if (magnitude(value) > max_orthogonal) { max_orthogonal = magnitude(value); }
    }
  }
  reconstruction = double(scale > T(0) ? max_error / scale : max_error);
  orthogonality = double(max_orthogonal);
  lower_triangle = double(scale > T(0) ? max_lower / scale : max_lower);
}

template <typename Solver, typename T, std::size_t M, std::size_t N>
void run(const Matrix<T, M, N> &A, const char *input_name, const char *method,
         bool explicit_q, T floor) {
  const std::size_t repetitions = N <= 4 ? 12000 : (N <= 8 ? 3000 : (N <= 16 ? 700 : 100));
  double elapsed = 1e100;
  for (std::size_t trial = 0; trial < 3; ++trial) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
      Solver solver;
      solver.division_min = floor;
      solver.solve(A);
      // The compiler must retain writes to all factors; no timing of metrics.
      asm volatile("" : : "g"(&solver) : "memory");
      if (explicit_q) {
        const auto Q = solver.get_Q();
        asm volatile("" : : "g"(&Q) : "memory");
      }
    }
    const auto end = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(end - start).count() /
                      double(repetitions);
    if (ns < elapsed) { elapsed = ns; }
  }
  Solver solver;
  solver.division_min = floor;
  solver.solve(A);
  double rec;
  double orth;
  double lower;
  errors(A, solver.get_Q(), solver.get_R(), rec, orth, lower);
  std::printf("%s,%zu,%zu,%s,%s,%.4g,%zu,%zu,%.3f,%.9g,%.9g,%.9g\n",
              std::is_same<T, float>::value ? "float" : "double", M, N,
              input_name, method, double(floor), sizeof(Solver),
              explicit_q ? sizeof(Matrix<T, M, M>) : std::size_t(0), elapsed, rec, orth, lower);
}

#ifdef ALGORITHM_IMPROVEMENTS
template <typename T, std::size_t M, std::size_t N>
struct HessenbergQR : QRDecomposition<T, M, N> {
  void solve(const Matrix<T, M, N> &A) { this->solve_hessenberg(A); }
};

template <typename T, std::size_t M, std::size_t N>
struct CheckedHessenbergQR : QRDecomposition<T, M, N> {
  // Trial only: scanning to validate structure removes most specialization gain.
  void solve(const Matrix<T, M, N> &A) {
    bool structure_valid = true;
    for (std::size_t column = 0; column < N; ++column) {
      for (std::size_t row = column + 2; row < M; ++row) {
        if (A(row, column) != T(0)) { structure_valid = false; }
      }
    }
    if (structure_valid) {
      this->solve_hessenberg(A);
    } else {
      QRDecomposition<T, M, N>::solve(A);
    }
  }
};

template <typename T, std::size_t M, std::size_t N>
struct HouseholderApplyQR : QRDecompositionHouseholder<T, M, N> {
  void solve(const Matrix<T, M, N> &A) {
    QRDecompositionHouseholder<T, M, N>::solve(A);
    Vector<T, M> b;
    for (std::size_t row = 0; row < M; ++row) { b[row] = T(row + 1); }
    this->apply_Q_transpose_in_place(b);
    asm volatile("" : : "g"(&b) : "memory");
  }
};
#endif

template <typename T, std::size_t M, std::size_t N> void cases() {
  const auto dense = input<T, M, N>(false, false);
  const auto hessenberg = input<T, M, N>(true, false);
  const auto almost_diagonal = input<T, M, N>(false, true);
  run<QRDecomposition<T, M, N>>(dense, "dense", "givens", false, T(1e-10));
  run<QRDecomposition<T, M, N>>(hessenberg, "hessenberg", "givens", false, T(1e-10));
  run<QRDecomposition<T, M, N>>(almost_diagonal, "near_diagonal", "givens", false, T(1e-10));
  run<QRDecomposition<T, M, N>>(almost_diagonal, "near_diagonal", "givens_skip", false, T(1e-2));
#ifdef ALGORITHM_IMPROVEMENTS
  run<HessenbergQR<T, M, N>>(hessenberg, "hessenberg", "givens_hessenberg", false, T(1e-10));
  run<CheckedHessenbergQR<T, M, N>>(hessenberg, "hessenberg", "givens_checked_trial", false, T(1e-10));
  run<QRDecompositionHouseholder<T, M, N>>(dense, "dense", "householder", false, T(1e-10));
  run<QRDecompositionHouseholder<T, M, N>>(dense, "dense", "householder_full_q", true, T(1e-10));
  run<HouseholderApplyQR<T, M, N>>(dense, "dense", "householder_apply_qt", false, T(1e-10));
#endif
}
} // namespace

int main() {
  std::printf("type,rows,cols,input,method,division_min,object_bytes,extra_output_bytes,time_ns,max_relative_reconstruction,max_orthogonality,max_relative_lower_triangle\n");
  cases<float, 3, 3>();
  cases<double, 3, 3>();
  cases<float, 8, 8>();
  cases<double, 8, 8>();
  cases<float, 16, 16>();
  cases<double, 16, 16>();
  cases<float, 32, 32>();
  cases<double, 32, 32>();
  cases<float, 16, 5>();
  cases<double, 16, 5>();
}
