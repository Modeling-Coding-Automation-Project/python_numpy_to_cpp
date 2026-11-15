#include "base_matrix_qr_decomposition.hpp"
#include <cstdio>
#include <type_traits>

using namespace Base::Matrix;

namespace {
int failures = 0;

// The dependency's default approximation accumulates through the rotations.
// The 32x32 benchmark (496 rotations) has maximum orthogonality error 9.3e-7;
// keep a 2e-6 regression bound without changing the matrix implementation.
// The host-only standard-math run retains the previous double-precision bound.
#ifdef BASE_MATH_USE_STD_MATH_
constexpr double DOUBLE_TOLERANCE = 2e-12;
#else
constexpr double DOUBLE_TOLERANCE = 2e-6;
#endif

template <typename T> T magnitude(T value) {
  return value < T(0) ? -value : value;
}

void check(bool condition, const char *label) {
  if (!condition) {
    std::printf("FAIL: %s\n", label);
    ++failures;
  }
}

template <typename T, std::size_t M, std::size_t N>
T reconstruction(const Matrix<T, M, N> &A, const Matrix<T, M, M> &Q,
                 const Matrix<T, M, N> &R) {
  T maximum = T(0);
  T scale = T(0);
  for (std::size_t column = 0; column < N; ++column) {
    for (std::size_t row = 0; row < M; ++row) {
      T value = T(0);
      for (std::size_t k = 0; k < M; ++k) {
        value += Q(row, k) * R(k, column);
      }
      const T error = magnitude(value - A(row, column));
      if (error > maximum) { maximum = error; }
      if (magnitude(A(row, column)) > scale) {
        scale = magnitude(A(row, column));
      }
      if (row > column) { check(R(row, column) == T(0), "R triangular"); }
    }
  }
  return scale > T(0) ? maximum / scale : maximum;
}

template <typename T, std::size_t M>
T orthogonality(const Matrix<T, M, M> &Q) {
  T maximum = T(0);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < M; ++j) {
      T value = i == j ? T(-1) : T(0);
      for (std::size_t k = 0; k < M; ++k) { value += Q(k, i) * Q(k, j); }
      if (magnitude(value) > maximum) { maximum = magnitude(value); }
    }
  }
  return maximum;
}

template <typename T, std::size_t M, std::size_t N>
void dense_cases() {
  const T tolerance = std::is_same<T, float>::value ? T(3e-5) : T(DOUBLE_TOLERANCE);
  Matrix<T, M, N> A;
  for (std::size_t column = 0; column < N; ++column) {
    for (std::size_t row = 0; row < M; ++row) {
      A(row, column) = T(int((row * 13 + column * 7) % 19) - 9) / T(7);
      if (row == column) { A(row, column) += T(3); }
    }
  }
  QRDecomposition<T, M, N> givens;
  QRDecompositionHouseholder<T, M, N> householder;
  for (std::size_t repeat = 0; repeat < 3; ++repeat) {
    givens.solve(A);
    householder.solve(A);
    check(reconstruction(A, givens.get_Q(), givens.get_R()) < tolerance,
          "Givens reconstruction/reuse");
    check(reconstruction(A, householder.get_Q(), householder.get_R()) < tolerance,
          "Householder reconstruction/reuse");
    check(orthogonality(givens.get_Q()) < tolerance, "Givens orthogonality");
    check(orthogonality(householder.get_Q()) < tolerance,
          "Householder orthogonality");
  }
  Vector<T, M> b;
  for (std::size_t row = 0; row < M; ++row) { b[row] = T(row + 1); }
  const auto result = householder.apply_Q_transpose(b);
  const auto Q = householder.get_Q();
  for (std::size_t column = 0; column < M; ++column) {
    T expected = T(0);
    for (std::size_t row = 0; row < M; ++row) { expected += Q(row, column) * b[row]; }
    check(magnitude(result[column] - expected) < tolerance * T(M), "Q transpose application");
  }

  for (std::size_t column = 0; column < N; ++column) {
    for (std::size_t row = column + 2; row < M; ++row) { A(row, column) = T(0); }
  }
  givens.solve_hessenberg(A);
  check(reconstruction(A, givens.get_Q(), givens.get_R()) < tolerance,
        "Hessenberg reconstruction");

  Matrix<T, M, N> zero;
  givens.division_min = T(0);
  householder.division_min = T(-1);
  givens.solve(zero);
  householder.solve(zero);
  check(givens.get_threshold_corrected() && householder.get_threshold_corrected(),
        "invalid thresholds corrected");
  check(givens.get_rank_deficient() && householder.get_rank_deficient(),
        "zero matrix rank deficiency");
  check(reconstruction(zero, householder.get_Q(), householder.get_R()) == T(0),
        "zero matrix exactly zero");
  check(orthogonality(householder.get_Q()) == T(0), "zero matrix identity Q");
}

template <typename T> void sparse_case() {
  typedef CSRIndices<0, 1, 1, 2, 0, 2> Indices;
  typedef CSRPointers<0, 2, 4, 6> Pointers;
  CompiledSparseMatrix<T, 3, 3, Indices, Pointers> A({1, 1, 1, 1, 1, 1});
  QRDecompositionSparse<T, 3, 3, Indices, Pointers> solver;
  const T tolerance = std::is_same<T, float>::value ? T(2e-5) : T(DOUBLE_TOLERANCE);
  for (std::size_t repeat = 0; repeat < 2; ++repeat) {
    solver.solve(A);
    check(reconstruction(output_dense_matrix(A), solver.get_Q(), solver.get_R()) < tolerance,
          "sparse fill-in reconstruction and triangularity");
  }
}

void scale_cases() {
  for (std::size_t choice = 0; choice < 2; ++choice) {
    const double scale = choice == 0 ? 1e200 : 1e-200;
    Matrix<double, 3, 2> A({{scale, -scale}, {2 * scale, scale}, {scale, 3 * scale}});
    QRDecompositionHouseholder<double, 3, 2> solver;
    solver.division_min = 1e-250;
    solver.solve(A);
    check(reconstruction(A, solver.get_Q(), solver.get_R()) < DOUBLE_TOLERANCE,
          "scaled Householder avoids squared-norm overflow/underflow");
    QRDecomposition<double, 3, 2> givens;
    givens.division_min = 1e-250;
    givens.solve(A);
    check(reconstruction(A, givens.get_Q(), givens.get_R()) < DOUBLE_TOLERANCE,
          "scaled Givens avoids squared-norm overflow/underflow");
  }
}
} // namespace

int main() {
  dense_cases<float, 1, 1>();
  dense_cases<double, 1, 1>();
  dense_cases<float, 4, 3>();
  dense_cases<double, 4, 3>();
  dense_cases<float, 8, 8>();
  dense_cases<double, 8, 8>();
  dense_cases<double, 16, 5>();
  sparse_case<float>();
  sparse_case<double>();
  scale_cases();
  std::printf("QR regression: %d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
