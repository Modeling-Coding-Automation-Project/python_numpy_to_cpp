// Host-only regression checks: standard math and diagnostics stay out of headers.
#include "base_matrix_lu_decomposition.hpp"
#include "base_matrix_cholesky_decomposition.hpp"
#include <cmath>
#include <cstdio>
#include <type_traits>

using namespace Base::Matrix;
int failures = 0;
void check(bool condition, const char *message) {
  if (!condition) {
    std::printf("FAIL: %s\n", message);
    ++failures;
  }
}

template <typename T, std::size_t N>
T error(const Matrix<T, N, N> &a, const Matrix<T, N, N> &b) {
  T result = T(0);
  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i < N; ++i) {
      T value = Base::Math::abs(a(i, j) - b(i, j));
      if (value > result) {
        result = value;
      }
    }
  }
  return result;
}

template <typename T> void regression() {
  T tolerance = std::is_same<T, float>::value ? T(2e-5) : T(2e-6);
  Matrix<T, 3, 3> a({{T(0), T(1), T(2)},
                      {T(1), T(3), T(1)},
                      {T(3), T(1), T(4)}});
  Vector<T, 3> truth({T(1), T(-2), T(3)});
  Vector<T, 3> b = a * truth;
  LUDecomposition<T, 3> lu;
  Vector<T, 3> x = lu.solve(a, b);
  check(lu.is_valid(), "LU valid");
  check(error(lu.get_P() * a, lu.get_L() * lu.get_U()) < tolerance,
        "pivoted PA=LU");
  check(Base::Math::abs(lu.get_determinant() - T(-17)) < tolerance,
        "LU determinant parity");
  for (std::size_t i = 0; i < 3; ++i) {
    check(Base::Math::abs(x[i] - truth[i]) < tolerance, "LU truth");
    check(lu.get_L()(i, i) == T(1), "LU unit diagonal");
  }
  LUDecomposition<T, 3> copied(lu);
  lu.solve_scaled(a);
  check(error(lu.get_P() * a, lu.get_L() * lu.get_U()) < tolerance,
        "scaled pivot PA=LU");
  LUDecomposition<T, 3, true> cached;
  cached.solve(a);
  Vector<T, 3> cached_x = cached.solve_rhs(b);
  Vector<T, 3> copied_x = copied.solve_rhs(b);
  for (std::size_t i = 0; i < 3; ++i) {
    check(Base::Math::abs(cached_x[i] - copied_x[i]) < tolerance,
          "cached inverse and copied factors");
  }
  Vector<T, 3> refined = copied.template refine<double>(a, b, copied_x, 2);
  for (std::size_t i = 0; i < 3; ++i) {
    check(Base::Math::abs(refined[i] - truth[i]) < tolerance,
          "fixed mixed precision refinement");
  }
  Matrix<T, 3, 3> zero;
  lu.solve(zero);
  check(lu.is_singular() && !lu.is_valid(), "LU singular status");
  check(lu.solve_rhs(b)[0] == T(0), "LU failed factors not divided");
  lu.division_min = T(-1);
  lu.solve(a);
  check(lu.threshold_was_adjusted() && lu.is_valid(), "LU invalid threshold");

  Matrix<T, 3, 3> spd({{T(4), T(2), T(0)},
                        {T(2), T(5), T(1)},
                        {T(0), T(1), T(3)}});
  Matrix<T, 3, 3> fallback = Matrix<T, 3, 3>::identity();
  auto result = cholesky_decomposition(spd, fallback, T(1e-10));
  check(!std::get<1>(result), "Cholesky success");
  auto upper = std::get<0>(result);
  check(error(output_matrix_transpose(upper) * upper, spd) < tolerance,
        "Cholesky reconstruction");
  Matrix<T, 3, 3> inplace = spd;
  check(!cholesky_decomposition_to(inplace, inplace, T(1e-10)) &&
            error(output_matrix_transpose(inplace) * inplace, spd) < tolerance,
        "Cholesky output-reference alias");
  PackedCholesky<T, 3> packed;
  packed.solve(spd);
  upper = packed.get_U();
  check(packed.is_valid() && error(output_matrix_transpose(upper) * upper, spd) < tolerance,
        "packed Cholesky reconstruction");
  Vector<T, 3> packed_x = packed.solve_rhs(spd * truth);
  for (std::size_t i = 0; i < 3; ++i) {
    check(Base::Math::abs(packed_x[i] - truth[i]) < tolerance,
          "packed Cholesky solve");
  }
  // Lower input alone suffices; the old sparse path also used this triangle.
  using Indices = CSRIndices<0, 0, 1, 0, 1, 2>;
  using Pointers = CSRPointers<0, 1, 3, 6>;
  CompiledSparseMatrix<T, 3, 3, Indices, Pointers> sparse(
      {T(4), T(2), T(5), T(0), T(1), T(3)});
  auto sparse_result = cholesky_decomposition_sparse(sparse, fallback, T(1e-10));
  check(!std::get<1>(sparse_result) &&
            error(std::get<0>(sparse_result), std::get<0>(result)) < tolerance,
        "sparse lower scatter");
  packed.solve(sparse);
  check(error(packed.get_U(), std::get<0>(result)) < tolerance,
        "packed sparse lower scatter");
  using DiagonalIndices = CSRIndices<0, 1, 2>;
  using DiagonalPointers = CSRPointers<0, 1, 2, 3>;
  CompiledSparseMatrix<T, 3, 3, DiagonalIndices, DiagonalPointers> sparse_diagonal(
      {T(1), T(4), T(9)});
  packed.solve(sparse_diagonal);
  check(packed.get_U()(0, 1) == T(0) && packed.get_U()(1, 2) == T(0),
        "packed sparse workspace reset");
  auto failed = cholesky_decomposition(zero, fallback, T(1e-10));
  check(std::get<1>(failed) && error(std::get<0>(failed), fallback) == T(0),
        "Cholesky fallback");
  failed = cholesky_decomposition(spd, fallback, T(0));
  check(std::get<1>(failed), "Cholesky invalid threshold reported");
  DiagMatrix<T, 3> diagonal({T(0), T(1e-12), T(4)});
  auto diag_result = cholesky_decomposition_diag(diagonal, diagonal);
  check(!std::get<1>(diag_result) && std::get<0>(diag_result)[0] == T(0),
        "diagonal exact zero");
  check(Base::Math::abs(std::get<0>(diag_result)[1] - T(1e-6)) < T(1e-10),
        "diagonal small positive has no unrelated floor");
  SPDLDLT<T, 3> ldlt;
  ldlt.solve(spd);
  auto lower = ldlt.get_L();
  Matrix<T, 3, 3> d = output_dense_matrix(ldlt.get_D());
  check(ldlt.is_valid() && error(lower * d * output_matrix_transpose(lower), spd) < tolerance,
        "SPD LDLT reconstruction");
  Vector<T, 3> ldlt_x = ldlt.solve_rhs(spd * truth);
  for (std::size_t i = 0; i < 3; ++i) {
    check(Base::Math::abs(ldlt_x[i] - truth[i]) < tolerance, "LDLT truth");
  }
  spd(0, 0) = T(-1);
  ldlt.solve(spd);
  check(!ldlt.is_valid(), "LDLT rejects indefinite");
  packed.solve(spd);
  check(!packed.is_valid(), "Cholesky rejects indefinite");
  Matrix<T, 1, 1> large({{T(1e30)}});
  PackedCholesky<T, 1> large_chol;
  large_chol.solve(large);
  check(large_chol.is_valid() &&
            Base::Math::abs(large_chol.get_U()(0, 0) / T(1e15) - T(1)) < tolerance,
        "rsqrt output is not clamped by input floor");
  PackedCholesky<T, 3> pivot_policy;
  pivot_policy.relative_pivot_min = T(2);
  pivot_policy.solve(Matrix<T, 3, 3>::identity());
  check(!pivot_policy.is_valid(), "relative pivot floor is configurable");
  pivot_policy.relative_pivot_min = T(-1);
  pivot_policy.division_min = T(0);
  pivot_policy.solve(Matrix<T, 3, 3>::identity());
  check(pivot_policy.threshold_was_adjusted() && pivot_policy.is_valid(),
        "packed invalid thresholds recorded");
  Matrix<T, 1, 1> one({{T(4)}});
  LUDecomposition<T, 1> one_lu;
  one_lu.solve(one);
  check(one_lu.solve_rhs(Vector<T, 1>({T(8)}))[0] == T(2), "1x1 LU");
  PackedCholesky<T, 1> one_chol;
  one_chol.solve(one);
  check(Base::Math::abs(one_chol.get_U()(0, 0) - T(2)) < tolerance,
        "1x1 Cholesky");
}

void subnormal_pivots() {
  Matrix<double, 2, 2> a({{4e-310, 1e-310}, {1e-310, 3e-310}});
  Vector<double, 2> truth({1, 2});
  Vector<double, 2> b = a * truth;
  LUDecomposition<double, 2> lu;
  LUDecomposition<double, 2, true> cached;
  SPDLDLT<double, 2> ldlt;
  lu.division_min = cached.division_min = ldlt.division_min = 1e-320;
  lu.solve(a);
  cached.solve(a);
  ldlt.solve(a);
  auto x = lu.solve_rhs(b);
  auto xc = cached.solve_rhs(b);
  auto xl = ldlt.solve_rhs(b);
  for (std::size_t i = 0; i < 2; ++i) {
    check(std::isfinite(x[i]) && std::abs(x[i] - truth[i]) < 1e-12,
          "LU avoids overflowing reciprocal");
    check(std::isfinite(xc[i]) && std::abs(xc[i] - truth[i]) < 1e-12,
          "LU cache falls back for subnormal pivot");
    check(std::isfinite(xl[i]) && std::abs(xl[i] - truth[i]) < 1e-12,
          "LDLT avoids overflowing reciprocal");
  }
}

int main() {
  regression<float>();
  regression<double>();
  subnormal_pivots();
  std::printf("LU/Cholesky regression failures: %d\n", failures);
  return failures != 0;
}
