// Host-only numerical regression tests; <cmath> is used only by the oracle.
#include "base_matrix_inverse.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace Base::Matrix;

static int failures = 0;
static void check(bool condition, const char *message) {
  if (!condition) {
    std::printf("FAIL: %s\n", message);
    ++failures;
  }
}

// The external default sqrt has about 9e-8 relative error on [1, 16].
// Allow it in direct norm/rotation checks; solution error checks retain their
// original strict bounds. Standard math keeps strict bounds for both.
template <typename T> T math_tolerance(T precise) {
  if (sizeof(T) == sizeof(float)) {
    return T(2e-5);
  }
#ifdef BASE_MATH_USE_STD_MATH_
  return precise;
#else
  static_cast<void>(precise);
  return T(2e-6);
#endif
}

// Independent host square root for solution and residual error checks.
template <typename T, std::size_t N>
typename Numerical::Scalar<T>::Real oracle_norm(const Vector<T, N> &input) {
  using Real = typename Numerical::Scalar<T>::Real;
  Real scale = Real(0);
  for (std::size_t i = 0; i < N; ++i) {
    Real candidate = Numerical::Scalar<T>::component_max(input[i]);
    if (candidate > scale) { scale = candidate; }
  }
  if (scale == Real(0)) { return Real(0); }
  Real square = Real(0);
  for (std::size_t i = 0; i < N; ++i) {
    square += Numerical::Scalar<T>::squared_abs(input[i] / scale);
  }
  return scale * std::sqrt(square);
}

template <typename T, std::size_t N> struct FullMatrixOperator {
  const Matrix<T, N, N> &matrix;
  void operator()(const Vector<T, N> &input, Vector<T, N> &output) const {
    output = matrix * input;
  }
};

template <typename T> void check_real() {
  const T tolerance = sizeof(T) == sizeof(float) ? T(2e-5) : T(1e-12);
  Matrix<T, 3, 3> matrix({{4, 1, 0}, {1, 3, 1}, {0, 1, 2}});
  Vector<T, 3> truth({1, -2, 3});
  Vector<T, 3> rhs = matrix * truth;
  Vector<T, 3> zero;
  GmresSolver<T, 3> solver;
  solver.division_min = T(1e-20);
  solver.relative_tolerance = tolerance;
  auto solution = solver.solve(matrix, rhs, zero);
  check(oracle_norm(solution - truth) < tolerance * T(10),
        "full real GMRES truth");
  T residual = oracle_norm(rhs - matrix * solution);
  check(std::abs(residual - solver.get_residual_norm()) < tolerance,
        "rho is true residual");
  check(solver.is_converged(), "convergence from true residual");
  auto unchanged = solver.solve(matrix, rhs, truth);
  check(oracle_norm(unchanged - truth) == T(0) &&
            solver.get_iterations() == 0,
        "exact initial solution unchanged");
  // An approximate sqrt need not produce an exactly unit-length basis.
  // Identity must still terminate after one Arnoldi step, with a correct solve.
  GmresSolver<T, 3> identity_solver;
  identity_solver.division_min = T(1e-5);
  identity_solver.relative_tolerance = tolerance;
  Vector<T, 3> identity_rhs({2, 3, 1});
  auto identity_solution = identity_solver.solve(
      Matrix<T, 3, 3>::identity(), identity_rhs, Vector<T, 3>());
  check(oracle_norm(identity_solution - identity_rhs) < tolerance * T(10) &&
            identity_solver.get_iterations() == 1,
        "identity handles approximately normalized Arnoldi basis in one step");
  Matrix<T, 3, 3> singular;
  unchanged = solver.solve(singular, rhs, truth);
  check(oracle_norm(unchanged - truth) == T(0) && solver.has_breakdown(),
        "singular system preserves last valid solution");
  solver.division_min = T(0);
  solver.solve(matrix, rhs, zero);
  check(solver.threshold_was_adjusted(),
        "zero threshold corrected with status");
  solver.division_min = T(-1);
  solver.solve(matrix, rhs, zero);
  check(solver.threshold_was_adjusted(),
        "negative threshold corrected with status");
  solver.cycles = 0;
  unchanged = solver.solve(matrix, rhs, truth);
  check(oracle_norm(unchanged - truth) == T(0),
        "zero cycle preserves initial value");

  Matrix<T, 3, 3> coupled({{4, 1, 2}, {1, 3, 2}, {2, 2, 5}});
  Vector<T, 3> partition_rhs({1, 2, 0});
  Vector<T, 3> partition_initial({0, 0, 7});
  solver.cycles = 1;
  solver.division_min = T(1e-20);
  auto partition = solver.solve(coupled, partition_rhs, partition_initial, 2);
  FullMatrixOperator<T, 3> callback = {coupled};
  auto callback_partition =
      solver.solve_operator(callback, partition_rhs, partition_initial, 2);
  check(
      oracle_norm(partition - callback_partition) < tolerance &&
          callback_partition[2] == T(7),
      "callback leading subspace masks input and output; inactive x unchanged");

  GmresSolver<T, 3, 1> restarted;
  restarted.division_min = T(1e-20);
  auto first = restarted.solve(matrix, rhs, zero);
  check(restarted.get_residual_norm() > tolerance,
        "K<N final subdiagonal retained");
  restarted.cycles = 32;
  auto improved = restarted.solve(matrix, rhs, zero);
  check(oracle_norm(improved - truth) < oracle_norm(first - truth),
        "restarts improve solution");

  Matrix<T, 3, 3> diagonal;
  diagonal(0, 0) = T(0.01);
  diagonal(1, 1) = T(1);
  diagonal(2, 2) = T(100);
  rhs = diagonal * truth;
  auto inverse_diagonal = gmres_inverse_diagonal(diagonal, T(1e-20));
  restarted.cycles = 1;
  restarted.relative_tolerance = tolerance;
  solution =
      restarted.solve_preconditioned(diagonal, rhs, zero, inverse_diagonal);
  check(oracle_norm(solution - truth) < tolerance * T(10),
        "fixed right diagonal preconditioning");

  Matrix<T, 1, 1> single;
  single(0, 0) = T(2);
  auto single_result =
      gmres_k(single, Vector<T, 1>({6}), Vector<T, 1>(), T(0), T(1e-20));
  check(std::abs(std::get<0>(single_result)[0] - T(3)) < tolerance,
        "one dimensional GMRES");

  Matrix<T, 3, 2> rectangular({{1, 0}, {0, 1}, {1, 1}});
  Vector<T, 3> inconsistent({1, 1, 3});
  auto least_squares =
      gmres_k_rect(rectangular, inconsistent, Vector<T, 2>(), T(0), T(1e-20));
  auto matrix_free = gmres_k_rect_matrix_free(rectangular, inconsistent,
                                              Vector<T, 2>(), T(0), T(1e-20));
  check(oracle_norm(std::get<0>(matrix_free) - std::get<0>(least_squares)) <
            tolerance,
        "explicit and matrix-free least squares agree");
  check(std::abs(std::get<0>(least_squares)[0] - T(4) / T(3)) <
            tolerance,
        "inconsistent least squares stationary truth");
  check(std::get<1>(least_squares) < tolerance * T(10),
        "rectangular rho is stationarity residual");

  using Sparse = CompiledSparseMatrix<T, 3, 2, CSRIndices<0, 1, 0, 1>,
                                      CSRPointers<0, 1, 2, 4>>;
  Sparse sparse({1, 1, 1, 1});
  auto sparse_result =
      sparse_gmres_k_rect(sparse, inconsistent, Vector<T, 2>(), T(0), T(1e-20));
  auto sparse_free = sparse_gmres_k_rect_matrix_free(
      sparse, inconsistent, Vector<T, 2>(), T(0), T(1e-20));
  check(oracle_norm(std::get<0>(sparse_free) - std::get<0>(least_squares)) <
            tolerance,
        "sparse matrix-free least squares agrees");
  check(oracle_norm(std::get<0>(sparse_result) -
                        std::get<0>(least_squares)) < tolerance,
        "sparse explicit normal operator");
}

template <typename T> inline void check_independent_gmres_tolerances() {
  // The small second Arnoldi direction is needed to recover x[1]. Use the
  // same thresholds for float and double: a type-dependent breakdown limit
  // previously discarded this direction for float after the first step.
  const T delta = T(2e-10);
  Matrix<T, 2, 2> matrix({{T(1), T(0)}, {delta, delta}});
  Vector<T, 2> truth({T(1), T(-1)});
  Vector<T, 2> rhs({T(1), T(0)});
  GmresSolver<T, 2> solver;
  solver.division_min = T(1e-20);
  solver.relative_tolerance = T(1e-12);
  auto solution = solver.solve(matrix, rhs, Vector<T, 2>());
  check(oracle_norm(solution - truth) < T(1e-6),
        "small Arnoldi direction recovers truth for both scalar types");
  check(solver.get_iterations() == 2 && solver.is_converged() &&
            !solver.has_breakdown() &&
            oracle_norm(rhs - matrix * solution) <=
                solver.relative_tolerance * oracle_norm(rhs),
        "strict residual tolerance retains the needed second GMRES step");

  // A loose residual tolerance deliberately accepts the first step. For this
  // ill-conditioned matrix, that small residual does not imply a small error
  // in x[1]; convergence and Arnoldi breakdown are separate decisions.
  solver.relative_tolerance = T(1e-6);
  solution = solver.solve(matrix, rhs, Vector<T, 2>());
  const T residual = oracle_norm(rhs - matrix * solution);
  check(solver.get_iterations() == 1 && solver.is_converged() &&
            !solver.has_breakdown() && residual > T(0) &&
            residual <= solver.relative_tolerance * oracle_norm(rhs),
        "loose residual tolerance permits early convergence without breakdown");
  check(oracle_norm(solution - truth) > T(0.5),
        "early residual convergence need not recover ill-conditioned truth");
}

template <typename T> void check_complex() {
  const T tolerance = sizeof(T) == sizeof(float) ? T(2e-5) : T(1e-12);
  const T root_tolerance = math_tolerance(T(1e-12));
  using C = Complex<T>;
  Vector<C, 2> norm_input({C(3, 4), C(0, 12)});
  check(complex_vector_norm(norm_input) == Base::Math::sqrt(T(169)),
        "complex norm delegates square-root policy to Base::Math");
  check(std::abs(complex_vector_norm(norm_input) - T(13)) <
            root_tolerance * T(13),
        "complex norm truth with configured math precision");
  Matrix<C, 2, 2> matrix;
  matrix(0, 0) = C(0, 1);
  matrix(1, 0) = C(1);
  matrix(1, 1) = C(1);
  Vector<C, 2> rhs({C(1), C(0)});
  GmresSolver<C, 2> solver;
  solver.relative_tolerance = tolerance;
  solver.division_min = T(1e-20);
  auto solution = solver.solve(matrix, rhs, Vector<C, 2>());
  Vector<C, 2> truth({C(0, -1), C(0, 1)});
  check(oracle_norm(solution - truth) < tolerance,
        "complex phase regression");
  check(oracle_norm(rhs - matrix * solution) < tolerance,
        "complex true residual");
  using Sparse =
      CompiledSparseMatrix<C, 2, 2, CSRIndices<0, 0, 1>, CSRPointers<0, 1, 3>>;
  Sparse sparse({C(0, 1), C(1), C(1)});
  auto sparse_result =
      complex_sparse_gmres_k(sparse, rhs, Vector<C, 2>(), tolerance, T(1e-20));
  check(oracle_norm(std::get<0>(sparse_result) - truth) < tolerance,
        "sparse complex phase and residual units");
  C phases[] = {C(), C(1), C(0, 1), C(-2, 3)};
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      T cosine;
      C sine;
      Numerical::givens(phases[i], phases[j], cosine, sine);
      C lower = -Numerical::Scalar<C>::conjugate(sine) * phases[i] +
                phases[j] * cosine;
      check(Numerical::Scalar<C>::magnitude(lower) < root_tolerance * T(10),
            "complex rotation annihilation");
      check(std::abs(cosine * cosine +
                            Numerical::Scalar<C>::squared_abs(sine) - T(1)) <
                root_tolerance,
            "complex rotation unitarity");
    }
  }
}

template <typename T> void check_manufactured_complex_systems() {
  using C = Complex<T>;
  const T tolerance = sizeof(T) == sizeof(float) ? T(2e-5) : T(2e-12);
  unsigned state = 17;
  for (std::size_t sample = 0; sample < 40; ++sample) {
    Matrix<C, 8, 8> matrix;
    Vector<C, 8> truth;
    for (std::size_t i = 0; i < 8; ++i) {
      truth[i] = C(T(i + 1) / T(8), T(8 - i) / T(16));
      for (std::size_t j = 0; j < 8; ++j) {
        state = 1664525u * state + 1013904223u;
        T real = T(int((state >> 16) % 101) - 50) / T(100);
        state = 1664525u * state + 1013904223u;
        T imag = T(int((state >> 16) % 101) - 50) / T(100);
        matrix(i, j) = C(real + (i == j ? T(8) : T(0)), imag);
      }
    }
    Vector<C, 8> rhs = matrix * truth;
    GmresSolver<C, 8> solver;
    solver.division_min = T(1e-20);
    solver.relative_tolerance = tolerance;
    auto solution = solver.solve(matrix, rhs, Vector<C, 8>());
    check(oracle_norm(solution - truth) < tolerance * T(20),
          "manufactured complex 8x8 truth over 40 deterministic fixtures");
  }
}

int main() {
  check_real<float>();
  check_real<double>();
  check_independent_gmres_tolerances<float>();
  check_independent_gmres_tolerances<double>();
  check_complex<float>();
  check_complex<double>();
  check_manufactured_complex_systems<float>();
  check_manufactured_complex_systems<double>();
  for (std::size_t scale_index = 0; scale_index < 2; ++scale_index) {
    const double scale = scale_index == 0 ? 1e150 : 1e-150;
    Matrix<double, 2, 2> matrix({{4 * scale, scale}, {scale, 3 * scale}});
    Vector<double, 2> truth({1, -2});
    GmresSolver<double, 2> solver;
    solver.division_min = 1e-200;
    solver.relative_tolerance = 1e-12;
    auto solution = solver.solve(matrix, matrix * truth, Vector<double, 2>());
    check(oracle_norm(solution - truth) < 1e-12,
          "GMRES uniformly scaled systems");
  }
#ifndef BASE_MATH_USE_STD_MATH_
  // A strict tolerance must agree with the independently measured residual.
  Matrix<double, 3, 3> strict_matrix({{4, 1, 0}, {1, 3, 1}, {0, 1, 2}});
  Vector<double, 3> strict_truth({1, -2, 3});
  Vector<double, 3> strict_rhs = strict_matrix * strict_truth;
  GmresSolver<double, 3> strict_solver;
  strict_solver.division_min = 1e-20;
  strict_solver.relative_tolerance = 1e-12;
  auto strict_solution =
      strict_solver.solve(strict_matrix, strict_rhs, Vector<double, 3>());
  double strict_residual = oracle_norm(strict_rhs - strict_matrix * strict_solution);
  check(strict_solver.is_converged() ==
            (strict_residual <= 1e-12 * oracle_norm(strict_rhs)),
        "strict convergence status agrees with independent residual");
#endif
  auto protected_quotient = Numerical::Scalar<Complex<double>>::divide(
      Complex<double>(1), Complex<double>(0, 1e-100), 1e-10);
  check(std::abs(protected_quotient.imag + 1e10) < 1e-4 &&
            protected_quotient.real == 0,
        "complex denominator floor preserves phase and magnitude units");
  auto large_quotient = Numerical::Scalar<Complex<double>>::divide(
      Complex<double>(1e308), Complex<double>(0.5, 0.5), 1e-20);
  check(std::isfinite(large_quotient.real) &&
            std::isfinite(large_quotient.imag) &&
            std::abs(large_quotient.real / 1e308 - 1) < 1e-14 &&
            std::abs(large_quotient.imag / 1e308 + 1) < 1e-14,
        "complex scaled divide avoids intermediate quotient overflow");
  Vector<double, 2> large({1e150, -1e150});
  Vector<double, 2> small({1e-150, -1e-150});
  check(std::abs(Numerical::norm(large) / 1e150 - std::sqrt(2.0)) < 1e-14,
        "scaled large norm");
  check(std::abs(Numerical::norm(small) / 1e-150 - std::sqrt(2.0)) < 1e-14,
        "scaled tiny norm");
  Matrix<double, 1, 1> identity;
  identity(0, 0) = 1;
  Vector<double, 1> tiny_rhs({1e-310});
  auto tiny = gmres_k(identity, tiny_rhs, Vector<double, 1>(), 0.0, 1e-320);
  check(std::get<0>(tiny)[0] == tiny_rhs[0],
        "subnormal normalization avoids reciprocal overflow");
  Matrix<double, 1, 1> tiny_diagonal;
  tiny_diagonal(0, 0) = 1e-320;
  check(gmres_inverse_diagonal(tiny_diagonal, 1e-321)[0] == 1.0,
        "unrepresentable preconditioner reciprocal uses identity");
  std::printf("GMRES regression failures: %d\n", failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
