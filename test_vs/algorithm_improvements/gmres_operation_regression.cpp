// Host-only tests for both fixed-loop and template-expanded GMRES operations.
// Build with and without BASE_MATRIX_USE_FOR_LOOP_OPERATION_.
#include "base_matrix_gmres.hpp"

#include <cmath>
#include <cstdio>

using namespace Base::Matrix;

static int failures = 0;

static inline void check(bool condition, const char *message) {
  if (!condition) {
    std::printf("FAIL: %s\n", message);
    ++failures;
  }
}

template <typename T> inline bool equal(T actual, T expected) {
  return actual == expected;
}

template <typename T>
inline bool equal(const Complex<T> &actual, const Complex<T> &expected) {
  return actual.real == expected.real && actual.imag == expected.imag;
}

template <typename T, std::size_t N>
inline void check_vector(const Vector<T, N> &actual,
                         const Vector<T, N> &expected, const char *message) {
  for (std::size_t i = 0; i < N; ++i) {
    check(equal(actual[i], expected[i]), message);
  }
}

template <typename T> inline bool near(T actual, T expected) {
  // The default external square root is approximate; use a relative bound.
  return std::abs(actual - expected) <= std::abs(expected) * T(2e-6);
}

template <typename T> inline void check_norms() {
  Vector<T, 3> zero;
  check(Numerical::norm(zero) == T(0), "zero norm");
  const T scales[] = {T(1), sizeof(T) == sizeof(float) ? T(1e-30) : T(1e-280),
                      sizeof(T) == sizeof(float) ? T(1e30) : T(1e280)};
  for (std::size_t i = 0; i < 3; ++i) {
    const T scale = scales[i];
    Vector<T, 3> value({T(3) * scale, T(-4) * scale, T(12) * scale});
    check(Numerical::norm(value, 0) == T(0), "empty active norm");
    check(near(Numerical::norm(value, 1), T(3) * scale), "one active norm");
    check(near(Numerical::norm(value, 2), T(5) * scale), "partial scaled norm");
    check(near(Numerical::norm(value), T(13) * scale), "full scaled norm");
  }
  Vector<T, 1> single({T(-7)});
  check(near(Numerical::norm(single), T(7)), "one dimensional norm");
  using C = Complex<T>;
  Vector<C, 3> complex({C(3, 4), C(0, -12), C(1e20, -1e20)});
  check(near(Numerical::norm(complex, 1), T(5)), "complex one active norm");
  check(near(Numerical::norm(complex, 2), T(13)), "complex norm masks tail");
  check(Numerical::norm(complex, 0) == T(0), "complex empty active norm");
}

template <typename T> struct FullCallback {
  inline void operator()(const Vector<T, 4> &input, Vector<T, 4> &output) const {
    const T sum = input[0] + input[1] + input[2] + input[3];
    for (std::size_t i = 0; i < 4; ++i) {
      output[i] = sum + T(i + 1);
    }
  }
};

template <typename T> inline void check_multiply() {
  Matrix<T, 4, 4> dense;
  dense(0, 0) = T(2);
  dense(0, 2) = T(-1);
  dense(2, 1) = T(3);
  dense(2, 3) = T(4);
  using Sparse = CompiledSparseMatrix<T, 4, 4, CSRIndices<0, 2, 1, 3>,
                                      CSRPointers<0, 2, 2, 4, 4>>;
  Sparse sparse({T(2), T(-1), T(3), T(4)});
  Vector<T, 4> input({T(1), T(2), T(3), T(4)});
  const Vector<T, 4> expected[] = {
      Vector<T, 4>(), Vector<T, 4>({T(2), T(0), T(0), T(0)}),
      Vector<T, 4>({T(2), T(0), T(0), T(0)}),
      Vector<T, 4>({T(-1), T(0), T(6), T(0)}),
      Vector<T, 4>({T(-1), T(0), T(22), T(0)})};
  for (std::size_t active = 0; active <= 4; ++active) {
    Vector<T, 4> output({T(99), T(99), T(99), T(99)});
    GmresOperation::multiply(dense, input, output, active);
    check_vector(output, expected[active], "dense leading subspace product");
    output = Vector<T, 4>({T(99), T(99), T(99), T(99)});
    GmresOperation::multiply(sparse, input, output, active);
    check_vector(output, expected[active], "sparse leading subspace product");

    FullCallback<T> callback;
    GmresOperation::apply_operator(callback, input, output, active);
    const T active_sum = T(active * (active + 1) / 2);
    for (std::size_t i = 0; i < 4; ++i) {
      check(equal(output[i], i < active ? active_sum + T(i + 1) : T()),
            "callback masks both input and output");
    }
  }
  using Empty = CompiledSparseMatrix<T, 4, 4, CSRIndices<>,
                                     CSRPointers<0, 0, 0, 0, 0>>;
  Empty empty;
  Vector<T, 4> output({T(99), T(99), T(99), T(99)});
  GmresOperation::multiply(empty, input, output, 4);
  check_vector(output, Vector<T, 4>(), "all-empty CSR product clears output");
  Matrix<T, 1, 1> single;
  single(0, 0) = T(-2);
  Vector<T, 1> single_output;
  GmresOperation::multiply(single, Vector<T, 1>({T(3)}), single_output, 1);
  check(equal(single_output[0], T(-6)), "one dimensional product");
}

template <typename MatrixType, typename T, std::size_t M, std::size_t N>
inline void check_normal(const MatrixType &matrix, const Vector<T, M> &rhs,
                         const Vector<T, N> &solution,
                         const Matrix<T, N, N> &expected_gram,
                         const Vector<T, N> &expected_rhs,
                         const Vector<T, N> &expected_product,
                         T expected_residual) {
  Matrix<T, N, N> gram;
  Vector<T, N> normal_rhs;
  for (std::size_t i = 0; i < N; ++i) {
    normal_rhs[i] = T(99);
    for (std::size_t j = 0; j < N; ++j) {
      gram(i, j) = T(99);
    }
  }
  GmresOperation::normal_equations(matrix, rhs, gram, normal_rhs);
  check_vector(normal_rhs, expected_rhs, "normal equation RHS");
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      check(gram(i, j) == expected_gram(i, j), "normal equation Gram");
    }
  }
  GmresOperation::NormalOperator<MatrixType, T, M, N> operation = {matrix};
  Vector<T, N> output;
  for (std::size_t i = 0; i < N; ++i) {
    output[i] = T(99);
  }
  operation.transpose_multiply(rhs, output);
  check_vector(output, expected_rhs, "normal transpose product clears output");
  operation(solution, output);
  check_vector(output, expected_product, "normal operator product");
  check(near(GmresOperation::normal_residual(matrix, rhs, solution),
             expected_residual), "normal residual from independent truth");
}

template <typename T> inline void check_rectangular() {
  // Closed-form results, including internal and trailing empty CSR rows.
  Matrix<T, 4, 2> tall({{1, 2}, {0, 0}, {-3, 4}, {0, 0}});
  using TallSparse = CompiledSparseMatrix<T, 4, 2, CSRIndices<0, 1, 0, 1>,
                                          CSRPointers<0, 2, 2, 4, 4>>;
  TallSparse tall_sparse({1, 2, -3, 4});
  Vector<T, 4> tall_rhs({3, 7, 5, -1});
  Vector<T, 2> tall_solution({2, -1});
  Matrix<T, 2, 2> tall_gram({{10, -10}, {-10, 20}});
  Vector<T, 2> tall_normal_rhs({-12, 26});
  Vector<T, 2> tall_product({30, -40});
  check_normal(tall, tall_rhs, tall_solution, tall_gram, tall_normal_rhs,
               tall_product, std::sqrt(T(6120)));
  check_normal(tall_sparse, tall_rhs, tall_solution, tall_gram, tall_normal_rhs,
               tall_product, std::sqrt(T(6120)));

  Matrix<T, 2, 4> wide({{1, 0, 2, 0}, {0, 0, -3, 4}});
  using WideSparse = CompiledSparseMatrix<T, 2, 4, CSRIndices<0, 2, 2, 3>,
                                          CSRPointers<0, 2, 4>>;
  WideSparse wide_sparse({1, 2, -3, 4});
  Vector<T, 2> wide_rhs({3, 5});
  Vector<T, 4> wide_solution({1, 2, -1, 3});
  Matrix<T, 4, 4> wide_gram({{1, 0, 2, 0}, {0, 0, 0, 0},
                            {2, 0, 13, -12}, {0, 0, -12, 16}});
  Vector<T, 4> wide_normal_rhs({3, 0, -9, 20});
  Vector<T, 4> wide_product({-1, 0, -47, 60});
  check_normal(wide, wide_rhs, wide_solution, wide_gram, wide_normal_rhs,
               wide_product, std::sqrt(T(3060)));
  check_normal(wide_sparse, wide_rhs, wide_solution, wide_gram, wide_normal_rhs,
               wide_product, std::sqrt(T(3060)));

  using Empty = CompiledSparseMatrix<T, 2, 4, CSRIndices<>, CSRPointers<0, 0, 0>>;
  Empty empty;
  check_normal(empty, wide_rhs, wide_solution, Matrix<T, 4, 4>(),
               Vector<T, 4>(), Vector<T, 4>(), T(0));
  Matrix<T, 1, 1> single({{2}});
  using SingleSparse =
      CompiledSparseMatrix<T, 1, 1, CSRIndices<0>, CSRPointers<0, 1>>;
  SingleSparse single_sparse({2});
  check_normal(single, Vector<T, 1>({6}), Vector<T, 1>({1}),
               Matrix<T, 1, 1>({{4}}), Vector<T, 1>({12}),
               Vector<T, 1>({4}), T(8));
  check_normal(single_sparse, Vector<T, 1>({6}), Vector<T, 1>({1}),
               Matrix<T, 1, 1>({{4}}), Vector<T, 1>({12}),
               Vector<T, 1>({4}), T(8));
}

template <typename T> inline void check_addition_order() {
  // Reversing accumulation loses the final unit to rounding.
  const T large = sizeof(T) == sizeof(float) ? T(33554432) : T(18014398509481984.0);
  Matrix<T, 3, 3> matrix;
  matrix(0, 0) = large;
  matrix(0, 1) = -large;
  matrix(0, 2) = T(1);
  using Sparse = CompiledSparseMatrix<T, 3, 3, CSRIndices<0, 1, 2>,
                                      CSRPointers<0, 3, 3, 3>>;
  Sparse sparse({large, -large, T(1)});
  Vector<T, 3> input({1, 1, 1});
  Vector<T, 3> output;
  GmresOperation::multiply(matrix, input, output, 3);
  check(output[0] == T(1), "dense product preserves accumulation order");
  GmresOperation::multiply(sparse, input, output, 3);
  check(output[0] == T(1), "sparse product preserves accumulation order");
  Matrix<T, 3, 1> column({{large}, {-large}, {1}});
  Matrix<T, 1, 1> gram;
  Vector<T, 1> normal_rhs;
  GmresOperation::normal_equations(column, input, gram, normal_rhs);
  check(normal_rhs[0] == T(1), "normal RHS preserves accumulation order");
}

template <typename T> inline void check_complex_product() {
  using C = Complex<T>;
  Matrix<C, 2, 2> matrix;
  matrix(0, 0) = C(0, 1);
  matrix(0, 1) = C(2, -1);
  matrix(1, 1) = C(-1, 2);
  using Sparse = CompiledSparseMatrix<C, 2, 2, CSRIndices<0, 1, 1>,
                                      CSRPointers<0, 2, 3>>;
  Sparse sparse({C(0, 1), C(2, -1), C(-1, 2)});
  Vector<C, 2> input({C(1, 2), C(3, -1)});
  Vector<C, 2> output;
  const Vector<C, 2> full_expected({C(3, -4), C(-1, 7)});
  const Vector<C, 2> partial_expected({C(-2, 1), C()});
  GmresOperation::multiply(matrix, input, output, 2);
  check_vector(output, full_expected, "dense complex phases");
  GmresOperation::multiply(sparse, input, output, 2);
  check_vector(output, full_expected, "sparse complex phases");
  GmresOperation::multiply(matrix, input, output, 1);
  check_vector(output, partial_expected, "dense complex partial product");
  GmresOperation::multiply(sparse, input, output, 1);
  check_vector(output, partial_expected, "sparse complex partial product");
}

int main() {
  check_norms<float>();
  check_norms<double>();
  check_multiply<float>();
  check_multiply<double>();
  check_multiply<Complex<float>>();
  check_multiply<Complex<double>>();
  check_rectangular<float>();
  check_rectangular<double>();
  check_addition_order<float>();
  check_addition_order<double>();
  check_complex_product<float>();
  check_complex_product<double>();
  std::printf("GMRES operation regression: %d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
