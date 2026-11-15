// Host-only allocation audit of the default fixed-storage public APIs.
// Also compile this TU with -fno-exceptions and a poisoned cmath header.
// The runner checks algorithm usage in repository sources, excluding submodules.
#include "python_numpy.hpp"
#include <cstdio>
#include <cstdlib>
#include <new>

namespace {
std::size_t allocations = 0;
bool count_allocations = false;

template <typename T> void consume(const T &value) {
  asm volatile("" : : "g"(&value) : "memory");
}
} // namespace

void *operator new(std::size_t size) {
  if (count_allocations) { ++allocations; }
  return std::malloc(size);
}
void *operator new[](std::size_t size) {
  if (count_allocations) { ++allocations; }
  return std::malloc(size);
}
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }

int main() {
#ifdef BASE_MATRIX_USE_STD_VECTOR_
  std::printf("Allocation audit skipped: explicitly dynamic compatibility configuration\n");
  return 0;
#else
  using namespace Base::Matrix;
  count_allocations = true;
  Matrix<double, 4, 4> A({{5, 1, 0, 0}, {1, 6, 1, 0},
                          {0, 1, 7, 1}, {0, 0, 1, 8}});
  Vector<double, 4> b({1, 2, 3, 4});
  Vector<double, 4> x0;
  {
    QRDecomposition<double, 4, 4> givens;
    givens.solve(A);
    consume(givens.get_Q());
    consume(givens.get_R());
    givens.solve_hessenberg(A);
    consume(givens);
    QRDecompositionHouseholder<double, 4, 4> householder;
    householder.solve(A);
    consume(householder.get_Q());
    consume(householder.get_R());
    consume(householder.apply_Q_transpose(b));
    typedef CSRIndices<0, 1, 1, 2, 0, 2> Indices;
    typedef CSRPointers<0, 2, 4, 6> Pointers;
    CompiledSparseMatrix<double, 3, 3, Indices, Pointers> sparse({1, 1, 1, 1, 1, 1});
    QRDecompositionSparse<double, 3, 3, Indices, Pointers> sparse_solver;
    sparse_solver.solve(sparse);
    consume(sparse_solver.get_Q());
    consume(sparse_solver.get_R());
  }
  {
    LUDecomposition<double, 4, true> lu;
    lu.solve(A);
    consume(lu.get_L());
    consume(lu.get_U());
    consume(lu.get_P());
    const auto x = lu.solve_rhs(b);
    consume(lu.refine<double>(A, b, x, 2));
    lu.solve_scaled(A);
    consume(lu.solve_rhs(b));
    PackedCholesky<double, 4> cholesky;
    cholesky.solve(A);
    consume(cholesky.get_U());
    consume(cholesky.solve_rhs(b));
    SPDLDLT<double, 4> ldlt;
    ldlt.solve(A);
    consume(ldlt.get_L());
    consume(ldlt.get_D());
    consume(ldlt.solve_rhs(b));
  }
  {
    GmresSolver<double, 4, 2> gmres;
    gmres.cycles = 3;
    gmres.orthogonalization_passes = 2;
    consume(gmres.solve(A, b, x0));
    const auto diagonal = gmres_inverse_diagonal(A, 1e-12);
    consume(gmres.solve_preconditioned(A, b, x0, diagonal));
    Matrix<double, 4, 4> inverse0;
    consume(gmres_k_matrix_inv(A, 1e-10, 1e-12, inverse0));
    Matrix<double, 6, 4> rectangular;
    Vector<double, 6> rectangular_b;
    for (std::size_t row = 0; row < 6; ++row) {
      rectangular_b[row] = double(row + 1);
      for (std::size_t column = 0; column < 4; ++column) {
        rectangular(row, column) = row == column ? 3.0 : 0.1;
      }
    }
    consume(gmres_k_rect(rectangular, rectangular_b, x0, 1e-10, 1e-12));
    consume(gmres_k_rect_matrix_free(rectangular, rectangular_b, x0, 1e-10, 1e-12));
    Matrix<Complex<double>, 4, 4> complex_A;
    Vector<Complex<double>, 4> complex_b;
    Vector<Complex<double>, 4> complex_x0;
    for (std::size_t row = 0; row < 4; ++row) {
      complex_b[row] = Complex<double>(b[row], -b[row]);
      for (std::size_t column = 0; column < 4; ++column) {
        complex_A(row, column) = Complex<double>(A(row, column), row == column ? 1.0 : 0.0);
      }
    }
    GmresSolver<Complex<double>, 4, 2> complex_gmres;
    complex_gmres.cycles = 3;
    complex_gmres.orthogonalization_passes = 2;
    consume(complex_gmres.solve(complex_A, complex_b, complex_x0));
  }
  {
    EigenSolverReal<double, 4> real_solver;
    real_solver.solve_eigen_values_and_vectors(A);
    consume(real_solver.get_eigen_values());
    consume(real_solver.get_eigen_vectors());
    EigenSolverComplex<double, 4> complex_solver;
    complex_solver.solve_eigen_values_and_vectors(A);
    consume(complex_solver.get_eigen_values());
    consume(complex_solver.get_eigen_vectors());
    EigenSolverSymmetric<double, 4, true> symmetric;
    symmetric.solve_eigen_values_and_vectors(A);
    consume(symmetric.get_eigen_values());
    consume(symmetric.get_eigen_vectors());
    EigenSolverSymmetric<double, 4, false> symmetric_values;
    symmetric_values.solve_eigen_values(A);
    consume(symmetric_values.get_eigen_values());
  }
  count_allocations = false;
  std::printf("Fixed-storage allocation audit: %zu allocations\n", allocations);
  return allocations == 0 ? 0 : 1;
#endif
}
