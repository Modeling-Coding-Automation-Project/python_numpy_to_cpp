// Compile with -O2 -fstack-usage. Factors and input are caller-owned here.
#include "base_matrix_qr_decomposition.hpp"
using namespace Base::Matrix;

extern "C" __attribute__((noinline)) void qr_givens_factor_16(
    QRDecomposition<float, 16, 16> &solver, const Matrix<float, 16, 16> &A) {
  solver.solve(A);
}

#ifdef ALGORITHM_IMPROVEMENTS
extern "C" __attribute__((noinline)) void qr_householder_factor_16(
    QRDecompositionHouseholder<float, 16, 16> &solver,
    const Matrix<float, 16, 16> &A) {
  solver.solve(A);
}
extern "C" __attribute__((noinline)) void qr_householder_q_16(
    const QRDecompositionHouseholder<float, 16, 16> &solver,
    Matrix<float, 16, 16> &Q) {
  solver.get_Q(Q);
}
#endif
