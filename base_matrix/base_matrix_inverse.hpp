/**
 * @file base_matrix_inverse.hpp
 * @brief GMRES-based matrix inverse and linear solver utilities for dense,
 * sparse, and complex matrices.
 *
 * This header provides a collection of template functions for solving linear
 * systems and computing matrix inverses using the GMRES (Generalized Minimal
 * Residual) iterative method. It supports dense matrices, sparse matrices
 * (CSR-like), and complex-valued matrices, as well as diagonal matrix
 * inversion. The implementation is generic and works with fixed-size matrices
 * and vectors, supporting both real and complex types.
 *
 * The GMRES wrappers use GmresSolver with K equal to the system dimension and
 * one cycle. decay_rate is a relative tolerance against the right-hand-side
 * norm. rho is the true residual norm for both real and complex arithmetic;
 * rectangular solves report the normal-equation residual norm. Earlier
 * versions used an initial-residual reference and inconsistent complex squared
 * norms. Use GmresSolver directly for restart, orthogonalization and status.
 *
 * @note
 * tparam M is the number of rows in the matrix.
 * tparam N is the number of columns in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef BASE_MATRIX_INVERSE_HPP_
#define BASE_MATRIX_INVERSE_HPP_

#include "base_matrix_macros.hpp"

#include "base_math.hpp"
#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_complex.hpp"
#include "base_matrix_gmres.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"

#include <cstddef>
#include <tuple>

namespace Base {
namespace Matrix {

/* GMRES K */

namespace InverseOperation {

/**
 * @brief Core function for GMRES K method for dense matrices.
 *
 * This function implements the GMRES K algorithm to solve the linear system
 * Ax = b, where A is a square matrix, b is a vector, and x_1 is an initial
 * guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The input matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param matrix_size The size of the matrix (M).
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
template <typename T, std::size_t M>
inline typename std::enable_if<(M > 1),
                               std::tuple<Vector<T, M>, T, std::size_t>>::type
gmres_k_core(const Matrix<T, M, M> &A, const Vector<T, M> &b,
             const Vector<T, M> &x_1, const T &decay_rate,
             const T &division_min, const std::size_t &matrix_size) {
  GmresSolver<T, M> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  Vector<T, M> solution = solver.solve(A, b, x_1, matrix_size);
  return std::make_tuple(solution, solver.get_residual_norm(),
                         solver.get_iterations());
}

/**
 * @brief Core function for GMRES K method for 1x1 matrices.
 *
 * This specialization handles the case where the matrix size is 1x1, which is a
 * trivial case. It simply computes the inverse of the single element in the
 * matrix.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M), must be 1 in this case.
 * @param A The input matrix A, which is expected to be 1x1.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param matrix_size The size of the matrix (M), must be 1 in this case.
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
template <typename T, std::size_t M>
inline typename std::enable_if<(M <= 1),
                               std::tuple<Vector<T, M>, T, std::size_t>>::type
gmres_k_core(const Matrix<T, M, M> &A, const Vector<T, M> &b,
             const Vector<T, M> &x_1, const T &decay_rate,
             const T &division_min, const std::size_t &matrix_size) {
  GmresSolver<T, M> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  Vector<T, M> solution = solver.solve(A, b, x_1, matrix_size);
  return std::make_tuple(solution, solver.get_residual_norm(),
                         solver.get_iterations());
}

} // namespace InverseOperation

/**
 * @brief GMRES K method for dense matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a square matrix, b is a vector, and x_1 is an initial guess.
 * It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The input matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
template <typename T, std::size_t M>
inline std::tuple<Vector<T, M>, T, std::size_t>
gmres_k(const Matrix<T, M, M> &A, const Vector<T, M> &b,
        const Vector<T, M> &x_1, const T &decay_rate, const T &division_min) {

  return InverseOperation::gmres_k_core<T, M>(A, b, x_1, decay_rate,
                                              division_min, M);
}

/**
 * @brief GMRES K method for partitioned matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a square matrix, b is a vector, and x_1 is an initial guess.
 * It supports partitioning of the matrix and returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The input matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param matrix_size The size of the matrix (M).
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
template <typename T, std::size_t M>
inline std::tuple<Vector<T, M>, T, std::size_t>
gmres_k_partition(const Matrix<T, M, M> &A, const Vector<T, M> &b,
                  const Vector<T, M> &x_1, const T &decay_rate,
                  const T &division_min, const std::size_t &matrix_size) {

  return InverseOperation::gmres_k_core<T, M>(A, b, x_1, decay_rate,
                                              division_min, matrix_size);
}

/* GMRES K for Matrix */

/**
 * @brief GMRES K method for matrices.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a square matrix, B is a matrix with multiple right-hand side
 * vectors, and X_1 is an initial guess. It returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam K The number of columns of matrix B.
 * @param A The input matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M, std::size_t K>
inline std::tuple<Matrix<T, M, K>, std::array<T, K>, std::array<std::size_t, K>>
gmres_k_matrix(const Matrix<T, M, M> &A, const Matrix<T, M, K> &B,
               const Matrix<T, M, K> &X_1, const T &decay_rate,
               const T &division_min) {

  Matrix<T, M, K> X = X_1;
  std::array<T, K> rho = {};
  std::array<std::size_t, K> rep_num = {};

  for (std::size_t i = 0; i < K; i++) {
    auto result = Base::Matrix::gmres_k(A, B.get_row(i), X.get_row(i),
                                        decay_rate, division_min);
    X.set_row(i, std::get<0>(result));
    rho[i] = std::get<1>(result);
    rep_num[i] = std::get<2>(result);
  }

  return std::make_tuple(X, rho, rep_num);
}

/**
 * @brief GMRES K method for matrices with diagonal matrix B.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a square matrix, B is a diagonal matrix, and X_1 is an initial
 * guess. It returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The input matrix A.
 * @param B The diagonal matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M>
inline std::tuple<Matrix<T, M, M>, std::array<T, M>, std::array<std::size_t, M>>
gmres_k_matrix(const Matrix<T, M, M> &A, const DiagMatrix<T, M> &B,
               const Matrix<T, M, M> &X_1, const T &decay_rate,
               const T &division_min) {

  Matrix<T, M, M> X = X_1;
  std::array<T, M> rho = {};
  std::array<std::size_t, M> rep_num = {};

  for (std::size_t i = 0; i < M; i++) {
    auto result = Base::Matrix::gmres_k(A, B.get_row(i), X.get_row(i),
                                        decay_rate, division_min);
    X.set_row(i, std::get<0>(result));
    rho[i] = std::get<1>(result);
    rep_num[i] = std::get<2>(result);
  }

  return std::make_tuple(X, rho, rep_num);
}

/**
 * @brief GMRES K method for partitioned matrices.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a square matrix, B is a matrix with multiple right-hand side
 * vectors, and X_1 is an initial guess. It supports partitioning of the matrix
 * and returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam K The number of columns of matrix B.
 * @param A The input matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param matrix_size The size of the matrix (M).
 * @return A tuple of (solution matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M, std::size_t K>
inline std::tuple<Matrix<T, M, K>, std::array<T, K>, std::array<std::size_t, K>>
gmres_k_partition_matrix(const Matrix<T, M, M> &A, const Matrix<T, M, K> &B,
                         const Matrix<T, M, K> &X_1, const T &decay_rate,
                         const T &division_min,
                         const std::size_t &matrix_size) {

  Matrix<T, M, K> X = X_1;
  std::array<T, K> rho = {};
  std::array<std::size_t, K> rep_num = {};

  std::size_t repeat_number;
  if (matrix_size < K) {
    repeat_number = matrix_size;
  } else {
    repeat_number = K;
  }

  for (std::size_t i = 0; i < repeat_number; i++) {
    auto result = Base::Matrix::gmres_k_partition(
        A, B.get_row(i), X.get_row(i), decay_rate, division_min, matrix_size);
    X.set_row(i, std::get<0>(result));
    rho[i] = std::get<1>(result);
    rep_num[i] = std::get<2>(result);
  }

  return std::make_tuple(X, rho, rep_num);
}

/**
 * @brief GMRES K method for partitioned matrices with diagonal matrix B.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a square matrix, B is a diagonal matrix, and X_1 is an initial
 * guess. It supports partitioning of the matrix and returns the solution
 * matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The input matrix A.
 * @param B The diagonal matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param matrix_size The size of the matrix (M).
 * @return A tuple of (solution matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M>
inline std::tuple<Matrix<T, M, M>, std::array<T, M>, std::array<std::size_t, M>>
gmres_k_partition_matrix(const Matrix<T, M, M> &A, const DiagMatrix<T, M> &B,
                         const Matrix<T, M, M> &X_1, const T &decay_rate,
                         const T &division_min,
                         const std::size_t &matrix_size) {

  Matrix<T, M, M> X = X_1;
  std::array<T, M> rho = {};
  std::array<std::size_t, M> rep_num = {};

  const std::size_t active_columns = matrix_size < M ? matrix_size : M;
  for (std::size_t i = 0; i < active_columns; i++) {
    auto result = Base::Matrix::gmres_k_partition(
        A, B.get_row(i), X.get_row(i), decay_rate, division_min, matrix_size);
    X.set_row(i, std::get<0>(result));
    rho[i] = std::get<1>(result);
    rep_num[i] = std::get<2>(result);
  }

  return std::make_tuple(X, rho, rep_num);
}

/* GMRES K for rectangular matrix */

/**
 * @brief GMRES K method for rectangular matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a rectangular matrix (M x N), b is a vector, and x_1 is an
 * initial guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The number of columns in the matrix A.
 * @tparam N The number of rows in the matrix A.
 * @param In_A The input matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
// Optional low-memory operator path; no explicit Gram matrix.
template <typename T, std::size_t M, std::size_t N>
inline std::tuple<Vector<T, N>, T, std::size_t>
gmres_k_rect_matrix_free(const Matrix<T, M, N> &In_A, const Vector<T, M> &b,
                         const Vector<T, N> &x_1, const T &decay_rate,
                         const T &division_min) {
  static_assert(M > N,
                "GMRES rectangular solve requires more rows than columns.");
  GmresOperation::NormalOperator<Matrix<T, M, N>, T, M, N> operation = {In_A};
  Vector<T, N> right_hand_side;
  operation.transpose_multiply(b, right_hand_side);
  GmresSolver<T, N> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  Vector<T, N> solution =
      solver.solve_operator(operation, right_hand_side, x_1);
  return std::make_tuple(solution,
                         GmresOperation::normal_residual(In_A, b, solution),
                         solver.get_iterations());
}

template <typename T, std::size_t M, std::size_t N>
inline std::tuple<Vector<T, N>, T, std::size_t>
gmres_k_rect(const Matrix<T, M, N> &In_A, const Vector<T, M> &b,
             const Vector<T, N> &x_1, const T &decay_rate,
             const T &division_min) {
  static_assert(M > N,
                "GMRES rectangular solve requires more rows than columns.");
  Matrix<T, N, N> gram;
  Vector<T, N> normal_rhs;
  GmresOperation::normal_equations(In_A, b, gram, normal_rhs);
  GmresSolver<T, N> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  Vector<T, N> solution = solver.solve(gram, normal_rhs, x_1);
  return std::make_tuple(solution,
                         GmresOperation::normal_residual(In_A, b, solution),
                         solver.get_iterations());
}

/**
 * @brief GMRES K method for rectangular matrices with multiple right-hand side
 * vectors.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a rectangular matrix (M x N), B is a matrix with multiple right-
 * hand side vectors, and X_1 is an initial guess. It returns the solution
 * matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The number of rows in the matrix A.
 * @tparam N The number of columns in the matrix A.
 * @tparam K The number of columns of matrix B.
 * @param A The input matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K>
inline std::tuple<Matrix<T, N, K>, std::array<T, K>, std::array<std::size_t, K>>
gmres_k_rect_matrix(const Matrix<T, M, N> &A, const Matrix<T, M, K> &B,
                    const Matrix<T, N, K> &X_1, const T &decay_rate,
                    const T &division_min) {

  Matrix<T, N, K> X = X_1;
  std::array<T, K> rho = {};
  std::array<std::size_t, K> rep_num = {};

  for (std::size_t i = 0; i < K; i++) {
    auto result = Base::Matrix::gmres_k_rect(A, B.get_row(i), X.get_row(i),
                                             decay_rate, division_min);
    X.set_row(i, std::get<0>(result));
    rho[i] = std::get<1>(result);
    rep_num[i] = std::get<2>(result);
  }

  return std::make_tuple(X, rho, rep_num);
}

template <typename T, std::size_t M, std::size_t N>
inline std::tuple<Matrix<T, N, M>, std::array<T, M>, std::array<std::size_t, M>>
gmres_k_rect_matrix(const Matrix<T, M, N> &A, const DiagMatrix<T, M> &B,
                    const Matrix<T, N, M> &X_1, const T &decay_rate,
                    const T &division_min) {

  Matrix<T, N, M> X = X_1;
  std::array<T, M> rho = {};
  std::array<std::size_t, M> rep_num = {};

  for (std::size_t i = 0; i < M; i++) {
    auto result = Base::Matrix::gmres_k_rect(A, B.get_row(i), X.get_row(i),
                                             decay_rate, division_min);
    X.set_row(i, std::get<0>(result));
    rho[i] = std::get<1>(result);
    rep_num[i] = std::get<2>(result);
  }

  return std::make_tuple(X, rho, rep_num);
}

/* GMRES K for matrix inverse */

/**
 * @brief GMRES K method for matrix inverse.
 *
 * This function computes the inverse of a square matrix A using the GMRES K
 * method, where A is a square matrix, and X_1 is an initial guess. It returns
 * the inverse matrix X.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The size of the square matrix (M x M).
 * @param In_A The input matrix A.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @return A tuple of (inverse matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M>
inline std::tuple<Matrix<T, M, M>, std::array<T, M>, std::array<std::size_t, M>>
gmres_k_matrix_inv(const Matrix<T, M, M> &In_A, const T &decay_rate,
                   const T &division_min, const Matrix<T, M, M> &X_1) {
  Matrix<T, M, M> X;
  std::array<T, M> rho = {};
  std::array<std::size_t, M> rep_num = {};
  GmresSolver<T, M> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  for (std::size_t i = 0; i < M; ++i) {
    Vector<T, M> unit;
    unit[i] = static_cast<T>(1);
    X.set_row(i, solver.solve(In_A, unit, X_1.get_row(i)));
    rho[i] = solver.get_residual_norm();
    rep_num[i] = solver.get_iterations();
  }
  return std::make_tuple(X, rho, rep_num);
}

/* Sparse GMRES K */
namespace InverseOperation {

/**
 * @brief Core function for GMRES K method for sparse matrices.
 *
 * This function implements the core logic of the GMRES K method for sparse
 * matrices, handling the case where the matrix size is larger than 1x1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam CSRIndices_A The type representing row indices of the sparse matrix.
 * @tparam CSRPointers_A The type representing row pointers of the sparse
 * matrix.
 * @param SA The compiled sparse matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param matrix_size The size of the matrix (M).
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
template <typename T, std::size_t M, typename CSRIndices_A,
          typename CSRPointers_A>
inline typename std::enable_if<(M > 1),
                               std::tuple<Vector<T, M>, T, std::size_t>>::type
sparse_gmres_k_core(
    const CompiledSparseMatrix<T, M, M, CSRIndices_A, CSRPointers_A> &SA,
    const Vector<T, M> &b, const Vector<T, M> &x_1, const T &decay_rate,
    const T &division_min, const std::size_t &matrix_size) {
  GmresSolver<T, M> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  Vector<T, M> solution = solver.solve(SA, b, x_1, matrix_size);
  return std::make_tuple(solution, solver.get_residual_norm(),
                         solver.get_iterations());
}

/**
 * @brief Core function for GMRES K method for sparse matrices with size 1x1.
 *
 * This function implements the core logic of the GMRES K method for sparse
 * matrices, handling the case where the matrix size is exactly 1x1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param SA The compiled sparse matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param matrix_size The size of the matrix (M).
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
template <typename T, std::size_t M, typename CSRIndices_A,
          typename CSRPointers_A>
inline typename std::enable_if<(M <= 1),
                               std::tuple<Vector<T, M>, T, std::size_t>>::type
sparse_gmres_k_core(
    const CompiledSparseMatrix<T, M, M, CSRIndices_A, CSRPointers_A> &SA,
    const Vector<T, M> &b, const Vector<T, M> &x_1, const T &decay_rate,
    const T &division_min, const std::size_t &matrix_size) {
  GmresSolver<T, M> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  Vector<T, M> solution = solver.solve(SA, b, x_1, matrix_size);
  return std::make_tuple(solution, solver.get_residual_norm(),
                         solver.get_iterations());
}

} // namespace InverseOperation

/**
 * @brief GMRES K method for sparse matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a sparse matrix, b is a vector, and x_1 is an initial guess.
 * It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param SA The compiled sparse matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
template <typename T, std::size_t M, typename CSRIndices_A,
          typename CSRPointers_A>
inline std::tuple<Vector<T, M>, T, std::size_t> sparse_gmres_k(
    const CompiledSparseMatrix<T, M, M, CSRIndices_A, CSRPointers_A> &SA,
    const Vector<T, M> &b, const Vector<T, M> &x_1, const T &decay_rate,
    const T &division_min) {

  return InverseOperation::sparse_gmres_k_core<T, M>(SA, b, x_1, decay_rate,
                                                     division_min, M);
}

/**
 * @brief GMRES K method for sparse matrices with partitioning.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a sparse matrix, b is a vector, and x_1 is an initial guess.
 * It supports partitioning of the matrix and returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param SA The compiled sparse matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param matrix_size The size of the matrix (M).
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
template <typename T, std::size_t M, typename CSRIndices_A,
          typename CSRPointers_A>
inline std::tuple<Vector<T, M>, T, std::size_t> sparse_gmres_k_partition(
    const CompiledSparseMatrix<T, M, M, CSRIndices_A, CSRPointers_A> &SA,
    const Vector<T, M> &b, const Vector<T, M> &x_1, const T &decay_rate,
    const T &division_min, const std::size_t &matrix_size) {

  return InverseOperation::sparse_gmres_k_core<T, M>(SA, b, x_1, decay_rate,
                                                     division_min, matrix_size);
}

/**
 * @brief GMRES K method for sparse matrices with multiple right-hand side
 * vectors.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a sparse matrix, B is a matrix with multiple right-hand side
 * vectors, and X_1 is an initial guess. It returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam K The number of columns of matrix B.
 * @param SA The compiled sparse matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M, std::size_t K, typename CSRIndices_A,
          typename CSRPointers_A>
inline std::tuple<Matrix<T, M, K>, std::array<T, K>, std::array<std::size_t, K>>
sparse_gmres_k_matrix(
    const CompiledSparseMatrix<T, M, M, CSRIndices_A, CSRPointers_A> &SA,
    const Matrix<T, M, K> &B, const Matrix<T, M, K> &X_1, const T &decay_rate,
    const T &division_min) {

  Matrix<T, M, K> X = X_1;
  std::array<T, K> rho = {};
  std::array<std::size_t, K> rep_num = {};

  for (std::size_t i = 0; i < K; i++) {
    auto result = Base::Matrix::sparse_gmres_k(SA, B.get_row(i), X.get_row(i),
                                               decay_rate, division_min);
    X.set_row(i, std::get<0>(result));
    rho[i] = std::get<1>(result);
    rep_num[i] = std::get<2>(result);
  }

  return std::make_tuple(X, rho, rep_num);
}

/**
 * @brief GMRES K method for sparse matrices with multiple right-hand side
 * vectors and partitioning.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a sparse matrix, B is a matrix with multiple right-hand side
 * vectors, and X_1 is an initial guess. It supports partitioning of the matrix
 * and returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam K The number of columns of matrix B.
 * @param SA The compiled sparse matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param matrix_size The size of the matrix (M).
 * @return A tuple of (solution matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M, std::size_t K, typename CSRIndices_A,
          typename CSRPointers_A>
inline std::tuple<Matrix<T, M, K>, std::array<T, K>, std::array<std::size_t, K>>
sparse_gmres_k_partition_matrix(
    const CompiledSparseMatrix<T, M, M, CSRIndices_A, CSRPointers_A> &SA,
    const Matrix<T, M, K> &B, const Matrix<T, M, K> &X_1, const T &decay_rate,
    const T &division_min, const std::size_t &matrix_size) {

  Matrix<T, M, K> X = X_1;
  std::array<T, K> rho = {};
  std::array<std::size_t, K> rep_num = {};

  std::size_t repeat_number;
  if (matrix_size < K) {
    repeat_number = matrix_size;
  } else {
    repeat_number = K;
  }

  for (std::size_t i = 0; i < repeat_number; i++) {
    auto result = Base::Matrix::sparse_gmres_k_partition(
        SA, B.get_row(i), X.get_row(i), decay_rate, division_min, matrix_size);
    X.set_row(i, std::get<0>(result));
    rho[i] = std::get<1>(result);
    rep_num[i] = std::get<2>(result);
  }

  return std::make_tuple(X, rho, rep_num);
}

/* Sparse GMRES K for rectangular matrix */

/**
 * @brief GMRES K method for rectangular matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a rectangular matrix (M x N), b is a vector, and x_1 is an
 * initial guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The number of columns in the matrix A.
 * @tparam N The number of rows in the matrix A.
 * @param In_SA The compiled sparse matrix A.
 * @param b The right-hand side vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
// Optional low-memory operator path; no explicit Gram matrix.
template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A>
inline std::tuple<Vector<T, N>, T, std::size_t> sparse_gmres_k_rect_matrix_free(
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &In_SA,
    const Vector<T, M> &b, const Vector<T, N> &x_1, const T &decay_rate,
    const T &division_min) {
  static_assert(M > N,
                "GMRES rectangular solve requires more rows than columns.");
  GmresOperation::NormalOperator<
      CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A>, T, M, N>
      operation = {In_SA};
  Vector<T, N> right_hand_side;
  operation.transpose_multiply(b, right_hand_side);
  GmresSolver<T, N> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  Vector<T, N> solution =
      solver.solve_operator(operation, right_hand_side, x_1);
  return std::make_tuple(solution,
                         GmresOperation::normal_residual(In_SA, b, solution),
                         solver.get_iterations());
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A>
inline std::tuple<Vector<T, N>, T, std::size_t> sparse_gmres_k_rect(
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &In_SA,
    const Vector<T, M> &b, const Vector<T, N> &x_1, const T &decay_rate,
    const T &division_min) {
  static_assert(M > N,
                "GMRES rectangular solve requires more rows than columns.");
  Matrix<T, N, N> gram;
  Vector<T, N> normal_rhs;
  GmresOperation::normal_equations(In_SA, b, gram, normal_rhs);
  GmresSolver<T, N> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  Vector<T, N> solution = solver.solve(gram, normal_rhs, x_1);
  return std::make_tuple(solution,
                         GmresOperation::normal_residual(In_SA, b, solution),
                         solver.get_iterations());
}

/**
 * @brief GMRES K method for rectangular matrices with multiple right-hand side
 * vectors.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a rectangular matrix (M x N), B is a matrix with multiple right-
 * hand side vectors, and X_1 is an initial guess. It returns the solution
 * matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The number of rows in the matrix A.
 * @tparam N The number of columns in the matrix A.
 * @tparam K The number of columns of matrix B.
 * @param In_SA The compiled sparse matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename CSRIndices_A, typename CSRPointers_A>
inline std::tuple<Matrix<T, N, K>, std::array<T, K>, std::array<std::size_t, K>>
sparse_gmres_k_rect_matrix(
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &In_SA,
    const Matrix<T, M, K> &B, const Matrix<T, N, K> &X_1, const T &decay_rate,
    const T &division_min) {

  Matrix<T, N, K> X = X_1;
  std::array<T, K> rho = {};
  std::array<std::size_t, K> rep_num = {};

  for (std::size_t i = 0; i < K; i++) {
    auto result = Base::Matrix::sparse_gmres_k_rect(
        In_SA, B.get_row(i), X.get_row(i), decay_rate, division_min);
    X.set_row(i, std::get<0>(result));
    rho[i] = std::get<1>(result);
    rep_num[i] = std::get<2>(result);
  }

  return std::make_tuple(X, rho, rep_num);
}

/**
 * @brief GMRES K method for rectangular matrices with multiple right-hand side
 * vectors and partitioning.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a rectangular matrix (M x N), B is a matrix with multiple right-
 * hand side vectors, and X_1 is an initial guess. It supports partitioning of
 * the matrix and returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The number of rows in the matrix A.
 * @tparam N The number of columns in the matrix A.
 * @tparam CSRIndices_A The type representing row indices of the sparse matrix.
 * @tparam CSRPointers_A The type representing row pointers of the sparse
 * matrix.
 * @param In_SA The compiled sparse matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A>
inline std::tuple<Matrix<T, N, M>, std::array<T, M>, std::array<std::size_t, M>>
sparse_gmres_k_rect_matrix(
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &In_SA,
    const DiagMatrix<T, M> &B, const Matrix<T, N, M> &X_1, const T &decay_rate,
    const T &division_min) {

  Matrix<T, N, M> X = X_1;
  std::array<T, M> rho = {};
  std::array<std::size_t, M> rep_num = {};

  for (std::size_t i = 0; i < M; i++) {
    auto result = Base::Matrix::sparse_gmres_k_rect(
        In_SA, B.get_row(i), X.get_row(i), decay_rate, division_min);
    X.set_row(i, std::get<0>(result));
    rho[i] = std::get<1>(result);
    rep_num[i] = std::get<2>(result);
  }

  return std::make_tuple(X, rho, rep_num);
}

/* Sparse GMRES K for matrix inverse */

/**
 * @brief GMRES K method for sparse matrix inverse.
 *
 * This function computes the inverse of a sparse matrix using the GMRES K
 * method, where A is a sparse matrix, and X_1 is an initial guess for the
 * solution matrix.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam CSRIndices_A The type representing row indices of the sparse matrix.
 * @tparam CSRPointers_A The type representing row pointers of the sparse
 * matrix.
 * @param In_A The compiled sparse matrix A.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @return A tuple of (inverse matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M, typename CSRIndices_A,
          typename CSRPointers_A>
inline std::tuple<Matrix<T, M, M>, std::array<T, M>, std::array<std::size_t, M>>
sparse_gmres_k_matrix_inv(
    const CompiledSparseMatrix<T, M, M, CSRIndices_A, CSRPointers_A> &In_A,
    const T &decay_rate, const T &division_min, const Matrix<T, M, M> &X_1) {
  Matrix<T, M, M> X;
  std::array<T, M> rho = {};
  std::array<std::size_t, M> rep_num = {};
  GmresSolver<T, M> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  for (std::size_t i = 0; i < M; ++i) {
    Vector<T, M> unit;
    unit[i] = static_cast<T>(1);
    X.set_row(i, solver.solve(In_A, unit, X_1.get_row(i)));
    rho[i] = solver.get_residual_norm();
    rep_num[i] = solver.get_iterations();
  }
  return std::make_tuple(X, rho, rep_num);
}

/* Complex GMRES K */

/**
 * @brief GMRES K method for complex matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a complex matrix, b is a complex vector, and x_1 is an initial
 * guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The complex matrix A.
 * @param b The right-hand side complex vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
template <typename T, std::size_t M>
inline typename std::enable_if<
    (M > 1), std::tuple<Vector<Complex<T>, M>, T, std::size_t>>::type
complex_gmres_k(const Matrix<Complex<T>, M, M> &A,
                const Vector<Complex<T>, M> &b,
                const Vector<Complex<T>, M> &x_1, const T &decay_rate,
                const T &division_min) {
  GmresSolver<Complex<T>, M> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  Vector<Complex<T>, M> solution = solver.solve(A, b, x_1, M);
  return std::make_tuple(solution, solver.get_residual_norm(),
                         solver.get_iterations());
}

/**
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a complex matrix of size 1x1, b is a complex vector, and x_1 is
 * an initial guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The complex matrix A.
 * @param b The right-hand side complex vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
template <typename T, std::size_t M>
inline typename std::enable_if<
    (M <= 1), std::tuple<Vector<Complex<T>, M>, T, std::size_t>>::type
complex_gmres_k(const Matrix<Complex<T>, M, M> &A,
                const Vector<Complex<T>, M> &b,
                const Vector<Complex<T>, M> &x_1, const T &decay_rate,
                const T &division_min) {
  GmresSolver<Complex<T>, M> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  Vector<Complex<T>, M> solution = solver.solve(A, b, x_1, M);
  return std::make_tuple(solution, solver.get_residual_norm(),
                         solver.get_iterations());
}

/**
 * @brief GMRES K method for complex matrices with multiple right-hand side
 * vectors.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a complex matrix, B is a matrix with multiple right-hand side
 * vectors, and X_1 is an initial guess. It returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam K The number of columns of matrix B.
 * @param A The complex matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M, std::size_t K>
inline std::tuple<Matrix<Complex<T>, M, K>, std::array<T, K>,
                  std::array<std::size_t, K>>
complex_gmres_k_matrix(const Matrix<Complex<T>, M, M> &A,
                       const Matrix<Complex<T>, M, K> &B,
                       const Matrix<Complex<T>, M, K> &X_1, const T &decay_rate,
                       const T &division_min) {

  Matrix<Complex<T>, M, K> X = X_1;
  std::array<T, K> rho = {};
  std::array<std::size_t, K> rep_num = {};

  for (std::size_t i = 0; i < K; i++) {
    auto result = Base::Matrix::complex_gmres_k(A, B.get_row(i), X.get_row(i),
                                                decay_rate, division_min);
    X.set_row(i, std::get<0>(result));
    rho[i] = std::get<1>(result);
    rep_num[i] = std::get<2>(result);
  }

  return std::make_tuple(X, rho, rep_num);
}

/**
 * @brief GMRES K method for complex matrices with multiple right-hand side
 * vectors and partitioning.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a complex matrix, B is a matrix with multiple right-hand side
 * vectors, and X_1 is an initial guess. It supports partitioning of the matrix
 * and returns the solution matrix X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param A The complex matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M>
inline std::tuple<Matrix<Complex<T>, M, M>, std::array<T, M>,
                  std::array<std::size_t, M>>
complex_gmres_k_matrix(const Matrix<Complex<T>, M, M> &A,
                       const DiagMatrix<Complex<T>, M> &B,
                       const Matrix<Complex<T>, M, M> &X_1, const T &decay_rate,
                       const T &division_min) {

  Matrix<Complex<T>, M, M> X = X_1;
  std::array<T, M> rho = {};
  std::array<std::size_t, M> rep_num = {};

  for (std::size_t i = 0; i < M; i++) {
    auto result = Base::Matrix::complex_gmres_k(A, B.get_row(i), X.get_row(i),
                                                decay_rate, division_min);
    X.set_row(i, std::get<0>(result));
    rho[i] = std::get<1>(result);
    rep_num[i] = std::get<2>(result);
  }

  return std::make_tuple(X, rho, rep_num);
}

/**
 * @brief GMRES K method for complex sparse matrices.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a complex sparse matrix, b is a complex vector, and x_1 is an
 * initial guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param SA The compiled sparse matrix A.
 * @param b The right-hand side complex vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
template <typename T, std::size_t M, typename CSRIndices_A,
          typename CSRPointers_A>
inline typename std::enable_if<
    (M > 1), std::tuple<Vector<Complex<T>, M>, T, std::size_t>>::type
complex_sparse_gmres_k(
    const CompiledSparseMatrix<Complex<T>, M, M, CSRIndices_A, CSRPointers_A>
        &SA,
    const Vector<Complex<T>, M> &b, const Vector<Complex<T>, M> &x_1,
    const T &decay_rate, const T &division_min) {
  GmresSolver<Complex<T>, M> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  Vector<Complex<T>, M> solution = solver.solve(SA, b, x_1, M);
  return std::make_tuple(solution, solver.get_residual_norm(),
                         solver.get_iterations());
}

/**
 * @brief GMRES K method for complex sparse matrices with size 1x1.
 *
 * This function solves the linear system Ax = b using the GMRES K method,
 * where A is a complex sparse matrix of size 1x1, b is a complex vector, and
 * x_1 is an initial guess. It returns the solution vector x.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param SA The compiled sparse matrix A.
 * @param b The right-hand side complex vector b.
 * @param x_1 The initial guess for the solution vector x.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution vector x, residual norm rho, number of
 * iterations rep_num).
 */
template <typename T, std::size_t M, typename CSRIndices_A,
          typename CSRPointers_A>
inline typename std::enable_if<
    (M <= 1), std::tuple<Vector<Complex<T>, M>, T, std::size_t>>::type
complex_sparse_gmres_k(
    const CompiledSparseMatrix<Complex<T>, M, M, CSRIndices_A, CSRPointers_A>
        &SA,
    const Vector<Complex<T>, M> &b, const Vector<Complex<T>, M> &x_1,
    const T &decay_rate, const T &division_min) {
  GmresSolver<Complex<T>, M> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  Vector<Complex<T>, M> solution = solver.solve(SA, b, x_1, M);
  return std::make_tuple(solution, solver.get_residual_norm(),
                         solver.get_iterations());
}

/**
 * @brief GMRES K method for complex sparse matrices with multiple right-hand
 * side vectors.
 *
 * This function solves the linear system Ax = B using the GMRES K method,
 * where A is a complex sparse matrix, B is a matrix with multiple right-hand
 * side vectors, and X_1 is an initial guess. It returns the solution matrix
 * X_1.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @tparam K The number of columns of matrix B.
 * @param In_A The compiled sparse matrix A.
 * @param B The right-hand side matrix B.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @return A tuple of (solution matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M, typename CSRIndices_A,
          typename CSRPointers_A>
inline std::tuple<Matrix<Complex<T>, M, M>, std::array<T, M>,
                  std::array<std::size_t, M>>
complex_sparse_gmres_k_matrix(
    const CompiledSparseMatrix<Complex<T>, M, M, CSRIndices_A, CSRPointers_A>
        &In_A,
    const T &decay_rate, const T &division_min,
    const Matrix<Complex<T>, M, M> &X_1) {
  Matrix<Complex<T>, M, M> X;
  std::array<T, M> rho = {};
  std::array<std::size_t, M> rep_num = {};
  GmresSolver<Complex<T>, M> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  for (std::size_t i = 0; i < M; ++i) {
    Vector<Complex<T>, M> unit;
    unit[i] = static_cast<T>(1);
    X.set_row(i, solver.solve(In_A, unit, X_1.get_row(i)));
    rho[i] = solver.get_residual_norm();
    rep_num[i] = solver.get_iterations();
  }
  return std::make_tuple(X, rho, rep_num);
}

/**
 * @brief GMRES K method for complex matrix inverse.
 *
 * This function computes the inverse of a complex matrix using the GMRES K
 * method, where A is a complex matrix, and X_1 is an initial guess for the
 * solution matrix.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param In_A The complex matrix A.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @return A tuple of (inverse matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M>
inline std::tuple<Matrix<Complex<T>, M, M>, std::array<T, M>,
                  std::array<std::size_t, M>>
complex_gmres_k_matrix_inv(const Matrix<Complex<T>, M, M> &In_A,
                           const T &decay_rate, const T &division_min,
                           const Matrix<Complex<T>, M, M> &X_1) {
  Matrix<Complex<T>, M, M> X;
  std::array<T, M> rho = {};
  std::array<std::size_t, M> rep_num = {};
  GmresSolver<Complex<T>, M> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  for (std::size_t i = 0; i < M; ++i) {
    Vector<Complex<T>, M> unit;
    unit[i] = static_cast<T>(1);
    X.set_row(i, solver.solve(In_A, unit, X_1.get_row(i)));
    rho[i] = solver.get_residual_norm();
    rep_num[i] = solver.get_iterations();
  }
  return std::make_tuple(X, rho, rep_num);
}

/**
 * @brief GMRES K method for complex sparse matrix inverse.
 *
 * This function computes the inverse of a complex sparse matrix using the
 * GMRES K method, where A is a complex sparse matrix, and X_1 is an initial
 * guess for the solution matrix.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @tparam M The size of the square matrix (M x M).
 * @param In_A The compiled sparse matrix A.
 * @param decay_rate The convergence criterion for the GMRES method.
 * @param division_min A small value to avoid division by zero.
 * @param X_1 The initial guess for the solution matrix X_1.
 * @return A tuple of (inverse matrix X, residual norms rho, iteration counts
 * rep_num).
 */
template <typename T, std::size_t M, typename CSRIndices_A,
          typename CSRPointers_A>
inline std::tuple<Matrix<Complex<T>, M, M>, std::array<T, M>,
                  std::array<std::size_t, M>>
complex_sparse_gmres_k_matrix_inv(
    const CompiledSparseMatrix<Complex<T>, M, M, CSRIndices_A, CSRPointers_A>
        &In_A,
    const T &decay_rate, const T &division_min,
    const Matrix<Complex<T>, M, M> &X_1) {
  Matrix<Complex<T>, M, M> X;
  std::array<T, M> rho = {};
  std::array<std::size_t, M> rep_num = {};
  GmresSolver<Complex<T>, M> solver;
  solver.division_min = division_min;
  solver.relative_tolerance = decay_rate;
  for (std::size_t i = 0; i < M; ++i) {
    Vector<Complex<T>, M> unit;
    unit[i] = static_cast<T>(1);
    X.set_row(i, solver.solve(In_A, unit, X_1.get_row(i)));
    rho[i] = solver.get_residual_norm();
    rep_num[i] = solver.get_iterations();
  }
  return std::make_tuple(X, rho, rep_num);
}

/* Diag Matrix inverse */

/**
 * @brief Inverse of a diagonal matrix.
 *
 * This function computes the inverse of a diagonal matrix, where each element
 * is divided by a small value to avoid division by zero.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The size of the diagonal matrix (M x M).
 * @param input The input diagonal matrix to be inverted.
 * @param division_min A small value to avoid division by zero.
 * @return The inverted diagonal matrix.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> inverse_diag_matrix(const DiagMatrix<T, M> &input,
                                            const T &division_min) {
  DiagMatrix<T, M> result;

  for (std::size_t i = 0; i < M; i++) {
    result[i] = static_cast<T>(1) /
                Base::Utility::avoid_zero_divide(input.data[i], division_min);
  }

  return result;
}

/**
 * @brief Inverse of a complex diagonal matrix.
 *
 * This function computes the inverse of a complex diagonal matrix, where each
 * element is divided by a small value to avoid division by zero.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The size of the diagonal matrix (M x M).
 * @param input The input complex diagonal matrix to be inverted.
 * @param division_min A small value to avoid division by zero.
 * @return The inverted complex diagonal matrix.
 */
template <typename T, std::size_t M>
inline DiagMatrix<Complex<T>, M>
inverse_complex_diag_matrix(const DiagMatrix<Complex<T>, M> &input,
                            const T &division_min) {
  DiagMatrix<Complex<T>, M> result;

  for (std::size_t i = 0; i < M; i++) {
    result[i] = Base::Matrix::complex_divide(static_cast<T>(1), input.data[i],
                                             division_min);
  }

  return result;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_INVERSE_HPP_
