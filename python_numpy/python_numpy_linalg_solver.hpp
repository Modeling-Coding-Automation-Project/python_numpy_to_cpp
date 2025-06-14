/**
 * @file python_numpy_linalg_solver.hpp
 * @brief Linear algebra solver utilities for Python-like Numpy matrix
 * operations in C++.
 *
 * This header provides a set of template classes and factory functions to
 * perform linear algebra operations such as solving linear systems, matrix
 * inversion, partitioned solving, and least-squares solutions. The solvers
 * support dense, diagonal, and sparse matrix types, and are designed to mimic
 * the behavior of Python's Numpy linalg module, but in a statically-typed,
 * template-based C++ environment.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __PYTHON_NUMPY_LINALG_SOLVER_HPP__
#define __PYTHON_NUMPY_LINALG_SOLVER_HPP__

#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace PythonNumpy {

const double DEFAULT_DIVISION_MIN_LINALG_SOLVER = 1.0e-23;

namespace InverseOperation {

template <typename T, typename Complex_T, std::size_t M, std::size_t K,
          bool IsComplex>
struct InverseDense {};

template <typename T, typename Complex_T, std::size_t M, std::size_t K>
struct InverseDense<T, Complex_T, M, K, true> {
  /**
   * @brief Computes the inverse of a complex matrix using a custom GMRES-based
   * solver.
   *
   * This static function applies a GMRES-based matrix inversion algorithm to
   * the input matrix `A`. It utilizes decay rate and division minimum
   * parameters to control the solver's behavior. The function updates the
   * provided `X_1` matrix with the computed result and returns it as a new
   * matrix object.
   *
   * @tparam DefDense   The matrix storage type.
   * @tparam Complex_T  The complex number type.
   * @tparam M          The number of rows and columns in the square matrix.
   * @tparam T          The floating-point type for decay and division
   * parameters.
   * @tparam K          The size of the `rho` and `rep_num` arrays.
   *
   * @param A           The input square matrix to invert (of size MxM).
   * @param decay_rate  The decay rate parameter for the GMRES solver.
   * @param division_min The minimum division threshold for numerical stability.
   * @param rho         Array of parameters influencing the solver (size K).
   * @param rep_num     Array of repetition numbers for the solver (size K).
   * @param X_1         Matrix to store the intermediate and final results (size
   * MxK).
   *
   * @return Matrix<DefDense, Complex_T, M, M> The computed inverse matrix.
   */
  static auto compute(const Matrix<DefDense, Complex_T, M, M> &A,
                      const T &decay_rate, const T &division_min,
                      std::array<T, K> &rho,
                      std::array<std::size_t, K> &rep_num,
                      Base::Matrix::Matrix<Complex_T, M, K> &X_1)
      -> Matrix<DefDense, Complex_T, M, M> {

    X_1 = Base::Matrix::complex_gmres_k_matrix_inv(
        A.matrix, decay_rate, division_min, rho, rep_num, X_1);

    return Matrix<DefDense, Complex_T, M, M>(X_1);
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t K>
struct InverseDense<T, Complex_T, M, K, false> {
  /**
   * @brief Computes the inverse of a real matrix using a custom GMRES-based
   * solver.
   *
   * This static function applies a GMRES-based matrix inversion algorithm to
   * the input matrix `A`. It utilizes decay rate and division minimum
   * parameters to control the solver's behavior. The function updates the
   * provided `X_1` matrix with the computed result and returns it as a new
   * matrix object.
   *
   * @tparam DefDense   The matrix storage type.
   * @tparam T          The floating-point type for decay and division
   * parameters.
   * @tparam M          The number of rows and columns in the square matrix.
   * @tparam K          The size of the `rho` and `rep_num` arrays.
   *
   * @param A           The input square matrix to invert (of size MxM).
   * @param decay_rate  The decay rate parameter for the GMRES solver.
   * @param division_min The minimum division threshold for numerical stability.
   * @param rho         Array of parameters influencing the solver (size K).
   * @param rep_num     Array of repetition numbers for the solver (size K).
   * @param X_1         Matrix to store the intermediate and final results (size
   * MxK).
   *
   * @return Matrix<DefDense, T, M, M> The computed inverse matrix.
   */
  static auto compute(const Matrix<DefDense, T, M, M> &A, const T &decay_rate,
                      const T &division_min, std::array<T, K> &rho,
                      std::array<std::size_t, K> &rep_num,
                      Base::Matrix::Matrix<T, M, K> &X_1)
      -> Matrix<DefDense, T, M, M> {

    X_1 = Base::Matrix::gmres_k_matrix_inv(A.matrix, decay_rate, division_min,
                                           rho, rep_num, X_1);

    return Matrix<DefDense, T, M, M>(X_1);
  }
};

template <typename T, typename Complex_T, std::size_t M, bool IsComplex>
struct InverseDiag {};

template <typename T, typename Complex_T, std::size_t M>
struct InverseDiag<T, Complex_T, M, true> {
  /**
   * @brief Computes the inverse of a complex diagonal matrix.
   *
   * This static function computes the inverse of a diagonal matrix `A` using
   * a custom method that handles complex numbers. It updates the provided
   * `X_1` matrix with the computed result and returns it as a new matrix
   * object.
   *
   * @tparam DefDiag    The matrix storage type for diagonal matrices.
   * @tparam Complex_T  The complex number type.
   * @tparam M          The size of the square diagonal matrix.
   * @tparam T          The floating-point type for division parameters.
   *
   * @param A           The input diagonal matrix to invert (of size MxM).
   * @param division_min The minimum division threshold for numerical stability.
   * @param X_1         Matrix to store the intermediate and final results (size
   * MxM).
   *
   * @return Matrix<DefDiag, Complex_T, M> The computed inverse diagonal matrix.
   */
  static auto compute(const Matrix<DefDiag, Complex_T, M> &A,
                      const T &division_min,
                      Base::Matrix::DiagMatrix<Complex_T, M> &X_1)
      -> Matrix<DefDiag, Complex_T, M> {

    X_1 = Base::Matrix::inverse_complex_diag_matrix(A.matrix, division_min);

    return Matrix<DefDiag, Complex_T, M>(X_1);
  }
};

template <typename T, typename Complex_T, std::size_t M>
struct InverseDiag<T, Complex_T, M, false> {
  /**
   * @brief Computes the inverse of a real diagonal matrix.
   *
   * This static function computes the inverse of a diagonal matrix `A` using
   * a custom method that handles real numbers. It updates the provided `X_1`
   * matrix with the computed result and returns it as a new matrix object.
   *
   * @tparam DefDiag    The matrix storage type for diagonal matrices.
   * @tparam T          The floating-point type for division parameters.
   * @tparam M          The size of the square diagonal matrix.
   *
   * @param A           The input diagonal matrix to invert (of size MxM).
   * @param division_min The minimum division threshold for numerical stability.
   * @param X_1         Matrix to store the intermediate and final results (size
   * MxM).
   *
   * @return Matrix<DefDiag, T, M> The computed inverse diagonal matrix.
   */
  static auto compute(const Matrix<DefDiag, T, M> &A, const T &division_min,
                      Base::Matrix::DiagMatrix<T, M> &X_1)
      -> Matrix<DefDiag, T, M> {

    X_1 = Base::Matrix::inverse_diag_matrix(A.matrix, division_min);

    return Matrix<DefDiag, T, M>(X_1);
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t K,
          typename SparseAvailable, bool IsComplex>
struct InverseSparse {};

template <typename T, typename Complex_T, std::size_t M, std::size_t K,
          typename SparseAvailable>
struct InverseSparse<T, Complex_T, M, K, SparseAvailable, true> {
  /**
   * @brief Computes the inverse of a complex sparse matrix using a custom GMRES
   * solver.
   *
   * This static function applies a GMRES-based matrix inversion algorithm to
   * the input sparse matrix `A`. It utilizes decay rate and division minimum
   * parameters to control the solver's behavior. The function updates the
   * provided `X_1` matrix with the computed result and returns it as a new
   * matrix object.
   *
   * @tparam DefSparse  The matrix storage type for sparse matrices.
   * @tparam Complex_T  The complex number type.
   * @tparam M          The number of rows and columns in the square matrix.
   * @tparam T          The floating-point type for decay and division
   * parameters.
   * @tparam K          The size of the `rho` and `rep_num` arrays.
   *
   * @param A           The input square sparse matrix to invert (of size MxM).
   * @param decay_rate  The decay rate parameter for the GMRES solver.
   * @param division_min The minimum division threshold for numerical stability.
   * @param rho         Array of parameters influencing the solver (size K).
   * @param rep_num     Array of repetition numbers for the solver (size K).
   * @param X_1         Matrix to store the intermediate and final results (size
   * MxK).
   *
   * @return Matrix<DefDense, Complex_T, M, M> The computed inverse matrix.
   */
  static auto
  compute(const Matrix<DefSparse, Complex_T, M, M, SparseAvailable> &A,
          const T &decay_rate, const T &division_min, std::array<T, K> &rho,
          std::array<std::size_t, K> &rep_num,
          Base::Matrix::Matrix<Complex_T, M, K> &X_1)
      -> Matrix<DefDense, Complex_T, M, M> {

    X_1 = Base::Matrix::complex_sparse_gmres_k_matrix_inv(
        A.matrix, decay_rate, division_min, rho, rep_num, X_1);

    return Matrix<DefDense, Complex_T, M, M>(X_1);
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t K,
          typename SparseAvailable>
struct InverseSparse<T, Complex_T, M, K, SparseAvailable, false> {
  /**
   * @brief Computes the inverse of a real sparse matrix using a custom GMRES
   * solver.
   *
   * This static function applies a GMRES-based matrix inversion algorithm to
   * the input sparse matrix `A`. It utilizes decay rate and division minimum
   * parameters to control the solver's behavior. The function updates the
   * provided `X_1` matrix with the computed result and returns it as a new
   * matrix object.
   *
   * @tparam DefSparse  The matrix storage type for sparse matrices.
   * @tparam T          The floating-point type for decay and division
   * parameters.
   * @tparam M          The number of rows and columns in the square matrix.
   * @tparam K          The size of the `rho` and `rep_num` arrays.
   *
   * @param A           The input square sparse matrix to invert (of size MxM).
   * @param decay_rate  The decay rate parameter for the GMRES solver.
   * @param division_min The minimum division threshold for numerical stability.
   * @param rho         Array of parameters influencing the solver (size K).
   * @param rep_num     Array of repetition numbers for the solver (size K).
   * @param X_1         Matrix to store the intermediate and final results (size
   * MxK).
   *
   * @return Matrix<DefDense, Complex_T, M, M> The computed inverse matrix.
   */
  static auto compute(const Matrix<DefSparse, T, M, M, SparseAvailable> &A,
                      const T &decay_rate, const T &division_min,
                      std::array<T, K> &rho,
                      std::array<std::size_t, K> &rep_num,
                      Base::Matrix::Matrix<T, M, K> &X_1)
      -> Matrix<DefDense, Complex_T, M, M> {

    X_1 = Base::Matrix::sparse_gmres_k_matrix_inv(
        A.matrix, decay_rate, division_min, rho, rep_num, X_1);

    return Matrix<DefDense, Complex_T, M, M>(X_1);
  }
};

} // namespace InverseOperation

/* Linalg Solver */

/**
 * @brief Linalg solver for various matrix types.
 *
 * This class provides methods to solve linear systems, compute inverses, and
 * perform other linear algebra operations on dense, diagonal, and sparse
 * matrices. It supports both real and complex number types (float or double).
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam M The number of rows and columns in the square matrix.
 * @tparam K The number of columns in the right-hand side matrix.
 * @tparam SparseAvailable_A Indicates if the first matrix is sparse.
 * @tparam SparseAvailable_B Indicates if the second matrix is sparse.
 */
template <typename T, std::size_t M, std::size_t K, typename SparseAvailable_A,
          typename SparseAvailable_B>
class LinalgSolver {
public:
  /* Type */
  using Value_Type = typename UnderlyingType<T>::Type;
  static_assert(std::is_same<Value_Type, double>::value ||
                    std::is_same<Value_Type, float>::value,
                "Value data type must be float or double.");

public:
  /* Constructor */
  LinalgSolver()
      : X_1(), decay_rate(static_cast<Value_Type>(0)),
        division_min(
            static_cast<Value_Type>(DEFAULT_DIVISION_MIN_LINALG_SOLVER)),
        rho({}), rep_num({}) {}

  /* Copy Constructor */
  LinalgSolver(
      const LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> &other)
      : X_1(other.X_1), decay_rate(other.decay_rate),
        division_min(other.division_min), rho(other.rho),
        rep_num(other.rep_num) {}

  LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> &
  operator=(const LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B>
                &other) {
    if (this != &other) {
      this->X_1 = other.X_1;
      this->decay_rate = other.decay_rate;
      this->division_min = other.division_min;
      this->rho = other.rho;
      this->rep_num = other.rep_num;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolver(LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B>
                   &&other) noexcept
      : X_1(std::move(other.X_1)), decay_rate(std::move(other.decay_rate)),
        division_min(std::move(other.division_min)), rho(std::move(other.rho)),
        rep_num(std::move(other.rep_num)) {}

  LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> &
  operator=(LinalgSolver<T, M, K, SparseAvailable_A, SparseAvailable_B>
                &&other) noexcept {
    if (this != &other) {
      this->X_1 = std::move(other.X_1);
      this->decay_rate = std::move(other.decay_rate);
      this->division_min = std::move(other.division_min);
      this->rho = std::move(other.rho);
      this->rep_num = std::move(other.rep_num);
    }
    return *this;
  }

public:
  /* Solve function */

  /**
   * @brief Solves the linear system Ax = B using GMRES method.
   *
   * This function applies the GMRES method to solve the linear system
   * represented by the matrix A and the right-hand side matrix B. It updates
   * the internal state of the solver with the computed solution.
   *
   * @param A The coefficient matrix (of size MxM).
   * @param B The right-hand side matrix (of size MxK).
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDense, T, M, M> &A,
                    const Matrix<DefDense, T, M, K> &B)
      -> Matrix<DefDense, T, M, K> {

    Base::Matrix::gmres_k_matrix(A.matrix, B.matrix, this->X_1,
                                 this->decay_rate, this->division_min,
                                 this->rho, this->rep_num);

    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B using GMRES method for diagonal
   * matrices.
   *
   * This function applies the GMRES method to solve the linear system
   * represented by the diagonal matrix A and the right-hand side matrix B.
   * It updates the internal state of the solver with the computed solution.
   *
   * @param A The coefficient diagonal matrix (of size MxM).
   * @param B The right-hand side diagonal matrix (of size M).
   * @return Matrix<DefDense, T, M, M> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDense, T, M, M> &A,
                    const Matrix<DefDiag, T, M> &B)
      -> Matrix<DefDense, T, M, M> {

    Base::Matrix::gmres_k_matrix(A.matrix, B.matrix, this->X_1,
                                 this->decay_rate, this->division_min,
                                 this->rho, this->rep_num);

    return Matrix<DefDense, T, M, M>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B using GMRES method for sparse
   * matrices.
   *
   * This function applies the GMRES method to solve the linear system
   * represented by the sparse matrix A and the right-hand side matrix B.
   * It updates the internal state of the solver with the computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxM).
   * @param B The right-hand side matrix (of size MxK).
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDense, T, M, M> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
      -> Matrix<DefDense, T, M, K> {

    Base::Matrix::Matrix<T, M, K> B_dense_matrix =
        Base::Matrix::output_dense_matrix(B.matrix);

    Base::Matrix::gmres_k_matrix(A.matrix, B_dense_matrix, this->X_1,
                                 this->decay_rate, this->division_min,
                                 this->rho, this->rep_num);

    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B using GMRES method for diagonal
   * matrices.
   *
   * This function applies the GMRES method to solve the linear system
   * represented by the diagonal matrix A and the right-hand side matrix B.
   * It updates the internal state of the solver with the computed solution.
   *
   * @param A The coefficient diagonal matrix (of size MxM).
   * @param B The right-hand side dense matrix (of size MxK).
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDiag, T, M> &A,
                    const Matrix<DefDense, T, M, K> &B)
      -> Matrix<DefDense, T, M, K> {

    this->X_1 = Base::Matrix::diag_inv_multiply_dense(A.matrix, B.matrix,
                                                      this->division_min);

    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B for diagonal matrices.
   *
   * This function computes the solution of the linear system represented by
   * the diagonal matrix A and the diagonal matrix B. It updates the internal
   * state of the solver with the computed solution.
   *
   * @param A The coefficient diagonal matrix (of size MxM).
   * @param B The right-hand side diagonal matrix (of size M).
   * @return Matrix<DefDiag, T, M> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDiag, T, M> &A,
                    const Matrix<DefDiag, T, M> &B) -> Matrix<DefDiag, T, M> {

    Base::Matrix::DiagMatrix<T, M> result =
        Base::Matrix::diag_divide_diag(B.matrix, A.matrix, this->division_min);

    this->X_1 = Base::Matrix::output_dense_matrix(result);

    return Matrix<DefDiag, T, M>(std::move(result));
  }

  /**
   * @brief Solves the linear system Ax = B for sparse matrices.
   *
   * This function computes the solution of the linear system represented by
   * the sparse matrix A and the dense matrix B. It updates the internal state
   * of the solver with the computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxM).
   * @param B The right-hand side dense matrix (of size MxK).
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDiag, T, M> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
      -> Matrix<DefDense, T, M, K> {

    using RowIndices_B = RowIndicesFromSparseAvailable<SparseAvailable_B>;
    using RowPointers_B = RowPointersFromSparseAvailable<SparseAvailable_B>;

    Base::Matrix::CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B>
        result = Base::Matrix::diag_inv_multiply_sparse(A.matrix, B.matrix,
                                                        this->division_min);

    this->X_1 = Base::Matrix::output_dense_matrix(result);

    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B for sparse matrices.
   *
   * This function computes the solution of the linear system represented by
   * the sparse matrix A and the sparse matrix B. It updates the internal state
   * of the solver with the computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxM).
   * @param B The right-hand side sparse matrix (of size MxK).
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                    const Matrix<DefDense, T, M, K> &B)
      -> Matrix<DefDense, T, M, K> {

    Base::Matrix::sparse_gmres_k_matrix(A.matrix, B.matrix, this->X_1,
                                        this->decay_rate, this->division_min,
                                        this->rho, this->rep_num);

    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B for sparse matrices.
   *
   * This function computes the solution of the linear system represented by
   * the sparse matrix A and the diagonal matrix B. It updates the internal
   * state of the solver with the computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxM).
   * @param B The right-hand side diagonal matrix (of size M).
   * @return Matrix<DefDense, T, M, M> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                    const Matrix<DefDiag, T, M> &B)
      -> Matrix<DefDense, T, M, M> {

    Base::Matrix::Matrix<T, M, M> B_dense_matrix =
        output_dense_matrix(B.matrix);

    Base::Matrix::sparse_gmres_k_matrix(A.matrix, B_dense_matrix, this->X_1,
                                        this->decay_rate, this->division_min,
                                        this->rho, this->rep_num);

    return Matrix<DefDense, T, M, M>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B for sparse matrices.
   *
   * This function computes the solution of the linear system represented by
   * the sparse matrix A and the sparse matrix B. It updates the internal state
   * of the solver with the computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxM).
   * @param B The right-hand side sparse matrix (of size MxK).
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
      -> Matrix<DefDense, T, M, K> {

    Base::Matrix::Matrix<T, M, K> B_dense_matrix =
        Base::Matrix::output_dense_matrix(B.matrix);

    Base::Matrix::sparse_gmres_k_matrix(A.matrix, B_dense_matrix, this->X_1,
                                        this->decay_rate, this->division_min,
                                        this->rho, this->rep_num);

    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /* Inv function */

  /**
   * @brief Computes the inverse of a dense matrix.
   *
   * This function computes the inverse of a dense square matrix A using a
   * custom GMRES-based solver. It updates the internal state with the computed
   * inverse and returns it as a new matrix object.
   *
   * @param A The input square matrix to invert (of size MxM).
   * @return Matrix<DefDense, T, M, M> The computed inverse matrix.
   */
  inline auto inv(const Matrix<DefDense, T, M, M> &A)
      -> Matrix<DefDense, T, M, M> {

    return InverseOperation::InverseDense<
        Value_Type, T, M, K, IS_COMPLEX>::compute(A, this->decay_rate,
                                                  this->division_min, this->rho,
                                                  this->rep_num, this->X_1);
  }

  /**
   * @brief Computes the inverse of a sparse matrix.
   *
   * This function computes the inverse of a sparse square matrix A using a
   * custom GMRES-based solver. It updates the internal state with the computed
   * inverse and returns it as a new matrix object.
   *
   * @param A The input square sparse matrix to invert (of size MxM).
   * @return Matrix<DefDense, T, M, M> The computed inverse matrix.
   */
  inline auto inv(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A)
      -> Matrix<DefDense, T, M, M> {

    return InverseOperation::InverseSparse<
        Value_Type, T, M, K, SparseAvailable_A,
        IS_COMPLEX>::compute(A, this->decay_rate, this->division_min, this->rho,
                             this->rep_num, X_1);
  }

  /**
   * @brief Computes the inverse of a diagonal matrix.
   *
   * This function computes the inverse of a diagonal matrix A using a custom
   * method that handles both real and complex numbers. It updates the internal
   * state with the computed inverse and returns it as a new matrix object.
   *
   * @param A The input diagonal matrix to invert (of size MxM).
   * @return Matrix<DefDiag, T, M> The computed inverse diagonal matrix.
   */
  inline auto get_answer(void) -> Matrix<DefDense, T, M, K> {
    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /**
   * @brief Sets the decay rate for the GMRES solver.
   *
   * This function sets the decay rate parameter used in the GMRES solver.
   *
   * @param decay_rate_in The decay rate value to set.
   */
  inline void set_decay_rate(const Value_Type &decay_rate_in) {
    this->decay_rate = decay_rate_in;
  }

  /**
   * @brief Sets the minimum division threshold for numerical stability.
   *
   * This function sets the minimum division threshold used in the solver to
   * prevent numerical instability during matrix inversion.
   *
   * @param division_min_in The minimum division value to set.
   */
  inline void set_division_min(const Value_Type &division_min_in) {
    this->division_min = division_min_in;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = K;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;

public:
  /* Variable */
  Base::Matrix::Matrix<T, M, K> X_1;

  Value_Type decay_rate;
  Value_Type division_min;
  std::array<Value_Type, K> rho;
  std::array<std::size_t, K> rep_num;
};

/** * @brief Linalg solver for inverse diagonal matrices.
 *
 * This class provides methods to compute the inverse of diagonal matrices.
 * It supports both real and complex number types (float or double).
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam M The number of rows and columns in the square matrix.
 */
template <typename T, std::size_t M> class LinalgInvDiag {
public:
  /* Type */
  using Value_Type = typename UnderlyingType<T>::Type;
  static_assert(std::is_same<Value_Type, double>::value ||
                    std::is_same<Value_Type, float>::value,
                "Value data type must be float or double.");

public:
  /* Constructor */
  LinalgInvDiag()
      : X_1(), division_min(static_cast<Value_Type>(
                   DEFAULT_DIVISION_MIN_LINALG_SOLVER)) {}

  /* Copy Constructor */
  LinalgInvDiag(const LinalgInvDiag<T, M> &other)
      : X_1(other.X_1), division_min(other.division_min) {}

  LinalgInvDiag<T, M> &operator=(const LinalgInvDiag<T, M> &other) {
    if (this != &other) {
      this->X_1 = other.X_1;
      this->division_min = other.division_min;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgInvDiag(LinalgInvDiag<T, M> &&other) noexcept
      : X_1(std::move(other.X_1)), division_min(std::move(other.division_min)) {
  }

  LinalgInvDiag<T, M> &operator=(LinalgInvDiag<T, M> &&other) {
    if (this != &other) {
      this->X_1 = std::move(other.X_1);
      this->division_min = std::move(other.division_min);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Computes the inverse of a diagonal matrix.
   *
   * This function computes the inverse of a diagonal matrix A using a custom
   * method that handles both real and complex numbers. It updates the internal
   * state with the computed inverse and returns it as a new matrix object.
   *
   * @param A The input diagonal matrix to invert (of size MxM).
   * @return Matrix<DefDiag, T, M> The computed inverse diagonal matrix.
   */
  inline auto inv(const Matrix<DefDiag, T, M> &A) -> Matrix<DefDiag, T, M> {

    return InverseOperation::InverseDiag<Value_Type, T, M, IS_COMPLEX>::compute(
        A, this->division_min, X_1);
  }

  /**
   * @brief Returns the computed inverse diagonal matrix.
   *
   * This function returns the internal state of the solver, which contains the
   * computed inverse diagonal matrix.
   *
   * @return Matrix<DefDiag, T, M> The computed inverse diagonal matrix.
   */
  inline auto get_answer(void) -> Matrix<DefDiag, T, M> {
    return Matrix<DefDiag, T, M>(this->X_1);
  }

  /**
   * @brief Sets the minimum division threshold for numerical stability.
   *
   * This function sets the minimum division threshold used in the solver to
   * prevent numerical instability during matrix inversion.
   *
   * @param division_min_in The minimum division value to set.
   */
  inline void set_division_min(const T &division_min_in) {
    this->division_min = division_min_in;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;

public:
  /* Variable */
  Base::Matrix::DiagMatrix<T, M> X_1;

  Value_Type division_min;
};

/* make LinalgSolver for inv */

/**
 * @brief Creates a LinalgSolver for matrix inversion.
 *
 * This function template creates an instance of LinalgSolver specialized for
 * matrix inversion based on the type of the input matrix A. It supports dense,
 * diagonal, and sparse matrices.
 *
 * @tparam A_Type The type of the input matrix (e.g., DenseMatrix, DiagMatrix,
 * SparseMatrix).
 * @return LinalgSolverInv_Type<A_Type> An instance of LinalgSolver for matrix
 * inversion.
 */
template <
    typename A_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverInv(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    A_Type::COLS, typename A_Type::SparseAvailable_Type,
                    typename A_Type::SparseAvailable_Type> {

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      A_Type::COLS, typename A_Type::SparseAvailable_Type,
                      typename A_Type::SparseAvailable_Type>();
}

/**
 * @brief Creates a LinalgSolver for diagonal matrix inversion.
 *
 * This function template creates an instance of LinalgInvDiag specialized for
 * diagonal matrices. It supports both real and complex number types.
 *
 * @tparam A_Type The type of the input diagonal matrix (e.g., DiagMatrix).
 * @return LinalgInvDiag<Type, COLS> An instance of LinalgInvDiag for diagonal
 * matrix inversion.
 */
template <typename A_Type, typename std::enable_if<
                               Is_Diag_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverInv(void)
    -> LinalgInvDiag<typename A_Type::Value_Complex_Type, A_Type::COLS> {

  return LinalgInvDiag<typename A_Type::Value_Complex_Type, A_Type::COLS>();
}

/**
 * @brief Creates a LinalgSolver for sparse matrix inversion.
 *
 * This function template creates an instance of LinalgSolver specialized for
 * sparse matrix inversion. It supports both real and complex number types.
 *
 * @tparam A_Type The type of the input sparse matrix (e.g., SparseMatrix).
 * @return LinalgSolverInv_Type<A_Type> An instance of LinalgSolver for sparse
 * matrix inversion.
 */
template <
    typename A_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverInv(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    A_Type::COLS, typename A_Type::SparseAvailable_Type,
                    typename A_Type::SparseAvailable_Type> {

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      A_Type::COLS, typename A_Type::SparseAvailable_Type,
                      typename A_Type::SparseAvailable_Type>();
}

/* LinalgSolver for inv Type */
template <typename A_Type>
using LinalgSolverInv_Type = decltype(make_LinalgSolverInv<A_Type>());

/* make LinalgSolver */

/**
 * @brief Creates a LinalgSolver for solving linear systems.
 *
 * This function template creates an instance of LinalgSolver specialized for
 * solving linear systems based on the types of the input matrices A and B. It
 * supports dense, diagonal, and sparse matrices.
 *
 * @tparam A_Type The type of the coefficient matrix (e.g., DenseMatrix,
 * DiagMatrix, SparseMatrix).
 * @tparam B_Type The type of the right-hand side matrix (e.g., DenseMatrix,
 * DiagMatrix, SparseMatrix).
 * @return LinalgSolver<Type, COLS, ROWS> An instance of LinalgSolver for
 * solving linear systems.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Dense_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Creates a LinalgSolver for solving linear systems with diagonal
 * matrices.
 *
 * This function template creates an instance of LinalgSolver specialized for
 * solving linear systems where the coefficient matrix is diagonal. It supports
 * both real and complex number types.
 *
 * @tparam A_Type The type of the coefficient diagonal matrix (e.g.,
 * DiagMatrix).
 * @tparam B_Type The type of the right-hand side matrix (e.g., DenseMatrix,
 * DiagMatrix, SparseMatrix).
 * @return LinalgSolver<Type, COLS, ROWS> An instance of LinalgSolver for
 * solving linear systems with diagonal matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Diag_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    A_Type::COLS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      A_Type::COLS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Creates a LinalgSolver for solving linear systems with sparse
 * matrices.
 *
 * This function template creates an instance of LinalgSolver specialized for
 * solving linear systems where the coefficient matrix is sparse. It supports
 * both real and complex number types.
 *
 * @tparam A_Type The type of the coefficient sparse matrix (e.g.,
 * SparseMatrix).
 * @tparam B_Type The type of the right-hand side matrix (e.g., DenseMatrix,
 * DiagMatrix, SparseMatrix).
 * @return LinalgSolver<Type, COLS, ROWS> An instance of LinalgSolver for
 * solving linear systems with sparse matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Sparse_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Creates a LinalgSolver for solving linear systems with diagonal
 * matrices.
 *
 * This function template creates an instance of LinalgSolver specialized for
 * solving linear systems where the coefficient matrix is diagonal. It supports
 * both real and complex number types.
 *
 * @tparam A_Type The type of the coefficient diagonal matrix (e.g.,
 * DiagMatrix).
 * @tparam B_Type The type of the right-hand side matrix (e.g., DenseMatrix,
 * DiagMatrix, SparseMatrix).
 * @return LinalgSolver<Type, COLS, ROWS> An instance of LinalgSolver for
 * solving linear systems with diagonal matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Diag_Matrix<A_Type>::value &&
                            Is_Dense_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Creates a LinalgSolver for solving linear systems with diagonal
 * matrices.
 *
 * This function template creates an instance of LinalgSolver specialized for
 * solving linear systems where both the coefficient matrix and the right-hand
 * side matrix are diagonal. It supports both real and complex number types.
 *
 * @tparam A_Type The type of the coefficient diagonal matrix (e.g.,
 * DiagMatrix).
 * @tparam B_Type The type of the right-hand side diagonal matrix (e.g.,
 * DiagMatrix).
 * @return LinalgSolver<Type, COLS, ROWS> An instance of LinalgSolver for
 * solving linear systems with diagonal matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Diag_Matrix<A_Type>::value &&
                            Is_Diag_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    A_Type::COLS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      A_Type::COLS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Creates a LinalgSolver for solving linear systems with diagonal and
 * sparse matrices.
 *
 * This function template creates an instance of LinalgSolver specialized for
 * solving linear systems where the coefficient matrix is diagonal and the
 * right-hand side matrix is sparse. It supports both real and complex number
 * types.
 *
 * @tparam A_Type The type of the coefficient diagonal matrix (e.g.,
 * DiagMatrix).
 * @tparam B_Type The type of the right-hand side sparse matrix (e.g.,
 * SparseMatrix).
 * @return LinalgSolver<Type, COLS, ROWS> An instance of LinalgSolver for
 * solving linear systems with diagonal and sparse matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Diag_Matrix<A_Type>::value &&
                            Is_Sparse_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Creates a LinalgSolver for solving linear systems with sparse
 * matrices.
 *
 * This function template creates an instance of LinalgSolver specialized for
 * solving linear systems where the coefficient matrix is sparse and the
 * right-hand side matrix can be dense, diagonal, or sparse. It supports both
 * real and complex number types.
 *
 * @tparam A_Type The type of the coefficient sparse matrix (e.g.,
 * SparseMatrix).
 * @tparam B_Type The type of the right-hand side matrix (e.g., DenseMatrix,
 * DiagMatrix, SparseMatrix).
 * @return LinalgSolver<Type, COLS, ROWS> An instance of LinalgSolver for
 * solving linear systems with sparse matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Dense_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Creates a LinalgSolver for solving linear systems with sparse and
 * diagonal matrices.
 *
 * This function template creates an instance of LinalgSolver specialized for
 * solving linear systems where the coefficient matrix is sparse and the
 * right-hand side matrix is diagonal. It supports both real and complex number
 * types.
 *
 * @tparam A_Type The type of the coefficient sparse matrix (e.g.,
 * SparseMatrix).
 * @tparam B_Type The type of the right-hand side diagonal matrix (e.g.,
 * DiagMatrix).
 * @return LinalgSolver<Type, COLS, ROWS> An instance of LinalgSolver for
 * solving linear systems with sparse and diagonal matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Diag_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    A_Type::COLS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      A_Type::COLS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Creates a LinalgSolver for solving linear systems with sparse
 * matrices.
 *
 * This function template creates an instance of LinalgSolver specialized for
 * solving linear systems where both the coefficient matrix and the right-hand
 * side matrix are sparse. It supports both real and complex number types.
 *
 * @tparam A_Type The type of the coefficient sparse matrix (e.g.,
 * SparseMatrix).
 * @tparam B_Type The type of the right-hand side sparse matrix (e.g.,
 * SparseMatrix).
 * @return LinalgSolver<Type, COLS, ROWS> An instance of LinalgSolver for
 * solving linear systems with sparse matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Sparse_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgSolver(void)
    -> LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                    B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                    typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                      B_Type::ROWS, typename A_Type::SparseAvailable_Type,
                      typename B_Type::SparseAvailable_Type>();
}

/* LinalgSolver Type */
template <typename A_Type, typename B_Type>
using LinalgSolver_Type = decltype(make_LinalgSolver<A_Type, B_Type>());

/* Linalg Partition Solver */

/** @brief Linalg solver for partitioned matrices.
 *
 * This class provides methods to solve linear systems using partitioned
 * matrices. It supports both dense and sparse matrices, and allows for
 * partitioning based on the number of rows and columns.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam M The number of columns in the square matrix.
 * @tparam K The number of rows in the right-hand side matrix.
 * @tparam SparseAvailable_A Indicates if the coefficient matrix A is sparse.
 * @tparam SparseAvailable_B Indicates if the right-hand side matrix B is
 * sparse.
 */
template <typename T, std::size_t M, std::size_t K, typename SparseAvailable_A,
          typename SparseAvailable_B>
class LinalgPartitionSolver {
public:
  /* Type */
  using Value_Type = typename UnderlyingType<T>::Type;
  static_assert(std::is_same<Value_Type, double>::value ||
                    std::is_same<Value_Type, float>::value,
                "Value data type must be float or double.");

public:
  /* Constructor */
  LinalgPartitionSolver()
      : X_1(), decay_rate(static_cast<Value_Type>(0)),
        division_min(
            static_cast<Value_Type>(DEFAULT_DIVISION_MIN_LINALG_SOLVER)),
        rho({}), rep_num({}) {}

  /* Copy Constructor */
  LinalgPartitionSolver(const LinalgPartitionSolver<T, M, K, SparseAvailable_A,
                                                    SparseAvailable_B> &other)
      : X_1(other.X_1), decay_rate(other.decay_rate),
        division_min(other.division_min), rho(other.rho),
        rep_num(other.rep_num) {}

  LinalgPartitionSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> &
  operator=(const LinalgPartitionSolver<T, M, K, SparseAvailable_A,
                                        SparseAvailable_B> &other) {
    if (this != &other) {
      this->X_1 = other.X_1;
      this->decay_rate = other.decay_rate;
      this->division_min = other.division_min;
      this->rho = other.rho;
      this->rep_num = other.rep_num;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgPartitionSolver(
      LinalgPartitionSolver<T, M, K, SparseAvailable_A, SparseAvailable_B>
          &&other) noexcept
      : X_1(std::move(other.X_1)), decay_rate(std::move(other.decay_rate)),
        division_min(std::move(other.division_min)), rho(std::move(other.rho)),
        rep_num(std::move(other.rep_num)) {}

  LinalgPartitionSolver<T, M, K, SparseAvailable_A, SparseAvailable_B> &
  operator=(LinalgPartitionSolver<T, M, K, SparseAvailable_A, SparseAvailable_B>
                &&other) noexcept {
    if (this != &other) {
      this->X_1 = std::move(other.X_1);
      this->decay_rate = std::move(other.decay_rate);
      this->division_min = std::move(other.division_min);
      this->rho = std::move(other.rho);
      this->rep_num = std::move(other.rep_num);
    }
    return *this;
  }

  /* Solve function */

  /**
   * @brief Solves the linear system Ax = B for dense matrices.
   *
   * This function computes the solution of the linear system represented by
   * the dense matrix A and the dense matrix B. It updates the internal state
   * of the solver with the computed solution.
   *
   * @param A The coefficient dense matrix (of size MxM).
   * @param B The right-hand side dense matrix (of size MxK).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDense, T, M, M> &A,
                    const Matrix<DefDense, T, M, K> &B, std::size_t matrix_size)
      -> Matrix<DefDense, T, M, K> {

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::gmres_k_partition_matrix(
        A.matrix, B.matrix, this->X_1, this->decay_rate, this->division_min,
        this->rho, this->rep_num, matrix_size);

    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B for dense matrices with cold start.
   *
   * This function computes the solution of the linear system represented by
   * the dense matrix A and the dense matrix B without using any previous
   * solution as a starting point. It returns a new matrix object with the
   * computed solution.
   *
   * @param A The coefficient dense matrix (of size MxM).
   * @param B The right-hand side dense matrix (of size MxK).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto cold_solve(const Matrix<DefDense, T, M, M> &A,
                         const Matrix<DefDense, T, M, K> &B,
                         std::size_t matrix_size) -> Matrix<DefDense, T, M, K> {

    Base::Matrix::Matrix<T, M, K> X_1_temporary;

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::gmres_k_partition_matrix(
        A.matrix, B.matrix, X_1_temporary, this->decay_rate, this->division_min,
        this->rho, this->rep_num, matrix_size);

    return Matrix<DefDense, T, M, K>(X_1_temporary);
  }

  /**
   * @brief Solves the linear system Ax = B for dense matrices with sparse B.
   *
   * This function computes the solution of the linear system represented by
   * the dense matrix A and the sparse matrix B. It updates the internal state
   * of the solver with the computed solution.
   *
   * @param A The coefficient dense matrix (of size MxM).
   * @param B The right-hand side sparse matrix (of size MxK).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDense, T, M, M> &A,
                    const Matrix<DefDiag, T, M> &B, std::size_t matrix_size)
      -> Matrix<DefDense, T, M, M> {

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::gmres_k_partition_matrix(
        A.matrix, B.matrix, this->X_1, this->decay_rate, this->division_min,
        this->rho, this->rep_num, matrix_size);

    return Matrix<DefDense, T, M, M>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B for dense matrices with sparse B
   * and cold start.
   *
   * This function computes the solution of the linear system represented by
   * the dense matrix A and the sparse matrix B without using any previous
   * solution as a starting point. It returns a new matrix object with the
   * computed solution.
   *
   * @param A The coefficient dense matrix (of size MxM).
   * @param B The right-hand side sparse matrix (of size MxK).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto cold_solve(const Matrix<DefDense, T, M, M> &A,
                         const Matrix<DefDiag, T, M> &B,
                         std::size_t matrix_size) -> Matrix<DefDense, T, M, M> {

    Base::Matrix::Matrix<T, M, K> X_1_temporary;

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::gmres_k_partition_matrix(
        A.matrix, B.matrix, X_1_temporary, this->decay_rate, this->division_min,
        this->rho, this->rep_num, matrix_size);

    return Matrix<DefDense, T, M, M>(X_1_temporary);
  }

  /**
   * @brief Solves the linear system Ax = B for dense matrices with sparse B.
   *
   * This function computes the solution of the linear system represented by
   * the dense matrix A and the sparse matrix B. It updates the internal state
   * of the solver with the computed solution.
   *
   * @param A The coefficient dense matrix (of size MxM).
   * @param B The right-hand side sparse matrix (of size MxK).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDense, T, M, M> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B,
                    std::size_t matrix_size) -> Matrix<DefDense, T, M, K> {

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::Matrix<T, M, K> B_dense_matrix =
        Base::Matrix::output_dense_matrix(B.matrix);

    Base::Matrix::gmres_k_partition_matrix(
        A.matrix, B_dense_matrix, this->X_1, this->decay_rate,
        this->division_min, this->rho, this->rep_num, matrix_size);

    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B for dense matrices with sparse B
   * and cold start.
   *
   * This function computes the solution of the linear system represented by
   * the dense matrix A and the sparse matrix B without using any previous
   * solution as a starting point. It returns a new matrix object with the
   * computed solution.
   *
   * @param A The coefficient dense matrix (of size MxM).
   * @param B The right-hand side sparse matrix (of size MxK).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto cold_solve(const Matrix<DefDense, T, M, M> &A,
                         const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B,
                         std::size_t matrix_size) -> Matrix<DefDense, T, M, K> {

    Base::Matrix::Matrix<T, M, K> X_1_temporary;

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::Matrix<T, M, K> B_dense_matrix =
        Base::Matrix::output_dense_matrix(B.matrix);

    Base::Matrix::gmres_k_partition_matrix(
        A.matrix, B_dense_matrix, X_1_temporary, this->decay_rate,
        this->division_min, this->rho, this->rep_num, matrix_size);

    return Matrix<DefDense, T, M, K>(X_1_temporary);
  }

  /**
   * @brief Solves the linear system Ax = B for diagonal matrices.
   *
   * This function computes the solution of the linear system represented by
   * the diagonal matrix A and the dense matrix B. It updates the internal state
   * of the solver with the computed solution.
   *
   * @param A The coefficient diagonal matrix (of size MxM).
   * @param B The right-hand side dense matrix (of size MxK).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDiag, T, M> &A,
                    const Matrix<DefDense, T, M, K> &B, std::size_t matrix_size)
      -> Matrix<DefDense, T, M, K> {

    if (matrix_size > M) {
      matrix_size = M;
    }

    this->X_1 = Base::Matrix::diag_inv_multiply_dense_partition(
        A.matrix, B.matrix, this->division_min, matrix_size);

    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B for diagonal matrices with cold
   * start.
   *
   * This function computes the solution of the linear system represented by
   * the diagonal matrix A and the dense matrix B without using any previous
   * solution as a starting point. It returns a new matrix object with the
   * computed solution.
   *
   * @param A The coefficient diagonal matrix (of size MxM).
   * @param B The right-hand side dense matrix (of size MxK).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto cold_solve(const Matrix<DefDiag, T, M> &A,
                         const Matrix<DefDense, T, M, K> &B,
                         std::size_t matrix_size) -> Matrix<DefDense, T, M, K> {

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::Matrix<T, M, K> X_1_temporary =
        Base::Matrix::diag_inv_multiply_dense_partition(
            A.matrix, B.matrix, this->division_min, matrix_size);

    return Matrix<DefDense, T, M, K>(X_1_temporary);
  }

  /**
   * @brief Solves the linear system Ax = B for diagonal matrices.
   *
   * This function computes the solution of the linear system represented by
   * the diagonal matrix A and the diagonal matrix B. It updates the internal
   * state of the solver with the computed solution.
   *
   * @param A The coefficient diagonal matrix (of size MxM).
   * @param B The right-hand side diagonal matrix (of size MxM).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDiag, T, M> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDiag, T, M> &A,
                    const Matrix<DefDiag, T, M> &B, std::size_t matrix_size)
      -> Matrix<DefDiag, T, M> {

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::DiagMatrix<T, M> result =
        Base::Matrix::diag_divide_diag_partition(
            B.matrix, A.matrix, this->division_min, matrix_size);

    return Matrix<DefDiag, T, M>(std::move(result));
  }

  /**
   * @brief Solves the linear system Ax = B for diagonal matrices with cold
   * start.
   *
   * This function computes the solution of the linear system represented by
   * the diagonal matrix A and the diagonal matrix B without using any previous
   * solution as a starting point. It returns a new matrix object with the
   * computed solution.
   *
   * @param A The coefficient diagonal matrix (of size MxM).
   * @param B The right-hand side diagonal matrix (of size MxM).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDiag, T, M> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDiag, T, M> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B,
                    std::size_t matrix_size) -> Matrix<DefDense, T, M, K> {

    if (matrix_size > M) {
      matrix_size = M;
    }

    this->X_1 = Base::Matrix::output_dense_matrix(
        Base::Matrix::diag_inv_multiply_sparse_partition(
            A.matrix, B.matrix, this->division_min, matrix_size));

    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B for diagonal matrices with cold
   * start.
   *
   * This function computes the solution of the linear system represented by
   * the diagonal matrix A and the sparse matrix B without using any previous
   * solution as a starting point. It returns a new matrix object with the
   * computed solution.
   *
   * @param A The coefficient diagonal matrix (of size MxM).
   * @param B The right-hand side sparse matrix (of size MxK).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto cold_solve(const Matrix<DefDiag, T, M> &A,
                         const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B,
                         std::size_t matrix_size) -> Matrix<DefDense, T, M, K> {

    using RowIndices_B = RowIndicesFromSparseAvailable<SparseAvailable_B>;
    using RowPointers_B = RowPointersFromSparseAvailable<SparseAvailable_B>;

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B>
        X_1_temporary = Base::Matrix::diag_inv_multiply_sparse_partition(
            A.matrix, B.matrix, this->division_min, matrix_size);

    return Matrix<DefDense, T, M, K>(
        Base::Matrix::output_dense_matrix(X_1_temporary));
  }

  /**
   * @brief Solves the linear system Ax = B for sparse matrices.
   *
   * This function computes the solution of the linear system represented by
   * the sparse matrix A and the dense matrix B. It updates the internal state
   * of the solver with the computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxM).
   * @param B The right-hand side dense matrix (of size MxK).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                    const Matrix<DefDense, T, M, K> &B, std::size_t matrix_size)
      -> Matrix<DefDense, T, M, K> {

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::sparse_gmres_k_partition_matrix(
        A.matrix, B.matrix, this->X_1, this->decay_rate, this->division_min,
        this->rho, this->rep_num, matrix_size);

    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B for sparse matrices with cold start.
   *
   * This function computes the solution of the linear system represented by
   * the sparse matrix A and the dense matrix B without using any previous
   * solution as a starting point. It returns a new matrix object with the
   * computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxM).
   * @param B The right-hand side dense matrix (of size MxK).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto cold_solve(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                         const Matrix<DefDense, T, M, K> &B,
                         std::size_t matrix_size) -> Matrix<DefDense, T, M, K> {

    Base::Matrix::Matrix<T, M, K> X_1_temporary;

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::sparse_gmres_k_partition_matrix(
        A.matrix, B.matrix, X_1_temporary, this->decay_rate, this->division_min,
        this->rho, this->rep_num, matrix_size);

    return Matrix<DefDense, T, M, K>(X_1_temporary);
  }

  /**
   * @brief Solves the linear system Ax = B for sparse matrices with diagonal B.
   *
   * This function computes the solution of the linear system represented by
   * the sparse matrix A and the diagonal matrix B. It updates the internal
   * state of the solver with the computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxM).
   * @param B The right-hand side diagonal matrix (of size MxM).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, M> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                    const Matrix<DefDiag, T, M> &B, std::size_t matrix_size)
      -> Matrix<DefDense, T, M, M> {

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::Matrix<T, M, M> B_dense_matrix =
        output_dense_matrix(B.matrix);

    Base::Matrix::sparse_gmres_k_partition_matrix(
        A.matrix, B_dense_matrix, this->X_1, this->decay_rate,
        this->division_min, this->rho, this->rep_num, matrix_size);

    return Matrix<DefDense, T, M, M>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B for sparse matrices with diagonal B
   * and cold start.
   *
   * This function computes the solution of the linear system represented by
   * the sparse matrix A and the diagonal matrix B without using any previous
   * solution as a starting point. It returns a new matrix object with the
   * computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxM).
   * @param B The right-hand side diagonal matrix (of size MxM).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, M> The computed solution matrix.
   */
  inline auto cold_solve(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                         const Matrix<DefDiag, T, M> &B,
                         std::size_t matrix_size) -> Matrix<DefDense, T, M, M> {

    Base::Matrix::Matrix<T, M, K> X_1_temporary;

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::Matrix<T, M, M> B_dense_matrix =
        output_dense_matrix(B.matrix);

    Base::Matrix::sparse_gmres_k_partition_matrix(
        A.matrix, B_dense_matrix, X_1_temporary, this->decay_rate,
        this->division_min, this->rho, this->rep_num, matrix_size);

    return Matrix<DefDense, T, M, M>(X_1_temporary);
  }

  /**
   * @brief Solves the linear system Ax = B for sparse matrices with sparse B.
   *
   * This function computes the solution of the linear system represented by
   * the sparse matrix A and the sparse matrix B. It updates the internal state
   * of the solver with the computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxM).
   * @param B The right-hand side sparse matrix (of size MxK).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B,
                    std::size_t matrix_size) -> Matrix<DefDense, T, M, K> {

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::Matrix<T, M, K> B_dense_matrix =
        Base::Matrix::output_dense_matrix(B.matrix);

    Base::Matrix::sparse_gmres_k_partition_matrix(
        A.matrix, B_dense_matrix, this->X_1, this->decay_rate,
        this->division_min, this->rho, this->rep_num, matrix_size);

    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /**
   * @brief Solves the linear system Ax = B for sparse matrices with sparse B
   * and cold start.
   *
   * This function computes the solution of the linear system represented by
   * the sparse matrix A and the sparse matrix B without using any previous
   * solution as a starting point. It returns a new matrix object with the
   * computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxM).
   * @param B The right-hand side sparse matrix (of size MxK).
   * @param matrix_size The size of the matrices to be processed.
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto cold_solve(const Matrix<DefSparse, T, M, M, SparseAvailable_A> &A,
                         const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B,
                         std::size_t matrix_size) -> Matrix<DefDense, T, M, K> {

    Base::Matrix::Matrix<T, M, K> X_1_temporary;

    if (matrix_size > M) {
      matrix_size = M;
    }

    Base::Matrix::Matrix<T, M, K> B_dense_matrix =
        Base::Matrix::output_dense_matrix(B.matrix);

    Base::Matrix::sparse_gmres_k_partition_matrix(
        A.matrix, B_dense_matrix, X_1_temporary, this->decay_rate,
        this->division_min, this->rho, this->rep_num, matrix_size);

    return Matrix<DefDense, T, M, K>(X_1_temporary);
  }

  /**
   * @brief Returns the computed solution matrix.
   *
   * This function returns the computed solution matrix stored in the solver.
   * It is typically called after a successful solve operation.
   *
   * @return Matrix<DefDense, T, M, K> The computed solution matrix.
   */
  inline auto get_answer(void) -> Matrix<DefDense, T, M, K> {
    return Matrix<DefDense, T, M, K>(this->X_1);
  }

  /**
   * @brief Sets the decay rate for the solver.
   *
   * This function sets the decay rate used in the solver's computations.
   * The decay rate influences the convergence behavior of the solver.
   *
   * @param decay_rate_in The decay rate to be set.
   */
  inline void set_decay_rate(const Value_Type &decay_rate_in) {
    this->decay_rate = decay_rate_in;
  }

  /**
   * @brief Sets the minimum division value for the solver.
   *
   * This function sets the minimum division value used in the solver's
   * computations. It prevents division by zero and controls numerical
   * stability.
   *
   * @param division_min_in The minimum division value to be set.
   */
  inline void set_division_min(const Value_Type &division_min_in) {
    this->division_min = division_min_in;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = K;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;

public:
  /* Variable */
  Base::Matrix::Matrix<T, M, K> X_1;

  Value_Type decay_rate;
  Value_Type division_min;
  std::array<Value_Type, K> rho;
  std::array<std::size_t, K> rep_num;
};

/* make LinalgPartitionSolver */

/**
 * @brief Factory function to create a LinalgPartitionSolver instance.
 *
 * This function creates an instance of LinalgPartitionSolver based on the
 * types of the input matrices A and B. It ensures that the value types of A and
 * B are compatible and returns a solver configured for the specific matrix
 * dimensions and sparsity.
 *
 * @tparam A_Type The type of the coefficient matrix A.
 * @tparam B_Type The type of the right-hand side matrix B.
 * @return LinalgPartitionSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Dense_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgPartitionSolver(void)
    -> LinalgPartitionSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                             B_Type::ROWS,
                             typename A_Type::SparseAvailable_Type,
                             typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgPartitionSolver<typename A_Type::Value_Complex_Type,
                               A_Type::COLS, B_Type::ROWS,
                               typename A_Type::SparseAvailable_Type,
                               typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Factory function to create a LinalgPartitionSolver instance for
 * diagonal matrices.
 *
 * This function creates an instance of LinalgPartitionSolver configured for
 * diagonal matrices A and B. It ensures that the value types of A and B are
 * compatible and returns a solver configured for the specific matrix dimensions
 * and sparsity.
 *
 * @tparam A_Type The type of the coefficient diagonal matrix A.
 * @tparam B_Type The type of the right-hand side diagonal matrix B.
 * @return LinalgPartitionSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Diag_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgPartitionSolver(void)
    -> LinalgPartitionSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                             A_Type::COLS,
                             typename A_Type::SparseAvailable_Type,
                             typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgPartitionSolver<typename A_Type::Value_Complex_Type,
                               A_Type::COLS, A_Type::COLS,
                               typename A_Type::SparseAvailable_Type,
                               typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Factory function to create a LinalgPartitionSolver instance for
 * sparse matrices.
 *
 * This function creates an instance of LinalgPartitionSolver configured for
 * sparse matrices A and B. It ensures that the value types of A and B are
 * compatible and returns a solver configured for the specific matrix dimensions
 * and sparsity.
 *
 * @tparam A_Type The type of the coefficient sparse matrix A.
 * @tparam B_Type The type of the right-hand side sparse matrix B.
 * @return LinalgPartitionSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Sparse_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgPartitionSolver(void)
    -> LinalgPartitionSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                             B_Type::ROWS,
                             typename A_Type::SparseAvailable_Type,
                             typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgPartitionSolver<typename A_Type::Value_Complex_Type,
                               A_Type::COLS, B_Type::ROWS,
                               typename A_Type::SparseAvailable_Type,
                               typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Factory function to create a LinalgPartitionSolver instance for
 * diagonal matrices with sparse B.
 *
 * This function creates an instance of LinalgPartitionSolver configured for
 * diagonal matrix A and sparse matrix B. It ensures that the value types of A
 * and B are compatible and returns a solver configured for the specific matrix
 * dimensions and sparsity.
 *
 * @tparam A_Type The type of the coefficient diagonal matrix A.
 * @tparam B_Type The type of the right-hand side sparse matrix B.
 * @return LinalgPartitionSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Diag_Matrix<A_Type>::value &&
                            Is_Dense_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgPartitionSolver(void)
    -> LinalgPartitionSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                             B_Type::ROWS,
                             typename A_Type::SparseAvailable_Type,
                             typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgPartitionSolver<typename A_Type::Value_Complex_Type,
                               A_Type::COLS, B_Type::ROWS,
                               typename A_Type::SparseAvailable_Type,
                               typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Factory function to create a LinalgPartitionSolver instance for
 * diagonal matrices with diagonal B.
 *
 * This function creates an instance of LinalgPartitionSolver configured for
 * diagonal matrices A and B. It ensures that the value types of A and B are
 * compatible and returns a solver configured for the specific matrix dimensions
 * and sparsity.
 *
 * @tparam A_Type The type of the coefficient diagonal matrix A.
 * @tparam B_Type The type of the right-hand side diagonal matrix B.
 * @return LinalgPartitionSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Diag_Matrix<A_Type>::value &&
                            Is_Diag_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgPartitionSolver(void)
    -> LinalgPartitionSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                             A_Type::COLS,
                             typename A_Type::SparseAvailable_Type,
                             typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgPartitionSolver<typename A_Type::Value_Complex_Type,
                               A_Type::COLS, A_Type::COLS,
                               typename A_Type::SparseAvailable_Type,
                               typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Factory function to create a LinalgPartitionSolver instance for
 * diagonal matrices with sparse B.
 *
 * This function creates an instance of LinalgPartitionSolver configured for
 * diagonal matrix A and sparse matrix B. It ensures that the value types of A
 * and B are compatible and returns a solver configured for the specific matrix
 * dimensions and sparsity.
 *
 * @tparam A_Type The type of the coefficient diagonal matrix A.
 * @tparam B_Type The type of the right-hand side sparse matrix B.
 * @return LinalgPartitionSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Diag_Matrix<A_Type>::value &&
                            Is_Sparse_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgPartitionSolver(void)
    -> LinalgPartitionSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                             B_Type::ROWS,
                             typename A_Type::SparseAvailable_Type,
                             typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgPartitionSolver<typename A_Type::Value_Complex_Type,
                               A_Type::COLS, B_Type::ROWS,
                               typename A_Type::SparseAvailable_Type,
                               typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Factory function to create a LinalgPartitionSolver instance for
 * sparse matrices with dense B.
 *
 * This function creates an instance of LinalgPartitionSolver configured for
 * sparse matrix A and dense matrix B. It ensures that the value types of A and
 * B are compatible and returns a solver configured for the specific matrix
 * dimensions and sparsity.
 *
 * @tparam A_Type The type of the coefficient sparse matrix A.
 * @tparam B_Type The type of the right-hand side dense matrix B.
 * @return LinalgPartitionSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Dense_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgPartitionSolver(void)
    -> LinalgPartitionSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                             B_Type::ROWS,
                             typename A_Type::SparseAvailable_Type,
                             typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgPartitionSolver<typename A_Type::Value_Complex_Type,
                               A_Type::COLS, B_Type::ROWS,
                               typename A_Type::SparseAvailable_Type,
                               typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Factory function to create a LinalgPartitionSolver instance for
 * sparse matrices with diagonal B.
 *
 * This function creates an instance of LinalgPartitionSolver configured for
 * sparse matrix A and diagonal matrix B. It ensures that the value types of A
 * and B are compatible and returns a solver configured for the specific matrix
 * dimensions and sparsity.
 *
 * @tparam A_Type The type of the coefficient sparse matrix A.
 * @tparam B_Type The type of the right-hand side diagonal matrix B.
 * @return LinalgPartitionSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Diag_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgPartitionSolver(void)
    -> LinalgPartitionSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                             A_Type::COLS,
                             typename A_Type::SparseAvailable_Type,
                             typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgPartitionSolver<typename A_Type::Value_Complex_Type,
                               A_Type::COLS, A_Type::COLS,
                               typename A_Type::SparseAvailable_Type,
                               typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Factory function to create a LinalgPartitionSolver instance for
 * sparse matrices with diagonal B.
 *
 * This function creates an instance of LinalgPartitionSolver configured for
 * sparse matrix A and diagonal matrix B. It ensures that the value types of A
 * and B are compatible and returns a solver configured for the specific matrix
 * dimensions and sparsity.
 *
 * @tparam A_Type The type of the coefficient sparse matrix A.
 * @tparam B_Type The type of the right-hand side diagonal matrix B.
 * @return LinalgPartitionSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Sparse_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgPartitionSolver(void)
    -> LinalgPartitionSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                             B_Type::ROWS,
                             typename A_Type::SparseAvailable_Type,
                             typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgPartitionSolver<typename A_Type::Value_Complex_Type,
                               A_Type::COLS, B_Type::ROWS,
                               typename A_Type::SparseAvailable_Type,
                               typename B_Type::SparseAvailable_Type>();
}

/* LinalgPartitionSolver Type */
template <typename A_Type, typename B_Type>
using LinalgPartitionSolver_Type =
    decltype(make_LinalgPartitionSolver<A_Type, B_Type>());

/* least-squares solution to a linear matrix equation */

/**
 * @brief Class for solving least-squares problems using GMRES.
 *
 * This class provides methods to solve least-squares problems of the form
 * Ax = B, where A is a matrix and B is a matrix or vector. It supports dense,
 * diagonal, and sparse matrices, and can handle both cold starts and warm
 * starts.
 *
 * @tparam T The value type of the matrices (float or double).
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrix A (and rows in matrix B).
 * @tparam K The number of rows in matrix B.
 * @tparam SparseAvailable_A Indicates if sparse matrices are available for A.
 * @tparam SparseAvailable_B Indicates if sparse matrices are available for B.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable_A, typename SparseAvailable_B>
class LinalgLstsqSolver {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<Value_Type, double>::value ||
                    std::is_same<Value_Type, float>::value,
                "Value data type must be float or double.");

  using SparseAvailable_A_Type = SparseAvailable_A;
  using SparseAvailable_B_Type = SparseAvailable_B;

public:
  /* Constructor */
  LinalgLstsqSolver()
      : X_1(), decay_rate(static_cast<Value_Type>(0)),
        division_min(
            static_cast<Value_Type>(DEFAULT_DIVISION_MIN_LINALG_SOLVER)),
        rho({}), rep_num({}) {}

  /* Copy Constructor */
  LinalgLstsqSolver(const LinalgLstsqSolver<T, M, N, K, SparseAvailable_A,
                                            SparseAvailable_B> &other)
      : X_1(other.X_1), decay_rate(other.decay_rate),
        division_min(other.division_min), rho(other.rho),
        rep_num(other.rep_num) {}

  LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B> &
  operator=(const LinalgLstsqSolver<T, M, N, K, SparseAvailable_A,
                                    SparseAvailable_B> &other) {
    if (this != &other) {
      this->X_1 = other.X_1;
      this->decay_rate = other.decay_rate;
      this->division_min = other.division_min;
      this->rho = other.rho;
      this->rep_num = other.rep_num;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgLstsqSolver(LinalgLstsqSolver<T, M, N, K, SparseAvailable_A,
                                      SparseAvailable_B> &&other) noexcept
      : X_1(std::move(other.X_1)), decay_rate(std::move(other.decay_rate)),
        division_min(std::move(other.division_min)), rho(std::move(other.rho)),
        rep_num(std::move(other.rep_num)) {}

  LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B> &
  operator=(LinalgLstsqSolver<T, M, N, K, SparseAvailable_A, SparseAvailable_B>
                &&other) {
    if (this != &other) {
      this->X_1 = std::move(other.X_1);
      this->decay_rate = std::move(other.decay_rate);
      this->division_min = std::move(other.division_min);
      this->rho = std::move(other.rho);
      this->rep_num = std::move(other.rep_num);
    }
    return *this;
  }

public:
  /* Solve method */

  /**
   * @brief Solves the least-squares problem Ax = B for dense matrices.
   *
   * This function computes the least-squares solution of the linear system
   * represented by the dense matrix A and the dense matrix B. It updates the
   * internal state of the solver with the computed solution.
   *
   * @param A The coefficient dense matrix (of size MxN).
   * @param B The right-hand side dense matrix (of size MxK).
   * @return Matrix<DefDense, T, N, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDense, T, M, N> &A,
                    const Matrix<DefDense, T, M, K> &B)
      -> Matrix<DefDense, T, N, K> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::gmres_k_rect_matrix(A.matrix, B.matrix, this->X_1,
                                      this->decay_rate, this->division_min,
                                      this->rho, this->rep_num);

    return Matrix<DefDense, T, N, K>(X_1);
  }

  /**
   * @brief Solves the least-squares problem Ax = B for dense matrices with
   * diagonal B.
   *
   * This function computes the least-squares solution of the linear system
   * represented by the dense matrix A and the diagonal matrix B. It updates
   * the internal state of the solver with the computed solution.
   *
   * @param A The coefficient dense matrix (of size MxN).
   * @param B The right-hand side diagonal matrix (of size MxM).
   * @return Matrix<DefDense, T, N, M> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDense, T, M, N> &A,
                    const Matrix<DefDiag, T, M> &B)
      -> Matrix<DefDense, T, N, M> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::gmres_k_rect_matrix(A.matrix, B.matrix, this->X_1,
                                      this->decay_rate, this->division_min,
                                      this->rho, this->rep_num);

    return Matrix<DefDense, T, N, M>(X_1);
  }

  /**
   * @brief Solves the least-squares problem Ax = B for dense matrices with
   * sparse B.
   *
   * This function computes the least-squares solution of the linear system
   * represented by the dense matrix A and the sparse matrix B. It updates the
   * internal state of the solver with the computed solution.
   *
   * @param A The coefficient dense matrix (of size MxN).
   * @param B The right-hand side sparse matrix (of size MxK).
   * @return Matrix<DefDense, T, N, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefDense, T, M, N> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
      -> Matrix<DefDense, T, N, K> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::Matrix<T, M, K> B_dense_matrix =
        Base::Matrix::output_dense_matrix(B.matrix);

    Base::Matrix::gmres_k_rect_matrix(A.matrix, B_dense_matrix, this->X_1,
                                      this->decay_rate, this->division_min,
                                      this->rho, this->rep_num);

    return Matrix<DefDense, T, N, K>(X_1);
  }

  /**
   * @brief Solves the least-squares problem Ax = B for dense matrices with
   * sparse B and cold start.
   *
   * This function computes the least-squares solution of the linear system
   * represented by the dense matrix A and the sparse matrix B without using
   * any previous solution as a starting point. It returns a new matrix object
   * with the computed solution.
   *
   * @param A The coefficient dense matrix (of size MxN).
   * @param B The right-hand side sparse matrix (of size MxK).
   * @return Matrix<DefDense, T, N, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                    const Matrix<DefDense, T, M, K> &B)
      -> Matrix<DefDense, T, N, K> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::sparse_gmres_k_rect_matrix(
        A.matrix, B.matrix, this->X_1, this->decay_rate, this->division_min,
        this->rho, this->rep_num);

    return Matrix<DefDense, T, N, K>(X_1);
  }

  /**
   * @brief Solves the least-squares problem Ax = B for sparse matrices with
   * diagonal B.
   *
   * This function computes the least-squares solution of the linear system
   * represented by the sparse matrix A and the diagonal matrix B. It updates
   * the internal state of the solver with the computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxN).
   * @param B The right-hand side diagonal matrix (of size MxM).
   * @return Matrix<DefDense, T, N, M> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                    const Matrix<DefDiag, T, M> &B)
      -> Matrix<DefDense, T, N, M> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::sparse_gmres_k_rect_matrix(
        A.matrix, B.matrix, this->X_1, this->decay_rate, this->division_min,
        this->rho, this->rep_num);

    return Matrix<DefDense, T, N, M>(X_1);
  }

  /**
   * @brief Solves the least-squares problem Ax = B for sparse matrices with
   * sparse B.
   *
   * This function computes the least-squares solution of the linear system
   * represented by the sparse matrix A and the sparse matrix B. It updates the
   * internal state of the solver with the computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxN).
   * @param B The right-hand side sparse matrix (of size MxK).
   * @return Matrix<DefDense, T, N, K> The computed solution matrix.
   */
  inline auto solve(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                    const Matrix<DefSparse, T, M, K, SparseAvailable_B> &B)
      -> Matrix<DefDense, T, N, K> {
    static_assert(M > N, "Class LinalgLstsqSolver argument 1, column number "
                         "must be larger than row number.");

    Base::Matrix::Matrix<T, M, K> B_dense_matrix =
        Base::Matrix::output_dense_matrix(B.matrix);

    Base::Matrix::sparse_gmres_k_rect_matrix(
        A.matrix, B_dense_matrix, this->X_1, this->decay_rate,
        this->division_min, this->rho, this->rep_num);

    return Matrix<DefDense, T, N, K>(X_1);
  }

  /**
   * @brief Solves the least-squares problem Ax = B for sparse matrices with
   * sparse B and cold start.
   *
   * This function computes the least-squares solution of the linear system
   * represented by the sparse matrix A and the sparse matrix B without using
   * any previous solution as a starting point. It returns a new matrix object
   * with the computed solution.
   *
   * @param A The coefficient sparse matrix (of size MxN).
   * @param B The right-hand side sparse matrix (of size MxK).
   * @return Matrix<DefDense, T, N, K> The computed solution matrix.
   */
  inline auto get_answer(void) -> Matrix<DefDense, T, N, K> {
    return Matrix<DefDense, T, N, K>(this->X_1);
  }

  /**
   * @brief Sets the decay rate for the solver.
   *
   * This function sets the decay rate used in the solver's computations.
   * The decay rate influences the convergence behavior of the solver.
   *
   * @param decay_rate_in The decay rate to be set.
   */
  inline void set_decay_rate(const Value_Type &decay_rate_in) {
    this->decay_rate = decay_rate_in;
  }

  /**
   * @brief Sets the minimum division value for the solver.
   *
   * This function sets the minimum division value used in the solver's
   * computations. It prevents division by zero and controls numerical
   * stability.
   *
   * @param division_min_in The minimum division value to be set.
   */
  inline void set_division_min(const Value_Type &division_min_in) {
    this->division_min = division_min_in;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = N;
  static constexpr std::size_t ROWS = K;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

public:
  /* Properties */
  Base::Matrix::Matrix<T, N, K> X_1;

  Value_Type decay_rate;
  Value_Type division_min;
  std::array<Value_Type, K> rho;
  std::array<std::size_t, K> rep_num;
};

/* make LinalgLstsqSolver */

/**
 * @brief Factory function to create a LinalgLstsqSolver instance.
 *
 * This function creates an instance of LinalgLstsqSolver based on the types of
 * the input matrices A and B. It ensures that the value types of A and B are
 * compatible and returns a solver configured for the specific matrix dimensions
 * and sparsity.
 *
 * @tparam A_Type The type of the coefficient matrix A.
 * @tparam B_Type The type of the right-hand side matrix B.
 * @return LinalgLstsqSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Dense_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgLstsqSolver(void)
    -> LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                         A_Type::ROWS, B_Type::ROWS,
                         typename A_Type::SparseAvailable_Type,
                         typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                           A_Type::ROWS, B_Type::ROWS,
                           typename A_Type::SparseAvailable_Type,
                           typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Factory function to create a LinalgLstsqSolver instance for diagonal
 * matrices.
 *
 * This function creates an instance of LinalgLstsqSolver configured for
 * diagonal matrices A and B. It ensures that the value types of A and B are
 * compatible and returns a solver configured for the specific matrix dimensions
 * and sparsity.
 *
 * @tparam A_Type The type of the coefficient diagonal matrix A.
 * @tparam B_Type The type of the right-hand side diagonal matrix B.
 * @return LinalgLstsqSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Diag_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgLstsqSolver(void)
    -> LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                         A_Type::ROWS, A_Type::COLS,
                         typename A_Type::SparseAvailable_Type,
                         typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                           A_Type::ROWS, A_Type::COLS,
                           typename A_Type::SparseAvailable_Type,
                           typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Factory function to create a LinalgLstsqSolver instance for sparse
 * matrices.
 *
 * This function creates an instance of LinalgLstsqSolver configured for sparse
 * matrices A and B. It ensures that the value types of A and B are compatible
 * and returns a solver configured for the specific matrix dimensions and
 * sparsity.
 *
 * @tparam A_Type The type of the coefficient sparse matrix A.
 * @tparam B_Type The type of the right-hand side sparse matrix B.
 * @return LinalgLstsqSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value &&
                            Is_Sparse_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgLstsqSolver(void)
    -> LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                         A_Type::ROWS, B_Type::ROWS,
                         typename A_Type::SparseAvailable_Type,
                         typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                           A_Type::ROWS, B_Type::ROWS,
                           typename A_Type::SparseAvailable_Type,
                           typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Factory function to create a LinalgLstsqSolver instance for diagonal
 * matrices with sparse B.
 *
 * This function creates an instance of LinalgLstsqSolver configured for
 * diagonal matrix A and sparse matrix B. It ensures that the value types of A
 * and B are compatible and returns a solver configured for the specific matrix
 * dimensions and sparsity.
 *
 * @tparam A_Type The type of the coefficient diagonal matrix A.
 * @tparam B_Type The type of the right-hand side sparse matrix B.
 * @return LinalgLstsqSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Dense_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgLstsqSolver(void)
    -> LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                         A_Type::ROWS, B_Type::ROWS,
                         typename A_Type::SparseAvailable_Type,
                         typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                           A_Type::ROWS, B_Type::ROWS,
                           typename A_Type::SparseAvailable_Type,
                           typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Factory function to create a LinalgLstsqSolver instance for diagonal
 * matrices with diagonal B.
 *
 * This function creates an instance of LinalgLstsqSolver configured for
 * diagonal matrices A and B. It ensures that the value types of A and B are
 * compatible and returns a solver configured for the specific matrix dimensions
 * and sparsity.
 *
 * @tparam A_Type The type of the coefficient diagonal matrix A.
 * @tparam B_Type The type of the right-hand side diagonal matrix B.
 * @return LinalgLstsqSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Diag_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgLstsqSolver(void)
    -> LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                         A_Type::ROWS, A_Type::COLS,
                         typename A_Type::SparseAvailable_Type,
                         typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                           A_Type::ROWS, A_Type::COLS,
                           typename A_Type::SparseAvailable_Type,
                           typename B_Type::SparseAvailable_Type>();
}

/**
 * @brief Factory function to create a LinalgLstsqSolver instance for sparse
 * matrices with dense B.
 *
 * This function creates an instance of LinalgLstsqSolver configured for sparse
 * matrix A and dense matrix B. It ensures that the value types of A and B are
 * compatible and returns a solver configured for the specific matrix dimensions
 * and sparsity.
 *
 * @tparam A_Type The type of the coefficient sparse matrix A.
 * @tparam B_Type The type of the right-hand side dense matrix B.
 * @return LinalgLstsqSolver instance configured for the given matrices.
 */
template <
    typename A_Type, typename B_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value &&
                            Is_Sparse_Matrix<B_Type>::value>::type * = nullptr>
inline auto make_LinalgLstsqSolver(void)
    -> LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                         A_Type::ROWS, B_Type::ROWS,
                         typename A_Type::SparseAvailable_Type,
                         typename B_Type::SparseAvailable_Type> {

  static_assert(std::is_same<typename A_Type::Value_Type,
                             typename B_Type::Value_Type>::value,
                "Value data type of A and B must be the same.");

  return LinalgLstsqSolver<typename A_Type::Value_Complex_Type, A_Type::COLS,
                           A_Type::ROWS, B_Type::ROWS,
                           typename A_Type::SparseAvailable_Type,
                           typename B_Type::SparseAvailable_Type>();
}

/* LinalgLstsqSolver Type */
template <typename A_Type, typename B_Type>
using LinalgLstsqSolver_Type =
    decltype(make_LinalgLstsqSolver<A_Type, B_Type>());

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_LINALG_SOLVER_HPP__
