/**
 * @file python_numpy_linalg_lu.hpp
 * @brief LU Decomposition Solver for Python-like Numpy Linear Algebra in C++
 *
 * This file defines the `PythonNumpy` namespace, which provides a generic LU
 * decomposition solver for dense, diagonal, and sparse matrices, mimicking
 * Python's numpy.linalg functionality in C++. The main class, `LinalgSolverLU`,
 * is a template class designed to work with various matrix types, supporting
 * both single and double precision floating point values. The solver provides
 * methods for factorization, solving, and extracting L and U matrices, as well
 * as determinant calculation.
 *
 * @note
 * tparam M is the number of rows in the matrix.
 * tparam N is the number of columns in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef PYTHON_NUMPY_LINALG_LU_HPP_
#define PYTHON_NUMPY_LINALG_LU_HPP_

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace PythonNumpy {

const double DEFAULT_DIVISION_MIN_LINALG_LU = 1.0e-10;

/**
 * @brief LinalgSolverLU class for LU decomposition and solving linear systems.
 *
 * This class provides methods to perform LU decomposition on matrices and solve
 * linear systems using the decomposed matrices. It supports dense, diagonal,
 * and sparse matrices, and can handle both single and double precision floating
 * point values.
 *
 * @tparam A_Type The type of the matrix (dense, diagonal, or sparse).
 */
template <typename A_Type> class LinalgSolverLU {
public:
  /* Type */
  using Value_Type = typename A_Type::Value_Type;
  static_assert(std::is_same<Value_Type, double>::value ||
                    std::is_same<Value_Type, float>::value,
                "Value data type must be float or double.");

  using SparseAvailable_Type = typename A_Type::SparseAvailable_Type;

  using UpperTriangular_SparseAvailable_Type =
      CreateSparseAvailableFromIndicesAndPointers<
          A_Type::ROWS, UpperTriangularCSRIndices<A_Type::ROWS, A_Type::ROWS>,
          UpperTriangularCSRPointers<A_Type::ROWS, A_Type::ROWS>>;

  using LowerTriangular_SparseAvailable_Type =
      CreateSparseAvailableFromIndicesAndPointers<
          A_Type::ROWS, LowerTriangularCSRIndices<A_Type::ROWS, A_Type::ROWS>,
          LowerTriangularCSRPointers<A_Type::ROWS, A_Type::ROWS>>;

protected:
  /* Type */
  using T_ = typename A_Type::Value_Type;

  using L_CSRIndices_ =
      Base::Matrix::LowerTriangularCSRIndices<A_Type::ROWS, A_Type::ROWS>;
  using L_CSRPointers_ =
      Base::Matrix::LowerTriangularCSRPointers<A_Type::ROWS, A_Type::ROWS>;

  using U_CSRIndices_ =
      Base::Matrix::UpperTriangularCSRIndices<A_Type::ROWS, A_Type::ROWS>;
  using U_CSRPointers_ =
      Base::Matrix::UpperTriangularCSRPointers<A_Type::ROWS, A_Type::ROWS>;

public:
  /* Constructor */
  template <
      typename U = A_Type,
      typename std::enable_if<Is_Dense_Matrix<U>::value>::type * = nullptr>
  LinalgSolverLU()
      : LU_decomposer_(),
        L_triangular_(
            Base::Matrix::create_LowerTriangularSparseMatrix<T_, A_Type::ROWS,
                                                             A_Type::ROWS>()),
        U_triangular_(
            Base::Matrix::create_UpperTriangularSparseMatrix<T_, A_Type::ROWS,
                                                             A_Type::ROWS>()) {
    this->LU_decomposer_ = Base::Matrix::LUDecomposition<T_, A_Type::ROWS>();
  }

  template <typename U = A_Type,
            typename std::enable_if<Is_Diag_Matrix<U>::value>::type * = nullptr>
  LinalgSolverLU()
      : LU_decomposer_(),
        L_triangular_(
            Base::Matrix::create_LowerTriangularSparseMatrix<T_, A_Type::ROWS,
                                                             A_Type::ROWS>()),
        U_triangular_(
            Base::Matrix::create_UpperTriangularSparseMatrix<T_, A_Type::ROWS,
                                                             A_Type::ROWS>()) {}

  template <
      typename U = A_Type,
      typename std::enable_if<Is_Sparse_Matrix<U>::value>::type * = nullptr>
  LinalgSolverLU()
      : LU_decomposer_(),
        L_triangular_(
            Base::Matrix::create_LowerTriangularSparseMatrix<T_, A_Type::ROWS,
                                                             A_Type::ROWS>()),
        U_triangular_(
            Base::Matrix::create_UpperTriangularSparseMatrix<T_, A_Type::ROWS,
                                                             A_Type::ROWS>()) {
    this->LU_decomposer_ = Base::Matrix::LUDecomposition<T_, A_Type::ROWS>();
  }

  /* Copy Constructor */
  LinalgSolverLU(const LinalgSolverLU<A_Type> &other)
      : LU_decomposer_(other.LU_decomposer_),
        L_triangular_(other.L_triangular_), U_triangular_(other.U_triangular_) {
  }

  LinalgSolverLU<A_Type> &operator=(const LinalgSolverLU<A_Type> &other) {
    if (this != &other) {
      this->LU_decomposer_ = other.LU_decomposer_;
      this->L_triangular_ = other.L_triangular_;
      this->U_triangular_ = other.U_triangular_;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverLU(LinalgSolverLU<A_Type> &&other) noexcept
      : LU_decomposer_(std::move(other.LU_decomposer_)),
        L_triangular_(std::move(other.L_triangular_)),
        U_triangular_(std::move(other.U_triangular_)) {}

  LinalgSolverLU<A_Type> &operator=(LinalgSolverLU<A_Type> &&other) noexcept {
    if (this != &other) {

      this->LU_decomposer_ = std::move(other.LU_decomposer_);
      this->L_triangular_ = std::move(other.L_triangular_);
      this->U_triangular_ = std::move(other.U_triangular_);
    }
    return *this;
  }

public:
  /* Solve function */

  /**
   * @brief Performs LU decomposition on the given matrix and prepares the
   * solver for solving linear systems.
   *
   * This function initializes the LU decomposition of the input matrix A,
   * allowing subsequent calls to `solve` to efficiently solve linear systems
   * involving A.
   *
   * @param A The input matrix for LU decomposition.
   */
  inline void solve(const Matrix<DefDense, T_, A_Type::ROWS, A_Type::ROWS> &A) {
    this->LU_decomposer_.solve(A.matrix);
  }

  /**
   * @brief Performs LU decomposition on the given diagonal matrix and prepares
   * the solver for solving linear systems.
   *
   * This function initializes the LU decomposition of the input diagonal matrix
   * A, allowing subsequent calls to `solve` to efficiently solve linear systems
   * involving A.
   *
   * @param A The input diagonal matrix for LU decomposition.
   */
  inline void solve(const Matrix<DefDiag, T_, A_Type::ROWS> &A) {
    this->LU_decomposer_ =
        Base::Matrix::LUDecomposition<T_, A_Type::ROWS>(A.matrix);
  }

  /**
   * @brief Performs LU decomposition on the given sparse matrix and prepares
   * the solver for solving linear systems.
   *
   * This function initializes the LU decomposition of the input sparse matrix
   * A, allowing subsequent calls to `solve` to efficiently solve linear systems
   * involving A.
   *
   * @param A The input sparse matrix for LU decomposition.
   */
  inline void solve(const Matrix<DefSparse, T_, A_Type::ROWS, A_Type::ROWS,
                                 SparseAvailable_Type> &A) {

    auto A_dense = A.matrix.create_dense();
    this->LU_decomposer_.solve(A_dense);
  }

  /* Get */

  /**
   * @brief Returns the lower triangular matrix (L) and upper triangular matrix
   * (U) from the LU decomposition.
   *
   * This function retrieves the L and U matrices from the LU decomposition
   * performed on the input matrix A. The matrices are returned as sparse
   * matrices if A is sparse, or as dense matrices otherwise.
   *
   * @return A pair containing the lower triangular matrix L and upper
   * triangular matrix U.
   */
  inline auto get_L() -> Matrix<DefSparse, T_, A_Type::ROWS, A_Type::ROWS,
                                LowerTriangular_SparseAvailable_Type> const {

    Base::Matrix::set_values_LowerTriangularSparseMatrix<T_, A_Type::ROWS,
                                                         A_Type::ROWS>(
        this->L_triangular_, this->LU_decomposer_.get_L());

    return Matrix<DefSparse, T_, A_Type::ROWS, A_Type::ROWS,
                  LowerTriangular_SparseAvailable_Type>(this->L_triangular_);
  }

  /**
   * @brief Returns the upper triangular matrix (U) from the LU decomposition.
   *
   * This function retrieves the U matrix from the LU decomposition performed on
   * the input matrix A. The matrix is returned as a sparse matrix if A is
   * sparse, or as a dense matrix otherwise.
   *
   * @return The upper triangular matrix U.
   */
  inline auto get_U() -> Matrix<DefSparse, T_, A_Type::ROWS, A_Type::ROWS,
                                UpperTriangular_SparseAvailable_Type> const {

    Base::Matrix::set_values_UpperTriangularSparseMatrix<T_, A_Type::ROWS,
                                                         A_Type::ROWS>(
        this->U_triangular_, this->LU_decomposer_.get_U());

    return Matrix<DefSparse, T_, A_Type::ROWS, A_Type::ROWS,
                  UpperTriangular_SparseAvailable_Type>(this->U_triangular_);
  }

  /**
   * @brief Solves the linear system Ax = b using the LU decomposition.
   *
   * This function solves the linear system represented by the matrix A and
   * vector b using the LU decomposition. It returns the solution vector x.
   *
   * @param b The right-hand side vector of the linear system.
   * @return The solution vector x such that Ax = b.
   */
  inline T_ get_det() { return this->LU_decomposer_.get_determinant(); }

  /* Set */

  /**
   * @brief Sets the minimum value for division in the LU decomposition.
   *
   * This function allows the user to specify a minimum value for division
   * during the LU decomposition process. It can help avoid numerical issues
   * when dealing with very small values.
   *
   * @param division_min The minimum value for division.
   */
  inline void set_division_min(const T_ &division_min) {
    this->LU_decomposer_.division_min = division_min;
  }

public:
  /* Constant */
  static constexpr std::size_t ROWS = A_Type::ROWS;
  static constexpr std::size_t COLS = A_Type::COLS;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T_>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

protected:
  /* Variable */
  Base::Matrix::LUDecomposition<T_, A_Type::ROWS> LU_decomposer_;

  Base::Matrix::CompiledSparseMatrix<T_, A_Type::ROWS, A_Type::ROWS,
                                     L_CSRIndices_, L_CSRPointers_>
      L_triangular_;

  Base::Matrix::CompiledSparseMatrix<T_, A_Type::ROWS, A_Type::ROWS,
                                     U_CSRIndices_, U_CSRPointers_>
      U_triangular_;
};

/* make LinalgSolverLU */

/**
 * @brief Creates an instance of LinalgSolverLU for the specified matrix type.
 *
 * This function is a factory function that creates and returns an instance of
 * LinalgSolverLU for the given matrix type A_Type. It is used to simplify the
 * creation of the solver object.
 *
 * @tparam A_Type The type of the matrix (dense, diagonal, or sparse).
 * @return An instance of LinalgSolverLU for the specified matrix type.
 */
template <typename A_Type>
inline auto make_LinalgSolverLU(void) -> LinalgSolverLU<A_Type> {

  return LinalgSolverLU<A_Type>();
}

/* LinalgSolverLU Type */
template <typename A_Type> using LinalgSolverLU_Type = LinalgSolverLU<A_Type>;

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_LINALG_LU_HPP_
