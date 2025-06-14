/**
 * @file python_numpy_linalg_cholesky.hpp
 * @brief Provides a C++ implementation of Cholesky decomposition solvers,
 * inspired by Python's NumPy linear algebra routines.
 *
 * This header defines the `PythonNumpy` namespace, which contains the
 * `LinalgSolverCholesky` template class and related utilities for performing
 * Cholesky decomposition on dense, diagonal, and sparse matrices. The
 * implementation is designed to be type-safe, supporting both `float` and
 * `double` value types, and is intended for use in numerical and scientific
 * computing applications where efficient linear algebra operations are
 * required.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __PYTHON_NUMPY_LINALG_CHOLESKY_HPP__
#define __PYTHON_NUMPY_LINALG_CHOLESKY_HPP__

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>

namespace PythonNumpy {

constexpr double DEFAULT_DIVISION_MIN_LINALG_CHOLESKY = 1.0e-10;

/**
 * @brief Cholesky decomposition linear algebra solver class template.
 *
 * This class provides methods to perform Cholesky decomposition on matrices of
 * various formats (dense, diagonal, sparse) and solve linear systems using the
 * decomposed matrix. The decomposition is only supported for real-valued (float
 * or double) matrices and does not support complex types.
 *
 * @tparam A_Type Matrix type traits, must define Value_Type, COLS, ROWS, and
 * SparseAvailable_Type.
 *
 * @note The class supports copy and move semantics.
 *
 * @section Usage
 * - Use the `solve` method with a supported matrix type to obtain the
 * upper-triangular Cholesky factor.
 * - Use `set_division_min` to set the minimum divisor threshold for numerical
 * stability.
 * - Use `get_zero_div_flag` to check if a zero division was detected during
 * decomposition.
 */
template <typename A_Type> class LinalgSolverCholesky {
public:
  /* Type */
  using Value_Type = typename A_Type::Value_Type;
  static_assert(std::is_same<Value_Type, double>::value ||
                    std::is_same<Value_Type, float>::value,
                "Value data type must be float or double.");

  using SparseAvailable_Type = typename A_Type::SparseAvailable_Type;

  using UpperTriangular_SparseAvailable_Type =
      CreateSparseAvailableFromIndicesAndPointers<
          A_Type::COLS, UpperTriangularRowIndices<A_Type::COLS, A_Type::COLS>,
          UpperTriangularRowPointers<A_Type::COLS, A_Type::COLS>>;

protected:
  /* Type */
  using _T = typename A_Type::Value_Type;

  using _CholeskyTriangularRowIndices =
      UpperTriangularRowIndices<A_Type::COLS, A_Type::COLS>;
  using _CholeskyTriangularRowPointers =
      UpperTriangularRowPointers<A_Type::COLS, A_Type::COLS>;

public:
  /* Constructor */
  LinalgSolverCholesky()
      : _cholesky_decomposed_matrix(),
        _cholesky_decomposed_triangular(
            Base::Matrix::TriangularSparse<_T, A_Type::COLS,
                                           A_Type::COLS>::create_upper()),
        _zero_div_flag(false) {}

  /* Copy Constructor */
  LinalgSolverCholesky(const LinalgSolverCholesky<A_Type> &other)
      : division_min(other.division_min),
        _cholesky_decomposed_matrix(other._cholesky_decomposed_matrix),
        _cholesky_decomposed_triangular(other._cholesky_decomposed_triangular),
        _zero_div_flag(other._zero_div_flag) {}

  LinalgSolverCholesky<A_Type> &
  operator=(const LinalgSolverCholesky<A_Type> &other) {
    if (this != &other) {
      this->division_min = other.division_min;
      this->_cholesky_decomposed_matrix = other._cholesky_decomposed_matrix;
      this->_cholesky_decomposed_triangular =
          other._cholesky_decomposed_triangular;
      this->_zero_div_flag = other._zero_div_flag;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverCholesky(LinalgSolverCholesky<A_Type> &&other) noexcept
      : division_min(std::move(other.division_min)),
        _cholesky_decomposed_matrix(
            std::move(other._cholesky_decomposed_matrix)),
        _cholesky_decomposed_triangular(
            std::move(other._cholesky_decomposed_triangular)),
        _zero_div_flag(std::move(other._zero_div_flag)) {}

  LinalgSolverCholesky<A_Type> &
  operator=(LinalgSolverCholesky<A_Type> &&other) noexcept {
    if (this != &other) {
      this->division_min = std::move(other.division_min);
      this->_cholesky_decomposed_matrix =
          std::move(other._cholesky_decomposed_matrix);
      this->_cholesky_decomposed_triangular =
          std::move(other._cholesky_decomposed_triangular);
      this->_zero_div_flag = std::move(other._zero_div_flag);
    }
    return *this;
  }

public:
  /* Solve function */

  /**
   * @brief
   * upper triangular sparse matrix.
   *
   * This function performs Cholesky decomposition on the input dense square
   * matrix A, stores the decomposed matrix, and sets the values of the upper
   * triangular part in a sparse matrix format. The resulting upper triangular
   * sparse matrix is returned.
   *
   * @tparam _T The data type of the matrix elements (e.g., float, double).
   * @tparam A_Type::COLS The number of columns (and rows) of the square matrix
   * A.
   * @param A The input dense square matrix to decompose.
   * @return Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
   * UpperTriangular_SparseAvailable_Type> The upper triangular matrix in sparse
   * format resulting from the Cholesky decomposition.
   */
  inline auto solve(const Matrix<DefDense, _T, A_Type::COLS, A_Type::COLS> &A)
      -> Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                UpperTriangular_SparseAvailable_Type> {

    this->_cholesky_decomposed_matrix =
        Base::Matrix::cholesky_decomposition<_T, A_Type::COLS>(
            A.matrix, this->_cholesky_decomposed_matrix, this->division_min,
            this->_zero_div_flag);

    Base::Matrix::TriangularSparse<_T, A_Type::COLS, A_Type::COLS>::
        set_values_upper(this->_cholesky_decomposed_triangular,
                         this->_cholesky_decomposed_matrix);

    return Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                  UpperTriangular_SparseAvailable_Type>(
        this->_cholesky_decomposed_triangular);
  }

  /**
   * @brief
   * diagonal matrix.
   *
   * This function performs Cholesky decomposition on the input diagonal matrix
   * A, stores the decomposed matrix, and returns a diagonal matrix containing
   * the square roots of the diagonal elements of A. The resulting matrix is
   * suitable for solving linear systems where A is a diagonal matrix.
   *
   * @tparam _T The data type of the matrix elements (e.g., float, double).
   * @tparam A_Type::COLS The number of columns (and rows) of the diagonal
   * matrix A.
   * @param A The input diagonal matrix to decompose.
   * @return Matrix<DefDiag, _T, A_Type::COLS> The diagonal matrix resulting
   * from the Cholesky decomposition.
   */
  inline auto solve(const Matrix<DefDiag, _T, A_Type::COLS> &A)
      -> Matrix<DefDiag, _T, A_Type::COLS> {

    Base::Matrix::DiagMatrix<_T, A_Type::COLS> Diag(
        this->_cholesky_decomposed_matrix(0));

    Diag = Base::Matrix::cholesky_decomposition_diag<_T, A_Type::COLS>(
        A.matrix, Diag, this->_zero_div_flag);

    this->_cholesky_decomposed_matrix(0) = Diag.data;

    return Matrix<DefDiag, _T, A_Type::COLS>(Diag);
  }

  /**
   * @brief
   * sparse matrix.
   *
   * This function performs Cholesky decomposition on the input sparse square
   * matrix A, stores the decomposed matrix, and sets the values of the upper
   * triangular part in a sparse matrix format. The resulting upper triangular
   * sparse matrix is returned.
   *
   * @tparam _T The data type of the matrix elements (e.g., float, double).
   * @tparam A_Type::COLS The number of columns (and rows) of the square matrix
   * A.
   * @param A The input sparse square matrix to decompose.
   * @return Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
   * UpperTriangular_SparseAvailable_Type> The upper triangular matrix in sparse
   * format resulting from the Cholesky decomposition.
   */
  inline auto solve(const Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                                 SparseAvailable_Type> &A)
      -> Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                UpperTriangular_SparseAvailable_Type> {

    this->_cholesky_decomposed_matrix =
        Base::Matrix::cholesky_decomposition_sparse<_T, A_Type::COLS>(
            A.matrix, this->_cholesky_decomposed_matrix, this->division_min,
            this->_zero_div_flag);

    Base::Matrix::TriangularSparse<_T, A_Type::COLS, A_Type::COLS>::
        set_values_upper(this->_cholesky_decomposed_triangular,
                         this->_cholesky_decomposed_matrix);

    return Matrix<DefSparse, _T, A_Type::COLS, A_Type::COLS,
                  UpperTriangular_SparseAvailable_Type>(
        this->_cholesky_decomposed_triangular);
  }

public:
  /* Function */

  /**
   * @brief Checks if a zero division occurred during the Cholesky
   * decomposition.
   *
   * This function returns a boolean flag indicating whether a zero division
   * was detected during the decomposition process. If true, it indicates that
   * the input matrix was not positive definite.
   *
   * @return true if a zero division occurred, false otherwise.
   */
  inline bool get_zero_div_flag() const { return this->_zero_div_flag; }

  /**
   * @brief Sets the minimum division threshold for numerical stability.
   *
   * This function allows the user to set a custom minimum value for division
   * operations during the Cholesky decomposition. This can help avoid numerical
   * instability in cases where matrix elements are very small.
   *
   * @param division_min_in The minimum value to set for division operations.
   */
  inline void set_division_min(const _T &division_min_in) {
    this->division_min = division_min_in;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = A_Type::COLS;
  static constexpr std::size_t ROWS = A_Type::ROWS;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<_T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

public:
  /* Variable */
  _T division_min = static_cast<_T>(DEFAULT_DIVISION_MIN_LINALG_CHOLESKY);

protected:
  /* Variable */
  Base::Matrix::Matrix<_T, A_Type::COLS, A_Type::COLS>
      _cholesky_decomposed_matrix;

  Base::Matrix::CompiledSparseMatrix<_T, A_Type::COLS, A_Type::COLS,
                                     _CholeskyTriangularRowIndices,
                                     _CholeskyTriangularRowPointers>
      _cholesky_decomposed_triangular;

  bool _zero_div_flag;
};

/* make LinalgSolverCholesky */

/**
 * @brief Factory function to create an instance of LinalgSolverCholesky.
 *
 * This function creates and returns an instance of the LinalgSolverCholesky
 * class template for the specified matrix type A_Type.
 *
 * @tparam A_Type The matrix type traits, must define Value_Type, COLS, ROWS,
 * and SparseAvailable_Type.
 * @return An instance of LinalgSolverCholesky<A_Type>.
 */
template <typename A_Type>
inline auto make_LinalgSolverCholesky(void) -> LinalgSolverCholesky<A_Type> {

  return LinalgSolverCholesky<A_Type>();
}

/* LinalgSolverCholesky Type */
template <typename A_Type>
using LinalgSolverCholesky_Type = LinalgSolverCholesky<A_Type>;

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_LINALG_CHOLESKY_HPP__
