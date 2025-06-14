/**
 * @file python_numpy_linalg_eig.hpp
 * @brief Linear algebra solvers for eigenvalues and eigenvectors, supporting
 * dense, diagonal, and sparse matrices with real and complex types.
 *
 * This header provides template classes and utility functions for computing
 * eigenvalues and eigenvectors of matrices in various storage formats (dense,
 * diagonal, sparse) and for both real and complex number types. It is designed
 * to be used as part of a Python/Numpy-like C++ numerical library.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __PYTHON_NUMPY_LINALG_EIG_HPP__
#define __PYTHON_NUMPY_LINALG_EIG_HPP__

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>

namespace PythonNumpy {

namespace ForLinalgSolverEigReal {

template <typename T, std::size_t M>
using EigenValues_Type = Matrix<DefDense, T, M, 1>;

template <typename T, std::size_t M>
using EigenVectors_Type = Matrix<DefDense, T, M, M>;

} // namespace ForLinalgSolverEigReal

/* Linalg solver for Real Eigen values and vectors of Dense Matrix */

/** * @brief Linalg solver for Real Eigen values and vectors of Dense Matrix.
 *
 * This class provides methods to compute eigenvalues and eigenvectors of dense
 * matrices with real number types (float or double). It uses the Eigen library
 * for the underlying computations.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam M The number of rows and columns in the square matrix.
 */
template <typename T, std::size_t M> class LinalgSolverEigRealDense {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using A_Type = Matrix<DefDense, T, M, M>;
  using EigenSolver_Type = Base::Matrix::EigenSolverReal<T, M>;

public:
  /* Constructor */
  LinalgSolverEigRealDense() : _Eigen_solver() {}

  /* Copy Constructor */
  LinalgSolverEigRealDense(const LinalgSolverEigRealDense<T, M> &other)
      : _Eigen_solver(other._Eigen_solver) {}

  LinalgSolverEigRealDense<T, M> &
  operator=(const LinalgSolverEigRealDense<T, M> &other) {
    if (this != &other) {
      this->_Eigen_solver = other._Eigen_solver;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigRealDense(LinalgSolverEigRealDense<T, M> &&other) noexcept
      : _Eigen_solver(std::move(other._Eigen_solver)) {}

  LinalgSolverEigRealDense<T, M> &
  operator=(LinalgSolverEigRealDense<T, M> &&other) noexcept {
    if (this != &other) {
      this->_Eigen_solver = std::move(other._Eigen_solver);
    }
    return *this;
  }

public:
  /* Solve function */

  /**
   * @brief Computes the eigenvalues of the given matrix.
   *
   * This function uses the internal Eigen solver to compute the eigenvalues
   * of the matrix contained in the input parameter A. The results are stored
   * within the solver for later retrieval.
   *
   * @tparam A_Type Type of the input matrix wrapper.
   * @param A An object containing the matrix for which eigenvalues are to be
   * computed.
   */
  inline void solve_eigen_values(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_values(A.matrix);
  }

  /**
   * @brief Continues solving for eigenvalues if the solver supports it.
   *
   * This function allows the solver to continue computing eigenvalues if it
   * has not yet completed the process. It is useful for iterative solvers.
   */
  inline void continue_solving_eigen_values(void) {
    this->_Eigen_solver.continue_solving_eigen_values();
  }

  /**
   * @brief Computes the eigenvectors of the given matrix.
   *
   * This function uses the internal Eigen solver to compute the eigenvectors
   * of the matrix contained in the input parameter A. The results are stored
   * within the solver for later retrieval.
   *
   * @tparam A_Type Type of the input matrix wrapper.
   * @param A An object containing the matrix for which eigenvectors are to be
   * computed.
   */
  inline void solve_eigen_vectors(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_vectors(A.matrix);
  }

  /* Get */

  /**
   * @brief Retrieves the computed eigenvalues.
   *
   * This function returns the eigenvalues computed by the solver as a
   * ForLinalgSolverEigReal::EigenValues_Type object.
   *
   * @return A ForLinalgSolverEigReal::EigenValues_Type object containing the
   * eigenvalues.
   */
  inline auto get_eigen_values(void)
      -> ForLinalgSolverEigReal::EigenValues_Type<T, M> {
    return ForLinalgSolverEigReal::EigenValues_Type<T, M>(
        Base::Matrix::Matrix<T, M, 1>(this->_Eigen_solver.get_eigen_values()));
  }

  /**
   * @brief Retrieves the computed eigenvectors.
   *
   * This function returns the eigenvectors computed by the solver as a
   * ForLinalgSolverEigReal::EigenVectors_Type object.
   *
   * @return A ForLinalgSolverEigReal::EigenVectors_Type object containing the
   * eigenvectors.
   */
  inline auto get_eigen_vectors(void)
      -> ForLinalgSolverEigReal::EigenVectors_Type<T, M> {
    return ForLinalgSolverEigReal::EigenVectors_Type<T, M>(
        this->_Eigen_solver.get_eigen_vectors());
  }

  /**
   * @brief Retrieves the maximum number of iterations allowed for the solver.
   *
   * This function returns the maximum number of iterations that the solver is
   * allowed to perform when computing eigenvalues and eigenvectors.
   *
   * @return The maximum number of iterations.
   */
  inline std::size_t get_iteration_max(void) {
    return this->_Eigen_solver.iteration_max;
  }

  /* Set */

  /**
   * @brief Sets the maximum number of iterations for the solver.
   *
   * This function allows the user to specify the maximum number of iterations
   * that the solver should perform when computing eigenvalues and
   * eigenvectors.
   *
   * @param iteration_max The maximum number of iterations to set.
   */
  inline void set_iteration_max(std::size_t iteration_max) {
    this->_Eigen_solver.iteration_max = iteration_max;
  }

  /**
   * @brief Sets the minimum division value for the Eigen solver.
   *
   * This function assigns the specified value to the `division_min` parameter
   * of the internal Eigen solver. The `division_min` parameter is typically
   * used to avoid division by very small numbers, which can lead to numerical
   * instability.
   *
   * @tparam T The type of the division minimum value.
   * @param division_min_in The minimum value to be used for divisions in the
   * Eigen solver.
   */
  inline void set_division_min(const T &division_min_in) {
    this->_Eigen_solver.division_min = division_min_in;
  }

  /**
   * @brief Sets a small value threshold for numerical stability.
   *
   * This function allows the user to specify a small value that can be used
   * to avoid numerical instability during computations, such as division by
   * very small numbers.
   *
   * @param small_value_in The small value to set for numerical stability.
   */
  inline void set_small_value(const T &small_value_in) {
    this->_Eigen_solver.small_value = small_value_in;
  }

  /* Check */

  /**
   * @brief Checks the validity of the eigenvalues and eigenvectors for the
   * given matrix.
   *
   * This function checks whether the computed eigenvalues and eigenvectors
   * are valid for the provided matrix A. It returns a
   * ForLinalgSolverEigReal::EigenVectors_Type object containing the results.
   *
   * @param A The input matrix for which validity is to be checked.
   * @return A ForLinalgSolverEigReal::EigenVectors_Type object containing the
   * validity check results.
   */
  inline auto check_validity(const A_Type &A)
      -> ForLinalgSolverEigReal::EigenVectors_Type<T, M> {

    return ForLinalgSolverEigReal::EigenVectors_Type<T, M>(
        this->_Eigen_solver.check_validity(A.matrix));
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

protected:
  /* Variable */
  EigenSolver_Type _Eigen_solver;
};

/* Linalg solver for Real Eigen values and vectors of Diag Matrix */

/**
 * @brief Linalg solver for Real Eigen values and vectors of Diag Matrix.
 *
 * This class provides methods to compute eigenvalues and eigenvectors of
 * diagonal matrices with real number types (float or double). It is designed
 * to work with matrices that have a diagonal structure.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam M The number of rows and columns in the square diagonal matrix.
 */
template <typename T, std::size_t M> class LinalgSolverEigRealDiag {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using A_Type = Matrix<DefDiag, T, M>;
  using EigenValues_Type = Matrix<DefDense, T, M, 1>;
  using EigenVectors_Type = Matrix<DefDiag, T, M>;

public:
  /* Constructor */
  LinalgSolverEigRealDiag() : _eigen_values() {}

  /* Copy Constructor */
  LinalgSolverEigRealDiag(const LinalgSolverEigRealDiag<T, M> &other)
      : _eigen_values(other._eigen_values) {}

  LinalgSolverEigRealDiag<T, M> &
  operator=(const LinalgSolverEigRealDiag<T, M> &other) {
    if (this != &other) {
      this->_eigen_values = other._eigen_values;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigRealDiag(LinalgSolverEigRealDiag<T, M> &&other) noexcept
      : _eigen_values(std::move(other._eigen_values)) {}

  LinalgSolverEigRealDiag<T, M> &
  operator=(LinalgSolverEigRealDiag<T, M> &&other) noexcept {
    if (this != &other) {
      this->_eigen_values = std::move(other._eigen_values);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Computes the eigenvalues of the given diagonal matrix.
   *
   * This function extracts the diagonal elements of the matrix A and stores
   * them as eigenvalues in the internal _eigen_values member variable.
   *
   * @param A The input diagonal matrix for which eigenvalues are to be
   * computed.
   */
  inline void solve_eigen_values(const A_Type &A) {
    this->_eigen_values = Base::Matrix::Matrix<T, M, 1>(A.matrix.data);
  }

  /**
   * @brief Continues solving for eigenvalues if the solver supports it.
   *
   * This function is a placeholder for compatibility with the interface but
   * does not perform any action since eigenvalues of diagonal matrices are
   * directly available.
   */
  inline void solve_eigen_vectors(const A_Type &A) {
    this->_eigen_values = Base::Matrix::Matrix<T, M, 1>(A.matrix.data);
  }

  /* Get */

  /**
   * @brief Retrieves the computed eigenvalues.
   *
   * This function returns the eigenvalues computed by the solver as an
   * EigenValues_Type object.
   *
   * @return An EigenValues_Type object containing the eigenvalues.
   */
  inline auto get_eigen_values(void) -> EigenValues_Type {
    return EigenValues_Type(this->_eigen_values);
  }

  /**
   * @brief Retrieves the eigenvectors of the diagonal matrix.
   *
   * Since the eigenvectors of a diagonal matrix are simply the identity
   * matrix, this function returns an EigenVectors_Type object representing
   * the identity matrix.
   *
   * @return An EigenVectors_Type object representing the identity matrix.
   */
  inline auto get_eigen_vectors(void) -> EigenVectors_Type {
    return EigenVectors_Type(Base::Matrix::DiagMatrix<T, M>::identity());
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

protected:
  /* Variable */
  Base::Matrix::Matrix<T, M, 1> _eigen_values;
};

/* Linalg solver for Real Eigen values and vectors of Sparse Matrix */

/** @brief Linalg solver for Real Eigen values and vectors of Sparse Matrix.
 *
 * This class provides methods to compute eigenvalues and eigenvectors of
 * sparse matrices with real number types (float or double). It is designed to
 * work with matrices that have a sparse structure, utilizing the Eigen library
 * for efficient computations.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam M The number of rows and columns in the square sparse matrix.
 * @tparam SparseAvailable A type indicating whether sparse operations are
 * available.
 */
template <typename T, std::size_t M, typename SparseAvailable>
class LinalgSolverEigRealSparse {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using A_Type = Matrix<DefSparse, T, M, M, SparseAvailable>;
  using EigenSolver_Type = Base::Matrix::EigenSolverReal<T, M>;

public:
  /* Constructor */
  LinalgSolverEigRealSparse() : _Eigen_solver() {}

  /* Copy Constructor */
  LinalgSolverEigRealSparse(
      const LinalgSolverEigRealSparse<T, M, SparseAvailable> &other)
      : _Eigen_solver(other._Eigen_solver) {}

  LinalgSolverEigRealSparse<T, M, SparseAvailable> &
  operator=(const LinalgSolverEigRealSparse<T, M, SparseAvailable> &other) {
    if (this != &other) {
      this->_Eigen_solver = other._Eigen_solver;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigRealSparse(
      LinalgSolverEigRealSparse<T, M, SparseAvailable> &&other) noexcept
      : _Eigen_solver(std::move(other._Eigen_solver)) {}

  LinalgSolverEigRealSparse<T, M, SparseAvailable> &
  operator=(LinalgSolverEigRealSparse<T, M, SparseAvailable> &&other) noexcept {
    if (this != &other) {
      this->_Eigen_solver = std::move(other._Eigen_solver);
    }
    return *this;
  }

  /* Solve method */

  /**
   * @brief Computes the eigenvalues of the given sparse matrix.
   *
   * This function uses the internal Eigen solver to compute the eigenvalues
   * of the sparse matrix contained in the input parameter A. The results are
   * stored within the solver for later retrieval.
   *
   * @param A An object containing the sparse matrix for which eigenvalues are
   * to be computed.
   */
  inline void solve_eigen_values(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_values(
        Base::Matrix::output_dense_matrix(A.matrix));
  }

  /**
   * @brief Continues solving for eigenvalues if the solver supports it.
   *
   * This function allows the solver to continue computing eigenvalues if it
   * has not yet completed the process. It is useful for iterative solvers.
   */
  inline void continue_solving_eigen_values(void) {
    this->_Eigen_solver.continue_solving_eigen_values();
  }

  /**
   * @brief Computes the eigenvectors of the given sparse matrix.
   *
   * This function uses the internal Eigen solver to compute the eigenvectors
   * of the sparse matrix contained in the input parameter A. The results are
   * stored within the solver for later retrieval.
   *
   * @param A An object containing the sparse matrix for which eigenvectors are
   * to be computed.
   */
  inline void solve_eigen_vectors(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_vectors(
        Base::Matrix::output_dense_matrix(A.matrix));
  }

  /* Get */

  /**
   * @brief Retrieves the computed eigenvalues.
   *
   * This function returns the eigenvalues computed by the solver as a
   * ForLinalgSolverEigReal::EigenValues_Type object.
   *
   * @return A ForLinalgSolverEigReal::EigenValues_Type object containing the
   * eigenvalues.
   */
  inline auto get_eigen_values(void)
      -> ForLinalgSolverEigReal::EigenValues_Type<T, M> {
    return ForLinalgSolverEigReal::EigenValues_Type<T, M>(
        Base::Matrix::Matrix<T, M, 1>(this->_Eigen_solver.get_eigen_values()));
  }

  /**
   * @brief Retrieves the computed eigenvectors.
   *
   * This function returns the eigenvectors computed by the solver as a
   * ForLinalgSolverEigReal::EigenVectors_Type object.
   *
   * @return A ForLinalgSolverEigReal::EigenVectors_Type object containing the
   * eigenvectors.
   */
  inline auto get_eigen_vectors(void)
      -> ForLinalgSolverEigReal::EigenVectors_Type<T, M> {
    return ForLinalgSolverEigReal::EigenVectors_Type<T, M>(
        this->_Eigen_solver.get_eigen_vectors());
  }

  /* Set */

  /**
   * @brief Sets the maximum number of iterations for the solver.
   *
   * This function allows the user to specify the maximum number of iterations
   * that the solver should perform when computing eigenvalues and
   * eigenvectors.
   *
   * @param iteration_max The maximum number of iterations to set.
   */
  inline void set_iteration_max(std::size_t iteration_max) {
    this->_Eigen_solver.iteration_max = iteration_max;
  }

  /**
   * @brief Sets the minimum division value for the Eigen solver.
   *
   * This function assigns the specified value to the `division_min` parameter
   * of the internal Eigen solver. The `division_min` parameter is typically
   * used to avoid division by very small numbers, which can lead to numerical
   * instability.
   *
   * @param division_min_in The minimum value to be used for divisions in the
   * Eigen solver.
   */
  inline void set_division_min(const T &division_min_in) {
    this->_Eigen_solver.division_min = division_min_in;
  }

  /**
   * @brief Sets a small value threshold for numerical stability.
   *
   * This function allows the user to specify a small value that can be used
   * to avoid numerical instability during computations, such as division by
   * very small numbers.
   *
   * @param small_value_in The small value to set for numerical stability.
   */
  inline void set_small_value(const T &small_value_in) {
    this->_Eigen_solver.small_value = small_value_in;
  }

  /* Check */

  /**
   * @brief Checks the validity of the eigenvalues and eigenvectors for the
   * given sparse matrix.
   *
   * This function checks whether the computed eigenvalues and eigenvectors
   * are valid for the provided matrix A. It returns a
   * ForLinalgSolverEigReal::EigenVectors_Type object containing the results.
   *
   * @param A The input sparse matrix for which validity is to be checked.
   * @return A ForLinalgSolverEigReal::EigenVectors_Type object containing the
   * validity check results.
   */
  inline auto check_validity(const A_Type &A)
      -> ForLinalgSolverEigReal::EigenVectors_Type<T, M> {

    return ForLinalgSolverEigReal::EigenVectors_Type<T, M>(
        this->_Eigen_solver.check_validity(
            Base::Matrix::output_dense_matrix(A.matrix)));
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

protected:
  /* Variable */
  Base::Matrix::EigenSolverReal<T, M> _Eigen_solver;
};

/* make LinalgSolverEig Real */

/**
 * @brief Factory function to create a LinalgSolverEigReal instance based on
 * the type of matrix provided.
 *
 * This function uses SFINAE (Substitution Failure Is Not An Error) to determine
 * the type of matrix and return the appropriate LinalgSolverEigReal instance.
 *
 * @tparam A_Type The type of the matrix for which the solver is to be created.
 * @return A LinalgSolverEigReal instance suitable for the given matrix type.
 */
template <
    typename A_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverEigReal()
    -> LinalgSolverEigRealDense<typename A_Type::Value_Complex_Type,
                                A_Type::COLS> {

  return LinalgSolverEigRealDense<typename A_Type::Value_Complex_Type,
                                  A_Type::COLS>();
}

/** @brief Factory function to create a LinalgSolverEigReal instance for
 * diagonal matrices.
 *
 * This function is specialized for diagonal matrices and returns a
 * LinalgSolverEigRealDiag instance.
 *
 * @tparam A_Type The type of the diagonal matrix for which the solver is to be
 * created.
 * @return A LinalgSolverEigRealDiag instance suitable for the given diagonal
 * matrix type.
 */
template <typename A_Type, typename std::enable_if<
                               Is_Diag_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverEigReal()
    -> LinalgSolverEigRealDiag<typename A_Type::Value_Complex_Type,
                               A_Type::COLS> {

  return LinalgSolverEigRealDiag<typename A_Type::Value_Complex_Type,
                                 A_Type::COLS>();
}

/** @brief Factory function to create a LinalgSolverEigReal instance for
 * sparse matrices.
 *
 * This function is specialized for sparse matrices and returns a
 * LinalgSolverEigRealSparse instance.
 *
 * @tparam A_Type The type of the sparse matrix for which the solver is to be
 * created.
 * @return A LinalgSolverEigRealSparse instance suitable for the given sparse
 * matrix type.
 */
template <
    typename A_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverEigReal()
    -> LinalgSolverEigRealSparse<typename A_Type::Value_Complex_Type,
                                 A_Type::COLS,
                                 typename A_Type::SparseAvailable_Type> {

  return LinalgSolverEigRealSparse<typename A_Type::Value_Complex_Type,
                                   A_Type::COLS,
                                   typename A_Type::SparseAvailable_Type>();
}

/* LinalgSolverEig Real Type */
template <typename A_Type>
using LinalgSolverEigReal_Type = decltype(make_LinalgSolverEigReal<A_Type>());

namespace ForLinalgSolverEig {

template <typename T, std::size_t M>
using EigenValues_Type = Matrix<DefDense, Base::Matrix::Complex<T>, M, 1>;

template <typename T, std::size_t M>
using EigenVectors_Type = Matrix<DefDense, Base::Matrix::Complex<T>, M, M>;

} // namespace ForLinalgSolverEig

/* Linalg solver for Complex Eigen values and vectors of Dense and Sparse Matrix
 */

/** @brief Linalg solver for Complex Eigen values and vectors of Dense Matrix.
 * This class provides methods to compute eigenvalues and eigenvectors of dense
 * matrices with complex number types (float or double). It uses the Eigen
 * library
 * for the underlying computations.
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam M The number of rows and columns in the square matrix.
 */
template <typename T, std::size_t M> class LinalgSolverEigDense {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using A_Type = Matrix<DefDense, T, M, M>;
  using EigenSolver_Type = Base::Matrix::EigenSolverComplex<T, M>;

public:
  /* Constructor */
  LinalgSolverEigDense() : _Eigen_solver() {}

  /* Copy Constructor */
  LinalgSolverEigDense(const LinalgSolverEigDense<T, M> &other)
      : _Eigen_solver(other._Eigen_solver) {}

  LinalgSolverEigDense<T, M> &
  operator=(const LinalgSolverEigDense<T, M> &other) {
    if (this != &other) {
      this->_Eigen_solver = other._Eigen_solver;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigDense(LinalgSolverEigDense<T, M> &&other) noexcept
      : _Eigen_solver(std::move(other._Eigen_solver)) {}

  LinalgSolverEigDense<T, M> &
  operator=(LinalgSolverEigDense<T, M> &&other) noexcept {
    if (this != &other) {
      this->_Eigen_solver = std::move(other._Eigen_solver);
    }
    return *this;
  }

  /* Solve method */

  /**
   * @brief Computes the eigenvalues of the given matrix.
   *
   * This function uses the internal Eigen solver to compute the eigenvalues
   * of the matrix contained in the input parameter A. The results are stored
   * within the solver for later retrieval.
   *
   * @param A An object containing the matrix for which eigenvalues are to be
   * computed.
   */
  inline void solve_eigen_values(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_values(A.matrix);
  }

  /**
   * @brief Continues solving for eigenvalues if the solver supports it.
   *
   * This function allows the solver to continue computing eigenvalues if it
   * has not yet completed the process. It is useful for iterative solvers.
   */
  inline void continue_solving_eigen_values(void) {
    this->_Eigen_solver.continue_solving_eigen_values();
  }

  /**
   * @brief Computes the eigenvectors of the given matrix.
   *
   * This function uses the internal Eigen solver to compute the eigenvectors
   * of the matrix contained in the input parameter A. The results are stored
   * within the solver for later retrieval.
   *
   * @param A An object containing the matrix for which eigenvectors are to be
   * computed.
   */
  inline void solve_eigen_vectors(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_vectors(A.matrix);
  }

  /* Get */

  /**
   * @brief Retrieves the computed eigenvalues.
   *
   * This function returns the eigenvalues computed by the solver as a
   * ForLinalgSolverEig::EigenValues_Type object.
   *
   * @return A ForLinalgSolverEig::EigenValues_Type object containing the
   * eigenvalues.
   */
  inline auto get_eigen_values(void)
      -> ForLinalgSolverEig::EigenValues_Type<T, M> {
    return ForLinalgSolverEig::EigenValues_Type<T, M>(
        Base::Matrix::Matrix<Base::Matrix::Complex<T>, M, 1>(
            this->_Eigen_solver.get_eigen_values()));
  }

  /**
   * @brief Retrieves the computed eigenvectors.
   *
   * This function returns the eigenvectors computed by the solver as a
   * ForLinalgSolverEig::EigenVectors_Type object.
   *
   * @return A ForLinalgSolverEig::EigenVectors_Type object containing the
   * eigenvectors.
   */
  inline auto get_eigen_vectors(void)
      -> ForLinalgSolverEig::EigenVectors_Type<T, M> {
    return ForLinalgSolverEig::EigenVectors_Type<T, M>(
        this->_Eigen_solver.get_eigen_vectors());
  }

  /**
   * @brief Retrieves the maximum number of iterations allowed for the solver.
   *
   * This function returns the maximum number of iterations that the solver is
   * allowed to perform when computing eigenvalues and eigenvectors.
   *
   * @return The maximum number of iterations.
   */
  inline std::size_t get_iteration_max(void) {
    return this->_Eigen_solver.iteration_max;
  }

  /* Set */

  /**
   * @brief Sets the maximum number of iterations for the solver.
   *
   * This function allows the user to specify the maximum number of iterations
   * that the solver should perform when computing eigenvalues and
   * eigenvectors.
   *
   * @param iteration_max The maximum number of iterations to set.
   */
  inline void set_iteration_max(std::size_t iteration_max) {
    this->_Eigen_solver.iteration_max = iteration_max;
  }

  /**
   * @brief Sets the maximum number of iterations for eigenvector computation.
   *
   * This function allows the user to specify the maximum number of iterations
   * that the solver should perform when computing eigenvectors.
   *
   * @param iteration_max_for_eigen_vector The maximum number of iterations to
   * set for eigenvector computation.
   */
  inline void set_iteration_max_for_eigen_vector(
      std::size_t iteration_max_for_eigen_vector) {
    this->_Eigen_solver.iteration_max_for_eigen_vector =
        iteration_max_for_eigen_vector;
  }

  /**
   * @brief Sets the minimum division value for the Eigen solver.
   *
   * This function assigns the specified value to the `division_min` parameter
   * of the internal Eigen solver. The `division_min` parameter is typically
   * used to avoid division by very small numbers, which can lead to numerical
   * instability.
   *
   * @param division_min_in The minimum value to be used for divisions in the
   * Eigen solver.
   */
  inline void set_division_min(const T &division_min_in) {
    this->_Eigen_solver.division_min = division_min_in;
  }

  /**
   * @brief Sets a small value threshold for numerical stability.
   *
   * This function allows the user to specify a small value that can be used
   * to avoid numerical instability during computations, such as division by
   * very small numbers.
   *
   * @param small_value_in The small value to set for numerical stability.
   */
  inline void set_small_value(const T &small_value_in) {
    this->_Eigen_solver.small_value = small_value_in;
  }

  /**
   * @brief Sets the decay rate for the GMRES method.
   *
   * This function allows the user to specify the decay rate for the GMRES
   * method used in the Eigen solver. The decay rate can affect the convergence
   * behavior of the solver.
   *
   * @param gmres_k_decay_rate_in The decay rate to set for the GMRES method.
   */
  inline void set_gmres_k_decay_rate(const T &gmres_k_decay_rate_in) {
    this->_Eigen_solver.gmres_k_decay_rate = gmres_k_decay_rate_in;
  }

  /* Check */

  /**
   * @brief Checks the validity of the eigenvalues and eigenvectors for the
   * given matrix.
   *
   * This function checks whether the computed eigenvalues and eigenvectors
   * are valid for the provided matrix A. It returns a
   * ForLinalgSolverEig::EigenVectors_Type object containing the results.
   *
   * @param A The input matrix for which validity is to be checked.
   * @return A ForLinalgSolverEig::EigenVectors_Type object containing the
   * validity check results.
   */
  inline auto check_validity(const A_Type &A)
      -> ForLinalgSolverEig::EigenVectors_Type<T, M> {

    return ForLinalgSolverEig::EigenVectors_Type<T, M>(
        this->_Eigen_solver.check_validity(A.matrix));
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

protected:
  /* Variable */
  EigenSolver_Type _Eigen_solver;
};

/* Linalg solver for Complex Eigen values and vectors of Diag Matrix */

/** @brief Linalg solver for Complex Eigen values and vectors of Diag Matrix.
 * This class provides methods to compute eigenvalues and eigenvectors of
 * diagonal matrices with complex number types (float or double). It is designed
 * to work with matrices that have a diagonal structure.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam M The number of rows and columns in the square diagonal matrix.
 */
template <typename T, std::size_t M> class LinalgSolverEigDiag {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using A_Type = Matrix<DefDiag, T, M>;
  using EigenValues_Type = Matrix<DefDense, Base::Matrix::Complex<T>, M, 1>;
  using EigenVectors_Type = Matrix<DefDiag, Base::Matrix::Complex<T>, M>;

public:
  /* Constructor */
  LinalgSolverEigDiag() : _eigen_values() {}

  /* Copy Constructor */
  LinalgSolverEigDiag(const LinalgSolverEigDiag<T, M> &other)
      : _eigen_values(other._eigen_values) {}

  LinalgSolverEigDiag<T, M> &operator=(const LinalgSolverEigDiag<T, M> &other) {
    if (this != &other) {
      this->_eigen_values = other._eigen_values;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigDiag(LinalgSolverEigDiag<T, M> &&other) noexcept
      : _eigen_values(std::move(other._eigen_values)) {}

  LinalgSolverEigDiag<T, M> &
  operator=(LinalgSolverEigDiag<T, M> &&other) noexcept {
    if (this != &other) {
      this->_eigen_values = std::move(other._eigen_values);
    }
    return *this;
  }

  /* Solve method */

  /**
   * @brief Computes the eigenvalues of the given diagonal matrix.
   *
   * This function extracts the diagonal elements of the matrix A and stores
   * them as eigenvalues in the internal _eigen_values member variable.
   *
   * @param A The input diagonal matrix for which eigenvalues are to be
   * computed.
   */
  inline void solve_eigen_values(const Matrix<DefDiag, T, M> &A) {
    this->_eigen_values =
        EigenValues_Type(Base::Matrix::convert_matrix_real_to_complex(
            Base::Matrix::Matrix<T, M, 1>(A.matrix.data)));
  }

  /**
   * @brief Computes the eigenvectors of the given diagonal matrix.
   *
   * This function extracts the diagonal elements of the matrix A and stores
   * them as eigenvectors in the internal _eigen_values member variable.
   *
   * @param A The input diagonal matrix for which eigenvectors are to be
   * computed.
   */
  inline void solve_eigen_vectors(const Matrix<DefDiag, T, M> &A) {
    this->_eigen_values =
        EigenValues_Type(Base::Matrix::convert_matrix_real_to_complex(
            Base::Matrix::Matrix<T, M, 1>(A.matrix.data)));
  }

  /* Get */

  /**
   * @brief Retrieves the computed eigenvalues.
   *
   * This function returns the eigenvalues computed by the solver as an
   * EigenValues_Type object.
   *
   * @return An EigenValues_Type object containing the eigenvalues.
   */
  inline auto get_eigen_values(void) -> EigenValues_Type {
    return EigenValues_Type(this->_eigen_values);
  }

  /**
   * @brief Retrieves the eigenvectors of the diagonal matrix.
   *
   * Since the eigenvectors of a diagonal matrix are simply the identity
   * matrix, this function returns an EigenVectors_Type object representing
   * the identity matrix.
   *
   * @return An EigenVectors_Type object representing the identity matrix.
   */
  inline auto get_eigen_vectors(void) -> EigenVectors_Type {
    return EigenVectors_Type(Base::Matrix::convert_matrix_real_to_complex(
        Base::Matrix::DiagMatrix<T, M>::identity()));
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

protected:
  /* Variable */
  EigenValues_Type _eigen_values;
};

/* Linalg solver for Complex Eigen values and vectors of Sparse Matrix */

/** @brief Linalg solver for Complex Eigen values and vectors of Sparse Matrix.
 * This class provides methods to compute eigenvalues and eigenvectors of
 * sparse matrices with complex number types (float or double). It is designed
 * to work with matrices that have a sparse structure, utilizing the Eigen
 * library for efficient computations.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam M The number of rows and columns in the square sparse matrix.
 * @tparam SparseAvailable A type indicating whether sparse operations are
 * available.
 */
template <typename T, std::size_t M, typename SparseAvailable>
class LinalgSolverEigSparse {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  using A_Type = Matrix<DefSparse, T, M, M, SparseAvailable>;
  using EigenSolver_Type = Base::Matrix::EigenSolverComplex<T, M>;

public:
  /* Constructor */
  LinalgSolverEigSparse() : _Eigen_solver() {}

  /* Copy Constructor */
  LinalgSolverEigSparse(
      const LinalgSolverEigSparse<T, M, SparseAvailable> &other)
      : _Eigen_solver(other._Eigen_solver) {}

  LinalgSolverEigSparse<T, M, SparseAvailable> &
  operator=(const LinalgSolverEigSparse<T, M, SparseAvailable> &other) {
    if (this != &other) {
      this->_Eigen_solver = other._Eigen_solver;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverEigSparse(
      LinalgSolverEigSparse<T, M, SparseAvailable> &&other) noexcept
      : _Eigen_solver(std::move(other._Eigen_solver)) {}

  LinalgSolverEigSparse<T, M, SparseAvailable> &
  operator=(LinalgSolverEigSparse<T, M, SparseAvailable> &&other) noexcept {
    if (this != &other) {
      this->_Eigen_solver = std::move(other._Eigen_solver);
    }
    return *this;
  }

public:
  /* Solve method */

  /**
   * @brief Computes the eigenvalues of the given sparse matrix.
   *
   * This function uses the internal Eigen solver to compute the eigenvalues
   * of the sparse matrix contained in the input parameter A. The results are
   * stored within the solver for later retrieval.
   *
   * @param A An object containing the sparse matrix for which eigenvalues are
   * to be computed.
   */
  inline void solve_eigen_values(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_values(
        Base::Matrix::output_dense_matrix(A.matrix));
  }

  /**
   * @brief Continues solving for eigenvalues if the solver supports it.
   *
   * This function allows the solver to continue computing eigenvalues if it
   * has not yet completed the process. It is useful for iterative solvers.
   */
  inline void continue_solving_eigen_values(void) {
    this->_Eigen_solver.continue_solving_eigen_values();
  }

  /**
   * @brief Computes the eigenvectors of the given sparse matrix.
   *
   * This function uses the internal Eigen solver to compute the eigenvectors
   * of the sparse matrix contained in the input parameter A. The results are
   * stored within the solver for later retrieval.
   *
   * @param A An object containing the sparse matrix for which eigenvectors are
   * to be computed.
   */
  inline void solve_eigen_vectors(const A_Type &A) {
    this->_Eigen_solver.solve_eigen_vectors(
        Base::Matrix::output_dense_matrix(A.matrix));
  }

  /* Get */

  /**
   * @brief Retrieves the computed eigenvalues.
   *
   * This function returns the eigenvalues computed by the solver as a
   * ForLinalgSolverEig::EigenValues_Type object.
   *
   * @return A ForLinalgSolverEig::EigenValues_Type object containing the
   * eigenvalues.
   */
  inline auto get_eigen_values(void)
      -> ForLinalgSolverEig::EigenValues_Type<T, M> {
    return ForLinalgSolverEig::EigenValues_Type<T, M>(
        Base::Matrix::Matrix<Base::Matrix::Complex<T>, M, 1>(
            this->_Eigen_solver.get_eigen_values()));
  }

  /**
   * @brief Retrieves the computed eigenvectors.
   *
   * This function returns the eigenvectors computed by the solver as a
   * ForLinalgSolverEig::EigenVectors_Type object.
   *
   * @return A ForLinalgSolverEig::EigenVectors_Type object containing the
   * eigenvectors.
   */
  inline auto get_eigen_vectors(void)
      -> ForLinalgSolverEig::EigenVectors_Type<T, M> {
    return ForLinalgSolverEig::EigenVectors_Type<T, M>(
        this->_Eigen_solver.get_eigen_vectors());
  }

  /* Set */

  /**
   * @brief Sets the maximum number of iterations for the solver.
   *
   * This function allows the user to specify the maximum number of iterations
   * that the solver should perform when computing eigenvalues and
   * eigenvectors.
   *
   * @param iteration_max The maximum number of iterations to set.
   */
  inline void set_iteration_max(const std::size_t &iteration_max) {
    this->_Eigen_solver.iteration_max = iteration_max;
  }

  /**
   * @brief Sets the maximum number of iterations for eigenvector computation.
   *
   * This function allows the user to specify the maximum number of iterations
   * that the solver should perform when computing eigenvectors.
   *
   * @param iteration_max_for_eigen_vector The maximum number of iterations to
   * set for eigenvector computation.
   */
  inline void set_iteration_max_for_eigen_vector(
      const std::size_t &iteration_max_for_eigen_vector) {
    this->_Eigen_solver.iteration_max_for_eigen_vector =
        iteration_max_for_eigen_vector;
  }

  /**
   * @brief Sets the minimum division value for the Eigen solver.
   *
   * This function assigns the specified value to the `division_min` parameter
   * of the internal Eigen solver. The `division_min` parameter is typically
   * used to avoid division by very small numbers, which can lead to numerical
   * instability.
   *
   * @param division_min_in The minimum value to be used for divisions in the
   * Eigen solver.
   */
  inline void set_division_min(const T &division_min_in) {
    this->_Eigen_solver.division_min = division_min_in;
  }

  /**
   * @brief Sets a small value threshold for numerical stability.
   *
   * This function allows the user to specify a small value that can be used
   * to avoid numerical instability during computations, such as division by
   * very small numbers.
   *
   * @param small_value_in The small value to set for numerical stability.
   */
  inline void set_small_value(const T &small_value_in) {
    this->_Eigen_solver.small_value = small_value_in;
  }

  /* Check */

  /**
   * @brief Checks the validity of the eigenvalues and eigenvectors for the
   * given sparse matrix.
   *
   * This function checks whether the computed eigenvalues and eigenvectors
   * are valid for the provided matrix A. It returns a
   * ForLinalgSolverEig::EigenVectors_Type object containing the results.
   *
   * @param A The input sparse matrix for which validity is to be checked.
   * @return A ForLinalgSolverEig::EigenVectors_Type object containing the
   * validity check results.
   */
  inline auto check_validity(const A_Type &A)
      -> ForLinalgSolverEig::EigenVectors_Type<T, M> {

    return ForLinalgSolverEig::EigenVectors_Type<T, M>(
        this->_Eigen_solver.check_validity(
            Base::Matrix::output_dense_matrix(A.matrix)));
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

protected:
  /* Variable */
  EigenSolver_Type _Eigen_solver;
};

/* make LinalgSolverEig Complex */

/**
 * @brief Factory function to create a LinalgSolverEig instance based on the
 * type of matrix provided.
 *
 * This function uses SFINAE (Substitution Failure Is Not An Error) to determine
 * the type of matrix and return the appropriate LinalgSolverEig instance.
 *
 * @tparam A_Type The type of the matrix for which the solver is to be created.
 * @return A LinalgSolverEig instance suitable for the given matrix type.
 */
template <
    typename A_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverEig()
    -> LinalgSolverEigDense<typename A_Type::Value_Complex_Type, A_Type::COLS> {

  return LinalgSolverEigDense<typename A_Type::Value_Complex_Type,
                              A_Type::COLS>();
}

/** @brief Factory function to create a LinalgSolverEig instance for diagonal
 * matrices.
 *
 * This function is specialized for diagonal matrices and returns a
 * LinalgSolverEigDiag instance.
 *
 * @tparam A_Type The type of the diagonal matrix for which the solver is to be
 * created.
 * @return A LinalgSolverEigDiag instance suitable for the given diagonal matrix
 * type.
 */
template <typename A_Type, typename std::enable_if<
                               Is_Diag_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverEig()
    -> LinalgSolverEigDiag<typename A_Type::Value_Complex_Type, A_Type::COLS> {

  return LinalgSolverEigDiag<typename A_Type::Value_Complex_Type,
                             A_Type::COLS>();
}

/** @brief Factory function to create a LinalgSolverEig instance for sparse
 * matrices.
 *
 * This function is specialized for sparse matrices and returns a
 * LinalgSolverEigSparse instance.
 *
 * @tparam A_Type The type of the sparse matrix for which the solver is to be
 * created.
 * @return A LinalgSolverEigSparse instance suitable for the given sparse matrix
 * type.
 */
template <
    typename A_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverEig()
    -> LinalgSolverEigSparse<typename A_Type::Value_Complex_Type, A_Type::COLS,
                             typename A_Type::SparseAvailable_Type> {

  return LinalgSolverEigSparse<typename A_Type::Value_Complex_Type,
                               A_Type::COLS,
                               typename A_Type::SparseAvailable_Type>();
}

/* LinalgSolverEig Complex Type */
template <typename A_Type>
using LinalgSolverEig_Type = decltype(make_LinalgSolverEig<A_Type>());

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_LINALG_EIG_HPP__
