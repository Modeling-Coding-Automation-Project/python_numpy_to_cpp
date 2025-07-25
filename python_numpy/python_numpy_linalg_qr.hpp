/**
 * @file python_numpy_linalg_qr.hpp
 * @brief QR decomposition solvers for dense, diagonal, and sparse matrices in a
 * NumPy-like C++ linear algebra library.
 *
 * This header defines template classes and factory functions for performing QR
 * decomposition on matrices of various types (dense, diagonal, sparse). The
 * solvers are designed to work with the matrix types defined in the library,
 * providing interfaces to compute and retrieve the Q and R factors.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __PYTHON_NUMPY_LINALG_QR_HPP__
#define __PYTHON_NUMPY_LINALG_QR_HPP__

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_base_simplification.hpp"
#include "python_numpy_base_simplified_action.hpp"
#include "python_numpy_linalg_solver.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace PythonNumpy {

const double DEFAULT_DIVISION_MIN_LINALG_QR = 1.0e-10;

namespace LinalgQR_Operation {

// inner loop: j loop
template <typename Upper_Triangular_Matrix_Type, typename Matrix_In_Type,
          typename Matrix_Out_Type, typename T, std::size_t Row_Index,
          std::size_t I, std::size_t J, int End_Index_Value>
struct BackwardSubstitution_J_Loop {
  /*
   * @brief Computes the backward substitution for the j loop in the QR
   * decomposition.
   *
   * This function computes the backward substitution for the j loop, updating
   * the output matrix with the computed values based on the upper triangular
   * matrix R and the input matrix.
   *
   * @param R The upper triangular matrix from QR decomposition.
   * @param matrix_in The input matrix used for substitution.
   * @param matrix_out The output matrix to store results.
   * @param sum The accumulated sum for the current row.
   * @param division_min The minimum value to avoid division by zero.
   */
  static void compute(const Upper_Triangular_Matrix_Type &R,
                      const Matrix_In_Type &matrix_in,
                      Matrix_Out_Type &matrix_out, T &sum,
                      const T &division_min) {

    sum -= Base::Matrix::get_sparse_matrix_value<I, J>(R) *
           matrix_out.template get<J, Row_Index>();

    BackwardSubstitution_J_Loop<Upper_Triangular_Matrix_Type, Matrix_In_Type,
                                Matrix_Out_Type, T, Row_Index, I, (J + 1),
                                (End_Index_Value - 1)>::compute(R, matrix_in,
                                                                matrix_out, sum,
                                                                division_min);
  }
};

// terminate j loop
template <typename Upper_Triangular_Matrix_Type, typename Matrix_In_Type,
          typename Matrix_Out_Type, typename T, std::size_t Row_Index,
          std::size_t I, std::size_t J>
struct BackwardSubstitution_J_Loop<Upper_Triangular_Matrix_Type, Matrix_In_Type,
                                   Matrix_Out_Type, T, Row_Index, I, J, 0> {
  /*
   * @brief Computes the backward substitution for the j loop in the QR
   * decomposition.
   *
   * This function computes the backward substitution for the j loop, updating
   * the output matrix with the computed values based on the upper triangular
   * matrix R and the input matrix.
   *
   * @param R The upper triangular matrix from QR decomposition.
   * @param matrix_in The input matrix used for substitution.
   * @param matrix_out The output matrix to store results.
   * @param sum The accumulated sum for the current row.
   * @param division_min The minimum value to avoid division by zero.
   */
  static void compute(const Upper_Triangular_Matrix_Type &R,
                      const Matrix_In_Type &matrix_in,
                      Matrix_Out_Type &matrix_out, T &sum,
                      const T &division_min) {

    // Do nothing
    static_cast<void>(R);
    static_cast<void>(matrix_in);
    static_cast<void>(matrix_out);
    static_cast<void>(sum);
    static_cast<void>(division_min);
  }
};

// inner loop: i loop
template <typename Upper_Triangular_Matrix_Type, typename Matrix_In_Type,
          typename Matrix_Out_Type, typename T, std::size_t Row_Index,
          std::size_t I_Count>
struct BackwardSubstitution_I_Loop {
  /*
   * @brief Computes the backward substitution for the i loop in the QR
   * decomposition.
   *
   * This function computes the backward substitution for the i loop, updating
   * the output matrix with the computed values based on the upper triangular
   * matrix R and the input matrix.
   *
   * @param R The upper triangular matrix from QR decomposition.
   * @param matrix_in The input matrix used for substitution.
   * @param matrix_out The output matrix to store results.
   * @param division_min The minimum value to avoid division by zero.
   */
  static void compute(const Upper_Triangular_Matrix_Type &R,
                      const Matrix_In_Type &matrix_in,
                      Matrix_Out_Type &matrix_out, const T &division_min) {

    constexpr std::size_t I = I_Count;

    compute_conditional(
        R, matrix_in, matrix_out, division_min,
        std::integral_constant<bool, (I + 1) < Matrix_Out_Type::ROWS>{});
  }

private:
  /**
   * @brief Computes the backward substitution conditionally based on the index.
   *
   * This function computes the backward substitution conditionally, either
   * performing the full computation or a simplified version based on the index.
   *
   * @param R The upper triangular matrix from QR decomposition.
   * @param matrix_in The input matrix used for substitution.
   * @param matrix_out The output matrix to store results.
   * @param division_min The minimum value to avoid division by zero.
   * @param is_full Whether to perform the full computation or a simplified one.
   */
  static void compute_conditional(const Upper_Triangular_Matrix_Type &R,
                                  const Matrix_In_Type &matrix_in,
                                  Matrix_Out_Type &matrix_out,
                                  const T &division_min, std::true_type) {

    constexpr std::size_t I = I_Count;
    constexpr int End_Index_Value = Matrix_Out_Type::COLS - (I + 1);

    T sum = matrix_in.template get<I, Row_Index>();

    BackwardSubstitution_J_Loop<Upper_Triangular_Matrix_Type, Matrix_In_Type,
                                Matrix_Out_Type, T, Row_Index, I, (I + 1),
                                End_Index_Value>::compute(R, matrix_in,
                                                          matrix_out, sum,
                                                          division_min);

    matrix_out.template set<I, Row_Index>(
        sum /
        Base::Utility::avoid_zero_divide(
            Base::Matrix::get_sparse_matrix_value<I, I>(R), division_min));

    BackwardSubstitution_I_Loop<Upper_Triangular_Matrix_Type, Matrix_In_Type,
                                Matrix_Out_Type, T, Row_Index,
                                (I_Count - 1)>::compute(R, matrix_in,
                                                        matrix_out,
                                                        division_min);
  }

  /**
   * @brief Computes the backward substitution conditionally based on the index.
   *
   * This function computes the backward substitution conditionally, either
   * performing the full computation or a simplified version based on the index.
   *
   * @param R The upper triangular matrix from QR decomposition.
   * @param matrix_in The input matrix used for substitution.
   * @param matrix_out The output matrix to store results.
   * @param division_min The minimum value to avoid division by zero.
   * @param is_full Whether to perform the full computation or a simplified one.
   */
  static void compute_conditional(const Upper_Triangular_Matrix_Type &R,
                                  const Matrix_In_Type &matrix_in,
                                  Matrix_Out_Type &matrix_out,
                                  const T &division_min, std::false_type) {

    constexpr std::size_t I = I_Count;

    T sum = matrix_in.template get<I, Row_Index>();

    matrix_out.template set<I, Row_Index>(
        sum /
        Base::Utility::avoid_zero_divide(
            Base::Matrix::get_sparse_matrix_value<I, I>(R), division_min));

    BackwardSubstitution_I_Loop<Upper_Triangular_Matrix_Type, Matrix_In_Type,
                                Matrix_Out_Type, T, Row_Index,
                                (I_Count - 1)>::compute(R, matrix_in,
                                                        matrix_out,
                                                        division_min);
  }
};

// terminate i loop
template <typename Upper_Triangular_Matrix_Type, typename Matrix_In_Type,
          typename Matrix_Out_Type, typename T, std::size_t Row_Index>
struct BackwardSubstitution_I_Loop<Upper_Triangular_Matrix_Type, Matrix_In_Type,
                                   Matrix_Out_Type, T, Row_Index, 0> {
  /**
   * @brief Computes the backward substitution for the i loop in the QR
   * decomposition.
   * This function computes the backward substitution for the i loop, updating
   * the output matrix with the computed values based on the upper triangular
   * matrix R and the input matrix.
   * @param R The upper triangular matrix from QR decomposition.
   * @param matrix_in The input matrix used for substitution.
   * @param matrix_out The output matrix to store results.
   * @param division_min The minimum value to avoid division by zero.
   */
  static void compute(const Upper_Triangular_Matrix_Type &R,
                      const Matrix_In_Type &matrix_in,
                      Matrix_Out_Type &matrix_out, const T &division_min) {

    constexpr int End_Index_Value = Matrix_Out_Type::COLS - 1;

    T sum = matrix_in.template get<0, Row_Index>();

    BackwardSubstitution_J_Loop<Upper_Triangular_Matrix_Type, Matrix_In_Type,
                                Matrix_Out_Type, T, Row_Index, 0, 1,
                                End_Index_Value>::compute(R, matrix_in,
                                                          matrix_out, sum,
                                                          division_min);

    matrix_out.template set<0, Row_Index>(
        sum /
        Base::Utility::avoid_zero_divide(
            Base::Matrix::get_sparse_matrix_value<0, 0>(R), division_min));
  }
};

// row_index loop
template <typename Upper_Triangular_Matrix_Type, typename Matrix_In_Type,
          typename Matrix_Out_Type, typename T, std::size_t Row_Index_Count>
struct BackwardSubstitution_RowLoop {
  /**
   * @brief Computes the backward substitution for the row loop in the QR
   * decomposition.
   * This function computes the backward substitution for the row loop, updating
   * the output matrix with the computed values based on the upper triangular
   * matrix R and the input matrix.
   * @param R The upper triangular matrix from QR decomposition.
   * @param matrix_in The input matrix used for substitution.
   * @param matrix_out The output matrix to store results.
   * @param division_min The minimum value to avoid division by zero.
   */
  static void compute(const Upper_Triangular_Matrix_Type &R,
                      const Matrix_In_Type &matrix_in,
                      Matrix_Out_Type &matrix_out, const T &division_min) {

    constexpr std::size_t Row_Index =
        (Matrix_In_Type::ROWS - 1) - Row_Index_Count;

    BackwardSubstitution_I_Loop<
        Upper_Triangular_Matrix_Type, Matrix_In_Type, Matrix_Out_Type, T,
        Row_Index, (Matrix_Out_Type::COLS - 1)>::compute(R, matrix_in,
                                                         matrix_out,
                                                         division_min);

    BackwardSubstitution_RowLoop<Upper_Triangular_Matrix_Type, Matrix_In_Type,
                                 Matrix_Out_Type, T,
                                 (Row_Index_Count - 1)>::compute(R, matrix_in,
                                                                 matrix_out,
                                                                 division_min);
  }
};

// row_index loop terminate
template <typename Upper_Triangular_Matrix_Type, typename Matrix_In_Type,
          typename Matrix_Out_Type, typename T>
struct BackwardSubstitution_RowLoop<Upper_Triangular_Matrix_Type,
                                    Matrix_In_Type, Matrix_Out_Type, T, 0> {

  /**
   * @brief Computes the backward substitution for the row loop in the QR
   * decomposition.
   * This function computes the backward substitution for the row loop, updating
   * the output matrix with the computed values based on the upper triangular
   * matrix R and the input matrix.
   * @param R The upper triangular matrix from QR decomposition.
   * @param matrix_in The input matrix used for substitution.
   * @param matrix_out The output matrix to store results.
   * @param division_min The minimum value to avoid division by zero.
   */
  static void compute(const Upper_Triangular_Matrix_Type &R,
                      const Matrix_In_Type &matrix_in,
                      Matrix_Out_Type &matrix_out, const T &division_min) {

    constexpr std::size_t Row_Index = Matrix_In_Type::ROWS - 1;

    BackwardSubstitution_I_Loop<
        Upper_Triangular_Matrix_Type, Matrix_In_Type, Matrix_Out_Type, T,
        Row_Index, (Matrix_Out_Type::COLS - 1)>::compute(R, matrix_in,
                                                         matrix_out,
                                                         division_min);
  }
};

/**
 * @brief Performs backward substitution on an upper triangular matrix.
 *
 * This function performs backward substitution on an upper triangular matrix
 * R, using the input matrix and storing the results in the output matrix.
 * It handles both dense and sparse matrices, ensuring that the dimensions
 * match appropriately.
 *
 * @tparam Upper_Triangular_Matrix_Type The type of the upper triangular matrix
 * R.
 * @tparam Matrix_In_Type The type of the input matrix.
 * @tparam Matrix_Out_Type The type of the output matrix.
 * @tparam T The data type of the matrix elements (e.g., float, double).
 *
 * @param R The upper triangular matrix from QR decomposition.
 * @param matrix_in The input matrix used for substitution.
 * @param matrix_out The output matrix to store results.
 * @param division_min The minimum value to avoid division by zero.
 */
template <typename Upper_Triangular_Matrix_Type, typename Matrix_In_Type,
          typename Matrix_Out_Type, typename T>
inline void backward_substitution(const Upper_Triangular_Matrix_Type &R,
                                  const Matrix_In_Type &matrix_in,
                                  Matrix_Out_Type &matrix_out,
                                  const T &division_min) {

  static_assert(
      std::is_same<typename Upper_Triangular_Matrix_Type::MatrixType,
                   Base::Matrix::Is_CompiledSparseMatrix>::value,
      "Upper_Triangular_Matrix_Type must be a compiled sparse matrix type.");

  static_assert(Upper_Triangular_Matrix_Type::COLS == Matrix_Out_Type::COLS,
                "The number of columns in the upper triangular matrix R must "
                "match the number of columns in the input matrix.");
  static_assert(Upper_Triangular_Matrix_Type::COLS >=
                    Upper_Triangular_Matrix_Type::ROWS,
                "The upper triangular matrix R must have at least as many "
                "columns as rows.");

  static_assert(Is_Dense_Matrix<Matrix_In_Type>::value ||
                    Is_Diag_Matrix<Matrix_In_Type>::value ||
                    Is_Sparse_Matrix<Matrix_In_Type>::value,
                "The input matrix must be either dense or sparse.");

  static_assert(Is_Dense_Matrix<Matrix_Out_Type>::value ||
                    Is_Diag_Matrix<Matrix_Out_Type>::value ||
                    Is_Sparse_Matrix<Matrix_Out_Type>::value,
                "The output matrix must be either dense or sparse.");

  BackwardSubstitution_RowLoop<
      Upper_Triangular_Matrix_Type, Matrix_In_Type, Matrix_Out_Type, T,
      (Matrix_Out_Type::ROWS - 1)>::compute(R, matrix_in, matrix_out,
                                            division_min);
}

} // namespace LinalgQR_Operation

/**
 * @brief LinalgSolverQR class for QR decomposition and solving linear systems.
 *
 * This class provides methods to perform QR decomposition on matrices and solve
 * linear systems using the decomposed matrices. It supports dense, diagonal,
 * and sparse matrices, and can handle both single and double precision floating
 * point values.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 */
template <typename T, std::size_t M, std::size_t N> class LinalgSolverQR {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  /* Check Compatibility */
  static_assert(M >= N, "only supports M >= N (columns >= rows) "
                        "for QR decomposition.");

protected:
  /* Type */
  using _R_TriangluarRowIndices = Base::Matrix::UpperTriangularRowIndices<M, N>;
  using _R_TriangluarRowPointers =
      Base::Matrix::UpperTriangularRowPointers<M, N>;

public:
  /* Constructor */
  LinalgSolverQR()
      : _QR_decomposer(),
        _R_triangular(
            Base::Matrix::create_UpperTriangularSparseMatrix<T, M, N>()) {}

  /* Copy Constructor */
  LinalgSolverQR(const LinalgSolverQR<T, M, N> &other)
      : _QR_decomposer(other._QR_decomposer),
        _R_triangular(other._R_triangular) {}

  LinalgSolverQR<T, M, N> &operator=(const LinalgSolverQR<T, M, N> &other) {
    if (this != &other) {
      this->_QR_decomposer = other._QR_decomposer;
      this->_R_triangular = other._R_triangular;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverQR(LinalgSolverQR<T, M, N> &&other) noexcept
      : _QR_decomposer(std::move(other._QR_decomposer)),
        _R_triangular(std::move(other._R_triangular)) {}

  LinalgSolverQR<T, M, N> &operator=(LinalgSolverQR<T, M, N> &&other) noexcept {
    if (this != &other) {
      this->_QR_decomposer = std::move(other._QR_decomposer);
      this->_R_triangular = std::move(other._R_triangular);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Performs backward substitution on the upper triangular matrix R.
   *
   * This function performs backward substitution on the upper triangular matrix
   * R, using the input matrix and storing the results in the output matrix.
   * It handles both dense and sparse matrices, ensuring that the dimensions
   * match appropriately.
   *
   * @param matrix_in The input matrix used for substitution.
   * @return A dense matrix containing the results of the backward substitution.
   */
  template <typename Matrix_In_Type>
  inline auto backward_substitution(const Matrix_In_Type &matrix_in)
      -> DenseMatrix_Type<T, Matrix_In_Type::COLS, Matrix_In_Type::ROWS> {
    static_assert(
        Matrix_In_Type::COLS == M,
        "The number of columns in the input matrix must match the number of "
        "columns in the diagonal matrix.");

    using Backward_Substitution_Out_Type =
        DenseMatrix_Type<T, Matrix_In_Type::COLS, Matrix_In_Type::ROWS>;

    Backward_Substitution_Out_Type matrix_out;

    LinalgQR_Operation::backward_substitution(
        this->_R_triangular, matrix_in, matrix_out,
        this->_QR_decomposer.division_min);

    return matrix_out;
  }

  /**
   * @brief Solves the linear system Ax = b using QR decomposition.
   *
   * This function performs QR decomposition on the matrix A and solves the
   * linear system Ax = b, where A is the input matrix and b is the right-hand
   * side vector. The solution is stored in the internal R triangular matrix.
   *
   * @param A The input matrix to decompose and solve.
   */
  inline void solve(const Matrix<DefDense, T, M, N> &A) {
    this->_QR_decomposer.solve(A.matrix);

    Base::Matrix::set_values_UpperTriangularSparseMatrix<T, M, N>(
        this->_R_triangular, this->_QR_decomposer.get_R());
  }

  /* Get Q, R */

  /**
   * @brief Returns the upper triangular matrix R from the QR decomposition.
   *
   * This function returns the upper triangular matrix R that was computed
   * during the QR decomposition process. The matrix is represented as a sparse
   * matrix.
   *
   * @return A sparse matrix representing the upper triangular part of R.
   */
  inline auto get_R(void) -> Matrix<
      DefSparse, T, M, N,
      CreateSparseAvailableFromIndicesAndPointers<
          N, _R_TriangluarRowIndices, _R_TriangluarRowPointers>> const {

    return Matrix<DefSparse, T, M, N,
                  CreateSparseAvailableFromIndicesAndPointers<
                      N, _R_TriangluarRowIndices, _R_TriangluarRowPointers>>(
        this->_R_triangular);
  }

  /**
   * @brief Returns the orthogonal matrix Q from the QR decomposition.
   *
   * This function returns the orthogonal matrix Q that was computed during
   * the QR decomposition process. The matrix is represented as a dense matrix.
   *
   * @return A dense matrix representing the orthogonal part of Q.
   */
  inline auto get_Q(void) -> Matrix<DefDense, T, M, M> const {
    return Matrix<DefDense, T, M, M>(this->_QR_decomposer.get_Q());
  }

  /**
   * @brief Sets the minimum division threshold for QR decomposition.
   *
   * This function allows the user to set a custom minimum division threshold
   * for the QR decomposition process. It is useful for controlling numerical
   * stability and precision.
   *
   * @param division_min The minimum division threshold to set.
   */
  inline void set_division_min(const T &division_min) {
    this->_QR_decomposer.division_min = division_min;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = N;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

protected:
  /* Properties */
  Base::Matrix::QRDecomposition<T, M, N> _QR_decomposer;

  Base::Matrix::CompiledSparseMatrix<T, M, N, _R_TriangluarRowIndices,
                                     _R_TriangluarRowPointers>
      _R_triangular;
};

/**
 * @brief LinalgSolverQRDiag class for QR decomposition of diagonal matrices.
 *
 * This class provides methods to perform QR decomposition specifically for
 * diagonal matrices. It supports both single and double precision floating
 * point values.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam M The number of columns in the diagonal matrix.
 */
template <typename T, std::size_t M> class LinalgSolverQRDiag {
public:
  /* Constructor */
  LinalgSolverQRDiag() : _R() {}

  /* Copy Constructor */
  LinalgSolverQRDiag(const LinalgSolverQRDiag<T, M> &other) : _R(other._R) {}

  LinalgSolverQRDiag<T, M> &operator=(const LinalgSolverQRDiag<T, M> &other) {
    if (this != &other) {
      this->_R = other._R;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverQRDiag(LinalgSolverQRDiag<T, M> &&other) noexcept
      : _R(std::move(other._R)) {}

  LinalgSolverQRDiag<T, M> &
  operator=(LinalgSolverQRDiag<T, M> &&other) noexcept {
    if (this != &other) {
      this->_R = std::move(other._R);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Performs backward substitution on the diagonal matrix R.
   *
   * This function performs backward substitution on the diagonal matrix R,
   * using the input matrix and storing the results in the output matrix.
   * It ensures that the dimensions match appropriately.
   *
   * @param matrix_in The input matrix used for substitution.
   * @return A dense matrix containing the results of the backward substitution.
   */
  template <typename Matrix_In_Type>
  inline auto backward_substitution(const Matrix_In_Type &matrix_in)
      -> DenseMatrix_Type<T, M, M> {
    static_assert(Matrix_In_Type::COLS == M,
                  "The number of columns in the input matrix must match the "
                  "number of columns in the diagonal matrix.");

    using Backward_Substitution_Out_Type = DenseMatrix_Type<T, M, M>;

    Backward_Substitution_Out_Type matrix_out;

    auto solver = make_LinalgSolver<decltype(this->_R), Matrix_In_Type>();

    matrix_out = solver.solve(this->_R, matrix_in);

    return matrix_out;
  }

  /**
   * @brief Solves the QR decomposition for the given diagonal matrix A.
   *
   * This function sets the internal R matrix to the provided diagonal matrix A.
   *
   * @param A The input diagonal matrix to decompose.
   */
  inline void solve(const Matrix<DefDiag, T, M> &A) { this->_R = A; }

  /**
   * @brief Returns the diagonal matrix R from the QR decomposition.
   *
   * This function returns the diagonal matrix R that was computed during
   * the QR decomposition process. The matrix is represented as a diagonal
   * matrix.
   *
   * @return A diagonal matrix representing R.
   */
  inline auto get_R(void) -> Matrix<DefDiag, T, M> const { return this->_R; }

  /**
   * @brief Returns the identity matrix Q from the QR decomposition.
   *
   * This function returns the identity matrix Q, which is a property of the
   * QR decomposition for diagonal matrices.
   *
   * @return An identity matrix representing Q.
   */
  inline auto get_Q(void) -> Matrix<DefDiag, T, M> const {
    return Matrix<DefDiag, T, M>::identity();
  }

protected:
  /* Properties */
  Matrix<DefDiag, T, M> _R;
};

/**
 * @brief LinalgSolverQRSparse class for QR decomposition of sparse matrices.
 *
 * This class provides methods to perform QR decomposition specifically for
 * sparse matrices. It supports both single and double precision floating
 * point values, and uses a sparse representation for the R matrix.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam M The number of columns in the sparse matrix.
 * @tparam N The number of rows in the sparse matrix.
 * @tparam SparseAvailable A type indicating the availability of sparse storage.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
class LinalgSolverQRSparse {
public:
  /* Type */
  using Value_Type = T;
  static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                "Value data type must be float or double.");

  /* Check Compatibility */
  static_assert(M >= N, "only supports M >= N (columns >= rows) "
                        "for QR decomposition.");

protected:
  /* Type */
  using _R_TriangluarRowIndices = Base::Matrix::UpperTriangularRowIndices<M, N>;
  using _R_TriangluarRowPointers =
      Base::Matrix::UpperTriangularRowPointers<M, N>;

public:
  /* Constructor */
  LinalgSolverQRSparse() {}

  /* Copy Constructor */
  LinalgSolverQRSparse(
      const LinalgSolverQRSparse<T, M, N, SparseAvailable> &other)
      : _QR_decomposer(other._QR_decomposer),
        _R_triangular(other._R_triangular) {}

  LinalgSolverQRSparse<T, M, N, SparseAvailable> &
  operator=(const LinalgSolverQRSparse<T, M, N, SparseAvailable> &other) {
    if (this != &other) {
      this->_QR_decomposer = other._QR_decomposer;
      this->_R_triangular = other._R_triangular;
    }
    return *this;
  }

  /* Move Constructor */
  LinalgSolverQRSparse(
      LinalgSolverQRSparse<T, M, N, SparseAvailable> &&other) noexcept
      : _QR_decomposer(std::move(other._QR_decomposer)),
        _R_triangular(std::move(other._R_triangular)) {}

  LinalgSolverQRSparse<T, M, N, SparseAvailable> &
  operator=(LinalgSolverQRSparse<T, M, N, SparseAvailable> &&other) noexcept {
    if (this != &other) {
      this->_QR_decomposer = std::move(other._QR_decomposer);
      this->_R_triangular = std::move(other._R_triangular);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Performs backward substitution on the diagonal matrix R.
   *
   * This function performs backward substitution on the diagonal matrix R,
   * using the input matrix and storing the results in the output matrix.
   * It ensures that the dimensions match appropriately.
   *
   * @param matrix_in The input matrix used for substitution.
   * @return A dense matrix containing the results of the backward substitution.
   */
  template <typename Matrix_In_Type>
  inline auto backward_substitution(const Matrix_In_Type &matrix_in)
      -> DenseMatrix_Type<T, Matrix_In_Type::COLS, Matrix_In_Type::ROWS> {
    static_assert(
        Matrix_In_Type::COLS == M,
        "The number of columns in the input matrix must match the number of "
        "columns in the diagonal matrix.");

    using Backward_Substitution_Out_Type =
        DenseMatrix_Type<T, Matrix_In_Type::COLS, Matrix_In_Type::ROWS>;

    Backward_Substitution_Out_Type matrix_out;

    LinalgQR_Operation::backward_substitution(
        this->_R_triangular, matrix_in, matrix_out,
        this->_QR_decomposer.division_min);

    return matrix_out;
  }

  /**
   * @brief Solves the QR decomposition for the given sparse matrix A.
   *
   * This function sets the internal R matrix to the provided sparse matrix A
   * and performs QR decomposition by calling the internal _decompose method.
   *
   * @param A The input sparse matrix to decompose.
   */
  inline void solve(const Matrix<DefSparse, T, M, N, SparseAvailable> &A) {
    this->_QR_decomposer.solve(A.matrix);

    Base::Matrix::set_values_UpperTriangularSparseMatrix<T, M, N>(
        this->_R_triangular, this->_QR_decomposer.get_R());
  }

  /**
   * @brief Returns the upper triangular matrix R from the QR decomposition.
   *
   * This function returns the upper triangular matrix R that was computed
   * during the QR decomposition process. The matrix is represented as a sparse
   * matrix with pre-defined row indices and pointers.
   *
   * @return A sparse matrix representing the upper triangular part of R.
   */
  inline auto get_R(void) -> Matrix<
      DefSparse, T, M, N,
      CreateSparseAvailableFromIndicesAndPointers<
          N, _R_TriangluarRowIndices, _R_TriangluarRowPointers>> const {

    return Matrix<DefSparse, T, M, N,
                  CreateSparseAvailableFromIndicesAndPointers<
                      N, _R_TriangluarRowIndices, _R_TriangluarRowPointers>>(
        this->_R_triangular);
  }

  /**
   * @brief Returns the orthogonal matrix Q from the QR decomposition.
   *
   * This function returns the orthogonal matrix Q that was computed during
   * the QR decomposition process. The matrix is represented as a dense matrix.
   *
   * @return A dense matrix representing the orthogonal part of Q.
   */
  inline auto get_Q(void) -> Matrix<DefDense, T, M, M> const {
    return Matrix<DefDense, T, M, M>(this->_QR_decomposer.get_Q());
  }

  /**
   * @brief Sets the minimum division threshold for QR decomposition.
   *
   * This function allows the user to set a custom minimum division threshold
   * for the QR decomposition process. It is useful for controlling numerical
   * stability and precision.
   *
   * @param division_min The minimum division threshold to set.
   */
  inline void set_division_min(const T &division_min) {
    this->_QR_decomposer.division_min = division_min;
  }

public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = N;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;
  static_assert(!IS_COMPLEX, "Complex type is not supported.");

protected:
  /* Variable */
  Base::Matrix::QRDecompositionSparse<
      T, M, N, Base::Matrix::RowIndicesFromSparseAvailable<SparseAvailable>,
      Base::Matrix::RowPointersFromSparseAvailable<SparseAvailable>>
      _QR_decomposer;

  Base::Matrix::CompiledSparseMatrix<T, M, N, _R_TriangluarRowIndices,
                                     _R_TriangluarRowPointers>
      _R_triangular =
          Base::Matrix::create_UpperTriangularSparseMatrix<T, M, N>();
};

/* make LinalgSolverQR */

/**
 * @brief Factory function to create a LinalgSolverQR instance based on the
 * matrix type.
 *
 * This function uses SFINAE to determine the appropriate LinalgSolverQR type
 * based on the input matrix type (dense, diagonal, or sparse).
 *
 * @tparam A_Type The type of the input matrix.
 * @return An instance of LinalgSolverQR, LinalgSolverQRDiag, or
 * LinalgSolverQRSparse.
 */
template <
    typename A_Type,
    typename std::enable_if<Is_Dense_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverQR(void)
    -> LinalgSolverQR<typename A_Type::Value_Type, A_Type::COLS, A_Type::ROWS> {

  return LinalgSolverQR<typename A_Type::Value_Type, A_Type::COLS,
                        A_Type::ROWS>();
}

/**
 * @brief Factory function to create a LinalgSolverQRDiag instance for diagonal
 * matrices.
 *
 * This function creates an instance of LinalgSolverQRDiag specifically for
 * diagonal matrices, using the value type and number of columns from the input
 * matrix type.
 *
 * @tparam A_Type The type of the input diagonal matrix.
 * @return An instance of LinalgSolverQRDiag.
 */
template <typename A_Type, typename std::enable_if<
                               Is_Diag_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverQR(void)
    -> LinalgSolverQRDiag<typename A_Type::Value_Type, A_Type::COLS> {

  return LinalgSolverQRDiag<typename A_Type::Value_Type, A_Type::COLS>();
}

/**
 * @brief Factory function to create a LinalgSolverQRSparse instance for sparse
 * matrices.
 *
 * This function creates an instance of LinalgSolverQRSparse specifically for
 * sparse matrices, using the value type, number of columns, rows, and sparse
 * availability from the input matrix type.
 *
 * @tparam A_Type The type of the input sparse matrix.
 * @return An instance of LinalgSolverQRSparse.
 */
template <
    typename A_Type,
    typename std::enable_if<Is_Sparse_Matrix<A_Type>::value>::type * = nullptr>
inline auto make_LinalgSolverQR(void)
    -> LinalgSolverQRSparse<typename A_Type::Value_Type, A_Type::COLS,
                            A_Type::ROWS,
                            typename A_Type::SparseAvailable_Type> {

  return LinalgSolverQRSparse<typename A_Type::Value_Type, A_Type::COLS,
                              A_Type::ROWS,
                              typename A_Type::SparseAvailable_Type>();
}

/* LinalgSolverQR Type */
template <typename A_Type>
using LinalgSolverQR_Type = decltype(make_LinalgSolverQR<A_Type>());

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_LINALG_QR_HPP__
