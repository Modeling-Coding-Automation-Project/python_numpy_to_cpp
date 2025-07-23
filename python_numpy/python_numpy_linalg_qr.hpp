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
#include "python_numpy_templates.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace PythonNumpy {

const double DEFAULT_DIVISION_MIN_LINALG_QR = 1.0e-10;

namespace LinalgQR_Operation {

// inner loop: j loop
template <typename Upper_Triangular_Matrix_Type, typename Matrix_Type,
          typename T, std::size_t Row_Index, std::size_t I, std::size_t J,
          int End_Index>
struct BackwardSubstitution_J_Loop {
  static void compute(const Upper_Triangular_Matrix_Type &R,
                      const Matrix_Type &matrix_in, Matrix_Type &matrix_out,
                      T &sum, const T &division_min) {

    sum -= R(I, J) * matrix_out(J, Row_Index);
    BackwardSubstitution_J_Loop<Upper_Triangular_Matrix_Type, Matrix_Type, T, I,
                                (J + 1), Row_Index,
                                (End_Index - 1)>::compute(R, matrix_in,
                                                          matrix_out, sum,
                                                          division_min);
  }
};

// terminate j loop
template <typename Upper_Triangular_Matrix_Type, typename Matrix_Type,
          typename T, std::size_t Row_Index, std::size_t I, std::size_t J>
struct BackwardSubstitution_J_Loop<Upper_Triangular_Matrix_Type, Matrix_Type, T,
                                   Row_Index, I, J, 0> {
  static void compute(const Upper_Triangular_Matrix_Type &R,
                      const Matrix_Type &matrix_in, Matrix_Type &matrix_out,
                      T &sum, const T &division_min) {
    // Do nothing
    static_cast<void>(R);
    static_cast<void>(matrix_in);
    static_cast<void>(matrix_out);
    static_cast<void>(sum);
    static_cast<void>(division_min);
  }
};

// inner loop: i loop
template <typename Upper_Triangular_Matrix_Type, typename Matrix_Type,
          typename T, std::size_t Row_Index, std::size_t I_Count>
struct BackwardSubstitution_I_Loop {
  static void compute(const Upper_Triangular_Matrix_Type &R,
                      const Matrix_Type &matrix_in, Matrix_Type &matrix_out,
                      const T &division_min) {

    constexpr std::size_t I = (Matrix_Type::COLS - 1) - I_Count;

    compute_conditional(
        R, matrix_in, matrix_out, division_min,
        std::integral_constant<bool, (I + 1) < Matrix_Type::ROWS>{});
  }

private:
  static void compute_conditional(const Upper_Triangular_Matrix_Type &R,
                                  const Matrix_Type &matrix_in,
                                  Matrix_Type &matrix_out,
                                  const T &division_min, std::true_type) {
    constexpr std::size_t I = (Matrix_Type::COLS - 1) - I_Count;
    constexpr int End_Index = Matrix_Type::ROWS - (I + 1);

    T sum = matrix_in(I, Row_Index);

    BackwardSubstitution_J_Loop<Upper_Triangular_Matrix_Type, Matrix_Type, T, I,
                                (I + 1), Row_Index,
                                End_Index>::compute(R, matrix_in, matrix_out,
                                                    sum, division_min);

    matrix_out(I, Row_Index) =
        sum / Base::Utility::avoid_zero_divide(R(I, I), division_min);

    BackwardSubstitution_I_Loop<Upper_Triangular_Matrix_Type, Matrix_Type, T,
                                Row_Index,
                                (I_Count - 1)>::compute(R, matrix_in,
                                                        matrix_out,
                                                        division_min);
  }

  static void compute_conditional(const Upper_Triangular_Matrix_Type &R,
                                  const Matrix_Type &matrix_in,
                                  Matrix_Type &matrix_out,
                                  const T &division_min, std::false_type) {
    // do nothing
    static_cast<void>(R);
    static_cast<void>(matrix_in);
    static_cast<void>(matrix_out);
    static_cast<void>(division_min);
  }
};

// terminate i loop
template <typename Upper_Triangular_Matrix_Type, typename Matrix_Type,
          typename T, std::size_t Row_Index>
struct BackwardSubstitution_I_Loop<Upper_Triangular_Matrix_Type, Matrix_Type, T,
                                   Row_Index, 0> {
  static void compute(const Upper_Triangular_Matrix_Type &R,
                      const Matrix_Type &matrix_in, Matrix_Type &matrix_out,
                      const T &division_min) {
    // Do nothing
    static_cast<void>(R);
    static_cast<void>(matrix_in);
    static_cast<void>(matrix_out);
    static_cast<void>(division_min);
  }
};

// row_index loop
template <typename Upper_Triangular_Matrix_Type, typename Matrix_Type,
          typename T, std::size_t Row_Index_Count>
struct BackwardSubstitution_RowLoop {
  static void compute(const Upper_Triangular_Matrix_Type &R,
                      const Matrix_Type &matrix_in, Matrix_Type &matrix_out,
                      const T &division_min) {

    constexpr std::size_t Row_Index = (Matrix_Type::ROWS - 1) - Row_Index_Count;

    BackwardSubstitution_I_Loop<Upper_Triangular_Matrix_Type, Matrix_Type, T,
                                Row_Index,
                                (Matrix_Type::COLS - 1)>::compute(R, matrix_in,
                                                                  matrix_out,
                                                                  division_min);

    BackwardSubstitution_RowLoop<Upper_Triangular_Matrix_Type, Matrix_Type, T,
                                 (Row_Index_Count - 1)>::compute(R, matrix_in,
                                                                 matrix_out,
                                                                 division_min);
  }
};

// row_index loop terminate
template <typename Upper_Triangular_Matrix_Type, typename Matrix_Type,
          typename T>
struct BackwardSubstitution_RowLoop<Upper_Triangular_Matrix_Type, Matrix_Type,
                                    T, 0> {
  static void compute(const Upper_Triangular_Matrix_Type &R,
                      const Matrix_Type &matrix_in, Matrix_Type &matrix_out,
                      const T &division_min) {

    constexpr std::size_t Row_Index = Matrix_Type::ROWS - 1;

    BackwardSubstitution_I_Loop<Upper_Triangular_Matrix_Type, Matrix_Type, T,
                                Row_Index,
                                (Matrix_Type::COLS - 1)>::compute(R, matrix_in,
                                                                  matrix_out,
                                                                  division_min);
  }
};

template <typename Upper_Triangular_Matrix_Type, typename Matrix_Type,
          typename T>
inline auto backward_substitution(const Upper_Triangular_Matrix_Type &R,
                                  const Matrix_Type &matrix_in,
                                  const T &division_min) -> Matrix_Type {
  static_assert(Upper_Triangular_Matrix_Type::COLS == Matrix_Type::COLS,
                "The number of columns in the upper triangular matrix R must "
                "match the number of columns in the input matrix.");
  static_assert(Upper_Triangular_Matrix_Type::COLS ==
                    Upper_Triangular_Matrix_Type::ROWS,
                "The upper triangular matrix R must be square.");

  Matrix_Type matrix_out;

  /*
    for (std::size_t row_index = 0; row_index < Matrix_Type::ROWS; ++row_index)
    {

      for (std::size_t i = Matrix_Type::COLS; i-- > 0;) {

        typename Matrix_Type::Value_Type sum = matrix_in(i, row_index);

        for (std::size_t j = i + 1; j < Matrix_Type::ROWS; ++j) {
          sum -= R(i, j) * matrix_out(j, row_index);
        }
        matrix_out(i, row_index) =
            sum / Base::Utility::avoid_zero_divide(R(i, i), division_min);
      }
    }
  */

  BackwardSubstitution_RowLoop<Upper_Triangular_Matrix_Type, Matrix_Type, T,
                               (Matrix_Type::ROWS - 1)>::compute(R, matrix_in,
                                                                 matrix_out,
                                                                 division_min);

  return matrix_out;
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

protected:
  /* Type */
  using _R_TriangluarRowIndices = Base::Matrix::UpperTriangularRowIndices<M, N>;
  using _R_TriangluarRowPointers =
      Base::Matrix::UpperTriangularRowPointers<M, N>;

public:
  /* Constructor */
  LinalgSolverQR()
      : _QR_decomposer(),
        _R_triangular(Base::Matrix::TriangularSparse<T, M, N>::create_upper()) {
  }

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
  /* Solve function */

  /**
   * @brief Solves the QR decomposition for the given matrix A.
   *
   * This function sets the internal R matrix to the provided matrix A
   * and performs QR decomposition by calling the internal _decompose method.
   *
   * @param A The input matrix to decompose.
   */
  inline void solve(const Matrix<DefDense, T, M, N> &A) {
    this->_QR_decomposer.solve(A.matrix);
  }

  /* Get Q, R */

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

    Base::Matrix::TriangularSparse<T, M, N>::set_values_upper(
        this->_R_triangular, this->_QR_decomposer.get_R());

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
   * @brief Solves the QR decomposition for the given sparse matrix A.
   *
   * This function sets the internal R matrix to the provided sparse matrix A
   * and performs QR decomposition by calling the internal _decompose method.
   *
   * @param A The input sparse matrix to decompose.
   */
  inline void solve(const Matrix<DefSparse, T, M, N, SparseAvailable> &A) {
    this->_QR_decomposer.solve(A.matrix);
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

    Base::Matrix::TriangularSparse<T, M, N>::set_values_upper(
        this->_R_triangular, this->_QR_decomposer.get_R());

    return Base::Matrix::CompiledSparseMatrix<
        T, M, N, Base::Matrix::UpperTriangularRowIndices<M, M>,
        _R_TriangluarRowPointers>(this->_R_triangular);
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
      _R_triangular = Base::Matrix::TriangularSparse<T, M, N>::create_upper();
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
