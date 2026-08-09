/**
 * @file python_numpy_base_simplification.hpp
 * @brief Provides a set of utility functions and type aliases for creating and
 * manipulating dense, diagonal, and sparse matrices in a style similar to
 * NumPy, using C++ templates.
 *
 * This header defines the `PythonNumpy` namespace, which contains a collection
 * of template functions and type aliases to facilitate the creation and
 * initialization of matrices with various storage types (dense, diagonal,
 * sparse). The utilities support zero, one, full, and custom value
 * initialization, as well as assignment of values to matrix elements in a
 * variadic, type-safe manner. The design is inspired by Python's NumPy library,
 * aiming to provide a familiar and expressive interface for matrix operations
 * in C++.
 *
 * @note
 * tparam M is the number of rows in the matrix.
 * tparam N is the number of columns in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP_
#define PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP_

#include "python_numpy_base.hpp"
#include "python_numpy_complex.hpp"
#include "python_numpy_templates.hpp"

#include "python_numpy_concatenate.hpp"

#include <initializer_list>
#include <tuple>
#include <type_traits>

namespace PythonNumpy {

// Helper struct to check if a type is std::array
template <typename T> struct is_std_array : std::false_type {};

template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

/**
 * @brief Creates a dense matrix of zeros.
 *
 * This function template constructs and returns a dense matrix of the specified
 * type and dimensions, initialized with zeros. The matrix type is determined by
 * the template parameters.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @return Matrix<DefDense, T, M, N> A dense matrix of zeros with dimensions M x
 * N.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto make_DenseMatrixZeros(void) -> Matrix<DefDense, T, M, N> {

  Matrix<DefDense, T, M, N> result;
  return result;
}

/**
 * @brief Creates a dense matrix of ones.
 *
 * This function template constructs and returns a dense matrix of the specified
 * type and dimensions, initialized with ones. The matrix type is determined by
 * the template parameters.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @return Matrix<DefDense, T, M, N> A dense matrix of ones with dimensions M x
 * N.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto make_DenseMatrixOnes(void) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>::ones();
}

/**
 * @brief Creates a dense matrix filled with a specified value.
 *
 * This function template constructs and returns a dense matrix of the specified
 * type and dimensions, initialized with a given value. The matrix type is
 * determined by the template parameters.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam T The data type of the matrix elements.
 * @param value The value to fill the matrix with.
 * @return Matrix<DefDense, T, M, N> A dense matrix filled with the specified
 * value, with dimensions M x N.
 */
template <std::size_t M, std::size_t N, typename T>
inline auto make_DenseMatrixFull(const T &value) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>::full(value);
}

/**
 * @brief Creates a diagonal matrix of zeros.
 *
 * This function template constructs and returns a diagonal matrix of the
 * specified type and dimensions, initialized with zeros. The matrix type is
 * determined by the template parameters.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @return Matrix<DefDiag, T, M> A diagonal matrix of zeros with dimensions M x
 * M.
 */
template <typename T, std::size_t M>
inline auto make_DiagMatrixZeros(void) -> Matrix<DefDiag, T, M> {

  Matrix<DefDiag, T, M> result;
  return result;
}

/**
 * @brief Creates a diagonal matrix of ones.
 *
 * This function template constructs and returns a diagonal matrix of the
 * specified type and dimensions, initialized with ones. The matrix type is
 * determined by the template parameters.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @return Matrix<DefDiag, T, M> A diagonal matrix of ones with dimensions M x
 * M.
 */
template <typename T, std::size_t M>
inline auto make_DiagMatrixIdentity(void) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>::identity();
}

/**
 * @brief Creates a diagonal matrix filled with a specified value.
 *
 * This function template constructs and returns a diagonal matrix of the
 * specified type and dimensions, initialized with a given value. The matrix
 * type is determined by the template parameters.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam T The data type of the matrix elements.
 * @param value The value to fill the diagonal of the matrix with.
 * @return Matrix<DefDiag, T, M> A diagonal matrix filled with the specified
 * value, with dimensions M x M.
 */
template <std::size_t M, typename T>
inline auto make_DiagMatrixFull(const T &value) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>::full(value);
}

/**
 * @brief Creates an empty sparse matrix.
 *
 * This function template constructs and returns an empty sparse matrix of the
 * specified type and dimensions. The matrix type is determined by the template
 * parameters.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @return Matrix<DefSparse, T, M, N, SparseAvailableEmpty<M, N>> An empty
 * sparse matrix with dimensions M x N.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto make_SparseMatrixEmpty(void)
    -> Matrix<DefSparse, T, M, N, SparseAvailableEmpty<M, N>> {

  return Matrix<DefSparse, T, M, N, SparseAvailableEmpty<M, N>>();
}

namespace MakeDenseMatrixOperation {

/**
 * @brief Assigns values to a dense matrix.
 *
 * This function template assigns values to a dense matrix at specified
 * indices. It supports variadic arguments for multiple values.
 *
 * @tparam RowCount The current row index for assignment.
 * @tparam ColumnCount The current column index for assignment.
 * @tparam DenseMatrix_Type The type of the dense matrix.
 * @tparam T The type of the first value to assign.
 * @param matrix The dense matrix to which values are assigned.
 * @param value_1 The first value to assign.
 */
template <std::size_t RowCount, std::size_t ColumnCount,
          typename DenseMatrix_Type, typename T>
inline void assign_values(DenseMatrix_Type &matrix, T value_1) {

  static_assert(RowCount < DenseMatrix_Type::ROWS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");
  static_assert(ColumnCount < DenseMatrix_Type::COLS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");

  matrix.template set<RowCount, ColumnCount>(value_1);
}

/**
 * @brief Assigns multiple values to a dense matrix.
 *
 * This function template assigns multiple values to a dense matrix at
 * specified indices. It supports variadic arguments for multiple values and
 * ensures that all values are of the same type.
 *
 * @tparam RowCount The current row index for assignment.
 * @tparam ColumnCount The current column index for assignment.
 * @tparam DenseMatrix_Type The type of the dense matrix.
 * @tparam T The type of the first value to assign.
 * @tparam U The type of the second value to assign.
 * @param matrix The dense matrix to which values are assigned.
 * @param value_1 The first value to assign.
 * @param value_2 The second value to assign.
 * @param args Additional values to assign, if any.
 */
template <std::size_t RowCount, std::size_t ColumnCount,
          typename DenseMatrix_Type, typename T, typename U, typename... Args>
inline void assign_values(DenseMatrix_Type &matrix, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(RowCount < DenseMatrix_Type::ROWS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");
  static_assert(ColumnCount < DenseMatrix_Type::COLS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");

  matrix.template set<RowCount, ColumnCount>(value_1);

  assign_values<RowCount + 1 * (ColumnCount == (DenseMatrix_Type::COLS - 1)),
                ((ColumnCount + 1) *
                 (ColumnCount != (DenseMatrix_Type::COLS - 1)))>(
      matrix, value_2, args...);
}

} // namespace MakeDenseMatrixOperation

/**
 * @brief Creates a dense matrix with specified values.
 *
 * This function template constructs and returns a dense matrix of the specified
 * type and dimensions, initialized with given values. The matrix type is
 * determined by the template parameters.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam T The data type of the matrix elements.
 * @param value_1 The first value to fill the matrix with.
 * @param args Additional values to fill the matrix with, if any.
 * @return Matrix<DefDense, T, M, N> A dense matrix filled with the specified
 * values, with dimensions M x N.
 */
template <std::size_t M, std::size_t N, typename T, typename... Args>
inline auto make_DenseMatrix(T value_1, Args... args)
    -> Matrix<DefDense, T, M, N> {

  Matrix<DefDense, T, M, N> result;

  MakeDenseMatrixOperation::assign_values<0, 0>(result, value_1, args...);

  return result;
}

namespace MakeDenseMatrixFromRowMajorArray {

/**
 * @brief Recursively assigns values to matrix columns from a std::array.
 *
 * This struct handles the column iteration when loading from row_major array.
 * It recursively processes columns from N-1 down to 0 for each row.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam I The current row index in the recursion.
 * @tparam J_idx The current column index in the recursion.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Row {
  /**
   * @brief Recursively assigns a value to the matrix from row_major array.
   *
   * @param row_major The source row-major array.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::array<std::array<T, N>, M> &row_major,
                      Matrix<DefDense, T, M, N> &result) {
    result.template set<I, J_idx>(row_major[I][J_idx]);
    Row<T, M, N, I, J_idx - 1>::compute(row_major, result);
  }
};

/**
 * @brief Base case for column recursion from row_major array.
 *
 * This specialization terminates the column recursion when J_idx == 0.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Row<T, M, N, I, 0> {
  /**
   * @brief Assigns the first column value and terminates recursion.
   *
   * @param row_major The source row-major array.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::array<std::array<T, N>, M> &row_major,
                      Matrix<DefDense, T, M, N> &result) {
    result.template set<I, 0>(row_major[I][0]);
  }
};

/**
 * @brief Recursively assigns values to matrix rows from a std::array.
 *
 * This struct handles the row iteration when loading from row_major array.
 * It processes rows from M-1 down to 0.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam I_idx The current row index in the recursion.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Column {
  /**
   * @brief Recursively processes a row and continues to the next.
   *
   * @param row_major The source row-major array.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::array<std::array<T, N>, M> &row_major,
                      Matrix<DefDense, T, M, N> &result) {
    Row<T, M, N, I_idx, N - 1>::compute(row_major, result);
    Column<T, M, N, I_idx - 1>::compute(row_major, result);
  }
};

/**
 * @brief Base case for row recursion from row_major array.
 *
 * This specialization terminates the row recursion when I_idx == 0.
 */
template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, 0> {
  /**
   * @brief Processes the first row and terminates recursion.
   *
   * @param row_major The source row-major array.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::array<std::array<T, N>, M> &row_major,
                      Matrix<DefDense, T, M, N> &result) {
    Row<T, M, N, 0, N - 1>::compute(row_major, result);
  }
};

/**
 * @brief Initiates the recursive assignment of values from row_major array.
 *
 * This function uses template metaprogramming to unroll loops and assign values
 * from a std::array to a dense matrix.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param row_major The source row-major array.
 * @param result The target matrix to assign values to.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const std::array<std::array<T, N>, M> &row_major,
                    Matrix<DefDense, T, M, N> &result) {
  Column<T, M, N, M - 1>::compute(row_major, result);
}

} // namespace MakeDenseMatrixFromRowMajorArray

namespace MakeDenseMatrixFromRowMajorVector {

/**
 * @brief Recursively assigns values to matrix columns from a std::vector.
 *
 * This struct handles the column iteration when loading from row_major vector.
 * It recursively processes columns from N-1 down to 0 for each row.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam I The current row index in the recursion.
 * @tparam J_idx The current column index in the recursion.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Row {
  /**
   * @brief Recursively assigns a value to the matrix from row_major vector.
   *
   * @param row_major The source row-major vector.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::vector<std::vector<T>> &row_major,
                      Matrix<DefDense, T, M, N> &result) {
    result.template set<I, J_idx>(row_major[I][J_idx]);
    Row<T, M, N, I, J_idx - 1>::compute(row_major, result);
  }
};

/**
 * @brief Base case for column recursion from row_major vector.
 *
 * This specialization terminates the column recursion when J_idx == 0.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Row<T, M, N, I, 0> {
  /**
   * @brief Assigns the first column value and terminates recursion.
   *
   * @param row_major The source row-major vector.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::vector<std::vector<T>> &row_major,
                      Matrix<DefDense, T, M, N> &result) {
    result.template set<I, 0>(row_major[I][0]);
  }
};

/**
 * @brief Recursively assigns values to matrix rows from a std::vector.
 *
 * This struct handles the row iteration when loading from row_major vector.
 * It processes rows from M-1 down to 0.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam I_idx The current row index in the recursion.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Column {
  /**
   * @brief Recursively processes a row and continues to the next.
   *
   * @param row_major The source row-major vector.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::vector<std::vector<T>> &row_major,
                      Matrix<DefDense, T, M, N> &result) {
    Row<T, M, N, I_idx, N - 1>::compute(row_major, result);
    Column<T, M, N, I_idx - 1>::compute(row_major, result);
  }
};

/**
 * @brief Base case for row recursion from row_major vector.
 *
 * This specialization terminates the row recursion when I_idx == 0.
 */
template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, 0> {
  /**
   * @brief Processes the first row and terminates recursion.
   *
   * @param row_major The source row-major vector.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::vector<std::vector<T>> &row_major,
                      Matrix<DefDense, T, M, N> &result) {
    Row<T, M, N, 0, N - 1>::compute(row_major, result);
  }
};

/**
 * @brief Initiates the recursive assignment of values from row_major vector.
 *
 * This function uses template metaprogramming to unroll loops and assign values
 * from a std::vector to a dense matrix.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param row_major The source row-major vector.
 * @param result The target matrix to assign values to.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const std::vector<std::vector<T>> &row_major,
                    Matrix<DefDense, T, M, N> &result) {
  Column<T, M, N, M - 1>::compute(row_major, result);
}

} // namespace MakeDenseMatrixFromRowMajorVector

/**
 * @brief Creates a dense matrix from a std::array in row-major format.
 *
 * This function template constructs and returns a dense matrix initialized
 * from a std::array<std::array<T, N>, M> with row-major layout. Uses template
 * metaprogramming to unroll the initialization loops at compile-time.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param row_major The source row-major array.
 * @return Matrix<DefDense, T, M, N> A dense matrix initialized from row_major.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto make_DenseMatrix_from_row_major(
    const std::array<std::array<T, N>, M> &row_major)
    -> Matrix<DefDense, T, M, N> {

  Matrix<DefDense, T, M, N> result;
  MakeDenseMatrixFromRowMajorArray::compute(row_major, result);
  return result;
}

namespace MakeDenseMatrixFromRowMajorArray1D {

// Vector Add Scalar Core Template: M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  /**
   * @brief Assigns a value from a 1D array to the matrix.
   *
   * This function recursively assigns values from a 1D std::array to a dense
   * matrix, starting from the last index and moving towards the first.
   *
   * @param row_major The source 1D array in row-major format.
   * @param result The target dense matrix to assign values to.
   */
  static void compute(const std::array<T, M> &row_major,
                      Matrix<DefDense, T, 1, M> &result) {
    result.template set<0, M_idx>(row_major[M_idx]);
    Core<T, M, M_idx - 1>::compute(row_major, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  /**
   * @brief Assigns the first value from a 1D array to the matrix.
   *
   * This function is called when the recursion reaches the first index, and it
   * assigns the first value from the 1D std::array to the dense matrix.
   *
   * @param row_major The source 1D array in row-major format.
   * @param result The target dense matrix to assign values to.
   */
  static void compute(const std::array<T, M> &row_major,
                      Matrix<DefDense, T, 1, M> &result) {
    result.template set<0, 0>(row_major[0]);
  }
};

/**
 * @brief Initiates the recursive assignment of values from a 1D array.
 *
 * This function uses template metaprogramming to unroll loops and assign values
 * from a 1D std::array to a dense matrix.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @param row_major The source 1D array in row-major format.
 * @param result The target dense matrix to assign values to.
 */
template <typename T, std::size_t M>
inline void compute(const std::array<T, M> &row_major,
                    Matrix<DefDense, T, 1, M> &result) {
  Core<T, M, M - 1>::compute(row_major, result);
}

} // namespace MakeDenseMatrixFromRowMajorArray1D

template <typename T, std::size_t M>
inline auto make_DenseMatrix_from_row_major(
    const std::array<T, M> &row_major,
    typename std::enable_if<!is_std_array<T>::value>::type * = nullptr)
    -> Matrix<DefDense, T, 1, M> {

  Matrix<DefDense, T, 1, M> result;
  MakeDenseMatrixFromRowMajorArray1D::compute(row_major, result);
  return result;
}

/**
 * @brief Creates a dense matrix from a std::vector in row-major format.
 *
 * This function template constructs and returns a dense matrix initialized
 * from a std::vector<std::vector<T>> with row-major layout. Uses template
 * metaprogramming to unroll the initialization loops at compile-time.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param row_major The source row-major vector.
 * @return Matrix<DefDense, T, M, N> A dense matrix initialized from row_major.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto
make_DenseMatrix_from_row_major(const std::vector<std::vector<T>> &row_major)
    -> Matrix<DefDense, T, M, N> {

  Matrix<DefDense, T, M, N> result;
  MakeDenseMatrixFromRowMajorVector::compute(row_major, result);
  return result;
}

namespace MakeDenseMatrixFromColMajorArray {

/**
 * @brief Recursively assigns values to matrix rows from a std::array.
 *
 * This struct handles the row iteration when loading from col_major array.
 * It recursively processes rows from M-1 down to 0 for each column.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam J The current column index in the recursion.
 * @tparam I_idx The current row index in the recursion.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t I_idx>
struct Row {
  /**
   * @brief Recursively assigns a value to the matrix from col_major array.
   *
   * @param col_major The source column-major array.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::array<std::array<T, M>, N> &col_major,
                      Matrix<DefDense, T, M, N> &result) {
    result.template set<I_idx, J>(col_major[J][I_idx]);
    Row<T, M, N, J, I_idx - 1>::compute(col_major, result);
  }
};

/**
 * @brief Base case for row recursion from col_major array.
 *
 * This specialization terminates the row recursion when I_idx == 0.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct Row<T, M, N, J, 0> {
  /**
   * @brief Assigns the first row value and terminates recursion.
   *
   * @param col_major The source column-major array.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::array<std::array<T, M>, N> &col_major,
                      Matrix<DefDense, T, M, N> &result) {
    result.template set<0, J>(col_major[J][0]);
  }
};

/**
 * @brief Recursively assigns values to matrix columns from a std::array.
 *
 * This struct handles the column iteration when loading from col_major array.
 * It processes columns from N-1 down to 0.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam J_idx The current column index in the recursion.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t J_idx>
struct Column {
  /**
   * @brief Recursively processes a column and continues to the next.
   *
   * @param col_major The source column-major array.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::array<std::array<T, M>, N> &col_major,
                      Matrix<DefDense, T, M, N> &result) {
    Row<T, M, N, J_idx, M - 1>::compute(col_major, result);
    Column<T, M, N, J_idx - 1>::compute(col_major, result);
  }
};

/**
 * @brief Base case for column recursion from col_major array.
 *
 * This specialization terminates the column recursion when J_idx == 0.
 */
template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, 0> {
  /**
   * @brief Processes the first column and terminates recursion.
   *
   * @param col_major The source column-major array.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::array<std::array<T, M>, N> &col_major,
                      Matrix<DefDense, T, M, N> &result) {
    Row<T, M, N, 0, M - 1>::compute(col_major, result);
  }
};

/**
 * @brief Initiates the recursive assignment of values from col_major array.
 *
 * This function uses template metaprogramming to unroll loops and assign values
 * from a std::array to a dense matrix.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param col_major The source column-major array.
 * @param result The target matrix to assign values to.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const std::array<std::array<T, M>, N> &col_major,
                    Matrix<DefDense, T, M, N> &result) {
  Column<T, M, N, N - 1>::compute(col_major, result);
}

} // namespace MakeDenseMatrixFromColMajorArray

namespace MakeDenseMatrixFromColMajorVector {

/**
 * @brief Recursively assigns values to matrix rows from a std::vector.
 *
 * This struct handles the row iteration when loading from col_major vector.
 * It recursively processes rows from M-1 down to 0 for each column.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam J The current column index in the recursion.
 * @tparam I_idx The current row index in the recursion.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t I_idx>
struct Row {
  /**
   * @brief Recursively assigns a value to the matrix from col_major vector.
   *
   * @param col_major The source column-major vector.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::vector<std::vector<T>> &col_major,
                      Matrix<DefDense, T, M, N> &result) {
    result.template set<I_idx, J>(col_major[J][I_idx]);
    Row<T, M, N, J, I_idx - 1>::compute(col_major, result);
  }
};

/**
 * @brief Base case for row recursion from col_major vector.
 *
 * This specialization terminates the row recursion when I_idx == 0.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct Row<T, M, N, J, 0> {
  /**
   * @brief Assigns the first row value and terminates recursion.
   *
   * @param col_major The source column-major vector.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::vector<std::vector<T>> &col_major,
                      Matrix<DefDense, T, M, N> &result) {
    result.template set<0, J>(col_major[J][0]);
  }
};

/**
 * @brief Recursively assigns values to matrix columns from a std::vector.
 *
 * This struct handles the column iteration when loading from col_major vector.
 * It processes columns from N-1 down to 0.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam J_idx The current column index in the recursion.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t J_idx>
struct Column {
  /**
   * @brief Recursively processes a column and continues to the next.
   *
   * @param col_major The source column-major vector.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::vector<std::vector<T>> &col_major,
                      Matrix<DefDense, T, M, N> &result) {
    Row<T, M, N, J_idx, M - 1>::compute(col_major, result);
    Column<T, M, N, J_idx - 1>::compute(col_major, result);
  }
};

/**
 * @brief Base case for column recursion from col_major vector.
 *
 * This specialization terminates the column recursion when J_idx == 0.
 */
template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, 0> {
  /**
   * @brief Processes the first column and terminates recursion.
   *
   * @param col_major The source column-major vector.
   * @param result The target matrix to assign values to.
   */
  static void compute(const std::vector<std::vector<T>> &col_major,
                      Matrix<DefDense, T, M, N> &result) {
    Row<T, M, N, 0, M - 1>::compute(col_major, result);
  }
};

/**
 * @brief Initiates the recursive assignment of values from col_major vector.
 *
 * This function uses template metaprogramming to unroll loops and assign values
 * from a std::vector to a dense matrix.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param col_major The source column-major vector.
 * @param result The target matrix to assign values to.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const std::vector<std::vector<T>> &col_major,
                    Matrix<DefDense, T, M, N> &result) {
  Column<T, M, N, N - 1>::compute(col_major, result);
}

} // namespace MakeDenseMatrixFromColMajorVector

/**
 * @brief Creates a dense matrix from a std::array in column-major format.
 *
 * This function template constructs and returns a dense matrix initialized
 * from a std::array<std::array<T, M>, N> with column-major layout. Uses
 * template metaprogramming to unroll the initialization loops at compile-time.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param col_major The source column-major array.
 * @return Matrix<DefDense, T, M, N> A dense matrix initialized from col_major.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto make_DenseMatrix_from_col_major(
    const std::array<std::array<T, M>, N> &col_major)
    -> Matrix<DefDense, T, M, N> {

  Matrix<DefDense, T, M, N> result;
  MakeDenseMatrixFromColMajorArray::compute(col_major, result);
  return result;
}

/**
 * @brief Creates a dense matrix from a std::vector in column-major format.
 *
 * This function template constructs and returns a dense matrix initialized
 * from a std::vector<std::vector<T>> with column-major layout. Uses template
 * metaprogramming to unroll the initialization loops at compile-time.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param col_major The source column-major vector.
 * @return Matrix<DefDense, T, M, N> A dense matrix initialized from col_major.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto
make_DenseMatrix_from_col_major(const std::vector<std::vector<T>> &col_major)
    -> Matrix<DefDense, T, M, N> {

  Matrix<DefDense, T, M, N> result;
  MakeDenseMatrixFromColMajorVector::compute(col_major, result);
  return result;
}

namespace MakeDiagMatrixOperation {

/**
 * @brief Assigns values to a diagonal matrix.
 *
 * This function template assigns values to a diagonal matrix at specified
 * indices. It supports variadic arguments for multiple values.
 *
 * @tparam IndexCount The current index for assignment.
 * @tparam DiagMatrix_Type The type of the diagonal matrix.
 * @tparam T The type of the first value to assign.
 * @param matrix The diagonal matrix to which values are assigned.
 * @param value_1 The first value to assign.
 */
template <std::size_t IndexCount, typename DiagMatrix_Type, typename T>
inline void assign_values(DiagMatrix_Type &matrix, T value_1) {

  static_assert(IndexCount < DiagMatrix_Type::ROWS,
                "Number of arguments must be less than the number of rows.");

  matrix.template set<IndexCount, IndexCount>(value_1);
}

/**
 * @brief Assigns multiple values to a diagonal matrix.
 *
 * This function template assigns multiple values to a diagonal matrix at
 * specified indices. It supports variadic arguments for multiple values and
 * ensures that all values are of the same type.
 *
 * @tparam IndexCount The current index for assignment.
 * @tparam DiagMatrix_Type The type of the diagonal matrix.
 * @tparam T The type of the first value to assign.
 * @tparam U The type of the second value to assign.
 * @param matrix The diagonal matrix to which values are assigned.
 * @param value_1 The first value to assign.
 * @param value_2 The second value to assign.
 * @param args Additional values to assign, if any.
 */
template <std::size_t IndexCount, typename DiagMatrix_Type, typename T,
          typename U, typename... Args>
inline void assign_values(DiagMatrix_Type &matrix, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(IndexCount < DiagMatrix_Type::ROWS,
                "Number of arguments must be less than the number of rows.");

  matrix.template set<IndexCount, IndexCount>(value_1);

  assign_values<IndexCount + 1>(matrix, value_2, args...);
}

} // namespace MakeDiagMatrixOperation

/**
 * @brief Creates a diagonal matrix with specified values.
 *
 * This function template constructs and returns a diagonal matrix of the
 * specified type and dimensions, initialized with given values. The matrix type
 * is determined by the template parameters.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam T The data type of the matrix elements.
 * @param value_1 The first value to fill the diagonal with.
 * @param args Additional values to fill the diagonal with, if any.
 * @return Matrix<DefDiag, T, M> A diagonal matrix filled with the specified
 * values, with dimensions M x M.
 */
template <std::size_t M, typename T, typename... Args>
inline auto make_DiagMatrix(T value_1, Args... args) -> Matrix<DefDiag, T, M> {

  Matrix<DefDiag, T, M> result;

  MakeDiagMatrixOperation::assign_values<0>(result, value_1, args...);

  return result;
}

/**
 * @brief Creates a sparse matrix of zeros.
 *
 * This function template constructs and returns a sparse matrix of the
 * specified type and dimensions, initialized with zeros. The matrix type is
 * determined by the template parameters.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam SparseAvailable The sparse matrix availability type.
 * @return Matrix<DefSparse, T, SparseAvailable::number_of_rows,
 * SparseAvailable::row_size, SparseAvailable> A sparse matrix of zeros with
 * dimensions defined by SparseAvailable.
 */
template <typename T, typename SparseAvailable>
inline auto make_SparseMatrixZeros(void)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_rows,
              SparseAvailable::row_size, SparseAvailable> {

  Matrix<DefSparse, T, SparseAvailable::number_of_rows,
         SparseAvailable::row_size, SparseAvailable>
      result;

  return result;
}

/**
 * @brief Creates a sparse matrix of ones.
 *
 * This function template constructs and returns a sparse matrix of the
 * specified type and dimensions, initialized with ones. The matrix type is
 * determined by the template parameters.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam SparseAvailable The sparse matrix availability type.
 * @return Matrix<DefSparse, T, SparseAvailable::number_of_rows,
 * SparseAvailable::row_size, SparseAvailable> A sparse matrix of ones with
 * dimensions defined by SparseAvailable.
 */
template <typename T, typename SparseAvailable>
inline auto make_SparseMatrixOnes(void)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_rows,
              SparseAvailable::row_size, SparseAvailable> {

  return Matrix<DefSparse, T, SparseAvailable::number_of_rows,
                SparseAvailable::row_size,
                SparseAvailable>::full(static_cast<T>(1));
}

/**
 * @brief Creates a sparse matrix filled with a specified value.
 *
 * This function template constructs and returns a sparse matrix of the
 * specified type and dimensions, initialized with a given value. The matrix
 * type is determined by the template parameters.
 *
 * @tparam SparseAvailable The sparse matrix availability type.
 * @tparam T The data type of the matrix elements.
 * @param value The value to fill the sparse matrix with.
 * @return Matrix<DefSparse, T, SparseAvailable::number_of_rows,
 * SparseAvailable::row_size, SparseAvailable> A sparse matrix filled with
 * the specified value, with dimensions defined by SparseAvailable.
 */
template <typename SparseAvailable, typename T>
inline auto make_SparseMatrixFull(const T &value)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_rows,
              SparseAvailable::row_size, SparseAvailable> {

  return Matrix<DefSparse, T, SparseAvailable::number_of_rows,
                SparseAvailable::row_size, SparseAvailable>::full(value);
}

namespace MakeSparseMatrixOperation {

/**
 * @brief Assigns values to a sparse matrix.
 *
 * This function template assigns values to a sparse matrix at specified
 * indices. It supports variadic arguments for multiple values.
 *
 * @tparam IndexCount The current index for assignment.
 * @tparam SparseMatrix_Type The type of the sparse matrix.
 * @tparam T The type of the first value to assign.
 * @param matrix The sparse matrix to which values are assigned.
 * @param value_1 The first value to assign.
 */
template <std::size_t IndexCount, typename SparseMatrix_Type, typename T>
inline void assign_values(SparseMatrix_Type &matrix, T value_1) {

  static_assert(IndexCount < SparseMatrix_Type::NumberOfValues,
                "Number of arguments must be the same or less than the number "
                "of elements of Sparse Matrix.");

  matrix.template set<IndexCount>(value_1);
}

/**
 * @brief Assigns multiple values to a sparse matrix.
 *
 * This function template assigns multiple values to a sparse matrix at
 * specified indices. It supports variadic arguments for multiple values and
 * ensures that all values are of the same type.
 *
 * @tparam IndexCount The current index for assignment.
 * @tparam SparseMatrix_Type The type of the sparse matrix.
 * @tparam T The type of the first value to assign.
 * @tparam U The type of the second value to assign.
 * @param matrix The sparse matrix to which values are assigned.
 * @param value_1 The first value to assign.
 * @param value_2 The second value to assign.
 * @param args Additional values to assign, if any.
 */
template <std::size_t IndexCount, typename SparseMatrix_Type, typename T,
          typename U, typename... Args>
inline void assign_values(SparseMatrix_Type &matrix, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");

  static_assert(IndexCount < SparseMatrix_Type::NumberOfValues,
                "Number of arguments must be the same or less than the number "
                "of elements of Sparse Matrix.");

  matrix.template set<IndexCount>(value_1);

  assign_values<IndexCount + 1>(matrix, value_2, args...);
}

} // namespace MakeSparseMatrixOperation

/**
 * @brief Creates a sparse matrix with specified values.
 *
 * This function template constructs and returns a sparse matrix of the
 * specified type and dimensions, initialized with given values. The matrix type
 * is determined by the template parameters.
 *
 * @tparam SparseAvailable The sparse matrix availability type.
 * @tparam T The data type of the matrix elements.
 * @param value_1 The first value to fill the sparse matrix with.
 * @param args Additional values to fill the sparse matrix with, if any.
 * @return Matrix<DefSparse, T, SparseAvailable::number_of_rows,
 * SparseAvailable::row_size, SparseAvailable> A sparse matrix filled with
 * the specified values, with dimensions defined by SparseAvailable.
 */
template <typename SparseAvailable, typename T, typename... Args>
inline auto make_SparseMatrix(T value_1, Args... args)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_rows,
              SparseAvailable::row_size, SparseAvailable> {

  Matrix<DefSparse, T, SparseAvailable::number_of_rows,
         SparseAvailable::row_size, SparseAvailable>
      result;

  MakeSparseMatrixOperation::assign_values<0>(result, value_1, args...);

  return result;
}

/**
 * @brief Creates a sparse matrix from a dense matrix.
 *
 * This function template constructs and returns a sparse matrix from a given
 * dense matrix. The dense matrix is converted to a sparse format, and the
 * resulting sparse matrix has the same dimensions as the dense matrix.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param dense_matrix The dense matrix to convert to a sparse matrix.
 * @return Matrix<DefSparse, T, M, N, DenseAvailable<M, N>> A sparse matrix
 * created from the dense matrix, with dimensions M x N.
 */
template <typename T, std::size_t M, std::size_t N, typename... Args>
inline auto
make_SparseMatrixFromDenseMatrix(Matrix<DefDense, T, M, N> &dense_matrix)
    -> Matrix<DefSparse, T, M, N, DenseAvailable<M, N>> {

  return Matrix<DefSparse, T, M, N, DenseAvailable<M, N>>(
      create_compiled_sparse(dense_matrix.matrix));
}

/* Type */
template <typename T, std::size_t M, std::size_t N>
using DenseMatrix_Type = Matrix<DefDense, T, M, N>;

template <typename T, std::size_t M>
using DiagMatrix_Type = Matrix<DefDiag, T, M>;

template <typename T, typename SparseAvailable>
using SparseMatrix_Type =
    decltype(make_SparseMatrixZeros<T, SparseAvailable>());

template <typename T, std::size_t M, std::size_t N>
using SparseMatrixEmpty_Type = decltype(make_SparseMatrixEmpty<T, M, N>());

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP_
