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
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__
#define __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__

#include "python_numpy_base.hpp"
#include "python_numpy_complex.hpp"
#include "python_numpy_templates.hpp"

#include "python_numpy_concatenate.hpp"

#include <initializer_list>
#include <tuple>
#include <type_traits>

namespace PythonNumpy {

/**
 * @brief Creates a dense matrix of zeros.
 *
 * This function template constructs and returns a dense matrix of the specified
 * type and dimensions, initialized with zeros. The matrix type is determined by
 * the template parameters.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
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
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
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
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
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
 * @tparam M The number of columns in the matrix.
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
 * @tparam M The number of columns in the matrix.
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
 * @tparam M The number of columns in the matrix.
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
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
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
 * @tparam ColumnCount The current column index for assignment.
 * @tparam RowCount The current row index for assignment.
 * @tparam DenseMatrix_Type The type of the dense matrix.
 * @tparam T The type of the first value to assign.
 * @param matrix The dense matrix to which values are assigned.
 * @param value_1 The first value to assign.
 */
template <std::size_t ColumnCount, std::size_t RowCount,
          typename DenseMatrix_Type, typename T>
inline void assign_values(DenseMatrix_Type &matrix, T value_1) {

  static_assert(ColumnCount < DenseMatrix_Type::COLS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");
  static_assert(RowCount < DenseMatrix_Type::ROWS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");

  matrix.template set<ColumnCount, RowCount>(value_1);
}

/**
 * @brief Assigns multiple values to a dense matrix.
 *
 * This function template assigns multiple values to a dense matrix at
 * specified indices. It supports variadic arguments for multiple values and
 * ensures that all values are of the same type.
 *
 * @tparam ColumnCount The current column index for assignment.
 * @tparam RowCount The current row index for assignment.
 * @tparam DenseMatrix_Type The type of the dense matrix.
 * @tparam T The type of the first value to assign.
 * @tparam U The type of the second value to assign.
 * @param matrix The dense matrix to which values are assigned.
 * @param value_1 The first value to assign.
 * @param value_2 The second value to assign.
 * @param args Additional values to assign, if any.
 */
template <std::size_t ColumnCount, std::size_t RowCount,
          typename DenseMatrix_Type, typename T, typename U, typename... Args>
inline void assign_values(DenseMatrix_Type &matrix, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(ColumnCount < DenseMatrix_Type::COLS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");
  static_assert(RowCount < DenseMatrix_Type::ROWS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");

  matrix.template set<ColumnCount, RowCount>(value_1);

  assign_values<ColumnCount + 1 * (RowCount == (DenseMatrix_Type::ROWS - 1)),
                ((RowCount + 1) * (RowCount != (DenseMatrix_Type::ROWS - 1)))>(
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
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
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

  static_assert(IndexCount < DiagMatrix_Type::COLS,
                "Number of arguments must be less than the number of columns.");

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
  static_assert(IndexCount < DiagMatrix_Type::COLS,
                "Number of arguments must be less than the number of columns.");

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
 * @tparam M The number of columns in the matrix.
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
 * @return Matrix<DefSparse, T, SparseAvailable::number_of_columns,
 * SparseAvailable::column_size, SparseAvailable> A sparse matrix of zeros with
 * dimensions defined by SparseAvailable.
 */
template <typename T, typename SparseAvailable>
inline auto make_SparseMatrixZeros(void)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  Matrix<DefSparse, T, SparseAvailable::number_of_columns,
         SparseAvailable::column_size, SparseAvailable>
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
 * @return Matrix<DefSparse, T, SparseAvailable::number_of_columns,
 * SparseAvailable::column_size, SparseAvailable> A sparse matrix of ones with
 * dimensions defined by SparseAvailable.
 */
template <typename T, typename SparseAvailable>
inline auto make_SparseMatrixOnes(void)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  return Matrix<DefSparse, T, SparseAvailable::number_of_columns,
                SparseAvailable::column_size,
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
 * @return Matrix<DefSparse, T, SparseAvailable::number_of_columns,
 * SparseAvailable::column_size, SparseAvailable> A sparse matrix filled with
 * the specified value, with dimensions defined by SparseAvailable.
 */
template <typename SparseAvailable, typename T>
inline auto make_SparseMatrixFull(const T &value)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  return Matrix<DefSparse, T, SparseAvailable::number_of_columns,
                SparseAvailable::column_size, SparseAvailable>::full(value);
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
 * @return Matrix<DefSparse, T, SparseAvailable::number_of_columns,
 * SparseAvailable::column_size, SparseAvailable> A sparse matrix filled with
 * the specified values, with dimensions defined by SparseAvailable.
 */
template <typename SparseAvailable, typename T, typename... Args>
inline auto make_SparseMatrix(T value_1, Args... args)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  Matrix<DefSparse, T, SparseAvailable::number_of_columns,
         SparseAvailable::column_size, SparseAvailable>
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
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
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

#endif // __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__
