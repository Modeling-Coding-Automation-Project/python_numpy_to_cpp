/**
 * @file python_numpy_complex.hpp
 * @brief Utilities for handling real and imaginary parts of complex matrices in
 * the PythonNumpy namespace.
 *
 * This header provides a set of template structures and type traits for
 * extracting real and imaginary components from dense, diagonal, and sparse
 * matrices that may contain complex numbers. The utilities are designed to work
 * with matrices defined in the Base::Matrix namespace and support both real and
 * complex types through template specialization and SFINAE.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __PYTHON_NUMPY_COMPLEX_HPP__
#define __PYTHON_NUMPY_COMPLEX_HPP__

#include "base_matrix.hpp"
#include "python_numpy_templates.hpp"

namespace PythonNumpy {

template <typename T> using Complex = Base::Matrix::Complex<T>;

template <typename T> struct UnderlyingType {
  using Type = T;
};

template <typename U> struct UnderlyingType<Complex<U>> {
  using Type = U;
};

namespace ComplexOperation {

template <typename T, typename Complex_T, std::size_t M, std::size_t N,
          bool IsComplex>
struct GetRealFromComplexDenseMatrix {};

template <typename T, typename Complex_T, std::size_t M, std::size_t N>
struct GetRealFromComplexDenseMatrix<T, Complex_T, M, N, true> {

  /**
   * @brief Extracts the real part from a matrix of complex numbers.
   *
   * This static function takes a matrix of complex numbers as input and returns
   * a matrix containing only the real parts of each element, preserving the
   * original matrix dimensions.
   *
   * @tparam T The type of the real part of the complex numbers.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param input The input matrix containing complex numbers.
   * @return Base::Matrix::Matrix<T, M, N> A matrix containing the real parts of
   * the input matrix.
   */
  static Base::Matrix::Matrix<T, M, N>
  get(const Base::Matrix::Matrix<Complex_T, M, N> &input) {

    return Base::Matrix::get_real_matrix_from_complex_matrix(input);
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t N>
struct GetRealFromComplexDenseMatrix<T, Complex_T, M, N, false> {
  /**
   * @brief Returns the input matrix as it is when it is not complex.
   *
   * This static function simply returns the input matrix without any changes,
   * as it is assumed to be a matrix of real numbers.
   *
   * @tparam T The type of the elements in the matrix.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param input The input matrix containing real numbers.
   * @return Base::Matrix::Matrix<T, M, N> The input matrix itself.
   */
  static Base::Matrix::Matrix<T, M, N>
  get(const Base::Matrix::Matrix<T, M, N> &input) {

    return input.matrix;
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t N,
          bool IsComplex>
struct GetImagFromComplexDenseMatrix {};

template <typename T, typename Complex_T, std::size_t M, std::size_t N>
struct GetImagFromComplexDenseMatrix<T, Complex_T, M, N, true> {
  /**
   * @brief Extracts the imaginary part from a matrix of complex numbers.
   *
   * This static function takes a matrix of complex numbers as input and returns
   * a matrix containing only the imaginary parts of each element, preserving
   * the original matrix dimensions.
   *
   * @tparam T The type of the imaginary part of the complex numbers.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param input The input matrix containing complex numbers.
   * @return Base::Matrix::Matrix<T, M, N> A matrix containing the imaginary
   * parts of the input matrix.
   */
  static Base::Matrix::Matrix<T, M, N>
  get(const Base::Matrix::Matrix<Complex_T, M, N> &input) {

    return Base::Matrix::get_imag_matrix_from_complex_matrix(input);
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t N>
struct GetImagFromComplexDenseMatrix<T, Complex_T, M, N, false> {
  /**
   * @brief Returns an empty matrix when the input is not complex.
   *
   * This static function returns an empty matrix of type T with dimensions M x
   * N when the input matrix is not complex, as there are no imaginary parts to
   * extract.
   *
   * @tparam T The type of the elements in the matrix.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param input The input matrix containing real numbers.
   * @return Base::Matrix::Matrix<T, M, N> An empty matrix of type T with
   * dimensions M x N.
   */
  static Base::Matrix::Matrix<T, M, N>
  get(const Base::Matrix::Matrix<T, M, N> &input) {

    Base::Matrix::Matrix<T, M, N> result;
    return result;
  }
};

template <typename T, typename Complex_T, std::size_t M, bool IsComplex>
struct GetRealFromComplexDiagMatrix {};

template <typename T, typename Complex_T, std::size_t M>
struct GetRealFromComplexDiagMatrix<T, Complex_T, M, true> {
  /**
   * @brief Extracts the real part from a diagonal matrix of complex numbers.
   *
   * This static function takes a diagonal matrix of complex numbers as input
   * and returns a diagonal matrix containing only the real parts of each
   * diagonal element, preserving the original matrix dimensions.
   *
   * @tparam T The type of the real part of the complex numbers.
   * @tparam M The size of the diagonal matrix.
   * @param input The input diagonal matrix containing complex numbers.
   * @return Base::Matrix::DiagMatrix<T, M> A diagonal matrix containing the
   * real parts of the input matrix.
   */
  static Base::Matrix::DiagMatrix<T, M>
  get(const Base::Matrix::DiagMatrix<Complex_T, M> &input) {

    return Base::Matrix::get_real_matrix_from_complex_matrix(input);
  }
};

template <typename T, typename Complex_T, std::size_t M>
struct GetRealFromComplexDiagMatrix<T, Complex_T, M, false> {
  /**
   * @brief Returns the input diagonal matrix as it is when it is not complex.
   *
   * This static function simply returns the input diagonal matrix without any
   * changes, as it is assumed to be a matrix of real numbers.
   *
   * @tparam T The type of the elements in the diagonal matrix.
   * @tparam M The size of the diagonal matrix.
   * @param input The input diagonal matrix containing real numbers.
   * @return Base::Matrix::DiagMatrix<T, M> The input diagonal matrix itself.
   */
  static Base::Matrix::DiagMatrix<T, M>
  get(const Base::Matrix::DiagMatrix<T, M> &input) {

    return input.matrix;
  }
};

template <typename T, typename Complex_T, std::size_t M, bool IsComplex>
struct GetImagFromComplexDiagMatrix {};

template <typename T, typename Complex_T, std::size_t M>
struct GetImagFromComplexDiagMatrix<T, Complex_T, M, true> {
  /**
   * @brief Extracts the imaginary part from a diagonal matrix of complex
   * numbers.
   *
   * This static function takes a diagonal matrix of complex numbers as input
   * and returns a diagonal matrix containing only the imaginary parts of each
   * diagonal element, preserving the original matrix dimensions.
   *
   * @tparam T The type of the imaginary part of the complex numbers.
   * @tparam M The size of the diagonal matrix.
   * @param input The input diagonal matrix containing complex numbers.
   * @return Base::Matrix::DiagMatrix<T, M> A diagonal matrix containing the
   * imaginary parts of the input matrix.
   */
  static Base::Matrix::DiagMatrix<T, M>
  get(const Base::Matrix::DiagMatrix<Complex_T, M> &input) {

    return Base::Matrix::get_imag_matrix_from_complex_matrix(input);
  }
};

template <typename T, typename Complex_T, std::size_t M>
struct GetImagFromComplexDiagMatrix<T, Complex_T, M, false> {
  /**
   * @brief Returns an empty diagonal matrix when the input is not complex.
   *
   * This static function returns an empty diagonal matrix of type T with size M
   * when the input matrix is not complex, as there are no imaginary parts to
   * extract.
   *
   * @tparam T The type of the elements in the diagonal matrix.
   * @tparam M The size of the diagonal matrix.
   * @param input The input diagonal matrix containing real numbers.
   * @return Base::Matrix::DiagMatrix<T, M> An empty diagonal matrix of type T
   * with size M.
   */
  static Base::Matrix::DiagMatrix<T, M>
  get(const Base::Matrix::DiagMatrix<T, M> &input) {

    Base::Matrix::DiagMatrix<T, M> result;
    return result;
  }
};

/** * @brief Type alias for a sparse matrix type based on the provided template
 * parameters.
 *
 * This type alias defines a sparse matrix type using the
 * Base::Matrix::CompiledSparseMatrix class template, with the specified element
 * type T, dimensions M and N, and row indices and pointers derived from the
 * SparseAvailable type.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of columns in the sparse matrix.
 * @tparam N The number of rows in the sparse matrix.
 * @tparam SparseAvailable A type that provides row indices and pointers for
 * the sparse matrix.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
using BaseMatrixSparseMatrix_Type = Base::Matrix::CompiledSparseMatrix<
    T, M, N, RowIndicesFromSparseAvailable<SparseAvailable>,
    RowPointersFromSparseAvailable<SparseAvailable>>;

template <typename T, typename Complex_T, std::size_t M, std::size_t N,
          typename SparseAvailable, bool IsComplex>
struct GetRealFromComplexSparseMatrix {};

template <typename T, typename Complex_T, std::size_t M, std::size_t N,
          typename SparseAvailable>
struct GetRealFromComplexSparseMatrix<T, Complex_T, M, N, SparseAvailable,
                                      true> {
  /**
   * @brief Extracts the real part from a sparse matrix of complex numbers.
   *
   * This static function takes a sparse matrix of complex numbers as input and
   * returns a sparse matrix containing only the real parts of each element,
   * preserving the original matrix dimensions.
   *
   * @tparam T The type of the real part of the complex numbers.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param input The input sparse matrix containing complex numbers.
   * @return BaseMatrixSparseMatrix_Type<T, M, N, SparseAvailable> A sparse
   * matrix containing the real parts of the input matrix.
   */
  static BaseMatrixSparseMatrix_Type<T, M, N, SparseAvailable>
  get(const BaseMatrixSparseMatrix_Type<Complex_T, M, N, SparseAvailable>
          &input) {

    return Base::Matrix::get_real_matrix_from_complex_matrix(input);
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t N,
          typename SparseAvailable>
struct GetRealFromComplexSparseMatrix<T, Complex_T, M, N, SparseAvailable,
                                      false> {
  /**
   * @brief Returns the input sparse matrix as it is when it is not complex.
   *
   * This static function simply returns the input sparse matrix without any
   * changes, as it is assumed to be a matrix of real numbers.
   *
   * @tparam T The type of the elements in the sparse matrix.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param input The input sparse matrix containing real numbers.
   * @return BaseMatrixSparseMatrix_Type<T, M, N, SparseAvailable> The input
   * sparse matrix itself.
   */
  static BaseMatrixSparseMatrix_Type<T, M, N, SparseAvailable>
  get(const BaseMatrixSparseMatrix_Type<T, M, N, SparseAvailable> &input) {

    return input.matrix;
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t N,
          typename SparseAvailable, bool IsComplex>
struct GetImagFromComplexSparseMatrix {};

template <typename T, typename Complex_T, std::size_t M, std::size_t N,
          typename SparseAvailable>
struct GetImagFromComplexSparseMatrix<T, Complex_T, M, N, SparseAvailable,
                                      true> {
  /**
   * @brief Extracts the imaginary part from a sparse matrix of complex numbers.
   *
   * This static function takes a sparse matrix of complex numbers as input and
   * returns a sparse matrix containing only the imaginary parts of each
   * element, preserving the original matrix dimensions.
   *
   * @tparam T The type of the imaginary part of the complex numbers.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param input The input sparse matrix containing complex numbers.
   * @return BaseMatrixSparseMatrix_Type<T, M, N, SparseAvailable> A sparse
   * matrix containing the imaginary parts of the input matrix.
   */
  static BaseMatrixSparseMatrix_Type<T, M, N, SparseAvailable>
  get(const BaseMatrixSparseMatrix_Type<Complex_T, M, N, SparseAvailable>
          &input) {

    return Base::Matrix::get_imag_matrix_from_complex_matrix(input);
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t N,
          typename SparseAvailable>
struct GetImagFromComplexSparseMatrix<T, Complex_T, M, N, SparseAvailable,
                                      false> {
  /**
   * @brief Returns an empty sparse matrix when the input is not complex.
   *
   * This static function returns an empty sparse matrix of type T with
   * dimensions M x N when the input matrix is not complex, as there are no
   * imaginary parts to extract.
   *
   * @tparam T The type of the elements in the sparse matrix.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param input The input sparse matrix containing real numbers.
   * @return BaseMatrixSparseMatrix_Type<T, M, N, SparseAvailable> An empty
   * sparse matrix of type T with dimensions M x N.
   */
  static BaseMatrixSparseMatrix_Type<T, M, N, SparseAvailable>
  get(const BaseMatrixSparseMatrix_Type<T, M, N, SparseAvailable> &input) {

    return input.matrix;
  }
};

} // namespace ComplexOperation

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_COMPLEX_HPP__
