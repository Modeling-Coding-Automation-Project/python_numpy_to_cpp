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
  static Base::Matrix::Matrix<T, M, N>
  get(const Base::Matrix::Matrix<Complex_T, M, N> &input) {

    return Base::Matrix::get_real_matrix_from_complex_matrix(input);
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t N>
struct GetRealFromComplexDenseMatrix<T, Complex_T, M, N, false> {
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
  static Base::Matrix::Matrix<T, M, N>
  get(const Base::Matrix::Matrix<Complex_T, M, N> &input) {

    return Base::Matrix::get_imag_matrix_from_complex_matrix(input);
  }
};

template <typename T, typename Complex_T, std::size_t M, std::size_t N>
struct GetImagFromComplexDenseMatrix<T, Complex_T, M, N, false> {
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
  static Base::Matrix::DiagMatrix<T, M>
  get(const Base::Matrix::DiagMatrix<Complex_T, M> &input) {

    return Base::Matrix::get_real_matrix_from_complex_matrix(input);
  }
};

template <typename T, typename Complex_T, std::size_t M>
struct GetRealFromComplexDiagMatrix<T, Complex_T, M, false> {
  static Base::Matrix::DiagMatrix<T, M>
  get(const Base::Matrix::DiagMatrix<T, M> &input) {

    return input.matrix;
  }
};

template <typename T, typename Complex_T, std::size_t M, bool IsComplex>
struct GetImagFromComplexDiagMatrix {};

template <typename T, typename Complex_T, std::size_t M>
struct GetImagFromComplexDiagMatrix<T, Complex_T, M, true> {
  static Base::Matrix::DiagMatrix<T, M>
  get(const Base::Matrix::DiagMatrix<Complex_T, M> &input) {

    return Base::Matrix::get_imag_matrix_from_complex_matrix(input);
  }
};

template <typename T, typename Complex_T, std::size_t M>
struct GetImagFromComplexDiagMatrix<T, Complex_T, M, false> {
  static Base::Matrix::DiagMatrix<T, M>
  get(const Base::Matrix::DiagMatrix<T, M> &input) {

    Base::Matrix::DiagMatrix<T, M> result;
    return result;
  }
};

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
  static BaseMatrixSparseMatrix_Type<T, M, N, SparseAvailable>
  get(const BaseMatrixSparseMatrix_Type<T, M, N, SparseAvailable> &input) {

    return input.matrix;
  }
};

} // namespace ComplexOperation

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_COMPLEX_HPP__
