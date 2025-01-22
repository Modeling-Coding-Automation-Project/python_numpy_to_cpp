#ifndef __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__
#define __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__

#include "python_numpy_base.hpp"
#include "python_numpy_complex.hpp"
#include "python_numpy_templates.hpp"

#include <initializer_list>

namespace PythonNumpy {

template <typename T, std::size_t M, std::size_t N>
inline auto make_MatrixZeros(void) -> Matrix<DefDense, T, M, N> {

  Matrix<DefDense, T, M, N> result;
  return result;
}

template <typename T, std::size_t M, std::size_t N>
inline auto make_MatrixOnes(void) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>::ones();
}

template <typename T, std::size_t M>
inline auto make_MatrixIdentity(void) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>::identity();
}

template <typename T, std::size_t M, std::size_t N>
inline auto make_MatrixEmpty(void)
    -> Matrix<DefSparse, T, M, N, SparseAvailableEmpty<M, N>> {

  return Matrix<DefSparse, T, M, N, SparseAvailableEmpty<M, N>>();
}

template <typename T, std::size_t M>
inline auto make_MatrixDiag(const std::initializer_list<T> &input)
    -> Matrix<DefDiag, T, M> {
  return Matrix<DefDiag, T, M>(input);
}

namespace MakeDiagMatrixOperation {

template <std::size_t IndexCount, typename DiagMatrix_Type, typename T>
void assign_diag_values(DiagMatrix_Type &matrix, T value_1) {

  matrix.template set<IndexCount, IndexCount>(value_1);
}

template <std::size_t IndexCount, typename DiagMatrix_Type, typename T,
          typename U, typename... Args>
void assign_diag_values(DiagMatrix_Type &matrix, T value_1, U value_2,
                        Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");

  matrix.template set<IndexCount, IndexCount>(value_1);

  assign_diag_values<IndexCount + 1>(matrix, value_2, args...);
}

} // namespace MakeDiagMatrixOperation

template <std::size_t M, typename T, typename... Args>
inline auto make_MatrixDiag(T value_1, Args... args) -> Matrix<DefDiag, T, M> {

  Matrix<DefDiag, T, M> result;

  MakeDiagMatrixOperation::assign_diag_values<0>(result, value_1, args...);

  return result;
}

template <typename T, std::size_t M, std::size_t N>
inline auto
make_Matrix(const std::initializer_list<std::initializer_list<T>> &input)
    -> Matrix<DefDense, T, M, N> {
  return Matrix<DefDense, T, M, N>(input);
}

template <typename T, typename SparseAvailable>
inline auto make_MatrixSparseZeros(void)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  Matrix<DefSparse, T, SparseAvailable::number_of_columns,
         SparseAvailable::column_size, SparseAvailable>
      result;

  return result;
}

template <typename T, typename SparseAvailable>
inline auto make_MatrixSparse(const std::initializer_list<T> &input)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  Matrix<DefSparse, T, SparseAvailable::number_of_columns,
         SparseAvailable::column_size, SparseAvailable>
      result(input);

  return result;
}

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__
