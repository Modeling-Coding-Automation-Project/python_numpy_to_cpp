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

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__
