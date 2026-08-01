#ifndef BASE_MATRIX_MATH_HPP_
#define BASE_MATRIX_MATH_HPP_

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"

#include "python_math.hpp"

namespace Base {
namespace Matrix {

/* abs */
template <typename T> inline T abs(const T &x) { return PythonMath::abs(x); }

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> abs(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i < M; ++i) {
      result.data[j][i] = PythonMath::abs(matrix.data[j][i]);
    }
  }

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> abs(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  for (std::size_t i = 0; i < M; ++i) {
    result.data[i] = PythonMath::abs(matrix.data[i]);
  }

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
abs(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  for (std::size_t i = 0; i < matrix.values.size(); ++i) {
    result.values[i] = PythonMath::abs(matrix.values[i]);
  }

  return result;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_MATH_HPP_
