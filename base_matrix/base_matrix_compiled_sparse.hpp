#ifndef BASE_MATRIX_COMPILED_SPARSE_HPP
#define BASE_MATRIX_COMPILED_SPARSE_HPP

#include "base_matrix_diagonal.hpp"
#include "base_matrix_macros.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

const double COMPILED_SPARSE_MATRIX_JUDGE_ZERO_LIMIT_VALUE = 1.0e-20;

template <std::size_t... Sizes> struct size_list_array {
  static const std::size_t size = sizeof...(Sizes);
  static const std::size_t value[size];
};

template <std::size_t... Sizes>
const std::size_t
    size_list_array<Sizes...>::value[size_list_array<Sizes...>::size] = {
        Sizes...};

template <typename Array> class CompiledSparseMatrixList {
public:
  typedef const std::size_t *size_list_type;
  static const size_list_type size_list;
  static const std::size_t size = Array::size;

  template <std::size_t I> static std::size_t get_each_size() {
    static_assert(I < Array::size, "Index out of bounds");
    return size_list[I];
  }
};

template <typename Array>
const typename CompiledSparseMatrixList<Array>::size_list_type
    CompiledSparseMatrixList<Array>::size_list = Array::value;

template <std::size_t... Sizes>
using RowIndices = CompiledSparseMatrixList<size_list_array<Sizes...>>;

template <std::size_t... Sizes>
using RowPointers = CompiledSparseMatrixList<size_list_array<Sizes...>>;

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
class CompiledSparseMatrix {
public:
#ifdef BASE_MATRIX_USE_STD_VECTOR

  CompiledSparseMatrix() : values(RowIndices::size, static_cast<T>(0)) {}

  CompiledSparseMatrix(const std::initializer_list<T> &values)
      : values(values) {}

  CompiledSparseMatrix(const std::vector<T> &values) : values(values) {}

#else

  CompiledSparseMatrix() : values{} {}

  CompiledSparseMatrix(const std::initializer_list<T> &values) : values{} {

    std::copy(values.begin(), values.end(), this->values.begin());
  }

  CompiledSparseMatrix(const std::array<T, RowIndices::size> &values)
      : values(values) {}

  CompiledSparseMatrix(const std::vector<T> &values) : values{} {

    std::copy(values.begin(), values.end(), this->values.begin());
  }

#endif

  /* Copy Constructor */
  CompiledSparseMatrix(
      const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &other)
      : values(other.values) {}

  CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &operator=(
      const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &other) {
    if (this != &other) {
      this->values = other.values;
    }
    return *this;
  }

  /* Move Constructor */
  CompiledSparseMatrix(
      CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &&other) noexcept
      : values(std::move(other.values)) {}

  CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &operator=(
      CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &&other) noexcept {
    if (this != &other) {
      this->values = std::move(other.values);
    }
    return *this;
  }

  /* Function */
  Matrix<T, M, N> create_dense() const { return output_dense_matrix(*this); }

  /* Variable */
#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values;
#else
  std::array<T, RowIndices::size> values;
#endif
};

/* Create dense matrix */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline Matrix<T, M, N> output_dense_matrix(
    const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat) {
  Matrix<T, M, N> result;

  for (std::size_t j = 0; j < M; j++) {
    for (std::size_t k = RowPointers::size_list[j];
         k < RowPointers::size_list[j + 1]; k++) {
      result(j, RowIndices::size_list[k]) = mat.values[k];
    }
  }

  return result;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_COMPILED_SPARSE_HPP
