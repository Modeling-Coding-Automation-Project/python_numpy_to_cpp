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

template <std::size_t... Sizes> struct sizes_array {
  static const std::size_t size = sizeof...(Sizes);
  static const std::size_t value[size];
};

template <std::size_t... Sizes>
const std::size_t sizes_array<Sizes...>::value[sizes_array<Sizes...>::size] = {
    Sizes...};

template <typename Array> class CompiledSparseMatrixList {
public:
  typedef const std::size_t *sizes_type;
  static const sizes_type sizes;
  static const std::size_t size = Array::size;

  template <std::size_t I> static std::size_t get_each_size() {
    static_assert(I < Array::size, "Index out of bounds");
    return sizes[I];
  }
};

template <typename Array>
const typename CompiledSparseMatrixList<Array>::sizes_type
    CompiledSparseMatrixList<Array>::sizes = Array::value;

template <std::size_t... Sizes>
using RowIndices = CompiledSparseMatrixList<sizes_array<Sizes...>>;

template <std::size_t... Sizes>
using RowPointers = CompiledSparseMatrixList<sizes_array<Sizes...>>;

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
class CompiledSparseMatrix {
public:
  CompiledSparseMatrix() {}

  std::size_t print_sizes() { return RowIndices::template get_each_size<1>(); }

/* Variable */
#ifdef BASE_MATRIX_USE_STD_VECTOR

#else
  std::array<T, RowIndices::size> values;
#endif
};

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_COMPILED_SPARSE_HPP
