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

template <typename Array> class CompiledSparseMatrixIndices {
public:
  typedef const std::size_t *sizes_type;
  static const sizes_type sizes;

  void example() {
    static const std::size_t size1 = sizes[0]; // 1
    static const std::size_t size2 = sizes[1]; // 2
    static const std::size_t size3 = sizes[2]; // 3

    for (std::size_t i = 0; i < Array::size; ++i) {
      std::size_t size = sizes[i];
    }
  }

  template <std::size_t I> static std::size_t get_size() {
    static_assert(I < Array::size, "Index out of bounds");
    return sizes[I];
  }
};

template <typename Array>
const typename CompiledSparseMatrixIndices<Array>::sizes_type
    CompiledSparseMatrixIndices<Array>::sizes = Array::value;

template <std::size_t... Sizes>
using Indices = CompiledSparseMatrixIndices<sizes_array<Sizes...>>;

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_COMPILED_SPARSE_HPP
