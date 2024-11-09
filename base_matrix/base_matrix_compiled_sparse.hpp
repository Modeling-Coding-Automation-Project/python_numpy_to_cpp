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

  template <std::size_t I> static std::size_t get_size() {
    static_assert(I < Array::size, "Index out of bounds");
    return sizes[I];
  }
};

template <typename Array>
const typename CompiledSparseMatrixList<Array>::sizes_type
    CompiledSparseMatrixList<Array>::sizes = Array::value;

template <std::size_t... Sizes>
using Indices = CompiledSparseMatrixList<sizes_array<Sizes...>>;

template <std::size_t... Sizes>
using Pointers = CompiledSparseMatrixList<sizes_array<Sizes...>>;

template <typename T, typename I> class CompiledSparseMatrix {
public:
  CompiledSparseMatrix() {}

  void print_sizes() { this->value = I::template get_size<1>(); }

  std::size_t value = static_cast<std::size_t>(0);
};

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_COMPILED_SPARSE_HPP
