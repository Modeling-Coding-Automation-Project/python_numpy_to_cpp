#ifndef PYTHON_NUMPY_AUGMENTED_MATRIX_HPP_
#define PYTHON_NUMPY_AUGMENTED_MATRIX_HPP_

#include "python_numpy_base.hpp"

#include <tuple>
#include <type_traits>

namespace PythonNumpy {

/* Check std::tuple */
namespace AugmentedMatrixAction {

template <typename T> struct is_tuple : std::false_type {};

template <typename... Args>
struct is_tuple<std::tuple<Args...>> : std::true_type {};

/* Check if a tuple has a perfect square number of elements */

constexpr std::size_t get_square_root_impl(std::size_t n, std::size_t i) {
  return (i * i == n) ? i : (i * i > n) ? 0 : get_square_root_impl(n, i + 1);
}

constexpr std::size_t get_square_root(std::size_t n) {
  return (n == 0) ? 0 : get_square_root_impl(n, 1);
}

template <typename T>
struct tuple_square_root : std::integral_constant<std::size_t, 0> {};

template <typename... Args>
struct tuple_square_root<std::tuple<Args...>>
    : std::integral_constant<std::size_t, get_square_root(sizeof...(Args))> {};

} // namespace AugmentedMatrixAction

/* Augmented Matrix */

template <typename Tuple_Type, std::size_t M, std::size_t N>
class AugmentedMatrix {
public:
  /* Check Compatibility */
  static_assert(AugmentedMatrixAction::is_tuple<Tuple_Type>::value,
                "Tuple_Type must be a std::tuple.");

  /* Type */
  static constexpr std::size_t ColRow_AugmentedMatrix =
      AugmentedMatrixAction::tuple_square_root<Tuple_Type>::value;

  static_assert(ColRow_AugmentedMatrix != 0,
                "Tuple_Type must have a perfect square number of elements.");

public:
  /* Variable */
  Tuple_Type matrix;
};

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_AUGMENTED_MATRIX_HPP_
