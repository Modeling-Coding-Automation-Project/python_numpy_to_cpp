#ifndef PYTHON_NUMPY_AUGMENTED_MATRIX_HPP_
#define PYTHON_NUMPY_AUGMENTED_MATRIX_HPP_

#include <array>
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

template <typename Tuple> struct matrix_shape_extractor;

template <typename... Args> struct matrix_shape_extractor<std::tuple<Args...>> {
  static constexpr std::array<std::size_t, sizeof...(Args)> ROWS = {
      Args::ROWS...};
  static constexpr std::array<std::size_t, sizeof...(Args)> COLS = {
      Args::COLS...};
};

/* Check Cols Rows sizes */

template <std::size_t N>
constexpr bool check_row_compatibility(const std::array<std::size_t, N> &rows,
                                       std::size_t dim, std::size_t i,
                                       std::size_t j) {
  return (j == dim) ? true
                    : (rows[i * dim + j] == rows[i * dim]) &&
                          check_row_compatibility(rows, dim, i, j + 1);
}

template <std::size_t N>
constexpr bool check_all_rows(const std::array<std::size_t, N> &rows,
                              std::size_t dim, std::size_t i) {
  return (i == dim) ? true
                    : check_row_compatibility(rows, dim, i, 1) &&
                          check_all_rows(rows, dim, i + 1);
}

template <std::size_t N>
constexpr bool check_col_compatibility(const std::array<std::size_t, N> &cols,
                                       std::size_t dim, std::size_t i,
                                       std::size_t j) {
  return (i == dim) ? true
                    : (cols[i * dim + j] == cols[j]) &&
                          check_col_compatibility(cols, dim, i + 1, j);
}

template <std::size_t N>
constexpr bool check_all_cols(const std::array<std::size_t, N> &cols,
                              std::size_t dim, std::size_t j) {
  return (j == dim) ? true
                    : check_col_compatibility(cols, dim, 1, j) &&
                          check_all_cols(cols, dim, j + 1);
}

template <std::size_t N>
constexpr bool
is_compatible_augmented_matrix(const std::array<std::size_t, N> &rows,
                               const std::array<std::size_t, N> &cols,
                               std::size_t dim) {

  return check_all_rows(rows, dim, 0) && check_all_cols(cols, dim, 0);
}

/* get */

template <std::size_t N>
constexpr std::size_t
get_block_row_idx(std::size_t global_row,
                  const std::array<std::size_t, N> &rows, std::size_t dim,
                  std::size_t current = 0, std::size_t accum = 0) {
  return (current == dim) ? 0
         : (global_row < accum + rows[current * dim])
             ? current
             : get_block_row_idx(global_row, rows, dim, current + 1,
                                 accum + rows[current * dim]);
}

template <std::size_t N>
constexpr std::size_t
get_local_row_idx(std::size_t global_row,
                  const std::array<std::size_t, N> &rows, std::size_t dim,
                  std::size_t current = 0, std::size_t accum = 0) {
  return (current == dim) ? 0
         : (global_row < accum + rows[current * dim])
             ? (global_row - accum)
             : get_local_row_idx(global_row, rows, dim, current + 1,
                                 accum + rows[current * dim]);
}

template <std::size_t N>
constexpr std::size_t
get_block_col_idx(std::size_t global_col,
                  const std::array<std::size_t, N> &cols, std::size_t dim,
                  std::size_t current = 0, std::size_t accum = 0) {
  return (current == dim) ? 0
         : (global_col < accum + cols[current])
             ? current
             : get_block_col_idx(global_col, cols, dim, current + 1,
                                 accum + cols[current]);
}

template <std::size_t N>
constexpr std::size_t
get_local_col_idx(std::size_t global_col,
                  const std::array<std::size_t, N> &cols, std::size_t dim,
                  std::size_t current = 0, std::size_t accum = 0) {
  return (current == dim) ? 0
         : (global_col < accum + cols[current])
             ? (global_col - accum)
             : get_local_col_idx(global_col, cols, dim, current + 1,
                                 accum + cols[current]);
}

} // namespace AugmentedMatrixAction

/* Augmented Matrix */

template <typename Tuple_Type, std::size_t M, std::size_t N>
class AugmentedMatrix {
public:
  /* Check Compatibility */
  static_assert(AugmentedMatrixAction::is_tuple<Tuple_Type>::value,
                "Tuple_Type must be a std::tuple.");

  /* Type */
  using T = typename std::tuple_element<0, Tuple_Type>::type::Value_Type;

  static constexpr std::size_t COLROW_AUGMENTED_MATRIX =
      AugmentedMatrixAction::tuple_square_root<Tuple_Type>::value;

  static_assert(COLROW_AUGMENTED_MATRIX != 0,
                "Tuple_Type must have a perfect square number of elements.");

  static constexpr auto ELEMENT_ROWS =
      AugmentedMatrixAction::matrix_shape_extractor<Tuple_Type>::ROWS;
  static constexpr auto ELEMENT_COLS =
      AugmentedMatrixAction::matrix_shape_extractor<Tuple_Type>::COLS;

  /* Check Compatibility */
  static_assert(AugmentedMatrixAction::is_compatible_augmented_matrix(
                    ELEMENT_ROWS, ELEMENT_COLS, COLROW_AUGMENTED_MATRIX),
                "The rows or columns of the nested matrices do not align "
                "perfectly in the augmented matrix.");

public:
  /* Constructor */
  AugmentedMatrix() {}

  /* Copy Constructor */
  AugmentedMatrix(const AugmentedMatrix<Tuple_Type, M, N> &input)
      : matrix(input.matrix) {}

  AugmentedMatrix<Tuple_Type, M, N> &
  operator=(const AugmentedMatrix<Tuple_Type, M, N> &input) {
    if (this != &input) {
      this->matrix = input.matrix;
    }
    return *this;
  }

  /* Move Constructor */
  AugmentedMatrix(AugmentedMatrix<Tuple_Type, M, N> &&input) noexcept
      : matrix(std::move(input.matrix)) {}

  AugmentedMatrix<Tuple_Type, M, N> &
  operator=(AugmentedMatrix<Tuple_Type, M, N> &&input) noexcept {
    if (this != &input) {
      this->matrix = std::move(input.matrix);
    }
    return *this;
  }

public:
  /* Function */
  template <std::size_t ROW, std::size_t COL> inline T get() const {

    constexpr std::size_t block_row = AugmentedMatrixAction::get_block_row_idx(
        ROW, ELEMENT_ROWS, COLROW_AUGMENTED_MATRIX);
    constexpr std::size_t block_col = AugmentedMatrixAction::get_block_col_idx(
        COL, ELEMENT_COLS, COLROW_AUGMENTED_MATRIX);

    constexpr std::size_t tuple_idx =
        block_row * COLROW_AUGMENTED_MATRIX + block_col;

    constexpr std::size_t local_row = AugmentedMatrixAction::get_local_row_idx(
        ROW, ELEMENT_ROWS, COLROW_AUGMENTED_MATRIX);
    constexpr std::size_t local_col = AugmentedMatrixAction::get_local_col_idx(
        COL, ELEMENT_COLS, COLROW_AUGMENTED_MATRIX);

    return std::get<tuple_idx>(this->matrix)
        .template get<local_row, local_col>();
  }

  template <std::size_t ROW, std::size_t COL> inline void set(const T &value) {

    constexpr std::size_t block_row = AugmentedMatrixAction::get_block_row_idx(
        ROW, ELEMENT_ROWS, COLROW_AUGMENTED_MATRIX);
    constexpr std::size_t block_col = AugmentedMatrixAction::get_block_col_idx(
        COL, ELEMENT_COLS, COLROW_AUGMENTED_MATRIX);

    constexpr std::size_t tuple_idx =
        block_row * COLROW_AUGMENTED_MATRIX + block_col;

    constexpr std::size_t local_row = AugmentedMatrixAction::get_local_row_idx(
        ROW, ELEMENT_ROWS, COLROW_AUGMENTED_MATRIX);
    constexpr std::size_t local_col = AugmentedMatrixAction::get_local_col_idx(
        COL, ELEMENT_COLS, COLROW_AUGMENTED_MATRIX);

    std::get<tuple_idx>(this->matrix).template set<local_row, local_col>(value);
  }

public:
  /* Variable */
  Tuple_Type matrix;
};

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_AUGMENTED_MATRIX_HPP_
