#ifndef PYTHON_NUMPY_AUGMENTED_MATRIX_HPP_
#define PYTHON_NUMPY_AUGMENTED_MATRIX_HPP_

#include "python_numpy_base.hpp"
#include "python_numpy_base_substitution.hpp"

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

/* Calculate total rows and cols for augmented matrix */

template <std::size_t Size>
constexpr std::size_t
calculate_total_rows(const std::array<std::size_t, Size> &rows, std::size_t dim,
                     std::size_t i = 0) {
  return (i == dim) ? 0
                    : rows[i * dim] + calculate_total_rows(rows, dim, i + 1);
}

template <std::size_t Size>
constexpr std::size_t
calculate_total_cols(const std::array<std::size_t, Size> &cols, std::size_t dim,
                     std::size_t j = 0) {
  return (j == dim) ? 0 : cols[j] + calculate_total_cols(cols, dim, j + 1);
}

/* get / set */

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

template <typename Tuple_Type> class AugmentedMatrix {
public:
  /* Check Compatibility */
  static_assert(AugmentedMatrixAction::is_tuple<Tuple_Type>::value,
                "Tuple_Type must be a std::tuple.");

  /* Type */
  using Value_Type =
      typename std::tuple_element<0, Tuple_Type>::type::Value_Type;

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

  /* Type */
  static constexpr std::size_t ROWS =
      AugmentedMatrixAction::calculate_total_rows(ELEMENT_ROWS,
                                                  COLROW_AUGMENTED_MATRIX);

  static constexpr std::size_t COLS =
      AugmentedMatrixAction::calculate_total_cols(ELEMENT_COLS,
                                                  COLROW_AUGMENTED_MATRIX);

protected:
  /* Type */
  using T_ = Value_Type;

public:
  /* Constructor */
  AugmentedMatrix() {}

  template <typename... Matrices,
            typename std::enable_if<sizeof...(Matrices) ==
                                        std::tuple_size<Tuple_Type>::value,
                                    int>::type = 0>
  explicit AugmentedMatrix(const Matrices &...inputs) : matrix(inputs...) {}

  /* Copy Constructor */
  AugmentedMatrix(const AugmentedMatrix<Tuple_Type> &input)
      : matrix(input.matrix) {}

  AugmentedMatrix<Tuple_Type> &
  operator=(const AugmentedMatrix<Tuple_Type> &input) {
    if (this != &input) {
      this->matrix = input.matrix;
    }
    return *this;
  }

  /* Move Constructor */
  AugmentedMatrix(AugmentedMatrix<Tuple_Type> &&input) noexcept
      : matrix(std::move(input.matrix)) {}

  AugmentedMatrix<Tuple_Type> &
  operator=(AugmentedMatrix<Tuple_Type> &&input) noexcept {
    if (this != &input) {
      this->matrix = std::move(input.matrix);
    }
    return *this;
  }

public:
  /* Function */

  constexpr std::size_t cols() const { return COLS; }

  constexpr std::size_t rows() const { return ROWS; }

  constexpr std::size_t size() const { return ROWS * COLS; }

  std::tuple<std::size_t, std::size_t> shape() const {
    return std::make_tuple(static_cast<std::size_t>(ROWS),
                           static_cast<std::size_t>(COLS));
  }

  std::size_t ndim() const { return 2; }

  template <std::size_t ROW_IN, std::size_t COL_IN> inline T_ get() const {
    static_assert(ROW_IN < ROWS && COL_IN < COLS,
                  "ROW and COL must be within the bounds of the matrix.");

    constexpr std::size_t block_row = AugmentedMatrixAction::get_block_row_idx(
        ROW_IN, ELEMENT_ROWS, COLROW_AUGMENTED_MATRIX);
    constexpr std::size_t block_col = AugmentedMatrixAction::get_block_col_idx(
        COL_IN, ELEMENT_COLS, COLROW_AUGMENTED_MATRIX);

    constexpr std::size_t tuple_idx =
        block_row * COLROW_AUGMENTED_MATRIX + block_col;

    constexpr std::size_t local_row = AugmentedMatrixAction::get_local_row_idx(
        ROW_IN, ELEMENT_ROWS, COLROW_AUGMENTED_MATRIX);
    constexpr std::size_t local_col = AugmentedMatrixAction::get_local_col_idx(
        COL_IN, ELEMENT_COLS, COLROW_AUGMENTED_MATRIX);

    return std::get<tuple_idx>(this->matrix)
        .template get<local_row, local_col>();
  }

  template <std::size_t ROW_IN, std::size_t COL_IN>
  inline void set(const T_ &value) {
    static_assert(ROW_IN < ROWS && COL_IN < COLS,
                  "ROW and COL must be within the bounds of the matrix.");

    constexpr std::size_t block_row = AugmentedMatrixAction::get_block_row_idx(
        ROW_IN, ELEMENT_ROWS, COLROW_AUGMENTED_MATRIX);
    constexpr std::size_t block_col = AugmentedMatrixAction::get_block_col_idx(
        COL_IN, ELEMENT_COLS, COLROW_AUGMENTED_MATRIX);

    constexpr std::size_t tuple_idx =
        block_row * COLROW_AUGMENTED_MATRIX + block_col;

    constexpr std::size_t local_row = AugmentedMatrixAction::get_local_row_idx(
        ROW_IN, ELEMENT_ROWS, COLROW_AUGMENTED_MATRIX);
    constexpr std::size_t local_col = AugmentedMatrixAction::get_local_col_idx(
        COL_IN, ELEMENT_COLS, COLROW_AUGMENTED_MATRIX);

    std::get<tuple_idx>(this->matrix).template set<local_row, local_col>(value);
  }

  template <typename Matrix_Type> inline auto to_matrix() const -> Matrix_Type {
    Matrix_Type dense_matrix;

    substitute_matrix(dense_matrix, *this);

    return dense_matrix;
  }

  T_ &operator()(std::size_t index) {
    if (index >= ROWS * COLS) {
      index = ROWS * COLS - 1;
    }

    std::size_t row = index / COLS;
    std::size_t col = index % COLS;

    Matrix<DefDense, T_, ROWS, COLS> dense_matrix;

    substitute_matrix(dense_matrix, *this);

    return dense_matrix(row, col);
  }

  T_ &operator()(std::size_t row, std::size_t col) {
    if (row >= ROWS) {
      row = ROWS - 1;
    }
    if (col >= COLS) {
      col = COLS - 1;
    }

    Matrix<DefDense, T_, ROWS, COLS> dense_matrix;

    substitute_matrix(dense_matrix, *this);

    return dense_matrix(row, col);
  }

public:
  /* Variable */
  Tuple_Type matrix;
};

/* Matrix Add AugmentedMatrix */

namespace AugmentedMatrixAddMatrix {

// when J_idx < N
template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I, std::size_t J_idx>
struct Row {
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    result.template set<I, J_idx>(A.template get<I, J_idx>() +
                                  B.template get<I, J_idx>());
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I,
        J_idx - 1>::compute(A, B, result);
  }
};

// column recursion termination
template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I>
struct Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I, 0> {
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    result.template set<I, 0>(A.template get<I, 0>() + B.template get<I, 0>());
  }
};

// when I_idx < M
template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I_idx>
struct Column {
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I_idx,
        Matrix_Result_Type::COLS - 1>::compute(A, B, result);
    Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type,
           I_idx - 1>::compute(A, B, result);
  }
};

// row recursion termination
template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type>
struct Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, 0> {
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, 0,
        Matrix_Result_Type::COLS - 1>::compute(A, B, result);
  }
};

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type>
inline void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                    Matrix_Result_Type &result) {
  Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type,
         Matrix_Result_Type::ROWS - 1>::compute(A, B, result);
}

} // namespace AugmentedMatrixAddMatrix

template <typename Matrix_Type, typename Tuple_Type>
inline auto operator+(const Matrix_Type &matrix,
                      const AugmentedMatrix<Tuple_Type> &augmented_matrix)
    -> Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
              Matrix_Type::COLS> {

  static_assert(
      std::is_same<typename Matrix_Type::Value_Type,
                   typename AugmentedMatrix<Tuple_Type>::Value_Type>::value,
      "Matrix_Type and AugmentedMatrix_Type must have the same value type.");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return matrix + augmented_matrix.template to_matrix<Matrix_Type>();

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
         Matrix_Type::COLS>
      result;
  AugmentedMatrixAddMatrix::compute(matrix, augmented_matrix, result);
  return result;

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

template <typename Matrix_Type, typename Tuple_Type>
inline auto operator+(const AugmentedMatrix<Tuple_Type> &augmented_matrix,
                      const Matrix_Type &matrix)
    -> Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
              Matrix_Type::COLS> {

  static_assert(
      std::is_same<typename Matrix_Type::Value_Type,
                   typename AugmentedMatrix<Tuple_Type>::Value_Type>::value,
      "Matrix_Type and AugmentedMatrix_Type must have the same value type.");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return augmented_matrix.template to_matrix<Matrix_Type>() + matrix;

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
         Matrix_Type::COLS>
      result;
  AugmentedMatrixAddMatrix::compute(augmented_matrix, matrix, result);
  return result;

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/* Matrix Sub AugmentedMatrix */

namespace AugmentedMatrixSubMatrix {

// when J_idx < N
template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I, std::size_t J_idx>
struct Row {
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    result.template set<I, J_idx>(A.template get<I, J_idx>() -
                                  B.template get<I, J_idx>());
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I,
        J_idx - 1>::compute(A, B, result);
  }
};

// column recursion termination
template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I>
struct Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I, 0> {
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    result.template set<I, 0>(A.template get<I, 0>() - B.template get<I, 0>());
  }
};

// when I_idx < M
template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I_idx>
struct Column {
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I_idx,
        Matrix_Result_Type::COLS - 1>::compute(A, B, result);
    Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type,
           I_idx - 1>::compute(A, B, result);
  }
};

// row recursion termination
template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type>
struct Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, 0> {
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, 0,
        Matrix_Result_Type::COLS - 1>::compute(A, B, result);
  }
};

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type>
inline void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                    Matrix_Result_Type &result) {
  Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type,
         Matrix_Result_Type::ROWS - 1>::compute(A, B, result);
}

} // namespace AugmentedMatrixSubMatrix

template <typename Matrix_Type, typename Tuple_Type>
inline auto operator-(const Matrix_Type &matrix,
                      const AugmentedMatrix<Tuple_Type> &augmented_matrix)
    -> Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
              Matrix_Type::COLS> {

  static_assert(
      std::is_same<typename Matrix_Type::Value_Type,
                   typename AugmentedMatrix<Tuple_Type>::Value_Type>::value,
      "Matrix_Type and AugmentedMatrix_Type must have the same value type.");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return matrix - augmented_matrix.template to_matrix<Matrix_Type>();

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
         Matrix_Type::COLS>
      result;
  AugmentedMatrixSubMatrix::compute(matrix, augmented_matrix, result);
  return result;

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

template <typename Matrix_Type, typename Tuple_Type>
inline auto operator-(const AugmentedMatrix<Tuple_Type> &augmented_matrix,
                      const Matrix_Type &matrix)
    -> Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
              Matrix_Type::COLS> {

  static_assert(
      std::is_same<typename Matrix_Type::Value_Type,
                   typename AugmentedMatrix<Tuple_Type>::Value_Type>::value,
      "Matrix_Type and AugmentedMatrix_Type must have the same value type.");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return augmented_matrix.template to_matrix<Matrix_Type>() - matrix;

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
         Matrix_Type::COLS>
      result;
  AugmentedMatrixSubMatrix::compute(augmented_matrix, matrix, result);
  return result;

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_AUGMENTED_MATRIX_HPP_
