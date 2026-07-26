/**
 * @file python_numpy_augmented_matrix.hpp
 * @brief Augmented matrix class template and operations for PythonNumpy C++
 * library.
 *
 * This file defines the AugmentedMatrix class template and its associated
 * operations for the PythonNumpy namespace. The AugmentedMatrix class allows
 * for the representation of a larger matrix composed of smaller matrices
 * arranged in a block structure. It provides functionality to access and
 * manipulate elements within the augmented matrix, as well as to perform matrix
 * multiplication with other matrices or augmented matrices.
 *
 * @note
 * tparam Tuple_Type is a std::tuple containing the smaller matrices that make
 * up the augmented matrix. The number of elements in the tuple must be a
 * perfect square to form a square block structure.
 */
#ifndef PYTHON_NUMPY_AUGMENTED_MATRIX_HPP_
#define PYTHON_NUMPY_AUGMENTED_MATRIX_HPP_

#include "python_numpy_base.hpp"
#include "python_numpy_base_substitution.hpp"

#include <array>
#include <tuple>
#include <type_traits>

namespace PythonNumpy {

namespace AugmentedMatrixAction {

/* Check std::tuple */

/**
 * @brief Checks if a type is a std::tuple.
 */
template <typename T> struct is_tuple : std::false_type {};

/**
 * @brief Specialization for std::tuple types.
 */
template <typename... Args>
struct is_tuple<std::tuple<Args...>> : std::true_type {};

/* Check if a tuple has a perfect square number of elements */

/**
 * @brief Computes the integer square root of a number at compile time.
 * @param n The number to compute the square root of.
 * @return The integer square root of n, or 0 if n is not a perfect square.
 */
constexpr std::size_t get_square_root_impl(std::size_t n, std::size_t i) {
  return (i * i == n) ? i : (i * i > n) ? 0 : get_square_root_impl(n, i + 1);
}

/**
 * @brief Computes the integer square root of a number at compile time.
 * @param n The number to compute the square root of.
 * @return The integer square root of n, or 0 if n is not a perfect square.
 */
constexpr std::size_t get_square_root(std::size_t n) {
  return (n == 0) ? 0 : get_square_root_impl(n, 1);
}

/**
 * @brief A helper struct to compute the square root of the size of a tuple.
 * @tparam T The type to check.
 */
template <typename T>
struct tuple_square_root : std::integral_constant<std::size_t, 0> {};

/**
 * @brief Specialization for std::tuple types.
 * @tparam Args The types contained in the tuple.
 */
template <typename... Args>
struct tuple_square_root<std::tuple<Args...>>
    : std::integral_constant<std::size_t, get_square_root(sizeof...(Args))> {};

/**
 * @brief A helper struct to extract the rows and columns of matrices contained
 * in a tuple.
 */
template <typename Tuple> struct matrix_shape_extractor;

/**
 * @brief Specialization for std::tuple types.
 * @tparam Args The types contained in the tuple.
 */
template <typename... Args> struct matrix_shape_extractor<std::tuple<Args...>> {
  static constexpr std::array<std::size_t, sizeof...(Args)> ROWS = {
      Args::ROWS...};
  static constexpr std::array<std::size_t, sizeof...(Args)> COLS = {
      Args::COLS...};
};

/* Check Cols Rows sizes */

/**
 * @brief Checks if the rows of a matrix are compatible with the augmented
 * matrix.
 * @tparam N The size of the rows array.
 * @param rows The array of rows.
 * @param dim The dimension of the augmented matrix.
 * @param i The current row index.
 * @param j The current column index.
 * @return True if the rows are compatible, false otherwise.
 */
template <std::size_t N>
constexpr bool check_row_compatibility(const std::array<std::size_t, N> &rows,
                                       std::size_t col_blocks, std::size_t i,
                                       std::size_t j) {
  return (j == col_blocks)
             ? true
             : (rows[i * col_blocks + j] == rows[i * col_blocks]) &&
                   check_row_compatibility(rows, col_blocks, i, j + 1);
}

/**
 * @brief Checks if all rows of a matrix are compatible with the augmented
 * matrix.
 * @tparam N The size of the rows array.
 * @param rows The array of rows.
 * @param dim The dimension of the augmented matrix.
 * @param i The current row index.
 * @return True if all rows are compatible, false otherwise.
 */
template <std::size_t N>
constexpr bool check_all_rows(const std::array<std::size_t, N> &rows,
                              std::size_t row_blocks, std::size_t col_blocks,
                              std::size_t i) {
  return (i == row_blocks)
             ? true
             : check_row_compatibility(rows, col_blocks, i, 1) &&
                   check_all_rows(rows, row_blocks, col_blocks, i + 1);
}

/**
 * @brief Checks if the columns of a matrix are compatible with the augmented
 * matrix.
 * @tparam N The size of the columns array.
 * @param cols The array of columns.
 * @param dim The dimension of the augmented matrix.
 * @param i The current row index.
 * @param j The current column index.
 * @return True if the columns are compatible, false otherwise.
 */
template <std::size_t N>
constexpr bool check_col_compatibility(const std::array<std::size_t, N> &cols,
                                       std::size_t row_blocks,
                                       std::size_t col_blocks, std::size_t i,
                                       std::size_t j) {
  return (i == row_blocks) ? true
                           : (cols[i * col_blocks + j] == cols[j]) &&
                                 check_col_compatibility(cols, row_blocks,
                                                         col_blocks, i + 1, j);
}

/**
 * @brief Checks if all columns of a matrix are compatible with the augmented
 * matrix.
 * @tparam N The size of the columns array.
 * @param cols The array of columns.
 * @param dim The dimension of the augmented matrix.
 * @param j The current column index.
 * @return True if all columns are compatible, false otherwise.
 */
template <std::size_t N>
constexpr bool check_all_cols(const std::array<std::size_t, N> &cols,
                              std::size_t row_blocks, std::size_t col_blocks,
                              std::size_t j) {
  return (j == col_blocks)
             ? true
             : check_col_compatibility(cols, row_blocks, col_blocks, 1, j) &&
                   check_all_cols(cols, row_blocks, col_blocks, j + 1);
}

/**
 * @brief Checks if the rows and columns of a matrix are compatible with the
 * augmented matrix.
 * @tparam N The size of the rows and columns arrays.
 * @param rows The array of rows.
 * @param cols The array of columns.
 * @param dim The dimension of the augmented matrix.
 * @return True if the rows and columns are compatible, false otherwise.
 */
template <std::size_t N>
constexpr bool
is_compatible_augmented_matrix(const std::array<std::size_t, N> &rows,
                               const std::array<std::size_t, N> &cols,
                               std::size_t row_blocks, std::size_t col_blocks) {

  return check_all_rows(rows, row_blocks, col_blocks, 0) &&
         check_all_cols(cols, row_blocks, col_blocks, 0);
}

/* Calculate total rows and cols for augmented matrix */

/**
 * @brief Calculates the total number of rows in the augmented matrix.
 * @tparam Size The size of the rows array.
 * @param rows The array of rows.
 * @param dim The dimension of the augmented matrix.
 * @param i The current row index.
 * @return The total number of rows in the augmented matrix.
 */
template <std::size_t Size>
constexpr std::size_t
calculate_total_rows(const std::array<std::size_t, Size> &rows,
                     std::size_t row_blocks, std::size_t col_blocks,
                     std::size_t i = 0) {
  return (i == row_blocks)
             ? 0
             : rows[i * col_blocks] +
                   calculate_total_rows(rows, row_blocks, col_blocks, i + 1);
}

/**
 * @brief Calculates the total number of columns in the augmented matrix.
 * @tparam Size The size of the columns array.
 * @param cols The array of columns.
 * @param dim The dimension of the augmented matrix.
 * @param j The current column index.
 * @return The total number of columns in the augmented matrix.
 */
template <std::size_t Size>
constexpr std::size_t
calculate_total_cols(const std::array<std::size_t, Size> &cols,
                     std::size_t col_blocks, std::size_t j = 0) {
  return (j == col_blocks)
             ? 0
             : cols[j] + calculate_total_cols(cols, col_blocks, j + 1);
}

/* get / set */

/**
 * @brief Gets the block row index for a given global row index.
 * @tparam N The size of the rows array.
 * @param global_row The global row index.
 * @param rows The array of rows.
 * @param dim The dimension of the augmented matrix.
 * @param current The current block row index (default is 0).
 * @param accum The accumulated number of rows (default is 0).
 * @return The block row index corresponding to the global row index.
 */
template <std::size_t N>
constexpr std::size_t
get_block_row_idx(std::size_t global_row,
                  const std::array<std::size_t, N> &rows,
                  std::size_t row_blocks, std::size_t col_blocks,
                  std::size_t current = 0, std::size_t accum = 0) {
  return (current == row_blocks) ? 0
         : (global_row < accum + rows[current * col_blocks])
             ? current
             : get_block_row_idx(global_row, rows, row_blocks, col_blocks,
                                 current + 1,
                                 accum + rows[current * col_blocks]);
}

/**
 * @brief Gets the local row index for a given global row index.
 * @tparam N The size of the rows array.
 * @param global_row The global row index.
 * @param rows The array of rows.
 * @param dim The dimension of the augmented matrix.
 * @param current The current block row index (default is 0).
 * @param accum The accumulated number of rows (default is 0).
 * @return The local row index corresponding to the global row index.
 */
template <std::size_t N>
constexpr std::size_t
get_local_row_idx(std::size_t global_row,
                  const std::array<std::size_t, N> &rows,
                  std::size_t row_blocks, std::size_t col_blocks,
                  std::size_t current = 0, std::size_t accum = 0) {
  return (current == row_blocks) ? 0
         : (global_row < accum + rows[current * col_blocks])
             ? (global_row - accum)
             : get_local_row_idx(global_row, rows, row_blocks, col_blocks,
                                 current + 1,
                                 accum + rows[current * col_blocks]);
}

/**
 * @brief Gets the block column index for a given global column index.
 * @tparam N The size of the columns array.
 * @param global_col The global column index.
 * @param cols The array of columns.
 * @param dim The dimension of the augmented matrix.
 * @param current The current block column index (default is 0).
 * @param accum The accumulated number of columns (default is 0).
 * @return The block column index corresponding to the global column index.
 */
template <std::size_t N>
constexpr std::size_t get_block_col_idx(std::size_t global_col,
                                        const std::array<std::size_t, N> &cols,
                                        std::size_t col_blocks,
                                        std::size_t current = 0,
                                        std::size_t accum = 0) {
  return (current == col_blocks) ? 0
         : (global_col < accum + cols[current])
             ? current
             : get_block_col_idx(global_col, cols, col_blocks, current + 1,
                                 accum + cols[current]);
}

/**
 * @brief Gets the local column index for a given global column index.
 * @tparam N The size of the columns array.
 * @param global_col The global column index.
 * @param cols The array of columns.
 * @param dim The dimension of the augmented matrix.
 * @param current The current block column index (default is 0).
 * @param accum The accumulated number of columns (default is 0).
 * @return The local column index corresponding to the global column index.
 */
template <std::size_t N>
constexpr std::size_t get_local_col_idx(std::size_t global_col,
                                        const std::array<std::size_t, N> &cols,
                                        std::size_t col_blocks,
                                        std::size_t current = 0,
                                        std::size_t accum = 0) {
  return (current == col_blocks) ? 0
         : (global_col < accum + cols[current])
             ? (global_col - accum)
             : get_local_col_idx(global_col, cols, col_blocks, current + 1,
                                 accum + cols[current]);
}

} // namespace AugmentedMatrixAction

/* Augmented Matrix */

/**
 * @brief A class representing an augmented matrix composed of smaller matrices.
 * @tparam Tuple_Type A std::tuple containing the smaller matrices.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
class AugmentedMatrix {
public:
  /* Check Compatibility */
  static_assert(AugmentedMatrixAction::is_tuple<Tuple_Type>::value,
                "Tuple_Type must be a std::tuple.");

  /* Type */
  using Value_Type =
      typename std::tuple_element<0, Tuple_Type>::type::Value_Type;

  static constexpr std::size_t ROW_BLOCKS =
      (Row_Blocks == 0 && Col_Blocks == 0)
          ? AugmentedMatrixAction::tuple_square_root<Tuple_Type>::value
          : Row_Blocks;

  static constexpr std::size_t COL_BLOCKS =
      (Row_Blocks == 0 && Col_Blocks == 0)
          ? AugmentedMatrixAction::tuple_square_root<Tuple_Type>::value
          : Col_Blocks;

  static_assert(ROW_BLOCKS * COL_BLOCKS == std::tuple_size<Tuple_Type>::value,
                "The number of elements in Tuple_Type does not match "
                "ROW_BLOCKS * COL_BLOCKS.");

  static constexpr auto ELEMENT_ROWS =
      AugmentedMatrixAction::matrix_shape_extractor<Tuple_Type>::ROWS;
  static constexpr auto ELEMENT_COLS =
      AugmentedMatrixAction::matrix_shape_extractor<Tuple_Type>::COLS;

  /* Check Compatibility */
  static_assert(AugmentedMatrixAction::is_compatible_augmented_matrix(
                    ELEMENT_ROWS, ELEMENT_COLS, ROW_BLOCKS, COL_BLOCKS),
                "The rows or columns of the nested matrices do not align "
                "perfectly in the augmented matrix.");

  /* Type */
  static constexpr std::size_t ROWS =
      AugmentedMatrixAction::calculate_total_rows(ELEMENT_ROWS, ROW_BLOCKS,
                                                  COL_BLOCKS);

  static constexpr std::size_t COLS =
      AugmentedMatrixAction::calculate_total_cols(ELEMENT_COLS, COL_BLOCKS);

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
  AugmentedMatrix(
      const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &input)
      : matrix(input.matrix) {}

  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &
  operator=(const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &input) {
    if (this != &input) {
      this->matrix = input.matrix;
    }
    return *this;
  }

  /* Move Constructor */
  AugmentedMatrix(
      AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &&input) noexcept
      : matrix(std::move(input.matrix)) {}

  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &operator=(
      AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &&input) noexcept {
    if (this != &input) {
      this->matrix = std::move(input.matrix);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Gets the number of columns of the augmented matrix.
   * @return The number of columns.
   */
  constexpr std::size_t cols() const { return COLS; }

  /**
   * @brief Gets the number of rows of the augmented matrix.
   * @return The number of rows.
   */
  constexpr std::size_t rows() const { return ROWS; }

  /**
   * @brief Gets the total size (number of elements) of the augmented matrix.
   * @return The total size.
   */
  constexpr std::size_t size() const { return ROWS * COLS; }

  /**
   * @brief Gets the shape of the augmented matrix as a tuple (rows, cols).
   * @return A tuple containing the number of rows and columns.
   */
  std::tuple<std::size_t, std::size_t> shape() const {
    return std::make_tuple(static_cast<std::size_t>(ROWS),
                           static_cast<std::size_t>(COLS));
  }

  /**
   * @brief Gets the number of dimensions of the augmented matrix.
   * @return The number of dimensions (always 2 for a matrix).
   */
  std::size_t ndim() const { return 2; }

  /**
   * @brief Gets the value at the specified row and column indices.
   * @tparam ROW_IN The row index (compile-time constant).
   * @tparam COL_IN The column index (compile-time constant).
   * @return The value at the specified indices.
   */
  template <std::size_t ROW_IN, std::size_t COL_IN> inline T_ get() const {
    static_assert(ROW_IN < ROWS && COL_IN < COLS,
                  "ROW and COL must be within the bounds of the matrix.");

    constexpr std::size_t block_row = AugmentedMatrixAction::get_block_row_idx(
        ROW_IN, ELEMENT_ROWS, ROW_BLOCKS, COL_BLOCKS);
    constexpr std::size_t block_col = AugmentedMatrixAction::get_block_col_idx(
        COL_IN, ELEMENT_COLS, COL_BLOCKS);

    constexpr std::size_t tuple_idx = block_row * COL_BLOCKS + block_col;

    constexpr std::size_t local_row = AugmentedMatrixAction::get_local_row_idx(
        ROW_IN, ELEMENT_ROWS, ROW_BLOCKS, COL_BLOCKS);
    constexpr std::size_t local_col = AugmentedMatrixAction::get_local_col_idx(
        COL_IN, ELEMENT_COLS, COL_BLOCKS);

    return std::get<tuple_idx>(this->matrix)
        .template get<local_row, local_col>();
  }

  /**
   * @brief Sets the value at the specified row and column indices.
   * @tparam ROW_IN The row index (compile-time constant).
   * @tparam COL_IN The column index (compile-time constant).
   * @param value The value to set at the specified indices.
   */
  template <std::size_t ROW_IN, std::size_t COL_IN>
  inline void set(const T_ &value) {
    static_assert(ROW_IN < ROWS && COL_IN < COLS,
                  "ROW and COL must be within the bounds of the matrix.");

    constexpr std::size_t block_row = AugmentedMatrixAction::get_block_row_idx(
        ROW_IN, ELEMENT_ROWS, ROW_BLOCKS, COL_BLOCKS);
    constexpr std::size_t block_col = AugmentedMatrixAction::get_block_col_idx(
        COL_IN, ELEMENT_COLS, COL_BLOCKS);

    constexpr std::size_t tuple_idx = block_row * COL_BLOCKS + block_col;

    constexpr std::size_t local_row = AugmentedMatrixAction::get_local_row_idx(
        ROW_IN, ELEMENT_ROWS, ROW_BLOCKS, COL_BLOCKS);
    constexpr std::size_t local_col = AugmentedMatrixAction::get_local_col_idx(
        COL_IN, ELEMENT_COLS, COL_BLOCKS);

    std::get<tuple_idx>(this->matrix).template set<local_row, local_col>(value);
  }

  template <typename Matrix_Type> inline auto to_matrix() const -> Matrix_Type {
    Matrix_Type dense_matrix;

    substitute_matrix(dense_matrix, *this);

    return dense_matrix;
  }

  /**
   * @brief Gets the value at the specified linear index.
   * @param index The linear index.
   * @return The value at the specified index.
   */
  T_ operator()(std::size_t index) const {
    if (index >= ROWS * COLS) {
      index = ROWS * COLS - 1;
    }

    std::size_t row = index / COLS;
    std::size_t col = index % COLS;

    Matrix<DefDense, T_, ROWS, COLS> dense_matrix;

    substitute_matrix(dense_matrix, *this);

    return dense_matrix(row, col);
  }

  /**
   * @brief Gets the value at the specified row and column indices.
   * @param row The row index.
   * @param col The column index.
   * @return The value at the specified indices.
   */
  T_ operator()(std::size_t row, std::size_t col) const {
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

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I, std::size_t J_idx>
struct Row {
  /**
   * @brief Computes the addition of two matrices for a specific row and column
   * index.
   * @param A The first matrix.
   * @param B The second matrix.
   * @param result The result matrix where the sum will be stored.
   */
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    result.template set<I, J_idx>(A.template get<I, J_idx>() +
                                  B.template get<I, J_idx>());
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I,
        J_idx - 1>::compute(A, B, result);
  }
};

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I>
struct Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I, 0> {
  /**
   * @brief Computes the addition of two matrices for a specific row and the
   * first column index (J_idx = 0).
   * @param A The first matrix.
   * @param B The second matrix.
   * @param result The result matrix where the sum will be stored.
   */
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    result.template set<I, 0>(A.template get<I, 0>() + B.template get<I, 0>());
  }
};

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I_idx>
struct Column {
  /**
   * @brief Computes the addition of two matrices for a specific column index
   * (I_idx) and all rows.
   * @param A The first matrix.
   * @param B The second matrix.
   * @param result The result matrix where the sum will be stored.
   */
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I_idx,
        Matrix_Result_Type::COLS - 1>::compute(A, B, result);
    Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type,
           I_idx - 1>::compute(A, B, result);
  }
};

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type>
struct Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, 0> {
  /**
   * @brief Computes the addition of two matrices for the first column index
   * (I_idx = 0) and all rows.
   * @param A The first matrix.
   * @param B The second matrix.
   * @param result The result matrix where the sum will be stored.
   */
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, 0,
        Matrix_Result_Type::COLS - 1>::compute(A, B, result);
  }
};

/**
 * @brief Computes the addition of two matrices and stores the result in a
 * third matrix.
 * @tparam Matrix_A_Type The type of the first matrix.
 * @tparam Matrix_B_Type The type of the second matrix.
 * @tparam Matrix_Result_Type The type of the result matrix.
 * @param A The first matrix.
 * @param B The second matrix.
 * @param result The result matrix where the sum will be stored.
 */
template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type>
inline void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                    Matrix_Result_Type &result) {
  Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type,
         Matrix_Result_Type::ROWS - 1>::compute(A, B, result);
}

} // namespace AugmentedMatrixAddMatrix

/**
 * @brief Overloaded operator+ to add a Matrix and an AugmentedMatrix.
 * @tparam Matrix_Type The type of the Matrix.
 * @tparam Tuple_Type The type of the AugmentedMatrix (tuple of matrices).
 * @param matrix The Matrix to be added.
 * @param augmented_matrix The AugmentedMatrix to be added.
 * @return A new Matrix containing the result of the addition.
 */
template <typename Matrix_Type, typename Tuple_Type, std::size_t Row_Blocks,
          std::size_t Col_Blocks>
inline auto operator+(
    const Matrix_Type &matrix,
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
              Matrix_Type::COLS> {

  static_assert(
      std::is_same<typename Matrix_Type::Value_Type,
                   typename AugmentedMatrix<Tuple_Type, Row_Blocks,
                                            Col_Blocks>::Value_Type>::value,
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

/**
 * @brief Overloaded operator+ to add two AugmentedMatrices.
 * @tparam Tuple_A_Type The type of the first AugmentedMatrix (tuple of
 * matrices).
 * @tparam Tuple_B_Type The type of the second AugmentedMatrix (tuple of
 * matrices).
 * @param augmented_matrix_a The first AugmentedMatrix to be added.
 * @param augmented_matrix_b The second AugmentedMatrix to be added.
 * @return A new Matrix containing the result of the addition.
 */
template <typename Matrix_Type, typename Tuple_Type, std::size_t Row_Blocks,
          std::size_t Col_Blocks>
inline auto operator+(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix,
    const Matrix_Type &matrix)
    -> Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
              Matrix_Type::COLS> {

  static_assert(
      std::is_same<typename Matrix_Type::Value_Type,
                   typename AugmentedMatrix<Tuple_Type, Row_Blocks,
                                            Col_Blocks>::Value_Type>::value,
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

/**
 * @brief Overloaded operator+ to add two AugmentedMatrices.
 * @tparam Tuple_A_Type The type of the first AugmentedMatrix (tuple of
 * matrices).
 * @tparam Tuple_B_Type The type of the second AugmentedMatrix (tuple of
 * matrices).
 * @param augmented_matrix_a The first AugmentedMatrix to be added.
 * @param augmented_matrix_b The second AugmentedMatrix to be added.
 * @return A new Matrix containing the result of the addition.
 */
template <typename Tuple_A_Type, std::size_t RA, std::size_t CA,
          typename Tuple_B_Type, std::size_t RB, std::size_t CB>
inline auto
operator+(const AugmentedMatrix<Tuple_A_Type, RA, CA> &augmented_matrix_a,
          const AugmentedMatrix<Tuple_B_Type, RB, CB> &augmented_matrix_b)
    -> Matrix<DefDense,
              typename AugmentedMatrix<Tuple_A_Type, RA, CA>::Value_Type,
              AugmentedMatrix<Tuple_A_Type, RA, CA>::ROWS,
              AugmentedMatrix<Tuple_A_Type, RA, CA>::COLS> {

  static_assert(
      std::is_same<
          typename AugmentedMatrix<Tuple_A_Type, RA, CA>::Value_Type,
          typename AugmentedMatrix<Tuple_B_Type, RB, CB>::Value_Type>::value,
      "Matrix_Type and AugmentedMatrix_Type must have the same value type.");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  using Matrix_Type =
      Matrix<DefDense,
             typename AugmentedMatrix<Tuple_A_Type, RA, CA>::Value_Type,
             AugmentedMatrix<Tuple_A_Type, RA, CA>::ROWS,
             AugmentedMatrix<Tuple_A_Type, RA, CA>::COLS>;

  return augmented_matrix_a.template to_matrix<Matrix_Type>() +
         augmented_matrix_b.template to_matrix<Matrix_Type>();

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  Matrix<DefDense, typename AugmentedMatrix<Tuple_A_Type, RA, CA>::Value_Type,
         AugmentedMatrix<Tuple_A_Type, RA, CA>::ROWS,
         AugmentedMatrix<Tuple_A_Type, RA, CA>::COLS>
      result;
  AugmentedMatrixAddMatrix::compute(augmented_matrix_a, augmented_matrix_b,
                                    result);
  return result;

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/* Matrix Sub AugmentedMatrix */

namespace AugmentedMatrixSubMatrix {

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I, std::size_t J_idx>
struct Row {
  /**
   * @brief Computes the subtraction of two matrices for a specific row and
   * column index.
   * @param A The first matrix.
   * @param B The second matrix.
   * @param result The result matrix where the difference will be stored.
   */
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    result.template set<I, J_idx>(A.template get<I, J_idx>() -
                                  B.template get<I, J_idx>());
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I,
        J_idx - 1>::compute(A, B, result);
  }
};

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I>
struct Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I, 0> {
  /**
   * @brief Computes the subtraction of two matrices for a specific row and the
   * first column index (J_idx = 0).
   * @param A The first matrix.
   * @param B The second matrix.
   * @param result The result matrix where the difference will be stored.
   */
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    result.template set<I, 0>(A.template get<I, 0>() - B.template get<I, 0>());
  }
};

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I_idx>
struct Column {
  /**
   * @brief Computes the subtraction of two matrices for a specific column index
   * (I_idx) and all rows.
   * @param A The first matrix.
   * @param B The second matrix.
   * @param result The result matrix where the difference will be stored.
   */
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I_idx,
        Matrix_Result_Type::COLS - 1>::compute(A, B, result);
    Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type,
           I_idx - 1>::compute(A, B, result);
  }
};

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type>
struct Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, 0> {
  /**
   * @brief Computes the subtraction of two matrices for the first column index
   * (I_idx = 0) and all rows.
   * @param A The first matrix.
   * @param B The second matrix.
   * @param result The result matrix where the difference will be stored.
   */
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, 0,
        Matrix_Result_Type::COLS - 1>::compute(A, B, result);
  }
};

/**
 * @brief Computes the subtraction of two matrices and stores the result in a
 * third matrix.
 * @tparam Matrix_A_Type The type of the first matrix.
 * @tparam Matrix_B_Type The type of the second matrix.
 * @tparam Matrix_Result_Type The type of the result matrix.
 * @param A The first matrix.
 * @param B The second matrix.
 * @param result The result matrix where the difference will be stored.
 */
template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type>
inline void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                    Matrix_Result_Type &result) {
  Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type,
         Matrix_Result_Type::ROWS - 1>::compute(A, B, result);
}

} // namespace AugmentedMatrixSubMatrix

/**
 * @brief Overloaded operator- to subtract an AugmentedMatrix from a Matrix.
 * @tparam Matrix_Type The type of the Matrix.
 * @tparam Tuple_Type The type of the AugmentedMatrix (tuple of matrices).
 * @param matrix The Matrix to be subtracted from.
 * @param augmented_matrix The AugmentedMatrix to subtract.
 * @return A new Matrix containing the result of the subtraction.
 */
template <typename Matrix_Type, typename Tuple_Type, std::size_t Row_Blocks,
          std::size_t Col_Blocks>
inline auto operator-(
    const Matrix_Type &matrix,
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
              Matrix_Type::COLS> {

  static_assert(
      std::is_same<typename Matrix_Type::Value_Type,
                   typename AugmentedMatrix<Tuple_Type, Row_Blocks,
                                            Col_Blocks>::Value_Type>::value,
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

/**
 * @brief Overloaded operator- to subtract a Matrix from an AugmentedMatrix.
 * @tparam Matrix_Type The type of the Matrix.
 * @tparam Tuple_Type The type of the AugmentedMatrix (tuple of matrices).
 * @param augmented_matrix The AugmentedMatrix to be subtracted from.
 * @param matrix The Matrix to subtract.
 * @return A new Matrix containing the result of the subtraction.
 */
template <typename Matrix_Type, typename Tuple_Type, std::size_t Row_Blocks,
          std::size_t Col_Blocks>
inline auto operator-(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix,
    const Matrix_Type &matrix)
    -> Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
              Matrix_Type::COLS> {

  static_assert(
      std::is_same<typename Matrix_Type::Value_Type,
                   typename AugmentedMatrix<Tuple_Type, Row_Blocks,
                                            Col_Blocks>::Value_Type>::value,
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

/**
 * @brief Overloaded operator- to subtract two AugmentedMatrices.
 * @tparam Tuple_A_Type The type of the first AugmentedMatrix (tuple of
 * matrices).
 * @tparam Tuple_B_Type The type of the second AugmentedMatrix (tuple of
 * matrices).
 * @param augmented_matrix_a The first AugmentedMatrix to be subtracted.
 * @param augmented_matrix_b The second AugmentedMatrix to subtract.
 * @return A new Matrix containing the result of the subtraction.
 */
template <typename Tuple_A_Type, std::size_t RA, std::size_t CA,
          typename Tuple_B_Type, std::size_t RB, std::size_t CB>
inline auto
operator-(const AugmentedMatrix<Tuple_A_Type, RA, CA> &augmented_matrix_a,
          const AugmentedMatrix<Tuple_B_Type, RB, CB> &augmented_matrix_b)
    -> Matrix<DefDense,
              typename AugmentedMatrix<Tuple_A_Type, RA, CA>::Value_Type,
              AugmentedMatrix<Tuple_A_Type, RA, CA>::ROWS,
              AugmentedMatrix<Tuple_A_Type, RA, CA>::COLS> {

  static_assert(
      std::is_same<
          typename AugmentedMatrix<Tuple_A_Type, RA, CA>::Value_Type,
          typename AugmentedMatrix<Tuple_B_Type, RB, CB>::Value_Type>::value,
      "Matrix_Type and AugmentedMatrix_Type must have the same value type.");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  using Matrix_Type =
      Matrix<DefDense,
             typename AugmentedMatrix<Tuple_A_Type, RA, CA>::Value_Type,
             AugmentedMatrix<Tuple_A_Type, RA, CA>::ROWS,
             AugmentedMatrix<Tuple_A_Type, RA, CA>::COLS>;

  return augmented_matrix_a.template to_matrix<Matrix_Type>() -
         augmented_matrix_b.template to_matrix<Matrix_Type>();

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  Matrix<DefDense, typename AugmentedMatrix<Tuple_A_Type, RA, CA>::Value_Type,
         AugmentedMatrix<Tuple_A_Type, RA, CA>::ROWS,
         AugmentedMatrix<Tuple_A_Type, RA, CA>::COLS>
      result;
  AugmentedMatrixSubMatrix::compute(augmented_matrix_a, augmented_matrix_b,
                                    result);
  return result;

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/* AugmentedMatrix Unary Minus */

namespace AugmentedMatrixUnaryMinus {

template <typename Matrix_A_Type, typename Matrix_Result_Type, std::size_t I,
          std::size_t J_idx>
struct Row {
  /**
   * @brief Computes the unary minus of a matrix for a specific row and column
   * index.
   * @param A The input matrix.
   * @param result The result matrix where the negated values will be stored.
   */
  static void compute(const Matrix_A_Type &A, Matrix_Result_Type &result) {
    result.template set<I, J_idx>(-A.template get<I, J_idx>());
    Row<Matrix_A_Type, Matrix_Result_Type, I, J_idx - 1>::compute(A, result);
  }
};

template <typename Matrix_A_Type, typename Matrix_Result_Type, std::size_t I>
struct Row<Matrix_A_Type, Matrix_Result_Type, I, 0> {
  /**
   * @brief Computes the unary minus of a matrix for a specific row and the
   * first column index (J_idx = 0).
   * @param A The input matrix.
   * @param result The result matrix where the negated values will be stored.
   */
  static void compute(const Matrix_A_Type &A, Matrix_Result_Type &result) {
    result.template set<I, 0>(-A.template get<I, 0>());
  }
};

template <typename Matrix_A_Type, typename Matrix_Result_Type,
          std::size_t I_idx>
struct Column {
  /**
   * @brief Computes the unary minus of a matrix for a specific column index
   * (I_idx) and all rows.
   * @param A The input matrix.
   * @param result The result matrix where the negated values will be stored.
   */
  static void compute(const Matrix_A_Type &A, Matrix_Result_Type &result) {
    Row<Matrix_A_Type, Matrix_Result_Type, I_idx,
        Matrix_Result_Type::COLS - 1>::compute(A, result);
    Column<Matrix_A_Type, Matrix_Result_Type, I_idx - 1>::compute(A, result);
  }
};

template <typename Matrix_A_Type, typename Matrix_Result_Type>
struct Column<Matrix_A_Type, Matrix_Result_Type, 0> {
  /**
   * @brief Computes the unary minus of a matrix for the first column index
   * (I_idx = 0) and all rows.
   * @param A The input matrix.
   * @param result The result matrix where the negated values will be stored.
   */
  static void compute(const Matrix_A_Type &A, Matrix_Result_Type &result) {
    Row<Matrix_A_Type, Matrix_Result_Type, 0,
        Matrix_Result_Type::COLS - 1>::compute(A, result);
  }
};

/**
 * @brief Computes the unary minus of a matrix and stores the result in a
 * second matrix.
 * @tparam Matrix_A_Type The type of the input matrix.
 * @tparam Matrix_Result_Type The type of the result matrix.
 * @param A The input matrix.
 * @param result The result matrix where the negated values will be stored.
 */
template <typename Matrix_A_Type, typename Matrix_Result_Type>
inline void compute(const Matrix_A_Type &A, Matrix_Result_Type &result) {
  Column<Matrix_A_Type, Matrix_Result_Type,
         Matrix_Result_Type::ROWS - 1>::compute(A, result);
}

} // namespace AugmentedMatrixUnaryMinus

/**
 * @brief Overloaded operator- to compute the unary minus of an AugmentedMatrix.
 * @tparam Tuple_Type The type of the AugmentedMatrix (tuple of matrices).
 * @param augmented_matrix The AugmentedMatrix to negate.
 * @return A new AugmentedMatrix containing the negated values.
 */
template <typename Tuple_Type, std::size_t Row_Blocks, std::size_t Col_Blocks>
inline auto operator-(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {

  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  AugmentedMatrixUnaryMinus::compute(augmented_matrix, result);

  return result;
}

/* Matrix Mul AugmentedMatrix */

namespace AugmentedMatrixMulMatrix {

template <typename Matrix_A_Type, typename Matrix_B_Type, typename Value_Type,
          std::size_t I, std::size_t J, std::size_t K_idx>
struct DotProduct {
  /**
   * @brief Computes the dot product of two matrices for a specific row and
   * column index.
   * @param A The first matrix.
   * @param B The second matrix.
   * @return The computed dot product value.
   */
  static Value_Type compute(const Matrix_A_Type &A, const Matrix_B_Type &B) {
    return A.template get<I, K_idx>() * B.template get<K_idx, J>() +
           DotProduct<Matrix_A_Type, Matrix_B_Type, Value_Type, I, J,
                      K_idx - 1>::compute(A, B);
  }
};

template <typename Matrix_A_Type, typename Matrix_B_Type, typename Value_Type,
          std::size_t I, std::size_t J>
struct DotProduct<Matrix_A_Type, Matrix_B_Type, Value_Type, I, J, 0> {
  /**
   * @brief Computes the dot product of two matrices for a specific row and
   * column index when K_idx = 0 (base case).
   * @param A The first matrix.
   * @param B The second matrix.
   * @return The computed dot product value.
   */
  static Value_Type compute(const Matrix_A_Type &A, const Matrix_B_Type &B) {
    return A.template get<I, 0>() * B.template get<0, J>();
  }
};

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I, std::size_t J_idx>
struct Row {
  /**
   * @brief Computes the dot product of two matrices for a specific row and
   * column index and stores the result in a third matrix.
   * @param A The first matrix.
   * @param B The second matrix.
   * @param result The result matrix where the computed values will be stored.
   */
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    result.template set<I, J_idx>(
        DotProduct<Matrix_A_Type, Matrix_B_Type,
                   typename Matrix_Result_Type::Value_Type, I, J_idx,
                   Matrix_A_Type::COLS - 1>::compute(A, B));
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I,
        J_idx - 1>::compute(A, B, result);
  }
};

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I>
struct Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I, 0> {
  /**
   * @brief Computes the dot product of two matrices for a specific row and the
   * first column index (J_idx = 0) and stores the result in a third matrix.
   * @param A The first matrix.
   * @param B The second matrix.
   * @param result The result matrix where the computed values will be stored.
   */
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    result.template set<I, 0>(
        DotProduct<Matrix_A_Type, Matrix_B_Type,
                   typename Matrix_Result_Type::Value_Type, I, 0,
                   Matrix_A_Type::COLS - 1>::compute(A, B));
  }
};

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type, std::size_t I_idx>
struct Column {
  /**
   * @brief Computes the dot product of two matrices for a specific column index
   * (I_idx) and all rows, storing the results in a third matrix.
   * @param A The first matrix.
   * @param B The second matrix.
   * @param result The result matrix where the computed values will be stored.
   */
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, I_idx,
        Matrix_Result_Type::COLS - 1>::compute(A, B, result);
    Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type,
           I_idx - 1>::compute(A, B, result);
  }
};

template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type>
struct Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, 0> {
  /**
   * @brief Computes the dot product of two matrices for the first column index
   * (I_idx = 0) and all rows, storing the results in a third matrix.
   * @param A The first matrix.
   * @param B The second matrix.
   * @param result The result matrix where the computed values will be stored.
   */
  static void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                      Matrix_Result_Type &result) {
    Row<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type, 0,
        Matrix_Result_Type::COLS - 1>::compute(A, B, result);
  }
};

/**
 * @brief Computes the dot product of two matrices and stores the result in a
 * third matrix.
 * @tparam Matrix_A_Type The type of the first matrix.
 * @tparam Matrix_B_Type The type of the second matrix.
 * @tparam Matrix_Result_Type The type of the result matrix.
 * @param A The first matrix.
 * @param B The second matrix.
 * @param result The result matrix where the computed values will be stored.
 */
template <typename Matrix_A_Type, typename Matrix_B_Type,
          typename Matrix_Result_Type>
inline void compute(const Matrix_A_Type &A, const Matrix_B_Type &B,
                    Matrix_Result_Type &result) {
  Column<Matrix_A_Type, Matrix_B_Type, Matrix_Result_Type,
         Matrix_Result_Type::ROWS - 1>::compute(A, B, result);
}

} // namespace AugmentedMatrixMulMatrix

/**
 * @brief Overloaded operator* to multiply an AugmentedMatrix with a Matrix.
 * @tparam Tuple_Type The type of the AugmentedMatrix (tuple of matrices).
 * @tparam Matrix_Type The type of the Matrix.
 * @param augmented_matrix The AugmentedMatrix to be multiplied.
 * @param matrix The Matrix to multiply with.
 * @return A new Matrix containing the result of the multiplication.
 */
template <typename Tuple_Type, std::size_t Row_Blocks, std::size_t Col_Blocks,
          typename Matrix_Type>
inline auto operator*(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix,
    const Matrix_Type &matrix)
    -> Matrix<DefDense, typename Matrix_Type::Value_Type,
              AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks>::ROWS,
              Matrix_Type::COLS> {

  static_assert(
      std::is_same<typename Matrix_Type::Value_Type,
                   typename AugmentedMatrix<Tuple_Type, Row_Blocks,
                                            Col_Blocks>::Value_Type>::value,
      "Matrix_Type and AugmentedMatrix_Type must have the same value type.");

  static_assert(AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks>::COLS ==
                    Matrix_Type::ROWS,
                "Inner matrix dimensions must agree for multiplication.");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  using Value_Type = typename Matrix_Type::Value_Type;

  Matrix<DefDense, typename Matrix_Type::Value_Type,
         AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks>::ROWS,
         Matrix_Type::COLS>
      result;

  for (std::size_t i = 0;
       i < AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks>::ROWS; ++i) {
    for (std::size_t j = 0; j < Matrix_Type::COLS; ++j) {
      Value_Type sum = 0;
      for (std::size_t k = 0;
           k < AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks>::COLS; ++k) {
        Value_Type a = augmented_matrix(i, k);
        Value_Type b = matrix(k, j);
        sum += a * b;
      }
      result(i, j) = sum;
    }
  }

  return result;

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  Matrix<DefDense, typename Matrix_Type::Value_Type,
         AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks>::ROWS,
         Matrix_Type::COLS>
      result;

  AugmentedMatrixMulMatrix::compute(augmented_matrix, matrix, result);

  return result;

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/**
 * @brief Overloaded operator* to multiply a Matrix with an AugmentedMatrix.
 * @tparam Matrix_Type The type of the Matrix.
 * @tparam Tuple_Type The type of the AugmentedMatrix (tuple of matrices).
 * @param matrix The Matrix to be multiplied.
 * @param augmented_matrix The AugmentedMatrix to multiply with.
 * @return A new Matrix containing the result of the multiplication.
 */
template <typename Matrix_Type, typename Tuple_Type, std::size_t Row_Blocks,
          std::size_t Col_Blocks>
inline auto operator*(
    const Matrix_Type &matrix,
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
              AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks>::COLS> {

  static_assert(
      std::is_same<typename Matrix_Type::Value_Type,
                   typename AugmentedMatrix<Tuple_Type, Row_Blocks,
                                            Col_Blocks>::Value_Type>::value,
      "Matrix_Type and AugmentedMatrix_Type must have the same value type.");

  static_assert(Matrix_Type::COLS ==
                    AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks>::ROWS,
                "Inner matrix dimensions must agree for multiplication.");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  using Value_Type = typename Matrix_Type::Value_Type;

  Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
         AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks>::COLS>
      result;

  for (std::size_t i = 0; i < Matrix_Type::ROWS; ++i) {
    for (std::size_t j = 0;
         j < AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks>::COLS; ++j) {
      Value_Type sum = 0;
      for (std::size_t k = 0; k < Matrix_Type::COLS; ++k) {
        Value_Type a = matrix(i, k);
        Value_Type b = augmented_matrix(k, j);
        sum += a * b;
      }
      result(i, j) = sum;
    }
  }

  return result;

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  Matrix<DefDense, typename Matrix_Type::Value_Type, Matrix_Type::ROWS,
         AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks>::COLS>
      result;

  AugmentedMatrixMulMatrix::compute(matrix, augmented_matrix, result);

  return result;

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/**
 * @brief Overloaded operator* to multiply two AugmentedMatrices.
 * @tparam Tuple_A_Type The type of the first AugmentedMatrix (tuple of
 * matrices).
 * @tparam Tuple_B_Type The type of the second AugmentedMatrix (tuple of
 * matrices).
 * @param augmented_matrix_a The first AugmentedMatrix to be multiplied.
 * @param augmented_matrix_b The second AugmentedMatrix to multiply with.
 * @return A new Matrix containing the result of the multiplication.
 */
template <typename Tuple_A_Type, std::size_t RA, std::size_t CA,
          typename Tuple_B_Type, std::size_t RB, std::size_t CB>
inline auto
operator*(const AugmentedMatrix<Tuple_A_Type, RA, CA> &augmented_matrix_a,
          const AugmentedMatrix<Tuple_B_Type, RB, CB> &augmented_matrix_b)
    -> Matrix<DefDense,
              typename AugmentedMatrix<Tuple_A_Type, RA, CA>::Value_Type,
              AugmentedMatrix<Tuple_A_Type, RA, CA>::ROWS,
              AugmentedMatrix<Tuple_B_Type, RB, CB>::COLS> {

  static_assert(
      std::is_same<
          typename AugmentedMatrix<Tuple_A_Type, RA, CA>::Value_Type,
          typename AugmentedMatrix<Tuple_B_Type, RB, CB>::Value_Type>::value,
      "AugmentedMatrix<Tuple_A_Type> and AugmentedMatrix<Tuple_B_Type> must "
      "have the same value type.");

  static_assert(AugmentedMatrix<Tuple_A_Type, RA, CA>::COLS ==
                    AugmentedMatrix<Tuple_B_Type, RB, CB>::ROWS,
                "Inner matrix dimensions must agree for multiplication.");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  Matrix<DefDense, typename AugmentedMatrix<Tuple_A_Type, RA, CA>::Value_Type,
         AugmentedMatrix<Tuple_A_Type, RA, CA>::ROWS,
         AugmentedMatrix<Tuple_B_Type, RB, CB>::COLS>
      result;

  for (std::size_t i = 0; i < AugmentedMatrix<Tuple_A_Type, RA, CA>::ROWS;
       ++i) {
    for (std::size_t j = 0; j < AugmentedMatrix<Tuple_B_Type, RB, CB>::COLS;
         ++j) {
      typename AugmentedMatrix<Tuple_A_Type, RA, CA>::Value_Type sum = 0;
      for (std::size_t k = 0; k < AugmentedMatrix<Tuple_A_Type, RA, CA>::COLS;
           ++k) {
        sum += augmented_matrix_a(i, k) * augmented_matrix_b(k, j);
      }
      result(i, j) = sum;
    }
  }

  return result;

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  Matrix<DefDense, typename AugmentedMatrix<Tuple_A_Type, RA, CA>::Value_Type,
         AugmentedMatrix<Tuple_A_Type, RA, CA>::ROWS,
         AugmentedMatrix<Tuple_B_Type, RB, CB>::COLS>
      result;

  AugmentedMatrixMulMatrix::compute(augmented_matrix_a, augmented_matrix_b,
                                    result);

  return result;

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/**
 * @brief Type alias for a tuple of AugmentedMatrix types.
 * @tparam Matrices The types of the matrices to be included in the tuple.
 */
template <typename... Matrices>
using AugmentedMatrix_Tuple_Type = std::tuple<Matrices...>;

/**
 * @brief Type alias for an AugmentedMatrix type constructed from a tuple of
 * matrices.
 * @tparam Matrices The types of the matrices to be included in the
 * AugmentedMatrix.
 */
template <typename... Matrices>
using AugmentedMatrix_Type = AugmentedMatrix<std::tuple<Matrices...>>;

/**
 * @brief Creates an AugmentedMatrix from the provided matrices.
 * @tparam Matrices The types of the matrices to be included in the
 * AugmentedMatrix.
 * @param inputs The matrices to be included in the AugmentedMatrix.
 * @return An AugmentedMatrix containing the provided matrices.
 */
template <std::size_t Row_Blocks = 0, std::size_t Col_Blocks = 0,
          typename... Matrices>
inline auto make_AugmentedMatrix(const Matrices &...inputs)
    -> AugmentedMatrix<std::tuple<Matrices...>, Row_Blocks, Col_Blocks> {
  return AugmentedMatrix<std::tuple<Matrices...>, Row_Blocks, Col_Blocks>(
      inputs...);
}

/**
 * @brief Creates an AugmentedMatrix filled with zeros from the provided
 * matrices.
 * @tparam Matrices The types of the matrices to be included in the
 * AugmentedMatrix.
 * @return An AugmentedMatrix filled with zeros.
 */
template <std::size_t Row_Blocks = 0, std::size_t Col_Blocks = 0,
          typename... Matrices>
inline auto make_AugmentedMatrixZeros()
    -> AugmentedMatrix<std::tuple<Matrices...>, Row_Blocks, Col_Blocks> {
  return AugmentedMatrix<std::tuple<Matrices...>, Row_Blocks, Col_Blocks>();
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_AUGMENTED_MATRIX_HPP_
