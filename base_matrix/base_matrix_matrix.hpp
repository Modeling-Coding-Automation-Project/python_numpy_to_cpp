/**
 * @file base_matrix_matrix.hpp
 * @brief Provides a generic, fixed-size or dynamic matrix class and a suite of
 * matrix operations for numerical computing.
 *
 * This header defines the Base::Matrix::Matrix class template, which supports
 * both static (std::array) and dynamic (std::vector) storage for matrices of
 * arbitrary type and size. It includes a comprehensive set of matrix operations
 * such as addition, subtraction, scalar multiplication, matrix multiplication,
 * transpose, trace, row/column swapping, and conversions between real and
 * complex matrices. The implementation leverages template metaprogramming for
 * compile-time recursion as well as runtime for-loop alternatives, controlled
 * by macros.
 *
 * @note
 * tparam M is the number of rows in the matrix.
 * tparam N is the number of columns in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef BASE_MATRIX_MATRIX_HPP_
#define BASE_MATRIX_MATRIX_HPP_

#include "base_matrix_macros.hpp"

#include "base_matrix_complex.hpp"
#include "base_matrix_vector.hpp"
#include "base_utility.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

/**
 * @brief A fixed-size Matrix class template supporting various storage
 backends.
 *
 * This Matrix class supports both std::vector and std::array as underlying
 storage,
 * controlled by the preprocessor macro BASE_MATRIX_USE_STD_VECTOR_.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 */
template <typename T, std::size_t M, std::size_t N> class Matrix {
public:
  /* Constant */
  static constexpr std::size_t ROWS = M;
  static constexpr std::size_t COLS = N;

public:
  /* Constructor */
#ifdef BASE_MATRIX_USE_STD_VECTOR_

  Matrix() : data(N, std::vector<T>(M, static_cast<T>(0))) {}

  Matrix(const std::initializer_list<std::initializer_list<T>> &input)
      : data(N, std::vector<T>(M, static_cast<T>(0))) {

    auto outer_it = input.begin();
    for (std::size_t i = 0; i < M; i++) {
      auto inner_it = outer_it->begin();
      for (std::size_t j = 0; j < N; j++) {
        this->data[j][i] = *inner_it;
        ++inner_it;
      }
      ++outer_it;
    }
  }

  Matrix(const std::vector<T> &input)
      : data(N, std::vector<T>(M, static_cast<T>(0))) {
    // This may cause runtime error if the size of values is larger than M.
    std::copy(input.begin(), input.end(), this->data[0].begin());
  }

  Matrix(T input[][N]) : data(N, std::vector<T>(M, static_cast<T>(0))) {

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        this->data[j][i] = input[i][j];
      }
    }
  }

#else // BASE_MATRIX_USE_STD_VECTOR_

  Matrix() : data{} {}

  Matrix(const std::initializer_list<std::initializer_list<T>> &input)
      : data{} {

    auto outer_it = input.begin();
    for (std::size_t i = 0; i < M; i++) {
      auto inner_it = outer_it->begin();
      for (std::size_t j = 0; j < N; j++) {
        this->data[j][i] = *inner_it;
        ++inner_it;
      }
      ++outer_it;
    }
  }

  Matrix(const std::array<std::array<T, N>, M> &input) : data(input) {}

  Matrix(const std::vector<T> &input) : data{} {
    // This may cause runtime error if the size of values is larger than M.
    std::copy(input.begin(), input.end(), this->data[0].begin());
  }

  Matrix(const std::array<T, M> &input) : data{} {
    Base::Utility::copy<T, 0, M, 0, M, M>(input, this->data[0]);
  }

  Matrix(T input[][N]) : data{} {

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        this->data[j][i] = input[i][j];
      }
    }
  }

#endif // BASE_MATRIX_USE_STD_VECTOR_

  /* Copy Constructor */
  Matrix(const Matrix<T, M, N> &other) : data(other.data) {}

  Matrix<T, M, N> &operator=(const Matrix<T, M, N> &other) {
    if (this != &other) {
      this->data = other.data;
    }
    return *this;
  }

  /* Move Constructor */
  Matrix(Matrix<T, M, N> &&other) noexcept : data(std::move(other.data)) {}

  Matrix<T, M, N> &operator=(Matrix<T, M, N> &&other) noexcept {
    if (this != &other) {
      this->data = std::move(other.data);
    }
    return *this;
  }

public:
  /* Function */

  /* Identity */

  template <typename U, std::size_t P, std::size_t Start, std::size_t End,
            typename Enable = void>
  struct CreateIdentityCore;

  template <typename U, std::size_t P, std::size_t Start, std::size_t End>
  struct CreateIdentityCore<U, P, Start, End,
                            typename std::enable_if<(End - Start > 1)>::type> {
    static constexpr std::size_t Mid = Start + (End - Start) / 2;

    static void compute(Matrix<U, P, P> &identity) {
      CreateIdentityCore<U, P, Start, Mid>::compute(identity);
      CreateIdentityCore<U, P, Mid, End>::compute(identity);
    }
  };

  template <typename U, std::size_t P, std::size_t Start, std::size_t End>
  struct CreateIdentityCore<U, P, Start, End,
                            typename std::enable_if<(End == Start)>::type> {
    static void compute(Matrix<U, P, P> &) {}
  };

  template <typename U, std::size_t P, std::size_t Start, std::size_t End>
  struct CreateIdentityCore<U, P, Start, End,
                            typename std::enable_if<(End - Start == 1)>::type> {
    static void compute(Matrix<U, P, P> &identity) {
      identity.template set<Start, Start>(static_cast<U>(1));
    }
  };

  /**
   * @brief Constructs an identity matrix of size P x P.
   *
   * This function uses template metaprogramming to recursively set the diagonal
   * elements of the identity matrix to 1.
   *
   * @tparam U The type of the matrix elements.
   * @tparam P The size of the identity matrix (P x P).
   * @param identity The identity matrix to be constructed.
   */
  template <typename U, std::size_t P>
  static inline void COMPILED_MATRIX_IDENTITY(Matrix<U, P, P> &identity) {
    CreateIdentityCore<U, P, 0, P>::compute(identity);
  }

  /**
   * @brief Creates an identity matrix of size M x M.
   *
   * This function constructs an identity matrix where all diagonal elements are
   * set to 1 and all off-diagonal elements are set to 0.
   *
   * @return Matrix<T, M, M> The identity matrix of size M x M.
   */
  static inline Matrix<T, M, M> identity() {
    Matrix<T, M, M> identity;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    for (std::size_t i = 0; i < M; i++) {
      identity(i, i) = static_cast<T>(1);
    }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    COMPILED_MATRIX_IDENTITY<T, M>(identity);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    return identity;
  }

  /* Full */
  template <typename U, std::size_t O, std::size_t P, std::size_t I,
            std::size_t Start, std::size_t End, typename Enable = void>
  struct MatrixFullColumn;

  template <typename U, std::size_t O, std::size_t P, std::size_t I,
            std::size_t Start, std::size_t End>
  struct MatrixFullColumn<U, O, P, I, Start, End,
                          typename std::enable_if<(End - Start > 1)>::type> {
    static constexpr std::size_t Mid = Start + (End - Start) / 2;

    static void compute(Matrix<U, O, P> &Full, const U &value) {
      MatrixFullColumn<U, O, P, I, Start, Mid>::compute(Full, value);
      MatrixFullColumn<U, O, P, I, Mid, End>::compute(Full, value);
    }
  };

  template <typename U, std::size_t O, std::size_t P, std::size_t I,
            std::size_t Start, std::size_t End>
  struct MatrixFullColumn<U, O, P, I, Start, End,
                          typename std::enable_if<(End == Start)>::type> {
    static void compute(Matrix<U, O, P> &, const U &) {}
  };

  template <typename U, std::size_t O, std::size_t P, std::size_t I,
            std::size_t Start, std::size_t End>
  struct MatrixFullColumn<U, O, P, I, Start, End,
                          typename std::enable_if<(End - Start == 1)>::type> {
    static void compute(Matrix<U, O, P> &Full, const U &value) {
      Full.template set<I, Start>(value);
    }
  };

  template <typename U, std::size_t O, std::size_t P, std::size_t Start,
            std::size_t End, typename Enable = void>
  struct MatrixFullRow;

  template <typename U, std::size_t O, std::size_t P, std::size_t Start,
            std::size_t End>
  struct MatrixFullRow<U, O, P, Start, End,
                       typename std::enable_if<(End - Start > 1)>::type> {
    static constexpr std::size_t Mid = Start + (End - Start) / 2;

    static void compute(Matrix<U, O, P> &Full, const U &value) {
      MatrixFullRow<U, O, P, Start, Mid>::compute(Full, value);
      MatrixFullRow<U, O, P, Mid, End>::compute(Full, value);
    }
  };

  template <typename U, std::size_t O, std::size_t P, std::size_t Start,
            std::size_t End>
  struct MatrixFullRow<U, O, P, Start, End,
                       typename std::enable_if<(End == Start)>::type> {
    static void compute(Matrix<U, O, P> &, const U &) {}
  };

  template <typename U, std::size_t O, std::size_t P, std::size_t Start,
            std::size_t End>
  struct MatrixFullRow<U, O, P, Start, End,
                       typename std::enable_if<(End - Start == 1)>::type> {
    static void compute(Matrix<U, O, P> &Full, const U &value) {
      MatrixFullColumn<U, O, P, Start, 0, P>::compute(Full, value);
    }
  };

  /**
   * @brief Constructs a full matrix of size O x P with all elements set to a
   * specified value.
   *
   * This function uses template metaprogramming to recursively set all elements
   * of the matrix to the given value.
   *
   * @tparam U The type of the matrix elements.
   * @tparam O The number of rows in the matrix.
   * @tparam P The number of columns in the matrix.
   * @param Full The full matrix to be constructed.
   * @param value The value to set in the matrix.
   */
  template <typename U, std::size_t O, std::size_t P>
  static inline void COMPILED_MATRIX_FULL(Matrix<U, O, P> &Full,
                                          const U &value) {
    MatrixFullRow<U, O, P, 0, O>::compute(Full, value);
  }

  /**
   * @brief Creates a full matrix of size M x N with all elements set to 1.
   *
   * This function constructs a matrix where all elements are initialized to
   * the value 1.
   *
   * @return Matrix<T, M, N> The full matrix of size M x N with all elements set
   * to 1.
   */
  static inline Matrix<T, M, N> ones() {
    Matrix<T, M, N> Ones;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        Ones(i, j) = static_cast<T>(1);
      }
    }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    COMPILED_MATRIX_FULL<T, M, N>(Ones, static_cast<T>(1));

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    return Ones;
  }

  /**
   * @brief Creates a full matrix of size M x N with all elements set to a
   * specified value.
   *
   * This function constructs a matrix where all elements are initialized to the
   * given value.
   *
   * @param value The value to set in all elements of the matrix.
   * @return Matrix<T, M, N> The full matrix of size M x N with all elements set
   * to the specified value.
   */
  static inline Matrix<T, M, N> full(const T &value) {
    Matrix<T, M, N> Full;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        Full(i, j) = value;
      }
    }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    COMPILED_MATRIX_FULL<T, M, N>(Full, value);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    return Full;
  }

  /**
   * @brief Creates a row vector from a specified row of the matrix.
   *
   * This function extracts a row from the matrix and returns it as a Vector
   * object.
   *
   * @param row The index of the column to extract (0-based).
   * @return Vector<T, M> The extracted row vector.
   */
  inline Vector<T, M> create_row_vector(std::size_t row) const {
    Vector<T, M> result;

    if (row >= M) {
      row = M - 1;
    }

    Base::Utility::copy<T, 0, M, 0, M, M>(this->data[row], result.data);

    return result;
  }

  /**
   * @brief Creates a column vector from a specified column of the matrix.
   *
   * This function extracts a column from the matrix and returns it as a Vector
   * object.
   *
   * @param col The index of the row to extract (0-based).
   * @return Vector<T, N> The extracted column vector.
   */
  T &operator()(std::size_t col, std::size_t row) {

    return this->data[row][col];
  }

  /**
   * @brief Accesses an element at the specified column and row indices.
   *
   * This function provides access to the element at the specified column and
   * row indices, allowing both read and write operations.
   *
   * @param col The index of the row (0-based).
   * @param row The index of the column (0-based).
   * @return T& Reference to the element at the specified position.
   */
  const T &operator()(std::size_t col, std::size_t row) const {

    return this->data[row][col];
  }

#ifdef BASE_MATRIX_USE_STD_VECTOR_

  /**
   * @brief Accesses a row of the matrix.
   *
   * This function provides access to a specific row of the matrix, returning a
   * reference to the vector representing that row. If the specified row index
   * is out of bounds, it defaults to the last row.
   *
   * @param row The index of the column to access (0-based).
   * @return std::vector<T>& Reference to the vector representing the specified
   * row.
   */
  std::vector<T> &operator()(std::size_t row) {
    if (row >= N) {
      row = N - 1;
    }

    return this->data[row];
  }

  /**
   * @brief Accesses a row of the matrix (const version).
   *
   * This function provides access to a specific row of the matrix, returning a
   * const reference to the vector representing that row. If the specified row
   * index is out of bounds, it defaults to the last row.
   *
   * @param row The index of the column to access (0-based).
   * @return const std::vector<T>& Const reference to the vector representing
   * the specified row.
   */
  const std::vector<T> &operator()(std::size_t row) const {
    if (row >= N) {
      row = N - 1;
    }

    return this->data[row];
  }

#else // BASE_MATRIX_USE_STD_VECTOR_

  /**
   * @brief Accesses a row of the matrix.
   *
   * This function provides access to a specific row of the matrix, returning a
   * reference to the array representing that row. If the specified row index is
   * out of bounds, it defaults to the last row.
   *
   * @param row The index of the column to access (0-based).
   * @return std::array<T, M>& Reference to the array representing the specified
   * row.
   */
  std::array<T, M> &operator()(std::size_t row) {
    if (row >= N) {
      row = N - 1;
    }

    return this->data[row];
  }

  /**
   * @brief Accesses a row of the matrix (const version).
   *
   * This function provides access to a specific row of the matrix, returning a
   * const reference to the array representing that row. If the specified row
   * index is out of bounds, it defaults to the last row.
   *
   * @param row The index of the column to access (0-based).
   * @return const std::array<T, M>& Const reference to the array representing
   * the specified row.
   */
  const std::array<T, M> &operator()(std::size_t row) const {
    if (row >= N) {
      row = N - 1;
    }

    return this->data[row];
  }

#endif // BASE_MATRIX_USE_STD_VECTOR_

  /**
   * @brief Returns the number of columns in the matrix.
   *
   * This function returns the number of columns in the matrix, which is a
   * compile-time constant.
   *
   * @return std::size_t The number of columns in the matrix.
   */
  constexpr std::size_t cols() const { return N; }

  /**
   * @brief Returns the number of rows in the matrix.
   *
   * This function returns the number of rows in the matrix, which is a
   * compile-time constant.
   *
   * @return std::size_t The number of rows in the matrix.
   */
  constexpr std::size_t rows() const { return M; }

  /**
   * @brief Returns the number of elements in the matrix.
   *
   * This function returns the total number of elements in the matrix, which is
   * the product of the number of columns and rows.
   *
   * @return std::size_t The total number of elements in the matrix.
   */
  inline Vector<T, M> get_row(std::size_t row) const {
    if (row >= N) {
      row = N - 1;
    }

    Vector<T, M> result;

    Base::Utility::copy<T, 0, M, 0, M, M>(this->data[row], result.data);

    return result;
  }

  /**
   * @brief Sets a specific row of the matrix to the values from a Vector.
   *
   * This function copies the values from the provided Vector into the specified
   * row of the matrix. If the row index is out of bounds, it defaults to the
   * last row.
   *
   * @param row The index of the column to set (0-based).
   * @param row_vector The Vector containing the values to set in the row.
   */
  inline void set_row(std::size_t row, const Vector<T, M> &row_vector) {
    if (row >= N) {
      row = N - 1;
    }

    Base::Utility::copy<T, 0, M, 0, M, M>(row_vector.data, this->data[row]);
  }

  /**
   * @brief Returns the number of elements in the matrix.
   *
   * This function returns the total number of elements in the matrix, which is
   * the product of the number of columns and rows.
   *
   * @return std::size_t The total number of elements in the matrix.
   */
  inline Matrix<T, M, M> inv() const {
    Matrix<T, M, M> X_temp = Matrix<T, M, M>::identity();
    std::array<T, M> rho;
    std::array<std::size_t, M> rep_num;

    Matrix<T, M, M> Inv;
    std::tie(Inv, rho, rep_num) = gmres_k_matrix_inv(
        *this, static_cast<T>(0.0), static_cast<T>(1.0e-10), X_temp);

    return Inv;
  }

  /* Get Dense Matrix value */

  /**
   * @brief Gets the value at the specified column and row indices.
   *
   * This function retrieves the value at the specified column and row indices
   * from the matrix. It performs compile-time checks to ensure that the indices
   * are within bounds.
   *
   * @tparam ROW The index of the row (0-based).
   * @tparam COL The index of the column (0-based).
   * @return T The value at the specified position in the matrix.
   */
  template <std::size_t ROW, std::size_t COL> inline T get() const {
    static_assert(ROW < M, "Row Index is out of range.");
    static_assert(COL < N, "Column Index is out of range.");

    return data[COL][ROW];
  }

  /* Set Dense Matrix value */

  /**
   * @brief Sets the value at the specified column and row indices.
   *
   * This function assigns a value to the specified column and row indices in
   * the matrix. It performs compile-time checks to ensure that the indices are
   * within bounds.
   *
   * @tparam ROW The index of the row (0-based).
   * @tparam COL The index of the column (0-based).
   * @param value The value to set at the specified position in the matrix.
   */
  template <std::size_t ROW, std::size_t COL> inline void set(const T &value) {
    static_assert(ROW < M, "Row Index is out of range.");
    static_assert(COL < N, "Column Index is out of range.");

    data[COL][ROW] = value;
  }

  /**
   * @brief Gets the trace of the matrix.
   *
   * This function calculates the trace of the matrix, which is the sum of the
   * diagonal elements. It performs a compile-time check to ensure that the
   * matrix is square (M == N).
   *
   * @return T The trace of the matrix.
   */
  inline T get_trace() const { return output_matrix_trace(*this); }

/* Variable */
#ifdef BASE_MATRIX_USE_STD_VECTOR_
  std::vector<std::vector<T>> data;
#else  // BASE_MATRIX_USE_STD_VECTOR_
  std::array<std::array<T, M>, N> data;
#endif // BASE_MATRIX_USE_STD_VECTOR_
};

/* swap rows */
namespace MatrixSwapRows {

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Core<T, M, N, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(std::size_t row_1, std::size_t row_2,
                      Matrix<T, M, N> &mat, T temp) {
    Core<T, M, N, Start, Mid>::compute(row_1, row_2, mat, temp);
    Core<T, M, N, Mid, End>::compute(row_1, row_2, mat, temp);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Core<T, M, N, Start, End,
            typename std::enable_if<(End == Start)>::type> {
  static void compute(std::size_t, std::size_t, Matrix<T, M, N> &, T) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Core<T, M, N, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(std::size_t row_1, std::size_t row_2,
                      Matrix<T, M, N> &mat, T temp) {
    temp = mat.data[Start][row_1];
    mat.data[Start][row_1] = mat.data[Start][row_2];
    mat.data[Start][row_2] = temp;
  }
};

/**
 * @brief Computes the column swap operation for a matrix.
 *
 * This function uses template metaprogramming to recursively swap two rows
 * in the matrix, starting from the last row and moving upwards.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param row_1 The index of the first row to swap.
 * @param row_2 The index of the second row to swap.
 * @param mat The matrix in which the rows are swapped.
 * @param temp A temporary variable to hold values during swapping.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(std::size_t row_1, std::size_t row_2, Matrix<T, M, N> &mat,
                    T &temp) {
  Core<T, M, N, 0, N>::compute(row_1, row_2, mat, temp);
}

} // namespace MatrixSwapRows

/**
 * @brief Swaps two rows in a matrix.
 *
 * This function swaps the elements of two specified rows in the matrix.
 * If the column indices are out of bounds, they are adjusted to the last valid
 * index.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param row_1 The index of the first row to swap (0-based).
 * @param row_2 The index of the second row to swap (0-based).
 * @param mat The matrix in which the rows are swapped.
 */
template <typename T, std::size_t M, std::size_t N>
inline void matrix_row_swap(std::size_t row_1, std::size_t row_2,
                            Matrix<T, M, N> &mat) {
  T temp = static_cast<T>(0);

  if (row_1 >= M) {
    row_1 = M - 1;
  }
  if (row_2 >= M) {
    row_2 = M - 1;
  }

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < N; i++) {

    temp = mat.data[i][row_1];
    mat.data[i][row_1] = mat.data[i][row_2];
    mat.data[i][row_2] = temp;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixSwapRows::compute<T, M, N>(row_1, row_2, mat, temp);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/* swap cols */

/**
 * @brief Swaps two cols in a matrix.
 *
 * This function swaps the elements of two specified cols in the matrix.
 * If the row indices are out of bounds, they are adjusted to the last valid
 * index.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param col_1 The index of the first column to swap (0-based).
 * @param col_2 The index of the second column to swap (0-based).
 * @param mat The matrix in which the cols are swapped.
 */
template <typename T, std::size_t M, std::size_t N>
inline void matrix_col_swap(std::size_t col_1, std::size_t col_2,
                            Matrix<T, M, N> &mat) {
  Vector<T, M> temp_vec;

  if (col_1 >= N) {
    col_1 = N - 1;
  }
  if (col_2 >= N) {
    col_2 = N - 1;
  }

  Base::Utility::copy<T, 0, M, 0, M, M>(mat(col_1), temp_vec.data);
  Base::Utility::copy<T, 0, M, 0, M, M>(mat(col_2), mat(col_1));
  Base::Utility::copy<T, 0, M, 0, M, M>(temp_vec.data, mat(col_2));
}

/* Trace */
namespace MatrixTrace {

template <typename T, std::size_t N, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t N, std::size_t Start, std::size_t End>
struct Core<T, N, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static T compute(const Matrix<T, N, N> &mat) {
    return Core<T, N, Start, Mid>::compute(mat) +
           Core<T, N, Mid, End>::compute(mat);
  }
};

template <typename T, std::size_t N, std::size_t Start, std::size_t End>
struct Core<T, N, Start, End, typename std::enable_if<(End == Start)>::type> {
  static T compute(const Matrix<T, N, N> &) { return static_cast<T>(0); }
};

template <typename T, std::size_t N, std::size_t Start, std::size_t End>
struct Core<T, N, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static T compute(const Matrix<T, N, N> &mat) {
    return mat.template get<Start, Start>();
  }
};

/**
 * @brief Computes the trace of a square matrix.
 *
 * This function uses template metaprogramming to recursively calculate the
 * trace of a square matrix by summing its diagonal elements.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix (must be equal to N).
 * @tparam N The number of columns in the matrix (must be equal to M).
 * @param mat The square matrix from which the trace is computed.
 * @return T The computed trace of the matrix.
 */
template <typename T, std::size_t M, std::size_t N>
inline T compute(const Matrix<T, M, N> &mat) {
  return Core<T, N, 0, N>::compute(mat);
}

} // namespace MatrixTrace

/**
 * @brief Computes the trace of a square matrix.
 *
 * This function calculates the trace of a square matrix by summing its diagonal
 * elements. It performs a compile-time check to ensure that the matrix is
 * square (M == N).
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix (must be equal to N).
 * @tparam N The number of columns in the matrix (must be equal to M).
 * @param mat The square matrix from which the trace is computed.
 * @return T The computed trace of the matrix.
 */
template <typename T, std::size_t M, std::size_t N>
inline T output_matrix_trace(const Matrix<T, M, N> &mat) {
  static_assert(M == N, "Matrix must be square matrix");
  T trace = static_cast<T>(0);

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; i++) {
    trace += mat(i, i);
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  trace = MatrixTrace::compute<T, M, N>(mat);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return trace;
}

/* Matrix Addition */
namespace MatrixAddMatrix {

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Row<T, M, N, I, Start, Mid>::compute(A, B, result);
    Row<T, M, N, I, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, const Matrix<T, M, N> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    result.template set<I, Start>(A.template get<I, Start>() +
                                  B.template get<I, Start>());
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, Start, Mid>::compute(A, B, result);
    Column<T, M, N, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, const Matrix<T, M, N> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Row<T, M, N, Start, 0, N>::compute(A, B, result);
  }
};

/**
 * @brief Computes the sum of two matrices.
 *
 * This function uses template metaprogramming to recursively add the elements
 * of two matrices and store the result in a third matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param A The first matrix to add.
 * @param B The second matrix to add.
 * @param result The matrix where the result is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                    Matrix<T, M, N> &result) {
  Column<T, M, N, 0, M>::compute(A, B, result);
}

} // namespace MatrixAddMatrix

/**
 * @brief Adds two matrices of the same size.
 *
 * This function computes the element-wise sum of two matrices A and B, both of
 * size M x N, and returns the resulting matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrices.
 * @tparam N The number of columns in the matrices.
 * @param A The first matrix to add.
 * @param B The second matrix to add.
 * @return Matrix<T, M, N> The resulting matrix after addition.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator+(const Matrix<T, M, N> &A,
                                 const Matrix<T, M, N> &B) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(i, j) = A(i, j) + B(i, j);
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixAddMatrix::compute<T, M, N>(A, B, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Matrix Subtraction */
namespace MatrixSubMatrix {

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Row<T, M, N, I, Start, Mid>::compute(A, B, result);
    Row<T, M, N, I, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, const Matrix<T, M, N> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    result.template set<I, Start>(A.template get<I, Start>() -
                                  B.template get<I, Start>());
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, Start, Mid>::compute(A, B, result);
    Column<T, M, N, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, const Matrix<T, M, N> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Row<T, M, N, Start, 0, N>::compute(A, B, result);
  }
};

/**
 * @brief Computes the difference of two matrices.
 *
 * This function uses template metaprogramming to recursively subtract the
 * elements of two matrices and store the result in a third matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param A The first matrix to subtract.
 * @param B The second matrix to subtract.
 * @param result The matrix where the result is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                    Matrix<T, M, N> &result) {
  Column<T, M, N, 0, M>::compute(A, B, result);
}

} // namespace MatrixSubMatrix

/**
 * @brief Subtracts two matrices of the same size.
 *
 * This function computes the element-wise difference of two matrices A and B,
 * both of size M x N, and returns the resulting matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrices.
 * @tparam N The number of columns in the matrices.
 * @param A The first matrix to subtract.
 * @param B The second matrix to subtract.
 * @return Matrix<T, M, N> The resulting matrix after subtraction.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator-(const Matrix<T, M, N> &A,
                                 const Matrix<T, M, N> &B) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(i, j) = A(i, j) - B(i, j);
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixSubMatrix::compute<T, M, N>(A, B, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

namespace MatrixMinus {

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
    Row<T, M, N, I, Start, Mid>::compute(A, result);
    Row<T, M, N, I, Mid, End>::compute(A, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
    result.template set<I, Start>(-A.template get<I, Start>());
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
    Column<T, M, N, Start, Mid>::compute(A, result);
    Column<T, M, N, Mid, End>::compute(A, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
    Row<T, M, N, Start, 0, N>::compute(A, result);
  }
};

/**
 * @brief Computes the negation of a matrix.
 *
 * This function uses template metaprogramming to recursively negate the
 * elements of a matrix and store the result in another matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param A The matrix to negate.
 * @param result The matrix where the result is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
  Column<T, M, N, 0, M>::compute(A, result);
}

} // namespace MatrixMinus

/**
 * @brief Negates a matrix.
 *
 * This function computes the negation of a matrix A, which means it multiplies
 * each element of the matrix by -1, and returns the resulting matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param A The matrix to negate.
 * @return Matrix<T, M, N> The resulting negated matrix.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator-(const Matrix<T, M, N> &A) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(i, j) = -A(i, j);
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixMinus::compute<T, M, N>(A, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* (Scalar) * (Matrix) */
namespace MatrixMultiplyScalar {

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {
    Row<T, M, N, I, Start, Mid>::compute(scalar, mat, result);
    Row<T, M, N, I, Mid, End>::compute(scalar, mat, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(const T &, const Matrix<T, M, N> &, Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {
    result.template set<I, Start>(scalar * mat.template get<I, Start>());
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, Start, Mid>::compute(scalar, mat, result);
    Column<T, M, N, Mid, End>::compute(scalar, mat, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const T &, const Matrix<T, M, N> &, Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {
    Row<T, M, N, Start, 0, N>::compute(scalar, mat, result);
  }
};

/**
 * @brief Computes the product of a scalar and a matrix.
 *
 * This function uses template metaprogramming to recursively multiply the
 * elements of a matrix by a scalar and store the result in another matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param scalar The scalar value to multiply with.
 * @param mat The matrix to multiply with the scalar.
 * @param result The matrix where the result is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const T &scalar, const Matrix<T, M, N> &mat,
                    Matrix<T, M, N> &result) {
  Column<T, M, N, 0, M>::compute(scalar, mat, result);
}

} // namespace MatrixMultiplyScalar

/**
 * @brief Multiplies a scalar with a matrix.
 *
 * This function computes the product of a scalar and a matrix, which means it
 * multiplies each element of the matrix by the scalar, and returns the
 * resulting matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param scalar The scalar value to multiply with.
 * @param mat The matrix to multiply with the scalar.
 * @return Matrix<T, M, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator*(const T &scalar, const Matrix<T, M, N> &mat) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = scalar * mat(j, k);
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixMultiplyScalar::compute<T, M, N>(scalar, mat, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/**
 * @brief Multiplies a matrix with a scalar.
 *
 * This function computes the product of a matrix and a scalar, which means it
 * multiplies each element of the matrix by the scalar, and returns the
 * resulting matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param mat The matrix to multiply with the scalar.
 * @param scalar The scalar value to multiply with.
 * @return Matrix<T, M, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator*(const Matrix<T, M, N> &mat, const T &scalar) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = scalar * mat(j, k);
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixMultiplyScalar::compute<T, M, N>(scalar, mat, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Matrix multiply Vector */
namespace MatrixMultiplyVector {

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Core<T, M, N, I, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec) {
    return Core<T, M, N, I, Start, Mid>::compute(mat, vec) +
           Core<T, M, N, I, Mid, End>::compute(mat, vec);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Core<T, M, N, I, Start, End,
            typename std::enable_if<(End == Start)>::type> {
  static T compute(const Matrix<T, M, N> &, const Vector<T, N> &) {
    return static_cast<T>(0);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Core<T, M, N, I, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec) {
    return mat.template get<I, Start>() * vec[Start];
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec,
                      Vector<T, M> &result) {
    Column<T, M, N, Start, Mid>::compute(mat, vec, result);
    Column<T, M, N, Mid, End>::compute(mat, vec, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, const Vector<T, N> &,
                      Vector<T, M> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec,
                      Vector<T, M> &result) {
    result[Start] = Core<T, M, N, Start, 0, N>::compute(mat, vec);
  }
};

/**
 * @brief Computes the product of a matrix and a vector.
 *
 * This function uses template metaprogramming to recursively compute the dot
 * product of each row of the matrix with the vector and store the results in a
 * new vector.
 *
 * @tparam T The type of the matrix and vector elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param mat The matrix to multiply with the vector.
 * @param vec The vector to multiply with the matrix.
 * @param result The vector where the result is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec,
                    Vector<T, M> &result) {
  Column<T, M, N, 0, M>::compute(mat, vec, result);
}

} // namespace MatrixMultiplyVector

/**
 * @brief Multiplies a matrix with a vector.
 *
 * This function computes the product of a matrix and a vector, which means it
 * performs the dot product of each row of the matrix with the vector, and
 * returns the resulting vector.
 *
 * @tparam T The type of the matrix and vector elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param mat The matrix to multiply with the vector.
 * @param vec The vector to multiply with the matrix.
 * @return Vector<T, M> The resulting vector after multiplication.
 */
template <typename T, std::size_t M, std::size_t N>
inline Vector<T, M> operator*(const Matrix<T, M, N> &mat,
                              const Vector<T, N> &vec) {
  Vector<T, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    T sum = 0;
    for (std::size_t j = 0; j < N; ++j) {
      sum += mat(i, j) * vec[j];
    }
    result[i] = sum;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixMultiplyVector::compute<T, M, N>(mat, vec, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

namespace VectorMultiplyMatrix {

template <typename T, std::size_t L, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Core;

template <typename T, std::size_t L, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End>
struct Core<T, L, N, J, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {
    Core<T, L, N, J, Start, Mid>::compute(vec, mat, result);
    Core<T, L, N, J, Mid, End>::compute(vec, mat, result);
  }
};

template <typename T, std::size_t L, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End>
struct Core<T, L, N, J, Start, End,
            typename std::enable_if<(End == Start)>::type> {
  static void compute(const Vector<T, L> &, const Matrix<T, 1, N> &,
                      Matrix<T, L, N> &) {}
};

template <typename T, std::size_t L, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End>
struct Core<T, L, N, J, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {
    result.template set<Start, J>(vec[Start] * mat.template get<0, J>());
  }
};

template <typename T, std::size_t L, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t L, std::size_t N, std::size_t Start,
          std::size_t End>
struct Row<T, L, N, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {
    Row<T, L, N, Start, Mid>::compute(vec, mat, result);
    Row<T, L, N, Mid, End>::compute(vec, mat, result);
  }
};

template <typename T, std::size_t L, std::size_t N, std::size_t Start,
          std::size_t End>
struct Row<T, L, N, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const Vector<T, L> &, const Matrix<T, 1, N> &,
                      Matrix<T, L, N> &) {}
};

template <typename T, std::size_t L, std::size_t N, std::size_t Start,
          std::size_t End>
struct Row<T, L, N, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {
    Core<T, L, N, Start, 0, L>::compute(vec, mat, result);
  }
};

/**
 * @brief Computes the product of a vector and a matrix.
 *
 * This function uses template metaprogramming to recursively compute the
 * product of each element of the vector with the corresponding column of the
 * matrix and store the results in a new matrix.
 *
 * @tparam T The type of the vector and matrix elements.
 * @tparam L The size of the vector.
 * @tparam M The number of columns in the matrix (should be 1).
 * @tparam N The number of rows in the matrix.
 * @param vec The vector to multiply with the matrix.
 * @param mat The matrix to multiply with the vector.
 * @param result The matrix where the result is stored.
 */
template <typename T, std::size_t L, std::size_t M, std::size_t N>
inline void compute(const Vector<T, L> &vec, const Matrix<T, M, N> &mat,
                    Matrix<T, L, N> &result) {
  Row<T, L, N, 0, N>::compute(vec, mat, result);
}

} // namespace VectorMultiplyMatrix

/**
 * @brief Multiplies a vector with a matrix.
 *
 * This function computes the product of a vector and a matrix, which means it
 * multiplies each element of the vector with the corresponding column of the
 * matrix, and returns the resulting matrix.
 *
 * @tparam T The type of the vector and matrix elements.
 * @tparam L The size of the vector.
 * @tparam M The number of rows in the matrix (should be 1).
 * @tparam N The number of columns in the matrix.
 * @param vec The vector to multiply with the matrix.
 * @param mat The matrix to multiply with the vector.
 * @return Matrix<T, L, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t L, std::size_t M, std::size_t N>
inline Matrix<T, L, N> operator*(const Vector<T, L> &vec,
                                 const Matrix<T, M, N> &mat) {
  static_assert(M == 1, "Invalid size.");
  Matrix<T, L, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t k = 0; k < L; ++k) {
      result(k, j) = vec[k] * mat(0, j);
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  VectorMultiplyMatrix::compute<T, L, M, N>(vec, mat, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* (Column Vector) * (Matrix) */
namespace ColumnVectorMultiplyMatrix {

template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End>
struct Core<T, M, N, J, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static T compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat) {
    return Core<T, M, N, J, Start, Mid>::compute(vec, mat) +
           Core<T, M, N, J, Mid, End>::compute(vec, mat);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End>
struct Core<T, M, N, J, Start, End,
            typename std::enable_if<(End == Start)>::type> {
  static T compute(const ColVector<T, M> &, const Matrix<T, M, N> &) {
    return static_cast<T>(0);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End>
struct Core<T, M, N, J, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static T compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat) {
    return vec[Start] * mat.template get<Start, J>();
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Row<T, M, N, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat,
                      ColVector<T, N> &result) {
    Row<T, M, N, Start, Mid>::compute(vec, mat, result);
    Row<T, M, N, Mid, End>::compute(vec, mat, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Row<T, M, N, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const ColVector<T, M> &, const Matrix<T, M, N> &,
                      ColVector<T, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Row<T, M, N, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat,
                      ColVector<T, N> &result) {
    result[Start] = Core<T, M, N, Start, 0, M>::compute(vec, mat);
  }
};

/**
 * @brief Computes the product of a column vector and a matrix.
 *
 * This function uses template metaprogramming to recursively compute the
 * product of each element of the column vector with the corresponding row of
 * the matrix and store the results in a new column vector.
 *
 * @tparam T The type of the column vector and matrix elements.
 * @tparam M The size of the column vector.
 * @tparam N The number of columns in the matrix.
 * @param vec The column vector to multiply with the matrix.
 * @param mat The matrix to multiply with the column vector.
 * @param result The column vector where the result is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat,
                    ColVector<T, N> &result) {
  Row<T, M, N, 0, N>::compute(vec, mat, result);
}

} // namespace ColumnVectorMultiplyMatrix

/**
 * @brief Multiplies a column vector with a matrix.
 *
 * This function computes the product of a column vector and a matrix, which
 * means it multiplies each element of the column vector with the corresponding
 * row of the matrix, and returns the resulting column vector.
 *
 * @tparam T The type of the column vector and matrix elements.
 * @tparam M The size of the column vector.
 * @tparam N The number of columns in the matrix.
 * @param vec The column vector to multiply with the matrix.
 * @param mat The matrix to multiply with the column vector.
 * @return ColVector<T, N> The resulting column vector after multiplication.
 */
template <typename T, std::size_t M, std::size_t N>
inline ColVector<T, N> operator*(const ColVector<T, M> &vec,
                                 const Matrix<T, M, N> &mat) {
  ColVector<T, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t j = 0; j < N; ++j) {
    T sum = 0;
    for (std::size_t i = 0; i < M; ++i) {
      sum += vec[i] * mat(i, j);
    }
    result[j] = sum;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  ColumnVectorMultiplyMatrix::compute<T, M, N>(vec, mat, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Matrix Multiply Matrix */
namespace MatrixMultiplyMatrix {

// Divide-and-conquer dot-product on k in [Start, End).
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End>
struct Core<T, M, K, N, I, J, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {
    return Core<T, M, K, N, I, J, Start, Mid>::compute(A, B) +
           Core<T, M, K, N, I, J, Mid, End>::compute(A, B);
  }
};

// Empty range: no contribution.
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End>
struct Core<T, M, K, N, I, J, Start, End,
            typename std::enable_if<(End == Start)>::type> {
  static T compute(const Matrix<T, M, K> &, const Matrix<T, K, N> &) {
    return static_cast<T>(0);
  }
};

// Single term range.
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End>
struct Core<T, M, K, N, I, J, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {
    return A.template get<I, Start>() * B.template get<Start, J>();
  }
};

// Divide-and-conquer over columns j in [Start, End) for a fixed row I.
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, M, K, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Row<T, M, K, N, I, Start, Mid>::compute(A, B, result);
    Row<T, M, K, N, I, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, M, K, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, K> &, const Matrix<T, K, N> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, M, K, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    result.template set<I, Start>(
        Core<T, M, K, N, I, Start, 0, K>::compute(A, B));
  }
};

// Divide-and-conquer over rows i in [Start, End).
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<T, M, K, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, Start, Mid>::compute(A, B, result);
    Column<T, M, K, N, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<T, M, K, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, K> &, const Matrix<T, K, N> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<T, M, K, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Row<T, M, K, N, Start, 0, N>::compute(A, B, result);
  }
};

/**
 * @brief Computes the product of two matrices.
 *
 * This function uses template metaprogramming to recursively compute the
 * product of each row of matrix A with each column of matrix B and store the
 * results in a new matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam K The number of columns in matrix A and rows in matrix B.
 * @tparam N The number of columns in matrix B.
 * @param A The first matrix to multiply.
 * @param B The second matrix to multiply.
 * @param result The resulting matrix where the product is stored.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                    Matrix<T, M, N> &result) {
  Column<T, M, K, N, 0, M>::compute(A, B, result);
}

} // namespace MatrixMultiplyMatrix

/**
 * @brief Multiplies two matrices.
 *
 * This function computes the product of two matrices A and B, which means it
 * performs matrix multiplication, and returns the resulting matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and cols in matrix B.
 * @tparam N The number of rows in matrix B.
 * @param A The first matrix to multiply.
 * @param B The second matrix to multiply.
 * @return Matrix<T, M, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline Matrix<T, M, N> operator*(const Matrix<T, M, K> &A,
                                 const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixMultiplyMatrix::compute<T, M, K, N>(A, B, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Transpose */
namespace MatrixTranspose {

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
    Row<T, M, N, I, Start, Mid>::compute(A, result);
    Row<T, M, N, I, Mid, End>::compute(A, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, Matrix<T, N, M> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
    result.template set<Start, I>(A.template get<I, Start>());
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
    Column<T, M, N, Start, Mid>::compute(A, result);
    Column<T, M, N, Mid, End>::compute(A, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, Matrix<T, N, M> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
    Row<T, M, N, Start, 0, N>::compute(A, result);
  }
};

/**
 * @brief Computes the transpose of a matrix.
 *
 * This function uses template metaprogramming to recursively compute the
 * transpose of a matrix by swapping cols and rows, and store the results in
 * a new matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param A The matrix to transpose.
 * @param result The resulting transposed matrix where the product is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
  Column<T, M, N, 0, M>::compute(A, result);
}

} // namespace MatrixTranspose

/**
 * @brief Transposes a matrix.
 *
 * This function computes the transpose of a matrix A, which means it swaps its
 * cols and rows, and returns the resulting transposed matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix A.
 * @tparam N The number of columns in the matrix A.
 * @param mat The matrix to transpose.
 * @return Matrix<T, N, M> The resulting transposed matrix.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, N, M> output_matrix_transpose(const Matrix<T, M, N> &mat) {
  Matrix<T, N, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(j, i) = mat(i, j);
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixTranspose::compute<T, M, N>(mat, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Upper Triangular Matrix Multiply Matrix */
namespace UpperTriangularMultiplyMatrix {

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End>
struct Core<T, M, K, N, I, J, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {
    return Core<T, M, K, N, I, J, Start, Mid>::compute(A, B) +
           Core<T, M, K, N, I, J, Mid, End>::compute(A, B);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End>
struct Core<T, M, K, N, I, J, Start, End,
            typename std::enable_if<(End == Start)>::type> {
  static T compute(const Matrix<T, M, K> &, const Matrix<T, K, N> &) {
    return static_cast<T>(0);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End>
struct Core<T, M, K, N, I, J, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {
    return (Start >= I)
               ? (A.template get<I, Start>() * B.template get<Start, J>())
               : static_cast<T>(0);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, M, K, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Row<T, M, K, N, I, Start, Mid>::compute(A, B, result);
    Row<T, M, K, N, I, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, M, K, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, K> &, const Matrix<T, K, N> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, M, K, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    result.template set<I, Start>(
        Core<T, M, K, N, I, Start, 0, K>::compute(A, B));
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<T, M, K, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, Start, Mid>::compute(A, B, result);
    Column<T, M, K, N, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<T, M, K, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, K> &, const Matrix<T, K, N> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<T, M, K, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Row<T, M, K, N, Start, 0, N>::compute(A, B, result);
  }
};

/**
 * @brief Computes the product of two matrices for upper triangular matrices.
 *
 * This function uses template metaprogramming to recursively compute the
 * product of each row of matrix A with each column of matrix B, considering
 * that A is an upper triangular matrix, and store the results in a new matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam K The number of columns in matrix A and rows in matrix B.
 * @tparam N The number of columns in matrix B.
 * @param A The first matrix to multiply (upper triangular).
 * @param B The second matrix to multiply.
 * @param result The resulting matrix where the product is stored.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                    Matrix<T, M, N> &result) {
  Column<T, M, K, N, 0, M>::compute(A, B, result);
}

} // namespace UpperTriangularMultiplyMatrix

/**
 * @brief Multiplies an upper triangular matrix A with a matrix B.
 *
 * This function computes the product of an upper triangular matrix A and a
 * matrix B, which means it performs matrix multiplication while considering
 * that A is upper triangular, and returns the resulting matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and cols in matrix B.
 * @tparam N The number of rows in matrix B.
 * @param A The upper triangular matrix to multiply.
 * @param B The matrix to multiply with the upper triangular matrix.
 * @return Matrix<T, M, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline Matrix<T, M, N>
matrix_multiply_Upper_triangular_A_mul_B(const Matrix<T, M, K> &A,
                                         const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = i; k < K; ++k) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  UpperTriangularMultiplyMatrix::compute<T, M, K, N>(A, B, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Matrix Transpose Multiply Matrix */
namespace MatrixTransposeMultiplyMatrix {

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End>
struct Core<T, M, K, N, I, J, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static T compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B) {
    return Core<T, M, K, N, I, J, Start, Mid>::compute(A, B) +
           Core<T, M, K, N, I, J, Mid, End>::compute(A, B);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End>
struct Core<T, M, K, N, I, J, Start, End,
            typename std::enable_if<(End == Start)>::type> {
  static T compute(const Matrix<T, K, M> &, const Matrix<T, K, N> &) {
    return static_cast<T>(0);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End>
struct Core<T, M, K, N, I, J, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static T compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B) {
    return A.template get<Start, I>() * B.template get<Start, J>();
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, M, K, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Row<T, M, K, N, I, Start, Mid>::compute(A, B, result);
    Row<T, M, K, N, I, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, M, K, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, K, M> &, const Matrix<T, K, N> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, M, K, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    result.template set<I, Start>(
        Core<T, M, K, N, I, Start, 0, K>::compute(A, B));
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<T, M, K, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, Start, Mid>::compute(A, B, result);
    Column<T, M, K, N, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<T, M, K, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, K, M> &, const Matrix<T, K, N> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<T, M, K, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Row<T, M, K, N, Start, 0, N>::compute(A, B, result);
  }
};

/**
 * @brief Computes the product of two matrices for matrix transpose
 * multiplication.
 *
 * This function uses template metaprogramming to recursively compute the
 * product of each row of matrix A with each column of matrix B, considering
 * that A is transposed, and store the results in a new matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam K The number of columns in matrix A and rows in matrix B.
 * @tparam N The number of columns in matrix B.
 * @param A The first matrix to multiply (transposed).
 * @param B The second matrix to multiply.
 * @param result The resulting matrix where the product is stored.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                    Matrix<T, M, N> &result) {
  Column<T, M, K, N, 0, M>::compute(A, B, result);
}

} // namespace MatrixTransposeMultiplyMatrix

/**
 * @brief Multiplies a transposed matrix A with a matrix B.
 *
 * This function computes the product of a transposed matrix A and a matrix B,
 * which means it performs matrix multiplication while considering that A is
 * transposed, and returns the resulting matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and cols in matrix B.
 * @tparam N The number of rows in matrix B.
 * @param A The transposed matrix to multiply.
 * @param B The matrix to multiply with the transposed matrix.
 * @return Matrix<T, M, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline Matrix<T, M, N> matrix_multiply_AT_mul_B(const Matrix<T, K, M> &A,
                                                const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(k, i) * B(k, j);
      }
      result(i, j) = sum;
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixTransposeMultiplyMatrix::compute<T, M, K, N>(A, B, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Transpose Matrix multiply Vector  */
namespace MatrixTransposeMultiplyVector {

template <typename T, std::size_t M, std::size_t N, std::size_t N_idx,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t N, std::size_t N_idx,
          std::size_t Start, std::size_t End>
struct Core<T, M, N, N_idx, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec) {
    return Core<T, M, N, N_idx, Start, Mid>::compute(mat, vec) +
           Core<T, M, N, N_idx, Mid, End>::compute(mat, vec);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t N_idx,
          std::size_t Start, std::size_t End>
struct Core<T, M, N, N_idx, Start, End,
            typename std::enable_if<(End == Start)>::type> {
  static T compute(const Matrix<T, M, N> &, const Vector<T, M> &) {
    return static_cast<T>(0);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t N_idx,
          std::size_t Start, std::size_t End>
struct Core<T, M, N, N_idx, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec) {
    return mat.template get<Start, N_idx>() * vec[Start];
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Row<T, M, N, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec,
                      Vector<T, N> &result) {
    Row<T, M, N, Start, Mid>::compute(mat, vec, result);
    Row<T, M, N, Mid, End>::compute(mat, vec, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Row<T, M, N, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, const Vector<T, M> &,
                      Vector<T, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Row<T, M, N, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec,
                      Vector<T, N> &result) {
    result[Start] = Core<T, M, N, Start, 0, M>::compute(mat, vec);
  }
};

/**
 * @brief Computes the product of a transposed matrix and a vector.
 *
 * This function uses template metaprogramming to recursively compute the
 * product of each column of the transposed matrix with the vector, and store
 * the results in a new vector.
 *
 * @tparam T The type of the matrix and vector elements.
 * @tparam M The number of rows of matrix A.
 * @tparam N The number of columns of matrix A.
 * @param mat The transposed matrix to multiply.
 * @param vec The vector to multiply with the transposed matrix.
 * @param result The resulting vector where the product is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec,
                    Vector<T, N> &result) {
  Row<T, M, N, 0, N>::compute(mat, vec, result);
}

} // namespace MatrixTransposeMultiplyVector

/**
 * @brief Multiplies a transposed matrix A with a vector b.
 *
 * This function computes the product of a transposed matrix A and a vector b,
 * which means it performs matrix-vector multiplication while considering that
 * A is transposed, and returns the resulting vector.
 *
 * @tparam T The type of the matrix and vector elements.
 * @tparam M The number of rows in matrix A.
 * @tparam N The number of columns in matrix A.
 * @param A The transposed matrix to multiply.
 * @param b The vector to multiply with the transposed matrix.
 * @return Vector<T, N> The resulting vector after multiplication.
 */
template <typename T, std::size_t M, std::size_t N>
inline Vector<T, N> matrix_multiply_AT_mul_b(const Matrix<T, M, N> &A,
                                             const Vector<T, M> &b) {
  Vector<T, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t n = 0; n < N; ++n) {
    T sum = 0;
    for (std::size_t m = 0; m < M; ++m) {
      sum += A(m, n) * b[m];
    }
    result[n] = sum;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixTransposeMultiplyVector::compute<T, M, N>(A, b, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Matrix multiply Transpose Matrix */
namespace MatrixMultiplyTransposeMatrix {

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End>
struct Core<T, M, K, N, I, J, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B) {
    return Core<T, M, K, N, I, J, Start, Mid>::compute(A, B) +
           Core<T, M, K, N, I, J, Mid, End>::compute(A, B);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End>
struct Core<T, M, K, N, I, J, Start, End,
            typename std::enable_if<(End == Start)>::type> {
  static T compute(const Matrix<T, M, K> &, const Matrix<T, N, K> &) {
    return static_cast<T>(0);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t Start, std::size_t End>
struct Core<T, M, K, N, I, J, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B) {
    return A.template get<I, Start>() * B.template get<J, Start>();
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, M, K, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {
    Row<T, M, K, N, I, Start, Mid>::compute(A, B, result);
    Row<T, M, K, N, I, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, M, K, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, K> &, const Matrix<T, N, K> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, M, K, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {
    result.template set<I, Start>(
        Core<T, M, K, N, I, Start, 0, K>::compute(A, B));
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<T, M, K, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, Start, Mid>::compute(A, B, result);
    Column<T, M, K, N, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<T, M, K, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, K> &, const Matrix<T, N, K> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<T, M, K, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {
    Row<T, M, K, N, Start, 0, N>::compute(A, B, result);
  }
};

/**
 * @brief Computes the product of two matrices for matrix multiplication with a
 * transposed matrix.
 *
 * This function uses template metaprogramming to recursively compute the
 * product of each row of matrix A with each column of matrix B, considering
 * that B is transposed, and store the results in a new matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam K The number of columns in matrix A and cols in matrix B.
 * @tparam N The number of rows in matrix B.
 * @param A The first matrix to multiply.
 * @param B The second matrix to multiply (transposed).
 * @param result The resulting matrix where the product is stored.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                    Matrix<T, M, N> &result) {
  Column<T, M, K, N, 0, M>::compute(A, B, result);
}

} // namespace MatrixMultiplyTransposeMatrix

/**
 * @brief Multiplies a matrix A with the transposed matrix B.
 *
 * This function computes the product of a matrix A and the transposed matrix
 * B, which means it performs matrix multiplication while considering that B is
 * transposed, and returns the resulting matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam K The number of columns in matrix A and cols in matrix B.
 * @tparam N The number of rows in matrix B.
 * @param A The matrix to multiply
 * @param B The transposed matrix to multiply with A.
 * @return Matrix<T, M, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline Matrix<T, M, N>
matrix_multiply_A_mul_BTranspose(const Matrix<T, M, K> &A,
                                 const Matrix<T, N, K> &B) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(i, k) * B(j, k);
      }
      result(i, j) = sum;
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixMultiplyTransposeMatrix::compute<T, M, K, N>(A, B, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Matrix real from complex */
namespace MatrixRealToComplex {

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {
    Row<T, M, N, I, Start, Mid>::compute(From_matrix, To_matrix);
    Row<T, M, N, I, Mid, End>::compute(From_matrix, To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, Matrix<Complex<T>, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {
    To_matrix(I, Start).real = From_matrix.template get<I, Start>();
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {
    Column<T, M, N, Start, Mid>::compute(From_matrix, To_matrix);
    Column<T, M, N, Mid, End>::compute(From_matrix, To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, Matrix<Complex<T>, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {
    Row<T, M, N, Start, 0, N>::compute(From_matrix, To_matrix);
  }
};

/**
 * @brief Computes the conversion from a real matrix to a complex matrix.
 *
 * This function uses template metaprogramming to recursively convert each
 * element of the real matrix to a complex number, where the imaginary part is
 * set to zero, and store the results in a new complex matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param From_matrix The real matrix to convert.
 * @param To_matrix The resulting complex matrix where the conversion is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &From_matrix,
                    Matrix<Complex<T>, M, N> &To_matrix) {
  Column<T, M, N, 0, M>::compute(From_matrix, To_matrix);
}

} // namespace MatrixRealToComplex

/**
 * @brief Converts a real matrix to a complex matrix.
 *
 * This function converts each element of the real matrix to a complex number,
 * where the imaginary part is set to zero, and returns the resulting complex
 * matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param From_matrix The real matrix to convert.
 * @return Matrix<Complex<T>, M, N> The resulting complex matrix after
 * conversion.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<Complex<T>, M, N>
convert_matrix_real_to_complex(const Matrix<T, M, N> &From_matrix) {

  Matrix<Complex<T>, M, N> To_matrix;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j).real = From_matrix(i, j);
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixRealToComplex::compute<T, M, N>(From_matrix, To_matrix);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return To_matrix;
}

/* Matrix real from complex */
namespace MatrixRealFromComplex {

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Row<T, M, N, I, Start, Mid>::compute(From_matrix, To_matrix);
    Row<T, M, N, I, Mid, End>::compute(From_matrix, To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<Complex<T>, M, N> &, Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    To_matrix.template set<I, Start>(From_matrix(I, Start).real);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Column<T, M, N, Start, Mid>::compute(From_matrix, To_matrix);
    Column<T, M, N, Mid, End>::compute(From_matrix, To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<Complex<T>, M, N> &, Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Row<T, M, N, Start, 0, N>::compute(From_matrix, To_matrix);
  }
};

/**
 * @brief Computes the conversion from a complex matrix to a real matrix.
 *
 * This function uses template metaprogramming to recursively convert each
 * element of the complex matrix to its real part and store the results in a new
 * real matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param From_matrix The complex matrix to convert.
 * @param To_matrix The resulting real matrix where the conversion is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                    Matrix<T, M, N> &To_matrix) {
  Column<T, M, N, 0, M>::compute(From_matrix, To_matrix);
}

} // namespace MatrixRealFromComplex

/**
 * @brief Converts a complex matrix to a real matrix.
 *
 * This function extracts the real part of each element in the complex matrix
 * and returns the resulting real matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the complex matrix.
 * @tparam N The number of columns in the complex matrix.
 * @param From_matrix The complex matrix to convert.
 * @return Matrix<T, M, N> The resulting real matrix after conversion.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> get_real_matrix_from_complex_matrix(
    const Matrix<Complex<T>, M, N> &From_matrix) {

  Matrix<T, M, N> To_matrix;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j) = From_matrix(i, j).real;
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixRealFromComplex::compute<T, M, N>(From_matrix, To_matrix);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return To_matrix;
}

/* Matrix imag from complex */
namespace MatrixImagFromComplex {

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Row<T, M, N, I, Start, Mid>::compute(From_matrix, To_matrix);
    Row<T, M, N, I, Mid, End>::compute(From_matrix, To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<Complex<T>, M, N> &, Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    To_matrix.template set<I, Start>(From_matrix(I, Start).imag);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Column<T, M, N, Start, Mid>::compute(From_matrix, To_matrix);
    Column<T, M, N, Mid, End>::compute(From_matrix, To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<Complex<T>, M, N> &, Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Row<T, M, N, Start, 0, N>::compute(From_matrix, To_matrix);
  }
};

/**
 * @brief Computes the conversion from a complex matrix to an imaginary part
 * matrix.
 *
 * This function uses template metaprogramming to recursively extract the
 * imaginary part of each element in the complex matrix and store the results
 * in a new imaginary part matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param From_matrix The complex matrix to convert.
 * @param To_matrix The resulting imaginary part matrix where the conversion is
 * stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                    Matrix<T, M, N> &To_matrix) {
  Column<T, M, N, 0, M>::compute(From_matrix, To_matrix);
}

} // namespace MatrixImagFromComplex

/**
 * @brief Converts a complex matrix to an imaginary part matrix.
 *
 * This function extracts the imaginary part of each element in the complex
 * matrix and returns the resulting imaginary part matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the complex matrix.
 * @tparam N The number of columns in the complex matrix.
 * @param From_matrix The complex matrix to convert.
 * @return Matrix<T, M, N> The resulting imaginary part matrix after conversion.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> get_imag_matrix_from_complex_matrix(
    const Matrix<Complex<T>, M, N> &From_matrix) {

  Matrix<T, M, N> To_matrix;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j) = From_matrix(i, j).imag;
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixImagFromComplex::compute<T, M, N>(From_matrix, To_matrix);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return To_matrix;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_MATRIX_HPP_
