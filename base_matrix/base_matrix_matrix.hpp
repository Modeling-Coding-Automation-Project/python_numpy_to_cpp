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
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __BASE_MATRIX_MATRIX_HPP__
#define __BASE_MATRIX_MATRIX_HPP__

#include "base_matrix_macros.hpp"

#include "base_matrix_complex.hpp"
#include "base_matrix_vector.hpp"
#include "base_utility.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
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
 * controlled by the preprocessor macro __BASE_MATRIX_USE_STD_VECTOR__.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 */
template <typename T, std::size_t M, std::size_t N> class Matrix {
public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = N;

public:
  /* Constructor */
#ifdef __BASE_MATRIX_USE_STD_VECTOR__

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

#else // __BASE_MATRIX_USE_STD_VECTOR__

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

#endif // __BASE_MATRIX_USE_STD_VECTOR__

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

  // P_idx < P
  template <typename U, std::size_t P, std::size_t P_idx>
  struct CreateIdentityCore {
    /**
     * @brief Recursively sets the diagonal element of the identity matrix.
     *
     * This function sets the diagonal element at position (P_idx, P_idx) to 1
     * and recursively calls itself to set the next diagonal element.
     *
     * @param identity The identity matrix being constructed.
     */
    static void compute(Matrix<U, M, M> &identity) {

      identity.template set<P_idx, P_idx>(static_cast<U>(1));
      CreateIdentityCore<U, P, P_idx - 1>::compute(identity);
    }
  };

  // Termination condition: P_idx == 0
  template <typename U, std::size_t P> struct CreateIdentityCore<U, P, 0> {
    /**
     * @brief Sets the first diagonal element of the identity matrix to 1.
     *
     * This function is the base case for the recursive identity matrix
     * construction, setting the element at (0, 0) to 1.
     *
     * @param identity The identity matrix being constructed.
     */
    static void compute(Matrix<U, P, P> &identity) {

      identity.template set<0, 0>(static_cast<U>(1));
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
    CreateIdentityCore<U, P, P - 1>::compute(identity);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    for (std::size_t i = 0; i < M; i++) {
      identity(i, i) = static_cast<T>(1);
    }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    COMPILED_MATRIX_IDENTITY<T, M>(identity);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    return identity;
  }

  /* Full */
  // when J_idx < P
  template <typename U, std::size_t O, std::size_t P, std::size_t I,
            std::size_t J_idx>
  struct MatrixFullColumn {
    /**
     * @brief Recursively sets the elements of a full matrix.
     *
     * This function sets the element at position (I, J_idx) to the specified
     * value and recursively calls itself to set the next element in the column.
     *
     * @param Full The full matrix being constructed.
     * @param value The value to set in the matrix.
     */
    static void compute(Matrix<U, O, P> &Full, const U &value) {

      Full.template set<I, J_idx>(value);
      MatrixFullColumn<U, O, P, I, J_idx - 1>::compute(Full, value);
    }
  };

  // column recursion termination
  template <typename U, std::size_t O, std::size_t P, std::size_t I>
  struct MatrixFullColumn<U, O, P, I, 0> {
    /**
     * @brief Sets the first element of a full matrix column.
     *
     * This function is the base case for the recursive column setting,
     * setting the element at position (I, 0) to the specified value.
     *
     * @param Full The full matrix being constructed.
     * @param value The value to set in the matrix.
     */
    static void compute(Matrix<U, O, P> &Full, const U &value) {

      Full.template set<I, 0>(value);
    }
  };

  // when I_idx < M
  template <typename U, std::size_t O, std::size_t P, std::size_t I_idx>
  struct MatrixFullRow {
    /**
     * @brief Recursively sets the elements of a full matrix row.
     *
     * This function sets the elements of the row at index I_idx to the
     * specified value by calling MatrixFullColumn for each column.
     *
     * @param Full The full matrix being constructed.
     * @param value The value to set in the matrix.
     */
    static void compute(Matrix<U, O, P> &Full, const U &value) {
      MatrixFullColumn<U, O, P, I_idx, P - 1>::compute(Full, value);
      MatrixFullRow<U, O, P, I_idx - 1>::compute(Full, value);
    }
  };

  // row recursion termination
  template <typename U, std::size_t O, std::size_t P>
  struct MatrixFullRow<U, O, P, 0> {
    /**
     * @brief Sets the first row of a full matrix.
     *
     * This function is the base case for the recursive row setting,
     * setting all elements in the first row to the specified value.
     *
     * @param Full The full matrix being constructed.
     * @param value The value to set in the matrix.
     */
    static void compute(Matrix<U, O, P> &Full, const U &value) {
      MatrixFullColumn<U, O, P, 0, P - 1>::compute(Full, value);
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
   * @tparam O The number of columns in the matrix.
   * @tparam P The number of rows in the matrix.
   * @param Full The full matrix to be constructed.
   * @param value The value to set in the matrix.
   */
  template <typename U, std::size_t O, std::size_t P>
  static inline void COMPILED_MATRIX_FULL(Matrix<U, O, P> &Full,
                                          const U &value) {
    MatrixFullRow<U, O, P, O - 1>::compute(Full, value);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        Ones(i, j) = static_cast<T>(1);
      }
    }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    COMPILED_MATRIX_FULL<T, M, N>(Ones, static_cast<T>(1));

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < N; j++) {
        Full(i, j) = value;
      }
    }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    COMPILED_MATRIX_FULL<T, M, N>(Full, value);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

    return Full;
  }

  /**
   * @brief Creates a row vector from a specified row of the matrix.
   *
   * This function extracts a row from the matrix and returns it as a Vector
   * object.
   *
   * @param row The index of the row to extract (0-based).
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
   * @param col The index of the column to extract (0-based).
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
   * @param col The index of the column (0-based).
   * @param row The index of the row (0-based).
   * @return T& Reference to the element at the specified position.
   */
  const T &operator()(std::size_t col, std::size_t row) const {

    return this->data[row][col];
  }

#ifdef __BASE_MATRIX_USE_STD_VECTOR__

  /**
   * @brief Accesses a row of the matrix.
   *
   * This function provides access to a specific row of the matrix, returning a
   * reference to the vector representing that row. If the specified row index
   * is out of bounds, it defaults to the last row.
   *
   * @param row The index of the row to access (0-based).
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
   * @param row The index of the row to access (0-based).
   * @return const std::vector<T>& Const reference to the vector representing
   * the specified row.
   */
  const std::vector<T> &operator()(std::size_t row) const {
    if (row >= N) {
      row = N - 1;
    }

    return this->data[row];
  }

#else // __BASE_MATRIX_USE_STD_VECTOR__

  /**
   * @brief Accesses a row of the matrix.
   *
   * This function provides access to a specific row of the matrix, returning a
   * reference to the array representing that row. If the specified row index is
   * out of bounds, it defaults to the last row.
   *
   * @param row The index of the row to access (0-based).
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
   * @param row The index of the row to access (0-based).
   * @return const std::array<T, M>& Const reference to the array representing
   * the specified row.
   */
  const std::array<T, M> &operator()(std::size_t row) const {
    if (row >= N) {
      row = N - 1;
    }

    return this->data[row];
  }

#endif // __BASE_MATRIX_USE_STD_VECTOR__

  /**
   * @brief Returns the number of rows in the matrix.
   *
   * This function returns the number of rows in the matrix, which is a
   * compile-time constant.
   *
   * @return std::size_t The number of rows in the matrix.
   */
  constexpr std::size_t rows() const { return N; }

  /**
   * @brief Returns the number of columns in the matrix.
   *
   * This function returns the number of columns in the matrix, which is a
   * compile-time constant.
   *
   * @return std::size_t The number of columns in the matrix.
   */
  constexpr std::size_t cols() const { return M; }

  /**
   * @brief Returns the number of elements in the matrix.
   *
   * This function returns the total number of elements in the matrix, which is
   * the product of the number of rows and columns.
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
   * @param row The index of the row to set (0-based).
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
   * the product of the number of rows and columns.
   *
   * @return std::size_t The total number of elements in the matrix.
   */
  inline Matrix<T, M, M> inv() const {
    Matrix<T, M, M> X_temp = Matrix<T, M, M>::identity();
    std::array<T, M> rho;
    std::array<std::size_t, M> rep_num;

    Matrix<T, M, M> Inv =
        gmres_k_matrix_inv(*this, static_cast<T>(0.0), static_cast<T>(1.0e-10),
                           rho, rep_num, X_temp);

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
   * @tparam COL The index of the column (0-based).
   * @tparam ROW The index of the row (0-based).
   * @return T The value at the specified position in the matrix.
   */
  template <std::size_t COL, std::size_t ROW> inline T get() const {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    return data[ROW][COL];
  }

  /* Set Dense Matrix value */

  /**
   * @brief Sets the value at the specified column and row indices.
   *
   * This function assigns a value to the specified column and row indices in
   * the matrix. It performs compile-time checks to ensure that the indices are
   * within bounds.
   *
   * @tparam COL The index of the column (0-based).
   * @tparam ROW The index of the row (0-based).
   * @param value The value to set at the specified position in the matrix.
   */
  template <std::size_t COL, std::size_t ROW> inline void set(const T &value) {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    data[ROW][COL] = value;
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
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<std::vector<T>> data;
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<std::array<T, M>, N> data;
#endif // __BASE_MATRIX_USE_STD_VECTOR__
};

/* swap columns */
namespace MatrixSwapColumns {

// Swap N_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t N_idx>
struct Core {
  /**
   * @brief Recursively swaps two columns in the matrix.
   *
   * This function swaps the elements of the specified columns in the matrix
   * for the current row index N_idx and recursively calls itself for the next
   * row index.
   *
   * @param col_1 The index of the first column to swap.
   * @param col_2 The index of the second column to swap.
   * @param mat The matrix in which the columns are swapped.
   * @param temp A temporary variable to hold values during swapping.
   */
  static void compute(std::size_t col_1, std::size_t col_2,
                      Matrix<T, M, N> &mat, T temp) {

    temp = mat.data[N_idx][col_1];
    mat.data[N_idx][col_1] = mat.data[N_idx][col_2];
    mat.data[N_idx][col_2] = temp;
    Core<T, M, N, N_idx - 1>::compute(col_1, col_2, mat, temp);
  }
};

// Termination condition: N_idx == 0
template <typename T, std::size_t M, std::size_t N> struct Core<T, M, N, 0> {
  /**
   * @brief Swaps two columns in the first row of the matrix.
   *
   * This function swaps the elements of the specified columns in the first row
   * of the matrix.
   *
   * @param col_1 The index of the first column to swap.
   * @param col_2 The index of the second column to swap.
   * @param mat The matrix in which the columns are swapped.
   * @param temp A temporary variable to hold values during swapping.
   */
  static void compute(std::size_t col_1, std::size_t col_2,
                      Matrix<T, M, N> &mat, T temp) {

    temp = mat.data[0][col_1];
    mat.data[0][col_1] = mat.data[0][col_2];
    mat.data[0][col_2] = temp;
  }
};

/**
 * @brief Computes the column swap operation for a matrix.
 *
 * This function uses template metaprogramming to recursively swap two columns
 * in the matrix, starting from the last row and moving upwards.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param col_1 The index of the first column to swap.
 * @param col_2 The index of the second column to swap.
 * @param mat The matrix in which the columns are swapped.
 * @param temp A temporary variable to hold values during swapping.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(std::size_t col_1, std::size_t col_2, Matrix<T, M, N> &mat,
                    T &temp) {
  Core<T, M, N, N - 1>::compute(col_1, col_2, mat, temp);
}

} // namespace MatrixSwapColumns

/**
 * @brief Swaps two columns in a matrix.
 *
 * This function swaps the elements of two specified columns in the matrix.
 * If the column indices are out of bounds, they are adjusted to the last valid
 * index.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param col_1 The index of the first column to swap (0-based).
 * @param col_2 The index of the second column to swap (0-based).
 * @param mat The matrix in which the columns are swapped.
 */
template <typename T, std::size_t M, std::size_t N>
inline void matrix_col_swap(std::size_t col_1, std::size_t col_2,
                            Matrix<T, M, N> &mat) {
  T temp = static_cast<T>(0);

  if (col_1 >= M) {
    col_1 = M - 1;
  }
  if (col_2 >= M) {
    col_2 = M - 1;
  }

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < N; i++) {

    temp = mat.data[i][col_1];
    mat.data[i][col_1] = mat.data[i][col_2];
    mat.data[i][col_2] = temp;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixSwapColumns::compute<T, M, N>(col_1, col_2, mat, temp);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

/* swap rows */

/**
 * @brief Swaps two rows in a matrix.
 *
 * This function swaps the elements of two specified rows in the matrix.
 * If the row indices are out of bounds, they are adjusted to the last valid
 * index.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param row_1 The index of the first row to swap (0-based).
 * @param row_2 The index of the second row to swap (0-based).
 * @param mat The matrix in which the rows are swapped.
 */
template <typename T, std::size_t M, std::size_t N>
inline void matrix_row_swap(std::size_t row_1, std::size_t row_2,
                            Matrix<T, M, N> &mat) {
  Vector<T, M> temp_vec;

  if (row_1 >= N) {
    row_1 = N - 1;
  }
  if (row_2 >= N) {
    row_2 = N - 1;
  }

  Base::Utility::copy<T, 0, M, 0, M, M>(mat(row_1), temp_vec.data);
  Base::Utility::copy<T, 0, M, 0, M, M>(mat(row_2), mat(row_1));
  Base::Utility::copy<T, 0, M, 0, M, M>(temp_vec.data, mat(row_2));
}

/* Trace */
namespace MatrixTrace {

// calculate trace of matrix
template <typename T, std::size_t N, std::size_t I> struct Core {
  /**
   * @brief Recursively computes the trace of a square matrix.
   *
   * This function calculates the trace by summing the diagonal elements
   * of the matrix. It uses template metaprogramming to recursively access the
   * diagonal elements based on the current index I.
   *
   * @tparam T The type of the matrix elements.
   * @tparam N The size of the square matrix (N x N).
   * @tparam I The current index in the recursion (starting from N - 1).
   * @param mat The square matrix from which the trace is computed.
   * @return T The computed trace of the matrix.
   */
  static T compute(const Matrix<T, N, N> &mat) {
    return mat.template get<I, I>() + Core<T, N, I - 1>::compute(mat);
  }
};

// if I == 0
template <typename T, std::size_t N> struct Core<T, N, 0> {
  /**
   * @brief Base case for the trace computation.
   *
   * This function returns the first diagonal element of the matrix when the
   * recursion reaches the base case (I == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam N The size of the square matrix (N x N).
   * @return T The first diagonal element of the matrix.
   */
  static T compute(const Matrix<T, N, N> &mat) {
    return mat.template get<0, 0>();
  }
};

/**
 * @brief Computes the trace of a square matrix.
 *
 * This function uses template metaprogramming to recursively calculate the
 * trace of a square matrix by summing its diagonal elements.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix (must be equal to N).
 * @tparam N The number of rows in the matrix (must be equal to M).
 * @param mat The square matrix from which the trace is computed.
 * @return T The computed trace of the matrix.
 */
template <typename T, std::size_t M, std::size_t N>
inline T compute(const Matrix<T, M, N> &mat) {
  return Core<T, N, N - 1>::compute(mat);
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
 * @tparam M The number of columns in the matrix (must be equal to N).
 * @tparam N The number of rows in the matrix (must be equal to M).
 * @param mat The square matrix from which the trace is computed.
 * @return T The computed trace of the matrix.
 */
template <typename T, std::size_t M, std::size_t N>
inline T output_matrix_trace(const Matrix<T, M, N> &mat) {
  static_assert(M == N, "Matrix must be square matrix");
  T trace = static_cast<T>(0);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; i++) {
    trace += mat(i, i);
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  trace = MatrixTrace::compute<T, M, N>(mat);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return trace;
}

/* Matrix Addition */
namespace MatrixAddMatrix {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  /**
   * @brief Recursively computes the sum of two matrices.
   *
   * This function adds the elements of two matrices at the specified indices
   * and stores the result in the result matrix. It uses template
   * metaprogramming to recursively access the elements based on the current
   * column index J_idx.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the recursion.
   * @tparam J_idx The current column index in the recursion.
   * @param A The first matrix to add.
   * @param B The second matrix to add.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, J_idx>(A.template get<I, J_idx>() +
                                  B.template get<I, J_idx>());
    Column<T, M, N, I, J_idx - 1>::compute(A, B, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  /**
   * @brief Base case for the column addition.
   *
   * This function adds the first element of the specified row in both matrices
   * and stores the result in the result matrix when the recursion reaches the
   * base case (J_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the recursion.
   * @param A The first matrix to add.
   * @param B The second matrix to add.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(A.template get<I, 0>() + B.template get<I, 0>());
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  /**
   * @brief Recursively computes the sum of two matrices row by row.
   *
   * This function adds the elements of two matrices for the current row index
   * I_idx and recursively calls itself for the next row index.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I_idx The current row index in the recursion.
   * @param A The first matrix to add.
   * @param B The second matrix to add.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, I_idx, N - 1>::compute(A, B, result);
    Row<T, M, N, I_idx - 1>::compute(A, B, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  /**
   * @brief Base case for the row addition.
   *
   * This function adds the elements of the first row in both matrices and
   * stores the result in the result matrix when the recursion reaches the base
   * case (I_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param A The first matrix to add.
   * @param B The second matrix to add.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, 0, N - 1>::compute(A, B, result);
  }
};

/**
 * @brief Computes the sum of two matrices.
 *
 * This function uses template metaprogramming to recursively add the elements
 * of two matrices and store the result in a third matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param A The first matrix to add.
 * @param B The second matrix to add.
 * @param result The matrix where the result is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                    Matrix<T, M, N> &result) {
  Row<T, M, N, M - 1>::compute(A, B, result);
}

} // namespace MatrixAddMatrix

/**
 * @brief Adds two matrices of the same size.
 *
 * This function computes the element-wise sum of two matrices A and B, both of
 * size M x N, and returns the resulting matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param A The first matrix to add.
 * @param B The second matrix to add.
 * @return Matrix<T, M, N> The resulting matrix after addition.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator+(const Matrix<T, M, N> &A,
                                 const Matrix<T, M, N> &B) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(i, j) = A(i, j) + B(i, j);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixAddMatrix::compute<T, M, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix Subtraction */
namespace MatrixSubMatrix {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  /**
   * @brief Recursively computes the difference of two matrices.
   *
   * This function subtracts the elements of two matrices at the specified
   * indices and stores the result in the result matrix. It uses template
   * metaprogramming to recursively access the elements based on the current
   * column index J_idx.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the recursion.
   * @tparam J_idx The current column index in the recursion.
   * @param A The first matrix to subtract.
   * @param B The second matrix to subtract.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, J_idx>(A.template get<I, J_idx>() -
                                  B.template get<I, J_idx>());
    Column<T, M, N, I, J_idx - 1>::compute(A, B, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  /**
   * @brief Base case for the column subtraction.
   *
   * This function subtracts the first element of the specified row in both
   * matrices and stores the result in the result matrix when the recursion
   * reaches the base case (J_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the recursion.
   * @param A The first matrix to subtract.
   * @param B The second matrix to subtract.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(A.template get<I, 0>() - B.template get<I, 0>());
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  /**
   * @brief Recursively computes the difference of two matrices row by row.
   *
   * This function subtracts the elements of two matrices for the current row
   * index I_idx and recursively calls itself for the next row index.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I_idx The current row index in the recursion.
   * @param A The first matrix to subtract.
   * @param B The second matrix to subtract.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, I_idx, N - 1>::compute(A, B, result);
    Row<T, M, N, I_idx - 1>::compute(A, B, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  /**
   * @brief Base case for the row subtraction.
   *
   * This function subtracts the elements of the first row in both matrices and
   * stores the result in the result matrix when the recursion reaches the base
   * case (I_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param A The first matrix to subtract.
   * @param B The second matrix to subtract.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, 0, N - 1>::compute(A, B, result);
  }
};

/**
 * @brief Computes the difference of two matrices.
 *
 * This function uses template metaprogramming to recursively subtract the
 * elements of two matrices and store the result in a third matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param A The first matrix to subtract.
 * @param B The second matrix to subtract.
 * @param result The matrix where the result is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &A, const Matrix<T, M, N> &B,
                    Matrix<T, M, N> &result) {
  Row<T, M, N, M - 1>::compute(A, B, result);
}

} // namespace MatrixSubMatrix

/**
 * @brief Subtracts two matrices of the same size.
 *
 * This function computes the element-wise difference of two matrices A and B,
 * both of size M x N, and returns the resulting matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param A The first matrix to subtract.
 * @param B The second matrix to subtract.
 * @return Matrix<T, M, N> The resulting matrix after subtraction.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator-(const Matrix<T, M, N> &A,
                                 const Matrix<T, M, N> &B) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(i, j) = A(i, j) - B(i, j);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixSubMatrix::compute<T, M, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

namespace MatrixMinus {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  /**
   * @brief Recursively computes the negation of a matrix.
   *
   * This function negates the elements of a matrix at the specified indices
   * and stores the result in the result matrix. It uses template
   * metaprogramming to recursively access the elements based on the current
   * column index J_idx.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the recursion.
   * @tparam J_idx The current column index in the recursion.
   * @param A The matrix to negate.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {

    result.template set<I, J_idx>(-A.template get<I, J_idx>());
    Column<T, M, N, I, J_idx - 1>::compute(A, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  /**
   * @brief Base case for the column negation.
   *
   * This function negates the first element of the specified row in the matrix
   * and stores the result in the result matrix when the recursion reaches the
   * base case (J_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the recursion.
   * @param A The matrix to negate.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {

    result.template set<I, 0>(-A.template get<I, 0>());
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  /**
   * @brief Recursively computes the negation of a matrix row by row.
   *
   * This function negates the elements of a matrix for the current row index
   * I_idx and recursively calls itself for the next row index.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I_idx The current row index in the recursion.
   * @param A The matrix to negate.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
    Column<T, M, N, I_idx, N - 1>::compute(A, result);
    Row<T, M, N, I_idx - 1>::compute(A, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  /**
   * @brief Base case for the row negation.
   *
   * This function negates the elements of the first row in the matrix and
   * stores the result in the result matrix when the recursion reaches the base
   * case (I_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param A The matrix to negate.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
    Column<T, M, N, 0, N - 1>::compute(A, result);
  }
};

/**
 * @brief Computes the negation of a matrix.
 *
 * This function uses template metaprogramming to recursively negate the
 * elements of a matrix and store the result in another matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param A The matrix to negate.
 * @param result The matrix where the result is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &A, Matrix<T, M, N> &result) {
  Row<T, M, N, M - 1>::compute(A, result);
}

} // namespace MatrixMinus

/**
 * @brief Negates a matrix.
 *
 * This function computes the negation of a matrix A, which means it multiplies
 * each element of the matrix by -1, and returns the resulting matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param A The matrix to negate.
 * @return Matrix<T, M, N> The resulting negated matrix.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator-(const Matrix<T, M, N> &A) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(i, j) = -A(i, j);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixMinus::compute<T, M, N>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* (Scalar) * (Matrix) */
namespace MatrixMultiplyScalar {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  /**
   * @brief Recursively computes the product of a scalar and a matrix.
   *
   * This function multiplies the elements of a matrix by a scalar at the
   * specified indices and stores the result in the result matrix. It uses
   * template metaprogramming to recursively access the elements based on the
   * current column index J_idx.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the recursion.
   * @tparam J_idx The current column index in the recursion.
   * @param scalar The scalar value to multiply with.
   * @param mat The matrix to multiply with the scalar.
   * @param result The matrix where the result is stored.
   */
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {

    result.template set<I, J_idx>(scalar * mat.template get<I, J_idx>());
    Column<T, M, N, I, J_idx - 1>::compute(scalar, mat, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  /**
   * @brief Base case for the column multiplication.
   *
   * This function multiplies the first element of the specified row in the
   * matrix by the scalar and stores the result in the result matrix when the
   * recursion reaches the base case (J_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the recursion.
   * @param scalar The scalar value to multiply with.
   * @param mat The matrix to multiply with the scalar.
   * @param result The matrix where the result is stored.
   */
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(scalar * mat.template get<I, 0>());
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  /**
   * @brief Recursively computes the product of a scalar and a matrix row by
   * row.
   *
   * This function multiplies the elements of a matrix for the current row index
   * I_idx by a scalar and recursively calls itself for the next row index.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I_idx The current row index in the recursion.
   * @param scalar The scalar value to multiply with.
   * @param mat The matrix to multiply with the scalar.
   * @param result The matrix where the result is stored.
   */
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, I_idx, N - 1>::compute(scalar, mat, result);
    Row<T, M, N, I_idx - 1>::compute(scalar, mat, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  /**
   * @brief Base case for the row multiplication.
   *
   * This function multiplies the elements of the first row in the matrix by
   * the scalar and stores the result in the result matrix when the recursion
   * reaches the base case (I_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param scalar The scalar value to multiply with.
   * @param mat The matrix to multiply with the scalar.
   * @param result The matrix where the result is stored.
   */
  static void compute(const T &scalar, const Matrix<T, M, N> &mat,
                      Matrix<T, M, N> &result) {
    Column<T, M, N, 0, N - 1>::compute(scalar, mat, result);
  }
};

/**
 * @brief Computes the product of a scalar and a matrix.
 *
 * This function uses template metaprogramming to recursively multiply the
 * elements of a matrix by a scalar and store the result in another matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param scalar The scalar value to multiply with.
 * @param mat The matrix to multiply with the scalar.
 * @param result The matrix where the result is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const T &scalar, const Matrix<T, M, N> &mat,
                    Matrix<T, M, N> &result) {
  Row<T, M, N, M - 1>::compute(scalar, mat, result);
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
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param scalar The scalar value to multiply with.
 * @param mat The matrix to multiply with the scalar.
 * @return Matrix<T, M, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator*(const T &scalar, const Matrix<T, M, N> &mat) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = scalar * mat(j, k);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixMultiplyScalar::compute<T, M, N>(scalar, mat, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param mat The matrix to multiply with the scalar.
 * @param scalar The scalar value to multiply with.
 * @return Matrix<T, M, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator*(const Matrix<T, M, N> &mat, const T &scalar) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = scalar * mat(j, k);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixMultiplyScalar::compute<T, M, N>(scalar, mat, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix multiply Vector */
namespace MatrixMultiplyVector {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J>
struct Core {
  /**
   * @brief Recursively computes the dot product of a matrix row and a vector.
   *
   * This function computes the dot product of the I-th row of the matrix with
   * the vector at index J and recursively calls itself for the next column
   * index J - 1.
   *
   * @tparam T The type of the matrix and vector elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the recursion.
   * @tparam J The current column index in the recursion.
   * @param mat The matrix to multiply with the vector.
   * @param vec The vector to multiply with the matrix.
   * @return T The computed dot product for the specified row and column.
   */
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec) {

    return mat.template get<I, J>() * vec[J] +
           Core<T, M, N, I, J - 1>::compute(mat, vec);
  }
};

// if J == 0
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Core<T, M, N, I, 0> {
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec) {
    return mat.template get<I, 0>() * vec[0];
  }
};

// column recursion
template <typename T, std::size_t M, std::size_t N, std::size_t I> struct Row {
  /**
   * @brief Recursively computes the dot product of a matrix row and a vector.
   *
   * This function computes the dot product for the I-th row of the matrix and
   * recursively calls itself for the next row index I - 1.
   *
   * @tparam T The type of the matrix and vector elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the recursion.
   * @param mat The matrix to multiply with the vector.
   * @param vec The vector to multiply with the matrix.
   * @param result The vector where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec,
                      Vector<T, M> &result) {
    result[I] = Core<T, M, N, I, N - 1>::compute(mat, vec);
    Row<T, M, N, I - 1>::compute(mat, vec, result);
  }
};

// if I == 0
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  /**
   * @brief Base case for the row multiplication.
   *
   * This function computes the dot product for the first row of the matrix and
   * stores the result in the result vector when the recursion reaches the base
   * case (I == 0).
   *
   * @tparam T The type of the matrix and vector elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param mat The matrix to multiply with the vector.
   * @param vec The vector to multiply with the matrix.
   * @param result The vector where the result is stored.
   */
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec,
                      Vector<T, M> &result) {
    result[0] = Core<T, M, N, 0, N - 1>::compute(mat, vec);
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
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param mat The matrix to multiply with the vector.
 * @param vec The vector to multiply with the matrix.
 * @param result The vector where the result is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &mat, const Vector<T, N> &vec,
                    Vector<T, M> &result) {
  Row<T, M, N, M - 1>::compute(mat, vec, result);
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
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param mat The matrix to multiply with the vector.
 * @param vec The vector to multiply with the matrix.
 * @return Vector<T, M> The resulting vector after multiplication.
 */
template <typename T, std::size_t M, std::size_t N>
inline Vector<T, M> operator*(const Matrix<T, M, N> &mat,
                              const Vector<T, N> &vec) {
  Vector<T, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    T sum = 0;
    for (std::size_t j = 0; j < N; ++j) {
      sum += mat(i, j) * vec[j];
    }
    result[i] = sum;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixMultiplyVector::compute<T, M, N>(mat, vec, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

namespace VectorMultiplyMatrix {

// calculate if K_idx > 0
template <typename T, std::size_t L, std::size_t N, std::size_t J,
          std::size_t K_idx>
struct Core {
  /**
   * @brief Recursively computes the product of a vector element and a matrix
   * column.
   *
   * This function multiplies the K_idx-th element of the vector with the J-th
   * column of the matrix and recursively calls itself for the next index K_idx
   * - 1.
   *
   * @tparam T The type of the vector and matrix elements.
   * @tparam L The size of the vector.
   * @tparam N The number of rows in the matrix.
   * @tparam J The current column index in the recursion.
   * @tparam K_idx The current index in the vector.
   * @param vec The vector to multiply with the matrix.
   * @param mat The matrix to multiply with the vector.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {

    result.template set<K_idx, J>(vec[K_idx] * mat.template get<0, J>());
    Core<T, L, N, J, K_idx - 1>::compute(vec, mat, result);
  }
};

// if K_idx = 0
template <typename T, std::size_t L, std::size_t N, std::size_t J>
struct Core<T, L, N, J, 0> {
  /**
   * @brief Base case for the vector-matrix multiplication.
   *
   * This function multiplies the first element of the vector with the J-th
   * column of the matrix and stores the result in the result matrix when the
   * recursion reaches the base case (K_idx == 0).
   *
   * @tparam T The type of the vector and matrix elements.
   * @tparam L The size of the vector.
   * @tparam N The number of rows in the matrix.
   * @tparam J The current column index in the recursion.
   * @param vec The vector to multiply with the matrix.
   * @param mat The matrix to multiply with the vector.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {
    result.template set<0, J>(vec[0] * mat.template get<0, J>());
  }
};

// row recursion
template <typename T, std::size_t L, std::size_t N, std::size_t J>
struct Column {
  /**
   * @brief Recursively computes the product of a vector and a matrix column.
   *
   * This function computes the product for the J-th column of the matrix and
   * recursively calls itself for the next column index J - 1.
   *
   * @tparam T The type of the vector and matrix elements.
   * @tparam L The size of the vector.
   * @tparam N The number of rows in the matrix.
   * @tparam J The current column index in the recursion.
   * @param vec The vector to multiply with the matrix.
   * @param mat The matrix to multiply with the vector.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {
    Core<T, L, N, J, L - 1>::compute(vec, mat, result);
    Column<T, L, N, J - 1>::compute(vec, mat, result);
  }
};

// if J = 0
template <typename T, std::size_t L, std::size_t N> struct Column<T, L, N, 0> {
  /**
   * @brief Base case for the column multiplication.
   *
   * This function computes the product for the first column of the matrix and
   * stores the result in the result matrix when the recursion reaches the base
   * case (J == 0).
   *
   * @tparam T The type of the vector and matrix elements.
   * @tparam L The size of the vector.
   * @tparam N The number of rows in the matrix.
   * @param vec The vector to multiply with the matrix.
   * @param mat The matrix to multiply with the vector.
   * @param result The matrix where the result is stored.
   */
  static void compute(const Vector<T, L> &vec, const Matrix<T, 1, N> &mat,
                      Matrix<T, L, N> &result) {
    Core<T, L, N, 0, L - 1>::compute(vec, mat, result);
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
 * @tparam M The number of rows in the matrix (should be 1).
 * @tparam N The number of columns in the matrix.
 * @param vec The vector to multiply with the matrix.
 * @param mat The matrix to multiply with the vector.
 * @param result The matrix where the result is stored.
 */
template <typename T, std::size_t L, std::size_t M, std::size_t N>
inline void compute(const Vector<T, L> &vec, const Matrix<T, M, N> &mat,
                    Matrix<T, L, N> &result) {
  Column<T, L, N, N - 1>::compute(vec, mat, result);
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
 * @tparam M The number of columns in the matrix (should be 1).
 * @tparam N The number of rows in the matrix.
 * @param vec The vector to multiply with the matrix.
 * @param mat The matrix to multiply with the vector.
 * @return Matrix<T, L, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t L, std::size_t M, std::size_t N>
inline Matrix<T, L, N> operator*(const Vector<T, L> &vec,
                                 const Matrix<T, M, N> &mat) {
  static_assert(M == 1, "Invalid size.");
  Matrix<T, L, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t k = 0; k < L; ++k) {
      result(k, j) = vec[k] * mat(0, j);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  VectorMultiplyMatrix::compute<T, L, M, N>(vec, mat, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* (Column Vector) * (Matrix) */
namespace ColumnVectorMultiplyMatrix {

// calculation when I > 0
template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t I>
struct Core {
  /**
   * @brief Recursively computes the product of a column vector element and a
   * matrix row.
   *
   * This function multiplies the I-th element of the column vector with the
   * J-th row of the matrix and recursively calls itself for the next index I
   * - 1.
   *
   * @tparam T The type of the column vector and matrix elements.
   * @tparam M The size of the column vector.
   * @tparam N The number of rows in the matrix.
   * @tparam J The current row index in the recursion.
   * @tparam I The current index in the column vector.
   * @param vec The column vector to multiply with the matrix.
   * @param mat The matrix to multiply with the column vector.
   * @return T The computed product for the specified row and index.
   */
  static T compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat) {
    return vec[I] * mat.template get<I, J>() +
           Core<T, M, N, J, I - 1>::compute(vec, mat);
  }
};

// if I = 0
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct Core<T, M, N, J, 0> {
  /**
   * @brief Base case for the column vector-matrix multiplication.
   *
   * This function multiplies the first element of the column vector with the
   * J-th row of the matrix and returns the result when the recursion reaches
   * the base case (I == 0).
   *
   * @tparam T The type of the column vector and matrix elements.
   * @tparam M The size of the column vector.
   * @tparam N The number of rows in the matrix.
   * @tparam J The current row index in the recursion.
   * @param vec The column vector to multiply with the matrix.
   * @param mat The matrix to multiply with the column vector.
   * @return T The computed product for the specified row and index.
   */
  static T compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat) {
    return vec[0] * mat.template get<0, J>();
  }
};

// row recursion
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct Column {
  /**
   * @brief Recursively computes the product of a column vector and a matrix
   * row by row.
   *
   * This function computes the product for the J-th row of the matrix and
   * recursively calls itself for the next row index J - 1.
   *
   * @tparam T The type of the column vector and matrix elements.
   * @tparam M The size of the column vector.
   * @tparam N The number of rows in the matrix.
   * @tparam J The current row index in the recursion.
   * @param vec The column vector to multiply with the matrix.
   * @param mat The matrix to multiply with the column vector.
   * @param result The column vector where the result is stored.
   */
  static void compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat,
                      ColVector<T, N> &result) {
    result[J] = Core<T, M, N, J, M - 1>::compute(vec, mat);
    Column<T, M, N, J - 1>::compute(vec, mat, result);
  }
};

// if J = 0
template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, 0> {
  /**
   * @brief Base case for the column multiplication.
   *
   * This function computes the product for the first row of the matrix and
   * stores the result in the result column vector when the recursion reaches
   * the base case (J == 0).
   *
   * @tparam T The type of the column vector and matrix elements.
   * @tparam M The size of the column vector.
   * @tparam N The number of rows in the matrix.
   * @param vec The column vector to multiply with the matrix.
   * @param mat The matrix to multiply with the column vector.
   * @param result The column vector where the result is stored.
   */
  static void compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat,
                      ColVector<T, N> &result) {
    result[0] = Core<T, M, N, 0, M - 1>::compute(vec, mat);
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
 * @tparam N The number of rows in the matrix.
 * @param vec The column vector to multiply with the matrix.
 * @param mat The matrix to multiply with the column vector.
 * @param result The column vector where the result is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const ColVector<T, M> &vec, const Matrix<T, M, N> &mat,
                    ColVector<T, N> &result) {
  Column<T, M, N, N - 1>::compute(vec, mat, result);
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
 * @tparam N The number of rows in the matrix.
 * @param vec The column vector to multiply with the matrix.
 * @param mat The matrix to multiply with the column vector.
 * @return ColVector<T, N> The resulting column vector after multiplication.
 */
template <typename T, std::size_t M, std::size_t N>
inline ColVector<T, N> operator*(const ColVector<T, M> &vec,
                                 const Matrix<T, M, N> &mat) {
  ColVector<T, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < N; ++j) {
    T sum = 0;
    for (std::size_t i = 0; i < M; ++i) {
      sum += vec[i] * mat(i, j);
    }
    result[j] = sum;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  ColumnVectorMultiplyMatrix::compute<T, M, N>(vec, mat, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix Multiply Matrix */
namespace MatrixMultiplyMatrix {

// when K_idx < K
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t K_idx>
struct Core {
  /**
   * @brief Recursively computes the product of two matrices.
   *
   * This function computes the dot product of the I-th row of matrix A and the
   * J-th column of matrix B, and recursively calls itself for the next index
   * K_idx - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @tparam J The current column index in matrix B.
   * @tparam K_idx The current index in the multiplication.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @return T The computed product for the specified row and column.
   */
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {

    return A.template get<I, K_idx>() * B.template get<K_idx, J>() +
           Core<T, M, K, N, I, J, K_idx - 1>::compute(A, B);
  }
};

// when K_idx reached 0
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Core<T, M, K, N, I, J, 0> {
  /**
   * @brief Base case for the matrix multiplication.
   *
   * This function computes the product of the I-th row of matrix A and the
   * J-th column of matrix B when the recursion reaches the base case (K_idx ==
   * 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @tparam J The current column index in matrix B.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @return T The computed product for the specified row and column.
   */
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {

    return A.template get<I, 0>() * B.template get<0, J>();
  }
};

// After completing the J column, go to the next row I
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Column {
  /**
   * @brief Recursively computes the product of two matrices column by column.
   *
   * This function computes the product for the J-th column of matrix B and
   * recursively calls itself for the next column index J - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @tparam J The current column index in matrix B.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, J>(Core<T, M, K, N, I, J, K - 1>::compute(A, B));
    Column<T, M, K, N, I, J - 1>::compute(A, B, result);
  }
};

// Row recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Column<T, M, K, N, I, 0> {
  /**
   * @brief Base case for the column multiplication.
   *
   * This function computes the product for the first column of matrix B and
   * stores the result in the result matrix when the recursion reaches the base
   * case (J == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(Core<T, M, K, N, I, 0, K - 1>::compute(A, B));
  }
};

// proceed to the next row after completing the I row
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Row {
  /**
   * @brief Recursively computes the product of two matrices row by row.
   *
   * This function computes the product for the I-th row of matrix A and
   * recursively calls itself for the next row index I - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, I, N - 1>::compute(A, B, result);
    Row<T, M, K, N, I - 1>::compute(A, B, result);
  }
};

// Column recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N>
struct Row<T, M, K, N, 0> {
  /**
   * @brief Base case for the row multiplication.
   *
   * This function computes the product for the first row of matrix A and
   * stores the result in the result matrix when the recursion reaches the base
   * case (I == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, 0, N - 1>::compute(A, B, result);
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
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and columns in matrix B.
 * @tparam N The number of rows in matrix B.
 * @param A The first matrix to multiply.
 * @param B The second matrix to multiply.
 * @param result The resulting matrix where the product is stored.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                    Matrix<T, M, N> &result) {
  Row<T, M, K, N, M - 1>::compute(A, B, result);
}

} // namespace MatrixMultiplyMatrix

/**
 * @brief Multiplies two matrices.
 *
 * This function computes the product of two matrices A and B, which means it
 * performs matrix multiplication, and returns the resulting matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam K The number of columns in matrix A and rows in matrix B.
 * @tparam N The number of columns in matrix B.
 * @param A The first matrix to multiply.
 * @param B The second matrix to multiply.
 * @return Matrix<T, M, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline Matrix<T, M, N> operator*(const Matrix<T, M, K> &A,
                                 const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixMultiplyMatrix::compute<T, M, K, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Transpose */
namespace MatrixTranspose {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  /**
   * @brief Recursively computes the transpose of a matrix column by column.
   *
   * This function computes the J_idx-th column of the transposed matrix
   * and recursively calls itself for the next column index J_idx - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the matrix.
   * @tparam J_idx The current column index in the recursion.
   * @param A The matrix to transpose.
   * @param result The resulting transposed matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {

    result.template set<J_idx, I>(A.template get<I, J_idx>());
    Column<T, M, N, I, J_idx - 1>::compute(A, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  /**
   * @brief Base case for the column transposition.
   *
   * This function computes the first column of the transposed matrix
   * and stores the result in the result matrix when the recursion reaches
   * the base case (J_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the matrix.
   * @param A The matrix to transpose.
   * @param result The resulting transposed matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {

    result.template set<0, I>(A.template get<I, 0>());
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  /**
   * @brief Recursively computes the transpose of a matrix row by row.
   *
   * This function computes the I_idx-th row of the transposed matrix
   * and recursively calls itself for the next row index I_idx - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I_idx The current row index in the recursion.
   * @param A The matrix to transpose.
   * @param result The resulting transposed matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
    Column<T, M, N, I_idx, N - 1>::compute(A, result);
    Row<T, M, N, I_idx - 1>::compute(A, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  /**
   * @brief Base case for the row transposition.
   *
   * This function computes the first row of the transposed matrix
   * and stores the result in the result matrix when the recursion reaches
   * the base case (I_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param A The matrix to transpose.
   * @param result The resulting transposed matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
    Column<T, M, N, 0, N - 1>::compute(A, result);
  }
};

/**
 * @brief Computes the transpose of a matrix.
 *
 * This function uses template metaprogramming to recursively compute the
 * transpose of a matrix by swapping rows and columns, and store the results in
 * a new matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param A The matrix to transpose.
 * @param result The resulting transposed matrix where the product is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &A, Matrix<T, N, M> &result) {
  Row<T, M, N, M - 1>::compute(A, result);
}

} // namespace MatrixTranspose

/**
 * @brief Transposes a matrix.
 *
 * This function computes the transpose of a matrix A, which means it swaps its
 * rows and columns, and returns the resulting transposed matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the matrix A.
 * @tparam N The number of rows in the matrix A.
 * @param mat The matrix to transpose.
 * @return Matrix<T, N, M> The resulting transposed matrix.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, N, M> output_matrix_transpose(const Matrix<T, M, N> &mat) {
  Matrix<T, N, M> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(j, i) = mat(i, j);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixTranspose::compute<T, M, N>(mat, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Upper Triangular Matrix Multiply Matrix */
namespace UpperTriangularMultiplyMatrix {

// when K_idx >= I
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t K_idx>
struct Core {
  /**
   * @brief Recursively computes the product of two matrices for upper
   * triangular matrices.
   *
   * This function computes the dot product of the I-th row of matrix A and the
   * J-th column of matrix B, and recursively calls itself for the next index
   * K_idx - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @tparam J The current column index in matrix B.
   * @tparam K_idx The current index in the multiplication.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @return T The computed product for the specified row and column.
   */
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {

    return (K_idx >= I)
               ? (A.template get<I, K_idx>() * B.template get<K_idx, J>() +
                  Core<T, M, K, N, I, J, K_idx - 1>::compute(A, B))
               : static_cast<T>(0);
  }
};

// recursion termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Core<T, M, K, N, I, J, static_cast<std::size_t>(-1)> {
  /**
   * @brief Base case for the upper triangular matrix multiplication.
   *
   * This function returns 0 when K_idx is less than I, indicating that the
   * multiplication does not contribute to the result in the upper triangular
   * matrix context.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @tparam J The current column index in matrix B.
   * @return T The computed product for the specified row and column, which is
   * 0.
   */
  static T compute(const Matrix<T, M, K> &, const Matrix<T, K, N> &) {

    return static_cast<T>(0);
  }
};

// when K_idx reaches I (base case)
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Core<T, M, K, N, I, J, I> {
  /**
   * @brief Base case for the upper triangular matrix multiplication.
   *
   * This function computes the product of the I-th row of matrix A and the
   * J-th column of matrix B when K_idx reaches I, indicating that the
   * multiplication is valid in the upper triangular matrix context.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @tparam J The current column index in matrix B.
   * @return T The computed product for the specified row and column.
   */
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B) {

    return A.template get<I, I>() * B.template get<I, J>();
  }
};

// Column-wise computation
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Column {
  /**
   * @brief Recursively computes the product of two matrices for upper
   * triangular matrices column by column.
   *
   * This function computes the product for the J-th column of matrix B and
   * recursively calls itself for the next column index J - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @tparam J The current column index in matrix B.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, J>(Core<T, M, K, N, I, J, K - 1>::compute(A, B));
    Column<T, M, K, N, I, J - 1>::compute(A, B, result);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Column<T, M, K, N, I, 0> {
  /**
   * @brief Base case for the column multiplication in upper triangular matrix
   * context.
   *
   * This function computes the product for the first column of matrix B and
   * stores the result in the result matrix when the recursion reaches the base
   * case (J == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(Core<T, M, K, N, I, 0, K - 1>::compute(A, B));
  }
};

// Row-wise computation
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Row {
  /**
   * @brief Recursively computes the product of two matrices for upper
   * triangular matrices row by row.
   *
   * This function computes the product for the I-th row of matrix A and
   * recursively calls itself for the next row index I - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, I, N - 1>::compute(A, B, result);
    Row<T, M, K, N, I - 1>::compute(A, B, result);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t K, std::size_t N>
struct Row<T, M, K, N, 0> {
  /**
   * @brief Base case for the row multiplication in upper triangular matrix
   * context.
   *
   * This function computes the product for the first row of matrix A and
   * stores the result in the result matrix when the recursion reaches the base
   * case (I == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, 0, N - 1>::compute(A, B, result);
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
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and columns in matrix B.
 * @tparam N The number of rows in matrix B.
 * @param A The first matrix to multiply (upper triangular).
 * @param B The second matrix to multiply.
 * @param result The resulting matrix where the product is stored.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline void compute(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
                    Matrix<T, M, N> &result) {
  Row<T, M, K, N, M - 1>::compute(A, B, result);
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
 * @tparam M The number of rows in matrix A.
 * @tparam K The number of columns in matrix A and rows in matrix B.
 * @tparam N The number of columns in matrix B.
 * @param A The upper triangular matrix to multiply.
 * @param B The matrix to multiply with the upper triangular matrix.
 * @return Matrix<T, M, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline Matrix<T, M, N>
matrix_multiply_Upper_triangular_A_mul_B(const Matrix<T, M, K> &A,
                                         const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = i; k < K; ++k) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  UpperTriangularMultiplyMatrix::compute<T, M, K, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix Transpose Multiply Matrix */
namespace MatrixTransposeMultiplyMatrix {

// when K_idx < K
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t K_idx>
struct Core {
  /**
   * @brief Recursively computes the product of two matrices for matrix
   * transpose multiplication.
   *
   * This function computes the dot product of the I-th row of matrix A and the
   * J-th column of matrix B, and recursively calls itself for the next index
   * K_idx - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @tparam J The current column index in matrix B.
   * @tparam K_idx The current index in the multiplication.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @return T The computed product for the specified row and column.
   */
  static T compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B) {

    return A.template get<K_idx, I>() * B.template get<K_idx, J>() +
           Core<T, M, K, N, I, J, K_idx - 1>::compute(A, B);
  }
};

// when K_idx reached 0
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Core<T, M, K, N, I, J, 0> {
  /**
   * @brief Base case for the matrix transpose multiplication.
   *
   * This function computes the product of the first row of matrix A and the
   * first column of matrix B when K_idx reaches 0, indicating that the
   * multiplication is valid in the context of matrix transpose multiplication.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @tparam J The current column index in matrix B.
   * @return T The computed product for the specified row and column.
   */
  static T compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B) {

    return A.template get<0, I>() * B.template get<0, J>();
  }
};

// After completing the J column, go to the next row I
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Column {
  /**
   * @brief Recursively computes the product of two matrices for matrix
   * transpose multiplication column by column.
   *
   * This function computes the product for the J-th column of matrix B and
   * recursively calls itself for the next column index J - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @tparam J The current column index in matrix B.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, J>(Core<T, M, K, N, I, J, K - 1>::compute(A, B));
    Column<T, M, K, N, I, J - 1>::compute(A, B, result);
  }
};

// Row recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Column<T, M, K, N, I, 0> {
  /**
   * @brief Base case for the column multiplication in matrix transpose
   * multiplication context.
   *
   * This function computes the product for the first column of matrix B and
   * stores the result in the result matrix when the recursion reaches the base
   * case (J == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(Core<T, M, K, N, I, 0, K - 1>::compute(A, B));
  }
};

// proceed to the next row after completing the I row
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Row {
  /**
   * @brief Recursively computes the product of two matrices for matrix
   * transpose multiplication row by row.
   *
   * This function computes the product for the I-th row of matrix A and
   * recursively calls itself for the next row index I - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @tparam I The current row index in matrix A.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, I, N - 1>::compute(A, B, result);
    Row<T, M, K, N, I - 1>::compute(A, B, result);
  }
};

// Column recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N>
struct Row<T, M, K, N, 0> {
  /**
   * @brief Base case for the row multiplication in matrix transpose
   * multiplication context.
   *
   * This function computes the product for the first row of matrix A and
   * stores the result in the result matrix when the recursion reaches the base
   * case (I == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and columns in matrix B.
   * @tparam N The number of rows in matrix B.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply.
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, 0, N - 1>::compute(A, B, result);
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
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and columns in matrix B.
 * @tparam N The number of rows in matrix B.
 * @param A The first matrix to multiply (transposed).
 * @param B The second matrix to multiply.
 * @param result The resulting matrix where the product is stored.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline void compute(const Matrix<T, K, M> &A, const Matrix<T, K, N> &B,
                    Matrix<T, M, N> &result) {
  Row<T, M, K, N, M - 1>::compute(A, B, result);
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
 * @tparam M The number of rows in matrix A.
 * @tparam K The number of columns in matrix A and rows in matrix B.
 * @tparam N The number of columns in matrix B.
 * @param A The transposed matrix to multiply.
 * @param B The matrix to multiply with the transposed matrix.
 * @return Matrix<T, M, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline Matrix<T, M, N> matrix_multiply_AT_mul_B(const Matrix<T, K, M> &A,
                                                const Matrix<T, K, N> &B) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(k, i) * B(k, j);
      }
      result(i, j) = sum;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixTransposeMultiplyMatrix::compute<T, M, K, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Transpose Matrix multiply Vector  */
namespace MatrixTransposeMultiplyVector {

// when M_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t N_idx,
          std::size_t M_idx>
struct Core {
  /**
   * @brief Recursively computes the product of a transposed matrix and a
   * vector.
   *
   * This function computes the dot product of the M_idx-th column of the
   * transposed matrix and the vector, and recursively calls itself for the
   * next index M_idx - 1.
   *
   * @tparam T The type of the matrix and vector elements.
   * @tparam M The number of columns of matrix A.
   * @tparam N The number of rows of matrix A.
   * @tparam N_idx The current column index in the transposed matrix.
   * @tparam M_idx The current row index in the transposed matrix.
   * @param mat The transposed matrix to multiply.
   * @param vec The vector to multiply with the transposed matrix.
   * @return T The computed product for the specified column and vector.
   */
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec) {

    return mat.template get<M_idx, N_idx>() * vec[M_idx] +
           Core<T, M, N, N_idx, M_idx - 1>::compute(mat, vec);
  }
};

// if M_idx == 0
template <typename T, std::size_t M, std::size_t N, std::size_t N_idx>
struct Core<T, M, N, N_idx, 0> {
  /**
   * @brief Base case for the product of a transposed matrix and a vector.
   *
   * This function computes the product of the first column of the transposed
   * matrix and the vector when M_idx reaches 0, indicating that the
   * multiplication is valid in the context of matrix transpose multiplication.
   *
   * @tparam T The type of the matrix and vector elements.
   * @tparam M The number of columns of matrix A.
   * @tparam N The number of rows of matrix A.
   * @tparam N_idx The current column index in the transposed matrix.
   * @return T The computed product for the specified column and vector.
   */
  static T compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec) {
    return mat.template get<0, N_idx>() * vec[0];
  }
};

// column recursion
template <typename T, std::size_t M, std::size_t N, std::size_t N_idx>
struct Column {
  /**
   * @brief Recursively computes the product of a transposed matrix and a vector
   * column by column.
   *
   * This function computes the product for the N_idx-th column of the
   * transposed matrix and recursively calls itself for the next column index
   * N_idx - 1.
   *
   * @tparam T The type of the matrix and vector elements.
   * @tparam M The number of columns of matrix A.
   * @tparam N The number of rows of matrix A.
   * @tparam N_idx The current column index in the transposed matrix.
   * @param mat The transposed matrix to multiply.
   * @param vec The vector to multiply with the transposed matrix.
   * @param result The resulting vector where the product is stored.
   */
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec,
                      Vector<T, N> &result) {
    result[N_idx] = Core<T, M, N, N_idx, M - 1>::compute(mat, vec);
    Column<T, M, N, N_idx - 1>::compute(mat, vec, result);
  }
};

// if N_idx == 0
template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, 0> {
  /**
   * @brief Base case for the column multiplication in matrix transpose
   * multiplication context.
   *
   * This function computes the product for the first column of the transposed
   * matrix and the vector, and stores the result in the result vector when the
   * recursion reaches the base case (N_idx == 0).
   *
   * @tparam T The type of the matrix and vector elements.
   * @tparam M The number of columns of matrix A.
   * @tparam N The number of rows of matrix A.
   * @param mat The transposed matrix to multiply.
   * @param vec The vector to multiply with the transposed matrix.
   * @param result The resulting vector where the product is stored.
   */
  static void compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec,
                      Vector<T, N> &result) {
    result[0] = Core<T, M, N, 0, M - 1>::compute(mat, vec);
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
 * @tparam M The number of columns of matrix A.
 * @tparam N The number of rows of matrix A.
 * @param mat The transposed matrix to multiply.
 * @param vec The vector to multiply with the transposed matrix.
 * @param result The resulting vector where the product is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &mat, const Vector<T, M> &vec,
                    Vector<T, N> &result) {
  Column<T, M, N, N - 1>::compute(mat, vec, result);
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
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrix A.
 * @param A The transposed matrix to multiply.
 * @param b The vector to multiply with the transposed matrix.
 * @return Vector<T, N> The resulting vector after multiplication.
 */
template <typename T, std::size_t M, std::size_t N>
inline Vector<T, N> matrix_multiply_AT_mul_b(const Matrix<T, M, N> &A,
                                             const Vector<T, M> &b) {
  Vector<T, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t n = 0; n < N; ++n) {
    T sum = 0;
    for (std::size_t m = 0; m < M; ++m) {
      sum += A(m, n) * b[m];
    }
    result[n] = sum;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixTransposeMultiplyVector::compute<T, M, N>(A, b, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix multiply Transpose Matrix */
namespace MatrixMultiplyTransposeMatrix {

// when K_idx < K
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J, std::size_t K_idx>
struct Core {
  /**
   * @brief Recursively computes the product of two matrices for matrix
   * multiplication with a transposed matrix.
   *
   * This function computes the dot product of the I-th row of matrix A and the
   * J-th column of matrix B, and recursively calls itself for the next index
   * K_idx - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and rows in matrix B.
   * @tparam N The number of columns in matrix B.
   * @tparam I The current row index in matrix A.
   * @tparam J The current column index in matrix B.
   * @tparam K_idx The current index in the multiplication.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply (transposed).
   * @return T The computed product for the specified row and column.
   */
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B) {

    return A.template get<I, K_idx>() * B.template get<J, K_idx>() +
           Core<T, M, K, N, I, J, K_idx - 1>::compute(A, B);
  }
};

// when K_idx reached 0
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Core<T, M, K, N, I, J, 0> {
  /**
   * @brief Base case for the matrix multiplication with a transposed matrix.
   *
   * This function computes the product of the first row of matrix A and the
   * first column of matrix B when K_idx reaches 0, indicating that the
   * multiplication is valid in the context of matrix multiplication with a
   * transposed matrix.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and rows in matrix B.
   * @tparam N The number of columns in matrix B.
   * @tparam I The current row index in matrix A.
   * @tparam J The current column index in matrix B.
   * @return T The computed product for the specified row and column.
   */
  static T compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B) {

    return A.template get<I, 0>() * B.template get<J, 0>();
  }
};

// After completing the J column, go to the next row I
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I, std::size_t J>
struct Column {
  /**
   * @brief Recursively computes the product of two matrices for matrix
   * multiplication with a transposed matrix column by column.
   *
   * This function computes the product for the J-th column of matrix B and
   * recursively calls itself for the next column index J - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and rows in matrix B.
   * @tparam N The number of columns in matrix B.
   * @tparam I The current row index in matrix A.
   * @tparam J The current column index in matrix B.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply (transposed).
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, J>(Core<T, M, K, N, I, J, K - 1>::compute(A, B));
    Column<T, M, K, N, I, J - 1>::compute(A, B, result);
  }
};

// Row recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Column<T, M, K, N, I, 0> {
  /**
   * @brief Base case for the column multiplication in matrix multiplication
   * with a transposed matrix context.
   *
   * This function computes the product for the first column of matrix B and
   * stores the result in the result matrix when the recursion reaches the base
   * case (J == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and rows in matrix B.
   * @tparam N The number of columns in matrix B.
   * @tparam I The current row index in matrix A.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply (transposed).
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {

    result.template set<I, 0>(Core<T, M, K, N, I, 0, K - 1>::compute(A, B));
  }
};

// proceed to the next row after completing the I row
template <typename T, std::size_t M, std::size_t K, std::size_t N,
          std::size_t I>
struct Row {
  /**
   * @brief Recursively computes the product of two matrices for matrix
   * multiplication with a transposed matrix row by row.
   *
   * This function computes the product for the I-th row of matrix A and
   * recursively calls itself for the next row index I - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and rows in matrix B.
   * @tparam N The number of columns in matrix B.
   * @tparam I The current row index in matrix A.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply (transposed).
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, I, N - 1>::compute(A, B, result);
    Row<T, M, K, N, I - 1>::compute(A, B, result);
  }
};

// Column recursive termination
template <typename T, std::size_t M, std::size_t K, std::size_t N>
struct Row<T, M, K, N, 0> {
  /**
   * @brief Base case for the row multiplication in matrix multiplication with a
   * transposed matrix context.
   *
   * This function computes the product for the first row of matrix A and
   * stores the result in the result matrix when the recursion reaches the base
   * case (I == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in matrix A.
   * @tparam K The number of rows in matrix A and rows in matrix B.
   * @tparam N The number of columns in matrix B.
   * @param A The first matrix to multiply.
   * @param B The second matrix to multiply (transposed).
   * @param result The resulting matrix where the product is stored.
   */
  static void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                      Matrix<T, M, N> &result) {
    Column<T, M, K, N, 0, N - 1>::compute(A, B, result);
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
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and rows in matrix B.
 * @tparam N The number of columns in matrix B.
 * @param A The first matrix to multiply.
 * @param B The second matrix to multiply (transposed).
 * @param result The resulting matrix where the product is stored.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline void compute(const Matrix<T, M, K> &A, const Matrix<T, N, K> &B,
                    Matrix<T, M, N> &result) {
  Row<T, M, K, N, M - 1>::compute(A, B, result);
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
 * @tparam M The number of columns in matrix A.
 * @tparam K The number of rows in matrix A and rows in matrix B.
 * @tparam N The number of columns in matrix B.
 * @param A The matrix to multiply
 * @param B The transposed matrix to multiply with A.
 * @return Matrix<T, M, N> The resulting matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t K, std::size_t N>
inline Matrix<T, M, N>
matrix_multiply_A_mul_BTranspose(const Matrix<T, M, K> &A,
                                 const Matrix<T, N, K> &B) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = 0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += A(i, k) * B(j, k);
      }
      result(i, j) = sum;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixMultiplyTransposeMatrix::compute<T, M, K, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix real from complex */
namespace MatrixRealToComplex {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  /**
   * @brief Recursively computes the conversion from a real matrix to a complex
   * matrix column by column.
   *
   * This function computes the conversion for the J_idx-th column of the real
   * matrix and recursively calls itself for the next column index J_idx - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the matrix.
   * @tparam J_idx The current column index in the matrix.
   * @param From_matrix The real matrix to convert.
   * @param To_matrix The resulting complex matrix where the conversion is
   * stored.
   */
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {

    To_matrix(I, J_idx).real = From_matrix.template get<I, J_idx>();
    Column<T, M, N, I, J_idx - 1>::compute(From_matrix, To_matrix);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  /**
   * @brief Base case for the column conversion in matrix real to complex
   * context.
   *
   * This function computes the conversion for the first column of the real
   * matrix and stores the result in the complex matrix when the recursion
   * reaches the base case (J_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the matrix.
   * @param From_matrix The real matrix to convert.
   * @param To_matrix The resulting complex matrix where the conversion is
   * stored.
   */
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {

    To_matrix(I, 0).real = From_matrix.template get<I, 0>();
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  /**
   * @brief Recursively computes the conversion from a real matrix to a complex
   * matrix row by row.
   *
   * This function computes the conversion for the I_idx-th row of the real
   * matrix and recursively calls itself for the next row index I_idx - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I_idx The current row index in the matrix.
   * @param From_matrix The real matrix to convert.
   * @param To_matrix The resulting complex matrix where the conversion is
   * stored.
   */
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {
    Column<T, M, N, I_idx, N - 1>::compute(From_matrix, To_matrix);
    Row<T, M, N, I_idx - 1>::compute(From_matrix, To_matrix);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  /**
   * @brief Base case for the row conversion in matrix real to complex context.
   *
   * This function computes the conversion for the first row of the real matrix
   * and stores the result in the complex matrix when the recursion reaches the
   * base case (I_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param From_matrix The real matrix to convert.
   * @param To_matrix The resulting complex matrix where the conversion is
   * stored.
   */
  static void compute(const Matrix<T, M, N> &From_matrix,
                      Matrix<Complex<T>, M, N> &To_matrix) {
    Column<T, M, N, 0, N - 1>::compute(From_matrix, To_matrix);
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
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param From_matrix The real matrix to convert.
 * @param To_matrix The resulting complex matrix where the conversion is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<T, M, N> &From_matrix,
                    Matrix<Complex<T>, M, N> &To_matrix) {
  Row<T, M, N, M - 1>::compute(From_matrix, To_matrix);
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
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param From_matrix The real matrix to convert.
 * @return Matrix<Complex<T>, M, N> The resulting complex matrix after
 * conversion.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<Complex<T>, M, N>
convert_matrix_real_to_complex(const Matrix<T, M, N> &From_matrix) {

  Matrix<Complex<T>, M, N> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j).real = From_matrix(i, j);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixRealToComplex::compute<T, M, N>(From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Matrix real from complex */
namespace MatrixRealFromComplex {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  /**
   * @brief Recursively computes the conversion from a complex matrix to a real
   * matrix column by column.
   *
   * This function computes the conversion for the J_idx-th column of the
   * complex matrix and recursively calls itself for the next column index J_idx
   * - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the matrix.
   * @tparam J_idx The current column index in the matrix.
   * @param From_matrix The complex matrix to convert.
   * @param To_matrix The resulting real matrix where the conversion is stored.
   */
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {

    To_matrix.template set<I, J_idx>(From_matrix(I, J_idx).real);
    Column<T, M, N, I, J_idx - 1>::compute(From_matrix, To_matrix);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  /**
   * @brief Base case for the column conversion in matrix complex to real
   * context.
   *
   * This function computes the conversion for the first column of the complex
   * matrix and stores the result in the real matrix when the recursion reaches
   * the base case (J_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the matrix.
   * @param From_matrix The complex matrix to convert.
   * @param To_matrix The resulting real matrix where the conversion is stored.
   */
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {

    To_matrix.template set<I, 0>(From_matrix(I, 0).real);
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  /**
   * @brief Recursively computes the conversion from a complex matrix to a real
   * matrix row by row.
   *
   * This function computes the conversion for the I_idx-th row of the complex
   * matrix and recursively calls itself for the next row index I_idx - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I_idx The current row index in the matrix.
   * @param From_matrix The complex matrix to convert.
   * @param To_matrix The resulting real matrix where the conversion is stored.
   */
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Column<T, M, N, I_idx, N - 1>::compute(From_matrix, To_matrix);
    Row<T, M, N, I_idx - 1>::compute(From_matrix, To_matrix);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  /**
   * @brief Base case for the row conversion in matrix complex to real context.
   *
   * This function computes the conversion for the first row of the complex
   * matrix and stores the result in the real matrix when the recursion reaches
   * the base case (I_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param From_matrix The complex matrix to convert.
   * @param To_matrix The resulting real matrix where the conversion is stored.
   */
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Column<T, M, N, 0, N - 1>::compute(From_matrix, To_matrix);
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
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param From_matrix The complex matrix to convert.
 * @param To_matrix The resulting real matrix where the conversion is stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                    Matrix<T, M, N> &To_matrix) {
  Row<T, M, N, M - 1>::compute(From_matrix, To_matrix);
}

} // namespace MatrixRealFromComplex

/**
 * @brief Converts a complex matrix to a real matrix.
 *
 * This function extracts the real part of each element in the complex matrix
 * and returns the resulting real matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the complex matrix.
 * @tparam N The number of rows in the complex matrix.
 * @param From_matrix The complex matrix to convert.
 * @return Matrix<T, M, N> The resulting real matrix after conversion.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> get_real_matrix_from_complex_matrix(
    const Matrix<Complex<T>, M, N> &From_matrix) {

  Matrix<T, M, N> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j) = From_matrix(i, j).real;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixRealFromComplex::compute<T, M, N>(From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Matrix imag from complex */
namespace MatrixImagFromComplex {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct Column {
  /**
   * @brief Recursively computes the conversion from a complex matrix to an
   * imaginary part matrix column by column.
   *
   * This function computes the conversion for the J_idx-th column of the
   * complex matrix and recursively calls itself for the next column index J_idx
   * - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the matrix.
   * @tparam J_idx The current column index in the matrix.
   * @param From_matrix The complex matrix to convert.
   * @param To_matrix The resulting imaginary part matrix where the conversion
   * is stored.
   */
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {

    To_matrix.template set<I, J_idx>(From_matrix(I, J_idx).imag);
    Column<T, M, N, I, J_idx - 1>::compute(From_matrix, To_matrix);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct Column<T, M, N, I, 0> {
  /**
   * @brief Base case for the column conversion in matrix complex to imaginary
   * part context.
   *
   * This function computes the conversion for the first column of the complex
   * matrix and stores the result in the imaginary part matrix when the
   * recursion reaches the base case (J_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I The current row index in the matrix.
   * @param From_matrix The complex matrix to convert.
   * @param To_matrix The resulting imaginary part matrix where the conversion
   * is stored.
   */
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {

    To_matrix.template set<I, 0>(From_matrix(I, 0).imag);
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, std::size_t I_idx>
struct Row {
  /**
   * @brief Recursively computes the conversion from a complex matrix to an
   * imaginary part matrix row by row.
   *
   * This function computes the conversion for the I_idx-th row of the complex
   * matrix and recursively calls itself for the next row index I_idx - 1.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @tparam I_idx The current row index in the matrix.
   * @param From_matrix The complex matrix to convert.
   * @param To_matrix The resulting imaginary part matrix where the conversion
   * is stored.
   */
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Column<T, M, N, I_idx, N - 1>::compute(From_matrix, To_matrix);
    Row<T, M, N, I_idx - 1>::compute(From_matrix, To_matrix);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  /**
   * @brief Base case for the row conversion in matrix complex to imaginary part
   * context.
   *
   * This function computes the conversion for the first row of the complex
   * matrix and stores the result in the imaginary part matrix when the
   * recursion reaches the base case (I_idx == 0).
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The number of columns in the matrix.
   * @tparam N The number of rows in the matrix.
   * @param From_matrix The complex matrix to convert.
   * @param To_matrix The resulting imaginary part matrix where the conversion
   * is stored.
   */
  static void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                      Matrix<T, M, N> &To_matrix) {
    Column<T, M, N, 0, N - 1>::compute(From_matrix, To_matrix);
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
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @param From_matrix The complex matrix to convert.
 * @param To_matrix The resulting imaginary part matrix where the conversion is
 * stored.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const Matrix<Complex<T>, M, N> &From_matrix,
                    Matrix<T, M, N> &To_matrix) {
  Row<T, M, N, M - 1>::compute(From_matrix, To_matrix);
}

} // namespace MatrixImagFromComplex

/**
 * @brief Converts a complex matrix to an imaginary part matrix.
 *
 * This function extracts the imaginary part of each element in the complex
 * matrix and returns the resulting imaginary part matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of columns in the complex matrix.
 * @tparam N The number of rows in the complex matrix.
 * @param From_matrix The complex matrix to convert.
 * @return Matrix<T, M, N> The resulting imaginary part matrix after conversion.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> get_imag_matrix_from_complex_matrix(
    const Matrix<Complex<T>, M, N> &From_matrix) {

  Matrix<T, M, N> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      To_matrix(i, j) = From_matrix(i, j).imag;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixImagFromComplex::compute<T, M, N>(From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_MATRIX_HPP__
