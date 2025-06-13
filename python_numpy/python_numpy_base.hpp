/**
 * @file python_numpy_base.hpp
 * @brief Core matrix class templates and operations for PythonNumpy C++
 * library.
 *
 * This file defines the main matrix class templates and arithmetic operations
 * for the PythonNumpy namespace, providing dense, diagonal, and sparse matrix
 * types with static shape and type information. The classes are designed to
 * mimic the behavior and flexibility of NumPy matrices in C++, supporting a
 * wide range of matrix operations (addition, subtraction, multiplication,
 * transpose, etc.) and conversions between matrix types.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __PYTHON_NUMPY_BASE_HPP__
#define __PYTHON_NUMPY_BASE_HPP__

#include "base_matrix.hpp"
#include "python_numpy_complex.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>
#include <initializer_list>
#include <utility>

namespace PythonNumpy {

/* Matrix class definition */
class DefDense {};

class DefDiag {};

class DefSparse {};

template <typename C, typename T, std::size_t M, std::size_t N = 1,
          typename SparseAvailable = void>
class Matrix;

/**
 * @brief Dense matrix class template specialization for DefDense storage.
 *
 * This class represents a fixed-size dense matrix with elements of type T,
 * with dimensions M (columns) x N (rows). It provides constructors for various
 * initialization methods, copy/move semantics, element access, and common
 * matrix operations such as transpose and conversion to complex types.
 *
 * @tparam T   Element type of the matrix (e.g., float, double, std::complex).
 * @tparam M   Number of columns in the matrix.
 * @tparam N   Number of rows in the matrix.
 *
 * @note
 * - The underlying storage is provided by Base::Matrix::Matrix<T, M, N>.
 * - Provides compile-time dimension checks for element access.
 * - Supports creation of zero, one, and full-valued matrices.
 * - Supports conversion to complex, and extraction of real/imaginary parts.
 */
template <typename T, std::size_t M, std::size_t N>
class Matrix<DefDense, T, M, N> {
public:
  /* Type */
  using Value_Type = typename UnderlyingType<T>::Type;
  using Value_Complex_Type = T;
  using Matrix_Type = DefDense;
  using SparseAvailable_Type = DenseAvailable<M, N>;

public:
  /* Constructor */
  Matrix() {}

  Matrix(const std::initializer_list<std::initializer_list<T>> &input)
      : matrix(input) {}

  Matrix(T input[][N]) : matrix(input) {}

  Matrix(Base::Matrix::Matrix<T, M, N> &input) : matrix(input) {}

  Matrix(Base::Matrix::Matrix<T, M, N> &&input) noexcept
      : matrix(std::move(input)) {}

  /* Copy Constructor */
  Matrix(const Matrix<DefDense, T, M, N> &input) : matrix(input.matrix) {}

  Matrix<DefDense, T, M, N> &operator=(const Matrix<DefDense, T, M, N> &input) {
    if (this != &input) {
      this->matrix = input.matrix;
    }
    return *this;
  }

  /* Move Constructor */
  Matrix(Matrix<DefDense, T, M, N> &&input) noexcept
      : matrix(std::move(input.matrix)) {}

  Matrix<DefDense, T, M, N> &
  operator=(Matrix<DefDense, T, M, N> &&input) noexcept {
    if (this != &input) {
      this->matrix = std::move(input.matrix);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Retrieves the element at the specified column and row indices from
   * the matrix.
   *
   * @tparam COL The zero-based column index (must be less than M).
   * @tparam ROW The zero-based row index (must be less than N).
   * @return T The value at the specified column and row.
   *
   * @note Compile-time assertions ensure that the provided indices are within
   * valid bounds.
   */
  template <std::size_t COL, std::size_t ROW> inline T get() const {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    return matrix.template get<COL, ROW>();
  }

  /**
   * @brief Sets the element at the specified column and row indices to the
   * provided value.
   *
   * @tparam COL The zero-based column index (must be less than M).
   * @tparam ROW The zero-based row index (must be less than N).
   * @param value The value to set at the specified column and row.
   *
   * @note Compile-time assertions ensure that the provided indices are within
   * valid bounds.
   */
  template <std::size_t COL, std::size_t ROW> inline void set(const T &value) {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    matrix.template set<COL, ROW>(value);
  }

  /**
   * @brief Retrieves the underlying matrix data.
   *
   * @return Base::Matrix::Matrix<T, M, N> The underlying matrix data.
   */
  constexpr std::size_t rows() const { return ROWS; }

  /**
   * @brief Retrieves the number of columns in the matrix.
   *
   * @return std::size_t The number of columns in the matrix.
   */
  constexpr std::size_t cols() const { return COLS; }

  /**
   * @brief Retrieves the number of elements in the matrix.
   *
   * @return std::size_t The total number of elements in the matrix (M * N).
   */
  T &operator()(std::size_t col, std::size_t row) {
    if (col >= M) {
      col = M - 1;
    }
    if (row >= N) {
      row = N - 1;
    }

    return this->matrix(col, row);
  }

  /**
   * @brief Retrieves the element at the specified column and row indices.
   *
   * @param col The zero-based column index (must be less than M).
   * @param row The zero-based row index (must be less than N).
   * @return const T& The value at the specified column and row.
   *
   * @note If the indices are out of bounds, they are clamped to the maximum
   * valid index.
   */
  const T &operator()(std::size_t col, std::size_t row) const {
    if (col >= M) {
      col = M - 1;
    }
    if (row >= N) {
      row = N - 1;
    }

    return this->matrix(col, row);
  }

  /**
   * @brief Accesses the element at the specified column and row indices.
   *
   * @param col The zero-based column index (must be less than M).
   * @param row The zero-based row index (must be less than N).
   * @return T& A reference to the value at the specified column and row.
   *
   * @note This method is fast but may cause segmentation faults if indices are
   * out of bounds.
   */
  inline T &access(const std::size_t &col, const std::size_t &row) {

    return this->matrix(row)[col];
  }

  /**
   * @brief Accesses the element at the specified column and row indices.
   *
   * @param col The zero-based column index (must be less than M).
   * @param row The zero-based row index (must be less than N).
   * @return const T& A constant reference to the value at the specified column
   * and row.
   *
   * @note This method is fast but may cause segmentation faults if indices are
   * out of bounds.
   */
  static inline auto zeros(void) -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>();
  }

  /**
   * @brief Creates and returns a dense matrix of ones.
   *
   * This static method constructs a dense matrix of size M x N,
   * where all elements are set to 1 (of type T).
   *
   * @return Matrix<DefDense, T, M, N> A dense matrix of ones with dimensions M
   * x N.
   */
  static inline auto ones(void) -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>(Base::Matrix::Matrix<T, M, N>::ones());
  }

  /**
   * @brief Creates and returns a dense matrix filled with a specified value.
   *
   * This static method constructs a dense matrix of size M x N,
   * where all elements are set to the specified value.
   *
   * @param value The value to fill the matrix with.
   * @return Matrix<DefDense, T, M, N> A dense matrix filled with the specified
   * value, with dimensions M x N.
   */
  static inline auto full(const T &value) -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>(
        Base::Matrix::Matrix<T, M, N>::full(value));
  }

  /**
   * @brief Transposes the matrix.
   *
   * This method returns a new matrix that is the transpose of the current
   * matrix, swapping rows and columns.
   *
   * @return Matrix<DefDense, T, N, M> A new matrix that is the transpose of the
   * current matrix.
   */
  inline auto transpose(void) const -> Matrix<DefDense, T, N, M> {
    return Matrix<DefDense, T, N, M>(
        Base::Matrix::output_matrix_transpose(this->matrix));
  }

  /**
   * @brief Creates a complex matrix from the current matrix.
   *
   * This method converts the current real matrix into a complex matrix,
   * where the real part is the current matrix and the imaginary part is zero.
   *
   * @return Matrix<DefDense, Complex<T>, M, N> A new complex matrix with the
   * same dimensions as the current matrix.
   */
  inline auto create_complex(void) const -> Matrix<DefDense, Complex<T>, M, N> {

    return Matrix<DefDense, Complex<T>, M, N>(
        Base::Matrix::convert_matrix_real_to_complex(this->matrix));
  }

  /**
   * @brief Extracts the real part of a complex matrix.
   *
   * This method returns a new matrix containing only the real part of the
   * current complex matrix.
   *
   * @return Matrix<DefDense, Value_Type, M, N> A new matrix containing the real
   * part of the current complex matrix.
   */
  inline auto real(void) const -> Matrix<DefDense, Value_Type, M, N> {
    return Matrix<DefDense, Value_Type, M, N>(
        ComplexOperation::GetRealFromComplexDenseMatrix<
            Value_Type, T, M, N, IS_COMPLEX>::get(this->matrix));
  }

  /**
   * @brief Extracts the imaginary part of a complex matrix.
   *
   * This method returns a new matrix containing only the imaginary part of the
   * current complex matrix.
   *
   * @return Matrix<DefDense, Value_Type, M, N> A new matrix containing the
   * imaginary part of the current complex matrix.
   */
  inline auto imag(void) const -> Matrix<DefDense, Value_Type, M, N> {
    return Matrix<DefDense, Value_Type, M, N>(
        ComplexOperation::GetImagFromComplexDenseMatrix<
            Value_Type, T, M, N, IS_COMPLEX>::get(this->matrix));
  }

public:
  /* Constant */
  static constexpr std::size_t ROWS = N;
  static constexpr std::size_t COLS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;

public:
  /* Variable */
  Base::Matrix::Matrix<T, M, N> matrix;
};

/**
 * @brief Diagonal matrix class template specialization for DefDiag storage.
 *
 * This class represents a fixed-size diagonal matrix with elements of type T,
 * with dimensions M x M. It provides constructors for various initialization
 * methods, copy/move semantics, element access, and common matrix operations
 * such as transpose and conversion to complex types.
 *
 * @tparam T   Element type of the matrix (e.g., float, double, std::complex).
 * @tparam M   Number of columns/rows in the square matrix.
 *
 * @note
 * - The underlying storage is provided by Base::Matrix::DiagMatrix<T, M>.
 * - Provides compile-time dimension checks for element access.
 * - Supports creation of identity, full-valued matrices.
 * - Supports conversion to complex, and extraction of real/imaginary parts.
 */
template <typename T, std::size_t M> class Matrix<DefDiag, T, M> {
public:
  /* Type */
  using Value_Type = typename UnderlyingType<T>::Type;
  using Value_Complex_Type = T;
  using Matrix_Type = DefDiag;
  using SparseAvailable_Type = DiagAvailable<M>;

public:
  /* Constructor */
  Matrix() {}

  Matrix(const std::initializer_list<T> &input) : matrix(input) {}

  Matrix(T input[M]) : matrix(input) {}

  Matrix(const Base::Matrix::Matrix<T, M, 1> &input) : matrix(input.data[0]) {}

  Matrix(const Matrix<DefDense, T, M, 1> &input)
      : matrix(input.matrix.data[0]) {}

  Matrix(Base::Matrix::DiagMatrix<T, M> &input) : matrix(input) {}

  Matrix(Base::Matrix::DiagMatrix<T, M> &&input) noexcept
      : matrix(std::move(input)) {}

  /* Copy Constructor */
  Matrix(const Matrix<DefDiag, T, M> &input) : matrix(input.matrix) {}

  Matrix<DefDiag, T, M> &operator=(const Matrix<DefDiag, T, M> &input) {
    if (this != &input) {
      this->matrix = input.matrix;
    }
    return *this;
  }

  /* Move Constructor */
  Matrix(Matrix<DefDiag, T, M> &&input) noexcept
      : matrix(std::move(input.matrix)) {}

  Matrix<DefDiag, T, M> &operator=(Matrix<DefDiag, T, M> &&input) noexcept {
    if (this != &input) {
      this->matrix = std::move(input.matrix);
    }
    return *this;
  }

public:
  /* Function */

  template <typename U, std::size_t P, std::size_t I_Col, std::size_t I_Col_Row>
  struct GetSetDiagMatrix {
    /** @brief Gets the value at the specified column and row indices from the
     * diagonal matrix.
     *
     * @param matrix The diagonal matrix to get the value from.
     * @return U The value at the specified column and row.
     *
     * @note This function returns zero for non-diagonal elements.
     */
    static U get_value(const Base::Matrix::DiagMatrix<U, P> &matrix) {
      static_cast<void>(matrix);
      return static_cast<U>(0);
    }

    /** @brief Sets the element at the specified column and row indices to the
     * provided value.
     *
     * @param matrix The diagonal matrix to set the value in.
     * @param value The value to set at the specified column and row.
     *
     * @note This function does nothing for non-diagonal elements.
     */
    static void set_value(Base::Matrix::DiagMatrix<U, P> &matrix, T value) {
      static_cast<void>(matrix);
      static_cast<void>(value);
    }
  };

  template <typename U, std::size_t P, std::size_t I_Col>
  struct GetSetDiagMatrix<U, P, I_Col, 0> {
    /**
     * @brief Gets the value at the specified column index from the diagonal
     * matrix.
     *
     * @param matrix The diagonal matrix to get the value from.
     * @return U The value at the specified column index.
     *
     * @note This function returns the diagonal element for the specified column
     * index.
     */
    static T get_value(const Base::Matrix::DiagMatrix<U, P> &matrix) {

      return matrix.data[I_Col];
    }

    /**
     * @brief Sets the element at the specified column index to the provided
     * value.
     *
     * @param matrix The diagonal matrix to set the value in.
     * @param value The value to set at the specified column index.
     *
     * @note This function sets the diagonal element for the specified column
     * index.
     */
    static void set_value(Base::Matrix::DiagMatrix<U, P> &matrix, T value) {

      matrix.data[I_Col] = value;
    }
  };

  /**
   * @brief Gets the value at the specified column and row indices from the
   * diagonal matrix.
   *
   * @tparam COL The zero-based column index (must be less than M).
   * @tparam ROW The zero-based row index (must be less than M).
   * @return T The value at the specified column and row.
   *
   * @note Compile-time assertions ensure that the provided indices are within
   * valid bounds.
   */
  template <std::size_t COL, std::size_t ROW> inline T get() const {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < M, "Row Index is out of range.");

    return GetSetDiagMatrix<T, M, COL, (COL - ROW)>::get_value(this->matrix);
  }

  /**
   * @brief Sets the element at the specified column and row indices to the
   * provided value.
   *
   * @tparam COL The zero-based column index (must be less than M).
   * @tparam ROW The zero-based row index (must be less than M).
   * @param value The value to set at the specified column and row.
   *
   * @note Compile-time assertions ensure that the provided indices are within
   * valid bounds.
   */
  template <std::size_t COL, std::size_t ROW> inline void set(const T &value) {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < M, "Row Index is out of range.");

    GetSetDiagMatrix<T, M, COL, (COL - ROW)>::set_value(this->matrix, value);
  }

  /**
   * @brief Retrieves the underlying diagonal matrix data.
   *
   * @return Base::Matrix::DiagMatrix<T, M> The underlying diagonal matrix data.
   */
  constexpr std::size_t rows() const { return ROWS; }

  /**
   * @brief Retrieves the number of columns in the diagonal matrix.
   *
   * @return std::size_t The number of columns in the diagonal matrix (equal to
   * M).
   */
  constexpr std::size_t cols() const { return COLS; }

  /**
   * @brief Retrieves the number of elements in the diagonal matrix.
   *
   * @return std::size_t The total number of elements in the diagonal matrix
   * (M).
   */
  T &operator()(std::size_t index) {
    if (index >= M) {
      index = M - 1;
    }

    return this->matrix[index];
  }

  /**
   * @brief Retrieves the element at the specified index in the diagonal matrix.
   *
   * @param index The zero-based index (must be less than M).
   * @return const T& The value at the specified index.
   *
   * @note If the index is out of bounds, it is clamped to the maximum valid
   * index (M - 1).
   */
  const T &operator()(std::size_t index) const {
    if (index >= M) {
      index = M - 1;
    }

    return this->matrix[index];
  }

  /**
   * @brief Accesses the element at the specified index in the diagonal matrix.
   *
   * @param index The zero-based index (must be less than M).
   * @return T& A reference to the value at the specified index.
   *
   * @note This method is fast but may cause segmentation faults if the index is
   * out of bounds.
   */
  inline T &access(const std::size_t &index) {
    // This is fast but may cause segmentation fault.

    return this->matrix[index];
  }

  /**
   * @brief Accesses the element at the specified index in the diagonal matrix.
   *
   * @param index The zero-based index (must be less than M).
   * @return const T& A constant reference to the value at the specified index.
   *
   * @note This method is fast but may cause segmentation faults if the index is
   * out of bounds.
   */
  static inline auto identity(void) -> Matrix<DefDiag, T, M> {
    return Matrix<DefDiag, T, M>(Base::Matrix::DiagMatrix<T, M>::identity());
  }

  /**
   * @brief Creates and returns a diagonal matrix filled with zeros.
   *
   * This static method constructs a diagonal matrix of size M x M,
   * where all diagonal elements are set to zero, and all off-diagonal
   * elements are also zero.
   *
   * @return Matrix<DefDiag, T, M> A diagonal matrix of zeros with dimensions M
   * x M.
   */
  static inline auto full(const T &value) -> Matrix<DefDiag, T, M> {
    return Matrix<DefDiag, T, M>(Base::Matrix::DiagMatrix<T, M>::full(value));
  }

  /**
   * @brief Creates a diagonal matrix filled with zeros.
   *
   * This static method constructs a diagonal matrix of size M x M,
   * where all diagonal elements are set to zero, and all off-diagonal
   * elements are also zero.
   *
   * @return Matrix<DefDiag, T, M> A diagonal matrix of zeros with dimensions M
   * x M.
   */
  inline auto create_dense(void) const -> Matrix<DefDense, T, M, M> {
    return Matrix<DefDense, T, M, M>(
        Base::Matrix::output_dense_matrix(this->matrix));
  }

  /**
   * @brief Transposes the diagonal matrix.
   *
   * This method returns a new diagonal matrix that is the transpose of the
   * current matrix. For diagonal matrices, the transpose is the same as the
   * original matrix.
   *
   * @return Matrix<DefDiag, T, M> A new diagonal matrix that is the transpose
   * of the current matrix.
   */
  inline auto transpose(void) const -> Matrix<DefDiag, T, M> { return *this; }

  /**
   * @brief Creates a complex diagonal matrix from the current diagonal matrix.
   *
   * This method converts the current real diagonal matrix into a complex
   * diagonal matrix, where the real part is the current matrix and the
   * imaginary part is zero.
   *
   * @return Matrix<DefDiag, Complex<T>, M> A new complex diagonal matrix with
   * the same dimensions as the current matrix.
   */
  inline auto create_complex(void) const -> Matrix<DefDiag, Complex<T>, M> {

    return Matrix<DefDiag, Complex<T>, M>(
        Base::Matrix::convert_matrix_real_to_complex(this->matrix));
  }

  /**
   * @brief Extracts the real part of a complex diagonal matrix.
   *
   * This method returns a new diagonal matrix containing only the real part of
   * the current complex diagonal matrix.
   *
   * @return Matrix<DefDiag, Value_Type, M> A new diagonal matrix containing the
   * real part of the current complex diagonal matrix.
   */
  inline auto real(void) const -> Matrix<DefDiag, Value_Type, M> {
    return Matrix<DefDiag, Value_Type, M>(
        ComplexOperation::GetRealFromComplexDiagMatrix<
            Value_Type, T, M, IS_COMPLEX>::get(this->matrix));
  }

  /**
   * @brief Extracts the imaginary part of a complex diagonal matrix.
   *
   * This method returns a new diagonal matrix containing only the imaginary
   * part of the current complex diagonal matrix.
   *
   * @return Matrix<DefDiag, Value_Type, M> A new diagonal matrix containing the
   * imaginary part of the current complex diagonal matrix.
   */
  inline auto imag(void) const -> Matrix<DefDiag, Value_Type, M> {
    return Matrix<DefDiag, Value_Type, M>(
        ComplexOperation::GetImagFromComplexDiagMatrix<
            Value_Type, T, M, IS_COMPLEX>::get(this->matrix));
  }

public:
  /* Constant */
  static constexpr std::size_t ROWS = M;
  static constexpr std::size_t COLS = M;

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;

public:
  /* Variable */
  Base::Matrix::DiagMatrix<T, M> matrix;
};

/**
 * @brief Sparse matrix class template specialization for DefSparse storage.
 *
 * This class represents a fixed-size sparse matrix with elements of type T,
 * with dimensions M x N. It provides constructors for various initialization
 * methods, copy/move semantics, element access, and common matrix operations
 * such as transpose and conversion to complex types.
 *
 * @tparam T   Element type of the matrix (e.g., float, double, std::complex).
 * @tparam M   Number of columns in the matrix.
 * @tparam N   Number of rows in the matrix.
 * @tparam SparseAvailable Type indicating the availability of sparse features.
 *
 * @note
 * - The underlying storage is provided by Base::Matrix::CompiledSparseMatrix<T,
 * M, N, RowIndices_Type, RowPointers_Type>.
 * - Provides compile-time dimension checks for element access.
 * - Supports creation of full-valued matrices.
 * - Supports conversion to complex, and extraction of real/imaginary parts.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
class Matrix<DefSparse, T, M, N, SparseAvailable> {
public:
  /* Type */
  using Value_Type = typename UnderlyingType<T>::Type;
  using Value_Complex_Type = T;
  using Matrix_Type = DefSparse;
  using SparseAvailable_Type = SparseAvailable;

protected:
  /* Type */
  using _ValidateSparseAvailable = ValidateSparseAvailable<SparseAvailable>;
  using _RowIndices_Type = RowIndicesFromSparseAvailable<SparseAvailable>;
  using _RowPointers_Type = RowPointersFromSparseAvailable<SparseAvailable>;

  using _BaseMatrix_Type =
      Base::Matrix::CompiledSparseMatrix<T, M, N, _RowIndices_Type,
                                         _RowPointers_Type>;

public:
  /* Constructor */
  Matrix() {}

  Matrix(const std::initializer_list<T> &values) : matrix(values) {}

  Matrix(_BaseMatrix_Type &input) : matrix(input) {}

  Matrix(_BaseMatrix_Type &&input) noexcept : matrix(std::move(input)) {}

  /* Copy Constructor */
  Matrix(const Matrix<DefSparse, T, M, N, SparseAvailable> &input)
      : matrix(input.matrix) {}

  Matrix<DefSparse, T, M, N, SparseAvailable> &
  operator=(const Matrix<DefSparse, T, M, N, SparseAvailable> &input) {
    if (this != &input) {
      this->matrix = input.matrix;
    }
    return *this;
  }

  /* Move Constructor */
  Matrix(Matrix<DefSparse, T, M, N, SparseAvailable> &&input) noexcept
      : matrix(std::move(input.matrix)) {}

  Matrix<DefSparse, T, M, N, SparseAvailable> &
  operator=(Matrix<DefSparse, T, M, N, SparseAvailable> &&input) noexcept {
    if (this != &input) {
      this->matrix = std::move(input.matrix);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Creates a dense matrix from the sparse matrix.
   *
   * This method converts the current sparse matrix into a dense matrix,
   * where all elements are represented explicitly.
   *
   * @return Matrix<DefDense, T, M, N> A new dense matrix with the same
   * dimensions as the current sparse matrix.
   */
  inline auto create_dense(void) const -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>(
        Base::Matrix::output_dense_matrix(this->matrix));
  }

  /**
   * @brief Retrieves the element at the specified column and row indices from
   * the sparse matrix.
   *
   * @tparam COL The zero-based column index (must be less than M).
   * @tparam ROW The zero-based row index (must be less than N).
   * @return T The value at the specified column and row.
   *
   * @note Compile-time assertions ensure that the provided indices are within
   * valid bounds.
   */
  template <std::size_t COL, std::size_t ROW> inline T get() const {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    return Base::Matrix::get_sparse_matrix_value<COL, ROW>(this->matrix);
  }

  /**
   * @brief Retrieves the value of a specific element in the sparse matrix.
   *
   * @tparam ELEMENT The zero-based index of the element (must be less than
   * NumberOfValues).
   * @return T The value of the specified element.
   *
   * @note Compile-time assertions ensure that the provided index is within
   * valid bounds.
   */
  template <std::size_t ELEMENT> inline T get() const {
    static_assert(ELEMENT < NumberOfValues,
                  "ELEMENT must be the same or less than the number "
                  "of elements of Sparse Matrix.");

    return Base::Matrix::get_sparse_matrix_element_value<ELEMENT>(this->matrix);
  }

  /**
   * @brief Sets the element at the specified column and row indices to the
   * provided value.
   *
   * @tparam COL The zero-based column index (must be less than M).
   * @tparam ROW The zero-based row index (must be less than N).
   * @param value The value to set at the specified column and row.
   *
   * @note Compile-time assertions ensure that the provided indices are within
   * valid bounds.
   */
  template <std::size_t COL, std::size_t ROW> inline void set(const T &value) {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    Base::Matrix::set_sparse_matrix_value<COL, ROW>(this->matrix, value);
  }

  /**
   * @brief Sets the value of a specific element in the sparse matrix.
   *
   * @tparam ELEMENT The zero-based index of the element (must be less than
   * NumberOfValues).
   * @param value The value to set for the specified element.
   *
   * @note Compile-time assertions ensure that the provided index is within
   * valid bounds.
   */
  template <std::size_t ELEMENT> inline void set(const T &value) {
    static_assert(ELEMENT < NumberOfValues,
                  "ELEMENT must be the same or less than the number "
                  "of elements of Sparse Matrix.");

    Base::Matrix::set_sparse_matrix_element_value<ELEMENT>(this->matrix, value);
  }

  /**
   * @brief Retrieves the underlying sparse matrix data.
   *
   * @return _BaseMatrix_Type The underlying sparse matrix data.
   */
  constexpr std::size_t rows() const { return ROWS; }

  /**
   * @brief Retrieves the number of columns in the sparse matrix.
   *
   * @return std::size_t The number of columns in the sparse matrix (equal to
   * M).
   */
  constexpr std::size_t cols() const { return COLS; }

  /**
   * @brief Retrieves the number of elements in the sparse matrix.
   *
   * @return std::size_t The total number of elements in the sparse matrix (M *
   * N).
   */
  T &operator()(std::size_t value_index) {
    if (value_index >= this->matrix.values.size()) {
      value_index = this->matrix.values.size() - 1;
    }

    return this->matrix[value_index];
  }

  /**
   * @brief Retrieves the element at the specified index in the sparse matrix.
   *
   * @param value_index The zero-based index of the element (must be less than
   * NumberOfValues).
   * @return const T& The value at the specified index.
   *
   * @note If the index is out of bounds, it is clamped to the maximum valid
   * index (NumberOfValues - 1).
   */
  const T &operator()(std::size_t value_index) const {
    if (value_index >= this->matrix.values.size()) {
      value_index = this->matrix.values.size() - 1;
    }

    return this->matrix[value_index];
  }

  /**
   * @brief Accesses the element at the specified index in the sparse matrix.
   *
   * @param value_index The zero-based index of the element (must be less than
   * NumberOfValues).
   * @return T& A reference to the value at the specified index.
   *
   * @note This method is fast but may cause segmentation faults if the index is
   * out of bounds.
   */
  inline T &access(const std::size_t &value_index) {
    // This is fast but may cause segmentation fault.

    return this->matrix[value_index];
  }

  /**
   * @brief Accesses the element at the specified index in the sparse matrix.
   *
   * @param value_index The zero-based index of the element (must be less than
   * NumberOfValues).
   * @return const T& A constant reference to the value at the specified index.
   *
   * @note This method is fast but may cause segmentation faults if the index is
   * out of bounds.
   */
  static inline auto full(const T &value)
      -> Matrix<DefSparse, T, M, N, SparseAvailable> {
    return Matrix<DefSparse, T, M, N, SparseAvailable>(
        _BaseMatrix_Type::full(value));
  }

  /**
   * @brief Transposes the sparse matrix.
   *
   * This method returns a new sparse matrix that is the transpose of the
   * current matrix, swapping rows and columns.
   *
   * @return Matrix<DefSparse, T, N, M,
   * SparseAvailableTranspose<SparseAvailable>> A new sparse matrix that is the
   * transpose of the current matrix.
   */
  inline auto transpose(void) const
      -> Matrix<DefSparse, T, N, M, SparseAvailableTranspose<SparseAvailable>> {

    return Matrix<DefSparse, T, N, M,
                  SparseAvailableTranspose<SparseAvailable>>(
        Base::Matrix::output_matrix_transpose(this->matrix));
  }

  /**
   * @brief Creates a complex sparse matrix from the current sparse matrix.
   *
   * This method converts the current real sparse matrix into a complex sparse
   * matrix, where the real part is the current matrix and the imaginary part is
   * zero.
   *
   * @return Matrix<DefSparse, Complex<T>, M, N, SparseAvailable> A new complex
   * sparse matrix with the same dimensions as the current matrix.
   */
  inline auto create_complex(void) const
      -> Matrix<DefSparse, Complex<T>, M, N, SparseAvailable> {

    return Matrix<DefSparse, Complex<T>, M, N, SparseAvailable>(
        Base::Matrix::convert_matrix_real_to_complex(this->matrix));
  }

  /**
   * @brief Extracts the real part of a complex sparse matrix.
   *
   * This method returns a new sparse matrix containing only the real part of
   * the current complex sparse matrix.
   *
   * @return Matrix<DefSparse, Value_Type, M, N, SparseAvailable> A new sparse
   * matrix containing the real part of the current complex sparse matrix.
   */
  inline auto real(void) const
      -> Matrix<DefSparse, Value_Type, M, N, SparseAvailable> {
    return Matrix<DefSparse, Value_Type, M, N, SparseAvailable>(
        ComplexOperation::GetRealFromComplexSparseMatrix<
            Value_Type, T, M, N, SparseAvailable,
            IS_COMPLEX>::get(this->matrix));
  }

  /**
   * @brief Extracts the imaginary part of a complex sparse matrix.
   *
   * This method returns a new sparse matrix containing only the imaginary part
   * of the current complex sparse matrix.
   *
   * @return Matrix<DefSparse, Value_Type, M, N, SparseAvailable> A new sparse
   * matrix containing the imaginary part of the current complex sparse matrix.
   */
  inline auto imag(void) const
      -> Matrix<DefSparse, Value_Type, M, N, SparseAvailable> {
    return Matrix<DefSparse, Value_Type, M, N, SparseAvailable>(
        ComplexOperation::GetImagFromComplexSparseMatrix<
            Value_Type, T, M, N, SparseAvailable,
            IS_COMPLEX>::get(this->matrix));
  }

public:
  /* Constant */
  static constexpr std::size_t ROWS = N;
  static constexpr std::size_t COLS = M;

  static constexpr std::size_t NumberOfValues = _RowPointers_Type::list[M];

  static constexpr bool IS_COMPLEX = Is_Complex_Type<T>::value;

public:
  /* Variable */
  _BaseMatrix_Type matrix;
};

/* Matrix Addition */

/**
 * @brief Adds two matrices together.
 *
 * This function overloads the '+' operator to add two matrices of the same type
 * and dimensions. It returns a new matrix containing the sum of the two input
 * matrices.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param A The first matrix to add.
 * @param B The second matrix to add.
 * @return Matrix<DefDense, T, M, N> A new matrix containing the sum of A and B.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto operator+(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

/**
 * @brief Adds a dense matrix and a diagonal matrix together.
 *
 * This function overloads the '+' operator to add a dense matrix and a
 * diagonal matrix of the same size. It returns a new dense matrix containing
 * the sum of the two matrices.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param A The dense matrix to add.
 * @param B The diagonal matrix to add.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the sum of A
 * and B.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto operator+(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

/**
 * @brief Adds a dense matrix and a sparse matrix together.
 *
 * This function overloads the '+' operator to add a dense matrix and a sparse
 * matrix of the same size. It returns a new dense matrix containing the sum of
 * the two matrices.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param A The dense matrix to add.
 * @param B The sparse matrix to add.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the sum of A
 * and B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator+(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

/**
 * @brief Adds a diagonal matrix and a dense matrix together.
 *
 * This function overloads the '+' operator to add a diagonal matrix and a
 * dense matrix of the same size. It returns a new dense matrix containing the
 * sum of the two matrices.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param A The diagonal matrix to add.
 * @param B The dense matrix to add.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the sum of A
 * and B.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto operator+(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

/**
 * @brief Adds two diagonal matrices together.
 *
 * This function overloads the '+' operator to add two diagonal matrices of the
 * same size. It returns a new diagonal matrix containing the sum of the two
 * matrices.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param A The first diagonal matrix to add.
 * @param B The second diagonal matrix to add.
 * @return Matrix<DefDiag, T, M> A new diagonal matrix containing the sum of A
 * and B.
 */
template <typename T, std::size_t M>
inline auto operator+(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefDiag, T, M> &B) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(A.matrix + B.matrix));
}

/**
 * @brief Adds a diagonal matrix and a sparse matrix together.
 *
 * This function overloads the '+' operator to add a diagonal matrix and a
 * sparse matrix of the same size. It returns a new sparse matrix containing the
 * sum of the two matrices.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param A The diagonal matrix to add.
 * @param B The sparse matrix to add.
 * @return Matrix<DefSparse, T, M, N> A new sparse matrix containing the sum of
 * A and B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator+(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefSparse, T, M, N,
              MatrixAddSubSparseAvailable<DiagAvailable<M>, SparseAvailable>> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefSparse, T, M, N,
                MatrixAddSubSparseAvailable<DiagAvailable<M>, SparseAvailable>>(
      std::move(A.matrix + B.matrix));
}

/**
 * @brief Adds a sparse matrix and a dense matrix together.
 *
 * This function overloads the '+' operator to add a sparse matrix and a dense
 * matrix of the same size. It returns a new dense matrix containing the sum of
 * the two matrices.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param A The sparse matrix to add.
 * @param B The dense matrix to add.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the sum of A
 * and B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator+(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

/**
 * @brief Adds a sparse matrix and a diagonal matrix together.
 *
 * This function overloads the '+' operator to add a sparse matrix and a
 * diagonal matrix of the same size. It returns a new sparse matrix containing
 * the sum of the two matrices.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param A The sparse matrix to add.
 * @param B The diagonal matrix to add.
 * @return Matrix<DefSparse, T, M, N> A new sparse matrix containing the sum of
 * A and B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator+(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
                      const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, M, N,
              MatrixAddSubSparseAvailable<SparseAvailable, DiagAvailable<M>>> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefSparse, T, M, N,
                MatrixAddSubSparseAvailable<SparseAvailable, DiagAvailable<M>>>(
      std::move(A.matrix + B.matrix));
}

/**
 * @brief Adds two sparse matrices together.
 *
 * This function overloads the '+' operator to add two sparse matrices of the
 * same type and dimensions. It returns a new sparse matrix containing the sum
 * of the two input matrices.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param A The first sparse matrix to add.
 * @param B The second sparse matrix to add.
 * @return Matrix<DefSparse, T, M, N> A new sparse matrix containing the sum of
 * A and B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A,
          typename SparseAvailable_B>
inline auto operator+(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                      const Matrix<DefSparse, T, M, N, SparseAvailable_B> &B)
    -> Matrix<
        DefSparse, T, M, N,
        MatrixAddSubSparseAvailable<SparseAvailable_A, SparseAvailable_B>> {

  return Matrix<
      DefSparse, T, M, N,
      MatrixAddSubSparseAvailable<SparseAvailable_A, SparseAvailable_B>>(
      std::move(A.matrix + B.matrix));
}

/* Matrix Subtraction */

/**
 * @brief Subtracts two matrices.
 *
 * This function overloads the '-' operator to subtract one matrix from another
 * of the same type and dimensions. It returns a new matrix containing the
 * result of the subtraction.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param A The matrix to subtract from.
 * @param B The matrix to subtract.
 * @return Matrix<DefDense, T, M, N> A new matrix containing the result of A -
 * B.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto operator-(const Matrix<DefDense, T, M, N> &A)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(-A.matrix));
}

/**
 * @brief Subtracts a dense matrix from a diagonal matrix.
 *
 * This function overloads the '-' operator to subtract a dense matrix from a
 * diagonal matrix of the same size. It returns a new dense matrix containing
 * the result of the subtraction.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param A The diagonal matrix to subtract from.
 * @param B The dense matrix to subtract.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * A - B.
 */
template <typename T, std::size_t M>
inline auto operator-(const Matrix<DefDiag, T, M> &A) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(-A.matrix));
}

/**
 * @brief Subtracts a dense matrix from a sparse matrix.
 *
 * This function overloads the '-' operator to subtract a dense matrix from a
 * sparse matrix of the same size. It returns a new dense matrix containing the
 * result of the subtraction.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param A The dense matrix to subtract from.
 * @param B The sparse matrix to subtract.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * A - B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator-(const Matrix<DefSparse, T, M, N, SparseAvailable> &A)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {

  return Matrix<DefSparse, T, M, N, SparseAvailable>(std::move(-A.matrix));
}

/**
 * @brief Subtracts a dense matrix from another dense matrix.
 *
 * This function overloads the '-' operator to subtract one dense matrix from
 * another of the same type and dimensions. It returns a new dense matrix
 * containing the result of the subtraction.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param A The first dense matrix to subtract from.
 * @param B The second dense matrix to subtract.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * A - B.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto operator-(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

/**
 * @brief Subtracts a diagonal matrix from a dense matrix.
 *
 * This function overloads the '-' operator to subtract a diagonal matrix from
 * a dense matrix of the same size. It returns a new dense matrix containing the
 * result of the subtraction.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param A The dense matrix to subtract from.
 * @param B The diagonal matrix to subtract.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * A - B.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto operator-(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

/**
 * @brief Subtracts a dense matrix from a sparse matrix.
 *
 * This function overloads the '-' operator to subtract a dense matrix from a
 * sparse matrix of the same size. It returns a new dense matrix containing the
 * result of the subtraction.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param A The dense matrix to subtract from.
 * @param B The sparse matrix to subtract.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * A - B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator-(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

/**
 * @brief Subtracts a diagonal matrix from a dense matrix.
 *
 * This function overloads the '-' operator to subtract a diagonal matrix from
 * a dense matrix of the same size. It returns a new dense matrix containing the
 * result of the subtraction.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param A The diagonal matrix to subtract from.
 * @param B The dense matrix to subtract.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * A - B.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto operator-(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

/**
 * @brief Subtracts two diagonal matrices.
 *
 * This function overloads the '-' operator to subtract one diagonal matrix
 * from another of the same type and dimensions. It returns a new diagonal
 * matrix containing the result of the subtraction.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param A The first diagonal matrix to subtract from.
 * @param B The second diagonal matrix to subtract.
 * @return Matrix<DefDiag, T, M> A new diagonal matrix containing the result of
 * A
 * - B.
 */
template <typename T, std::size_t M>
inline auto operator-(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefDiag, T, M> &B) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(A.matrix - B.matrix));
}

/**
 * @brief Subtracts a diagonal matrix from a sparse matrix.
 *
 * This function overloads the '-' operator to subtract a diagonal matrix from
 * a sparse matrix of the same size. It returns a new sparse matrix containing
 * the result of the subtraction.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param A The diagonal matrix to subtract from.
 * @param B The sparse matrix to subtract.
 * @return Matrix<DefSparse, T, M, N> A new sparse matrix containing the result
 * of A - B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator-(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefSparse, T, M, N,
              MatrixAddSubSparseAvailable<DiagAvailable<M>, SparseAvailable>> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefSparse, T, M, N,
                MatrixAddSubSparseAvailable<DiagAvailable<M>, SparseAvailable>>(
      std::move(A.matrix - B.matrix));
}

/**
 * @brief Subtracts a sparse matrix from a dense matrix.
 *
 * This function overloads the '-' operator to subtract a sparse matrix from a
 * dense matrix of the same size. It returns a new dense matrix containing the
 * result of the subtraction.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param A The sparse matrix to subtract from.
 * @param B The dense matrix to subtract.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * A - B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator-(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

/**
 * @brief Subtracts a sparse matrix from a dense matrix.
 *
 * This function overloads the '-' operator to subtract a dense matrix from a
 * sparse matrix of the same size. It returns a new sparse matrix containing the
 * result of the subtraction.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param A The sparse matrix to subtract from.
 * @param B The dense matrix to subtract.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * A - B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator-(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
                      const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, M, N,
              MatrixAddSubSparseAvailable<SparseAvailable, DiagAvailable<M>>> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefSparse, T, M, N,
                MatrixAddSubSparseAvailable<SparseAvailable, DiagAvailable<M>>>(
      std::move(A.matrix - B.matrix));
}

/**
 * @brief Subtracts two sparse matrices.
 *
 * This function overloads the '-' operator to subtract one sparse matrix from
 * another of the same type and dimensions. It returns a new sparse matrix
 * containing the result of the subtraction.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param A The first sparse matrix to subtract from.
 * @param B The second sparse matrix to subtract.
 * @return Matrix<DefSparse, T, M, N> A new sparse matrix containing the result
 * of A - B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A,
          typename SparseAvailable_B>
inline auto operator-(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                      const Matrix<DefSparse, T, M, N, SparseAvailable_B> &B)
    -> Matrix<
        DefSparse, T, M, N,
        MatrixAddSubSparseAvailable<SparseAvailable_A, SparseAvailable_B>> {

  return Matrix<
      DefSparse, T, M, N,
      MatrixAddSubSparseAvailable<SparseAvailable_A, SparseAvailable_B>>(
      std::move(A.matrix - B.matrix));
}

/* Matrix Multiply Scalar */

/**
 * @brief Multiplies a scalar with a dense matrix.
 *
 * This function overloads the '*' operator to multiply a scalar with a dense
 * matrix. It returns a new dense matrix containing the result of the
 * multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param a The scalar to multiply with.
 * @param B The dense matrix to multiply.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * a * B.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto operator*(const T &a, const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(a * B.matrix));
}

/**
 * @brief Multiplies a dense matrix with a scalar.
 *
 * This function overloads the '*' operator to multiply a dense matrix with a
 * scalar. It returns a new dense matrix containing the result of the
 * multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param B The dense matrix to multiply.
 * @param a The scalar to multiply with.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * B * a.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto operator*(const Matrix<DefDense, T, M, N> &B, const T &a)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(B.matrix * a));
}

/**
 * @brief Multiplies a scalar with a diagonal matrix.
 *
 * This function overloads the '*' operator to multiply a scalar with a
 * diagonal matrix. It returns a new diagonal matrix containing the result of
 * the multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param a The scalar to multiply with.
 * @param B The diagonal matrix to multiply.
 * @return Matrix<DefDiag, T, M> A new diagonal matrix containing the result of
 * a * B.
 */
template <typename T, std::size_t M>
inline auto operator*(const T &a, const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(a * B.matrix));
}

/**
 * @brief Multiplies a diagonal matrix with a scalar.
 *
 * This function overloads the '*' operator to multiply a diagonal matrix with
 * a scalar. It returns a new diagonal matrix containing the result of the
 * multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param B The diagonal matrix to multiply.
 * @param a The scalar to multiply with.
 * @return Matrix<DefDiag, T, M> A new diagonal matrix containing the result of
 * B * a.
 */
template <typename T, std::size_t M>
inline auto operator*(const Matrix<DefDiag, T, M> &B, const T &a)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(B.matrix * a));
}

/**
 * @brief Multiplies a scalar with a sparse matrix.
 *
 * This function overloads the '*' operator to multiply a scalar with a sparse
 * matrix. It returns a new sparse matrix containing the result of the
 * multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param a The scalar to multiply with.
 * @param B The sparse matrix to multiply.
 * @return Matrix<DefSparse, T, M, N> A new sparse matrix containing the result
 * of a * B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator*(const T &a,
                      const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {

  return Matrix<DefSparse, T, M, N, SparseAvailable>(std::move(a * B.matrix));
}

/**
 * @brief Multiplies a sparse matrix with a scalar.
 *
 * This function overloads the '*' operator to multiply a sparse matrix with a
 * scalar. It returns a new sparse matrix containing the result of the
 * multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the matrices.
 * @tparam N The number of rows in the matrices.
 * @param B The sparse matrix to multiply.
 * @param a The scalar to multiply with.
 * @return Matrix<DefSparse, T, M, N> A new sparse matrix containing the result
 * of B * a.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator*(const Matrix<DefSparse, T, M, N, SparseAvailable> &B,
                      const T &a)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {

  return Matrix<DefSparse, T, M, N, SparseAvailable>(std::move(B.matrix * a));
}

/* Matrix Multiply Matrix */

/**
 * @brief Multiplies two dense matrices.
 *
 * This function overloads the '*' operator to multiply two dense matrices of
 * compatible dimensions. It returns a new dense matrix containing the result of
 * the multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the first matrix.
 * @tparam N The number of rows in the first matrix and columns in the second
 * matrix.
 * @tparam K The number of rows in the second matrix.
 * @param A The first dense matrix to multiply.
 * @param B The second dense matrix to multiply.
 * @return Matrix<DefDense, T, M, K> A new dense matrix containing the result of
 * A * B.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K>
inline auto operator*(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(std::move(A.matrix * B.matrix));
}

/**
 * @brief Multiplies a dense matrix and a diagonal matrix.
 *
 * This function overloads the '*' operator to multiply a dense matrix and a
 * diagonal matrix of compatible dimensions. It returns a new dense matrix
 * containing the result of the multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the dense matrix and rows in the diagonal
 * matrix.
 * @tparam N The number of rows in the dense matrix and columns in the diagonal
 * matrix.
 * @param A The dense matrix to multiply.
 * @param B The diagonal matrix to multiply.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * A * B.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto operator*(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefDiag, T, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix * B.matrix));
}

/**
 * @brief Multiplies a dense matrix and a sparse matrix.
 *
 * This function overloads the '*' operator to multiply a dense matrix and a
 * sparse matrix of compatible dimensions. It returns a new dense matrix
 * containing the result of the multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the dense matrix and rows in the sparse
 * matrix.
 * @tparam N The number of rows in the dense matrix and columns in the sparse
 * matrix.
 * @param A The dense matrix to multiply.
 * @param B The sparse matrix to multiply.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * A * B.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable>
inline auto operator*(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefSparse, T, N, K, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(std::move(A.matrix * B.matrix));
}

/**
 * @brief Multiplies a diagonal matrix and a dense matrix.
 *
 * This function overloads the '*' operator to multiply a diagonal matrix and a
 * dense matrix of compatible dimensions. It returns a new dense matrix
 * containing the result of the multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the diagonal matrix and rows in the dense
 * matrix.
 * @tparam N The number of rows in the diagonal matrix and columns in the dense
 * matrix.
 * @param A The diagonal matrix to multiply.
 * @param B The dense matrix to multiply.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * A * B.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto operator*(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix * B.matrix));
}

/**
 * @brief Multiplies two diagonal matrices.
 *
 * This function overloads the '*' operator to multiply two diagonal matrices
 * of the same size. It returns a new diagonal matrix containing the result of
 * the multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param A The first diagonal matrix to multiply.
 * @param B The second diagonal matrix to multiply.
 * @return Matrix<DefDiag, T, M> A new diagonal matrix containing the result of
 * A * B.
 */
template <typename T, std::size_t M>
inline auto operator*(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefDiag, T, M> &B) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(A.matrix * B.matrix));
}

/**
 * @brief Multiplies a diagonal matrix and a sparse matrix.
 *
 * This function overloads the '*' operator to multiply a diagonal matrix and a
 * sparse matrix of compatible dimensions. It returns a new sparse matrix
 * containing the result of the multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns/rows in the square matrices.
 * @param A The diagonal matrix to multiply.
 * @param B The sparse matrix to multiply.
 * @return Matrix<DefSparse, T, M, N> A new sparse matrix containing the result
 * of A * B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator*(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {

  return Matrix<DefSparse, T, M, N, SparseAvailable>(
      std::move(A.matrix * B.matrix));
}

/**
 * @brief Multiplies a sparse matrix and a dense matrix.
 *
 * This function overloads the '*' operator to multiply a sparse matrix and a
 * dense matrix of compatible dimensions. It returns a new dense matrix
 * containing the result of the multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the sparse matrix and rows in the dense
 * matrix.
 * @tparam N The number of rows in the sparse matrix and columns in the dense
 * matrix.
 * @param A The sparse matrix to multiply.
 * @param B The dense matrix to multiply.
 * @return Matrix<DefDense, T, M, N> A new dense matrix containing the result of
 * A * B.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable>
inline auto operator*(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
                      const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(std::move(A.matrix * B.matrix));
}

/**
 * @brief Multiplies a sparse matrix and a diagonal matrix.
 *
 * This function overloads the '*' operator to multiply a sparse matrix and a
 * diagonal matrix of compatible dimensions. It returns a new sparse matrix
 * containing the result of the multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the sparse matrix and rows in the diagonal
 * matrix.
 * @tparam N The number of rows in the sparse matrix and columns in the diagonal
 * matrix.
 * @param A The sparse matrix to multiply.
 * @param B The diagonal matrix to multiply.
 * @return Matrix<DefSparse, T, M, N> A new sparse matrix containing the result
 * of A * B.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator*(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
                      const Matrix<DefDiag, T, N> &B)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {

  return Matrix<DefSparse, T, M, N, SparseAvailable>(
      std::move(A.matrix * B.matrix));
}

/**
 * @brief Multiplies two sparse matrices.
 *
 * This function overloads the '*' operator to multiply two sparse matrices of
 * compatible dimensions. It returns a new sparse matrix containing the result
 * of the multiplication.
 *
 * @tparam T The type of elements in the matrices (e.g., float, double).
 * @tparam M The number of columns in the first sparse matrix.
 * @tparam N The number of rows in the first sparse matrix and columns in the
 * second sparse matrix.
 * @tparam K The number of rows in the second sparse matrix.
 * @param A The first sparse matrix to multiply.
 * @param B The second sparse matrix to multiply.
 * @return Matrix<DefSparse, T, M, K> A new sparse matrix containing the result
 * of A * B.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline auto operator*(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                      const Matrix<DefSparse, T, N, K, SparseAvailable_B> &B)
    -> Matrix<
        DefSparse, T, M, K,
        SparseAvailableMatrixMultiply<SparseAvailable_A, SparseAvailable_B>> {

  return Matrix<
      DefSparse, T, M, K,
      SparseAvailableMatrixMultiply<SparseAvailable_A, SparseAvailable_B>>(
      std::move(A.matrix * B.matrix));
}

/* Type */
template <typename A_Type, typename B_Type>
using A_Multiply_B_Type =
    decltype(std::declval<A_Type>() * std::declval<B_Type>());

template <typename A_Type>
using Transpose_Type = decltype(std::declval<A_Type>().transpose());

/* Matrix Type Checker */
template <typename MatrixInput>
using Is_Dense_Matrix =
    std::is_same<typename MatrixInput::Matrix_Type, DefDense>;

template <typename MatrixInput>
using Is_Diag_Matrix = std::is_same<typename MatrixInput::Matrix_Type, DefDiag>;

template <typename MatrixInput>
using Is_Sparse_Matrix =
    std::is_same<typename MatrixInput::Matrix_Type, DefSparse>;

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_BASE_HPP__
