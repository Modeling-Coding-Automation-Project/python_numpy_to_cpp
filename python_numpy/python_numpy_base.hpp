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

  /* Function */
  /* Get Dense Matrix value */
  template <std::size_t COL, std::size_t ROW> inline T get() const {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    return matrix.data[ROW][COL];
  }

  /* Set Dense Matrix value */
  template <std::size_t COL, std::size_t ROW> inline void set(const T &value) {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    matrix.data[ROW][COL] = value;
  }

  constexpr std::size_t rows() const { return ROWS; }

  constexpr std::size_t cols() const { return COLS; }

  T &operator()(std::size_t col, std::size_t row) {
    if (col >= M) {
      col = M - 1;
    }
    if (row >= N) {
      row = N - 1;
    }

    return this->matrix(col, row);
  }

  const T &operator()(std::size_t col, std::size_t row) const {
    if (col >= M) {
      col = M - 1;
    }
    if (row >= N) {
      row = N - 1;
    }

    return this->matrix(col, row);
  }

  inline T &access(const std::size_t &col, const std::size_t &row) {
    // This is fast but may cause segmentation fault.

    return this->matrix(row)[col];
  }

  static inline auto zeros(void) -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>();
  }

  static inline auto ones(void) -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>(Base::Matrix::Matrix<T, M, N>::ones());
  }

  static inline auto full(const T &value) -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>(
        Base::Matrix::Matrix<T, M, N>::full(value));
  }

  inline auto transpose(void) const -> Matrix<DefDense, T, N, M> {
    return Matrix<DefDense, T, N, M>(
        Base::Matrix::output_matrix_transpose(this->matrix));
  }

  inline auto create_complex(void) const -> Matrix<DefDense, Complex<T>, M, N> {

    return Matrix<DefDense, Complex<T>, M, N>(
        Base::Matrix::convert_matrix_real_to_complex(this->matrix));
  }

  inline auto real(void) const -> Matrix<DefDense, Value_Type, M, N> {
    return Matrix<DefDense, Value_Type, M, N>(
        ComplexOperation::GetRealFromComplexDenseMatrix<
            Value_Type, T, M, N, IS_COMPLEX>::get(this->matrix));
  }

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

  /* Function */
  /* Get Diag Matrix value */
  template <typename U, std::size_t P, std::size_t I_Col, std::size_t I_Col_Row>
  struct GetSetDiagMatrix {
    static U get_value(const Base::Matrix::DiagMatrix<U, P> &matrix) {
      static_cast<void>(matrix);
      return static_cast<U>(0);
    }

    static void set_value(Base::Matrix::DiagMatrix<U, P> &matrix, T value) {
      static_cast<void>(matrix);
      static_cast<void>(value);
    }
  };

  template <typename U, std::size_t P, std::size_t I_Col>
  struct GetSetDiagMatrix<U, P, I_Col, 0> {
    static T get_value(const Base::Matrix::DiagMatrix<U, P> &matrix) {

      return matrix.data[I_Col];
    }

    static void set_value(Base::Matrix::DiagMatrix<U, P> &matrix, T value) {

      matrix.data[I_Col] = value;
    }
  };

  template <std::size_t COL, std::size_t ROW> inline T get() const {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < M, "Row Index is out of range.");

    return GetSetDiagMatrix<T, M, COL, (COL - ROW)>::get_value(this->matrix);
  }

  /* Set Diag Matrix value */
  template <std::size_t COL, std::size_t ROW> inline void set(const T &value) {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < M, "Row Index is out of range.");

    GetSetDiagMatrix<T, M, COL, (COL - ROW)>::set_value(this->matrix, value);
  }

  constexpr std::size_t rows() const { return ROWS; }

  constexpr std::size_t cols() const { return COLS; }

  T &operator()(std::size_t index) {
    if (index >= M) {
      index = M - 1;
    }

    return this->matrix[index];
  }

  const T &operator()(std::size_t index) const {
    if (index >= M) {
      index = M - 1;
    }

    return this->matrix[index];
  }

  inline T &access(const std::size_t &index) {
    // This is fast but may cause segmentation fault.

    return this->matrix[index];
  }

  static inline auto identity(void) -> Matrix<DefDiag, T, M> {
    return Matrix<DefDiag, T, M>(Base::Matrix::DiagMatrix<T, M>::identity());
  }

  static inline auto full(const T &value) -> Matrix<DefDiag, T, M> {
    return Matrix<DefDiag, T, M>(Base::Matrix::DiagMatrix<T, M>::full(value));
  }

  inline auto create_dense(void) const -> Matrix<DefDense, T, M, M> {
    return Matrix<DefDense, T, M, M>(
        Base::Matrix::output_dense_matrix(this->matrix));
  }

  inline auto transpose(void) const -> Matrix<DefDiag, T, M> { return *this; }

  inline auto create_complex(void) const -> Matrix<DefDiag, Complex<T>, M> {

    return Matrix<DefDiag, Complex<T>, M>(
        Base::Matrix::convert_matrix_real_to_complex(this->matrix));
  }

  inline auto real(void) const -> Matrix<DefDiag, Value_Type, M> {
    return Matrix<DefDiag, Value_Type, M>(
        ComplexOperation::GetRealFromComplexDiagMatrix<
            Value_Type, T, M, IS_COMPLEX>::get(this->matrix));
  }

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

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
class Matrix<DefSparse, T, M, N, SparseAvailable> {
public:
  /* Type */
  using Value_Type = typename UnderlyingType<T>::Type;
  using Value_Complex_Type = T;
  using Matrix_Type = DefSparse;
  using SparseAvailable_Type = SparseAvailable;

private:
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

  /* Function */
  inline auto create_dense(void) const -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>(
        Base::Matrix::output_dense_matrix(this->matrix));
  }

  template <std::size_t COL, std::size_t ROW> inline T get() const {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    return Base::Matrix::get_sparse_matrix_value<COL, ROW>(this->matrix);
  }

  template <std::size_t ELEMENT> inline T get() const {
    static_assert(ELEMENT < NumberOfValues,
                  "ELEMENT must be the same or less than the number "
                  "of elements of Sparse Matrix.");

    return Base::Matrix::get_sparse_matrix_element_value<ELEMENT>(this->matrix);
  }

  template <std::size_t COL, std::size_t ROW> inline void set(const T &value) {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    Base::Matrix::set_sparse_matrix_value<COL, ROW>(this->matrix, value);
  }

  template <std::size_t ELEMENT> inline void set(const T &value) {
    static_assert(ELEMENT < NumberOfValues,
                  "ELEMENT must be the same or less than the number "
                  "of elements of Sparse Matrix.");

    Base::Matrix::set_sparse_matrix_element_value<ELEMENT>(this->matrix, value);
  }

  constexpr std::size_t rows() const { return ROWS; }

  constexpr std::size_t cols() const { return COLS; }

  T &operator()(std::size_t value_index) {
    if (value_index >= this->matrix.values.size()) {
      value_index = this->matrix.values.size() - 1;
    }

    return this->matrix[value_index];
  }

  const T &operator()(std::size_t value_index) const {
    if (value_index >= this->matrix.values.size()) {
      value_index = this->matrix.values.size() - 1;
    }

    return this->matrix[value_index];
  }

  inline auto transpose(void) const
      -> Matrix<DefSparse, T, N, M, SparseAvailableTranspose<SparseAvailable>> {

    return Matrix<DefSparse, T, N, M,
                  SparseAvailableTranspose<SparseAvailable>>(
        Base::Matrix::output_matrix_transpose(this->matrix));
  }

  inline auto create_complex(void) const
      -> Matrix<DefSparse, Complex<T>, M, N, SparseAvailable> {

    return Matrix<DefSparse, Complex<T>, M, N, SparseAvailable>(
        Base::Matrix::convert_matrix_real_to_complex(this->matrix));
  }

  inline auto real(void) const
      -> Matrix<DefSparse, Value_Type, M, N, SparseAvailable> {
    return Matrix<DefSparse, Value_Type, M, N, SparseAvailable>(
        ComplexOperation::GetRealFromComplexSparseMatrix<
            Value_Type, T, M, N, SparseAvailable,
            IS_COMPLEX>::get(this->matrix));
  }

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
template <typename T, std::size_t M, std::size_t N>
inline auto operator+(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
inline auto operator+(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator+(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
inline auto operator+(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M>
inline auto operator+(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefDiag, T, M> &B) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(A.matrix + B.matrix));
}

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

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator+(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

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
template <typename T, std::size_t M, std::size_t N>
inline auto operator-(const Matrix<DefDense, T, M, N> &A)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(-A.matrix));
}

template <typename T, std::size_t M>
inline auto operator-(const Matrix<DefDiag, T, M> &A) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(-A.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator-(const Matrix<DefSparse, T, M, N, SparseAvailable> &A)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {

  return Matrix<DefSparse, T, M, N, SparseAvailable>(std::move(-A.matrix));
}

template <typename T, std::size_t M, std::size_t N>
inline auto operator-(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
inline auto operator-(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator-(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
inline auto operator-(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M>
inline auto operator-(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefDiag, T, M> &B) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(A.matrix - B.matrix));
}

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

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator-(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

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
template <typename T, std::size_t M, std::size_t N>
inline auto operator*(const T &a, const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(a * B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
inline auto operator*(const Matrix<DefDense, T, M, N> &B, const T &a)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(B.matrix * a));
}

template <typename T, std::size_t M>
inline auto operator*(const T &a, const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(a * B.matrix));
}

template <typename T, std::size_t M>
inline auto operator*(const Matrix<DefDiag, T, M> &B, const T &a)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(B.matrix * a));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator*(const T &a,
                      const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {

  return Matrix<DefSparse, T, M, N, SparseAvailable>(std::move(a * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator*(const Matrix<DefSparse, T, M, N, SparseAvailable> &B,
                      const T &a)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {

  return Matrix<DefSparse, T, M, N, SparseAvailable>(std::move(B.matrix * a));
}

/* Matrix Multiply Matrix */
template <typename T, std::size_t M, std::size_t N, std::size_t K>
inline auto operator*(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
inline auto operator*(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefDiag, T, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable>
inline auto operator*(const Matrix<DefDense, T, M, N> &A,
                      const Matrix<DefSparse, T, N, K, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
inline auto operator*(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M>
inline auto operator*(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefDiag, T, M> &B) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator*(const Matrix<DefDiag, T, M> &A,
                      const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {

  return Matrix<DefSparse, T, M, N, SparseAvailable>(
      std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable>
inline auto operator*(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
                      const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto operator*(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
                      const Matrix<DefDiag, T, N> &B)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {

  return Matrix<DefSparse, T, M, N, SparseAvailable>(
      std::move(A.matrix * B.matrix));
}

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
