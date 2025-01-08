#ifndef PYTHON_NUMPY_BASE_HPP
#define PYTHON_NUMPY_BASE_HPP

#include "base_matrix.hpp"
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

    return this->matrix.data[row][col];
  }

  const T &operator()(std::size_t col, std::size_t row) const {
    if (col >= M) {
      col = M - 1;
    }
    if (row >= N) {
      row = N - 1;
    }

    return this->matrix.data[row][col];
  }

  static inline auto zeros(void) -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>();
  }

  static inline auto ones(void) -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>(Base::Matrix::Matrix<T, M, N>::ones());
  }

  inline auto transpose(void) -> Matrix<DefDense, T, N, M> {
    return Matrix<DefDense, T, N, M>(
        Base::Matrix::output_matrix_transpose(this->matrix));
  }

  inline auto create_complex(void)
      -> Matrix<DefDense, Base::Matrix::Complex<T>, M, N> {

    Matrix<DefDense, Base::Matrix::Complex<T>, M, N> Complex_matrix(
        Base::Matrix::convert_matrix_real_to_complex(this->matrix));

    return Complex_matrix;
  }

  /* Variable */
  static constexpr std::size_t ROWS = N;
  static constexpr std::size_t COLS = M;

  Base::Matrix::Matrix<T, M, N> matrix;
};

template <typename T, std::size_t M> class Matrix<DefDiag, T, M> {
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

  T &operator()(std::size_t col) {
    if (col >= M) {
      col = M - 1;
    }
    return this->matrix.data[col];
  }

  const T &operator()(std::size_t col, std::size_t row) const {
    if (col >= M) {
      col = M - 1;
    }
    return this->matrix.data[col];
  }

  static inline auto identity(void) -> Matrix<DefDiag, T, M> {
    return Matrix<DefDiag, T, M>(Base::Matrix::DiagMatrix<T, M>::identity());
  }

  inline auto create_dense(void) -> Matrix<DefDense, T, M, M> {
    return Matrix<DefDense, T, M, M>(
        Base::Matrix::output_dense_matrix(this->matrix));
  }

  inline auto transpose(void) -> Matrix<DefDiag, T, M> { return *this; }

  /* Variable */
  static constexpr std::size_t ROWS = M;
  static constexpr std::size_t COLS = M;

  Base::Matrix::DiagMatrix<T, M> matrix;
};

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
class Matrix<DefSparse, T, M, N, SparseAvailable> {
public:
  /* Constructor */
  Matrix() {}

  Matrix(const std::initializer_list<T> &values) : matrix(values) {}

  Matrix(Base::Matrix::CompiledSparseMatrix<
         T, M, N, RowIndicesFromSparseAvailable<SparseAvailable>,
         RowPointersFromSparseAvailable<SparseAvailable>> &input)
      : matrix(input) {}

  Matrix(Base::Matrix::CompiledSparseMatrix<
         T, M, N, RowIndicesFromSparseAvailable<SparseAvailable>,
         RowPointersFromSparseAvailable<SparseAvailable>> &&input) noexcept
      : matrix(std::move(input)) {}

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
  inline auto create_dense(void) -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>(
        Base::Matrix::output_dense_matrix(this->matrix));
  }

  template <std::size_t COL, std::size_t ROW> inline T get() const {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    return Base::Matrix::get_sparse_matrix_value<COL, ROW>(this->matrix);
  }

  template <std::size_t COL, std::size_t ROW> inline void set(const T &value) {
    static_assert(COL < M, "Column Index is out of range.");
    static_assert(ROW < N, "Row Index is out of range.");

    Base::Matrix::set_sparse_matrix_value<COL, ROW>(this->matrix, value);
  }

  constexpr std::size_t rows() const { return ROWS; }

  constexpr std::size_t cols() const { return COLS; }

  T &operator()(std::size_t value_index) {
    if (value_index >= this->matrix.values.size()) {
      value_index = this->matrix.values.size() - 1;
    }

    return this->matrix.values[value_index];
  }

  const T &operator()(std::size_t value_index) const {
    if (value_index >= this->matrix.values.size()) {
      value_index = this->matrix.values.size() - 1;
    }

    return this->matrix.values[value_index];
  }

  inline auto transpose(void) -> Matrix<DefDense, T, N, M> {
    return Matrix<DefDense, T, N, M>(
        Base::Matrix::output_matrix_transpose(this->matrix));
  }

  /* Variable */
  static constexpr std::size_t ROWS = N;
  static constexpr std::size_t COLS = M;

  Base::Matrix::CompiledSparseMatrix<
      T, M, N, RowIndicesFromSparseAvailable<SparseAvailable>,
      RowPointersFromSparseAvailable<SparseAvailable>>
      matrix;
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
              CreateSparseAvailableFromIndicesAndPointers<
                  N,
                  RowIndicesFromSparseAvailable<MatrixAddSubSparseAvailable<
                      SparseAvailable, DiagAvailable<M>>>,
                  RowPointersFromSparseAvailable<MatrixAddSubSparseAvailable<
                      SparseAvailable, DiagAvailable<M>>>>> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<
      DefSparse, T, M, N,
      CreateSparseAvailableFromIndicesAndPointers<
          N,
          RowIndicesFromSparseAvailable<
              MatrixAddSubSparseAvailable<SparseAvailable, DiagAvailable<M>>>,
          RowPointersFromSparseAvailable<
              MatrixAddSubSparseAvailable<SparseAvailable, DiagAvailable<M>>>>>(
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
              CreateSparseAvailableFromIndicesAndPointers<
                  N,
                  RowIndicesFromSparseAvailable<MatrixAddSubSparseAvailable<
                      SparseAvailable, DiagAvailable<M>>>,
                  RowPointersFromSparseAvailable<MatrixAddSubSparseAvailable<
                      SparseAvailable, DiagAvailable<M>>>>> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<
      DefSparse, T, M, N,
      CreateSparseAvailableFromIndicesAndPointers<
          N,
          RowIndicesFromSparseAvailable<
              MatrixAddSubSparseAvailable<SparseAvailable, DiagAvailable<M>>>,
          RowPointersFromSparseAvailable<
              MatrixAddSubSparseAvailable<SparseAvailable, DiagAvailable<M>>>>>(
      std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A,
          typename SparseAvailable_B>
inline auto operator+(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                      const Matrix<DefSparse, T, M, N, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, M, N,
              CreateSparseAvailableFromIndicesAndPointers<
                  N,
                  RowIndicesFromSparseAvailable<MatrixAddSubSparseAvailable<
                      SparseAvailable_A, SparseAvailable_B>>,
                  RowPointersFromSparseAvailable<MatrixAddSubSparseAvailable<
                      SparseAvailable_A, SparseAvailable_B>>>> {

  return Matrix<DefSparse, T, M, N,
                CreateSparseAvailableFromIndicesAndPointers<
                    N,
                    RowIndicesFromSparseAvailable<MatrixAddSubSparseAvailable<
                        SparseAvailable_A, SparseAvailable_B>>,
                    RowPointersFromSparseAvailable<MatrixAddSubSparseAvailable<
                        SparseAvailable_A, SparseAvailable_B>>>>(
      std::move(A.matrix + B.matrix));
}

/* Matrix Subtraction */
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
              CreateSparseAvailableFromIndicesAndPointers<
                  N,
                  RowIndicesFromSparseAvailable<MatrixAddSubSparseAvailable<
                      SparseAvailable, DiagAvailable<M>>>,
                  RowPointersFromSparseAvailable<MatrixAddSubSparseAvailable<
                      SparseAvailable, DiagAvailable<M>>>>> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<
      DefSparse, T, M, N,
      CreateSparseAvailableFromIndicesAndPointers<
          N,
          RowIndicesFromSparseAvailable<
              MatrixAddSubSparseAvailable<SparseAvailable, DiagAvailable<M>>>,
          RowPointersFromSparseAvailable<
              MatrixAddSubSparseAvailable<SparseAvailable, DiagAvailable<M>>>>>(
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
              CreateSparseAvailableFromIndicesAndPointers<
                  N,
                  RowIndicesFromSparseAvailable<MatrixAddSubSparseAvailable<
                      SparseAvailable, DiagAvailable<M>>>,
                  RowPointersFromSparseAvailable<MatrixAddSubSparseAvailable<
                      SparseAvailable, DiagAvailable<M>>>>> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<
      DefSparse, T, M, N,
      CreateSparseAvailableFromIndicesAndPointers<
          N,
          RowIndicesFromSparseAvailable<
              MatrixAddSubSparseAvailable<SparseAvailable, DiagAvailable<M>>>,
          RowPointersFromSparseAvailable<
              MatrixAddSubSparseAvailable<SparseAvailable, DiagAvailable<M>>>>>(
      std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A,
          typename SparseAvailable_B>
inline auto operator-(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                      const Matrix<DefSparse, T, M, N, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, M, N,
              CreateSparseAvailableFromIndicesAndPointers<
                  N,
                  RowIndicesFromSparseAvailable<MatrixAddSubSparseAvailable<
                      SparseAvailable_A, SparseAvailable_B>>,
                  RowPointersFromSparseAvailable<MatrixAddSubSparseAvailable<
                      SparseAvailable_A, SparseAvailable_B>>>> {

  return Matrix<DefSparse, T, M, N,
                CreateSparseAvailableFromIndicesAndPointers<
                    N,
                    RowIndicesFromSparseAvailable<MatrixAddSubSparseAvailable<
                        SparseAvailable_A, SparseAvailable_B>>,
                    RowPointersFromSparseAvailable<MatrixAddSubSparseAvailable<
                        SparseAvailable_A, SparseAvailable_B>>>>(
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

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_BASE_HPP
