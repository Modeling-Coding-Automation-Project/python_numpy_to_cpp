#ifndef PYTHON_NUMPY_BASE_HPP
#define PYTHON_NUMPY_BASE_HPP

#include "base_matrix.hpp"
#include <cstddef>
#include <initializer_list>
#include <utility>

namespace PythonNumpy {

/* Compiled Sparse Matrix Templates */
template <std::size_t... Sizes>
using RowIndices = Base::Matrix::RowIndices<Sizes...>;

template <std::size_t... Sizes>
using RowPointers = Base::Matrix::RowPointers<Sizes...>;

template <bool... Flags>
using ColumnAvailable = Base::Matrix::ColumnAvailable<Flags...>;

template <typename... Columns>
using SparseAvailable = Base::Matrix::SparseAvailable<Columns...>;

template <std::size_t M, std::size_t N>
using DenseAvailable = Base::Matrix::DenseAvailable<M, N>;

template <std::size_t M> using DiagAvailable = Base::Matrix::DiagAvailable<M>;

template <typename SparseAvailable>
using RowIndicesFromSparseAvailable =
    Base::Matrix::RowIndicesFromSparseAvailable<SparseAvailable>;

template <typename SparseAvailable>
using RowPointersFromSparseAvailable =
    Base::Matrix::RowPointersFromSparseAvailable<SparseAvailable>;

template <std::size_t N, typename RowIndices, typename RowPointers>
using CreateSparseAvailableFromIndicesAndPointers =
    Base::Matrix::CreateSparseAvailableFromIndicesAndPointers<N, RowIndices,
                                                              RowPointers>;

template <typename SparseAvailable_A, typename SparseAvailable_B>
using ConcatenateSparseAvailableVertically =
    Base::Matrix::ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                       SparseAvailable_B>;

template <typename SparseAvailable_A, typename SparseAvailable_B>
using ConcatenateSparseAvailableHorizontally =
    Base::Matrix::ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                         SparseAvailable_B>;

using SparseAvailable_NoUse =
    SparseAvailable<ColumnAvailable<true, false>, ColumnAvailable<false, true>>;

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
  std::size_t rows() const { return N; }

  std::size_t cols() const { return M; }

  static auto zeros(void) -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>();
  }

  static auto ones(void) -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>(Base::Matrix::Matrix<T, M, N>::ones());
  }

  auto transpose(void) -> Matrix<DefDense, T, N, M> {
    return Matrix<DefDense, T, N, M>(this->matrix.transpose());
  }

  auto create_complex(void)
      -> Matrix<DefDense, Base::Matrix::Complex<T>, M, N> {

    Matrix<DefDense, Base::Matrix::Complex<T>, M, N> Complex_matrix(
        Base::Matrix::convert_matrix_real_to_complex(this->matrix));

    return Complex_matrix;
  }

  /* Variable */
  Base::Matrix::Matrix<T, M, N> matrix;
};

template <typename T, std::size_t M> class Matrix<DefDiag, T, M> {
public:
  /* Constructor */
  Matrix() {}

  Matrix(const std::initializer_list<T> &input) : matrix(input) {}

  Matrix(T input[M]) : matrix(input) {}

  Matrix(const Base::Matrix::Matrix<T, M, 1> &input) : matrix(input.data[0]) {}

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
  std::size_t rows() const { return M; }

  std::size_t cols() const { return M; }

  static auto identity(void) -> Matrix<DefDiag, T, M> {
    return Matrix<DefDiag, T, M>(Base::Matrix::DiagMatrix<T, M>::identity());
  }

  auto create_dense(void) -> Matrix<DefDense, T, M, M> {
    return Matrix<DefDense, T, M, M>(this->matrix.create_dense());
  }

  auto transpose(void) -> Matrix<DefDiag, T, M> { return *this; }

  /* Variable */
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
  auto create_dense(void) -> Matrix<DefDense, T, M, N> {
    return Matrix<DefDense, T, M, N>(this->matrix.create_dense());
  }

  std::size_t rows() const { return N; }

  std::size_t cols() const { return M; }

  auto transpose(void) -> Matrix<DefDense, T, N, M> {
    return Matrix<DefDense, T, N, M>(this->matrix.transpose());
  }

  /* Variable */
  Base::Matrix::CompiledSparseMatrix<
      T, M, N, RowIndicesFromSparseAvailable<SparseAvailable>,
      RowPointersFromSparseAvailable<SparseAvailable>>
      matrix;
};

/* Matrix Addition */
template <typename T, std::size_t M, std::size_t N>
auto operator+(const Matrix<DefDense, T, M, N> &A,
               const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
auto operator+(const Matrix<DefDense, T, M, N> &A,
               const Matrix<DefDiag, T, M> &B) -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator+(const Matrix<DefDense, T, M, N> &A,
               const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
auto operator+(const Matrix<DefDiag, T, M> &A,
               const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M>
auto operator+(const Matrix<DefDiag, T, M> &A, const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator+(const Matrix<DefDiag, T, M> &A,
               const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator+(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
               const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator+(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
               const Matrix<DefDiag, T, M> &B) -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator+(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
               const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

/* Matrix Subtraction */
template <typename T, std::size_t M, std::size_t N>
auto operator-(const Matrix<DefDense, T, M, N> &A,
               const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
auto operator-(const Matrix<DefDense, T, M, N> &A,
               const Matrix<DefDiag, T, M> &B) -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator-(const Matrix<DefDense, T, M, N> &A,
               const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
auto operator-(const Matrix<DefDiag, T, M> &A,
               const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M>
auto operator-(const Matrix<DefDiag, T, M> &A, const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator-(const Matrix<DefDiag, T, M> &A,
               const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator-(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
               const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator-(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
               const Matrix<DefDiag, T, M> &B) -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator-(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
               const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

/* Matrix Multiply Scalar */
template <typename T, std::size_t M, std::size_t N>
auto operator*(const T &a, const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(a * B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
auto operator*(const Matrix<DefDense, T, M, N> &B, const T &a)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(B.matrix * a));
}

template <typename T, std::size_t M>
auto operator*(const T &a, const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(a * B.matrix));
}

template <typename T, std::size_t M>
auto operator*(const Matrix<DefDiag, T, M> &B, const T &a)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(B.matrix * a));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator*(const T &a, const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {

  return Matrix<DefSparse, T, M, N, SparseAvailable>(std::move(a * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator*(const Matrix<DefSparse, T, M, N, SparseAvailable> &B, const T &a)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {

  return Matrix<DefSparse, T, M, N, SparseAvailable>(std::move(B.matrix * a));
}

/* Matrix Multiply Matrix */
template <typename T, std::size_t M, std::size_t N, std::size_t K>
auto operator*(const Matrix<DefDense, T, M, N> &A,
               const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
auto operator*(const Matrix<DefDense, T, M, N> &A,
               const Matrix<DefDiag, T, N> &B) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable>
auto operator*(const Matrix<DefDense, T, M, N> &A,
               const Matrix<DefSparse, T, N, K, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N>
auto operator*(const Matrix<DefDiag, T, M> &A,
               const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M>
auto operator*(const Matrix<DefDiag, T, M> &A, const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator*(const Matrix<DefDiag, T, M> &A,
               const Matrix<DefSparse, T, M, N, SparseAvailable> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable>
auto operator*(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
               const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
auto operator*(const Matrix<DefSparse, T, M, N, SparseAvailable> &A,
               const Matrix<DefDiag, T, N> &B) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename SparseAvailable_A, typename SparseAvailable_B>
auto operator*(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
               const Matrix<DefSparse, T, N, K, SparseAvailable_B> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(std::move(A.matrix * B.matrix));
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_BASE_HPP
