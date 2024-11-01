#ifndef PYTHON_NUMPY_BASE_HPP
#define PYTHON_NUMPY_BASE_HPP

#include "base_matrix.hpp"
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <utility>

namespace PythonNumpy {

class DefDense {};

class DefDiag {};

class DefSparse {};

template <typename C, typename T, std::size_t M, std::size_t N = 1,
          std::size_t V = 1>
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

    Matrix<DefDense, Base::Matrix::Complex<T>, M, N> Complex_matrix;

    Base::Matrix::copy_matrix_real_to_complex(Complex_matrix.matrix,
                                              this->matrix);
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

  auto transpose(void) -> Matrix<DefDiag, T, M> { return *this; }

  /* Variable */
  Base::Matrix::DiagMatrix<T, M> matrix;
};

template <typename T, std::size_t M, std::size_t N, std::size_t V>
class Matrix<DefSparse, T, M, N, V> {
public:
  /* Constructor */
  Matrix() {}

  Matrix(const std::initializer_list<T> &values,
         const std::initializer_list<std::size_t> &row_indices,
         const std::initializer_list<std::size_t> &row_pointers)
      : matrix(values, row_indices, row_pointers) {}

  Matrix(Base::Matrix::SparseMatrix<T, M, N, V> &input) : matrix(input) {}

  Matrix(Base::Matrix::SparseMatrix<T, M, N, V> &&input) noexcept
      : matrix(std::move(input)) {}

  Matrix(Base::Matrix::Matrix<T, M, N> &input) : matrix(input) {}

  /* Copy Constructor */
  Matrix(const Matrix<DefSparse, T, M, N, V> &input) : matrix(input.matrix) {}

  Matrix<DefSparse, T, M, N, V> &
  operator=(const Matrix<DefSparse, T, M, N, V> &input) {
    if (this != &input) {
      this->matrix = input.matrix;
    }
    return *this;
  }

  /* Move Constructor */
  Matrix(Matrix<DefSparse, T, M, N, V> &&input) noexcept
      : matrix(std::move(input.matrix)) {}

  Matrix<DefSparse, T, M, N, V> &
  operator=(Matrix<DefSparse, T, M, N, V> &&input) noexcept {
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
  Base::Matrix::SparseMatrix<T, M, N, V> matrix;
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

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator+(const Matrix<DefDense, T, M, N> &A,
               const Matrix<DefSparse, T, M, N, V> &B)
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

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator+(const Matrix<DefDiag, T, M> &A,
               const Matrix<DefSparse, T, M, N, V> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator+(const Matrix<DefSparse, T, M, N, V> &A,
               const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator+(const Matrix<DefSparse, T, M, N, V> &A,
               const Matrix<DefDiag, T, M> &B) -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix + B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator+(const Matrix<DefSparse, T, M, N, V> &A,
               const Matrix<DefSparse, T, M, N, V> &B)
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

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator-(const Matrix<DefDense, T, M, N> &A,
               const Matrix<DefSparse, T, M, N, V> &B)
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

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator-(const Matrix<DefDiag, T, M> &A,
               const Matrix<DefSparse, T, M, N, V> &B)
    -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator-(const Matrix<DefSparse, T, M, N, V> &A,
               const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator-(const Matrix<DefSparse, T, M, N, V> &A,
               const Matrix<DefDiag, T, M> &B) -> Matrix<DefDense, T, M, N> {
  static_assert(M == N, "Argument is not square matrix.");

  return Matrix<DefDense, T, M, N>(std::move(A.matrix - B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator-(const Matrix<DefSparse, T, M, N, V> &A,
               const Matrix<DefSparse, T, M, N, V> &B)
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

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator*(const T &a, const Matrix<DefSparse, T, M, N, V> &B)
    -> Matrix<DefSparse, T, M, N, V> {

  return Matrix<DefSparse, T, M, N, V>(std::move(a * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator*(const Matrix<DefSparse, T, M, N, V> &B, const T &a)
    -> Matrix<DefSparse, T, M, N, V> {

  return Matrix<DefSparse, T, M, N, V>(std::move(B.matrix * a));
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
          std::size_t V>
auto operator*(const Matrix<DefDense, T, M, N> &A,
               const Matrix<DefSparse, T, N, K, V> &B)
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

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator*(const Matrix<DefDiag, T, M> &A,
               const Matrix<DefSparse, T, M, N, V> &B)
    -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V>
auto operator*(const Matrix<DefSparse, T, M, N, V> &A,
               const Matrix<DefDense, T, N, K> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
auto operator*(const Matrix<DefSparse, T, M, N, V> &A,
               const Matrix<DefDiag, T, N> &B) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>(std::move(A.matrix * B.matrix));
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          std::size_t V, std::size_t W>
auto operator*(const Matrix<DefSparse, T, M, N, V> &A,
               const Matrix<DefSparse, T, N, K, W> &B)
    -> Matrix<DefDense, T, M, K> {

  return Matrix<DefDense, T, M, K>(std::move(A.matrix * B.matrix));
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_BASE_HPP
