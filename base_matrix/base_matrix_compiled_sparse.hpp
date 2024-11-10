#ifndef BASE_MATRIX_COMPILED_SPARSE_HPP
#define BASE_MATRIX_COMPILED_SPARSE_HPP

#include "base_matrix_diagonal.hpp"
#include "base_matrix_macros.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

const double COMPILED_SPARSE_MATRIX_JUDGE_ZERO_LIMIT_VALUE = 1.0e-20;

template <std::size_t... Sizes> struct size_list_array {
  static constexpr std::size_t size = sizeof...(Sizes);
  static constexpr std::size_t value[size] = {Sizes...};
};

template <std::size_t... Sizes>
constexpr std::size_t
    size_list_array<Sizes...>::value[size_list_array<Sizes...>::size];

template <typename Array> class CompiledSparseMatrixList {
public:
  typedef const std::size_t *size_list_type;
  static constexpr size_list_type size_list = Array::value;
  static constexpr std::size_t size = Array::size;
};

template <std::size_t... Sizes>
using RowIndices = CompiledSparseMatrixList<size_list_array<Sizes...>>;

template <std::size_t... Sizes>
using RowPointers = CompiledSparseMatrixList<size_list_array<Sizes...>>;

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
class CompiledSparseMatrix {
public:
#ifdef BASE_MATRIX_USE_STD_VECTOR

  CompiledSparseMatrix() : values(RowIndices::size, static_cast<T>(0)) {}

  CompiledSparseMatrix(const std::initializer_list<T> &values)
      : values(values) {}

  CompiledSparseMatrix(const std::vector<T> &values) : values(values) {}

#else

  CompiledSparseMatrix() : values{} {}

  CompiledSparseMatrix(const std::initializer_list<T> &values) : values{} {

    std::copy(values.begin(), values.end(), this->values.begin());
  }

  CompiledSparseMatrix(const std::array<T, RowIndices::size> &values)
      : values(values) {}

  CompiledSparseMatrix(const std::vector<T> &values) : values{} {

    std::copy(values.begin(), values.end(), this->values.begin());
  }

#endif

  /* Copy Constructor */
  CompiledSparseMatrix(
      const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &other)
      : values(other.values) {}

  CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &operator=(
      const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &other) {
    if (this != &other) {
      this->values = other.values;
    }
    return *this;
  }

  /* Move Constructor */
  CompiledSparseMatrix(
      CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &&other) noexcept
      : values(std::move(other.values)) {}

  CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &operator=(
      CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &&other) noexcept {
    if (this != &other) {
      this->values = std::move(other.values);
    }
    return *this;
  }

  /* Function */
  Matrix<T, M, N> create_dense() const { return output_dense_matrix(*this); }

  /* Variable */
#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values;
#else
  std::array<T, RowIndices::size> values;
  std::array<T, RowPointers::size_list[M]> test_pointers;
#endif
};

/* Create dense matrix */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K>
struct OutputDenseMatrixCore {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {
    result(J, RowIndices::size_list[K]) = mat.values[K];
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t L, std::size_t P>
struct OutputDenseMatrixList {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {
    OutputDenseMatrixCore<T, M, N, RowIndices, RowPointers, M - 1 - P,
                          L>::compute(mat, result);

    (RowPointers::size_list[M - 1 - P] == L)
        ? OutputDenseMatrixList<T, M, N, RowIndices, RowPointers, L - 1,
                                P + 1>::compute(mat, result)
        : OutputDenseMatrixList<T, M, N, RowIndices, RowPointers, L - 1,
                                P>::compute(mat, result);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t P>
struct OutputDenseMatrixList<T, M, N, RowIndices, RowPointers, 0, P> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {
    OutputDenseMatrixCore<T, M, N, RowIndices, RowPointers, M - 1 - P,
                          0>::compute(mat, result);
  }
};

#define BASE_MATRIX_COMPILED_SPARSE_MATRIX_CREATE_DENSE(                       \
    T, M, N, RowIndices, RowPointers, mat, result)                             \
  OutputDenseMatrixList<T, M, N, RowIndices, RowPointers, M + 1, 0>::compute(  \
      mat, result);

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline Matrix<T, M, N> output_dense_matrix(
    const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; j++) {
    for (std::size_t k = RowPointers::size_list[j];
         k < RowPointers::size_list[j + 1]; k++) {
      result(j, RowIndices::size_list[k]) = mat.values[k];
    }
  }

#else

  BASE_MATRIX_COMPILED_SPARSE_MATRIX_CREATE_DENSE(T, M, N, RowIndices,
                                                  RowPointers, mat, result)

#endif

  return result;
}

/* Sparse Matrix multiply Matrix */
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t Start, std::size_t End>
struct MatrixMultiplicationLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, T &sum) {
    sum += A.values[Start] * B(RowIndices_A::size_list[Start], I);
    MatrixMultiplicationLoop<T, M, N, K, RowIndices_A, RowPointers_A, J, I,
                             Start + 1, End>::compute(A, B, sum);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t End>
struct MatrixMultiplicationLoop<T, M, N, K, RowIndices_A, RowPointers_A, J, I,
                                End, End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, T &sum) {
    // End of loop, do nothing
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct MatrixMultiplicationCore {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    T sum = static_cast<T>(0);
    MatrixMultiplicationLoop<T, M, N, K, RowIndices_A, RowPointers_A, J, I,
                             RowPointers_A::size_list[J],
                             RowPointers_A::size_list[J + 1]>::compute(A, B,
                                                                       sum);
    Y(J, I) = sum;
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct MatrixMultiplicationList {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    MatrixMultiplicationCore<T, M, N, K, RowIndices_A, RowPointers_A, J,
                             I>::compute(A, B, Y);
    MatrixMultiplicationList<T, M, N, K, RowIndices_A, RowPointers_A, J - 1,
                             I>::compute(A, B, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct MatrixMultiplicationList<T, M, N, K, RowIndices_A, RowPointers_A, 0, I> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    MatrixMultiplicationCore<T, M, N, K, RowIndices_A, RowPointers_A, 0,
                             I>::compute(A, B, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct MatrixMultiplicationColumn {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    MatrixMultiplicationList<T, M, N, K, RowIndices_A, RowPointers_A, M - 1,
                             I>::compute(A, B, Y);
    MatrixMultiplicationColumn<T, M, N, K, RowIndices_A, RowPointers_A,
                               I - 1>::compute(A, B, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A>
struct MatrixMultiplicationColumn<T, M, N, K, RowIndices_A, RowPointers_A, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    MatrixMultiplicationList<T, M, N, K, RowIndices_A, RowPointers_A, M - 1,
                             0>::compute(A, B, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
Matrix<T, M, K>
operator*(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B) {
  Matrix<T, M, K> Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < K; i++) {
    for (std::size_t j = 0; j < M; j++) {
      T sum = static_cast<T>(0);
      for (std::size_t k = RowPointers_A::size_list[j];
           k < RowPointers_A::size_list[j + 1]; k++) {
        sum += A.values[k] * B(RowIndices_A::size_list[k], i);
      }
      Y(j, i) = sum;
    }
  }

#else

  MatrixMultiplicationColumn<T, M, N, K, RowIndices_A, RowPointers_A,
                             K - 1>::compute(A, B, Y);

#endif

  return Y;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_COMPILED_SPARSE_HPP
