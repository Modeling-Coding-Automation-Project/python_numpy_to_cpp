#ifndef BASE_MATRIX_COMPILED_SPARSE_OPERATION_HPP
#define BASE_MATRIX_COMPILED_SPARSE_OPERATION_HPP

#include "base_matrix_compiled_sparse.hpp"
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

/* Sparse Matrix multiply Dense Matrix */
// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t Start, std::size_t End>
struct SparseMatrixMultiplyDenseLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, T &sum) {
    sum += A.values[Start] * B(RowIndices_A::size_list[Start], I);
    SparseMatrixMultiplyDenseLoop<T, M, N, K, RowIndices_A, RowPointers_A, J, I,
                                  Start + 1, End>::compute(A, B, sum);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t End>
struct SparseMatrixMultiplyDenseLoop<T, M, N, K, RowIndices_A, RowPointers_A, J,
                                     I, End, End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, T &sum) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(sum);
    // End of loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct SparseMatrixMultiplyDenseCore {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    T sum = static_cast<T>(0);
    SparseMatrixMultiplyDenseLoop<
        T, M, N, K, RowIndices_A, RowPointers_A, J, I,
        RowPointers_A::size_list[J],
        RowPointers_A::size_list[J + 1]>::compute(A, B, sum);
    Y(J, I) = sum;
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct SparseMatrixMultiplyDenseList {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    SparseMatrixMultiplyDenseCore<T, M, N, K, RowIndices_A, RowPointers_A, J,
                                  I>::compute(A, B, Y);
    SparseMatrixMultiplyDenseList<T, M, N, K, RowIndices_A, RowPointers_A,
                                  J - 1, I>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct SparseMatrixMultiplyDenseList<T, M, N, K, RowIndices_A, RowPointers_A, 0,
                                     I> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    SparseMatrixMultiplyDenseCore<T, M, N, K, RowIndices_A, RowPointers_A, 0,
                                  I>::compute(A, B, Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct SparseMatrixMultiplyDenseColumn {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    SparseMatrixMultiplyDenseList<T, M, N, K, RowIndices_A, RowPointers_A,
                                  M - 1, I>::compute(A, B, Y);
    SparseMatrixMultiplyDenseColumn<T, M, N, K, RowIndices_A, RowPointers_A,
                                    I - 1>::compute(A, B, Y);
  }
};

// End of column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A>
struct SparseMatrixMultiplyDenseColumn<T, M, N, K, RowIndices_A, RowPointers_A,
                                       0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    SparseMatrixMultiplyDenseList<T, M, N, K, RowIndices_A, RowPointers_A,
                                  M - 1, 0>::compute(A, B, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
static inline void COMPILED_SPARSE_MATRIX_MULTIPLY_DENSE(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
  SparseMatrixMultiplyDenseColumn<T, M, N, K, RowIndices_A, RowPointers_A,
                                  K - 1>::compute(A, B, Y);
}

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

  COMPILED_SPARSE_MATRIX_MULTIPLY_DENSE<T, M, N, RowIndices_A, RowPointers_A,
                                        K>(A, B, Y);

#endif

  return Y;
}

/* Dense Matrix multiply Sparse Matrix */
// Core loop for multiplication
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t I, std::size_t Start, std::size_t End>
struct DenseMatrixMultiplySparseLoop {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    Y(I, RowIndices_B::size_list[Start]) += B.values[Start] * A(I, J);
    DenseMatrixMultiplySparseLoop<T, M, N, K, RowIndices_B, RowPointers_B, J, I,
                                  Start + 1, End>::compute(A, B, Y);
  }
};

// End of core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t I, std::size_t End>
struct DenseMatrixMultiplySparseLoop<T, M, N, K, RowIndices_B, RowPointers_B, J,
                                     I, End, End> {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t I>
struct DenseMatrixMultiplySparseCore {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DenseMatrixMultiplySparseLoop<T, M, N, K, RowIndices_B, RowPointers_B, J, I,
                                  RowPointers_B::size_list[J],
                                  RowPointers_B::size_list[J + 1]>::compute(A,
                                                                            B,
                                                                            Y);
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t I>
struct DenseMatrixMultiplySparseList {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DenseMatrixMultiplySparseCore<T, M, N, K, RowIndices_B, RowPointers_B, J,
                                  I>::compute(A, B, Y);
    DenseMatrixMultiplySparseList<T, M, N, K, RowIndices_B, RowPointers_B, J,
                                  I - 1>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J>
struct DenseMatrixMultiplySparseList<T, M, N, K, RowIndices_B, RowPointers_B, J,
                                     0> {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DenseMatrixMultiplySparseCore<T, M, N, K, RowIndices_B, RowPointers_B, J,
                                  0>::compute(A, B, Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J>
struct DenseMatrixMultiplySparseColumn {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DenseMatrixMultiplySparseList<T, M, N, K, RowIndices_B, RowPointers_B, J,
                                  N - 1>::compute(A, B, Y);
    DenseMatrixMultiplySparseColumn<T, M, N, K, RowIndices_B, RowPointers_B,
                                    J - 1>::compute(A, B, Y);
  }
};

// End of column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B>
struct DenseMatrixMultiplySparseColumn<T, M, N, K, RowIndices_B, RowPointers_B,
                                       0> {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DenseMatrixMultiplySparseList<T, M, N, K, RowIndices_B, RowPointers_B, 0,
                                  N - 1>::compute(A, B, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t K>
static inline void COMPILED_DENSE_MATRIX_MULTIPLY_SPARSE(
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
    Matrix<T, M, K> &Y) {
  DenseMatrixMultiplySparseColumn<T, M, N, K, RowIndices_B, RowPointers_B,
                                  M - 1>::compute(A, B, Y);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t K>
Matrix<T, M, K>
operator*(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, M, K> Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < N; j++) {
    for (std::size_t k = RowPointers_B::size_list[j];
         k < RowPointers_B::size_list[j + 1]; k++) {
      for (std::size_t i = 0; i < M; i++) {
        Y(i, RowIndices_B::size_list[k]) += B.values[k] * A(i, j);
      }
    }
  }

#else

  COMPILED_DENSE_MATRIX_MULTIPLY_SPARSE<T, M, N, RowIndices_B, RowPointers_B,
                                        K>(A, B, Y);

#endif

  return Y;
}

/* Sparse Matrix add Dense Matrix */
// Core loop for addition
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K,
          std::size_t Start, std::size_t End>
struct SparseMatrixAddDenseLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {
    Y(J, RowIndices_A::size_list[Start]) += A.values[Start];
    SparseMatrixAddDenseLoop<T, M, N, RowIndices_A, RowPointers_A, J, K,
                             Start + 1, End>::compute(A, Y);
  }
};

// End of core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K, std::size_t End>
struct SparseMatrixAddDenseLoop<T, M, N, RowIndices_A, RowPointers_A, J, K, End,
                                End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {
    static_cast<void>(A);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct SparseMatrixAddDenseRow {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {
    SparseMatrixAddDenseLoop<T, M, N, RowIndices_A, RowPointers_A, J, 0,
                             RowPointers_A::size_list[J],
                             RowPointers_A::size_list[J + 1]>::compute(A, Y);
    SparseMatrixAddDenseRow<T, M, N, RowIndices_A, RowPointers_A,
                            J - 1>::compute(A, Y);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct SparseMatrixAddDenseRow<T, M, N, RowIndices_A, RowPointers_A, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {
    SparseMatrixAddDenseLoop<T, M, N, RowIndices_A, RowPointers_A, 0, 0,
                             RowPointers_A::size_list[0],
                             RowPointers_A::size_list[1]>::compute(A, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
static inline void COMPILED_SPARSE_MATRIX_ADD_DENSE(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    Matrix<T, M, N> &Y) {
  SparseMatrixAddDenseRow<T, M, N, RowIndices_A, RowPointers_A, M - 1>::compute(
      A, Y);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
Matrix<T, M, N>
operator+(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, M, N> &B) {
  Matrix<T, M, N> Y = B;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::size_list[j];
         k < RowPointers_A::size_list[j + 1]; ++k) {
      Y(j, RowIndices_A::size_list[k]) += A.values[k];
    }
  }

#else

  COMPILED_SPARSE_MATRIX_ADD_DENSE<T, M, N, RowIndices_A, RowPointers_A>(A, Y);

#endif

  return Y;
}

/* Sparse Matrix Add Diag Matrix */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
Matrix<T, M, M>
operator+(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, M> &B) {
  Matrix<T, M, M> Y = B.create_dense();

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::size_list[j];
         k < RowPointers_A::size_list[j + 1]; ++k) {
      Y(j, RowIndices_A::size_list[k]) += A.values[k];
    }
  }

#else

  COMPILED_SPARSE_MATRIX_ADD_DENSE<T, M, N, RowIndices_A, RowPointers_A>(A, Y);

#endif

  return Y;
}

/* Diag Matrix Add Sparse Matrix */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
Matrix<T, M, M>
operator+(const DiagMatrix<T, M> &B,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
  Matrix<T, M, M> Y = B.create_dense();

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::size_list[j];
         k < RowPointers_A::size_list[j + 1]; ++k) {
      Y(j, RowIndices_A::size_list[k]) += A.values[k];
    }
  }

#else

  COMPILED_SPARSE_MATRIX_ADD_DENSE<T, M, N, RowIndices_A, RowPointers_A>(A, Y);

#endif

  return Y;
}

/* Sparse Matrix Add Sparse Matrix */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_B, typename RowPointers_B>
Matrix<T, M, N>
operator+(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, M, M> Y = B.create_dense();

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::size_list[j];
         k < RowPointers_A::size_list[j + 1]; ++k) {
      Y(j, RowIndices_A::size_list[k]) += A.values[k];
    }
  }

#else

  COMPILED_SPARSE_MATRIX_ADD_DENSE<T, M, N, RowIndices_A, RowPointers_A>(A, Y);

#endif

  return Y;
}

/* Sparse Matrix Subtract Dense Matrix */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
Matrix<T, M, N>
operator-(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, M, N> &B) {
  Matrix<T, M, N> Y = -B;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::size_list[j];
         k < RowPointers_A::size_list[j + 1]; ++k) {
      Y(j, RowIndices_A::size_list[k]) += A.values[k];
    }
  }

#else

  COMPILED_SPARSE_MATRIX_ADD_DENSE<T, M, N, RowIndices_A, RowPointers_A>(A, Y);

#endif

  return Y;
}

/* Sparse Matrix Subtract Diag Matrix */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
Matrix<T, M, M>
operator-(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, M> &B) {
  Matrix<T, M, M> Y = -(B.create_dense());

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::size_list[j];
         k < RowPointers_A::size_list[j + 1]; ++k) {
      Y(j, RowIndices_A::size_list[k]) += A.values[k];
    }
  }

#else

  COMPILED_SPARSE_MATRIX_ADD_DENSE<T, M, N, RowIndices_A, RowPointers_A>(A, Y);

#endif

  return Y;
}

/* Diag Matrix Subtract Sparse Matrix */
// Core loop for addition
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K,
          std::size_t Start, std::size_t End>
struct SparseMatrixSubDenseLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {
    Y(J, RowIndices_A::size_list[Start]) -= A.values[Start];
    SparseMatrixSubDenseLoop<T, M, N, RowIndices_A, RowPointers_A, J, K,
                             Start + 1, End>::compute(A, Y);
  }
};

// End of core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K, std::size_t End>
struct SparseMatrixSubDenseLoop<T, M, N, RowIndices_A, RowPointers_A, J, K, End,
                                End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {
    static_cast<void>(A);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct SparseMatrixSubDenseRow {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {
    SparseMatrixSubDenseLoop<T, M, N, RowIndices_A, RowPointers_A, J, 0,
                             RowPointers_A::size_list[J],
                             RowPointers_A::size_list[J + 1]>::compute(A, Y);
    SparseMatrixSubDenseRow<T, M, N, RowIndices_A, RowPointers_A,
                            J - 1>::compute(A, Y);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct SparseMatrixSubDenseRow<T, M, N, RowIndices_A, RowPointers_A, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          Matrix<T, M, N> &Y) {
    SparseMatrixSubDenseLoop<T, M, N, RowIndices_A, RowPointers_A, 0, 0,
                             RowPointers_A::size_list[0],
                             RowPointers_A::size_list[1]>::compute(A, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
static inline void COMPILED_SPARSE_MATRIX_SUB_DENSE(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    Matrix<T, M, N> &Y) {
  SparseMatrixSubDenseRow<T, M, N, RowIndices_A, RowPointers_A, M - 1>::compute(
      A, Y);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
Matrix<T, M, M>
operator-(const DiagMatrix<T, M> &B,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
  Matrix<T, M, M> Y = B.create_dense();

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::size_list[j];
         k < RowPointers_A::size_list[j + 1]; ++k) {
      Y(j, RowIndices_A::size_list[k]) -= A.values[k];
    }
  }

#else

  COMPILED_SPARSE_MATRIX_SUB_DENSE<T, M, N, RowIndices_A, RowPointers_A>(A, Y);

#endif

  return Y;
}

/* Sparse Matrix Subtract Sparse Matrix */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, typename RowIndices_B, typename RowPointers_B>
Matrix<T, M, N>
operator-(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, M, N> Y = A.create_dense();

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_B::size_list[j];
         k < RowPointers_B::size_list[j + 1]; ++k) {
      Y(j, RowIndices_B::size_list[k]) -= B.values[k];
    }
  }

#else

  COMPILED_SPARSE_MATRIX_SUB_DENSE<T, M, N, RowIndices_B, RowPointers_B>(B, Y);

#endif

  return Y;
}

/* Sparse Matrix multiply Scalar */
// Core loop for scalar multiplication
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I, std::size_t End>
struct SparseMatrixMultiplyScalarLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &scalar,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {
    Y.values[I] = scalar * A.values[I];
    SparseMatrixMultiplyScalarLoop<T, M, N, RowIndices_A, RowPointers_A, I + 1,
                                   End>::compute(A, scalar, Y);
  }
};

// End of loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t End>
struct SparseMatrixMultiplyScalarLoop<T, M, N, RowIndices_A, RowPointers_A, End,
                                      End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &scalar,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {
    static_cast<void>(A);
    static_cast<void>(scalar);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
static inline void COMPILED_SPARSE_MATRIX_MULTIPLY_SCALAR(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const T &scalar,
    CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {
  SparseMatrixMultiplyScalarLoop<T, M, N, RowIndices_A, RowPointers_A, 0,
                                 RowIndices_A::size>::compute(A, scalar, Y);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A>
operator*(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &scalar) {
  CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> Y = A;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < RowIndices_A::size; i++) {
    Y.values[i] = scalar * A.values[i];
  }

#else

  COMPILED_SPARSE_MATRIX_MULTIPLY_SCALAR<T, M, N, RowIndices_A, RowPointers_A>(
      A, scalar, Y);

#endif

  return Y;
}

/* Scalar multiply Sparse Matrix */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A>
operator*(const T &scalar,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
  CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> Y = A;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < RowIndices_A::size; i++) {
    Y.values[i] = scalar * A.values[i];
  }

#else

  COMPILED_SPARSE_MATRIX_MULTIPLY_SCALAR<T, M, N, RowIndices_A, RowPointers_A>(
      A, scalar, Y);

#endif

  return Y;
}

/* Sparse Matrix multiply Vector */
// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t Start,
          std::size_t End>
struct SparseMatrixMultiplyVectorLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Vector<T, N> &b, T &sum) {
    sum += A.values[Start] * b[RowIndices_A::size_list[Start]];
    SparseMatrixMultiplyVectorLoop<T, M, N, RowIndices_A, RowPointers_A, J,
                                   Start + 1, End>::compute(A, b, sum);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t End>
struct SparseMatrixMultiplyVectorLoop<T, M, N, RowIndices_A, RowPointers_A, J,
                                      End, End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Vector<T, N> &b, T &sum) {
    static_cast<void>(A);
    static_cast<void>(b);
    static_cast<void>(sum);
    // End of loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct SparseMatrixMultiplyVectorCore {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Vector<T, N> &b, Vector<T, M> &y) {
    T sum = static_cast<T>(0);
    SparseMatrixMultiplyVectorLoop<
        T, M, N, RowIndices_A, RowPointers_A, J, RowPointers_A::size_list[J],
        RowPointers_A::size_list[J + 1]>::compute(A, b, sum);
    y[J] = sum;
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct SparseMatrixMultiplyVectorList {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Vector<T, N> &b, Vector<T, M> &y) {
    SparseMatrixMultiplyVectorCore<T, M, N, RowIndices_A, RowPointers_A,
                                   J>::compute(A, b, y);
    SparseMatrixMultiplyVectorList<T, M, N, RowIndices_A, RowPointers_A,
                                   J - 1>::compute(A, b, y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct SparseMatrixMultiplyVectorList<T, M, N, RowIndices_A, RowPointers_A, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Vector<T, N> &b, Vector<T, M> &y) {
    SparseMatrixMultiplyVectorCore<T, M, N, RowIndices_A, RowPointers_A,
                                   0>::compute(A, b, y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
static inline void COMPILED_SPARSE_MATRIX_MULTIPLY_VECTOR(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const Vector<T, N> &b, Vector<T, M> &y) {
  SparseMatrixMultiplyVectorList<T, M, N, RowIndices_A, RowPointers_A,
                                 M - 1>::compute(A, b, y);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
Vector<T, M>
operator*(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Vector<T, N> &b) {
  Vector<T, M> y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; j++) {
    T sum = static_cast<T>(0);
    for (std::size_t k = RowPointers_A::size_list[j];
         k < RowPointers_A::size_list[j + 1]; k++) {
      sum += A.values[k] * b[RowIndices_A::size_list[k]];
    }
    y[j] = sum;
  }

#else

  COMPILED_SPARSE_MATRIX_MULTIPLY_VECTOR<T, M, N, RowIndices_A, RowPointers_A>(
      A, b, y);

#endif

  return y;
}

/* ColVector multiply Sparse Matrix */
// Core loop
template <typename T, std::size_t N, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t Start,
          std::size_t End>
struct ColVectorMultiplySparseLoop {
  static void
  compute(const ColVector<T, N> &a,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          ColVector<T, K> &y) {
    y[RowIndices_B::size_list[Start]] += B.values[Start] * a[J];
    ColVectorMultiplySparseLoop<T, N, K, RowIndices_B, RowPointers_B, J,
                                Start + 1, End>::compute(a, B, y);
  }
};

// End of core loop
template <typename T, std::size_t N, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t End>
struct ColVectorMultiplySparseLoop<T, N, K, RowIndices_B, RowPointers_B, J, End,
                                   End> {
  static void
  compute(const ColVector<T, N> &a,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          ColVector<T, K> &y) {
    static_cast<void>(a);
    static_cast<void>(B);
    static_cast<void>(y);
    // End of loop, do nothing
  }
};

// List loop
template <typename T, std::size_t N, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J>
struct ColVectorMultiplySparseList {
  static void
  compute(const ColVector<T, N> &a,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          ColVector<T, K> &y) {
    ColVectorMultiplySparseLoop<T, N, K, RowIndices_B, RowPointers_B, J,
                                RowPointers_B::size_list[J],
                                RowPointers_B::size_list[J + 1]>::compute(a, B,
                                                                          y);
    ColVectorMultiplySparseList<T, N, K, RowIndices_B, RowPointers_B,
                                J - 1>::compute(a, B, y);
  }
};

// End of list loop
template <typename T, std::size_t N, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
struct ColVectorMultiplySparseList<T, N, K, RowIndices_B, RowPointers_B, 0> {
  static void
  compute(const ColVector<T, N> &a,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          ColVector<T, K> &y) {
    ColVectorMultiplySparseLoop<T, N, K, RowIndices_B, RowPointers_B, 0,
                                RowPointers_B::size_list[0],
                                RowPointers_B::size_list[1]>::compute(a, B, y);
  }
};

template <typename T, std::size_t N, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
static inline void COMPILED_COLVECTOR_MULTIPLY_SPARSE(
    const ColVector<T, N> &a,
    const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
    ColVector<T, K> &y) {
  ColVectorMultiplySparseList<T, N, K, RowIndices_B, RowPointers_B,
                              N - 1>::compute(a, B, y);
}

template <typename T, std::size_t N, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
ColVector<T, K> colVector_a_mul_SparseB(
    const ColVector<T, N> &a,
    const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B) {
  ColVector<T, K> y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < N; j++) {
    for (std::size_t k = RowPointers_B::size_list[j];
         k < RowPointers_B::size_list[j + 1]; k++) {
      y[RowIndices_B::size_list[k]] += B.values[k] * a[j];
    }
  }

#else

  COMPILED_COLVECTOR_MULTIPLY_SPARSE<T, N, K, RowIndices_B, RowPointers_B>(a, B,
                                                                           y);

#endif

  return y;
}

/* Sparse Matrix multiply Dense Matrix Transpose */
// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t Start, std::size_t End>
struct SparseMatrixMultiplyDenseTransposeLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, T &sum) {
    sum += A.values[Start] * B(I, RowIndices_A::size_list[Start]);
    SparseMatrixMultiplyDenseTransposeLoop<T, M, N, K, RowIndices_A,
                                           RowPointers_A, J, I, Start + 1,
                                           End>::compute(A, B, sum);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t End>
struct SparseMatrixMultiplyDenseTransposeLoop<T, M, N, K, RowIndices_A,
                                              RowPointers_A, J, I, End, End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, T &sum) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(sum);
    // End of loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct SparseMatrixMultiplyDenseTransposeCore {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, Matrix<T, M, K> &Y) {
    T sum = static_cast<T>(0);
    SparseMatrixMultiplyDenseTransposeLoop<
        T, M, N, K, RowIndices_A, RowPointers_A, J, I,
        RowPointers_A::size_list[J],
        RowPointers_A::size_list[J + 1]>::compute(A, B, sum);
    Y(J, I) = sum;
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct SparseMatrixMultiplyDenseTransposeList {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, Matrix<T, M, K> &Y) {
    SparseMatrixMultiplyDenseTransposeCore<T, M, N, K, RowIndices_A,
                                           RowPointers_A, J, I>::compute(A, B,
                                                                         Y);
    SparseMatrixMultiplyDenseTransposeList<T, M, N, K, RowIndices_A,
                                           RowPointers_A, J - 1, I>::compute(A,
                                                                             B,
                                                                             Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct SparseMatrixMultiplyDenseTransposeList<T, M, N, K, RowIndices_A,
                                              RowPointers_A, 0, I> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, Matrix<T, M, K> &Y) {
    SparseMatrixMultiplyDenseTransposeCore<T, M, N, K, RowIndices_A,
                                           RowPointers_A, 0, I>::compute(A, B,
                                                                         Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct SparseMatrixMultiplyDenseTransposeColumn {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, Matrix<T, M, K> &Y) {
    SparseMatrixMultiplyDenseTransposeList<T, M, N, K, RowIndices_A,
                                           RowPointers_A, M - 1, I>::compute(A,
                                                                             B,
                                                                             Y);
    SparseMatrixMultiplyDenseTransposeColumn<T, M, N, K, RowIndices_A,
                                             RowPointers_A, I - 1>::compute(A,
                                                                            B,
                                                                            Y);
  }
};

// End of column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A>
struct SparseMatrixMultiplyDenseTransposeColumn<T, M, N, K, RowIndices_A,
                                                RowPointers_A, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, K, N> &B, Matrix<T, M, K> &Y) {
    SparseMatrixMultiplyDenseTransposeList<T, M, N, K, RowIndices_A,
                                           RowPointers_A, M - 1, 0>::compute(A,
                                                                             B,
                                                                             Y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
static inline void COMPILED_SPARSE_MATRIX_MULTIPLY_DENSE_TRANSPOSE(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, K, N> &B, Matrix<T, M, K> &Y) {
  SparseMatrixMultiplyDenseTransposeColumn<T, M, N, K, RowIndices_A,
                                           RowPointers_A, K - 1>::compute(A, B,
                                                                          Y);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
Matrix<T, M, K> matrix_multiply_SparseA_mul_BTranspose(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, K, N> &B) {
  Matrix<T, M, K> Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < K; i++) {
    for (std::size_t j = 0; j < M; j++) {
      T sum = static_cast<T>(0);
      for (std::size_t k = RowPointers_A::size_list[j];
           k < RowPointers_A::size_list[j + 1]; k++) {
        sum += A.values[k] * B(i, RowIndices_A::size_list[k]);
      }
      Y(j, i) = sum;
    }
  }

#else

  COMPILED_SPARSE_MATRIX_MULTIPLY_DENSE_TRANSPOSE<T, M, N, RowIndices_A,
                                                  RowPointers_A, K>(A, B, Y);

#endif

  return Y;
}

/* Sparse Matrix multiply Sparse Matrix */
// Inner loop for Sparse Matrix multiply Sparse Matrix
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t Start,
          std::size_t L, std::size_t LEnd>
struct SparseMatrixMultiplySparseInnerLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    Y(J, RowIndices_B::size_list[L]) += A.values[Start] * B.values[L];
    SparseMatrixMultiplySparseInnerLoop<T, M, N, K, RowIndices_A, RowPointers_A,
                                        RowIndices_B, RowPointers_B, J, Start,
                                        L + 1, LEnd>::compute(A, B, Y);
  }
};

// End of Inner loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t Start,
          std::size_t LEnd>
struct SparseMatrixMultiplySparseInnerLoop<
    T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, J,
    Start, LEnd, LEnd> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of inner loop, do nothing
  }
};

// Outer loop for Sparse Matrix multiply Sparse Matrix
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t Start,
          std::size_t End>
struct SparseMatrixMultiplySparseLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixMultiplySparseInnerLoop<
        T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, J,
        Start, RowPointers_B::size_list[RowIndices_A::size_list[Start]],
        RowPointers_B::size_list[RowIndices_A::size_list[Start] +
                                 1]>::compute(A, B, Y);
    SparseMatrixMultiplySparseLoop<T, M, N, K, RowIndices_A, RowPointers_A,
                                   RowIndices_B, RowPointers_B, J, Start + 1,
                                   End>::compute(A, B, Y);
  }
};

// End of Outer loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t End>
struct SparseMatrixMultiplySparseLoop<T, M, N, K, RowIndices_A, RowPointers_A,
                                      RowIndices_B, RowPointers_B, J, End,
                                      End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of outer loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t J>
struct SparseMatrixMultiplySparseCore {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixMultiplySparseLoop<T, M, N, K, RowIndices_A, RowPointers_A,
                                   RowIndices_B, RowPointers_B, J,
                                   RowPointers_A::size_list[J],
                                   RowPointers_A::size_list[J + 1]>::compute(A,
                                                                             B,
                                                                             Y);
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t J>
struct SparseMatrixMultiplySparseList {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixMultiplySparseCore<T, M, N, K, RowIndices_A, RowPointers_A,
                                   RowIndices_B, RowPointers_B, J>::compute(A,
                                                                            B,
                                                                            Y);
    SparseMatrixMultiplySparseList<T, M, N, K, RowIndices_A, RowPointers_A,
                                   RowIndices_B, RowPointers_B,
                                   J - 1>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
struct SparseMatrixMultiplySparseList<T, M, N, K, RowIndices_A, RowPointers_A,
                                      RowIndices_B, RowPointers_B, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixMultiplySparseCore<T, M, N, K, RowIndices_A, RowPointers_A,
                                   RowIndices_B, RowPointers_B, 0>::compute(A,
                                                                            B,
                                                                            Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
struct SparseMatrixMultiplySparseColumn {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixMultiplySparseList<T, M, N, K, RowIndices_A, RowPointers_A,
                                   RowIndices_B, RowPointers_B,
                                   M - 1>::compute(A, B, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
static inline void COMPILED_SPARSE_MATRIX_MULTIPLY_SPARSE(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
    Matrix<T, M, K> &Y) {
  SparseMatrixMultiplySparseColumn<T, M, N, K, RowIndices_A, RowPointers_A,
                                   RowIndices_B, RowPointers_B>::compute(A, B,
                                                                         Y);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
Matrix<T, M, K>
operator*(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, M, K> Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_A::size_list[j];
         k < RowPointers_A::size_list[j + 1]; ++k) {
      for (std::size_t l = RowPointers_B::size_list[RowIndices_A::size_list[k]];
           l < RowPointers_B::size_list[RowIndices_A::size_list[k] + 1]; ++l) {
        Y(j, RowIndices_B::size_list[l]) += A.values[k] * B.values[l];
      }
    }
  }

#else

  COMPILED_SPARSE_MATRIX_MULTIPLY_SPARSE<T, M, N, RowIndices_A, RowPointers_A,
                                         K, RowIndices_B, RowPointers_B>(A, B,
                                                                         Y);

#endif

  return Y;
}

/* Sparse Matrix Transpose multiply Sparse Matrix */
// Inner loop for Sparse Matrix Transpose multiply Sparse Matrix
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t Start,
          std::size_t L, std::size_t LEnd>
struct SparseMatrixTransposeMultiplySparseInnerLoop {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    Y(RowIndices_A::size_list[Start], RowIndices_B::size_list[L]) +=
        A.values[Start] * B.values[L];
    SparseMatrixTransposeMultiplySparseInnerLoop<
        T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, I,
        Start, L + 1, LEnd>::compute(A, B, Y);
  }
};

// End of Inner loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t Start,
          std::size_t LEnd>
struct SparseMatrixTransposeMultiplySparseInnerLoop<
    T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, I,
    Start, LEnd, LEnd> {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of inner loop, do nothing
  }
};

// Outer loop for Sparse Matrix Transpose multiply Sparse Matrix
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t Start,
          std::size_t End>
struct SparseMatrixTransposeMultiplySparseLoop {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixTransposeMultiplySparseInnerLoop<
        T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, I,
        Start, RowPointers_B::size_list[I],
        RowPointers_B::size_list[I + 1]>::compute(A, B, Y);
    SparseMatrixTransposeMultiplySparseLoop<
        T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, I,
        Start + 1, End>::compute(A, B, Y);
  }
};

// End of Outer loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t End>
struct SparseMatrixTransposeMultiplySparseLoop<T, M, N, K, RowIndices_A,
                                               RowPointers_A, RowIndices_B,
                                               RowPointers_B, I, End, End> {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of outer loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I>
struct SparseMatrixTransposeMultiplySparseCore {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixTransposeMultiplySparseLoop<
        T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, I,
        RowPointers_A::size_list[I],
        RowPointers_A::size_list[I + 1]>::compute(A, B, Y);
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I>
struct SparseMatrixTransposeMultiplySparseList {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixTransposeMultiplySparseCore<T, M, N, K, RowIndices_A,
                                            RowPointers_A, RowIndices_B,
                                            RowPointers_B, I>::compute(A, B, Y);
    SparseMatrixTransposeMultiplySparseList<T, M, N, K, RowIndices_A,
                                            RowPointers_A, RowIndices_B,
                                            RowPointers_B, I - 1>::compute(A, B,
                                                                           Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
struct SparseMatrixTransposeMultiplySparseList<
    T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, 0> {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixTransposeMultiplySparseCore<T, M, N, K, RowIndices_A,
                                            RowPointers_A, RowIndices_B,
                                            RowPointers_B, 0>::compute(A, B, Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
struct SparseMatrixTransposeMultiplySparseColumn {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixTransposeMultiplySparseList<T, M, N, K, RowIndices_A,
                                            RowPointers_A, RowIndices_B,
                                            RowPointers_B, N - 1>::compute(A, B,
                                                                           Y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
static inline void COMPILED_SPARSE_MATRIX_TRANSPOSE_MULTIPLY_SPARSE(
    const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
    Matrix<T, M, K> &Y) {
  SparseMatrixTransposeMultiplySparseColumn<T, M, N, K, RowIndices_A,
                                            RowPointers_A, RowIndices_B,
                                            RowPointers_B>::compute(A, B, Y);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
Matrix<T, M, K> matrix_multiply_SparseATranspose_mul_SparseB(
    const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, M, K> Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t k = RowPointers_A::size_list[i];
         k < RowPointers_A::size_list[i + 1]; k++) {
      for (std::size_t j = RowPointers_B::size_list[i];
           j < RowPointers_B::size_list[i + 1]; j++) {
        Y(RowIndices_A::size_list[k], RowIndices_B::size_list[j]) +=
            A.values[k] * B.values[j];
      }
    }
  }

#else

  COMPILED_SPARSE_MATRIX_TRANSPOSE_MULTIPLY_SPARSE<
      T, M, N, RowIndices_A, RowPointers_A, K, RowIndices_B, RowPointers_B>(
      A, B, Y);

#endif

  return Y;
}

/* Sparse Matrix multiply Sparse Matrix Transpose */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
Matrix<T, M, K> matrix_multiply_SparseA_mul_SparseBTranspose(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, M, K> Y;

  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < K; j++) {
      for (std::size_t l = RowPointers_A::size_list[i];
           l < RowPointers_A::size_list[i + 1]; l++) {
        for (std::size_t o = RowPointers_B::size_list[j];
             o < RowPointers_B::size_list[j + 1]; o++) {
          if (RowIndices_A::size_list[l] == RowIndices_B::size_list[o]) {
            Y(i, j) += A.values[l] * B.values[o];
          }
        }
      }
    }
  }

  return Y;
}

/* Sparse Matrix multiply Diag Matrix */
// Core loop for Sparse Matrix multiply Diag Matrix
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t I,
          std::size_t Start, std::size_t End>
struct SparseMatrixMultiplyDiagLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, N> &B, T &sum) {
    if (RowIndices_A::size_list[Start] == I) {
      sum += A.values[Start] * B[I];
    }
    SparseMatrixMultiplyDiagLoop<T, M, N, RowIndices_A, RowPointers_A, J, I,
                                 Start + 1, End>::compute(A, B, sum);
  }
};

// End of Core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t I, std::size_t End>
struct SparseMatrixMultiplyDiagLoop<T, M, N, RowIndices_A, RowPointers_A, J, I,
                                    End, End> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, N> &B, T &sum) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(sum);
    // End of loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t I>
struct SparseMatrixMultiplyDiagCore {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, N> &B, Matrix<T, M, N> &Y) {
    T sum = static_cast<T>(0);
    SparseMatrixMultiplyDiagLoop<T, M, N, RowIndices_A, RowPointers_A, J, I,
                                 RowPointers_A::size_list[J],
                                 RowPointers_A::size_list[J + 1]>::compute(A, B,
                                                                           sum);
    Y(J, I) = sum;
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t I>
struct SparseMatrixMultiplyDiagList {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, N> &B, Matrix<T, M, N> &Y) {
    SparseMatrixMultiplyDiagCore<T, M, N, RowIndices_A, RowPointers_A, J,
                                 I>::compute(A, B, Y);
    SparseMatrixMultiplyDiagList<T, M, N, RowIndices_A, RowPointers_A, J - 1,
                                 I>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I>
struct SparseMatrixMultiplyDiagList<T, M, N, RowIndices_A, RowPointers_A, 0,
                                    I> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, N> &B, Matrix<T, M, N> &Y) {
    SparseMatrixMultiplyDiagCore<T, M, N, RowIndices_A, RowPointers_A, 0,
                                 I>::compute(A, B, Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I>
struct SparseMatrixMultiplyDiagColumn {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, N> &B, Matrix<T, M, N> &Y) {
    SparseMatrixMultiplyDiagList<T, M, N, RowIndices_A, RowPointers_A, M - 1,
                                 I>::compute(A, B, Y);
    SparseMatrixMultiplyDiagColumn<T, M, N, RowIndices_A, RowPointers_A,
                                   I - 1>::compute(A, B, Y);
  }
};

// End of column loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct SparseMatrixMultiplyDiagColumn<T, M, N, RowIndices_A, RowPointers_A, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, N> &B, Matrix<T, M, N> &Y) {
    SparseMatrixMultiplyDiagList<T, M, N, RowIndices_A, RowPointers_A, M - 1,
                                 0>::compute(A, B, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
static inline void COMPILED_SPARSE_MATRIX_MULTIPLY_DIAG(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const DiagMatrix<T, N> &B, Matrix<T, M, N> &Y) {
  SparseMatrixMultiplyDiagColumn<T, M, N, RowIndices_A, RowPointers_A,
                                 N - 1>::compute(A, B, Y);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
Matrix<T, M, N>
operator*(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const DiagMatrix<T, N> &B) {
  Matrix<T, M, N> Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < M; j++) {
      T sum = static_cast<T>(0);
      for (std::size_t k = RowPointers_A::size_list[j];
           k < RowPointers_A::size_list[j + 1]; k++) {
        if (RowIndices_A::size_list[k] == i) {
          sum += A.values[k] * B[i];
        }
        Y(j, i) = sum;
      }
    }
  }

#else

  COMPILED_SPARSE_MATRIX_MULTIPLY_DIAG<T, M, N, RowIndices_A, RowPointers_A>(
      A, B, Y);

#endif

  return Y;
}

/* Diag Matrix multiply Sparse Matrix */
// Conditional operation False
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t KStart_End,
          std::size_t I_J>
struct DiagMatrixMultiplySparseConditionalOperation {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // Do nothing
  }
};

// Conditional operation True
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t KStart_End>
struct DiagMatrixMultiplySparseConditionalOperation<
    T, M, K, RowIndices_B, RowPointers_B, I, KStart_End, 0> {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    Y(I, RowIndices_B::size_list[KStart_End]) += B.values[KStart_End] * A[I];
  }
};

// Core loop (Inner loop)
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t I,
          std::size_t KStart, std::size_t KEnd>
struct DiagMatrixMultiplySparseInnerLoop {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DiagMatrixMultiplySparseConditionalOperation<
        T, M, K, RowIndices_B, RowPointers_B, I, KStart, (I - J)>::compute(A, B,
                                                                           Y);
    DiagMatrixMultiplySparseInnerLoop<T, M, K, RowIndices_B, RowPointers_B, J,
                                      I, KStart + 1, KEnd>::compute(A, B, Y);
  }
};

// End of inner loop
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t I,
          std::size_t KEnd>
struct DiagMatrixMultiplySparseInnerLoop<T, M, K, RowIndices_B, RowPointers_B,
                                         J, I, KEnd, KEnd> {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DiagMatrixMultiplySparseConditionalOperation<
        T, M, K, RowIndices_B, RowPointers_B, I, KEnd, (I - J)>::compute(A, B,
                                                                         Y);
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t I>
struct DiagMatrixMultiplySparseCore {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DiagMatrixMultiplySparseInnerLoop<
        T, M, K, RowIndices_B, RowPointers_B, J, I, RowPointers_B::size_list[J],
        RowPointers_B::size_list[J + 1] - 1>::compute(A, B, Y);
  }
};

// List loop
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J>
struct DiagMatrixMultiplySparseList {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DiagMatrixMultiplySparseCore<T, M, K, RowIndices_B, RowPointers_B, J,
                                 J>::compute(A, B, Y);
    DiagMatrixMultiplySparseList<T, M, K, RowIndices_B, RowPointers_B,
                                 J - 1>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
struct DiagMatrixMultiplySparseList<T, M, K, RowIndices_B, RowPointers_B, 0> {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DiagMatrixMultiplySparseCore<T, M, K, RowIndices_B, RowPointers_B, 0,
                                 0>::compute(A, B, Y);
  }
};

template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
static inline void COMPILED_DIAG_MATRIX_MULTIPLY_SPARSE(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
    Matrix<T, M, K> &Y) {
  DiagMatrixMultiplySparseList<T, M, K, RowIndices_B, RowPointers_B,
                               M - 1>::compute(A, B, Y);
}

template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
Matrix<T, M, K>
operator*(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, M, K> Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; j++) {
    for (std::size_t k = RowPointers_B::size_list[j];
         k < RowPointers_B::size_list[j + 1]; k++) {
      for (std::size_t i = 0; i < M; i++) {
        if (i == j) {
          Y(i, RowIndices_B::size_list[k]) += B.values[k] * A[i];
        }
      }
    }
  }

#else

  COMPILED_DIAG_MATRIX_MULTIPLY_SPARSE<T, M, K, RowIndices_B, RowPointers_B>(
      A, B, Y);

#endif

  return Y;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_COMPILED_SPARSE_OPERATION_HPP