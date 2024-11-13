#ifndef BASE_MATRIX_COMPILED_SPARSE_OPERATION_HPP
#define BASE_MATRIX_COMPILED_SPARSE_OPERATION_HPP

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_macros.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"
#include <array>
#include <cstddef>
#include <initializer_list>
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

/* Dense Matrix add Sparse Matrix */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
Matrix<T, M, N>
operator+(const Matrix<T, M, N> &B,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
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

/* Sparse Matrix add Diag Matrix */
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

/* Diag Matrix add Sparse Matrix */
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

/* Sparse Matrix add Sparse Matrix */
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

/* Sparse Matrix subtract Dense Matrix */
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

/* Dense Matrix subtract Sparse Matrix */
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
Matrix<T, M, N>
operator-(const Matrix<T, M, N> &B,
          const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
  Matrix<T, M, N> Y = B;

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

/* Sparse Matrix subtract Diag Matrix */
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

/* Diag Matrix subtract Sparse Matrix */
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

/* Sparse Matrix subtract Sparse Matrix */
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
// Core conditional operation for Sparse Matrix multiply Sparse Matrix Transpose
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t J, std::size_t L,
          std::size_t O, std::size_t L_O>
struct SparseMatrixMultiplySparseTransposeCoreConditional {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of conditional operation, do nothing
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t J, std::size_t L,
          std::size_t O>
struct SparseMatrixMultiplySparseTransposeCoreConditional<
    T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, I, J,
    L, O, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    Y(I, J) += A.values[L] * B.values[O];
  }
};

// Core inner loop for Sparse Matrix multiply Sparse Matrix Transpose
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t J, std::size_t L,
          std::size_t O, std::size_t O_End>
struct SparseMatrixMultiplySparseTransposeInnerLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixMultiplySparseTransposeCoreConditional<
        T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, I,
        J, L, O,
        (RowIndices_A::size_list[L] - RowIndices_B::size_list[O])>::compute(A,
                                                                            B,
                                                                            Y);

    SparseMatrixMultiplySparseTransposeInnerLoop<
        T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, I,
        J, L, (O + 1), (O_End - 1)>::compute(A, B, Y);
  }
};

// Core inner loop for Sparse Matrix multiply Sparse Matrix Transpose
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t J, std::size_t L,
          std::size_t O>
struct SparseMatrixMultiplySparseTransposeInnerLoop<
    T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, I, J,
    L, O, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of inner loop, do nothing
  }
};

// Core outer loop for Sparse Matrix multiply Sparse Matrix Transpose
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t J, std::size_t L,
          std::size_t L_End>
struct SparseMatrixMultiplySparseTransposeOuterLoop {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixMultiplySparseTransposeInnerLoop<
        T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, I,
        J, L, RowPointers_B::size_list[J],
        (RowPointers_B::size_list[J + 1] -
         RowPointers_B::size_list[J])>::compute(A, B, Y);

    SparseMatrixMultiplySparseTransposeOuterLoop<
        T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, I,
        J, (L + 1), (L_End - 1)>::compute(A, B, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t J, std::size_t L>
struct SparseMatrixMultiplySparseTransposeOuterLoop<T, M, N, K, RowIndices_A,
                                                    RowPointers_A, RowIndices_B,
                                                    RowPointers_B, I, J, L, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of outer loop, do nothing
  }
};

// Core loop for Sparse Matrix multiply Sparse Matrix Transpose
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t J>
struct SparseMatrixMultiplySparseTransposeCore {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixMultiplySparseTransposeOuterLoop<
        T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, I,
        J, RowPointers_A::size_list[I],
        (RowPointers_A::size_list[I + 1] -
         RowPointers_A::size_list[I])>::compute(A, B, Y);
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t J>
struct SparseMatrixMultiplySparseTransposeList {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixMultiplySparseTransposeCore<T, M, N, K, RowIndices_A,
                                            RowPointers_A, RowIndices_B,
                                            RowPointers_B, I, J>::compute(A, B,
                                                                          Y);
    SparseMatrixMultiplySparseTransposeList<
        T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
        I - 1, J>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t J>
struct SparseMatrixMultiplySparseTransposeList<T, M, N, K, RowIndices_A,
                                               RowPointers_A, RowIndices_B,
                                               RowPointers_B, 0, J> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixMultiplySparseTransposeCore<T, M, N, K, RowIndices_A,
                                            RowPointers_A, RowIndices_B,
                                            RowPointers_B, 0, J>::compute(A, B,
                                                                          Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B, std::size_t J>
struct SparseMatrixMultiplySparseTransposeColumn {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixMultiplySparseTransposeList<
        T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
        M - 1, J>::compute(A, B, Y);
    SparseMatrixMultiplySparseTransposeColumn<T, M, N, K, RowIndices_A,
                                              RowPointers_A, RowIndices_B,
                                              RowPointers_B, J - 1>::compute(A,
                                                                             B,
                                                                             Y);
  }
};

// End of column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, typename RowIndices_B,
          typename RowPointers_B>
struct SparseMatrixMultiplySparseTransposeColumn<
    T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B, 0> {
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    SparseMatrixMultiplySparseTransposeList<
        T, M, N, K, RowIndices_A, RowPointers_A, RowIndices_B, RowPointers_B,
        M - 1, 0>::compute(A, B, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
static inline void COMPILED_SPARSE_MATRIX_MULTIPLY_SPARSE_TRANSPOSE(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B,
    Matrix<T, M, K> &Y) {
  SparseMatrixMultiplySparseTransposeColumn<T, M, N, K, RowIndices_A,
                                            RowPointers_A, RowIndices_B,
                                            RowPointers_B, K - 1>::compute(A, B,
                                                                           Y);
}

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
Matrix<T, M, K> matrix_multiply_SparseA_mul_SparseBTranspose(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const CompiledSparseMatrix<T, K, N, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, M, K> Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

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

#else

  COMPILED_SPARSE_MATRIX_MULTIPLY_SPARSE_TRANSPOSE<
      T, M, N, RowIndices_A, RowPointers_A, K, RowIndices_B, RowPointers_B>(
      A, B, Y);

#endif

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

/* Transpose (Diag Matrix multiply Sparse Matrix) */
// Conditional operation False
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t KStart_End,
          std::size_t I_J>
struct TransposeDiagMatrixMultiplySparseConditionalOperation {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, K, M> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // Do nothing
  }
};

// Conditional operation True
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t I, std::size_t KStart_End>
struct TransposeDiagMatrixMultiplySparseConditionalOperation<
    T, M, K, RowIndices_B, RowPointers_B, I, KStart_End, 0> {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, K, M> &Y) {
    Y(RowIndices_B::size_list[KStart_End], I) += B.values[KStart_End] * A[I];
  }
};

// Core loop (Inner loop)
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t I,
          std::size_t KStart, std::size_t KEnd>
struct TransposeDiagMatrixMultiplySparseInnerLoop {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, K, M> &Y) {
    TransposeDiagMatrixMultiplySparseConditionalOperation<
        T, M, K, RowIndices_B, RowPointers_B, I, KStart, (I - J)>::compute(A, B,
                                                                           Y);
    TransposeDiagMatrixMultiplySparseInnerLoop<T, M, K, RowIndices_B,
                                               RowPointers_B, J, I, KStart + 1,
                                               KEnd>::compute(A, B, Y);
  }
};

// End of inner loop
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t I,
          std::size_t KEnd>
struct TransposeDiagMatrixMultiplySparseInnerLoop<
    T, M, K, RowIndices_B, RowPointers_B, J, I, KEnd, KEnd> {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, K, M> &Y) {
    TransposeDiagMatrixMultiplySparseConditionalOperation<
        T, M, K, RowIndices_B, RowPointers_B, I, KEnd, (I - J)>::compute(A, B,
                                                                         Y);
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t I>
struct TransposeDiagMatrixMultiplySparseCore {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, K, M> &Y) {
    TransposeDiagMatrixMultiplySparseInnerLoop<
        T, M, K, RowIndices_B, RowPointers_B, J, I, RowPointers_B::size_list[J],
        RowPointers_B::size_list[J + 1] - 1>::compute(A, B, Y);
  }
};

// List loop
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B, std::size_t J>
struct TransposeDiagMatrixMultiplySparseList {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, K, M> &Y) {
    TransposeDiagMatrixMultiplySparseCore<T, M, K, RowIndices_B, RowPointers_B,
                                          J, J>::compute(A, B, Y);
    TransposeDiagMatrixMultiplySparseList<T, M, K, RowIndices_B, RowPointers_B,
                                          J - 1>::compute(A, B, Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
struct TransposeDiagMatrixMultiplySparseList<T, M, K, RowIndices_B,
                                             RowPointers_B, 0> {
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, K, M> &Y) {
    TransposeDiagMatrixMultiplySparseCore<T, M, K, RowIndices_B, RowPointers_B,
                                          0, 0>::compute(A, B, Y);
  }
};

template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
static inline void COMPILED_TRANSPOSE_DIAG_MATRIX_MULTIPLY_SPARSE(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
    Matrix<T, K, M> &Y) {
  TransposeDiagMatrixMultiplySparseList<T, M, K, RowIndices_B, RowPointers_B,
                                        M - 1>::compute(A, B, Y);
}

template <typename T, std::size_t M, std::size_t K, typename RowIndices_B,
          typename RowPointers_B>
Matrix<T, K, M> matrix_multiply_Transpose_DiagA_mul_SparseB(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, K, M> Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < M; j++) {
    for (std::size_t k = RowPointers_B::size_list[j];
         k < RowPointers_B::size_list[j + 1]; k++) {
      for (std::size_t i = 0; i < M; i++) {
        if (i == j) {
          Y(RowIndices_B::size_list[k], i) += B.values[k] * A[i];
        }
      }
    }
  }

#else

  COMPILED_TRANSPOSE_DIAG_MATRIX_MULTIPLY_SPARSE<T, M, K, RowIndices_B,
                                                 RowPointers_B>(A, B, Y);

#endif

  return Y;
}

/* Dense Matrix multiply Sparse Matrix Transpose */
// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t I,
          std::size_t J, std::size_t Start, std::size_t End>
struct DenseMatrixMultiplySparseTransposeLoop {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          T &sum) {
    sum += B.values[Start] * A(I, RowIndices_B::size_list[Start]);
    DenseMatrixMultiplySparseTransposeLoop<T, M, N, K, RowIndices_B,
                                           RowPointers_B, I, J, Start + 1,
                                           End>::compute(A, B, sum);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t I,
          std::size_t J, std::size_t End>
struct DenseMatrixMultiplySparseTransposeLoop<T, M, N, K, RowIndices_B,
                                              RowPointers_B, I, J, End, End> {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          T &sum) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(sum);
    // End of loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t I,
          std::size_t J>
struct DenseMatrixMultiplySparseTransposeCore {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, N, K> &Y) {
    T sum = static_cast<T>(0);
    DenseMatrixMultiplySparseTransposeLoop<
        T, M, N, K, RowIndices_B, RowPointers_B, I, J,
        RowPointers_B::size_list[J],
        RowPointers_B::size_list[J + 1]>::compute(A, B, sum);
    Y(I, J) = sum;
  }
};

// List loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t I,
          std::size_t J>
struct DenseMatrixMultiplySparseTransposeList {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, N, K> &Y) {
    DenseMatrixMultiplySparseTransposeCore<T, M, N, K, RowIndices_B,
                                           RowPointers_B, I, J>::compute(A, B,
                                                                         Y);
    DenseMatrixMultiplySparseTransposeList<T, M, N, K, RowIndices_B,
                                           RowPointers_B, I - 1, J>::compute(A,
                                                                             B,
                                                                             Y);
  }
};

// End of list loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J>
struct DenseMatrixMultiplySparseTransposeList<T, M, N, K, RowIndices_B,
                                              RowPointers_B, 0, J> {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, N, K> &Y) {
    DenseMatrixMultiplySparseTransposeCore<T, M, N, K, RowIndices_B,
                                           RowPointers_B, 0, J>::compute(A, B,
                                                                         Y);
  }
};

// Column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J>
struct DenseMatrixMultiplySparseTransposeColumn {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, N, K> &Y) {
    DenseMatrixMultiplySparseTransposeList<T, M, N, K, RowIndices_B,
                                           RowPointers_B, N - 1, J>::compute(A,
                                                                             B,
                                                                             Y);
    DenseMatrixMultiplySparseTransposeColumn<T, M, N, K, RowIndices_B,
                                             RowPointers_B, J - 1>::compute(A,
                                                                            B,
                                                                            Y);
  }
};

// End of column loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B>
struct DenseMatrixMultiplySparseTransposeColumn<T, M, N, K, RowIndices_B,
                                                RowPointers_B, 0> {
  static void
  compute(const Matrix<T, M, N> &A,
          const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, N, K> &Y) {
    DenseMatrixMultiplySparseTransposeList<T, M, N, K, RowIndices_B,
                                           RowPointers_B, N - 1, 0>::compute(A,
                                                                             B,
                                                                             Y);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t K>
static inline void COMPILED_DENSE_MATRIX_MULTIPLY_SPARSE_TRANSPOSE(
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B,
    Matrix<T, N, K> &Y) {
  DenseMatrixMultiplySparseTransposeColumn<T, M, N, K, RowIndices_B,
                                           RowPointers_B, K - 1>::compute(A, B,
                                                                          Y);
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B>
Matrix<T, N, K> matrix_multiply_A_mul_SparseBTranspose(
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, M, K, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, N, K> Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < K; ++j) {
      T sum = static_cast<T>(0);
      for (std::size_t k = RowPointers_B::size_list[j];
           k < RowPointers_B::size_list[j + 1]; ++k) {
        sum += B.values[k] * A(i, RowIndices_B::size_list[k]);
      }
      Y(i, j) = sum;
    }
  }

#else

  COMPILED_DENSE_MATRIX_MULTIPLY_SPARSE_TRANSPOSE<T, M, N, RowIndices_B,
                                                  RowPointers_B, K>(A, B, Y);

#endif

  return Y;
}

/* Dense Transpose Matrix multiply Sparse Matrix */
// Inner loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t KIndex, std::size_t I, std::size_t Start, std::size_t End>
struct DenseTransposeMultiplySparseInnerLoop {
  static void
  compute(const Matrix<T, N, M> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    Y(I, RowIndices_B::size_list[Start]) += B.values[Start] * A(J, I);
    DenseTransposeMultiplySparseInnerLoop<T, M, N, K, RowIndices_B,
                                          RowPointers_B, J, KIndex, I,
                                          Start + 1, End>::compute(A, B, Y);
  }
};

// End of inner loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t KIndex, std::size_t I, std::size_t End>
struct DenseTransposeMultiplySparseInnerLoop<
    T, M, N, K, RowIndices_B, RowPointers_B, J, KIndex, I, End, End> {
  static void
  compute(const Matrix<T, N, M> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Middle loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t KIndex, std::size_t I>
struct DenseTransposeMultiplySparseMiddleLoop {
  static void
  compute(const Matrix<T, N, M> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DenseTransposeMultiplySparseInnerLoop<
        T, M, N, K, RowIndices_B, RowPointers_B, J, KIndex, I,
        RowPointers_B::size_list[J],
        RowPointers_B::size_list[J + 1]>::compute(A, B, Y);
    DenseTransposeMultiplySparseMiddleLoop<
        T, M, N, K, RowIndices_B, RowPointers_B, J, KIndex, I - 1>::compute(A,
                                                                            B,
                                                                            Y);
  }
};

// End of middle loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t KIndex>
struct DenseTransposeMultiplySparseMiddleLoop<T, M, N, K, RowIndices_B,
                                              RowPointers_B, J, KIndex, 0> {
  static void
  compute(const Matrix<T, N, M> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DenseTransposeMultiplySparseInnerLoop<
        T, M, N, K, RowIndices_B, RowPointers_B, J, KIndex, 0,
        RowPointers_B::size_list[J],
        RowPointers_B::size_list[J + 1]>::compute(A, B, Y);
  }
};

// Outer loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t J,
          std::size_t KIndex>
struct DenseTransposeMultiplySparseOuterLoop {
  static void
  compute(const Matrix<T, N, M> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DenseTransposeMultiplySparseMiddleLoop<
        T, M, N, K, RowIndices_B, RowPointers_B, J, KIndex, M - 1>::compute(A,
                                                                            B,
                                                                            Y);
    DenseTransposeMultiplySparseOuterLoop<
        T, M, N, K, RowIndices_B, RowPointers_B, J - 1, KIndex>::compute(A, B,
                                                                         Y);
  }
};

// End of outer loop
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B, std::size_t KIndex>
struct DenseTransposeMultiplySparseOuterLoop<T, M, N, K, RowIndices_B,
                                             RowPointers_B, 0, KIndex> {
  static void
  compute(const Matrix<T, N, M> &A,
          const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
          Matrix<T, M, K> &Y) {
    DenseTransposeMultiplySparseMiddleLoop<
        T, M, N, K, RowIndices_B, RowPointers_B, 0, KIndex, M - 1>::compute(A,
                                                                            B,
                                                                            Y);
  }
};

// Main function
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B>
static inline void COMPILED_DENSE_TRANSPOSE_MATRIX_MULTIPLY_SPARSE(
    const Matrix<T, N, M> &A,
    const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B,
    Matrix<T, M, K> &Y) {
  DenseTransposeMultiplySparseOuterLoop<T, M, N, K, RowIndices_B, RowPointers_B,
                                        N - 1, K - 1>::compute(A, B, Y);
}

template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename RowIndices_B, typename RowPointers_B>
Matrix<T, M, K> matrix_multiply_ATranspose_mul_SparseB(
    const Matrix<T, N, M> &A,
    const CompiledSparseMatrix<T, N, K, RowIndices_B, RowPointers_B> &B) {
  Matrix<T, M, K> Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < N; j++) {
    for (std::size_t k = RowPointers_B::size_list[j];
         k < RowPointers_B::size_list[j + 1]; k++) {
      for (std::size_t i = 0; i < M; i++) {
        Y(i, RowIndices_B::size_list[k]) += B.values[k] * A(j, i);
      }
    }
  }

#else

  COMPILED_DENSE_TRANSPOSE_MATRIX_MULTIPLY_SPARSE<T, M, N, K, RowIndices_B,
                                                  RowPointers_B>(A, B, Y);

#endif

  return Y;
}

/* Sparse Matrix transpose multiply Dense Matrix */
// Start < End (Core)
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t Start, std::size_t End>
struct SparseTransposeMatrixMultiplyDenseLoop {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    Y(RowIndices_A::size_list[Start], I) += A.values[Start] * B(J, I);
    SparseTransposeMatrixMultiplyDenseLoop<T, N, M, K, RowIndices_A,
                                           RowPointers_A, J, I, Start + 1,
                                           End>::compute(A, B, Y);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I, std::size_t End>
struct SparseTransposeMatrixMultiplyDenseLoop<T, N, M, K, RowIndices_A,
                                              RowPointers_A, J, I, End, End> {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(Y);
    // End of loop, do nothing
  }
};

// Core loop
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct SparseTransposeMatrixMultiplyDenseCore {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    SparseTransposeMatrixMultiplyDenseLoop<
        T, N, M, K, RowIndices_A, RowPointers_A, J, I,
        RowPointers_A::size_list[J],
        RowPointers_A::size_list[J + 1]>::compute(A, B, Y);
  }
};

// List loop
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t J,
          std::size_t I>
struct SparseTransposeMatrixMultiplyDenseList {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    SparseTransposeMatrixMultiplyDenseCore<T, N, M, K, RowIndices_A,
                                           RowPointers_A, J, I>::compute(A, B,
                                                                         Y);
    SparseTransposeMatrixMultiplyDenseList<T, N, M, K, RowIndices_A,
                                           RowPointers_A, J - 1, I>::compute(A,
                                                                             B,
                                                                             Y);
  }
};

// End of list loop
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct SparseTransposeMatrixMultiplyDenseList<T, N, M, K, RowIndices_A,
                                              RowPointers_A, 0, I> {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    SparseTransposeMatrixMultiplyDenseCore<T, N, M, K, RowIndices_A,
                                           RowPointers_A, 0, I>::compute(A, B,
                                                                         Y);
  }
};

// Column loop
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A, std::size_t I>
struct SparseTransposeMatrixMultiplyDenseColumn {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    SparseTransposeMatrixMultiplyDenseList<T, N, M, K, RowIndices_A,
                                           RowPointers_A, N - 1, I>::compute(A,
                                                                             B,
                                                                             Y);
    SparseTransposeMatrixMultiplyDenseColumn<T, N, M, K, RowIndices_A,
                                             RowPointers_A, I - 1>::compute(A,
                                                                            B,
                                                                            Y);
  }
};

// End of column loop
template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A>
struct SparseTransposeMatrixMultiplyDenseColumn<T, N, M, K, RowIndices_A,
                                                RowPointers_A, 0> {
  static void
  compute(const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
          const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    SparseTransposeMatrixMultiplyDenseList<T, N, M, K, RowIndices_A,
                                           RowPointers_A, N - 1, 0>::compute(A,
                                                                             B,
                                                                             Y);
  }
};

template <typename T, std::size_t N, std::size_t M, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
static inline void COMPILED_SPARSE_TRANSPOSE_MATRIX_MULTIPLY_DENSE(
    const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
  SparseTransposeMatrixMultiplyDenseColumn<T, N, M, K, RowIndices_A,
                                           RowPointers_A, K - 1>::compute(A, B,
                                                                          Y);
}

template <typename T, std::size_t N, std::size_t M, std::size_t K,
          typename RowIndices_A, typename RowPointers_A>
Matrix<T, M, K> matrix_multiply_SparseAT_mul_B(
    const CompiledSparseMatrix<T, N, M, RowIndices_A, RowPointers_A> &A,
    const Matrix<T, N, K> &B) {
  Matrix<T, M, K> Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

  for (std::size_t j = 0; j < N; j++) {
    for (std::size_t k = RowPointers_A::size_list[j];
         k < RowPointers_A::size_list[j + 1]; k++) {
      for (std::size_t i = 0; i < M; i++) {
        Y(RowIndices_A::size_list[k], i) += A.values[k] * B(j, i);
      }
    }
  }

#else

  COMPILED_SPARSE_TRANSPOSE_MATRIX_MULTIPLY_DENSE<T, N, M, RowIndices_A,
                                                  RowPointers_A, K>(A, B, Y);

#endif

  return Y;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_COMPILED_SPARSE_OPERATION_HPP
