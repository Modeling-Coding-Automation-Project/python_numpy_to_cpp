/**
 * @file base_matrix_concatenate.hpp
 * @brief Provides template functions and structures for concatenating matrices
 * (dense, diagonal, and sparse) both vertically and horizontally.
 *
 * This header defines a comprehensive set of template-based utilities for
 * concatenating matrices of various types (dense, diagonal, and compiled
 * sparse) in both vertical and horizontal directions. The code supports
 * compile-time matrix size and type deduction, and provides both loop-based and
 * template-recursive implementations for efficient operations. It also includes
 * helper structures for managing sparse matrix metadata during concatenation.
 *
 * @note
 * tparam M is the number of rows in the matrix.
 * tparam N is the number of columns in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef BASE_MATRIX_CONCATENATE_HPP_
#define BASE_MATRIX_CONCATENATE_HPP_

#include "base_matrix_macros.hpp"

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_templates.hpp"
#include "base_utility.hpp"

#include <cstddef>
#include <iostream>
#include <tuple>

namespace Base {
namespace Matrix {

/* Functions: Concatenate vertically */
template <typename T, std::size_t M, std::size_t P, std::size_t N,
          std::size_t Start, std::size_t End, typename Enable = void>
struct VerticalConcatenateLoop;

template <typename T, std::size_t M, std::size_t P, std::size_t N,
          std::size_t Start, std::size_t End>
struct VerticalConcatenateLoop<
    T, M, P, N, Start, End, typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(const Matrix<T, M, N> &A, const Matrix<T, P, N> &B,
                      Matrix<T, M + P, N> &Y) {
    VerticalConcatenateLoop<T, M, P, N, Start, Mid>::compute(A, B, Y);
    VerticalConcatenateLoop<T, M, P, N, Mid, End>::compute(A, B, Y);
  }
};

template <typename T, std::size_t M, std::size_t P, std::size_t N,
          std::size_t Start, std::size_t End>
struct VerticalConcatenateLoop<T, M, P, N, Start, End,
                               typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, const Matrix<T, P, N> &,
                      Matrix<T, M + P, N> &) {}
};

template <typename T, std::size_t M, std::size_t P, std::size_t N,
          std::size_t Start, std::size_t End>
struct VerticalConcatenateLoop<
    T, M, P, N, Start, End, typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &A, const Matrix<T, P, N> &B,
                      Matrix<T, M + P, N> &Y) {
    Base::Utility::copy<T, 0, M, 0, M, (M + P)>(A.data[Start], Y.data[Start]);
    Base::Utility::copy<T, 0, P, M, P, (M + P)>(B.data[Start], Y.data[Start]);
  }
};

/**
 * @brief Concatenates two matrices vertically and stores the result in a third
 * matrix.
 *
 * This function takes two matrices A and B, concatenates them vertically, and
 * stores the result in matrix Y. The operation is performed using a template
 * loop to handle the row-wise copying of elements.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam N The number of columns in matrix A.
 * @tparam P The number of rows in matrix B.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 * @param Y The output matrix with M + P cols and N rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
static inline void
COMPILED_SPARSE_VERTICAL_CONCATENATE(const Matrix<T, M, N> &A,
                                     const Matrix<T, P, N> &B,
                                     Matrix<T, M + P, N> &Y) {
  VerticalConcatenateLoop<T, M, P, N, 0, N>::compute(A, B, Y);
}

/**
 * @brief Updates a vertically concatenated matrix with two input matrices.
 *
 * This function takes two matrices A and B, and updates the output matrix Y
 * by concatenating A on top of B. The operation is performed using either a
 * loop-based or template-based approach, depending on the compilation flags.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam N The number of columns in both matrices A and B.
 * @tparam P The number of rows in matrix B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline void update_vertically_concatenated_matrix(Matrix<T, M + P, N> &Y,
                                                  const Matrix<T, M, N> &A,
                                                  const Matrix<T, P, N> &B) {

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t row = 0; row < N; row++) {
    Base::Utility::copy<T, 0, M, 0, M, (M + P)>(A.data[row], Y.data[row]);
    Base::Utility::copy<T, 0, P, M, P, (M + P)>(B.data[row], Y.data[row]);
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  Base::Matrix::COMPILED_SPARSE_VERTICAL_CONCATENATE<T, M, N, P>(A, B, Y);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/**
 * @brief Concatenates two matrices vertically and returns the result as a new
 * matrix.
 *
 * This function takes two matrices A and B, concatenates them vertically, and
 * returns a new matrix Y containing the result. The operation is performed
 * using a template loop to handle the row-wise copying of elements.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam N The number of columns in matrix A.
 * @tparam P The number of rows in matrix B.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 * @return A new matrix Y with M + P cols and N rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline auto concatenate_vertically(const Matrix<T, M, N> &A,
                                   const Matrix<T, P, N> &B)
    -> Matrix<T, M + P, N> {
  Matrix<T, M + P, N> Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t N> struct DenseAndDiag {

  using CSRIndices =
      CSRIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DenseAvailable<M, N>, DiagAvailable<N>>>;

  using CSRPointers =
      CSRPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
          DenseAvailable<M, N>, DiagAvailable<N>>>;

  using Y_Type = CompiledSparseMatrix<T, (M + N), N, CSRIndices, CSRPointers>;
};

} // namespace ConcatenateVertically

/**
 * @brief Updates a vertically concatenated matrix with a dense matrix and a
 * diagonal matrix.
 *
 * This function takes a dense matrix A and a diagonal matrix B, and updates the
 * output matrix Y by concatenating A on top of B. The operation is performed
 * using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam N The number of columns in both matrices A and B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 */
template <typename T, std::size_t M, std::size_t N>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::DenseAndDiag<T, M, N>::Y_Type &Y,
    const Matrix<T, M, N> &A, const DiagMatrix<T, N> &B) {

  auto sparse_A = create_compiled_sparse(A);
  Base::Utility::copy<T, 0, (M * N), 0, (M * N), ((M * N) + N)>(sparse_A.values,
                                                                Y.values);

  auto sparse_B = create_compiled_sparse(B);
  Base::Utility::copy<T, 0, N, (M * N), N, ((M * N) + N)>(sparse_B.values,
                                                          Y.values);
}

/**
 * @brief Concatenates a dense matrix and a diagonal matrix vertically and
 * returns the result as a new matrix.
 *
 * This function takes a dense matrix A and a diagonal matrix B, concatenates
 * them vertically, and returns a new matrix Y containing the result. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam N The number of columns in both matrices A and B.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 * @return A new matrix Y with M + N cols and N rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_vertically(const Matrix<T, M, N> &A,
                                   const DiagMatrix<T, N> &B) ->
    typename ConcatenateVertically::DenseAndDiag<T, M, N>::Y_Type {

  typename ConcatenateVertically::DenseAndDiag<T, M, N>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename CSRIndices_B, typename CSRPointers_B>
struct DenseAndSparse {

  using SparseAvailable_A = DenseAvailable<M, N>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<N, CSRIndices_B,
                                                  CSRPointers_B>;

  using SparseAvailable_A_v_B =
      ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                           SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_A_v_B>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_A_v_B>;

  using Y_Type =
      CompiledSparseMatrix<T, (M + P), N, CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateVertically

/**
 * @brief Updates a vertically concatenated matrix with a dense matrix and a
 * compiled sparse matrix.
 *
 * This function takes a dense matrix A and a compiled sparse matrix B, and
 * updates the output matrix Y by concatenating A on top of B. The operation is
 * performed using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam N The number of columns in both matrices A and B.
 * @tparam P The number of rows in matrix B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename CSRIndices_B, typename CSRPointers_B>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::DenseAndSparse<T, M, N, P, CSRIndices_B,
                                                   CSRPointers_B>::Y_Type &Y,
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, P, N, CSRIndices_B, CSRPointers_B> &B) {

  auto sparse_A = Base::Matrix::create_compiled_sparse(A);
  Base::Utility::copy<T, 0, (M * N), 0, (M * N),
                      ((M * N) + CSRPointers_B::list[P])>(sparse_A.values,
                                                          Y.values);

  Base::Utility::copy<T, 0, CSRPointers_B::list[P], (M * N),
                      CSRPointers_B::list[P],
                      ((M * N) + CSRPointers_B::list[P])>(B.values, Y.values);
}

/**
 * @brief Concatenates a dense matrix and a compiled sparse matrix vertically
 * and returns the result as a new matrix.
 *
 * This function takes a dense matrix A and a compiled sparse matrix B,
 * concatenates them vertically, and returns a new matrix Y containing the
 * result. The operation is performed using compiled sparse operations to
 * efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam N The number of columns in both matrices A and B.
 * @tparam P The number of rows in matrix B.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 * @return A new matrix Y with M + P cols and N rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename CSRIndices_B, typename CSRPointers_B>
inline auto concatenate_vertically(
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, P, N, CSRIndices_B, CSRPointers_B> &B) ->
    typename ConcatenateVertically::DenseAndSparse<T, M, N, P, CSRIndices_B,
                                                   CSRPointers_B>::Y_Type {

  typename ConcatenateVertically::DenseAndSparse<T, M, N, P, CSRIndices_B,
                                                 CSRPointers_B>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t P> struct DiagAndDense {

  using SparseAvailable_A = DiagAvailable<M>;

  using SparseAvailable_B = DenseAvailable<P, M>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                           SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_Y>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, (M + P), M, CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateVertically

/**
 * @brief Updates a vertically concatenated matrix with a diagonal matrix and a
 * dense matrix.
 *
 * This function takes a diagonal matrix A and a dense matrix B, and updates the
 * output matrix Y by concatenating A on top of B. The operation is performed
 * using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam P The number of rows in matrix B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 */
template <typename T, std::size_t M, std::size_t P>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::DiagAndDense<T, M, P>::Y_Type &Y,
    const DiagMatrix<T, M> &A, const Matrix<T, P, M> &B) {

  auto sparse_A = Base::Matrix::create_compiled_sparse(A);
  Base::Utility::copy<T, 0, M, 0, M, (M + (M * P))>(sparse_A.values, Y.values);

  auto sparse_B = create_compiled_sparse(B);
  Base::Utility::copy<T, 0, (M * P), M, (M * P), (M + (M * P))>(sparse_B.values,
                                                                Y.values);
}

/**
 * @brief Concatenates a diagonal matrix and a dense matrix vertically and
 * returns the result as a new matrix.
 *
 * This function takes a diagonal matrix A and a dense matrix B, concatenates
 * them vertically, and returns a new matrix Y containing the result. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam P The number of rows in matrix B.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 * @return A new matrix Y with M + P cols and M rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t P>
inline auto concatenate_vertically(const DiagMatrix<T, M> &A,
                                   const Matrix<T, P, M> &B) ->
    typename ConcatenateVertically::DiagAndDense<T, M, P>::Y_Type {

  typename ConcatenateVertically::DiagAndDense<T, M, P>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M> struct DiagAndDiag {

  using SparseAvailable_A = DiagAvailable<M>;

  using SparseAvailable_B = DiagAvailable<M>;

  using CSRIndices_Y =
      CSRIndicesFromSparseAvailable<ConcatenateSparseAvailableVertically<
          SparseAvailable_A, SparseAvailable_B>>;

  using CSRPointers_Y =
      CSRPointersFromSparseAvailable<ConcatenateSparseAvailableVertically<
          SparseAvailable_A, SparseAvailable_B>>;

  using Y_Type =
      CompiledSparseMatrix<T, (2 * M), M, CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateVertically

/**
 * @brief Updates a vertically concatenated matrix with two diagonal matrices.
 *
 * This function takes two diagonal matrices A and B, and updates the output
 * matrix Y by concatenating A on top of B. The operation is performed using
 * compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 */
template <typename T, std::size_t M>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::DiagAndDiag<T, M>::Y_Type &Y,
    const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B) {

  auto sparse_A = Base::Matrix::create_compiled_sparse(A);
  Base::Utility::copy<T, 0, M, 0, M, (2 * M)>(sparse_A.values, Y.values);

  auto sparse_B = Base::Matrix::create_compiled_sparse(B);
  Base::Utility::copy<T, 0, M, M, M, (2 * M)>(sparse_B.values, Y.values);
}

/**
 * @brief Concatenates two diagonal matrices vertically and returns the result
 * as a new matrix.
 *
 * This function takes two diagonal matrices A and B, concatenates them
 * vertically, and returns a new matrix Y containing the result. The operation
 * is performed using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 * @return A new matrix Y with 2 * M cols and M rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M>
inline auto concatenate_vertically(const DiagMatrix<T, M> &A,
                                   const DiagMatrix<T, M> &B) ->
    typename ConcatenateVertically::DiagAndDiag<T, M>::Y_Type {

  typename ConcatenateVertically::DiagAndDiag<T, M>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t P, typename CSRIndices_B,
          typename CSRPointers_B>
struct DiagAndSparse {

  using SparseAvailable_A = DiagAvailable<M>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<M, CSRIndices_B,
                                                  CSRPointers_B>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                           SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_Y>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, (M + P), M, CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateVertically

/**
 * @brief Updates a vertically concatenated matrix with a diagonal matrix and a
 * compiled sparse matrix.
 *
 * This function takes a diagonal matrix A and a compiled sparse matrix B, and
 * updates the output matrix Y by concatenating A on top of B. The operation is
 * performed using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam P The number of rows in matrix B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 */
template <typename T, std::size_t M, std::size_t P, typename CSRIndices_B,
          typename CSRPointers_B>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::DiagAndSparse<T, M, P, CSRIndices_B,
                                                  CSRPointers_B>::Y_Type &Y,
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, P, M, CSRIndices_B, CSRPointers_B> &B) {

  auto sparse_A = Base::Matrix::create_compiled_sparse(A);
  Base::Utility::copy<T, 0, M, 0, M, (M + CSRPointers_B::list[P])>(
      sparse_A.values, Y.values);

  Base::Utility::copy<T, 0, CSRPointers_B::list[P], M, CSRPointers_B::list[P],
                      (M + CSRPointers_B::list[P])>(B.values, Y.values);
}

/**
 * @brief Concatenates a diagonal matrix and a compiled sparse matrix
 * vertically and returns the result as a new matrix.
 *
 * This function takes a diagonal matrix A and a compiled sparse matrix B,
 * concatenates them vertically, and returns a new matrix Y containing the
 * result. The operation is performed using compiled sparse operations to
 * efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam P The number of rows in matrix B.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 * @return A new matrix Y with M + P cols and M rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t P, typename CSRIndices_B,
          typename CSRPointers_B>
inline auto concatenate_vertically(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, P, M, CSRIndices_B, CSRPointers_B> &B) ->
    typename ConcatenateVertically::DiagAndSparse<T, M, P, CSRIndices_B,
                                                  CSRPointers_B>::Y_Type {

  typename ConcatenateVertically::DiagAndSparse<T, M, P, CSRIndices_B,
                                                CSRPointers_B>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A, std::size_t P>
struct SparseAndDense {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<N, CSRIndices_A,
                                                  CSRPointers_A>;

  using SparseAvailable_B = DenseAvailable<P, N>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                           SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_Y>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, (M + P), N, CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateVertically

/**
 * @brief Updates a vertically concatenated matrix with a compiled sparse matrix
 * and a dense matrix.
 *
 * This function takes a compiled sparse matrix A and a dense matrix B, and
 * updates the output matrix Y by concatenating A on top of B. The operation is
 * performed using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam N The number of columns in both matrices A and B.
 * @tparam P The number of rows in matrix B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A, std::size_t P>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::SparseAndDense<T, M, N, CSRIndices_A,
                                                   CSRPointers_A, P>::Y_Type &Y,
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A,
    const Matrix<T, P, N> &B) {

  Base::Utility::copy<T, 0, CSRPointers_A::list[P], 0, CSRPointers_A::list[P],
                      (CSRPointers_A::list[P] + (P * N))>(A.values, Y.values);

  auto sparse_B = Base::Matrix::create_compiled_sparse(B);
  Base::Utility::copy<T, 0, (P * N), CSRPointers_A::list[P], (P * N),
                      (CSRPointers_A::list[P] + (P * N))>(sparse_B.values,
                                                          Y.values);
}

/**
 * @brief Concatenates a compiled sparse matrix and a dense matrix vertically
 * and returns the result as a new matrix.
 *
 * This function takes a compiled sparse matrix A and a dense matrix B,
 * concatenates them vertically, and returns a new matrix Y containing the
 * result. The operation is performed using compiled sparse operations to
 * efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam N The number of columns in both matrices A and B.
 * @tparam P The number of rows in matrix B.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 * @return A new matrix Y with M + P cols and N rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A, std::size_t P>
inline auto concatenate_vertically(
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A,
    const Matrix<T, P, N> &B) ->
    typename ConcatenateVertically::SparseAndDense<T, M, N, CSRIndices_A,
                                                   CSRPointers_A, P>::Y_Type {

  typename ConcatenateVertically::SparseAndDense<T, M, N, CSRIndices_A,
                                                 CSRPointers_A, P>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A>
struct SparseAndDiag {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<N, CSRIndices_A,
                                                  CSRPointers_A>;

  using SparseAvailable_B = DiagAvailable<N>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                           SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_Y>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, (M + N), N, CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateVertically

/**
 * @brief Updates a vertically concatenated matrix with a compiled sparse matrix
 * and a diagonal matrix.
 *
 * This function takes a compiled sparse matrix A and a diagonal matrix B, and
 * updates the output matrix Y by concatenating A on top of B. The operation is
 * performed using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam N The number of columns in both matrices A and B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::SparseAndDiag<T, M, N, CSRIndices_A,
                                                  CSRPointers_A>::Y_Type &Y,
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A,
    const DiagMatrix<T, N> &B) {

  Base::Utility::copy<T, 0, CSRPointers_A::list[M], 0, CSRPointers_A::list[M],
                      (CSRPointers_A::list[M] + N)>(A.values, Y.values);

  auto sparse_B = Base::Matrix::create_compiled_sparse(B);
  Base::Utility::copy<T, 0, N, CSRPointers_A::list[M], N,
                      (CSRPointers_A::list[M] + N)>(sparse_B.values, Y.values);
}

/**
 * @brief Concatenates a compiled sparse matrix and a diagonal matrix vertically
 * and returns the result as a new matrix.
 *
 * This function takes a compiled sparse matrix A and a diagonal matrix B,
 * concatenates them vertically, and returns a new matrix Y containing the
 * result. The operation is performed using compiled sparse operations to
 * efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in matrix A.
 * @tparam N The number of columns in both matrices A and B.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 * @return A new matrix Y with M + N cols and N rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A>
inline auto concatenate_vertically(
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A,
    const DiagMatrix<T, N> &B) ->
    typename ConcatenateVertically::SparseAndDiag<T, M, N, CSRIndices_A,
                                                  CSRPointers_A>::Y_Type {

  typename ConcatenateVertically::SparseAndDiag<T, M, N, CSRIndices_A,
                                                CSRPointers_A>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateVertically {

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename CSRIndices_A, typename CSRPointers_A, typename CSRIndices_B,
          typename CSRPointers_B>
struct SparseAndSparse {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<N, CSRIndices_A,
                                                  CSRPointers_A>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<N, CSRIndices_B,
                                                  CSRPointers_B>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                           SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_Y>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, (M + P), N, CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateVertically

/**
 * @brief Updates a vertically concatenated matrix with two compiled sparse
 * matrices.
 *
 * This function takes two compiled sparse matrices A and B, and updates the
 * output matrix Y by concatenating A on top of B. The operation is performed
 * using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in both matrices A and B.
 * @tparam P The number of rows in matrix B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename CSRIndices_A, typename CSRPointers_A, typename CSRIndices_B,
          typename CSRPointers_B>
inline void update_vertically_concatenated_matrix(
    typename ConcatenateVertically::SparseAndSparse<T, M, N, P, CSRIndices_A,
                                                    CSRPointers_A, CSRIndices_B,
                                                    CSRPointers_B>::Y_Type &Y,
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A,
    const CompiledSparseMatrix<T, P, N, CSRIndices_B, CSRPointers_B> &B) {

  Base::Utility::copy<T, 0, CSRPointers_A::list[M], 0, CSRPointers_A::list[M],
                      (CSRPointers_A::list[M] + CSRPointers_B::list[P])>(
      A.values, Y.values);

  Base::Utility::copy<T, 0, CSRPointers_B::list[P], CSRPointers_A::list[M],
                      CSRPointers_B::list[P],
                      (CSRPointers_A::list[M] + CSRPointers_B::list[P])>(
      B.values, Y.values);
}

/**
 * @brief Concatenates two compiled sparse matrices vertically and returns the
 * result as a new matrix.
 *
 * This function takes two compiled sparse matrices A and B, concatenates them
 * vertically, and returns a new matrix Y containing the result. The operation
 * is performed using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in both matrices A and B.
 * @tparam P The number of rows in matrix B.
 * @param A The first input matrix (top part of the result).
 * @param B The second input matrix (bottom part of the result).
 * @return A new matrix Y with M + P cols and N rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename CSRIndices_A, typename CSRPointers_A, typename CSRIndices_B,
          typename CSRPointers_B>
inline auto concatenate_vertically(
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A,
    const CompiledSparseMatrix<T, P, N, CSRIndices_B, CSRPointers_B> &B) ->
    typename ConcatenateVertically::SparseAndSparse<T, M, N, P, CSRIndices_A,
                                                    CSRPointers_A, CSRIndices_B,
                                                    CSRPointers_B>::Y_Type {

  typename ConcatenateVertically::SparseAndSparse<T, M, N, P, CSRIndices_A,
                                                  CSRPointers_A, CSRIndices_B,
                                                  CSRPointers_B>::Y_Type Y;

  Base::Matrix::update_vertically_concatenated_matrix(Y, A, B);

  return Y;
}

/* Functions: Concatenate horizontally */

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Start, std::size_t End, typename Enable = void>
struct CopyColumnsFirstLoop;

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Start, std::size_t End>
struct CopyColumnsFirstLoop<T, M, N, P, Start, End,
                            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N + P> &Y) {
    CopyColumnsFirstLoop<T, M, N, P, Start, Mid>::compute(A, Y);
    CopyColumnsFirstLoop<T, M, N, P, Mid, End>::compute(A, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Start, std::size_t End>
struct CopyColumnsFirstLoop<T, M, N, P, Start, End,
                            typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, N> &, Matrix<T, M, N + P> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Start, std::size_t End>
struct CopyColumnsFirstLoop<T, M, N, P, Start, End,
                            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, N> &A, Matrix<T, M, N + P> &Y) {
    Base::Utility::copy<T, 0, M, 0, M, M>(A(Start), Y(Start));
  }
};

/**
 * @brief Concatenates matrix A into the first N rows of matrix Y.
 *
 * This function copies the contents of matrix A into the corresponding cols and
 * rows of matrix Y, starting from the first row. The resulting matrix Y
 * has additional rows (N + P) to accommodate further concatenation.
 *
 * @tparam T   The data type of the matrix elements.
 * @tparam M   The number of rows in the matrices.
 * @tparam N   The number of columns in matrix A.
 * @tparam P   The number of additional cols in matrix Y.
 * @param A    The input matrix to be concatenated.
 * @param Y    The output matrix with concatenated rows.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
static inline void
COMPILED_SPARSE_HORIZONTAL_CONCATENATE_1(const Matrix<T, M, N> &A,
                                         Matrix<T, M, N + P> &Y) {
  CopyColumnsFirstLoop<T, M, N, P, 0, N>::compute(A, Y);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Start, std::size_t End, typename Enable = void>
struct CopyColumnsSecondLoop;

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Start, std::size_t End>
struct CopyColumnsSecondLoop<T, M, N, P, Start, End,
                             typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(const Matrix<T, M, P> &B, Matrix<T, M, N + P> &Y) {
    CopyColumnsSecondLoop<T, M, N, P, Start, Mid>::compute(B, Y);
    CopyColumnsSecondLoop<T, M, N, P, Mid, End>::compute(B, Y);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Start, std::size_t End>
struct CopyColumnsSecondLoop<T, M, N, P, Start, End,
                             typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, M, P> &, Matrix<T, M, N + P> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t Start, std::size_t End>
struct CopyColumnsSecondLoop<
    T, M, N, P, Start, End, typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, M, P> &B, Matrix<T, M, N + P> &Y) {
    Base::Utility::copy<T, 0, M, 0, M, M>(B(Start), Y(N + Start));
  }
};

/**
 * @brief Concatenates matrix B into the last P rows of matrix Y.
 *
 * This function copies the contents of matrix B into the corresponding cols and
 * rows of matrix Y, starting from the column index N. The resulting matrix Y
 * has additional rows (N + P) to accommodate the concatenation.
 *
 * @tparam T   The data type of the matrix elements.
 * @tparam M   The number of rows in the matrices.
 * @tparam N   The number of columns in matrix A.
 * @tparam P   The number of additional cols in matrix Y.
 * @param B    The input matrix to be concatenated.
 * @param Y    The output matrix with concatenated rows.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
static inline void
COMPILED_SPARSE_HORIZONTAL_CONCATENATE_2(const Matrix<T, M, P> &B,
                                         Matrix<T, M, N + P> &Y) {
  CopyColumnsSecondLoop<T, M, N, P, 0, P>::compute(B, Y);
}

/**
 * @brief Updates a horizontally concatenated matrix with two matrices.
 *
 * This function takes two matrices A and B, and updates the output matrix Y by
 * concatenating A on the left and B on the right. The operation is performed
 * using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix A.
 * @tparam P The number of columns in matrix B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline void update_horizontally_concatenated_matrix(Matrix<T, M, N + P> &Y,
                                                    const Matrix<T, M, N> &A,
                                                    const Matrix<T, M, P> &B) {

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t row = 0; row < N; row++) {
    std::copy(A(row).begin(), A(row).end(), Y(row).begin());
  }

  std::size_t B_row = 0;
  for (std::size_t row = N; row < N + P; row++) {
    std::copy(B(B_row).begin(), B(B_row).end(), Y(row).begin());
    B_row++;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  Base::Matrix::COMPILED_SPARSE_HORIZONTAL_CONCATENATE_1<T, M, N, P>(A, Y);
  Base::Matrix::COMPILED_SPARSE_HORIZONTAL_CONCATENATE_2<T, M, N, P>(B, Y);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/**
 * @brief Concatenates two matrices horizontally and returns the result as a new
 * matrix.
 *
 * This function takes two matrices A and B, concatenates them horizontally, and
 * returns a new matrix Y containing the result. The operation is performed
 * using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix A.
 * @tparam P The number of columns in matrix B.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 * @return A new matrix Y with M cols and N + P rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline auto concatenate_horizontally(const Matrix<T, M, N> &A,
                                     const Matrix<T, M, P> &B)
    -> Matrix<T, M, N + P> {
  Matrix<T, M, N + P> Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

/* Copy DenseMatrix to horizontally concatenated matrix */
template <typename T, std::size_t M, std::size_t N, std::size_t Y_Row,
          std::size_t Y_Col, std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct ConcatMatrixSetFromDenseColumn;

template <typename T, std::size_t M, std::size_t N, std::size_t Y_Row,
          std::size_t Y_Col, std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t I,
          std::size_t Start, std::size_t End>
struct ConcatMatrixSetFromDenseColumn<
    T, M, N, Y_Row, Y_Col, Column_Offset, Col_Offset, CSRIndices_Y,
    CSRPointers_Y, I, Start, End,
    typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &Y,
          const Matrix<T, M, N> &A) {
    ConcatMatrixSetFromDenseColumn<T, M, N, Y_Row, Y_Col, Column_Offset,
                                   Col_Offset, CSRIndices_Y, CSRPointers_Y, I,
                                   Start, Mid>::compute(Y, A);
    ConcatMatrixSetFromDenseColumn<T, M, N, Y_Row, Y_Col, Column_Offset,
                                   Col_Offset, CSRIndices_Y, CSRPointers_Y, I,
                                   Mid, End>::compute(Y, A);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Y_Row,
          std::size_t Y_Col, std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t I,
          std::size_t Start, std::size_t End>
struct ConcatMatrixSetFromDenseColumn<
    T, M, N, Y_Row, Y_Col, Column_Offset, Col_Offset, CSRIndices_Y,
    CSRPointers_Y, I, Start, End,
    typename std::enable_if<(End == Start)>::type> {
  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &,
          const Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Y_Row,
          std::size_t Y_Col, std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t I,
          std::size_t Start, std::size_t End>
struct ConcatMatrixSetFromDenseColumn<
    T, M, N, Y_Row, Y_Col, Column_Offset, Col_Offset, CSRIndices_Y,
    CSRPointers_Y, I, Start, End,
    typename std::enable_if<(End - Start == 1)>::type> {
  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &Y,
          const Matrix<T, M, N> &A) {
    Base::Matrix::set_sparse_matrix_value<(I + Column_Offset),
                                          (Start + Col_Offset)>(Y, A(I, Start));
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Y_Row,
          std::size_t Y_Col, std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t Start,
          std::size_t End, typename Enable = void>
struct ConcatMatrixSetFromDenseRow;

template <typename T, std::size_t M, std::size_t N, std::size_t Y_Row,
          std::size_t Y_Col, std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t Start,
          std::size_t End>
struct ConcatMatrixSetFromDenseRow<
    T, M, N, Y_Row, Y_Col, Column_Offset, Col_Offset, CSRIndices_Y,
    CSRPointers_Y, Start, End,
    typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &Y,
          const Matrix<T, M, N> &A) {
    ConcatMatrixSetFromDenseRow<T, M, N, Y_Row, Y_Col, Column_Offset,
                                Col_Offset, CSRIndices_Y, CSRPointers_Y, Start,
                                Mid>::compute(Y, A);
    ConcatMatrixSetFromDenseRow<T, M, N, Y_Row, Y_Col, Column_Offset,
                                Col_Offset, CSRIndices_Y, CSRPointers_Y, Mid,
                                End>::compute(Y, A);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Y_Row,
          std::size_t Y_Col, std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t Start,
          std::size_t End>
struct ConcatMatrixSetFromDenseRow<
    T, M, N, Y_Row, Y_Col, Column_Offset, Col_Offset, CSRIndices_Y,
    CSRPointers_Y, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &,
          const Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Y_Row,
          std::size_t Y_Col, std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t Start,
          std::size_t End>
struct ConcatMatrixSetFromDenseRow<
    T, M, N, Y_Row, Y_Col, Column_Offset, Col_Offset, CSRIndices_Y,
    CSRPointers_Y, Start, End,
    typename std::enable_if<(End - Start == 1)>::type> {
  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &Y,
          const Matrix<T, M, N> &A) {
    ConcatMatrixSetFromDenseColumn<T, M, N, Y_Row, Y_Col, Column_Offset,
                                   Col_Offset, CSRIndices_Y, CSRPointers_Y,
                                   Start, 0, N>::compute(Y, A);
  }
};

/* Copy DiagMatrix to horizontally concatenated matrix */
template <typename T, std::size_t M, std::size_t Y_Row, std::size_t Y_Col,
          std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t Start,
          std::size_t End, typename Enable = void>
struct ConcatMatrixSetFromDiagRow;

template <typename T, std::size_t M, std::size_t Y_Row, std::size_t Y_Col,
          std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t Start,
          std::size_t End>
struct ConcatMatrixSetFromDiagRow<
    T, M, Y_Row, Y_Col, Column_Offset, Col_Offset, CSRIndices_Y, CSRPointers_Y,
    Start, End, typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &Y,
          const DiagMatrix<T, M> &B) {
    ConcatMatrixSetFromDiagRow<T, M, Y_Row, Y_Col, Column_Offset, Col_Offset,
                               CSRIndices_Y, CSRPointers_Y, Start,
                               Mid>::compute(Y, B);
    ConcatMatrixSetFromDiagRow<T, M, Y_Row, Y_Col, Column_Offset, Col_Offset,
                               CSRIndices_Y, CSRPointers_Y, Mid,
                               End>::compute(Y, B);
  }
};

template <typename T, std::size_t M, std::size_t Y_Row, std::size_t Y_Col,
          std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t Start,
          std::size_t End>
struct ConcatMatrixSetFromDiagRow<
    T, M, Y_Row, Y_Col, Column_Offset, Col_Offset, CSRIndices_Y, CSRPointers_Y,
    Start, End, typename std::enable_if<(End == Start)>::type> {
  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &,
          const DiagMatrix<T, M> &) {}
};

template <typename T, std::size_t M, std::size_t Y_Row, std::size_t Y_Col,
          std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t Start,
          std::size_t End>
struct ConcatMatrixSetFromDiagRow<
    T, M, Y_Row, Y_Col, Column_Offset, Col_Offset, CSRIndices_Y, CSRPointers_Y,
    Start, End, typename std::enable_if<(End - Start == 1)>::type> {
  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &Y,
          const DiagMatrix<T, M> &B) {
    Base::Matrix::set_sparse_matrix_value<(Start + Column_Offset),
                                          (Start + Col_Offset)>(Y, B[Start]);
  }
};

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N> struct DenseAndDiag {

  using SparseAvailable_A = DenseAvailable<M, N>;

  using SparseAvailable_B = DiagAvailable<M>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_Y>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (M + N), CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateHorizontally

/**
 * @brief Updates a horizontally concatenated matrix with a dense matrix and a
 * diagonal matrix.
 *
 * This function takes a dense matrix A and a diagonal matrix B, and updates the
 * output matrix Y by concatenating A on the left and B on the right. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix A.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 */
template <typename T, std::size_t M, std::size_t N>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::DenseAndDiag<T, M, N>::Y_Type &Y,
    const Matrix<T, M, N> &A, const DiagMatrix<T, M> &B) {

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < N) {

        Y.values[value_count] = A(i, j);
        value_count++;

      } else if ((j - N) == i) {

        Y.values[value_count] = B[i];
        value_count++;

      } else {
        /* Do nothing */
      }
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  using CSRIndices_Y =
      typename ConcatenateHorizontally::DenseAndDiag<T, M, N>::CSRIndices_Y;

  using CSRPointers_Y =
      typename ConcatenateHorizontally::DenseAndDiag<T, M, N>::CSRPointers_Y;

  ConcatMatrixSetFromDenseRow<T, M, N, M, (M + N), 0, 0, CSRIndices_Y,
                              CSRPointers_Y, 0, M>::compute(Y, A);

  ConcatMatrixSetFromDiagRow<T, M, M, (M + N), 0, N, CSRIndices_Y,
                             CSRPointers_Y, 0, M>::compute(Y, B);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/**
 * @brief Concatenates a dense matrix and a diagonal matrix horizontally and
 * returns the result as a new matrix.
 *
 * This function takes a dense matrix A and a diagonal matrix B, concatenates
 * them horizontally, and returns a new matrix Y containing the result. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix A.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 * @return A new matrix Y with M cols and N + M rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_horizontally(const Matrix<T, M, N> &A,
                                     const DiagMatrix<T, M> &B) ->
    typename ConcatenateHorizontally::DenseAndDiag<T, M, N>::Y_Type {

  typename ConcatenateHorizontally::DenseAndDiag<T, M, N>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

/* Copy SparseMatrix to horizontally concatenated matrix */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A, std::size_t Y_Row, std::size_t Y_Col,
          std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct ConcatMatrixSetFromSparseColumn;

template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A, std::size_t Y_Row, std::size_t Y_Col,
          std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t I,
          std::size_t Start, std::size_t End>
struct ConcatMatrixSetFromSparseColumn<
    T, M, N, CSRIndices_A, CSRPointers_A, Y_Row, Y_Col, Column_Offset,
    Col_Offset, CSRIndices_Y, CSRPointers_Y, I, Start, End,
    typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &Y,
          const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A) {
    ConcatMatrixSetFromSparseColumn<
        T, M, N, CSRIndices_A, CSRPointers_A, Y_Row, Y_Col, Column_Offset,
        Col_Offset, CSRIndices_Y, CSRPointers_Y, I, Start, Mid>::compute(Y, A);
    ConcatMatrixSetFromSparseColumn<
        T, M, N, CSRIndices_A, CSRPointers_A, Y_Row, Y_Col, Column_Offset,
        Col_Offset, CSRIndices_Y, CSRPointers_Y, I, Mid, End>::compute(Y, A);
  }
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A, std::size_t Y_Row, std::size_t Y_Col,
          std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t I,
          std::size_t Start, std::size_t End>
struct ConcatMatrixSetFromSparseColumn<
    T, M, N, CSRIndices_A, CSRPointers_A, Y_Row, Y_Col, Column_Offset,
    Col_Offset, CSRIndices_Y, CSRPointers_Y, I, Start, End,
    typename std::enable_if<(End == Start)>::type> {
  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &,
          const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &) {}
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A, std::size_t Y_Row, std::size_t Y_Col,
          std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t I,
          std::size_t Start, std::size_t End>
struct ConcatMatrixSetFromSparseColumn<
    T, M, N, CSRIndices_A, CSRPointers_A, Y_Row, Y_Col, Column_Offset,
    Col_Offset, CSRIndices_Y, CSRPointers_Y, I, Start, End,
    typename std::enable_if<(End - Start == 1)>::type> {
  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &Y,
          const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A) {
    Base::Matrix::set_sparse_matrix_value<(I + Column_Offset),
                                          (Start + Col_Offset)>(
        Y, Base::Matrix::get_sparse_matrix_value<I, Start>(A));
  }
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A, std::size_t Y_Row, std::size_t Y_Col,
          std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t Start,
          std::size_t End, typename Enable = void>
struct ConcatMatrixSetFromSparseRow;

template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A, std::size_t Y_Row, std::size_t Y_Col,
          std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t Start,
          std::size_t End>
struct ConcatMatrixSetFromSparseRow<
    T, M, N, CSRIndices_A, CSRPointers_A, Y_Row, Y_Col, Column_Offset,
    Col_Offset, CSRIndices_Y, CSRPointers_Y, Start, End,
    typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &Y,
          const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A) {
    ConcatMatrixSetFromSparseRow<T, M, N, CSRIndices_A, CSRPointers_A, Y_Row,
                                 Y_Col, Column_Offset, Col_Offset, CSRIndices_Y,
                                 CSRPointers_Y, Start, Mid>::compute(Y, A);
    ConcatMatrixSetFromSparseRow<T, M, N, CSRIndices_A, CSRPointers_A, Y_Row,
                                 Y_Col, Column_Offset, Col_Offset, CSRIndices_Y,
                                 CSRPointers_Y, Mid, End>::compute(Y, A);
  }
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A, std::size_t Y_Row, std::size_t Y_Col,
          std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t Start,
          std::size_t End>
struct ConcatMatrixSetFromSparseRow<
    T, M, N, CSRIndices_A, CSRPointers_A, Y_Row, Y_Col, Column_Offset,
    Col_Offset, CSRIndices_Y, CSRPointers_Y, Start, End,
    typename std::enable_if<(End == Start)>::type> {
  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &,
          const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &) {}
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A, std::size_t Y_Row, std::size_t Y_Col,
          std::size_t Column_Offset, std::size_t Col_Offset,
          typename CSRIndices_Y, typename CSRPointers_Y, std::size_t Start,
          std::size_t End>
struct ConcatMatrixSetFromSparseRow<
    T, M, N, CSRIndices_A, CSRPointers_A, Y_Row, Y_Col, Column_Offset,
    Col_Offset, CSRIndices_Y, CSRPointers_Y, Start, End,
    typename std::enable_if<(End - Start == 1)>::type> {
  static void
  compute(CompiledSparseMatrix<T, Y_Row, Y_Col, CSRIndices_Y, CSRPointers_Y> &Y,
          const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A) {
    ConcatMatrixSetFromSparseColumn<
        T, M, N, CSRIndices_A, CSRPointers_A, Y_Row, Y_Col, Column_Offset,
        Col_Offset, CSRIndices_Y, CSRPointers_Y, Start, 0, N>::compute(Y, A);
  }
};

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename CSRIndices_B, typename CSRPointers_B>
struct DenseAndSparse {

  using SparseAvailable_A = DenseAvailable<M, N>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<L, CSRIndices_B,
                                                  CSRPointers_B>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_Y>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (N + L), CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateHorizontally

/**
 * @brief Updates a horizontally concatenated matrix with a dense matrix and a
 * sparse matrix.
 *
 * This function takes a dense matrix A and a sparse matrix B, and updates the
 * output matrix Y by concatenating A on the left and B on the right. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix A.
 * @tparam L The number of columns in matrix B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename CSRIndices_B, typename CSRPointers_B>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::DenseAndSparse<T, M, N, L, CSRIndices_B,
                                                     CSRPointers_B>::Y_Type &Y,
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, M, L, CSRIndices_B, CSRPointers_B> &B) {

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  std::size_t value_count = 0;
  std::size_t sparse_value_count = 0;
  std::size_t sparse_row_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + L); j++) {
      if (j < N) {

        Y.values[value_count] = A(i, j);
        value_count++;

      } else if ((CSRPointers_B::list[i + 1] - CSRPointers_B::list[i] >
                  sparse_row_count) &&
                 (sparse_value_count < CSRIndices_B::size)) {

        if ((j - N) == CSRIndices_B::list[sparse_value_count]) {
          Y.values[value_count] = B.values[sparse_value_count];

          value_count++;
          sparse_value_count++;
          sparse_row_count++;
        }
      }
    }
    sparse_row_count = 0;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  using CSRIndices_Y = typename ConcatenateHorizontally::DenseAndSparse<
      T, M, N, L, CSRIndices_B, CSRPointers_B>::CSRIndices_Y;

  using CSRPointers_Y = typename ConcatenateHorizontally::DenseAndSparse<
      T, M, N, L, CSRIndices_B, CSRPointers_B>::CSRPointers_Y;

  ConcatMatrixSetFromDenseRow<T, M, N, M, (N + L), 0, 0, CSRIndices_Y,
                              CSRPointers_Y, 0, M>::compute(Y, A);

  ConcatMatrixSetFromSparseRow<T, M, L, CSRIndices_B, CSRPointers_B, M, (N + L),
                               0, N, CSRIndices_Y, CSRPointers_Y, 0,
                               M>::compute(Y, B);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/**
 * @brief Concatenates a dense matrix and a sparse matrix horizontally and
 * returns the result as a new matrix.
 *
 * This function takes a dense matrix A and a sparse matrix B, concatenates
 * them horizontally, and returns a new matrix Y containing the result. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix A.
 * @tparam L The number of columns in matrix B.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 * @return A new matrix Y with M cols and N + L rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename CSRIndices_B, typename CSRPointers_B>
inline auto concatenate_horizontally(
    const Matrix<T, M, N> &A,
    const CompiledSparseMatrix<T, M, L, CSRIndices_B, CSRPointers_B> &B) ->
    typename ConcatenateHorizontally::DenseAndSparse<T, M, N, L, CSRIndices_B,
                                                     CSRPointers_B>::Y_Type {

  typename ConcatenateHorizontally::DenseAndSparse<T, M, N, L, CSRIndices_B,
                                                   CSRPointers_B>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N> struct DiagAndDense {

  using SparseAvailable_A = DiagAvailable<M>;

  using SparseAvailable_B = DenseAvailable<M, N>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_Y>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (M + N), CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateHorizontally

/**
 * @brief Updates a horizontally concatenated matrix with a diagonal matrix and
 * a dense matrix.
 *
 * This function takes a diagonal matrix A and a dense matrix B, and updates the
 * output matrix Y by concatenating A on the left and B on the right. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 */
template <typename T, std::size_t M, std::size_t N>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::DiagAndDense<T, M, N>::Y_Type &Y,
    const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B) {

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < M) {
        if (i == j) {
          Y.values[value_count] = A[i];

          value_count++;
        }
      } else {

        Y.values[value_count] = B(i, j - M);

        value_count++;
      }
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  using CSRIndices_Y =
      typename ConcatenateHorizontally::DiagAndDense<T, M, N>::CSRIndices_Y;

  using CSRPointers_Y =
      typename ConcatenateHorizontally::DiagAndDense<T, M, N>::CSRPointers_Y;

  ConcatMatrixSetFromDiagRow<T, M, M, (M + N), 0, 0, CSRIndices_Y,
                             CSRPointers_Y, 0, M>::compute(Y, A);

  ConcatMatrixSetFromDenseRow<T, M, N, M, (M + N), 0, M, CSRIndices_Y,
                              CSRPointers_Y, 0, M>::compute(Y, B);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/**
 * @brief Concatenates a diagonal matrix and a dense matrix horizontally and
 * returns the result as a new matrix.
 *
 * This function takes a diagonal matrix A and a dense matrix B, concatenates
 * them horizontally, and returns a new matrix Y containing the result. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix B.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 * @return A new matrix Y with M cols and N + M rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_horizontally(const DiagMatrix<T, M> &A,
                                     const Matrix<T, M, N> &B) ->
    typename ConcatenateHorizontally::DiagAndDense<T, M, N>::Y_Type {

  typename ConcatenateHorizontally::DiagAndDense<T, M, N>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateHorizontally {

template <typename T, std::size_t M> struct DiagAndDiag {

  using SparseAvailable_A = DiagAvailable<M>;

  using SparseAvailable_B = DiagAvailable<M>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_Y>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (2 * M), CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateHorizontally

/**
 * @brief Updates a horizontally concatenated matrix with two diagonal matrices.
 *
 * This function takes two diagonal matrices A and B, and updates the output
 * matrix Y by concatenating A on the left and B on the right. The operation is
 * performed using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 */
template <typename T, std::size_t M>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::DiagAndDiag<T, M>::Y_Type &Y,
    const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B) {

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (2 * M); j++) {
      if (j < M) {
        if (i == j) {

          Y.values[value_count] = A[i];
          value_count++;
        }
      } else {
        if ((j - M) == i) {

          Y.values[value_count] = B[i];
          value_count++;
        }
      }
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  using CSRIndices_Y =
      typename ConcatenateHorizontally::DiagAndDiag<T, M>::CSRIndices_Y;

  using CSRPointers_Y =
      typename ConcatenateHorizontally::DiagAndDiag<T, M>::CSRPointers_Y;

  ConcatMatrixSetFromDiagRow<T, M, M, (2 * M), 0, 0, CSRIndices_Y,
                             CSRPointers_Y, 0, M>::compute(Y, A);

  ConcatMatrixSetFromDiagRow<T, M, M, (2 * M), 0, M, CSRIndices_Y,
                             CSRPointers_Y, 0, M>::compute(Y, B);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/**
 * @brief Concatenates two diagonal matrices horizontally and returns the result
 * as a new matrix.
 *
 * This function takes two diagonal matrices A and B, concatenates them
 * horizontally, and returns a new matrix Y containing the result. The operation
 * is performed using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 * @return A new matrix Y with M cols and 2 * M rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M>
inline auto concatenate_horizontally(const DiagMatrix<T, M> &A,
                                     const DiagMatrix<T, M> &B) ->
    typename ConcatenateHorizontally::DiagAndDiag<T, M>::Y_Type {

  typename ConcatenateHorizontally::DiagAndDiag<T, M>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N, typename CSRIndices_B,
          typename CSRPointers_B>
struct DiagAndSparse {

  using SparseAvailable_A = DiagAvailable<M>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<N, CSRIndices_B,
                                                  CSRPointers_B>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_Y>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (M + N), CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateHorizontally

/**
 * @brief Updates a horizontally concatenated matrix with a diagonal matrix and
 * a sparse matrix.
 *
 * This function takes a diagonal matrix A and a sparse matrix B, and updates
 * the output matrix Y by concatenating A on the left and B on the right. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices_B,
          typename CSRPointers_B>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::DiagAndSparse<T, M, N, CSRIndices_B,
                                                    CSRPointers_B>::Y_Type &Y,
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, N, CSRIndices_B, CSRPointers_B> &B) {

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  std::size_t value_count = 0;
  std::size_t sparse_value_count = 0;
  std::size_t sparse_row_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < M) {
        if (i == j) {

          Y.values[value_count] = A[i];
          value_count++;
        }
      } else if ((CSRPointers_B::list[i + 1] - CSRPointers_B::list[i] >
                  sparse_row_count) &&
                 (sparse_value_count < CSRIndices_B::size)) {

        if ((j - M) == CSRIndices_B::list[sparse_value_count]) {
          Y.values[value_count] = B.values[sparse_value_count];

          value_count++;
          sparse_value_count++;
          sparse_row_count++;
        }
      }
    }
    sparse_row_count = 0;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  using CSRIndices_Y = typename ConcatenateHorizontally::DiagAndSparse<
      T, M, N, CSRIndices_B, CSRPointers_B>::CSRIndices_Y;

  using CSRPointers_Y = typename ConcatenateHorizontally::DiagAndSparse<
      T, M, N, CSRIndices_B, CSRPointers_B>::CSRPointers_Y;

  ConcatMatrixSetFromDiagRow<T, M, M, (M + N), 0, 0, CSRIndices_Y,
                             CSRPointers_Y, 0, M>::compute(Y, A);

  ConcatMatrixSetFromSparseRow<T, M, N, CSRIndices_B, CSRPointers_B, M, (M + N),
                               0, M, CSRIndices_Y, CSRPointers_Y, 0,
                               M>::compute(Y, B);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/**
 * @brief Concatenates a diagonal matrix and a sparse matrix horizontally and
 * returns the result as a new matrix.
 *
 * This function takes a diagonal matrix A and a sparse matrix B, concatenates
 * them horizontally, and returns a new matrix Y containing the result. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix B.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 * @return A new matrix Y with M cols and M + N rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices_B,
          typename CSRPointers_B>
inline auto concatenate_horizontally(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, N, CSRIndices_B, CSRPointers_B> &B) ->
    typename ConcatenateHorizontally::DiagAndSparse<T, M, N, CSRIndices_B,
                                                    CSRPointers_B>::Y_Type {

  typename ConcatenateHorizontally::DiagAndSparse<T, M, N, CSRIndices_B,
                                                  CSRPointers_B>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename CSRIndices_A, typename CSRPointers_A>
struct SparseAndDense {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<L, CSRIndices_A,
                                                  CSRPointers_A>;

  using SparseAvailable_B = DenseAvailable<M, N>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_Y>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (N + L), CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateHorizontally

/**
 * @brief Updates a horizontally concatenated matrix with a sparse matrix and a
 * dense matrix.
 *
 * This function takes a sparse matrix A and a dense matrix B, and updates the
 * output matrix Y by concatenating A on the left and B on the right. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix B.
 * @tparam L The number of columns in matrix A.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename CSRIndices_A, typename CSRPointers_A>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::SparseAndDense<T, M, N, L, CSRIndices_A,
                                                     CSRPointers_A>::Y_Type &Y,
    const CompiledSparseMatrix<T, M, L, CSRIndices_A, CSRPointers_A> &A,
    const Matrix<T, M, N> &B) {

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  std::size_t value_count = 0;
  std::size_t sparse_value_count = 0;
  std::size_t sparse_row_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + L); j++) {
      if (j < N) {

        if ((CSRPointers_A::list[i + 1] - CSRPointers_A::list[i] >
             sparse_row_count) &&
            (sparse_value_count < CSRIndices_A::size)) {

          if (j == CSRIndices_A::list[sparse_value_count]) {
            Y.values[value_count] = A.values[sparse_value_count];

            value_count++;
            sparse_value_count++;
            sparse_row_count++;
          }
        }
      } else {

        Y.values[value_count] = B(i, j - N);
        value_count++;
      }
    }
    sparse_row_count = 0;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  using CSRIndices_Y = typename ConcatenateHorizontally::SparseAndDense<
      T, M, N, L, CSRIndices_A, CSRPointers_A>::CSRIndices_Y;

  using CSRPointers_Y = typename ConcatenateHorizontally::SparseAndDense<
      T, M, N, L, CSRIndices_A, CSRPointers_A>::CSRPointers_Y;

  ConcatMatrixSetFromSparseRow<T, M, L, CSRIndices_A, CSRPointers_A, M, (N + L),
                               0, 0, CSRIndices_Y, CSRPointers_Y, 0,
                               M>::compute(Y, A);

  ConcatMatrixSetFromDenseRow<T, M, N, M, (N + L), 0, N, CSRIndices_Y,
                              CSRPointers_Y, 0, M>::compute(Y, B);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/**
 * @brief Concatenates a sparse matrix and a dense matrix horizontally and
 * returns the result as a new matrix.
 *
 * This function takes a sparse matrix A and a dense matrix B, concatenates
 * them horizontally, and returns a new matrix Y containing the result. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix B.
 * @tparam L The number of columns in matrix A.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 * @return A new matrix Y with M cols and N + L rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename CSRIndices_A, typename CSRPointers_A>
inline auto concatenate_horizontally(
    const CompiledSparseMatrix<T, M, L, CSRIndices_A, CSRPointers_A> &A,
    const Matrix<T, M, N> &B) ->
    typename ConcatenateHorizontally::SparseAndDense<T, M, N, L, CSRIndices_A,
                                                     CSRPointers_A>::Y_Type {

  typename ConcatenateHorizontally::SparseAndDense<T, M, N, L, CSRIndices_A,
                                                   CSRPointers_A>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A>
struct SparseAndDiag {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<N, CSRIndices_A,
                                                  CSRPointers_A>;

  using SparseAvailable_B = DiagAvailable<M>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_Y>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (M + N), CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateHorizontally

/**
 * @brief Updates a horizontally concatenated matrix with a sparse matrix and a
 * diagonal matrix.
 *
 * This function takes a sparse matrix A and a diagonal matrix B, and updates
 * the output matrix Y by concatenating A on the left and B on the right. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix A.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::SparseAndDiag<T, M, N, CSRIndices_A,
                                                    CSRPointers_A>::Y_Type &Y,
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A,
    const DiagMatrix<T, M> &B) {

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  std::size_t value_count = 0;
  std::size_t sparse_value_count = 0;
  std::size_t sparse_row_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < N) {

        if ((CSRPointers_A::list[i + 1] - CSRPointers_A::list[i] >
             sparse_row_count) &&
            (sparse_value_count < CSRIndices_A::size)) {

          if (j == CSRIndices_A::list[sparse_value_count]) {
            Y.values[value_count] = A.values[sparse_value_count];

            value_count++;
            sparse_value_count++;
            sparse_row_count++;
          }
        }

      } else {
        if (i == (j - N)) {

          Y.values[value_count] = B[i];
          value_count++;
        }
      }
    }
    sparse_row_count = 0;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  using CSRIndices_Y = typename ConcatenateHorizontally::SparseAndDiag<
      T, M, N, CSRIndices_A, CSRPointers_A>::CSRIndices_Y;

  using CSRPointers_Y = typename ConcatenateHorizontally::SparseAndDiag<
      T, M, N, CSRIndices_A, CSRPointers_A>::CSRPointers_Y;

  ConcatMatrixSetFromSparseRow<T, M, N, CSRIndices_A, CSRPointers_A, M, (M + N),
                               0, 0, CSRIndices_Y, CSRPointers_Y, 0,
                               M>::compute(Y, A);

  ConcatMatrixSetFromDiagRow<T, M, M, (M + N), 0, N, CSRIndices_Y,
                             CSRPointers_Y, 0, M>::compute(Y, B);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/**
 * @brief Concatenates a sparse matrix and a diagonal matrix horizontally and
 * returns the result as a new matrix.
 *
 * This function takes a sparse matrix A and a diagonal matrix B, concatenates
 * them horizontally, and returns a new matrix Y containing the result. The
 * operation is performed using compiled sparse operations to efficiently copy
 * the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix A.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 * @return A new matrix Y with M cols and M + N rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices_A,
          typename CSRPointers_A>
inline auto concatenate_horizontally(
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A,
    const DiagMatrix<T, M> &B) ->
    typename ConcatenateHorizontally::SparseAndDiag<T, M, N, CSRIndices_A,
                                                    CSRPointers_A>::Y_Type {

  typename ConcatenateHorizontally::SparseAndDiag<T, M, N, CSRIndices_A,
                                                  CSRPointers_A>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

namespace ConcatenateHorizontally {

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename CSRIndices_A, typename CSRPointers_A, typename CSRIndices_B,
          typename CSRPointers_B>
struct SparseAndSparse {

  using SparseAvailable_A =
      CreateSparseAvailableFromIndicesAndPointers<N, CSRIndices_A,
                                                  CSRPointers_A>;

  using SparseAvailable_B =
      CreateSparseAvailableFromIndicesAndPointers<N, CSRIndices_B,
                                                  CSRPointers_B>;

  using SparseAvailable_Y =
      ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                             SparseAvailable_B>;

  using CSRIndices_Y = CSRIndicesFromSparseAvailable<SparseAvailable_Y>;

  using CSRPointers_Y = CSRPointersFromSparseAvailable<SparseAvailable_Y>;

  using Y_Type =
      CompiledSparseMatrix<T, M, (N + L), CSRIndices_Y, CSRPointers_Y>;
};

} // namespace ConcatenateHorizontally

/**
 * @brief Updates a horizontally concatenated matrix with two sparse matrices.
 *
 * This function takes two sparse matrices A and B, and updates the output
 * matrix Y by concatenating A on the left and B on the right. The operation is
 * performed using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix A.
 * @tparam L The number of columns in matrix B.
 * @param Y The output matrix to be updated with the concatenated result.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename CSRIndices_A, typename CSRPointers_A, typename CSRIndices_B,
          typename CSRPointers_B>
inline void update_horizontally_concatenated_matrix(
    typename ConcatenateHorizontally::SparseAndSparse<
        T, M, N, L, CSRIndices_A, CSRPointers_A, CSRIndices_B,
        CSRPointers_B>::Y_Type &Y,
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A,
    const CompiledSparseMatrix<T, M, L, CSRIndices_B, CSRPointers_B> &B) {

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  std::size_t value_count = 0;
  std::size_t sparse_value_count_A = 0;
  std::size_t sparse_row_count_A = 0;
  std::size_t sparse_value_count_B = 0;
  std::size_t sparse_row_count_B = 0;

  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if ((j < N) &&
          (CSRPointers_A::list[i + 1] - CSRPointers_A::list[i] >
           sparse_row_count_A) &&
          (sparse_value_count_A < CSRIndices_A::size)) {

        if (j == CSRIndices_A::list[sparse_value_count_A]) {
          Y.values[value_count] = A.values[sparse_value_count_A];

          value_count++;
          sparse_value_count_A++;
          sparse_row_count_A++;
        }
      } else if ((CSRPointers_B::list[i + 1] - CSRPointers_B::list[i] >
                  sparse_row_count_B) &&
                 (sparse_value_count_B < CSRIndices_B::size)) {

        if ((j - N) == CSRIndices_B::list[sparse_value_count_B]) {
          Y.values[value_count] = B.values[sparse_value_count_B];

          value_count++;
          sparse_value_count_B++;
          sparse_row_count_B++;
        }
      }
    }
    sparse_row_count_A = 0;
    sparse_row_count_B = 0;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  using CSRIndices_Y = typename ConcatenateHorizontally::SparseAndSparse<
      T, M, N, L, CSRIndices_A, CSRPointers_A, CSRIndices_B,
      CSRPointers_B>::CSRIndices_Y;

  using CSRPointers_Y = typename ConcatenateHorizontally::SparseAndSparse<
      T, M, N, L, CSRIndices_A, CSRPointers_A, CSRIndices_B,
      CSRPointers_B>::CSRPointers_Y;

  ConcatMatrixSetFromSparseRow<T, M, N, CSRIndices_A, CSRPointers_A, M, (N + L),
                               0, 0, CSRIndices_Y, CSRPointers_Y, 0,
                               M>::compute(Y, A);

  ConcatMatrixSetFromSparseRow<T, M, L, CSRIndices_B, CSRPointers_B, M, (N + L),
                               0, N, CSRIndices_Y, CSRPointers_Y, 0,
                               M>::compute(Y, B);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/**
 * @brief Concatenates two sparse matrices horizontally and returns the result
 * as a new matrix.
 *
 * This function takes two sparse matrices A and B, concatenates them
 * horizontally, and returns a new matrix Y containing the result. The operation
 * is performed using compiled sparse operations to efficiently copy the values.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in both matrices A and B.
 * @tparam N The number of columns in matrix A.
 * @tparam L The number of columns in matrix B.
 * @param A The first input matrix (left part of the result).
 * @param B The second input matrix (right part of the result).
 * @return A new matrix Y with M cols and N + L rows, containing the
 * concatenated result.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename CSRIndices_A, typename CSRPointers_A, typename CSRIndices_B,
          typename CSRPointers_B>
inline auto concatenate_horizontally(
    const CompiledSparseMatrix<T, M, N, CSRIndices_A, CSRPointers_A> &A,
    const CompiledSparseMatrix<T, M, L, CSRIndices_B, CSRPointers_B> &B) ->
    typename ConcatenateHorizontally::SparseAndSparse<
        T, M, N, L, CSRIndices_A, CSRPointers_A, CSRIndices_B,
        CSRPointers_B>::Y_Type {

  typename ConcatenateHorizontally::SparseAndSparse<T, M, N, L, CSRIndices_A,
                                                    CSRPointers_A, CSRIndices_B,
                                                    CSRPointers_B>::Y_Type Y;

  Base::Matrix::update_horizontally_concatenated_matrix(Y, A, B);

  return Y;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_CONCATENATE_HPP_
