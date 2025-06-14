/**
 * @file python_numpy_concatenate.hpp
 * @brief Provides template utilities for concatenating matrices in a manner
 * similar to NumPy's concatenate functionality.
 *
 * This header defines a set of template functions and type aliases within the
 * PythonNumpy namespace to perform vertical and horizontal concatenation of
 * matrices. The concatenation supports various matrix types, including dense,
 * diagonal, and sparse matrices, and handles all combinations of these types.
 * The resulting matrix type is deduced at compile time based on the input
 * types, ensuring efficient and type-safe concatenation.
 *
 * The main features include:
 * - Vertical concatenation (stacking matrices row-wise) for all combinations of
 * dense, diagonal, and sparse matrices.
 * - Horizontal concatenation (stacking matrices column-wise) for all
 * combinations of dense, diagonal, and sparse matrices.
 * - Type deduction utilities for determining the result type of concatenation
 * at compile time.
 * - Block 2x2 concatenation, allowing construction of a larger matrix from four
 * submatrices.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __PYTHON_NUMPY_CONCATENATE_HPP__
#define __PYTHON_NUMPY_CONCATENATE_HPP__

#include "base_matrix.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_templates.hpp"

#include <cstddef>
#include <utility>

namespace PythonNumpy {

/* Concatenate vertically */

/**
 * @brief Updates a matrix Y to be the vertical concatenation of matrices A and
 * B.
 *
 * This function takes two input matrices, A and B, with the same number of
 * columns (N) and concatenates them vertically to form the output matrix Y. The
 * resulting matrix Y will have (M + P) rows and N columns, where M and P are
 * the number of rows in A and B, respectively.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A, B, and Y.
 * @tparam P The number of columns in matrix B.
 * @param[out] Y The output matrix to store the vertically concatenated result
 * (size: (M + P) x N).
 * @param[in] A The first input matrix (size: M x N).
 * @param[in] B The second input matrix (size: P x N).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline void
update_vertically_concatenated_matrix(Matrix<DefDense, T, (M + P), N> &Y,
                                      const Matrix<DefDense, T, M, N> &A,
                                      const Matrix<DefDense, T, P, N> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

/**
 * @brief Concatenates two matrices A and B vertically to form a new matrix.
 *
 * This function takes two input matrices, A and B, with the same number of
 * columns (N) and concatenates them vertically to form a new matrix. The
 * resulting matrix will have (M + P) rows and N columns, where M and P are the
 * number of rows in A and B, respectively.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A, B, and the result.
 * @tparam P The number of columns in matrix B.
 * @param[in] A The first input matrix (size: M x N).
 * @param[in] B The second input matrix (size: P x N).
 * @return A new matrix containing the vertically concatenated result (size:
 * (M + P) x N).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline auto concatenate_vertically(const Matrix<DefDense, T, M, N> &A,
                                   const Matrix<DefDense, T, P, N> &B)
    -> Matrix<DefDense, T, (M + P), N> {

  return Matrix<DefDense, T, (M + P), N>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the vertical concatenation of a dense
 * matrix A and a diagonal matrix B.
 *
 * This function takes a dense matrix A and a diagonal matrix B, and updates the
 * sparse matrix Y to be their vertical concatenation. The resulting matrix Y
 * will have (M + N) rows and N columns, where M is the number of rows in A and
 * N is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @param[out] Y The output sparse matrix to store the vertically concatenated
 * result (size: (M + N) x N).
 * @param[in] A The input dense matrix (size: M x N).
 * @param[in] B The input diagonal matrix (size: N x N).
 */
template <typename T, std::size_t M, std::size_t N>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + N), N,
           ConcatenateSparseAvailableVertically<DenseAvailable<M, N>,
                                                DiagAvailable<N>>> &Y,
    const Matrix<DefDense, T, M, N> &A, const Matrix<DefDiag, T, N> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

/**
 * @brief Concatenates a dense matrix A and a diagonal matrix B vertically to
 * form a new sparse matrix.
 *
 * This function takes a dense matrix A and a diagonal matrix B, and returns a
 * new sparse matrix that is their vertical concatenation. The resulting matrix
 * will have (M + N) rows and N columns, where M is the number of rows in A and
 * N is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @param[in] A The input dense matrix (size: M x N).
 * @param[in] B The input diagonal matrix (size: N x N).
 * @return A new sparse matrix containing the vertically concatenated result
 * (size: (M + N) x N).
 */
template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_vertically(const Matrix<DefDense, T, M, N> &A,
                                   const Matrix<DefDiag, T, N> &B)
    -> Matrix<DefSparse, T, (M + N), N,
              ConcatenateSparseAvailableVertically<DenseAvailable<M, N>,
                                                   DiagAvailable<N>>> {

  return Matrix<DefSparse, T, (M + N), N,
                ConcatenateSparseAvailableVertically<DenseAvailable<M, N>,
                                                     DiagAvailable<N>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the vertical concatenation of a dense
 * matrix A and a sparse matrix B.
 *
 * This function takes a dense matrix A and a sparse matrix B, and updates the
 * sparse matrix Y to be their vertical concatenation. The resulting matrix Y
 * will have (M + P) rows and N columns, where M is the number of rows in A and
 * P is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @tparam P The number of columns in matrix B.
 * @param[out] Y The output sparse matrix to store the vertically concatenated
 * result (size: (M + P) x N).
 * @param[in] A The input dense matrix (size: M x N).
 * @param[in] B The input sparse matrix (size: P x N).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename SparseAvailable_B>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + P), N,
           ConcatenateSparseAvailableVertically<DenseAvailable<M, N>,
                                                SparseAvailable_B>> &Y,
    const Matrix<DefDense, T, M, N> &A,
    const Matrix<DefSparse, T, P, N, SparseAvailable_B> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

/**
 * @brief Concatenates a dense matrix A and a sparse matrix B vertically to form
 * a new sparse matrix.
 *
 * This function takes a dense matrix A and a sparse matrix B, and returns a new
 * sparse matrix that is their vertical concatenation. The resulting matrix will
 * have (M + P) rows and N columns, where M is the number of rows in A and P is
 * the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @tparam P The number of columns in matrix B.
 * @param[in] A The input dense matrix (size: M x N).
 * @param[in] B The input sparse matrix (size: P x N).
 * @return A new sparse matrix containing the vertically concatenated result
 * (size: (M + P) x N).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename SparseAvailable_B>
inline auto
concatenate_vertically(const Matrix<DefDense, T, M, N> &A,
                       const Matrix<DefSparse, T, P, N, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, (M + P), N,
              ConcatenateSparseAvailableVertically<DenseAvailable<M, N>,
                                                   SparseAvailable_B>> {

  return Matrix<DefSparse, T, (M + P), N,
                ConcatenateSparseAvailableVertically<DenseAvailable<M, N>,
                                                     SparseAvailable_B>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the vertical concatenation of a
 * diagonal matrix A and a dense matrix B.
 *
 * This function takes a diagonal matrix A and a dense matrix B, and updates the
 * sparse matrix Y to be their vertical concatenation. The resulting matrix Y
 * will have (M + P) rows and M columns, where M is the number of rows in A and
 * P is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam P The number of columns in matrix B.
 * @param[out] Y The output sparse matrix to store the vertically concatenated
 * result (size: (M + P) x M).
 * @param[in] A The input diagonal matrix (size: M x M).
 * @param[in] B The input dense matrix (size: P x M).
 */
template <typename T, std::size_t M, std::size_t P>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + P), M,
           ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                DenseAvailable<P, M>>> &Y,
    const Matrix<DefDiag, T, M> &A, const Matrix<DefDense, T, P, M> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

/**
 * @brief Concatenates a diagonal matrix A and a dense matrix B vertically to
 * form a new sparse matrix.
 *
 * This function takes a diagonal matrix A and a dense matrix B, and returns a
 * new sparse matrix that is their vertical concatenation. The resulting matrix
 * will have (M + P) rows and M columns, where M is the number of rows in A and
 * P is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam P The number of columns in matrix B.
 * @param[in] A The input diagonal matrix (size: M x M).
 * @param[in] B The input dense matrix (size: P x M).
 * @return A new sparse matrix containing the vertically concatenated result
 * (size: (M + P) x M).
 */
template <typename T, std::size_t M, std::size_t P>
inline auto concatenate_vertically(const Matrix<DefDiag, T, M> &A,
                                   const Matrix<DefDense, T, P, M> &B)
    -> Matrix<DefSparse, T, (M + P), M,
              ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                   DenseAvailable<P, M>>> {

  return Matrix<DefSparse, T, (M + P), M,
                ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                     DenseAvailable<P, M>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the vertical concatenation of a
 * diagonal matrix A and a diagonal matrix B.
 *
 * This function takes two diagonal matrices A and B, and updates the sparse
 * matrix Y to be their vertical concatenation. The resulting matrix Y will have
 * (2 * M) rows and M columns, where M is the number of rows in both A and B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrices A and B.
 * @param[out] Y The output sparse matrix to store the vertically concatenated
 * result (size: (2 * M) x M).
 * @param[in] A The input diagonal matrix (size: M x M).
 * @param[in] B The input diagonal matrix (size: M x M).
 */
template <typename T, std::size_t M>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (2 * M), M,
           ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                DiagAvailable<M>>> &Y,
    const Matrix<DefDiag, T, M> &A, const Matrix<DefDiag, T, M> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

/**
 * @brief Concatenates two diagonal matrices A and B vertically to form a new
 * sparse matrix.
 *
 * This function takes two diagonal matrices A and B, and returns a new sparse
 * matrix that is their vertical concatenation. The resulting matrix will have
 * (2 * M) rows and M columns, where M is the number of rows in both A and B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrices A and B.
 * @param[in] A The input diagonal matrix (size: M x M).
 * @param[in] B The input diagonal matrix (size: M x M).
 * @return A new sparse matrix containing the vertically concatenated result
 * (size: (2 * M) x M).
 */
template <typename T, std::size_t M>
inline auto concatenate_vertically(const Matrix<DefDiag, T, M> &A,
                                   const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, (2 * M), M,
              ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                   DiagAvailable<M>>> {

  return Matrix<
      DefSparse, T, (2 * M), M,
      ConcatenateSparseAvailableVertically<DiagAvailable<M>, DiagAvailable<M>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the vertical concatenation of a
 * diagonal matrix A and a sparse matrix B.
 *
 * This function takes a diagonal matrix A and a sparse matrix B, and updates
 * the sparse matrix Y to be their vertical concatenation. The resulting matrix
 * Y will have (M + P) rows and M columns, where M is the number of rows in A
 * and P is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam P The number of columns in matrix B.
 * @param[out] Y The output sparse matrix to store the vertically concatenated
 * result (size: (M + P) x M).
 * @param[in] A The input diagonal matrix (size: M x M).
 * @param[in] B The input sparse matrix (size: P x M).
 */
template <typename T, std::size_t M, std::size_t P, typename SparseAvailable_B>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + P), M,
           ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                SparseAvailable_B>> &Y,
    const Matrix<DefDiag, T, M> &A,
    const Matrix<DefSparse, T, P, M, SparseAvailable_B> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

/**
 * @brief Concatenates a diagonal matrix A and a sparse matrix B vertically to
 * form a new sparse matrix.
 *
 * This function takes a diagonal matrix A and a sparse matrix B, and returns a
 * new sparse matrix that is their vertical concatenation. The resulting matrix
 * will have (M + P) rows and M columns, where M is the number of rows in A and
 * P is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam P The number of columns in matrix B.
 * @param[in] A The input diagonal matrix (size: M x M).
 * @param[in] B The input sparse matrix (size: P x M).
 * @return A new sparse matrix containing the vertically concatenated result
 * (size: (M + P) x M).
 */
template <typename T, std::size_t M, std::size_t P, typename SparseAvailable_B>
inline auto
concatenate_vertically(const Matrix<DefDiag, T, M> &A,
                       const Matrix<DefSparse, T, P, M, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, (M + P), M,
              ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                   SparseAvailable_B>> {

  return Matrix<DefSparse, T, (M + P), M,
                ConcatenateSparseAvailableVertically<DiagAvailable<M>,
                                                     SparseAvailable_B>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the vertical concatenation of two
 * sparse matrices A and B.
 *
 * This function takes two sparse matrices A and B, and updates the sparse
 * matrix Y to be their vertical concatenation. The resulting matrix Y will have
 * (M + P) rows and N columns, where M is the number of rows in A and P is the
 * number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @tparam P The number of columns in matrix B.
 * @param[out] Y The output sparse matrix to store the vertically concatenated
 * result (size: (M + P) x N).
 * @param[in] A The input sparse matrix (size: M x N).
 * @param[in] B The input sparse matrix (size: P x N).
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A,
          std::size_t P>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + P), N,
           ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                DenseAvailable<P, N>>> &Y,
    const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
    const Matrix<DefDense, T, P, N> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

/**
 * @brief Concatenates a sparse matrix A and a dense matrix B vertically to form
 * a new sparse matrix.
 *
 * This function takes a sparse matrix A and a dense matrix B, and returns a new
 * sparse matrix that is their vertical concatenation. The resulting matrix will
 * have (M + P) rows and N columns, where M is the number of rows in A and P is
 * the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @tparam P The number of columns in matrix B.
 * @param[in] A The input sparse matrix (size: M x N).
 * @param[in] B The input dense matrix (size: P x N).
 * @return A new sparse matrix containing the vertically concatenated result
 * (size: (M + P) x N).
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A,
          std::size_t P>
inline auto
concatenate_vertically(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                       const Matrix<DefDense, T, P, N> &B)
    -> Matrix<DefSparse, T, (M + P), N,
              ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                   DenseAvailable<P, N>>> {

  return Matrix<DefSparse, T, (M + P), N,
                ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                     DenseAvailable<P, N>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the vertical concatenation of a sparse
 * matrix A and a diagonal matrix B.
 *
 * This function takes a sparse matrix A and a diagonal matrix B, and updates
 * the sparse matrix Y to be their vertical concatenation. The resulting matrix
 * Y will have (M + N) rows and N columns, where M is the number of rows in A
 * and N is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @param[out] Y The output sparse matrix to store the vertically concatenated
 * result (size: (M + N) x N).
 * @param[in] A The input sparse matrix (size: M x N).
 * @param[in] B The input diagonal matrix (size: N x N).
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + N), N,
           ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                DiagAvailable<N>>> &Y,
    const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
    const Matrix<DefDiag, T, N> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

/**
 * @brief Concatenates a sparse matrix A and a diagonal matrix B vertically to
 * form a new sparse matrix.
 *
 * This function takes a sparse matrix A and a diagonal matrix B, and returns a
 * new sparse matrix that is their vertical concatenation. The resulting matrix
 * will have (M + N) rows and N columns, where M is the number of rows in A and
 * N is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @param[in] A The input sparse matrix (size: M x N).
 * @param[in] B The input diagonal matrix (size: N x N).
 * @return A new sparse matrix containing the vertically concatenated result
 * (size: (M + N) x N).
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A>
inline auto
concatenate_vertically(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                       const Matrix<DefDiag, T, N> &B)
    -> Matrix<DefSparse, T, (M + N), N,
              ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                   DiagAvailable<N>>> {

  return Matrix<DefSparse, T, (M + N), N,
                ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                     DiagAvailable<N>>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the vertical concatenation of two
 * sparse matrices A and B.
 *
 * This function takes two sparse matrices A and B, and updates the sparse
 * matrix Y to be their vertical concatenation. The resulting matrix Y will have
 * (M + P) rows and N columns, where M is the number of rows in A and P is the
 * number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @tparam P The number of columns in matrix B.
 * @param[out] Y The output sparse matrix to store the vertically concatenated
 * result (size: (M + P) x N).
 * @param[in] A The input sparse matrix (size: M x N).
 * @param[in] B The input sparse matrix (size: P x N).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline void update_vertically_concatenated_matrix(
    Matrix<DefSparse, T, (M + P), N,
           ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                SparseAvailable_B>> &Y,
    const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
    const Matrix<DefSparse, T, P, N, SparseAvailable_B> &B) {

  Base::Matrix::update_vertically_concatenated_matrix(Y.matrix, A.matrix,
                                                      B.matrix);
}

/**
 * @brief Concatenates two sparse matrices A and B vertically to form a new
 * sparse matrix.
 *
 * This function takes two sparse matrices A and B, and returns a new sparse
 * matrix that is their vertical concatenation. The resulting matrix will have
 * (M + P) rows and N columns, where M is the number of rows in A and P is the
 * number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @tparam P The number of columns in matrix B.
 * @param[in] A The input sparse matrix (size: M x N).
 * @param[in] B The input sparse matrix (size: P x N).
 * @return A new sparse matrix containing the vertically concatenated result
 * (size: (M + P) x N).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t P,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline auto
concatenate_vertically(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                       const Matrix<DefSparse, T, P, N, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, (M + P), N,
              ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                   SparseAvailable_B>> {

  return Matrix<DefSparse, T, (M + P), N,
                ConcatenateSparseAvailableVertically<SparseAvailable_A,
                                                     SparseAvailable_B>>(
      Base::Matrix::concatenate_vertically(A.matrix, B.matrix));
}

/* Concatenate horizontally */

/**
 * @brief Updates a matrix Y to be the horizontal concatenation of matrices A
 * and B.
 *
 * This function takes two input matrices, A and B, with the same number of rows
 * (M) and concatenates them horizontally to form the output matrix Y. The
 * resulting matrix Y will have M rows and (N + L) columns, where N and L are
 * the number of columns in A and B, respectively.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrices A, B, and Y.
 * @tparam N The number of rows in matrix A.
 * @tparam L The number of rows in matrix B.
 * @param[out] Y The output matrix to store the horizontally concatenated result
 * (size: M x (N + L)).
 * @param[in] A The first input matrix (size: M x N).
 * @param[in] B The second input matrix (size: M x L).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L>
inline void
update_horizontally_concatenated_matrix(Matrix<DefDense, T, M, (N + L)> &Y,
                                        const Matrix<DefDense, T, M, N> &A,
                                        const Matrix<DefDense, T, M, L> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

/**
 * @brief Concatenates two matrices A and B horizontally to form a new matrix.
 *
 * This function takes two input matrices, A and B, with the same number of rows
 * (M) and concatenates them horizontally to form a new matrix. The resulting
 * matrix will have M rows and (N + L) columns, where N and L are the number of
 * columns in A and B, respectively.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrices A, B, and the result.
 * @tparam N The number of rows in matrix A.
 * @tparam L The number of rows in matrix B.
 * @param[in] A The first input matrix (size: M x N).
 * @param[in] B The second input matrix (size: M x L).
 * @return A new matrix containing the horizontally concatenated result (size:
 * M x (N + L)).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L>
inline auto concatenate_horizontally(const Matrix<DefDense, T, M, N> &A,
                                     const Matrix<DefDense, T, M, L> &B)
    -> Matrix<DefDense, T, M, (N + L)> {

  return Matrix<DefDense, T, M, (N + L)>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the horizontal concatenation of a
 * dense matrix A and a diagonal matrix B.
 *
 * This function takes a dense matrix A and a diagonal matrix B, and updates the
 * sparse matrix Y to be their horizontal concatenation. The resulting matrix Y
 * will have M rows and (M + N) columns, where M is the number of rows in A and
 * N is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @param[out] Y The output sparse matrix to store the horizontally concatenated
 * result (size: M x (M + N)).
 * @param[in] A The input dense matrix (size: M x N).
 * @param[in] B The input diagonal matrix (size: M x M).
 */
template <typename T, std::size_t M, std::size_t N>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (M + N),
           ConcatenateSparseAvailableHorizontally<DenseAvailable<M, N>,
                                                  DiagAvailable<M>>> &Y,
    const Matrix<DefDense, T, M, N> &A, const Matrix<DefDiag, T, M> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

/**
 * @brief Concatenates a dense matrix A and a diagonal matrix B horizontally to
 * form a new sparse matrix.
 *
 * This function takes a dense matrix A and a diagonal matrix B, and returns a
 * new sparse matrix that is their horizontal concatenation. The resulting
 * matrix will have M rows and (M + N) columns, where M is the number of rows in
 * A and N is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @param[in] A The input dense matrix (size: M x N).
 * @param[in] B The input diagonal matrix (size: M x M).
 * @return A new sparse matrix containing the horizontally concatenated result
 * (size: M x (M + N)).
 */
template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_horizontally(const Matrix<DefDense, T, M, N> &A,
                                     const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, M, (M + N),
              ConcatenateSparseAvailableHorizontally<DenseAvailable<M, N>,
                                                     DiagAvailable<M>>> {

  return Matrix<DefSparse, T, M, (M + N),
                ConcatenateSparseAvailableHorizontally<DenseAvailable<M, N>,
                                                       DiagAvailable<M>>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the horizontal concatenation of a
 * diagonal matrix A and a dense matrix B.
 *
 * This function takes a diagonal matrix A and a dense matrix B, and updates the
 * sparse matrix Y to be their horizontal concatenation. The resulting matrix Y
 * will have M rows and (M + N) columns, where M is the number of rows in A and
 * N is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @param[out] Y The output sparse matrix to store the horizontally concatenated
 * result (size: M x (M + N)).
 * @param[in] A The input diagonal matrix (size: M x M).
 * @param[in] B The input dense matrix (size: M x N).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_B>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (N + L),
           ConcatenateSparseAvailableHorizontally<DenseAvailable<M, N>,
                                                  SparseAvailable_B>> &Y,
    const Matrix<DefDense, T, M, N> &A,
    const Matrix<DefSparse, T, M, L, SparseAvailable_B> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

/**
 * @brief Concatenates a dense matrix A and a sparse matrix B horizontally to
 * form a new sparse matrix.
 *
 * This function takes a dense matrix A and a sparse matrix B, and returns a new
 * sparse matrix that is their horizontal concatenation. The resulting matrix
 * will have M rows and (N + L) columns, where M is the number of rows in A and
 * L is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @tparam L The number of columns in matrix B.
 * @param[in] A The input dense matrix (size: M x N).
 * @param[in] B The input sparse matrix (size: M x L).
 * @return A new sparse matrix containing the horizontally concatenated result
 * (size: M x (N + L)).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_B>
inline auto
concatenate_horizontally(const Matrix<DefDense, T, M, N> &A,
                         const Matrix<DefSparse, T, M, L, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, M, (N + L),
              ConcatenateSparseAvailableHorizontally<DenseAvailable<M, N>,
                                                     SparseAvailable_B>> {

  return Matrix<DefSparse, T, M, (N + L),
                ConcatenateSparseAvailableHorizontally<DenseAvailable<M, N>,
                                                       SparseAvailable_B>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the horizontal concatenation of a
 * diagonal matrix A and a diagonal matrix B.
 *
 * This function takes two diagonal matrices A and B, and updates the sparse
 * matrix Y to be their horizontal concatenation. The resulting matrix Y will
 * have M rows and (2 * M) columns, where M is the number of rows in both A and
 * B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrices A and B.
 * @param[out] Y The output sparse matrix to store the horizontally concatenated
 * result (size: M x (2 * M)).
 * @param[in] A The input diagonal matrix (size: M x M).
 * @param[in] B The input diagonal matrix (size: M x M).
 */
template <typename T, std::size_t M, std::size_t N>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (M + N),
           ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                  DenseAvailable<M, N>>> &Y,
    const Matrix<DefDiag, T, M> &A, const Matrix<DefDense, T, M, N> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

/**
 * @brief Concatenates a diagonal matrix A and a dense matrix B horizontally to
 * form a new sparse matrix.
 *
 * This function takes a diagonal matrix A and a dense matrix B, and returns a
 * new sparse matrix that is their horizontal concatenation. The resulting
 * matrix will have M rows and (M + N) columns, where M is the number of rows in
 * A and N is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @param[in] A The input diagonal matrix (size: M x M).
 * @param[in] B The input dense matrix (size: M x N).
 * @return A new sparse matrix containing the horizontally concatenated result
 * (size: M x (M + N)).
 */
template <typename T, std::size_t M, std::size_t N>
inline auto concatenate_horizontally(const Matrix<DefDiag, T, M> &A,
                                     const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefSparse, T, M, (M + N),
              ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                     DenseAvailable<M, N>>> {

  return Matrix<DefSparse, T, M, (M + N),
                ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                       DenseAvailable<M, N>>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the horizontal concatenation of a
 * diagonal matrix A and a diagonal matrix B.
 *
 * This function takes two diagonal matrices A and B, and updates the sparse
 * matrix Y to be their horizontal concatenation. The resulting matrix Y will
 * have M rows and (2 * M) columns, where M is the number of rows in both A and
 * B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrices A and B.
 * @param[out] Y The output sparse matrix to store the horizontally concatenated
 * result (size: M x (2 * M)).
 * @param[in] A The input diagonal matrix (size: M x M).
 * @param[in] B The input diagonal matrix (size: M x M).
 */
template <typename T, std::size_t M>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (2 * M),
           ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                  DiagAvailable<M>>> &Y,
    const Matrix<DefDiag, T, M> &A, const Matrix<DefDiag, T, M> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

/**
 * @brief Concatenates two diagonal matrices A and B horizontally to form a new
 * sparse matrix.
 *
 * This function takes two diagonal matrices A and B, and returns a new sparse
 * matrix that is their horizontal concatenation. The resulting matrix will have
 * M rows and (2 * M) columns, where M is the number of rows in both A and B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrices A and B.
 * @param[in] A The input diagonal matrix (size: M x M).
 * @param[in] B The input diagonal matrix (size: M x M).
 * @return A new sparse matrix containing the horizontally concatenated result
 * (size: M x (2 * M)).
 */
template <typename T, std::size_t M>
inline auto concatenate_horizontally(const Matrix<DefDiag, T, M> &A,
                                     const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, M, (2 * M),
              ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                     DiagAvailable<M>>> {

  return Matrix<DefSparse, T, M, (2 * M),
                ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                       DiagAvailable<M>>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the horizontal concatenation of a
 * diagonal matrix A and a sparse matrix B.
 *
 * This function takes a diagonal matrix A and a sparse matrix B, and updates
 * the sparse matrix Y to be their horizontal concatenation. The resulting
 * matrix Y will have M rows and (M + N) columns, where M is the number of rows
 * in A and N is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @param[out] Y The output sparse matrix to store the horizontally concatenated
 * result (size: M x (M + N)).
 * @param[in] A The input diagonal matrix (size: M x M).
 * @param[in] B The input sparse matrix (size: M x N).
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_B>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (M + N),
           ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                  SparseAvailable_B>> &Y,
    const Matrix<DefDiag, T, M> &A,
    const Matrix<DefSparse, T, M, N, SparseAvailable_B> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

/**
 * @brief Concatenates a diagonal matrix A and a sparse matrix B horizontally to
 * form a new sparse matrix.
 *
 * This function takes a diagonal matrix A and a sparse matrix B, and returns a
 * new sparse matrix that is their horizontal concatenation. The resulting
 * matrix will have M rows and (M + N) columns, where M is the number of rows in
 * A and N is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @param[in] A The input diagonal matrix (size: M x M).
 * @param[in] B The input sparse matrix (size: M x N).
 * @return A new sparse matrix containing the horizontally concatenated result
 * (size: M x (M + N)).
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_B>
inline auto
concatenate_horizontally(const Matrix<DefDiag, T, M> &A,
                         const Matrix<DefSparse, T, M, N, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, M, (M + N),
              ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                     SparseAvailable_B>> {

  return Matrix<DefSparse, T, M, (M + N),
                ConcatenateSparseAvailableHorizontally<DiagAvailable<M>,
                                                       SparseAvailable_B>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the horizontal concatenation of a
 * sparse matrix A and a dense matrix B.
 *
 * This function takes a sparse matrix A and a dense matrix B, and updates the
 * sparse matrix Y to be their horizontal concatenation. The resulting matrix Y
 * will have M rows and (N + L) columns, where M is the number of rows in A and
 * L is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @tparam L The number of roes in matrix A.
 * @param[out] Y The output sparse matrix to store the horizontally concatenated
 * result (size: M x (N + L)).
 * @param[in] A The input sparse matrix (size: M x N).
 * @param[in] B The input dense matrix (size: M x L).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_A>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (N + L),
           ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                  DenseAvailable<M, N>>> &Y,
    const Matrix<DefSparse, T, M, L, SparseAvailable_A> &A,
    const Matrix<DefDense, T, M, N> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

/**
 * @brief Concatenates a sparse matrix A and a dense matrix B horizontally to
 * form a new sparse matrix.
 *
 * This function takes a sparse matrix A and a dense matrix B, and returns a new
 * sparse matrix that is their horizontal concatenation. The resulting matrix
 * will have M rows and (N + L) columns, where M is the number of rows in A and
 * L is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @tparam L The number of rows in matrix B.
 * @param[in] A The input sparse matrix (size: M x L).
 * @param[in] B The input dense matrix (size: M x N).
 * @return A new sparse matrix containing the horizontally concatenated result
 * (size: M x (N + L)).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_A>
inline auto
concatenate_horizontally(const Matrix<DefSparse, T, M, L, SparseAvailable_A> &A,
                         const Matrix<DefDense, T, M, N> &B)
    -> Matrix<DefSparse, T, M, (N + L),
              ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                     DenseAvailable<M, N>>> {

  return Matrix<DefSparse, T, M, (N + L),
                ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                       DenseAvailable<M, N>>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the horizontal concatenation of a
 * sparse matrix A and a diagonal matrix B.
 *
 * This function takes a sparse matrix A and a diagonal matrix B, and updates
 * the sparse matrix Y to be their horizontal concatenation. The resulting
 * matrix Y will have M rows and (M + N) columns, where M is the number of rows
 * in A and N is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @param[out] Y The output sparse matrix to store the horizontally concatenated
 * result (size: M x (M + N)).
 * @param[in] A The input sparse matrix (size: M x N).
 * @param[in] B The input diagonal matrix (size: M x M).
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (M + N),
           ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                  DiagAvailable<M>>> &Y,
    const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
    const Matrix<DefDiag, T, M> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

/**
 * @brief Concatenates a sparse matrix A and a diagonal matrix B horizontally to
 * form a new sparse matrix.
 *
 * This function takes a sparse matrix A and a diagonal matrix B, and returns a
 * new sparse matrix that is their horizontal concatenation. The resulting
 * matrix will have M rows and (M + N) columns, where M is the number of rows in
 * A and N is the number of rows in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrix A.
 * @tparam N The number of rows in matrices A and B.
 * @param[in] A The input sparse matrix (size: M x N).
 * @param[in] B The input diagonal matrix (size: M x M).
 * @return A new sparse matrix containing the horizontally concatenated result
 * (size: M x (M + N)).
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable_A>
inline auto
concatenate_horizontally(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                         const Matrix<DefDiag, T, M> &B)
    -> Matrix<DefSparse, T, M, (M + N),
              ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                     DiagAvailable<M>>> {

  return Matrix<DefSparse, T, M, (M + N),
                ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                       DiagAvailable<M>>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

/**
 * @brief Updates a sparse matrix Y to be the horizontal concatenation of two
 * sparse matrices A and B.
 *
 * This function takes two sparse matrices A and B, and updates the sparse
 * matrix Y to be their horizontal concatenation. The resulting matrix Y will
 * have M rows and (N + L) columns, where M is the number of rows in A and B, N
 * is the number of columns in A, and L is the number of columns in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrices A, B, and Y.
 * @tparam N The number of rows in matrix A.
 * @tparam L The number of rows in matrix B.
 * @param[out] Y The output sparse matrix to store the horizontally concatenated
 * result (size: M x (N + L)).
 * @param[in] A The input sparse matrix (size: M x N).
 * @param[in] B The input sparse matrix (size: M x L).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline void update_horizontally_concatenated_matrix(
    Matrix<DefSparse, T, M, (N + L),
           ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                  SparseAvailable_B>> &Y,
    const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
    const Matrix<DefSparse, T, M, L, SparseAvailable_B> &B) {

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A.matrix,
                                                        B.matrix);
}

/**
 * @brief Concatenates two sparse matrices A and B horizontally to form a new
 * sparse matrix.
 *
 * This function takes two sparse matrices A and B, and returns a new sparse
 * matrix that is their horizontal concatenation. The resulting matrix will have
 * M rows and (N + L) columns, where M is the number of rows in A and B, N is
 * the number of columns in A, and L is the number of columns in B.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of columns in matrices A, B, and the result.
 * @tparam N The number of rows in matrix A.
 * @tparam L The number of rows in matrix B.
 * @param[in] A The input sparse matrix (size: M x N).
 * @param[in] B The input sparse matrix (size: M x L).
 * @return A new sparse matrix containing the horizontally concatenated result
 * (size: M x (N + L)).
 */
template <typename T, std::size_t M, std::size_t N, std::size_t L,
          typename SparseAvailable_A, typename SparseAvailable_B>
inline auto
concatenate_horizontally(const Matrix<DefSparse, T, M, N, SparseAvailable_A> &A,
                         const Matrix<DefSparse, T, M, L, SparseAvailable_B> &B)
    -> Matrix<DefSparse, T, M, (N + L),
              ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                     SparseAvailable_B>> {

  return Matrix<DefSparse, T, M, (N + L),
                ConcatenateSparseAvailableHorizontally<SparseAvailable_A,
                                                       SparseAvailable_B>>(
      Base::Matrix::concatenate_horizontally(A.matrix, B.matrix));
}

/* Concatenation Type */
template <typename A_Type, typename B_Type>
using ConcatenateVertically_Type = decltype(concatenate_vertically(
    std::declval<A_Type>(), std::declval<B_Type>()));

template <typename A_Type, typename B_Type>
using ConcatenateHorizontally_Type = decltype(concatenate_horizontally(
    std::declval<A_Type>(), std::declval<B_Type>()));

/* Concatenate block Type */
template <typename A_Type, typename B_Type, typename C_Type, typename D_Type>
using ConcatenateBlock2X2_Type =
    ConcatenateHorizontally_Type<ConcatenateVertically_Type<A_Type, C_Type>,
                                 ConcatenateVertically_Type<B_Type, D_Type>>;

/**
 * @brief Updates a concatenated block 2x2 matrix Y with matrices A, B, C, and
 * D.
 *
 * This function takes four matrices A, B, C, and D, and updates the
 * concatenated block 2x2 matrix Y to be their concatenation. The resulting
 * matrix Y will have (M + P) rows and (N + L) columns, where M is the number of
 * rows in A and C, N is the number of columns in A and B, P is the number of
 * rows in B and D, and L is the number of columns in B and D.
 *
 * @tparam A_Type The type of matrix A.
 * @tparam B_Type The type of matrix B.
 * @tparam C_Type The type of matrix C.
 * @tparam D_Type The type of matrix D.
 * @param[out] Y The output concatenated block 2x2 matrix to store the result.
 * @param[in] A The input matrix A.
 * @param[in] B The input matrix B.
 * @param[in] C The input matrix C.
 * @param[in] D The input matrix D.
 */
template <typename A_Type, typename B_Type, typename C_Type, typename D_Type>
inline void update_block_2x2_concatenated_matrix(
    ConcatenateBlock2X2_Type<A_Type, B_Type, C_Type, D_Type> &Y,
    const A_Type &A, const B_Type &B, const C_Type &C, const D_Type &D) {

  ConcatenateVertically_Type<A_Type, C_Type> A_v_C =
      concatenate_vertically(A, C);
  ConcatenateVertically_Type<B_Type, D_Type> B_v_D =
      concatenate_vertically(B, D);

  Base::Matrix::update_horizontally_concatenated_matrix(Y.matrix, A_v_C.matrix,
                                                        B_v_D.matrix);
}

/**
 * @brief Concatenates four matrices A, B, C, and D into a block 2x2 matrix.
 *
 * This function takes four matrices A, B, C, and D, and returns a new
 * concatenated block 2x2 matrix that is their concatenation. The resulting
 * matrix will have (M + P) rows and (N + L) columns, where M is the number of
 * rows in A and C, N is the number of columns in A and B, P is the number of
 * rows in B and D, and L is the number of columns in B and D.
 *
 * @tparam A_Type The type of matrix A.
 * @tparam B_Type The type of matrix B.
 * @tparam C_Type The type of matrix C.
 * @tparam D_Type The type of matrix D.
 * @param[in] A The input matrix A.
 * @param[in] B The input matrix B.
 * @param[in] C The input matrix C.
 * @param[in] D The input matrix D.
 * @return A new concatenated block 2x2 matrix containing the result.
 */
template <typename A_Type, typename B_Type, typename C_Type, typename D_Type>
inline auto concatenate_block_2x2(const A_Type &A, const B_Type &B,
                                  const C_Type &C, const D_Type &D)
    -> ConcatenateBlock2X2_Type<A_Type, B_Type, C_Type, D_Type> {

  ConcatenateBlock2X2_Type<A_Type, B_Type, C_Type, D_Type> Y;

  update_block_2x2_concatenated_matrix(Y, A, B, C, D);

  return Y;
}

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_CONCATENATE_HPP__
