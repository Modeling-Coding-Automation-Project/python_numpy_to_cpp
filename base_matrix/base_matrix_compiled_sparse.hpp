/**
 * @file base_matrix_compiled_sparse.hpp
 * @brief Provides a highly generic, template-based implementation of sparse
 * matrix operations for fixed-size matrices.
 *
 * This file defines the `Base::Matrix` namespace, which contains the
 * `CompiledSparseMatrix` class template and a suite of supporting functions and
 * meta-programming utilities for manipulating sparse matrices in a highly
 * efficient and type-safe manner. The implementation supports both standard
 * vector and fixed-size array storage, and provides compile-time and runtime
 * algorithms for:
 *   - Construction and assignment of sparse matrices
 *   - Conversion between dense and sparse representations
 *   - Element-wise and block-wise access and modification
 *   - Transposition, real/complex conversion, and diagonal operations
 *   - Efficient loop unrolling via template meta-programming for performance
 *
 * The code is designed for use in high-performance scientific computing, code
 * generation, or embedded systems where matrix sparsity patterns are known at
 * compile time.
 *
 * Classes and Main Components:
 *
 * - CompiledSparseMatrix<T, M, N, RowIndices, RowPointers>:
 *     Represents a sparse matrix with compile-time fixed dimensions and
 * sparsity pattern.
 *     - T: Element type (e.g., double, Complex<double>)
 *     - M: Number of rows
 *     - N: Number of columns
 *     - RowIndices: Type encoding the row indices of nonzero elements
 *     - RowPointers: Type encoding the start/end of each row's nonzero elements
 *     Provides constructors, copy/move semantics, element access, and static
 * creation utilities.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __BASE_MATRIX_COMPILED_SPARSE_HPP__
#define __BASE_MATRIX_COMPILED_SPARSE_HPP__

#include "base_matrix_macros.hpp"

#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_templates.hpp"
#include "base_matrix_vector.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

/*
 * @class CompiledSparseMatrix
 * @brief A fixed-size, template-based sparse matrix class for efficient storage
 * and operations.
 *
 * This class represents a sparse matrix with compile-time fixed dimensions and
 * a compile-time sparsity pattern, specified by RowIndices and RowPointers
 * types. It supports both std::vector and std::array storage for the nonzero
 * values, depending on the compile-time macro.
 *
 * Key Features:
 * - Efficient storage of only nonzero elements, with access via operator[].
 * - Copy/move constructors and assignment operators.
 * - Static creation utilities for full, dense, and diagonal matrices.
 * - Conversion to dense matrix representation.
 * - Compile-time and runtime algorithms for element access, assignment, and
 * manipulation.
 *
 * Template Parameters:
 * @tparam T           Element type (e.g., double, Complex<double>)
 * @tparam M           Number of columns (mathematical convention)
 * @tparam N           Number of rows
 * @tparam RowIndices  Type encoding the row indices of nonzero elements
 * @tparam RowPointers Type encoding the start/end of each row's nonzero
 * elements
 *
 * Usage:
 *   - For high-performance scientific computing, code generation, or embedded
 * systems where the sparsity pattern is known at compile time.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
class CompiledSparseMatrix {
public:
#ifdef __BASE_MATRIX_USE_STD_VECTOR__

  CompiledSparseMatrix() : values(RowPointers::list[M], static_cast<T>(0)) {}

  CompiledSparseMatrix(const std::initializer_list<T> &values)
      : values(values) {}

  CompiledSparseMatrix(const std::vector<T> &values) : values(values) {}

#else // __BASE_MATRIX_USE_STD_VECTOR__

  CompiledSparseMatrix() : values{} {}

  CompiledSparseMatrix(const std::initializer_list<T> &values) : values{} {

    // This may cause runtime error if the size of values is larger than
    // RowIndices::size.
    std::copy(values.begin(), values.end(), this->values.begin());
  }

  CompiledSparseMatrix(const std::array<T, RowIndices::size> &values)
      : values(values) {}

  CompiledSparseMatrix(const std::vector<T> &values) : values{} {

    // This may cause runtime error if the size of values is larger than
    // RowIndices::size.
    std::copy(values.begin(), values.end(), this->values.begin());
  }

#endif // __BASE_MATRIX_USE_STD_VECTOR__

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

  /**
   * @brief Provides access to the element at the specified index in the matrix
   * values.
   *
   * @param index The position of the element to access.
   * @return Reference to the element of type T at the given index.
   */
  T &operator[](std::size_t index) { return this->values[index]; }

  /**
   * @brief Provides constant access to the element at the specified index in
   * the matrix values.
   *
   * @param index The position of the element to access.
   * @return Constant reference to the element of type T at the given index.
   */
  const T &operator[](std::size_t index) const { return this->values[index]; }

  /**
   * @brief Creates a CompiledSparseMatrix where all elements are initialized to
   * the given value.
   *
   * This static inline function constructs and returns a CompiledSparseMatrix
   * object with all entries set to the specified value. The size of the
   * underlying storage is determined by the number of non-zero elements as
   * indicated by RowPointers::list[M].
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of rows in the matrix.
   * @tparam N            The number of columns in the matrix.
   * @tparam RowIndices   The type representing row indices.
   * @tparam RowPointers  The type representing row pointers.
   * @param value         The value to initialize all elements of the matrix
   * with.
   * @return CompiledSparseMatrix<T, M, N, RowIndices, RowPointers>
   *         A sparse matrix with all elements set to the specified value.
   */
  static inline CompiledSparseMatrix<T, M, N, RowIndices, RowPointers>
  full(const T &value) {
    CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> full(
        std::vector<T>(RowPointers::list[M], value));

    return full;
  }

  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = N;

  /* Variable */
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<T> values;
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<T, RowPointers::list[M]> values;
#endif // __BASE_MATRIX_USE_STD_VECTOR__
};

/* Output dense matrix */
namespace OutputDenseMatrix {

// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K, std::size_t Start,
          std::size_t End>
struct Loop {
  /**
   * @brief Core loop for computing the output dense matrix from a compiled
   * sparse matrix.
   *
   * This template struct recursively computes the values of the output dense
   * matrix by iterating over the non-zero elements of the compiled sparse
   * matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices.
   * @tparam RowPointers  The type representing row pointers.
   * @tparam J            Current row index in the output dense matrix.
   * @tparam K            Current column index in the output dense matrix.
   * @tparam Start        Starting index for the current row's non-zero
   * elements.
   * @tparam End          Ending index for the current row's non-zero elements.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {

    result(J, RowIndices::list[Start]) = mat.values[Start];
    Loop<T, M, N, RowIndices, RowPointers, J, K, Start + 1, End>::compute(
        mat, result);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K, std::size_t End>
struct Loop<T, M, N, RowIndices, RowPointers, J, K, End, End> {
  /**
   * @brief End of the core loop for computing the output dense matrix.
   *
   * This template struct represents the termination condition of the core loop
   * for computing the output dense matrix from a compiled sparse matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices.
   * @tparam RowPointers  The type representing row pointers.
   * @tparam J            Current row index in the output dense matrix.
   * @tparam K            Current column index in the output dense matrix.
   * @tparam End          Ending index for the current row's non-zero elements.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {

    static_cast<void>(mat);
    static_cast<void>(result);
    // End of loop, do nothing
  }
};

// Pointer loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J, std::size_t K>
struct Core {
  /**
   * @brief Core loop for computing the output dense matrix from a compiled
   * sparse matrix.
   *
   * This template struct recursively computes the values of the output dense
   * matrix by iterating over the non-zero elements of the compiled sparse
   * matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices.
   * @tparam RowPointers  The type representing row pointers.
   * @tparam J            Current row index in the output dense matrix.
   * @tparam K            Current column index in the output dense matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {

    Loop<T, M, N, RowIndices, RowPointers, J, K, RowPointers::list[J],
         RowPointers::list[J + 1]>::compute(mat, result);
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t J>
struct Row {
  /**
   * @brief Row loop for computing the output dense matrix from a compiled
   * sparse matrix.
   *
   * This template struct recursively computes the values of the output dense
   * matrix by iterating over the rows of the compiled sparse matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices.
   * @tparam RowPointers  The type representing row pointers.
   * @tparam J            Current row index in the output dense matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {

    Core<T, M, N, RowIndices, RowPointers, J, 0>::compute(mat, result);
    Row<T, M, N, RowIndices, RowPointers, J - 1>::compute(mat, result);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct Row<T, M, N, RowIndices, RowPointers, 0> {
  /**
   * @brief End of the row loop for computing the output dense matrix.
   *
   * This template struct represents the termination condition of the row loop
   * for computing the output dense matrix from a compiled sparse matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices.
   * @tparam RowPointers  The type representing row pointers.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Matrix<T, M, N> &result) {

    Core<T, M, N, RowIndices, RowPointers, 0, 0>::compute(mat, result);
  }
};

/**
 * @brief Computes the output dense matrix from a compiled sparse matrix.
 *
 * This function computes the output dense matrix by iterating over the
 * non-zero elements of the compiled sparse matrix and filling in the
 * corresponding entries in the result matrix.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices   The type representing row indices.
 * @tparam RowPointers  The type representing row pointers.
 * @param mat           The compiled sparse matrix to convert to dense format.
 * @param result        The resulting dense matrix to fill with computed values.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
        Matrix<T, M, N> &result) {

  Row<T, M, N, RowIndices, RowPointers, M - 1>::compute(mat, result);
}

} // namespace OutputDenseMatrix

/**
 * @brief Converts a compiled sparse matrix to a dense matrix.
 *
 * This function takes a compiled sparse matrix and converts it to a dense
 * matrix by iterating over the non-zero elements and filling in the
 * corresponding entries in the result matrix.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices   The type representing row indices.
 * @tparam RowPointers  The type representing row pointers.
 * @param mat           The compiled sparse matrix to convert to dense format.
 * @return Matrix<T, M, N> A dense matrix containing the values from the sparse
 * matrix.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline Matrix<T, M, N> output_dense_matrix(
    const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat) {
  Matrix<T, M, N> result;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; j++) {
    for (std::size_t k = RowPointers::list[j]; k < RowPointers::list[j + 1];
         k++) {
      result(j, RowIndices::list[k]) = mat.values[k];
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  OutputDenseMatrix::compute<T, M, N, RowIndices, RowPointers>(mat, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Substitute Dense Matrix to Sparse Matrix */
namespace SubstituteDenseMatrixToSparseMatrix {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I, std::size_t J_idx>
struct Column {
  /**
   * @brief Recursive computation of a column in the sparse matrix.
   *
   * This template struct recursively computes the values of a specific column
   * in the sparse matrix by accessing the corresponding elements in the dense
   * matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam I            Current row index in the sparse matrix.
   * @tparam J_idx        Current column index in the dense matrix.
   */
  static void
  compute(const Matrix<T, M, N> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

    Y.values[I * N + J_idx] = A.template get<I, J_idx>();
    Column<T, M, N, RowIndices_A, RowPointers_A, I, J_idx - 1>::compute(A, Y);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I>
struct Column<T, M, N, RowIndices_A, RowPointers_A, I, 0> {
  /**
   * @brief Termination condition for the column computation.
   *
   * This template struct represents the termination condition of the column
   * computation, where J_idx is 0. It sets the first element of the column in
   * the sparse matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam I            Current row index in the sparse matrix.
   */
  static void
  compute(const Matrix<T, M, N> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

    Y.values[I * N] = A.template get<I, 0>();
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t I_idx>
struct Row {
  /**
   * @brief Recursive computation of a row in the sparse matrix.
   *
   * This template struct recursively computes the values of a specific row in
   * the sparse matrix by accessing the corresponding elements in the dense
   * matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam I_idx        Current row index in the sparse matrix.
   */
  static void
  compute(const Matrix<T, M, N> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {

    Column<T, M, N, RowIndices_A, RowPointers_A, I_idx, N - 1>::compute(A, Y);
    Row<T, M, N, RowIndices_A, RowPointers_A, I_idx - 1>::compute(A, Y);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
struct Row<T, M, N, RowIndices_A, RowPointers_A, 0> {
  /**
   * @brief Termination condition for the row computation.
   *
   * This template struct represents the termination condition of the row
   * computation, where I_idx is 0. It computes the last column of the sparse
   * matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   */
  static void
  compute(const Matrix<T, M, N> &A,
          CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {
    Column<T, M, N, RowIndices_A, RowPointers_A, 0, N - 1>::compute(A, Y);
  }
};

/**
 * @brief Computes the sparse matrix from a dense matrix.
 *
 * This function computes the sparse matrix by iterating over the rows and
 * columns of the dense matrix and filling in the corresponding entries in the
 * sparse matrix.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices_A The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers_A The type representing row pointers of the sparse
 * matrix.
 * @param A             The dense matrix to convert to sparse format.
 * @param Y             The resulting sparse matrix to fill with computed
 * values.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
static inline void
compute(const Matrix<T, M, N> &A,
        CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &Y) {
  Row<T, M, N, RowIndices_A, RowPointers_A, M - 1>::compute(A, Y);
}

} // namespace SubstituteDenseMatrixToSparseMatrix

/**
 * @brief Creates a compiled sparse matrix from a dense matrix.
 *
 * This function constructs a compiled sparse matrix from a given dense matrix
 * by iterating over its elements and filling in the non-zero values in the
 * sparse matrix representation.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @param A             The dense matrix to convert to sparse format.
 * @return CompiledSparseMatrix<T, M, N, RowIndices, RowPointers>
 *         A sparse matrix representation of the input dense matrix.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto create_compiled_sparse(const Matrix<T, M, N> &A)
    -> CompiledSparseMatrix<T, M, N, DenseMatrixRowIndices<M, N>,
                            DenseMatrixRowPointers<M, N>> {
  CompiledSparseMatrix<T, M, N, DenseMatrixRowIndices<M, N>,
                       DenseMatrixRowPointers<M, N>>
      Y;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  std::size_t consecutive_index = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < N; j++) {
      Y.values[consecutive_index] = A(i, j);
      consecutive_index++;
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SubstituteDenseMatrixToSparseMatrix::compute<
      T, M, N, DenseMatrixRowIndices<M, N>, DenseMatrixRowPointers<M, N>>(A, Y);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return Y;
}

/* Create Sparse Matrix from Diag Matrix */
template <std::size_t M>
using DiagMatrixRowIndices = typename TemplatesOperation::ToRowIndices<
    TemplatesOperation::MatrixRowNumbers<M>>::type;

template <std::size_t M>
using DiagMatrixRowPointers = typename TemplatesOperation::ToRowIndices<
    TemplatesOperation::MatrixRowNumbers<(M + 1)>>::type;

/**
 * @brief Creates a compiled sparse matrix from a diagonal matrix.
 *
 * This function constructs a compiled sparse matrix from a given diagonal
 * matrix by copying the diagonal elements into the sparse matrix
 * representation.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of rows and columns in the diagonal matrix.
 * @param A             The diagonal matrix to convert to sparse format.
 * @return CompiledSparseMatrix<T, M, M, DiagMatrixRowIndices<M>,
 *         DiagMatrixRowPointers<M>>
 *         A sparse matrix representation of the input diagonal matrix.
 */
template <typename T, std::size_t M>
inline auto create_compiled_sparse(const DiagMatrix<T, M> &A)
    -> CompiledSparseMatrix<T, M, M, DiagMatrixRowIndices<M>,
                            DiagMatrixRowPointers<M>> {
  CompiledSparseMatrix<T, M, M, DiagMatrixRowIndices<M>,
                       DiagMatrixRowPointers<M>>
      Y;

  Y.values = A.data;

  return Y;
}

/* Create Compiled Sparse Matrix from SparseAvailable */

/**
 * @brief Creates a compiled sparse matrix from a sparse available type.
 *
 * This function constructs a compiled sparse matrix from a given sparse
 * available type, which contains the necessary information about the sparsity
 * pattern and the number of columns and rows.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam SparseAvailable The type representing the sparsity pattern and size.
 * @param values        An initializer list of values to fill the sparse matrix.
 * @return CompiledSparseMatrix<T, SparseAvailable::number_of_columns,
 *         SparseAvailable::column_size,
 * RowIndicesFromSparseAvailable<SparseAvailable>,
 *         RowPointersFromSparseAvailable<SparseAvailable>>
 *         A sparse matrix representation of the input values.
 */
template <typename T, typename SparseAvailable>
inline auto create_compiled_sparse(std::initializer_list<T> values)
    -> CompiledSparseMatrix<T, SparseAvailable::number_of_columns,
                            SparseAvailable::column_size,
                            RowIndicesFromSparseAvailable<SparseAvailable>,
                            RowPointersFromSparseAvailable<SparseAvailable>> {
  CompiledSparseMatrix<T, SparseAvailable::number_of_columns,
                       SparseAvailable::column_size,
                       RowIndicesFromSparseAvailable<SparseAvailable>,
                       RowPointersFromSparseAvailable<SparseAvailable>>
      Y;

  // This may cause runtime error if the size of values is larger than
  // RowIndices::size.
  std::copy(values.begin(),
            values.begin() +
                RowIndicesFromSparseAvailable<SparseAvailable>::size,
            Y.values.begin());

  return Y;
}

/* Set Sparse Matrix Value */
namespace SetSparseMatrixValue {

// check if RowToSet == RowIndices_A::list[K]
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, std::size_t RowToGet_I>
struct CoreIf {
  /**
   * @brief Core conditional operation for setting sparse matrix value.
   *
   * This template struct checks if the current row index matches the specified
   * row index and sets the value accordingly.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam K            Current index in the row indices list.
   * @tparam RowToGet_I   The row index to check against.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {
    /* Do nothing */
    static_cast<void>(A);
    static_cast<void>(value);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
struct CoreIf<T, M, N, RowIndices_A, RowPointers_A, K, 0> {
  /**
   * @brief Core conditional operation for setting sparse matrix value when
   * RowToSet == RowIndices_A::list[K].
   *
   * This template struct sets the value in the sparse matrix if the current
   * row index matches the specified row index.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam K            Current index in the row indices list.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    A.values[K] = value;
  }
};

// Core conditional operation for setting sparse matrix value
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K, std::size_t L>
struct CoreConditional {
  /**
   * @brief Core conditional operation for setting sparse matrix value.
   *
   * This template struct checks if the current row index matches the specified
   * row index and sets the value accordingly.
   *
   * @tparam ColumnToSet  The column index to set.
   * @tparam RowToSet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current index in the row indices list.
   * @tparam K            Current index in the row pointers list.
   * @tparam L            Difference between RowToSet and RowIndices_A::list[K].
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {
    /* Do nothing */
    static_cast<void>(A);
    static_cast<void>(value);
  }
};

template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K>
struct CoreConditional<ColumnToSet, RowToSet, T, M, N, RowIndices_A,
                       RowPointers_A, J, K, 0> {
  /**
   * @brief Core conditional operation for setting sparse matrix value when
   * RowToSet == RowIndices_A::list[K].
   *
   * This template struct sets the value in the sparse matrix if the current
   * row index matches the specified row index.
   *
   * @tparam ColumnToSet  The column index to set.
   * @tparam RowToSet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current index in the row indices list.
   * @tparam K            Current index in the row pointers list.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    CoreIf<T, M, N, RowIndices_A, RowPointers_A, K,
           (RowToSet - RowIndices_A::list[K])>::compute(A, value);
  }
};

// Core inner loop for setting sparse matrix value
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K,
          std::size_t K_End>
struct InnerLoop {
  /**
   * @brief Core inner loop for setting sparse matrix value.
   *
   * This template struct iterates over the non-zero elements of the sparse
   * matrix and sets the value at the specified column and row index.
   *
   * @tparam ColumnToSet  The column index to set.
   * @tparam RowToSet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current index in the row indices list.
   * @tparam K            Current index in the row pointers list.
   * @tparam K_End        Ending index for the current row's non-zero elements.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    CoreConditional<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A,
                    J, K, (RowToSet - RowIndices_A::list[K])>::compute(A,
                                                                       value);

    InnerLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A, J,
              (K + 1), (K_End - 1)>::compute(A, value);
  }
};

// End of inner loop
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K>
struct InnerLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A, J,
                 K, 0> {
  /**
   * @brief End of the inner loop for setting sparse matrix value.
   *
   * This template struct represents the termination condition of the inner
   * loop for setting sparse matrix value, where K_End is 0. It does nothing
   * as there are no more elements to process.
   *
   * @tparam ColumnToSet  The column index to set.
   * @tparam RowToSet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current index in the row indices list.
   * @tparam K            Current index in the row pointers list.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of inner loop, do nothing
  }
};

// Conditional operation for ColumnSet != J
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t C_J, std::size_t J,
          std::size_t J_End>
struct OuterConditional {
  /**
   * @brief Conditional operation for setting sparse matrix value.
   *
   * This template struct checks if the current column index matches the
   * specified column index and performs the necessary operations accordingly.
   *
   * @tparam ColumnToSet  The column index to set.
   * @tparam RowToSet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam C_J          Current column index in the output dense matrix.
   * @tparam J            Current row index in the output dense matrix.
   * @tparam J_End        Ending index for the current row's non-zero elements.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of conditional operation, do nothing
  }
};

// Conditional operation for ColumnSet == J
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t J_End>
struct OuterConditional<ColumnToSet, RowToSet, T, M, N, RowIndices_A,
                        RowPointers_A, 0, J, J_End> {
  /**
   * @brief Conditional operation for setting sparse matrix value when
   * ColumnToSet == J.
   *
   * This template struct sets the value in the sparse matrix if the current
   * column index matches the specified column index.
   *
   * @tparam ColumnToSet  The column index to set.
   * @tparam RowToSet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current row index in the output dense matrix.
   * @tparam J_End        Ending index for the current row's non-zero elements.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    InnerLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A, J,
              RowPointers_A::list[J],
              (RowPointers_A::list[J + 1] -
               RowPointers_A::list[J])>::compute(A, value);
  }
};

// Core outer loop for setting sparse matrix value
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t J_End>
struct OuterLoop {
  /**
   * @brief Core outer loop for setting sparse matrix value.
   *
   * This template struct iterates over the columns of the sparse matrix and
   * performs the necessary operations to set the value at the specified column
   * and row index.
   *
   * @tparam ColumnToSet  The column index to set.
   * @tparam RowToSet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current index in the row indices list.
   * @tparam J_End        Ending index for the current row's non-zero elements.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {

    OuterConditional<ColumnToSet, RowToSet, T, M, N, RowIndices_A,
                     RowPointers_A, (ColumnToSet - J), J,
                     J_End>::compute(A, value);

    OuterLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A,
              (J + 1), (J_End - 1)>::compute(A, value);
  }
};

// End of outer loop
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct OuterLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A, J,
                 0> {
  /**
   * @brief End of the outer loop for setting sparse matrix value.
   *
   * This template struct represents the termination condition of the outer
   * loop for setting sparse matrix value, where J_End is 0. It does nothing as
   * there are no more columns to process.
   *
   * @tparam ColumnToSet  The column index to set.
   * @tparam RowToSet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current index in the row indices list.
   */
  static void
  compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          const T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of outer loop, do nothing
  }
};

/**
 * @brief Computes the sparse matrix value at a specific column and row index.
 *
 * This function sets the value at the specified column and row index in the
 * sparse matrix by iterating over the non-zero elements and updating the
 * corresponding entry.
 *
 * @tparam ColumnToSet  The column index to set.
 * @tparam RowToSet     The row index to check against.
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices_A The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers_A The type representing row pointers of the sparse
 * matrix.
 * @param A             The sparse matrix to update.
 * @param value         The value to set at the specified column and row index.
 */
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void
compute(CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        const T &value) {

  OuterLoop<ColumnToSet, RowToSet, T, M, N, RowIndices_A, RowPointers_A, 0,
            M>::compute(A, value);
}

} // namespace SetSparseMatrixValue

/**
 * @brief Sets a value in the sparse matrix at a specific column and row index.
 *
 * This function updates the value at the specified column and row index in the
 * sparse matrix. It uses a compile-time loop to find the correct position in
 * the sparse matrix and set the value.
 *
 * @tparam ColumnToSet  The column index to set.
 * @tparam RowToSet     The row index to check against.
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices_A The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers_A The type representing row pointers of the sparse
 * matrix.
 * @param A             The sparse matrix to update.
 * @param value         The value to set at the specified column and row index.
 */
template <std::size_t ColumnToSet, std::size_t RowToSet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void set_sparse_matrix_value(
    CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const T &value) {
  static_assert(ColumnToSet < M, "Column number must be less than M");
  static_assert(RowToSet < N, "Row number must be less than N");

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    if (ColumnToSet == j) {

      for (std::size_t k = RowPointers_A::list[j];
           k < RowPointers_A::list[j + 1]; ++k) {
        if (RowToSet == RowIndices_A::list[k]) {

          A.values[k] = value;
        }
      }
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  SetSparseMatrixValue::compute<ColumnToSet, RowToSet, T, M, N, RowIndices_A,
                                RowPointers_A>(A, value);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__
}

/* Set Sparse Matrix each element values */

/**
 * @brief Sets the value of a specific element in the sparse matrix.
 *
 * This function updates the value at the specified element index in the sparse
 * matrix. It uses a static assertion to ensure that the element index is valid
 * and then sets the value at that index.
 *
 * @tparam ElementToSet The index of the element to set.
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices_A The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers_A The type representing row pointers of the sparse
 * matrix.
 * @param A             The sparse matrix to update.
 * @param value         The value to set at the specified element index.
 */
template <std::size_t ElementToSet, typename T, std::size_t M, std::size_t N,
          typename RowIndices_A, typename RowPointers_A>
inline void set_sparse_matrix_element_value(
    CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
    const T &value) {

  static_assert(ElementToSet < RowPointers_A::list[M],
                "Element number must be less than RowPointers::list[M]");

  A.values[ElementToSet] = value;
}

/* Get Sparse Matrix Value */
namespace GetSparseMatrixValue {

// check if RowToGet == RowIndices_A::list[K]
template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K, std::size_t RowToGet_I>
struct CoreIf {
  /**
   * @brief Core conditional operation for getting sparse matrix value.
   *
   * This template struct checks if the current row index matches the specified
   * row index and retrieves the value accordingly.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam K            Current index in the row indices list.
   * @tparam RowToGet_I   The row index to check against.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {
    /* Do nothing */
    static_cast<void>(A);
    static_cast<void>(value);
  }
};

template <typename T, std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t K>
struct CoreIf<T, M, N, RowIndices_A, RowPointers_A, K, 0> {
  /**
   * @brief Core conditional operation for getting sparse matrix value when
   * RowToGet == RowIndices_A::list[K].
   *
   * This template struct retrieves the value from the sparse matrix if the
   * current row index matches the specified row index.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam K            Current index in the row indices list.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    value = A.values[K];
  }
};

// Core conditional operation for setting sparse matrix value
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K, std::size_t L>
struct CoreConditional {
  /**
   * @brief Core conditional operation for getting sparse matrix value.
   *
   * This template struct checks if the current row index matches the specified
   * row index and retrieves the value accordingly.
   *
   * @tparam ColumnToGet  The column index to get.
   * @tparam RowToGet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current index in the row indices list.
   * @tparam K            Current index in the row pointers list.
   * @tparam L            Difference between RowToGet and RowIndices_A::list[K].
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {
    /* Do nothing */
    static_cast<void>(A);
    static_cast<void>(value);
  }
};

template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K>
struct CoreConditional<ColumnToGet, RowToGet, T, M, N, RowIndices_A,
                       RowPointers_A, J, K, 0> {
  /**
   * @brief Core conditional operation for getting sparse matrix value when
   * RowToGet == RowIndices_A::list[K].
   *
   * This template struct retrieves the value from the sparse matrix if the
   * current row index matches the specified row index.
   *
   * @tparam ColumnToGet  The column index to get.
   * @tparam RowToGet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current index in the row indices list.
   * @tparam K            Current index in the row pointers list.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    CoreIf<T, M, N, RowIndices_A, RowPointers_A, K,
           (RowToGet - RowIndices_A::list[K])>::compute(A, value);
  }
};

// Core inner loop for setting sparse matrix value
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K,
          std::size_t K_End>
struct InnerLoop {
  /**
   * @brief Core inner loop for getting sparse matrix value.
   *
   * This template struct iterates over the non-zero elements of the sparse
   * matrix and retrieves the value at the specified column and row index.
   *
   * @tparam ColumnToGet  The column index to get.
   * @tparam RowToGet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current index in the row indices list.
   * @tparam K            Current index in the row pointers list.
   * @tparam K_End        Ending index for the current row's non-zero elements.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    CoreConditional<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A,
                    J, K, (RowToGet - RowIndices_A::list[K])>::compute(A,
                                                                       value);

    InnerLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A, J,
              (K + 1), (K_End - 1)>::compute(A, value);
  }
};

// End of inner loop
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t K>
struct InnerLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A, J,
                 K, 0> {
  /**
   * @brief End of the inner loop for getting sparse matrix value.
   *
   * This template struct represents the termination condition of the inner
   * loop for getting sparse matrix value, where K_End is 0. It does nothing
   * as there are no more elements to process.
   *
   * @tparam ColumnToGet  The column index to get.
   * @tparam RowToGet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current index in the row indices list.
   * @tparam K            Current index in the row pointers list.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of inner loop, do nothing
  }
};

// Conditional operation for ColumnGet != J
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t C_J, std::size_t J,
          std::size_t J_End>
struct OuterConditional {
  /**
   * @brief Conditional operation for getting sparse matrix value.
   *
   * This template struct checks if the current column index matches the
   * specified column index and performs the necessary operations accordingly.
   *
   * @tparam ColumnToGet  The column index to get.
   * @tparam RowToGet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam C_J          Current column index in the output dense matrix.
   * @tparam J            Current row index in the output dense matrix.
   * @tparam J_End        Ending index for the current row's non-zero elements.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of conditional operation, do nothing
  }
};

// Conditional operation for ColumnGet == J
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t J_End>
struct OuterConditional<ColumnToGet, RowToGet, T, M, N, RowIndices_A,
                        RowPointers_A, 0, J, J_End> {
  /**
   * @brief Conditional operation for getting sparse matrix value when
   * ColumnToGet == J.
   *
   * This template struct retrieves the value from the sparse matrix if the
   * current column index matches the specified column index.
   *
   * @tparam ColumnToGet  The column index to get.
   * @tparam RowToGet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current row index in the output dense matrix.
   * @tparam J_End        Ending index for the current row's non-zero elements.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    InnerLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A, J,
              RowPointers_A::list[J],
              (RowPointers_A::list[J + 1] -
               RowPointers_A::list[J])>::compute(A, value);
  }
};

// Core outer loop for setting sparse matrix value
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J, std::size_t J_End>
struct OuterLoop {
  /**
   * @brief Core outer loop for getting sparse matrix value.
   *
   * This template struct iterates over the columns of the sparse matrix and
   * performs the necessary operations to retrieve the value at the specified
   * column and row index.
   *
   * @tparam ColumnToGet  The column index to get.
   * @tparam RowToGet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current index in the row indices list.
   * @tparam J_End        Ending index for the current row's non-zero elements.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {

    OuterConditional<ColumnToGet, RowToGet, T, M, N, RowIndices_A,
                     RowPointers_A, (ColumnToGet - J), J,
                     J_End>::compute(A, value);

    OuterLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A,
              (J + 1), (J_End - 1)>::compute(A, value);
  }
};

// End of outer loop
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A, std::size_t J>
struct OuterLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A, J,
                 0> {
  /**
   * @brief End of the outer loop for getting sparse matrix value.
   *
   * This template struct represents the termination condition of the outer
   * loop for getting sparse matrix value, where J_End is 0. It does nothing as
   * there are no more columns to process.
   *
   * @tparam ColumnToGet  The column index to get.
   * @tparam RowToGet     The row index to check against.
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_A The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers_A The type representing row pointers of the sparse
   * matrix.
   * @tparam J            Current index in the row indices list.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
          T &value) {
    static_cast<void>(A);
    static_cast<void>(value);
    // End of outer loop, do nothing
  }
};

/**
 * @brief Computes the sparse matrix value at a specific column and row index.
 *
 * This function retrieves the value at the specified column and row index in
 * the sparse matrix by iterating over the non-zero elements and returning the
 * corresponding entry.
 *
 * @tparam ColumnToGet  The column index to get.
 * @tparam RowToGet     The row index to check against.
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices_A The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers_A The type representing row pointers of the sparse
 * matrix.
 * @param A             The sparse matrix to query.
 * @param value         Reference to store the retrieved value.
 */
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A,
        T &value) {
  OuterLoop<ColumnToGet, RowToGet, T, M, N, RowIndices_A, RowPointers_A, 0,
            M>::compute(A, value);
}

} // namespace GetSparseMatrixValue

/**
 * @brief Retrieves the value at a specific column and row index in the sparse
 * matrix.
 *
 * This function uses compile-time loops to find the value at the specified
 * column and row index in the sparse matrix. It returns the value if found,
 * otherwise returns zero.
 *
 * @tparam ColumnToGet  The column index to get.
 * @tparam RowToGet     The row index to check against.
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices_A The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers_A The type representing row pointers of the sparse
 * matrix.
 * @param A             The sparse matrix to query.
 * @return The value at the specified column and row index, or zero if not
 * found.
 */
template <std::size_t ColumnToGet, std::size_t RowToGet, typename T,
          std::size_t M, std::size_t N, typename RowIndices_A,
          typename RowPointers_A>
inline T get_sparse_matrix_value(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
  static_assert(ColumnToGet < M, "Column number must be less than M");
  static_assert(RowToGet < N, "Row number must be less than N");

  T value = static_cast<T>(0);

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    if (ColumnToGet == j) {

      for (std::size_t k = RowPointers_A::list[j];
           k < RowPointers_A::list[j + 1]; ++k) {
        if (RowToGet == RowIndices_A::list[k]) {

          value = A.values[k];
        }
      }
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  GetSparseMatrixValue::compute<ColumnToGet, RowToGet, T, M, N, RowIndices_A,
                                RowPointers_A>(A, value);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return value;
}

/* Get Sparse Matrix each element values */

/**
 * @brief Retrieves the value of a specific element in the sparse matrix.
 *
 * This function returns the value at the specified element index in the sparse
 * matrix. It uses a static assertion to ensure that the element index is valid.
 *
 * @tparam ElementToGet The index of the element to get.
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices_A The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers_A The type representing row pointers of the sparse
 * matrix.
 * @param A             The sparse matrix to query.
 * @return The value at the specified element index.
 */
template <std::size_t ElementToGet, typename T, std::size_t M, std::size_t N,
          typename RowIndices_A, typename RowPointers_A>
inline T get_sparse_matrix_element_value(
    const CompiledSparseMatrix<T, M, N, RowIndices_A, RowPointers_A> &A) {
  static_assert(ElementToGet < RowPointers_A::list[M],
                "Element number must be less than RowPointers::list[M]");

  return A.values[ElementToGet];
}

/* Output transpose matrix */
namespace OutputTransposeMatrix {

// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, typename Result_Type, std::size_t J,
          std::size_t K, std::size_t Start, std::size_t End>
struct OutputTransposeMatrixLoop {
  /**
   * @brief Core loop for outputting the transpose of a sparse matrix.
   *
   * This template struct iterates over the non-zero elements of the sparse
   * matrix and sets the corresponding values in the result matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers  The type representing row pointers of the sparse
   * matrix.
   * @tparam Result_Type  The type of the result matrix.
   * @tparam J            Current index in the row indices list.
   * @tparam K            Current index in the row pointers list.
   * @tparam Start        Starting index for the current row's non-zero
   * elements.
   * @tparam End          Ending index for the current row's non-zero elements.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Result_Type &result) {

    set_sparse_matrix_value<RowIndices::list[Start], J>(result,
                                                        mat.values[Start]);
    OutputTransposeMatrixLoop<T, M, N, RowIndices, RowPointers, Result_Type, J,
                              K, Start + 1, End>::compute(mat, result);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, typename Result_Type, std::size_t J,
          std::size_t K, std::size_t End>
struct OutputTransposeMatrixLoop<T, M, N, RowIndices, RowPointers, Result_Type,
                                 J, K, End, End> {
  /**
   * @brief End of the core loop for outputting the transpose of a sparse
   * matrix.
   *
   * This template struct represents the termination condition of the core loop
   * for outputting the transpose of a sparse matrix, where Start == End. It
   * does nothing as there are no more elements to process.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers  The type representing row pointers of the sparse
   * matrix.
   * @tparam Result_Type  The type of the result matrix.
   * @tparam J            Current index in the row indices list.
   * @tparam K            Current index in the row pointers list.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Result_Type &result) {
    static_cast<void>(mat);
    static_cast<void>(result);
    // End of loop, do nothing
  }
};

// Pointer loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, typename Result_Type, std::size_t J,
          std::size_t K>
struct OutputTransposeMatrixCore {
  /**
   * @brief Core loop for outputting the transpose of a sparse matrix.
   *
   * This template struct iterates over the rows of the sparse matrix and
   * processes the non-zero elements to set the corresponding values in the
   * result matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers  The type representing row pointers of the sparse
   * matrix.
   * @tparam Result_Type  The type of the result matrix.
   * @tparam J            Current index in the row indices list.
   * @tparam K            Current index in the row pointers list.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Result_Type &result) {

    OutputTransposeMatrixLoop<T, M, N, RowIndices, RowPointers, Result_Type, J,
                              K, RowPointers::list[J],
                              RowPointers::list[J + 1]>::compute(mat, result);
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, typename Result_Type, std::size_t J>
struct OutputTransposeMatrixRow {
  /**
   * @brief Row loop for outputting the transpose of a sparse matrix.
   *
   * This template struct iterates over the rows of the sparse matrix and
   * processes each row to set the corresponding values in the result matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers  The type representing row pointers of the sparse
   * matrix.
   * @tparam Result_Type  The type of the result matrix.
   * @tparam J            Current index in the row indices list.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Result_Type &result) {

    OutputTransposeMatrixCore<T, M, N, RowIndices, RowPointers, Result_Type, J,
                              0>::compute(mat, result);
    OutputTransposeMatrixRow<T, M, N, RowIndices, RowPointers, Result_Type,
                             J - 1>::compute(mat, result);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, typename Result_Type>
struct OutputTransposeMatrixRow<T, M, N, RowIndices, RowPointers, Result_Type,
                                0> {
  /**
   * @brief End of the row loop for outputting the transpose of a sparse matrix.
   *
   * This template struct represents the termination condition of the row loop
   * for outputting the transpose of a sparse matrix, where J is 0. It does
   * nothing as there are no more rows to process.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers  The type representing row pointers of the sparse
   * matrix.
   * @tparam Result_Type  The type of the result matrix.
   */
  static void
  compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
          Result_Type &result) {

    OutputTransposeMatrixCore<T, M, N, RowIndices, RowPointers, Result_Type, 0,
                              0>::compute(mat, result);
  }
};

/**
 * @brief Computes the transpose of a sparse matrix and stores it in the result
 * matrix.
 *
 * This function uses compile-time loops to iterate over the rows and non-zero
 * elements of the sparse matrix, setting the corresponding values in the
 * result matrix.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices   The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers  The type representing row pointers of the sparse
 * matrix.
 * @tparam Result_Type  The type of the result matrix.
 * @param mat           The sparse matrix to transpose.
 * @param result        Reference to store the transposed result.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, typename Result_Type>
inline void
compute(const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat,
        Result_Type &result) {

  OutputTransposeMatrixRow<T, M, N, RowIndices, RowPointers, Result_Type,
                           M - 1>::compute(mat, result);
}

} // namespace OutputTransposeMatrix

namespace CompiledSparseOperation {

template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct Transpose {

  using SparseAvailable_In =
      CreateSparseAvailableFromIndicesAndPointers<N, RowIndices, RowPointers>;

  using SparseAvailable_Out = SparseAvailableTranspose<SparseAvailable_In>;

  using RowIndices_Out = RowIndicesFromSparseAvailable<SparseAvailable_Out>;

  using RowPointers_Out = RowPointersFromSparseAvailable<SparseAvailable_Out>;

  using Result_Type =
      CompiledSparseMatrix<T, N, M, RowIndices_Out, RowPointers_Out>;
};

} // namespace CompiledSparseOperation

/**
 * @brief Outputs the transpose of a sparse matrix.
 *
 * This function computes the transpose of the given sparse matrix and returns
 * it as a new sparse matrix with transposed dimensions.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the input matrix.
 * @tparam N            The number of rows in the input matrix.
 * @tparam RowIndices   The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers  The type representing row pointers of the sparse
 * matrix.
 * @param mat           The sparse matrix to transpose.
 * @return A new sparse matrix representing the transpose of the input matrix.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline auto output_matrix_transpose(
    const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &mat) ->
    typename CompiledSparseOperation::Transpose<T, M, N, RowIndices,
                                                RowPointers>::Result_Type {

  using Result_Type =
      typename CompiledSparseOperation::Transpose<T, M, N, RowIndices,
                                                  RowPointers>::Result_Type;

  Result_Type result;

  OutputTransposeMatrix::compute<T, M, N, RowIndices, RowPointers, Result_Type>(
      mat, result);

  return result;
}

/* Convert Real Matrix to Complex Matrix */
namespace ConvertRealSparseMatrixToComplex {

/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t I>
struct Loop {
  /**
   * @brief Core loop for converting a real sparse matrix to a complex sparse
   * matrix.
   *
   * This template struct iterates over the non-zero elements of the real sparse
   * matrix and sets the corresponding values in the complex sparse matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers  The type representing row pointers of the sparse
   * matrix.
   * @tparam I            Current index in the values list.
   */
  static void compute(
      const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &From_matrix,
      CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
          &To_matrix) {

    To_matrix.values[I - 1].real = From_matrix.values[I - 1];
    Loop<T, M, N, RowIndices, RowPointers, I - 1>::compute(From_matrix,
                                                           To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct Loop<T, M, N, RowIndices, RowPointers, 0> {
  /**
   * @brief End of the core loop for converting a real sparse matrix to a
   * complex sparse matrix.
   *
   * This template struct represents the termination condition of the core loop
   * for converting a real sparse matrix to a complex sparse matrix, where I is
   * 0. It does nothing as there are no more elements to process.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers  The type representing row pointers of the sparse
   * matrix.
   */
  static void compute(
      const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &From_matrix,
      CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
          &To_matrix) {
    /* Do Nothing. */
    static_cast<void>(From_matrix);
    static_cast<void>(To_matrix);
  }
};

/**
 * @brief Computes the conversion from a real sparse matrix to a complex sparse
 * matrix.
 *
 * This function uses compile-time loops to iterate over the non-zero elements
 * of the real sparse matrix and sets the corresponding values in the complex
 * sparse matrix.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices   The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers  The type representing row pointers of the sparse
 * matrix.
 * @param From_matrix   The real sparse matrix to convert.
 * @param To_matrix     Reference to store the converted complex sparse matrix.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline void compute(
    const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &From_matrix,
    CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
        &To_matrix) {

  Loop<T, M, N, RowIndices, RowPointers, RowPointers::list[M]>::compute(
      From_matrix, To_matrix);
}

} // namespace ConvertRealSparseMatrixToComplex

/**
 * @brief Converts a real sparse matrix to a complex sparse matrix.
 *
 * This function takes a real sparse matrix and converts it to a complex sparse
 * matrix by setting the real part of each element in the complex matrix to the
 * corresponding value in the real matrix.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices   The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers  The type representing row pointers of the sparse
 * matrix.
 * @param From_matrix   The real sparse matrix to convert.
 * @return A new complex sparse matrix with the converted values.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
convert_matrix_real_to_complex(
    const CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &From_matrix) {

  CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < RowPointers::list[M]; ++i) {
    To_matrix.values[i].real = From_matrix.values[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  ConvertRealSparseMatrixToComplex::compute<T, M, N, RowIndices, RowPointers>(
      From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Get Real Matrix from Complex Matrix */
namespace GetRealSparseMatrixFromComplex {

/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t I>
struct Loop {
  /**
   * @brief Core loop for extracting the real part from a complex sparse matrix.
   *
   * This template struct iterates over the non-zero elements of the complex
   * sparse matrix and sets the corresponding values in the real sparse matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers  The type representing row pointers of the sparse
   * matrix.
   * @tparam I            Current index in the values list.
   */
  static void
  compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
              &From_matrix,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

    To_matrix.values[I - 1] = From_matrix.values[I - 1].real;
    Loop<T, M, N, RowIndices, RowPointers, I - 1>::compute(From_matrix,
                                                           To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct Loop<T, M, N, RowIndices, RowPointers, 0> {
  /**
   * @brief End of the core loop for extracting the real part from a complex
   * sparse matrix.
   *
   * This template struct represents the termination condition of the core loop
   * for extracting the real part from a complex sparse matrix, where I is 0. It
   * does nothing as there are no more elements to process.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers  The type representing row pointers of the sparse
   * matrix.
   */
  static void
  compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
              &From_matrix,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

    /* Do Nothing. */
    static_cast<void>(From_matrix);
    static_cast<void>(To_matrix);
  }
};

/**
 * @brief Computes the conversion from a complex sparse matrix to a real sparse
 * matrix.
 *
 * This function uses compile-time loops to iterate over the non-zero elements
 * of the complex sparse matrix and sets the corresponding values in the real
 * sparse matrix.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices   The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers  The type representing row pointers of the sparse
 * matrix.
 * @param From_matrix   The complex sparse matrix to convert.
 * @param To_matrix     Reference to store the converted real sparse matrix.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline void
compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
            &From_matrix,
        CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

  Loop<T, M, N, RowIndices, RowPointers, RowPointers::list[M]>::compute(
      From_matrix, To_matrix);
}

} // namespace GetRealSparseMatrixFromComplex

/**
 * @brief Converts a complex sparse matrix to a real sparse matrix.
 *
 * This function takes a complex sparse matrix and extracts the real part of
 * each element, returning a new real sparse matrix with the converted values.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices   The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers  The type representing row pointers of the sparse
 * matrix.
 * @param From_matrix   The complex sparse matrix to convert.
 * @return A new real sparse matrix with the converted values.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline CompiledSparseMatrix<T, M, N, RowIndices, RowPointers>
get_real_matrix_from_complex_matrix(
    const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
        &From_matrix) {

  CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < RowPointers::list[M]; ++i) {
    To_matrix.values[i] = From_matrix.values[i].real;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  GetRealSparseMatrixFromComplex::compute<T, M, N, RowIndices, RowPointers>(
      From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Get Imag Matrix from Complex Matrix */
namespace GetImagSparseMatrixFromComplex {

/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers, std::size_t I>
struct Loop {
  /**
   * @brief Core loop for extracting the imaginary part from a complex sparse
   * matrix.
   *
   * This template struct iterates over the non-zero elements of the complex
   * sparse matrix and sets the corresponding values in the imaginary sparse
   * matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers  The type representing row pointers of the sparse
   * matrix.
   * @tparam I            Current index in the values list.
   */
  static void
  compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
              &From_matrix,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

    To_matrix.values[I - 1] = From_matrix.values[I - 1].imag;
    Loop<T, M, N, RowIndices, RowPointers, I - 1>::compute(From_matrix,
                                                           To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
struct Loop<T, M, N, RowIndices, RowPointers, 0> {
  /**
   * @brief End of the core loop for extracting the imaginary part from a
   * complex sparse matrix.
   *
   * This template struct represents the termination condition of the core loop
   * for extracting the imaginary part from a complex sparse matrix, where I is
   * 0. It does nothing as there are no more elements to process.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices   The type representing row indices of the sparse
   * matrix.
   * @tparam RowPointers  The type representing row pointers of the sparse
   * matrix.
   */
  static void
  compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
              &From_matrix,
          CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

    /* Do Nothing. */
    static_cast<void>(From_matrix);
    static_cast<void>(To_matrix);
  }
};

/**
 * @brief Computes the conversion from a complex sparse matrix to an imaginary
 * sparse matrix.
 *
 * This function uses compile-time loops to iterate over the non-zero elements
 * of the complex sparse matrix and sets the corresponding values in the
 * imaginary sparse matrix.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices   The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers  The type representing row pointers of the sparse
 * matrix.
 * @param From_matrix   The complex sparse matrix to convert.
 * @param To_matrix     Reference to store the converted imaginary sparse
 * matrix.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline void
compute(const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
            &From_matrix,
        CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> &To_matrix) {

  Loop<T, M, N, RowIndices, RowPointers, RowPointers::list[M]>::compute(
      From_matrix, To_matrix);
}

} // namespace GetImagSparseMatrixFromComplex

/**
 * @brief Converts a complex sparse matrix to an imaginary sparse matrix.
 *
 * This function takes a complex sparse matrix and extracts the imaginary part
 * of each element, returning a new imaginary sparse matrix with the converted
 * values.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices   The type representing row indices of the sparse
 * matrix.
 * @tparam RowPointers  The type representing row pointers of the sparse
 * matrix.
 * @param From_matrix   The complex sparse matrix to convert.
 * @return A new imaginary sparse matrix with the converted values.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices,
          typename RowPointers>
inline CompiledSparseMatrix<T, M, N, RowIndices, RowPointers>
get_imag_matrix_from_complex_matrix(
    const CompiledSparseMatrix<Complex<T>, M, N, RowIndices, RowPointers>
        &From_matrix) {

  CompiledSparseMatrix<T, M, N, RowIndices, RowPointers> To_matrix;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < RowPointers::list[M]; ++i) {
    To_matrix.values[i] = From_matrix.values[i].imag;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  GetImagSparseMatrixFromComplex::compute<T, M, N, RowIndices, RowPointers>(
      From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Diagonal Inverse Multiply Sparse */
namespace DiagonalInverseMultiplySparse {

// Start < End (Core)
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t K,
          std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief Core loop for diagonal inverse multiplication of a sparse matrix.
   *
   * This template struct iterates over the non-zero elements of the sparse
   * matrix and performs the diagonal inverse multiplication with the diagonal
   * matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_B The type representing row indices of the sparse
   * matrix B.
   * @tparam RowPointers_B The type representing row pointers of the sparse
   * matrix B.
   * @tparam J            Current index in the row indices list of B.
   * @tparam K            Current index in the row pointers list of B.
   * @tparam Start        Starting index for the current row's non-zero
   * elements.
   * @tparam End          Ending index for the current row's non-zero elements.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          const T &division_min,
          CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &result) {

    result.values[Start] =
        B.values[Start] / Base::Utility::avoid_zero_divide(A[J], division_min);

    Loop<T, M, N, RowIndices_B, RowPointers_B, J, K, Start + 1, End>::compute(
        A, B, division_min, result);
  }
};

// Start == End (End of Core Loop)
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t K, std::size_t End>
struct Loop<T, M, N, RowIndices_B, RowPointers_B, J, K, End, End> {
  /**
   * @brief End of the core loop for diagonal inverse multiplication of a sparse
   * matrix.
   *
   * This template struct represents the termination condition of the core loop
   * for diagonal inverse multiplication of a sparse matrix, where Start == End.
   * It does nothing as there are no more elements to process.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_B The type representing row indices of the sparse
   * matrix B.
   * @tparam RowPointers_B The type representing row pointers of the sparse
   * matrix B.
   * @tparam J            Current index in the row indices list of B.
   * @tparam K            Current index in the row pointers list of B.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          const T &division_min,
          CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &result) {

    static_cast<void>(A);
    static_cast<void>(B);
    static_cast<void>(division_min);
    static_cast<void>(result);
    // End of loop, do nothing
  }
};

// Pointer loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t J, std::size_t K>
struct Core {
  /**
   * @brief Core loop for diagonal inverse multiplication of a sparse matrix.
   *
   * This template struct iterates over the non-zero elements of the sparse
   * matrix and performs the diagonal inverse multiplication with the diagonal
   * matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_B The type representing row indices of the sparse
   * matrix B.
   * @tparam RowPointers_B The type representing row pointers of the sparse
   * matrix B.
   * @tparam J            Current index in the row indices list of B.
   * @tparam K            Current index in the row pointers list of B.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          const T &division_min,
          CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &result) {

    Loop<T, M, N, RowIndices_B, RowPointers_B, J, K, RowPointers_B::list[J],
         RowPointers_B::list[J + 1]>::compute(A, B, division_min, result);
  }
};

// Row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B, std::size_t J>
struct Row {
  /**
   * @brief Row loop for diagonal inverse multiplication of a sparse matrix.
   *
   * This template struct iterates over the rows of the sparse matrix and
   * processes each row to perform the diagonal inverse multiplication with the
   * diagonal matrix.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_B The type representing row indices of the sparse
   * matrix B.
   * @tparam RowPointers_B The type representing row pointers of the sparse
   * matrix B.
   * @tparam J            Current index in the row indices list of B.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          const T &division_min,
          CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &result) {

    Core<T, M, N, RowIndices_B, RowPointers_B, J, 0>::compute(
        A, B, division_min, result);
    Row<T, M, N, RowIndices_B, RowPointers_B, J - 1>::compute(
        A, B, division_min, result);
  }
};

// End of row loop
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
struct Row<T, M, N, RowIndices_B, RowPointers_B, 0> {
  /**
   * @brief End of the row loop for diagonal inverse multiplication of a sparse
   * matrix.
   *
   * This template struct represents the termination condition of the row loop
   * for diagonal inverse multiplication of a sparse matrix, where J is 0. It
   * does nothing as there are no more rows to process.
   *
   * @tparam T            The type of the matrix elements.
   * @tparam M            The number of columns in the matrix.
   * @tparam N            The number of rows in the matrix.
   * @tparam RowIndices_B The type representing row indices of the sparse
   * matrix B.
   * @tparam RowPointers_B The type representing row pointers of the sparse
   * matrix B.
   */
  static void
  compute(const DiagMatrix<T, M> &A,
          const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
          const T &division_min,
          CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &result) {

    Core<T, M, N, RowIndices_B, RowPointers_B, 0, 0>::compute(
        A, B, division_min, result);
  }
};

/**
 * @brief Computes the diagonal inverse multiplication of a sparse matrix.
 *
 * This function uses compile-time loops to iterate over the rows and non-zero
 * elements of the sparse matrix, performing the diagonal inverse multiplication
 * with the diagonal matrix.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices_B The type representing row indices of the sparse matrix
 * B.
 * @tparam RowPointers_B The type representing row pointers of the sparse matrix
 * B.
 * @param A             The diagonal matrix to multiply with.
 * @param B             The sparse matrix to multiply.
 * @param division_min  Minimum value to avoid division by zero.
 * @param result        Reference to store the result of the multiplication.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
inline void
compute(const DiagMatrix<T, M> &A,
        const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
        const T &division_min,
        CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &result) {

  Row<T, M, N, RowIndices_B, RowPointers_B, M - 1>::compute(A, B, division_min,
                                                            result);
}

} // namespace DiagonalInverseMultiplySparse

/**
 * @brief Performs diagonal inverse multiplication of a sparse matrix.
 *
 * This function takes a diagonal matrix and a sparse matrix, and performs the
 * diagonal inverse multiplication, returning a new sparse matrix with the
 * results.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices_B The type representing row indices of the sparse matrix
 * B.
 * @tparam RowPointers_B The type representing row pointers of the sparse matrix
 * B.
 * @param A             The diagonal matrix to multiply with.
 * @param B             The sparse matrix to multiply.
 * @param division_min  Minimum value to avoid division by zero.
 * @return A new sparse matrix resulting from the diagonal inverse
 * multiplication.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
inline CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B>
diag_inv_multiply_sparse(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
    const T &division_min) {

  CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> result = B;

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = RowPointers_B::list[j]; k < RowPointers_B::list[j + 1];
         ++k) {

      result.values[k] =
          B.values[k] / Base::Utility::avoid_zero_divide(A[j], division_min);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagonalInverseMultiplySparse::compute<T, M, N, RowIndices_B, RowPointers_B>(
      A, B, division_min, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/**
 * @brief Performs diagonal inverse multiplication of a sparse matrix with a
 * partitioned approach.
 *
 * This function takes a diagonal matrix and a sparse matrix, and performs the
 * diagonal inverse multiplication, returning a new sparse matrix with the
 * results. It is optimized for partitioned sparse matrices.
 *
 * @tparam T            The type of the matrix elements.
 * @tparam M            The number of columns in the matrix.
 * @tparam N            The number of rows in the matrix.
 * @tparam RowIndices_B The type representing row indices of the sparse matrix
 * B.
 * @tparam RowPointers_B The type representing row pointers of the sparse matrix
 * B.
 * @param A             The diagonal matrix to multiply with.
 * @param B             The sparse matrix to multiply.
 * @param division_min  Minimum value to avoid division by zero.
 * @param matrix_size   Size of the matrix for partitioning.
 * @return A new sparse matrix resulting from the diagonal inverse
 * multiplication.
 */
template <typename T, std::size_t M, std::size_t N, typename RowIndices_B,
          typename RowPointers_B>
inline CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B>
diag_inv_multiply_sparse_partition(
    const DiagMatrix<T, M> &A,
    const CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> &B,
    const T &division_min, const std::size_t &matrix_size) {

  CompiledSparseMatrix<T, M, N, RowIndices_B, RowPointers_B> result;

  for (std::size_t j = 0; j < matrix_size; ++j) {
    for (std::size_t k = RowPointers_B::list[j]; k < RowPointers_B::list[j + 1];
         ++k) {

      if (RowIndices_B::list[k] < matrix_size) {

        result.values[k] =
            B.values[k] / Base::Utility::avoid_zero_divide(A[j], division_min);
      }
    }
  }

  return result;
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_COMPILED_SPARSE_HPP__
