/**
 * @file python_numpy_base_simplified_action.hpp
 * @brief Provides a set of template-based utilities for matrix manipulation,
 * inspired by Python's NumPy, for C++ static matrices.
 *
 * This header defines the PythonNumpy namespace, which contains a comprehensive
 * suite of template meta-programming utilities for manipulating matrices at
 * compile time. The utilities support dense, diagonal, and sparse matrices, and
 * provide operations such as row extraction, row setting, matrix substitution,
 * block concatenation, tiling, and reshaping. The design is highly generic and
 * leverages C++ templates for static size and type safety, enabling efficient
 * and flexible matrix operations similar to those found in Python's NumPy
 * library, but at compile time.
 *
 * @note
 * tparam M is the number of rows in the matrix.
 * tparam N is the number of columns in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __PYTHON_NUMPY_BASE_SIMPLIFIED_ACTION_HPP__
#define __PYTHON_NUMPY_BASE_SIMPLIFIED_ACTION_HPP__

#include "python_math.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_base_simplification.hpp"
#include "python_numpy_complex.hpp"
#include "python_numpy_templates.hpp"

#include "python_numpy_concatenate.hpp"

#include <initializer_list>
#include <tuple>
#include <type_traits>

namespace PythonNumpy {

/* Element wise multiply */

namespace ElementWiseMultiplyOperation {

template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t I, std::size_t J_idx>
struct Row {
  /**
   * @brief Computes the product of corresponding elements from input matrices A
   * and B at position (I, J_idx), and sets the result in the output matrix Out
   * at the same position.
   *
   * This function recursively processes rows by calling the compute method
   * of the Column class for the previous column index (J_idx - 1).
   *
   * @tparam Out_Type Type of the output matrix.
   * @tparam In_A_Type Type of the first input matrix.
   * @tparam In_B_Type Type of the second input matrix.
   * @tparam M Number of columns in the matrices.
   * @tparam N Number of rows in the matrices.
   * @tparam I Current row index.
   * @tparam J_idx Current column index.
   *
   * @param Out Reference to the output matrix.
   * @param A Reference to the first input matrix.
   * @param B Reference to the second input matrix.
   */
  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {

    Out.template set<I, J_idx>(A.template get<I, J_idx>() *
                               B.template get<I, J_idx>());

    Row<Out_Type, In_A_Type, In_B_Type, M, N, I, (J_idx - 1)>::compute(Out, A,
                                                                       B);
  }
};

template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t I>
/**
 * @brief Specialization of the Column struct for the case when J (the last
 * template parameter) is 0.
 *
 * This struct provides a static compute function that multiplies elements from
 * two input types (A and B) at position <I, 0> and stores the result in the
 * output type (Out) at the same position.
 *
 * @tparam Out_Type Type of the output container.
 * @tparam In_A_Type Type of the first input container.
 * @tparam In_B_Type Type of the second input container.
 * @tparam M Number of columns (unused in this specialization).
 * @tparam N Number of rows (unused in this specialization).
 * @tparam I Row index for the operation.
 * @tparam 0 Column index (specialized to 0).
 */
struct Row<Out_Type, In_A_Type, In_B_Type, M, N, I, 0> {
  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {

    Out.template set<I, 0>(A.template get<I, 0>() * B.template get<I, 0>());
  }
};

template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t I_idx>
struct Column {
  /**
   * @brief Performs computation by invoking the compute methods of Column and
   * Row classes.
   *
   * This static function calls the compute method of the Column class with
   * template parameters (Out_Type, In_A_Type, In_B_Type, M, N, I_idx, N - 1)
   * and the compute method of the Row class with template parameters (Out_Type,
   * In_A_Type, In_B_Type, M, N, I_idx - 1).
   *
   * @tparam Out_Type Type of the output.
   * @tparam In_A_Type Type of the first input.
   * @tparam In_B_Type Type of the second input.
   * @tparam M Number of columns.
   * @tparam N Number of rows.
   * @tparam I_idx Current index for computation.
   * @param Out Reference to the output object.
   * @param A Reference to the first input object.
   * @param B Reference to the second input object.
   */
  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {
    Row<Out_Type, In_A_Type, In_B_Type, M, N, I_idx, (N - 1)>::compute(Out, A,
                                                                       B);
    Column<Out_Type, In_A_Type, In_B_Type, M, N, (I_idx - 1)>::compute(Out, A,
                                                                       B);
  }
};

template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N>
struct Column<Out_Type, In_A_Type, In_B_Type, M, N, 0> {
  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {
    Row<Out_Type, In_A_Type, In_B_Type, M, N, 0, (N - 1)>::compute(Out, A, B);
  }
};

/**
 * @brief Computes the output by processing two input types element-wise.
 *
 * This function template performs a computation on two input objects, `A` and
 * `B`, and stores the result in `Out`. It ensures at compile-time that both
 * input types have the same number of columns and rows using static assertions.
 *
 * @tparam Out_Type Type of the output object.
 * @tparam In_A_Type Type of the first input object.
 * @tparam In_B_Type Type of the second input object.
 * @param[out] Out Reference to the output object where the result will be
 * stored.
 * @param[in] A Constant reference to the first input object.
 * @param[in] B Constant reference to the second input object.
 *
 * @note The function relies on a helper template `Row` to perform the actual
 * computation. The dimensions (cols and rows) are determined at compile-time
 * from `In_A_Type`. Both input types must have matching dimensions.
 */
template <typename Out_Type, typename In_A_Type, typename In_B_Type>
inline void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {
  static_assert(In_A_Type::ROWS == In_B_Type::ROWS,
                "In_A_Type::ROWS != In_B_Type::ROWS");
  static_assert(In_A_Type::COLS == In_B_Type::COLS,
                "In_A_Type::COLS != In_B_Type::COLS");

  constexpr std::size_t M = In_A_Type::ROWS;
  constexpr std::size_t N = In_A_Type::COLS;

  Column<Out_Type, In_A_Type, In_B_Type, M, N, (M - 1)>::compute(Out, A, B);
}

} // namespace ElementWiseMultiplyOperation

/**
 * @brief Performs element-wise multiplication of two matrices or arrays.
 *
 * This function multiplies each corresponding element of input matrices or
 * arrays `A` and `B`, storing the result in `Out`. The input types must have
 * the same number of columns and rows, enforced at compile time via static
 * assertions.
 *
 * @tparam Out_Type Type of the output matrix or array.
 * @tparam In_A_Type Type of the first input matrix or array.
 * @tparam In_B_Type Type of the second input matrix or array.
 * @param Out Reference to the output matrix or array where the result is
 * stored.
 * @param A Constant reference to the first input matrix or array.
 * @param B Constant reference to the second input matrix or array.
 *
 * @note The actual computation is delegated to
 * ElementWiseMultiplyOperation::compute.
 * @throws static_assert if the dimensions of A and B do not match.
 */
template <typename Out_Type, typename In_A_Type, typename In_B_Type>
inline void element_wise_multiply(Out_Type &Out, const In_A_Type &A,
                                  const In_B_Type &B) {

  static_assert(In_A_Type::ROWS == In_B_Type::ROWS,
                "In_A_Type::ROWS != In_B_Type::ROWS");
  static_assert(In_A_Type::COLS == In_B_Type::COLS,
                "In_A_Type::COLS != In_B_Type::COLS");

  ElementWiseMultiplyOperation::compute(Out, A, B);
}

/* Inner product */

namespace InnerProductOperation {

template <typename T, typename In_A_Type, typename In_B_Type, std::size_t M,
          std::size_t N, std::size_t I, std::size_t J_idx>
struct Row {
  /**
   * @brief Accumulates the product of corresponding elements from input
   * matrices A and B at position (I, J_idx) into the result, then recursively
   * processes the previous column index (J_idx - 1).
   *
   * @tparam T The scalar type for accumulation.
   * @tparam In_A_Type Type of the first input matrix.
   * @tparam In_B_Type Type of the second input matrix.
   * @tparam M Number of rows in the matrices.
   * @tparam N Number of columns in the matrices.
   * @tparam I Current column index.
   * @tparam J_idx Current row index.
   *
   * @param result Reference to the accumulated result.
   * @param A Reference to the first input matrix.
   * @param B Reference to the second input matrix.
   */
  static void compute(T &result, const In_A_Type &A, const In_B_Type &B) {

    result += A.template get<I, J_idx>() * B.template get<I, J_idx>();

    Row<T, In_A_Type, In_B_Type, M, N, I, (J_idx - 1)>::compute(result, A, B);
  }
};

template <typename T, typename In_A_Type, typename In_B_Type, std::size_t M,
          std::size_t N, std::size_t I>
/**
 * @brief Specialization of the Column struct for the case when J_idx is 0.
 *
 * This struct provides a static compute function that accumulates the product
 * of elements from two input matrices (A and B) at position <I, 0> into the
 * result.
 *
 * @tparam T The scalar type for accumulation.
 * @tparam In_A_Type Type of the first input container.
 * @tparam In_B_Type Type of the second input container.
 * @tparam M Number of rows (unused in this specialization).
 * @tparam N Number of columns (unused in this specialization).
 * @tparam I Column index for the operation.
 * @tparam 0 Row index (specialized to 0).
 */
struct Row<T, In_A_Type, In_B_Type, M, N, I, 0> {
  static void compute(T &result, const In_A_Type &A, const In_B_Type &B) {

    result += A.template get<I, 0>() * B.template get<I, 0>();
  }
};

template <typename T, typename In_A_Type, typename In_B_Type, std::size_t M,
          std::size_t N, std::size_t I_idx>
struct Column {
  /**
   * @brief Performs accumulation by invoking the compute methods of Column and
   * Row classes.
   *
   * This static function calls the compute method of the Column class with
   * template parameters (T, In_A_Type, In_B_Type, M, N, I_idx, N - 1) and the
   * compute method of the Row class with template parameters (T, In_A_Type,
   * In_B_Type, M, N, I_idx - 1).
   *
   * @tparam T The scalar type for accumulation.
   * @tparam In_A_Type Type of the first input.
   * @tparam In_B_Type Type of the second input.
   * @tparam M Number of rows.
   * @tparam N Number of columns.
   * @tparam I_idx Current column index for computation.
   * @param result Reference to the accumulated result.
   * @param A Reference to the first input object.
   * @param B Reference to the second input object.
   */
  static void compute(T &result, const In_A_Type &A, const In_B_Type &B) {
    Row<T, In_A_Type, In_B_Type, M, N, I_idx, (N - 1)>::compute(result, A, B);
    Column<T, In_A_Type, In_B_Type, M, N, (I_idx - 1)>::compute(result, A, B);
  }
};

template <typename T, typename In_A_Type, typename In_B_Type, std::size_t M,
          std::size_t N>
struct Column<T, In_A_Type, In_B_Type, M, N, 0> {
  static void compute(T &result, const In_A_Type &A, const In_B_Type &B) {
    Row<T, In_A_Type, In_B_Type, M, N, 0, (N - 1)>::compute(result, A, B);
  }
};

/**
 * @brief Computes the inner product of two input matrices element-wise.
 *
 * This function template performs an inner product computation on two input
 * objects, `A` and `B`, accumulating the sum of element-wise products. It
 * ensures at compile-time that both input types have the same number of columns
 * and rows using static assertions.
 *
 * @tparam T The scalar type for the result.
 * @tparam In_A_Type Type of the first input object.
 * @tparam In_B_Type Type of the second input object.
 * @param[in] A Constant reference to the first input object.
 * @param[in] B Constant reference to the second input object.
 * @return The inner product of A and B as a scalar of type T.
 *
 * @note The function relies on helper templates `Row` and `Column` to perform
 * the actual computation. The dimensions (rows and cols) are determined at
 * compile-time from `In_A_Type`. Both input types must have matching
 * dimensions.
 */
template <typename T, typename In_A_Type, typename In_B_Type>
inline T compute(const In_A_Type &A, const In_B_Type &B) {
  static_assert(In_A_Type::ROWS == In_B_Type::ROWS,
                "In_A_Type::ROWS != In_B_Type::ROWS");
  static_assert(In_A_Type::COLS == In_B_Type::COLS,
                "In_A_Type::COLS != In_B_Type::COLS");

  constexpr std::size_t M = In_A_Type::ROWS;
  constexpr std::size_t N = In_A_Type::COLS;

  T result = static_cast<T>(0);

  Column<T, In_A_Type, In_B_Type, M, N, (M - 1)>::compute(result, A, B);

  return result;
}

} // namespace InnerProductOperation

/**
 * @brief Computes the inner product (dot product) of two matrices or arrays.
 *
 * This function calculates the inner product of two input matrices or arrays
 * `A` and `B` by summing the products of their corresponding elements. The
 * input types must have the same number of columns and rows, which is enforced
 * at compile time via static assertions.
 */
template <typename In_A_Type, typename In_B_Type>
inline auto inner_product(const In_A_Type &A, const In_B_Type &B) ->
    typename In_A_Type::Value_Type {

  static_assert(
      Base::Matrix::Is_Complex_Type<typename In_A_Type::Value_Type>::value ==
          false,
      "Complex types are not supported");
  static_assert(
      Base::Matrix::Is_Complex_Type<typename In_B_Type::Value_Type>::value ==
          false,
      "Complex types are not supported");

  static_assert(In_A_Type::ROWS == In_B_Type::ROWS,
                "In_A_Type::ROWS != In_B_Type::ROWS");
  static_assert(In_A_Type::COLS == In_B_Type::COLS,
                "In_A_Type::COLS != In_B_Type::COLS");

  return InnerProductOperation::compute<typename In_A_Type::Value_Type>(A, B);
}

/* Get */
namespace GetDenseMatrixOperation {

template <typename T, std::size_t M, std::size_t N, std::size_t COL,
          std::size_t COL_Index>
struct GetRow_Loop {
  /**
   * @brief Computes a specific operation on the input matrix and stores the
   * result in the provided result vector.
   *
   * This static function extracts a particular column (indexed by COL_Index)
   * from the input matrix and assigns it to the result vector. It then
   * recursively processes the remaining rows by invoking GetRow_Loop with a
   * decremented column index.
   *
   * @tparam T         The data type of the matrix elements.
   * @tparam M         The number of rows in the matrix.
   * @tparam N         The number of columns in the matrix.
   * @tparam COL       The row index to operate on.
   * @tparam COL_Index The current column index being processed.
   * @param matrix     The input dense matrix of type DenseMatrix_Type<T, M, N>.
   * @param result     The output dense matrix (vector) of type
   * DenseMatrix_Type<T, M, 1> to store the computed result.
   */
  static void compute(const DenseMatrix_Type<T, M, N> &matrix,
                      DenseMatrix_Type<T, M, 1> &result) {

    result.template set<COL_Index, 0>(matrix.template get<COL_Index, COL>());
    GetRow_Loop<T, M, N, COL, COL_Index - 1>::compute(matrix, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t COL>
struct GetRow_Loop<T, M, N, COL, 0> {
  /**
   * @brief Computes a specific operation on the input matrix and stores the
   * result in the provided result vector.
   *
   * This static function extracts the first row (index 0) from the input
   * matrix and assigns it to the result vector. It serves as the base case for
   * the recursive processing of rows.
   *
   * @tparam T   The data type of the matrix elements.
   * @tparam M   The number of rows in the matrix.
   * @tparam N   The number of columns in the matrix.
   * @tparam COL The row index to operate on.
   * @param matrix The input dense matrix of type DenseMatrix_Type<T, M, N>.
   * @param result The output dense matrix (vector) of type
   * DenseMatrix_Type<T, M, 1> to store the computed result.
   */
  static void compute(const DenseMatrix_Type<T, M, N> &matrix,
                      DenseMatrix_Type<T, M, 1> &result) {

    result.template set<0, 0>(matrix.template get<0, COL>());
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t COL>
using GetRow = GetRow_Loop<T, M, N, COL, M - 1>;

} // namespace GetDenseMatrixOperation

/**
 * @brief Extracts a specific row from a dense matrix.
 *
 * This function template retrieves a specified row from a dense matrix and
 * returns it as a new dense matrix (vector) of size M x 1.
 *
 * @tparam COL The index of the column to extract.
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input dense matrix from which to extract the row.
 * @return DenseMatrix_Type<T, M, 1> A dense matrix (vector) containing the
 * extracted row.
 */
template <std::size_t COL, typename T, std::size_t M, std::size_t N>
inline auto get_row(const DenseMatrix_Type<T, M, N> &matrix)
    -> DenseMatrix_Type<T, M, 1> {

  DenseMatrix_Type<T, M, 1> result;

  GetDenseMatrixOperation::GetRow<T, M, N, COL>::compute(matrix, result);

  return result;
}

namespace GetDiagMatrixOperation {

/**
 * @brief Retrieves a specific row from a diagonal matrix.
 *
 * This function template extracts a specified row from a diagonal matrix and
 * returns it as a new sparse matrix (vector) of size M x 1.
 *
 * @tparam M The number of rows in the diagonal matrix.
 * @tparam Index The index of the column to extract.
 * @return SparseAvailableGetRow<M, DiagAvailable<M>, Index> A sparse matrix
 * (vector) containing the extracted row.
 */
template <std::size_t M, std::size_t Index>
using DiagAvailableRow = SparseAvailableGetRow<M, DiagAvailable<M>, Index>;

/**
 * @brief Defines the type of a sparse matrix that can hold a specific row from
 * a diagonal matrix.
 *
 * This type alias creates a sparse matrix type that can hold a specific row
 * from a diagonal matrix, indexed by Index.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the diagonal matrix.
 * @tparam Index The index of the column to extract.
 */
template <typename T, std::size_t M, std::size_t Index>
using DiagAvailableRow_Type = SparseMatrix_Type<T, DiagAvailableRow<M, Index>>;

template <typename T, std::size_t M, std::size_t COL, std::size_t COL_Index>
struct GetRow_Loop {
  /**
   * @brief Computes a specific operation on the input diagonal matrix and
   * stores the result in the provided result vector.
   *
   * This static function extracts a particular column (indexed by COL_Index)
   * from the input diagonal matrix and assigns it to the result vector. It then
   * recursively processes the remaining rows by invoking GetRow_Loop with a
   * decremented column index.
   *
   * @tparam T         The data type of the matrix elements.
   * @tparam M         The number of rows in the matrix.
   * @tparam COL       The row index to operate on.
   * @tparam COL_Index The current column index being processed.
   * @param matrix     The input diagonal matrix of type DiagMatrix_Type<T, M>.
   * @param result     The output diagonal matrix (vector) of type
   * DiagAvailableRow_Type<T, M, COL> to store the computed result.
   */
  static void compute(const DiagMatrix_Type<T, M> &matrix,
                      DiagAvailableRow_Type<T, M, COL> &result) {

    result.template set<COL_Index, 0>(matrix.template get<COL_Index, COL>());
    GetRow_Loop<T, M, COL, COL_Index - 1>::compute(matrix, result);
  }
};

template <typename T, std::size_t M, std::size_t COL>
struct GetRow_Loop<T, M, COL, 0> {
  /**
   * @brief Computes a specific operation on the input diagonal matrix and
   * stores the result in the provided result vector.
   *
   * This static function extracts the first row (index 0) from the input
   * diagonal matrix and assigns it to the result vector. It serves as the base
   * case for the recursive processing of rows.
   *
   * @tparam T   The data type of the matrix elements.
   * @tparam M   The number of rows in the matrix.
   * @tparam COL The row index to operate on.
   * @param matrix The input diagonal matrix of type DiagMatrix_Type<T, M>.
   * @param result The output diagonal matrix (vector) of type
   * DiagAvailableRow_Type<T, M, COL> to store the computed result.
   */
  static void compute(const DiagMatrix_Type<T, M> &matrix,
                      DiagAvailableRow_Type<T, M, COL> &result) {

    result.template set<0, 0>(matrix.template get<0, COL>());
  }
};

/**
 * @brief Retrieves a specific row from a diagonal matrix.
 *
 * This function template extracts a specified row from a diagonal matrix and
 * returns it as a new sparse matrix (vector) of size M x 1.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the diagonal matrix.
 * @tparam COL The index of the column to extract.
 * @return DiagAvailableRow_Type<T, M, COL> A sparse matrix (vector) containing
 * the extracted row.
 */
template <typename T, std::size_t M, std::size_t COL>
using GetRow = GetRow_Loop<T, M, COL, M - 1>;

} // namespace GetDiagMatrixOperation

/**
 * @brief Extracts a specific row from a diagonal matrix.
 *
 * This function template retrieves a specified row from a diagonal matrix and
 * returns it as a new sparse matrix (vector) of size M x 1.
 *
 * @tparam COL The index of the column to extract.
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the diagonal matrix.
 * @param matrix The input diagonal matrix from which to extract the row.
 * @return GetDiagMatrixOperation::DiagAvailableRow_Type<T, M, COL> A sparse
 * matrix (vector) containing the extracted row.
 */
template <std::size_t COL, typename T, std::size_t M>
inline auto get_row(const DiagMatrix_Type<T, M> &matrix)
    -> GetDiagMatrixOperation::DiagAvailableRow_Type<T, M, COL> {

  GetDiagMatrixOperation::DiagAvailableRow_Type<T, M, COL> result;

  GetDiagMatrixOperation::GetRow<T, M, COL>::compute(matrix, result);

  return result;
}

namespace GetSparseMatrixOperation {

/**
 * @brief Retrieves a specific row from a sparse matrix.
 *
 * This function template extracts a specified row from a sparse matrix and
 * returns it as a new sparse matrix (vector) of size M x 1.
 *
 * @tparam M The number of rows in the sparse matrix.
 * @tparam Index The index of the column to extract.
 * @return SparseAvailableGetRow<M, SparseAvailable, Index> A sparse matrix
 * (vector) containing the extracted row.
 */
template <std::size_t M, std::size_t Index, typename SparseAvailable>
using SparseAvailableRow = SparseAvailableGetRow<M, SparseAvailable, Index>;

/**
 * @brief Defines the type of a sparse matrix that can hold a specific row from
 * a sparse matrix.
 *
 * This type alias creates a sparse matrix type that can hold a specific row
 * from a sparse matrix, indexed by Index.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam Index The index of the column to extract.
 * @tparam SparseAvailable The sparse matrix availability type.
 */
template <typename T, std::size_t M, std::size_t Index,
          typename SparseAvailable>
using SparseCol_Type =
    SparseMatrix_Type<T, SparseAvailableRow<M, Index, SparseAvailable>>;

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t COL, std::size_t COL_Index>
struct GetRow_Loop {
  /**
   * @brief Computes a specific operation on the input sparse matrix and stores
   * the result in the provided result vector.
   *
   * This static function extracts a particular column (indexed by COL_Index)
   * from the input sparse matrix and assigns it to the result vector. It then
   * recursively processes the remaining rows by invoking GetRow_Loop with a
   * decremented column index.
   *
   * @tparam T         The data type of the matrix elements.
   * @tparam M         The number of rows in the matrix.
   * @tparam N         The number of columns in the matrix.
   * @tparam SparseAvailable The sparse matrix availability type.
   * @tparam COL       The row index to operate on.
   * @tparam COL_Index The current column index being processed.
   * @param matrix     The input sparse matrix of type Matrix<DefSparse, T, M,
   * N, SparseAvailable>.
   * @param result     The output sparse matrix (vector) of type
   * SparseCol_Type<T, M, COL, SparseAvailable> to store the computed result.
   */
  static void compute(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                      SparseCol_Type<T, M, COL, SparseAvailable> &result) {

    result.template set<COL_Index, 0>(matrix.template get<COL_Index, COL>());
    GetRow_Loop<T, M, N, SparseAvailable, COL, COL_Index - 1>::compute(matrix,
                                                                       result);
  }
};

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t COL>
struct GetRow_Loop<T, M, N, SparseAvailable, COL, 0> {
  /**
   * @brief Computes a specific operation on the input sparse matrix and stores
   * the result in the provided result vector.
   *
   * This static function extracts the first row (index 0) from the input
   * sparse matrix and assigns it to the result vector. It serves as the base
   * case for the recursive processing of rows.
   *
   * @tparam T   The data type of the matrix elements.
   * @tparam M   The number of rows in the matrix.
   * @tparam N   The number of columns in the matrix.
   * @tparam SparseAvailable The sparse matrix availability type.
   * @tparam COL The row index to operate on.
   * @param matrix The input sparse matrix of type Matrix<DefSparse, T, M, N,
   * SparseAvailable>.
   * @param result The output sparse matrix (vector) of type
   * SparseCol_Type<T, M, COL, SparseAvailable> to store the computed result.
   */
  static void compute(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                      SparseCol_Type<T, M, COL, SparseAvailable> &result) {

    result.template set<0, 0>(matrix.template get<0, COL>());
  }
};

/**
 * @brief Retrieves a specific row from a sparse matrix.
 *
 * This function template extracts a specified row from a sparse matrix and
 * returns it as a new sparse matrix (vector) of size M x 1.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable The sparse matrix availability type.
 * @tparam COL The index of the column to extract.
 * @return SparseCol_Type<T, M, COL, SparseAvailable> A sparse matrix (vector)
 * containing the extracted row.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t COL>
using GetRow = GetRow_Loop<T, M, N, SparseAvailable, COL, M - 1>;

} // namespace GetSparseMatrixOperation

/**
 * @brief Extracts a specific row from a sparse matrix.
 *
 * This function template retrieves a specified row from a sparse matrix and
 * returns it as a new sparse matrix (vector) of size M x 1.
 *
 * @tparam COL The index of the column to extract.
 * @tparam T The data type of the matrix elements.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable The sparse matrix availability type.
 * @param matrix The input sparse matrix from which to extract the row.
 * @return GetSparseMatrixOperation::SparseCol_Type<T, M, COL,
 * SparseAvailable> A sparse matrix (vector) containing the extracted row.
 */
template <std::size_t COL, typename T, std::size_t M, std::size_t N,
          typename SparseAvailable>
inline auto get_row(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> GetSparseMatrixOperation::SparseCol_Type<T, M, COL, SparseAvailable> {

  GetSparseMatrixOperation::SparseCol_Type<T, M, COL, SparseAvailable> result;

  GetSparseMatrixOperation::GetRow<T, M, N, SparseAvailable, COL>::compute(
      matrix, result);

  return result;
}

/* Set */
namespace SetMatrixOperation {

template <typename Matrix_Type, typename RowVector_Type, std::size_t COL,
          std::size_t COL_Index>
struct SetRow_Loop {
  /**
   * @brief Computes a specific operation on the input matrix and sets the
   * corresponding row vector values.
   *
   * This static function assigns a particular column (indexed by COL_Index) of
   * the row vector to the specified row of the matrix. It then recursively
   * processes the remaining rows by invoking SetRow_Loop with a decremented
   * column index.
   *
   * @tparam Matrix_Type The type of the matrix to set values in.
   * @tparam RowVector_Type The type of the row vector containing values to set.
   * @tparam COL The row index to operate on.
   * @tparam COL_Index The current column index being processed.
   * @param matrix The input matrix of type Matrix_Type.
   * @param row_vector The row vector containing values to set in the matrix.
   */
  static void compute(Matrix_Type &matrix, const RowVector_Type &row_vector) {

    matrix.template set<COL_Index, COL>(
        row_vector.template get<COL_Index, 0>());
    SetRow_Loop<Matrix_Type, RowVector_Type, COL, COL_Index - 1>::compute(
        matrix, row_vector);
  }
};

template <typename Matrix_Type, typename RowVector_Type, std::size_t COL>
struct SetRow_Loop<Matrix_Type, RowVector_Type, COL, 0> {
  /**
   * @brief Computes a specific operation on the input matrix and sets the
   * corresponding row vector values.
   *
   * This static function assigns the first row (index 0) of the row vector
   * to the specified row of the matrix. It serves as the base case for the
   * recursive processing of rows.
   *
   * @tparam Matrix_Type The type of the matrix to set values in.
   * @tparam RowVector_Type The type of the row vector containing values to set.
   * @tparam COL The row index to operate on.
   * @param matrix The input matrix of type Matrix_Type.
   * @param row_vector The row vector containing values to set in the matrix.
   */
  static void compute(Matrix_Type &matrix, const RowVector_Type &row_vector) {

    matrix.template set<0, COL>(row_vector.template get<0, 0>());
  }
};

/**
 * @brief Sets a specific row in a matrix with values from a row vector.
 *
 * This type alias defines a loop structure that sets the specified row of the
 * matrix with values from the provided row vector. It processes all rows
 * of the row vector.
 *
 * @tparam Matrix_Type The type of the matrix to set values in.
 * @tparam RowVector_Type The type of the row vector containing values to set.
 * @tparam COL The row index to operate on.
 */
template <typename Matrix_Type, typename RowVector_Type, std::size_t COL>
using SetRow =
    SetRow_Loop<Matrix_Type, RowVector_Type, COL, (Matrix_Type::ROWS - 1)>;

} // namespace SetMatrixOperation

/**
 * @brief Sets a specific row in a matrix with values from a row vector.
 *
 * This function template assigns the values from a row vector to a specified
 * row in the matrix. It processes all rows of the row vector.
 *
 * @tparam COL The index of the column to set in the matrix.
 * @tparam Matrix_Type The type of the matrix to set values in.
 * @tparam RowVector_Type The type of the row vector containing values to set.
 * @param matrix The input matrix where the row will be set.
 * @param row_vector The row vector containing values to set in the matrix.
 */
template <std::size_t COL, typename Matrix_Type, typename RowVector_Type>
inline void set_row(Matrix_Type &matrix, const RowVector_Type &row_vector) {

  SetMatrixOperation::SetRow<Matrix_Type, RowVector_Type, COL>::compute(
      matrix, row_vector);
}

/* Part matrix substitute */
namespace PartMatrixOperation {

// when J_idx < N
template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct SubstituteColumn {
  /**
   * @brief Computes a specific operation on the input part matrix and
   * substitutes its values into the corresponding position in the All matrix.
   *
   * This static function assigns a particular value from the part matrix
   * (indexed by I and J_idx) to the All matrix at the position (Row_Offset + I,
   * Col_Offset + J_idx). It then recursively processes the remaining rows by
   * invoking SubstituteColumn with a decremented column index.
   *
   * @tparam Row_Offset The column offset for substitution.
   * @tparam Col_Offset The row offset for substitution.
   * @tparam All_Type The type of the All matrix.
   * @tparam Part_Type The type of the part matrix.
   * @tparam M The number of rows in the part matrix.
   * @tparam N The number of columns in the part matrix.
   * @tparam I The current row index being processed.
   * @tparam J_idx The current column index being processed.
   * @param All The All matrix where values are substituted.
   * @param Part The part matrix containing values to substitute.
   */
  static void compute(All_Type &All, const Part_Type &Part) {

    All.template set<(Row_Offset + I), (Col_Offset + J_idx)>(
        Part.template get<I, J_idx>());

    SubstituteColumn<Row_Offset, Col_Offset, All_Type, Part_Type, M, N, I,
                     (J_idx - 1)>::compute(All, Part);
  }
};

// column recursion termination
template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t I>
struct SubstituteColumn<Row_Offset, Col_Offset, All_Type, Part_Type, M, N, I,
                        0> {
  /**
   * @brief Computes a specific operation on the input part matrix and
   * substitutes its values into the corresponding position in the All matrix.
   *
   * This static function assigns the first row (index 0) from the part
   * matrix to the All matrix at the position (Row_Offset + I, Col_Offset).
   *
   * @tparam Row_Offset The column offset for substitution.
   * @tparam Col_Offset The row offset for substitution.
   * @tparam All_Type The type of the All matrix.
   * @tparam Part_Type The type of the part matrix.
   * @tparam M The number of rows in the part matrix.
   * @tparam N The number of columns in the part matrix.
   * @tparam I The current row index being processed.
   * @param All The All matrix where values are substituted.
   * @param Part The part matrix containing values to substitute.
   */
  static void compute(All_Type &All, const Part_Type &Part) {

    All.template set<(Row_Offset + I), Col_Offset>(Part.template get<I, 0>());
  }
};

// when I_idx < M
template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t I_idx>
struct SubstituteRow {
  /**
   * @brief Computes a specific operation on the input part matrix and
   * substitutes its values into the corresponding position in the All matrix.
   *
   * This static function assigns a particular value from the part matrix
   * (indexed by I_idx and J_idx) to the All matrix at the position (Row_Offset,
   * Col_Offset + I_idx). It then recursively processes the remaining cols by
   * invoking SubstituteRow with a decremented row index.
   *
   * @tparam Row_Offset The column offset for substitution.
   * @tparam Col_Offset The row offset for substitution.
   * @tparam All_Type The type of the All matrix.
   * @tparam Part_Type The type of the part matrix.
   * @tparam M The number of rows in the part matrix.
   * @tparam N The number of columns in the part matrix.
   * @tparam I_idx The current row index being processed.
   * @param All The All matrix where values are substituted.
   * @param Part The part matrix containing values to substitute.
   */
  static void compute(All_Type &All, const Part_Type &Part) {

    SubstituteColumn<Row_Offset, Col_Offset, All_Type, Part_Type, M, N, I_idx,
                     (N - 1)>::compute(All, Part);
    SubstituteRow<Row_Offset, Col_Offset, All_Type, Part_Type, M, N,
                  (I_idx - 1)>::compute(All, Part);
  }
};

// row recursion termination
template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N>
struct SubstituteRow<Row_Offset, Col_Offset, All_Type, Part_Type, M, N, 0> {
  /**
   * @brief Computes a specific operation on the input part matrix and
   * substitutes its values into the corresponding position in the All matrix.
   *
   * This static function assigns the first column (index 0) from the part
   * matrix to the All matrix at the position (Row_Offset, Col_Offset).
   *
   * @tparam Row_Offset The column offset for substitution.
   * @tparam Col_Offset The row offset for substitution.
   * @tparam All_Type The type of the All matrix.
   * @tparam Part_Type The type of the part matrix.
   * @tparam M The number of rows in the part matrix.
   * @tparam N The number of columns in the part matrix.
   * @param All The All matrix where values are substituted.
   * @param Part The part matrix containing values to substitute.
   */
  static void compute(All_Type &All, const Part_Type &Part) {

    SubstituteColumn<Row_Offset, Col_Offset, All_Type, Part_Type, M, N, 0,
                     (N - 1)>::compute(All, Part);
  }
};

/**
 * @brief Substitutes a part matrix into a larger matrix at specified offsets.
 *
 * This function template substitutes the values from a part matrix into a
 * larger matrix (All) at specified column and row offsets. It ensures that the
 * All matrix has enough space to accommodate the part matrix.
 *
 * @tparam Row_Offset The column offset for substitution.
 * @tparam Col_Offset The row offset for substitution.
 * @tparam All_Type The type of the All matrix.
 * @tparam Part_Type The type of the part matrix.
 * @param All The All matrix where values are substituted.
 * @param Part The part matrix containing values to substitute.
 */
template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type>
inline void substitute_each(All_Type &All, const Part_Type &Part) {

  static_assert(
      All_Type::ROWS >= (Part_Type::ROWS + Row_Offset),
      "All matrix must have enough rows to substitute the part matrix.");
  static_assert(
      All_Type::COLS >= (Part_Type::COLS + Col_Offset),
      "All matrix must have enough cols to substitute the part matrix.");

  SubstituteRow<Row_Offset, Col_Offset, All_Type, Part_Type, Part_Type::ROWS,
                Part_Type::COLS, (Part_Type::ROWS - 1)>::compute(All, Part);
}

template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t TupleRow_Count,
          std::size_t TupleRow_Offset, std::size_t TupleCol_Offset,
          std::size_t TupleCol_Index>
struct TupleColumn {
  /**
   * @brief Substitutes a specific column of a tuple into the All matrix.
   *
   * This static function substitutes the values from a specific column of the
   * tuple (indexed by THIS_TUPLE_INDEX) into the All matrix at the specified
   * offsets. It then recursively processes the remaining cols by invoking
   * substitute with a decremented row index.
   *
   * @tparam M The number of rows in the matrix.
   * @tparam N The number of columns in the matrix.
   * @tparam All_Type The type of the All matrix.
   * @tparam ArgsTuple_Type The type of the tuple containing arguments.
   * @tparam TupleRow_Count The number of rows in the tuple.
   * @tparam TupleRow_Offset The column offset for substitution.
   * @tparam TupleCol_Offset The row offset for substitution.
   * @tparam TupleCol_Index The current row index being processed.
   * @param All The All matrix where values are substituted.
   * @param args The tuple containing arguments to substitute.
   */
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {

    constexpr std::size_t THIS_TUPLE_INDEX =
        N - TupleCol_Index + (TupleRow_Count * N);

    using ArgType =
        typename std::remove_reference<decltype(std::get<THIS_TUPLE_INDEX>(
            args))>::type;

    constexpr std::size_t EACH_COLUMN_SIZE = ArgType::COLS;

    substitute_each<TupleRow_Offset, TupleCol_Offset>(
        All, std::get<THIS_TUPLE_INDEX>(args));
    TupleColumn<M, N, All_Type, ArgsTuple_Type, TupleRow_Count, TupleRow_Offset,
                (TupleCol_Offset + EACH_COLUMN_SIZE),
                (TupleCol_Index - 1)>::substitute(All, args);
  }
};

template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t TupleRow_Count,
          std::size_t TupleRow_Offset, std::size_t TupleCol_Offset>
struct TupleColumn<M, N, All_Type, ArgsTuple_Type, TupleRow_Count,
                   TupleRow_Offset, TupleCol_Offset, 0> {
  /**
   * @brief Substitutes a specific column of a tuple into the All matrix.
   *
   * This static function does nothing when the row index is 0, serving as the
   * base case for the recursive processing of cols.
   *
   * @tparam M The number of rows in the matrix.
   * @tparam N The number of columns in the matrix.
   * @tparam All_Type The type of the All matrix.
   * @tparam ArgsTuple_Type The type of the tuple containing arguments.
   * @tparam TupleRow_Count The number of rows in the tuple.
   * @tparam TupleRow_Offset The column offset for substitution.
   * @tparam TupleCol_Offset The row offset for substitution.
   * @param All The All matrix where values would be substituted (not used
   * here).
   * @param args The tuple containing arguments to substitute (not used here).
   */
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {
    // Do Nothing
    static_cast<void>(All);
    static_cast<void>(args);
  }
};

template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t TupleRow_Offset,
          std::size_t TupleRow_Index>
struct TupleRow {
  /**
   * @brief Substitutes a specific row of a tuple into the All matrix.
   *
   * This static function substitutes the values from a specific row of the
   * tuple (indexed by THIS_TUPLE_INDEX) into the All matrix at the specified
   * offsets. It then recursively processes the remaining rows by invoking
   * substitute with a decremented column index.
   *
   * @tparam M The number of rows in the matrix.
   * @tparam N The number of columns in the matrix.
   * @tparam All_Type The type of the All matrix.
   * @tparam ArgsTuple_Type The type of the tuple containing arguments.
   * @tparam TupleRow_Offset The column offset for substitution.
   * @tparam TupleRow_Index The current column index being processed.
   * @param All The All matrix where values are substituted.
   * @param args The tuple containing arguments to substitute.
   */
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {

    constexpr std::size_t TUPLECOL_COUNT = M - TupleRow_Index;

    constexpr std::size_t THIS_TUPLE_INDEX = TUPLECOL_COUNT * N;

    using ArgType =
        typename std::remove_reference<decltype(std::get<THIS_TUPLE_INDEX>(
            args))>::type;

    constexpr std::size_t EACH_ROW_SIZE = ArgType::ROWS;

    TupleColumn<M, N, All_Type, ArgsTuple_Type, TUPLECOL_COUNT, TupleRow_Offset,
                0, N>::substitute(All, args);

    TupleRow<M, N, All_Type, ArgsTuple_Type, TupleRow_Offset + EACH_ROW_SIZE,
             (TupleRow_Index - 1)>::substitute(All, args);
  }
};

template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t TupleRow_Offset>
struct TupleRow<M, N, All_Type, ArgsTuple_Type, TupleRow_Offset, 0> {
  /**
   * @brief Substitutes a specific row of a tuple into the All matrix.
   *
   * This static function does nothing when the column index is 0, serving as
   * the base case for the recursive processing of rows.
   *
   * @tparam M The number of rows in the matrix.
   * @tparam N The number of columns in the matrix.
   * @tparam All_Type The type of the All matrix.
   * @tparam ArgsTuple_Type The type of the tuple containing arguments.
   * @tparam TupleRow_Offset The column offset for substitution.
   * @param All The All matrix where values would be substituted (not used
   * here).
   * @param args The tuple containing arguments to substitute (not used here).
   */
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {
    // Do Nothing
    static_cast<void>(All);
    static_cast<void>(args);
  }
};

} // namespace PartMatrixOperation

/* Substitute same size Matrix */

/**
 * @brief Substitutes a source matrix into a destination matrix of the same
 * size.
 *
 * This function template substitutes the values from a source matrix into a
 * destination matrix, ensuring that both matrices have the same number of
 * elements.
 *
 * @tparam From_Type The type of the source matrix.
 * @tparam To_Type The type of the destination matrix.
 * @param to_matrix The destination matrix where values are substituted.
 * @param from_matrix The source matrix containing values to substitute.
 */
template <typename From_Type, typename To_Type>
inline void substitute_matrix(To_Type &to_matrix,
                              const From_Type &from_matrix) {

  static_assert(From_Type::ROWS * From_Type::COLS ==
                    To_Type::ROWS * To_Type::COLS,
                "The number of elements in the source and destination matrices "
                "must be the same.");

  PartMatrixOperation::substitute_each<0, 0>(to_matrix, from_matrix);
}

/* Substitute small size Matrix to large size Matrix */

/**
 * @brief Substitutes a small matrix into a larger matrix at specified offsets.
 *
 * This function template substitutes the values from a small matrix into a
 * larger matrix at specified column and row offsets. It ensures that the large
 * matrix has enough space to accommodate the small matrix.
 *
 * @tparam Row_Offset The column offset for substitution.
 * @tparam Col_Offset The row offset for substitution.
 * @tparam Large_Type The type of the large matrix.
 * @tparam Small_Type The type of the small matrix.
 * @param Large The large matrix where values are substituted.
 * @param Small The small matrix containing values to substitute.
 */
template <std::size_t Row_Offset, std::size_t Col_Offset, typename Large_Type,
          typename Small_Type>
inline void substitute_part_matrix(Large_Type &Large, const Small_Type &Small) {

  static_assert(Large_Type::ROWS >= (Small_Type::ROWS + Row_Offset),
                "Large matrix must have enough rows to substitute the small "
                "matrix.");
  static_assert(Large_Type::COLS >= (Small_Type::COLS + Col_Offset),
                "Large matrix must have enough cols to substitute the small "
                "matrix.");

  PartMatrixOperation::substitute_each<Row_Offset, Col_Offset>(Large, Small);
}

/* Concatenate block, any size */
namespace ConcatenateBlockOperation {

template <std::size_t Row_Offset, std::size_t N, typename ArgsTuple_Type,
          std::size_t Col_Index>
struct ConcatenateBlockRows {

  using Arg_Type = typename std::tuple_element<(Row_Offset + Col_Index),
                                               ArgsTuple_Type>::type;

  using type = ConcatenateHorizontally_Type<
      typename ConcatenateBlockRows<Row_Offset, N, ArgsTuple_Type,
                                    (Col_Index - 1)>::type,
      Arg_Type>;
};

template <std::size_t Row_Offset, std::size_t N, typename ArgsTuple_Type>
struct ConcatenateBlockRows<Row_Offset, N, ArgsTuple_Type, 0> {

  using type = typename std::tuple_element<Row_Offset, ArgsTuple_Type>::type;
};

template <std::size_t N, typename ArgsTuple_Type, std::size_t Row_Index>
struct ConcatenateBlockColumns {

  using Arg_Type = typename ConcatenateBlockRows<(Row_Index * N), N,
                                                 ArgsTuple_Type, (N - 1)>::type;

  using type =
      ConcatenateVertically_Type<typename ConcatenateBlockColumns<
                                     N, ArgsTuple_Type, (Row_Index - 1)>::type,
                                 Arg_Type>;
};

template <std::size_t N, typename ArgsTuple_Type>
struct ConcatenateBlockColumns<N, ArgsTuple_Type, 0> {

  using type =
      typename ConcatenateBlockRows<0, N, ArgsTuple_Type, (N - 1)>::type;
};

template <std::size_t M, std::size_t N, typename ArgsTuple_Type>
struct ConcatenateBlock {

  using type =
      typename ConcatenateBlockColumns<N, ArgsTuple_Type, (M - 1)>::type;
};

template <std::size_t M, std::size_t N, typename Tuple, typename... Args>
struct ConcatenateArgsType {
  using type = typename ConcatenateBlock<
      M, N,
      decltype(std::tuple_cat(std::declval<Tuple>(),
                              std::make_tuple(std::declval<Args>()...)))>::type;
};

/**
 * @brief Type alias for the concatenated arguments type.
 *
 * This type alias defines the type of the concatenated arguments based on the
 * specified M, N, Tuple, and Args.
 *
 * @tparam M The number of rows in the block.
 * @tparam N The number of columns in the block.
 * @tparam Tuple The tuple containing previous arguments.
 * @tparam Args The additional arguments to concatenate.
 */
template <std::size_t M, std::size_t N, typename Tuple, typename... Args>
using ArgsType_t = typename ConcatenateArgsType<M, N, Tuple, Args...>::type;

/**
 * @brief Concatenates arguments into a block matrix.
 *
 * This function template concatenates the provided arguments into a block
 * matrix of size M x N, ensuring that the total number of arguments matches M
 * * N.
 *
 * @tparam M The number of rows in the block.
 * @tparam N The number of columns in the block.
 * @tparam Concatenate_Type The type of the concatenated matrix.
 * @tparam Tuple The tuple containing previous arguments.
 * @param Concatenated The concatenated matrix to update.
 * @param previousArgs The tuple containing previously concatenated arguments.
 */
template <std::size_t M, std::size_t N, typename Concatenate_Type,
          typename Tuple, typename Last>
inline void concatenate_args(Concatenate_Type &Concatenated,
                             const Tuple &previousArgs, Last last) {

  auto all_args = std::tuple_cat(previousArgs, std::make_tuple(last));

  using UpdatedArgsType = decltype(all_args);

  constexpr std::size_t TUPLE_SIZE = std::tuple_size<UpdatedArgsType>::value;

  static_assert(TUPLE_SIZE == (M * N),
                "Number of arguments must be equal to M * N.");

  using ConcatenateMatrix_Type =
      typename ConcatenateBlock<M, N, UpdatedArgsType>::type;

  PartMatrixOperation::TupleRow<M, N, ConcatenateMatrix_Type, UpdatedArgsType,
                                0, M>::substitute(Concatenated, all_args);
}

/**
 * @brief Concatenates multiple arguments into a block matrix.
 *
 * This function template recursively concatenates the provided arguments into
 * a block matrix of size M x N, ensuring that the total number of arguments
 * matches M * N.
 *
 * @tparam M The number of rows in the block.
 * @tparam N The number of columns in the block.
 * @tparam Concatenate_Type The type of the concatenated matrix.
 * @tparam Tuple The tuple containing previous arguments.
 * @tparam First The first argument to concatenate.
 * @tparam Rest The remaining arguments to concatenate.
 */
template <std::size_t M, std::size_t N, typename Concatenate_Type,
          typename Tuple, typename First, typename... Rest>
inline void concatenate_args(Concatenate_Type &Concatenated,
                             const Tuple &previousArgs, First first,
                             Rest... rest) {

  auto updatedArgs = std::tuple_cat(previousArgs, std::make_tuple(first));

  return concatenate_args<M, N>(Concatenated, updatedArgs, rest...);
}

/**
 * @brief Calculates the concatenated block matrix from the provided arguments.
 *
 * This function template initializes the concatenated block matrix and
 * recursively concatenates the provided arguments into it.
 *
 * @tparam M The number of rows in the block.
 * @tparam N The number of columns in the block.
 * @tparam Args The types of the arguments to concatenate.
 * @param Concatenated The concatenated matrix to update.
 * @param args The arguments to concatenate into the block matrix.
 */
template <std::size_t M, std::size_t N, typename... Args>
inline void calculate(ArgsType_t<M, N, std::tuple<>, Args...> &Concatenated,
                      Args... args) {
  static_assert(M > 0, "M must be greater than 0.");
  static_assert(N > 0, "N must be greater than 0.");

  return concatenate_args<M, N>(Concatenated, std::make_tuple(), args...);
}

} // namespace ConcatenateBlockOperation

/**
 * @brief Updates a concatenated block matrix with new arguments.
 *
 * This function template updates the provided concatenated block matrix with
 * the specified arguments, ensuring that the total number of arguments matches
 * M * N.
 *
 * @tparam M The number of rows in the block.
 * @tparam N The number of columns in the block.
 * @tparam Concatenate_Type The type of the concatenated matrix.
 * @tparam Args The types of the arguments to concatenate.
 * @param Concatenated The concatenated matrix to update.
 * @param args The arguments to concatenate into the block matrix.
 */
template <std::size_t M, std::size_t N, typename Concatenate_Type,
          typename... Args>
inline void update_block_concatenated_matrix(Concatenate_Type &Concatenated,
                                             Args... args) {

  static_assert(M > 0, "M must be greater than 0.");
  static_assert(N > 0, "N must be greater than 0.");

  ConcatenateBlockOperation::calculate<M, N>(Concatenated, args...);
}

/**
 * @brief Concatenates multiple arguments into a block matrix.
 *
 * This function template concatenates the provided arguments into a block
 * matrix of size M x N, ensuring that the total number of arguments matches M
 * * N.
 *
 * @tparam M The number of rows in the block.
 * @tparam N The number of columns in the block.
 * @tparam Args The types of the arguments to concatenate.
 * @return ConcatenateBlockOperation::ArgsType_t<M, N, std::tuple<>, Args...>
 * A concatenated block matrix containing the provided arguments.
 */
template <std::size_t M, std::size_t N, typename... Args>
inline auto concatenate_block(Args... args)
    -> ConcatenateBlockOperation::ArgsType_t<M, N, std::tuple<>, Args...> {

  ConcatenateBlockOperation::ArgsType_t<M, N, std::tuple<>, Args...>
      Concatenated;

  ConcatenateBlockOperation::calculate<M, N>(Concatenated, args...);

  return Concatenated;
}

/* Concatenate block Type */
template <std::size_t M, std::size_t N, typename... Args>
using ConcatenateBlock_Type =
    typename ConcatenateBlockOperation::ArgsType_t<M, N, std::tuple<>, Args...>;

/* tile */
namespace TileOperation {

template <std::size_t M, std::size_t N, std::size_t Count, typename MATRIX_Type,
          typename... Args>
struct GenerateTileTypes {
  using type = typename GenerateTileTypes<M, N, (Count - 1), MATRIX_Type,
                                          MATRIX_Type, Args...>::type;
};

template <std::size_t M, std::size_t N, typename MATRIX_Type, typename... Args>
struct GenerateTileTypes<M, N, 0, MATRIX_Type, Args...> {
  using type = ConcatenateBlock_Type<M, N, Args...>;
};

} // namespace TileOperation

/**
 * @brief Type alias for a tile type with specified dimensions and matrix type.
 *
 * This type alias defines a tile type based on the specified M, N, and
 * MATRIX_Type. It uses the TileOperation::GenerateTileTypes to generate the
 * appropriate tile type.
 *
 * @tparam M The number of rows in the tile.
 * @tparam N The number of columns in the tile.
 * @tparam MATRIX_Type The type of the matrix used in the tile.
 */
template <std::size_t M, std::size_t N, typename MATRIX_Type>
using Tile_Type =
    typename TileOperation::GenerateTileTypes<M, N, M * N, MATRIX_Type>::type;

namespace TileOperation {

/**
 * @brief Index sequence for tile operations.
 *
 * This struct defines an index sequence for tile operations, allowing for
 * compile-time generation of indices for tile elements.
 *
 * @tparam Indices The indices for the tile elements.
 */
template <std::size_t... Indices> struct index_sequence_for_tile {};

/**
 * @brief Helper struct to generate an index sequence for tile operations.
 *
 * This struct recursively generates an index sequence for tile operations,
 * allowing for compile-time generation of indices based on the specified size
 * N.
 *
 * @tparam N The size of the index sequence to generate.
 * @tparam Indices The indices for the tile elements (used in recursion).
 */
template <std::size_t N, std::size_t... Indices>
struct make_index_sequence_for_tile_impl
    : make_index_sequence_for_tile_impl<N - 1, N - 1, Indices...> {};

/**
 * @brief Specialization of make_index_sequence_for_tile_impl for the base case.
 *
 * This specialization defines the index sequence when N is 0, providing the
 * final type for the index sequence.
 *
 * @tparam Indices The indices for the tile elements (used in recursion).
 */
template <std::size_t... Indices>
struct make_index_sequence_for_tile_impl<0, Indices...> {
  using type = index_sequence_for_tile<Indices...>;
};

/**
 * @brief Generates an index sequence for tile operations.
 *
 * This type alias generates an index sequence for tile operations based on the
 * specified size N. It uses the make_index_sequence_for_tile_impl to create
 * the appropriate index sequence type.
 *
 * @tparam N The size of the index sequence to generate.
 */
template <std::size_t N>
using make_index_sequence_for_tile =
    typename make_index_sequence_for_tile_impl<N>::type;

/**
 * @brief Repeats a matrix type a specified number of times.
 *
 * This struct recursively generates a type that represents a tuple of the
 * specified matrix type repeated a given number of times.
 *
 * @tparam Count The number of times to repeat the matrix type.
 * @tparam MATRIX_Type The type of the matrix to repeat.
 * @tparam Args The types of the arguments to include in the tuple.
 */
template <std::size_t Count, typename MATRIX_Type, typename... Args>
struct RepeatMatrix {
  using type =
      typename RepeatMatrix<Count - 1, MATRIX_Type, MATRIX_Type, Args...>::type;
};

/**
 * @brief Specialization of RepeatMatrix for the base case.
 *
 * This specialization defines the type when Count is 0, providing a tuple of
 * the specified argument types.
 *
 * @tparam MATRIX_Type The type of the matrix to repeat (not used here).
 * @tparam Args The types of the arguments to include in the tuple.
 */
template <typename MATRIX_Type, typename... Args>
struct RepeatMatrix<0, MATRIX_Type, Args...> {
  using type = std::tuple<Args...>;
};

/**
 * @brief Updates a concatenated matrix with multiple matrices.
 *
 * This function template updates the provided concatenated matrix with the
 * specified matrices, ensuring that the total number of matrices matches M * N.
 *
 * @tparam M The number of rows in the tile.
 * @tparam N The number of columns in the tile.
 * @tparam MATRIX_Type The type of the matrices to concatenate.
 * @tparam Indices The indices for the tile elements (used in recursion).
 * @param Concatenated The concatenated matrix to update.
 * @param matrix The matrices to concatenate into the block matrix.
 */
template <std::size_t M, std::size_t N, typename MATRIX_Type,
          std::size_t... Indices>
inline void calculate(Tile_Type<M, N, MATRIX_Type> &Concatenated,
                      const MATRIX_Type &matrix,
                      index_sequence_for_tile<Indices...>) {

  update_block_concatenated_matrix<M, N>(
      Concatenated, (static_cast<void>(Indices), matrix)...);
}

} // namespace TileOperation

/**
 * @brief Updates a concatenated matrix with a single matrix.
 *
 * This function template updates the provided concatenated matrix with the
 * specified matrix, ensuring that the total number of elements matches M * N.
 *
 * @tparam M The number of rows in the tile.
 * @tparam N The number of columns in the tile.
 * @tparam MATRIX_Type The type of the matrix to concatenate.
 * @param Concatenated The concatenated matrix to update.
 * @param matrix The matrix to concatenate into the block matrix.
 */
template <std::size_t M, std::size_t N, typename MATRIX_Type>
inline void
update_tile_concatenated_matrix(Tile_Type<M, N, MATRIX_Type> &Concatenated,
                                const MATRIX_Type &matrix) {

  static_assert(M > 0, "M must be greater than 0.");
  static_assert(N > 0, "N must be greater than 0.");

  constexpr std::size_t TotalCount = M * N;

  TileOperation::calculate<M, N>(
      Concatenated, matrix,
      TileOperation::make_index_sequence_for_tile<TotalCount>{});
}

/**
 * @brief Concatenates a matrix into a tile of specified dimensions.
 *
 * This function template concatenates the provided matrix into a tile of size
 * M x N, ensuring that the total number of elements matches M * N.
 *
 * @tparam M The number of rows in the tile.
 * @tparam N The number of columns in the tile.
 * @tparam MATRIX_Type The type of the matrix to concatenate.
 * @param matrix The matrix to concatenate into the tile.
 * @return Tile_Type<M, N, MATRIX_Type> A tile containing the concatenated
 * matrix.
 */
template <std::size_t M, std::size_t N, typename MATRIX_Type>
inline auto concatenate_tile(const MATRIX_Type &matrix)
    -> Tile_Type<M, N, MATRIX_Type> {

  Tile_Type<M, N, MATRIX_Type> Concatenated;

  update_tile_concatenated_matrix<M, N>(Concatenated, matrix);

  return Concatenated;
}

namespace ReshapeOperation {

// when J_idx < N
template <typename To_Type, typename From_Type, std::size_t I,
          std::size_t J_idx>
struct Row {
  /**
   * @brief Substitutes a specific column of the from_matrix into the to_matrix.
   *
   * This static function substitutes the values from a specific column of the
   * from_matrix (indexed by I and J_idx) into the to_matrix at the
   * corresponding position. It then recursively processes the remaining rows
   * by invoking substitute with a decremented column index.
   *
   * @tparam To_Type The type of the destination matrix.
   * @tparam From_Type The type of the source matrix.
   * @tparam I The current row index being processed.
   * @tparam J_idx The current column index being processed.
   * @param to_matrix The destination matrix where values are substituted.
   * @param from_matrix The source matrix containing values to substitute.
   */
  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {

    constexpr std::size_t NUMBER_OF_ELEMENT = I * From_Type::COLS + J_idx;

    constexpr std::size_t FROM_MATRIX_ROW_INDEX =
        static_cast<std::size_t>(NUMBER_OF_ELEMENT / From_Type::ROWS);
    constexpr std::size_t FROM_MATRIX_COL_INDEX =
        NUMBER_OF_ELEMENT - FROM_MATRIX_ROW_INDEX * From_Type::ROWS;

    static_assert(FROM_MATRIX_COL_INDEX < From_Type::ROWS,
                  "The column index is out of range for the from_matrix.");
    static_assert(FROM_MATRIX_ROW_INDEX < From_Type::COLS,
                  "The row index is out of range for the from_matrix.");

    constexpr std::size_t TO_MATRIX_ROW_INDEX =
        static_cast<std::size_t>(NUMBER_OF_ELEMENT / To_Type::ROWS);
    constexpr std::size_t TO_MATRIX_COL_INDEX =
        NUMBER_OF_ELEMENT - TO_MATRIX_ROW_INDEX * To_Type::ROWS;

    static_assert(TO_MATRIX_COL_INDEX < To_Type::ROWS,
                  "The column index is out of range for the to_matrix.");
    static_assert(TO_MATRIX_ROW_INDEX < To_Type::COLS,
                  "The row index is out of range for the to_matrix.");

    to_matrix.template set<TO_MATRIX_COL_INDEX, TO_MATRIX_ROW_INDEX>(
        from_matrix
            .template get<FROM_MATRIX_COL_INDEX, FROM_MATRIX_ROW_INDEX>());
    Row<To_Type, From_Type, I, J_idx - 1>::substitute(to_matrix, from_matrix);
  }
};

// column recursion termination
template <typename To_Type, typename From_Type, std::size_t I>
struct Row<To_Type, From_Type, I, 0> {
  /**
   * @brief Substitutes the first row of the from_matrix into the to_matrix.
   *
   * This static function substitutes the values from the first row (index 0)
   * of the from_matrix into the to_matrix at the corresponding position.
   *
   * @tparam To_Type The type of the destination matrix.
   * @tparam From_Type The type of the source matrix.
   * @param to_matrix The destination matrix where values are substituted.
   * @param from_matrix The source matrix containing values to substitute.
   */
  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {

    constexpr std::size_t NUMBER_OF_ELEMENT = I * From_Type::COLS;

    constexpr std::size_t FROM_MATRIX_ROW_INDEX =
        static_cast<std::size_t>(NUMBER_OF_ELEMENT / From_Type::ROWS);
    constexpr std::size_t FROM_MATRIX_COL_INDEX =
        NUMBER_OF_ELEMENT - FROM_MATRIX_ROW_INDEX * From_Type::ROWS;

    static_assert(FROM_MATRIX_COL_INDEX < From_Type::ROWS,
                  "The column index is out of range for the from_matrix.");
    static_assert(FROM_MATRIX_ROW_INDEX < From_Type::COLS,
                  "The row index is out of range for the from_matrix.");

    constexpr std::size_t TO_MATRIX_ROW_INDEX =
        static_cast<std::size_t>(NUMBER_OF_ELEMENT / To_Type::ROWS);
    constexpr std::size_t TO_MATRIX_COL_INDEX =
        NUMBER_OF_ELEMENT - TO_MATRIX_ROW_INDEX * To_Type::ROWS;

    static_assert(TO_MATRIX_COL_INDEX < To_Type::ROWS,
                  "The column index is out of range for the to_matrix.");
    static_assert(TO_MATRIX_ROW_INDEX < To_Type::COLS,
                  "The row index is out of range for the to_matrix.");

    to_matrix.template set<TO_MATRIX_COL_INDEX, TO_MATRIX_ROW_INDEX>(
        from_matrix
            .template get<FROM_MATRIX_COL_INDEX, FROM_MATRIX_ROW_INDEX>());
  }
};

// when I_idx < M
template <typename To_Type, typename From_Type, std::size_t M, std::size_t N,
          std::size_t I_idx>
struct Column {
  /**
   * @brief Substitutes a specific row of the from_matrix into the to_matrix.
   *
   * This static function substitutes the values from a specific row of the
   * from_matrix (indexed by I_idx) into the to_matrix at the corresponding
   * position. It then recursively processes the remaining cols by invoking
   * substitute with a decremented row index.
   *
   * @tparam To_Type The type of the destination matrix.
   * @tparam From_Type The type of the source matrix.
   * @tparam M The number of rows in the matrices.
   * @tparam N The number of columns in the matrices.
   * @tparam I_idx The current row index being processed.
   * @param to_matrix The destination matrix where values are substituted.
   * @param from_matrix The source matrix containing values to substitute.
   */
  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {
    Row<To_Type, From_Type, I_idx, N - 1>::substitute(to_matrix, from_matrix);
    Column<To_Type, From_Type, M, N, I_idx - 1>::substitute(to_matrix,
                                                            from_matrix);
  }
};

// row recursion termination
template <typename To_Type, typename From_Type, std::size_t M, std::size_t N>
struct Column<To_Type, From_Type, M, N, 0> {
  /**
   * @brief Substitutes the first column of the from_matrix into the to_matrix.
   *
   * This static function substitutes the values from the first column (index 0)
   * of the from_matrix into the to_matrix at the corresponding position.
   *
   * @tparam To_Type The type of the destination matrix.
   * @tparam From_Type The type of the source matrix.
   * @param to_matrix The destination matrix where values are substituted.
   * @param from_matrix The source matrix containing values to substitute.
   */
  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {
    Row<To_Type, From_Type, 0, N - 1>::substitute(to_matrix, from_matrix);
  }
};

/**
 * @brief Substitutes values from a source matrix into a destination matrix.
 *
 * This function template substitutes the values from a source matrix into a
 * destination matrix, ensuring that both matrices have the same number of
 * elements.
 *
 * @tparam To_Type The type of the destination matrix.
 * @tparam From_Type The type of the source matrix.
 * @param to_matrix The destination matrix where values are substituted.
 * @param from_matrix The source matrix containing values to substitute.
 */
template <typename To_Type, typename From_Type>
inline void substitute(To_Type &to_matrix, const From_Type &from_matrix) {
  Column<To_Type, From_Type, From_Type::ROWS, From_Type::COLS,
         (From_Type::ROWS - 1)>::substitute(to_matrix, from_matrix);
}

} // namespace ReshapeOperation

/**
 * @brief Updates a reshaped matrix with values from a source matrix.
 *
 * This function template updates the provided reshaped matrix with the values
 * from a source matrix, ensuring that both matrices have the same number of
 * elements.
 *
 * @tparam To_Type The type of the destination matrix.
 * @tparam From_Type The type of the source matrix.
 * @param to_matrix The destination matrix where values are substituted.
 * @param from_matrix The source matrix containing values to substitute.
 */
template <typename To_Type, typename From_Type>
inline void update_reshaped_matrix(To_Type &to_matrix,
                                   const From_Type &from_matrix) {

  static_assert(From_Type::ROWS * From_Type::COLS ==
                    To_Type::ROWS * To_Type::COLS,
                "The number of elements in the source and destination matrices "
                "must be the same.");

  ReshapeOperation::substitute(to_matrix, from_matrix);
}

/* Reshape (So far, it can only outputs Dense Matrix.) */

/**
 * @brief Reshapes a source matrix into a destination matrix of specified size.
 *
 * This function template reshapes the provided source matrix into a new
 * destination matrix of size M x N, ensuring that the total number of elements
 * matches between the source and destination matrices.
 *
 * @tparam M The number of rows in the destination matrix.
 * @tparam N The number of columns in the destination matrix.
 * @tparam From_Type The type of the source matrix.
 * @param from_matrix The source matrix to reshape.
 * @return DenseMatrix_Type<typename From_Type::Value_Complex_Type, M, N>
 * A reshaped matrix containing the values from the source matrix.
 */
template <std::size_t M, std::size_t N, typename From_Type>
inline auto reshape(const From_Type &from_matrix)
    -> DenseMatrix_Type<typename From_Type::Value_Complex_Type, M, N> {

  static_assert(From_Type::ROWS * From_Type::COLS == M * N,
                "The number of elements in the source and destination matrices "
                "must be the same.");

  using To_Type =
      DenseMatrix_Type<typename From_Type::Value_Complex_Type, M, N>;

  To_Type to_matrix;

  ReshapeOperation::substitute(to_matrix, from_matrix);

  return to_matrix;
}

namespace NormalizationOperation {

/* Real value operation */

template <typename T, typename Matrix_Type, std::size_t I, std::size_t J,
          bool Value_Exists>
struct RealSumSquaresTemplate {};

template <typename T, typename Matrix_Type, std::size_t I, std::size_t J>
struct RealSumSquaresTemplate<T, Matrix_Type, I, J, false> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    static_cast<void>(sum_of_squares);
    static_cast<void>(matrix);
    // Do nothing.
  }
};

template <typename T, typename Matrix_Type, std::size_t I, std::size_t J>
struct RealSumSquaresTemplate<T, Matrix_Type, I, J, true> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    T value = matrix.template get<I, J>();
    sum_of_squares += value * value;
  }
};

// when J_idx < N
template <typename T, typename Matrix_Type, std::size_t I, std::size_t J_idx>
struct RealColumn {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    RealSumSquaresTemplate<T, Matrix_Type, I, J_idx,
                           Matrix_Type::SparseAvailable_Type::lists[I][J_idx]>::
        sum_squares(sum_of_squares, matrix);

    RealColumn<T, Matrix_Type, I, J_idx - 1>::sum_squares(sum_of_squares,
                                                          matrix);
  }
};

// column recursion termination
template <typename T, typename Matrix_Type, std::size_t I>
struct RealColumn<T, Matrix_Type, I, 0> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    RealSumSquaresTemplate<T, Matrix_Type, I, 0,
                           Matrix_Type::SparseAvailable_Type::lists[I][0]>::
        sum_squares(sum_of_squares, matrix);
  }
};

// when I_idx < M
template <typename T, typename Matrix_Type, std::size_t M, std::size_t N,
          std::size_t I_idx>
struct RealRow {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    RealColumn<T, Matrix_Type, I_idx, N - 1>::sum_squares(sum_of_squares,
                                                          matrix);
    RealRow<T, Matrix_Type, M, N, I_idx - 1>::sum_squares(sum_of_squares,
                                                          matrix);
  }
};

// row recursion termination
template <typename T, typename Matrix_Type, std::size_t M, std::size_t N>
struct RealRow<T, Matrix_Type, M, N, 0> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    RealColumn<T, Matrix_Type, 0, N - 1>::sum_squares(sum_of_squares, matrix);
  }
};

/* Complex value operation */

template <typename T, typename Matrix_Type, std::size_t I, std::size_t J,
          bool Value_Exists>
struct ComplexSumSquaresTemplate {};

template <typename T, typename Matrix_Type, std::size_t I, std::size_t J>
struct ComplexSumSquaresTemplate<T, Matrix_Type, I, J, false> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    static_cast<void>(sum_of_squares);
    static_cast<void>(matrix);
    // Do nothing.
  }
};

template <typename T, typename Matrix_Type, std::size_t I, std::size_t J>
struct ComplexSumSquaresTemplate<T, Matrix_Type, I, J, true> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    using Value_Complex_Type = typename Matrix_Type::Value_Complex_Type;

    Value_Complex_Type value = matrix.template get<I, J>();
    sum_of_squares += value.real * value.real + value.imag * value.imag;
  }
};

// when J_idx < N
template <typename T, typename Matrix_Type, std::size_t I, std::size_t J_idx>
struct ComplexColumn {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    ComplexSumSquaresTemplate<T, Matrix_Type, I, J_idx,
                              Matrix_Type::SparseAvailable_Type::lists
                                  [I][J_idx]>::sum_squares(sum_of_squares,
                                                           matrix);

    ComplexColumn<T, Matrix_Type, I, J_idx - 1>::sum_squares(sum_of_squares,
                                                             matrix);
  }
};

// column recursion termination
template <typename T, typename Matrix_Type, std::size_t I>
struct ComplexColumn<T, Matrix_Type, I, 0> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    ComplexSumSquaresTemplate<T, Matrix_Type, I, 0,
                              Matrix_Type::SparseAvailable_Type::lists[I][0]>::
        sum_squares(sum_of_squares, matrix);
  }
};

// when I_idx < M
template <typename T, typename Matrix_Type, std::size_t M, std::size_t N,
          std::size_t I_idx>
struct ComplexRow {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    ComplexColumn<T, Matrix_Type, I_idx, N - 1>::sum_squares(sum_of_squares,
                                                             matrix);
    ComplexRow<T, Matrix_Type, M, N, I_idx - 1>::sum_squares(sum_of_squares,
                                                             matrix);
  }
};

// row recursion termination
template <typename T, typename Matrix_Type, std::size_t M, std::size_t N>
struct ComplexRow<T, Matrix_Type, M, N, 0> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    ComplexColumn<T, Matrix_Type, 0, N - 1>::sum_squares(sum_of_squares,
                                                         matrix);
  }
};

template <typename Matrix_Type, typename isComplex> struct Normalizer {};

template <typename Matrix_Type>
struct Normalizer<Matrix_Type, std::false_type> {
  static auto norm(const Matrix_Type &matrix) ->
      typename Matrix_Type::Value_Type {

    using ValueType = typename Matrix_Type::Value_Type;

    ValueType sum_of_squares = static_cast<ValueType>(0);

    RealRow<ValueType, Matrix_Type, Matrix_Type::ROWS, Matrix_Type::COLS,
            (Matrix_Type::ROWS - 1)>::sum_squares(sum_of_squares, matrix);

    return PythonMath::sqrt(sum_of_squares);
  }
};

template <typename Matrix_Type> struct Normalizer<Matrix_Type, std::true_type> {
  static auto norm(const Matrix_Type &matrix) ->
      typename Matrix_Type::Value_Type {

    using ValueType = typename Matrix_Type::Value_Type;

    ValueType sum_of_squares = static_cast<ValueType>(0);

    ComplexRow<ValueType, Matrix_Type, Matrix_Type::ROWS, Matrix_Type::COLS,
               (Matrix_Type::ROWS - 1)>::sum_squares(sum_of_squares, matrix);

    return PythonMath::sqrt(sum_of_squares);
  }
};

} // namespace NormalizationOperation

template <typename Matrix_Type>
inline auto norm(const Matrix_Type &matrix) ->
    typename Matrix_Type::Value_Type {

  using IsComplexTrait =
      Base::Matrix::Is_Complex_Type<typename Matrix_Type::Value_Complex_Type>;

  using _Is_Complex_Type =
      typename std::conditional<IsComplexTrait::value, std::true_type,
                                std::false_type>::type;

  return NormalizationOperation::Normalizer<Matrix_Type, _Is_Complex_Type>()
      .norm(matrix);
}

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_BASE_SIMPLIFIED_ACTION_HPP__
