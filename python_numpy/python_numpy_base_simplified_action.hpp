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
#ifndef PYTHON_NUMPY_BASE_SIMPLIFIED_ACTION_HPP_
#define PYTHON_NUMPY_BASE_SIMPLIFIED_ACTION_HPP_

#include "python_math.hpp"
#include "python_numpy_base.hpp"
#include "python_numpy_base_simplification.hpp"
#include "python_numpy_base_substitution.hpp"
#include "python_numpy_complex.hpp"
#include "python_numpy_templates.hpp"

#include <initializer_list>
#include <tuple>
#include <type_traits>

namespace PythonNumpy {

/* Element wise multiply */

namespace ElementWiseMultiplyOperation {

template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t I, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Row;

template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t I, std::size_t Start,
          std::size_t End>
struct Row<Out_Type, In_A_Type, In_B_Type, M, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {
    Row<Out_Type, In_A_Type, In_B_Type, M, N, I, Start, Mid>::compute(Out, A,
                                                                      B);
    Row<Out_Type, In_A_Type, In_B_Type, M, N, I, Mid, End>::compute(Out, A, B);
  }
};

template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t I, std::size_t Start,
          std::size_t End>
struct Row<Out_Type, In_A_Type, In_B_Type, M, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {
    static_cast<void>(Out);
    static_cast<void>(A);
    static_cast<void>(B);
  }
};

template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t I, std::size_t Start,
          std::size_t End>
struct Row<Out_Type, In_A_Type, In_B_Type, M, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {
    Out.template set<I, Start>(A.template get<I, Start>() *
                               B.template get<I, Start>());
  }
};

template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Column;

template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t Start, std::size_t End>
struct Column<Out_Type, In_A_Type, In_B_Type, M, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {
    Column<Out_Type, In_A_Type, In_B_Type, M, N, Start, Mid>::compute(Out, A,
                                                                      B);
    Column<Out_Type, In_A_Type, In_B_Type, M, N, Mid, End>::compute(Out, A, B);
  }
};

template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t Start, std::size_t End>
struct Column<Out_Type, In_A_Type, In_B_Type, M, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {
    static_cast<void>(Out);
    static_cast<void>(A);
    static_cast<void>(B);
  }
};

template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t Start, std::size_t End>
struct Column<Out_Type, In_A_Type, In_B_Type, M, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {
    Row<Out_Type, In_A_Type, In_B_Type, M, N, Start, 0, N>::compute(Out, A, B);
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

  Column<Out_Type, In_A_Type, In_B_Type, M, N, 0, M>::compute(Out, A, B);
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
          std::size_t N, std::size_t I, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Row;

template <typename T, typename In_A_Type, typename In_B_Type, std::size_t M,
          std::size_t N, std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, In_A_Type, In_B_Type, M, N, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(T &result, const In_A_Type &A, const In_B_Type &B) {
    Row<T, In_A_Type, In_B_Type, M, N, I, Start, Mid>::compute(result, A, B);
    Row<T, In_A_Type, In_B_Type, M, N, I, Mid, End>::compute(result, A, B);
  }
};

template <typename T, typename In_A_Type, typename In_B_Type, std::size_t M,
          std::size_t N, std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, In_A_Type, In_B_Type, M, N, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(T &result, const In_A_Type &A, const In_B_Type &B) {
    static_cast<void>(result);
    static_cast<void>(A);
    static_cast<void>(B);
  }
};

template <typename T, typename In_A_Type, typename In_B_Type, std::size_t M,
          std::size_t N, std::size_t I, std::size_t Start, std::size_t End>
struct Row<T, In_A_Type, In_B_Type, M, N, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(T &result, const In_A_Type &A, const In_B_Type &B) {
    result += A.template get<I, Start>() * B.template get<I, Start>();
  }
};

template <typename T, typename In_A_Type, typename In_B_Type, std::size_t M,
          std::size_t N, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Column;

template <typename T, typename In_A_Type, typename In_B_Type, std::size_t M,
          std::size_t N, std::size_t Start, std::size_t End>
struct Column<T, In_A_Type, In_B_Type, M, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(T &result, const In_A_Type &A, const In_B_Type &B) {
    Column<T, In_A_Type, In_B_Type, M, N, Start, Mid>::compute(result, A, B);
    Column<T, In_A_Type, In_B_Type, M, N, Mid, End>::compute(result, A, B);
  }
};

template <typename T, typename In_A_Type, typename In_B_Type, std::size_t M,
          std::size_t N, std::size_t Start, std::size_t End>
struct Column<T, In_A_Type, In_B_Type, M, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(T &result, const In_A_Type &A, const In_B_Type &B) {
    static_cast<void>(result);
    static_cast<void>(A);
    static_cast<void>(B);
  }
};

template <typename T, typename In_A_Type, typename In_B_Type, std::size_t M,
          std::size_t N, std::size_t Start, std::size_t End>
struct Column<T, In_A_Type, In_B_Type, M, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(T &result, const In_A_Type &A, const In_B_Type &B) {
    Row<T, In_A_Type, In_B_Type, M, N, Start, 0, N>::compute(result, A, B);
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

  Column<T, In_A_Type, In_B_Type, M, N, 0, M>::compute(result, A, B);

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
          std::size_t Start, std::size_t End, typename Enable = void>
struct GetRow_Loop {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(const DenseMatrix_Type<T, M, N> &matrix,
                      DenseMatrix_Type<T, M, 1> &result) {

    GetRow_Loop<T, M, N, COL, Start, Mid>::compute(matrix, result);
    GetRow_Loop<T, M, N, COL, Mid, End>::compute(matrix, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t COL,
          std::size_t Start, std::size_t End>
struct GetRow_Loop<T, M, N, COL, Start, End,
                   typename std::enable_if<(End == Start)>::type> {
  static void compute(const DenseMatrix_Type<T, M, N> &matrix,
                      DenseMatrix_Type<T, M, 1> &result) {
    static_cast<void>(matrix);
    static_cast<void>(result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t COL,
          std::size_t Start, std::size_t End>
struct GetRow_Loop<T, M, N, COL, Start, End,
                   typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DenseMatrix_Type<T, M, N> &matrix,
                      DenseMatrix_Type<T, M, 1> &result) {
    result.template set<Start, 0>(matrix.template get<Start, COL>());
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t COL>
using GetRow = GetRow_Loop<T, M, N, COL, 0, M>;

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

template <typename T, std::size_t M, std::size_t COL, std::size_t Start,
          std::size_t End, typename Enable = void>
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
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(const DiagMatrix_Type<T, M> &matrix,
                      DiagAvailableRow_Type<T, M, COL> &result) {

    GetRow_Loop<T, M, COL, Start, Mid>::compute(matrix, result);
    GetRow_Loop<T, M, COL, Mid, End>::compute(matrix, result);
  }
};

template <typename T, std::size_t M, std::size_t COL, std::size_t Start,
          std::size_t End>
struct GetRow_Loop<T, M, COL, Start, End,
                   typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix_Type<T, M> &matrix,
                      DiagAvailableRow_Type<T, M, COL> &result) {
    static_cast<void>(matrix);
    static_cast<void>(result);
  }
};

template <typename T, std::size_t M, std::size_t COL, std::size_t Start,
          std::size_t End>
struct GetRow_Loop<T, M, COL, Start, End,
                   typename std::enable_if<(End - Start == 1)>::type> {
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
    result.template set<Start, 0>(matrix.template get<Start, COL>());
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
using GetRow = GetRow_Loop<T, M, COL, 0, M>;

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
          std::size_t COL, std::size_t Start, std::size_t End,
          typename Enable = void>
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
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                      SparseCol_Type<T, M, COL, SparseAvailable> &result) {

    GetRow_Loop<T, M, N, SparseAvailable, COL, Start, Mid>::compute(matrix,
                                                                    result);
    GetRow_Loop<T, M, N, SparseAvailable, COL, Mid, End>::compute(matrix,
                                                                  result);
  }
};

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t COL, std::size_t Start, std::size_t End>
struct GetRow_Loop<T, M, N, SparseAvailable, COL, Start, End,
                   typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                      SparseCol_Type<T, M, COL, SparseAvailable> &result) {
    static_cast<void>(matrix);
    static_cast<void>(result);
  }
};

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t COL, std::size_t Start, std::size_t End>
struct GetRow_Loop<T, M, N, SparseAvailable, COL, Start, End,
                   typename std::enable_if<(End - Start == 1)>::type> {
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
    result.template set<Start, 0>(matrix.template get<Start, COL>());
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
using GetRow = GetRow_Loop<T, M, N, SparseAvailable, COL, 0, M>;

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
          std::size_t Start, std::size_t End, typename Enable = void>
struct SetRow_Loop {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void compute(Matrix_Type &matrix, const RowVector_Type &row_vector) {
    SetRow_Loop<Matrix_Type, RowVector_Type, COL, Start, Mid>::compute(
        matrix, row_vector);
    SetRow_Loop<Matrix_Type, RowVector_Type, COL, Mid, End>::compute(
        matrix, row_vector);
  }
};

template <typename Matrix_Type, typename RowVector_Type, std::size_t COL,
          std::size_t Start, std::size_t End>
struct SetRow_Loop<Matrix_Type, RowVector_Type, COL, Start, End,
                   typename std::enable_if<(End == Start)>::type> {
  static void compute(Matrix_Type &matrix, const RowVector_Type &row_vector) {
    static_cast<void>(matrix);
    static_cast<void>(row_vector);
  }
};

template <typename Matrix_Type, typename RowVector_Type, std::size_t COL,
          std::size_t Start, std::size_t End>
struct SetRow_Loop<Matrix_Type, RowVector_Type, COL, Start, End,
                   typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(Matrix_Type &matrix, const RowVector_Type &row_vector) {
    matrix.template set<Start, COL>(row_vector.template get<Start, 0>());
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
    SetRow_Loop<Matrix_Type, RowVector_Type, COL, 0, Matrix_Type::ROWS>;

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

/* Concatenate block, any size */
namespace ConcatenateBlockOperation {

/**
 * @brief Concatenates two matrices horizontally.
 *
 * This type alias defines the type of the resulting matrix when two matrices
 * are concatenated horizontally. The resulting matrix has the same number of
 * rows as the input matrices and a number of columns equal to the sum of the
 * columns of the input matrices.
 *
 * @tparam Left_Type The type of the left matrix to concatenate.
 * @tparam Right_Type The type of the right matrix to concatenate.
 */
template <std::size_t Row_Offset, std::size_t N, typename ArgsTuple_Type,
          std::size_t Start, std::size_t End, typename Enable = void>
struct ConcatenateBlockColumnsRange;

template <std::size_t Row_Offset, std::size_t N, typename ArgsTuple_Type,
          std::size_t Start, std::size_t End>
struct ConcatenateBlockColumnsRange<
    Row_Offset, N, ArgsTuple_Type, Start, End,
    typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  using type = ConcatenateHorizontally_Type<
      typename ConcatenateBlockColumnsRange<Row_Offset, N, ArgsTuple_Type,
                                            Start, Mid>::type,
      typename ConcatenateBlockColumnsRange<Row_Offset, N, ArgsTuple_Type, Mid,
                                            End>::type>;
};

template <std::size_t Row_Offset, std::size_t N, typename ArgsTuple_Type,
          std::size_t Start, std::size_t End>
struct ConcatenateBlockColumnsRange<
    Row_Offset, N, ArgsTuple_Type, Start, End,
    typename std::enable_if<(End - Start == 1)>::type> {
  using type =
      typename std::tuple_element<Row_Offset + Start, ArgsTuple_Type>::type;
};

template <std::size_t Row_Offset, std::size_t N, typename ArgsTuple_Type,
          std::size_t Col_Index>
struct ConcatenateBlockColumns {
  using type =
      typename ConcatenateBlockColumnsRange<Row_Offset, N, ArgsTuple_Type, 0,
                                            (Col_Index + 1)>::type;
};

/**
 * @brief Concatenates two matrices vertically.
 *
 * This type alias defines the type of the resulting matrix when two matrices
 * are concatenated vertically. The resulting matrix has the same number of
 * columns as the input matrices and a number of rows equal to the sum of the
 * rows of the input matrices.
 *
 * @tparam Top_Type The type of the top matrix to concatenate.
 * @tparam Bottom_Type The type of the bottom matrix to concatenate.
 */
template <std::size_t N, typename ArgsTuple_Type, std::size_t Row_Index>
struct ConcatenateBlockRows {
  template <std::size_t Start, std::size_t End, typename Enable = void>
  struct Range;

  template <std::size_t Start, std::size_t End>
  struct Range<Start, End, typename std::enable_if<(End - Start > 1)>::type> {
    static constexpr std::size_t Mid = Start + (End - Start) / 2;
    using type = ConcatenateVertically_Type<typename Range<Start, Mid>::type,
                                            typename Range<Mid, End>::type>;
  };

  template <std::size_t Start, std::size_t End>
  struct Range<Start, End, typename std::enable_if<(End - Start == 1)>::type> {
    using type =
        typename ConcatenateBlockColumns<(Start * N), N, ArgsTuple_Type,
                                         (N - 1)>::type;
  };

  using type = typename Range<0, (Row_Index + 1)>::type;
};

/**
 * @brief Concatenates a block of matrices based on the provided arguments.
 *
 * This type alias defines the type of the resulting matrix when a block of
 * matrices is concatenated based on the specified M, N, and ArgsTuple_Type.
 * The resulting matrix has M rows and N columns, where each element is
 * determined by the corresponding argument in the tuple.
 *
 * @tparam M The number of rows in the block.
 * @tparam N The number of columns in the block.
 * @tparam ArgsTuple_Type The type of the tuple containing arguments for
 * concatenation.
 */
template <std::size_t M, std::size_t N, typename ArgsTuple_Type>
struct ConcatenateBlock {

  using type = typename ConcatenateBlockRows<N, ArgsTuple_Type, (M - 1)>::type;
};

/**
 * @brief Concatenates a block of matrices based on the provided arguments.
 *
 * This type alias defines the type of the resulting matrix when a block of
 * matrices is concatenated based on the specified M, N, and ArgsTuple_Type.
 * The resulting matrix has M rows and N columns, where each element is
 * determined by the corresponding argument in the tuple.
 *
 * @tparam M The number of rows in the block.
 * @tparam N The number of columns in the block.
 * @tparam ArgsTuple_Type The type of the tuple containing arguments for
 * concatenation.
 */
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

/**
 * @brief Generates tile types for a block of matrices.
 *
 * This struct recursively generates a type that represents a block of
 * matrices (tiles) based on the specified dimensions M and N, and the
 * matrix type. It uses the ConcatenateBlock_Type to create the appropriate
 * tile type.
 *
 * @tparam M The number of rows in the tile.
 * @tparam N The number of columns in the tile.
 * @tparam Count The number of matrices to concatenate (used in recursion).
 * @tparam MATRIX_Type The type of the matrix used in the tile.
 * @tparam Args The types of the arguments to include in the tuple.
 */
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

template <typename To_Type, typename From_Type, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Row;

template <typename To_Type, typename From_Type, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<To_Type, From_Type, I, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {
    Row<To_Type, From_Type, I, Start, Mid>::substitute(to_matrix, from_matrix);
    Row<To_Type, From_Type, I, Mid, End>::substitute(to_matrix, from_matrix);
  }
};

template <typename To_Type, typename From_Type, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<To_Type, From_Type, I, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {
    static_cast<void>(to_matrix);
    static_cast<void>(from_matrix);
  }
};

template <typename To_Type, typename From_Type, std::size_t I,
          std::size_t Start, std::size_t End>
struct Row<To_Type, From_Type, I, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {
    constexpr std::size_t NUMBER_OF_ELEMENT = I * From_Type::COLS + Start;

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

template <typename To_Type, typename From_Type, std::size_t M, std::size_t N,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Column;

template <typename To_Type, typename From_Type, std::size_t M, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<To_Type, From_Type, M, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {
    Column<To_Type, From_Type, M, N, Start, Mid>::substitute(to_matrix,
                                                             from_matrix);
    Column<To_Type, From_Type, M, N, Mid, End>::substitute(to_matrix,
                                                           from_matrix);
  }
};

template <typename To_Type, typename From_Type, std::size_t M, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<To_Type, From_Type, M, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {
    static_cast<void>(to_matrix);
    static_cast<void>(from_matrix);
  }
};

template <typename To_Type, typename From_Type, std::size_t M, std::size_t N,
          std::size_t Start, std::size_t End>
struct Column<To_Type, From_Type, M, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {
    Row<To_Type, From_Type, Start, 0, N>::substitute(to_matrix, from_matrix);
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
  Column<To_Type, From_Type, From_Type::ROWS, From_Type::COLS, 0,
         From_Type::ROWS>::substitute(to_matrix, from_matrix);
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
  /**
   * @brief Adds the square of a non-existent value to the sum of squares.
   *
   * This static function is a specialization for the case when the value at
   * position (I, J) does not exist in the matrix. In this case, it does not
   * contribute to the sum of squares, and the function simply returns without
   * modifying the sum_of_squares variable.
   *
   * @tparam T The type of the sum of squares variable.
   * @tparam Matrix_Type The type of the matrix being processed.
   * @tparam I The row index being processed.
   * @tparam J The column index being processed.
   */
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    static_cast<void>(sum_of_squares);
    static_cast<void>(matrix);
    // Do nothing.
  }
};

template <typename T, typename Matrix_Type, std::size_t I, std::size_t J>
struct RealSumSquaresTemplate<T, Matrix_Type, I, J, true> {
  /**
   * @brief Adds the square of an existing value to the sum of squares.
   *
   * This static function is a specialization for the case when the value at
   * position (I, J) exists in the matrix. It retrieves the value, computes its
   * square, and adds it to the sum_of_squares variable.
   *
   * @tparam T The type of the sum of squares variable.
   * @tparam Matrix_Type The type of the matrix being processed.
   * @tparam I The row index being processed.
   * @tparam J The column index being processed.
   */
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    T value = matrix.template get<I, J>();
    sum_of_squares += value * value;
  }
};

template <typename T, typename Matrix_Type, std::size_t I, std::size_t Start,
          std::size_t End, typename Enable = void>
struct RealColumn;

template <typename T, typename Matrix_Type, std::size_t I, std::size_t Start,
          std::size_t End>
struct RealColumn<T, Matrix_Type, I, Start, End,
                  typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {
    RealColumn<T, Matrix_Type, I, Start, Mid>::sum_squares(sum_of_squares,
                                                           matrix);
    RealColumn<T, Matrix_Type, I, Mid, End>::sum_squares(sum_of_squares,
                                                         matrix);
  }
};

template <typename T, typename Matrix_Type, std::size_t I, std::size_t Start,
          std::size_t End>
struct RealColumn<T, Matrix_Type, I, Start, End,
                  typename std::enable_if<(End == Start)>::type> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {
    static_cast<void>(sum_of_squares);
    static_cast<void>(matrix);
  }
};

template <typename T, typename Matrix_Type, std::size_t I, std::size_t Start,
          std::size_t End>
struct RealColumn<T, Matrix_Type, I, Start, End,
                  typename std::enable_if<(End - Start == 1)>::type> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {
    RealSumSquaresTemplate<T, Matrix_Type, I, Start,
                           Matrix_Type::SparseAvailable_Type::lists[I][Start]>::
        sum_squares(sum_of_squares, matrix);
  }
};

template <typename T, typename Matrix_Type, std::size_t M, std::size_t N,
          std::size_t Start, std::size_t End, typename Enable = void>
struct RealRow;

template <typename T, typename Matrix_Type, std::size_t M, std::size_t N,
          std::size_t Start, std::size_t End>
struct RealRow<T, Matrix_Type, M, N, Start, End,
               typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {
    RealRow<T, Matrix_Type, M, N, Start, Mid>::sum_squares(sum_of_squares,
                                                           matrix);
    RealRow<T, Matrix_Type, M, N, Mid, End>::sum_squares(sum_of_squares,
                                                         matrix);
  }
};

template <typename T, typename Matrix_Type, std::size_t M, std::size_t N,
          std::size_t Start, std::size_t End>
struct RealRow<T, Matrix_Type, M, N, Start, End,
               typename std::enable_if<(End == Start)>::type> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {
    static_cast<void>(sum_of_squares);
    static_cast<void>(matrix);
  }
};

template <typename T, typename Matrix_Type, std::size_t M, std::size_t N,
          std::size_t Start, std::size_t End>
struct RealRow<T, Matrix_Type, M, N, Start, End,
               typename std::enable_if<(End - Start == 1)>::type> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {
    RealColumn<T, Matrix_Type, Start, 0, N>::sum_squares(sum_of_squares,
                                                         matrix);
  }
};

/* Complex value operation */

template <typename T, typename Matrix_Type, std::size_t I, std::size_t J,
          bool Value_Exists>
struct ComplexSumSquaresTemplate {};

template <typename T, typename Matrix_Type, std::size_t I, std::size_t J>
struct ComplexSumSquaresTemplate<T, Matrix_Type, I, J, false> {
  /**
   * @brief Adds the square of a non-existent complex value to the sum of
   * squares.
   *
   * This static function is a specialization for the case when the complex
   * value at position (I, J) does not exist in the matrix. In this case, it
   * does not contribute to the sum of squares, and the function simply returns
   * without modifying the sum_of_squares variable.
   *
   * @tparam T The type of the sum of squares variable.
   * @tparam Matrix_Type The type of the matrix being processed.
   * @tparam I The row index being processed.
   * @tparam J The column index being processed.
   */
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    static_cast<void>(sum_of_squares);
    static_cast<void>(matrix);
    // Do nothing.
  }
};

template <typename T, typename Matrix_Type, std::size_t I, std::size_t J>
struct ComplexSumSquaresTemplate<T, Matrix_Type, I, J, true> {
  /**
   * @brief Adds the square of an existing complex value to the sum of squares.
   *
   * This static function is a specialization for the case when the complex
   * value at position (I, J) exists in the matrix. It retrieves the complex
   * value, computes the sum of squares of its real and imaginary parts, and
   * adds it to the sum_of_squares variable.
   *
   * @tparam T The type of the sum of squares variable.
   * @tparam Matrix_Type The type of the matrix being processed.
   * @tparam I The row index being processed.
   * @tparam J The column index being processed.
   */
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {

    using Value_Complex_Type = typename Matrix_Type::Value_Complex_Type;

    Value_Complex_Type value = matrix.template get<I, J>();
    sum_of_squares += value.real * value.real + value.imag * value.imag;
  }
};

template <typename T, typename Matrix_Type, std::size_t I, std::size_t Start,
          std::size_t End, typename Enable = void>
struct ComplexColumn;

template <typename T, typename Matrix_Type, std::size_t I, std::size_t Start,
          std::size_t End>
struct ComplexColumn<T, Matrix_Type, I, Start, End,
                     typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {
    ComplexColumn<T, Matrix_Type, I, Start, Mid>::sum_squares(sum_of_squares,
                                                              matrix);
    ComplexColumn<T, Matrix_Type, I, Mid, End>::sum_squares(sum_of_squares,
                                                            matrix);
  }
};

template <typename T, typename Matrix_Type, std::size_t I, std::size_t Start,
          std::size_t End>
struct ComplexColumn<T, Matrix_Type, I, Start, End,
                     typename std::enable_if<(End == Start)>::type> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {
    static_cast<void>(sum_of_squares);
    static_cast<void>(matrix);
  }
};

template <typename T, typename Matrix_Type, std::size_t I, std::size_t Start,
          std::size_t End>
struct ComplexColumn<T, Matrix_Type, I, Start, End,
                     typename std::enable_if<(End - Start == 1)>::type> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {
    ComplexSumSquaresTemplate<T, Matrix_Type, I, Start,
                              Matrix_Type::SparseAvailable_Type::lists
                                  [I][Start]>::sum_squares(sum_of_squares,
                                                           matrix);
  }
};

template <typename T, typename Matrix_Type, std::size_t M, std::size_t N,
          std::size_t Start, std::size_t End, typename Enable = void>
struct ComplexRow;

template <typename T, typename Matrix_Type, std::size_t M, std::size_t N,
          std::size_t Start, std::size_t End>
struct ComplexRow<T, Matrix_Type, M, N, Start, End,
                  typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {
    ComplexRow<T, Matrix_Type, M, N, Start, Mid>::sum_squares(sum_of_squares,
                                                              matrix);
    ComplexRow<T, Matrix_Type, M, N, Mid, End>::sum_squares(sum_of_squares,
                                                            matrix);
  }
};

template <typename T, typename Matrix_Type, std::size_t M, std::size_t N,
          std::size_t Start, std::size_t End>
struct ComplexRow<T, Matrix_Type, M, N, Start, End,
                  typename std::enable_if<(End == Start)>::type> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {
    static_cast<void>(sum_of_squares);
    static_cast<void>(matrix);
  }
};

template <typename T, typename Matrix_Type, std::size_t M, std::size_t N,
          std::size_t Start, std::size_t End>
struct ComplexRow<T, Matrix_Type, M, N, Start, End,
                  typename std::enable_if<(End - Start == 1)>::type> {
  static void sum_squares(T &sum_of_squares, const Matrix_Type &matrix) {
    ComplexColumn<T, Matrix_Type, Start, 0, N>::sum_squares(sum_of_squares,
                                                            matrix);
  }
};

template <typename Matrix_Type, typename isComplex> struct Normalizer {};

template <typename Matrix_Type>
struct Normalizer<Matrix_Type, std::false_type> {
  /**
   * @brief Computes the norm of a matrix with real values.
   *
   * This static function computes the norm of a matrix by calculating the sum
   * of squares of its real values and then taking the square root of that sum.
   *
   * @tparam Matrix_Type The type of the matrix for which to compute the norm.
   * @param matrix The matrix for which to compute the norm.
   * @return typename Matrix_Type::Value_Type The computed norm of the matrix.
   */
  static auto norm(const Matrix_Type &matrix) ->
      typename Matrix_Type::Value_Type {

    using ValueType = typename Matrix_Type::Value_Type;

    ValueType sum_of_squares = static_cast<ValueType>(0);

    RealRow<ValueType, Matrix_Type, Matrix_Type::ROWS, Matrix_Type::COLS, 0,
            Matrix_Type::ROWS>::sum_squares(sum_of_squares, matrix);

    return PythonMath::sqrt(sum_of_squares);
  }
};

template <typename Matrix_Type> struct Normalizer<Matrix_Type, std::true_type> {
  /**
   * @brief Computes the norm of a matrix with complex values.
   *
   * This static function computes the norm of a matrix by calculating the sum
   * of squares of its complex values (considering both real and imaginary
   * parts) and then taking the square root of that sum.
   *
   * @tparam Matrix_Type The type of the matrix for which to compute the norm.
   * @param matrix The matrix for which to compute the norm.
   * @return typename Matrix_Type::Value_Type The computed norm of the matrix.
   */
  static auto norm(const Matrix_Type &matrix) ->
      typename Matrix_Type::Value_Type {

    using ValueType = typename Matrix_Type::Value_Type;

    ValueType sum_of_squares = static_cast<ValueType>(0);

    ComplexRow<ValueType, Matrix_Type, Matrix_Type::ROWS, Matrix_Type::COLS, 0,
               Matrix_Type::ROWS>::sum_squares(sum_of_squares, matrix);

    return PythonMath::sqrt(sum_of_squares);
  }
};

} // namespace NormalizationOperation

/**
 * @brief Computes the norm of a matrix.
 *
 * This function template computes the norm of a matrix by determining whether
 * the matrix contains complex values or not, and then invoking the appropriate
 * normalization operation to calculate the norm based on the type of values in
 * the matrix.
 *
 * @tparam Matrix_Type The type of the matrix for which to compute the norm.
 * @param matrix The matrix for which to compute the norm.
 * @return typename Matrix_Type::Value_Type The computed norm of the matrix.
 */
template <typename Matrix_Type>
inline auto norm(const Matrix_Type &matrix) ->
    typename Matrix_Type::Value_Type {

  using IsComplexTrait =
      Base::Matrix::Is_Complex_Type<typename Matrix_Type::Value_Complex_Type>;

  using Is_Complex_Type_ =
      typename std::conditional<IsComplexTrait::value, std::true_type,
                                std::false_type>::type;

  return NormalizationOperation::Normalizer<Matrix_Type, Is_Complex_Type_>()
      .norm(matrix);
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_BASE_SIMPLIFIED_ACTION_HPP_
