#ifndef __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__
#define __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__

#include "python_numpy_base.hpp"
#include "python_numpy_complex.hpp"
#include "python_numpy_templates.hpp"

#include "python_numpy_concatenate.hpp"

#include <initializer_list>
#include <tuple>
#include <type_traits>

namespace PythonNumpy {

template <typename T, std::size_t M, std::size_t N>
inline auto make_DenseMatrixZeros(void) -> Matrix<DefDense, T, M, N> {

  Matrix<DefDense, T, M, N> result;
  return result;
}

template <typename T, std::size_t M, std::size_t N>
inline auto make_DenseMatrixOnes(void) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>::ones();
}

template <std::size_t M, std::size_t N, typename T>
inline auto make_DenseMatrixFull(const T &value) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>::full(value);
}

template <typename T, std::size_t M>
inline auto make_DiagMatrixZeros(void) -> Matrix<DefDiag, T, M> {

  Matrix<DefDiag, T, M> result;
  return result;
}

template <typename T, std::size_t M>
inline auto make_DiagMatrixIdentity(void) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>::identity();
}

template <std::size_t M, typename T>
inline auto make_DiagMatrixFull(const T &value) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>::full(value);
}

template <typename T, std::size_t M, std::size_t N>
inline auto make_SparseMatrixEmpty(void)
    -> Matrix<DefSparse, T, M, N, SparseAvailableEmpty<M, N>> {

  return Matrix<DefSparse, T, M, N, SparseAvailableEmpty<M, N>>();
}

namespace MakeDenseMatrixOperation {

template <std::size_t ColumnCount, std::size_t RowCount,
          typename DenseMatrix_Type, typename T>
inline void assign_values(DenseMatrix_Type &matrix, T value_1) {

  static_assert(ColumnCount < DenseMatrix_Type::COLS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");
  static_assert(RowCount < DenseMatrix_Type::ROWS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");

  matrix.template set<ColumnCount, RowCount>(value_1);
}

template <std::size_t ColumnCount, std::size_t RowCount,
          typename DenseMatrix_Type, typename T, typename U, typename... Args>
inline void assign_values(DenseMatrix_Type &matrix, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(ColumnCount < DenseMatrix_Type::COLS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");
  static_assert(RowCount < DenseMatrix_Type::ROWS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");

  matrix.template set<ColumnCount, RowCount>(value_1);

  assign_values<ColumnCount + 1 * (RowCount == (DenseMatrix_Type::ROWS - 1)),
                ((RowCount + 1) * (RowCount != (DenseMatrix_Type::ROWS - 1)))>(
      matrix, value_2, args...);
}

} // namespace MakeDenseMatrixOperation

template <std::size_t M, std::size_t N, typename T, typename... Args>
inline auto make_DenseMatrix(T value_1, Args... args)
    -> Matrix<DefDense, T, M, N> {

  Matrix<DefDense, T, M, N> result;

  MakeDenseMatrixOperation::assign_values<0, 0>(result, value_1, args...);

  return result;
}

namespace MakeDiagMatrixOperation {

template <std::size_t IndexCount, typename DiagMatrix_Type, typename T>
inline void assign_values(DiagMatrix_Type &matrix, T value_1) {

  static_assert(IndexCount < DiagMatrix_Type::COLS,
                "Number of arguments must be less than the number of columns.");

  matrix.template set<IndexCount, IndexCount>(value_1);
}

template <std::size_t IndexCount, typename DiagMatrix_Type, typename T,
          typename U, typename... Args>
inline void assign_values(DiagMatrix_Type &matrix, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(IndexCount < DiagMatrix_Type::COLS,
                "Number of arguments must be less than the number of columns.");

  matrix.template set<IndexCount, IndexCount>(value_1);

  assign_values<IndexCount + 1>(matrix, value_2, args...);
}

} // namespace MakeDiagMatrixOperation

template <std::size_t M, typename T, typename... Args>
inline auto make_DiagMatrix(T value_1, Args... args) -> Matrix<DefDiag, T, M> {

  Matrix<DefDiag, T, M> result;

  MakeDiagMatrixOperation::assign_values<0>(result, value_1, args...);

  return result;
}

template <typename T, typename SparseAvailable>
inline auto make_SparseMatrixZeros(void)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  Matrix<DefSparse, T, SparseAvailable::number_of_columns,
         SparseAvailable::column_size, SparseAvailable>
      result;

  return result;
}

template <typename T, typename SparseAvailable>
inline auto make_SparseMatrixOnes(void)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  return Matrix<DefSparse, T, SparseAvailable::number_of_columns,
                SparseAvailable::column_size,
                SparseAvailable>::full(static_cast<T>(1));
}

template <typename SparseAvailable, typename T>
inline auto make_SparseMatrixFull(const T &value)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  return Matrix<DefSparse, T, SparseAvailable::number_of_columns,
                SparseAvailable::column_size, SparseAvailable>::full(value);
}

namespace MakeSparseMatrixOperation {

template <std::size_t IndexCount, typename SparseMatrix_Type, typename T>
inline void assign_values(SparseMatrix_Type &matrix, T value_1) {

  static_assert(IndexCount < SparseMatrix_Type::NumberOfValues,
                "Number of arguments must be the same or less than the number "
                "of elements of Sparse Matrix.");

  matrix.template set<IndexCount>(value_1);
}

template <std::size_t IndexCount, typename SparseMatrix_Type, typename T,
          typename U, typename... Args>
inline void assign_values(SparseMatrix_Type &matrix, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");

  static_assert(IndexCount < SparseMatrix_Type::NumberOfValues,
                "Number of arguments must be the same or less than the number "
                "of elements of Sparse Matrix.");

  matrix.template set<IndexCount>(value_1);

  assign_values<IndexCount + 1>(matrix, value_2, args...);
}

} // namespace MakeSparseMatrixOperation

template <typename SparseAvailable, typename T, typename... Args>
inline auto make_SparseMatrix(T value_1, Args... args)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  Matrix<DefSparse, T, SparseAvailable::number_of_columns,
         SparseAvailable::column_size, SparseAvailable>
      result;

  MakeSparseMatrixOperation::assign_values<0>(result, value_1, args...);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename... Args>
inline auto
make_SparseMatrixFromDenseMatrix(Matrix<DefDense, T, M, N> &dense_matrix)
    -> Matrix<DefSparse, T, M, N, DenseAvailable<M, N>> {

  return Matrix<DefSparse, T, M, N, DenseAvailable<M, N>>(
      create_compiled_sparse(dense_matrix.matrix));
}

/* Type */
template <typename T, std::size_t M, std::size_t N>
using DenseMatrix_Type = Matrix<DefDense, T, M, N>;

template <typename T, std::size_t M>
using DiagMatrix_Type = Matrix<DefDiag, T, M>;

template <typename T, typename SparseAvailable>
using SparseMatrix_Type =
    decltype(make_SparseMatrixZeros<T, SparseAvailable>());

template <typename T, std::size_t M, std::size_t N>
using SparseMatrixEmpty_Type = decltype(make_SparseMatrixEmpty<T, M, N>());

/* Get */
namespace GetDenseMatrixOperation {

template <typename T, std::size_t M, std::size_t N, std::size_t ROW,
          std::size_t COL_Index>
struct GetRow_Loop {
  static void compute(const DenseMatrix_Type<T, M, N> &matrix,
                      DenseMatrix_Type<T, M, 1> &result) {

    result.template set<COL_Index, 0>(matrix.template get<COL_Index, ROW>());
    GetRow_Loop<T, M, N, ROW, COL_Index - 1>::compute(matrix, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t ROW>
struct GetRow_Loop<T, M, N, ROW, 0> {
  static void compute(const DenseMatrix_Type<T, M, N> &matrix,
                      DenseMatrix_Type<T, M, 1> &result) {

    result.template set<0, 0>(matrix.template get<0, ROW>());
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t ROW>
using GetRow = GetRow_Loop<T, M, N, ROW, M - 1>;

} // namespace GetDenseMatrixOperation

template <std::size_t ROW, typename T, std::size_t M, std::size_t N>
inline auto get_row(const DenseMatrix_Type<T, M, N> &matrix)
    -> DenseMatrix_Type<T, M, 1> {

  DenseMatrix_Type<T, M, 1> result;

  GetDenseMatrixOperation::GetRow<T, M, N, ROW>::compute(matrix, result);

  return result;
}

namespace GetDiagMatrixOperation {

template <std::size_t M, std::size_t Index>
using DiagAvailableRow = SparseAvailableGetRow<M, DiagAvailable<M>, Index>;

template <typename T, std::size_t M, std::size_t Index>
using DiagAvailableRow_Type = SparseMatrix_Type<T, DiagAvailableRow<M, Index>>;

template <typename T, std::size_t M, std::size_t ROW, std::size_t COL_Index>
struct GetRow_Loop {
  static void compute(const DiagMatrix_Type<T, M> &matrix,
                      DiagAvailableRow_Type<T, M, ROW> &result) {

    result.template set<COL_Index, 0>(matrix.template get<COL_Index, ROW>());
    GetRow_Loop<T, M, ROW, COL_Index - 1>::compute(matrix, result);
  }
};

template <typename T, std::size_t M, std::size_t ROW>
struct GetRow_Loop<T, M, ROW, 0> {
  static void compute(const DiagMatrix_Type<T, M> &matrix,
                      DiagAvailableRow_Type<T, M, ROW> &result) {

    result.template set<0, 0>(matrix.template get<0, ROW>());
  }
};

template <typename T, std::size_t M, std::size_t ROW>
using GetRow = GetRow_Loop<T, M, ROW, M - 1>;

} // namespace GetDiagMatrixOperation

template <std::size_t ROW, typename T, std::size_t M>
inline auto get_row(const DiagMatrix_Type<T, M> &matrix)
    -> GetDiagMatrixOperation::DiagAvailableRow_Type<T, M, ROW> {

  GetDiagMatrixOperation::DiagAvailableRow_Type<T, M, ROW> result;

  GetDiagMatrixOperation::GetRow<T, M, ROW>::compute(matrix, result);

  return result;
}

namespace GetSparseMatrixOperation {

template <std::size_t M, std::size_t Index, typename SparseAvailable>
using SparseAvailableRow = SparseAvailableGetRow<M, SparseAvailable, Index>;

template <typename T, std::size_t M, std::size_t Index,
          typename SparseAvailable>
using SparseRow_Type =
    SparseMatrix_Type<T, SparseAvailableRow<M, Index, SparseAvailable>>;

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t ROW, std::size_t COL_Index>
struct GetRow_Loop {
  static void compute(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                      SparseRow_Type<T, M, ROW, SparseAvailable> &result) {

    result.template set<COL_Index, 0>(matrix.template get<COL_Index, ROW>());
    GetRow_Loop<T, M, N, SparseAvailable, ROW, COL_Index - 1>::compute(matrix,
                                                                       result);
  }
};

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t ROW>
struct GetRow_Loop<T, M, N, SparseAvailable, ROW, 0> {
  static void compute(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                      SparseRow_Type<T, M, ROW, SparseAvailable> &result) {

    result.template set<0, 0>(matrix.template get<0, ROW>());
  }
};

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t ROW>
using GetRow = GetRow_Loop<T, M, N, SparseAvailable, ROW, M - 1>;

} // namespace GetSparseMatrixOperation

template <std::size_t ROW, typename T, std::size_t M, std::size_t N,
          typename SparseAvailable>
inline auto get_row(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> GetSparseMatrixOperation::SparseRow_Type<T, M, ROW, SparseAvailable> {

  GetSparseMatrixOperation::SparseRow_Type<T, M, ROW, SparseAvailable> result;

  GetSparseMatrixOperation::GetRow<T, M, N, SparseAvailable, ROW>::compute(
      matrix, result);

  return result;
}

/* Set */
namespace SetMatrixOperation {

template <typename Matrix_Type, typename RowVector_Type, std::size_t ROW,
          std::size_t COL_Index>
struct SetRow_Loop {
  static void compute(Matrix_Type &matrix, const RowVector_Type &row_vector) {

    matrix.template set<COL_Index, ROW>(
        row_vector.template get<COL_Index, 0>());
    SetRow_Loop<Matrix_Type, RowVector_Type, ROW, COL_Index - 1>::compute(
        matrix, row_vector);
  }
};

template <typename Matrix_Type, typename RowVector_Type, std::size_t ROW>
struct SetRow_Loop<Matrix_Type, RowVector_Type, ROW, 0> {
  static void compute(Matrix_Type &matrix, const RowVector_Type &row_vector) {

    matrix.template set<0, ROW>(row_vector.template get<0, 0>());
  }
};

template <typename Matrix_Type, typename RowVector_Type, std::size_t ROW>
using SetRow =
    SetRow_Loop<Matrix_Type, RowVector_Type, ROW, (Matrix_Type::COLS - 1)>;

} // namespace SetMatrixOperation

template <std::size_t ROW, typename Matrix_Type, typename RowVector_Type>
inline void set_row(Matrix_Type &matrix, const RowVector_Type &row_vector) {

  SetMatrixOperation::SetRow<Matrix_Type, RowVector_Type, ROW>::compute(
      matrix, row_vector);
}

/* Part matrix substitute */
namespace PartMatrixOperation {

// when J_idx < N
template <std::size_t Col_Offset, std::size_t Row_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct SubstituteColumn {
  static void compute(All_Type &All, const Part_Type &Part) {

    All.template set<(Col_Offset + I), (Row_Offset + J_idx)>(
        Part.template get<I, J_idx>());

    SubstituteColumn<Col_Offset, Row_Offset, All_Type, Part_Type, M, N, I,
                     (J_idx - 1)>::compute(All, Part);
  }
};

// column recursion termination
template <std::size_t Col_Offset, std::size_t Row_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t I>
struct SubstituteColumn<Col_Offset, Row_Offset, All_Type, Part_Type, M, N, I,
                        0> {
  static void compute(All_Type &All, const Part_Type &Part) {

    All.template set<(Col_Offset + I), Row_Offset>(Part.template get<I, 0>());
  }
};

// when I_idx < M
template <std::size_t Col_Offset, std::size_t Row_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t I_idx>
struct SubstituteRow {
  static void compute(All_Type &All, const Part_Type &Part) {

    SubstituteColumn<Col_Offset, Row_Offset, All_Type, Part_Type, M, N, I_idx,
                     (N - 1)>::compute(All, Part);
    SubstituteRow<Col_Offset, Row_Offset, All_Type, Part_Type, M, N,
                  (I_idx - 1)>::compute(All, Part);
  }
};

// row recursion termination
template <std::size_t Col_Offset, std::size_t Row_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N>
struct SubstituteRow<Col_Offset, Row_Offset, All_Type, Part_Type, M, N, 0> {
  static void compute(All_Type &All, const Part_Type &Part) {

    SubstituteColumn<Col_Offset, Row_Offset, All_Type, Part_Type, M, N, 0,
                     (N - 1)>::compute(All, Part);
  }
};

template <std::size_t Col_Offset, std::size_t Row_Offset, typename All_Type,
          typename Part_Type>
inline void substitute_each(All_Type &All, const Part_Type &Part) {

  static_assert(
      All_Type::COLS >= (Part_Type::COLS + Col_Offset),
      "All matrix must have enough columns to substitute the part matrix.");
  static_assert(
      All_Type::ROWS >= (Part_Type::ROWS + Row_Offset),
      "All matrix must have enough rows to substitute the part matrix.");

  SubstituteRow<Col_Offset, Row_Offset, All_Type, Part_Type, Part_Type::COLS,
                Part_Type::ROWS, (Part_Type::COLS - 1)>::compute(All, Part);
}

template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t TupleCol_Count,
          std::size_t TupleCol_Offset, std::size_t TupleRow_Offset,
          std::size_t TupleRow_Index>
struct TupleColumn {
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {

    constexpr std::size_t THIS_TUPLE_INDEX =
        N - TupleRow_Index + (TupleCol_Count * N);

    using ArgType =
        typename std::remove_reference<decltype(std::get<THIS_TUPLE_INDEX>(
            args))>::type;

    constexpr std::size_t EACH_ROW_SIZE = ArgType::ROWS;

    substitute_each<TupleCol_Offset, TupleRow_Offset>(
        All, std::get<THIS_TUPLE_INDEX>(args));
    TupleColumn<M, N, All_Type, ArgsTuple_Type, TupleCol_Count, TupleCol_Offset,
                (TupleRow_Offset + EACH_ROW_SIZE),
                (TupleRow_Index - 1)>::substitute(All, args);
  }
};

template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t TupleCol_Count,
          std::size_t TupleCol_Offset, std::size_t TupleRow_Offset>
struct TupleColumn<M, N, All_Type, ArgsTuple_Type, TupleCol_Count,
                   TupleCol_Offset, TupleRow_Offset, 0> {
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {
    // Do Nothing
    static_cast<void>(All);
    static_cast<void>(args);
  }
};

template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t TupleCol_Offset,
          std::size_t TupleCol_Index>
struct TupleRow {
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {

    constexpr std::size_t TUPLECOL_COUNT = M - TupleCol_Index;

    constexpr std::size_t THIS_TUPLE_INDEX = TUPLECOL_COUNT * N;

    using ArgType =
        typename std::remove_reference<decltype(std::get<THIS_TUPLE_INDEX>(
            args))>::type;

    constexpr std::size_t EACH_COLUMN_SIZE = ArgType::COLS;

    TupleColumn<M, N, All_Type, ArgsTuple_Type, TUPLECOL_COUNT, TupleCol_Offset,
                0, N>::substitute(All, args);

    TupleRow<M, N, All_Type, ArgsTuple_Type, TupleCol_Offset + EACH_COLUMN_SIZE,
             (TupleCol_Index - 1)>::substitute(All, args);
  }
};

template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t TupleCol_Offset>
struct TupleRow<M, N, All_Type, ArgsTuple_Type, TupleCol_Offset, 0> {
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {
    // Do Nothing
    static_cast<void>(All);
    static_cast<void>(args);
  }
};

} // namespace PartMatrixOperation

/* Concatenate block, any size */
namespace ConcatenateBlockOperation {

template <std::size_t Col_Offset, std::size_t N, typename ArgsTuple_Type,
          std::size_t Row_Index>
struct ConcatenateBlockRows {

  using Arg_Type = typename std::tuple_element<(Col_Offset + Row_Index),
                                               ArgsTuple_Type>::type;

  using type = ConcatenateHorizontally_Type<
      typename ConcatenateBlockRows<Col_Offset, N, ArgsTuple_Type,
                                    (Row_Index - 1)>::type,
      Arg_Type>;
};

template <std::size_t Col_Offset, std::size_t N, typename ArgsTuple_Type>
struct ConcatenateBlockRows<Col_Offset, N, ArgsTuple_Type, 0> {

  using type = typename std::tuple_element<Col_Offset, ArgsTuple_Type>::type;
};

template <std::size_t N, typename ArgsTuple_Type, std::size_t Col_Index>
struct ConcatenateBlockColumns {

  using Arg_Type = typename ConcatenateBlockRows<(Col_Index * N), N,
                                                 ArgsTuple_Type, (N - 1)>::type;

  using type =
      ConcatenateVertically_Type<typename ConcatenateBlockColumns<
                                     N, ArgsTuple_Type, (Col_Index - 1)>::type,
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

template <std::size_t M, std::size_t N, typename Tuple, typename... Args>
using ConcatenateArgsType_t =
    typename ConcatenateArgsType<M, N, Tuple, Args...>::type;

template <std::size_t M, std::size_t N, typename Tuple, typename Last>
inline auto concatenate_args(const Tuple &previousArgs, Last last) ->
    typename ConcatenateBlock<
        M, N,
        decltype(std::tuple_cat(previousArgs, std::make_tuple(last)))>::type {

  auto all_args = std::tuple_cat(previousArgs, std::make_tuple(last));

  using UpdatedArgsType = decltype(all_args);

  constexpr std::size_t TUPLE_SIZE = std::tuple_size<decltype(all_args)>::value;

  static_assert(TUPLE_SIZE == (M * N),
                "Number of arguments must be equal to M * N.");

  typename ConcatenateBlock<M, N, UpdatedArgsType>::type result;

  PartMatrixOperation::TupleRow<M, N, decltype(result), decltype(all_args), 0,
                                M>::substitute(result, all_args);

  return result;
}

template <std::size_t M, std::size_t N, typename Tuple, typename First,
          typename... Rest>
inline auto concatenate_args(const Tuple &previousArgs, First first,
                             Rest... rest)
    -> ConcatenateArgsType_t<
        M, N, decltype(std::tuple_cat(previousArgs, std::make_tuple(first))),
        Rest...> {

  auto updatedArgs = std::tuple_cat(previousArgs, std::make_tuple(first));

  return concatenate_args<M, N>(updatedArgs, rest...);
}

template <std::size_t M, std::size_t N, typename... Args>
inline auto calculate(Args... args)
    -> ConcatenateArgsType_t<M, N, std::tuple<>, Args...> {
  static_assert(M > 0, "M must be greater than 0.");
  static_assert(N > 0, "N must be greater than 0.");

  return concatenate_args<M, N>(std::make_tuple(), args...);
}

} // namespace ConcatenateBlockOperation

template <std::size_t M, std::size_t N, typename... Args>
inline auto concatenate_block(Args... args)
    -> ConcatenateBlockOperation::ConcatenateArgsType_t<M, N, std::tuple<>,
                                                        Args...> {

  return ConcatenateBlockOperation::calculate<M, N>(args...);
}

/* Concatenate block Type */
template <std::size_t M, std::size_t N, typename... Args>
using ConcatenateBlock_Type =
    typename ConcatenateBlockOperation::ConcatenateArgsType_t<
        M, N, std::tuple<>, Args...>;

/* repmat */
namespace ReatmatOperation {

template <std::size_t M, std::size_t N, std::size_t Count, typename MATRIX_Type,
          typename... Args>
struct GenerateRepmatTypes {
  using type = typename GenerateRepmatTypes<M, N, (Count - 1), MATRIX_Type,
                                            MATRIX_Type, Args...>::type;
};

template <std::size_t M, std::size_t N, typename MATRIX_Type, typename... Args>
struct GenerateRepmatTypes<M, N, 0, MATRIX_Type, Args...> {
  using type = ConcatenateBlock_Type<M, N, Args...>;
};

} // namespace ReatmatOperation

template <std::size_t M, std::size_t N, typename MATRIX_Type>
using Repmat_Type =
    typename ReatmatOperation::GenerateRepmatTypes<M, N, M * N,
                                                   MATRIX_Type>::type;

namespace ReatmatOperation {

template <std::size_t... Indices> struct index_sequence_for_repmat {};

template <std::size_t N, std::size_t... Indices>
struct make_index_sequence_for_repmat_impl
    : make_index_sequence_for_repmat_impl<N - 1, N - 1, Indices...> {};

template <std::size_t... Indices>
struct make_index_sequence_for_repmat_impl<0, Indices...> {
  using type = index_sequence_for_repmat<Indices...>;
};

template <std::size_t N>
using make_index_sequence_for_repmat =
    typename make_index_sequence_for_repmat_impl<N>::type;

template <std::size_t Count, typename MATRIX_Type, typename... Args>
struct RepeatMatrix {
  using type =
      typename RepeatMatrix<Count - 1, MATRIX_Type, MATRIX_Type, Args...>::type;
};

template <typename MATRIX_Type, typename... Args>
struct RepeatMatrix<0, MATRIX_Type, Args...> {
  using type = std::tuple<Args...>;
};

template <std::size_t M, std::size_t N, typename MATRIX_Type,
          std::size_t... Indices>
inline auto implement(const MATRIX_Type &matrix,
                      index_sequence_for_repmat<Indices...>)
    -> Repmat_Type<M, N, MATRIX_Type> {

  return concatenate_block<M, N>((static_cast<void>(Indices), matrix)...);
}

} // namespace ReatmatOperation

template <std::size_t M, std::size_t N, typename MATRIX_Type>
inline auto repmat(const MATRIX_Type &matrix)
    -> Repmat_Type<M, N, MATRIX_Type> {

  static_assert(M > 0, "M must be greater than 0.");
  static_assert(N > 0, "N must be greater than 0.");

  constexpr std::size_t TotalCount = M * N;

  return ReatmatOperation::implement<M, N>(
      matrix, ReatmatOperation::make_index_sequence_for_repmat<TotalCount>{});
}

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__
