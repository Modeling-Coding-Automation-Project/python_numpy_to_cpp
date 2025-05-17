#ifndef __PYTHON_NUMPY_BASE_SIMPLIFIED_ACTION_HPP__
#define __PYTHON_NUMPY_BASE_SIMPLIFIED_ACTION_HPP__

#include "python_numpy_base.hpp"
#include "python_numpy_base_simplification.hpp"
#include "python_numpy_complex.hpp"
#include "python_numpy_templates.hpp"

#include "python_numpy_concatenate.hpp"

#include <initializer_list>
#include <tuple>
#include <type_traits>

namespace PythonNumpy {

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

/* Substitute same size Matrix */
template <typename From_Type, typename To_Type>
inline void substitute_matrix(To_Type &to_matrix,
                              const From_Type &from_matrix) {

  static_assert(From_Type::COLS * From_Type::ROWS ==
                    To_Type::COLS * To_Type::ROWS,
                "The number of elements in the source and destination matrices "
                "must be the same.");

  PartMatrixOperation::substitute_each<0, 0>(to_matrix, from_matrix);
}

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
using ArgsType_t = typename ConcatenateArgsType<M, N, Tuple, Args...>::type;

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

template <std::size_t M, std::size_t N, typename Concatenate_Type,
          typename Tuple, typename First, typename... Rest>
inline void concatenate_args(Concatenate_Type &Concatenated,
                             const Tuple &previousArgs, First first,
                             Rest... rest) {

  auto updatedArgs = std::tuple_cat(previousArgs, std::make_tuple(first));

  return concatenate_args<M, N>(Concatenated, updatedArgs, rest...);
}

template <std::size_t M, std::size_t N, typename... Args>
inline void calculate(ArgsType_t<M, N, std::tuple<>, Args...> &Concatenated,
                      Args... args) {
  static_assert(M > 0, "M must be greater than 0.");
  static_assert(N > 0, "N must be greater than 0.");

  return concatenate_args<M, N>(Concatenated, std::make_tuple(), args...);
}

} // namespace ConcatenateBlockOperation

template <std::size_t M, std::size_t N, typename Concatenate_Type,
          typename... Args>
inline void update_block_concatenated_matrix(Concatenate_Type &Concatenated,
                                             Args... args) {

  static_assert(M > 0, "M must be greater than 0.");
  static_assert(N > 0, "N must be greater than 0.");

  ConcatenateBlockOperation::calculate<M, N>(Concatenated, args...);
}

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

template <std::size_t M, std::size_t N, typename MATRIX_Type>
using Tile_Type =
    typename TileOperation::GenerateTileTypes<M, N, M * N, MATRIX_Type>::type;

namespace TileOperation {

template <std::size_t... Indices> struct index_sequence_for_tile {};

template <std::size_t N, std::size_t... Indices>
struct make_index_sequence_for_tile_impl
    : make_index_sequence_for_tile_impl<N - 1, N - 1, Indices...> {};

template <std::size_t... Indices>
struct make_index_sequence_for_tile_impl<0, Indices...> {
  using type = index_sequence_for_tile<Indices...>;
};

template <std::size_t N>
using make_index_sequence_for_tile =
    typename make_index_sequence_for_tile_impl<N>::type;

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
inline void calculate(Tile_Type<M, N, MATRIX_Type> &Concatenated,
                      const MATRIX_Type &matrix,
                      index_sequence_for_tile<Indices...>) {

  update_block_concatenated_matrix<M, N>(
      Concatenated, (static_cast<void>(Indices), matrix)...);
}

} // namespace TileOperation

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
struct Column {
  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {

    constexpr std::size_t NUMBER_OF_ELEMENT = I * From_Type::ROWS + J_idx;

    constexpr std::size_t FROM_MATRIX_ROW_INDEX =
        static_cast<std::size_t>(NUMBER_OF_ELEMENT / From_Type::COLS);
    constexpr std::size_t FROM_MATRIX_COL_INDEX =
        NUMBER_OF_ELEMENT - FROM_MATRIX_ROW_INDEX * From_Type::COLS;

    static_assert(FROM_MATRIX_COL_INDEX < From_Type::COLS,
                  "The column index is out of range for the from_matrix.");
    static_assert(FROM_MATRIX_ROW_INDEX < From_Type::ROWS,
                  "The row index is out of range for the from_matrix.");

    constexpr std::size_t TO_MATRIX_ROW_INDEX =
        static_cast<std::size_t>(NUMBER_OF_ELEMENT / To_Type::COLS);
    constexpr std::size_t TO_MATRIX_COL_INDEX =
        NUMBER_OF_ELEMENT - TO_MATRIX_ROW_INDEX * To_Type::COLS;

    static_assert(TO_MATRIX_COL_INDEX < To_Type::COLS,
                  "The column index is out of range for the to_matrix.");
    static_assert(TO_MATRIX_ROW_INDEX < To_Type::ROWS,
                  "The row index is out of range for the to_matrix.");

    to_matrix.template set<TO_MATRIX_COL_INDEX, TO_MATRIX_ROW_INDEX>(
        from_matrix
            .template get<FROM_MATRIX_COL_INDEX, FROM_MATRIX_ROW_INDEX>());
    Column<To_Type, From_Type, I, J_idx - 1>::substitute(to_matrix,
                                                         from_matrix);
  }
};

// column recursion termination
template <typename To_Type, typename From_Type, std::size_t I>
struct Column<To_Type, From_Type, I, 0> {
  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {

    constexpr std::size_t NUMBER_OF_ELEMENT = I * From_Type::ROWS;

    constexpr std::size_t FROM_MATRIX_ROW_INDEX =
        static_cast<std::size_t>(NUMBER_OF_ELEMENT / From_Type::COLS);
    constexpr std::size_t FROM_MATRIX_COL_INDEX =
        NUMBER_OF_ELEMENT - FROM_MATRIX_ROW_INDEX * From_Type::COLS;

    static_assert(FROM_MATRIX_COL_INDEX < From_Type::COLS,
                  "The column index is out of range for the from_matrix.");
    static_assert(FROM_MATRIX_ROW_INDEX < From_Type::ROWS,
                  "The row index is out of range for the from_matrix.");

    constexpr std::size_t TO_MATRIX_ROW_INDEX =
        static_cast<std::size_t>(NUMBER_OF_ELEMENT / To_Type::COLS);
    constexpr std::size_t TO_MATRIX_COL_INDEX =
        NUMBER_OF_ELEMENT - TO_MATRIX_ROW_INDEX * To_Type::COLS;

    static_assert(TO_MATRIX_COL_INDEX < To_Type::COLS,
                  "The column index is out of range for the to_matrix.");
    static_assert(TO_MATRIX_ROW_INDEX < To_Type::ROWS,
                  "The row index is out of range for the to_matrix.");

    to_matrix.template set<TO_MATRIX_COL_INDEX, TO_MATRIX_ROW_INDEX>(
        from_matrix
            .template get<FROM_MATRIX_COL_INDEX, FROM_MATRIX_ROW_INDEX>());
  }
};

// when I_idx < M
template <typename To_Type, typename From_Type, std::size_t M, std::size_t N,
          std::size_t I_idx>
struct Row {
  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {
    Column<To_Type, From_Type, I_idx, N - 1>::substitute(to_matrix,
                                                         from_matrix);
    Row<To_Type, From_Type, M, N, I_idx - 1>::substitute(to_matrix,
                                                         from_matrix);
  }
};

// row recursion termination
template <typename To_Type, typename From_Type, std::size_t M, std::size_t N>
struct Row<To_Type, From_Type, M, N, 0> {
  static void substitute(To_Type &to_matrix, const From_Type &from_matrix) {
    Column<To_Type, From_Type, 0, N - 1>::substitute(to_matrix, from_matrix);
  }
};

template <typename To_Type, typename From_Type>
inline void substitute(To_Type &to_matrix, const From_Type &from_matrix) {
  Row<To_Type, From_Type, From_Type::COLS, From_Type::ROWS,
      (From_Type::COLS - 1)>::substitute(to_matrix, from_matrix);
}

} // namespace ReshapeOperation

template <typename To_Type, typename From_Type>
inline void update_reshaped_matrix(To_Type &to_matrix,
                                   const From_Type &from_matrix) {

  static_assert(From_Type::COLS * From_Type::ROWS ==
                    To_Type::COLS * To_Type::ROWS,
                "The number of elements in the source and destination matrices "
                "must be the same.");

  ReshapeOperation::substitute(to_matrix, from_matrix);
}

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_BASE_SIMPLIFIED_ACTION_HPP__
