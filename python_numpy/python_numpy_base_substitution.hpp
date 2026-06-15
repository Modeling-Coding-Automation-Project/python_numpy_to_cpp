/**
 * @file python_numpy_base_substitution.hpp
 *
 * @brief Substitution operations for matrix manipulation in the PythonNumpy C++
 * library.
 * This file defines the substitution operations for matrix manipulation,
 * allowing parts of a matrix to be replaced with values from another matrix.
 *
 * @note
 * tparam M is the number of rows in the matrix.
 * tparam N is the number of columns in the matrix.
 */
#ifndef PYTHON_NUMPY_BASE_SUBSTITUTION_HPP_
#define PYTHON_NUMPY_BASE_SUBSTITUTION_HPP_

#include "python_math.hpp"
#include "python_numpy_complex.hpp"
#include "python_numpy_templates.hpp"

#include <initializer_list>
#include <tuple>
#include <type_traits>

namespace PythonNumpy {

/* Part matrix substitute */
namespace PartMatrixOperation {

/** * @brief Substitutes a part matrix into a larger matrix at specified
 * offsets.
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
          typename Part_Type, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct SubstituteColumnRange;

/**
 * @brief Substitutes a range of columns from the part matrix into the All
 * matrix.
 *
 * This struct template recursively substitutes a range of columns from the part
 * matrix into the All matrix at the specified offsets. It divides the column
 * range until it reaches individual columns, which are then substituted
 * directly.
 *
 * @tparam Row_Offset The column offset for substitution.
 * @tparam Col_Offset The row offset for substitution.
 * @tparam All_Type The type of the All matrix.
 * @tparam Part_Type The type of the part matrix.
 * @tparam M The number of rows in the part matrix.
 * @tparam N The number of columns in the part matrix.
 * @tparam I The current row index for substitution.
 * @tparam Start The starting index of the column range to substitute.
 * @tparam End The ending index of the column range to substitute.
 */
template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct SubstituteColumnRange<Row_Offset, Col_Offset, All_Type, Part_Type, M, N,
                             I, Start, End,
                             typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  /**
   * @brief Substitutes a range of columns from the part matrix into the All
   * matrix.
   *
   * This static function substitutes a range of columns from the part matrix
   * into the All matrix at the specified offsets. It recursively divides the
   * column range until it reaches individual columns, which are then
   * substituted directly.
   *
   * @param All The All matrix where values are substituted.
   * @param Part The part matrix containing values to substitute.
   */
  static void compute(All_Type &All, const Part_Type &Part) {
    SubstituteColumnRange<Row_Offset, Col_Offset, All_Type, Part_Type, M, N, I,
                          Start, Mid>::compute(All, Part);
    SubstituteColumnRange<Row_Offset, Col_Offset, All_Type, Part_Type, M, N, I,
                          Mid, End>::compute(All, Part);
  }
};

template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct SubstituteColumnRange<Row_Offset, Col_Offset, All_Type, Part_Type, M, N,
                             I, Start, End,
                             typename std::enable_if<(End == Start)>::type> {
  /**
   * @brief Base case for substituting a range of columns when the range is
   * empty.
   *
   * This static function serves as the base case for the recursive substitution
   * of columns. When the column range is empty (End == Start), it does nothing.
   *
   * @param All The All matrix where values would be substituted (not used
   * here).
   * @param Part The part matrix containing values to substitute (not used
   * here).
   */
  static void compute(All_Type &All, const Part_Type &Part) {
    static_cast<void>(All);
    static_cast<void>(Part);
  }
};

template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t I,
          std::size_t Start, std::size_t End>
struct SubstituteColumnRange<
    Row_Offset, Col_Offset, All_Type, Part_Type, M, N, I, Start, End,
    typename std::enable_if<(End - Start == 1)>::type> {
  /**
   * @brief Substitutes a single column from the part matrix into the All
   * matrix.
   * This static function substitutes a single column (indexed by Start) from
   * the part matrix into the All matrix at the specified offsets. It directly
   * sets the value from the part matrix into the All matrix for the specified
   * row and column.
   * @param All The All matrix where values are substituted.
   * @param Part The part matrix containing values to substitute.
   */
  static void compute(All_Type &All, const Part_Type &Part) {
    All.template set<(Row_Offset + I), (Col_Offset + Start)>(
        Part.template get<I, Start>());
  }
};

/**
 * @brief Substitutes a range of rows from the part matrix into the All matrix.
 *
 * This struct template recursively substitutes a range of rows from the part
 * matrix into the All matrix at the specified offsets. It divides the row
 * range until it reaches individual rows, which are then substituted directly.
 *
 * @tparam Row_Offset The column offset for substitution.
 * @tparam Col_Offset The row offset for substitution.
 * @tparam All_Type The type of the All matrix.
 * @tparam Part_Type The type of the part matrix.
 * @tparam M The number of rows in the part matrix.
 * @tparam N The number of columns in the part matrix.
 * @tparam Start The starting index of the row range to substitute.
 * @tparam End The ending index of the row range to substitute.
 */
template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct SubstituteRowRange;

/**
 * @brief Substitutes a range of rows from the part matrix into the All
 * matrix.
 *
 * This static function substitutes a range of rows from the part matrix into
 * the All matrix at the specified offsets. It recursively divides the row
 * range until it reaches individual rows, which are then substituted
 * directly.
 *
 * @param All The All matrix where values are substituted.
 * @param Part The part matrix containing values to substitute.
 */
template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct SubstituteRowRange<Row_Offset, Col_Offset, All_Type, Part_Type, M, N,
                          Start, End,
                          typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;

  /**
   * @brief Substitutes a range of rows from the part matrix into the All
   * matrix.
   *
   * This static function substitutes a range of rows from the part matrix into
   * the All matrix at the specified offsets. It recursively divides the row
   * range until it reaches individual rows, which are then substituted
   * directly.
   *
   * @param All The All matrix where values are substituted.
   * @param Part The part matrix containing values to substitute.
   */
  static void compute(All_Type &All, const Part_Type &Part) {
    SubstituteRowRange<Row_Offset, Col_Offset, All_Type, Part_Type, M, N, Start,
                       Mid>::compute(All, Part);
    SubstituteRowRange<Row_Offset, Col_Offset, All_Type, Part_Type, M, N, Mid,
                       End>::compute(All, Part);
  }
};

template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct SubstituteRowRange<Row_Offset, Col_Offset, All_Type, Part_Type, M, N,
                          Start, End,
                          typename std::enable_if<(End == Start)>::type> {
  /**
   * @brief Base case for substituting a range of rows when the range is empty.
   *
   * This static function serves as the base case for the recursive substitution
   * of rows. When the row range is empty (End == Start), it does nothing.
   *
   * @param All The All matrix where values would be substituted (not used
   * here).
   * @param Part The part matrix containing values to substitute (not used
   * here).
   */
  static void compute(All_Type &All, const Part_Type &Part) {
    static_cast<void>(All);
    static_cast<void>(Part);
  }
};

template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct SubstituteRowRange<Row_Offset, Col_Offset, All_Type, Part_Type, M, N,
                          Start, End,
                          typename std::enable_if<(End - Start == 1)>::type> {
  /**
   * @brief Substitutes a single row from the part matrix into the All
   * matrix.
   * This static function substitutes a single row (indexed by Start) from the
   * part matrix into the All matrix at the specified offsets. It directly sets
   * the value from the part matrix into the All matrix for the specified row
   * and column.
   * @param All The All matrix where values are substituted.
   * @param Part The part matrix containing values to substitute.
   */
  static void compute(All_Type &All, const Part_Type &Part) {
    SubstituteColumnRange<Row_Offset, Col_Offset, All_Type, Part_Type, M, N,
                          Start, 0, N>::compute(All, Part);
  }
};

template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J_idx>
struct SubstituteColumn {
  /**
   * @brief Substitutes a specific column from the part matrix into the All
   * matrix.
   *
   * This static function substitutes a specific column (indexed by J_idx) from
   * the part matrix into the All matrix at the specified offsets. It then
   * recursively processes the remaining rows by invoking compute with a
   * decremented row index.
   *
   * @param All The All matrix where values are substituted.
   * @param Part The part matrix containing values to substitute.
   */
  static void compute(All_Type &All, const Part_Type &Part) {
    SubstituteColumnRange<Row_Offset, Col_Offset, All_Type, Part_Type, M, N, I,
                          0, (J_idx + 1)>::compute(All, Part);
  }
};

template <std::size_t Row_Offset, std::size_t Col_Offset, typename All_Type,
          typename Part_Type, std::size_t M, std::size_t N, std::size_t I_idx>
struct SubstituteRow {
  /**
   * @brief Substitutes a specific row from the part matrix into the All
   * matrix.
   *
   * This static function substitutes a specific row (indexed by I_idx) from the
   * part matrix into the All matrix at the specified offsets. It then
   * recursively processes the remaining columns by invoking compute with a
   * decremented column index.
   *
   * @param All The All matrix where values are substituted.
   * @param Part The part matrix containing values to substitute.
   */
  static void compute(All_Type &All, const Part_Type &Part) {
    SubstituteRowRange<Row_Offset, Col_Offset, All_Type, Part_Type, M, N, 0,
                       (I_idx + 1)>::compute(All, Part);
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

  SubstituteRowRange<Row_Offset, Col_Offset, All_Type, Part_Type,
                     Part_Type::ROWS, Part_Type::COLS, 0,
                     Part_Type::ROWS>::compute(All, Part);
}

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
template <std::size_t M, std::size_t N, typename ArgsTuple_Type,
          std::size_t TupleRow_Index, std::size_t Start, std::size_t End,
          typename Enable = void>
struct TupleColumnWidthSum;

/**
 * @brief Computes the total width of a range of columns in a tuple of
 * arguments.
 *
 * This struct template recursively computes the total width (number of columns)
 * of a range of columns in a tuple of arguments. It divides the column range
 * until it reaches individual columns, at which point it retrieves the width
 * from the corresponding argument type.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam ArgsTuple_Type The type of the tuple containing arguments.
 * @tparam TupleRow_Index The index of the current row in the tuple.
 * @tparam Start The starting index of the column range to compute.
 * @tparam End The ending index of the column range to compute.
 */
template <std::size_t M, std::size_t N, typename ArgsTuple_Type,
          std::size_t TupleRow_Index, std::size_t Start, std::size_t End>
struct TupleColumnWidthSum<M, N, ArgsTuple_Type, TupleRow_Index, Start, End,
                           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static constexpr std::size_t value =
      TupleColumnWidthSum<M, N, ArgsTuple_Type, TupleRow_Index, Start,
                          Mid>::value +
      TupleColumnWidthSum<M, N, ArgsTuple_Type, TupleRow_Index, Mid,
                          End>::value;
};

/**
 * @brief Base case for computing the total width of a range of columns when the
 * range is empty.
 *
 * This struct template serves as the base case for the recursive computation of
 * column widths. When the column range is empty (End == Start), it defines the
 * width as 0.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam ArgsTuple_Type The type of the tuple containing arguments.
 * @tparam TupleRow_Index The index of the current row in the tuple.
 * @tparam Start The starting index of the column range to compute.
 * @tparam End The ending index of the column range to compute.
 */
template <std::size_t M, std::size_t N, typename ArgsTuple_Type,
          std::size_t TupleRow_Index, std::size_t Start, std::size_t End>
struct TupleColumnWidthSum<M, N, ArgsTuple_Type, TupleRow_Index, Start, End,
                           typename std::enable_if<(End == Start)>::type> {
  static constexpr std::size_t value = 0;
};

/**
 * @brief Base case for computing the total width of a range of columns when the
 * range has only one column.
 *
 * This struct template serves as the base case for the recursive computation of
 * column widths. When the column range has only one column (End - Start == 1),
 * it retrieves the width from the corresponding argument type in the tuple.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam ArgsTuple_Type The type of the tuple containing arguments.
 * @tparam TupleRow_Index The index of the current row in the tuple.
 * @tparam Start The starting index of the column range to compute.
 * @tparam End The ending index of the column range to compute.
 */
template <std::size_t M, std::size_t N, typename ArgsTuple_Type,
          std::size_t TupleRow_Index, std::size_t Start, std::size_t End>
struct TupleColumnWidthSum<M, N, ArgsTuple_Type, TupleRow_Index, Start, End,
                           typename std::enable_if<(End - Start == 1)>::type> {
  static constexpr std::size_t TupleIndex = (TupleRow_Index * N) + Start;
  using ArgType = typename std::tuple_element<TupleIndex, ArgsTuple_Type>::type;
  static constexpr std::size_t value = ArgType::COLS;
};

/**
 * @brief Computes the total height of a range of rows in a tuple of arguments.
 *
 * This struct template recursively computes the total height (number of rows)
 * of a range of rows in a tuple of arguments. It divides the row range until it
 * reaches individual rows, at which point it retrieves the height from the
 * corresponding argument type.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam ArgsTuple_Type The type of the tuple containing arguments.
 * @tparam Start The starting index of the row range to compute.
 * @tparam End The ending index of the row range to compute.
 */
template <std::size_t M, std::size_t N, typename ArgsTuple_Type,
          std::size_t Start, std::size_t End, typename Enable = void>
struct TupleRowHeightSum;

/**
 * @brief Recursively computes the total height of a range of rows in a tuple of
 * arguments when the range has more than one row.
 *
 * This struct template recursively computes the total height (number of rows)
 * of a range of rows in a tuple of arguments. It divides the row range until it
 * reaches individual rows, at which point it retrieves the height from the
 * corresponding argument type.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam ArgsTuple_Type The type of the tuple containing arguments.
 * @tparam Start The starting index of the row range to compute.
 * @tparam End The ending index of the row range to compute.
 */
template <std::size_t M, std::size_t N, typename ArgsTuple_Type,
          std::size_t Start, std::size_t End>
struct TupleRowHeightSum<M, N, ArgsTuple_Type, Start, End,
                         typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static constexpr std::size_t value =
      TupleRowHeightSum<M, N, ArgsTuple_Type, Start, Mid>::value +
      TupleRowHeightSum<M, N, ArgsTuple_Type, Mid, End>::value;
};

/**
 * @brief Base case for computing the total height of a range of rows when the
 * range is empty.
 *
 * This struct template serves as the base case for the recursive computation of
 * row heights. When the row range is empty (End == Start), it defines the
 * height as 0.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam ArgsTuple_Type The type of the tuple containing arguments.
 * @tparam Start The starting index of the row range to compute.
 * @tparam End The ending index of the row range to compute.
 */
template <std::size_t M, std::size_t N, typename ArgsTuple_Type,
          std::size_t Start, std::size_t End>
struct TupleRowHeightSum<M, N, ArgsTuple_Type, Start, End,
                         typename std::enable_if<(End == Start)>::type> {
  static constexpr std::size_t value = 0;
};

/**
 * @brief Base case for computing the total height of a range of rows when the
 * range has only one row.
 *
 * This struct template serves as the base case for the recursive computation of
 * row heights. When the row range has only one row (End - Start == 1), it
 * retrieves the height from the corresponding argument type in the tuple.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam ArgsTuple_Type The type of the tuple containing arguments.
 * @tparam Start The starting index of the row range to compute.
 * @tparam End The ending index of the row range to compute.
 */
template <std::size_t M, std::size_t N, typename ArgsTuple_Type,
          std::size_t Start, std::size_t End>
struct TupleRowHeightSum<M, N, ArgsTuple_Type, Start, End,
                         typename std::enable_if<(End - Start == 1)>::type> {
  static constexpr std::size_t TupleIndex = Start * N;
  using ArgType = typename std::tuple_element<TupleIndex, ArgsTuple_Type>::type;
  static constexpr std::size_t value = ArgType::ROWS;
};

/**
 * @brief Substitutes a range of columns from a tuple of arguments into the All
 * matrix.
 *
 * This struct template recursively substitutes a range of columns from a tuple
 * of arguments into the All matrix at the specified offsets. It divides the
 * column range until it reaches individual columns, which are then substituted
 * directly.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam All_Type The type of the All matrix.
 * @tparam ArgsTuple_Type The type of the tuple containing arguments.
 * @tparam TupleRow_Index The index of the current row in the tuple.
 * @tparam TupleRow_Offset The column offset for substitution.
 * @tparam Start The starting index of the column range to substitute.
 * @tparam End The ending index of the column range to substitute.
 * @tparam TupleCol_Offset The row offset for substitution.
 */
template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t TupleRow_Index,
          std::size_t TupleRow_Offset, std::size_t Start, std::size_t End,
          std::size_t TupleCol_Offset, typename Enable = void>
struct TupleColumnRange;

/**
 * @brief Recursively substitutes a range of columns from a tuple of arguments
 * into the All matrix when the range has more than one column.
 *
 * This struct template recursively substitutes a range of columns from a tuple
 * of arguments into the All matrix at the specified offsets. It divides the
 * column range until it reaches individual columns, which are then substituted
 * directly.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam All_Type The type of the All matrix.
 * @tparam ArgsTuple_Type The type of the tuple containing arguments.
 * @tparam TupleRow_Index The index of the current row in the tuple.
 * @tparam TupleRow_Offset The column offset for substitution.
 * @tparam Start The starting index of the column range to substitute.
 * @tparam End The ending index of the column range to substitute.
 * @tparam TupleCol_Offset The row offset for substitution.
 */
template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t TupleRow_Index,
          std::size_t TupleRow_Offset, std::size_t Start, std::size_t End,
          std::size_t TupleCol_Offset>
struct TupleColumnRange<M, N, All_Type, ArgsTuple_Type, TupleRow_Index,
                        TupleRow_Offset, Start, End, TupleCol_Offset,
                        typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static constexpr std::size_t LEFT_WIDTH =
      TupleColumnWidthSum<M, N, ArgsTuple_Type, TupleRow_Index, Start,
                          Mid>::value;

  /**
   * @brief Substitutes a range of columns from a tuple of arguments into the
   * All matrix.
   *
   * This static function substitutes a range of columns from a tuple of
   * arguments into the All matrix at the specified offsets. It recursively
   * divides the column range until it reaches individual columns, which are
   * then substituted directly.
   *
   * @param All The All matrix where values are substituted.
   * @param args The tuple containing arguments to substitute.
   */
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {
    TupleColumnRange<M, N, All_Type, ArgsTuple_Type, TupleRow_Index,
                     TupleRow_Offset, Start, Mid,
                     TupleCol_Offset>::substitute(All, args);
    TupleColumnRange<M, N, All_Type, ArgsTuple_Type, TupleRow_Index,
                     TupleRow_Offset, Mid, End,
                     (TupleCol_Offset + LEFT_WIDTH)>::substitute(All, args);
  }
};

template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t TupleRow_Index,
          std::size_t TupleRow_Offset, std::size_t Start, std::size_t End,
          std::size_t TupleCol_Offset>
struct TupleColumnRange<M, N, All_Type, ArgsTuple_Type, TupleRow_Index,
                        TupleRow_Offset, Start, End, TupleCol_Offset,
                        typename std::enable_if<(End == Start)>::type> {
  /**
   * @brief Specialization for an empty column range.
   *
   * This struct template specialization handles the case where the column range
   * is empty (End == Start). In this case, no substitution is performed.
   */
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {
    static_cast<void>(All);
    static_cast<void>(args);
  }
};

template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t TupleRow_Index,
          std::size_t TupleRow_Offset, std::size_t Start, std::size_t End,
          std::size_t TupleCol_Offset>
struct TupleColumnRange<M, N, All_Type, ArgsTuple_Type, TupleRow_Index,
                        TupleRow_Offset, Start, End, TupleCol_Offset,
                        typename std::enable_if<(End - Start == 1)>::type> {
  static constexpr std::size_t THIS_TUPLE_INDEX = (TupleRow_Index * N) + Start;

  /**
   * @brief Substitutes a single column from a tuple of arguments into the All
   * matrix.
   *
   * This static function substitutes a single column (indexed by
   * THIS_TUPLE_INDEX) from a tuple of arguments into the All matrix at the
   * specified offsets. It directly sets the value from the corresponding
   * argument type in the tuple into the All matrix for the specified row and
   * column.
   *
   * @param All The All matrix where values are substituted.
   * @param args The tuple containing arguments to substitute.
   */
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {
    substitute_each<TupleRow_Offset, TupleCol_Offset>(
        All, std::get<THIS_TUPLE_INDEX>(args));
  }
};

/**
 * @brief Substitutes a range of rows from a tuple of arguments into the All
 * matrix.
 *
 * This struct template recursively substitutes a range of rows from a tuple of
 * arguments into the All matrix at the specified offsets. It divides the row
 * range until it reaches individual rows, which are then substituted directly.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam All_Type The type of the All matrix.
 * @tparam ArgsTuple_Type The type of the tuple containing arguments.
 * @tparam Start The starting index of the row range to substitute.
 * @tparam End The ending index of the row range to substitute.
 * @tparam TupleRow_Offset The column offset for substitution.
 */
template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t Start, std::size_t End,
          std::size_t TupleRow_Offset, typename Enable = void>
struct TupleRowRange;

template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t Start, std::size_t End,
          std::size_t TupleRow_Offset>
struct TupleRowRange<M, N, All_Type, ArgsTuple_Type, Start, End,
                     TupleRow_Offset,
                     typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static constexpr std::size_t LEFT_HEIGHT =
      TupleRowHeightSum<M, N, ArgsTuple_Type, Start, Mid>::value;

  /**
   * @brief Substitutes a range of rows from a tuple of arguments into the All
   * matrix.
   * This static function substitutes a range of rows from a tuple of arguments
   * into the All matrix at the specified offsets. It recursively divides the
   * row range until it reaches individual rows, which are then substituted
   * directly.
   * @param All The All matrix where values are substituted.
   * @param args The tuple containing arguments to substitute.
   */
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {
    TupleRowRange<M, N, All_Type, ArgsTuple_Type, Start, Mid,
                  TupleRow_Offset>::substitute(All, args);
    TupleRowRange<M, N, All_Type, ArgsTuple_Type, Mid, End,
                  (TupleRow_Offset + LEFT_HEIGHT)>::substitute(All, args);
  }
};

template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t Start, std::size_t End,
          std::size_t TupleRow_Offset>
struct TupleRowRange<M, N, All_Type, ArgsTuple_Type, Start, End,
                     TupleRow_Offset,
                     typename std::enable_if<(End == Start)>::type> {
  /**
   * @brief Specialization for an empty row range.
   *
   * This struct template specialization handles the case where the row range is
   * empty (End == Start). In this case, no substitution is performed.
   */
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {
    static_cast<void>(All);
    static_cast<void>(args);
  }
};

template <std::size_t M, std::size_t N, typename All_Type,
          typename ArgsTuple_Type, std::size_t Start, std::size_t End,
          std::size_t TupleRow_Offset>
struct TupleRowRange<M, N, All_Type, ArgsTuple_Type, Start, End,
                     TupleRow_Offset,
                     typename std::enable_if<(End - Start == 1)>::type> {
  /**
   * @brief Specialization for a single row range.
   *
   * This struct template specialization handles the case where the row range
   * contains exactly one row (End - Start == 1). In this case, the substitution
   * is performed directly for that row.
   */
  static void substitute(All_Type &All, const ArgsTuple_Type &args) {
    TupleColumnRange<M, N, All_Type, ArgsTuple_Type, Start, TupleRow_Offset, 0,
                     N, 0>::substitute(All, args);
  }
};

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
    constexpr std::size_t Start = N - TupleCol_Index;
    TupleColumnRange<M, N, All_Type, ArgsTuple_Type, TupleRow_Count,
                     TupleRow_Offset, Start, N,
                     TupleCol_Offset>::substitute(All, args);
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
    constexpr std::size_t Start = M - TupleRow_Index;
    TupleRowRange<M, N, All_Type, ArgsTuple_Type, Start, M,
                  TupleRow_Offset>::substitute(All, args);
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

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_BASE_SUBSTITUTION_HPP_
