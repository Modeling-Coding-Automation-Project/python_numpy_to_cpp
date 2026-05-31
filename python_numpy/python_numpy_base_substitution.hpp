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

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_BASE_SUBSTITUTION_HPP_
