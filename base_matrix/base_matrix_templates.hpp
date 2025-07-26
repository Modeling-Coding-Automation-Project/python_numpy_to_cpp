/**
 * @file base_matrix_templates.hpp
 * @brief Template metaprogramming utilities for compile-time sparse and dense
 * matrix structure manipulation.
 *
 * This header provides a comprehensive set of C++ template metaprogramming
 * tools for representing, constructing, and manipulating matrix structures
 * (both sparse and dense) at compile time. It enables the definition of matrix
 * sparsity patterns, row/column indices, pointers, and various matrix
 * operations (addition, multiplication, concatenation, transpose, triangular
 * extraction, etc.) entirely through type-level programming.
 *
 * The utilities are designed to support static analysis and code generation for
 * matrix operations, particularly for applications such as scientific
 * computing, code automation, and symbolic manipulation of matrix structures.
 *
 * @note
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __BASE_MATRIX_TEMPLATES_HPP__
#define __BASE_MATRIX_TEMPLATES_HPP__

#include "base_matrix_macros.hpp"

#include <tuple>

namespace Base {
namespace Matrix {

/**
 * @namespace TemplatesOperation
 * @brief Contains template utilities for compile-time array and sparse matrix
 * operations.
 *
 * This namespace provides template structures and classes to facilitate
 * compile-time manipulation of arrays and sparse matrix representations.
 */
namespace TemplatesOperation {

/**
 * @struct list_array
 * @brief Represents a compile-time array of std::size_t values.
 *
 * @tparam Sizes Variadic template parameter pack representing the array
 * elements.
 *
 * Provides:
 * - A static constexpr member `size` indicating the number of elements.
 * - A static constexpr array `value` containing the elements.
 */
template <std::size_t... Sizes> struct list_array {
  static constexpr std::size_t size = sizeof...(Sizes);
  static constexpr std::size_t value[size] = {Sizes...};
};

/**
 * @class CompiledSparseMatrixList
 * @brief Provides a compile-time interface to access a list of indices for
 * sparse matrices.
 *
 * @tparam Array A type that provides static constexpr members `value` (pointer
 * to array) and `size`.
 *
 * Provides:
 * - A typedef `list_type` for a pointer to const std::size_t.
 * - A static constexpr member `list` pointing to the array of indices.
 * - A static constexpr member `size` indicating the number of indices.
 */
template <std::size_t... Sizes>
constexpr std::size_t list_array<Sizes...>::value[list_array<Sizes...>::size];

template <typename Array> class CompiledSparseMatrixList {
public:
  typedef const std::size_t *list_type;
  static constexpr list_type list = Array::value;
  static constexpr std::size_t size = Array::size;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template for creating a compiled sparse matrix list of row
 * indices.
 *
 * This alias uses the TemplatesOperation::CompiledSparseMatrixList template,
 * parameterized with a list of sizes provided as template arguments.
 * It is typically used to represent a list of row indices for sparse matrix
 * operations.
 *
 * @tparam Sizes Variadic template parameter pack representing the row indices.
 */
template <std::size_t... Sizes>
using RowIndices = TemplatesOperation::CompiledSparseMatrixList<
    TemplatesOperation::list_array<Sizes...>>;

/**
 * @brief Alias template for representing a list of row pointers for sparse
 * matrices.
 *
 * This alias uses the TemplatesOperation::CompiledSparseMatrixList template,
 * parameterized with a list of sizes provided via
 * TemplatesOperation::list_array. It is typically used to define compile-time
 * row pointer structures for sparse matrix representations, where the sizes
 * correspond to the dimensions or non-zero counts per row.
 *
 * @tparam Sizes Variadic list of size_t values representing row sizes or
 * indices.
 */
template <std::size_t... Sizes>
using RowPointers = TemplatesOperation::CompiledSparseMatrixList<
    TemplatesOperation::list_array<Sizes...>>;

namespace TemplatesOperation {

/* Create Sparse Matrix from Matrix Element List */

/**
 * @brief A template struct that represents a compile-time list of boolean
 * flags.
 *
 * This struct provides a static constexpr array `value` containing the boolean
 * flags passed as template parameters, and a static constexpr `size` indicating
 * the number of flags.
 *
 * @tparam Flags Variadic boolean template parameters representing the list of
 * flags.
 *
 * Example usage:
 * @code
 * using MyFlags = available_list_array<true, false, true>;
 * // MyFlags::size == 3
 * // MyFlags::value == {true, false, true}
 * @endcode
 */
template <bool... Flags> struct available_list_array {
  static constexpr std::size_t size = sizeof...(Flags);
  static constexpr bool value[size] = {Flags...};
};

/**
 * @brief Static constexpr array indicating the availability of each flag in the
 * template parameter pack.
 *
 * This member array, `value`, is defined for the `available_list_array`
 * template class, parameterized by a variadic boolean template parameter pack
 * (`Flags...`). The array has a size equal to the number of flags provided and
 * contains the boolean values from the parameter pack, representing the
 * availability or state of each corresponding flag.
 *
 * @tparam Flags Variadic boolean template parameters representing individual
 * flags.
 * @see available_list_array
 */
template <bool... Flags>
constexpr bool
    available_list_array<Flags...>::value[available_list_array<Flags...>::size];

/**
 * @brief Template class for representing a compiled list of sparse matrix
 * elements.
 *
 * This class template provides a static interface to access a list of boolean
 * values (typically representing the presence or absence of elements in a
 * sparse matrix) and its size at compile time. The list and its size are
 * extracted from the provided Array type, which must define a static constexpr
 * pointer `value` to a boolean array and a static constexpr `size` indicating
 * the number of elements.
 *
 * @tparam Array The type providing the static boolean array and its size.
 *
 * @note This class is intended for use with compile-time constant data.
 */
template <typename Array> class CompiledSparseMatrixElementList {
public:
  typedef const bool *list_type;
  static constexpr list_type list = Array::value;
  static constexpr std::size_t size = Array::size;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template for creating a compiled sparse matrix element list
 * based on column availability flags.
 *
 * This template alias uses a parameter pack of boolean flags to specify the
 * availability of columns. It leverages
 * `TemplatesOperation::available_list_array` to generate a list array from the
 * provided flags, and then passes this list to
 * `TemplatesOperation::CompiledSparseMatrixElementList` to create the
 * corresponding type.
 *
 * @tparam Flags Boolean flags indicating the availability of each column.
 *
 * @see TemplatesOperation::available_list_array
 * @see TemplatesOperation::CompiledSparseMatrixElementList
 */
template <bool... Flags>
using ColumnAvailable = TemplatesOperation::CompiledSparseMatrixElementList<
    TemplatesOperation::available_list_array<Flags...>>;

namespace TemplatesOperation {

/**
 * @brief Template struct to represent a collection of available columns in a
 * sparse matrix.
 *
 * This struct aggregates information about multiple columns, each represented
 * by a type. It provides compile-time access to the number of columns, the
 * lists associated with each column, and the size of the first column.
 *
 * @tparam Columns Variadic template parameter pack representing column types.
 *
 * Members:
 * - number_of_columns: The total number of columns provided.
 * - ExtractList: Helper struct to extract the list from each column type.
 * - lists: Compile-time array of lists, one for each column.
 * - column_size: The size of the first column type.
 */
template <typename... Columns> struct SparseAvailableColumns {
  static constexpr std::size_t number_of_columns = sizeof...(Columns);

  /**
   * @brief Helper struct to extract the list type from a given Column type.
   *
   * This template struct defines a static constexpr member `value` that
   * retrieves the `list` member from the specified `Column` type. The type of
   * `value` is determined by `Column::list_type`.
   *
   * @tparam Column The type from which to extract the `list` member and its
   * type.
   */
  template <typename Column> struct ExtractList {
    static constexpr typename Column::list_type value = Column::list;
  };

  /**
   * @brief Array of pointers to constant boolean lists, one for each column.
   *
   * This static constexpr array holds pointers to boolean lists extracted for
   * each column using the ExtractList template. The size of the array is
   * determined by the number of columns. Each entry in the array corresponds to
   * the value provided by ExtractList<Columns>::value.
   *
   * @tparam Columns Template parameter pack representing the columns.
   * @tparam number_of_columns The total number of columns.
   */
  static constexpr const bool *lists[number_of_columns] = {
      ExtractList<Columns>::value...};

  /**
   * @brief Represents an unsigned integral type used for sizes and counts.
   *
   * std::size_t is the unsigned integer type returned by the sizeof operator
   * and is commonly used for array indexing and loop counting. It is guaranteed
   * to be able to represent the size of any object in bytes.
   */
  static constexpr std::size_t column_size =
      std::tuple_element<0, std::tuple<Columns...>>::type::size;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to determine the availability of sparse columns.
 *
 * This alias uses the `SparseAvailableColumns` metafunction from the
 * `TemplatesOperation` namespace to check or operate on the provided column
 * types (`Columns...`). It is typically used in template metaprogramming to
 * enable or disable features based on the properties of the given columns.
 *
 * @tparam Columns Variadic template parameter pack representing column types to
 * be checked for sparse availability.
 */
template <typename... Columns>
using SparseAvailable = TemplatesOperation::SparseAvailableColumns<Columns...>;

namespace TemplatesOperation {

/**
 * @brief Compile-time logical OR operation for boolean template parameters.
 *
 * This struct computes the logical OR of two boolean template parameters `A`
 * and `B` at compile time and exposes the result as a static constexpr boolean
 * member `value`.
 *
 * @tparam A First boolean value.
 * @tparam B Second boolean value.
 *
 * @note Useful for template metaprogramming where compile-time boolean logic is
 * required.
 */
template <bool A, bool B> struct LogicalOr {
  static constexpr bool value = A || B;
};

/* SparseAvailable check empty */

/**
 * @brief Template struct to compute the logical OR of multiple boolean values
 * at compile time.
 *
 * This variadic template struct takes a parameter pack of boolean values and
 * can be specialized to evaluate whether at least one of the provided values is
 * true.
 *
 * @tparam Values Variadic list of boolean values to be logically OR'ed
 * together.
 */
template <bool... Values> struct LogicalOrMultiple;

/**
 * @brief Template specialization for LogicalOrMultiple with a single boolean
 * value.
 *
 * This struct defines a static constexpr boolean member `value` that is set to
 * the template parameter `Value`. It is typically used as a base case in
 * recursive template metaprogramming to compute the logical OR of multiple
 * boolean values at compile time.
 *
 * @tparam Value The boolean value to be represented.
 */
template <bool Value> struct LogicalOrMultiple<Value> {
  static constexpr bool value = Value;
};

/**
 * @brief Computes the logical OR of multiple boolean template parameters.
 *
 * This template recursively evaluates the logical OR operation across all
 * provided boolean template arguments. It uses the LogicalOr template to
 * combine the first argument with the result of recursively applying
 * LogicalOrMultiple to the rest.
 *
 * @tparam First The first boolean value in the parameter pack.
 * @tparam Rest The remaining boolean values in the parameter pack.
 *
 * @note The base case for this recursion should be defined elsewhere to
 * terminate the recursion.
 */
template <bool First, bool... Rest> struct LogicalOrMultiple<First, Rest...> {
  static constexpr bool value =
      LogicalOr<First, LogicalOrMultiple<Rest...>::value>::value;
};

/**
 * @brief Template struct to check if any of the sparse columns are available.
 *
 * This struct is used to determine if at least one column in a sparse matrix is
 * available (i.e., has non-zero entries). It is specialized for both
 * ColumnAvailable and SparseAvailable types.
 *
 * @tparam SparseAvailable A type representing the availability of sparse
 * columns.
 */
template <typename SparseAvailable> struct CheckSparseAvailableEmpty;

/**
 * @brief Specialization of CheckSparseAvailableEmpty for ColumnAvailable.
 *
 * This template struct checks if any of the boolean template parameters in
 * ColumnAvailable are true. It uses TemplatesOperation::LogicalOrMultiple to
 * compute the logical OR of all Values.
 *
 * @tparam Values Variadic boolean template parameters representing column
 * availability.
 *
 * The static constexpr bool 'value' will be true if at least one of the Values
 * is true, indicating that at least one column is available (not empty).
 */
template <bool... Values>
struct CheckSparseAvailableEmpty<ColumnAvailable<Values...>> {
  static constexpr bool value =
      TemplatesOperation::LogicalOrMultiple<Values...>::value;
};

/**
 * @brief Trait to check if any of the provided columns are considered "sparse
 * available and empty".
 *
 * This specialization of CheckSparseAvailableEmpty for SparseAvailable types
 * recursively checks each column type in the parameter pack. It uses
 * TemplatesOperation::LogicalOrMultiple to compute the logical OR of the
 * CheckSparseAvailableEmpty value for each column.
 *
 * @tparam Columns Variadic template parameter pack representing column types.
 *
 * @note The resulting static constexpr bool 'value' will be true if at least
 * one column is considered sparse available and empty, false otherwise.
 */
template <typename... Columns>
struct CheckSparseAvailableEmpty<SparseAvailable<Columns...>> {
  static constexpr bool value = TemplatesOperation::LogicalOrMultiple<
      CheckSparseAvailableEmpty<Columns>::value...>::value;
};

/* Create Dense Available */

/**
 * @brief Metafunction to generate a parameter pack of boolean flags, all set to
 * false.
 *
 * This template recursively constructs a type alias containing a parameter pack
 * of N boolean values, all initialized to false. It is typically used in
 * template metaprogramming to generate compile-time sequences of boolean flags.
 *
 * @tparam N The number of boolean flags to generate.
 * @tparam Flags The accumulated boolean flags (used internally during
 * recursion).
 */
template <std::size_t N, bool... Flags> struct GenerateFalseFlags {
  using type = typename GenerateFalseFlags<N - 1, false, Flags...>::type;
};

/**
 * @brief Template specialization for GenerateFalseFlags when the first template
 * parameter is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * ColumnAvailable<Flags...>, effectively propagating the boolean parameter pack
 * 'Flags...' to the ColumnAvailable template.
 *
 * @tparam Flags Variadic boolean template parameters representing flag values.
 */
template <bool... Flags> struct GenerateFalseFlags<0, Flags...> {
  using type = ColumnAvailable<Flags...>;
};

/**
 * @brief Alias template to generate a type representing a sequence of 'false'
 * flags for columns.
 *
 * This alias uses the GenerateFalseFlags template to create a type with N
 * 'false' values, typically used to indicate the availability status of columns
 * in a matrix (all unavailable).
 *
 * @tparam N The number of columns (flags) to generate.
 * @see GenerateFalseFlags
 */
template <std::size_t N>
using GenerateFalseColumnAvailable = typename GenerateFalseFlags<N>::type;

/**
 * @brief Metafunction to generate a parameter pack of boolean template
 * arguments, prepending 'true' N times to the pack.
 *
 * @tparam N The number of 'true' flags to prepend.
 * @tparam Flags The existing boolean flags in the parameter pack.
 *
 * This primary template recursively prepends 'true' to the Flags parameter pack
 * N times. The recursion is expected to be terminated by a specialization for N
 * == 0.
 */
template <std::size_t N, bool... Flags> struct GenerateTrueFlags {
  using type = typename GenerateTrueFlags<N - 1, true, Flags...>::type;
};

/**
 * @brief Specialization of the GenerateTrueFlags template for the case when the
 * first template parameter is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * ColumnAvailable<Flags...>, effectively forwarding the remaining boolean
 * template parameters (Flags...) to the ColumnAvailable template.
 *
 * @tparam Flags Variadic boolean template parameters representing flag values.
 */
template <bool... Flags> struct GenerateTrueFlags<0, Flags...> {
  using type = ColumnAvailable<Flags...>;
};

/**
 * @brief Alias template to generate a type representing a sequence of 'true'
 * flags for a given size.
 *
 * This alias uses the GenerateTrueFlags metafunction to create a type
 * (typically a type list or similar) containing N 'true' values, which can be
 * used for compile-time flag management or template metaprogramming.
 *
 * @tparam N The number of 'true' flags to generate.
 *
 * @see GenerateTrueFlags
 */
template <std::size_t N>
using GenerateTrueColumnAvailable = typename GenerateTrueFlags<N>::type;

/**
 * @brief Recursively constructs a type by repeating the ColumnAvailable type M
 * times in a parameter pack.
 *
 * @tparam M The number of times to repeat the ColumnAvailable type.
 * @tparam ColumnAvailable The type to be repeated.
 * @tparam Columns The parameter pack accumulating the repeated types.
 *
 * This template recursively instantiates itself, decrementing M each time,
 * and prepending ColumnAvailable to the Columns parameter pack, until a base
 * case is reached.
 */
template <std::size_t M, typename ColumnAvailable, typename... Columns>
struct RepeatColumnAvailable {
  using type =
      typename RepeatColumnAvailable<M - 1, ColumnAvailable, ColumnAvailable,
                                     Columns...>::type;
};

/**
 * @brief Specialization of RepeatColumnAvailable for the case when the first
 * template parameter is 0.
 *
 * This specialization defines a type alias `type` that is set to
 * `TemplatesOperation::SparseAvailableColumns<Columns...>`, effectively
 * collecting the remaining columns into a sparse column representation.
 *
 * @tparam ColumnAvailable Unused in this specialization, but required for
 * template matching.
 * @tparam Columns Variadic template parameter pack representing the remaining
 * columns.
 */
template <typename ColumnAvailable, typename... Columns>
struct RepeatColumnAvailable<0, ColumnAvailable, Columns...> {
  using type = TemplatesOperation::SparseAvailableColumns<Columns...>;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to determine the availability of dense matrix columns.
 *
 * This alias uses TemplatesOperation utilities to generate a compile-time
 * type indicating which columns are available in a dense matrix of size MxN.
 *
 * @tparam M Number of rows in the matrix.
 * @tparam N Number of columns in the matrix.
 *
 * The resulting type is computed by repeating the column availability
 * (as generated by GenerateTrueColumnAvailable<N>) for M rows using
 * RepeatColumnAvailable.
 */
template <std::size_t M, std::size_t N>
using DenseAvailable = typename TemplatesOperation::RepeatColumnAvailable<
    M, TemplatesOperation::GenerateTrueColumnAvailable<N>>::type;

/**
 * @brief Alias template for creating a dense matrix availability mask with all
 * entries set to false.
 *
 * This alias uses TemplatesOperation utilities to generate a column
 * availability type for a matrix of size M x N, where all columns are marked as
 * unavailable (false).
 *
 * @tparam M Number of rows in the matrix.
 * @tparam N Number of columns in the matrix.
 *
 * @see TemplatesOperation::RepeatColumnAvailable
 * @see TemplatesOperation::GenerateFalseColumnAvailable
 */
template <std::size_t M, std::size_t N>
using DenseAvailableEmpty = typename TemplatesOperation::RepeatColumnAvailable<
    M, TemplatesOperation::GenerateFalseColumnAvailable<N>>::type;

/* Create Diag Available */

namespace TemplatesOperation {

/**
 * @brief Metafunction to generate a parameter pack of boolean flags with a
 * single true value at a specified index.
 *
 * This template recursively constructs a boolean parameter pack of length N,
 * where only the element at position Index is true, and all other elements are
 * false. The generated type can be used for compile-time selection or
 * specialization based on index.
 *
 * @tparam N The total number of boolean flags to generate.
 * @tparam Index The index at which the flag should be set to true.
 * @tparam Flags The accumulated boolean flags during recursion.
 */
template <std::size_t N, std::size_t Index, bool... Flags>
struct GenerateIndexedTrueFlags {
  using type = typename GenerateIndexedTrueFlags<
      N - 1, Index, (N - 1 == Index ? true : false), Flags...>::type;
};

/**
 * @brief Specialization of the GenerateIndexedTrueFlags template for the case
 * when the first template parameter is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * ColumnAvailable<Flags...>, effectively forwarding the boolean parameter pack
 * Flags to the ColumnAvailable template.
 *
 * @tparam Index The index value (unused in this specialization).
 * @tparam Flags A parameter pack of boolean values representing column
 * availability.
 */
template <std::size_t Index, bool... Flags>
struct GenerateIndexedTrueFlags<0, Index, Flags...> {
  using type = ColumnAvailable<Flags...>;
};

/**
 * @brief Alias template to generate a type representing a sequence of boolean
 * flags, where the flag at position Index is set to true and all others are
 * false.
 *
 * @tparam N     The total number of flags (columns).
 * @tparam Index The index of the flag to be set to true.
 *
 * This alias uses GenerateIndexedTrueFlags to produce a type (typically a
 * std::integer_sequence or similar) with N elements, where only the element at
 * position Index is true.
 */
template <std::size_t N, std::size_t Index>
using GenerateIndexedTrueColumnAvailable =
    typename GenerateIndexedTrueFlags<N, Index>::type;

/**
 * @brief Recursively constructs a type list by repeating a column availability
 * type for a matrix.
 *
 * This template recursively builds a type list representing the availability of
 * columns in a matrix, by prepending a new column availability type at each
 * recursion step. The recursion continues until the base case (not shown here)
 * is reached.
 *
 * @tparam M The number of columns remaining to process.
 * @tparam N The number of rows in the matrix.
 * @tparam Index The current index for which the column availability is being
 * generated.
 * @tparam ColumnAvailable The type representing the availability of the current
 * column.
 * @tparam Columns The accumulated list of column availability types.
 *
 * The resulting type is accessible via the nested ::type member.
 */
template <std::size_t M, std::size_t N, std::size_t Index,
          typename ColumnAvailable, typename... Columns>
struct IndexedRepeatColumnAvailable {
  using type = typename IndexedRepeatColumnAvailable<
      (M - 1), N, (Index - 1),
      GenerateIndexedTrueColumnAvailable<N, (Index - 1)>, ColumnAvailable,
      Columns...>::type;
};

/**
 * @brief Specialization of IndexedRepeatColumnAvailable for the case when M is
 * 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * TemplatesOperation::SparseAvailableColumns<Columns...>, effectively
 * collecting the repeated column availability types into a sparse column
 * representation.
 *
 * @tparam N The number of rows in the matrix (unused in this specialization).
 * @tparam Index The current index (unused in this specialization).
 * @tparam ColumnAvailable The type representing the availability of the current
 * column (unused in this specialization).
 * @tparam Columns Variadic template parameter pack representing the accumulated
 * column availability types.
 */
template <std::size_t N, std::size_t Index, typename ColumnAvailable,
          typename... Columns>
struct IndexedRepeatColumnAvailable<0, N, Index, ColumnAvailable, Columns...> {
  using type = TemplatesOperation::SparseAvailableColumns<Columns...>;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to create a diagonal availability mask for a square
 * matrix of size MxM.
 *
 * This alias uses the TemplatesOperation::IndexedRepeatColumnAvailable to
 * generate a type representing the availability of columns in a diagonal
 * matrix, where only the diagonal elements are available (true).
 *
 * @tparam M The size of the square matrix (number of rows and columns).
 *
 * The resulting type is a SparseAvailableColumns type with M columns, where
 * each column has only one true value at its diagonal position.
 */
template <std::size_t M>
using DiagAvailable = typename TemplatesOperation::IndexedRepeatColumnAvailable<
    M, M, (M - 1),
    TemplatesOperation::GenerateIndexedTrueColumnAvailable<M, (M - 1)>>::type;

/* Create Sparse Available from Indices and Pointers */

namespace TemplatesOperation {

/**
 * @brief Metafunction to generate a compile-time list of boolean flags by
 * performing a logical OR operation on two ColumnAvailable types.
 *
 * This template recursively combines the boolean flags from two ColumnAvailable
 * types, producing a new ColumnAvailable type that reflects the logical OR of
 * the flags at each index.
 *
 * @tparam ColumnAvailable_A The first ColumnAvailable type.
 * @tparam ColumnAvailable_B The second ColumnAvailable type.
 * @tparam N The number of columns (size of the lists).
 * @tparam Flags The accumulated boolean flags during recursion.
 */
template <typename ColumnAvailable_A, typename ColumnAvailable_B, std::size_t N,
          bool... Flags>
struct GenerateORTrueFlagsLoop {
  using type = typename GenerateORTrueFlagsLoop<
      ColumnAvailable_A, ColumnAvailable_B, N - 1,
      (ColumnAvailable_A::list[N - 1] | ColumnAvailable_B::list[N - 1]),
      Flags...>::type;
};

/**
 * @brief Specialization of GenerateORTrueFlagsLoop for the case when N is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * ColumnAvailable<Flags...>, effectively collecting the accumulated boolean
 * flags into a ColumnAvailable type.
 *
 * @tparam ColumnAvailable_A The first ColumnAvailable type (unused in this
 * specialization).
 * @tparam ColumnAvailable_B The second ColumnAvailable type (unused in this
 * specialization).
 * @tparam Flags The accumulated boolean flags representing the logical OR of
 * the two ColumnAvailable types.
 */
template <typename ColumnAvailable_A, typename ColumnAvailable_B, bool... Flags>
struct GenerateORTrueFlagsLoop<ColumnAvailable_A, ColumnAvailable_B, 0,
                               Flags...> {
  using type = ColumnAvailable<Flags...>;
};

/**
 * @brief Alias template to generate a ColumnAvailable type representing the
 * logical OR of two ColumnAvailable types.
 *
 * This alias uses the GenerateORTrueFlagsLoop to compute the logical OR of the
 * flags in ColumnAvailable_A and ColumnAvailable_B, producing a new
 * ColumnAvailable type that reflects the combined availability of both columns.
 *
 * @tparam ColumnAvailable_A The first ColumnAvailable type.
 * @tparam ColumnAvailable_B The second ColumnAvailable type.
 *
 * The resulting type is a ColumnAvailable type with the logical OR of the
 * flags from both input ColumnAvailable types.
 */
template <typename ColumnAvailable_A, typename ColumnAvailable_B>
using GenerateORTrueFlagsColumnAvailable =
    typename GenerateORTrueFlagsLoop<ColumnAvailable_A, ColumnAvailable_B,
                                     ColumnAvailable_A::size>::type;

/**
 * @brief Metafunction to create a sparse pointers row loop for generating
 * ColumnAvailable types based on row indices and pointers.
 *
 * This template recursively constructs a ColumnAvailable type for each row,
 * based on the provided row indices and pointers. It generates a type that
 * indicates which columns are available (true) or not (false) for each row.
 *
 * @tparam RowIndices A type representing the row indices.
 * @tparam N The number of columns in the matrix.
 * @tparam RowIndicesIndex The current index in the RowIndices list being
 * processed.
 * @tparam RowEndCount The number of rows remaining to process.
 */
template <typename RowIndices, std::size_t N, std::size_t RowIndicesIndex,
          std::size_t RowEndCount>
struct CreateSparsePointersRowLoop {
  using type = typename GenerateORTrueFlagsLoop<
      GenerateIndexedTrueColumnAvailable<N, RowIndices::list[RowIndicesIndex]>,
      typename CreateSparsePointersRowLoop<RowIndices, N, (RowIndicesIndex + 1),
                                           (RowEndCount - 1)>::type,
      N>::type;
};

/**
 * @brief Specialization of CreateSparsePointersRowLoop for the case when
 * RowEndCount is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * GenerateFalseColumnAvailable<N>, effectively creating a ColumnAvailable type
 * with all columns marked as unavailable (false).
 *
 * @tparam RowIndices A type representing the row indices (unused in this
 * specialization).
 * @tparam N The number of columns in the matrix.
 * @tparam RowIndicesIndex The current index in the RowIndices list (unused in
 * this specialization).
 */
template <typename RowIndices, std::size_t N, std::size_t RowIndicesIndex>
struct CreateSparsePointersRowLoop<RowIndices, N, RowIndicesIndex, 0> {
  using type = GenerateFalseColumnAvailable<N>;
};

/**
 * @brief Metafunction to create a loop for generating sparse pointers based on
 * row indices and pointers.
 *
 * This template recursively constructs a SparseAvailableColumns type by
 * iterating over the row pointers and generating ColumnAvailable types for each
 * row based on the row indices and pointers.
 *
 * @tparam N The number of columns in the matrix.
 * @tparam RowIndices A type representing the row indices.
 * @tparam RowPointers A type representing the row pointers.
 * @tparam EndCount The number of rows remaining to process.
 * @tparam ConsecutiveIndex The current index in the RowPointers list being
 * processed.
 * @tparam Columns The accumulated ColumnAvailable types during recursion.
 */
template <std::size_t N, typename RowIndices, typename RowPointers,
          std::size_t EndCount, std::size_t ConsecutiveIndex,
          typename... Columns>
struct CreateSparsePointersLoop {
  using type = typename CreateSparsePointersLoop<
      N, RowIndices, RowPointers, (EndCount - 1),
      RowPointers::list[RowPointers::size - EndCount], Columns...,
      typename CreateSparsePointersRowLoop<
          RowIndices, N, RowPointers::list[RowPointers::size - EndCount - 1],
          (RowPointers::list[RowPointers::size - EndCount] -
           RowPointers::list[RowPointers::size - EndCount - 1])>::type>::type;
};

/**
 * @brief Specialization of CreateSparsePointersLoop for the case when EndCount
 * is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailableColumns<Columns...>, effectively collecting the accumulated
 * ColumnAvailable types into a sparse column representation.
 *
 * @tparam N The number of columns in the matrix (unused in this
 * specialization).
 * @tparam RowIndices A type representing the row indices (unused in this
 * specialization).
 * @tparam RowPointers A type representing the row pointers (unused in this
 * specialization).
 * @tparam ConsecutiveIndex The current index in the RowPointers list (unused in
 * this specialization).
 * @tparam Columns Variadic template parameter pack representing the accumulated
 * ColumnAvailable types.
 */
template <std::size_t N, typename RowIndices, typename RowPointers,
          std::size_t ConsecutiveIndex, typename... Columns>
struct CreateSparsePointersLoop<N, RowIndices, RowPointers, 0, ConsecutiveIndex,
                                Columns...> {
  using type = SparseAvailableColumns<Columns...>;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to create a sparse matrix availability type from row
 * indices and row pointers.
 *
 * This alias uses the TemplatesOperation::CreateSparsePointersLoop to generate
 * a SparseAvailableColumns type based on the provided row indices and row
 * pointers. It is typically used to represent the availability of columns in a
 * sparse matrix, where each column's availability is determined by the row
 * indices and pointers.
 *
 * @tparam N The number of columns in the matrix.
 * @tparam RowIndices A type representing the row indices.
 * @tparam RowPointers A type representing the row pointers.
 *
 * The resulting type is a SparseAvailableColumns type with the availability of
 * each column determined by the provided row indices and pointers.
 */
template <std::size_t N, typename RowIndices, typename RowPointers>
using CreateSparseAvailableFromIndicesAndPointers =
    typename TemplatesOperation::CreateSparsePointersLoop<
        N, RowIndices, RowPointers, (RowPointers::size - 1), 0>::type;

namespace TemplatesOperation {

/* Create Sparse Matrix from Dense Matrix */

/**
 * @brief A template struct that represents a compile-time list of indices.
 *
 * This struct provides a static constexpr array `list` containing the indices
 * passed as template parameters, and a static constexpr `size` indicating the
 * number of indices.
 *
 * @tparam Seq Variadic template parameters representing the list of indices.
 *
 * Example usage:
 * @code
 * using MyIndices = IndexSequence<0, 1, 2>;
 * // MyIndices::size == 3
 * // MyIndices::list == {0, 1, 2}
 * @endcode
 */
template <std::size_t... Seq> struct IndexSequence {
  static constexpr std::size_t size = sizeof...(Seq);
  static constexpr std::size_t list[size] = {Seq...};
};

/**
 * @brief A template struct that represents an invalid sequence of indices.
 *
 * This struct provides a static constexpr array `list` containing the indices
 * passed as template parameters, and a static constexpr `size` indicating the
 * number of indices. It is used to represent an invalid or empty sequence.
 *
 * @tparam Seq Variadic template parameters representing the list of indices.
 */
template <std::size_t... Seq> struct InvalidSequence {
  static constexpr std::size_t size = sizeof...(Seq);
  static constexpr std::size_t list[size] = {Seq...};
};

/**
 * @brief A template struct to create a compile-time index sequence of size N.
 *
 * This struct recursively generates an IndexSequence of size N, where each
 * index is filled with the corresponding value from 0 to N-1.
 *
 * @tparam N The size of the index sequence to be generated.
 * @tparam Seq The accumulated indices during recursion.
 */
template <std::size_t N, std::size_t... Seq>
struct MakeIndexSequence : MakeIndexSequence<N - 1, N - 1, Seq...> {};

/**
 * @brief Specialization of MakeIndexSequence for the base case when N is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * IndexSequence<Seq...>, effectively collecting the accumulated indices into an
 * IndexSequence.
 *
 * @tparam Seq The accumulated indices during recursion.
 */
template <std::size_t... Seq> struct MakeIndexSequence<0, Seq...> {
  using type = IndexSequence<Seq...>;
};

/**
 * @brief A template struct to create a compile-time index sequence of size N.
 *
 * This struct provides a type alias 'type' that is set to the result of
 * MakeIndexSequence<N>::type, effectively generating an IndexSequence of size
 * N.
 *
 * @tparam N The size of the index sequence to be generated.
 */
template <std::size_t N> struct IntegerSequenceList {
  using type = typename MakeIndexSequence<N>::type;
};

/**
 * @brief A template struct to create a compile-time list of row indices for a
 * matrix with N columns.
 *
 * This struct provides a type alias 'type' that is set to IndexSequence<0, 1,
 * ..., N-1>, effectively generating a list of row indices for a matrix with N
 * columns.
 *
 * @tparam N The number of columns in the matrix.
 */
template <std::size_t N>
using MatrixRowNumbers = typename IntegerSequenceList<N>::type;

/**
 * @brief A template struct to concatenate two index sequences or invalid
 * sequences.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * two IndexSequence or InvalidSequence types. If one of the sequences is an
 * InvalidSequence, it will return the other sequence.
 *
 * @tparam Seq1 The first index sequence.
 * @tparam Seq2 The second index sequence.
 */
template <typename Seq1, typename Seq2> struct Concatenate;

/**
 * @brief Specialization of Concatenate for two IndexSequence types.
 *
 * This specialization defines a type alias 'type' that is set to
 * IndexSequence<Seq1..., Seq2...>, effectively concatenating the two index
 * sequences.
 *
 * @tparam Seq1 The first index sequence.
 * @tparam Seq2 The second index sequence.
 */
template <std::size_t... Seq1, std::size_t... Seq2>
struct Concatenate<IndexSequence<Seq1...>, IndexSequence<Seq2...>> {
  using type = IndexSequence<Seq1..., Seq2...>;
};

/**
 * @brief Specialization of Concatenate for an IndexSequence and an
 * InvalidSequence.
 *
 * This specialization defines a type alias 'type' that is set to
 * IndexSequence<Seq1...>, effectively returning the first sequence when the
 * second sequence is invalid.
 *
 * @tparam Seq1 The first index sequence.
 * @tparam Seq2 The second invalid sequence.
 */
template <std::size_t... Seq1, std::size_t... Seq2>
struct Concatenate<IndexSequence<Seq1...>, InvalidSequence<Seq2...>> {
  using type = IndexSequence<Seq1...>;
};

/**
 * @brief Specialization of Concatenate for an InvalidSequence and an
 * IndexSequence.
 *
 * This specialization defines a type alias 'type' that is set to
 * IndexSequence<Seq2...>, effectively returning the second sequence when the
 * first sequence is invalid.
 *
 * @tparam Seq1 The first invalid sequence.
 * @tparam Seq2 The second index sequence.
 */
template <std::size_t... Seq1, std::size_t... Seq2>
struct Concatenate<InvalidSequence<Seq1...>, IndexSequence<Seq2...>> {
  using type = IndexSequence<Seq2...>;
};

/* Create Dense Matrix Row Indices and Pointers */

/**
 * @brief Concatenates two index sequences representing matrix row numbers.
 *
 * This alias template uses the Concatenate metafunction to combine two index
 * sequences, resulting in a new index sequence that contains all indices from
 * both input sequences.
 *
 * @tparam IndexSequence_1 The first index sequence to concatenate.
 * @tparam IndexSequence_2 The second index sequence to concatenate.
 *
 * The resulting type is a concatenated IndexSequence containing all indices
 * from both input sequences.
 */
template <typename IndexSequence_1, typename IndexSequence_2>
using ConcatenateMatrixRowNumbers =
    typename Concatenate<IndexSequence_1, IndexSequence_2>::type;

/**
 * @brief Recursively concatenates a given index sequence M times.
 *
 * This template recursively constructs a new index sequence by repeating the
 * provided MatrixRowNumbers type M times. It is used to generate a sequence of
 * row indices for a matrix with M rows and N columns.
 *
 * @tparam M The number of times to repeat the index sequence.
 * @tparam MatrixRowNumbers The index sequence to be repeated.
 */
template <std::size_t M, typename MatrixRowNumbers>
struct RepeatConcatenateIndexSequence;

/**
 * @brief Specialization of RepeatConcatenateIndexSequence for the case when M
 * is 1.
 *
 * This specialization defines a type alias 'type' that is set to
 * MatrixRowNumbers, effectively returning the original index sequence when M is
 * 1.
 *
 * @tparam MatrixRowNumbers The index sequence to be returned.
 */
template <typename MatrixRowNumbers>
struct RepeatConcatenateIndexSequence<1, MatrixRowNumbers> {
  using type = MatrixRowNumbers;
};

/**
 * @brief Specialization of RepeatConcatenateIndexSequence for the case when M
 * is greater than 1.
 *
 * This specialization defines a type alias 'type' that is set to
 * ConcatenateMatrixRowNumbers, effectively concatenating the MatrixRowNumbers
 * with the result of recursively calling RepeatConcatenateIndexSequence with M
 * decremented by 1.
 *
 * @tparam M The number of times to repeat the index sequence (M > 1).
 * @tparam MatrixRowNumbers The index sequence to be repeated.
 */
template <std::size_t M, typename MatrixRowNumbers>
struct RepeatConcatenateIndexSequence {
  using type = ConcatenateMatrixRowNumbers<
      MatrixRowNumbers,
      typename RepeatConcatenateIndexSequence<M - 1, MatrixRowNumbers>::type>;
};

/**
 * @brief A template struct to convert an IndexSequence into a RowIndices type.
 *
 * This struct provides a type alias 'type' that is set to RowIndices<Seq...>,
 * effectively converting the IndexSequence into a RowIndices type.
 *
 * @tparam Seq The index sequence to be converted.
 */
template <typename Seq> struct ToRowIndices;

/**
 * @brief Specialization of ToRowIndices for IndexSequence types.
 *
 * This specialization defines a type alias 'type' that is set to
 * RowIndices<Seq...>, effectively converting the IndexSequence into a
 * RowIndices type.
 *
 * @tparam Seq The index sequence to be converted.
 */
template <std::size_t... Seq> struct ToRowIndices<IndexSequence<Seq...>> {
  using type = RowIndices<Seq...>;
};

/**
 * @brief A template struct to generate a sequence of row numbers for a matrix
 * with M rows and N columns.
 *
 * This struct provides a type alias 'type' that is set to a repeated
 * concatenation of MatrixRowNumbers<N> M times, effectively generating a list
 * of row indices for the matrix.
 *
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 */
template <std::size_t M, std::size_t N>
using MatrixColumnRowNumbers =
    typename RepeatConcatenateIndexSequence<M, MatrixRowNumbers<N>>::type;

} // namespace TemplatesOperation

/**
 * @brief Alias template to create a list of row indices for a dense matrix of
 * size MxN.
 *
 * This alias uses the TemplatesOperation::MatrixColumnRowNumbers to generate
 * a type representing the row indices for a dense matrix with M rows and N
 * columns.
 *
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 *
 * The resulting type is an IndexSequence containing the row indices for the
 * dense matrix.
 */
template <std::size_t M, std::size_t N>
using DenseMatrixRowIndices = typename TemplatesOperation::ToRowIndices<
    TemplatesOperation::MatrixColumnRowNumbers<M, N>>::type;

namespace TemplatesOperation {

/**
 * @brief A template struct to create a list of pointers for a dense matrix of
 * size MxN.
 *
 * This struct recursively generates a list of pointers for each row in the
 * matrix, where each pointer points to the start of a row in the matrix.
 *
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 * @tparam Seq The accumulated pointers during recursion.
 *
 * The resulting type is an IndexSequence containing the pointers for each row
 * in the dense matrix.
 */
template <std::size_t M, std::size_t N, std::size_t... Seq>
struct MakePointerList : MakePointerList<M - 1, N, (M * N), Seq...> {};

/**
 * @brief Specialization of MakePointerList for the case when M is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * IndexSequence<0, Seq...>, effectively collecting the accumulated pointers
 * into an IndexSequence.
 *
 * @tparam N The number of columns in the matrix (unused in this
 * specialization).
 * @tparam Seq The accumulated pointers during recursion.
 */
template <std::size_t N, std::size_t... Seq>
struct MakePointerList<0, N, Seq...> {
  using type = IndexSequence<0, Seq...>;
};

/**
 * @brief A template struct to create a list of pointers for a dense matrix of
 * size MxN.
 *
 * This struct provides a type alias 'type' that is set to the result of
 * MakePointerList<M, N>::type, effectively generating a list of pointers for
 * each row in the dense matrix.
 *
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 */
template <std::size_t M, std::size_t N> struct MatrixDensePointerList {
  using type = typename MakePointerList<M, N>::type;
};

/**
 * @brief A template struct to create a list of row pointers for a dense matrix
 * of size MxN.
 *
 * This struct provides a type alias 'type' that is set to RowPointers<Seq...>,
 * effectively converting the IndexSequence generated by MakePointerList into a
 * RowPointers type.
 *
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 */
template <std::size_t M, std::size_t N>
using MatrixColumnRowPointers = typename MatrixDensePointerList<M, N>::type;

/**
 * @brief A template struct to convert an IndexSequence into a RowPointers type.
 *
 * This struct provides a type alias 'type' that is set to RowPointers<Seq...>,
 * effectively converting the IndexSequence into a RowPointers type.
 *
 * @tparam Seq The index sequence to be converted.
 */
template <typename Seq> struct ToRowPointers;

/**
 * @brief Specialization of ToRowPointers for IndexSequence types.
 *
 * This specialization defines a type alias 'type' that is set to
 * RowPointers<Seq...>, effectively converting the IndexSequence into a
 * RowPointers type.
 *
 * @tparam Seq The index sequence to be converted.
 */
template <std::size_t... Seq>
struct ToRowPointers<TemplatesOperation::IndexSequence<Seq...>> {
  using type = RowPointers<Seq...>;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to create a list of row pointers for a dense matrix of
 * size MxN.
 *
 * This alias uses the TemplatesOperation::MatrixColumnRowPointers to generate
 * a type representing the row pointers for a dense matrix with M rows and N
 * columns.
 *
 * @tparam M The number of columns in the matrix.
 * @tparam N The number of rows in the matrix.
 *
 * The resulting type is a RowPointers type containing the row pointers for the
 * dense matrix.
 */
template <std::size_t M, std::size_t N>
using DenseMatrixRowPointers = typename TemplatesOperation::ToRowIndices<
    TemplatesOperation::MatrixColumnRowPointers<M, N>>::type;

/* Concatenate ColumnAvailable */

namespace TemplatesOperation {

/**
 * @brief A template struct to concatenate two ColumnAvailable types into a new
 * ColumnAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * two ColumnAvailable types, effectively combining their boolean flags into a
 * single ColumnAvailable type.
 *
 * @tparam ColumnAvailable_A The first ColumnAvailable type.
 * @tparam ColumnAvailable_B The second ColumnAvailable type.
 *
 * The resulting type is a ColumnAvailable type that contains the combined
 * boolean flags from both input ColumnAvailable types.
 */
template <typename Column1, typename Column2>
struct ConcatenateColumnAvailableLists;

/**
 * @brief Specialization of ConcatenateColumnAvailableLists for two
 * ColumnAvailable types.
 *
 * This specialization defines a type alias 'type' that is set to
 * ColumnAvailable<Flags1..., Flags2...>, effectively concatenating the boolean
 * flags from both ColumnAvailable types into a new ColumnAvailable type.
 *
 * @tparam Flags1 The boolean flags from the first ColumnAvailable type.
 * @tparam Flags2 The boolean flags from the second ColumnAvailable type.
 */
template <bool... Flags1, bool... Flags2>
struct ConcatenateColumnAvailableLists<ColumnAvailable<Flags1...>,
                                       ColumnAvailable<Flags2...>> {
  using type = ColumnAvailable<Flags1..., Flags2...>;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to concatenate two ColumnAvailable types into a new
 * ColumnAvailable type.
 *
 * This alias uses the TemplatesOperation::ConcatenateColumnAvailableLists to
 * combine two ColumnAvailable types, effectively merging their boolean flags
 * into a single ColumnAvailable type.
 *
 * @tparam ColumnAvailable_A The first ColumnAvailable type.
 * @tparam ColumnAvailable_B The second ColumnAvailable type.
 *
 * The resulting type is a ColumnAvailable type that contains the combined
 * boolean flags from both input ColumnAvailable types.
 */
template <typename ColumnAvailable_A, typename ColumnAvailable_B>
using ConcatenateColumnAvailable =
    typename TemplatesOperation::ConcatenateColumnAvailableLists<
        ColumnAvailable_A, ColumnAvailable_B>::type;

/* Get ColumnAvailable from SparseAvailable */

namespace TemplatesOperation {

/**
 * @brief A template struct to get the ColumnAvailable type at a specific index
 * from a SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is set to the ColumnAvailable
 * type at index N in the SparseAvailable type, effectively extracting the
 * availability information for that specific column.
 *
 * @tparam N The index of the column to retrieve.
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 */
template <std::size_t N, typename SparseAvailable> struct GetColumnAvailable;

/**
 * @brief Specialization of GetColumnAvailable for SparseAvailable types.
 *
 * This specialization defines a type alias 'type' that is set to the
 * std::tuple_element<N, std::tuple<Columns...>>::type, effectively extracting
 * the ColumnAvailable type at index N from the SparseAvailable type.
 *
 * @tparam N The index of the column to retrieve.
 * @tparam Columns The variadic template parameter pack representing the columns
 * in the SparseAvailable type.
 */
template <std::size_t N, typename... Columns>
struct GetColumnAvailable<N, SparseAvailable<Columns...>> {
  using type = typename std::tuple_element<N, std::tuple<Columns...>>::type;
};

} // namespace TemplatesOperation

/* Concatenate SparseAvailable vertically */

/**
 * @brief A template struct to concatenate two SparseAvailable types vertically.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * two SparseAvailable types, effectively merging their columns into a new
 * SparseAvailable type.
 *
 * @tparam SparseAvailable1 The first SparseAvailable type.
 * @tparam SparseAvailable2 The second SparseAvailable type.
 *
 * The resulting type is a SparseAvailable type that contains all columns from
 * both input SparseAvailable types.
 */
template <typename SparseAvailable1, typename SparseAvailable2>
struct ConcatenateSparseAvailable;

/**
 * @brief Specialization of ConcatenateSparseAvailable for two SparseAvailable
 * types.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailableColumns<Columns1..., Columns2...>, effectively concatenating
 * the columns from both SparseAvailable types into a new SparseAvailable type.
 *
 * @tparam Columns1 The variadic template parameter pack representing the
 * columns in the first SparseAvailable type.
 * @tparam Columns2 The variadic template parameter pack representing the
 * columns in the second SparseAvailable type.
 */
template <typename... Columns1, typename... Columns2>
struct ConcatenateSparseAvailable<
    TemplatesOperation::SparseAvailableColumns<Columns1...>,
    TemplatesOperation::SparseAvailableColumns<Columns2...>> {
  using type =
      TemplatesOperation::SparseAvailableColumns<Columns1..., Columns2...>;
};

/**
 * @brief Alias template to concatenate two SparseAvailable types vertically.
 *
 * This alias uses the ConcatenateSparseAvailable to combine two SparseAvailable
 * types, effectively merging their columns into a single SparseAvailable type.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 *
 * The resulting type is a SparseAvailable type that contains all columns from
 * both input SparseAvailable types.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B>
using ConcatenateSparseAvailableVertically =
    typename ConcatenateSparseAvailable<SparseAvailable_A,
                                        SparseAvailable_B>::type;

/* Concatenate SparseAvailable horizontally */

namespace TemplatesOperation {

/**
 * @brief A template struct to concatenate two SparseAvailable types
 * horizontally.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * two SparseAvailable types horizontally, effectively merging their columns
 * into a new SparseAvailableColumns type.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t ColumnCount>
struct ConcatenateSparseAvailableHorizontallyLoop {
  using type = typename ConcatenateSparseAvailable<
      typename ConcatenateSparseAvailableHorizontallyLoop<
          SparseAvailable_A, SparseAvailable_B, (ColumnCount - 1)>::type,
      SparseAvailableColumns<typename ConcatenateColumnAvailableLists<
          typename GetColumnAvailable<ColumnCount, SparseAvailable_A>::type,
          typename GetColumnAvailable<ColumnCount,
                                      SparseAvailable_B>::type>::type>>::type;
};

/**
 * @brief Specialization of ConcatenateSparseAvailableHorizontallyLoop for the
 * base case when ColumnCount is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailableColumns, effectively creating a SparseAvailableColumns type
 * with the combined availability of the first columns from both SparseAvailable
 * types.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type (unused in this
 * specialization).
 * @tparam SparseAvailable_B The second SparseAvailable type (unused in
 * this specialization).
 */
template <typename SparseAvailable_A, typename SparseAvailable_B>
struct ConcatenateSparseAvailableHorizontallyLoop<SparseAvailable_A,
                                                  SparseAvailable_B, 0> {
  using type = SparseAvailableColumns<typename ConcatenateColumnAvailableLists<
      typename GetColumnAvailable<0, SparseAvailable_A>::type,
      typename GetColumnAvailable<0, SparseAvailable_B>::type>::type>;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to concatenate two SparseAvailable types horizontally.
 *
 * This alias uses the
 * TemplatesOperation::ConcatenateSparseAvailableHorizontallyLoop to combine two
 * SparseAvailable types, effectively merging their columns into a single
 * SparseAvailableColumns type.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 *
 * The resulting type is a SparseAvailableColumns type that contains all columns
 * from both input SparseAvailable types, concatenated horizontally.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B>
using ConcatenateSparseAvailableHorizontally =
    typename TemplatesOperation::ConcatenateSparseAvailableHorizontallyLoop<
        SparseAvailable_A, SparseAvailable_B,
        SparseAvailable_A::number_of_columns - 1>::type;

/* Concatenate ColumnAvailable with SparseAvailable  */

namespace TemplatesOperation {

/**
 * @brief A template struct to concatenate a ColumnAvailable type with a
 * SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * a ColumnAvailable type with a SparseAvailable type, effectively merging their
 * columns into a new SparseAvailable type.
 *
 * @tparam Column The ColumnAvailable type to be concatenated.
 * @tparam Sparse The SparseAvailable type to be concatenated.
 */
template <typename...> struct ConcatTuple;

/**
 * @brief Specialization of ConcatTuple for two std::tuple types.
 *
 * This specialization defines a type alias 'type' that is set to
 * std::tuple<Ts1..., Ts2...>, effectively concatenating the elements of two
 * tuples into a new tuple.
 *
 * @tparam Ts1 The variadic template parameter pack representing the first
 * tuple.
 * @tparam Ts2 The variadic template parameter pack representing the second
 * tuple.
 */
template <typename... Ts1, typename... Ts2>
struct ConcatTuple<std::tuple<Ts1...>, std::tuple<Ts2...>> {
  using type = std::tuple<Ts1..., Ts2...>;
};

/**
 * @brief A template struct to concatenate a ColumnAvailable type with a
 * SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * a ColumnAvailable type with a SparseAvailable type, effectively merging their
 * columns into a new SparseAvailable type.
 *
 * @tparam Column The ColumnAvailable type to be concatenated.
 * @tparam Sparse The SparseAvailable type to be concatenated.
 */
template <typename Column, typename Sparse> struct ConcatColumnSparse;

/**
 * @brief Specialization of ConcatColumnSparse for a ColumnAvailable type and a
 * SparseAvailable type.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailableColumns<ConcatTuple<Column, Columns>::type...>, effectively
 * concatenating the ColumnAvailable type with each column in the
 * SparseAvailable type.
 *
 * @tparam Column The ColumnAvailable type to be concatenated.
 * @tparam Columns The variadic template parameter pack representing the columns
 * in the SparseAvailable type.
 */
template <typename Column, typename... Columns>
struct ConcatColumnSparse<Column, SparseAvailable<Columns...>> {
  using type = SparseAvailable<Column, Columns...>;
};

/* Get rest of SparseAvailable */

/**
 * @brief A template struct to get the rest of SparseAvailable starting from a
 * specific column index.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * the ColumnAvailable type at the specified column index with the rest of the
 * SparseAvailable columns.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple
 * columns.
 * @tparam Col_Index The index of the column to start from.
 * @tparam Residual The number of remaining columns to process.
 */
template <typename SparseAvailable_In, std::size_t Col_Index,
          std::size_t Residual>
struct GetRestOfSparseAvailableLoop {
  using type = typename ConcatColumnSparse<
      typename GetColumnAvailable<Col_Index, SparseAvailable_In>::type,
      typename GetRestOfSparseAvailableLoop<SparseAvailable_In, (Col_Index + 1),
                                            (Residual - 1)>::type>::type;
};

/**
 * @brief Specialization of GetRestOfSparseAvailableLoop for the case when
 * Residual is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailable<ColumnAvailable>, effectively creating a SparseAvailable type
 * with the ColumnAvailable type at the specified column index.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple
 * columns.
 * @tparam Col_Index The index of the column to start from.
 */
template <typename SparseAvailable_In, std::size_t Col_Index>
struct GetRestOfSparseAvailableLoop<SparseAvailable_In, Col_Index, 0> {
  using type = SparseAvailable<
      typename GetColumnAvailable<Col_Index, SparseAvailable_In>::type>;
};

/**
 * @brief A template alias to get the rest of SparseAvailable starting from a
 * specific column index.
 *
 * This alias uses the GetRestOfSparseAvailableLoop to generate a
 * SparseAvailable type that contains the ColumnAvailable types starting from
 * the specified column index.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple
 * columns.
 * @tparam Col_Index The index of the column to start from.
 *
 * The resulting type is a SparseAvailable type that contains all columns from
 * the specified index onward.
 */
template <typename SparseAvailable_In, std::size_t Col_Index>
using GetRestOfSparseAvailable = typename GetRestOfSparseAvailableLoop<
    SparseAvailable_In, Col_Index,
    ((SparseAvailable_In::number_of_columns - 1) - Col_Index)>::type;

/**
 * @brief A template struct to check if a SparseAvailable type is empty.
 *
 * This struct provides a static constexpr boolean value 'value' that is
 * true if the SparseAvailable type has no columns, and false otherwise.
 *
 * @tparam SparseAvailable The SparseAvailable type to check.
 */
template <typename SparseAvailable_In, std::size_t Col_Index, bool NotEmpty>
struct AvoidEmptyColumnsSparseAvailableLoop;

/**
 * @brief Specialization of AvoidEmptyColumnsSparseAvailableLoop for the case
 * when the SparseAvailable type is empty.
 *
 * This specialization defines a type alias 'type' that is set to
 * GetRestOfSparseAvailable<SparseAvailable_In, Col_Index>, effectively
 * returning the rest of the SparseAvailable starting from the specified column
 * index when it is not empty.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple
 * columns.
 * @tparam Col_Index The index of the column to start from.
 */
template <typename SparseAvailable_In, std::size_t Col_Index>
struct AvoidEmptyColumnsSparseAvailableLoop<SparseAvailable_In, Col_Index,
                                            true> {
  using type = GetRestOfSparseAvailable<SparseAvailable_In, Col_Index>;
};

/**
 * @brief Specialization of AvoidEmptyColumnsSparseAvailableLoop for the case
 * when the SparseAvailable type is not empty.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * recursively calling AvoidEmptyColumnsSparseAvailableLoop with the next column
 * index, effectively skipping empty columns in the SparseAvailable type.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple
 * columns.
 * @tparam Col_Index The index of the column to start from.
 */
template <typename SparseAvailable_In, std::size_t Col_Index>
struct AvoidEmptyColumnsSparseAvailableLoop<SparseAvailable_In, Col_Index,
                                            false> {
  using type = typename AvoidEmptyColumnsSparseAvailableLoop<
      SparseAvailable_In, (Col_Index + 1),
      CheckSparseAvailableEmpty<typename GetColumnAvailable<
          (Col_Index + 1), SparseAvailable_In>::type>::value>::type;
};

/**
 * @brief A template alias to avoid empty columns in a SparseAvailable type.
 *
 * This alias uses the AvoidEmptyColumnsSparseAvailableLoop to generate a
 * SparseAvailable type that contains only the non-empty columns from the input
 * SparseAvailable type.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple
 * columns.
 *
 * The resulting type is a SparseAvailable type that contains only the columns
 * that are not empty, effectively filtering out any empty columns.
 */
template <typename SparseAvailable_In>
using AvoidEmptyColumnsSparseAvailable =
    typename AvoidEmptyColumnsSparseAvailableLoop<
        SparseAvailable_In, 0,
        CheckSparseAvailableEmpty<TemplatesOperation::SparseAvailableColumns<
            typename GetColumnAvailable<0, SparseAvailable_In>::type>>::value>::
        type;

/* Create Row Indices */

/**
 * @brief A template struct to create a sequence of row indices for a sparse
 * matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating a sequence of row indices for each column in the SparseAvailable
 * type.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 */
template <typename SparseAvailable, std::size_t ColumnElementNumber,
          bool Active, std::size_t RowElementNumber>
struct AssignSparseMatrixRowLoop;

/**
 * @brief Specialization of AssignSparseMatrixRowLoop for the case when the row
 * is active.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * concatenating the current row index with the result of recursively calling
 * AssignSparseMatrixRowLoop with the next row index.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam RowElementNumber The index of the row being processed.
 */
template <typename SparseAvailable, std::size_t ColumnElementNumber,
          std::size_t RowElementNumber>
struct AssignSparseMatrixRowLoop<SparseAvailable, ColumnElementNumber, true,
                                 RowElementNumber> {
  using type = typename Concatenate<
      typename AssignSparseMatrixRowLoop<
          SparseAvailable, ColumnElementNumber,
          SparseAvailable::lists[ColumnElementNumber][RowElementNumber - 1],
          RowElementNumber - 1>::type,
      IndexSequence<RowElementNumber>>::type;
};

/**
 * @brief Specialization of AssignSparseMatrixRowLoop for the case when the row
 * is not active.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * recursively calling AssignSparseMatrixRowLoop with the next row index,
 * effectively skipping the current row.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam RowElementNumber The index of the row being processed.
 */
template <typename SparseAvailable, std::size_t ColumnElementNumber,
          std::size_t RowElementNumber>
struct AssignSparseMatrixRowLoop<SparseAvailable, ColumnElementNumber, false,
                                 RowElementNumber> {
  using type = typename AssignSparseMatrixRowLoop<
      SparseAvailable, ColumnElementNumber,
      SparseAvailable::lists[ColumnElementNumber][RowElementNumber - 1],
      RowElementNumber - 1>::type;
};

/**
 * @brief Specialization of AssignSparseMatrixRowLoop for the case when there
 * are no more rows to process.
 *
 * This specialization defines a type alias 'type' that is set to an
 * InvalidSequence, effectively indicating that there are no valid row indices
 * to return.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 * @tparam ColumnElementNumber The index of the column being processed.
 */
template <typename SparseAvailable, std::size_t ColumnElementNumber>
struct AssignSparseMatrixRowLoop<SparseAvailable, ColumnElementNumber, false,
                                 0> {
  using type = InvalidSequence<0>;
};

/**
 * @brief Specialization of AssignSparseMatrixRowLoop for the case when there
 * are no more rows to process and the row is active.
 *
 * This specialization defines a type alias 'type' that is set to
 * IndexSequence<0>, effectively indicating that there are no valid row indices
 * to return.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 * @tparam ColumnElementNumber The index of the column being processed.
 */
template <typename SparseAvailable, std::size_t ColumnElementNumber>
struct AssignSparseMatrixRowLoop<SparseAvailable, ColumnElementNumber, true,
                                 0> {
  using type = IndexSequence<0>;
};

/**
 * @brief A template struct to assign row indices for each column in a sparse
 * matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating row indices for each column in the SparseAvailable type.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 * @tparam ColumnElementNumber The index of the column being processed.
 */
template <typename SparseAvailable, std::size_t ColumnElementNumber>
struct AssignSparseMatrixColumnLoop {
  using type = typename Concatenate<
      typename AssignSparseMatrixColumnLoop<SparseAvailable,
                                            ColumnElementNumber - 1>::type,
      typename AssignSparseMatrixRowLoop<
          SparseAvailable, ColumnElementNumber,
          SparseAvailable::lists[ColumnElementNumber]
                                [SparseAvailable::column_size - 1],
          (SparseAvailable::column_size - 1)>::type>::type;
};

/**
 * @brief Specialization of AssignSparseMatrixColumnLoop for the case when
 * ColumnElementNumber is 0.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * AssignSparseMatrixRowLoop for the first column, effectively generating row
 * indices for the first column in the SparseAvailable type.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 */
template <typename SparseAvailable>
struct AssignSparseMatrixColumnLoop<SparseAvailable, 0> {
  using type = typename AssignSparseMatrixRowLoop<
      SparseAvailable, 0,
      SparseAvailable::lists[0][SparseAvailable::column_size - 1],
      (SparseAvailable::column_size - 1)>::type;
};

/**
 * @brief A template struct to create a sequence of row indices from a
 * SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * sequence of row indices based on the SparseAvailable type, taking into
 * account whether the SparseAvailable type is empty or not.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 * @tparam NotEmpty A boolean value indicating whether the SparseAvailable type
 * is empty or not.
 */
template <typename SparseAvailable, bool NotEmpty>
struct RowIndicesSequenceFromSparseAvailable;

/**
 * @brief Specialization of RowIndicesSequenceFromSparseAvailable for the case
 * when the SparseAvailable type is not empty.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * AssignSparseMatrixColumnLoop for the last column in the SparseAvailable type,
 * effectively generating a sequence of row indices for the non-empty columns.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 */
template <typename SparseAvailable>
struct RowIndicesSequenceFromSparseAvailable<SparseAvailable, true> {
  using type = typename AssignSparseMatrixColumnLoop<
      AvoidEmptyColumnsSparseAvailable<SparseAvailable>,
      (AvoidEmptyColumnsSparseAvailable<SparseAvailable>::number_of_columns -
       1)>::type;
};

/**
 * @brief Specialization of RowIndicesSequenceFromSparseAvailable for the case
 * when the SparseAvailable type is empty.
 *
 * This specialization defines a type alias 'type' that is set to
 * IndexSequence<0>, effectively indicating that there are no valid row indices
 * to return when the SparseAvailable type is empty.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 */
template <typename SparseAvailable>
struct RowIndicesSequenceFromSparseAvailable<SparseAvailable, false> {
  using type = IndexSequence<0>;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to create a sequence of row indices from a
 * SparseAvailable type.
 *
 * This alias uses the TemplatesOperation::RowIndicesSequenceFromSparseAvailable
 * to generate a type representing the row indices for the SparseAvailable type,
 * taking into account whether it is empty or not.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 *
 * The resulting type is an IndexSequence containing the row indices for the
 * SparseAvailable type.
 */
template <typename SparseAvailable>
using RowIndicesFromSparseAvailable = typename TemplatesOperation::ToRowIndices<
    typename TemplatesOperation::RowIndicesSequenceFromSparseAvailable<
        SparseAvailable, TemplatesOperation::CheckSparseAvailableEmpty<
                             SparseAvailable>::value>::type>::type;

/* Create Row Pointers */

namespace TemplatesOperation {

/**
 * @brief A template struct to count the number of elements in each row of a
 * sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * counting the number of elements in each row for each column in the
 * SparseAvailable type.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 * @tparam ElementCount The current count of elements processed.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam Active A boolean indicating whether the current row is active or not.
 * @tparam RowElementNumber The index of the row being processed.
 */
template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber, bool Active,
          std::size_t RowElementNumber>
struct CountSparseMatrixRowLoop;

/**
 * @brief Specialization of CountSparseMatrixRowLoop for the case when the row
 * is active.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * recursively counting the number of elements in the current row, incrementing
 * the ElementCount if the row is active.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 * @tparam ElementCount The current count of elements processed.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam RowElementNumber The index of the row being processed.
 */
template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber, std::size_t RowElementNumber>
struct CountSparseMatrixRowLoop<SparseAvailable, ElementCount,
                                ColumnElementNumber, true, RowElementNumber> {
  using type = typename CountSparseMatrixRowLoop<
      SparseAvailable, (ElementCount + 1), ColumnElementNumber,
      SparseAvailable::lists[ColumnElementNumber][RowElementNumber - 1],
      RowElementNumber - 1>::type;
};

/**
 * @brief Specialization of CountSparseMatrixRowLoop for the case when the row
 * is not active.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * recursively counting the number of elements in the current row without
 * incrementing the ElementCount if the row is not active.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 * @tparam ElementCount The current count of elements processed.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam RowElementNumber The index of the row being processed.
 */
template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber, std::size_t RowElementNumber>
struct CountSparseMatrixRowLoop<SparseAvailable, ElementCount,
                                ColumnElementNumber, false, RowElementNumber> {
  using type = typename CountSparseMatrixRowLoop<
      SparseAvailable, ElementCount, ColumnElementNumber,
      SparseAvailable::lists[ColumnElementNumber][RowElementNumber - 1],
      RowElementNumber - 1>::type;
};

/**
 * @brief Specialization of CountSparseMatrixRowLoop for the case when there
 * are no more rows to process.
 *
 * This specialization defines a type alias 'type' that is set to an
 * IndexSequence containing the ElementCount, effectively indicating that there
 * are no more rows to process and returning the count of elements in the last
 * row.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 * @tparam ElementCount The current count of elements processed.
 * @tparam ColumnElementNumber The index of the column being processed.
 */
template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber>
struct CountSparseMatrixRowLoop<SparseAvailable, ElementCount,
                                ColumnElementNumber, false, 0> {
  using type = IndexSequence<ElementCount>;
};

/**
 * @brief Specialization of CountSparseMatrixRowLoop for the case when there
 * are no more rows to process and the row is active.
 *
 * This specialization defines a type alias 'type' that is set to an
 * IndexSequence containing (ElementCount + 1), effectively indicating that
 * there are no more rows to process and returning the count of elements in the
 * last row incremented by one.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 * @tparam ElementCount The current count of elements processed.
 * @tparam ColumnElementNumber The index of the column being processed.
 */
template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber>
struct CountSparseMatrixRowLoop<SparseAvailable, ElementCount,
                                ColumnElementNumber, true, 0> {
  using type = IndexSequence<(ElementCount + 1)>;
};

/**
 * @brief A template struct to count the number of elements in each column of a
 * sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * counting the number of elements in each column for the SparseAvailable type.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 * @tparam ColumnElementNumber The index of the column being processed.
 */
template <typename SparseAvailable, std::size_t ColumnElementNumber>
struct CountSparseMatrixColumnLoop {
  using type = typename Concatenate<
      typename CountSparseMatrixColumnLoop<SparseAvailable,
                                           ColumnElementNumber - 1>::type,
      typename CountSparseMatrixRowLoop<
          SparseAvailable, 0, ColumnElementNumber,
          SparseAvailable::lists[ColumnElementNumber]
                                [SparseAvailable::column_size - 1],
          (SparseAvailable::column_size - 1)>::type>::type;
};

/**
 * @brief Specialization of CountSparseMatrixColumnLoop for the case when
 * ColumnElementNumber is 0.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * CountSparseMatrixRowLoop for the first column, effectively counting the
 * number of elements in the first column of the SparseAvailable type.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 */
template <typename SparseAvailable>
struct CountSparseMatrixColumnLoop<SparseAvailable, 0> {
  using type = typename Concatenate<
      IndexSequence<0>,
      typename CountSparseMatrixRowLoop<
          SparseAvailable, 0, 0,
          SparseAvailable::lists[0][SparseAvailable::column_size - 1],
          (SparseAvailable::column_size - 1)>::type>::type;
};

/**
 * @brief A template struct to count the number of elements in each column of a
 * sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * counting the number of elements in each column for the SparseAvailable type,
 * taking into account whether the SparseAvailable type is empty or not.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 */
template <typename CountSparseMatrixColumnLoop, std::size_t ColumnElementNumber>
struct AccumulateElementNumberLoop {
  static constexpr std::size_t compute() {
    return CountSparseMatrixColumnLoop::list[ColumnElementNumber] +
           AccumulateElementNumberLoop<CountSparseMatrixColumnLoop,
                                       ColumnElementNumber - 1>::compute();
  }
};

/**
 * @brief Specialization of AccumulateElementNumberLoop for the case when
 * ColumnElementNumber is 0.
 *
 * This specialization defines a static constexpr function 'compute' that
 * returns 0, effectively indicating that there are no elements to accumulate
 * when the column index is 0.
 *
 * @tparam CountSparseMatrixColumnLoop The CountSparseMatrixColumnLoop type
 * containing the counts of elements in each column.
 */
template <typename CountSparseMatrixColumnLoop>
struct AccumulateElementNumberLoop<CountSparseMatrixColumnLoop, 0> {
  static constexpr std::size_t compute() { return 0; }
};

/**
 * @brief A template struct to accumulate the number of elements in each column
 * of a sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * accumulating the number of elements in each column for the SparseAvailable
 * type.
 *
 * @tparam CountSparseMatrixColumnLoop The CountSparseMatrixColumnLoop type
 * containing the counts of elements in each column.
 * @tparam ColumnElementNumber The index of the column being processed.
 */
template <typename CountSparseMatrixColumnLoop, std::size_t ColumnElementNumber>
struct AccumulateSparseMatrixElementNumberLoop {
  using type = typename Concatenate<
      typename AccumulateSparseMatrixElementNumberLoop<
          CountSparseMatrixColumnLoop, ColumnElementNumber - 1>::type,
      IndexSequence<AccumulateElementNumberLoop<
          CountSparseMatrixColumnLoop, ColumnElementNumber>::compute()>>::type;
};

/**
 * @brief Specialization of AccumulateSparseMatrixElementNumberLoop for the case
 * when ColumnElementNumber is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * IndexSequence<CountSparseMatrixColumnLoop::list[0]>, effectively returning
 * the count of elements in the first column of the SparseAvailable type.
 *
 * @tparam CountSparseMatrixColumnLoop The CountSparseMatrixColumnLoop type
 * containing the counts of elements in each column.
 */
template <typename CountSparseMatrixColumnLoop>
struct AccumulateSparseMatrixElementNumberLoop<CountSparseMatrixColumnLoop, 0> {
  using type = IndexSequence<CountSparseMatrixColumnLoop::list[0]>;
};

/**
 * @brief A template struct to accumulate the number of elements in each column
 * of a sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * accumulating the number of elements in each column for the SparseAvailable
 * type, taking into account whether the SparseAvailable type is empty or not.
 *
 * @tparam CountSparseMatrixColumnLoop The CountSparseMatrixColumnLoop type
 * containing the counts of elements in each column.
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 */
template <typename CountSparseMatrixColumnLoop, typename SparseAvailable>
struct AccumulateSparseMatrixElementNumberStruct {
  using type = typename AccumulateSparseMatrixElementNumberLoop<
      CountSparseMatrixColumnLoop, SparseAvailable::number_of_columns>::type;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to create row pointers from a SparseAvailable type.
 *
 * This alias uses the
 * TemplatesOperation::AccumulateSparseMatrixElementNumberStruct to generate a
 * type representing the row pointers for the SparseAvailable type.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 *
 * The resulting type is a RowPointers type that contains the accumulated number
 * of elements in each row of the SparseAvailable type.
 */
template <typename SparseAvailable>
using RowPointersFromSparseAvailable =
    typename TemplatesOperation::ToRowPointers<
        typename TemplatesOperation::AccumulateSparseMatrixElementNumberStruct<
            typename TemplatesOperation::CountSparseMatrixColumnLoop<
                SparseAvailable,
                (SparseAvailable::number_of_columns - 1)>::type,
            SparseAvailable>::type>::type;

namespace TemplatesOperation {

/* Sequence for Triangular */

/** * @brief A template struct to create a triangular index sequence.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating a triangular index sequence for a given range.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 * @tparam E_S The current size of the sequence.
 */
template <std::size_t Start, std::size_t End, std::size_t E_S>
struct MakeTriangularIndexSequence {
  using type = typename Concatenate<
      typename MakeTriangularIndexSequence<Start, (End - 1), (E_S - 1)>::type,
      IndexSequence<(End - 1)>>::type;
};

/**
 * @brief Specialization of MakeTriangularIndexSequence for the case when E_S is
 * 0.
 *
 * This specialization defines a type alias 'type' that is set to IndexSequence
 * containing (End - 1), effectively creating a triangular index sequence for
 * the specified range.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 */
template <std::size_t Start, std::size_t End>
struct MakeTriangularIndexSequence<Start, End, 0> {
  using type = IndexSequence<(End - 1)>;
};

/**
 * @brief A template struct to create a triangular sequence list.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * triangular index sequence for a given range, starting from Start and ending
 * at End.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 */
template <std::size_t Start, std::size_t End> struct TriangularSequenceList {
  using type =
      typename MakeTriangularIndexSequence<Start, End, (End - Start)>::type;
};

/* Count for Triangular */

/**
 * @brief A template struct to create a triangular count sequence.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating a triangular count sequence for a given range.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 * @tparam E_S The current size of the sequence.
 */
template <std::size_t Start, std::size_t End, std::size_t E_S>
struct MakeTriangularCountSequence {
  using type =
      typename Concatenate<IndexSequence<End>,
                           typename MakeTriangularCountSequence<
                               Start, (End - 1), (E_S - 1)>::type>::type;
};

/**
 * @brief Specialization of MakeTriangularCountSequence for the case when E_S is
 * 0.
 *
 * This specialization defines a type alias 'type' that is set to IndexSequence
 * containing End, effectively creating a triangular count sequence for the
 * specified range.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 */
template <std::size_t Start, std::size_t End>
struct MakeTriangularCountSequence<Start, End, 0> {
  using type = IndexSequence<End>;
};

/**
 * @brief A template struct to create a triangular count list.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * triangular count sequence for a given range, starting from Start and ending
 * at End.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 */
template <std::size_t Start, std::size_t End> struct TriangularCountList {
  using type =
      typename MakeTriangularCountSequence<Start, End, (End - Start)>::type;
};

/**
 * @brief A template struct to count the number of elements in a triangular
 * sequence.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * triangular count sequence for a given range, starting from 0 and ending at N,
 * and concatenating it with an IndexSequence containing 0.
 *
 * @tparam M The starting index of the sequence.
 * @tparam N The ending index of the sequence.
 */
template <std::size_t M, std::size_t N> struct TriangularCountNumbers {
  using type =
      typename Concatenate<IndexSequence<0>,
                           typename TriangularCountList<
                               ((N - ((N < M) ? N : M)) + 1), N>::type>::type;
};

/* Create Upper Triangular Sparse Matrix Row Indices */

/**
 * @brief A template struct to create a sequence of upper triangular row
 * numbers.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating a sequence of upper triangular row numbers for a given range.
 *
 * @tparam M The starting index of the sequence.
 * @tparam N The ending index of the sequence.
 */
template <std::size_t M, std::size_t N>
struct ConcatenateUpperTriangularRowNumbers {
  using type = typename Concatenate<
      typename ConcatenateUpperTriangularRowNumbers<(M - 1), N>::type,
      typename TriangularSequenceList<M, N>::type>::type;
};

/**
 * @brief Specialization of ConcatenateUpperTriangularRowNumbers for the case
 * when M is 1.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * TriangularSequenceList<1, N>, effectively creating a sequence of upper
 * triangular row numbers for the specified range.
 *
 * @tparam N The ending index of the sequence.
 */
template <std::size_t N> struct ConcatenateUpperTriangularRowNumbers<1, N> {
  using type = typename TriangularSequenceList<1, N>::type;
};

/**
 * @brief A template alias to create a sequence of upper triangular row numbers
 * for a given range.
 *
 * This alias uses the ConcatenateUpperTriangularRowNumbers to generate a type
 * representing the upper triangular row numbers for the specified range.
 *
 * @tparam M The starting index of the sequence.
 * @tparam N The ending index of the sequence.
 *
 * The resulting type is a sequence of upper triangular row numbers for the
 * specified range.
 */
template <std::size_t M, std::size_t N>
using UpperTriangularRowNumbers =
    typename ConcatenateUpperTriangularRowNumbers<((N < M) ? N : M), N>::type;

template <typename TriangularCountNumbers, std::size_t M>
struct AccumulateTriangularElementNumberStruct {
  using type =
      typename AccumulateSparseMatrixElementNumberLoop<TriangularCountNumbers,
                                                       M>::type;
};

/**
 * @brief A template struct to extend the
 * AccumulateTriangularElementNumberStruct for a specific range.
 *
 * This struct provides a type alias 'type' that is the result of generating an
 * extended sequence of triangular element numbers for a given range, taking
 * into account the difference between M and N.
 *
 * @tparam TriangularCountNumbers The TriangularCountNumbers type containing the
 * counts of elements in each row.
 * @tparam M The starting index of the sequence.
 * @tparam N The ending index of the sequence.
 * @tparam Dif The difference between M and N.
 */
template <typename TriangularCountNumbers, std::size_t M, std::size_t N,
          std::size_t Dif>
struct AccumulateTriangularElementNumberStructExtend {
  using _Sequence =
      typename TemplatesOperation::AccumulateTriangularElementNumberStruct<
          typename TemplatesOperation::TriangularCountNumbers<N, N>::type,
          N>::type;

  using type = typename Concatenate<
      _Sequence, typename RepeatConcatenateIndexSequence<
                     Dif, IndexSequence<_Sequence::list[N]>>::type>::type;
};

/**
 * @brief Specialization of AccumulateTriangularElementNumberStructExtend for
 * the case when Dif is 0.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * AccumulateTriangularElementNumberStruct for the TriangularCountNumbers type
 * and M, effectively generating an extended sequence of triangular element
 * numbers for the specified range when Dif is 0.
 *
 * @tparam TriangularCountNumbers The TriangularCountNumbers type containing the
 * counts of elements in each row.
 * @tparam M The starting index of the sequence.
 * @tparam N The ending index of the sequence.
 */
template <typename TriangularCountNumbers, std::size_t M, std::size_t N>
struct AccumulateTriangularElementNumberStructExtend<TriangularCountNumbers, M,
                                                     N, 0> {
  using type =
      typename TemplatesOperation::AccumulateTriangularElementNumberStruct<
          typename TemplatesOperation::TriangularCountNumbers<M, N>::type,
          M>::type;
};

} // namespace TemplatesOperation

/* Create Upper Triangular Sparse Matrix Row Indices and Pointers */

/**
 * @brief A template alias to create a sequence of upper triangular row indices
 * for a given range.
 *
 * This alias uses the TemplatesOperation::UpperTriangularRowNumbers to
 * generate a type representing the upper triangular row indices for the
 * specified range.
 *
 * @tparam M The starting index of the sequence.
 * @tparam N The ending index of the sequence.
 *
 * The resulting type is an IndexSequence containing the upper triangular row
 * indices for the specified range.
 */
template <std::size_t M, std::size_t N>
using UpperTriangularRowIndices = typename TemplatesOperation::ToRowIndices<
    TemplatesOperation::UpperTriangularRowNumbers<M, N>>::type;

/**
 * @brief A template alias to create row pointers for an upper triangular sparse
 * matrix.
 *
 * This alias uses the
 * TemplatesOperation::AccumulateTriangularElementNumberStruct to generate a
 * type representing the row pointers for the upper triangular sparse matrix.
 *
 * @tparam M The starting index of the sequence.
 * @tparam N The ending index of the sequence.
 *
 * The resulting type is a RowPointers type that contains the accumulated number
 * of elements in each row of the upper triangular sparse matrix.
 */
template <std::size_t M, std::size_t N>
using UpperTriangularRowPointers = typename TemplatesOperation::ToRowPointers<
    typename TemplatesOperation::AccumulateTriangularElementNumberStructExtend<
        typename TemplatesOperation::TriangularCountNumbers<M, N>::type, M, N,
        (M <= N ? 0 : M - N)>::type>::type;

namespace TemplatesOperation {

/* Create Lower Triangular Sparse Matrix Row Indices */

/** @brief A template struct to create a lower triangular index sequence.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating a lower triangular index sequence for a given range.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 * @tparam E_S The current size of the sequence.
 */
template <std::size_t Start, std::size_t End, std::size_t E_S>
struct MakeLowerTriangularIndexSequence {
  using type = typename Concatenate<typename MakeLowerTriangularIndexSequence<
                                        Start, (End - 1), (E_S - 1)>::type,
                                    IndexSequence<E_S>>::type;
};

/**
 * @brief Specialization of MakeLowerTriangularIndexSequence for the case when
 * E_S is 0.
 *
 * This specialization defines a type alias 'type' that is set to IndexSequence
 * containing 0, effectively creating a lower triangular index sequence for the
 * specified range.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 */
template <std::size_t Start, std::size_t End>
struct MakeLowerTriangularIndexSequence<Start, End, 0> {
  using type = IndexSequence<0>;
};

/**
 * @brief A template struct to create a lower triangular sequence list.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * lower triangular index sequence for a given range, starting from Start and
 * ending at End.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 */
template <std::size_t Start, std::size_t End>
struct LowerTriangularSequenceList {
  using type = typename MakeLowerTriangularIndexSequence<Start, End,
                                                         (End - Start)>::type;
};

/**
 * @brief A template struct to create a sequence of lower triangular row
 * numbers.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating a sequence of lower triangular row numbers for a given range.
 *
 * @tparam M The starting index of the sequence.
 * @tparam N The ending index of the sequence.
 */
template <std::size_t M, std::size_t N>
struct ConcatenateLowerTriangularRowNumbers {
  static_assert(M <= N, "So far, M must be less than or equal to N");

  using type = typename Concatenate<
      typename LowerTriangularSequenceList<M, N>::type,
      typename ConcatenateLowerTriangularRowNumbers<(M - 1), N>::type>::type;
};

/**
 * @brief Specialization of ConcatenateLowerTriangularRowNumbers for the case
 * when M is 1.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * LowerTriangularSequenceList<1, N>, effectively creating a sequence of lower
 * triangular row numbers for the specified range.
 *
 * @tparam N The ending index of the sequence.
 */
template <std::size_t N> struct ConcatenateLowerTriangularRowNumbers<1, N> {
  using type = typename LowerTriangularSequenceList<1, N>::type;
};

/**
 * @brief A template alias to create a sequence of lower triangular row numbers
 * for a given range.
 *
 * This alias uses the ConcatenateLowerTriangularRowNumbers to generate a type
 * representing the lower triangular row numbers for the specified range.
 *
 * @tparam M The starting index of the sequence.
 * @tparam N The ending index of the sequence.
 *
 * The resulting type is a sequence of lower triangular row numbers for the
 * specified range.
 */
template <std::size_t M, std::size_t N>
using LowerTriangularRowNumbers =
    typename TemplatesOperation::ConcatenateLowerTriangularRowNumbers<
        M, ((M < N) ? M : N)>::type;

} // namespace TemplatesOperation

/**
 * @brief A template alias to create a sequence of lower triangular row indices
 * for a given range.
 *
 * This alias uses the TemplatesOperation::LowerTriangularRowNumbers to
 * generate a type representing the lower triangular row indices for the
 * specified range.
 *
 * @tparam M The starting index of the sequence.
 * @tparam N The ending index of the sequence.
 *
 * The resulting type is an IndexSequence containing the lower triangular row
 * indices for the specified range.
 */
template <std::size_t M, std::size_t N>
using LowerTriangularRowIndices = typename TemplatesOperation::ToRowIndices<
    TemplatesOperation::LowerTriangularRowNumbers<M, N>>::type;

namespace TemplatesOperation {

/* Create Lower Triangular Sparse Matrix Row Pointers */

/**
 * @brief A template struct to create a sequence of lower triangular count
 * numbers.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating a lower triangular count sequence for a given range.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 * @tparam E_S The current size of the sequence.
 */
template <std::size_t Start, std::size_t End, std::size_t E_S>
struct MakeLowerTriangularCountSequence {
  using type =
      typename Concatenate<IndexSequence<Start>,
                           typename MakeLowerTriangularCountSequence<
                               (Start + 1), (End - 1), (E_S - 1)>::type>::type;
};

/**
 * @brief Specialization of MakeLowerTriangularCountSequence for the case when
 * E_S is 0.
 *
 * This specialization defines a type alias 'type' that is set to IndexSequence
 * containing Start, effectively creating a lower triangular count sequence for
 * the specified range.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 */
template <std::size_t Start, std::size_t End>
struct MakeLowerTriangularCountSequence<Start, End, 0> {
  using type = IndexSequence<Start>;
};

/**
 * @brief A template struct to create a lower triangular count list.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * lower triangular count sequence for a given range, starting from Start and
 * ending at End.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 */
template <std::size_t Start, std::size_t End> struct LowerTriangularCountList {
  using type = typename MakeLowerTriangularCountSequence<Start, End,
                                                         (End - Start)>::type;
};

/**
 * @brief A template struct to count the number of elements in a lower
 * triangular sequence.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * lower triangular count sequence for a given range, starting from 0 and
 * ending at N, and concatenating it with an IndexSequence containing 0.
 *
 * @tparam M The starting index of the sequence.
 * @tparam N The ending index of the sequence.
 */
template <std::size_t M, std::size_t N> struct LowerTriangularCountNumbers {
  using type = typename Concatenate<
      IndexSequence<0>,
      typename LowerTriangularCountList<1, ((M < N) ? M : N)>::type>::type;
};

/**
 * @brief A template struct to accumulate the number of elements in each column
 * of a lower triangular sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * accumulating the number of elements in each column for the lower triangular
 * sparse matrix.
 *
 * @tparam LowerTriangularCountNumbers The LowerTriangularCountNumbers type
 * containing the counts of elements in each column.
 * @tparam M The index of the column being processed.
 */
template <typename LowerTriangularCountNumbers, std::size_t M>
struct AccumulateLowerTriangularElementNumberStruct {
  using type = typename AccumulateSparseMatrixElementNumberLoop<
      LowerTriangularCountNumbers, M>::type;
};

} // namespace TemplatesOperation

/**
 * @brief A template alias to create row pointers for a lower triangular sparse
 * matrix.
 *
 * This alias uses the
 * TemplatesOperation::AccumulateLowerTriangularElementNumberStruct to generate
 * a type representing the row pointers for the lower triangular sparse matrix.
 *
 * @tparam M The starting index of the sequence.
 * @tparam N The ending index of the sequence.
 *
 * The resulting type is a RowPointers type that contains the accumulated number
 * of elements in each row of the lower triangular sparse matrix.
 */
template <std::size_t M, std::size_t N>
using LowerTriangularRowPointers = typename TemplatesOperation::ToRowPointers<
    typename TemplatesOperation::AccumulateLowerTriangularElementNumberStruct<
        typename TemplatesOperation::LowerTriangularCountNumbers<M, N>::type,
        M>::type>::type;

/* SparseAvailable Addition and Subtraction */

namespace TemplatesOperation {

/**
 * @brief A helper template to determine if the addition or subtraction of two
 * SparseAvailable types is available.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * checking if the addition or subtraction of two SparseAvailable types is
 * available for each column.
 *
 * @tparam MatrixA The first SparseAvailable type.
 * @tparam MatrixB The second SparseAvailable type.
 */
template <typename MatrixA, typename MatrixB>
struct MatrixAddSubSparseAvailableHelper;

/**
 * @brief Partial specialization of MatrixAddSubSparseAvailableHelper for the
 * case when both SparseAvailable types are ColumnAvailable.
 *
 * This specialization defines a type alias 'type' that is set to a
 * ColumnAvailable type containing the logical OR of the availability of each
 * column in the two SparseAvailable types.
 *
 * @tparam ValuesA The boolean values representing the availability of columns
 * in the first SparseAvailable type.
 * @tparam ValuesB The boolean values representing the availability of columns
 * in the second SparseAvailable type.
 */
template <bool... ValuesA, bool... ValuesB>
struct MatrixAddSubSparseAvailableHelper<ColumnAvailable<ValuesA...>,
                                         ColumnAvailable<ValuesB...>> {
  using type = ColumnAvailable<
      TemplatesOperation::LogicalOr<ValuesA, ValuesB>::value...>;
};

/**
 * @brief Partial specialization of MatrixAddSubSparseAvailableHelper for the
 * case when one of the SparseAvailable types is ColumnAvailable and the other
 * is SparseAvailable.
 *
 * This specialization defines a type alias 'type' that is set to a
 * SparseAvailable type containing the logical OR of the availability of each
 * column in the two SparseAvailable types.
 *
 * @tparam ValuesA The boolean values representing the availability of columns
 * in the first SparseAvailable type.
 * @tparam ValuesB The boolean values representing the availability of columns
 * in the second SparseAvailable type.
 */
template <typename... ColumnsA, typename... ColumnsB>
struct MatrixAddSubSparseAvailableHelper<SparseAvailable<ColumnsA...>,
                                         SparseAvailable<ColumnsB...>> {
  using type = SparseAvailable<
      typename MatrixAddSubSparseAvailableHelper<ColumnsA, ColumnsB>::type...>;
};

} // namespace TemplatesOperation

/**
 * @brief A template alias to determine if the addition or subtraction of two
 * SparseAvailable types is available.
 *
 * This alias uses the TemplatesOperation::MatrixAddSubSparseAvailableHelper to
 * generate a type representing the availability of addition or subtraction for
 * the two SparseAvailable types.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 *
 * The resulting type is a SparseAvailable type containing the logical OR of
 * the availability of each column in the two SparseAvailable types.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B>
using MatrixAddSubSparseAvailable =
    typename TemplatesOperation::MatrixAddSubSparseAvailableHelper<
        SparseAvailable_A, SparseAvailable_B>::type;

/* SparseAvailable Multiply */

namespace TemplatesOperation {

/**
 * @brief A template struct to perform logical OR operation on two boolean
 * values.
 *
 * This struct provides a static constexpr member 'value' that is the result of
 * performing a logical OR operation on the two boolean values A and B.
 *
 * @tparam A The first boolean value.
 * @tparam B The second boolean value.
 */
template <bool A, bool B> struct LogicalAnd {
  static constexpr bool value = A && B;
};

/**
 * @brief A template struct to perform logical OR operation on two boolean
 * values.
 *
 * This struct provides a static constexpr member 'value' that is the result of
 * performing a logical OR operation on the two boolean values A and B.
 *
 * @tparam A The first boolean value.
 * @tparam B The second boolean value.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t COL, std::size_t ROW, std::size_t N_Idx>
struct SparseAvailableMatrixMultiplyElement {
  static constexpr bool value =
      LogicalAnd<SparseAvailable_A::lists[COL][N_Idx],
                 SparseAvailable_B::lists[N_Idx][ROW]>::value;
};

/**
 * @brief A template struct to perform the multiplication of two SparseAvailable
 * types.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * multiplying the elements of two SparseAvailable types for each column and
 * row.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam COL The index of the column being processed.
 * @tparam ROW The index of the row being processed.
 * @tparam N_Idx The index of the element being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t COL, std::size_t ROW, std::size_t N_Idx>
struct SparseAvailableMatrixMultiplyMultiplyLoop {
  using type = ConcatenateColumnAvailable<
      typename SparseAvailableMatrixMultiplyMultiplyLoop<
          SparseAvailable_A, SparseAvailable_B, COL, ROW, (N_Idx - 1)>::type,
      ColumnAvailable<SparseAvailableMatrixMultiplyElement<
          SparseAvailable_A, SparseAvailable_B, COL, ROW, N_Idx>::value>>;
};

/**
 * @brief Specialization of SparseAvailableMatrixMultiplyMultiplyLoop for the
 * case when N_Idx is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * ColumnAvailable type containing the result of the multiplication for the
 * first element.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam COL The index of the column being processed.
 * @tparam ROW The index of the row being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t COL, std::size_t ROW>
struct SparseAvailableMatrixMultiplyMultiplyLoop<
    SparseAvailable_A, SparseAvailable_B, COL, ROW, 0> {
  using type = ColumnAvailable<SparseAvailableMatrixMultiplyElement<
      SparseAvailable_A, SparseAvailable_B, COL, ROW, 0>::value>;
};

/**
 * @brief A template struct to concatenate multiple ColumnAvailable types into a
 * single SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * multiple ColumnAvailable types into a SparseAvailable type.
 *
 * @tparam ColumnAvailableList The list of ColumnAvailable types to be
 * concatenated.
 */
template <typename ColumnAvailable, std::size_t N_Idx>
struct ColumnAvailableElementWiseOr {
  static constexpr bool value = LogicalOr<
      ColumnAvailable::list[N_Idx],
      ColumnAvailableElementWiseOr<ColumnAvailable, (N_Idx - 1)>::value>::value;
};

/**
 * @brief Specialization of ColumnAvailableElementWiseOr for the case when N_Idx
 * is 0.
 *
 * This specialization defines a static constexpr member 'value' that is set to
 * the first element of the ColumnAvailable type, effectively returning the
 * availability of the first column.
 *
 * @tparam ColumnAvailable The ColumnAvailable type containing multiple columns.
 */
template <typename ColumnAvailable>
struct ColumnAvailableElementWiseOr<ColumnAvailable, 0> {
  static constexpr bool value = ColumnAvailable::list[0];
};

/**
 * @brief A template struct to concatenate multiple ColumnAvailable types into a
 * single SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * multiple ColumnAvailable types into a SparseAvailable type.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 */
template <typename SparseAvailable, std::size_t ROW, std::size_t M_Idx>
struct ColumnAvailableFromSparseAvailableColumLoop {
  using type = ConcatenateColumnAvailable<
      typename ColumnAvailableFromSparseAvailableColumLoop<SparseAvailable, ROW,
                                                           (M_Idx - 1)>::type,
      ColumnAvailable<SparseAvailable::lists[M_Idx][ROW]>>;
};

/**
 * @brief Specialization of ColumnAvailableFromSparseAvailableColumLoop for the
 * case when M_Idx is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * ColumnAvailable type containing the first column of the SparseAvailable type,
 * effectively returning the availability of the first column.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple columns.
 * @tparam ROW The index of the row being processed.
 */
template <typename SparseAvailable, std::size_t ROW>
struct ColumnAvailableFromSparseAvailableColumLoop<SparseAvailable, ROW, 0> {
  using type = ColumnAvailable<SparseAvailable::lists[0][ROW]>;
};

/**
 * @brief A template struct to concatenate multiple ColumnAvailable types into a
 * SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * multiple ColumnAvailable types into a SparseAvailable type.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam COL The index of the column being processed.
 * @tparam J_Idx The index of the row being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t COL, std::size_t J_Idx>
struct SparseAvailableMatrixMultiplyRowLoop {
  using type = ConcatenateColumnAvailable<
      typename SparseAvailableMatrixMultiplyRowLoop<
          SparseAvailable_A, SparseAvailable_B, COL, (J_Idx - 1)>::type,
      ColumnAvailable<ColumnAvailableElementWiseOr<
          typename SparseAvailableMatrixMultiplyMultiplyLoop<
              SparseAvailable_A, SparseAvailable_B, COL, J_Idx,
              (SparseAvailable_A::column_size - 1)>::type,
          (SparseAvailable_A::column_size - 1)>::value>>;
};

/**
 * @brief Specialization of SparseAvailableMatrixMultiplyRowLoop for the case
 * when J_Idx is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * ColumnAvailable type containing the result of the multiplication for the
 * first row.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam COL The index of the column being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t COL>
struct SparseAvailableMatrixMultiplyRowLoop<SparseAvailable_A,
                                            SparseAvailable_B, COL, 0> {
  using type = ColumnAvailable<ColumnAvailableElementWiseOr<
      typename SparseAvailableMatrixMultiplyMultiplyLoop<
          SparseAvailable_A, SparseAvailable_B, COL, 0,
          (SparseAvailable_A::column_size - 1)>::type,
      (SparseAvailable_A::column_size - 1)>::value>;
};

/**
 * @brief A template struct to concatenate multiple SparseAvailable types into a
 * single SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * multiple SparseAvailable types into a SparseAvailable type.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam I_Idx The index of the column being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t I_Idx>
struct SparseAvailableMatrixMultiplyColumnLoop {
  using type = ConcatenateSparseAvailableVertically<
      typename SparseAvailableMatrixMultiplyColumnLoop<
          SparseAvailable_A, SparseAvailable_B, (I_Idx - 1)>::type,
      SparseAvailableColumns<typename SparseAvailableMatrixMultiplyRowLoop<
          SparseAvailable_A, SparseAvailable_B, I_Idx,
          (SparseAvailable_B::column_size - 1)>::type>>;
};

/**
 * @brief Specialization of SparseAvailableMatrixMultiplyColumnLoop for the case
 * when I_Idx is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * SparseAvailableColumns type containing the result of the multiplication for
 * the first column.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B>
struct SparseAvailableMatrixMultiplyColumnLoop<SparseAvailable_A,
                                               SparseAvailable_B, 0> {
  using type =
      SparseAvailableColumns<typename SparseAvailableMatrixMultiplyRowLoop<
          SparseAvailable_A, SparseAvailable_B, 0,
          (SparseAvailable_B::column_size - 1)>::type>;
};

} // namespace TemplatesOperation

/**
 * @brief A template alias to perform matrix multiplication of two
 * SparseAvailable types.
 *
 * This alias uses the
 * TemplatesOperation::SparseAvailableMatrixMultiplyColumnLoop to generate a
 * type representing the result of multiplying two SparseAvailable types.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 *
 * The resulting type is a SparseAvailable type containing the result of the
 * multiplication for each column.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B>
using SparseAvailableMatrixMultiply =
    typename TemplatesOperation::SparseAvailableMatrixMultiplyColumnLoop<
        SparseAvailable_A, SparseAvailable_B,
        (SparseAvailable_A::number_of_columns - 1)>::type;

/* SparseAvailable Multiply Transpose
 * (SparseAvailable_BT will be calculated as transpose) */

namespace TemplatesOperation {

/**
 * @brief A template struct to perform logical AND operation on two boolean
 * values.
 *
 * This struct provides a static constexpr member 'value' that is the result of
 * performing a logical AND operation on the two boolean values A and B.
 *
 * @tparam A The first boolean value.
 * @tparam B The second boolean value.
 */
template <typename SparseAvailable_A, typename SparseAvailable_BT,
          std::size_t COL, std::size_t ROW, std::size_t N_Idx>
struct SparseAvailableMatrixMultiplyTransposeElement {
  static constexpr bool value =
      LogicalAnd<SparseAvailable_A::lists[COL][N_Idx],
                 SparseAvailable_BT::lists[ROW][N_Idx]>::value;
};

/**
 * @brief A template struct to perform the multiplication of two SparseAvailable
 * types with transpose.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * multiplying the elements of two SparseAvailable types for each column and
 * row, considering the transpose of the second SparseAvailable type.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_BT The second SparseAvailable type (transposed).
 * @tparam COL The index of the column being processed.
 * @tparam ROW The index of the row being processed.
 * @tparam N_Idx The index of the element being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_BT,
          std::size_t COL, std::size_t ROW, std::size_t N_Idx>
struct SparseAvailableMatrixMultiplyTransposeMultiplyLoop {
  using type = ConcatenateColumnAvailable<
      typename SparseAvailableMatrixMultiplyTransposeMultiplyLoop<
          SparseAvailable_A, SparseAvailable_BT, COL, ROW, (N_Idx - 1)>::type,
      ColumnAvailable<SparseAvailableMatrixMultiplyTransposeElement<
          SparseAvailable_A, SparseAvailable_BT, COL, ROW, N_Idx>::value>>;
};

/**
 * @brief Specialization of SparseAvailableMatrixMultiplyTransposeMultiplyLoop
 * for the case when N_Idx is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * ColumnAvailable type containing the result of the multiplication for the
 * first element.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_BT The second SparseAvailable type (transposed).
 * @tparam COL The index of the column being processed.
 * @tparam ROW The index of the row being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_BT,
          std::size_t COL, std::size_t ROW>
struct SparseAvailableMatrixMultiplyTransposeMultiplyLoop<
    SparseAvailable_A, SparseAvailable_BT, COL, ROW, 0> {
  using type = ColumnAvailable<SparseAvailableMatrixMultiplyTransposeElement<
      SparseAvailable_A, SparseAvailable_BT, COL, ROW, 0>::value>;
};

/**
 * @brief A template struct to concatenate multiple ColumnAvailable types into a
 * SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * multiple ColumnAvailable types into a SparseAvailable type.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_BT The second SparseAvailable type (transposed).
 * @tparam COL The index of the column being processed.
 * @tparam J_Idx The index of the row being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_BT,
          std::size_t COL, std::size_t J_Idx>
struct SparseAvailableMatrixMultiplyTransposeRowLoop {
  using type = ConcatenateColumnAvailable<
      typename SparseAvailableMatrixMultiplyTransposeRowLoop<
          SparseAvailable_A, SparseAvailable_BT, COL, (J_Idx - 1)>::type,
      ColumnAvailable<ColumnAvailableElementWiseOr<
          typename SparseAvailableMatrixMultiplyTransposeMultiplyLoop<
              SparseAvailable_A, SparseAvailable_BT, COL, J_Idx,
              (SparseAvailable_A::column_size - 1)>::type,
          (SparseAvailable_A::column_size - 1)>::value>>;
};

/**
 * @brief Specialization of SparseAvailableMatrixMultiplyTransposeRowLoop for
 * the case when J_Idx is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * ColumnAvailable type containing the result of the multiplication for the
 * first row.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_BT The second SparseAvailable type (transposed).
 * @tparam COL The index of the column being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_BT,
          std::size_t COL>
struct SparseAvailableMatrixMultiplyTransposeRowLoop<
    SparseAvailable_A, SparseAvailable_BT, COL, 0> {
  using type = ColumnAvailable<ColumnAvailableElementWiseOr<
      typename SparseAvailableMatrixMultiplyTransposeMultiplyLoop<
          SparseAvailable_A, SparseAvailable_BT, COL, 0,
          (SparseAvailable_A::column_size - 1)>::type,
      (SparseAvailable_A::column_size - 1)>::value>;
};

/**
 * @brief A template struct to concatenate multiple SparseAvailable types into a
 * SparseAvailableColumns type.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * multiple SparseAvailable types into a SparseAvailableColumns type.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_BT The second SparseAvailable type (transposed).
 * @tparam I_Idx The index of the column being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_BT,
          std::size_t I_Idx>
struct SparseAvailableMatrixMultiplyTransposeColumnLoop {
  using type = ConcatenateSparseAvailableVertically<
      typename SparseAvailableMatrixMultiplyTransposeColumnLoop<
          SparseAvailable_A, SparseAvailable_BT, (I_Idx - 1)>::type,
      TemplatesOperation::SparseAvailableColumns<
          typename SparseAvailableMatrixMultiplyTransposeRowLoop<
              SparseAvailable_A, SparseAvailable_BT, I_Idx,
              (SparseAvailable_BT::number_of_columns - 1)>::type>>;
};

/**
 * @brief Specialization of SparseAvailableMatrixMultiplyTransposeColumnLoop
 * for the case when I_Idx is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * SparseAvailableColumns type containing the result of the multiplication for
 * the first column.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_BT The second SparseAvailable type (transposed).
 */
template <typename SparseAvailable_A, typename SparseAvailable_BT>
struct SparseAvailableMatrixMultiplyTransposeColumnLoop<SparseAvailable_A,
                                                        SparseAvailable_BT, 0> {
  using type = TemplatesOperation::SparseAvailableColumns<
      typename SparseAvailableMatrixMultiplyTransposeRowLoop<
          SparseAvailable_A, SparseAvailable_BT, 0,
          (SparseAvailable_BT::number_of_columns - 1)>::type>;
};

/**
 * @brief A template alias to perform matrix multiplication of a SparseAvailable
 * type with the transpose of another SparseAvailable type.
 *
 * This alias uses the
 * TemplatesOperation::SparseAvailableMatrixMultiplyTransposeColumnLoop to
 * generate a type representing the result of multiplying a SparseAvailable type
 * with the transpose of another SparseAvailable type.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_BT The second SparseAvailable type (transposed).
 *
 * The resulting type is a SparseAvailable type containing the result of the
 * multiplication for each column.
 */
template <typename SparseAvailable_A, typename SparseAvailable_BT>
using SparseAvailableMatrixMultiplyTranspose =
    typename SparseAvailableMatrixMultiplyTransposeColumnLoop<
        SparseAvailable_A, SparseAvailable_BT,
        (SparseAvailable_A::number_of_columns - 1)>::type;

} // namespace TemplatesOperation

/**
 *  @brief A template alias to perform matrix multiplication of a
 * SparseAvailable type with the transpose of another SparseAvailable type,
 * where the first SparseAvailable type is diagonal.
 *
 * This alias uses the
 * TemplatesOperation::SparseAvailableMatrixMultiplyTranspose to generate a type
 * representing the result of multiplying a diagonal SparseAvailable type with
 * the transpose of another SparseAvailable type.
 *
 * @tparam SparseAvailable The SparseAvailable type to be multiplied.
 *
 * The resulting type is a SparseAvailable type containing the result of the
 * multiplication for each column.
 */
template <typename SparseAvailable>
using SparseAvailableTranspose =
    TemplatesOperation::SparseAvailableMatrixMultiplyTranspose<
        DiagAvailable<SparseAvailable::column_size>, SparseAvailable>;

/* Check SparseAvailable is valid or not */

namespace TemplatesOperation {

/**
 * @brief A template struct to validate the SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * validating the SparseAvailable type for each column.
 *
 * @tparam SparseAvailable The SparseAvailable type to be validated.
 * @tparam NumberOfRowsFirst The number of rows in the first column.
 * @tparam ColumnIndex The index of the column being processed.
 */
template <typename SparseAvailable, std::size_t NumberOfRowsFirst,
          std::size_t ColumnIndex>
struct ValidateSparseAvailableLoop {
  static_assert(SparseAvailable::column_size == NumberOfRowsFirst,
                "Each ColumnAvailable size of SparseAvailable is not the same");

  // static_assert(NumberOfRowsFirst > 3, "NumberOfRowsFirst value");

  using type = typename ValidateSparseAvailableLoop<
      GetRestOfSparseAvailable<SparseAvailable, 1>, NumberOfRowsFirst,
      (ColumnIndex - 1)>::type;
};

/**
 * @brief Specialization of ValidateSparseAvailableLoop for the case when
 * ColumnIndex is 0.
 *
 * This specialization defines a type alias 'type' that is set to the
 * SparseAvailable type itself, effectively validating the SparseAvailable type
 * for the first column.
 *
 * @tparam SparseAvailable The SparseAvailable type to be validated.
 * @tparam NumberOfRowsFirst The number of rows in the first column.
 */
template <typename SparseAvailable, std::size_t NumberOfRowsFirst>
struct ValidateSparseAvailableLoop<SparseAvailable, NumberOfRowsFirst, 0> {
  static_assert(SparseAvailable::column_size == NumberOfRowsFirst,
                "Each ColumnAvailable size of SparseAvailable is not the same");

  using type = SparseAvailable;
};

} // namespace TemplatesOperation

/**
 * @brief A template alias to validate the SparseAvailable type.
 *
 * This alias uses the TemplatesOperation::ValidateSparseAvailableLoop to
 * generate a type representing the validated SparseAvailable type.
 *
 * @tparam SparseAvailable The SparseAvailable type to be validated.
 *
 * The resulting type is the validated SparseAvailable type if it is valid, or
 * it will trigger a static assertion if it is not valid.
 */
template <typename SparseAvailable>
using ValidateSparseAvailable =
    typename TemplatesOperation::ValidateSparseAvailableLoop<
        SparseAvailable, SparseAvailable::column_size,
        (SparseAvailable::number_of_columns - 1)>::type;

/* SparseAvailable get row */
namespace TemplatesOperation {

/** @brief A template struct to generate a ColumnAvailable type with a
 * specific index set to true.
 *
 * This struct provides a type alias 'type' that is set to a ColumnAvailable
 * type with the specified index set to true, and all other indices set to
 * false.
 *
 * @tparam M The total number of columns.
 * @tparam Index The index to be set to true.
 */
template <std::size_t M, std::size_t Index>
using GenerateIndexedRowTrueColumnAvailable =
    ColumnAvailable<(M == Index ? true : false)>;

/**
 * @brief A template struct to recursively generate a ColumnAvailable type with
 * a specific index set to true for each column.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating a ColumnAvailable type with the specified index set to true for
 * each column.
 *
 * @tparam M The total number of columns.
 * @tparam Index The index to be set to true.
 */
template <std::size_t M, std::size_t Index, typename ColumnAvailable,
          typename... Columns>
struct IndexedRowRepeatColumnAvailable {
  using type = typename IndexedRowRepeatColumnAvailable<
      (M - 1), Index, GenerateIndexedRowTrueColumnAvailable<(M - 1), Index>,
      ColumnAvailable, Columns...>::type;
};

/**
 * @brief Specialization of IndexedRowRepeatColumnAvailable for the case when M
 * is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * SparseAvailable type containing the ColumnAvailable types generated for each
 * column.
 *
 * @tparam Index The index to be set to true.
 * @tparam ColumnAvailable The ColumnAvailable type generated so far.
 * @tparam Columns The remaining ColumnAvailable types.
 */
template <std::size_t Index, typename ColumnAvailable, typename... Columns>
struct IndexedRowRepeatColumnAvailable<0, Index, ColumnAvailable, Columns...> {
  using type = TemplatesOperation::SparseAvailableColumns<
      GenerateIndexedRowTrueColumnAvailable<0, Index>, Columns...>;
};

} // namespace TemplatesOperation

/**
 * @brief A template alias to generate a SparseAvailable type with a specific
 * row index set to true.
 *
 * This alias uses the TemplatesOperation::IndexedRowRepeatColumnAvailable to
 * generate a type representing the SparseAvailable type with the specified row
 * index set to true for each column.
 *
 * @tparam M The total number of columns.
 * @tparam Index The index of the row to be set to true.
 *
 * The resulting type is a SparseAvailable type containing the ColumnAvailable
 * types generated for each column, with the specified row index set to true.
 */
template <std::size_t M, std::size_t Index>
using RowAvailable =
    typename TemplatesOperation::IndexedRowRepeatColumnAvailable<
        (M - 1), Index,
        TemplatesOperation::GenerateIndexedRowTrueColumnAvailable<(M - 1),
                                                                  Index>>::type;

/**
 * @brief A template alias to perform matrix multiplication of a SparseAvailable
 * type with a RowAvailable type.
 *
 * This alias uses the TemplatesOperation::SparseAvailableMatrixMultiply to
 * generate a type representing the result of multiplying a SparseAvailable type
 * with a RowAvailable type.
 *
 * @tparam SparseAvailable The SparseAvailable type to be multiplied.
 * @tparam M The total number of columns in the RowAvailable type.
 * @tparam Index The index of the row to be set to true in the RowAvailable
 * type.
 *
 * The resulting type is a SparseAvailable type containing the result of the
 * multiplication for each column, where the specified row index in the
 * RowAvailable type is set to true.
 */
template <std::size_t M, typename SparseAvailable, std::size_t Index>
using SparseAvailableGetRow =
    SparseAvailableMatrixMultiply<SparseAvailable, RowAvailable<M, Index>>;

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_TEMPLATES_HPP__
