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
 * tparam M is the number of rows in the matrix.
 * tparam N is the number of columns in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef BASE_MATRIX_TEMPLATES_HPP_
#define BASE_MATRIX_TEMPLATES_HPP_

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
using CSRIndices = TemplatesOperation::CompiledSparseMatrixList<
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
using CSRPointers = TemplatesOperation::CompiledSparseMatrixList<
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
 * availability of rows. It leverages
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
 * @brief Template struct to represent a collection of available rows in a
 * sparse matrix.
 *
 * This struct aggregates information about multiple rows, each represented
 * by a type. It provides compile-time access to the number of rows, the
 * lists associated with each column, and the size of the first row.
 *
 * @tparam Columns Variadic template parameter pack representing column types.
 *
 * Members:
 * - number_of_rows: The total number of rows provided.
 * - ExtractList: Helper struct to extract the list from each column type.
 * - lists: Compile-time array of lists, one for each column.
 * - row_size: The size of the first row type.
 */
template <typename... Columns> struct SparseAvailableColumns {
  static constexpr std::size_t number_of_rows = sizeof...(Columns);

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
   * determined by the number of rows. Each entry in the array corresponds to
   * the value provided by ExtractList<Columns>::value.
   *
   * @tparam Columns Template parameter pack representing the rows.
   * @tparam number_of_rows The total number of rows.
   */
  static constexpr const bool *lists[number_of_rows] = {
      ExtractList<Columns>::value...};

  /**
   * @brief Represents an unsigned integral type used for sizes and counts.
   *
   * std::size_t is the unsigned integer type returned by the sizeof operator
   * and is commonly used for array indexing and loop counting. It is guaranteed
   * to be able to represent the size of any object in bytes.
   */
  static constexpr std::size_t row_size =
      std::tuple_element<0, std::tuple<Columns...>>::type::size;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to determine the availability of sparse rows.
 *
 * This alias uses the `SparseAvailableColumns` metafunction from the
 * `TemplatesOperation` namespace to check or operate on the provided column
 * types (`Columns...`). It is typically used in template metaprogramming to
 * enable or disable features based on the properties of the given rows.
 *
 * @tparam Columns Variadic template parameter pack representing column types to
 * be checked for sparse availability.
 */
template <typename... Columns>
using SparseAvailable = TemplatesOperation::SparseAvailableColumns<Columns...>;

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

/**
 * @brief A template struct to concatenate two SparseAvailable types vertically.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * two SparseAvailable types, effectively merging their rows into a new
 * SparseAvailable type.
 *
 * @tparam SparseAvailable1 The first SparseAvailable type.
 * @tparam SparseAvailable2 The second SparseAvailable type.
 *
 * The resulting type is a SparseAvailable type that contains all rows from
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
 * the rows from both SparseAvailable types into a new SparseAvailable type.
 *
 * @tparam Columns1 The variadic template parameter pack representing the
 * rows in the first SparseAvailable type.
 * @tparam Columns2 The variadic template parameter pack representing the
 * rows in the second SparseAvailable type.
 */
template <typename... Columns1, typename... Columns2>
struct ConcatenateSparseAvailable<
    TemplatesOperation::SparseAvailableColumns<Columns1...>,
    TemplatesOperation::SparseAvailableColumns<Columns2...>> {
  using type =
      TemplatesOperation::SparseAvailableColumns<Columns1..., Columns2...>;
};

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
 * @brief Template struct to check if any of the sparse rows are available.
 *
 * This struct is used to determine if at least one column in a sparse matrix is
 * available (i.e., has non-zero entries). It is specialized for both
 * ColumnAvailable and SparseAvailable types.
 *
 * @tparam SparseAvailable A type representing the availability of sparse
 * rows.
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
 * @brief Trait to check if any of the provided rows are considered "sparse
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
 * @brief Metafunction to generate a parameter pack of boolean template
 * arguments, prepending 'false' N times to the pack.
 *
 * @tparam N The number of 'false' flags to prepend.
 *
 * This primary template recursively divides the problem into two halves to
 * reduce recursion depth. It concatenates the results from both halves using
 * ConcatenateColumnAvailable.
 */
template <std::size_t N> struct GenerateFalseFlags {
  // Divide at the midpoint to reduce recursion depth
  static constexpr std::size_t Mid = N / 2;

  using type =
      ConcatenateColumnAvailable<typename GenerateFalseFlags<Mid>::type,
                                 typename GenerateFalseFlags<N - Mid>::type>;
};

/**
 * @brief Specialization of GenerateFalseFlags for the case when N is 1.
 *
 * This specialization defines a type alias 'type' that is set to
 * ColumnAvailable<false>.
 */
template <> struct GenerateFalseFlags<1> {
  using type = ColumnAvailable<false>;
};

/**
 * @brief Specialization of GenerateFalseFlags for the case when N is 0.
 *
 * This specialization defines an empty ColumnAvailable type.
 */
template <> struct GenerateFalseFlags<0> {
  using type = ColumnAvailable<>;
};

/**
 * @brief Alias template to generate a type representing a sequence of 'false'
 * flags for a given size.
 *
 * This alias uses the GenerateFalseFlags metafunction to create a type
 * (typically a type list or similar) containing N 'false' values, which can be
 * used for compile-time flag management or template metaprogramming.
 *
 * @tparam N The number of 'false' flags to generate.
 *
 * @see GenerateFalseFlags
 */
template <std::size_t N>
using GenerateFalseColumnAvailable = typename GenerateFalseFlags<N>::type;

/**
 * @brief Metafunction to generate a parameter pack of boolean template
 * arguments, prepending 'true' N times to the pack.
 *
 * @tparam N The number of 'true' flags to prepend.
 *
 * This primary template recursively divides the problem into two halves to
 * reduce recursion depth. It concatenates the results from both halves using
 * ConcatenateColumnAvailable.
 */
template <std::size_t N> struct GenerateTrueFlags {
  // Divide at the midpoint to reduce recursion depth
  static constexpr std::size_t Mid = N / 2;

  using type =
      ConcatenateColumnAvailable<typename GenerateTrueFlags<Mid>::type,
                                 typename GenerateTrueFlags<N - Mid>::type>;
};

/**
 * @brief Specialization of GenerateTrueFlags for the case when N is 1.
 *
 * This specialization defines a type alias 'type' that is set to
 * ColumnAvailable<true>.
 */
template <> struct GenerateTrueFlags<1> {
  using type = ColumnAvailable<true>;
};

/**
 * @brief Specialization of GenerateTrueFlags for the case when N is 0.
 *
 * This specialization defines an empty ColumnAvailable type.
 */
template <> struct GenerateTrueFlags<0> {
  using type = ColumnAvailable<>;
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
 * @brief Metafunction to generate a type list by repeating a column
 * availability type for a specified number of times.
 *
 * This template recursively constructs a type list representing the
 * availability of rows in a matrix, by prepending a new column availability
 * type at each recursion step. The recursion continues until the base case (not
 * shown here) is reached.
 *
 * @tparam M The number of rows remaining to process.
 * @tparam ColumnAvailableType The type representing the availability of the
 * current column.
 *
 * The resulting type is accessible via the nested ::type member.
 */
template <std::size_t M, typename ColumnAvailableType>
struct RepeatColumnAvailable {
  static constexpr std::size_t Mid = M / 2;

  using type = typename ConcatenateSparseAvailable<
      typename RepeatColumnAvailable<Mid, ColumnAvailableType>::type,
      typename RepeatColumnAvailable<M - Mid, ColumnAvailableType>::type>::type;
};

/**
 * @brief Specialization of RepeatColumnAvailable for the case when M is 1.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailableColumns<ColumnAvailableType>, effectively creating a sparse
 * column representation for a single row.
 *
 * @tparam ColumnAvailableType The type representing the availability of the
 * current column.
 */
template <typename ColumnAvailableType>
struct RepeatColumnAvailable<1, ColumnAvailableType> {
  using type = SparseAvailableColumns<ColumnAvailableType>;
};

/**
 * @brief Specialization of RepeatColumnAvailable for the case when M is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailableColumns<>, effectively creating an empty sparse column
 * representation.
 *
 * @tparam ColumnAvailableType The type representing the availability of the
 * current column (unused in this specialization).
 */
template <typename ColumnAvailableType>
struct RepeatColumnAvailable<0, ColumnAvailableType> {
  using type = SparseAvailableColumns<>;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to determine the availability of dense matrix rows.
 *
 * This alias uses TemplatesOperation utilities to generate a compile-time
 * type indicating which rows are available in a dense matrix of size MxN.
 *
 * @tparam M Number of columns in the matrix.
 * @tparam N Number of rows in the matrix.
 *
 * The resulting type is computed by repeating the column availability
 * (as generated by GenerateTrueColumnAvailable<N>) for M cols using
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
 * availability type for a matrix of size M x N, where all rows are marked as
 * unavailable (false).
 *
 * @tparam M Number of columns in the matrix.
 * @tparam N Number of rows in the matrix.
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
 * @brief Metafunction to generate a ColumnAvailable type with 'true' flags at
 * specific indices.
 *
 * This template recursively constructs a ColumnAvailable type where the flags
 * are set to 'true' at the specified index and 'false' elsewhere. It divides
 * the range of indices into two halves to reduce recursion depth and combines
 * the results using ConcatenateColumnAvailable.
 *
 * @tparam Start The starting index for the current range.
 * @tparam Count The number of indices in the current range.
 * @tparam Index The specific index at which the flag should be set to 'true'.
 *
 * The resulting type is accessible via the nested ::type member, which will be
 * a ColumnAvailable type with 'true' at the specified index and 'false'
 * elsewhere.
 */
template <std::size_t Start, std::size_t Count, std::size_t Index>
struct GenerateIndexedTrueFlagsRange {

  static constexpr std::size_t MidCount = Count / 2;

  using type = ConcatenateColumnAvailable<
      typename GenerateIndexedTrueFlagsRange<Start, MidCount, Index>::type,
      typename GenerateIndexedTrueFlagsRange<Start + MidCount, Count - MidCount,
                                             Index>::type>;
};

/**
 * @brief Specialization of GenerateIndexedTrueFlagsRange for the case when
 * Count is 1.
 *
 * This specialization defines a type alias 'type' that is set to
 * ColumnAvailable<(Start == Index)>, effectively creating a ColumnAvailable
 * type where the flag is 'true' if the current index (Start) matches the
 * specified Index, and 'false' otherwise.
 *
 * @tparam Start The starting index for the current range.
 * @tparam Index The specific index at which the flag should be set to 'true'.
 */
template <std::size_t Start, std::size_t Index>
struct GenerateIndexedTrueFlagsRange<Start, 1, Index> {

  using type = ColumnAvailable<(Start == Index)>;
};

/**
 * @brief Specialization of GenerateIndexedTrueFlagsRange for the case when
 * Count is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * ColumnAvailable<>, effectively creating an empty ColumnAvailable type when
 * there are no indices to process.
 *
 * @tparam Start The starting index for the current range (unused in this
 * specialization).
 * @tparam Index The specific index at which the flag should be set to 'true'
 * (unused in this specialization).
 */
template <std::size_t Start, std::size_t Index>
struct GenerateIndexedTrueFlagsRange<Start, 0, Index> {

  using type = ColumnAvailable<>;
};

/**
 * @brief Alias template to generate a ColumnAvailable type with 'true' flags at
 * a specific index for a range of size N.
 *
 * This alias uses the GenerateIndexedTrueFlagsRange metafunction to create a
 * ColumnAvailable type where the flag is 'true' at the specified Index and
 * 'false' elsewhere for a total of N indices.
 *
 * @tparam N The total number of indices (size of the range).
 * @tparam Index The specific index at which the flag should be set to 'true'.
 *
 * The resulting type is a ColumnAvailable type with 'true' at the specified
 * index and 'false' elsewhere, accessible via the nested ::type member.
 */
template <std::size_t N, std::size_t Index>
using GenerateIndexedTrueColumnAvailable =
    typename GenerateIndexedTrueFlagsRange<0, N, Index>::type;

/**
 * @brief Metafunction to generate a block of diagonal columns for a square
 * matrix.
 *
 * This template recursively constructs a SparseAvailable type representing the
 * diagonal columns of a square matrix. It divides the range of rows into two
 * halves to reduce recursion depth and combines the results using
 * ConcatenateSparseAvailable.
 *
 * @tparam TotalCols The total number of columns in the matrix (also the number
 * of rows for a square matrix).
 * @tparam StartRow The starting row index for the current block.
 * @tparam RowCount The number of rows in the current block.
 *
 * The resulting type is accessible via the nested ::type member, which will be
 * a SparseAvailable type representing the diagonal columns for the specified
 * block of rows.
 */
template <std::size_t TotalCols, std::size_t StartRow, std::size_t RowCount>
struct GenerateDiagColumnsBlock {
  static constexpr std::size_t Mid = RowCount / 2;

  using type = typename ConcatenateSparseAvailable<
      typename GenerateDiagColumnsBlock<TotalCols, StartRow, Mid>::type,
      typename GenerateDiagColumnsBlock<TotalCols, StartRow + Mid,
                                        RowCount - Mid>::type>::type;
};

/**
 * @brief Specialization of GenerateDiagColumnsBlock for the case when RowCount
 * is 1.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailableColumns<GenerateIndexedTrueColumnAvailable<TotalCols,
 * StartRow>>, effectively creating a SparseAvailable type representing a single
 * diagonal column for the specified row.
 *
 * @tparam TotalCols The total number of columns in the matrix (also the number
 * of rows for a square matrix).
 * @tparam StartRow The starting row index for the current block.
 */
template <std::size_t TotalCols, std::size_t StartRow>
struct GenerateDiagColumnsBlock<TotalCols, StartRow, 1> {
  using type = SparseAvailableColumns<
      GenerateIndexedTrueColumnAvailable<TotalCols, StartRow>>;
};

/**
 * @brief Specialization of GenerateDiagColumnsBlock for the case when RowCount
 * is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailableColumns<>, effectively creating an empty SparseAvailable type
 * when there are no rows to process.
 *
 * @tparam TotalCols The total number of columns in the matrix (also the number
 * of rows for a square matrix) (unused in this specialization).
 * @tparam StartRow The starting row index for the current block (unused in this
 * specialization).
 */
template <std::size_t TotalCols, std::size_t StartRow>
struct GenerateDiagColumnsBlock<TotalCols, StartRow, 0> {
  using type = SparseAvailableColumns<>;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to generate a SparseAvailable type representing the
 * diagonal columns of a square matrix of size MxM.
 *
 * This alias uses the GenerateDiagColumnsBlock metafunction to create a
 * SparseAvailable type where the diagonal columns are marked as available
 * (true) and the off-diagonal columns are marked as unavailable (false) for a
 * square matrix of size MxM.
 *
 * @tparam M The number of rows and columns in the square matrix.
 *
 * The resulting type is a SparseAvailable type representing the diagonal
 * columns for a square matrix of size MxM, accessible via the nested ::type
 * member.
 */
template <std::size_t M>
using DiagAvailable =
    typename TemplatesOperation::GenerateDiagColumnsBlock<M, 0, M>::type;

/* Create Sparse Available from Indices and Pointers */

namespace TemplatesOperation {

/* Generate OR of ColumnAvailable */
template <typename ColA, typename ColB> struct GenerateORTrueFlags;

/**
 * @brief Specialization of GenerateORTrueFlags for two ColumnAvailable types.
 *
 * This specialization defines a type alias 'type' that is set to
 * ColumnAvailable<(FlagsA | FlagsB)...>, effectively creating a new
 * ColumnAvailable type where each flag is the logical OR of the corresponding
 * flags from the two input ColumnAvailable types.
 *
 * @tparam FlagsA The boolean flags from the first ColumnAvailable type.
 * @tparam FlagsB The boolean flags from the second ColumnAvailable type.
 */
template <bool... FlagsA, bool... FlagsB>
struct GenerateORTrueFlags<ColumnAvailable<FlagsA...>,
                           ColumnAvailable<FlagsB...>> {

  using type = ColumnAvailable<(FlagsA | FlagsB)...>;
};

/**
 * @brief Alias template to generate a ColumnAvailable type by performing a
 * logical OR operation on two ColumnAvailable types.
 *
 * This alias uses the GenerateORTrueFlags metafunction to create a new
 * ColumnAvailable type where each flag is the logical OR of the corresponding
 * flags from the two input ColumnAvailable types.
 *
 * @tparam ColumnAvailable_A The first ColumnAvailable type.
 * @tparam ColumnAvailable_B The second ColumnAvailable type.
 *
 * The resulting type is a ColumnAvailable type where each flag is the logical
 * OR of the corresponding flags from both input types, accessible via the
 * nested ::type member.
 */
template <typename ColumnAvailable_A, typename ColumnAvailable_B>
using GenerateORTrueFlagsColumnAvailable =
    typename GenerateORTrueFlags<ColumnAvailable_A, ColumnAvailable_B>::type;

/**
 * @brief Template struct to create a block of available columns for a sparse
 * matrix based on given indices.
 *
 * This template recursively constructs a ColumnAvailable type representing the
 * availability of columns for a specific row in a sparse matrix. It divides
 * the range of column indices into two halves to reduce recursion depth and
 * combines the results using GenerateORTrueFlagsColumnAvailable.
 *
 * @tparam CSRIndices The type representing the column indices for the sparse
 * matrix (typically an IndexSequence).
 * @tparam N The total number of columns in the matrix.
 * @tparam StartIndex The starting index for the current block of columns.
 * @tparam Count The number of columns in the current block.
 *
 * The resulting type is accessible via the nested ::type member, which will be
 * a ColumnAvailable type representing the availability of columns for the
 * specified block based on the provided indices.
 */
template <typename CSRIndices, std::size_t N, std::size_t StartIndex,
          std::size_t Count>
struct CreateSparseAvailableRowBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type = GenerateORTrueFlagsColumnAvailable<
      typename CreateSparseAvailableRowBlock<CSRIndices, N, StartIndex,
                                             MidCount>::type,
      typename CreateSparseAvailableRowBlock<
          CSRIndices, N, StartIndex + MidCount, Count - MidCount>::type>;
};

/**
 * @brief Specialization of CreateSparseAvailableRowBlock for the case when
 * Count is 1.
 *
 * This specialization defines a type alias 'type' that is set to
 * GenerateIndexedTrueColumnAvailable<N, CSRIndices::list[StartIndex]>,
 * effectively creating a ColumnAvailable type where the flag is 'true' at the
 * index specified by CSRIndices::list[StartIndex] and 'false' elsewhere.
 *
 * @tparam CSRIndices The type representing the column indices for the sparse
 * matrix (typically an IndexSequence).
 * @tparam N The total number of columns in the matrix.
 * @tparam StartIndex The starting index for the current block of columns.
 */
template <typename CSRIndices, std::size_t N, std::size_t StartIndex>
struct CreateSparseAvailableRowBlock<CSRIndices, N, StartIndex, 1> {

  using type =
      GenerateIndexedTrueColumnAvailable<N, CSRIndices::list[StartIndex]>;
};

/**
 * @brief Specialization of CreateSparseAvailableRowBlock for the case when
 * Count is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * GenerateFalseColumnAvailable<N>, effectively creating a ColumnAvailable type
 * where all flags are 'false' when there are no columns to process.
 *
 * @tparam CSRIndices The type representing the column indices for the sparse
 * matrix (typically an IndexSequence) (unused in this specialization).
 * @tparam N The total number of columns in the matrix.
 * @tparam StartIndex The starting index for the current block of columns
 * (unused in this specialization).
 */
template <typename CSRIndices, std::size_t N, std::size_t StartIndex>
struct CreateSparseAvailableRowBlock<CSRIndices, N, StartIndex, 0> {
  using type = GenerateFalseColumnAvailable<N>;
};

/**
 * @brief Template struct to create a SparseAvailable type representing the
 * available rows in a sparse matrix based on given indices and pointers.
 *
 * This template recursively constructs a SparseAvailable type by processing
 * blocks of rows defined by StartRow and RowCount. It divides the range of rows
 * into two halves to reduce recursion depth and combines the results using
 * ConcatenateSparseAvailable.
 *
 * @tparam N The total number of columns in the matrix.
 * @tparam CSRIndices The type representing the column indices for the sparse
 * matrix (typically an IndexSequence).
 * @tparam CSRPointers The type representing the row pointers for the sparse
 * matrix (typically an IndexSequence).
 * @tparam StartRow The starting row index for the current block of rows.
 * @tparam RowCount The number of rows in the current block.
 *
 * The resulting type is accessible via the nested ::type member, which will be
 * a SparseAvailable type representing the available rows for the specified
 * block based on the provided indices and pointers.
 */
template <std::size_t N, typename CSRIndices, typename CSRPointers,
          std::size_t StartRow, std::size_t RowCount>
struct CreateSparseAvailableRows {
  static constexpr std::size_t Mid = RowCount / 2;

  using type = typename ConcatenateSparseAvailable<
      typename CreateSparseAvailableRows<N, CSRIndices, CSRPointers, StartRow,
                                         Mid>::type,
      typename CreateSparseAvailableRows<N, CSRIndices, CSRPointers,
                                         StartRow + Mid,
                                         RowCount - Mid>::type>::type;
};

/**
 * @brief Specialization of CreateSparseAvailableRows for the case when RowCount
 * is 1.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailableColumns<typename CreateSparseAvailableRowBlock<CSRIndices, N,
 * StartIdx, Count>::type>, effectively creating a SparseAvailable type
 * representing the available columns for a single row based on the provided
 * indices and pointers.
 *
 * @tparam N The total number of columns in the matrix.
 * @tparam CSRIndices The type representing the column indices for the sparse
 * matrix (typically an IndexSequence).
 * @tparam CSRPointers The type representing the row pointers for the sparse
 * matrix (typically an IndexSequence).
 * @tparam StartRow The starting row index for the current block of rows.
 */
template <std::size_t N, typename CSRIndices, typename CSRPointers,
          std::size_t StartRow>
struct CreateSparseAvailableRows<N, CSRIndices, CSRPointers, StartRow, 1> {

  static constexpr std::size_t StartIdx = CSRPointers::list[StartRow];
  static constexpr std::size_t Count =
      CSRPointers::list[StartRow + 1] - StartIdx;

  using type = SparseAvailableColumns<typename CreateSparseAvailableRowBlock<
      CSRIndices, N, StartIdx, Count>::type>;
};

/**
 * @brief Specialization of CreateSparseAvailableRows for the case when RowCount
 * is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailableColumns<>, effectively creating an empty SparseAvailable type
 * when there are no rows to process.
 *
 * @tparam N The total number of columns in the matrix.
 * @tparam CSRIndices The type representing the column indices for the sparse
 * matrix (typically an IndexSequence) (unused in this specialization).
 * @tparam CSRPointers The type representing the row pointers for the sparse
 * matrix (typically an IndexSequence) (unused in this specialization).
 * @tparam StartRow The starting row index for the current block of rows
 * (unused in this specialization).
 */
template <std::size_t N, typename CSRIndices, typename CSRPointers,
          std::size_t StartRow>
struct CreateSparseAvailableRows<N, CSRIndices, CSRPointers, StartRow, 0> {
  using type = SparseAvailableColumns<>;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to create a SparseAvailable type representing the
 * available rows in a sparse matrix based on given indices and pointers.
 *
 * This alias uses the CreateSparseAvailableRows metafunction to generate a
 * SparseAvailable type that indicates which rows are available in a sparse
 * matrix of size MxN, based on the provided column indices and row pointers.
 *
 * @tparam N The total number of columns in the matrix.
 * @tparam CSRIndices The type representing the column indices for the sparse
 * matrix (typically an IndexSequence).
 * @tparam CSRPointers The type representing the row pointers for the sparse
 * matrix (typically an IndexSequence).
 *
 * The resulting type is a SparseAvailable type representing the available rows
 * for the specified matrix, accessible via the nested ::type member.
 */
template <std::size_t N, typename CSRIndices, typename CSRPointers>
using CreateSparseAvailableFromIndicesAndPointers =
    typename TemplatesOperation::CreateSparseAvailableRows<
        N, CSRIndices, CSRPointers, 0, (CSRPointers::size - 1)>::type;

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
 * matrix with N rows.
 *
 * This struct provides a type alias 'type' that is set to IndexSequence<0, 1,
 * ..., N-1>, effectively generating a list of row indices for a matrix with N
 * rows.
 *
 * @tparam N The number of rows in the matrix.
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

/**
 * @brief Specialization of Concatenate for two InvalidSequence types.
 *
 * This specialization defines a type alias 'type' that is set to
 * InvalidSequence<0>, effectively returning an invalid sequence when both
 * sequences are invalid.
 *
 * @tparam Seq1 The first invalid sequence.
 * @tparam Seq2 The second invalid sequence.
 */
template <std::size_t... Seq1, std::size_t... Seq2>
struct Concatenate<InvalidSequence<Seq1...>, InvalidSequence<Seq2...>> {
  using type = InvalidSequence<0>;
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
 * row indices for a matrix with M cols and N rows.
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
 * @brief A template struct to convert an IndexSequence into a CSRIndices type.
 *
 * This struct provides a type alias 'type' that is set to CSRIndices<Seq...>,
 * effectively converting the IndexSequence into a CSRIndices type.
 *
 * @tparam Seq The index sequence to be converted.
 */
template <typename Seq> struct ToCSRIndices;

/**
 * @brief Specialization of ToCSRIndices for IndexSequence types.
 *
 * This specialization defines a type alias 'type' that is set to
 * CSRIndices<Seq...>, effectively converting the IndexSequence into a
 * CSRIndices type.
 *
 * @tparam Seq The index sequence to be converted.
 */
template <std::size_t... Seq> struct ToCSRIndices<IndexSequence<Seq...>> {
  using type = CSRIndices<Seq...>;
};

/**
 * @brief A template struct to generate a sequence of row numbers for a matrix
 * with M cols and N rows.
 *
 * This struct provides a type alias 'type' that is set to a repeated
 * concatenation of MatrixRowNumbers<N> M times, effectively generating a list
 * of row indices for the matrix.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
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
 * a type representing the row indices for a dense matrix with M cols and N
 * rows.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 *
 * The resulting type is an IndexSequence containing the row indices for the
 * dense matrix.
 */
template <std::size_t M, std::size_t N>
using DenseMatrixCSRIndices = typename TemplatesOperation::ToCSRIndices<
    TemplatesOperation::MatrixColumnRowNumbers<M, N>>::type;

namespace TemplatesOperation {

/* Create Dense Matrix Column Pointers */
template <typename Seq1, typename Seq2, std::size_t Offset>
struct MergeAndOffsetSequence;

/**
 * @brief A template struct to merge two index sequences and apply an offset to
 * the second sequence.
 *
 * This struct provides a type alias 'type' that is set to a new IndexSequence
 * containing all indices from the first sequence followed by the indices from
 * the second sequence with an added offset. It is used to generate a sequence
 * of column pointers for a dense matrix.
 *
 * @tparam Seq1 The first index sequence to be merged.
 * @tparam Seq2 The second index sequence to be merged, which will have an
 * offset applied to its indices.
 * @tparam Offset The value to be added to each index in the second sequence.
 *
 * The resulting type is an IndexSequence containing the merged and offset
 * indices.
 */
template <std::size_t... Is1, std::size_t... Is2, std::size_t Offset>
struct MergeAndOffsetSequence<IndexSequence<Is1...>, IndexSequence<Is2...>,
                              Offset> {

  using type = IndexSequence<Is1..., (Is2 + Offset)...>;
};

/**
 * @brief A template struct to create a compile-time index sequence of size
 * Count.
 *
 * This struct recursively generates an IndexSequence of size Count by dividing
 * the range of indices into two halves to reduce recursion depth. It uses
 * MergeAndOffsetSequence to combine the results from the two halves.
 *
 * @tparam Count The size of the index sequence to be generated.
 */
template <std::size_t Count> struct MakeIndexSequenceImpl {
  static constexpr std::size_t Mid = Count / 2;

  using type = typename MergeAndOffsetSequence<
      typename MakeIndexSequenceImpl<Mid>::type,
      typename MakeIndexSequenceImpl<Count - Mid>::type, Mid>::type;
};

/* Specializations for base cases when Count is 0 or 1 */
template <> struct MakeIndexSequenceImpl<1> {
  using type = IndexSequence<0>;
};

/* Specialization for the base case when Count is 0, resulting in an empty
 * sequence. */
template <> struct MakeIndexSequenceImpl<0> {
  using type = IndexSequence<>;
};

/* Multiply each index in the sequence by N */
template <typename Seq, std::size_t N> struct MultiplySequence;

/**
 * @brief Specialization of MultiplySequence for IndexSequence types.
 *
 * This specialization defines a type alias 'type' that is set to
 * IndexSequence<(Is * N)...>, effectively creating a new IndexSequence where
 * each index from the input sequence is multiplied by N.
 *
 * @tparam Is The indices from the input sequence to be multiplied.
 * @tparam N The constant factor by which each index in the sequence will be
 * multiplied.
 */
template <std::size_t... Is, std::size_t N>
struct MultiplySequence<IndexSequence<Is...>, N> {

  using type = IndexSequence<(Is * N)...>;
};

/**
 * @brief A template struct to create a list of pointers for a dense matrix of
 * size MxN.
 *
 * This struct provides a type alias 'type' that is set to the result of
 * multiplying the index sequence generated by MakeIndexSequenceImpl<M + 1> by
 * N, effectively generating a list of pointers for each row in the dense
 * matrix.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 */
template <std::size_t M, std::size_t N> struct MakePointerList {
  using type = typename TemplatesOperation::MultiplySequence<
      typename TemplatesOperation::MakeIndexSequenceImpl<M + 1>::type, N>::type;
};

/**
 * @brief A template struct to create a list of pointers for a dense matrix of
 * size MxN.
 *
 * This struct provides a type alias 'type' that is set to the result of
 * MakePointerList<M, N>::type, effectively generating a list of pointers for
 * each row in the dense matrix.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 */
template <std::size_t M, std::size_t N> struct MatrixDensePointerList {
  using type = typename MakePointerList<M, N>::type;
};

/**
 * @brief A template struct to create a list of row pointers for a dense matrix
 * of size MxN.
 *
 * This struct provides a type alias 'type' that is set to CSRPointers<Seq...>,
 * effectively converting the IndexSequence generated by MakePointerList into a
 * CSRPointers type.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 */
template <std::size_t M, std::size_t N>
using MatrixColumnCSRPointers = typename MatrixDensePointerList<M, N>::type;

/**
 * @brief A template struct to convert an IndexSequence into a CSRPointers type.
 *
 * This struct provides a type alias 'type' that is set to CSRPointers<Seq...>,
 * effectively converting the IndexSequence into a CSRPointers type.
 *
 * @tparam Seq The index sequence to be converted.
 */
template <typename Seq> struct ToCSRPointers;

/**
 * @brief Specialization of ToCSRPointers for IndexSequence types.
 *
 * This specialization defines a type alias 'type' that is set to
 * CSRPointers<Seq...>, effectively converting the IndexSequence into a
 * CSRPointers type.
 *
 * @tparam Seq The index sequence to be converted.
 */
template <std::size_t... Seq>
struct ToCSRPointers<TemplatesOperation::IndexSequence<Seq...>> {
  using type = CSRPointers<Seq...>;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to create a list of row pointers for a dense matrix of
 * size MxN.
 *
 * This alias uses the TemplatesOperation::MatrixColumnCSRPointers to generate
 * a type representing the row pointers for a dense matrix with M cols and N
 * rows.
 *
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 *
 * The resulting type is a CSRPointers type containing the row pointers for the
 * dense matrix.
 */
template <std::size_t M, std::size_t N>
using DenseMatrixCSRPointers = typename TemplatesOperation::ToCSRIndices<
    TemplatesOperation::MatrixColumnCSRPointers<M, N>>::type;

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
 * @tparam N The index of the row to retrieve.
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 */
template <std::size_t N, typename SparseAvailable> struct GetColumnAvailable;

/**
 * @brief Specialization of GetColumnAvailable for SparseAvailable types.
 *
 * This specialization defines a type alias 'type' that is set to the
 * std::tuple_element<N, std::tuple<Columns...>>::type, effectively extracting
 * the ColumnAvailable type at index N from the SparseAvailable type.
 *
 * @tparam N The index of the row to retrieve.
 * @tparam Columns The variadic template parameter pack representing the rows
 * in the SparseAvailable type.
 */
template <std::size_t N, typename... Columns>
struct GetColumnAvailable<N, SparseAvailable<Columns...>> {
  using type = typename std::tuple_element<N, std::tuple<Columns...>>::type;
};

} // namespace TemplatesOperation

/* Concatenate SparseAvailable vertically */

/**
 * @brief Alias template to concatenate two SparseAvailable types vertically.
 *
 * This alias uses the ConcatenateSparseAvailable to combine two SparseAvailable
 * types, effectively merging their rows into a single SparseAvailable type.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 *
 * The resulting type is a SparseAvailable type that contains all rows from
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
horizontally
 * in blocks.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * two SparseAvailable types horizontally in blocks, effectively merging their
 * columns for a specific range of rows.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam StartRow The starting row index for the block to concatenate.
 * @tparam Count The number of rows in the block to concatenate.
 * The resulting type is a SparseAvailable type that contains the concatenated
 * columns for the specified block of rows from both input SparseAvailable
types.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t StartRow, std::size_t Count>
struct ConcatenateSparseAvailableHorizontallyBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type = typename ConcatenateSparseAvailable<
      typename ConcatenateSparseAvailableHorizontallyBlock<
          SparseAvailable_A, SparseAvailable_B, StartRow, MidCount>::type,
      typename ConcatenateSparseAvailableHorizontallyBlock<
          SparseAvailable_A, SparseAvailable_B, StartRow + MidCount,
          Count - MidCount>::type>::type;
};

/**
 * @brief Specialization of ConcatenateSparseAvailableHorizontallyBlock for the
 * case when Count is 1.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailableColumns<typename ConcatenateColumnAvailableLists<
 * typename GetColumnAvailable<StartRow, SparseAvailable_A>::type,
 * typename GetColumnAvailable<StartRow, SparseAvailable_B>::type>::type>,
 * effectively concatenating the ColumnAvailable types from both SparseAvailable
 * types for the specific row index when there is only one row to process.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam StartRow The starting row index for the block to concatenate.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t StartRow>
struct ConcatenateSparseAvailableHorizontallyBlock<
    SparseAvailable_A, SparseAvailable_B, StartRow, 1> {

  using type = SparseAvailableColumns<typename ConcatenateColumnAvailableLists<
      typename GetColumnAvailable<StartRow, SparseAvailable_A>::type,
      typename GetColumnAvailable<StartRow, SparseAvailable_B>::type>::type>;
};

/**
 * @brief Specialization of ConcatenateSparseAvailableHorizontallyBlock for the
 * case when Count is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailableColumns<>, effectively creating an empty SparseAvailable type
 * when there are no rows to process.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam StartRow The starting row index for the block to concatenate.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t StartRow>
struct ConcatenateSparseAvailableHorizontallyBlock<
    SparseAvailable_A, SparseAvailable_B, StartRow, 0> {
  using type = SparseAvailableColumns<>;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to concatenate two SparseAvailable types horizontally.
 *
 * This alias uses the ConcatenateSparseAvailableHorizontallyBlock to combine
 * two SparseAvailable types horizontally, effectively merging their columns for
 * all rows. It specifies the starting row index as 0 and the count as the
 * total number of rows in the first SparseAvailable type.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam StartRow The starting row index for the block to concatenate.
 * @tparam Count The number of rows in the block to concatenate.
 * The resulting type is a SparseAvailable type that contains the concatenated
 * columns for all rows from both input SparseAvailable types.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B>
using ConcatenateSparseAvailableHorizontally =
    typename TemplatesOperation::ConcatenateSparseAvailableHorizontallyBlock<
        SparseAvailable_A, SparseAvailable_B, 0,
        SparseAvailable_A::number_of_rows>::type;

/* Concatenate ColumnAvailable with SparseAvailable  */

namespace TemplatesOperation {

/**
 * @brief A template struct to concatenate a ColumnAvailable type with a
 * SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * a ColumnAvailable type with a SparseAvailable type, effectively merging their
 * rows into a new SparseAvailable type.
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
 * @tparam Columns The variadic template parameter pack representing the rows
 * in the SparseAvailable type.
 */
template <typename Column, typename... Columns>
struct ConcatColumnSparse<Column, SparseAvailable<Columns...>> {
  using type = SparseAvailable<Column, Columns...>;
};

/* Get rest of SparseAvailable */

/**
 * @brief A template struct to get the rest of a SparseAvailable type starting
 * from a specific row index.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * extracting the ColumnAvailable types from the SparseAvailable type starting
 * from the specified row index, effectively creating a new SparseAvailable type
 * that contains only the rows from the specified index onward.
 *
 * @tparam SparseAvailable_In The input SparseAvailable type containing multiple
 * rows.
 * @tparam StartRow The index of the row to start from.
 * @tparam Count The number of rows to include in the resulting SparseAvailable
 * type.
 *
 * The resulting type is a SparseAvailable type that contains only the rows
 * starting from the specified index, accessible via the nested ::type member.
 */
template <typename SparseAvailable_In, std::size_t StartRow, std::size_t Count>
struct GetRestOfSparseAvailableBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type = typename ConcatenateSparseAvailable<
      typename GetRestOfSparseAvailableBlock<SparseAvailable_In, StartRow,
                                             MidCount>::type,
      typename GetRestOfSparseAvailableBlock<SparseAvailable_In,
                                             StartRow + MidCount,
                                             Count - MidCount>::type>::type;
};

/**
 * @brief Specialization of GetRestOfSparseAvailableBlock for the case when
 * Count is 1.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailable<typename GetColumnAvailable<StartRow,
 * SparseAvailable_In>::type>, effectively extracting the ColumnAvailable type
 * at the specified row index and creating a new SparseAvailable type with that
 * single row when there is only one row to process.
 *
 * @tparam SparseAvailable_In The input SparseAvailable type containing multiple
 * rows.
 * @tparam StartRow The index of the row to start from.
 */
template <typename SparseAvailable_In, std::size_t StartRow>
struct GetRestOfSparseAvailableBlock<SparseAvailable_In, StartRow, 1> {

  using type = SparseAvailable<
      typename GetColumnAvailable<StartRow, SparseAvailable_In>::type>;
};

/**
 * @brief Specialization of GetRestOfSparseAvailableBlock for the case when
 * Count is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailable<>, effectively creating an empty SparseAvailable type when
 * there are no rows to process.
 *
 * @tparam SparseAvailable_In The input SparseAvailable type containing multiple
 * rows.
 * @tparam StartRow The index of the row to start from.
 */
template <typename SparseAvailable_In, std::size_t StartRow>
struct GetRestOfSparseAvailableBlock<SparseAvailable_In, StartRow, 0> {

  using type = SparseAvailable<>;
};

/**
 * @brief Alias template to get the rest of a SparseAvailable type starting from
 * a specific row index.
 *
 * This alias uses the GetRestOfSparseAvailableBlock to extract the
 * ColumnAvailable types from the SparseAvailable type starting from the
 * specified row index, effectively creating a new SparseAvailable type that
 * contains only the rows from the specified index onward.
 *
 * @tparam SparseAvailable_In The input SparseAvailable type containing multiple
 * rows.
 * @tparam Row_Index The index of the row to start from.
 *
 * The resulting type is a SparseAvailable type that contains only the rows
 * starting from the specified index, accessible via the nested ::type member.
 */
template <typename SparseAvailable_In, std::size_t Row_Index>
using GetRestOfSparseAvailable =
    typename TemplatesOperation::GetRestOfSparseAvailableBlock<
        SparseAvailable_In, Row_Index,
        (SparseAvailable_In::number_of_rows - Row_Index)>::type;

template <typename SparseAvailable_In, std::size_t StartRow, std::size_t Count>
struct FindFirstNotEmptyColumn {
  static constexpr std::size_t MidCount = Count / 2;

  static constexpr std::size_t LeftResult =
      FindFirstNotEmptyColumn<SparseAvailable_In, StartRow, MidCount>::value;

  template <bool Found, std::size_t Dummy = 0> struct EvaluateRight {
    static constexpr std::size_t value = LeftResult;
  };

  template <std::size_t Dummy> struct EvaluateRight<false, Dummy> {
    static constexpr std::size_t value =
        FindFirstNotEmptyColumn<SparseAvailable_In, StartRow + MidCount,
                                Count - MidCount>::value;
  };

  static constexpr std::size_t value =
      EvaluateRight<(LeftResult != static_cast<std::size_t>(-1))>::value;
};

/**
 * @brief Specialization of FindFirstNotEmptyColumn for the case when Count
 * is 1.
 *
 * This specialization defines a static constexpr boolean 'NotEmpty' that checks
 * if the ColumnAvailable type at the specified row index is not empty. It also
 * defines a static constexpr size_t 'value' that is set to the row index if the
 * column is not empty, or -1 if it is empty.
 *
 * @tparam SparseAvailable_In The input SparseAvailable type containing multiple
 * rows.
 * @tparam StartRow The index of the row to check.
 */
template <typename SparseAvailable_In, std::size_t StartRow>
struct FindFirstNotEmptyColumn<SparseAvailable_In, StartRow, 1> {
  static constexpr bool NotEmpty = CheckSparseAvailableEmpty<
      typename GetColumnAvailable<StartRow, SparseAvailable_In>::type>::value;

  static constexpr std::size_t value =
      NotEmpty ? StartRow : static_cast<std::size_t>(-1);
};

/**
 * @brief Specialization of FindFirstNotEmptyColumn for the case when Count is
 * 0.
 *
 * This specialization defines a static constexpr size_t 'value' that is set to
 * -1, effectively indicating that there are no non-empty columns to be found
 * when there are no rows to check.
 *
 * @tparam SparseAvailable_In The input SparseAvailable type containing multiple
 * rows.
 * @tparam StartRow The index of the row to check.
 */
template <typename SparseAvailable_In, std::size_t StartRow>
struct FindFirstNotEmptyColumn<SparseAvailable_In, StartRow, 0> {
  static constexpr std::size_t value = static_cast<std::size_t>(-1);
};

/**
 * @brief A template struct to apply the result of FindFirstNotEmptyColumn to
 * get the rest of a SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is set to the result of
 * GetRestOfSparseAvailable if a non-empty column is found, or an empty
 * SparseAvailable type if no non-empty columns are found.
 *
 * @tparam SparseAvailable_In The input SparseAvailable type containing multiple
 * rows.
 * @tparam FoundRow The index of the first non-empty column found, or -1 if no
 * non-empty columns are found.
 *
 * The resulting type is a SparseAvailable type that contains only the rows
 * starting from the first non-empty column, or an empty SparseAvailable type
 * if no non-empty columns are found.
 */
template <typename SparseAvailable_In, std::size_t FoundRow>
struct ApplyRestOfSparseAvailable {
  using type = GetRestOfSparseAvailable<SparseAvailable_In, FoundRow>;
};

/**
 * @brief Specialization of ApplyRestOfSparseAvailable for the case when no
 * non-empty columns are found.
 *
 * This specialization defines a type alias 'type' that is set to
 * SparseAvailable<>, effectively creating an empty SparseAvailable type when no
 * non-empty columns are found.
 *
 * @tparam SparseAvailable_In The input SparseAvailable type containing multiple
 * rows.
 */
template <typename SparseAvailable_In>
struct ApplyRestOfSparseAvailable<SparseAvailable_In,
                                  static_cast<std::size_t>(-1)> {
  using type = SparseAvailable<>;
};

/**
 * @brief Alias template to avoid empty columns in a SparseAvailable type.
 *
 * This alias uses the ApplyRestOfSparseAvailable to extract the non-empty
 * columns from the SparseAvailable type, effectively creating a new
 * SparseAvailable type that contains only the non-empty columns.
 *
 * @tparam SparseAvailable_In The input SparseAvailable type containing multiple
 * rows.
 *
 * The resulting type is a SparseAvailable type that contains only the non-empty
 * columns from the input SparseAvailable type, accessible via the nested ::type
 * member.
 */
template <typename SparseAvailable_In>
using AvoidEmptyColumnsSparseAvailable = typename ApplyRestOfSparseAvailable<
    SparseAvailable_In,
    FindFirstNotEmptyColumn<SparseAvailable_In, 0,
                            SparseAvailable_In::number_of_rows>::value>::type;

/* Create Row Indices */

/**
 * @brief A template struct to assign row indices for a block of rows in a
 * sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating row indices for a block of rows in the SparseAvailable type,
 * effectively creating a sequence of row indices for the specified range of
 * rows.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple rows.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam StartRow The starting row index for the block to assign.
 * @tparam Count The number of rows in the block to assign.
 *
 * The resulting type is an IndexSequence containing the row indices for the
 * specified block of rows, accessible via the nested ::type member.
 */
template <typename SparseAvailable_In, std::size_t ColumnElementNumber,
          std::size_t StartRow, std::size_t Count>
struct AssignSparseMatrixRowBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type = typename Concatenate<
      typename AssignSparseMatrixRowBlock<
          SparseAvailable_In, ColumnElementNumber, StartRow, MidCount>::type,
      typename AssignSparseMatrixRowBlock<
          SparseAvailable_In, ColumnElementNumber, StartRow + MidCount,
          Count - MidCount>::type>::type;
};

/**
 * @brief Specialization of AssignSparseMatrixRowBlock for the case when Count
 * is 1.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * AssignSparseMatrixRowSingle, effectively generating a single row index for
 * the specified row when there is only one row to process.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple rows.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam StartRow The starting row index for the block to assign.
 */
template <typename SparseAvailable_In, std::size_t ColumnElementNumber,
          std::size_t StartRow, bool Active>
struct AssignSparseMatrixRowSingle;

/**
 * @brief Specialization of AssignSparseMatrixRowSingle for the case when the
 * column is active.
 *
 * This specialization defines a type alias 'type' that is set to
 * IndexSequence<StartRow>, effectively generating a single row index for the
 * specified row when the column is active.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple rows.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam StartRow The starting row index for the block to assign.
 */
template <typename SparseAvailable_In, std::size_t ColumnElementNumber,
          std::size_t StartRow>
struct AssignSparseMatrixRowSingle<SparseAvailable_In, ColumnElementNumber,
                                   StartRow, true> {
  using type = IndexSequence<StartRow>;
};

/**
 * @brief Specialization of AssignSparseMatrixRowSingle for the case when the
 * column is not active.
 *
 * This specialization defines a type alias 'type' that is set to
 * InvalidSequence<0>, effectively indicating that there are no valid row
 * indices to return for the specified row when the column is not active.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple rows.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam StartRow The starting row index for the block to assign.
 */
template <typename SparseAvailable_In, std::size_t ColumnElementNumber,
          std::size_t StartRow>
struct AssignSparseMatrixRowSingle<SparseAvailable_In, ColumnElementNumber,
                                   StartRow, false> {
  using type = InvalidSequence<0>;
};

/**
 * @brief Specialization of AssignSparseMatrixRowBlock for the case when Count
 * is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * InvalidSequence<0>, effectively indicating that there are no valid row
 * indices to return when there are no rows to process.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple rows.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam StartRow The starting row index for the block to assign.
 */
template <typename SparseAvailable_In, std::size_t ColumnElementNumber,
          std::size_t StartRow>
struct AssignSparseMatrixRowBlock<SparseAvailable_In, ColumnElementNumber,
                                  StartRow, 1> {
  using type = typename AssignSparseMatrixRowSingle<
      SparseAvailable_In, ColumnElementNumber, StartRow,
      SparseAvailable_In::lists[ColumnElementNumber][StartRow]>::type;
};

/**
 * @brief Specialization of AssignSparseMatrixRowBlock for the case when Count
 * is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * InvalidSequence<0>, effectively indicating that there are no valid row
 * indices to return when there are no rows to process.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple rows.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam StartRow The starting row index for the block to assign.
 */
template <typename SparseAvailable_In, std::size_t ColumnElementNumber,
          std::size_t StartRow>
struct AssignSparseMatrixRowBlock<SparseAvailable_In, ColumnElementNumber,
                                  StartRow, 0> {
  using type = InvalidSequence<0>;
};

/**
 * @brief A template struct to assign row indices for a block of columns in a
 * sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating row indices for a block of columns in the SparseAvailable type,
 * effectively creating a sequence of row indices for the specified range of
 * columns.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple rows.
 * @tparam StartColumn The starting column index for the block to assign.
 * @tparam Count The number of columns in the block to assign.
 *
 * The resulting type is an IndexSequence containing the row indices for the
 * specified block of columns, accessible via the nested ::type member.
 */
template <typename SparseAvailable_In, std::size_t StartColumn,
          std::size_t Count>
struct AssignSparseMatrixColumnBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type =
      typename Concatenate<typename AssignSparseMatrixColumnBlock<
                               SparseAvailable_In, StartColumn, MidCount>::type,
                           typename AssignSparseMatrixColumnBlock<
                               SparseAvailable_In, StartColumn + MidCount,
                               Count - MidCount>::type>::type;
};

/**
 * @brief Specialization of AssignSparseMatrixColumnBlock for the case when
 * Count is 1.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * AssignSparseMatrixRowBlock for the specified column index, effectively
 * generating row indices for that single column when there is only one column
 * to process.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple rows.
 * @tparam StartColumn The starting column index for the block to assign.
 */
template <typename SparseAvailable_In, std::size_t StartColumn>
struct AssignSparseMatrixColumnBlock<SparseAvailable_In, StartColumn, 1> {

  using type =
      typename AssignSparseMatrixRowBlock<SparseAvailable_In, StartColumn, 0,
                                          SparseAvailable_In::row_size>::type;
};

/**
 * @brief Specialization of AssignSparseMatrixColumnBlock for the case when
 * Count is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * InvalidSequence<0>, effectively indicating that there are no valid row
 * indices to return when there are no columns to process.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple rows.
 * @tparam StartColumn The starting column index for the block to assign.
 */
template <typename SparseAvailable_In, std::size_t StartColumn>
struct AssignSparseMatrixColumnBlock<SparseAvailable_In, StartColumn, 0> {

  using type = InvalidSequence<0>;
};

/**
 * @brief A template struct to assign row indices for all columns in a sparse
 * matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating row indices for all columns in the SparseAvailable type,
 * effectively creating a sequence of row indices for the entire matrix.
 *
 * @tparam SparseAvailable_In The SparseAvailable type containing multiple rows.
 * @tparam ColumnElementNumber The index of the column being processed.
 *
 * The resulting type is an IndexSequence containing the row indices for all
 * columns in the SparseAvailable type, accessible via the nested ::type member.
 */
template <typename SparseAvailable_In, std::size_t ColumnElementNumber>
struct AssignSparseMatrixColumnLoop {

  using type =
      typename AssignSparseMatrixColumnBlock<SparseAvailable_In, 0,
                                             ColumnElementNumber + 1>::type;
};

/**
 * @brief A template struct to generate a sequence of row indices from a
 * SparseAvailable type, taking into account whether it is empty or not.
 *
 * This struct provides a type alias 'type' that is set to the result of
 * AssignSparseMatrixColumnLoop if the SparseAvailable type is not empty, or an
 * IndexSequence<0> if it is empty, effectively generating a sequence of row
 * indices for the SparseAvailable type while handling the case of an empty
 * matrix.
 *
 * @tparam SparseAvailable_In The input SparseAvailable type containing multiple
 * rows.
 * @tparam NotEmpty A boolean indicating whether the SparseAvailable type is
 * empty or not.
 *
 * The resulting type is an IndexSequence containing the row indices for the
 * SparseAvailable type if it is not empty, or an IndexSequence<0> if it is
 * empty, accessible via the nested ::type member.
 */
template <typename SparseAvailable_In, bool NotEmpty>
struct CSRIndicesSequenceFromSparseAvailable;

/**
 * @brief Specialization of CSRIndicesSequenceFromSparseAvailable for the case
 * when the SparseAvailable type is not empty.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * AssignSparseMatrixColumnLoop for the non-empty SparseAvailable type,
 * effectively generating a sequence of row indices for the SparseAvailable type
 * when it is not empty.
 *
 * @tparam SparseAvailable_In The input SparseAvailable type containing multiple
 * rows.
 */
template <typename SparseAvailable_In>
struct CSRIndicesSequenceFromSparseAvailable<SparseAvailable_In, true> {
  using type = typename AssignSparseMatrixColumnLoop<
      AvoidEmptyColumnsSparseAvailable<SparseAvailable_In>,
      (AvoidEmptyColumnsSparseAvailable<SparseAvailable_In>::number_of_rows -
       1)>::type;
};

/**
 * @brief Specialization of CSRIndicesSequenceFromSparseAvailable for the case
 * when the SparseAvailable type is empty.
 *
 * This specialization defines a type alias 'type' that is set to
 * IndexSequence<0>, effectively indicating that there are no valid row indices
 * to return when the SparseAvailable type is empty.
 *
 * @tparam SparseAvailable_In The input SparseAvailable type containing multiple
 * rows.
 */
template <typename SparseAvailable_In>
struct CSRIndicesSequenceFromSparseAvailable<SparseAvailable_In, false> {
  using type = IndexSequence<0>;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to create a sequence of row indices from a
 * SparseAvailable type.
 *
 * This alias uses the TemplatesOperation::CSRIndicesSequenceFromSparseAvailable
 * to generate a type representing the row indices for the SparseAvailable type,
 * taking into account whether it is empty or not.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 *
 * The resulting type is an IndexSequence containing the row indices for the
 * SparseAvailable type.
 */
template <typename SparseAvailable_In>
using CSRIndicesFromSparseAvailable = typename TemplatesOperation::ToCSRIndices<
    typename TemplatesOperation::CSRIndicesSequenceFromSparseAvailable<
        SparseAvailable_In, TemplatesOperation::CheckSparseAvailableEmpty<
                                SparseAvailable_In>::value>::type>::type;

/* Create Row Pointers */

namespace TemplatesOperation {

/**
 * @brief A template struct to count the number of elements in a block of rows
 * for a specific column in a sparse matrix.
 *
 * This struct provides a static constexpr size_t 'value' that is the result
 * of recursively counting the number of elements in a block of rows for a
 * specific column in the SparseAvailable type, effectively calculating the
 * number of non-empty entries for that column within the specified range of
 * rows.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam StartRow The starting row index for the block to count.
 * @tparam Count The number of rows in the block to count.
 * The resulting value is the total count of non-empty entries for the specified
 * column within the specified range of rows.
 */
template <typename SparseAvailable, std::size_t ColumnElementNumber,
          std::size_t StartRow, std::size_t Count>
struct CountSparseMatrixRowBlock {
  static constexpr std::size_t MidCount = Count / 2;

  static constexpr std::size_t value =
      CountSparseMatrixRowBlock<SparseAvailable, ColumnElementNumber, StartRow,
                                MidCount>::value +
      CountSparseMatrixRowBlock<SparseAvailable, ColumnElementNumber,
                                StartRow + MidCount, Count - MidCount>::value;
};

/**
 * @brief Specialization of CountSparseMatrixRowBlock for the case when Count is
 * 1.
 *
 * This specialization defines a static constexpr size_t 'value' that checks if
 * the entry at the specified column and row index in the SparseAvailable type
 * is non-empty, returning 1 if it is non-empty and 0 if it is empty.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam StartRow The starting row index for the block to count.
 */
template <typename SparseAvailable, std::size_t ColumnElementNumber,
          std::size_t StartRow>
struct CountSparseMatrixRowBlock<SparseAvailable, ColumnElementNumber, StartRow,
                                 1> {

  static constexpr std::size_t value =
      SparseAvailable::lists[ColumnElementNumber][StartRow] ? 1 : 0;
};

/**
 * @brief Specialization of CountSparseMatrixRowBlock for the case when Count is
 * 0.
 *
 * This specialization defines a static constexpr size_t 'value' that is always
 * 0, indicating that there are no elements in the specified block.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam StartRow The starting row index for the block to count.
 */
template <typename SparseAvailable, std::size_t ColumnElementNumber,
          std::size_t StartRow>
struct CountSparseMatrixRowBlock<SparseAvailable, ColumnElementNumber, StartRow,
                                 0> {
  static constexpr std::size_t value = 0;
};

/**
 * @brief A template struct to count the number of elements in a block of
 * columns for a specific row in a sparse matrix.
 *
 * This struct provides a static constexpr size_t 'AddedCount' that is the
 * result of recursively counting the number of elements in a block of columns
 * for a specific row in the SparseAvailable type, effectively calculating the
 * total count of non-empty entries for that row across the specified range of
 * columns. It also provides a type alias 'type' that is an IndexSequence with
 * the total count added to the existing ElementCount.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 * @tparam ElementCount The current count of elements before adding the count
 * for the current block of columns.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam RowElementNumber The index of the row being processed.
 *
 * The resulting value is the total count of non-empty entries for the specified
 * row across the specified range of columns, and the resulting type is an
 * IndexSequence containing that total count added to the existing ElementCount.
 */
template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber, std::size_t RowElementNumber>
struct CountSparseMatrixRowLoop {

  static constexpr std::size_t AddedCount =
      CountSparseMatrixRowBlock<SparseAvailable, ColumnElementNumber, 0,
                                RowElementNumber + 1>::value;

  using type = IndexSequence<ElementCount + AddedCount>;
};

/**
 * @brief A template struct to count the number of elements in a block of
 * columns for a specific row in a sparse matrix.
 *
 * This struct provides a static constexpr size_t 'value' that is the result of
 * recursively counting the number of elements in a block of columns for a
 * specific row in the SparseAvailable type, effectively calculating the total
 * count of non-empty entries for that row across the specified range of
 * columns.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 * @tparam ColumnElementNumber The index of the column being processed.
 * @tparam RowElementNumber The index of the row being processed.
 *
 * The resulting value is the total count of non-empty entries for the specified
 * row across the specified range of columns.
 */
template <typename SparseAvailable, std::size_t StartColumn, std::size_t Count>
struct CountSparseMatrixColumnBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type =
      typename Concatenate<typename CountSparseMatrixColumnBlock<
                               SparseAvailable, StartColumn, MidCount>::type,
                           typename CountSparseMatrixColumnBlock<
                               SparseAvailable, StartColumn + MidCount,
                               Count - MidCount>::type>::type;
};

/**
 * @brief Specialization of CountSparseMatrixColumnBlock for the case when Count
 * is 1.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * CountSparseMatrixRowLoop for the specified column index, effectively
 * generating the count of non-empty entries for that column across all rows
 * when there is only one column to process.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 * @tparam StartColumn The starting column index for the block to count.
 */
template <typename SparseAvailable, std::size_t StartColumn>
struct CountSparseMatrixColumnBlock<SparseAvailable, StartColumn, 1> {

  using type =
      typename CountSparseMatrixRowLoop<SparseAvailable, 0, StartColumn,
                                        (SparseAvailable::row_size - 1)>::type;
};

/**
 * @brief Specialization of CountSparseMatrixColumnBlock for the case when Count
 * is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * InvalidSequence<0>, effectively indicating that there are no valid counts to
 * return when there are no columns to process.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 * @tparam StartColumn The starting column index for the block to count.
 */
template <typename SparseAvailable, std::size_t StartColumn>
struct CountSparseMatrixColumnBlock<SparseAvailable, StartColumn, 0> {

  using type = InvalidSequence<0>;
};

/**
 * @brief A template struct to count the number of elements in all columns for
 * all rows in a sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * counting the number of elements in all columns for all rows in the
 * SparseAvailable type, effectively creating a sequence of counts for each
 * column across all rows in the matrix.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 * @tparam ColumnElementNumber The index of the column being processed.
 *
 * The resulting type is an IndexSequence containing the counts of non-empty
 * entries for each column across all rows in the SparseAvailable type,
 * accessible via the nested ::type member.
 */
template <typename SparseAvailable, std::size_t ColumnElementNumber>
struct CountSparseMatrixColumnLoop {

  using type = typename Concatenate<
      IndexSequence<0>,
      typename CountSparseMatrixColumnBlock<
          SparseAvailable, 0, ColumnElementNumber + 1>::type>::type;
};

/**
 * @brief A template struct to accumulate the number of elements in a block of
 * columns for a specific row in a sparse matrix.
 *
 * This struct provides a static constexpr size_t 'value' that is the result of
 * recursively accumulating the number of elements in a block of columns for a
 * specific row in the SparseAvailable type, effectively calculating the total
 * count of non-empty entries for that row across the specified range of
 * columns.
 *
 * @tparam CountSparseMatrixColumnLoop The type containing the counts for each
 * column across all rows in the SparseAvailable type.
 * @tparam StartIndex The starting index for the block to accumulate.
 * @tparam Count The number of columns in the block to accumulate.
 *
 * The resulting value is the total count of non-empty entries for the specified
 * row across the specified range of columns.
 */
template <typename CountSparseMatrixColumnLoop, std::size_t StartIndex,
          std::size_t Count>
struct AccumulateElementNumberBlock {
  static constexpr std::size_t MidCount = Count / 2;

  static constexpr std::size_t compute() {
    return AccumulateElementNumberBlock<CountSparseMatrixColumnLoop, StartIndex,
                                        MidCount>::compute() +
           AccumulateElementNumberBlock<CountSparseMatrixColumnLoop,
                                        StartIndex + MidCount,
                                        Count - MidCount>::compute();
  }
};

/**
 * @brief Specialization of AccumulateElementNumberBlock for the case when Count
 * is 1.
 *
 * This specialization defines a static constexpr size_t 'value' that is set to
 * the count at the specified index in the CountSparseMatrixColumnLoop type,
 * effectively returning the count for that specific column when there is only
 * one column to process.
 *
 * @tparam CountSparseMatrixColumnLoop The type containing the counts for each
 * column across all rows in the SparseAvailable type.
 * @tparam StartIndex The starting index for the block to accumulate.
 */
template <typename CountSparseMatrixColumnLoop, std::size_t StartIndex>
struct AccumulateElementNumberBlock<CountSparseMatrixColumnLoop, StartIndex,
                                    1> {
  static constexpr std::size_t compute() {
    return CountSparseMatrixColumnLoop::list[StartIndex];
  }
};

/**
 * @brief Specialization of AccumulateElementNumberBlock for the case when Count
 * is 0.
 *
 * This specialization defines a static constexpr size_t 'value' that is always
 * 0, indicating that there are no elements to accumulate when there are no
 * columns to process.
 *
 * @tparam CountSparseMatrixColumnLoop The type containing the counts for each
 * column across all rows in the SparseAvailable type.
 * @tparam StartIndex The starting index for the block to accumulate.
 */
template <typename CountSparseMatrixColumnLoop, std::size_t StartIndex>
struct AccumulateElementNumberBlock<CountSparseMatrixColumnLoop, StartIndex,
                                    0> {
  static constexpr std::size_t compute() { return 0; }
};

/**
 * @brief A template struct to accumulate the number of elements in all columns
 * for a specific row in a sparse matrix.
 *
 * This struct provides a static constexpr size_t 'value' that is the result of
 * recursively accumulating the number of elements in all columns for a specific
 * row in the SparseAvailable type, effectively calculating the total count of
 * non-empty entries for that row across all columns in the matrix.
 *
 * @tparam CountSparseMatrixColumnLoop The type containing the counts for each
 * column across all rows in the SparseAvailable type.
 * @tparam ColumnElementNumber The index of the column being processed.
 *
 * The resulting value is the total count of non-empty entries for the specified
 * row across all columns in the SparseAvailable type.
 */
template <typename CountSparseMatrixColumnLoop, std::size_t ColumnElementNumber>
struct AccumulateElementNumberLoop {

  static constexpr std::size_t compute() {
    return AccumulateElementNumberBlock<CountSparseMatrixColumnLoop, 1,
                                        ColumnElementNumber>::compute();
  }
};

/**
 * @brief A template struct to accumulate the number of elements in all columns
 * for all rows in a sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * accumulating the number of elements in all columns for all rows in the
 * SparseAvailable type, effectively creating a sequence of accumulated counts
 * for each column across all rows in the matrix.
 *
 * @tparam CountSparseMatrixColumnLoop The type containing the counts for each
 * column across all rows in the SparseAvailable type.
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 *
 * The resulting type is an IndexSequence containing the accumulated counts of
 * non-empty entries for each column across all rows in the SparseAvailable
 * type, accessible via the nested ::type member.
 */
template <typename CountSparseMatrixColumnLoop, std::size_t StartColumn,
          std::size_t Count>
struct AccumulateSparseMatrixElementNumberBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type = typename Concatenate<
      typename AccumulateSparseMatrixElementNumberBlock<
          CountSparseMatrixColumnLoop, StartColumn, MidCount>::type,
      typename AccumulateSparseMatrixElementNumberBlock<
          CountSparseMatrixColumnLoop, StartColumn + MidCount,
          Count - MidCount>::type>::type;
};

/**
 * @brief Specialization of AccumulateSparseMatrixElementNumberBlock for the
 * case when Count is 1.
 *
 * This specialization defines a type alias 'type' that is set to an
 * IndexSequence containing the result of AccumulateElementNumberLoop for the
 * specified column index, effectively returning the accumulated count for that
 * specific column when there is only one column to process.
 *
 * @tparam CountSparseMatrixColumnLoop The type containing the counts for each
 * column across all rows in the SparseAvailable type.
 * @tparam StartColumn The starting index for the block to accumulate.
 */
template <typename CountSparseMatrixColumnLoop, std::size_t StartColumn>
struct AccumulateSparseMatrixElementNumberBlock<CountSparseMatrixColumnLoop,
                                                StartColumn, 1> {

  using type =
      IndexSequence<AccumulateElementNumberLoop<CountSparseMatrixColumnLoop,
                                                StartColumn>::compute()>;
};

/**
 * @brief Specialization of AccumulateSparseMatrixElementNumberBlock for the
 * case when Count is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * InvalidSequence<0>, effectively indicating that there are no valid counts to
 * return when there are no columns to process.
 *
 * @tparam CountSparseMatrixColumnLoop The type containing the counts for each
 * column across all rows in the SparseAvailable type.
 * @tparam StartColumn The starting index for the block to accumulate.
 */
template <typename CountSparseMatrixColumnLoop>
struct AccumulateSparseMatrixElementNumberBlock<CountSparseMatrixColumnLoop, 0,
                                                1> {

  using type = IndexSequence<CountSparseMatrixColumnLoop::list[0]>;
};

/**
 * @brief Specialization of AccumulateSparseMatrixElementNumberBlock for the
 * case when Count is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * InvalidSequence<0>, effectively indicating that there are no valid counts to
 * return when there are no columns to process.
 *
 * @tparam CountSparseMatrixColumnLoop The type containing the counts for each
 * column across all rows in the SparseAvailable type.
 */
template <typename CountSparseMatrixColumnLoop, std::size_t StartColumn>
struct AccumulateSparseMatrixElementNumberBlock<CountSparseMatrixColumnLoop,
                                                StartColumn, 0> {
  using type = InvalidSequence<0>;
};

/**
 * @brief A template struct to accumulate the number of elements in all columns
 * for all rows in a sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * accumulating the number of elements in all columns for all rows in the
 * SparseAvailable type, effectively creating a sequence of accumulated counts
 * for each column across all rows in the matrix.
 *
 * @tparam CountSparseMatrixColumnLoop The type containing the counts for each
 * column across all rows in the SparseAvailable type.
 * @tparam ColumnElementNumber The index of the column being processed.
 *
 * The resulting type is an IndexSequence containing the accumulated counts of
 * non-empty entries for each column across all rows in the SparseAvailable
 * type, accessible via the nested ::type member.
 */
template <typename CountSparseMatrixColumnLoop, std::size_t ColumnElementNumber>
struct AccumulateSparseMatrixElementNumberLoop {

  using type = typename AccumulateSparseMatrixElementNumberBlock<
      CountSparseMatrixColumnLoop, 0, ColumnElementNumber + 1>::type;
};

/**
 * @brief A template struct to accumulate the number of elements in all columns
 * for all rows in a sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * accumulating the number of elements in all columns for all rows in the
 * SparseAvailable type, effectively creating a sequence of accumulated counts
 * for each column across all rows in the matrix.
 *
 * @tparam CountSparseMatrixColumnLoop The type containing the counts for each
 * column across all rows in the SparseAvailable type.
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 *
 * The resulting type is an IndexSequence containing the accumulated counts of
 * non-empty entries for each column across all rows in the SparseAvailable
 * type, accessible via the nested ::type member.
 */
template <typename CountSparseMatrixColumnLoop, typename SparseAvailable>
struct AccumulateSparseMatrixElementNumberStruct {
  using type = typename AccumulateSparseMatrixElementNumberLoop<
      CountSparseMatrixColumnLoop, SparseAvailable::number_of_rows>::type;
};

} // namespace TemplatesOperation

/**
 * @brief Alias template to create row pointers from a SparseAvailable type.
 *
 * This alias uses the
 * TemplatesOperation::AccumulateSparseMatrixElementNumberStruct to generate a
 * type representing the row pointers for the SparseAvailable type.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 *
 * The resulting type is a CSRPointers type that contains the accumulated number
 * of elements in each row of the SparseAvailable type.
 */
template <typename SparseAvailable>
using CSRPointersFromSparseAvailable =
    typename TemplatesOperation::ToCSRPointers<
        typename TemplatesOperation::AccumulateSparseMatrixElementNumberStruct<
            typename TemplatesOperation::CountSparseMatrixColumnLoop<
                SparseAvailable, (SparseAvailable::number_of_rows - 1)>::type,
            SparseAvailable>::type>::type;

namespace TemplatesOperation {

/* Sequence for Triangular */

/**
 * @brief A template struct to create a triangular index block.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating a triangular index block for a given starting index and count,
 * effectively creating a sequence of indices that form a triangular pattern.
 *
 * @tparam SeqStart The starting index for the block.
 * @tparam Count The number of indices in the block.
 *
 * The resulting type is an IndexSequence containing the indices for the
 * triangular block, accessible via the nested ::type member.
 */
template <std::size_t SeqStart, std::size_t Count>
struct MakeTriangularIndexBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type = typename Concatenate<
      typename MakeTriangularIndexBlock<SeqStart, MidCount>::type,
      typename MakeTriangularIndexBlock<SeqStart + MidCount,
                                        Count - MidCount>::type>::type;
};

/**
 * @brief Specialization of MakeTriangularIndexBlock for the case when Count is
 * 1.
 *
 * This specialization defines a type alias 'type' that is set to an
 * IndexSequence containing the SeqStart index, effectively creating a
 * triangular index block for that single index when there is only one index to
 * process.
 *
 * @tparam SeqStart The starting index for the block.
 */
template <std::size_t SeqStart> struct MakeTriangularIndexBlock<SeqStart, 1> {
  using type = IndexSequence<SeqStart>;
};

/**
 * @brief Specialization of MakeTriangularIndexBlock for the case when Count is
 * 0.
 *
 * This specialization defines a type alias 'type' that is set to an
 * InvalidSequence, effectively creating a triangular index block for that
 * single index when there are no indices to process.
 *
 * @tparam SeqStart The starting index for the block.
 */
template <std::size_t SeqStart> struct MakeTriangularIndexBlock<SeqStart, 0> {

  using type = InvalidSequence<0>;
};

/**
 * @brief A template struct to create a triangular index sequence.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * triangular index block for a given range, starting from Start and ending at
 * End, with an element size of E_S.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 * @tparam E_S The current size of the sequence.
 *
 * The resulting type is an IndexSequence containing the indices for the
 * triangular sequence, accessible via the nested ::type member.
 */
template <std::size_t Start, std::size_t End, std::size_t E_S>
struct MakeTriangularIndexSequence {

  using type =
      typename MakeTriangularIndexBlock<(End - E_S - 1), (E_S + 1)>::type;
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
 *
 * The resulting type is an IndexSequence containing the indices for the
 * triangular sequence, accessible via the nested ::type member.
 */
template <std::size_t Start, std::size_t End> struct TriangularSequenceList {
  using type =
      typename MakeTriangularIndexSequence<Start, End, (End - Start)>::type;
};

/* Count for Triangular */

/**
 * @brief A template struct to create a triangular count block.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating a triangular count block for a given ending index and count,
 * effectively creating a sequence of counts that form a triangular pattern.
 *
 * @tparam End The ending index for the block.
 * @tparam Count The number of counts in the block.
 *
 * The resulting type is an IndexSequence containing the counts for the
 * triangular block, accessible via the nested ::type member.
 */
template <std::size_t End, std::size_t Count> struct MakeTriangularCountBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type = typename Concatenate<
      typename MakeTriangularCountBlock<End, MidCount>::type,
      typename MakeTriangularCountBlock<End - MidCount,
                                        Count - MidCount>::type>::type;
};

/**
 * @brief Specialization of MakeTriangularCountBlock for the case when Count is
 * 1.
 *
 * This specialization defines a type alias 'type' that is set to an
 * IndexSequence containing the End index, effectively creating a triangular
 * count block for that single index when there is only one count to process.
 *
 * @tparam End The ending index for the block.
 */
template <std::size_t End> struct MakeTriangularCountBlock<End, 1> {
  using type = IndexSequence<End>;
};

/**
 * @brief Specialization of MakeTriangularCountBlock for the case when Count is
 * 0.
 *
 * This specialization defines a type alias 'type' that is set to an
 * InvalidSequence, effectively creating a triangular count block for that
 * single index when there are no counts to process.
 *
 * @tparam End The ending index for the block.
 */
template <std::size_t End> struct MakeTriangularCountBlock<End, 0> {

  using type = InvalidSequence<0>;
};

/**
 * @brief A template struct to create a triangular count sequence.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * triangular count block for a given range, starting from Start and ending at
 * End, with an element size of E_S.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 * @tparam E_S The current size of the sequence.
 *
 * The resulting type is an IndexSequence containing the counts for the
 * triangular sequence, accessible via the nested ::type member.
 */
template <std::size_t Start, std::size_t End, std::size_t E_S>
struct MakeTriangularCountSequence {

  using type = typename MakeTriangularCountBlock<End, E_S + 1>::type;
};

/**
 * @brief A template struct to create a triangular count sequence list.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * triangular count sequence for a given range, starting from Start and ending
 * at End.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 *
 * The resulting type is an IndexSequence containing the counts for the
 * triangular sequence, accessible via the nested ::type member.
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
 * @brief A template struct to concatenate upper triangular row numbers for a
 * specific block of rows in a sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * concatenating upper triangular row numbers for a specific block of rows in
 * the matrix, effectively creating a sequence of row indices that correspond to
 * the upper triangular part of the matrix for the specified range of rows.
 *
 * @tparam Start The starting index of the block of rows.
 * @tparam Count The number of rows in the block.
 * @tparam N The total number of columns in the matrix.
 *
 * The resulting type is an IndexSequence containing the row indices for the
 * upper triangular part of the matrix for the specified range of rows,
 * accessible via the nested ::type member.
 */
template <std::size_t Start, std::size_t Count, std::size_t N>
struct ConcatenateUpperTriangularRowNumbersBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type = typename Concatenate<
      typename ConcatenateUpperTriangularRowNumbersBlock<Start, MidCount,
                                                         N>::type,
      typename ConcatenateUpperTriangularRowNumbersBlock<
          Start + MidCount, Count - MidCount, N>::type>::type;
};

/**
 * @brief Specialization of ConcatenateUpperTriangularRowNumbersBlock for the
 * case when Count is 1.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * TriangularSequenceList for the specified starting index and total number of
 * columns, effectively creating a sequence of row indices for the upper
 * triangular part of the matrix for that single row when there is only one row
 * to process.
 *
 * @tparam Start The starting index of the block of rows.
 * @tparam N The total number of columns in the matrix.
 */
template <std::size_t Start, std::size_t N>
struct ConcatenateUpperTriangularRowNumbersBlock<Start, 1, N> {

  using type = typename TriangularSequenceList<Start, N>::type;
};

/**
 * @brief Specialization of ConcatenateUpperTriangularRowNumbersBlock for the
 * case when Count is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * InvalidSequence<0>, effectively indicating that there are no valid row
 * indices to return when there are no rows to process.
 *
 * @tparam Start The starting index of the block of rows.
 * @tparam N The total number of columns in the matrix.
 */
template <std::size_t Start, std::size_t N>
struct ConcatenateUpperTriangularRowNumbersBlock<Start, 0, N> {

  using type = InvalidSequence<0>;
};

/**
 * @brief A template struct to concatenate upper triangular row numbers for a
 * specific range of rows in a sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * concatenated sequence of upper triangular row numbers for a given range of
 * rows, starting from 1 and ending at M, with respect to the total number of
 * columns N.
 *
 * @tparam M The starting index of the range of rows.
 * @tparam N The total number of columns in the matrix.
 *
 * The resulting type is an IndexSequence containing the row indices for the
 * upper triangular part of the matrix for the specified range of rows,
 * accessible via the nested ::type member.
 */
template <std::size_t M, std::size_t N>
struct ConcatenateUpperTriangularRowNumbers {

  using type =
      typename ConcatenateUpperTriangularRowNumbersBlock<1, M, N>::type;
};

/**
 * @brief A template alias to create a sequence of upper triangular row numbers
 * for a given range.
 *
 * This alias uses the TemplatesOperation::ConcatenateUpperTriangularRowNumbers
 * to generate a type representing the upper triangular row numbers for the
 * specified range.
 *
 * @tparam M The starting index of the range of rows.
 * @tparam N The total number of columns in the matrix.
 *
 * The resulting type is an IndexSequence containing the row indices for the
 * upper triangular part of the matrix for the specified range of rows.
 */
template <std::size_t M, std::size_t N>
using UpperTriangularRowNumbers =
    typename ConcatenateUpperTriangularRowNumbers<((N < M) ? N : M), N>::type;

/**
 * @brief A template struct to accumulate the number of elements in a triangular
 * sequence.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * triangular count sequence for a given range, starting from 0 and ending at M,
 * and concatenating it with an IndexSequence containing 0.
 *
 * @tparam TriangularCountNumbers The TriangularCountNumbers type containing the
 * counts of elements in each row.
 * @tparam M The starting index of the sequence.
 * @tparam N The ending index of the sequence.
 * @tparam Dif The difference between M and N.
 * The resulting type is an IndexSequence containing the accumulated counts of
 * non-empty entries for each row across the specified range, accessible via the
 * nested ::type member.
 */
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
  using Sequence_ =
      typename TemplatesOperation::AccumulateTriangularElementNumberStruct<
          typename TemplatesOperation::TriangularCountNumbers<N, N>::type,
          N>::type;

  using type = typename Concatenate<
      Sequence_, typename RepeatConcatenateIndexSequence<
                     Dif, IndexSequence<Sequence_::list[N]>>::type>::type;
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
using UpperTriangularCSRIndices = typename TemplatesOperation::ToCSRIndices<
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
 * The resulting type is a CSRPointers type that contains the accumulated number
 * of elements in each row of the upper triangular sparse matrix.
 */
template <std::size_t M, std::size_t N>
using UpperTriangularCSRPointers = typename TemplatesOperation::ToCSRPointers<
    typename TemplatesOperation::AccumulateTriangularElementNumberStructExtend<
        typename TemplatesOperation::TriangularCountNumbers<M, N>::type, M, N,
        (M <= N ? 0 : M - N)>::type>::type;

namespace TemplatesOperation {

/* Create Lower Triangular Sparse Matrix Row Indices */

/**
 * @brief A template struct to concatenate lower triangular row numbers for a
 * specific block of rows in a sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * concatenating lower triangular row numbers for a specific block of rows in
 * the matrix, effectively creating a sequence of row indices that correspond to
 * the lower triangular part of the matrix for the specified range of rows.
 *
 * @tparam Start The starting index of the block of rows.
 * @tparam Count The number of rows in the block.
 *
 * The resulting type is an IndexSequence containing the row indices for the
 * lower triangular part of the matrix for the specified range of rows,
 * accessible via the nested ::type member.
 */
template <std::size_t StartVal, std::size_t Count>
struct MakeLowerTriangularIndexSequenceBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type = typename Concatenate<
      typename MakeLowerTriangularIndexSequenceBlock<StartVal, MidCount>::type,
      typename MakeLowerTriangularIndexSequenceBlock<
          StartVal + MidCount, Count - MidCount>::type>::type;
};

/**
 * @brief Specialization of MakeLowerTriangularIndexSequenceBlock for the case
 * when Count is 1.
 *
 * This specialization defines a type alias 'type' that is set to an
 * IndexSequence containing the StartVal index, effectively creating a
 * lower triangular index block for that single index when there is only one row
 * to process.
 *
 * @tparam StartVal The starting index of the block of rows.
 */
template <std::size_t StartVal>
struct MakeLowerTriangularIndexSequenceBlock<StartVal, 1> {
  using type = IndexSequence<StartVal>;
};

/**
 * @brief Specialization of MakeLowerTriangularIndexSequenceBlock for the case
 * when Count is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * InvalidSequence<0>, effectively indicating that there are no valid row
 * indices to return when there are no rows to process.
 *
 * @tparam StartVal The starting index of the block of rows.
 */
template <std::size_t StartVal>
struct MakeLowerTriangularIndexSequenceBlock<StartVal, 0> {
  using type = InvalidSequence<0>;
};

/**
 * @brief A template struct to create a lower triangular index sequence.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * lower triangular index block for a given range, starting from Start and
 * ending at End, with an element size of E_S.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 * @tparam E_S The current size of the sequence.
 *
 * The resulting type is an IndexSequence containing the indices for the
 * lower triangular sequence, accessible via the nested ::type member.
 */
template <std::size_t Start, std::size_t End, std::size_t E_S>
struct MakeLowerTriangularIndexSequence {

  using type = typename MakeLowerTriangularIndexSequenceBlock<0, E_S + 1>::type;
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
 *
 * The resulting type is an IndexSequence containing the indices for the
 * lower triangular sequence, accessible via the nested ::type member.
 */
template <std::size_t Start, std::size_t End>
struct LowerTriangularSequenceList {
  using type = typename MakeLowerTriangularIndexSequence<Start, End,
                                                         (End - Start)>::type;
};

/**
 * @brief A template struct to concatenate lower triangular row numbers for a
 * specific range of rows in a sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * concatenated sequence of lower triangular row numbers for a given range of
 * rows, starting from 1 and ending at M, with respect to the total number of
 * columns N.
 *
 * @tparam M The starting index of the range of rows.
 * @tparam N The total number of columns in the matrix.
 *
 * The resulting type is an IndexSequence containing the row indices for the
 * lower triangular part of the matrix for the specified range of rows,
 * accessible via the nested ::type member.
 */
template <std::size_t StartM, std::size_t Count, std::size_t N>
struct ConcatenateLowerTriangularBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type =
      typename Concatenate<typename ConcatenateLowerTriangularBlock<
                               StartM + MidCount, Count - MidCount, N>::type,
                           typename ConcatenateLowerTriangularBlock<
                               StartM, MidCount, N>::type>::type;
};

/**
 * @brief Specialization of ConcatenateLowerTriangularBlock for the case when
 * Count is 1.
 *
 * This specialization defines a type alias 'type' that is set to the result of
 * LowerTriangularSequenceList for the specified starting index and total number
 * of columns, effectively creating a sequence of row indices for the lower
 * triangular part of the matrix for that single row when there is only one row
 * to process.
 *
 * @tparam StartM The starting index of the range of rows.
 * @tparam N The total number of columns in the matrix.
 */
template <std::size_t StartM, std::size_t N>
struct ConcatenateLowerTriangularBlock<StartM, 1, N> {
  using type = typename LowerTriangularSequenceList<StartM, N>::type;
};

/**
 * @brief Specialization of ConcatenateLowerTriangularBlock for the case when
 * Count is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * InvalidSequence<0>, effectively indicating that there are no valid row
 * indices to return when there are no rows to process.
 *
 * @tparam StartM The starting index of the range of rows.
 * @tparam N The total number of columns in the matrix.
 */
template <std::size_t StartM, std::size_t N>
struct ConcatenateLowerTriangularBlock<StartM, 0, N> {
  using type = InvalidSequence<0>;
};

/**
 * @brief A template struct to concatenate lower triangular row numbers for a
 * specific range of rows in a sparse matrix.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * concatenated sequence of lower triangular row numbers for a given range of
 * rows, starting from 1 and ending at M, with respect to the total number of
 * columns N.
 *
 * @tparam M The starting index of the range of rows.
 * @tparam N The total number of columns in the matrix.
 *
 * The resulting type is an IndexSequence containing the row indices for the
 * lower triangular part of the matrix for the specified range of rows,
 * accessible via the nested ::type member.
 */
template <std::size_t M, std::size_t N>
struct ConcatenateLowerTriangularRowNumbers {
  static_assert(M <= N, "So far, M must be less than or equal to N");
  using type = typename ConcatenateLowerTriangularBlock<1, M, N>::type;
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
using LowerTriangularCSRIndices = typename TemplatesOperation::ToCSRIndices<
    TemplatesOperation::LowerTriangularRowNumbers<M, N>>::type;

namespace TemplatesOperation {

/* Create Lower Triangular Sparse Matrix Row Pointers */

/**
 * @brief A template struct to create a lower triangular count block.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * generating a lower triangular count block for a given starting index and
 * count, effectively creating a sequence of counts that form a lower triangular
 * pattern.
 *
 * @tparam Start The starting index for the block.
 * @tparam Count The number of counts in the block.
 *
 * The resulting type is an IndexSequence containing the counts for the
 * lower triangular block, accessible via the nested ::type member.
 */
template <std::size_t Start, std::size_t Count>
struct MakeLowerTriangularCountSequenceBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type = typename Concatenate<
      typename MakeLowerTriangularCountSequenceBlock<Start, MidCount>::type,
      typename MakeLowerTriangularCountSequenceBlock<
          Start + MidCount, Count - MidCount>::type>::type;
};

/**
 * @brief Specialization of MakeLowerTriangularCountSequenceBlock for the case
 * when Count is 1.
 *
 * This specialization defines a type alias 'type' that is set to an
 * IndexSequence containing the Start index, effectively creating a lower
 * triangular count block for that single index when there is only one count to
 * process.
 *
 * @tparam Start The starting index for the block.
 */
template <std::size_t Start>
struct MakeLowerTriangularCountSequenceBlock<Start, 1> {

  using type = IndexSequence<Start>;
};

/**
 * @brief Specialization of MakeLowerTriangularCountSequenceBlock for the case
 * when Count is 0.
 *
 * This specialization defines a type alias 'type' that is set to
 * InvalidSequence<0>, effectively indicating that there are no valid counts to
 * return when there are no counts to process.
 *
 * @tparam Start The starting index for the block.
 */
template <std::size_t Start>
struct MakeLowerTriangularCountSequenceBlock<Start, 0> {

  using type = InvalidSequence<0>;
};

/**
 * @brief A template struct to create a lower triangular count sequence.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * lower triangular count block for a given range, starting from Start and
 * ending at End, with an element size of E_S.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 * @tparam E_S The current size of the sequence.
 *
 * The resulting type is an IndexSequence containing the counts for the
 * lower triangular sequence, accessible via the nested ::type member.
 */
template <std::size_t Start, std::size_t End, std::size_t E_S>
struct MakeLowerTriangularCountSequence {

  using type =
      typename MakeLowerTriangularCountSequenceBlock<Start, E_S + 1>::type;
};

/**
 * @brief A template struct to create a lower triangular count sequence list.
 *
 * This struct provides a type alias 'type' that is the result of generating a
 * lower triangular count sequence for a given range, starting from Start and
 * ending at End.
 *
 * @tparam Start The starting index of the sequence.
 * @tparam End The ending index of the sequence.
 *
 * The resulting type is an IndexSequence containing the counts for the
 * lower triangular sequence, accessible via the nested ::type member.
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
 * @tparam M The index of the row being processed.
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
 * The resulting type is a CSRPointers type that contains the accumulated number
 * of elements in each row of the lower triangular sparse matrix.
 */
template <std::size_t M, std::size_t N>
using LowerTriangularCSRPointers = typename TemplatesOperation::ToCSRPointers<
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
 * @tparam ValuesA The boolean values representing the availability of rows
 * in the first SparseAvailable type.
 * @tparam ValuesB The boolean values representing the availability of rows
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
 * @tparam ValuesA The boolean values representing the availability of rows
 * in the first SparseAvailable type.
 * @tparam ValuesB The boolean values representing the availability of rows
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
          std::size_t ROW, std::size_t COL, std::size_t N_Idx>
struct SparseAvailableMatrixMultiplyElement {
  static constexpr bool value =
      LogicalAnd<SparseAvailable_A::lists[ROW][N_Idx],
                 SparseAvailable_B::lists[N_Idx][COL]>::value;
};

/**
 * @brief A template struct to concatenate multiple ColumnAvailable types into a
 * single SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * multiple ColumnAvailable types into a SparseAvailable type.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam ROW The index of the row being processed.
 * @tparam COL The index of the column being processed.
 * @tparam StartIdx The starting index for the multiplication.
 * @tparam Count The number of indices to process for the multiplication.
 * The resulting type is a SparseAvailable type containing the logical OR of
 * the availability of each column in the two SparseAvailable types for the
 * specified row and column, accessible via the nested ::type member.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t ROW, std::size_t COL, std::size_t StartIdx,
          std::size_t Count>
struct SparseAvailableMatrixMultiplyMultiplyBlock {
  static constexpr std::size_t MidCount = Count / 2;

  using type = ConcatenateColumnAvailable<
      typename SparseAvailableMatrixMultiplyMultiplyBlock<
          SparseAvailable_A, SparseAvailable_B, ROW, COL, StartIdx,
          MidCount>::type,
      typename SparseAvailableMatrixMultiplyMultiplyBlock<
          SparseAvailable_A, SparseAvailable_B, ROW, COL, StartIdx + MidCount,
          Count - MidCount>::type>;
};

/**
 * @brief Specialization of SparseAvailableMatrixMultiplyMultiplyBlock for the
 * case when Count is 1.
 *
 * This specialization defines a type alias 'type' that is set to a
 * ColumnAvailable type containing the result of the multiplication for the
 * specified row and column when there is only one index to process.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam ROW The index of the row being processed.
 * @tparam COL The index of the column being processed.
 * @tparam StartIdx The starting index for the multiplication.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t ROW, std::size_t COL, std::size_t StartIdx>
struct SparseAvailableMatrixMultiplyMultiplyBlock<
    SparseAvailable_A, SparseAvailable_B, ROW, COL, StartIdx, 1> {

  using type = ColumnAvailable<SparseAvailableMatrixMultiplyElement<
      SparseAvailable_A, SparseAvailable_B, ROW, COL, StartIdx>::value>;
};

/**
 * @brief Specialization of SparseAvailableMatrixMultiplyMultiplyBlock for the
 * case when Count is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * ColumnAvailable type containing no values, effectively indicating that there
 * are no valid indices to process for the multiplication when Count is 0.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam ROW The index of the row being processed.
 * @tparam COL The index of the column being processed.
 * @tparam StartIdx The starting index for the multiplication.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t ROW, std::size_t COL, std::size_t StartIdx>
struct SparseAvailableMatrixMultiplyMultiplyBlock<
    SparseAvailable_A, SparseAvailable_B, ROW, COL, StartIdx, 0> {

  using type = ColumnAvailable<>;
};

/**
 * @brief A template struct to concatenate multiple ColumnAvailable types into a
 * single SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * multiple ColumnAvailable types into a SparseAvailable type.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam ROW The index of the row being processed.
 * @tparam COL The index of the column being processed.
 * @tparam N_Idx The index of the current multiplication being processed.
 * The resulting type is a ColumnAvailable type containing the logical OR of
 * the availability of each column in the two SparseAvailable types for the
 * specified row and column, accessible via the nested ::type member.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t ROW, std::size_t COL, std::size_t N_Idx>
struct SparseAvailableMatrixMultiplyMultiplyLoop {

  using type = typename SparseAvailableMatrixMultiplyMultiplyBlock<
      SparseAvailable_A, SparseAvailable_B, ROW, COL, 0, N_Idx + 1>::type;
};

/**
 * @brief A template struct to perform element-wise logical OR operation on a
 * specific range of columns for a given row and column in the multiplication of
 * two SparseAvailable types.
 *
 * This struct provides a static constexpr member 'value' that is the result of
 * performing an element-wise logical OR operation on the availability of each
 * column in the two SparseAvailable types for the specified row and column,
 * effectively determining if the multiplication is available for that specific
 * range of columns.
 *
 * @tparam ColumnAvailable The ColumnAvailable type containing the availability
 * values for the specified row and column.
 * @tparam StartIdx The starting index for the logical OR operation.
 * @tparam Count The number of indices to process for the logical OR operation.
 */
template <typename ColumnAvailable, std::size_t StartIdx, std::size_t Count>
struct ColumnAvailableElementWiseOrBlock {
  static constexpr std::size_t MidCount = Count / 2;

  static constexpr bool LeftResult =
      ColumnAvailableElementWiseOrBlock<ColumnAvailable, StartIdx,
                                        MidCount>::value;

  template <bool LeftVal, std::size_t Dummy = 0> struct EvaluateRight {

    static constexpr bool value = true;
  };

  template <std::size_t Dummy> struct EvaluateRight<false, Dummy> {

    static constexpr bool value =
        ColumnAvailableElementWiseOrBlock<ColumnAvailable, StartIdx + MidCount,
                                          Count - MidCount>::value;
  };

  static constexpr bool value = EvaluateRight<LeftResult>::value;
};

/**
 * @brief Specialization of ColumnAvailableElementWiseOrBlock for the case when
 * Count is 1.
 *
 * This specialization defines a static constexpr member 'value' that is set to
 * the availability value for the specified index in the ColumnAvailable type,
 * effectively determining if the multiplication is available for that specific
 * column when there is only one index to process.
 *
 * @tparam ColumnAvailable The ColumnAvailable type containing the availability
 * values for the specified row and column.
 * @tparam StartIdx The starting index for the logical OR operation.
 */
template <typename ColumnAvailable, std::size_t StartIdx>
struct ColumnAvailableElementWiseOrBlock<ColumnAvailable, StartIdx, 1> {
  static constexpr bool value = ColumnAvailable::list[StartIdx];
};

/**
 * @brief Specialization of ColumnAvailableElementWiseOrBlock for the case when
 * Count is 0.
 *
 * This specialization defines a static constexpr member 'value' that is set to
 * false, effectively indicating that there are no valid indices to process for
 * the logical OR operation when Count is 0.
 *
 * @tparam ColumnAvailable The ColumnAvailable type containing the availability
 * values for the specified row and column.
 * @tparam StartIdx The starting index for the logical OR operation.
 */
template <typename ColumnAvailable, std::size_t StartIdx>
struct ColumnAvailableElementWiseOrBlock<ColumnAvailable, StartIdx, 0> {
  static constexpr bool value = false;
};

/**
 * @brief A template struct to perform element-wise logical OR operation on a
 * specific range of columns for a given row and column in the multiplication of
 * two SparseAvailable types.
 *
 * This struct provides a static constexpr member 'value' that is the result of
 * performing an element-wise logical OR operation on the availability of each
 * column in the two SparseAvailable types for the specified row and column,
 * effectively determining if the multiplication is available for that specific
 * range of columns.
 *
 * @tparam ColumnAvailable The ColumnAvailable type containing the availability
 * values for the specified row and column.
 * @tparam N_Idx The index of the current multiplication being processed.
 */
template <typename ColumnAvailable, std::size_t N_Idx>
struct ColumnAvailableElementWiseOr {

  static constexpr bool value =
      ColumnAvailableElementWiseOrBlock<ColumnAvailable, 0, N_Idx + 1>::value;
};

/**
 * @brief A template struct to concatenate multiple ColumnAvailable types into a
 * single SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of concatenating
 * multiple ColumnAvailable types into a SparseAvailable type.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 * @tparam COL The index of the column being processed.
 * @tparam M_Idx The index of the current row being processed.
 * The resulting type is a ColumnAvailable type containing the logical OR of
 * the availability of each column in the two SparseAvailable types for the
 * specified row and column, accessible via the nested ::type member.
 */
template <typename SparseAvailable, std::size_t COL, std::size_t M_Idx>
struct ColumnAvailableFromSparseAvailableColumLoop {
  using type = ConcatenateColumnAvailable<
      typename ColumnAvailableFromSparseAvailableColumLoop<SparseAvailable, COL,
                                                           (M_Idx - 1)>::type,
      ColumnAvailable<SparseAvailable::lists[M_Idx][COL]>>;
};

/**
 * @brief Specialization of ColumnAvailableFromSparseAvailableColumLoop for the
 * case when M_Idx is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * ColumnAvailable type containing the first row of the SparseAvailable type,
 * effectively returning the availability of the first row.
 *
 * @tparam SparseAvailable The SparseAvailable type containing multiple rows.
 * @tparam COL The index of the column being processed.
 */
template <typename SparseAvailable, std::size_t COL>
struct ColumnAvailableFromSparseAvailableColumLoop<SparseAvailable, COL, 0> {
  using type = ColumnAvailable<SparseAvailable::lists[0][COL]>;
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
 * @tparam ROW The index of the row being processed.
 * @tparam J_Idx The index of the column being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t ROW, std::size_t J_Idx>
struct SparseAvailableMatrixMultiplyRowLoop {
  using type = ConcatenateColumnAvailable<
      typename SparseAvailableMatrixMultiplyRowLoop<
          SparseAvailable_A, SparseAvailable_B, ROW, (J_Idx - 1)>::type,
      ColumnAvailable<ColumnAvailableElementWiseOr<
          typename SparseAvailableMatrixMultiplyMultiplyLoop<
              SparseAvailable_A, SparseAvailable_B, ROW, J_Idx,
              (SparseAvailable_A::row_size - 1)>::type,
          (SparseAvailable_A::row_size - 1)>::value>>;
};

/**
 * @brief Specialization of SparseAvailableMatrixMultiplyRowLoop for the case
 * when J_Idx is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * ColumnAvailable type containing the result of the multiplication for the
 * first column.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_B The second SparseAvailable type.
 * @tparam ROW The index of the row being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t ROW>
struct SparseAvailableMatrixMultiplyRowLoop<SparseAvailable_A,
                                            SparseAvailable_B, ROW, 0> {
  using type = ColumnAvailable<ColumnAvailableElementWiseOr<
      typename SparseAvailableMatrixMultiplyMultiplyLoop<
          SparseAvailable_A, SparseAvailable_B, ROW, 0,
          (SparseAvailable_A::row_size - 1)>::type,
      (SparseAvailable_A::row_size - 1)>::value>;
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
 * @tparam I_Idx The index of the row being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t I_Idx>
struct SparseAvailableMatrixMultiplyColumnLoop {
  using type = ConcatenateSparseAvailableVertically<
      typename SparseAvailableMatrixMultiplyColumnLoop<
          SparseAvailable_A, SparseAvailable_B, (I_Idx - 1)>::type,
      SparseAvailableColumns<typename SparseAvailableMatrixMultiplyRowLoop<
          SparseAvailable_A, SparseAvailable_B, I_Idx,
          (SparseAvailable_B::row_size - 1)>::type>>;
};

/**
 * @brief Specialization of SparseAvailableMatrixMultiplyColumnLoop for the case
 * when I_Idx is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * SparseAvailableColumns type containing the result of the multiplication for
 * the first row.
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
          (SparseAvailable_B::row_size - 1)>::type>;
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
        (SparseAvailable_A::number_of_rows - 1)>::type;

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
          std::size_t ROW, std::size_t COL, std::size_t N_Idx>
struct SparseAvailableMatrixMultiplyTransposeElement {
  static constexpr bool value =
      LogicalAnd<SparseAvailable_A::lists[ROW][N_Idx],
                 SparseAvailable_BT::lists[COL][N_Idx]>::value;
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
 * @tparam ROW The index of the row being processed.
 * @tparam COL The index of the column being processed.
 * @tparam N_Idx The index of the element being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_BT,
          std::size_t ROW, std::size_t COL, std::size_t N_Idx>
struct SparseAvailableMatrixMultiplyTransposeMultiplyLoop {
  using type = ConcatenateColumnAvailable<
      typename SparseAvailableMatrixMultiplyTransposeMultiplyLoop<
          SparseAvailable_A, SparseAvailable_BT, ROW, COL, (N_Idx - 1)>::type,
      ColumnAvailable<SparseAvailableMatrixMultiplyTransposeElement<
          SparseAvailable_A, SparseAvailable_BT, ROW, COL, N_Idx>::value>>;
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
 * @tparam ROW The index of the row being processed.
 * @tparam COL The index of the column being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_BT,
          std::size_t ROW, std::size_t COL>
struct SparseAvailableMatrixMultiplyTransposeMultiplyLoop<
    SparseAvailable_A, SparseAvailable_BT, ROW, COL, 0> {
  using type = ColumnAvailable<SparseAvailableMatrixMultiplyTransposeElement<
      SparseAvailable_A, SparseAvailable_BT, ROW, COL, 0>::value>;
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
 * @tparam ROW The index of the row being processed.
 * @tparam J_Idx The index of the column being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_BT,
          std::size_t ROW, std::size_t J_Idx>
struct SparseAvailableMatrixMultiplyTransposeRowLoop {
  using type = ConcatenateColumnAvailable<
      typename SparseAvailableMatrixMultiplyTransposeRowLoop<
          SparseAvailable_A, SparseAvailable_BT, ROW, (J_Idx - 1)>::type,
      ColumnAvailable<ColumnAvailableElementWiseOr<
          typename SparseAvailableMatrixMultiplyTransposeMultiplyLoop<
              SparseAvailable_A, SparseAvailable_BT, ROW, J_Idx,
              (SparseAvailable_A::row_size - 1)>::type,
          (SparseAvailable_A::row_size - 1)>::value>>;
};

/**
 * @brief Specialization of SparseAvailableMatrixMultiplyTransposeRowLoop for
 * the case when J_Idx is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * ColumnAvailable type containing the result of the multiplication for the
 * first column.
 *
 * @tparam SparseAvailable_A The first SparseAvailable type.
 * @tparam SparseAvailable_BT The second SparseAvailable type (transposed).
 * @tparam ROW The index of the row being processed.
 */
template <typename SparseAvailable_A, typename SparseAvailable_BT,
          std::size_t ROW>
struct SparseAvailableMatrixMultiplyTransposeRowLoop<
    SparseAvailable_A, SparseAvailable_BT, ROW, 0> {
  using type = ColumnAvailable<ColumnAvailableElementWiseOr<
      typename SparseAvailableMatrixMultiplyTransposeMultiplyLoop<
          SparseAvailable_A, SparseAvailable_BT, ROW, 0,
          (SparseAvailable_A::row_size - 1)>::type,
      (SparseAvailable_A::row_size - 1)>::value>;
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
 * @tparam I_Idx The index of the row being processed.
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
              (SparseAvailable_BT::number_of_rows - 1)>::type>>;
};

/**
 * @brief Specialization of SparseAvailableMatrixMultiplyTransposeColumnLoop
 * for the case when I_Idx is 0.
 *
 * This specialization defines a type alias 'type' that is set to a
 * SparseAvailableColumns type containing the result of the multiplication for
 * the first row.
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
          (SparseAvailable_BT::number_of_rows - 1)>::type>;
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
        (SparseAvailable_A::number_of_rows - 1)>::type;

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
        DiagAvailable<SparseAvailable::row_size>, SparseAvailable>;

/* Check SparseAvailable is valid or not */

namespace TemplatesOperation {

/**
 * @brief A template struct to validate the SparseAvailable type.
 *
 * This struct provides a type alias 'type' that is the result of recursively
 * validating the SparseAvailable type for each column.
 *
 * @tparam SparseAvailable The SparseAvailable type to be validated.
 * @tparam NumberOfRowsFirst The number of rows in the first row.
 * @tparam ColumnIndex The index of the row being processed.
 */
template <typename SparseAvailable, std::size_t NumberOfRowsFirst,
          std::size_t ColumnIndex>
struct ValidateSparseAvailableLoop {
  static_assert(SparseAvailable::row_size == NumberOfRowsFirst,
                "Each ColumnAvailable size of SparseAvailable is not the same");

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
 * for the first row.
 *
 * @tparam SparseAvailable The SparseAvailable type to be validated.
 * @tparam NumberOfRowsFirst The number of rows in the first row.
 */
template <typename SparseAvailable, std::size_t NumberOfRowsFirst>
struct ValidateSparseAvailableLoop<SparseAvailable, NumberOfRowsFirst, 0> {
  static_assert(SparseAvailable::row_size == NumberOfRowsFirst,
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
        SparseAvailable, SparseAvailable::row_size,
        (SparseAvailable::number_of_rows - 1)>::type;

/* SparseAvailable get row */
namespace TemplatesOperation {

/** @brief A template struct to generate a ColumnAvailable type with a
 * specific index set to true.
 *
 * This struct provides a type alias 'type' that is set to a ColumnAvailable
 * type with the specified index set to true, and all other indices set to
 * false.
 *
 * @tparam M The total number of rows.
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
 * @tparam M The total number of rows.
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
 * @tparam M The total number of rows.
 * @tparam Index The index of the column to be set to true.
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
 * @tparam M The total number of rows in the RowAvailable type.
 * @tparam Index The index of the column to be set to true in the RowAvailable
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

#endif // BASE_MATRIX_TEMPLATES_HPP_
