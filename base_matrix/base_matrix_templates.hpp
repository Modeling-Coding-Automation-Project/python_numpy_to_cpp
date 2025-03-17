#ifndef __BASE_MATRIX_TEMPLATES_HPP__
#define __BASE_MATRIX_TEMPLATES_HPP__

#include "base_matrix_macros.hpp"

#include <tuple>

namespace Base {
namespace Matrix {

namespace TemplatesOperation {

template <std::size_t... Sizes> struct list_array {
  static constexpr std::size_t size = sizeof...(Sizes);
  static constexpr std::size_t value[size] = {Sizes...};
};

template <std::size_t... Sizes>
constexpr std::size_t list_array<Sizes...>::value[list_array<Sizes...>::size];

template <typename Array> class CompiledSparseMatrixList {
public:
  typedef const std::size_t *list_type;
  static constexpr list_type list = Array::value;
  static constexpr std::size_t size = Array::size;
};

} // namespace TemplatesOperation

template <std::size_t... Sizes>
using RowIndices = TemplatesOperation::CompiledSparseMatrixList<
    TemplatesOperation::list_array<Sizes...>>;

template <std::size_t... Sizes>
using RowPointers = TemplatesOperation::CompiledSparseMatrixList<
    TemplatesOperation::list_array<Sizes...>>;

namespace TemplatesOperation {

/* Create Sparse Matrix from Matrix Element List */
template <bool... Flags> struct available_list_array {
  static constexpr std::size_t size = sizeof...(Flags);
  static constexpr bool value[size] = {Flags...};
};

template <bool... Flags>
constexpr bool
    available_list_array<Flags...>::value[available_list_array<Flags...>::size];

template <typename Array> class CompiledSparseMatrixElementList {
public:
  typedef const bool *list_type;
  static constexpr list_type list = Array::value;
  static constexpr std::size_t size = Array::size;
};

} // namespace TemplatesOperation

template <bool... Flags>
using ColumnAvailable = TemplatesOperation::CompiledSparseMatrixElementList<
    TemplatesOperation::available_list_array<Flags...>>;

namespace TemplatesOperation {

template <typename... Columns> struct SparseAvailableColumns {
  static constexpr std::size_t number_of_columns = sizeof...(Columns);

  // Helper struct to extract the list from each ColumnAvailable
  template <typename Column> struct ExtractList {
    static constexpr typename Column::list_type value = Column::list;
  };

  // Array of lists from each ColumnAvailable
  static constexpr const bool *lists[number_of_columns] = {
      ExtractList<Columns>::value...};

  // Function to get the size of a specific column
  static constexpr std::size_t column_size =
      std::tuple_element<0, std::tuple<Columns...>>::type::size;
};

} // namespace TemplatesOperation

template <typename... Columns>
using SparseAvailable = TemplatesOperation::SparseAvailableColumns<Columns...>;

namespace TemplatesOperation {

// helper template to calculate the logical OR
template <bool A, bool B> struct LogicalOr {
  static constexpr bool value = A || B;
};

/* SparseAvailable check empty */
// helper template to calculate the logical OR for multiple values
template <bool... Values> struct LogicalOrMultiple;

// base case for a single value
template <bool Value> struct LogicalOrMultiple<Value> {
  static constexpr bool value = Value;
};

// recursive case for multiple values
template <bool First, bool... Rest> struct LogicalOrMultiple<First, Rest...> {
  static constexpr bool value =
      LogicalOr<First, LogicalOrMultiple<Rest...>::value>::value;
};

// helper template to calculate the logical OR for SparseAvailable
template <typename SparseAvailable> struct CheckSparseAvailableEmpty;

// partial specialization for ColumnAvailable
template <bool... Values>
struct CheckSparseAvailableEmpty<ColumnAvailable<Values...>> {
  static constexpr bool value =
      TemplatesOperation::LogicalOrMultiple<Values...>::value;
};

// partial specialization for SparseAvailable
template <typename... Columns>
struct CheckSparseAvailableEmpty<SparseAvailable<Columns...>> {
  static constexpr bool value = TemplatesOperation::LogicalOrMultiple<
      CheckSparseAvailableEmpty<Columns>::value...>::value;
};

/* Create Dense Available */

// Generate false flags
// base case: N = 0
template <std::size_t N, bool... Flags> struct GenerateFalseFlags {
  using type = typename GenerateFalseFlags<N - 1, false, Flags...>::type;
};

// recursive termination case: N = 0
template <bool... Flags> struct GenerateFalseFlags<0, Flags...> {
  using type = ColumnAvailable<Flags...>;
};

template <std::size_t N>
using GenerateFalseColumnAvailable = typename GenerateFalseFlags<N>::type;

// Generate true flags
// base case: N = 0
template <std::size_t N, bool... Flags> struct GenerateTrueFlags {
  using type = typename GenerateTrueFlags<N - 1, true, Flags...>::type;
};

// recursive termination case: N = 0
template <bool... Flags> struct GenerateTrueFlags<0, Flags...> {
  using type = ColumnAvailable<Flags...>;
};

template <std::size_t N>
using GenerateTrueColumnAvailable = typename GenerateTrueFlags<N>::type;

// concatenate true flags vertically
// base case: N = 0
template <std::size_t M, typename ColumnAvailable, typename... Columns>
struct RepeatColumnAvailable {
  using type =
      typename RepeatColumnAvailable<M - 1, ColumnAvailable, ColumnAvailable,
                                     Columns...>::type;
};

// recursive termination case: N = 0
template <typename ColumnAvailable, typename... Columns>
struct RepeatColumnAvailable<0, ColumnAvailable, Columns...> {
  using type = TemplatesOperation::SparseAvailableColumns<Columns...>;
};

} // namespace TemplatesOperation

// repeat ColumnAvailable M times
template <std::size_t M, std::size_t N>
using DenseAvailable = typename TemplatesOperation::RepeatColumnAvailable<
    M, TemplatesOperation::GenerateTrueColumnAvailable<N>>::type;

template <std::size_t M, std::size_t N>
using DenseAvailableEmpty = typename TemplatesOperation::RepeatColumnAvailable<
    M, TemplatesOperation::GenerateFalseColumnAvailable<N>>::type;

/* Create Diag Available */

namespace TemplatesOperation {

// base case: N = 0
template <std::size_t N, std::size_t Index, bool... Flags>
struct GenerateIndexedTrueFlags {
  using type = typename GenerateIndexedTrueFlags<
      N - 1, Index, (N - 1 == Index ? true : false), Flags...>::type;
};

// recursive termination case: N = 0
template <std::size_t Index, bool... Flags>
struct GenerateIndexedTrueFlags<0, Index, Flags...> {
  using type = ColumnAvailable<Flags...>;
};

template <std::size_t N, std::size_t Index>
using GenerateIndexedTrueColumnAvailable =
    typename GenerateIndexedTrueFlags<N, Index>::type;

// concatenate indexed true flags vertically
// base case: N = 0
template <std::size_t M, std::size_t N, std::size_t Index,
          typename ColumnAvailable, typename... Columns>
struct IndexedRepeatColumnAvailable {
  using type = typename IndexedRepeatColumnAvailable<
      (M - 1), N, (Index - 1),
      GenerateIndexedTrueColumnAvailable<N, (Index - 1)>, ColumnAvailable,
      Columns...>::type;
};

// recursive termination case: N = 0
template <std::size_t N, std::size_t Index, typename ColumnAvailable,
          typename... Columns>
struct IndexedRepeatColumnAvailable<0, N, Index, ColumnAvailable, Columns...> {
  using type = TemplatesOperation::SparseAvailableColumns<Columns...>;
};

} // namespace TemplatesOperation

// repeat ColumnAvailable M times
template <std::size_t M>
using DiagAvailable = typename TemplatesOperation::IndexedRepeatColumnAvailable<
    M, M, (M - 1),
    TemplatesOperation::GenerateIndexedTrueColumnAvailable<M, (M - 1)>>::type;

/* Create Sparse Available from Indices and Pointers */

namespace TemplatesOperation {

// base case: N = 0
template <typename ColumnAvailable_A, typename ColumnAvailable_B, std::size_t N,
          bool... Flags>
struct GenerateORTrueFlagsLoop {
  using type = typename GenerateORTrueFlagsLoop<
      ColumnAvailable_A, ColumnAvailable_B, N - 1,
      (ColumnAvailable_A::list[N - 1] | ColumnAvailable_B::list[N - 1]),
      Flags...>::type;
};

// recursive termination case: N = 0
template <typename ColumnAvailable_A, typename ColumnAvailable_B, bool... Flags>
struct GenerateORTrueFlagsLoop<ColumnAvailable_A, ColumnAvailable_B, 0,
                               Flags...> {
  using type = ColumnAvailable<Flags...>;
};

template <typename ColumnAvailable_A, typename ColumnAvailable_B>
using GenerateORTrueFlagsColumnAvailable =
    typename GenerateORTrueFlagsLoop<ColumnAvailable_A, ColumnAvailable_B,
                                     ColumnAvailable_A::size>::type;

template <typename RowIndices, std::size_t N, std::size_t RowIndicesIndex,
          std::size_t RowEndCount>
struct CreateSparsePointersRowLoop {
  using type = typename GenerateORTrueFlagsLoop<
      GenerateIndexedTrueColumnAvailable<N, RowIndices::list[RowIndicesIndex]>,
      typename CreateSparsePointersRowLoop<RowIndices, N, (RowIndicesIndex + 1),
                                           (RowEndCount - 1)>::type,
      N>::type;
};

template <typename RowIndices, std::size_t N, std::size_t RowIndicesIndex>
struct CreateSparsePointersRowLoop<RowIndices, N, RowIndicesIndex, 0> {
  using type = GenerateFalseColumnAvailable<N>;
};

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

template <std::size_t N, typename RowIndices, typename RowPointers,
          std::size_t ConsecutiveIndex, typename... Columns>
struct CreateSparsePointersLoop<N, RowIndices, RowPointers, 0, ConsecutiveIndex,
                                Columns...> {
  using type = SparseAvailableColumns<Columns...>;
};

} // namespace TemplatesOperation

template <std::size_t N, typename RowIndices, typename RowPointers>
using CreateSparseAvailableFromIndicesAndPointers =
    typename TemplatesOperation::CreateSparsePointersLoop<
        N, RowIndices, RowPointers, (RowPointers::size - 1), 0>::type;

namespace TemplatesOperation {

/* Create Sparse Matrix from Dense Matrix */
template <std::size_t... Seq> struct IndexSequence {
  static constexpr std::size_t size = sizeof...(Seq);
  static constexpr std::size_t list[size] = {Seq...};
};

template <std::size_t... Seq> struct InvalidSequence {
  static constexpr std::size_t size = sizeof...(Seq);
  static constexpr std::size_t list[size] = {Seq...};
};

template <std::size_t N, std::size_t... Seq>
struct MakeIndexSequence : MakeIndexSequence<N - 1, N - 1, Seq...> {};

template <std::size_t... Seq> struct MakeIndexSequence<0, Seq...> {
  using type = IndexSequence<Seq...>;
};

template <std::size_t N> struct IntegerSequenceList {
  using type = typename MakeIndexSequence<N>::type;
};

template <std::size_t N>
using MatrixRowNumbers = typename IntegerSequenceList<N>::type;

template <typename Seq1, typename Seq2> struct Concatenate;

template <std::size_t... Seq1, std::size_t... Seq2>
struct Concatenate<IndexSequence<Seq1...>, IndexSequence<Seq2...>> {
  using type = IndexSequence<Seq1..., Seq2...>;
};

template <std::size_t... Seq1, std::size_t... Seq2>
struct Concatenate<IndexSequence<Seq1...>, InvalidSequence<Seq2...>> {
  using type = IndexSequence<Seq1...>;
};

template <std::size_t... Seq1, std::size_t... Seq2>
struct Concatenate<InvalidSequence<Seq1...>, IndexSequence<Seq2...>> {
  using type = IndexSequence<Seq2...>;
};

/* Create Dense Matrix Row Indices and Pointers */
template <typename IndexSequence_1, typename IndexSequence_2>
using ConcatenateMatrixRowNumbers =
    typename Concatenate<IndexSequence_1, IndexSequence_2>::type;

template <std::size_t M, typename MatrixRowNumbers>
struct RepeatConcatenateIndexSequence;

template <typename MatrixRowNumbers>
struct RepeatConcatenateIndexSequence<1, MatrixRowNumbers> {
  using type = MatrixRowNumbers;
};

template <std::size_t M, typename MatrixRowNumbers>
struct RepeatConcatenateIndexSequence {
  using type = ConcatenateMatrixRowNumbers<
      MatrixRowNumbers,
      typename RepeatConcatenateIndexSequence<M - 1, MatrixRowNumbers>::type>;
};

template <typename Seq> struct ToRowIndices;

template <std::size_t... Seq> struct ToRowIndices<IndexSequence<Seq...>> {
  using type = RowIndices<Seq...>;
};

template <std::size_t M, std::size_t N>
using MatrixColumnRowNumbers =
    typename RepeatConcatenateIndexSequence<M, MatrixRowNumbers<N>>::type;

} // namespace TemplatesOperation

template <std::size_t M, std::size_t N>
using DenseMatrixRowIndices = typename TemplatesOperation::ToRowIndices<
    TemplatesOperation::MatrixColumnRowNumbers<M, N>>::type;

namespace TemplatesOperation {

template <std::size_t M, std::size_t N, std::size_t... Seq>
struct MakePointerList : MakePointerList<M - 1, N, (M * N), Seq...> {};

template <std::size_t N, std::size_t... Seq>
struct MakePointerList<0, N, Seq...> {
  using type = IndexSequence<0, Seq...>;
};

template <std::size_t M, std::size_t N> struct MatrixDensePointerList {
  using type = typename MakePointerList<M, N>::type;
};

template <std::size_t M, std::size_t N>
using MatrixColumnRowPointers = typename MatrixDensePointerList<M, N>::type;

template <typename Seq> struct ToRowPointers;

template <std::size_t... Seq>
struct ToRowPointers<TemplatesOperation::IndexSequence<Seq...>> {
  using type = RowPointers<Seq...>;
};

} // namespace TemplatesOperation

template <std::size_t M, std::size_t N>
using DenseMatrixRowPointers = typename TemplatesOperation::ToRowIndices<
    TemplatesOperation::MatrixColumnRowPointers<M, N>>::type;

/* Concatenate ColumnAvailable */

namespace TemplatesOperation {

template <typename Column1, typename Column2>
struct ConcatenateColumnAvailableLists;

template <bool... Flags1, bool... Flags2>
struct ConcatenateColumnAvailableLists<ColumnAvailable<Flags1...>,
                                       ColumnAvailable<Flags2...>> {
  using type = ColumnAvailable<Flags1..., Flags2...>;
};

} // namespace TemplatesOperation

template <typename ColumnAvailable_A, typename ColumnAvailable_B>
using ConcatenateColumnAvailable =
    typename TemplatesOperation::ConcatenateColumnAvailableLists<
        ColumnAvailable_A, ColumnAvailable_B>::type;

/* Get ColumnAvailable from SparseAvailable */

namespace TemplatesOperation {

template <std::size_t N, typename SparseAvailable> struct GetColumnAvailable;

template <std::size_t N, typename... Columns>
struct GetColumnAvailable<N, SparseAvailable<Columns...>> {
  using type = typename std::tuple_element<N, std::tuple<Columns...>>::type;
};

} // namespace TemplatesOperation

/* Concatenate SparseAvailable vertically */
template <typename SparseAvailable1, typename SparseAvailable2>
struct ConcatenateSparseAvailable;

template <typename... Columns1, typename... Columns2>
struct ConcatenateSparseAvailable<
    TemplatesOperation::SparseAvailableColumns<Columns1...>,
    TemplatesOperation::SparseAvailableColumns<Columns2...>> {
  using type =
      TemplatesOperation::SparseAvailableColumns<Columns1..., Columns2...>;
};

template <typename SparseAvailable_A, typename SparseAvailable_B>
using ConcatenateSparseAvailableVertically =
    typename ConcatenateSparseAvailable<SparseAvailable_A,
                                        SparseAvailable_B>::type;

/* Concatenate SparseAvailable horizontally */

namespace TemplatesOperation {

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

template <typename SparseAvailable_A, typename SparseAvailable_B>
struct ConcatenateSparseAvailableHorizontallyLoop<SparseAvailable_A,
                                                  SparseAvailable_B, 0> {
  using type = SparseAvailableColumns<typename ConcatenateColumnAvailableLists<
      typename GetColumnAvailable<0, SparseAvailable_A>::type,
      typename GetColumnAvailable<0, SparseAvailable_B>::type>::type>;
};

} // namespace TemplatesOperation

template <typename SparseAvailable_A, typename SparseAvailable_B>
using ConcatenateSparseAvailableHorizontally =
    typename TemplatesOperation::ConcatenateSparseAvailableHorizontallyLoop<
        SparseAvailable_A, SparseAvailable_B,
        SparseAvailable_A::number_of_columns - 1>::type;

/* Concatenate ColumnAvailable with SparseAvailable  */

namespace TemplatesOperation {

// Helper template to concatenate two template parameter packs
template <typename...> struct ConcatTuple;

template <typename... Ts1, typename... Ts2>
struct ConcatTuple<std::tuple<Ts1...>, std::tuple<Ts2...>> {
  using type = std::tuple<Ts1..., Ts2...>;
};

// Template to concatenate ColumnAvailable and SparseAvailable
template <typename Column, typename Sparse> struct ConcatColumnSparse;

template <typename Column, typename... Columns>
struct ConcatColumnSparse<Column, SparseAvailable<Columns...>> {
  using type = SparseAvailable<Column, Columns...>;
};

/* Get rest of SparseAvailable */
template <typename SparseAvailable_In, std::size_t Col_Index,
          std::size_t Residual>
struct GetRestOfSparseAvailableLoop {
  using type = typename ConcatColumnSparse<
      typename GetColumnAvailable<Col_Index, SparseAvailable_In>::type,
      typename GetRestOfSparseAvailableLoop<SparseAvailable_In, (Col_Index + 1),
                                            (Residual - 1)>::type>::type;
};

template <typename SparseAvailable_In, std::size_t Col_Index>
struct GetRestOfSparseAvailableLoop<SparseAvailable_In, Col_Index, 0> {
  using type = SparseAvailable<
      typename GetColumnAvailable<Col_Index, SparseAvailable_In>::type>;
};

template <typename SparseAvailable_In, std::size_t Col_Index>
using GetRestOfSparseAvailable = typename GetRestOfSparseAvailableLoop<
    SparseAvailable_In, Col_Index,
    ((SparseAvailable_In::number_of_columns - 1) - Col_Index)>::type;

template <typename SparseAvailable_In, std::size_t Col_Index, bool NotEmpty>
struct AvoidEmptyColumnsSparseAvailableLoop;

template <typename SparseAvailable_In, std::size_t Col_Index>
struct AvoidEmptyColumnsSparseAvailableLoop<SparseAvailable_In, Col_Index,
                                            true> {
  using type = GetRestOfSparseAvailable<SparseAvailable_In, Col_Index>;
};

template <typename SparseAvailable_In, std::size_t Col_Index>
struct AvoidEmptyColumnsSparseAvailableLoop<SparseAvailable_In, Col_Index,
                                            false> {
  using type = typename AvoidEmptyColumnsSparseAvailableLoop<
      SparseAvailable_In, (Col_Index + 1),
      CheckSparseAvailableEmpty<typename GetColumnAvailable<
          (Col_Index + 1), SparseAvailable_In>::type>::value>::type;
};

template <typename SparseAvailable_In>
using AvoidEmptyColumnsSparseAvailable =
    typename AvoidEmptyColumnsSparseAvailableLoop<
        SparseAvailable_In, 0,
        CheckSparseAvailableEmpty<TemplatesOperation::SparseAvailableColumns<
            typename GetColumnAvailable<0, SparseAvailable_In>::type>>::value>::
        type;

/* Create Row Indices */
template <typename SparseAvailable, std::size_t ColumnElementNumber,
          bool Active, std::size_t RowElementNumber>
struct AssignSparseMatrixRowLoop;

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

template <typename SparseAvailable, std::size_t ColumnElementNumber,
          std::size_t RowElementNumber>
struct AssignSparseMatrixRowLoop<SparseAvailable, ColumnElementNumber, false,
                                 RowElementNumber> {
  using type = typename AssignSparseMatrixRowLoop<
      SparseAvailable, ColumnElementNumber,
      SparseAvailable::lists[ColumnElementNumber][RowElementNumber - 1],
      RowElementNumber - 1>::type;
};

template <typename SparseAvailable, std::size_t ColumnElementNumber>
struct AssignSparseMatrixRowLoop<SparseAvailable, ColumnElementNumber, false,
                                 0> {
  using type = InvalidSequence<0>;
};

template <typename SparseAvailable, std::size_t ColumnElementNumber>
struct AssignSparseMatrixRowLoop<SparseAvailable, ColumnElementNumber, true,
                                 0> {
  using type = IndexSequence<0>;
};

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

template <typename SparseAvailable>
struct AssignSparseMatrixColumnLoop<SparseAvailable, 0> {
  using type = typename AssignSparseMatrixRowLoop<
      SparseAvailable, 0,
      SparseAvailable::lists[0][SparseAvailable::column_size - 1],
      (SparseAvailable::column_size - 1)>::type;
};

template <typename SparseAvailable, bool NotEmpty>
struct RowIndicesSequenceFromSparseAvailable;

template <typename SparseAvailable>
struct RowIndicesSequenceFromSparseAvailable<SparseAvailable, true> {
  using type = typename AssignSparseMatrixColumnLoop<
      AvoidEmptyColumnsSparseAvailable<SparseAvailable>,
      (AvoidEmptyColumnsSparseAvailable<SparseAvailable>::number_of_columns -
       1)>::type;
};

template <typename SparseAvailable>
struct RowIndicesSequenceFromSparseAvailable<SparseAvailable, false> {
  using type = IndexSequence<0>;
};

} // namespace TemplatesOperation

template <typename SparseAvailable>
using RowIndicesFromSparseAvailable = typename TemplatesOperation::ToRowIndices<
    typename TemplatesOperation::RowIndicesSequenceFromSparseAvailable<
        SparseAvailable, TemplatesOperation::CheckSparseAvailableEmpty<
                             SparseAvailable>::value>::type>::type;

/* Create Row Pointers */

namespace TemplatesOperation {

template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber, bool Active,
          std::size_t RowElementNumber>
struct CountSparseMatrixRowLoop;

template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber, std::size_t RowElementNumber>
struct CountSparseMatrixRowLoop<SparseAvailable, ElementCount,
                                ColumnElementNumber, true, RowElementNumber> {
  using type = typename CountSparseMatrixRowLoop<
      SparseAvailable, (ElementCount + 1), ColumnElementNumber,
      SparseAvailable::lists[ColumnElementNumber][RowElementNumber - 1],
      RowElementNumber - 1>::type;
};

template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber, std::size_t RowElementNumber>
struct CountSparseMatrixRowLoop<SparseAvailable, ElementCount,
                                ColumnElementNumber, false, RowElementNumber> {
  using type = typename CountSparseMatrixRowLoop<
      SparseAvailable, ElementCount, ColumnElementNumber,
      SparseAvailable::lists[ColumnElementNumber][RowElementNumber - 1],
      RowElementNumber - 1>::type;
};

template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber>
struct CountSparseMatrixRowLoop<SparseAvailable, ElementCount,
                                ColumnElementNumber, false, 0> {
  using type = IndexSequence<ElementCount>;
};

template <typename SparseAvailable, std::size_t ElementCount,
          std::size_t ColumnElementNumber>
struct CountSparseMatrixRowLoop<SparseAvailable, ElementCount,
                                ColumnElementNumber, true, 0> {
  using type = IndexSequence<(ElementCount + 1)>;
};

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

template <typename SparseAvailable>
struct CountSparseMatrixColumnLoop<SparseAvailable, 0> {
  using type = typename Concatenate<
      IndexSequence<0>,
      typename CountSparseMatrixRowLoop<
          SparseAvailable, 0, 0,
          SparseAvailable::lists[0][SparseAvailable::column_size - 1],
          (SparseAvailable::column_size - 1)>::type>::type;
};

template <typename CountSparseMatrixColumnLoop, std::size_t ColumnElementNumber>
struct AccumulateElementNumberLoop {
  static constexpr std::size_t compute() {
    return CountSparseMatrixColumnLoop::list[ColumnElementNumber] +
           AccumulateElementNumberLoop<CountSparseMatrixColumnLoop,
                                       ColumnElementNumber - 1>::compute();
  }
};

template <typename CountSparseMatrixColumnLoop>
struct AccumulateElementNumberLoop<CountSparseMatrixColumnLoop, 0> {
  static constexpr std::size_t compute() { return 0; }
};

template <typename CountSparseMatrixColumnLoop, std::size_t ColumnElementNumber>
struct AccumulateSparseMatrixElementNumberLoop {
  using type = typename Concatenate<
      typename AccumulateSparseMatrixElementNumberLoop<
          CountSparseMatrixColumnLoop, ColumnElementNumber - 1>::type,
      IndexSequence<AccumulateElementNumberLoop<
          CountSparseMatrixColumnLoop, ColumnElementNumber>::compute()>>::type;
};

template <typename CountSparseMatrixColumnLoop>
struct AccumulateSparseMatrixElementNumberLoop<CountSparseMatrixColumnLoop, 0> {
  using type = IndexSequence<CountSparseMatrixColumnLoop::list[0]>;
};

template <typename CountSparseMatrixColumnLoop, typename SparseAvailable>
struct AccumulateSparseMatrixElementNumberStruct {
  using type = typename AccumulateSparseMatrixElementNumberLoop<
      CountSparseMatrixColumnLoop, SparseAvailable::number_of_columns>::type;
};

} // namespace TemplatesOperation

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
template <std::size_t Start, std::size_t End, std::size_t E_S>
struct MakeTriangularIndexSequence {
  using type = typename Concatenate<
      typename MakeTriangularIndexSequence<Start, (End - 1), (E_S - 1)>::type,
      IndexSequence<(End - 1)>>::type;
};

template <std::size_t Start, std::size_t End>
struct MakeTriangularIndexSequence<Start, End, 0> {
  using type = IndexSequence<(End - 1)>;
};

template <std::size_t Start, std::size_t End> struct TriangularSequenceList {
  using type =
      typename MakeTriangularIndexSequence<Start, End, (End - Start)>::type;
};

/* Count for Triangular */
template <std::size_t Start, std::size_t End, std::size_t E_S>
struct MakeTriangularCountSequence {
  using type =
      typename Concatenate<IndexSequence<End>,
                           typename MakeTriangularCountSequence<
                               Start, (End - 1), (E_S - 1)>::type>::type;
};

template <std::size_t Start, std::size_t End>
struct MakeTriangularCountSequence<Start, End, 0> {
  using type = IndexSequence<End>;
};

template <std::size_t Start, std::size_t End> struct TriangularCountList {
  using type =
      typename MakeTriangularCountSequence<Start, End, (End - Start)>::type;
};

template <std::size_t M, std::size_t N> struct TriangularCountNumbers {
  using type =
      typename Concatenate<IndexSequence<0>,
                           typename TriangularCountList<
                               ((N - ((N < M) ? N : M)) + 1), N>::type>::type;
};

/* Create Upper Triangular Sparse Matrix Row Indices */
template <std::size_t M, std::size_t N>
struct ConcatenateUpperTriangularRowNumbers {
  using type = typename Concatenate<
      typename ConcatenateUpperTriangularRowNumbers<(M - 1), N>::type,
      typename TriangularSequenceList<M, N>::type>::type;
};

template <std::size_t N> struct ConcatenateUpperTriangularRowNumbers<1, N> {
  using type = typename TriangularSequenceList<1, N>::type;
};

template <std::size_t M, std::size_t N>
using UpperTriangularRowNumbers =
    typename ConcatenateUpperTriangularRowNumbers<((N < M) ? N : M), N>::type;

template <typename TriangularCountNumbers, std::size_t M>
struct AccumulateTriangularElementNumberStruct {
  using type =
      typename AccumulateSparseMatrixElementNumberLoop<TriangularCountNumbers,
                                                       M>::type;
};

} // namespace TemplatesOperation

/* Create Upper Triangular Sparse Matrix Row Indices and Pointers */

template <std::size_t M, std::size_t N>
using UpperTriangularRowIndices = typename TemplatesOperation::ToRowIndices<
    TemplatesOperation::UpperTriangularRowNumbers<M, N>>::type;

template <std::size_t M, std::size_t N>
using UpperTriangularRowPointers = typename TemplatesOperation::ToRowPointers<
    typename TemplatesOperation::AccumulateTriangularElementNumberStruct<
        typename TemplatesOperation::TriangularCountNumbers<M, N>::type,
        M>::type>::type;

namespace TemplatesOperation {

/* Create Lower Triangular Sparse Matrix Row Indices */
template <std::size_t Start, std::size_t End, std::size_t E_S>
struct MakeLowerTriangularIndexSequence {
  using type = typename Concatenate<typename MakeLowerTriangularIndexSequence<
                                        Start, (End - 1), (E_S - 1)>::type,
                                    IndexSequence<E_S>>::type;
};

template <std::size_t Start, std::size_t End>
struct MakeLowerTriangularIndexSequence<Start, End, 0> {
  using type = IndexSequence<0>;
};

template <std::size_t Start, std::size_t End>
struct LowerTriangularSequenceList {
  using type = typename MakeLowerTriangularIndexSequence<Start, End,
                                                         (End - Start)>::type;
};

template <std::size_t M, std::size_t N>
struct ConcatenateLowerTriangularRowNumbers {
  using type = typename Concatenate<
      typename LowerTriangularSequenceList<M, N>::type,
      typename ConcatenateLowerTriangularRowNumbers<(M - 1), N>::type>::type;
};

template <std::size_t N> struct ConcatenateLowerTriangularRowNumbers<1, N> {
  using type = typename LowerTriangularSequenceList<1, N>::type;
};

template <std::size_t M, std::size_t N>
using LowerTriangularRowNumbers =
    typename TemplatesOperation::ConcatenateLowerTriangularRowNumbers<
        M, ((M < N) ? M : N)>::type;

} // namespace TemplatesOperation

template <std::size_t M, std::size_t N>
using LowerTriangularRowIndices = typename TemplatesOperation::ToRowIndices<
    TemplatesOperation::LowerTriangularRowNumbers<M, N>>::type;

namespace TemplatesOperation {

/* Create Lower Triangular Sparse Matrix Row Pointers */
template <std::size_t Start, std::size_t End, std::size_t E_S>
struct MakeLowerTriangularCountSequence {
  using type =
      typename Concatenate<IndexSequence<Start>,
                           typename MakeLowerTriangularCountSequence<
                               (Start + 1), (End - 1), (E_S - 1)>::type>::type;
};

template <std::size_t Start, std::size_t End>
struct MakeLowerTriangularCountSequence<Start, End, 0> {
  using type = IndexSequence<Start>;
};

template <std::size_t Start, std::size_t End> struct LowerTriangularCountList {
  using type = typename MakeLowerTriangularCountSequence<Start, End,
                                                         (End - Start)>::type;
};

template <std::size_t M, std::size_t N> struct LowerTriangularCountNumbers {
  using type = typename Concatenate<
      IndexSequence<0>,
      typename LowerTriangularCountList<1, ((M < N) ? M : N)>::type>::type;
};

template <typename LowerTriangularCountNumbers, std::size_t M>
struct AccumulateLowerTriangularElementNumberStruct {
  using type = typename AccumulateSparseMatrixElementNumberLoop<
      LowerTriangularCountNumbers, M>::type;
};

} // namespace TemplatesOperation

template <std::size_t M, std::size_t N>
using LowerTriangularRowPointers = typename TemplatesOperation::ToRowPointers<
    typename TemplatesOperation::AccumulateLowerTriangularElementNumberStruct<
        typename TemplatesOperation::LowerTriangularCountNumbers<M, N>::type,
        M>::type>::type;

/* SparseAvailable Addition and Subtraction */

namespace TemplatesOperation {

// helper template to calculate the matrix product
template <typename MatrixA, typename MatrixB>
struct MatrixAddSubSparseAvailableHelper;

// partial specialization for ColumnAvailable
template <bool... ValuesA, bool... ValuesB>
struct MatrixAddSubSparseAvailableHelper<ColumnAvailable<ValuesA...>,
                                         ColumnAvailable<ValuesB...>> {
  using type = ColumnAvailable<
      TemplatesOperation::LogicalOr<ValuesA, ValuesB>::value...>;
};

// partial specialization for SparseAvailable
template <typename... ColumnsA, typename... ColumnsB>
struct MatrixAddSubSparseAvailableHelper<SparseAvailable<ColumnsA...>,
                                         SparseAvailable<ColumnsB...>> {
  using type = SparseAvailable<
      typename MatrixAddSubSparseAvailableHelper<ColumnsA, ColumnsB>::type...>;
};

} // namespace TemplatesOperation

// template to check if the matrix product is available
template <typename SparseAvailable_A, typename SparseAvailable_B>
using MatrixAddSubSparseAvailable =
    typename TemplatesOperation::MatrixAddSubSparseAvailableHelper<
        SparseAvailable_A, SparseAvailable_B>::type;

/* SparseAvailable Multiply */

namespace TemplatesOperation {

// helper template to calculate the logical AND
template <bool A, bool B> struct LogicalAnd {
  static constexpr bool value = A && B;
};

template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t COL, std::size_t ROW, std::size_t N_Idx>
struct SparseAvailableMatrixMultiplyElement {
  static constexpr bool value =
      LogicalAnd<SparseAvailable_A::lists[COL][N_Idx],
                 SparseAvailable_B::lists[N_Idx][ROW]>::value;
};

template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t COL, std::size_t ROW, std::size_t N_Idx>
struct SparseAvailableMatrixMultiplyMultiplyLoop {
  using type = ConcatenateColumnAvailable<
      typename SparseAvailableMatrixMultiplyMultiplyLoop<
          SparseAvailable_A, SparseAvailable_B, COL, ROW, (N_Idx - 1)>::type,
      ColumnAvailable<SparseAvailableMatrixMultiplyElement<
          SparseAvailable_A, SparseAvailable_B, COL, ROW, N_Idx>::value>>;
};

template <typename SparseAvailable_A, typename SparseAvailable_B,
          std::size_t COL, std::size_t ROW>
struct SparseAvailableMatrixMultiplyMultiplyLoop<
    SparseAvailable_A, SparseAvailable_B, COL, ROW, 0> {
  using type = ColumnAvailable<SparseAvailableMatrixMultiplyElement<
      SparseAvailable_A, SparseAvailable_B, COL, ROW, 0>::value>;
};

template <typename ColumnAvailable, std::size_t N_Idx>
struct ColumnAvailableElementWiseOr {
  static constexpr bool value = LogicalOr<
      ColumnAvailable::list[N_Idx],
      ColumnAvailableElementWiseOr<ColumnAvailable, (N_Idx - 1)>::value>::value;
};

template <typename ColumnAvailable>
struct ColumnAvailableElementWiseOr<ColumnAvailable, 0> {
  static constexpr bool value = ColumnAvailable::list[0];
};

// ColumnAvailable from SparseAvailable Row vector
template <typename SparseAvailable, std::size_t ROW, std::size_t M_Idx>
struct ColumnAvailableFromSparseAvailableColumLoop {
  using type = ConcatenateColumnAvailable<
      typename ColumnAvailableFromSparseAvailableColumLoop<SparseAvailable, ROW,
                                                           (M_Idx - 1)>::type,
      ColumnAvailable<SparseAvailable::lists[M_Idx][ROW]>>;
};

template <typename SparseAvailable, std::size_t ROW>
struct ColumnAvailableFromSparseAvailableColumLoop<SparseAvailable, ROW, 0> {
  using type = ColumnAvailable<SparseAvailable::lists[0][ROW]>;
};

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

template <typename SparseAvailable_A, typename SparseAvailable_B>
struct SparseAvailableMatrixMultiplyColumnLoop<SparseAvailable_A,
                                               SparseAvailable_B, 0> {
  using type =
      SparseAvailableColumns<typename SparseAvailableMatrixMultiplyRowLoop<
          SparseAvailable_A, SparseAvailable_B, 0,
          (SparseAvailable_B::column_size - 1)>::type>;
};

} // namespace TemplatesOperation

template <typename SparseAvailable_A, typename SparseAvailable_B>
using SparseAvailableMatrixMultiply =
    typename TemplatesOperation::SparseAvailableMatrixMultiplyColumnLoop<
        SparseAvailable_A, SparseAvailable_B,
        (SparseAvailable_A::number_of_columns - 1)>::type;

/* SparseAvailable Multiply Transpose
 * (SparseAvailable_BT will be calculated as transpose) */

namespace TemplatesOperation {

template <typename SparseAvailable_A, typename SparseAvailable_BT,
          std::size_t COL, std::size_t ROW, std::size_t N_Idx>
struct SparseAvailableMatrixMultiplyTransposeElement {
  static constexpr bool value =
      LogicalAnd<SparseAvailable_A::lists[COL][N_Idx],
                 SparseAvailable_BT::lists[ROW][N_Idx]>::value;
};

template <typename SparseAvailable_A, typename SparseAvailable_BT,
          std::size_t COL, std::size_t ROW, std::size_t N_Idx>
struct SparseAvailableMatrixMultiplyTransposeMultiplyLoop {
  using type = ConcatenateColumnAvailable<
      typename SparseAvailableMatrixMultiplyTransposeMultiplyLoop<
          SparseAvailable_A, SparseAvailable_BT, COL, ROW, (N_Idx - 1)>::type,
      ColumnAvailable<SparseAvailableMatrixMultiplyTransposeElement<
          SparseAvailable_A, SparseAvailable_BT, COL, ROW, N_Idx>::value>>;
};

template <typename SparseAvailable_A, typename SparseAvailable_BT,
          std::size_t COL, std::size_t ROW>
struct SparseAvailableMatrixMultiplyTransposeMultiplyLoop<
    SparseAvailable_A, SparseAvailable_BT, COL, ROW, 0> {
  using type = ColumnAvailable<SparseAvailableMatrixMultiplyTransposeElement<
      SparseAvailable_A, SparseAvailable_BT, COL, ROW, 0>::value>;
};

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

template <typename SparseAvailable_A, typename SparseAvailable_BT>
struct SparseAvailableMatrixMultiplyTransposeColumnLoop<SparseAvailable_A,
                                                        SparseAvailable_BT, 0> {
  using type = TemplatesOperation::SparseAvailableColumns<
      typename SparseAvailableMatrixMultiplyTransposeRowLoop<
          SparseAvailable_A, SparseAvailable_BT, 0,
          (SparseAvailable_BT::number_of_columns - 1)>::type>;
};

template <typename SparseAvailable_A, typename SparseAvailable_BT>
using SparseAvailableMatrixMultiplyTranspose =
    typename SparseAvailableMatrixMultiplyTransposeColumnLoop<
        SparseAvailable_A, SparseAvailable_BT,
        (SparseAvailable_A::number_of_columns - 1)>::type;

} // namespace TemplatesOperation

template <typename SparseAvailable>
using SparseAvailableTranspose =
    TemplatesOperation::SparseAvailableMatrixMultiplyTranspose<
        DiagAvailable<SparseAvailable::column_size>, SparseAvailable>;

/* Check SparseAvailable is valid or not */

namespace TemplatesOperation {

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

template <typename SparseAvailable, std::size_t NumberOfRowsFirst>
struct ValidateSparseAvailableLoop<SparseAvailable, NumberOfRowsFirst, 0> {
  static_assert(SparseAvailable::column_size == NumberOfRowsFirst,
                "Each ColumnAvailable size of SparseAvailable is not the same");

  using type = SparseAvailable;
};

} // namespace TemplatesOperation

template <typename SparseAvailable>
using ValidateSparseAvailable =
    typename TemplatesOperation::ValidateSparseAvailableLoop<
        SparseAvailable, SparseAvailable::column_size,
        (SparseAvailable::number_of_columns - 1)>::type;

/* SparseAvailable get row */
namespace TemplatesOperation {

template <std::size_t M, std::size_t Index>
using GenerateIndexedRowTrueColumnAvailable =
    ColumnAvailable<(M == Index ? true : false)>;

// concatenate indexed true flags vertically
// base case: N = 0
template <std::size_t M, std::size_t Index, typename ColumnAvailable,
          typename... Columns>
struct IndexedRowRepeatColumnAvailable {
  using type = typename IndexedRowRepeatColumnAvailable<
      (M - 1), Index, GenerateIndexedRowTrueColumnAvailable<(M - 1), Index>,
      ColumnAvailable, Columns...>::type;
};

// recursive termination case: N = 0
template <std::size_t Index, typename ColumnAvailable, typename... Columns>
struct IndexedRowRepeatColumnAvailable<0, Index, ColumnAvailable, Columns...> {
  using type = TemplatesOperation::SparseAvailableColumns<
      GenerateIndexedRowTrueColumnAvailable<0, Index>, Columns...>;
};

} // namespace TemplatesOperation

template <std::size_t M, std::size_t Index>
using RowAvailable =
    typename TemplatesOperation::IndexedRowRepeatColumnAvailable<
        (M - 1), Index,
        TemplatesOperation::GenerateIndexedRowTrueColumnAvailable<(M - 1),
                                                                  Index>>::type;

template <std::size_t M, typename SparseAvailable, std::size_t Index>
using SparseAvailableGetRow =
    SparseAvailableMatrixMultiply<SparseAvailable, RowAvailable<M, Index>>;

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_TEMPLATES_HPP__
