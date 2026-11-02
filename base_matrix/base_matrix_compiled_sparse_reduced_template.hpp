/********************************************************************************
@file base_matrix_compiled_sparse_reduced_template.hpp
@brief
This file provides reduced-template kernels for compiled sparse addition,
subtraction, scaling, and matrix/vector products, including transposed forms.
It uses compile-time index packs to unroll each sparse row while keeping the
complete CSR pattern out of the arithmetic kernels.

@details
The file defines:
- Row kernels with fixed input/output offsets and ordered arithmetic.
- Compile-time sparse coordinate mapping, including duplicate indices.
- Compile-time checks corresponding to the original dense/sparse accessors.
- Dispatch helpers for BASE_MATRIX_REDUCE_TEMPLATE_PATTERN_.

The existing recursive helpers remain available, including for bool storage.
Sequencing arrays preserve the operation order in unoptimized builds and can be
discarded by optimizing compilers.

@note
tparam M is the number of rows in the matrix.
tparam N is the number of columns in the matrix.
Somehow programming custom is vice versa,
but in this project, we use the mathematical custom.
********************************************************************************/
#ifndef BASE_MATRIX_COMPILED_SPARSE_REDUCED_TEMPLATE_HPP_
#define BASE_MATRIX_COMPILED_SPARSE_REDUCED_TEMPLATE_HPP_

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_templates.hpp"
#include "base_matrix_vector.hpp"

#include <type_traits>
#include <utility>

namespace Base {
namespace Matrix {

namespace RowUnroll {

using TemplatesOperation::SparseDetail::IndexPack;
using TemplatesOperation::SparseDetail::MakeIndexPack;

// Preserve the original += contract for ordinary scalars. Only bool requires
// a promoted sum followed by conversion to avoid MSVC's C4804 warning.
template <bool IsBool> struct Accumulate {
  template <typename T, typename Product>
  static void run(T &sum, Product &&product) {
    sum += std::forward<Product>(product);
  }
};
template <> struct Accumulate<true> {
  template <typename T, typename Product>
  static void run(T &sum, Product &&product) {
    sum = static_cast<T>(sum + std::forward<Product>(product));
  }
};

template <typename Columns, typename Positions> struct RowKernel;

template <std::size_t... Columns, std::size_t... Positions>
struct RowKernel<IndexPack<Columns...>, IndexPack<Positions...>> {
  /**
   * @brief Computes one sparse matrix row against one dense vector.
   *
   * The products are accumulated in CSR order.
   *
   * @tparam T Value type of the matrix and vector.
   * @tparam Columns Compile-time dense vector indices for the row entries.
   * @tparam Positions Compile-time CSR positions for the row entries.
   * @param[in]  a Pointer to the values in the sparse matrix row.
   * @param[in]  b Pointer to the dense input vector.
   * @param[out] out Output value for this row.
   */
  template <typename T> static void run(const T *a, const T *b, T &out) {
    // Empty rows do not read either input pointer.
    static_cast<void>(a);
    static_cast<void>(b);
    T sum = static_cast<T>(0);

    // C++11 list initialization sequences the additions in CSR order. Cast
    // to void to exclude a user-defined comma operator. Read b by value, as
    // Matrix::get does, and preserve Matrix::set's const assignment operand.
    const int sequenced[] = {
        0, (static_cast<void>(sum += a[Positions] * static_cast<T>(b[Columns])),
            0)...};
    static_cast<void>(sequenced);
    out = static_cast<const T &>(sum);
  }
};

// This compile-time adapter maps the complete CSR pattern to a row kernel that
// can be shared by matching rows, other matrix sizes, and every output column.
template <typename Indices, typename Pointers, std::size_t Row,
          typename Positions>
struct RowDescriptor;

template <typename Indices, typename Pointers, std::size_t Row,
          std::size_t... Positions>
struct RowDescriptor<Indices, Pointers, Row, IndexPack<Positions...>> {
  using type =
      RowKernel<IndexPack<Indices::list[Pointers::list[Row] + Positions]...>,
                IndexPack<Positions...>>;
};

template <typename Indices, typename Pointers, std::size_t Row>
using RowKernelFor = typename RowDescriptor<
    Indices, Pointers, Row,
    typename MakeIndexPack<Pointers::list[Row + 1] -
                           Pointers::list[Row]>::type>::type;

template <typename Indices, typename Pointers, std::size_t M, typename Rows>
struct ColumnKernel;

template <typename Indices, typename Pointers, std::size_t M,
          std::size_t... Rows>
struct ColumnKernel<Indices, Pointers, M, IndexPack<Rows...>> {
  /**
   * @brief Computes one output column for all rows of a sparse matrix.
   *
   * Rows are processed in descending order, matching the original traversal.
   *
   * @tparam T Value type of the matrices.
   * @tparam Indices Type representing the CSR column indices.
   * @tparam Pointers Type representing the CSR row pointers.
   * @tparam M Number of rows in the sparse matrix.
   * @tparam Rows Compile-time row traversal indices.
   * @param[in]  a Pointer to the sparse matrix values.
   * @param[in]  b Pointer to the dense input column.
   * @param[out] out Pointer to the output column.
   */
  template <typename T> static void run(const T *a, const T *b, T *out) {
    // Match the original descending row traversal. Avoid even null + 0 for
    // the empty std::vector used to store an entirely empty sparse matrix.
    const int sequenced[] = {
        0, (RowKernelFor<Indices, Pointers, M - 1 - Rows>::run(
                Pointers::list[M - 1 - Rows] == 0
                    ? a
                    : a + Pointers::list[M - 1 - Rows],
                b, out[M - 1 - Rows]),
            0)...};
    static_cast<void>(sequenced);
  }
};

// Raw-pointer kernels must retain Matrix::get's compile-time column bounds
// check. Balanced constexpr recursion avoids per-element validation types and
// keeps the evaluation depth small. Only indices in used row spans are checked.
template <typename Indices, typename Pointers, std::size_t Bound>
struct UsedIndexBounds {
  /**
   * @brief Checks whether the CSR indices in a range are within a bound.
   *
   * @tparam Indices Type representing the CSR column indices.
   * @tparam Pointers Type representing the CSR row pointers.
   * @tparam Bound Exclusive upper bound for valid indices.
   * @param[in] begin Inclusive start position in the CSR index array.
   * @param[in] end Exclusive end position in the CSR index array.
   * @return True if all indices in the range are less than Bound.
   */
  static constexpr bool span(std::size_t begin, std::size_t end) {
    return begin > end || end > Indices::size ? false
           : begin == end                     ? true
           : end - begin == 1                 ? Indices::list[begin] < Bound
                              : span(begin, begin + (end - begin) / 2) &&
                                    span(begin + (end - begin) / 2, end);
  }

  /**
   * @brief Checks whether all row spans contain valid CSR column indices.
   *
   * @tparam Indices Type representing the CSR column indices.
   * @tparam Pointers Type representing the CSR row pointers.
   * @tparam Bound Exclusive upper bound for valid indices.
   * @param[in] begin Inclusive start row.
   * @param[in] end Exclusive end row.
   * @return True if all indices in the specified rows are less than Bound.
   */
  static constexpr bool rows(std::size_t begin, std::size_t end) {
    return begin == end ? true
           : end - begin == 1
               ? span(Pointers::list[begin], Pointers::list[begin + 1])
               : rows(begin, begin + (end - begin) / 2) &&
                     rows(begin + (end - begin) / 2, end);
  }
};

template <typename T> struct RunDispatch {
  /**
   * @brief Multiplies a compiled sparse matrix by a dense matrix.
   *
   * The unrolled kernels process each output column using compile-time CSR
   * indices after validating the used column indices.
   *
   * @tparam T Value type of the matrices.
   * @tparam M Number of rows in the sparse matrix.
   * @tparam N Number of columns in the sparse matrix.
   * @tparam K Number of columns in the dense input and output matrices.
   * @tparam Indices Type representing the CSR column indices.
   * @tparam Pointers Type representing the CSR row pointers.
   * @tparam Columns Compile-time output column traversal indices.
   * @param[in]  A Input sparse matrix.
   * @param[in]  B Input dense matrix.
   * @param[out] Y Output dense matrix containing the product.
   * @param[in]  IndexPack Compile-time output column indices.
   */
  template <std::size_t M, std::size_t N, std::size_t K, typename Indices,
            typename Pointers, std::size_t... Columns>
  static void run(const CompiledSparseMatrix<T, M, N, Indices, Pointers> &A,
                  const Matrix<T, N, K> &B, Matrix<T, M, K> &Y,
                  IndexPack<Columns...>) {
    static_assert(UsedIndexBounds<Indices, Pointers, N>::rows(0, M),
                  "CSR column index outside dense input row range");
    using Kernel =
        ColumnKernel<Indices, Pointers, M, typename MakeIndexPack<M>::type>;

    // Each dense column is contiguous for both array and vector storage.
    // Preserve the original descending output-column traversal.
    const int sequenced[] = {
        0, (Kernel::run(A.values.data(), B.data[K - 1 - Columns].data(),
                        Y.data[K - 1 - Columns].data()),
            0)...};
    static_cast<void>(sequenced);
  }
};

// vector<bool> has no usable data(), so bool must not use RunDispatch. The
// caller selects the recursive helpers for bool.
} // namespace RowUnroll

namespace SparseReduced {

using TemplatesOperation::SparseDetail::IndexPack;
using TemplatesOperation::SparseDetail::MakeIndexPack;

// Generic Y_Type entry points also accept adapted or mixed-scalar outputs.
// Keep those on the existing path rather than narrowing their public contract.
template <typename Output, typename T, std::size_t Rows, std::size_t Columns>
struct SupportsOutput : std::false_type {};
template <typename T, std::size_t M, std::size_t N, typename Indices,
          typename Pointers>
struct SupportsOutput<CompiledSparseMatrix<T, M, N, Indices, Pointers>, T, M, N>
    : std::true_type {};

// Select before instantiating a pointer kernel: vector<bool> has proxy storage.
template <bool UseLegacy> struct Select;
template <> struct Select<false> {
  template <typename Kernel, typename Legacy, typename... Args>
  static void run(Args &...args) {
    Kernel::run(args...);
  }
};
template <> struct Select<true> {
  template <typename Kernel, typename Legacy, typename... Args>
  static void run(Args &...args) {
    Legacy::compute(args...);
  }
};

template <bool UseLegacy> struct SelectApply : Select<false> {};
template <> struct SelectApply<true> {
  template <typename Kernel, typename Legacy, typename... Args>
  static void run(Args &...args) {
    Legacy::apply(args...);
  }
};

// Legacy inclusive-end loops cannot represent an empty CSR row. Select before
// instantiating their body, including when the reduced kernels are disabled.
template <bool HasValues> struct NonEmptyRow {
  template <typename Loop, typename... Args> static void run(Args &...args) {
    Loop::compute(args...);
  }
};
template <> struct NonEmptyRow<false> {
  template <typename Loop, typename... Args> static void run(Args &...) {}
};

/* Elementwise operations and diagonal products */

namespace Elementwise {

// These helpers carry offsets for one row, not the matrix's complete CSR type.
// A zero offset must also be valid for an empty vector's null data pointer.
template <std::size_t Offset, typename T> inline T *offset(T *values) {
  return Offset == 0 ? values : values + Offset;
}

template <typename T> inline void assign(T &out, const T &value) {
  out = value;
}

template <std::size_t Position> struct Read {
  template <typename T> static T run(const T *values) {
    T value = static_cast<T>(0);
    value = values[Position];
    return value;
  }
};
template <> struct Read<StaticSparseLookup::missing> {
  template <typename T> static T run(const T *) {
    T value = static_cast<T>(0);
    return value;
  }
};

template <typename Positions> struct Write;
template <std::size_t... Positions> struct Write<IndexPack<Positions...>> {
  template <typename T> static void run(T *out, const T &value) {
    static_cast<void>(out);
    static_cast<void>(value);
    const int sequenced[] = {0,
                             (static_cast<void>(out[Positions] = value), 0)...};
    static_cast<void>(sequenced);
  }
};

// Normal CSR has one match. Duplicate indices retain ascending writes to all
// matches; an absent output coordinate still evaluates the arithmetic once.
template <
    typename Indices, std::size_t Column, std::size_t Begin, std::size_t End,
    std::size_t First = StaticSparseLookup::first<Indices>(Column, Begin, End),
    std::size_t Last = StaticSparseLookup::last<Indices>(Column, Begin, End)>
struct MatchSlots {
  static constexpr std::size_t middle = First + (Last - First) / 2 + 1;
  using type = typename TemplatesOperation::SparseDetail::ConcatenateIndexPacks<
      typename MatchSlots<Indices, Column, First, middle>::type,
      typename MatchSlots<Indices, Column, middle, Last + 1>::type>::type;
};
template <typename Indices, std::size_t Column, std::size_t Begin,
          std::size_t End, std::size_t Position>
struct MatchSlots<Indices, Column, Begin, End, Position, Position> {
  using type = IndexPack<Position>;
};
template <typename Indices, std::size_t Column, std::size_t Begin,
          std::size_t End>
struct MatchSlots<Indices, Column, Begin, End, StaticSparseLookup::missing,
                  StaticSparseLookup::missing> {
  using type = IndexPack<>;
};

struct Add {
  template <typename T> static void dense(const T &a, T &y) {
    Elementwise::assign<T>(y, static_cast<T>(static_cast<const T &>(y)) + a);
  }
  template <typename Reader, typename Writer, typename T>
  static void sparse(const T &a, T *y) {
    Writer::run(y, Reader::run(y) + a);
  }
};
struct Subtract {
  template <typename T> static void dense(const T &a, T &y) {
    Elementwise::assign<T>(y, static_cast<T>(static_cast<const T &>(y)) - a);
  }
  template <typename Reader, typename Writer, typename T>
  static void sparse(const T &a, T *y) {
    Writer::run(y, Reader::run(y) - a);
  }
};

template <typename Positions> struct NegateRow;
template <std::size_t... Positions> struct NegateRow<IndexPack<Positions...>> {
  template <typename T> static void run(const T *a, T *y) {
    static_cast<void>(a);
    static_cast<void>(y);
    const int sequenced[] = {
        0, (static_cast<void>(y[Positions] = -a[Positions]), 0)...};
    static_cast<void>(sequenced);
  }
};

template <typename Pointers, std::size_t M, typename Rows> struct NegateRows;
template <typename Pointers, std::size_t M, std::size_t... Rows>
struct NegateRows<Pointers, M, IndexPack<Rows...>> {
  template <typename T> static void run(const T *a, T *y) {
    const int sequenced[] = {
        0,
        (NegateRow<typename MakeIndexPack<Pointers::list[M - Rows] -
                                          Pointers::list[M - 1 - Rows]>::type>::
             run(Elementwise::offset<Pointers::list[M - 1 - Rows]>(a),
                 Elementwise::offset<Pointers::list[M - 1 - Rows]>(y)),
         0)...};
    static_cast<void>(sequenced);
  }
};
struct Negate {
  template <typename T, std::size_t M, std::size_t N, typename Indices,
            typename Pointers>
  static void run(const CompiledSparseMatrix<T, M, N, Indices, Pointers> &a,
                  CompiledSparseMatrix<T, M, N, Indices, Pointers> &y) {
    NegateRows<Pointers, M, typename MakeIndexPack<M>::type>::run(
        a.values.data(), y.values.data());
  }
};

template <typename Positions> struct ScaleValues;
template <std::size_t... Positions>
struct ScaleValues<IndexPack<Positions...>> {
  template <typename T> static void run(const T *a, const T &scalar, T *y) {
    static_cast<void>(a);
    static_cast<void>(scalar);
    static_cast<void>(y);
    const int sequenced[] = {
        0, (static_cast<void>(y[Positions] = scalar * a[Positions]), 0)...};
    static_cast<void>(sequenced);
  }
};
struct Scale {
  template <typename T, std::size_t M, std::size_t N, typename Indices,
            typename Pointers>
  static void run(const CompiledSparseMatrix<T, M, N, Indices, Pointers> &a,
                  const T &scalar,
                  CompiledSparseMatrix<T, M, N, Indices, Pointers> &y) {
    ScaleValues<typename MakeIndexPack<Pointers::list[M]>::type>::run(
        a.values.data(), scalar, y.values.data());
  }
};

template <typename Operation, std::size_t Row, typename Columns,
          typename Positions>
struct DenseRow;
template <typename Operation, std::size_t Row, std::size_t... Columns,
          std::size_t... Positions>
struct DenseRow<Operation, Row, IndexPack<Columns...>,
                IndexPack<Positions...>> {
  template <typename T, typename Storage>
  static void run(const T *a, Storage &y) {
    static_cast<void>(a);
    static_cast<void>(y);
    const int sequenced[] = {
        0, (Operation::dense(a[Positions], y[Columns][Row]), 0)...};
    static_cast<void>(sequenced);
  }
  template <typename T, typename Storage>
  static void copy(T *a, const Storage &b) {
    static_cast<void>(a);
    static_cast<void>(b);
    const int sequenced[] = {
        0, (static_cast<void>(a[Positions] = static_cast<T>(b[Columns][Row])),
            0)...};
    static_cast<void>(sequenced);
  }
};
template <typename Operation, typename Indices, typename Pointers,
          std::size_t Row, typename Positions>
struct DenseRowDescriptor;
template <typename Operation, typename Indices, typename Pointers,
          std::size_t Row, std::size_t... Positions>
struct DenseRowDescriptor<Operation, Indices, Pointers, Row,
                          IndexPack<Positions...>> {
  using type =
      DenseRow<Operation, Row,
               IndexPack<Indices::list[Pointers::list[Row] + Positions]...>,
               IndexPack<Positions...>>;
};
template <typename Operation, typename Indices, typename Pointers,
          std::size_t Row>
using DenseRowFor = typename DenseRowDescriptor<
    Operation, Indices, Pointers, Row,
    typename MakeIndexPack<Pointers::list[Row + 1] -
                           Pointers::list[Row]>::type>::type;

template <typename Operation, typename Indices, typename Pointers,
          std::size_t M, typename Rows>
struct DenseRows;
template <typename Operation, typename Indices, typename Pointers,
          std::size_t M, std::size_t... Rows>
struct DenseRows<Operation, Indices, Pointers, M, IndexPack<Rows...>> {
  template <typename T, typename Storage>
  static void run(const T *a, Storage &y) {
    const int sequenced[] = {
        0, (DenseRowFor<Operation, Indices, Pointers, M - 1 - Rows>::run(
                Elementwise::offset<Pointers::list[M - 1 - Rows]>(a), y),
            0)...};
    static_cast<void>(sequenced);
  }
  template <typename T, typename Storage>
  static void copy(T *a, const Storage &b) {
    const int sequenced[] = {
        0, (DenseRowFor<Operation, Indices, Pointers, M - 1 - Rows>::copy(
                Elementwise::offset<Pointers::list[M - 1 - Rows]>(a), b),
            0)...};
    static_cast<void>(sequenced);
  }
};
template <typename Operation> struct DenseUpdate {
  template <typename T, std::size_t M, std::size_t N, typename Indices,
            typename Pointers>
  static void run(const CompiledSparseMatrix<T, M, N, Indices, Pointers> &a,
                  Matrix<T, M, N> &y) {
    static_assert(RowUnroll::UsedIndexBounds<Indices, Pointers, N>::rows(0, M),
                  "CSR column index outside dense output range");
    DenseRows<Operation, Indices, Pointers, M,
              typename MakeIndexPack<M>::type>::run(a.values.data(), y.data);
  }
};
struct CopyDense {
  template <typename T, std::size_t M, std::size_t N, typename Indices,
            typename Pointers>
  static void run(CompiledSparseMatrix<T, M, N, Indices, Pointers> &a,
                  const Matrix<T, M, N> &b) {
    static_assert(RowUnroll::UsedIndexBounds<Indices, Pointers, N>::rows(0, M),
                  "CSR column index outside dense input range");
    DenseRows<Add, Indices, Pointers, M, typename MakeIndexPack<M>::type>::copy(
        a.values.data(), b.data);
  }
};

template <typename Operation, std::size_t Position, typename Slots>
struct UpdateCell {
  template <typename T> static void run(const T &a, T *y) {
    Operation::template sparse<Read<Position>, Write<Slots>>(a, y);
  }
};
template <typename Positions, typename... Cells> struct UpdateRow;
template <std::size_t... Positions, typename... Cells>
struct UpdateRow<IndexPack<Positions...>, Cells...> {
  template <typename T> static void run(const T *a, T *y) {
    static_cast<void>(a);
    static_cast<void>(y);
    const int sequenced[] = {0, (Cells::run(a[Positions], y), 0)...};
    static_cast<void>(sequenced);
  }
};
template <typename Operation, typename Indices, typename Pointers,
          typename OutputIndices, typename OutputPointers, std::size_t Row,
          typename Positions>
struct UpdateRowDescriptor;
template <typename Operation, typename Indices, typename Pointers,
          typename OutputIndices, typename OutputPointers, std::size_t Row,
          std::size_t... Positions>
struct UpdateRowDescriptor<Operation, Indices, Pointers, OutputIndices,
                           OutputPointers, Row, IndexPack<Positions...>> {
  using type = UpdateRow<
      IndexPack<Positions...>,
      UpdateCell<
          Operation,
          StaticSparseLookup::last<OutputIndices>(
              Indices::list[Pointers::list[Row] + Positions],
              OutputPointers::list[Row], OutputPointers::list[Row + 1]),
          typename MatchSlots<OutputIndices,
                              Indices::list[Pointers::list[Row] + Positions],
                              OutputPointers::list[Row],
                              OutputPointers::list[Row + 1]>::type>...>;
};
template <typename Operation, typename Indices, typename Pointers,
          typename OutputIndices, typename OutputPointers, std::size_t Row>
using UpdateRowFor = typename UpdateRowDescriptor<
    Operation, Indices, Pointers, OutputIndices, OutputPointers, Row,
    typename MakeIndexPack<Pointers::list[Row + 1] -
                           Pointers::list[Row]>::type>::type;

template <typename Operation, typename Indices, typename Pointers,
          typename OutputIndices, typename OutputPointers, std::size_t M,
          typename Rows>
struct UpdateRows;
template <typename Operation, typename Indices, typename Pointers,
          typename OutputIndices, typename OutputPointers, std::size_t M,
          std::size_t... Rows>
struct UpdateRows<Operation, Indices, Pointers, OutputIndices, OutputPointers,
                  M, IndexPack<Rows...>> {
  template <typename T> static void run(const T *a, T *y) {
    const int sequenced[] = {
        0, (UpdateRowFor<Operation, Indices, Pointers, OutputIndices,
                         OutputPointers, M - 1 - Rows>::
                run(Elementwise::offset<Pointers::list[M - 1 - Rows]>(a), y),
            0)...};
    static_cast<void>(sequenced);
  }
};
template <typename Operation> struct SparseUpdate {
  template <typename T, std::size_t M, std::size_t N, typename Indices,
            typename Pointers, typename OutputIndices, typename OutputPointers>
  static void
  run(const CompiledSparseMatrix<T, M, N, Indices, Pointers> &a,
      CompiledSparseMatrix<T, M, N, OutputIndices, OutputPointers> &y) {
    static_assert(RowUnroll::UsedIndexBounds<Indices, Pointers, N>::rows(0, M),
                  "CSR column index outside sparse output range");
    UpdateRows<Operation, Indices, Pointers, OutputIndices, OutputPointers, M,
               typename MakeIndexPack<M>::type>::run(a.values.data(),
                                                     y.values.data());
  }
};

template <typename Positions, typename Columns> struct DiagonalRow;
template <std::size_t... Positions, std::size_t... Columns>
struct DiagonalRow<IndexPack<Positions...>, IndexPack<Columns...>> {
  template <typename T> static void right(const T *a, const T *b, T *y) {
    static_cast<void>(a);
    static_cast<void>(b);
    static_cast<void>(y);
    const int sequenced[] = {
        0, (static_cast<void>(y[Positions] = a[Positions] * b[Columns]), 0)...};
    static_cast<void>(sequenced);
  }
  template <typename T> static void left(const T &a, const T *b, T *y) {
    static_cast<void>(a);
    static_cast<void>(b);
    static_cast<void>(y);
    const int sequenced[] = {
        0, (static_cast<void>(y[Positions] = b[Positions] * a), 0)...};
    static_cast<void>(sequenced);
  }
};
template <typename Indices, typename Pointers, std::size_t Row,
          typename Positions>
struct DiagonalRowDescriptor;
template <typename Indices, typename Pointers, std::size_t Row,
          std::size_t... Positions>
struct DiagonalRowDescriptor<Indices, Pointers, Row, IndexPack<Positions...>> {
  using type =
      DiagonalRow<IndexPack<Positions...>,
                  IndexPack<Indices::list[Pointers::list[Row] + Positions]...>>;
};
template <typename Indices, typename Pointers, std::size_t Row>
using DiagonalRowFor = typename DiagonalRowDescriptor<
    Indices, Pointers, Row,
    typename MakeIndexPack<Pointers::list[Row + 1] -
                           Pointers::list[Row]>::type>::type;
template <typename Indices, typename Pointers, std::size_t M, typename Rows>
struct DiagonalRows;
template <typename Indices, typename Pointers, std::size_t M,
          std::size_t... Rows>
struct DiagonalRows<Indices, Pointers, M, IndexPack<Rows...>> {
  template <typename T> static void right(const T *a, const T *b, T *y) {
    const int sequenced[] = {
        0, (DiagonalRowFor<Indices, Pointers, M - 1 - Rows>::right(
                Elementwise::offset<Pointers::list[M - 1 - Rows]>(a), b,
                Elementwise::offset<Pointers::list[M - 1 - Rows]>(y)),
            0)...};
    static_cast<void>(sequenced);
  }
  template <typename T> static void left(const T *a, const T *b, T *y) {
    const int sequenced[] = {
        0, (DiagonalRowFor<Indices, Pointers, M - 1 - Rows>::left(
                a[M - 1 - Rows],
                Elementwise::offset<Pointers::list[M - 1 - Rows]>(b),
                Elementwise::offset<Pointers::list[M - 1 - Rows]>(y)),
            0)...};
    static_cast<void>(sequenced);
  }
};
struct RightDiagonal {
  template <typename T, std::size_t M, std::size_t N, typename Indices,
            typename Pointers>
  static void run(const CompiledSparseMatrix<T, M, N, Indices, Pointers> &a,
                  const DiagMatrix<T, N> &b,
                  CompiledSparseMatrix<T, M, N, Indices, Pointers> &y) {
    DiagonalRows<Indices, Pointers, M, typename MakeIndexPack<M>::type>::right(
        a.values.data(), b.data.data(), y.values.data());
  }
};
struct LeftDiagonal {
  template <typename T, std::size_t M, std::size_t N, typename Indices,
            typename Pointers>
  static void run(const DiagMatrix<T, M> &a,
                  const CompiledSparseMatrix<T, M, N, Indices, Pointers> &b,
                  CompiledSparseMatrix<T, M, N, Indices, Pointers> &y) {
    DiagonalRows<Indices, Pointers, M, typename MakeIndexPack<M>::type>::left(
        a.data.data(), b.values.data(), y.values.data());
  }
};

template <typename Indices, typename Pointers, std::size_t M, typename Rows>
struct SetDiagonalRows;
template <typename Indices, typename Pointers, std::size_t M,
          std::size_t... Rows>
struct SetDiagonalRows<Indices, Pointers, M, IndexPack<Rows...>> {
  template <typename T> static void run(T *y, const T *b) {
    const int sequenced[] = {
        0, (Write<typename MatchSlots<
                Indices, M - 1 - Rows, Pointers::list[M - 1 - Rows],
                Pointers::list[M - Rows]>::type>::run(y, b[M - 1 - Rows]),
            0)...};
    static_cast<void>(sequenced);
  }
};
struct SetDiagonal {
  template <typename T, std::size_t M, typename Indices, typename Pointers>
  static void run(CompiledSparseMatrix<T, M, M, Indices, Pointers> &y,
                  const DiagMatrix<T, M> &b) {
    SetDiagonalRows<Indices, Pointers, M, typename MakeIndexPack<M>::type>::run(
        y.values.data(), b.data.data());
  }
};

template <std::size_t Position, typename Slots> struct TransposeDiagonalCell {
  template <typename T> static void run(const T &a, const T &b, T *y) {
    Write<Slots>::run(y, Read<Position>::run(y) + b * a);
  }
};
template <typename Positions, typename... Cells> struct TransposeDiagonalRow;
template <std::size_t... Positions, typename... Cells>
struct TransposeDiagonalRow<IndexPack<Positions...>, Cells...> {
  template <typename T> static void run(const T &a, const T *b, T *y) {
    static_cast<void>(a);
    static_cast<void>(b);
    static_cast<void>(y);
    const int sequenced[] = {0, (Cells::run(a, b[Positions], y), 0)...};
    static_cast<void>(sequenced);
  }
};
template <typename Indices, typename Pointers, typename OutputIndices,
          typename OutputPointers, std::size_t Row, typename Positions>
struct TransposeDiagonalRowDescriptor;
template <typename Indices, typename Pointers, typename OutputIndices,
          typename OutputPointers, std::size_t Row, std::size_t... Positions>
struct TransposeDiagonalRowDescriptor<Indices, Pointers, OutputIndices,
                                      OutputPointers, Row,
                                      IndexPack<Positions...>> {
  using type = TransposeDiagonalRow<
      IndexPack<Positions...>,
      TransposeDiagonalCell<
          StaticSparseLookup::last<OutputIndices>(
              Row,
              OutputPointers::list[Indices::list[Pointers::list[Row] +
                                                 Positions]],
              OutputPointers::list
                  [Indices::list[Pointers::list[Row] + Positions] + 1]),
          typename MatchSlots<
              OutputIndices, Row,
              OutputPointers::list[Indices::list[Pointers::list[Row] +
                                                 Positions]],
              OutputPointers::list[Indices::list[Pointers::list[Row] +
                                                 Positions] +
                                   1]>::type>...>;
};
template <typename Indices, typename Pointers, typename OutputIndices,
          typename OutputPointers, std::size_t M, typename Rows>
struct TransposeDiagonalRows;
template <typename Indices, typename Pointers, typename OutputIndices,
          typename OutputPointers, std::size_t M, std::size_t... Rows>
struct TransposeDiagonalRows<Indices, Pointers, OutputIndices, OutputPointers,
                             M, IndexPack<Rows...>> {
  template <typename T> static void run(const T *a, const T *b, T *y) {
    const int sequenced[] = {
        0,
        (TransposeDiagonalRowDescriptor<
             Indices, Pointers, OutputIndices, OutputPointers, M - 1 - Rows,
             typename MakeIndexPack<Pointers::list[M - Rows] -
                                    Pointers::list[M - 1 - Rows]>::type>::type::
             run(a[M - 1 - Rows],
                 Elementwise::offset<Pointers::list[M - 1 - Rows]>(b), y),
         0)...};
    static_cast<void>(sequenced);
  }
};
struct TransposeDiagonal {
  template <typename T, std::size_t M, std::size_t N, typename Indices,
            typename Pointers, typename OutputIndices, typename OutputPointers>
  static void
  run(const DiagMatrix<T, M> &a,
      const CompiledSparseMatrix<T, M, N, Indices, Pointers> &b,
      CompiledSparseMatrix<T, N, M, OutputIndices, OutputPointers> &y) {
    static_assert(RowUnroll::UsedIndexBounds<Indices, Pointers, N>::rows(0, M),
                  "CSR column index outside sparse output range");
    TransposeDiagonalRows<Indices, Pointers, OutputIndices, OutputPointers, M,
                          typename MakeIndexPack<M>::type>::run(a.data.data(),
                                                                b.values.data(),
                                                                y.values
                                                                    .data());
  }
};

} // namespace Elementwise

/* Products with dense matrices and vectors */

namespace DenseProduct {

// Keep complete CSR types in these adapters; arithmetic kernels carry only
// the column indices and relative positions of the row being processed.
template <template <typename, typename> class Kernel, typename Indices,
          typename Pointers, std::size_t Row, typename Positions>
struct RowDescriptor;

template <template <typename, typename> class Kernel, typename Indices,
          typename Pointers, std::size_t Row, std::size_t... Positions>
struct RowDescriptor<Kernel, Indices, Pointers, Row, IndexPack<Positions...>> {
  using type =
      Kernel<IndexPack<Indices::list[Pointers::list[Row] + Positions]...>,
             IndexPack<Positions...>>;
};

template <template <typename, typename> class Kernel, typename Indices,
          typename Pointers, std::size_t Row>
using RowKernel = typename RowDescriptor<
    Kernel, Indices, Pointers, Row,
    typename MakeIndexPack<Pointers::list[Row + 1] -
                           Pointers::list[Row]>::type>::type;

// An entirely empty sparse matrix may use a null vector data pointer.
template <typename T>
inline const T *row_values(const T *values, std::size_t start) {
  return start == 0 ? values : values + start;
}

// Matrix::set assigns through a const T reference, including when the
// expression first undergoes an implicit conversion to T.
template <typename T> inline void assign(T &output, const T &value) {
  output = value;
}

template <typename Columns, typename Positions> struct VectorDot;

template <std::size_t... Columns, std::size_t... Positions>
struct VectorDot<IndexPack<Columns...>, IndexPack<Positions...>> {
  template <typename T, std::size_t N>
  static void run(const T *values, const Vector<T, N> &input, T &output) {
    static_cast<void>(values);
    static_cast<void>(input);
    T sum = static_cast<T>(0);
    // Vector::operator[] intentionally retains its out-of-range clamping.
    const int sequenced[] = {
        0,
        (static_cast<void>(sum += values[Positions] * input[Columns]), 0)...};
    static_cast<void>(sequenced);
    output = sum;
  }
};

template <typename Columns, typename Positions> struct VectorScatter;

template <std::size_t... Columns, std::size_t... Positions>
struct VectorScatter<IndexPack<Columns...>, IndexPack<Positions...>> {
  template <typename T, std::size_t N, std::size_t K>
  static void run(const T *values, const ColVector<T, N> &input,
                  ColVector<T, K> &output, std::size_t row) {
    static_cast<void>(values);
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(row);
    const int sequenced[] = {
        0,
        (static_cast<void>(output[Columns] += values[Positions] * input[row]),
         0)...};
    static_cast<void>(sequenced);
  }
};

// Sparse*dense-transpose reads through Matrix::get (a value), whereas
// dense*sparse-transpose reads through operator() (a const reference).
template <bool ByValue> struct DenseRead;
template <> struct DenseRead<true> {
  template <typename T, std::size_t M, std::size_t N>
  static T get(const Matrix<T, M, N> &matrix, std::size_t row,
               std::size_t column) {
    return matrix(row, column);
  }
};
template <> struct DenseRead<false> {
  template <typename T, std::size_t M, std::size_t N>
  static const T &get(const Matrix<T, M, N> &matrix, std::size_t row,
                      std::size_t column) {
    return matrix(row, column);
  }
};

template <bool ByValue, typename Columns, typename Positions> struct MatrixDot;

template <bool ByValue, std::size_t... Columns, std::size_t... Positions>
struct MatrixDot<ByValue, IndexPack<Columns...>, IndexPack<Positions...>> {
  template <typename T, std::size_t M, std::size_t N>
  static void run(const T *values, const Matrix<T, M, N> &input,
                  std::size_t row, T &output) {
    static_cast<void>(values);
    static_cast<void>(input);
    static_cast<void>(row);
    T sum = static_cast<T>(0);
    const int sequenced[] = {
        0,
        (static_cast<void>(sum += values[Positions] *
                                  DenseRead<ByValue>::get(input, row, Columns)),
         0)...};
    static_cast<void>(sequenced);
    output = sum;
  }

  template <typename T, std::size_t M, std::size_t N, std::size_t... Rows>
  static void rows(const T *values, const Matrix<T, M, N> &input, T *output,
                   IndexPack<Rows...>) {
    const int sequenced[] = {
        0, (run(values, input, M - 1 - Rows, output[M - 1 - Rows]), 0)...};
    static_cast<void>(sequenced);
  }
};

template <typename Columns, typename Positions>
using MatrixDotValue = MatrixDot<true, Columns, Positions>;
template <typename Columns, typename Positions>
using MatrixDotReference = MatrixDot<false, Columns, Positions>;

template <bool TransposeInput, typename Columns, typename Positions>
struct MatrixScatter;

template <bool TransposeInput, std::size_t... Columns, std::size_t... Positions>
struct MatrixScatter<TransposeInput, IndexPack<Columns...>,
                     IndexPack<Positions...>> {
  template <typename T, std::size_t ARow, std::size_t AColumn, std::size_t M,
            std::size_t K>
  static void row(const T *values, const Matrix<T, ARow, AColumn> &input,
                  Matrix<T, M, K> &output, std::size_t inner,
                  std::size_t output_row) {
    static_cast<void>(values);
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(inner);
    static_cast<void>(output_row);
    // Do not hoist the dense value: the legacy loop reads it for each entry,
    // which also preserves behavior when input and output alias.
    const int sequenced[] = {
        0,
        (DenseProduct::assign<T>(
             output(output_row, Columns),
             static_cast<T>(
                 static_cast<const T &>(output(output_row, Columns))) +
                 values[Positions] *
                     static_cast<T>(TransposeInput ? input(inner, output_row)
                                                   : input(output_row, inner))),
         0)...};
    static_cast<void>(sequenced);
  }

  template <typename T, std::size_t ARow, std::size_t AColumn, std::size_t M,
            std::size_t K, std::size_t... Rows>
  static void run(const T *values, const Matrix<T, ARow, AColumn> &input,
                  Matrix<T, M, K> &output, std::size_t inner,
                  IndexPack<Rows...>) {
    const int sequenced[] = {
        0, (row(values, input, output, inner, M - 1 - Rows), 0)...};
    static_cast<void>(sequenced);
  }
};

template <typename Columns, typename Positions> struct TransposeScatter;

template <std::size_t... Columns, std::size_t... Positions>
struct TransposeScatter<IndexPack<Columns...>, IndexPack<Positions...>> {
  template <typename T, std::size_t N, std::size_t K>
  static void run(const T *values, const Matrix<T, N, K> &input, T *output,
                  std::size_t row, std::size_t column) {
    static_cast<void>(values);
    static_cast<void>(input);
    static_cast<void>(output);
    static_cast<void>(row);
    static_cast<void>(column);
    const int sequenced[] = {
        0, (DenseProduct::assign<T>(
                output[Columns],
                static_cast<T>(static_cast<const T &>(output[Columns])) +
                    values[Positions] * static_cast<T>(input(row, column))),
            0)...};
    static_cast<void>(sequenced);
  }
};

struct SparseVector {
  template <typename T, std::size_t M, std::size_t N, typename Indices,
            typename Pointers, std::size_t... Rows>
  static void rows(const CompiledSparseMatrix<T, M, N, Indices, Pointers> &A,
                   const Vector<T, N> &b, Vector<T, M> &y, IndexPack<Rows...>) {
    const int sequenced[] = {
        0, (RowKernel<VectorDot, Indices, Pointers, M - 1 - Rows>::run(
                DenseProduct::row_values(A.values.data(),
                                         Pointers::list[M - 1 - Rows]),
                b, y[M - 1 - Rows]),
            0)...};
    static_cast<void>(sequenced);
  }

  template <typename T, std::size_t M, std::size_t N, typename Indices,
            typename Pointers>
  static void run(const CompiledSparseMatrix<T, M, N, Indices, Pointers> &A,
                  const Vector<T, N> &b, Vector<T, M> &y) {
    rows(A, b, y, typename MakeIndexPack<M>::type{});
  }
};

struct ColVectorSparse {
  template <typename T, std::size_t N, std::size_t K, typename Indices,
            typename Pointers, std::size_t... Rows>
  static void rows(const ColVector<T, N> &a,
                   const CompiledSparseMatrix<T, N, K, Indices, Pointers> &B,
                   ColVector<T, K> &y, IndexPack<Rows...>) {
    const int sequenced[] = {
        0, (RowKernel<VectorScatter, Indices, Pointers, N - 1 - Rows>::run(
                DenseProduct::row_values(B.values.data(),
                                         Pointers::list[N - 1 - Rows]),
                a, y, N - 1 - Rows),
            0)...};
    static_cast<void>(sequenced);
  }

  template <typename T, std::size_t N, std::size_t K, typename Indices,
            typename Pointers>
  static void run(const ColVector<T, N> &a,
                  const CompiledSparseMatrix<T, N, K, Indices, Pointers> &B,
                  ColVector<T, K> &y) {
    rows(a, B, y, typename MakeIndexPack<N>::type{});
  }
};

template <bool TransposeInput> struct DenseSparse {
  template <typename Columns, typename Positions>
  using Scatter = MatrixScatter<TransposeInput, Columns, Positions>;

  template <typename T, std::size_t ARow, std::size_t AColumn, std::size_t N,
            std::size_t M, std::size_t K, typename Indices, typename Pointers,
            std::size_t... Rows>
  static void rows(const Matrix<T, ARow, AColumn> &A,
                   const CompiledSparseMatrix<T, N, K, Indices, Pointers> &B,
                   Matrix<T, M, K> &Y, IndexPack<Rows...>) {
    const int sequenced[] = {
        0, (RowKernel<Scatter, Indices, Pointers, N - 1 - Rows>::run(
                DenseProduct::row_values(B.values.data(),
                                         Pointers::list[N - 1 - Rows]),
                A, Y, N - 1 - Rows, typename MakeIndexPack<M>::type{}),
            0)...};
    static_cast<void>(sequenced);
  }

  template <typename T, std::size_t ARow, std::size_t AColumn, std::size_t N,
            std::size_t M, std::size_t K, typename Indices, typename Pointers>
  static void run(const Matrix<T, ARow, AColumn> &A,
                  const CompiledSparseMatrix<T, N, K, Indices, Pointers> &B,
                  Matrix<T, M, K> &Y) {
    static_assert(RowUnroll::UsedIndexBounds<Indices, Pointers, K>::rows(0, N),
                  "CSR column index outside dense output column range");
    rows(A, B, Y, typename MakeIndexPack<N>::type{});
  }
};

struct SparseDenseTranspose {
  template <typename T, std::size_t M, std::size_t N, std::size_t K,
            typename Indices, typename Pointers, std::size_t... Rows>
  static void column(const CompiledSparseMatrix<T, M, N, Indices, Pointers> &A,
                     const Matrix<T, K, N> &B, T *output,
                     std::size_t column_index, IndexPack<Rows...>) {
    const int sequenced[] = {
        0, (RowKernel<MatrixDotValue, Indices, Pointers, M - 1 - Rows>::run(
                DenseProduct::row_values(A.values.data(),
                                         Pointers::list[M - 1 - Rows]),
                B, column_index, output[M - 1 - Rows]),
            0)...};
    static_cast<void>(sequenced);
  }

  template <typename T, std::size_t M, std::size_t N, std::size_t K,
            typename Indices, typename Pointers, std::size_t... Columns>
  static void columns(const CompiledSparseMatrix<T, M, N, Indices, Pointers> &A,
                      const Matrix<T, K, N> &B, Matrix<T, M, K> &Y,
                      IndexPack<Columns...>) {
    const int sequenced[] = {
        0, (column(A, B, Y.data[K - 1 - Columns].data(), K - 1 - Columns,
                   typename MakeIndexPack<M>::type{}),
            0)...};
    static_cast<void>(sequenced);
  }

  template <typename T, std::size_t M, std::size_t N, std::size_t K,
            typename Indices, typename Pointers>
  static void run(const CompiledSparseMatrix<T, M, N, Indices, Pointers> &A,
                  const Matrix<T, K, N> &B, Matrix<T, M, K> &Y) {
    static_assert(RowUnroll::UsedIndexBounds<Indices, Pointers, N>::rows(0, M),
                  "CSR column index outside dense input column range");
    columns(A, B, Y, typename MakeIndexPack<K>::type{});
  }
};

struct DenseSparseTranspose {
  template <typename T, std::size_t M, std::size_t N, std::size_t K,
            typename Indices, typename Pointers, std::size_t... Rows>
  static void rows(const Matrix<T, M, K> &A,
                   const CompiledSparseMatrix<T, N, K, Indices, Pointers> &B,
                   Matrix<T, M, N> &Y, IndexPack<Rows...>) {
    const int sequenced[] = {
        0,
        (RowKernel<MatrixDotReference, Indices, Pointers, N - 1 - Rows>::rows(
             DenseProduct::row_values(B.values.data(),
                                      Pointers::list[N - 1 - Rows]),
             A, Y.data[N - 1 - Rows].data(), typename MakeIndexPack<M>::type{}),
         0)...};
    static_cast<void>(sequenced);
  }

  template <typename T, std::size_t M, std::size_t N, std::size_t K,
            typename Indices, typename Pointers>
  static void run(const Matrix<T, M, K> &A,
                  const CompiledSparseMatrix<T, N, K, Indices, Pointers> &B,
                  Matrix<T, M, N> &Y) {
    rows(A, B, Y, typename MakeIndexPack<N>::type{});
  }
};

struct SparseTransposeDense {
  template <typename T, std::size_t N, std::size_t M, std::size_t K,
            typename Indices, typename Pointers, std::size_t... Rows>
  static void column(const CompiledSparseMatrix<T, N, M, Indices, Pointers> &A,
                     const Matrix<T, N, K> &B, T *output,
                     std::size_t column_index, IndexPack<Rows...>) {
    const int sequenced[] = {
        0, (RowKernel<TransposeScatter, Indices, Pointers, N - 1 - Rows>::run(
                DenseProduct::row_values(A.values.data(),
                                         Pointers::list[N - 1 - Rows]),
                B, output, N - 1 - Rows, column_index),
            0)...};
    static_cast<void>(sequenced);
  }

  template <typename T, std::size_t N, std::size_t M, std::size_t K,
            typename Indices, typename Pointers, std::size_t... Columns>
  static void columns(const CompiledSparseMatrix<T, N, M, Indices, Pointers> &A,
                      const Matrix<T, N, K> &B, Matrix<T, M, K> &Y,
                      IndexPack<Columns...>) {
    const int sequenced[] = {
        0, (column(A, B, Y.data[K - 1 - Columns].data(), K - 1 - Columns,
                   typename MakeIndexPack<N>::type{}),
            0)...};
    static_cast<void>(sequenced);
  }

  template <typename T, std::size_t N, std::size_t M, std::size_t K,
            typename Indices, typename Pointers>
  static void run(const CompiledSparseMatrix<T, N, M, Indices, Pointers> &A,
                  const Matrix<T, N, K> &B, Matrix<T, M, K> &Y) {
    static_assert(RowUnroll::UsedIndexBounds<Indices, Pointers, M>::rows(0, N),
                  "CSR column index outside dense output row range");
    columns(A, B, Y, typename MakeIndexPack<K>::type{});
  }
};

} // namespace DenseProduct

/* Products of two sparse matrices */

/** @brief Compile-time contribution plans for the three sparse products. */
namespace ProductDetail {

enum Kind { Ordinary, TransposeLeft, TransposeRight };

template <typename Matrix> struct Descriptor;
template <typename T, std::size_t M, std::size_t N, typename Indices,
          typename Pointers>
struct Descriptor<CompiledSparseMatrix<T, M, N, Indices, Pointers>> {
  using Index = Indices;
  using Pointer = Pointers;
  static constexpr std::size_t rows = M;
  static constexpr std::size_t columns = N;
};

// Function arguments, rather than template positions, perform the searches.
// Repeated input columns remain separate contributions in their stored order.
template <typename Indices> struct Occurrences {
  static constexpr std::size_t count(std::size_t column, std::size_t begin,
                                     std::size_t end) {
    return begin == end ? 0
           : end - begin == 1
               ? (Indices::list[begin] == column ? 1 : 0)
               : count(column, begin, begin + (end - begin) / 2) +
                     count(column, begin + (end - begin) / 2, end);
  }

  static constexpr std::size_t position(std::size_t column, std::size_t ordinal,
                                        std::size_t begin, std::size_t end) {
    return end - begin == 1
               ? begin
               : select(column, ordinal, begin, end,
                        count(column, begin, begin + (end - begin) / 2));
  }

  static constexpr std::size_t select(std::size_t column, std::size_t ordinal,
                                      std::size_t begin, std::size_t end,
                                      std::size_t left_count) {
    return ordinal < left_count
               ? position(column, ordinal, begin, begin + (end - begin) / 2)
               : position(column, ordinal - left_count,
                          begin + (end - begin) / 2, end);
  }
};

template <typename Positions> struct WriteSlots;
template <std::size_t... Positions> struct WriteSlots<IndexPack<Positions...>> {
  template <typename T> static void run(T *out, const T &value) {
    static_cast<void>(out);
    static_cast<void>(value);
    const int sequenced[] = {0,
                             (static_cast<void>(out[Positions] = value), 0)...};
    static_cast<void>(sequenced);
  }
};

template <std::size_t Position> struct ReadSlot {
  template <typename T> static T run(const T *out) {
    // Match get_sparse_matrix_value: initialization followed by const-source
    // assignment is observably different from copy construction for some T.
    T value = static_cast<T>(0);
    value = out[Position];
    return value;
  }
};
template <> struct ReadSlot<StaticSparseLookup::missing> {
  template <typename T> static T run(const T *) {
    T value = static_cast<T>(0);
    return value;
  }
};

template <std::size_t Read, typename Writes> struct Location {
  static constexpr std::size_t read = Read;
  using Writer = WriteSlots<Writes>;
};

template <typename Indices, std::size_t Column, std::size_t Begin,
          std::size_t End, typename Ordinals>
struct WriteLocations;
template <typename Indices, std::size_t Column, std::size_t Begin,
          std::size_t End, std::size_t... Ordinals>
struct WriteLocations<Indices, Column, Begin, End, IndexPack<Ordinals...>> {
  using type = IndexPack<Occurrences<Indices>::position(Column, Ordinals, Begin,
                                                        End)...>;
};

template <typename Output, std::size_t Row, std::size_t Column>
struct OutputLocation {
  static_assert(Row < Output::rows, "Column number must be less than M");
  static_assert(Column < Output::columns, "Row number must be less than N");
  using Index = typename Output::Index;
  using Pointer = typename Output::Pointer;
  using Writes = typename WriteLocations<
      Index, Column, Pointer::list[Row], Pointer::list[Row + 1],
      typename MakeIndexPack<Occurrences<Index>::count(
          Column, Pointer::list[Row], Pointer::list[Row + 1])>::type>::type;
  using type = Location<StaticSparseLookup::last<Index>(
                            Column, Pointer::list[Row], Pointer::list[Row + 1]),
                        Writes>;
};

// Arithmetic helpers contain only value-slot indices and output locations;
// none of the input or output matrix-wide CSR types appears in their types.
template <std::size_t APosition, std::size_t BPosition, typename Output>
struct Contribution {
  template <typename T> static void run(const T *a, const T *b, T *out) {
    // Keep operator+ (not +=), a by-value sparse read, and const input values.
    // Even a missing output position must evaluate this product and addition.
    using Writer = typename Output::Writer;
    Writer::run(out, ReadSlot<Output::read>::run(static_cast<const T *>(out)) +
                         a[APosition] * b[BPosition]);
  }
};

template <typename... Contributions> struct RowKernel {
  template <typename T> static void run(const T *a, const T *b, T *out) {
    static_cast<void>(a);
    static_cast<void>(b);
    static_cast<void>(out);
    const int sequenced[] = {0, (Contributions::run(a, b, out), 0)...};
    static_cast<void>(sequenced);
  }
};

/**
 * @brief Selects the ordinal-th contributing pair without per-pair type scans.
 *
 * Ordinary and left-transposed products visit all stored entries of the
 * corresponding B row. Right-transposed products select matching columns only,
 * including every duplicate, in the same A-major/B-minor order as the legacy
 * conditional recursion. Empty plans never request a pair position.
 */
template <Kind Operation, typename A, typename B, typename Output,
          std::size_t ARow, std::size_t BRow>
struct Plan {
  using AI = typename A::Index;
  using AP = typename A::Pointer;
  using BI = typename B::Index;
  using BP = typename B::Pointer;
  using OutputDescriptor = Output;

  static constexpr std::size_t b_row(std::size_t a_position) {
    return Operation == Ordinary        ? AI::list[a_position]
           : Operation == TransposeLeft ? ARow
                                        : BRow;
  }
  static constexpr std::size_t count_b(std::size_t a_position) {
    return Operation == TransposeRight
               ? Occurrences<BI>::count(AI::list[a_position],
                                        BP::list[b_row(a_position)],
                                        BP::list[b_row(a_position) + 1])
               : BP::list[b_row(a_position) + 1] - BP::list[b_row(a_position)];
  }
  static constexpr std::size_t count(std::size_t begin, std::size_t end) {
    return begin == end       ? 0
           : end - begin == 1 ? count_b(begin)
                              : count(begin, begin + (end - begin) / 2) +
                                    count(begin + (end - begin) / 2, end);
  }
  static constexpr std::size_t select_a(std::size_t ordinal, std::size_t begin,
                                        std::size_t end) {
    return end - begin == 1
               ? begin
               : select_half(ordinal, begin, end,
                             count(begin, begin + (end - begin) / 2));
  }
  static constexpr std::size_t select_half(std::size_t ordinal,
                                           std::size_t begin, std::size_t end,
                                           std::size_t left_count) {
    return ordinal < left_count
               ? select_a(ordinal, begin, begin + (end - begin) / 2)
               : select_a(ordinal - left_count, begin + (end - begin) / 2, end);
  }
  static constexpr std::size_t a_position(std::size_t ordinal) {
    return select_a(ordinal, AP::list[ARow], AP::list[ARow + 1]);
  }
  static constexpr std::size_t select_b(std::size_t ordinal,
                                        std::size_t a_position) {
    return Operation == TransposeRight
               ? Occurrences<BI>::position(
                     AI::list[a_position],
                     ordinal - count(AP::list[ARow], a_position),
                     BP::list[b_row(a_position)],
                     BP::list[b_row(a_position) + 1])
               : BP::list[b_row(a_position)] + ordinal -
                     count(AP::list[ARow], a_position);
  }
  static constexpr std::size_t b_position(std::size_t ordinal) {
    return select_b(ordinal, a_position(ordinal));
  }
  static constexpr std::size_t output_row(std::size_t ordinal) {
    return Operation == TransposeLeft ? AI::list[a_position(ordinal)] : ARow;
  }
  static constexpr std::size_t output_column(std::size_t ordinal) {
    return Operation == TransposeRight ? BRow : BI::list[b_position(ordinal)];
  }
  static constexpr std::size_t size = count(AP::list[ARow], AP::list[ARow + 1]);
};

template <typename Plan, typename Ordinals> struct ExpandPlan;
template <typename Plan, std::size_t... Ordinals>
struct ExpandPlan<Plan, IndexPack<Ordinals...>> {
  using type = RowKernel<Contribution<
      Plan::a_position(Ordinals), Plan::b_position(Ordinals),
      typename OutputLocation<typename Plan::OutputDescriptor,
                              Plan::output_row(Ordinals),
                              Plan::output_column(Ordinals)>::type>...>;
};

template <Kind Operation, typename A, typename B, typename Output,
          std::size_t ARow, std::size_t BRow>
using KernelFor = typename ExpandPlan<
    Plan<Operation, A, B, Output, ARow, BRow>,
    typename MakeIndexPack<
        Plan<Operation, A, B, Output, ARow, BRow>::size>::type>::type;

template <Kind Operation, typename A, typename B, typename Output,
          std::size_t BRow, typename Rows>
struct RowTraversal;
template <Kind Operation, typename A, typename B, typename Output,
          std::size_t BRow, std::size_t... Rows>
struct RowTraversal<Operation, A, B, Output, BRow, IndexPack<Rows...>> {
  template <typename T> static void run(const T *a, const T *b, T *out) {
    static_cast<void>(a);
    static_cast<void>(b);
    static_cast<void>(out);
    const int sequenced[] = {
        0, (KernelFor<Operation, A, B, Output, A::rows - 1 - Rows, BRow>::run(
                a, b, out),
            0)...};
    static_cast<void>(sequenced);
  }
};

template <Kind Operation> struct Dispatch {
  template <typename A, typename B, typename Output, std::size_t... Columns>
  static void columns(const A &a, const B &b, Output &out,
                      IndexPack<Columns...>) {
    using AD = Descriptor<A>;
    using BD = Descriptor<B>;
    using OD = Descriptor<Output>;
    constexpr std::size_t column_count =
        Operation == TransposeRight ? BD::rows : 1;
    const int sequenced[] = {
        0, (RowTraversal<Operation, AD, BD, OD, column_count - 1 - Columns,
                         typename MakeIndexPack<AD::rows>::type>::
                run(a.values.data(), b.values.data(), out.values.data()),
            0)...};
    static_cast<void>(sequenced);
  }

  template <typename A, typename B, typename Output>
  static void run(const A &a, const B &b, Output &out) {
    columns(a, b, out,
            typename MakeIndexPack < Operation == TransposeRight
                ? Descriptor<B>::rows
                : 1 > ::type{});
  }
};

} // namespace ProductDetail

} // namespace SparseReduced

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_COMPILED_SPARSE_REDUCED_TEMPLATE_HPP_
