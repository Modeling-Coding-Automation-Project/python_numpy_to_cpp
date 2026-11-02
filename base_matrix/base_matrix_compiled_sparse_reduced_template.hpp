/********************************************************************************
@file base_matrix_compiled_sparse_reduced_template.hpp
@brief
This file provides reduced-template kernels for multiplying a compiled sparse
matrix by a dense matrix. It uses compile-time index packs to unroll the
arithmetic for each sparse row while keeping the complete CSR pattern out of
the row kernels.

@details
The file defines:
- Row kernels that compute a sparse row and one dense column.
- Column kernels that apply the row kernels to all matrix rows.
- Compile-time checks for column indices used by the sparse matrix.
- Dispatch for dense matrix multiplication using the unrolled kernels.

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

namespace Base {
namespace Matrix {

namespace RowUnroll {

using TemplatesOperation::SparseDetail::IndexPack;
using TemplatesOperation::SparseDetail::MakeIndexPack;

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

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_COMPILED_SPARSE_REDUCED_TEMPLATE_HPP_
