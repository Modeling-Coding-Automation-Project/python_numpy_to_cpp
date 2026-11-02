#ifndef BASE_MATRIX_COMPILED_SPARSE_REDUCED_TEMPLATE_HPP_
#define BASE_MATRIX_COMPILED_SPARSE_REDUCED_TEMPLATE_HPP_

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_templates.hpp"
#include "base_matrix_vector.hpp"

/**
 * @brief Fully unrolled kernels whose arithmetic types describe one row.
 *
 * Keeping the complete CSR pattern out of each arithmetic helper avoids
 * repeating large template arguments in thousands of function specializations.
 * The existing recursive helpers remain available, including for bool storage.
 * Sequencing arrays can remain in unoptimized builds; optimizing compilers can
 * discard them while retaining the ordered arithmetic expressions.
 */
namespace Base {
namespace Matrix {

namespace SparseMatrixMultiplyDenseMatrix {

// Defined in base_matrix_compiled_sparse_operation.hpp.
template <typename T, std::size_t M, std::size_t N, std::size_t K,
          typename CSRIndices_A, typename CSRPointers_A, std::size_t I>
struct Row;

} // namespace SparseMatrixMultiplyDenseMatrix

namespace RowUnroll {

using TemplatesOperation::SparseDetail::IndexPack;
using TemplatesOperation::SparseDetail::MakeIndexPack;

template <typename Columns, typename Positions> struct RowKernel;

template <std::size_t... Columns, std::size_t... Positions>
struct RowKernel<IndexPack<Columns...>, IndexPack<Positions...>> {
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
  static constexpr bool span(std::size_t begin, std::size_t end) {
    return begin > end || end > Indices::size ? false
           : begin == end                     ? true
           : end - begin == 1                 ? Indices::list[begin] < Bound
                              : span(begin, begin + (end - begin) / 2) &&
                                    span(begin + (end - begin) / 2, end);
  }

  static constexpr bool rows(std::size_t begin, std::size_t end) {
    return begin == end ? true
           : end - begin == 1
               ? span(Pointers::list[begin], Pointers::list[begin + 1])
               : rows(begin, begin + (end - begin) / 2) &&
                     rows(begin + (end - begin) / 2, end);
  }
};

template <typename T> struct RunDispatch {
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

// vector<bool> uses proxy elements and has no usable data(). Specialization
// prevents the pointer path from being instantiated for either bool backend.
template <> struct RunDispatch<bool> {
  template <std::size_t M, std::size_t N, std::size_t K, typename Indices,
            typename Pointers, std::size_t... Columns>
  static void run(const CompiledSparseMatrix<bool, M, N, Indices, Pointers> &A,
                  const Matrix<bool, N, K> &B, Matrix<bool, M, K> &Y,
                  IndexPack<Columns...>) {
    SparseMatrixMultiplyDenseMatrix::Row<bool, M, N, K, Indices, Pointers,
                                         K - 1>::compute(A, B, Y);
  }
};

} // namespace RowUnroll

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_COMPILED_SPARSE_REDUCED_TEMPLATE_HPP_
