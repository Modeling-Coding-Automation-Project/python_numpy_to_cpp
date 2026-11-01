#ifndef CHECK_STATIC_SPARSE_HPP_
#define CHECK_STATIC_SPARSE_HPP_

#include "MCAP_tester.hpp"
#include "python_numpy.hpp"

#include <cstddef>
#include <type_traits>

namespace StaticSparseTests {

using namespace Base::Matrix;

template <typename A, typename B>
using LegacyProduct =
    typename TemplatesOperation::SparseAvailableMatrixMultiplyColumnLoop<
        A, B, A::number_of_rows - 1>::type;

template <typename A>
using LegacyTranspose =
    TemplatesOperation::SparseAvailableMatrixMultiplyTranspose<
        DiagAvailable<A::row_size>, A>;

using PatternA = SparseAvailable<ColumnAvailable<true, false, true>,
                                 ColumnAvailable<false, false, false>>;
using PatternB = SparseAvailable<ColumnAvailable<true, false>,
                                 ColumnAvailable<false, false>,
                                 ColumnAvailable<false, true>>;
using ProductExpected = SparseAvailable<ColumnAvailable<true, true>,
                                        ColumnAvailable<false, false>>;
using TransposeExpected = SparseAvailable<ColumnAvailable<true, false>,
                                          ColumnAvailable<false, false>,
                                          ColumnAvailable<true, false>>;

static_assert(std::is_same<SparseAvailableMatrixMultiply<PatternA, PatternB>,
                           ProductExpected>::value,
              "Rectangular product must retain the expected sparse pattern");
static_assert(std::is_same<SparseAvailableTranspose<PatternA>,
                           TransposeExpected>::value,
              "Rectangular transpose must retain empty rows and columns");
static_assert(std::is_same<SparseAvailableTranspose<
                              SparseAvailableTranspose<PatternA>>,
                           PatternA>::value,
              "Double transpose must recover the exact original type");

template <typename A, typename B> struct CheckProductType {
  static_assert(std::is_same<SparseAvailableMatrixMultiply<A, B>,
                             LegacyProduct<A, B>>::value,
                "Product metadata must retain its exact legacy type");
};

template <typename A> struct CheckTransposeType {
  static_assert(std::is_same<SparseAvailableTranspose<A>,
                             LegacyTranspose<A>>::value,
                "Transpose metadata must retain its exact legacy type");
};

// Instantiate each comparison, including positive-size all-zero patterns.
static_assert(sizeof(CheckProductType<PatternA, PatternB>) > 0, "rectangular");
static_assert(sizeof(CheckProductType<DenseAvailable<3, 4>,
                                      DenseAvailable<4, 2>>) > 0,
              "dense");
static_assert(sizeof(CheckProductType<DiagAvailable<4>,
                                      DenseAvailable<4, 3>>) > 0,
              "diagonal");
static_assert(sizeof(CheckProductType<DenseAvailableEmpty<2, 3>,
                                      DenseAvailable<3, 4>>) > 0,
              "empty left operand");
static_assert(sizeof(CheckProductType<DenseAvailable<2, 3>,
                                      DenseAvailableEmpty<3, 4>>) > 0,
              "empty right operand");
static_assert(sizeof(CheckTransposeType<PatternA>) > 0, "rectangular");
static_assert(sizeof(CheckTransposeType<DenseAvailable<3, 4>>) > 0, "dense");
static_assert(sizeof(CheckTransposeType<DiagAvailable<4>>) > 0, "diagonal");
static_assert(sizeof(CheckTransposeType<DenseAvailableEmpty<2, 3>>) > 0,
              "empty");
static_assert(sizeof(CheckProductType<DiagAvailable<1>, DiagAvailable<1>>) > 0,
              "one element");
static_assert(sizeof(CheckTransposeType<DiagAvailable<1>>) > 0, "one element");

// Lightweight metadata descriptors avoid a long parameter-pack construction in
// the test itself. Only the operations under test traverse the 1024 entries.
template <std::size_t Size> struct SequentialIndices {
  struct List {
    constexpr std::size_t operator[](std::size_t index) const { return index; }
  };
  static constexpr std::size_t size = Size;
  static constexpr List list{};
};

template <std::size_t Size>
constexpr typename SequentialIndices<Size>::List SequentialIndices<Size>::list;

template <std::size_t Rows, std::size_t Columns, std::size_t ActiveRow,
          std::size_t ActiveColumn>
struct SingleActivePattern {
  struct Row {
    std::size_t index;
    constexpr bool operator[](std::size_t column) const {
      return index == ActiveRow && column == ActiveColumn;
    }
  };
  struct Lists {
    constexpr Row operator[](std::size_t row) const { return Row{row}; }
  };
  static constexpr std::size_t number_of_rows = Rows;
  static constexpr std::size_t row_size = Columns;
  static constexpr Lists lists{};
};

template <std::size_t Rows, std::size_t Columns, std::size_t ActiveRow,
          std::size_t ActiveColumn>
constexpr typename SingleActivePattern<Rows, Columns, ActiveRow,
                                      ActiveColumn>::Lists
    SingleActivePattern<Rows, Columns, ActiveRow, ActiveColumn>::lists;

using LongLeft = SingleActivePattern<1, 1024, 0, 1023>;
using LongRight = SingleActivePattern<1024, 1, 1023, 0>;
using LongDisjointRight = SingleActivePattern<1024, 1, 1022, 0>;
using LongTranspose = SparseAvailableTranspose<LongLeft>;
static_assert(LongTranspose::number_of_rows == 1024 &&
                  LongTranspose::row_size == 1 &&
                  !LongTranspose::lists[0][0] &&
                  LongTranspose::lists[1023][0],
              "Wide transpose must fit default template-depth limits");
static_assert(std::is_same<SparseAvailableMatrixMultiply<LongLeft, LongRight>,
                           DiagAvailable<1>>::value,
              "A late inner-product match must fit default constexpr limits");
static_assert(std::is_same<SparseAvailableMatrixMultiply<LongLeft,
                                                        LongDisjointRight>,
                           DenseAvailableEmpty<1, 1>>::value,
              "A long disjoint product must remain structurally empty");

template <typename T> void check_access() {
  Tester::MCAPTester<T> tester;
  const T exact = static_cast<T>(0);

  using Small = CompiledSparseMatrix<T, 3, 4, CSRIndices<0, 3, 1, 3>,
                                     CSRPointers<0, 2, 2, 4>>;
  Small matrix{T(1), T(2), T(3), T(4)};
  set_sparse_matrix_value<0, 0>(matrix, T(0));
  set_sparse_matrix_value<2, 3>(matrix, T(9));
  set_sparse_matrix_value<0, 2>(matrix, T(99));
  set_sparse_matrix_value<1, 1>(matrix, T(99));
  tester.expect_near(get_sparse_matrix_value<0, 0>(matrix), T(0), exact,
                     "Static sparse access must allow writing zero");
  tester.expect_near(get_sparse_matrix_value<2, 3>(matrix), T(9), exact,
                     "Static sparse access must find the last stored entry");
  tester.expect_near(get_sparse_matrix_value<0, 2>(matrix), T(0), exact,
                     "Missing sparse entries must read zero");
  tester.expect_near(get_sparse_matrix_value<1, 1>(matrix), T(0), exact,
                     "Empty sparse rows must read zero");
  tester.expect_near(matrix.values[1], T(2), exact,
                     "Missing writes must leave other entries unchanged");
  tester.expect_near(matrix.values[2], T(3), exact,
                     "Empty-row writes must leave other rows unchanged");

  using Empty = CompiledSparseMatrix<T, 2, 2, CSRIndices<0>,
                                     CSRPointers<0, 0, 0>>;
  Empty empty;
  set_sparse_matrix_value<1, 1>(empty, T(7));
  tester.expect_near(get_sparse_matrix_value<1, 1>(empty), T(0), exact,
                     "A matrix without stored entries must read zero");
  tester.expect_near(static_cast<T>(empty.values.size()), T(0), exact,
                     "Writing a structural zero must not allocate storage");

  using Duplicates = CompiledSparseMatrix<T, 1, 3, CSRIndices<2, 0, 2, 1>,
                                          CSRPointers<0, 4>>;
  Duplicates duplicates{T(1), T(2), T(3), T(4)};
  tester.expect_near(get_sparse_matrix_value<0, 2>(duplicates), T(3), exact,
                     "Duplicate CSR indices must read the last matching value");
  set_sparse_matrix_value<0, 2>(duplicates, T(8));
  tester.expect_near(duplicates.values[0], T(8), exact,
                     "Duplicate CSR writes must update the first match");
  tester.expect_near(duplicates.values[2], T(8), exact,
                     "Duplicate CSR writes must update the last match");
  tester.expect_near(duplicates.values[1], T(2), exact,
                     "Unsorted CSR writes must preserve nonmatching entries");
  tester.expect_near(duplicates.values[3], T(4), exact,
                     "Unsorted CSR writes must preserve following entries");

  using Wide = CompiledSparseMatrix<T, 1, 1025, SequentialIndices<1024>,
                                    CSRPointers<0, 1024>>;
  Wide wide;
  set_sparse_matrix_value<0, 0>(wide, T(5));
  set_sparse_matrix_value<0, 1023>(wide, T(7));
  set_sparse_matrix_value<0, 1024>(wide, T(99));
  tester.expect_near(get_sparse_matrix_value<0, 0>(wide), T(5), exact,
                     "Long sparse rows must find their first entry");
  tester.expect_near(get_sparse_matrix_value<0, 1023>(wide), T(7), exact,
                     "Long sparse rows must find their last entry");
  tester.expect_near(get_sparse_matrix_value<0, 1024>(wide), T(0), exact,
                     "Long sparse rows must handle missing entries");

  // Exercise the Python-facing wrapper used by generated model headers.
  PythonNumpy::Matrix<PythonNumpy::DefSparse, T, 2, 3, PatternA> wrapper;
  wrapper.template set<0, 2>(T(6));
  wrapper.template set<1, 1>(T(99));
  tester.expect_near(wrapper.template get<0, 2>(), T(6), exact,
                     "PythonNumpy static access must retain stored values");
  tester.expect_near(wrapper.template get<1, 1>(), T(0), exact,
                     "PythonNumpy static access must retain structural zeros");

  tester.throw_error_if_test_failed();
}

} // namespace StaticSparseTests

#endif // CHECK_STATIC_SPARSE_HPP_
