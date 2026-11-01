#ifndef CHECK_STATIC_SPARSE_HPP_
#define CHECK_STATIC_SPARSE_HPP_

#include "MCAP_tester.hpp"
#include "python_numpy.hpp"

#include <cstddef>
#include <type_traits>

namespace StaticSparseLinkTests {

using Pattern = Base::Matrix::SparseAvailable<
    Base::Matrix::ColumnAvailable<true, false, true>,
    Base::Matrix::ColumnAvailable<false, true, false>,
    Base::Matrix::ColumnAvailable<false, false, false>>;
using Indices = Base::Matrix::CSRIndicesFromSparseAvailable<Pattern>;
using Pointers = Base::Matrix::CSRPointersFromSparseAvailable<Pattern>;

static_assert(std::is_same<Indices, Base::Matrix::CSRIndices<0, 2, 1>>::value,
              "Both translation units must instantiate the same CSR type");

inline std::size_t row_end(std::size_t row) { return Pointers::list[row + 1]; }

inline double multiply() {
  Base::Matrix::CompiledSparseMatrix<double, 3, 3, Indices, Pointers> a{
      2.0, 3.0, 4.0};
  Base::Matrix::Matrix<double, 3, 2> b{{5.0, 1.0}, {6.0, 2.0}, {7.0, 3.0}};
  const auto result = a * b;
  return result.template get<0, 0>() + result.template get<1, 1>() +
         result.template get<2, 0>();
}

double from_other_translation_unit();

} // namespace StaticSparseLinkTests

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
using PatternB =
    SparseAvailable<ColumnAvailable<true, false>, ColumnAvailable<false, false>,
                    ColumnAvailable<false, true>>;
using ProductExpected =
    SparseAvailable<ColumnAvailable<true, true>, ColumnAvailable<false, false>>;
using TransposeExpected =
    SparseAvailable<ColumnAvailable<true, false>, ColumnAvailable<false, false>,
                    ColumnAvailable<true, false>>;

static_assert(std::is_same<SparseAvailableMatrixMultiply<PatternA, PatternB>,
                           ProductExpected>::value,
              "Rectangular product must retain the expected sparse pattern");
static_assert(
    std::is_same<SparseAvailableTranspose<PatternA>, TransposeExpected>::value,
    "Rectangular transpose must retain empty rows and columns");
static_assert(
    std::is_same<SparseAvailableTranspose<SparseAvailableTranspose<PatternA>>,
                 PatternA>::value,
    "Double transpose must recover the exact original type");

template <typename A, typename B> struct CheckProductType {
  static_assert(std::is_same<SparseAvailableMatrixMultiply<A, B>,
                             LegacyProduct<A, B>>::value,
                "Product metadata must retain its exact legacy type");
};

template <typename A> struct CheckTransposeType {
  static_assert(
      std::is_same<SparseAvailableTranspose<A>, LegacyTranspose<A>>::value,
      "Transpose metadata must retain its exact legacy type");
};

// Instantiate each comparison, including positive-size all-zero patterns.
static_assert(sizeof(CheckProductType<PatternA, PatternB>) > 0, "rectangular");
static_assert(
    sizeof(CheckProductType<DenseAvailable<3, 4>, DenseAvailable<4, 2>>) > 0,
    "dense");
static_assert(sizeof(CheckProductType<DiagAvailable<4>, DenseAvailable<4, 3>>) >
                  0,
              "diagonal");
static_assert(
    sizeof(CheckProductType<DenseAvailableEmpty<2, 3>, DenseAvailable<3, 4>>) >
        0,
    "empty left operand");
static_assert(
    sizeof(CheckProductType<DenseAvailable<2, 3>, DenseAvailableEmpty<3, 4>>) >
        0,
    "empty right operand");
static_assert(sizeof(CheckTransposeType<PatternA>) > 0, "rectangular");
static_assert(sizeof(CheckTransposeType<DenseAvailable<3, 4>>) > 0, "dense");
static_assert(sizeof(CheckTransposeType<DiagAvailable<4>>) > 0, "diagonal");
static_assert(sizeof(CheckTransposeType<DenseAvailableEmpty<2, 3>>) > 0,
              "empty");
static_assert(sizeof(CheckProductType<DiagAvailable<1>, DiagAvailable<1>>) > 0,
              "one element");
static_assert(sizeof(CheckTransposeType<DiagAvailable<1>>) > 0, "one element");

template <typename Pattern, typename Indices, typename Pointers,
          typename LastRow>
struct CheckCSRType {
  static_assert(
      std::is_same<CSRIndicesFromSparseAvailable<Pattern>, Indices>::value,
      "CSR column indices must retain their canonical public type");
  static_assert(
      std::is_same<CSRPointersFromSparseAvailable<Pattern>, Pointers>::value,
      "CSR row pointers must retain their canonical public type");
  static_assert(std::is_same<ValidateSparseAvailable<Pattern>,
                             SparseAvailable<LastRow>>::value,
                "Validation must retain its existing last-row result type");
  static_assert(std::is_same<CreateSparseAvailableFromIndicesAndPointers<
                                 Pattern::row_size, Indices, Pointers>,
                             Pattern>::value,
                "Canonical CSR metadata must reconstruct the original type");
};

using EmptyRow4 = ColumnAvailable<false, false, false, false>;
using EmptyEdges =
    SparseAvailable<EmptyRow4, ColumnAvailable<true, false, false, true>,
                    EmptyRow4, ColumnAvailable<false, true, true, false>,
                    EmptyRow4>;

static_assert(
    sizeof(CheckCSRType<PatternA, CSRIndices<0, 2>, CSRPointers<0, 2, 2>,
                        ColumnAvailable<false, false, false>>) > 0,
    "rectangular CSR with trailing empty row");
static_assert(sizeof(CheckCSRType<EmptyEdges, CSRIndices<0, 3, 1, 2>,
                                  CSRPointers<0, 0, 2, 2, 4, 4>, EmptyRow4>) >
                  0,
              "CSR with leading, internal and trailing empty rows");
static_assert(
    sizeof(
        CheckCSRType<DenseAvailable<2, 3>, CSRIndices<0, 1, 2, 0, 1, 2>,
                     CSRPointers<0, 3, 6>, ColumnAvailable<true, true, true>>) >
        0,
    "dense rectangular CSR");
static_assert(sizeof(CheckCSRType<DenseAvailableEmpty<3, 2>, CSRIndices<0>,
                                  CSRPointers<0, 0, 0, 0>,
                                  ColumnAvailable<false, false>>) > 0,
              "empty CSR must preserve the single-zero index sentinel");
static_assert(sizeof(CheckCSRType<DiagAvailable<3>, CSRIndices<0, 1, 2>,
                                  CSRPointers<0, 1, 2, 3>,
                                  ColumnAvailable<false, false, true>>) > 0,
              "diagonal CSR");
static_assert(sizeof(CheckCSRType<DiagAvailable<1>, CSRIndices<0>,
                                  CSRPointers<0, 1>, ColumnAvailable<true>>) >
                  0,
              "singleton CSR and validation");

// Build a wide test input with logarithmic recursion, independently of the
// library's sequence helpers. The operation under test must handle 1024 columns
// with the compiler's default template/constexpr depth limits.
template <std::size_t... I> struct TestIndices {};
template <typename A, typename B> struct JoinTestIndices;
template <std::size_t... A, std::size_t... B>
struct JoinTestIndices<TestIndices<A...>, TestIndices<B...>> {
  using type = TestIndices<A..., (sizeof...(A) + B)...>;
};
template <std::size_t N> struct MakeTestIndices {
  using type =
      typename JoinTestIndices<typename MakeTestIndices<N / 2>::type,
                               typename MakeTestIndices<N - N / 2>::type>::type;
};
template <> struct MakeTestIndices<0> {
  using type = TestIndices<>;
};
template <> struct MakeTestIndices<1> {
  using type = TestIndices<0>;
};
template <typename Sequence> struct WideCSRPattern;
template <std::size_t... I> struct WideCSRPattern<TestIndices<I...>> {
  using type = SparseAvailable<ColumnAvailable<(I == 0 || I == 1023)...>>;
};
using WideCSR =
    typename WideCSRPattern<typename MakeTestIndices<1024>::type>::type;
static_assert(std::is_same<CSRIndicesFromSparseAvailable<WideCSR>,
                           CSRIndices<0, 1023>>::value,
              "Wide CSR generation must preserve the last active column");
static_assert(std::is_same<CSRPointersFromSparseAvailable<WideCSR>,
                           CSRPointers<0, 2>>::value,
              "Wide CSR row counts must fit default depth limits");
static_assert(std::is_same<ValidateSparseAvailable<WideCSR>, WideCSR>::value,
              "Wide singleton validation must preserve its exact type");

// Existing descriptor adapters are valid inputs even when the descriptor is
// not itself a canonical SparseAvailable instantiation.
using CustomCanonical = SparseAvailable<ColumnAvailable<true, false, true>,
                                        ColumnAvailable<false, false, false>,
                                        ColumnAvailable<false, true, false>>;
struct CustomPattern : CustomCanonical {};
struct CustomSingleRow : SparseAvailable<ColumnAvailable<true, false, true>> {};
struct CustomEmpty : DenseAvailableEmpty<2, 2> {};

} // namespace StaticSparseTests

namespace Base {
namespace Matrix {
namespace TemplatesOperation {

template <>
struct CheckSparseAvailableEmpty<StaticSparseTests::CustomPattern>
    : CheckSparseAvailableEmpty<StaticSparseTests::CustomCanonical> {};
template <std::size_t I>
struct GetColumnAvailable<I, StaticSparseTests::CustomPattern>
    : GetColumnAvailable<I, StaticSparseTests::CustomCanonical> {};
template <> struct CheckSparseAvailableEmpty<StaticSparseTests::CustomEmpty> {
  static constexpr bool value = false;
};

} // namespace TemplatesOperation
} // namespace Matrix
} // namespace Base

namespace StaticSparseTests {

static_assert(std::is_same<CSRIndicesFromSparseAvailable<CustomPattern>,
                           CSRIndices<0, 2, 1>>::value,
              "Adapted descriptors must retain CSR index generation");
static_assert(std::is_same<CSRPointersFromSparseAvailable<CustomPattern>,
                           CSRPointers<0, 2, 2, 3>>::value,
              "Adapted descriptors must retain CSR pointer generation");
static_assert(
    std::is_same<ValidateSparseAvailable<CustomPattern>,
                 SparseAvailable<ColumnAvailable<false, true, false>>>::value,
    "Adapted descriptors must retain the validation result");
static_assert(
    std::is_same<ValidateSparseAvailable<CustomSingleRow>,
                 CustomSingleRow>::value,
    "Singleton validation must preserve a custom descriptor's identity");
static_assert(std::is_same<CSRIndicesFromSparseAvailable<CustomEmpty>,
                           CSRIndices<0>>::value,
              "Custom empty descriptors must retain the CSR index sentinel");
static_assert(std::is_same<CSRPointersFromSparseAvailable<CustomEmpty>,
                           CSRPointers<0, 0, 0>>::value,
              "Custom empty descriptors must retain empty row pointers");

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
constexpr
    typename SingleActivePattern<Rows, Columns, ActiveRow, ActiveColumn>::Lists
        SingleActivePattern<Rows, Columns, ActiveRow, ActiveColumn>::lists;

using LongLeft = SingleActivePattern<1, 1024, 0, 1023>;
using LongRight = SingleActivePattern<1024, 1, 1023, 0>;
using LongDisjointRight = SingleActivePattern<1024, 1, 1022, 0>;
using LongTranspose = SparseAvailableTranspose<LongLeft>;
static_assert(LongTranspose::number_of_rows == 1024 &&
                  LongTranspose::row_size == 1 && !LongTranspose::lists[0][0] &&
                  LongTranspose::lists[1023][0],
              "Wide transpose must fit default template-depth limits");
static_assert(std::is_same<SparseAvailableMatrixMultiply<LongLeft, LongRight>,
                           DiagAvailable<1>>::value,
              "A late inner-product match must fit default constexpr limits");
static_assert(
    std::is_same<SparseAvailableMatrixMultiply<LongLeft, LongDisjointRight>,
                 DenseAvailableEmpty<1, 1>>::value,
    "A long disjoint product must remain structurally empty");

static_assert(
    std::is_same<
        CSRPointersFromSparseAvailable<SingleActivePattern<3, 4, 2, 3>>,
        CSRPointers<0, 0, 0, 1>>::value,
    "CSR pointers must still accept generic metadata descriptors");

template <typename T> struct TestValue {
  static T make(int real, int) { return static_cast<T>(real); }
};
template <typename T> struct TestValue<Complex<T>> {
  static Complex<T> make(int real, int imag) {
    return Complex<T>(static_cast<T>(real), static_cast<T>(imag));
  }
};

template <typename T> bool exactly_equal(const T &a, const T &b) {
  return a == b;
}
template <typename T>
bool exactly_equal(const Complex<T> &a, const Complex<T> &b) {
  return a.real == b.real && a.imag == b.imag;
}
template <typename T>
void expect_equal(Tester::MCAPTester<double> &tester, const T &actual,
                  const T &expected, const char *message) {
  tester.expect_near(exactly_equal(actual, expected) ? 1.0 : 0.0, 1.0, 0.0,
                     message);
}

template <typename T> void check_multiply() {
  Tester::MCAPTester<double> tester;
  // Row 1 is deliberately unsorted and contains column 2 twice. Neither
  // duplicate entries nor empty rows may be discarded by the unrolled kernel.
  using Sparse = CompiledSparseMatrix<T, 5, 5, CSRIndices<2, 0, 2, 4, 1, 3, 0>,
                                      CSRPointers<0, 0, 4, 4, 7, 7>>;
  Sparse a{T(2), T(-1), T(3), T(4), T(5), T(-2), T(1)};
  Matrix<T, 5, 3> b;
  Matrix<T, 5, 3> overwritten;
  for (std::size_t c = 0; c < 3; ++c) {
    for (std::size_t r = 0; r < 5; ++r) {
      b(r, c) = TestValue<T>::make(static_cast<int>((r + 1) * (c + 1)),
                                   static_cast<int>(r) - static_cast<int>(c));
      overwritten(r, c) = T(99);
    }
  }
  const auto result = a * b;
  SparseMatrixMultiplyDenseMatrix::compute(a, b, overwritten);
  for (std::size_t c = 0; c < 3; ++c) {
    T row1 = T(0);
    row1 += T(2) * b(2, c);
    row1 += T(-1) * b(0, c);
    row1 += T(3) * b(2, c);
    row1 += T(4) * b(4, c);
    T row3 = T(0);
    row3 += T(5) * b(1, c);
    row3 += T(-2) * b(3, c);
    row3 += b(0, c);
    for (std::size_t r = 0; r < 5; ++r) {
      const T expected = r == 1 ? row1 : r == 3 ? row3 : T(0);
      expect_equal(
          tester, result(r, c), expected,
          "Sparse product must preserve unsorted and duplicate entries");
      expect_equal(
          tester, overwritten(r, c), expected,
          "Sparse compute must overwrite every output, including empty rows");
    }
  }

  CompiledSparseMatrix<T, 3, 5, CSRIndices<0>, CSRPointers<0, 0, 0, 0>> empty;
  const auto zeros = empty * b;
  for (std::size_t c = 0; c < 3; ++c) {
    for (std::size_t r = 0; r < 3; ++r) {
      expect_equal(tester, zeros(r, c), T(0),
                   "An entirely empty sparse product must be zero");
    }
  }

  // Reassociating these terms can lose the final 3 for both float and double.
  CompiledSparseMatrix<T, 1, 3, CSRIndices<0, 1, 2>, CSRPointers<0, 3>> cancel{
      T(1.0e30), T(-1.0e30), T(3)};
  Matrix<T, 3, 1> ones{{T(1)}, {T(1)}, {T(1)}};
  const auto cancellation_result = cancel * ones;
  expect_equal(tester, cancellation_result.template get<0, 0>(), T(3),
               "Sparse product must accumulate in the original CSR order");
  tester.throw_error_if_test_failed();
}

// Deliberately observable overloads expose changes to comma sequencing and to
// the value categories formerly supplied by Matrix::get()/set().
struct ContractScalar {
  int value;
  ContractScalar(int v = 0) : value(v) {}
  ContractScalar(const ContractScalar &) = default;
  ContractScalar &operator=(ContractScalar &) {
    value = -777;
    return *this;
  }
  ContractScalar &operator=(const ContractScalar &other) {
    value = other.value;
    return *this;
  }
  ContractScalar operator*(ContractScalar &&other) const {
    return ContractScalar(value * other.value);
  }
  ContractScalar operator*(const ContractScalar &) const {
    return ContractScalar(-999);
  }
  ContractScalar &operator+=(const ContractScalar &other) {
    value += other.value;
    return *this;
  }
  int operator,(int) {
    ++comma_calls();
    return 0;
  }
  static int &comma_calls() {
    static int count = 0;
    return count;
  }
};

inline void check_multiply_contracts() {
  Tester::MCAPTester<double> tester;
  using Number = ContractScalar;
  Number::comma_calls() = 0;
  CompiledSparseMatrix<Number, 2, 3, CSRIndices<0, 2, 1>, CSRPointers<0, 2, 3>>
      a;
  a.values[0] = Number(2);
  a.values[1] = Number(3);
  a.values[2] = Number(4);
  Matrix<Number, 3, 1> b;
  b.template set<0, 0>(Number(5));
  b.template set<1, 0>(Number(6));
  b.template set<2, 0>(Number(7));
  Matrix<Number, 2, 1> y;
  // Test the unrolled compute entry point directly, including when the rest of
  // the existing test suite is built with the optional for-loop macro.
  SparseMatrixMultiplyDenseMatrix::compute(a, b, y);
  tester.expect_near(y.template get<0, 0>().value, 31.0, 0.0,
                     "Sparse compute must retain get/set value categories");
  tester.expect_near(y.template get<1, 0>().value, 24.0, 0.0,
                     "Sparse compute must retain const output assignment");
  tester.expect_near(Number::comma_calls(), 0.0, 0.0,
                     "Pack sequencing must not invoke an overloaded comma");

  CompiledSparseMatrix<bool, 2, 2, CSRIndices<0, 1>, CSRPointers<0, 1, 2>>
      flags;
  flags.values[0] = true;
  flags.values[1] = true;
  Matrix<bool, 2, 2> bits;
  bits.template set<0, 0>(true);
  bits.template set<1, 0>(false);
  bits.template set<0, 1>(false);
  bits.template set<1, 1>(true);
  Matrix<bool, 2, 2> product;
  SparseMatrixMultiplyDenseMatrix::compute(flags, bits, product);
  tester.expect_near(
      product.template get<0, 0>() ? 1.0 : 0.0, 1.0, 0.0,
      "Bool multiplication must work with array and vector storage");
  tester.expect_near(product.template get<1, 0>() ? 1.0 : 0.0, 0.0, 0.0,
                     "Bool multiplication must preserve false values");
  tester.expect_near(
      product.template get<0, 1>() ? 1.0 : 0.0, 0.0, 0.0,
      "Bool multiplication must process multiple output columns");
  tester.expect_near(product.template get<1, 1>() ? 1.0 : 0.0, 1.0, 0.0,
                     "Bool multiplication must preserve the last column");
  tester.throw_error_if_test_failed();
}

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

  using Empty =
      CompiledSparseMatrix<T, 2, 2, CSRIndices<0>, CSRPointers<0, 0, 0>>;
  Empty empty;
  set_sparse_matrix_value<1, 1>(empty, T(7));
  tester.expect_near(get_sparse_matrix_value<1, 1>(empty), T(0), exact,
                     "A matrix without stored entries must read zero");
  tester.expect_near(static_cast<T>(empty.values.size()), T(0), exact,
                     "Writing a structural zero must not allocate storage");

  using Duplicates =
      CompiledSparseMatrix<T, 1, 3, CSRIndices<2, 0, 2, 1>, CSRPointers<0, 4>>;
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
