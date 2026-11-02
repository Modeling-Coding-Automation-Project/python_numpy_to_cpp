#ifndef CHECK_STATIC_SPARSE_HPP_
#define CHECK_STATIC_SPARSE_HPP_

#include "MCAP_tester.hpp"
#include "python_numpy.hpp"

#include <cstddef>
#include <type_traits>
#include <vector>
#include <string>

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

double StaticSparseLinkTests::from_other_translation_unit() {
  return row_end(1) == 3 ? multiply() : -1.0;
}

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
  const T large = TestValue<T>::make(1000000000, 0);
  CompiledSparseMatrix<T, 1, 3, CSRIndices<0, 1, 2>, CSRPointers<0, 3>> cancel{
      large, -large, T(3)};
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

// Keep the recursive helper calls as independent reference implementations.
// These tests compare stored values AND the arithmetic event sequence: a
// faster template expansion must not introduce extra arithmetic or reassociate
// expressions, even when duplicate/missing sparse output positions are used.
namespace ExtendedKernelTests {

struct Event {
  char operation;
  double left;
  double right;
  bool operator==(const Event &other) const {
    return operation == other.operation && left == other.left &&
           right == other.right;
  }
};

inline std::vector<Event> &events() {
  static std::vector<Event> value;
  return value;
}

struct Number {
  double value;
  Number(double v = 0) : value(v) {}
  // Matrix::get() copies from const storage. A raw-pointer replacement must
  // retain that category even when it reads an output accumulator.
  Number(Number &other) : value(other.value + 1000) {}
  Number(const Number &other) : value(other.value) {}
  Number &operator=(const Number &) = default;
  static Number binary(char operation, double a, double b, double result) {
    events().push_back(Event{operation, a, b});
    return Number(result);
  }
  Number operator+(const Number &other) const {
    return binary('+', value, other.value, value + other.value);
  }
  Number operator-(const Number &other) const {
    return binary('-', value, other.value, value - other.value);
  }
  Number operator*(const Number &other) const {
    return binary('*', value, other.value, value * other.value);
  }
  Number operator*(Number &&other) const {
    return binary('R', value, other.value, value * other.value);
  }
  Number operator-() const { return binary('N', value, 0, -value); }
  Number &operator+=(const Number &other) {
    events().push_back(Event{'A', value, other.value});
    value += other.value;
    return *this;
  }
  Number &operator-=(const Number &other) {
    events().push_back(Event{'S', value, other.value});
    value -= other.value;
    return *this;
  }
  int operator,(int) {
    events().push_back(Event{'C', value, 0});
    return 0;
  }
  bool operator==(const Number &other) const { return value == other.value; }
};

template <typename T, std::size_t M, std::size_t N>
void fill(Matrix<T, M, N> &matrix) {
  for (std::size_t c = 0; c < N; ++c)
    for (std::size_t r = 0; r < M; ++r)
      matrix(r, c) =
          TestValue<T>::make(static_cast<int>(1 + r + 3 * c),
                             static_cast<int>(r) - static_cast<int>(c));
}
template <typename T, std::size_t M, std::size_t N, typename CI, typename CP>
void fill(CompiledSparseMatrix<T, M, N, CI, CP> &matrix) {
  for (std::size_t i = 0; i < matrix.values.size(); ++i)
    matrix.values[i] = TestValue<T>::make(static_cast<int>(i + 2),
                                          static_cast<int>(i % 3) - 1);
}
template <typename T, std::size_t N> void fill(Vector<T, N> &vector) {
  for (std::size_t i = 0; i < N; ++i)
    vector[i] = T(i + 2);
}
template <typename T, std::size_t N> void fill(ColVector<T, N> &vector) {
  for (std::size_t i = 0; i < N; ++i)
    vector[i] = T(i + 2);
}
template <typename T, std::size_t N> void fill(DiagMatrix<T, N> &matrix) {
  for (std::size_t i = 0; i < N; ++i)
    matrix[i] = T(i + 2);
}

template <typename T, std::size_t M, std::size_t N>
bool same(const Matrix<T, M, N> &a, const Matrix<T, M, N> &b) {
  for (std::size_t c = 0; c < N; ++c)
    for (std::size_t r = 0; r < M; ++r)
      if (!exactly_equal(a(r, c), b(r, c)))
        return false;
  return true;
}
template <typename T, std::size_t M, std::size_t N, typename CI, typename CP>
bool same(const CompiledSparseMatrix<T, M, N, CI, CP> &a,
          const CompiledSparseMatrix<T, M, N, CI, CP> &b) {
  for (std::size_t i = 0; i < a.values.size(); ++i)
    if (!exactly_equal(T(a.values[i]), T(b.values[i])))
      return false;
  return true;
}
template <typename T, std::size_t N>
bool same(const Vector<T, N> &a, const Vector<T, N> &b) {
  for (std::size_t i = 0; i < N; ++i)
    if (!exactly_equal(a[i], b[i]))
      return false;
  return true;
}
template <typename T, std::size_t N>
bool same(const ColVector<T, N> &a, const ColVector<T, N> &b) {
  for (std::size_t i = 0; i < N; ++i)
    if (!exactly_equal(a[i], b[i]))
      return false;
  return true;
}

template <typename Output, typename Current, typename Legacy>
void compare(Tester::MCAPTester<double> &tester, const Output &seed,
             Current current, Legacy legacy, const char *name) {
  Output actual(seed), expected(seed);
  events().clear();
  current(actual);
  const auto actual_events = events();
  events().clear();
  legacy(expected);
  tester.expect_near(same(actual, expected) ? 1.0 : 0.0, 1.0, 0.0,
                     std::string(name) + ": stored results");
  tester.expect_near(actual_events == events() ? 1.0 : 0.0, 1.0, 0.0,
                     std::string(name) +
                         ": arithmetic counts, order and operand categories");
  events().clear();
}

template <typename T, std::size_t M, std::size_t N>
using FullSparse =
    CompiledSparseMatrix<T, M, N,
                         CSRIndicesFromSparseAvailable<DenseAvailable<M, N>>,
                         CSRPointersFromSparseAvailable<DenseAvailable<M, N>>>;

template <typename T> void check_elementwise_and_mixed() {
  Tester::MCAPTester<double> tester;
  using CI = CSRIndices<3, 1, 3, 0>;
  using CP = CSRPointers<0, 3, 3, 4>;
  using Sparse = CompiledSparseMatrix<T, 3, 4, CI, CP>;
  Sparse a, sparse_seed;
  fill(a);
  fill(sparse_seed);
  Matrix<T, 3, 4> dense_seed;
  fill(dense_seed);
  const T scale(3);

  compare(
      tester, sparse_seed, [&](Sparse &y) { SparseMatrixMinus::compute(a, y); },
      [&](Sparse &y) {
        SparseMatrixMinus::Column<T, 3, 4, CI, CP, 2>::compute(a, y);
      },
      "Sparse unary minus");
  compare(
      tester, sparse_seed,
      [&](Sparse &y) { SparseMatrixMultiplyScalar::compute(a, scale, y); },
      [&](Sparse &y) {
        SparseMatrixMultiplyScalar::Loop<T, 3, 4, CI, CP, 0, CI::size>::compute(
            a, scale, y);
      },
      "Sparse scalar product");
  compare(
      tester, sparse_seed,
      [&](Sparse &y) {
        DenseMatrixSubstituteSparseMatrix::compute(y, dense_seed);
      },
      [&](Sparse &y) {
        DenseMatrixSubstituteSparseMatrix::Column<T, 3, 4, CI, CP, 2>::compute(
            y, dense_seed);
      },
      "Dense assignment to sparse");
  compare(
      tester, dense_seed,
      [&](Matrix<T, 3, 4> &y) { SparseMatrixAddDenseMatrix::compute(a, y); },
      [&](Matrix<T, 3, 4> &y) {
        SparseMatrixAddDenseMatrix::Column<T, 3, 4, CI, CP, 2>::compute(a, y);
      },
      "Sparse add dense");
  compare(
      tester, dense_seed,
      [&](Matrix<T, 3, 4> &y) { DenseMatrixSubSparseMatrix::compute(a, y); },
      [&](Matrix<T, 3, 4> &y) {
        DenseMatrixSubSparseMatrix::Column<T, 3, 4, CI, CP, 2>::compute(a, y);
      },
      "Dense subtract sparse");
  compare(
      tester, sparse_seed,
      [&](Sparse &y) { SparseMatrixAddSparseMatrix::compute(a, y); },
      [&](Sparse &y) {
        SparseMatrixAddSparseMatrix::Column<T, 3, 4, CI, CP, CI, CP,
                                            2>::compute(a, y);
      },
      "Sparse add sparse with duplicate destination");
  compare(
      tester, sparse_seed,
      [&](Sparse &y) { SparseMatrixSubSparseMatrix::compute(a, y); },
      [&](Sparse &y) {
        SparseMatrixSubSparseMatrix::Column<T, 3, 4, CI, CP, CI, CP,
                                            2>::compute(a, y);
      },
      "Sparse subtract sparse with duplicate destination");
  compare(
      tester, sparse_seed,
      [&](Sparse &y) { DiagMatrixSubSparseMatrix::compute(a, y); },
      [&](Sparse &y) {
        DiagMatrixSubSparseMatrix::Column<T, 3, 4, CI, CP, CI, CP, 2>::compute(
            a, y);
      },
      "Diagonal subtract sparse update");

  DiagMatrix<T, 3> d3;
  fill(d3);
  DiagMatrix<T, 4> d4;
  fill(d4);
  compare(
      tester, sparse_seed,
      [&](Sparse &y) { SparseMatrixMultiplyDiagMatrix::compute(a, d4, y); },
      [&](Sparse &y) {
        SparseMatrixMultiplyDiagMatrix::Column<T, 3, 4, CI, CP, 2>::compute(
            a, d4, y);
      },
      "Sparse times diagonal");
  compare(
      tester, sparse_seed,
      [&](Sparse &y) { DiagMatrixMultiplySparseMatrix::compute(d3, a, y); },
      [&](Sparse &y) {
        DiagMatrixMultiplySparseMatrix::Column<T, 3, 4, CI, CP, 2>::compute(
            d3, a, y);
      },
      "Diagonal times sparse");
  using Square = FullSparse<T, 3, 3>;
  using SQI = CSRIndicesFromSparseAvailable<DenseAvailable<3, 3>>;
  using SQP = CSRPointersFromSparseAvailable<DenseAvailable<3, 3>>;
  Square square;
  fill(square);
  compare(
      tester, square,
      [&](Square &y) {
        SetDiagMatrixToValuesSparseMatrix::compute<T, 3, 3, SQI, SQP>(y, d3);
      },
      [&](Square &y) {
        SetDiagMatrixToValuesSparseMatrix::Core<T, 3, 3, SQI, SQP, 2>::apply(
            y, d3);
      },
      "Set sparse diagonal");
  using Transposed = FullSparse<T, 4, 3>;
  Transposed transposed;
  fill(transposed);
  // Cover both ordinary rows and the empty-row guard in the recursive helper.
  using TP = CSRPointers<0, 2, 3, 4>;
  CompiledSparseMatrix<T, 3, 4, CI, TP> no_empty_rows;
  fill(no_empty_rows);
  compare(
      tester, transposed,
      [&](Transposed &y) {
        TransposeOfDiagMatrixMultiplySparse::compute(d3, no_empty_rows, y);
      },
      [&](Transposed &y) {
        TransposeOfDiagMatrixMultiplySparse::List<T, 3, 4, CI, TP, Transposed,
                                                  2>::compute(d3, no_empty_rows,
                                                              y);
      },
      "Transpose of diagonal times sparse");
  compare(
      tester, transposed,
      [&](Transposed &y) {
        TransposeOfDiagMatrixMultiplySparse::compute(d3, a, y);
      },
      [&](Transposed &y) {
        TransposeOfDiagMatrixMultiplySparse::List<T, 3, 4, CI, CP, Transposed,
                                                  2>::compute(d3, a, y);
      },
      "Transpose of diagonal times sparse with an empty row");

  Matrix<T, 4, 2> b42;
  fill(b42);
  Matrix<T, 2, 3> b23;
  fill(b23);
  Matrix<T, 2, 4> b24;
  fill(b24);
  Matrix<T, 3, 2> b32;
  fill(b32);
  compare(
      tester, b32,
      [&](Matrix<T, 3, 2> &y) {
        SparseMatrixMultiplyDenseMatrix::compute(a, b42, y);
      },
      [&](Matrix<T, 3, 2> &y) {
        SparseMatrixMultiplyDenseMatrix::Row<T, 3, 4, 2, CI, CP, 1>::compute(
            a, b42, y);
      },
      "Sparse times dense");
  compare(
      tester, b24,
      [&](Matrix<T, 2, 4> &y) {
        DenseMatrixMultiplySparseMatrix::compute(b23, a, y);
      },
      [&](Matrix<T, 2, 4> &y) {
        DenseMatrixMultiplySparseMatrix::Row<T, 2, 3, 4, CI, CP, 2>::compute(
            b23, a, y);
      },
      "Dense times sparse");
  compare(
      tester, b32,
      [&](Matrix<T, 3, 2> &y) {
        SparseMatrixMultiplyDenseTranspose::compute(a, b24, y);
      },
      [&](Matrix<T, 3, 2> &y) {
        SparseMatrixMultiplyDenseTranspose::Row<T, 3, 4, 2, CI, CP, 1>::compute(
            a, b24, y);
      },
      "Sparse times dense transpose");
  compare(
      tester, b23,
      [&](Matrix<T, 2, 3> &y) {
        DenseMatrixMultiplySparseTranspose::compute(b24, a, y);
      },
      [&](Matrix<T, 2, 3> &y) {
        DenseMatrixMultiplySparseTranspose::Row<T, 2, 3, 4, CI, CP, 2>::compute(
            b24, a, y);
      },
      "Dense times sparse transpose");
  compare(
      tester, b24,
      [&](Matrix<T, 2, 4> &y) {
        DenseMatrixTransposeMultiplySparse::compute(b32, a, y);
      },
      [&](Matrix<T, 2, 4> &y) {
        DenseMatrixTransposeMultiplySparse::OuterLoop<T, 2, 3, 4, CI, CP, 2,
                                                      3>::compute(b32, a, y);
      },
      "Dense transpose times sparse");
  compare(
      tester, b42,
      [&](Matrix<T, 4, 2> &y) {
        SparseTransposeMatrixMultiplyDenseMatrix::compute(a, b32, y);
      },
      [&](Matrix<T, 4, 2> &y) {
        SparseTransposeMatrixMultiplyDenseMatrix::Row<T, 3, 4, 2, CI, CP,
                                                      1>::compute(a, b32, y);
      },
      "Sparse transpose times dense");

  Vector<T, 4> v4;
  fill(v4);
  Vector<T, 3> v3;
  fill(v3);
  ColVector<T, 3> c3;
  fill(c3);
  ColVector<T, 4> c4;
  fill(c4);
  compare(
      tester, v3,
      [&](Vector<T, 3> &y) { SparseMatrixMultiplyVector::compute(a, v4, y); },
      [&](Vector<T, 3> &y) {
        SparseMatrixMultiplyVector::List<T, 3, 4, CI, CP, 2>::compute(a, v4, y);
      },
      "Sparse times vector");
  compare(
      tester, c4,
      [&](ColVector<T, 4> &y) {
        ColVectorMultiplySparseMatrix::compute(c3, a, y);
      },
      [&](ColVector<T, 4> &y) {
        ColVectorMultiplySparseMatrix::List<T, 3, 4, CI, CP, 2>::compute(c3, a,
                                                                         y);
      },
      "Column vector times sparse");

  // Vector indexing historically clamps out-of-range indices to the last slot.
  using ClampCI = CSRIndices<4>;
  using ClampCP = CSRPointers<0, 0, 1, 1>;
  CompiledSparseMatrix<T, 3, 4, ClampCI, ClampCP> clamp;
  fill(clamp);
  compare(
      tester, v3,
      [&](Vector<T, 3> &y) {
        SparseMatrixMultiplyVector::compute(clamp, v4, y);
      },
      [&](Vector<T, 3> &y) {
        SparseMatrixMultiplyVector::List<T, 3, 4, ClampCI, ClampCP, 2>::compute(
            clamp, v4, y);
      },
      "Sparse vector input index clamp");
  compare(
      tester, c4,
      [&](ColVector<T, 4> &y) {
        ColVectorMultiplySparseMatrix::compute(c3, clamp, y);
      },
      [&](ColVector<T, 4> &y) {
        ColVectorMultiplySparseMatrix::List<T, 3, 4, ClampCI, ClampCP,
                                            2>::compute(c3, clamp, y);
      },
      "Sparse vector output index clamp");
  tester.throw_error_if_test_failed();
}

template <typename T> void check_sparse_products() {
  Tester::MCAPTester<double> tester;
  using AI = CSRIndices<3, 1, 3, 0>;
  using AP = CSRPointers<0, 3, 3, 4>;
  using BI = CSRIndices<1, 0, 1, 0>;
  using BP = CSRPointers<0, 1, 1, 3, 4>;
  using CI = CSRIndices<0, 1, 0>;
  using CP = CSRPointers<0, 1, 1, 3>;
  using DI = CSRIndices<3, 1, 0>;
  using DP = CSRPointers<0, 2, 3>;
  CompiledSparseMatrix<T, 3, 4, AI, AP> a;
  fill(a);
  CompiledSparseMatrix<T, 4, 2, BI, BP> b;
  fill(b);
  CompiledSparseMatrix<T, 3, 2, CI, CP> c;
  fill(c);
  CompiledSparseMatrix<T, 2, 4, DI, DP> d;
  fill(d);
  using YI = CSRIndicesFromSparseAvailable<DenseAvailable<3, 2>>;
  using YP = CSRPointersFromSparseAvailable<DenseAvailable<3, 2>>;
  using Y = CompiledSparseMatrix<T, 3, 2, YI, YP>;
  Y seed;
  fill(seed);
  compare(
      tester, seed, [&](Y &y) { SparseMatrixMultiplySparse::compute(a, b, y); },
      [&](Y &y) {
        SparseMatrixMultiplySparse::Row<T, 3, 4, 2, AI, AP, BI, BP, YI,
                                        YP>::compute(a, b, y);
      },
      "Sparse times sparse");
  compare(
      tester, seed,
      [&](Y &y) { SparseMatrixMultiplySparseTranspose::compute(a, d, y); },
      [&](Y &y) {
        SparseMatrixMultiplySparseTranspose::Row<T, 3, 4, 2, AI, AP, DI, DP, Y,
                                                 1>::compute(a, d, y);
      },
      "Sparse times sparse transpose");
  using YT = FullSparse<T, 4, 2>;
  YT seed_t;
  fill(seed_t);
  compare(
      tester, seed_t,
      [&](YT &y) { SparseMatrixTransposeMultiplySparse::compute(a, c, y); },
      [&](YT &y) {
        SparseMatrixTransposeMultiplySparse::Row<T, 4, 3, 2, AI, AP, CI, CP,
                                                 YT>::compute(a, c, y);
      },
      "Sparse transpose times sparse");

  // Missing destinations still incur the same arithmetic as the recursive
  // helpers; duplicate destinations retain last-read/all-write semantics.
  using HI = CSRIndices<1, 1>;
  using HP = CSRPointers<0, 2, 2, 2>;
  using Holes = CompiledSparseMatrix<T, 3, 2, HI, HP>;
  Holes holes;
  fill(holes);
  compare(
      tester, holes,
      [&](Holes &y) { SparseMatrixMultiplySparse::compute(a, b, y); },
      [&](Holes &y) {
        SparseMatrixMultiplySparse::Row<T, 3, 4, 2, AI, AP, BI, BP, HI,
                                        HP>::compute(a, b, y);
      },
      "Sparse product missing and duplicate output positions");
  compare(
      tester, holes,
      [&](Holes &y) { SparseMatrixMultiplySparseTranspose::compute(a, d, y); },
      [&](Holes &y) {
        SparseMatrixMultiplySparseTranspose::Row<T, 3, 4, 2, AI, AP, DI, DP,
                                                 Holes, 1>::compute(a, d, y);
      },
      "Sparse transpose-right product missing output positions");
  using HTP = CSRPointers<0, 2, 2, 2, 2>;
  using HolesT = CompiledSparseMatrix<T, 4, 2, HI, HTP>;
  HolesT holes_t;
  fill(holes_t);
  compare(
      tester, holes_t,
      [&](HolesT &y) { SparseMatrixTransposeMultiplySparse::compute(a, c, y); },
      [&](HolesT &y) {
        SparseMatrixTransposeMultiplySparse::Row<T, 4, 3, 2, AI, AP, CI, CP,
                                                 HolesT>::compute(a, c, y);
      },
      "Sparse transpose-left product missing output positions");

  using SI = CSRIndices<0, 1, 0, 1>;
  using SP = CSRPointers<0, 2, 4>;
  using Square = CompiledSparseMatrix<T, 2, 2, SI, SP>;
  Square square_a, square_b;
  fill(square_a);
  fill(square_b);
  compare(
      tester, square_a,
      [&](Square &y) { SparseMatrixMultiplySparse::compute(y, square_b, y); },
      [&](Square &y) {
        SparseMatrixMultiplySparse::Row<T, 2, 2, 2, SI, SP, SI, SP, SI,
                                        SP>::compute(y, square_b, y);
      },
      "Sparse product output aliases left input");
  compare(
      tester, square_b,
      [&](Square &y) { SparseMatrixMultiplySparse::compute(square_a, y, y); },
      [&](Square &y) {
        SparseMatrixMultiplySparse::Row<T, 2, 2, 2, SI, SP, SI, SP, SI,
                                        SP>::compute(square_a, y, y);
      },
      "Sparse product output aliases right input");

  using EI = CSRIndices<0>;
  using EP = CSRPointers<0, 0, 0, 0>;
  using Empty = CompiledSparseMatrix<T, 3, 4, EI, EP>;
  Empty empty;
  const T scale(3);
  compare(
      tester, empty,
      [&](Empty &y) {
        SparseMatrixMultiplyScalar::compute(empty, scale, y);
        tester.expect_near(
            static_cast<double>(y.values.size()), 0.0, 0.0,
            "Empty scalar product must not allocate a sentinel value");
        tester.expect_near(static_cast<double>(events().size()), 0.0, 0.0,
                           "Empty scalar product must perform no arithmetic");
      },
      [&](Empty &y) {
        // CSRIndices<0> has size 1, but the matrix stores zero coefficients.
        SparseMatrixMultiplyScalar::Loop<T, 3, 4, EI, EP, 0,
                                         EP::list[3]>::compute(empty, scale, y);
      },
      "Empty sparse scalar product must use the actual stored value count");
  events().clear();
  const auto scaled_right = empty * scale;
  const auto scaled_left = scale * empty;
  tester.expect_near(static_cast<double>(scaled_right.values.size()), 0.0, 0.0,
                     "Empty sparse times scalar must preserve empty storage");
  tester.expect_near(static_cast<double>(scaled_left.values.size()), 0.0, 0.0,
                     "Scalar times empty sparse must preserve empty storage");
  tester.expect_near(
      static_cast<double>(events().size()), 0.0, 0.0,
      "Both empty scalar-product operators must perform no arithmetic");
  events().clear();

  DiagMatrix<T, 3> diagonal;
  fill(diagonal);
  using Transposed = FullSparse<T, 4, 3>;
  Transposed transposed;
  fill(transposed);
  compare(
      tester, transposed,
      [&](Transposed &y) {
        TransposeOfDiagMatrixMultiplySparse::compute(diagonal, empty, y);
        tester.expect_near(
            static_cast<double>(events().size()), 0.0, 0.0,
            "Empty transpose-diagonal product must perform no arithmetic");
      },
      [&](Transposed &y) {
        TransposeOfDiagMatrixMultiplySparse::List<T, 3, 4, EI, EP, Transposed,
                                                  2>::compute(diagonal, empty,
                                                              y);
      },
      "Empty transpose-diagonal product must preserve existing output values");
  using ETP = CSRPointers<0, 0, 0, 0, 0>;
  using EmptyTransposed = CompiledSparseMatrix<T, 4, 3, EI, ETP>;
  EmptyTransposed empty_transposed;
  compare(
      tester, empty_transposed,
      [&](EmptyTransposed &y) {
        TransposeOfDiagMatrixMultiplySparse::compute(diagonal, empty, y);
      },
      [&](EmptyTransposed &y) {
        TransposeOfDiagMatrixMultiplySparse::List<
            T, 3, 4, EI, EP, EmptyTransposed, 2>::compute(diagonal, empty, y);
      },
      "Empty transpose-diagonal product with zero-size output storage");
  compare(
      tester, empty, [&](Empty &y) { SparseMatrixMinus::compute(empty, y); },
      [&](Empty &y) {
        SparseMatrixMinus::Column<T, 3, 4, EI, EP, 2>::compute(empty, y);
      },
      "Empty sparse unary minus");
  compare(
      tester, seed,
      [&](Y &y) { SparseMatrixMultiplySparse::compute(empty, b, y); },
      [&](Y &y) {
        SparseMatrixMultiplySparse::Row<T, 3, 4, 2, EI, EP, BI, BP, YI,
                                        YP>::compute(empty, b, y);
      },
      "Empty sparse product must leave accumulated output untouched");
  compare(
      tester, seed,
      [&](Y &y) { SparseMatrixMultiplySparseTranspose::compute(empty, d, y); },
      [&](Y &y) {
        SparseMatrixMultiplySparseTranspose::Row<T, 3, 4, 2, EI, EP, DI, DP, Y,
                                                 1>::compute(empty, d, y);
      },
      "Empty sparse transpose-right product");
  compare(
      tester, seed_t,
      [&](YT &y) { SparseMatrixTransposeMultiplySparse::compute(empty, c, y); },
      [&](YT &y) {
        SparseMatrixTransposeMultiplySparse::Row<T, 4, 3, 2, EI, EP, CI, CP,
                                                 YT>::compute(empty, c, y);
      },
      "Empty sparse transpose-left product");
  Matrix<T, 3, 4> dense;
  fill(dense);
  compare(
      tester, dense,
      [&](Matrix<T, 3, 4> &y) {
        SparseMatrixAddDenseMatrix::compute(empty, y);
      },
      [&](Matrix<T, 3, 4> &y) {
        SparseMatrixAddDenseMatrix::Column<T, 3, 4, EI, EP, 2>::compute(empty,
                                                                        y);
      },
      "Empty sparse addition must not add zeros");

  Matrix<T, 2, 3> dense23;
  fill(dense23);
  Matrix<T, 2, 4> dense24;
  fill(dense24);
  Matrix<T, 3, 2> dense32;
  fill(dense32);
  Matrix<T, 4, 2> dense42;
  fill(dense42);
  compare(
      tester, dense24,
      [&](Matrix<T, 2, 4> &y) {
        DenseMatrixMultiplySparseMatrix::compute(dense23, empty, y);
      },
      [&](Matrix<T, 2, 4> &y) {
        DenseMatrixMultiplySparseMatrix::Row<T, 2, 3, 4, EI, EP, 2>::compute(
            dense23, empty, y);
      },
      "Empty sparse scatter must preserve dense output");
  compare(
      tester, dense32,
      [&](Matrix<T, 3, 2> &y) {
        SparseMatrixMultiplyDenseTranspose::compute(empty, dense24, y);
      },
      [&](Matrix<T, 3, 2> &y) {
        SparseMatrixMultiplyDenseTranspose::Row<T, 3, 4, 2, EI, EP, 1>::compute(
            empty, dense24, y);
      },
      "Empty sparse gather must write zero dense output");
  compare(
      tester, dense42,
      [&](Matrix<T, 4, 2> &y) {
        SparseTransposeMatrixMultiplyDenseMatrix::compute(empty, dense32, y);
      },
      [&](Matrix<T, 4, 2> &y) {
        SparseTransposeMatrixMultiplyDenseMatrix::Row<T, 3, 4, 2, EI, EP,
                                                      1>::compute(empty,
                                                                  dense32, y);
      },
      "Empty transpose scatter must preserve dense output");
  Vector<T, 4> vector4;
  fill(vector4);
  Vector<T, 3> vector3;
  fill(vector3);
  compare(
      tester, vector3,
      [&](Vector<T, 3> &y) {
        SparseMatrixMultiplyVector::compute(empty, vector4, y);
      },
      [&](Vector<T, 3> &y) {
        SparseMatrixMultiplyVector::List<T, 3, 4, EI, EP, 2>::compute(
            empty, vector4, y);
      },
      "Empty sparse vector gather must write zeros");
  tester.throw_error_if_test_failed();
}

inline void check_bool_elementwise() {
  Tester::MCAPTester<double> tester;
  using CI = CSRIndices<0, 1>;
  using CP = CSRPointers<0, 1, 1, 2>;
  using Sparse = CompiledSparseMatrix<bool, 3, 2, CI, CP>;
  Sparse a;
  a.values[0] = true;
  a.values[1] = false;
  compare(
      tester, a, [&](Sparse &y) { SparseMatrixMinus::compute(a, y); },
      [&](Sparse &y) {
        SparseMatrixMinus::Column<bool, 3, 2, CI, CP, 2>::compute(a, y);
      },
      "Bool sparse unary minus, including vector<bool> storage");
  using EI = CSRIndices<0>;
  using EP = CSRPointers<0, 0, 0, 0>;
  CompiledSparseMatrix<bool, 3, 2, EI, EP> empty;
  // Nonempty bool sparse addition is not supported by the recursive API:
  // integer promotion conflicts with set_sparse_matrix_value's T deduction.
  compare(
      tester, a,
      [&](Sparse &y) { SparseMatrixAddSparseMatrix::compute(empty, y); },
      [&](Sparse &y) {
        SparseMatrixAddSparseMatrix::Column<bool, 3, 2, EI, EP, CI, CP,
                                            2>::compute(empty, y);
      },
      "Empty bool sparse addition, including vector<bool> storage");
  tester.throw_error_if_test_failed();
}

} // namespace ExtendedKernelTests

inline void run_all() {
  check_access<double>();
  check_access<float>();
  check_multiply<double>();
  check_multiply<float>();
  check_multiply<Complex<double>>();
  check_multiply<Complex<float>>();
  check_multiply_contracts();
  ExtendedKernelTests::check_elementwise_and_mixed<double>();
  ExtendedKernelTests::check_elementwise_and_mixed<float>();
  ExtendedKernelTests::check_elementwise_and_mixed<Complex<double>>();
  ExtendedKernelTests::check_elementwise_and_mixed<Complex<float>>();
  ExtendedKernelTests::check_elementwise_and_mixed<
      ExtendedKernelTests::Number>();
  ExtendedKernelTests::check_sparse_products<double>();
  ExtendedKernelTests::check_sparse_products<float>();
  ExtendedKernelTests::check_sparse_products<Complex<double>>();
  ExtendedKernelTests::check_sparse_products<Complex<float>>();
  ExtendedKernelTests::check_sparse_products<ExtendedKernelTests::Number>();
  ExtendedKernelTests::check_bool_elementwise();
}

} // namespace StaticSparseTests

#endif // CHECK_STATIC_SPARSE_HPP_
