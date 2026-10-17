/**
 * @file base_matrix_convolve.hpp
 * @brief Provides one-dimensional convolution operations for dense and
 * compiled sparse matrices.
 *
 * This header defines full, same, and valid convolution functions for
 * one-dimensional matrices. It supports row and column vectors, with
 * compile-time validation of input shapes and orientation. The computation
 * uses either for loops or template-recursive expansion, depending on
 * BASE_MATRIX_USE_FOR_LOOP_OPERATION_.
 *
 * @note
 * tparam M is the number of rows in the matrix.
 * tparam N is the number of columns in the matrix.
 */
#ifndef BASE_MATRIX_CONVOLVE_HPP_
#define BASE_MATRIX_CONVOLVE_HPP_

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_matrix.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace Base {
namespace Matrix {

namespace Convolve {

template <typename T, std::size_t M, std::size_t N>
inline T get_element(const Matrix<T, M, N> &matrix, std::size_t index,
                     bool is_row) {
  return is_row ? matrix(0, index) : matrix(index, 0);
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline T get_element(
    const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix,
    std::size_t index, bool is_row) {
  const std::size_t row = is_row ? 0 : index;
  const std::size_t col = is_row ? index : 0;
  const std::size_t value_index = matrix.get_value_index(row, col);
  return value_index == static_cast<std::size_t>(-1) ? static_cast<T>(0)
                                                     : matrix[value_index];
}

template <std::size_t M, std::size_t N, std::size_t P, std::size_t Q>
inline void validate_inputs() {
  static_assert((M == 1) != (N == 1),
                "Each input must be a non-scalar one-dimensional matrix.");
  static_assert((P == 1) != (Q == 1),
                "Each input must be a non-scalar one-dimensional matrix.");
  static_assert((M == 1) == (P == 1),
                "Both inputs must have the same orientation.");
  static_assert((M == 1 ? N : M) > 0 && (P == 1 ? Q : P) > 0,
                "Input arrays must not be empty.");
}

template <bool IsValid, bool IsRow, std::size_t FullIndex, std::size_t AIndex,
          typename A, typename B, typename ResultValue>
struct ConvolveTerm {
  static void compute(const A &a, const B &b, ResultValue &value) {
    static_cast<void>(a);
    static_cast<void>(b);
    static_cast<void>(value);
  }
};

template <bool IsRow, std::size_t FullIndex, std::size_t AIndex, typename A,
          typename B, typename ResultValue>
struct ConvolveTerm<true, IsRow, FullIndex, AIndex, A, B, ResultValue> {
  static void compute(const A &a, const B &b, ResultValue &value) {
    value += get_element(a, AIndex, IsRow) *
             get_element(b, FullIndex - AIndex, IsRow);
  }
};

template <bool IsRow, std::size_t FullIndex, std::size_t AIndex,
          std::size_t LengthA, std::size_t LengthB, typename A, typename B,
          typename ResultValue>
struct ConvolveTerms {
  static void compute(const A &a, const B &b, ResultValue &value) {
    ConvolveTerm<(FullIndex >= AIndex && FullIndex - AIndex < LengthB), IsRow,
                 FullIndex, AIndex, A, B, ResultValue>::compute(a, b, value);
    ConvolveTerms<IsRow, FullIndex, AIndex + 1, LengthA, LengthB, A, B,
                  ResultValue>::compute(a, b, value);
  }
};

template <bool IsRow, std::size_t FullIndex, std::size_t LengthA,
          std::size_t LengthB, typename A, typename B, typename ResultValue>
struct ConvolveTerms<IsRow, FullIndex, LengthA, LengthA, LengthB, A, B,
                     ResultValue> {
  static void compute(const A &a, const B &b, ResultValue &value) {
    static_cast<void>(a);
    static_cast<void>(b);
    static_cast<void>(value);
  }
};

template <bool IsRow, std::size_t OutputIndex, typename Result,
          typename ResultValue>
struct SetConvolveOutput {
  static void compute(Result &result, const ResultValue &value) {
    result(OutputIndex, 0) = value;
  }
};

template <std::size_t OutputIndex, typename Result, typename ResultValue>
struct SetConvolveOutput<true, OutputIndex, Result, ResultValue> {
  static void compute(Result &result, const ResultValue &value) {
    result(0, OutputIndex) = value;
  }
};

template <bool IsRow, std::size_t OutputIndex, std::size_t OutputLength,
          std::size_t Start, std::size_t LengthA, std::size_t LengthB,
          typename A, typename B, typename Result, typename ResultValue>
struct ConvolveOutputs {
  static void compute(const A &a, const B &b, Result &result) {
    ResultValue value = static_cast<ResultValue>(0);
    ConvolveTerms<IsRow, Start + OutputIndex, 0, LengthA, LengthB, A, B,
                  ResultValue>::compute(a, b, value);

    SetConvolveOutput<IsRow, OutputIndex, Result, ResultValue>::compute(result,
                                                                        value);

    ConvolveOutputs<IsRow, OutputIndex + 1, OutputLength, Start, LengthA,
                    LengthB, A, B, Result, ResultValue>::compute(a, b, result);
  }
};

template <bool IsRow, std::size_t OutputLength, std::size_t Start,
          std::size_t LengthA, std::size_t LengthB, typename A, typename B,
          typename Result, typename ResultValue>
struct ConvolveOutputs<IsRow, OutputLength, OutputLength, Start, LengthA,
                       LengthB, A, B, Result, ResultValue> {
  static void compute(const A &a, const B &b, Result &result) {
    static_cast<void>(a);
    static_cast<void>(b);
    static_cast<void>(result);
  }
};

template <bool IsRow, std::size_t OutputLength, std::size_t Start,
          std::size_t LengthA, std::size_t LengthB, typename A, typename B>
inline auto convolve_impl(const A &a, const B &b)
    -> Matrix<typename std::decay<decltype(get_element(a, 0, IsRow) *
                                           get_element(b, 0, IsRow))>::type,
              (IsRow ? 1 : OutputLength), (IsRow ? OutputLength : 1)> {
  typedef
      typename std::decay<decltype(get_element(a, 0, IsRow) *
                                   get_element(b, 0, IsRow))>::type ResultValue;
  Matrix<ResultValue, (IsRow ? 1 : OutputLength), (IsRow ? OutputLength : 1)>
      result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t output_index = 0; output_index < OutputLength;
       ++output_index) {
    const std::size_t full_index = Start + output_index;
    ResultValue value = static_cast<ResultValue>(0);

    for (std::size_t a_index = 0; a_index < LengthA; ++a_index) {
      if (full_index >= a_index && full_index - a_index < LengthB) {
        value += get_element(a, a_index, IsRow) *
                 get_element(b, full_index - a_index, IsRow);
      }
    }

    if (IsRow) {
      result(0, output_index) = value;
    } else {
      result(output_index, 0) = value;
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  ConvolveOutputs<IsRow, 0, OutputLength, Start, LengthA, LengthB, A, B,
                  decltype(result), ResultValue>::compute(a, b, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

} // namespace Convolve

/* (Dense Matrix) * (Dense Matrix) */
template <typename TA, std::size_t M, std::size_t N, typename TB, std::size_t P,
          std::size_t Q>
inline auto convolve_full(const Matrix<TA, M, N> &a, const Matrix<TB, P, Q> &b)
    -> decltype(Convolve::convolve_impl<
                (M == 1), (M == 1 ? N + Q - 1 : M + P - 1), 0, (M == 1 ? N : M),
                (P == 1 ? Q : P)>(a, b)) {
  Convolve::validate_inputs<M, N, P, Q>();
  return Convolve::convolve_impl<(M == 1), (M == 1 ? N + Q - 1 : M + P - 1), 0,
                                 (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b);
}

/* (Dense Matrix) * (Compiled Sparse Matrix) */
template <typename TA, std::size_t M, std::size_t N, typename TB, std::size_t P,
          std::size_t Q, typename CSRIndicesB, typename CSRPointersB>
inline auto convolve_full(
    const Matrix<TA, M, N> &a,
    const CompiledSparseMatrix<TB, P, Q, CSRIndicesB, CSRPointersB> &b)
    -> decltype(Convolve::convolve_impl<
                (M == 1), (M == 1 ? N + Q - 1 : M + P - 1), 0, (M == 1 ? N : M),
                (P == 1 ? Q : P)>(a, b)) {
  Convolve::validate_inputs<M, N, P, Q>();
  return Convolve::convolve_impl<(M == 1), (M == 1 ? N + Q - 1 : M + P - 1), 0,
                                 (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b);
}

/* (Compiled Sparse Matrix) * (Dense Matrix) */
template <typename TA, std::size_t M, std::size_t N, typename CSRIndicesA,
          typename CSRPointersA, typename TB, std::size_t P, std::size_t Q>
inline auto convolve_full(
    const CompiledSparseMatrix<TA, M, N, CSRIndicesA, CSRPointersA> &a,
    const Matrix<TB, P, Q> &b)
    -> decltype(Convolve::convolve_impl<
                (M == 1), (M == 1 ? N + Q - 1 : M + P - 1), 0, (M == 1 ? N : M),
                (P == 1 ? Q : P)>(a, b)) {
  Convolve::validate_inputs<M, N, P, Q>();
  return Convolve::convolve_impl<(M == 1), (M == 1 ? N + Q - 1 : M + P - 1), 0,
                                 (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b);
}

/* (Compiled Sparse Matrix) * (Compiled Sparse Matrix) */
template <typename TA, std::size_t M, std::size_t N, typename CSRIndicesA,
          typename CSRPointersA, typename TB, std::size_t P, std::size_t Q,
          typename CSRIndicesB, typename CSRPointersB>
inline auto convolve_full(
    const CompiledSparseMatrix<TA, M, N, CSRIndicesA, CSRPointersA> &a,
    const CompiledSparseMatrix<TB, P, Q, CSRIndicesB, CSRPointersB> &b)
    -> decltype(Convolve::convolve_impl<
                (M == 1), (M == 1 ? N + Q - 1 : M + P - 1), 0, (M == 1 ? N : M),
                (P == 1 ? Q : P)>(a, b)) {
  Convolve::validate_inputs<M, N, P, Q>();
  return Convolve::convolve_impl<(M == 1), (M == 1 ? N + Q - 1 : M + P - 1), 0,
                                 (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b);
}

/* (Dense Matrix) * (Dense Matrix) */
template <typename TA, std::size_t M, std::size_t N, typename TB, std::size_t P,
          std::size_t Q>
inline auto convolve_same(const Matrix<TA, M, N> &a, const Matrix<TB, P, Q> &b)
    -> decltype(Convolve::convolve_impl<
                (M == 1), (M == 1 ? (N > Q ? N : Q) : (M > P ? M : P)),
                ((M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) - 1) / 2,
                (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b)) {
  Convolve::validate_inputs<M, N, P, Q>();
  return Convolve::convolve_impl<
      (M == 1), (M == 1 ? (N > Q ? N : Q) : (M > P ? M : P)),
      ((M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) - 1) / 2, (M == 1 ? N : M),
      (P == 1 ? Q : P)>(a, b);
}

/* (Dense Matrix) * (Compiled Sparse Matrix) */
template <typename TA, std::size_t M, std::size_t N, typename TB, std::size_t P,
          std::size_t Q, typename CSRIndicesB, typename CSRPointersB>
inline auto convolve_same(
    const Matrix<TA, M, N> &a,
    const CompiledSparseMatrix<TB, P, Q, CSRIndicesB, CSRPointersB> &b)
    -> decltype(Convolve::convolve_impl<
                (M == 1), (M == 1 ? (N > Q ? N : Q) : (M > P ? M : P)),
                ((M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) - 1) / 2,
                (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b)) {
  Convolve::validate_inputs<M, N, P, Q>();
  return Convolve::convolve_impl<
      (M == 1), (M == 1 ? (N > Q ? N : Q) : (M > P ? M : P)),
      ((M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) - 1) / 2, (M == 1 ? N : M),
      (P == 1 ? Q : P)>(a, b);
}

/* (Compiled Sparse Matrix) * (Dense Matrix) */
template <typename TA, std::size_t M, std::size_t N, typename CSRIndicesA,
          typename CSRPointersA, typename TB, std::size_t P, std::size_t Q>
inline auto convolve_same(
    const CompiledSparseMatrix<TA, M, N, CSRIndicesA, CSRPointersA> &a,
    const Matrix<TB, P, Q> &b)
    -> decltype(Convolve::convolve_impl<
                (M == 1), (M == 1 ? (N > Q ? N : Q) : (M > P ? M : P)),
                ((M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) - 1) / 2,
                (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b)) {
  Convolve::validate_inputs<M, N, P, Q>();
  return Convolve::convolve_impl<
      (M == 1), (M == 1 ? (N > Q ? N : Q) : (M > P ? M : P)),
      ((M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) - 1) / 2, (M == 1 ? N : M),
      (P == 1 ? Q : P)>(a, b);
}

/* (Compiled Sparse Matrix) * (Compiled Sparse Matrix) */
template <typename TA, std::size_t M, std::size_t N, typename CSRIndicesA,
          typename CSRPointersA, typename TB, std::size_t P, std::size_t Q,
          typename CSRIndicesB, typename CSRPointersB>
inline auto convolve_same(
    const CompiledSparseMatrix<TA, M, N, CSRIndicesA, CSRPointersA> &a,
    const CompiledSparseMatrix<TB, P, Q, CSRIndicesB, CSRPointersB> &b)
    -> decltype(Convolve::convolve_impl<
                (M == 1), (M == 1 ? (N > Q ? N : Q) : (M > P ? M : P)),
                ((M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) - 1) / 2,
                (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b)) {
  Convolve::validate_inputs<M, N, P, Q>();
  return Convolve::convolve_impl<
      (M == 1), (M == 1 ? (N > Q ? N : Q) : (M > P ? M : P)),
      ((M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) - 1) / 2, (M == 1 ? N : M),
      (P == 1 ? Q : P)>(a, b);
}

/* (Dense Matrix) * (Dense Matrix) */
template <typename TA, std::size_t M, std::size_t N, typename TB, std::size_t P,
          std::size_t Q>
inline auto convolve_valid(const Matrix<TA, M, N> &a, const Matrix<TB, P, Q> &b)
    -> decltype(Convolve::convolve_impl<
                (M == 1),
                (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1)
                        : (M > P ? M - P + 1 : P - M + 1)),
                (M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) - 1,
                (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b)) {
  Convolve::validate_inputs<M, N, P, Q>();
  return Convolve::convolve_impl<(M == 1),
                                 (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1)
                                         : (M > P ? M - P + 1 : P - M + 1)),
                                 (M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) -
                                     1,
                                 (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b);
}

/* (Dense Matrix) * (Compiled Sparse Matrix) */
template <typename TA, std::size_t M, std::size_t N, typename TB, std::size_t P,
          std::size_t Q, typename CSRIndicesB, typename CSRPointersB>
inline auto convolve_valid(
    const Matrix<TA, M, N> &a,
    const CompiledSparseMatrix<TB, P, Q, CSRIndicesB, CSRPointersB> &b)
    -> decltype(Convolve::convolve_impl<
                (M == 1),
                (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1)
                        : (M > P ? M - P + 1 : P - M + 1)),
                (M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) - 1,
                (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b)) {
  Convolve::validate_inputs<M, N, P, Q>();
  return Convolve::convolve_impl<(M == 1),
                                 (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1)
                                         : (M > P ? M - P + 1 : P - M + 1)),
                                 (M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) -
                                     1,
                                 (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b);
}

/* (Compiled Sparse Matrix) * (Dense Matrix) */
template <typename TA, std::size_t M, std::size_t N, typename CSRIndicesA,
          typename CSRPointersA, typename TB, std::size_t P, std::size_t Q>
inline auto convolve_valid(
    const CompiledSparseMatrix<TA, M, N, CSRIndicesA, CSRPointersA> &a,
    const Matrix<TB, P, Q> &b)
    -> decltype(Convolve::convolve_impl<
                (M == 1),
                (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1)
                        : (M > P ? M - P + 1 : P - M + 1)),
                (M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) - 1,
                (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b)) {
  Convolve::validate_inputs<M, N, P, Q>();
  return Convolve::convolve_impl<(M == 1),
                                 (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1)
                                         : (M > P ? M - P + 1 : P - M + 1)),
                                 (M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) -
                                     1,
                                 (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b);
}

/* (Compiled Sparse Matrix) * (Compiled Sparse Matrix) */
template <typename TA, std::size_t M, std::size_t N, typename CSRIndicesA,
          typename CSRPointersA, typename TB, std::size_t P, std::size_t Q,
          typename CSRIndicesB, typename CSRPointersB>
inline auto convolve_valid(
    const CompiledSparseMatrix<TA, M, N, CSRIndicesA, CSRPointersA> &a,
    const CompiledSparseMatrix<TB, P, Q, CSRIndicesB, CSRPointersB> &b)
    -> decltype(Convolve::convolve_impl<
                (M == 1),
                (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1)
                        : (M > P ? M - P + 1 : P - M + 1)),
                (M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) - 1,
                (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b)) {
  Convolve::validate_inputs<M, N, P, Q>();
  return Convolve::convolve_impl<(M == 1),
                                 (M == 1 ? (N > Q ? N - Q + 1 : Q - N + 1)
                                         : (M > P ? M - P + 1 : P - M + 1)),
                                 (M == 1 ? (N < Q ? N : Q) : (M < P ? M : P)) -
                                     1,
                                 (M == 1 ? N : M), (P == 1 ? Q : P)>(a, b);
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_CONVOLVE_HPP_
