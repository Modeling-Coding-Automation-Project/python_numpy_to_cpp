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

/**
 * @brief Retrieves an element from a one-dimensional dense matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input dense matrix.
 * @param index The index of the element to retrieve.
 * @param is_row True if the matrix is a row vector, false if it is a column
 * vector.
 * @return The value of the element at the specified index.
 */
template <typename T, std::size_t M, std::size_t N>
inline T get_element(const Matrix<T, M, N> &matrix, std::size_t index,
                     bool is_row) {
  return is_row ? matrix(0, index) : matrix(index, 0);
}

/**
 * @brief Retrieves an element from a one-dimensional compiled sparse matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @tparam CSRIndices The type of the CSR indices.
 * @tparam CSRPointers The type of the CSR pointers.
 * @param matrix The input compiled sparse matrix.
 * @param index The index of the element to retrieve.
 * @param is_row True if the matrix is a row vector, false if it is a column
 * vector.
 * @return The value of the element at the specified index.
 */
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

/**
 * @brief Validates the inputs for one-dimensional convolution.
 *
 * @tparam M The number of rows in the first matrix.
 * @tparam N The number of columns in the first matrix.
 * @tparam P The number of rows in the second matrix.
 * @tparam Q The number of columns in the second matrix.
 *
 * @note This function performs compile-time checks to ensure that the inputs
 * are non-scalar one-dimensional matrices with the same orientation and that
 * they are not empty.
 */
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
  /**
   * @brief Computes the contribution of a single term in the convolution sum.
   *
   * @param a The first input matrix.
   * @param b The second input matrix.
   * @param value The result value to which the term is added.
   */
  static void compute(const A &a, const B &b, ResultValue &value) {
    static_cast<void>(a);
    static_cast<void>(b);
    static_cast<void>(value);
  }
};

template <bool IsRow, std::size_t FullIndex, std::size_t AIndex, typename A,
          typename B, typename ResultValue>
struct ConvolveTerm<true, IsRow, FullIndex, AIndex, A, B, ResultValue> {
  /**
   * @brief Specialization of ConvolveTerm for valid terms in the convolution
   * sum.
   *
   * @tparam IsRow True if the matrix is a row vector, false if it is a column
   * vector.
   * @tparam FullIndex The index in the full convolution output.
   * @tparam AIndex The index in the first input matrix.
   * @tparam A The type of the first input matrix.
   * @tparam B The type of the second input matrix.
   * @tparam ResultValue The type of the result value.
   */
  static void compute(const A &a, const B &b, ResultValue &value) {
    value += get_element(a, AIndex, IsRow) *
             get_element(b, FullIndex - AIndex, IsRow);
  }
};

template <bool IsRow, std::size_t FullIndex, std::size_t AIndex,
          std::size_t LengthA, std::size_t LengthB, typename A, typename B,
          typename ResultValue>
struct ConvolveTerms {
  /**
   * @brief Computes the contributions of all valid terms in the convolution sum
   * for a given index in the full output.
   *
   * @param a The first input matrix.
   * @param b The second input matrix.
   * @param value The result value to which the terms are added.
   */
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
  /**
   * @brief Specialization of ConvolveTerms for the case when all terms have
   * been processed.
   *
   * @param a The first input matrix.
   * @param b The second input matrix.
   * @param value The result value to which the terms are added.
   */
  static void compute(const A &a, const B &b, ResultValue &value) {
    static_cast<void>(a);
    static_cast<void>(b);
    static_cast<void>(value);
  }
};

template <bool IsRow, std::size_t OutputIndex, typename Result,
          typename ResultValue>
struct SetConvolveOutput {
  /**
   * @brief Sets the value of a specific output element in the convolution
   * result.
   *
   * @param result The result matrix.
   * @param value The value to set at the specified output index.
   */
  static void compute(Result &result, const ResultValue &value) {
    result(OutputIndex, 0) = value;
  }
};

template <std::size_t OutputIndex, typename Result, typename ResultValue>
struct SetConvolveOutput<true, OutputIndex, Result, ResultValue> {
  /**
   * @brief Specialization of SetConvolveOutput for row vectors.
   *
   * @param result The result matrix.
   * @param value The value to set at the specified output index.
   */
  static void compute(Result &result, const ResultValue &value) {
    result(0, OutputIndex) = value;
  }
};

template <bool IsRow, std::size_t OutputIndex, std::size_t OutputLength,
          std::size_t Start, std::size_t LengthA, std::size_t LengthB,
          typename A, typename B, typename Result, typename ResultValue>
struct ConvolveOutputs {
  /**
   * @brief Computes the convolution outputs for a given range of output
   * indices.
   *
   * @param a The first input matrix.
   * @param b The second input matrix.
   * @param result The result matrix where the convolution outputs are stored.
   */
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
  /**
   * @brief Specialization of ConvolveOutputs for the case when all output
   * indices have been processed.
   *
   * @param a The first input matrix.
   * @param b The second input matrix.
   * @param result The result matrix where the convolution outputs are stored.
   */
  static void compute(const A &a, const B &b, Result &result) {
    static_cast<void>(a);
    static_cast<void>(b);
    static_cast<void>(result);
  }
};

/**
 * @brief Implementation of the full convolution operation.
 *
 * @tparam IsRow Indicates whether the output is a row vector.
 * @tparam OutputLength The length of the output vector.
 * @tparam Start The starting index for the convolution.
 * @tparam LengthA The length of the first input matrix.
 * @tparam LengthB The length of the second input matrix.
 * @tparam A The type of the first input matrix.
 * @tparam B The type of the second input matrix.
 * @param a The first input matrix.
 * @param b The second input matrix.
 * @return The result matrix containing the full convolution output.
 */
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

/**
 * @brief Computes the full convolution of two dense matrices.
 *
 * @tparam TA The type of the elements in the first input matrix.
 * @tparam M The number of rows in the first input matrix.
 * @tparam N The number of columns in the first input matrix.
 * @tparam TB The type of the elements in the second input matrix.
 * @tparam P The number of rows in the second input matrix.
 * @tparam Q The number of columns in the second input matrix.
 * @param a The first input matrix.
 * @param b The second input matrix.
 * @return The result matrix containing the full convolution output.
 */
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

/**
 * @brief Computes the full convolution of a dense matrix with a compiled sparse
 * matrix.
 *
 * @tparam TA The type of the elements in the dense input matrix.
 * @tparam M The number of rows in the dense input matrix.
 * @tparam N The number of columns in the dense input matrix.
 * @tparam TB The type of the elements in the compiled sparse input matrix.
 * @tparam P The number of rows in the compiled sparse input matrix.
 * @tparam Q The number of columns in the compiled sparse input matrix.
 * @tparam CSRIndicesB The type of the CSR indices for the compiled sparse input
 * matrix.
 * @tparam CSRPointersB The type of the CSR pointers for the compiled sparse
 * input matrix.
 * @param a The dense input matrix.
 * @param b The compiled sparse input matrix.
 * @return The result matrix containing the full convolution output.
 */
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

/**
 * @brief Computes the full convolution of a compiled sparse matrix with a dense
 * matrix.
 *
 * @tparam TA The type of the elements in the compiled sparse input matrix.
 * @tparam M The number of rows in the compiled sparse input matrix.
 * @tparam N The number of columns in the compiled sparse input matrix.
 * @tparam CSRIndicesA The type of the CSR indices for the compiled sparse input
 * matrix.
 * @tparam CSRPointersA The type of the CSR pointers for the compiled sparse
 * input matrix.
 * @tparam TB The type of the elements in the dense input matrix.
 * @tparam P The number of rows in the dense input matrix.
 * @tparam Q The number of columns in the dense input matrix.
 * @param a The compiled sparse input matrix.
 * @param b The dense input matrix.
 * @return The result matrix containing the full convolution output.
 */
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

/**
 * @brief Computes the full convolution of two compiled sparse matrices.
 *
 * @tparam TA The type of the elements in the first compiled sparse input
 * matrix.
 * @tparam M The number of rows in the first compiled sparse input matrix.
 * @tparam N The number of columns in the first compiled sparse input matrix.
 * @tparam CSRIndicesA The type of the CSR indices for the first compiled sparse
 * input matrix.
 * @tparam CSRPointersA The type of the CSR pointers for the first compiled
 * sparse input matrix.
 * @tparam TB The type of the elements in the second compiled sparse input
 * matrix.
 * @tparam P The number of rows in the second compiled sparse input matrix.
 * @tparam Q The number of columns in the second compiled sparse input matrix.
 * @tparam CSRIndicesB The type of the CSR indices for the second compiled
 * sparse input matrix.
 * @tparam CSRPointersB The type of the CSR pointers for the second compiled
 * sparse input matrix.
 * @param a The first compiled sparse input matrix.
 * @param b The second compiled sparse input matrix.
 * @return The result matrix containing the full convolution output.
 */
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

/**
 * @brief Computes the same convolution of two dense matrices.
 *
 * @tparam TA The type of the elements in the first dense input matrix.
 * @tparam M The number of rows in the first dense input matrix.
 * @tparam N The number of columns in the first dense input matrix.
 * @tparam TB The type of the elements in the second dense input matrix.
 * @tparam P The number of rows in the second dense input matrix.
 * @tparam Q The number of columns in the second dense input matrix.
 * @param a The first dense input matrix.
 * @param b The second dense input matrix.
 * @return The result matrix containing the same convolution output.
 */
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

/**
 * @brief Computes the same convolution of a dense matrix with a compiled sparse
 * matrix.
 *
 * @tparam TA The type of the elements in the dense input matrix.
 * @tparam M The number of rows in the dense input matrix.
 * @tparam N The number of columns in the dense input matrix.
 * @tparam TB The type of the elements in the compiled sparse input matrix.
 * @tparam P The number of rows in the compiled sparse input matrix.
 * @tparam Q The number of columns in the compiled sparse input matrix.
 * @tparam CSRIndicesB The type of the CSR indices for the compiled sparse input
 * matrix.
 * @tparam CSRPointersB The type of the CSR pointers for the compiled sparse
 * input matrix.
 * @param a The dense input matrix.
 * @param b The compiled sparse input matrix.
 * @return The result matrix containing the same convolution output.
 */
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

/**
 * @brief Computes the same convolution of a compiled sparse matrix with a dense
 * matrix.
 *
 * @tparam TA The type of the elements in the compiled sparse input matrix.
 * @tparam M The number of rows in the compiled sparse input matrix.
 * @tparam N The number of columns in the compiled sparse input matrix.
 * @tparam CSRIndicesA The type of the CSR indices for the compiled sparse input
 * matrix.
 * @tparam CSRPointersA The type of the CSR pointers for the compiled sparse
 * input matrix.
 * @tparam TB The type of the elements in the dense input matrix.
 * @tparam P The number of rows in the dense input matrix.
 * @tparam Q The number of columns in the dense input matrix.
 * @param a The compiled sparse input matrix.
 * @param b The dense input matrix.
 * @return The result matrix containing the same convolution output.
 */
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

/**
 * @brief Computes the same convolution of two compiled sparse matrices.
 *
 * @tparam TA The type of the elements in the first compiled sparse input
 * matrix.
 * @tparam M The number of rows in the first compiled sparse input matrix.
 * @tparam N The number of columns in the first compiled sparse input matrix.
 * @tparam CSRIndicesA The type of the CSR indices for the first compiled sparse
 * input matrix.
 * @tparam CSRPointersA The type of the CSR pointers for the first compiled
 * sparse input matrix.
 * @tparam TB The type of the elements in the second compiled sparse input
 * matrix.
 * @tparam P The number of rows in the second compiled sparse input matrix.
 * @tparam Q The number of columns in the second compiled sparse input matrix.
 * @tparam CSRIndicesB The type of the CSR indices for the second compiled
 * sparse input matrix.
 * @tparam CSRPointersB The type of the CSR pointers for the second compiled
 * sparse input matrix.
 * @param a The first compiled sparse input matrix.
 * @param b The second compiled sparse input matrix.
 * @return The result matrix containing the same convolution output.
 */
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

/**
 * @brief Computes the valid convolution of two dense matrices.
 *
 * @tparam TA The type of the elements in the first dense input matrix.
 * @tparam M The number of rows in the first dense input matrix.
 * @tparam N The number of columns in the first dense input matrix.
 * @tparam TB The type of the elements in the second dense input matrix.
 * @tparam P The number of rows in the second dense input matrix.
 * @tparam Q The number of columns in the second dense input matrix.
 * @param a The first dense input matrix.
 * @param b The second dense input matrix.
 * @return The result matrix containing the valid convolution output.
 */
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

/**
 * @brief Computes the valid convolution of a dense matrix with a compiled
 * sparse matrix.
 *
 * @tparam TA The type of the elements in the dense input matrix.
 * @tparam M The number of rows in the dense input matrix.
 * @tparam N The number of columns in the dense input matrix.
 * @tparam TB The type of the elements in the compiled sparse input matrix.
 * @tparam P The number of rows in the compiled sparse input matrix.
 * @tparam Q The number of columns in the compiled sparse input matrix.
 * @tparam CSRIndicesB The type of the CSR indices for the compiled sparse input
 * matrix.
 * @tparam CSRPointersB The type of the CSR pointers for the compiled sparse
 * input matrix.
 * @param a The dense input matrix.
 * @param b The compiled sparse input matrix.
 * @return The result matrix containing the valid convolution output.
 */
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

/**
 * @brief Computes the valid convolution of a compiled sparse matrix with a
 * dense matrix.
 *
 * @tparam TA The type of the elements in the compiled sparse input matrix.
 * @tparam M The number of rows in the compiled sparse input matrix.
 * @tparam N The number of columns in the compiled sparse input matrix.
 * @tparam CSRIndicesA The type of the CSR indices for the compiled sparse input
 * matrix.
 * @tparam CSRPointersA The type of the CSR pointers for the compiled sparse
 * input matrix.
 * @tparam TB The type of the elements in the dense input matrix.
 * @tparam P The number of rows in the dense input matrix.
 * @tparam Q The number of columns in the dense input matrix.
 * @param a The compiled sparse input matrix.
 * @param b The dense input matrix.
 * @return The result matrix containing the valid convolution output.
 */
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

/**
 * @brief Computes the valid convolution of two compiled sparse matrices.
 *
 * @tparam TA The type of the elements in the first compiled sparse input
 * matrix.
 * @tparam M The number of rows in the first compiled sparse input matrix.
 * @tparam N The number of columns in the first compiled sparse input matrix.
 * @tparam CSRIndicesA The type of the CSR indices for the first compiled sparse
 * input matrix.
 * @tparam CSRPointersA The type of the CSR pointers for the first compiled
 * sparse input matrix.
 * @tparam TB The type of the elements in the second compiled sparse input
 * matrix.
 * @tparam P The number of rows in the second compiled sparse input matrix.
 * @tparam Q The number of columns in the second compiled sparse input matrix.
 * @tparam CSRIndicesB The type of the CSR indices for the second compiled
 * sparse input matrix.
 * @tparam CSRPointersB The type of the CSR pointers for the second compiled
 * sparse input matrix.
 * @param a The first compiled sparse input matrix.
 * @param b The second compiled sparse input matrix.
 * @return The result matrix containing the valid convolution output.
 */
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
