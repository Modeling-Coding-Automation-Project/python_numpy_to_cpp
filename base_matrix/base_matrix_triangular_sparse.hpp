#ifndef BASE_MATRIX_TRIANGULAR_SPARSE_HPP
#define BASE_MATRIX_TRIANGULAR_SPARSE_HPP

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_vector.hpp"
#include <array>
#include <cstddef>
#include <vector>

namespace Base {
namespace Matrix {

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

template <std::size_t M, std::size_t N>
using UpperTriangularRowIndices =
    typename ToRowIndices<UpperTriangularRowNumbers<M, N>>::type;

/* Create Upper Triangular Sparse Matrix Row Pointers */
template <typename TriangularCountNumbers, std::size_t M>
struct AccumulateTriangularElementNumberStruct {
  using type =
      typename AccumulateSparseMatrixElementNumberLoop<TriangularCountNumbers,
                                                       M>::type;
};

template <std::size_t M, std::size_t N>
using UpperTriangularRowPointers =
    typename ToRowPointers<typename AccumulateTriangularElementNumberStruct<
        typename TriangularCountNumbers<M, N>::type, M>::type>::type;

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
    typename ConcatenateLowerTriangularRowNumbers<M, ((M < N) ? M : N)>::type;

template <std::size_t M, std::size_t N>
using LowerTriangularRowIndices =
    typename ToRowIndices<LowerTriangularRowNumbers<M, N>>::type;

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

template <std::size_t M, std::size_t N>
using LowerTriangularRowPointers = typename ToRowPointers<
    typename AccumulateLowerTriangularElementNumberStruct<
        typename LowerTriangularCountNumbers<M, N>::type, M>::type>::type;

/* Set values for Upper Triangular Sparse Matrix */
// Calculate consecutive index at compile time
template <std::size_t I, std::size_t J, std::size_t N> struct ConsecutiveIndex {
  static constexpr std::size_t value = (I * (2 * N - I + 1)) / 2 + (J - I);
};

// Specialization for the base case
template <std::size_t N> struct ConsecutiveIndex<0, 0, N> {
  static constexpr std::size_t value = 0;
};

// Set values in the upper triangular matrix
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J>
struct SetUpperValues {
  static void
  compute(CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                               UpperTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    constexpr std::size_t index = ConsecutiveIndex<I, J, N>::value;
    A.values[index] = B(I, J);
    SetUpperValues<T, M, N, I, J - 1>::compute(A, B);
  }
};

// Specialization for the end of a row
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct SetUpperValues<T, M, N, I, I> {
  static void
  compute(CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                               UpperTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    constexpr std::size_t index = ConsecutiveIndex<I, I, N>::value;
    A.values[index] = B(I, I);
  }
};

// Set values for each row
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct SetUpperRow {
  static void
  compute(CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                               UpperTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    SetUpperValues<T, M, N, I, N - 1>::compute(A, B);
    SetUpperRow<T, M, N, I - 1>::compute(A, B);
  }
};

// Specialization for the first row
template <typename T, std::size_t M, std::size_t N>
struct SetUpperRow<T, M, N, 0> {
  static void
  compute(CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                               UpperTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    SetUpperValues<T, M, N, 0, N - 1>::compute(A, B);
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline void SET_UPPER_TRIANGULAR_VALUES(
    CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                         UpperTriangularRowPointers<M, N>> &A,
    const Matrix<T, M, N> &B) {
  SetUpperRow<T, M, N, M - 1>::compute(A, B);
}

/* Set values for Lower Triangular Sparse Matrix */
// Calculate consecutive index at compile time for lower triangular matrix
template <std::size_t I, std::size_t J, std::size_t N>
struct ConsecutiveIndexLower {
  static constexpr std::size_t value = (I * (I + 1)) / 2 + J;
};

// Specialization for the base case
template <std::size_t N> struct ConsecutiveIndexLower<0, 0, N> {
  static constexpr std::size_t value = 0;
};

// Set values in the lower triangular matrix
template <typename T, std::size_t M, std::size_t N, std::size_t I,
          std::size_t J>
struct SetLowerValues {
  static void
  compute(CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                               LowerTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    constexpr std::size_t index = ConsecutiveIndexLower<I, J, N>::value;
    A.values[index] = B(I, J);
    SetLowerValues<T, M, N, I, J - 1>::compute(A, B);
  }
};

// Specialization for the end of a row
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct SetLowerValues<T, M, N, I, 0> {
  static void
  compute(CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                               LowerTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    constexpr std::size_t index = ConsecutiveIndexLower<I, 0, N>::value;
    A.values[index] = B(I, 0);
  }
};

// Set values for each row
template <typename T, std::size_t M, std::size_t N, std::size_t I>
struct SetLowerRow {
  static void
  compute(CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                               LowerTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    SetLowerValues<T, M, N, I, I>::compute(A, B);
    SetLowerRow<T, M, N, I - 1>::compute(A, B);
  }
};

// Specialization for the first row
template <typename T, std::size_t M, std::size_t N>
struct SetLowerRow<T, M, N, 0> {
  static void
  compute(CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                               LowerTriangularRowPointers<M, N>> &A,
          const Matrix<T, M, N> &B) {
    SetLowerValues<T, M, N, 0, 0>::compute(A, B);
  }
};

template <typename T, std::size_t M, std::size_t N>
static inline void SET_LOWER_TRIANGULAR_VALUES(
    CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                         LowerTriangularRowPointers<M, N>> &A,
    const Matrix<T, M, N> &B) {
  SetLowerRow<T, M, N, M - 1>::compute(A, B);
}

template <typename T, std::size_t M, std::size_t N> class TriangularSparse {
public:
  TriangularSparse() {}

  /* Upper */
  static auto create_upper(void)
      -> CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                              UpperTriangularRowPointers<M, N>> {
    // Currently, only support M >= N.
    static_assert(M >= N, "M must be greater than or equal to N");

    CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                         UpperTriangularRowPointers<M, N>>
        Y;

    return Y;
  }

  static auto create_upper(const Matrix<T, M, N> &A)
      -> CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                              UpperTriangularRowPointers<M, N>> {
    // Currently, only support M >= N.
    static_assert(M >= N, "M must be greater than or equal to N");

    CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                         UpperTriangularRowPointers<M, N>>
        Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

    std::size_t consecutive_index = 0;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = i; j < N; j++) {
        Y.values[consecutive_index] = A(i, j);

        consecutive_index++;
      }
    }

#else

    SET_UPPER_TRIANGULAR_VALUES<T, M, N>(Y, A);

#endif

    return Y;
  }

  static void set_values_upper(
      CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                           UpperTriangularRowPointers<M, N>> &A,
      const Matrix<T, M, N> &B) {
    // Currently, only support M >= N.
    static_assert(M >= N, "M must be greater than or equal to N");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

    std::size_t consecutive_index = 0;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = i; j < N; j++) {
        A.values[consecutive_index] = B(i, j);
        consecutive_index++;
      }
    }

#else

    SET_UPPER_TRIANGULAR_VALUES<T, M, N>(A, B);

#endif
  }

  /* Lower */
  static auto create_lower(void)
      -> CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                              LowerTriangularRowPointers<M, N>> {
    // Currently, only support M <= N.
    static_assert(M <= N, "M must be smaller than or equal to N");

    CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                         LowerTriangularRowPointers<M, N>>
        Y;

    return Y;
  }

  static auto create_lower(const Matrix<T, M, N> &A)
      -> CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                              LowerTriangularRowPointers<M, N>> {
    // Currently, only support M <= N.
    static_assert(M <= N, "M must be smaller than or equal to N");

    CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                         LowerTriangularRowPointers<M, N>>
        Y;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

    std::size_t consecutive_index = 0;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < i + 1; j++) {
        Y.values[consecutive_index] = A(i, j);

        consecutive_index++;
      }
    }

#else

    SET_LOWER_TRIANGULAR_VALUES<T, M, N>(Y, A);

#endif

    return Y;
  }

  static void set_values_lower(
      CompiledSparseMatrix<T, M, N, LowerTriangularRowIndices<M, N>,
                           LowerTriangularRowPointers<M, N>> &A,
      const Matrix<T, M, N> &B) {
    // Currently, only support M <= N.
    static_assert(M <= N, "M must be smaller than or equal to N");

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION

    std::size_t consecutive_index = 0;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < i + 1; j++) {
        A.values[consecutive_index] = B(i, j);
        consecutive_index++;
      }
    }

#else

    SET_LOWER_TRIANGULAR_VALUES<T, M, N>(A, B);

#endif
  }
};

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_TRIANGULAR_SPARSE_HPP
