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

/* Calculate triangular matrix size */
template <std::size_t X, std::size_t Y> struct CalculateTriangularSize {
  static constexpr std::size_t value =
      (Y > 0) ? Y + CalculateTriangularSize<X, Y - 1>::value : 0;
};

template <std::size_t X> struct CalculateTriangularSize<X, 0> {
  static constexpr std::size_t value = 0;
};

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

    std::size_t consecutive_index = 0;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = i; j < N; j++) {
        Y.values[consecutive_index] = A(i, j);

        consecutive_index++;
      }
    }

    return Y;
  }

  static void set_values_upper(
      CompiledSparseMatrix<T, M, N, UpperTriangularRowIndices<M, N>,
                           UpperTriangularRowPointers<M, N>> &A,
      const Matrix<T, M, N> &B) {
    // Currently, only support M >= N.
    static_assert(M >= N, "M must be greater than or equal to N");

    std::size_t consecutive_index = 0;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = i; j < N; j++) {
        A.values[consecutive_index] = B(i, j);
        consecutive_index++;
      }
    }
  }

  /* Lower */
  static auto create_lower(void)
      -> SparseMatrix<T, M, N,
                      CalculateTriangularSize<M, ((N < M) ? N : M)>::value> {

    std::size_t consecutive_index = 0;

#ifdef BASE_MATRIX_USE_STD_VECTOR
    std::vector<T> values(CalculateTriangularSize<M, ((N < M) ? N : M)>::value,
                          static_cast<T>(0));
    std::vector<std::size_t> row_indices(
        CalculateTriangularSize<M, ((N < M) ? N : M)>::value,
        static_cast<std::size_t>(0));
    std::vector<std::size_t> row_pointers(M + 1, static_cast<std::size_t>(0));
#else
    std::array<T, CalculateTriangularSize<M, ((N < M) ? N : M)>::value> values =
        {};
    std::array<std::size_t,
               CalculateTriangularSize<M, ((N < M) ? N : M)>::value>
        row_indices = {};
    std::array<std::size_t, M + 1> row_pointers = {};
#endif

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < i + 1; j++) {
        row_indices[consecutive_index] = j;

        consecutive_index++;
        row_pointers[i + 1] = consecutive_index;
      }
    }

    return SparseMatrix<T, M, N,
                        CalculateTriangularSize<M, ((N < M) ? N : M)>::value>(
        values, row_indices, row_pointers);
  }

  static auto create_lower(const Matrix<T, M, N> &A)
      -> SparseMatrix<T, M, N,
                      CalculateTriangularSize<M, ((N < M) ? N : M)>::value> {

    std::size_t consecutive_index = 0;

#ifdef BASE_MATRIX_USE_STD_VECTOR
    std::vector<T> values(CalculateTriangularSize<M, ((N < M) ? N : M)>::value,
                          static_cast<T>(0));
    std::vector<std::size_t> row_indices(
        CalculateTriangularSize<M, ((N < M) ? N : M)>::value,
        static_cast<std::size_t>(0));
    std::vector<std::size_t> row_pointers(M + 1, static_cast<std::size_t>(0));
#else
    std::array<T, CalculateTriangularSize<M, ((N < M) ? N : M)>::value> values =
        {};
    std::array<std::size_t,
               CalculateTriangularSize<M, ((N < M) ? N : M)>::value>
        row_indices = {};
    std::array<std::size_t, M + 1> row_pointers = {};
#endif

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < i + 1; j++) {
        values[consecutive_index] = A(i, j);
        row_indices[consecutive_index] = j;

        consecutive_index++;
        row_pointers[i + 1] = consecutive_index;
      }
    }

    return SparseMatrix<T, M, N,
                        CalculateTriangularSize<M, ((N < M) ? N : M)>::value>(
        values, row_indices, row_pointers);
  }

  static void set_values_lower(
      SparseMatrix<T, M, N,
                   CalculateTriangularSize<M, ((N < M) ? N : M)>::value> &A,
      const Matrix<T, M, N> &B) {
    std::size_t consecutive_index = 0;

    for (std::size_t i = 0; i < M; i++) {
      for (std::size_t j = 0; j < i + 1; j++) {
        A.values[consecutive_index] = B(i, j);
        consecutive_index++;
      }
    }
  }
};

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_TRIANGULAR_SPARSE_HPP
