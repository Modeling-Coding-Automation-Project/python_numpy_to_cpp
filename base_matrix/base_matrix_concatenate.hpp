#ifndef BASE_MATRIX_CONCATENATE_HPP
#define BASE_MATRIX_CONCATENATE_HPP

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_compiled_sparse_operation.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include <cstddef>

namespace Base {
namespace Matrix {

/* Concatenate ColumnAvailable vertically */
template <typename Column1, typename Column2>
struct ConcatenateColumnAvailableLists;

template <bool... Flags1, bool... Flags2>
struct ConcatenateColumnAvailableLists<ColumnAvailable<Flags1...>,
                                       ColumnAvailable<Flags2...>> {
  using type = ColumnAvailable<Flags1..., Flags2...>;
};

template <typename ColumnAvailable_A, typename ColumnAvailable_B>
using ConcatenateColumnAvailableVertically =
    typename ConcatenateColumnAvailableLists<ColumnAvailable_A,
                                             ColumnAvailable_B>::type;

/* Concatenate SparseAvailable vertically */
template <typename SparseAvailable1, typename SparseAvailable2>
struct ConcatenateSparseAvailable;

template <typename... Columns1, typename... Columns2>
struct ConcatenateSparseAvailable<SparseAvailableColumns<Columns1...>,
                                  SparseAvailableColumns<Columns2...>> {
  using type = SparseAvailableColumns<Columns1..., Columns2...>;
};

/* Functions: Concatenate vertically */
template <typename T, std::size_t M, std::size_t N, std::size_t P>
Matrix<T, M + P, N> concatenate_vertically(const Matrix<T, M, N> &A,
                                           const Matrix<T, P, N> &B) {
  Matrix<T, M + P, N> result;

  for (std::size_t row = 0; row < N; row++) {
    std::copy(A(row).begin(), A(row).end(), result(row).begin());
    std::copy(B(row).begin(), B(row).end(), result(row).begin() + M);
  }

  return result;
}

template <typename T, std::size_t M, std::size_t N>
SparseMatrix<T, (M + N), N, ((M + 1) * N)>
concatenate_vertically(const Matrix<T, M, N> &A, const DiagMatrix<T, N> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values((M + 1) * N);
  std::vector<std::size_t> row_indices((M + 1) * N);
  std::vector<std::size_t> row_pointers(M + N + 1);
#else
  std::array<T, (M + 1) * N> values;
  std::array<std::size_t, ((M + 1) * N)> row_indices;
  std::array<std::size_t, (M + N + 1)> row_pointers;
#endif

  /* A */
  SparseMatrix<T, M, N, (M * N)> sparse_A = create_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), values.begin());
  std::copy(sparse_A.row_indices.begin(), sparse_A.row_indices.end(),
            row_indices.begin());
  std::copy(sparse_A.row_pointers.begin(), sparse_A.row_pointers.end(),
            row_pointers.begin());

  /* B */
  SparseMatrix<T, N, N, N> sparse_B = create_sparse(B);
  std::copy(sparse_B.values.begin(), sparse_B.values.end(),
            values.begin() + M * N);
  std::copy(sparse_B.row_indices.begin(), sparse_B.row_indices.end(),
            row_indices.begin() + M * N);

  std::size_t pointer_index = 1;
  for (std::size_t i = M + 1; i < (M + N + 1); i++) {
    row_pointers[i] = row_pointers[M] + sparse_B.row_pointers[pointer_index];
    pointer_index++;
  }

  /* Result */
  return SparseMatrix<T, (M + N), N, ((M + 1) * N)>(values, row_indices,
                                                    row_pointers);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t V>
SparseMatrix<T, (M + P), N, ((M * N) + V)>
concatenate_vertically(const Matrix<T, M, N> &A,
                       const SparseMatrix<T, P, N, V> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values((M * N) + V);
  std::vector<std::size_t> row_indices((M * N) + V);
  std::vector<std::size_t> row_pointers(M + P + 1);
#else
  std::array<T, ((M * N) + V)> values;
  std::array<std::size_t, ((M * N) + V)> row_indices;
  std::array<std::size_t, (M + P + 1)> row_pointers;
#endif

  /* A */
  SparseMatrix<T, M, N, (M * N)> sparse_A = create_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), values.begin());
  std::copy(sparse_A.row_indices.begin(), sparse_A.row_indices.end(),
            row_indices.begin());
  std::copy(sparse_A.row_pointers.begin(), sparse_A.row_pointers.end(),
            row_pointers.begin());

  /* B */
  std::copy(B.values.begin(), B.values.end(), values.begin() + M * N);
  std::copy(B.row_indices.begin(), B.row_indices.end(),
            row_indices.begin() + M * N);

  std::size_t pointer_index = 1;
  for (std::size_t i = M + 1; i < (M + P + 1); i++) {
    row_pointers[i] = row_pointers[M] + B.row_pointers[pointer_index];
    pointer_index++;
  }

  /* Result */
  return SparseMatrix<T, (M + P), N, ((M * N) + V)>(values, row_indices,
                                                    row_pointers);
}

template <typename T, std::size_t M, std::size_t P>
SparseMatrix<T, (M + P), M, ((P + 1) * M)>
concatenate_vertically(const DiagMatrix<T, M> &A, const Matrix<T, P, M> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values((P + 1) * M);
  std::vector<std::size_t> row_indices((P + 1) * M);
  std::vector<std::size_t> row_pointers(M + P + 1);
#else
  std::array<T, ((P + 1) * M)> values;
  std::array<std::size_t, ((P + 1) * M)> row_indices;
  std::array<std::size_t, (M + P + 1)> row_pointers;
#endif

  /* A */
  SparseMatrix<T, M, M, M> sparse_A = create_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), values.begin());
  std::copy(sparse_A.row_indices.begin(), sparse_A.row_indices.end(),
            row_indices.begin());
  std::copy(sparse_A.row_pointers.begin(), sparse_A.row_pointers.end(),
            row_pointers.begin());

  /* B */
  SparseMatrix<T, P, M, (P * M)> sparse_B = create_sparse(B);
  std::copy(sparse_B.values.begin(), sparse_B.values.end(), values.begin() + M);
  std::copy(sparse_B.row_indices.begin(), sparse_B.row_indices.end(),
            row_indices.begin() + M);

  std::size_t pointer_index = 1;
  for (std::size_t i = M + 1; i < (M + P + 1); i++) {
    row_pointers[i] = row_pointers[M] + sparse_B.row_pointers[pointer_index];
    pointer_index++;
  }

  /* Result */
  return SparseMatrix<T, (M + P), M, ((P + 1) * M)>(values, row_indices,
                                                    row_pointers);
}

template <typename T, std::size_t M>
auto concatenate_vertically(const DiagMatrix<T, M> &A,
                            const DiagMatrix<T, M> &B)
    -> SparseMatrix<T, (2 * M), M, (2 * M)> {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values(2 * M);
  std::vector<std::size_t> row_indices(2 * M);
  std::vector<std::size_t> row_pointers(2 * M + 1);
#else
  std::array<T, (2 * M)> values;
  std::array<std::size_t, (2 * M)> row_indices;
  std::array<std::size_t, (2 * M + 1)> row_pointers;
#endif

  /* A */
  SparseMatrix<T, M, M, M> sparse_A = create_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), values.begin());
  std::copy(sparse_A.row_indices.begin(), sparse_A.row_indices.end(),
            row_indices.begin());
  std::copy(sparse_A.row_pointers.begin(), sparse_A.row_pointers.end(),
            row_pointers.begin());

  /* B */
  SparseMatrix<T, M, M, M> sparse_B = create_sparse(B);
  std::copy(sparse_B.values.begin(), sparse_B.values.end(), values.begin() + M);
  std::copy(sparse_B.row_indices.begin(), sparse_B.row_indices.end(),
            row_indices.begin() + M);

  std::size_t pointer_index = 1;
  for (std::size_t i = M + 1; i < (2 * M + 1); i++) {
    row_pointers[i] = row_pointers[M] + sparse_B.row_pointers[pointer_index];
    pointer_index++;
  }

  /* Result */
  return SparseMatrix<T, (2 * M), M, (2 * M)>(values, row_indices,
                                              row_pointers);
}

template <typename T, std::size_t M, std::size_t P, std::size_t V>
SparseMatrix<T, (M + P), M, (M + V)>
concatenate_vertically(const DiagMatrix<T, M> &A,
                       const SparseMatrix<T, P, M, V> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values(M + V);
  std::vector<std::size_t> row_indices(M + V);
  std::vector<std::size_t> row_pointers(M + P + 1);
#else
  std::array<T, (M + V)> values;
  std::array<std::size_t, (M + V)> row_indices;
  std::array<std::size_t, (M + P + 1)> row_pointers;
#endif

  /* A */
  SparseMatrix<T, M, M, M> sparse_A = create_sparse(A);
  std::copy(sparse_A.values.begin(), sparse_A.values.end(), values.begin());
  std::copy(sparse_A.row_indices.begin(), sparse_A.row_indices.end(),
            row_indices.begin());
  std::copy(sparse_A.row_pointers.begin(), sparse_A.row_pointers.end(),
            row_pointers.begin());

  /* B */
  std::copy(B.values.begin(), B.values.end(), values.begin() + M);
  std::copy(B.row_indices.begin(), B.row_indices.end(),
            row_indices.begin() + M);

  std::size_t pointer_index = 1;
  for (std::size_t i = M + 1; i < (M + P + 1); i++) {
    row_pointers[i] = row_pointers[M] + B.row_pointers[pointer_index];
    pointer_index++;
  }

  /* Result */
  return SparseMatrix<T, (M + P), M, (M + V)>(values, row_indices,
                                              row_pointers);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t V, std::size_t W>
SparseMatrix<T, (M + P), N, (V + W)>
concatenate_vertically(const SparseMatrix<T, M, N, V> &A,
                       const SparseMatrix<T, P, N, W> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values(V + W);
  std::vector<std::size_t> row_indices(V + W);
  std::vector<std::size_t> row_pointers(M + P + 1);
#else
  std::array<T, (V + W)> values;
  std::array<std::size_t, (V + W)> row_indices;
  std::array<std::size_t, (M + P + 1)> row_pointers;
#endif

  /* A */
  std::copy(A.values.begin(), A.values.end(), values.begin());
  std::copy(A.row_indices.begin(), A.row_indices.end(), row_indices.begin());
  std::copy(A.row_pointers.begin(), A.row_pointers.end(), row_pointers.begin());

  /* B */
  std::copy(B.values.begin(), B.values.end(), values.begin() + V);
  std::copy(B.row_indices.begin(), B.row_indices.end(),
            row_indices.begin() + V);

  std::size_t pointer_index = 1;
  for (std::size_t i = M + 1; i < (M + P + 1); i++) {
    row_pointers[i] = row_pointers[M] + B.row_pointers[pointer_index];
    pointer_index++;
  }

  /* Result */
  return SparseMatrix<T, (M + P), N, (V + W)>(values, row_indices,
                                              row_pointers);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P>
Matrix<T, M, N + P> concatenate_horizontally(const Matrix<T, M, N> &A,
                                             const Matrix<T, M, P> &B) {
  Matrix<T, M, N + P> result;

  for (std::size_t row = 0; row < N; row++) {
    std::copy(A(row).begin(), A(row).end(), result(row).begin());
  }

  std::size_t B_row = 0;
  for (std::size_t row = N; row < N + P; row++) {
    std::copy(B(B_row).begin(), B(B_row).end(), result(row).begin());
    B_row++;
  }

  return result;
}

template <typename T, std::size_t M, std::size_t N>
SparseMatrix<T, M, (M + N), ((N + 1) * M)>
concatenate_horizontally(const Matrix<T, M, N> &A, const DiagMatrix<T, M> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values((N + 1) * M);
  std::vector<std::size_t> row_indices((N + 1) * M);
  std::vector<std::size_t> row_pointers(M + 1);
#else
  std::array<T, ((N + 1) * M)> values;
  std::array<std::size_t, ((N + 1) * M)> row_indices;
  std::array<std::size_t, (M + 1)> row_pointers;
#endif

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < N) {

        values[value_count] = A(i, j);
        row_indices[value_count] = j;

        value_count++;

      } else if ((j - N) == i) {

        values[value_count] = B[i];
        row_indices[value_count] = j;

        value_count++;
      }
    }
    row_pointers[i + 1] = value_count;
  }

  return SparseMatrix<T, M, (M + N), ((N + 1) * M)>(values, row_indices,
                                                    row_pointers);
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          std::size_t V>
SparseMatrix<T, M, (N + L), ((M * N) + V)>
concatenate_horizontally(const Matrix<T, M, N> &A,
                         const SparseMatrix<T, M, L, V> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values((M * N) + V);
  std::vector<std::size_t> row_indices((M * N) + V);
  std::vector<std::size_t> row_pointers(M + 1);
#else
  std::array<T, ((M * N) + V)> values;
  std::array<std::size_t, ((M * N) + V)> row_indices;
  std::array<std::size_t, (M + 1)> row_pointers;
#endif

  std::size_t value_count = 0;
  std::size_t sparse_value_count = 0;
  std::size_t sparse_col_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + L); j++) {
      if (j < N) {

        values[value_count] = A(i, j);
        row_indices[value_count] = j;

        value_count++;

      } else if ((B.row_pointers[i + 1] - B.row_pointers[i] >
                  sparse_col_count) &&
                 (sparse_value_count < V)) {

        if ((j - N) == B.row_indices[sparse_value_count]) {
          values[value_count] = B.values[sparse_value_count];
          row_indices[value_count] = j;

          value_count++;
          sparse_value_count++;
          sparse_col_count++;
        }
      }
    }

    row_pointers[i + 1] = value_count;
    sparse_col_count = 0;
  }

  return SparseMatrix<T, M, (N + L), ((M * N) + V)>(values, row_indices,
                                                    row_pointers);
}

template <typename T, std::size_t M, std::size_t N>
SparseMatrix<T, M, (M + N), ((N + 1) * M)>
concatenate_horizontally(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values((N + 1) * M);
  std::vector<std::size_t> row_indices((N + 1) * M);
  std::vector<std::size_t> row_pointers(M + 1);
#else
  std::array<T, ((N + 1) * M)> values;
  std::array<std::size_t, ((N + 1) * M)> row_indices;
  std::array<std::size_t, (M + 1)> row_pointers;
#endif

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < N) {
        if (i == j) {
          values[value_count] = A[i];
          row_indices[value_count] = j;

          value_count++;
        }
      } else {

        values[value_count] = B(i, j - N);
        row_indices[value_count] = j;

        value_count++;
      }
    }
    row_pointers[i + 1] = value_count;
  }

  return SparseMatrix<T, M, (M + N), ((N + 1) * M)>(values, row_indices,
                                                    row_pointers);
}

template <typename T, std::size_t M>
SparseMatrix<T, M, (2 * M), (2 * M)>
concatenate_horizontally(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values(2 * M);
  std::vector<std::size_t> row_indices(2 * M);
  std::vector<std::size_t> row_pointers(M + 1);
#else
  std::array<T, (2 * M)> values;
  std::array<std::size_t, (2 * M)> row_indices;
  std::array<std::size_t, (M + 1)> row_pointers;
#endif

  std::size_t value_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (2 * M); j++) {
      if (j < M) {
        if (i == j) {
          values[value_count] = A[i];
          row_indices[value_count] = j;

          value_count++;
        }
      } else {
        if ((j - M) == i) {
          values[value_count] = B[i];
          row_indices[value_count] = j;

          value_count++;
        }
      }
    }
    row_pointers[i + 1] = value_count;
  }

  return SparseMatrix<T, M, (2 * M), (2 * M)>(values, row_indices,
                                              row_pointers);
}

template <typename T, std::size_t M, std::size_t N, std::size_t V>
SparseMatrix<T, M, (M + N), (M + V)>
concatenate_horizontally(const DiagMatrix<T, M> &A,
                         const SparseMatrix<T, M, N, V> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values(M + V);
  std::vector<std::size_t> row_indices(M + V);
  std::vector<std::size_t> row_pointers(M + 1);
#else
  std::array<T, (M + V)> values;
  std::array<std::size_t, (M + V)> row_indices;
  std::array<std::size_t, (M + 1)> row_pointers;
#endif

  std::size_t value_count = 0;
  std::size_t sparse_value_count = 0;
  std::size_t sparse_col_count = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if (j < N) {
        if (i == j) {
          values[value_count] = A[i];
          row_indices[value_count] = j;

          value_count++;
        }
      } else if ((B.row_pointers[i + 1] - B.row_pointers[i] >
                  sparse_col_count) &&
                 (sparse_value_count < V)) {

        if ((j - N) == B.row_indices[sparse_value_count]) {
          values[value_count] = B.values[sparse_value_count];
          row_indices[value_count] = j;

          value_count++;
          sparse_value_count++;
          sparse_col_count++;
        }
      }
    }

    row_pointers[i + 1] = value_count;
    sparse_col_count = 0;
  }

  return SparseMatrix<T, M, (M + N), (M + V)>(values, row_indices,
                                              row_pointers);
}

template <typename T, std::size_t M, std::size_t N, std::size_t L,
          std::size_t V, std::size_t W>
SparseMatrix<T, M, (N + L), (V + W)>
concatenate_horizontally(const SparseMatrix<T, M, N, V> &A,
                         const SparseMatrix<T, M, L, W> &B) {

#ifdef BASE_MATRIX_USE_STD_VECTOR
  std::vector<T> values(V + W);
  std::vector<std::size_t> row_indices(V + W);
  std::vector<std::size_t> row_pointers(M + 1);
#else
  std::array<T, (V + W)> values;
  std::array<std::size_t, (V + W)> row_indices;
  std::array<std::size_t, (M + 1)> row_pointers;
#endif

  std::size_t value_count = 0;
  std::size_t sparse_value_count_A = 0;
  std::size_t sparse_col_count_A = 0;
  std::size_t sparse_value_count_B = 0;
  std::size_t sparse_col_count_B = 0;

  row_pointers[0] = 0;
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < (N + M); j++) {
      if ((j < N) &&
          (A.row_pointers[i + 1] - A.row_pointers[i] > sparse_col_count_A) &&
          (sparse_value_count_A < V)) {

        if (j == A.row_indices[sparse_value_count_A]) {
          values[value_count] = A.values[sparse_value_count_A];
          row_indices[value_count] = j;

          value_count++;
          sparse_value_count_A++;
          sparse_col_count_A++;
        }
      } else if ((B.row_pointers[i + 1] - B.row_pointers[i] >
                  sparse_col_count_B) &&
                 (sparse_value_count_B < W)) {

        if ((j - N) == B.row_indices[sparse_value_count_B]) {
          values[value_count] = B.values[sparse_value_count_B];
          row_indices[value_count] = j;

          value_count++;
          sparse_value_count_B++;
          sparse_col_count_B++;
        }
      }
    }

    row_pointers[i + 1] = value_count;
    sparse_col_count_A = 0;
    sparse_col_count_B = 0;
  }

  return SparseMatrix<T, M, (N + L), (V + W)>(values, row_indices,
                                              row_pointers);
}

template <typename T, std::size_t M, std::size_t N, std::size_t P,
          std::size_t K>
Matrix<T, M + K, N + P>
concatenate_square(const Matrix<T, M, N> &A, const Matrix<T, M, P> &B,
                   const Matrix<T, K, N> &C, const Matrix<T, K, P> &D) {
  Matrix<T, M + K, N + P> result;

  for (std::size_t i = 0; i < N; ++i) {
    std::copy(A(i).begin(), A(i).end(), result(i).begin());
    std::copy(C(i).begin(), C(i).end(), result(i).begin() + M);
  }

  std::size_t B_row = 0;
  for (std::size_t i = N; i < N + P; ++i) {
    std::copy(B(B_row).begin(), B(B_row).end(), result(i).begin());
    std::copy(D(B_row).begin(), D(B_row).end(), result(i).begin() + M);
    B_row++;
  }

  return result;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_CONCATENATE_HPP
