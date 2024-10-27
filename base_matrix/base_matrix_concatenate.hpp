#ifndef BASE_MATRIX_CONCATENATE_HPP
#define BASE_MATRIX_CONCATENATE_HPP

#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include <cstddef>
#include <cstring>

namespace Base {
namespace Matrix {

template <typename T, std::size_t M, std::size_t N, std::size_t P>
Matrix<T, M + P, N> concatenate_vertically(const Matrix<T, M, N> &A,
                                           const Matrix<T, P, N> &B) {
  Matrix<T, M + P, N> result;

  for (std::size_t row = 0; row < N; row++) {
    std::memcpy(&result(row)[0], &A(row)[0], M * sizeof(result(row)[0]));
    std::memcpy(&result(row)[M], &B(row)[0], P * sizeof(result(row)[M]));
  }

  return result;
}

template <typename T, std::size_t M, std::size_t N>
SparseMatrix<T, (M + N), N, ((M + 1) * N)>
concatenate_vertically(const Matrix<T, M, N> &A, const DiagMatrix<T, N> &B) {

  std::vector<T> values((M + 1) * N);
  std::vector<std::size_t> row_indices((M + 1) * N);
  std::vector<std::size_t> row_pointers(M + N + 1);

  /* A */
  SparseMatrix<T, M, N, (M * N)> sparse_A = create_sparse(A);
  std::memcpy(&values[0], &sparse_A.values[0], (M * N) * sizeof(values[0]));
  std::memcpy(&row_indices[0], &sparse_A.row_indices[0],
              (M * N) * sizeof(row_indices[0]));
  std::memcpy(&row_pointers[0], &sparse_A.row_pointers[0],
              (M + 1) * sizeof(row_pointers[0]));

  /* B */
  SparseMatrix<T, N, N, N> sparse_B = create_sparse(B);
  std::memcpy(&values[M * N], &sparse_B.values[0], N * sizeof(values[M * N]));
  std::memcpy(&row_indices[M * N], &sparse_B.row_indices[0],
              N * sizeof(row_indices[M * N]));

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

  std::vector<T> values((M * N) + V);
  std::vector<std::size_t> row_indices((M * N) + V);
  std::vector<std::size_t> row_pointers(M + P + 1);

  /* A */
  SparseMatrix<T, M, N, (M * N)> sparse_A = create_sparse(A);
  std::memcpy(&values[0], &sparse_A.values[0], (M * N) * sizeof(T));
  std::memcpy(&row_indices[0], &sparse_A.row_indices[0],
              (M * N) * sizeof(values[0]));
  std::memcpy(&row_pointers[0], &sparse_A.row_pointers[0],
              (M + 1) * sizeof(row_pointers[0]));

  /* B */
  std::memcpy(&values[M * N], &B.values[0], V * sizeof(values[M * N]));
  std::memcpy(&row_indices[M * N], &B.row_indices[0],
              V * sizeof(row_indices[M * N]));

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

  std::vector<T> values((P + 1) * M);
  std::vector<std::size_t> row_indices((P + 1) * M);
  std::vector<std::size_t> row_pointers(M + P + 1);

  /* A */
  SparseMatrix<T, M, M, M> sparse_A = create_sparse(A);
  std::memcpy(&values[0], &sparse_A.values[0], M * sizeof(values[0]));
  std::memcpy(&row_indices[0], &sparse_A.row_indices[0],
              M * sizeof(row_indices[0]));
  std::memcpy(&row_pointers[0], &sparse_A.row_pointers[0],
              (M + 1) * sizeof(row_pointers[0]));

  /* B */
  SparseMatrix<T, P, M, (P * M)> sparse_B = create_sparse(B);
  std::memcpy(&values[M], &sparse_B.values[0], (P * M) * sizeof(values[M]));
  std::memcpy(&row_indices[M], &sparse_B.row_indices[0],
              (P * M) * sizeof(row_indices[M]));

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

  std::vector<T> values(2 * M);
  std::vector<std::size_t> row_indices(2 * M);
  std::vector<std::size_t> row_pointers(2 * M + 1);

  /* A */
  SparseMatrix<T, M, M, M> sparse_A = create_sparse(A);
  std::memcpy(&values[0], &sparse_A.values[0], M * sizeof(values[0]));
  std::memcpy(&row_indices[0], &sparse_A.row_indices[0],
              M * sizeof(row_indices[0]));
  std::memcpy(&row_pointers[0], &sparse_A.row_pointers[0],
              (M + 1) * sizeof(row_pointers[0]));

  /* B */
  SparseMatrix<T, M, M, M> sparse_B = create_sparse(B);
  std::memcpy(&values[M], &sparse_B.values[0], M * sizeof(values[M]));
  std::memcpy(&row_indices[M], &sparse_B.row_indices[0],
              M * sizeof(row_indices[M]));

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

  std::vector<T> values(M + V);
  std::vector<std::size_t> row_indices(M + V);
  std::vector<std::size_t> row_pointers(M + P + 1);

  /* A */
  SparseMatrix<T, M, M, M> sparse_A = create_sparse(A);
  std::memcpy(&values[0], &sparse_A.values[0], M * sizeof(values[0]));
  std::memcpy(&row_indices[0], &sparse_A.row_indices[0],
              M * sizeof(row_indices[0]));
  std::memcpy(&row_pointers[0], &sparse_A.row_pointers[0],
              (M + 1) * sizeof(row_pointers[0]));

  /* B */
  std::memcpy(&values[M], &B.values[0], V * sizeof(values[M]));
  std::memcpy(&row_indices[M], &B.row_indices[0], V * sizeof(row_indices[M]));

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

  std::vector<T> values(V + W);
  std::vector<std::size_t> row_indices(V + W);
  std::vector<std::size_t> row_pointers(M + P + 1);

  /* A */
  std::memcpy(&values[0], &A.values[0], V * sizeof(values[0]));
  std::memcpy(&row_indices[0], &A.row_indices[0], V * sizeof(row_indices[0]));
  std::memcpy(&row_pointers[0], &A.row_pointers[0],
              (M + 1) * sizeof(row_pointers[0]));

  /* B */
  std::memcpy(&values[V], &B.values[0], W * sizeof(values[V]));
  std::memcpy(&row_indices[V], &B.row_indices[0], W * sizeof(row_indices[V]));

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
    std::memcpy(&result(row)[0], &A(row)[0], M * sizeof(result(row)[0]));
  }

  std::size_t B_row = 0;
  for (std::size_t row = N; row < N + P; row++) {
    std::memcpy(&result(row)[0], &B(B_row)[0], M * sizeof(result(row)[0]));
    B_row++;
  }

  return result;
}

template <typename T, std::size_t M, std::size_t N>
SparseMatrix<T, M, (M + N), ((N + 1) * M)>
concatenate_horizontally(const Matrix<T, M, N> &A, const DiagMatrix<T, M> &B) {

  std::vector<T> values((N + 1) * M);
  std::vector<std::size_t> row_indices((N + 1) * M);
  std::vector<std::size_t> row_pointers(M + 1);

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

  std::vector<T> values((M * N) + V);
  std::vector<std::size_t> row_indices((M * N) + V);
  std::vector<std::size_t> row_pointers(M + 1);

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

  std::vector<T> values((N + 1) * M);
  std::vector<std::size_t> row_indices((N + 1) * M);
  std::vector<std::size_t> row_pointers(M + 1);

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

  std::vector<T> values(2 * M);
  std::vector<std::size_t> row_indices(2 * M);
  std::vector<std::size_t> row_pointers(M + 1);

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

  std::vector<T> values(M + V);
  std::vector<std::size_t> row_indices(M + V);
  std::vector<std::size_t> row_pointers(M + 1);

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

  std::vector<T> values(V + W);
  std::vector<std::size_t> row_indices(V + W);
  std::vector<std::size_t> row_pointers(M + 1);

  std::size_t value_count = 0;
  std::size_t sparse_value_count_A = 0;
  std::size_t sparse_col_count_A = 0;
  std::size_t sparse_value_count_B = 0;
  std::size_t sparse_col_count_B = 0;
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
    std::memcpy(&result(i)[0], &A(i)[0], M * sizeof(result(i)[0]));
    std::memcpy(&result(i)[M], &C(i)[0], K * sizeof(result(i)[M]));
  }

  std::size_t B_row = 0;
  for (std::size_t i = N; i < N + P; ++i) {
    std::memcpy(&result(i)[0], &B(B_row)[0], M * sizeof(result(i)[0]));
    std::memcpy(&result(i)[M], &D(B_row)[0], K * sizeof(result(i)[M]));
    B_row++;
  }

  return result;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_CONCATENATE_HPP
