#ifndef BASE_MATRIX_TRIANGULAR_SPARSE_HPP
#define BASE_MATRIX_TRIANGULAR_SPARSE_HPP

#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_sparse.hpp"
#include "base_matrix_vector.hpp"
#include <cstddef>
#include <vector>

namespace Base {
namespace Matrix {

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
      for (std::size_t j = i; j < N; j++) {
        row_indices[consecutive_index] = j;

        consecutive_index++;
        row_pointers[i + 1] = consecutive_index;
      }
    }

    return SparseMatrix<T, M, N,
                        CalculateTriangularSize<M, ((N < M) ? N : M)>::value>(
        values, row_indices, row_pointers);
  }

  static auto create_upper(const Matrix<T, M, N> &A)
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
      for (std::size_t j = i; j < N; j++) {
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

  static void set_values_upper(
      SparseMatrix<T, M, N,
                   CalculateTriangularSize<M, ((N < M) ? N : M)>::value> &A,
      const Matrix<T, M, N> &B) {
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
