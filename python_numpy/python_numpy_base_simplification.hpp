#ifndef __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__
#define __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__

#include "python_numpy_base.hpp"
#include "python_numpy_complex.hpp"
#include "python_numpy_templates.hpp"

#include <initializer_list>

namespace PythonNumpy {

template <typename T, std::size_t M, std::size_t N>
inline auto make_DenseMatrixZeros(void) -> Matrix<DefDense, T, M, N> {

  Matrix<DefDense, T, M, N> result;
  return result;
}

template <typename T, std::size_t M, std::size_t N>
inline auto make_DenseMatrixOnes(void) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>::ones();
}

template <typename T, std::size_t M>
inline auto make_DiagMatrixIdentity(void) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>::identity();
}

template <typename T, std::size_t M, std::size_t N>
inline auto make_SparseMatrixEmpty(void)
    -> Matrix<DefSparse, T, M, N, SparseAvailableEmpty<M, N>> {

  return Matrix<DefSparse, T, M, N, SparseAvailableEmpty<M, N>>();
}

namespace MakeDenseMatrixOperation {

template <std::size_t ColumnCount, std::size_t RowCount,
          typename DenseMatrix_Type, typename T>
inline void assign_values(DenseMatrix_Type &matrix, T value_1) {

  static_assert(ColumnCount < DenseMatrix_Type::COLS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");
  static_assert(RowCount < DenseMatrix_Type::ROWS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");

  matrix.template set<ColumnCount, RowCount>(value_1);
}

template <std::size_t ColumnCount, std::size_t RowCount,
          typename DenseMatrix_Type, typename T, typename U, typename... Args>
inline void assign_values(DenseMatrix_Type &matrix, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(ColumnCount < DenseMatrix_Type::COLS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");
  static_assert(RowCount < DenseMatrix_Type::ROWS,
                "Number of arguments must be less than the number of elements "
                "of Dense Matrix.");

  matrix.template set<ColumnCount, RowCount>(value_1);

  assign_values<ColumnCount + 1 * (RowCount == (DenseMatrix_Type::ROWS - 1)),
                ((RowCount + 1) * (RowCount != (DenseMatrix_Type::ROWS - 1)))>(
      matrix, value_2, args...);
}

} // namespace MakeDenseMatrixOperation

template <std::size_t M, std::size_t N, typename T, typename... Args>
inline auto make_DenseMatrix(T value_1, Args... args)
    -> Matrix<DefDense, T, M, N> {

  Matrix<DefDense, T, M, N> result;

  MakeDenseMatrixOperation::assign_values<0, 0>(result, value_1, args...);

  return result;
}

namespace MakeDiagMatrixOperation {

template <std::size_t IndexCount, typename DiagMatrix_Type, typename T>
inline void assign_values(DiagMatrix_Type &matrix, T value_1) {

  static_assert(IndexCount < DiagMatrix_Type::COLS,
                "Number of arguments must be less than the number of columns.");

  matrix.template set<IndexCount, IndexCount>(value_1);
}

template <std::size_t IndexCount, typename DiagMatrix_Type, typename T,
          typename U, typename... Args>
inline void assign_values(DiagMatrix_Type &matrix, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");
  static_assert(IndexCount < DiagMatrix_Type::COLS,
                "Number of arguments must be less than the number of columns.");

  matrix.template set<IndexCount, IndexCount>(value_1);

  assign_values<IndexCount + 1>(matrix, value_2, args...);
}

} // namespace MakeDiagMatrixOperation

template <std::size_t M, typename T, typename... Args>
inline auto make_DiagMatrix(T value_1, Args... args) -> Matrix<DefDiag, T, M> {

  Matrix<DefDiag, T, M> result;

  MakeDiagMatrixOperation::assign_values<0>(result, value_1, args...);

  return result;
}

template <typename T, typename SparseAvailable>
inline auto make_SparseMatrixZeros(void)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  Matrix<DefSparse, T, SparseAvailable::number_of_columns,
         SparseAvailable::column_size, SparseAvailable>
      result;

  return result;
}

namespace MakeSparseMatrixOperation {

template <std::size_t IndexCount, typename SparseMatrix_Type, typename T>
inline void assign_values(SparseMatrix_Type &matrix, T value_1) {

  static_assert(IndexCount < SparseMatrix_Type::NumberOfValues,
                "Number of arguments must be the same or less than the number "
                "of elements of Sparse Matrix.");

  matrix.template set<IndexCount>(value_1);
}

template <std::size_t IndexCount, typename SparseMatrix_Type, typename T,
          typename U, typename... Args>
inline void assign_values(SparseMatrix_Type &matrix, T value_1, U value_2,
                          Args... args) {

  static_assert(std::is_same<T, U>::value, "Arguments must be the same type.");

  static_assert(IndexCount < SparseMatrix_Type::NumberOfValues,
                "Number of arguments must be the same or less than the number "
                "of elements of Sparse Matrix.");

  matrix.template set<IndexCount>(value_1);

  assign_values<IndexCount + 1>(matrix, value_2, args...);
}

} // namespace MakeSparseMatrixOperation

template <typename SparseAvailable, typename T, typename... Args>
inline auto make_SparseMatrix(T value_1, Args... args)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  Matrix<DefSparse, T, SparseAvailable::number_of_columns,
         SparseAvailable::column_size, SparseAvailable>
      result;

  MakeSparseMatrixOperation::assign_values<0>(result, value_1, args...);

  return result;
}

namespace MakeSparseMatrixOperation {

// when J_idx < N
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t I, std::size_t J_idx>
struct SparseMatrixSetDenseMatrixColumn {
  static void
  compute(Matrix<DefSparse, T, M, N, SparseAvailable> &sparse_matrix,
          const Matrix<DefDense, T, M, N> &dense_matrix) {

    sparse_matrix.template set<I, J_idx>(dense_matrix.template get<I, J_idx>());

    SparseMatrixSetDenseMatrixColumn<T, M, N, SparseAvailable, I,
                                     J_idx - 1>::compute(sparse_matrix,
                                                         dense_matrix);
  }
};

// column recursion termination
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t I>
struct SparseMatrixSetDenseMatrixColumn<T, M, N, SparseAvailable, I, 0> {
  static void
  compute(Matrix<DefSparse, T, M, N, SparseAvailable> &sparse_matrix,
          const Matrix<DefDense, T, M, N> &dense_matrix) {

    sparse_matrix.template set<I, 0>(dense_matrix.template get<I, 0>());
  }
};

// when I_idx < M
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t I_idx>
struct SparseMatrixSetDenseMatrixRow {
  static void
  compute(Matrix<DefSparse, T, M, N, SparseAvailable> &sparse_matrix,
          const Matrix<DefDense, T, M, N> &dense_matrix) {
    SparseMatrixSetDenseMatrixColumn<T, M, N, SparseAvailable, I_idx,
                                     N - 1>::compute(sparse_matrix,
                                                     dense_matrix);
    SparseMatrixSetDenseMatrixRow<T, M, N, SparseAvailable, I_idx - 1>::compute(
        sparse_matrix, dense_matrix);
  }
};

// row recursion termination
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
struct SparseMatrixSetDenseMatrixRow<T, M, N, SparseAvailable, 0> {
  static void
  compute(Matrix<DefSparse, T, M, N, SparseAvailable> &sparse_matrix,
          const Matrix<DefDense, T, M, N> &dense_matrix) {
    SparseMatrixSetDenseMatrixColumn<T, M, N, SparseAvailable, 0,
                                     N - 1>::compute(sparse_matrix,
                                                     dense_matrix);
  }
};

} // namespace MakeSparseMatrixOperation

template <typename T, std::size_t M, std::size_t N, typename... Args>
inline auto
make_SparseMatrixFromDenseMatrix(Matrix<DefDense, T, M, N> &dense_matrix)
    -> Matrix<DefSparse, T, M, N, DenseAvailable<M, N>> {

  Matrix<DefSparse, T, M, N, DenseAvailable<M, N>> result;

  MakeSparseMatrixOperation::SparseMatrixSetDenseMatrixRow<
      T, M, N, DenseAvailable<M, N>, M - 1>::compute(result, dense_matrix);

  return result;
}

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__
