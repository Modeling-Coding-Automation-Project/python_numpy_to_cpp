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

template <std::size_t M, std::size_t N, typename T>
inline auto make_DenseMatrixFull(const T &value) -> Matrix<DefDense, T, M, N> {

  return Matrix<DefDense, T, M, N>::full(value);
}

template <typename T, std::size_t M>
inline auto make_DiagMatrixZeros(void) -> Matrix<DefDiag, T, M> {

  Matrix<DefDiag, T, M> result;
  return result;
}

template <typename T, std::size_t M>
inline auto make_DiagMatrixIdentity(void) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>::identity();
}

template <std::size_t M, typename T>
inline auto make_DiagMatrixFull(const T &value) -> Matrix<DefDiag, T, M> {

  return Matrix<DefDiag, T, M>::full(value);
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

template <typename T, typename SparseAvailable>
inline auto make_SparseMatrixOnes(void)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  return Matrix<DefSparse, T, SparseAvailable::number_of_columns,
                SparseAvailable::column_size,
                SparseAvailable>::full(static_cast<T>(1));
}

template <typename SparseAvailable, typename T>
inline auto make_SparseMatrixFull(const T &value)
    -> Matrix<DefSparse, T, SparseAvailable::number_of_columns,
              SparseAvailable::column_size, SparseAvailable> {

  return Matrix<DefSparse, T, SparseAvailable::number_of_columns,
                SparseAvailable::column_size, SparseAvailable>::full(value);
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

template <typename T, std::size_t M, std::size_t N, typename... Args>
inline auto
make_SparseMatrixFromDenseMatrix(Matrix<DefDense, T, M, N> &dense_matrix)
    -> Matrix<DefSparse, T, M, N, DenseAvailable<M, N>> {

  return Matrix<DefSparse, T, M, N, DenseAvailable<M, N>>(
      create_compiled_sparse(dense_matrix.matrix));
}

/* Type */
template <typename T, std::size_t M, std::size_t N>
using DenseMatrix_Type = Matrix<DefDense, T, M, N>;

template <typename T, std::size_t M>
using DiagMatrix_Type = Matrix<DefDiag, T, M>;

template <typename T, typename SparseAvailable>
using SparseMatrix_Type =
    decltype(make_SparseMatrixZeros<T, SparseAvailable>());

template <typename T, std::size_t M, std::size_t N>
using SparseMatrixEmpty_Type = decltype(make_SparseMatrixEmpty<T, M, N>());

/* Get */
namespace GetDenseMatrixOperation {

template <typename T, std::size_t M, std::size_t N, std::size_t ROW,
          std::size_t COL_Index>
struct GetRow_Loop {
  static void compute(const DenseMatrix_Type<T, M, N> &matrix,
                      DenseMatrix_Type<T, M, 1> &result) {

    result.template set<COL_Index, 0>(matrix.template get<COL_Index, ROW>());
    GetRow_Loop<T, M, N, ROW, COL_Index - 1>::compute(matrix, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t ROW>
struct GetRow_Loop<T, M, N, ROW, 0> {
  static void compute(const DenseMatrix_Type<T, M, N> &matrix,
                      DenseMatrix_Type<T, M, 1> &result) {

    result.template set<0, 0>(matrix.template get<0, ROW>());
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t ROW>
using GetRow = GetRow_Loop<T, M, N, ROW, M - 1>;

} // namespace GetDenseMatrixOperation

template <std::size_t ROW, typename T, std::size_t M, std::size_t N>
inline auto get_row(const DenseMatrix_Type<T, M, N> &matrix)
    -> DenseMatrix_Type<T, M, 1> {

  DenseMatrix_Type<T, M, 1> result;

  GetDenseMatrixOperation::GetRow<T, M, N, ROW>::compute(matrix, result);

  return result;
}

namespace GetDiagMatrixOperation {

template <std::size_t M, std::size_t Index>
using DiagAvailableRow = SparseAvailableGetRow<M, DiagAvailable<M>, Index>;

template <typename T, std::size_t M, std::size_t Index>
using DiagAvailableRow_Type = SparseMatrix_Type<T, DiagAvailableRow<M, Index>>;

template <typename T, std::size_t M, std::size_t ROW, std::size_t COL_Index>
struct GetRow_Loop {
  static void compute(const DiagMatrix_Type<T, M> &matrix,
                      DiagAvailableRow_Type<T, M, ROW> &result) {

    result.template set<COL_Index, 0>(matrix.template get<COL_Index, ROW>());
    GetRow_Loop<T, M, ROW, COL_Index - 1>::compute(matrix, result);
  }
};

template <typename T, std::size_t M, std::size_t ROW>
struct GetRow_Loop<T, M, ROW, 0> {
  static void compute(const DiagMatrix_Type<T, M> &matrix,
                      DiagAvailableRow_Type<T, M, ROW> &result) {

    result.template set<0, 0>(matrix.template get<0, ROW>());
  }
};

template <typename T, std::size_t M, std::size_t ROW>
using GetRow = GetRow_Loop<T, M, ROW, M - 1>;

} // namespace GetDiagMatrixOperation

template <std::size_t ROW, typename T, std::size_t M>
inline auto get_row(const DiagMatrix_Type<T, M> &matrix)
    -> GetDiagMatrixOperation::DiagAvailableRow_Type<T, M, ROW> {

  GetDiagMatrixOperation::DiagAvailableRow_Type<T, M, ROW> result;

  GetDiagMatrixOperation::GetRow<T, M, ROW>::compute(matrix, result);

  return result;
}

namespace GetSparseMatrixOperation {

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t ROW, std::size_t COL_Index>
struct GetRow_Loop {
  static void compute(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                      DenseMatrix_Type<T, M, 1> &result) {

    result.template set<COL_Index, 0>(matrix.template get<COL_Index, ROW>());
    GetRow_Loop<T, M, N, SparseAvailable, ROW, COL_Index - 1>::compute(matrix,
                                                                       result);
  }
};

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t ROW>
struct GetRow_Loop<T, M, N, SparseAvailable, ROW, 0> {
  static void compute(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                      DenseMatrix_Type<T, M, 1> &result) {

    result.template set<0, 0>(matrix.template get<0, ROW>());
  }
};

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t ROW>
using GetRow = GetRow_Loop<T, M, N, SparseAvailable, ROW, M - 1>;

} // namespace GetSparseMatrixOperation

template <std::size_t ROW, typename T, std::size_t M, std::size_t N,
          typename SparseAvailable>
inline auto get_row(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> DenseMatrix_Type<T, M, 1> {

  DenseMatrix_Type<T, M, 1> result;

  GetSparseMatrixOperation::GetRow<T, M, N, SparseAvailable, ROW>::compute(
      matrix, result);

  return result;
}

/* Set */
namespace SetDenseMatrixOperation {

template <typename T, std::size_t M, std::size_t N, std::size_t ROW,
          std::size_t COL_Index>
struct SetRow_Loop {
  static void compute(DenseMatrix_Type<T, M, N> &matrix,
                      const DenseMatrix_Type<T, M, 1> &row_vector) {

    matrix.template set<COL_Index, ROW>(
        row_vector.template get<COL_Index, 0>());
    SetRow_Loop<T, M, N, ROW, COL_Index - 1>::compute(matrix, row_vector);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t ROW>
struct SetRow_Loop<T, M, N, ROW, 0> {
  static void compute(DenseMatrix_Type<T, M, N> &matrix,
                      const DenseMatrix_Type<T, M, 1> &row_vector) {

    matrix.template set<0, ROW>(row_vector.template get<0, 0>());
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t ROW>
using SetRow = SetRow_Loop<T, M, N, ROW, M - 1>;

} // namespace SetDenseMatrixOperation

template <std::size_t ROW, typename T, std::size_t M, std::size_t N>
inline void set_row(DenseMatrix_Type<T, M, N> &matrix,
                    const DenseMatrix_Type<T, M, 1> &row_vector) {

  SetDenseMatrixOperation::SetRow<T, M, N, ROW>::compute(matrix, row_vector);
}

namespace SetDiagMatrixOperation {

template <typename T, std::size_t M, std::size_t ROW, std::size_t COL_Index>
struct SetRow_Loop {
  static void compute(DiagMatrix_Type<T, M> &matrix,
                      const DenseMatrix_Type<T, M, 1> &row_vector) {

    matrix.template set<COL_Index, ROW>(
        row_vector.template get<COL_Index, 0>());
    SetRow_Loop<T, M, ROW, COL_Index - 1>::compute(matrix, row_vector);
  }
};

template <typename T, std::size_t M, std::size_t ROW>
struct SetRow_Loop<T, M, ROW, 0> {
  static void compute(DiagMatrix_Type<T, M> &matrix,
                      const DenseMatrix_Type<T, M, 1> &row_vector) {

    matrix.template set<0, ROW>(row_vector.template get<0, 0>());
  }
};

template <typename T, std::size_t M, std::size_t ROW>
using SetRow = SetRow_Loop<T, M, ROW, M - 1>;

} // namespace SetDiagMatrixOperation

template <std::size_t ROW, typename T, std::size_t M>
inline void set_row(DiagMatrix_Type<T, M> &matrix,
                    const DenseMatrix_Type<T, M, 1> &row_vector) {

  SetDiagMatrixOperation::SetRow<T, M, ROW>::compute(matrix, row_vector);
}

namespace SetSparseMatrixOperation {

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t ROW, std::size_t COL_Index>
struct SetRow_Loop {
  static void compute(Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                      const DenseMatrix_Type<T, M, 1> &row_vector) {

    matrix.template set<COL_Index, ROW>(
        row_vector.template get<COL_Index, 0>());
    SetRow_Loop<T, M, N, SparseAvailable, ROW, COL_Index - 1>::compute(
        matrix, row_vector);
  }
};

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t ROW>
struct SetRow_Loop<T, M, N, SparseAvailable, ROW, 0> {
  static void compute(Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                      const DenseMatrix_Type<T, M, 1> &row_vector) {

    matrix.template set<0, ROW>(row_vector.template get<0, 0>());
  }
};

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable,
          std::size_t ROW>
using SetRow = SetRow_Loop<T, M, N, SparseAvailable, ROW, M - 1>;

} // namespace SetSparseMatrixOperation

template <std::size_t ROW, typename T, std::size_t M, std::size_t N,
          typename SparseAvailable>
inline void set_row(Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                    const DenseMatrix_Type<T, M, 1> &row_vector) {

  SetSparseMatrixOperation::SetRow<T, M, N, SparseAvailable, ROW>::compute(
      matrix, row_vector);
}

} // namespace PythonNumpy

#endif // __PYTHON_NUMPY_BASE_SIMPLIFICATION_HPP__
