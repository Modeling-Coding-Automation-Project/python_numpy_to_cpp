/**
 * @file base_matrix_gmres.hpp
 * @brief Fixed-capacity restarted GMRES with real or complex arithmetic.
 */
#ifndef BASE_MATRIX_GMRES_HPP_
#define BASE_MATRIX_GMRES_HPP_

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_numerical.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace Base {
namespace Matrix {

// Default work budget and convergence thresholds for GMRES.
constexpr std::size_t DEFAULT_CYCLES_GMRES = 1;
constexpr std::size_t DEFAULT_ORTHOGONALIZATION_PASSES_GMRES = 1;
constexpr double DEFAULT_DIVISION_MIN_GMRES = 1.0e-10;
// With both tolerances zero, only an exact zero residual meets convergence.
constexpr double DEFAULT_RELATIVE_TOLERANCE_GMRES = 0.0;
constexpr double DEFAULT_ABSOLUTE_TOLERANCE_GMRES = 0.0;
// Arnoldi breakdown limit: relative tolerance * ||A*D^-1*q||.
// Keep this independent of the requested residual tolerance.
constexpr double GMRES_BREAKDOWN_RELATIVE_TOLERANCE = 8.0e-12;
static_assert(GMRES_BREAKDOWN_RELATIVE_TOLERANCE >= 0.0,
              "The GMRES breakdown tolerance must be nonnegative.");

namespace GmresOperation {

/* Vector initialization and projection onto the leading subspace. */
namespace VectorSubspace {

template <typename T, std::size_t N, std::size_t Index> struct Loop {
  static inline void clear(Vector<T, N> &output) {
    output[Index] = T();
    Loop<T, N, Index + 1>::clear(output);
  }

  static inline void copy_active(const Vector<T, N> &input,
                                 Vector<T, N> &output,
                                 std::size_t active_size) {
    if (Index < active_size) {
      output[Index] = input[Index];
    }
    Loop<T, N, Index + 1>::copy_active(input, output, active_size);
  }

  static inline void clear_inactive(Vector<T, N> &output,
                                    std::size_t active_size) {
    if (Index >= active_size) {
      output[Index] = T();
    }
    Loop<T, N, Index + 1>::clear_inactive(output, active_size);
  }
};

// Termination condition: Index == N.
template <typename T, std::size_t N> struct Loop<T, N, N> {
  static inline void clear(Vector<T, N> &) {}
  static inline void copy_active(const Vector<T, N> &, Vector<T, N> &,
                                 std::size_t) {}
  static inline void clear_inactive(Vector<T, N> &, std::size_t) {}
};

} // namespace VectorSubspace

/* Column-major multiplication; preserve the ascending accumulation order. */
namespace DenseMultiply {

template <typename T, std::size_t N, std::size_t ColumnIndex,
          std::size_t RowIndex, bool FullSize>
struct Row {
  static inline void compute(const Matrix<T, N, N> &matrix,
                             const Vector<T, N> &input, Vector<T, N> &output,
                             std::size_t active_size) {
    if (FullSize || (RowIndex < active_size && ColumnIndex < active_size)) {
      output[RowIndex] += matrix(RowIndex, ColumnIndex) * input[ColumnIndex];
    }
    Row<T, N, ColumnIndex, RowIndex + 1, FullSize>::compute(
        matrix, input, output, active_size);
  }
};

template <typename T, std::size_t N, std::size_t ColumnIndex, bool FullSize>
struct Row<T, N, ColumnIndex, N, FullSize> {
  static inline void compute(const Matrix<T, N, N> &, const Vector<T, N> &,
                             Vector<T, N> &, std::size_t) {}
};

template <typename T, std::size_t N, std::size_t ColumnIndex, bool FullSize>
struct Column {
  static inline void compute(const Matrix<T, N, N> &matrix,
                             const Vector<T, N> &input, Vector<T, N> &output,
                             std::size_t active_size) {
    Row<T, N, ColumnIndex, 0, FullSize>::compute(matrix, input, output,
                                                 active_size);
    Column<T, N, ColumnIndex + 1, FullSize>::compute(matrix, input, output,
                                                     active_size);
  }
};

template <typename T, std::size_t N, bool FullSize>
struct Column<T, N, N, FullSize> {
  static inline void compute(const Matrix<T, N, N> &, const Vector<T, N> &,
                             Vector<T, N> &, std::size_t) {}
};

template <typename T, std::size_t N>
inline void compute(const Matrix<T, N, N> &matrix, const Vector<T, N> &input,
                    Vector<T, N> &output, std::size_t active_size) {
  VectorSubspace::Loop<T, N, 0>::clear(output);
  if (active_size == N) {
    Column<T, N, 0, true>::compute(matrix, input, output, active_size);
  } else {
    Column<T, N, 0, false>::compute(matrix, input, output, active_size);
  }
}

} // namespace DenseMultiply

/* A^T times a vector; used for both the normal operator and its RHS. */
namespace DenseTransposeMultiply {

template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex,
          std::size_t RowIndex>
struct Row {
  static inline void compute(const Matrix<T, M, N> &matrix,
                             const Vector<T, M> &input, Vector<T, N> &output) {
    output[ColumnIndex] += matrix(RowIndex, ColumnIndex) * input[RowIndex];
    Row<T, M, N, ColumnIndex, RowIndex + 1>::compute(matrix, input, output);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex>
struct Row<T, M, N, ColumnIndex, M> {
  static inline void compute(const Matrix<T, M, N> &, const Vector<T, M> &,
                             Vector<T, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex>
struct Column {
  static inline void compute(const Matrix<T, M, N> &matrix,
                             const Vector<T, M> &input, Vector<T, N> &output) {
    output[ColumnIndex] = T();
    Row<T, M, N, ColumnIndex, 0>::compute(matrix, input, output);
    Column<T, M, N, ColumnIndex + 1>::compute(matrix, input, output);
  }
};

template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, N> {
  static inline void compute(const Matrix<T, M, N> &, const Vector<T, M> &,
                             Vector<T, N> &) {}
};

} // namespace DenseTransposeMultiply

/* Explicit dense normal equations: fill one Gram column and RHS entry. */
namespace DenseNormalEquations {

template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex,
          std::size_t GramRowIndex, std::size_t RowIndex>
struct Product {
  static inline void compute(const Matrix<T, M, N> &matrix,
                             Matrix<T, N, N> &gram) {
    gram(GramRowIndex, ColumnIndex) +=
        matrix(RowIndex, GramRowIndex) * matrix(RowIndex, ColumnIndex);
    Product<T, M, N, ColumnIndex, GramRowIndex, RowIndex + 1>::compute(matrix,
                                                                       gram);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex,
          std::size_t GramRowIndex>
struct Product<T, M, N, ColumnIndex, GramRowIndex, M> {
  static inline void compute(const Matrix<T, M, N> &, Matrix<T, N, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex,
          std::size_t GramRowIndex>
struct Row {
  static inline void compute(const Matrix<T, M, N> &matrix,
                             Matrix<T, N, N> &gram) {
    gram(GramRowIndex, ColumnIndex) = T();
    Product<T, M, N, ColumnIndex, GramRowIndex, 0>::compute(matrix, gram);
    Row<T, M, N, ColumnIndex, GramRowIndex + 1>::compute(matrix, gram);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex>
struct Row<T, M, N, ColumnIndex, N> {
  static inline void compute(const Matrix<T, M, N> &, Matrix<T, N, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex>
struct Column {
  static inline void compute(const Matrix<T, M, N> &matrix,
                             const Vector<T, M> &right_hand_side,
                             Matrix<T, N, N> &gram, Vector<T, N> &normal_rhs) {
    Row<T, M, N, ColumnIndex, 0>::compute(matrix, gram);
    normal_rhs[ColumnIndex] = T();
    DenseTransposeMultiply::Row<T, M, N, ColumnIndex, 0>::compute(
        matrix, right_hand_side, normal_rhs);
    Column<T, M, N, ColumnIndex + 1>::compute(matrix, right_hand_side, gram,
                                              normal_rhs);
  }
};

template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, N> {
  static inline void compute(const Matrix<T, M, N> &, const Vector<T, M> &,
                             Matrix<T, N, N> &, Vector<T, N> &) {}
};

} // namespace DenseNormalEquations

/* Clear the Gram matrix and normal RHS before accumulating sparse rows. */
namespace ZeroNormalEquations {

template <typename T, std::size_t N, std::size_t ColumnIndex,
          std::size_t RowIndex>
struct Row {
  static inline void compute(Matrix<T, N, N> &gram) {
    gram(RowIndex, ColumnIndex) = T();
    Row<T, N, ColumnIndex, RowIndex + 1>::compute(gram);
  }
};

template <typename T, std::size_t N, std::size_t ColumnIndex>
struct Row<T, N, ColumnIndex, N> {
  static inline void compute(Matrix<T, N, N> &) {}
};

template <typename T, std::size_t N, std::size_t ColumnIndex> struct Column {
  static inline void compute(Matrix<T, N, N> &gram, Vector<T, N> &normal_rhs) {
    normal_rhs[ColumnIndex] = T();
    Row<T, N, ColumnIndex, 0>::compute(gram);
    Column<T, N, ColumnIndex + 1>::compute(gram, normal_rhs);
  }
};

template <typename T, std::size_t N> struct Column<T, N, N> {
  static inline void compute(Matrix<T, N, N> &, Vector<T, N> &) {}
};

} // namespace ZeroNormalEquations

/* Direct evaluation of A^T(b-Ax), keeping each subtraction in row order. */
namespace DenseNormalResidual {

template <typename T, std::size_t M, std::size_t N, std::size_t RowIndex,
          std::size_t ColumnIndex>
struct Column {
  static inline void subtract(const Matrix<T, M, N> &matrix,
                              const Vector<T, N> &solution, T &residual) {
    residual -= matrix(RowIndex, ColumnIndex) * solution[ColumnIndex];
    Column<T, M, N, RowIndex, ColumnIndex + 1>::subtract(matrix, solution,
                                                         residual);
  }

  static inline void accumulate(const Matrix<T, M, N> &matrix,
                                const T &residual, Vector<T, N> &stationarity) {
    stationarity[ColumnIndex] += matrix(RowIndex, ColumnIndex) * residual;
    Column<T, M, N, RowIndex, ColumnIndex + 1>::accumulate(matrix, residual,
                                                           stationarity);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t RowIndex>
struct Column<T, M, N, RowIndex, N> {
  static inline void subtract(const Matrix<T, M, N> &, const Vector<T, N> &,
                              T &) {}
  static inline void accumulate(const Matrix<T, M, N> &, const T &,
                                Vector<T, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t RowIndex>
struct Row {
  static inline void compute(const Matrix<T, M, N> &matrix,
                             const Vector<T, M> &right_hand_side,
                             const Vector<T, N> &solution,
                             Vector<T, N> &stationarity) {
    T residual = right_hand_side[RowIndex];
    Column<T, M, N, RowIndex, 0>::subtract(matrix, solution, residual);
    Column<T, M, N, RowIndex, 0>::accumulate(matrix, residual, stationarity);
    Row<T, M, N, RowIndex + 1>::compute(matrix, right_hand_side, solution,
                                        stationarity);
  }
};

template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, M> {
  static inline void compute(const Matrix<T, M, N> &, const Vector<T, M> &,
                             const Vector<T, N> &, Vector<T, N> &) {}
};

} // namespace DenseNormalResidual

/* First product A*x for the dense normal operator. */
namespace DenseNormalProduct {

template <typename T, std::size_t M, std::size_t N, std::size_t RowIndex,
          std::size_t ColumnIndex>
struct Column {
  static inline void compute(const Matrix<T, M, N> &matrix,
                             const Vector<T, N> &input,
                             Vector<T, M> &intermediate) {
    intermediate[RowIndex] +=
        matrix(RowIndex, ColumnIndex) * input[ColumnIndex];
    Column<T, M, N, RowIndex, ColumnIndex + 1>::compute(matrix, input,
                                                        intermediate);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t RowIndex>
struct Column<T, M, N, RowIndex, N> {
  static inline void compute(const Matrix<T, M, N> &, const Vector<T, N> &,
                             Vector<T, M> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t RowIndex>
struct Row {
  static inline void compute(const Matrix<T, M, N> &matrix,
                             const Vector<T, N> &input,
                             Vector<T, M> &intermediate) {
    Column<T, M, N, RowIndex, 0>::compute(matrix, input, intermediate);
    Row<T, M, N, RowIndex + 1>::compute(matrix, input, intermediate);
  }
};

template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, M> {
  static inline void compute(const Matrix<T, M, N> &, const Vector<T, N> &,
                             Vector<T, M> &) {}
};

} // namespace DenseNormalProduct

/* Sparse matrix-vector multiplication in the leading active subspace. */
namespace SparseMultiply {

template <typename T, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex, std::size_t Start,
          std::size_t End>
struct Loop {
  static inline void
  compute(const CompiledSparseMatrix<T, N, N, CSRIndicesType, CSRPointersType>
              &matrix,
          const Vector<T, N> &input, Vector<T, N> &output,
          std::size_t active_size) {
    if (RowIndex < active_size && CSRIndicesType::list[Start] < active_size) {
      output[RowIndex] +=
          matrix.values[Start] * input[CSRIndicesType::list[Start]];
    }
    Loop<T, N, CSRIndicesType, CSRPointersType, RowIndex, Start + 1,
         End>::compute(matrix, input, output, active_size);
  }
};

// Empty rows and the end of each row require no element access.
template <typename T, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex, std::size_t End>
struct Loop<T, N, CSRIndicesType, CSRPointersType, RowIndex, End, End> {
  static inline void compute(
      const CompiledSparseMatrix<T, N, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, N> &, Vector<T, N> &, std::size_t) {}
};

template <typename T, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex>
struct Row {
  static inline void
  compute(const CompiledSparseMatrix<T, N, N, CSRIndicesType, CSRPointersType>
              &matrix,
          const Vector<T, N> &input, Vector<T, N> &output,
          std::size_t active_size) {
    output[RowIndex] = T();
    Loop<T, N, CSRIndicesType, CSRPointersType, RowIndex,
         CSRPointersType::list[RowIndex],
         CSRPointersType::list[RowIndex + 1]>::compute(matrix, input, output,
                                                       active_size);
    Row<T, N, CSRIndicesType, CSRPointersType, RowIndex + 1>::compute(
        matrix, input, output, active_size);
  }
};

template <typename T, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
struct Row<T, N, CSRIndicesType, CSRPointersType, N> {
  static inline void compute(
      const CompiledSparseMatrix<T, N, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, N> &, Vector<T, N> &, std::size_t) {}
};

} // namespace SparseMultiply

/* Add each sparse row's contribution to A^T A and A^T b. */
namespace SparseNormalEquations {

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t FirstIndex, std::size_t Start,
          std::size_t End>
struct SecondEntry {
  static inline void
  compute(const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>
              &matrix,
          Matrix<T, N, N> &gram) {
    gram(CSRIndicesType::list[FirstIndex], CSRIndicesType::list[Start]) +=
        matrix.values[FirstIndex] * matrix.values[Start];
    SecondEntry<T, M, N, CSRIndicesType, CSRPointersType, FirstIndex, Start + 1,
                End>::compute(matrix, gram);
  }
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t FirstIndex, std::size_t End>
struct SecondEntry<T, M, N, CSRIndicesType, CSRPointersType, FirstIndex, End,
                   End> {
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      Matrix<T, N, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex, std::size_t Start,
          std::size_t End>
struct FirstEntry {
  static inline void
  compute(const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>
              &matrix,
          const Vector<T, M> &right_hand_side, Matrix<T, N, N> &gram,
          Vector<T, N> &normal_rhs) {
    normal_rhs[CSRIndicesType::list[Start]] +=
        matrix.values[Start] * right_hand_side[RowIndex];
    SecondEntry<T, M, N, CSRIndicesType, CSRPointersType, Start,
                CSRPointersType::list[RowIndex], End>::compute(matrix, gram);
    FirstEntry<T, M, N, CSRIndicesType, CSRPointersType, RowIndex, Start + 1,
               End>::compute(matrix, right_hand_side, gram, normal_rhs);
  }
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex, std::size_t End>
struct FirstEntry<T, M, N, CSRIndicesType, CSRPointersType, RowIndex, End,
                  End> {
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, M> &, Matrix<T, N, N> &, Vector<T, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex>
struct Row {
  static inline void
  compute(const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>
              &matrix,
          const Vector<T, M> &right_hand_side, Matrix<T, N, N> &gram,
          Vector<T, N> &normal_rhs) {
    FirstEntry<T, M, N, CSRIndicesType, CSRPointersType, RowIndex,
               CSRPointersType::list[RowIndex],
               CSRPointersType::list[RowIndex + 1]>::compute(matrix,
                                                             right_hand_side,
                                                             gram, normal_rhs);
    Row<T, M, N, CSRIndicesType, CSRPointersType, RowIndex + 1>::compute(
        matrix, right_hand_side, gram, normal_rhs);
  }
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
struct Row<T, M, N, CSRIndicesType, CSRPointersType, M> {
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, M> &, Matrix<T, N, N> &, Vector<T, N> &) {}
};

} // namespace SparseNormalEquations

/* Scatter a sparse row times a scalar, preserving stored entry order. */
namespace SparseTransposeMultiply {

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t Start, std::size_t End>
struct Loop {
  static inline void
  compute(const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>
              &matrix,
          const T &input, Vector<T, N> &output) {
    output[CSRIndicesType::list[Start]] += matrix.values[Start] * input;
    Loop<T, M, N, CSRIndicesType, CSRPointersType, Start + 1, End>::compute(
        matrix, input, output);
  }
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t End>
struct Loop<T, M, N, CSRIndicesType, CSRPointersType, End, End> {
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const T &, Vector<T, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex>
struct Row {
  static inline void
  compute(const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>
              &matrix,
          const Vector<T, M> &input, Vector<T, N> &output) {
    Loop<T, M, N, CSRIndicesType, CSRPointersType,
         CSRPointersType::list[RowIndex],
         CSRPointersType::list[RowIndex + 1]>::compute(matrix, input[RowIndex],
                                                       output);
    Row<T, M, N, CSRIndicesType, CSRPointersType, RowIndex + 1>::compute(
        matrix, input, output);
  }
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
struct Row<T, M, N, CSRIndicesType, CSRPointersType, M> {
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, M> &, Vector<T, N> &) {}
};

} // namespace SparseTransposeMultiply

/* Form b-Ax one row at a time, then accumulate A^T(b-Ax). */
namespace SparseNormalResidual {

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t Start, std::size_t End>
struct Loop {
  static inline void
  compute(const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>
              &matrix,
          const Vector<T, N> &solution, T &residual) {
    residual -= matrix.values[Start] * solution[CSRIndicesType::list[Start]];
    Loop<T, M, N, CSRIndicesType, CSRPointersType, Start + 1, End>::compute(
        matrix, solution, residual);
  }
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t End>
struct Loop<T, M, N, CSRIndicesType, CSRPointersType, End, End> {
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, N> &, T &) {}
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex>
struct Row {
  static inline void
  compute(const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>
              &matrix,
          const Vector<T, M> &right_hand_side, const Vector<T, N> &solution,
          Vector<T, N> &stationarity) {
    T residual = right_hand_side[RowIndex];
    Loop<T, M, N, CSRIndicesType, CSRPointersType,
         CSRPointersType::list[RowIndex],
         CSRPointersType::list[RowIndex + 1]>::compute(matrix, solution,
                                                       residual);
    SparseTransposeMultiply::Loop<
        T, M, N, CSRIndicesType, CSRPointersType,
        CSRPointersType::list[RowIndex],
        CSRPointersType::list[RowIndex + 1]>::compute(matrix, residual,
                                                      stationarity);
    Row<T, M, N, CSRIndicesType, CSRPointersType, RowIndex + 1>::compute(
        matrix, right_hand_side, solution, stationarity);
  }
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
struct Row<T, M, N, CSRIndicesType, CSRPointersType, M> {
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, M> &, const Vector<T, N> &, Vector<T, N> &) {}
};

} // namespace SparseNormalResidual

/* Accumulate Ax into the zero-initialized normal-operator intermediate. */
namespace SparseNormalProduct {

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t Start, std::size_t End>
struct Loop {
  static inline void
  compute(const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>
              &matrix,
          const Vector<T, N> &input, T &output) {
    output += matrix.values[Start] * input[CSRIndicesType::list[Start]];
    Loop<T, M, N, CSRIndicesType, CSRPointersType, Start + 1, End>::compute(
        matrix, input, output);
  }
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t End>
struct Loop<T, M, N, CSRIndicesType, CSRPointersType, End, End> {
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, N> &, T &) {}
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex>
struct Row {
  static inline void
  compute(const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>
              &matrix,
          const Vector<T, N> &input, Vector<T, M> &output) {
    Loop<T, M, N, CSRIndicesType, CSRPointersType,
         CSRPointersType::list[RowIndex],
         CSRPointersType::list[RowIndex + 1]>::compute(matrix, input,
                                                       output[RowIndex]);
    Row<T, M, N, CSRIndicesType, CSRPointersType, RowIndex + 1>::compute(
        matrix, input, output);
  }
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
struct Row<T, M, N, CSRIndicesType, CSRPointersType, M> {
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, N> &, Vector<T, M> &) {}
};

} // namespace SparseNormalProduct

template <typename T, std::size_t N>
inline void multiply(const Matrix<T, N, N> &matrix, const Vector<T, N> &input,
                     Vector<T, N> &output, std::size_t active_size) {
#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < N; ++i) {
    output[i] = T();
  }
  if (active_size == N) {
    // Matrix storage is column-major; visit each contiguous column once.
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t i = 0; i < N; ++i) {
        output[i] += matrix(i, j) * input[j];
      }
    }
  } else {
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t i = 0; i < N; ++i) {
        if (i < active_size && j < active_size) {
          output[i] += matrix(i, j) * input[j];
        }
      }
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DenseMultiply::compute(matrix, input, output, active_size);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

template <typename T, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
inline void multiply(const CompiledSparseMatrix<T, N, N, CSRIndicesType,
                                                CSRPointersType> &matrix,
                     const Vector<T, N> &input, Vector<T, N> &output,
                     std::size_t active_size) {
#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < N; ++i) {
    output[i] = T();
    // An empty compile-time pattern has no index or value to access.
    if (CSRIndicesType::size > 0) {
      for (std::size_t j = CSRPointersType::list[i];
           j < CSRPointersType::list[i + 1]; ++j) {
        if (i < active_size && CSRIndicesType::list[j] < active_size) {
          output[i] += matrix.values[j] * input[CSRIndicesType::list[j]];
        }
      }
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  SparseMultiply::Row<T, N, CSRIndicesType, CSRPointersType, 0>::compute(
      matrix, input, output, active_size);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

template <typename MatrixType, typename T, std::size_t N>
struct MatrixOperator {
  const MatrixType &matrix;
  std::size_t active_size;
  inline void operator()(const Vector<T, N> &input,
                         Vector<T, N> &output) const {
    multiply(this->matrix, input, output, this->active_size);
  }
};

/** Keep a public callback inside the requested leading subspace. */
template <typename Operator, typename T, std::size_t N>
inline void apply_operator(const Operator &operation, const Vector<T, N> &input,
                           Vector<T, N> &output, std::size_t active_size) {
  if (active_size == N) {
    operation(input, output);
  } else {
    Vector<T, N> active_input;
#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    for (std::size_t i = 0; i < N; ++i) {
      if (i < active_size) {
        active_input[i] = input[i];
      }
    }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    VectorSubspace::Loop<T, N, 0>::copy_active(input, active_input,
                                               active_size);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
    operation(active_input, output);
#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    for (std::size_t i = 0; i < N; ++i) {
      if (i >= active_size) {
        output[i] = T();
      }
    }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    VectorSubspace::Loop<T, N, 0>::clear_inactive(output, active_size);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
  }
}

template <typename T> struct IdentityPreconditioner {
  inline T operator[](std::size_t) const { return static_cast<T>(1); }
};

/** Explicit normal equations are useful for tall inputs or repeated RHS. */
template <typename T, std::size_t M, std::size_t N>
inline void normal_equations(const Matrix<T, M, N> &matrix,
                             const Vector<T, M> &right_hand_side,
                             Matrix<T, N, N> &gram, Vector<T, N> &normal_rhs) {
#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i < N; ++i) {
      gram(i, j) = T();
      for (std::size_t row = 0; row < M; ++row) {
        gram(i, j) += matrix(row, i) * matrix(row, j);
      }
    }
    normal_rhs[j] = T();
    for (std::size_t row = 0; row < M; ++row) {
      normal_rhs[j] += matrix(row, j) * right_hand_side[row];
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DenseNormalEquations::Column<T, M, N, 0>::compute(matrix, right_hand_side,
                                                    gram, normal_rhs);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
inline void
normal_equations(const CompiledSparseMatrix<T, M, N, CSRIndicesType,
                                            CSRPointersType> &matrix,
                 const Vector<T, M> &right_hand_side, Matrix<T, N, N> &gram,
                 Vector<T, N> &normal_rhs) {
#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t j = 0; j < N; ++j) {
    normal_rhs[j] = T();
    for (std::size_t i = 0; i < N; ++i) {
      gram(i, j) = T();
    }
  }
  // Skip entry access when the compile-time CSR pattern is empty.
  if (CSRIndicesType::size > 0) {
    for (std::size_t row = 0; row < M; ++row) {
      for (std::size_t first = CSRPointersType::list[row];
           first < CSRPointersType::list[row + 1]; ++first) {
        normal_rhs[CSRIndicesType::list[first]] +=
            matrix.values[first] * right_hand_side[row];
        for (std::size_t second = CSRPointersType::list[row];
             second < CSRPointersType::list[row + 1]; ++second) {
          gram(CSRIndicesType::list[first], CSRIndicesType::list[second]) +=
              matrix.values[first] * matrix.values[second];
        }
      }
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  ZeroNormalEquations::Column<T, N, 0>::compute(gram, normal_rhs);
  SparseNormalEquations::Row<T, M, N, CSRIndicesType, CSRPointersType,
                             0>::compute(matrix, right_hand_side, gram,
                                         normal_rhs);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
}

/** Evaluate A^T(b-Ax) directly, including for an explicitly formed Gram. */
template <typename T, std::size_t M, std::size_t N>
inline T normal_residual(const Matrix<T, M, N> &matrix,
                         const Vector<T, M> &right_hand_side,
                         const Vector<T, N> &solution) {
  Vector<T, N> stationarity;
#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t row = 0; row < M; ++row) {
    T residual = right_hand_side[row];
    for (std::size_t j = 0; j < N; ++j) {
      residual -= matrix(row, j) * solution[j];
    }
    for (std::size_t j = 0; j < N; ++j) {
      stationarity[j] += matrix(row, j) * residual;
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DenseNormalResidual::Row<T, M, N, 0>::compute(matrix, right_hand_side,
                                                solution, stationarity);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
  return Numerical::norm(stationarity);
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
inline T normal_residual(const CompiledSparseMatrix<T, M, N, CSRIndicesType,
                                                    CSRPointersType> &matrix,
                         const Vector<T, M> &right_hand_side,
                         const Vector<T, N> &solution) {
  Vector<T, N> stationarity;
#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  // Skip entry access when the compile-time CSR pattern is empty.
  if (CSRIndicesType::size > 0) {
    for (std::size_t row = 0; row < M; ++row) {
      T residual = right_hand_side[row];
      for (std::size_t entry = CSRPointersType::list[row];
           entry < CSRPointersType::list[row + 1]; ++entry) {
        residual -=
            matrix.values[entry] * solution[CSRIndicesType::list[entry]];
      }
      for (std::size_t entry = CSRPointersType::list[row];
           entry < CSRPointersType::list[row + 1]; ++entry) {
        stationarity[CSRIndicesType::list[entry]] +=
            matrix.values[entry] * residual;
      }
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  SparseNormalResidual::Row<T, M, N, CSRIndicesType, CSRPointersType,
                            0>::compute(matrix, right_hand_side, solution,
                                        stationarity);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
  return Numerical::norm(stationarity);
}

/** The normal operator avoids storing A^T A; it is still GMRES on that system.
 */
template <typename MatrixType, typename T, std::size_t M, std::size_t N>
struct NormalOperator;

template <typename T, std::size_t M, std::size_t N>
struct NormalOperator<Matrix<T, M, N>, T, M, N> {
  const Matrix<T, M, N> &matrix;
  inline void transpose_multiply(const Vector<T, M> &input,
                                 Vector<T, N> &output) const {
#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    for (std::size_t j = 0; j < N; ++j) {
      output[j] = T();
      for (std::size_t i = 0; i < M; ++i) {
        output[j] += this->matrix(i, j) * input[i];
      }
    }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    DenseTransposeMultiply::Column<T, M, N, 0>::compute(this->matrix, input,
                                                        output);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
  }
  inline void operator()(const Vector<T, N> &input,
                         Vector<T, N> &output) const {
    Vector<T, M> intermediate;
#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        intermediate[i] += this->matrix(i, j) * input[j];
      }
    }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    DenseNormalProduct::Row<T, M, N, 0>::compute(this->matrix, input,
                                                 intermediate);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
    this->transpose_multiply(intermediate, output);
  }
};

template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
struct NormalOperator<
    CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>, T, M, N> {
  const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &matrix;
  inline void transpose_multiply(const Vector<T, M> &input,
                                 Vector<T, N> &output) const {
#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    for (std::size_t j = 0; j < N; ++j) {
      output[j] = T();
    }
    // Skip entry access when the compile-time CSR pattern is empty.
    if (CSRIndicesType::size > 0) {
      for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = CSRPointersType::list[i];
             j < CSRPointersType::list[i + 1]; ++j) {
          output[CSRIndicesType::list[j]] += this->matrix.values[j] * input[i];
        }
      }
    }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    VectorSubspace::Loop<T, N, 0>::clear(output);
    SparseTransposeMultiply::Row<T, M, N, CSRIndicesType, CSRPointersType,
                                 0>::compute(this->matrix, input, output);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
  }
  inline void operator()(const Vector<T, N> &input,
                         Vector<T, N> &output) const {
    Vector<T, M> intermediate;
#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    // Skip entry access when the compile-time CSR pattern is empty.
    if (CSRIndicesType::size > 0) {
      for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = CSRPointersType::list[i];
             j < CSRPointersType::list[i + 1]; ++j) {
          intermediate[i] +=
              this->matrix.values[j] * input[CSRIndicesType::list[j]];
        }
      }
    }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

    SparseNormalProduct::Row<T, M, N, CSRIndicesType, CSRPointersType,
                             0>::compute(this->matrix, input, intermediate);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_
    this->transpose_multiply(intermediate, output);
  }
};

} // namespace GmresOperation

/**
 * @brief Reusable GMRES workspace; K controls storage, cycles controls
 * restarts.
 *
 * All cycle/Arnoldi slots are visited; completed slots skip numerical updates.
 * Iteration counts report actual Arnoldi updates, not the configured slots.
 * rho is always the norm of the true final residual (also for complex inputs).
 * No non-finite checks are made here. Products and final norms must fit Real.
 */
template <typename T, std::size_t N, std::size_t K = N> class GmresSolver {
public:
  using Real = typename Numerical::Scalar<T>::Real;
  static_assert(K > 0 && K <= N, "GMRES requires 1 <= K <= N.");
  static_assert(std::is_floating_point<Real>::value,
                "GMRES requires floating-point real components.");

  std::size_t cycles = DEFAULT_CYCLES_GMRES;
  std::size_t krylov_dimension = K;
  std::size_t orthogonalization_passes = DEFAULT_ORTHOGONALIZATION_PASSES_GMRES;
  Real division_min = static_cast<Real>(DEFAULT_DIVISION_MIN_GMRES);
  Real relative_tolerance = static_cast<Real>(DEFAULT_RELATIVE_TOLERANCE_GMRES);
  Real absolute_tolerance = static_cast<Real>(DEFAULT_ABSOLUTE_TOLERANCE_GMRES);

  template <typename MatrixType>
  inline Vector<T, N>
  solve(const MatrixType &matrix, const Vector<T, N> &right_hand_side,
        const Vector<T, N> &initial, std::size_t active_size = N) {
    active_size = active_size > N ? N : active_size;
    GmresOperation::MatrixOperator<MatrixType, T, N> operation = {matrix,
                                                                  active_size};
    return this->solve_operator(operation, right_hand_side, initial,
                                active_size);
  }

  template <typename MatrixType>
  inline Vector<T, N> solve_preconditioned(
      const MatrixType &matrix, const Vector<T, N> &right_hand_side,
      const Vector<T, N> &initial, const Vector<T, N> &inverse_diagonal) {
    GmresOperation::MatrixOperator<MatrixType, T, N> operation = {matrix, N};
    return this->_solve(operation, right_hand_side, initial, inverse_diagonal,
                        N);
  }

  template <typename Operator>
  inline Vector<T, N>
  solve_operator(const Operator &operation, const Vector<T, N> &right_hand_side,
                 const Vector<T, N> &initial, std::size_t active_size = N) {
    GmresOperation::IdentityPreconditioner<T> identity;
    return this->_solve(operation, right_hand_side, initial, identity,
                        active_size > N ? N : active_size);
  }

  inline Real get_residual_norm() const { return this->residual_norm_; }
  inline std::size_t get_iterations() const { return this->iterations_; }
  inline bool is_converged() const { return this->converged_; }
  inline bool has_breakdown() const { return this->breakdown_; }
  inline bool threshold_was_adjusted() const {
    return this->threshold_adjusted_;
  }

private:
  std::array<T, N * K> basis_ = {};
  std::array<T, K *(K + 1) / 2> triangular_ = {};
  std::array<T, K + 1> column_ = {};
  std::array<T, K + 1> transformed_rhs_ = {};
  std::array<T, K> sine_ = {};
  std::array<Real, K> cosine_ = {};
  std::array<T, K> coefficients_ = {};
  Real residual_norm_ = static_cast<Real>(0);
  std::size_t iterations_ = 0;
  bool converged_ = false;
  bool breakdown_ = false;
  bool threshold_adjusted_ = false;

  inline T &_r(std::size_t row, std::size_t column) {
    return this->triangular_[column * (column + 1) / 2 + row];
  }

  template <typename Operator, typename Preconditioner>
  inline Vector<T, N>
  _solve(const Operator &operation, const Vector<T, N> &right_hand_side,
         const Vector<T, N> &initial, const Preconditioner &inverse_diagonal,
         std::size_t active_size) {
    using Scalar = Numerical::Scalar<T>;

    // Initialize solver state and correct the denominator threshold.
    const Real zero = static_cast<Real>(0);
    const Real floor = Numerical::positive_threshold(this->division_min);
    this->threshold_adjusted_ = this->division_min <= zero;
    this->iterations_ = 0;
    this->breakdown_ = false;
    this->converged_ = false;

    // Bound the Krylov dimension and select the orthogonalization passes.
    std::size_t dimension = this->krylov_dimension;
    dimension = dimension == 0 ? 1 : (dimension > K ? K : dimension);
    if (dimension > active_size) {
      dimension = active_size;
    }
    const std::size_t passes = this->orthogonalization_passes > 1 ? 2 : 1;

    // Convergence threshold: absolute tolerance + relative tolerance * ||b||.
    const Real relative =
        this->relative_tolerance > zero ? this->relative_tolerance : zero;
    const Real absolute =
        this->absolute_tolerance > zero ? this->absolute_tolerance : zero;
    const Real tolerance =
        absolute + relative * Numerical::norm(right_hand_side, active_size);

    // Initial residual: r = b - A*x. Check for an already converged solution.
    Vector<T, N> solution = initial;
    Vector<T, N> residual;
    GmresOperation::apply_operator(operation, solution, residual, active_size);
    for (std::size_t i = 0; i < N; ++i) {
      residual[i] = i < active_size ? right_hand_side[i] - residual[i] : T();
    }
    this->residual_norm_ = Numerical::norm(residual, active_size);
    this->converged_ = this->residual_norm_ <= tolerance;

    // Run the first cycle and any restarts within the configured cycle budget.
    for (std::size_t cycle = 0; cycle < this->cycles; ++cycle) {
      if (!this->converged_ && !this->breakdown_ && dimension > 0) {
        // Initialize the small least-squares right-hand side: b_hat = ||r||*e1.
        const Real initial_norm = this->residual_norm_;
        for (std::size_t j = 0; j <= K; ++j) {
          this->transformed_rhs_[j] = T();
        }
        this->transformed_rhs_[0] = initial_norm;

        // Normalize r to form the first basis vector of this cycle.
        // initial_norm > tolerance >= 0, so this denominator is nonzero.
        // Use a shared reciprocal only when it is representable.
        const bool initial_reciprocal_safe =
            initial_norm >= std::numeric_limits<Real>::min();
        const Real initial_inverse = initial_reciprocal_safe
                                         ? static_cast<Real>(1) / initial_norm
                                         : zero;
        for (std::size_t i = 0; i < N; ++i) {
          this->basis_[i] = initial_reciprocal_safe
                                ? residual[i] * initial_inverse
                                : residual[i] / initial_norm;
        }

        std::size_t used = 0;
        bool active = true;
        bool arnoldi_breakdown = false;
        Vector<T, N> input;
        Vector<T, N> product;

        // Build the Krylov basis and Hessenberg matrix within the step budget.
        for (std::size_t column = 0; column < K; ++column) {
          if (active && column < dimension) {
            // Apply right preconditioning, then multiply: w = A*(D^-1*q_k).
            // D^-1 is the identity when no preconditioner is requested.
            for (std::size_t i = 0; i < N; ++i) {
              input[i] = this->basis_[column * N + i] * inverse_diagonal[i];
            }
            GmresOperation::apply_operator(operation, input, product,
                                           active_size);
            Real product_norm = Numerical::norm(product, active_size);

            // Reset the work column of H before orthogonalization.
            for (std::size_t j = 0; j <= K; ++j) {
              this->column_[j] = T();
            }

            // Orthogonalize w against the stored basis and fill H(j, k).
            // Modified Gram-Schmidt; the second pass adds its corrections.
            for (std::size_t pass = 0; pass < passes; ++pass) {
              for (std::size_t j = 0; j <= column; ++j) {
                T projection = T();
                Real basis_square = zero;
                for (std::size_t i = 0; i < N; ++i) {
                  const T basis_value = this->basis_[j * N + i];
                  projection += Scalar::conjugate(basis_value) * product[i];
                  basis_square += Scalar::squared_abs(basis_value);
                }
                // A dependency-provided norm may be approximate. Project onto
                // the actual nonzero basis vector instead of assuming unit
                // length, which could create a spurious Arnoldi direction.
                // Stored vectors come from nonzero residuals, so their
                // squared lengths are positive and this divisor is nonzero.
                projection = projection / basis_square;
                this->column_[j] += projection;
                for (std::size_t i = 0; i < N; ++i) {
                  product[i] -= this->basis_[j * N + i] * projection;
                }
              }
            }

            // Store H(k+1, k) = ||w|| and detect Arnoldi breakdown.
            Real next_norm = Numerical::norm(product, active_size);
            this->column_[column + 1] = next_norm;
            Real breakdown_limit =
                static_cast<Real>(GMRES_BREAKDOWN_RELATIVE_TOLERANCE) *
                product_norm;
            arnoldi_breakdown = next_norm <= breakdown_limit;

            // Normalize the next basis vector only if another step can use it.
            // The last column still needs its subdiagonal and rotation.
            if (!arnoldi_breakdown && column + 1 < dimension) {
              const bool reciprocal_safe =
                  next_norm >= std::numeric_limits<Real>::min();
              const Real inverse_norm =
                  reciprocal_safe ? static_cast<Real>(1) / next_norm : zero;
              for (std::size_t i = 0; i < N; ++i) {
                this->basis_[(column + 1) * N + i] =
                    reciprocal_safe ? product[i] * inverse_norm
                                    : product[i] / next_norm;
              }
            }

            // Apply previous Givens rotations to the current column of H.
            for (std::size_t j = 0; j < column; ++j) {
              T upper = this->column_[j];
              T lower = this->column_[j + 1];
              this->column_[j] =
                  upper * this->cosine_[j] + this->sine_[j] * lower;
              this->column_[j + 1] =
                  -Scalar::conjugate(this->sine_[j]) * upper +
                  lower * this->cosine_[j];
            }

            // Eliminate H(k+1, k) with a new rotation and store column k of R.
            Numerical::givens(this->column_[column], this->column_[column + 1],
                              this->cosine_[column], this->sine_[column]);
            this->column_[column] =
                this->column_[column] * this->cosine_[column] +
                this->sine_[column] * this->column_[column + 1];
            for (std::size_t j = 0; j <= column; ++j) {
              this->_r(j, column) = this->column_[j];
            }

            // Rotate b_hat; the next entry's magnitude estimates ||r||.
            this->transformed_rhs_[column + 1] =
                -Scalar::conjugate(this->sine_[column]) *
                this->transformed_rhs_[column];
            this->transformed_rhs_[column] *= this->cosine_[column];
            used = column + 1;
            ++this->iterations_;

            // End Arnoldi updates on breakdown or a sufficiently small
            // estimate. Final convergence is checked later using the actual
            // residual.
            active = !arnoldi_breakdown &&
                     Scalar::magnitude(this->transformed_rhs_[column + 1]) >
                         tolerance;
          }
        }

        // Back substitution: solve the used triangular system R*y = b_hat.
        bool triangular_valid = true;
        for (std::size_t offset = 0; offset < K; ++offset) {
          const std::size_t row = K - 1 - offset;
          this->coefficients_[row] = T();
          if (row < used && triangular_valid) {
            T value = this->transformed_rhs_[row];
            for (std::size_t j = row + 1; j < K; ++j) {
              if (j < used) {
                value -= this->_r(row, j) * this->coefficients_[j];
              }
            }
            // Reject the correction if a diagonal pivot cannot be used safely.
            if (Scalar::magnitude(this->_r(row, row)) < floor) {
              triangular_valid = false;
            } else {
              this->coefficients_[row] =
                  Scalar::divide(value, this->_r(row, row), floor);
            }
          }
        }

        // Form the candidate solution: x_candidate = x + D^-1*Q*y.
        if (triangular_valid) {
          Vector<T, N> candidate = solution;
          for (std::size_t i = 0; i < N; ++i) {
            T correction = T();
            for (std::size_t j = 0; j < K; ++j) {
              if (j < used) {
                correction += this->basis_[j * N + i] * this->coefficients_[j];
              }
            }
            candidate[i] += inverse_diagonal[i] * correction;
          }

          // Recompute r_candidate = b - A*x_candidate to verify the correction.
          GmresOperation::apply_operator(operation, candidate, residual,
                                         active_size);
          for (std::size_t i = 0; i < N; ++i) {
            residual[i] =
                i < active_size ? right_hand_side[i] - residual[i] : T();
          }

          // Keep the candidate only when its actual residual does not increase.
          Real candidate_norm = Numerical::norm(residual, active_size);
          if (candidate_norm <= initial_norm) {
            solution = candidate;
            this->residual_norm_ = candidate_norm;
          } else {
            // Restore the retained solution's residual for a possible restart.
            GmresOperation::apply_operator(operation, solution, residual,
                                           active_size);
            for (std::size_t i = 0; i < N; ++i) {
              residual[i] =
                  i < active_size ? right_hand_side[i] - residual[i] : T();
            }
          }

          // Confirm convergence from the retained solution's actual residual.
          this->converged_ = this->residual_norm_ <= tolerance;
        }

        // A happy breakdown is successful only when the true residual agrees.
        // Otherwise, or if the triangular solve failed, disable later cycles.
        this->breakdown_ =
            !triangular_valid || (arnoldi_breakdown && !this->converged_);
      }
    }
    return solution;
  }
};

/** Build an optional fixed right preconditioner; tiny diagonals use identity.
 */
template <typename T, std::size_t N>
inline Vector<T, N>
gmres_inverse_diagonal(const Matrix<T, N, N> &matrix,
                       typename Numerical::Scalar<T>::Real division_min) {
  using Scalar = Numerical::Scalar<T>;
  using Real = typename Scalar::Real;
  const Real floor = Numerical::positive_threshold(division_min);
  Vector<T, N> inverse;
  for (std::size_t i = 0; i < N; ++i) {
    const Real magnitude = Scalar::magnitude(matrix(i, i));
    inverse[i] =
        magnitude < floor || magnitude < std::numeric_limits<Real>::min()
            ? static_cast<T>(1)
            : Scalar::divide(static_cast<T>(1), matrix(i, i), floor);
  }
  return inverse;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_GMRES_HPP_
