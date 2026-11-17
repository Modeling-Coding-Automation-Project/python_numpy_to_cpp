/**
 * @file base_matrix_gmres.hpp
 * @brief Fixed-capacity restarted GMRES with real or complex arithmetic.
 *
 * Provides reusable solver storage, dense and compiled-CSR products, optional
 * right diagonal preconditioning, and real normal-equation operators. The
 * default fixed-size container configuration uses bounded solver storage.
 * Helper operations support for loops or recursive template expansion through
 * BASE_MATRIX_USE_FOR_LOOP_OPERATION_. The caller checks non-finite inputs
 * and outputs; intermediate products and final norms must
 * fit their scalar type.
 */
#ifndef BASE_MATRIX_GMRES_HPP_
#define BASE_MATRIX_GMRES_HPP_

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_matrix.hpp"
#include "base_matrix_numerical.hpp"

#include <array>
#include <cstddef>
#include <type_traits>

namespace Base {
namespace Matrix {

// Default work budget and convergence thresholds for GMRES.
constexpr std::size_t DEFAULT_CYCLES_GMRES = 1;
constexpr std::size_t DEFAULT_ORTHOGONALIZATION_PASSES_GMRES = 1;
constexpr double DEFAULT_DIVISION_MIN_GMRES = 1.0e-10;
// Smaller nonzero norms use direct division; this is not a convergence
// tolerance.
constexpr double NORMALIZATION_RECIPROCAL_MIN_GMRES = 1.0e-30;
// Smaller diagonal magnitudes use identity in the optional preconditioner.
constexpr double PRECONDITIONER_RECIPROCAL_MIN_GMRES = 1.0e-30;
// Validate the shared float/double settings using the narrower type.
static_assert(
    static_cast<float>(NORMALIZATION_RECIPROCAL_MIN_GMRES) > 0.0f &&
        1.0f / static_cast<float>(NORMALIZATION_RECIPROCAL_MIN_GMRES) > 0.0f,
    "The GMRES normalization threshold must have a positive, "
    "representable float reciprocal.");
static_assert(
    static_cast<float>(PRECONDITIONER_RECIPROCAL_MIN_GMRES) > 0.0f &&
        1.0f / static_cast<float>(PRECONDITIONER_RECIPROCAL_MIN_GMRES) > 0.0f,
    "The GMRES preconditioner threshold must have a positive, "
    "representable float reciprocal.");
// Conservative shared default; callers can request a different residual target.
// With both tolerances zero, only an exact zero residual meets convergence.
// Legacy APIs explicitly replace this value with their decay_rate argument.
constexpr double DEFAULT_RELATIVE_TOLERANCE_GMRES = 1.0e-12;
constexpr double DEFAULT_ABSOLUTE_TOLERANCE_GMRES = 0.0;
// Arnoldi breakdown limit: relative tolerance * ||A*D^-1*q||.
// Keep this independent of the requested residual tolerance.
constexpr double GMRES_BREAKDOWN_RELATIVE_TOLERANCE = 8.0e-12;
static_assert(GMRES_BREAKDOWN_RELATIVE_TOLERANCE >= 0.0,
              "The GMRES breakdown tolerance must be nonnegative.");

namespace GmresOperation {

/* Vector initialization and projection onto the leading subspace. */
namespace VectorSubspace {

/**
 * @brief Visits vector components for initialization and subspace projection.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The vector capacity.
 * @tparam Index The current vector index.
 */
template <typename T, std::size_t N, std::size_t Index> struct Loop {
  /**
   * @brief Clears this component and all remaining components.
   *
   * @param output The vector to modify; components outside this
   * operation are retained.
   */
  static inline void clear(Vector<T, N> &output) {
    output[Index] = T();
    Loop<T, N, Index + 1>::clear(output);
  }

  /**
   * @brief Copies the remaining components that belong to
   * the active subspace.
   *
   * @param input The vector whose active components are copied.
   * @param output The vector to modify; components outside this
   * operation are retained.
   * @param active_size The number of active leading components, at most N.
   */
  static inline void copy_active(const Vector<T, N> &input,
                                 Vector<T, N> &output,
                                 std::size_t active_size) {
    if (Index < active_size) {
      output[Index] = input[Index];
    }
    Loop<T, N, Index + 1>::copy_active(input, output, active_size);
  }

  /**
   * @brief Clears the remaining components outside the active subspace.
   *
   * @param output The vector to modify; components outside this
   * operation are retained.
   * @param active_size The number of active leading components, at most N.
   */
  static inline void clear_inactive(Vector<T, N> &output,
                                    std::size_t active_size) {
    if (Index >= active_size) {
      output[Index] = T();
    }
    Loop<T, N, Index + 1>::clear_inactive(output, active_size);
  }
};

// Termination condition: Index == N.
/**
 * @brief Terminates VectorSubspace recursion without accessing storage.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The vector capacity.
 */
template <typename T, std::size_t N> struct Loop<T, N, N> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void clear(Vector<T, N> &) {}
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void copy_active(const Vector<T, N> &, Vector<T, N> &,
                                 std::size_t) {}
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void clear_inactive(Vector<T, N> &, std::size_t) {}
};

} // namespace VectorSubspace

/* Column-major multiplication; preserve the ascending accumulation order. */
namespace DenseMultiply {

/**
 * @brief Accumulates one dense column into the active output rows.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The square matrix size and vector capacity.
 * @tparam ColumnIndex The current matrix column.
 * @tparam RowIndex The current matrix row.
 * @tparam FullSize Whether every matrix row and column is active.
 */
template <typename T, std::size_t N, std::size_t ColumnIndex,
          std::size_t RowIndex, bool FullSize>
struct Row {
  /**
   * @brief Adds the current column contribution to each remaining active row.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param input The source vector for the matrix product.
   * @param output The result storage updated by this recursion.
   * @param active_size The number of active leading components, at most N.
   */
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

/**
 * @brief Terminates DenseMultiply recursion without accessing storage.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The square matrix size and vector capacity.
 * @tparam ColumnIndex The current matrix column.
 * @tparam FullSize Whether every matrix row and column is active.
 */
template <typename T, std::size_t N, std::size_t ColumnIndex, bool FullSize>
struct Row<T, N, ColumnIndex, N, FullSize> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(const Matrix<T, N, N> &, const Vector<T, N> &,
                             Vector<T, N> &, std::size_t) {}
};

/**
 * @brief Visits dense columns in ascending order for a matrix-vector product.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The square matrix size and vector capacity.
 * @tparam ColumnIndex The current matrix column.
 * @tparam FullSize Whether every matrix row and column is active.
 */
template <typename T, std::size_t N, std::size_t ColumnIndex, bool FullSize>
struct Column {
  /**
   * @brief Accumulates the current and remaining dense matrix columns.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param input The source vector for the matrix product.
   * @param output The result storage updated by this recursion.
   * @param active_size The number of active leading components, at most N.
   */
  static inline void compute(const Matrix<T, N, N> &matrix,
                             const Vector<T, N> &input, Vector<T, N> &output,
                             std::size_t active_size) {
    Row<T, N, ColumnIndex, 0, FullSize>::compute(matrix, input, output,
                                                 active_size);
    Column<T, N, ColumnIndex + 1, FullSize>::compute(matrix, input, output,
                                                     active_size);
  }
};

/**
 * @brief Terminates DenseMultiply recursion without accessing storage.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The square matrix size and vector capacity.
 * @tparam FullSize Whether every matrix row and column is active.
 */
template <typename T, std::size_t N, bool FullSize>
struct Column<T, N, N, FullSize> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(const Matrix<T, N, N> &, const Vector<T, N> &,
                             Vector<T, N> &, std::size_t) {}
};

/**
 * @brief Computes the dense product in the leading active subspace.
 *
 * Output is cleared first. Input and output must refer to different vectors.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The square matrix size and vector capacity.
 * @param matrix The input dense matrix.
 * @param input The source vector.
 * @param output The product; inactive components are set to zero.
 * @param active_size The number of active leading rows
 * and columns, at most N.
 */
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

/**
 * @brief Accumulates one output component of a dense transpose product.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam ColumnIndex The current matrix column.
 * @tparam RowIndex The current matrix row.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex,
          std::size_t RowIndex>
struct Row {
  /**
   * @brief Adds the remaining row contributions to one
   * transpose-product component.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param input The source vector for the matrix product.
   * @param output The result storage updated by this recursion.
   */
  static inline void compute(const Matrix<T, M, N> &matrix,
                             const Vector<T, M> &input, Vector<T, N> &output) {
    output[ColumnIndex] += matrix(RowIndex, ColumnIndex) * input[RowIndex];
    Row<T, M, N, ColumnIndex, RowIndex + 1>::compute(matrix, input, output);
  }
};

/**
 * @brief Terminates DenseTransposeMultiply recursion
 * without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam ColumnIndex The current matrix column.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex>
struct Row<T, M, N, ColumnIndex, M> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(const Matrix<T, M, N> &, const Vector<T, M> &,
                             Vector<T, N> &) {}
};

/**
 * @brief Initializes and computes each component of a
 * dense transpose product.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam ColumnIndex The current matrix column.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex>
struct Column {
  /**
   * @brief Overwrites the remaining output components with
   * the transpose product.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param input The source vector for the matrix product.
   * @param output The result storage updated by this recursion.
   */
  static inline void compute(const Matrix<T, M, N> &matrix,
                             const Vector<T, M> &input, Vector<T, N> &output) {
    output[ColumnIndex] = T();
    Row<T, M, N, ColumnIndex, 0>::compute(matrix, input, output);
    Column<T, M, N, ColumnIndex + 1>::compute(matrix, input, output);
  }
};

/**
 * @brief Terminates DenseTransposeMultiply recursion
 * without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 */
template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, N> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(const Matrix<T, M, N> &, const Vector<T, M> &,
                             Vector<T, N> &) {}
};

} // namespace DenseTransposeMultiply

/* Explicit dense normal equations: fill one Gram column and RHS entry. */
namespace DenseNormalEquations {

/**
 * @brief Accumulates one entry of the dense Gram matrix.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam ColumnIndex The current matrix column.
 * @tparam GramRowIndex The current row of the Gram matrix.
 * @tparam RowIndex The current matrix row.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex,
          std::size_t GramRowIndex, std::size_t RowIndex>
struct Product {
  /**
   * @brief Adds the remaining row products to one Gram-matrix entry.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param gram The Gram-matrix storage updated by this recursion.
   */
  static inline void compute(const Matrix<T, M, N> &matrix,
                             Matrix<T, N, N> &gram) {
    gram(GramRowIndex, ColumnIndex) +=
        matrix(RowIndex, GramRowIndex) * matrix(RowIndex, ColumnIndex);
    Product<T, M, N, ColumnIndex, GramRowIndex, RowIndex + 1>::compute(matrix,
                                                                       gram);
  }
};

/**
 * @brief Terminates DenseNormalEquations recursion without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam ColumnIndex The current matrix column.
 * @tparam GramRowIndex The current row of the Gram matrix.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex,
          std::size_t GramRowIndex>
struct Product<T, M, N, ColumnIndex, GramRowIndex, M> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(const Matrix<T, M, N> &, Matrix<T, N, N> &) {}
};

/**
 * @brief Initializes and fills one column of the dense Gram matrix.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam ColumnIndex The current matrix column.
 * @tparam GramRowIndex The current row of the Gram matrix.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex,
          std::size_t GramRowIndex>
struct Row {
  /**
   * @brief Overwrites the remaining Gram entries in the current column.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param gram The Gram-matrix storage updated by this recursion.
   */
  static inline void compute(const Matrix<T, M, N> &matrix,
                             Matrix<T, N, N> &gram) {
    gram(GramRowIndex, ColumnIndex) = T();
    Product<T, M, N, ColumnIndex, GramRowIndex, 0>::compute(matrix, gram);
    Row<T, M, N, ColumnIndex, GramRowIndex + 1>::compute(matrix, gram);
  }
};

/**
 * @brief Terminates DenseNormalEquations recursion without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam ColumnIndex The current matrix column.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex>
struct Row<T, M, N, ColumnIndex, N> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(const Matrix<T, M, N> &, Matrix<T, N, N> &) {}
};

/**
 * @brief Computes dense normal equations one Gram column at a time.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam ColumnIndex The current matrix column.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t ColumnIndex>
struct Column {
  /**
   * @brief Fills the remaining Gram columns and normal
   * right-hand-side entries.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param right_hand_side The original right-hand side b.
   * @param gram The Gram-matrix storage updated by this recursion.
   * @param normal_rhs The normal-equation right-hand-side storage for A^T*b.
   */
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

/**
 * @brief Terminates DenseNormalEquations recursion without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 */
template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, N> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(const Matrix<T, M, N> &, const Vector<T, M> &,
                             Matrix<T, N, N> &, Vector<T, N> &) {}
};

} // namespace DenseNormalEquations

/* Clear the Gram matrix and normal RHS before accumulating sparse rows. */
namespace ZeroNormalEquations {

/**
 * @brief Clears the entries in one Gram-matrix column.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The Gram-matrix size and normal-vector capacity.
 * @tparam ColumnIndex The current matrix column.
 * @tparam RowIndex The current matrix row.
 */
template <typename T, std::size_t N, std::size_t ColumnIndex,
          std::size_t RowIndex>
struct Row {
  /**
   * @brief Zeros the remaining entries in the current Gram column.
   *
   * @param gram The Gram-matrix storage updated by this recursion.
   */
  static inline void compute(Matrix<T, N, N> &gram) {
    gram(RowIndex, ColumnIndex) = T();
    Row<T, N, ColumnIndex, RowIndex + 1>::compute(gram);
  }
};

/**
 * @brief Terminates ZeroNormalEquations recursion without accessing storage.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The Gram-matrix size and normal-vector capacity.
 * @tparam ColumnIndex The current matrix column.
 */
template <typename T, std::size_t N, std::size_t ColumnIndex>
struct Row<T, N, ColumnIndex, N> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(Matrix<T, N, N> &) {}
};

/**
 * @brief Clears the Gram matrix and the normal-equation right-hand side.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The Gram-matrix size and normal-vector capacity.
 * @tparam ColumnIndex The current matrix column.
 */
template <typename T, std::size_t N, std::size_t ColumnIndex> struct Column {
  /**
   * @brief Zeros the remaining Gram columns and right-hand-side entries.
   *
   * @param gram The Gram-matrix storage updated by this recursion.
   * @param normal_rhs The normal-equation right-hand-side storage for A^T*b.
   */
  static inline void compute(Matrix<T, N, N> &gram, Vector<T, N> &normal_rhs) {
    normal_rhs[ColumnIndex] = T();
    Row<T, N, ColumnIndex, 0>::compute(gram);
    Column<T, N, ColumnIndex + 1>::compute(gram, normal_rhs);
  }
};

/**
 * @brief Terminates ZeroNormalEquations recursion without accessing storage.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The Gram-matrix size and normal-vector capacity.
 */
template <typename T, std::size_t N> struct Column<T, N, N> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(Matrix<T, N, N> &, Vector<T, N> &) {}
};

} // namespace ZeroNormalEquations

/* Direct evaluation of A^T(b-Ax), keeping each subtraction in row order. */
namespace DenseNormalResidual {

/**
 * @brief Visits a dense row to form its residual and transpose contribution.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam RowIndex The current matrix row.
 * @tparam ColumnIndex The current matrix column.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t RowIndex,
          std::size_t ColumnIndex>
struct Column {
  /**
   * @brief Subtracts the remaining dense row products from a residual.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param solution The current candidate solution x.
   * @param residual The row residual, initialized from the
   * corresponding entry of b.
   */
  static inline void subtract(const Matrix<T, M, N> &matrix,
                              const Vector<T, N> &solution, T &residual) {
    residual -= matrix(RowIndex, ColumnIndex) * solution[ColumnIndex];
    Column<T, M, N, RowIndex, ColumnIndex + 1>::subtract(matrix, solution,
                                                         residual);
  }

  /**
   * @brief Adds a row residual multiplied by the remaining matrix entries.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param residual The already computed residual for the current input row.
   * @param stationarity The accumulator for A^T*(b-A*x);
   * initialize it to zero first.
   */
  static inline void accumulate(const Matrix<T, M, N> &matrix,
                                const T &residual, Vector<T, N> &stationarity) {
    stationarity[ColumnIndex] += matrix(RowIndex, ColumnIndex) * residual;
    Column<T, M, N, RowIndex, ColumnIndex + 1>::accumulate(matrix, residual,
                                                           stationarity);
  }
};

/**
 * @brief Terminates DenseNormalResidual recursion without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam RowIndex The current matrix row.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t RowIndex>
struct Column<T, M, N, RowIndex, N> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void subtract(const Matrix<T, M, N> &, const Vector<T, N> &,
                              T &) {}
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void accumulate(const Matrix<T, M, N> &, const T &,
                                Vector<T, N> &) {}
};

/**
 * @brief Accumulates the dense normal residual one input row at a time.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam RowIndex The current matrix row.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t RowIndex>
struct Row {
  /**
   * @brief Adds the remaining row contributions to A^T*(b-A*x).
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param right_hand_side The original right-hand side b.
   * @param solution The current candidate solution x.
   * @param stationarity The accumulator for A^T*(b-A*x);
   * initialize it to zero first.
   */
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

/**
 * @brief Terminates DenseNormalResidual recursion without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 */
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, M> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(const Matrix<T, M, N> &, const Vector<T, M> &,
                             const Vector<T, N> &, Vector<T, N> &) {}
};

} // namespace DenseNormalResidual

/* First product A*x for the dense normal operator. */
namespace DenseNormalProduct {

/**
 * @brief Accumulates one component of the dense product A*x.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam RowIndex The current matrix row.
 * @tparam ColumnIndex The current matrix column.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t RowIndex,
          std::size_t ColumnIndex>
struct Column {
  /**
   * @brief Adds the remaining column products to one intermediate component.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param input The source vector for the matrix product.
   * @param intermediate The accumulator for A*x; initialize it to zero first.
   */
  static inline void compute(const Matrix<T, M, N> &matrix,
                             const Vector<T, N> &input,
                             Vector<T, M> &intermediate) {
    intermediate[RowIndex] +=
        matrix(RowIndex, ColumnIndex) * input[ColumnIndex];
    Column<T, M, N, RowIndex, ColumnIndex + 1>::compute(matrix, input,
                                                        intermediate);
  }
};

/**
 * @brief Terminates DenseNormalProduct recursion without accessing storage.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam RowIndex The current matrix row.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t RowIndex>
struct Column<T, M, N, RowIndex, N> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(const Matrix<T, M, N> &, const Vector<T, N> &,
                             Vector<T, M> &) {}
};

/**
 * @brief Visits the rows of the dense product used by the normal operator.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam RowIndex The current matrix row.
 */
template <typename T, std::size_t M, std::size_t N, std::size_t RowIndex>
struct Row {
  /**
   * @brief Accumulates each remaining row of the intermediate product A*x.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param input The source vector for the matrix product.
   * @param intermediate The accumulator for A*x; initialize it to zero first.
   */
  static inline void compute(const Matrix<T, M, N> &matrix,
                             const Vector<T, N> &input,
                             Vector<T, M> &intermediate) {
    Column<T, M, N, RowIndex, 0>::compute(matrix, input, intermediate);
    Row<T, M, N, RowIndex + 1>::compute(matrix, input, intermediate);
  }
};

/**
 * @brief Terminates DenseNormalProduct recursion without accessing storage.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 */
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, M> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(const Matrix<T, M, N> &, const Vector<T, N> &,
                             Vector<T, M> &) {}
};

} // namespace DenseNormalProduct

/* Sparse matrix-vector multiplication in the leading active subspace. */
namespace SparseMultiply {

/**
 * @brief Accumulates active stored entries from one CSR row.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The square matrix size and vector capacity.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam RowIndex The current matrix row.
 * @tparam Start The current stored-entry index.
 * @tparam End The exclusive end of the stored-entry range.
 */
template <typename T, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex, std::size_t Start,
          std::size_t End>
struct Loop {
  /**
   * @brief Adds the remaining stored entries whose row and column are active.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param input The source vector for the matrix product.
   * @param output The result storage updated by this recursion.
   * @param active_size The number of active leading components, at most N.
   */
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
/**
 * @brief Terminates SparseMultiply recursion without accessing storage.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The square matrix size and vector capacity.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam RowIndex The current matrix row.
 * @tparam End The exclusive end of the stored-entry range.
 */
template <typename T, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex, std::size_t End>
struct Loop<T, N, CSRIndicesType, CSRPointersType, RowIndex, End, End> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(
      const CompiledSparseMatrix<T, N, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, N> &, Vector<T, N> &, std::size_t) {}
};

/**
 * @brief Initializes each output row and evaluates the active sparse product.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The square matrix size and vector capacity.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam RowIndex The current matrix row.
 */
template <typename T, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex>
struct Row {
  /**
   * @brief Zeros and computes the remaining output rows
   * of the sparse product.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param input The source vector for the matrix product.
   * @param output The result storage updated by this recursion.
   * @param active_size The number of active leading components, at most N.
   */
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

/**
 * @brief Terminates SparseMultiply recursion without accessing storage.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The square matrix size and vector capacity.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 */
template <typename T, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
struct Row<T, N, CSRIndicesType, CSRPointersType, N> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(
      const CompiledSparseMatrix<T, N, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, N> &, Vector<T, N> &, std::size_t) {}
};

} // namespace SparseMultiply

/* Add each sparse row's contribution to A^T A and A^T b. */
namespace SparseNormalEquations {

/**
 * @brief Accumulates Gram entries paired with one stored row entry.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam FirstIndex The stored entry paired with the remaining row entries.
 * @tparam Start The current stored-entry index.
 * @tparam End The exclusive end of the stored-entry range.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t FirstIndex, std::size_t Start,
          std::size_t End>
struct SecondEntry {
  /**
   * @brief Adds products pairing FirstIndex with the
   * remaining stored row entries.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param gram The Gram-matrix storage updated by this recursion.
   */
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

/**
 * @brief Terminates SparseNormalEquations recursion
 * without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam FirstIndex The stored entry paired with the remaining row entries.
 * @tparam End The exclusive end of the stored-entry range.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t FirstIndex, std::size_t End>
struct SecondEntry<T, M, N, CSRIndicesType, CSRPointersType, FirstIndex, End,
                   End> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      Matrix<T, N, N> &) {}
};

/**
 * @brief Adds one stored entry to the normal right-hand side and Gram matrix.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam RowIndex The current matrix row.
 * @tparam Start The current stored-entry index.
 * @tparam End The exclusive end of the stored-entry range.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex, std::size_t Start,
          std::size_t End>
struct FirstEntry {
  /**
   * @brief Adds the remaining entries of a CSR row to A^T*A and A^T*b.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param right_hand_side The original right-hand side b.
   * @param gram The Gram-matrix storage updated by this recursion.
   * @param normal_rhs The normal-equation right-hand-side storage for A^T*b.
   */
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

/**
 * @brief Terminates SparseNormalEquations recursion
 * without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam RowIndex The current matrix row.
 * @tparam End The exclusive end of the stored-entry range.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex, std::size_t End>
struct FirstEntry<T, M, N, CSRIndicesType, CSRPointersType, RowIndex, End,
                  End> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, M> &, Matrix<T, N, N> &, Vector<T, N> &) {}
};

/**
 * @brief Visits CSR rows to accumulate the explicit normal equations.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam RowIndex The current matrix row.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex>
struct Row {
  /**
   * @brief Adds the remaining CSR rows to A^T*A and A^T*b.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param right_hand_side The original right-hand side b.
   * @param gram The Gram-matrix storage updated by this recursion.
   * @param normal_rhs The normal-equation right-hand-side storage for A^T*b.
   */
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

/**
 * @brief Terminates SparseNormalEquations recursion
 * without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
struct Row<T, M, N, CSRIndicesType, CSRPointersType, M> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, M> &, Matrix<T, N, N> &, Vector<T, N> &) {}
};

} // namespace SparseNormalEquations

/* Scatter a sparse row times a scalar, preserving stored entry order. */
namespace SparseTransposeMultiply {

/**
 * @brief Scatters one sparse row scaled by an input component.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam Start The current stored-entry index.
 * @tparam End The exclusive end of the stored-entry range.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief Adds the remaining scaled row entries to their output columns.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param input The scalar multiplying each entry in the current row.
   * @param output The result storage updated by this recursion.
   */
  static inline void
  compute(const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>
              &matrix,
          const T &input, Vector<T, N> &output) {
    output[CSRIndicesType::list[Start]] += matrix.values[Start] * input;
    Loop<T, M, N, CSRIndicesType, CSRPointersType, Start + 1, End>::compute(
        matrix, input, output);
  }
};

/**
 * @brief Terminates SparseTransposeMultiply recursion
 * without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam End The exclusive end of the stored-entry range.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t End>
struct Loop<T, M, N, CSRIndicesType, CSRPointersType, End, End> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const T &, Vector<T, N> &) {}
};

/**
 * @brief Visits CSR rows to accumulate a sparse transpose product.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam RowIndex The current matrix row.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex>
struct Row {
  /**
   * @brief Adds the remaining CSR rows to the transpose product.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param input The vector supplying one scale for each input row.
   * @param output The result storage updated by this recursion.
   */
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

/**
 * @brief Terminates SparseTransposeMultiply recursion
 * without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
struct Row<T, M, N, CSRIndicesType, CSRPointersType, M> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, M> &, Vector<T, N> &) {}
};

} // namespace SparseTransposeMultiply

/* Form b-Ax one row at a time, then accumulate A^T(b-Ax). */
namespace SparseNormalResidual {

/**
 * @brief Subtracts a sparse row product from its residual accumulator.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam Start The current stored-entry index.
 * @tparam End The exclusive end of the stored-entry range.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief Subtracts the remaining stored row products from a residual.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param solution The current candidate solution x.
   * @param residual The row residual, initialized from the
   * corresponding entry of b.
   */
  static inline void
  compute(const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>
              &matrix,
          const Vector<T, N> &solution, T &residual) {
    residual -= matrix.values[Start] * solution[CSRIndicesType::list[Start]];
    Loop<T, M, N, CSRIndicesType, CSRPointersType, Start + 1, End>::compute(
        matrix, solution, residual);
  }
};

/**
 * @brief Terminates SparseNormalResidual recursion without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam End The exclusive end of the stored-entry range.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t End>
struct Loop<T, M, N, CSRIndicesType, CSRPointersType, End, End> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, N> &, T &) {}
};

/**
 * @brief Accumulates the sparse normal residual one input row at a time.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam RowIndex The current matrix row.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex>
struct Row {
  /**
   * @brief Adds the remaining CSR row contributions to A^T*(b-A*x).
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param right_hand_side The original right-hand side b.
   * @param solution The current candidate solution x.
   * @param stationarity The accumulator for A^T*(b-A*x);
   * initialize it to zero first.
   */
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

/**
 * @brief Terminates SparseNormalResidual recursion without accessing storage.
 *
 * Transpose operations do not conjugate complex entries.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
struct Row<T, M, N, CSRIndicesType, CSRPointersType, M> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, M> &, const Vector<T, N> &, Vector<T, N> &) {}
};

} // namespace SparseNormalResidual

/* Accumulate Ax into the zero-initialized normal-operator intermediate. */
namespace SparseNormalProduct {

/**
 * @brief Accumulates one sparse row product into a scalar result.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam Start The current stored-entry index.
 * @tparam End The exclusive end of the stored-entry range.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t Start, std::size_t End>
struct Loop {
  /**
   * @brief Adds the remaining stored row products to a scalar output.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param input The source vector for the matrix product.
   * @param output The result storage updated by this recursion.
   */
  static inline void
  compute(const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>
              &matrix,
          const Vector<T, N> &input, T &output) {
    output += matrix.values[Start] * input[CSRIndicesType::list[Start]];
    Loop<T, M, N, CSRIndicesType, CSRPointersType, Start + 1, End>::compute(
        matrix, input, output);
  }
};

/**
 * @brief Terminates SparseNormalProduct recursion without accessing storage.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam End The exclusive end of the stored-entry range.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t End>
struct Loop<T, M, N, CSRIndicesType, CSRPointersType, End, End> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, N> &, T &) {}
};

/**
 * @brief Visits CSR rows to form the intermediate product A*x.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @tparam RowIndex The current matrix row.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType, std::size_t RowIndex>
struct Row {
  /**
   * @brief Accumulates the remaining CSR row products into the
   * intermediate vector.
   *
   * @param matrix The input matrix, whose entries are unchanged.
   * @param input The source vector for the matrix product.
   * @param output The result storage updated by this recursion.
   */
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

/**
 * @brief Terminates SparseNormalProduct recursion without accessing storage.
 *
 * @tparam T The matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
struct Row<T, M, N, CSRIndicesType, CSRPointersType, M> {
  /**
   * @brief Ends this recursion; all arguments are unused.
   */
  static inline void compute(
      const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &,
      const Vector<T, N> &, Vector<T, M> &) {}
};

} // namespace SparseNormalProduct

/**
 * @brief Computes a dense matrix-vector product in the
 * active leading subspace.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The square matrix size and vector capacity.
 * @param matrix The input matrix.
 * @param input The source vector; it must not alias output.
 * @param output The product, with inactive components set to zero.
 * @param active_size The number of leading rows and
 * columns to use, at most N.
 */
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

/**
 * @brief Computes a compiled-CSR matrix-vector product in
 * the active subspace.
 *
 * @tparam T The matrix and vector element type.
 * @tparam N The square matrix size and vector capacity.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @param matrix The input matrix.
 * @param input The source vector; it must not alias output.
 * @param output The product, with inactive components set to zero.
 * @param active_size The number of leading rows and
 * columns to use, at most N.
 */
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

/**
 * @brief Adapts a borrowed matrix and active size to the
 * GMRES callback interface.
 *
 * The referenced matrix must outlive every use of this wrapper.
 *
 * @tparam MatrixType A matrix type supported by multiply.
 * @tparam T The matrix and vector element type.
 * @tparam N The vector capacity.
 */
template <typename MatrixType, typename T, std::size_t N>
struct MatrixOperator {
  const MatrixType &matrix;
  std::size_t active_size;
  /**
   * @brief Applies the borrowed matrix in its configured leading subspace.
   *
   * @param input The source vector; it must not alias output.
   * @param output The product, including zeroed inactive components.
   */
  inline void operator()(const Vector<T, N> &input,
                         Vector<T, N> &output) const {
    multiply(this->matrix, input, output, this->active_size);
  }
};

/**
 * @brief Applies an operator while masking components outside
 * the active subspace.
 *
 * The callback must overwrite its output and accept distinct input and output
 * vectors of capacity N. Partial-subspace calls use a temporary zero-padded
 * input and clear inactive output components after the callback.
 *
 * @tparam Operator A callable accepting input and output vector references.
 * @tparam T The vector element type.
 * @tparam N The vector capacity.
 * @param operation The callback to invoke.
 * @param input The source vector.
 * @param output The callback result with inactive components cleared.
 * @param active_size The number of active leading components, at most N.
 */
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

/**
 * @brief Supplies unit diagonal entries without storing a
 * preconditioning vector.
 *
 * @tparam T The scalar type returned for each diagonal entry.
 */
template <typename T> struct IdentityPreconditioner {
  /**
   * @brief Returns the multiplicative identity for any requested index.
   *
   * The index is unused; no storage is accessed.
   *
   * @return A scalar value equal to one.
   */
  inline T operator[](std::size_t) const { return static_cast<T>(1); }
};

/**
 * @brief Forms explicit normal equations from a dense matrix.
 *
 * Explicit normal equations may be useful for tall matrices or repeated
 * right-hand sides. This operation uses an ordinary transpose, without
 * conjugation, and is intended for real least-squares systems. Output storage
 * must not alias the inputs.
 *
 * @tparam T The real matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @param matrix The input matrix A.
 * @param right_hand_side The original right-hand side b.
 * @param gram The overwritten dense matrix A^T*A.
 * @param normal_rhs The overwritten vector A^T*b.
 */
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

/**
 * @brief Forms dense normal equations by accumulating compiled-CSR rows.
 *
 * Explicit normal equations may be useful for tall matrices or repeated
 * right-hand sides. This operation uses an ordinary transpose, without
 * conjugation, and is intended for real least-squares systems. Output storage
 * must not alias the inputs.
 *
 * @tparam T The real matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @param matrix The input matrix A.
 * @param right_hand_side The original right-hand side b.
 * @param gram The overwritten dense matrix A^T*A.
 * @param normal_rhs The overwritten vector A^T*b.
 */
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

/**
 * @brief Computes the normal-equation residual norm from a dense matrix.
 *
 * Evaluates the original matrix directly rather than using a stored Gram
 * matrix. The transpose is not conjugated; this helper is intended for real
 * least-squares systems.
 *
 * @tparam T The real matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @param matrix The original matrix A.
 * @param right_hand_side The original right-hand side b.
 * @param solution The candidate solution x.
 * @return The Euclidean norm of A^T*(b-A*x), evaluated with scaling.
 */
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

/**
 * @brief Computes the normal-equation residual norm from a
 * compiled-CSR matrix.
 *
 * Evaluates the original matrix directly rather than using a stored Gram
 * matrix. The transpose is not conjugated; this helper is intended for real
 * least-squares systems.
 *
 * @tparam T The real matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 * @param matrix The original matrix A.
 * @param right_hand_side The original right-hand side b.
 * @param solution The candidate solution x.
 * @return The Euclidean norm of A^T*(b-A*x), evaluated with scaling.
 */
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

/**
 * @brief Declares a matrix-free adapter for applying the
 * normal operator A^T*A.
 *
 * Specializations borrow the original matrix and form products without
 * storing a Gram matrix. Applying GMRES to this operator still solves the
 * normal equations; their conditioning is not changed by avoiding Gram
 * storage. Transposes do not conjugate entries, so the adapter is intended
 * for real least-squares systems.
 *
 * @tparam MatrixType The supported dense or compiled-CSR matrix type.
 * @tparam T The real matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 */
template <typename MatrixType, typename T, std::size_t M, std::size_t N>
struct NormalOperator;

/**
 * @brief Applies a dense normal operator while borrowing the original matrix.
 *
 * The matrix must outlive this wrapper. An intermediate vector of length M is
 * used for each application; the dense Gram matrix is not stored.
 *
 * @tparam T The real matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 */
template <typename T, std::size_t M, std::size_t N>
struct NormalOperator<Matrix<T, M, N>, T, M, N> {
  const Matrix<T, M, N> &matrix;
  /**
   * @brief Overwrites the output with the ordinary
   * transpose product A^T*input.
   *
   * @param input The vector of length M; it must not alias output.
   * @param output The transpose product of length N.
   */
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
  /**
   * @brief Applies A^T*A without constructing the Gram matrix.
   *
   * @param input The vector multiplied by the normal operator.
   * @param output The overwritten normal-operator product.
   */
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

/**
 * @brief Applies a compiled-CSR normal operator while borrowing its matrix.
 *
 * The matrix must outlive this wrapper. The stored CSR entries are reused for
 * both products, with an intermediate vector of length M.
 *
 * @tparam T The real matrix and vector element type.
 * @tparam M The number of matrix rows.
 * @tparam N The number of matrix columns.
 * @tparam CSRIndicesType The compile-time CSR column-index sequence.
 * @tparam CSRPointersType The compile-time CSR row-pointer sequence.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndicesType,
          typename CSRPointersType>
struct NormalOperator<
    CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType>, T, M, N> {
  const CompiledSparseMatrix<T, M, N, CSRIndicesType, CSRPointersType> &matrix;
  /**
   * @brief Overwrites the output with the ordinary
   * transpose product A^T*input.
   *
   * @param input The vector of length M; it must not alias output.
   * @param output The transpose product of length N.
   */
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
  /**
   * @brief Applies A^T*A without constructing the Gram matrix.
   *
   * @param input The vector multiplied by the normal operator.
   * @param output The overwritten normal-operator product.
   */
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
 * @brief Reusable GMRES workspace; K controls storage,
 * cycles controls restarts.
 *
 * Cycle and Arnoldi loops have fixed upper bounds; inactive numerical
 * updates are skipped. Iteration counts report actual Arnoldi updates, not
 * the configured slots. rho is always the norm of the true final residual
 * (also for complex inputs). No non-finite checks are made here. Products and
 * final norms must fit Real. The true residual is checked after each
 * candidate correction; corrections that increase it are discarded. This is
 * not a forward-error guarantee.
 *
 * @tparam T The real or complex matrix and vector element type.
 * @tparam N The matrix dimension and vector capacity.
 * @tparam K The stored Krylov dimension, with 1 <= K <= N.
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

  /**
   * @brief Solves a dense or compiled-CSR system within the work budget.
   *
   * Only the leading active subspace participates in the residual and
   * correction. The active size is clamped to N; inactive initial components
   * are retained. Inspect the status getters to distinguish convergence,
   * breakdown, and budget exhaustion.
   *
   * @tparam MatrixType A matrix type supported by GmresOperation::multiply.
   * @param matrix The coefficient matrix, borrowed for this solve.
   * @param right_hand_side The right-hand side b.
   * @param initial The initial solution estimate.
   * @param active_size The number of leading components to solve for.
   * @return The retained solution after the configured cycles.
   */
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

  /**
   * @brief Solves the full system with a fixed right diagonal preconditioner.
   *
   * The supplied inverse diagonal is used in A*D^-1 and in the final
   * correction. It is borrowed for this solve and is not
   * recomputed or validated here.
   *
   * @tparam MatrixType A matrix type supported by GmresOperation::multiply.
   * @param matrix The coefficient matrix.
   * @param right_hand_side The right-hand side b.
   * @param initial The initial solution estimate.
   * @param inverse_diagonal The fixed inverse diagonal D^-1.
   * @return The retained solution in the original,
   * unpreconditioned coordinates.
   */
  template <typename MatrixType>
  inline Vector<T, N> solve_preconditioned(
      const MatrixType &matrix, const Vector<T, N> &right_hand_side,
      const Vector<T, N> &initial, const Vector<T, N> &inverse_diagonal) {
    GmresOperation::MatrixOperator<MatrixType, T, N> operation = {matrix, N};
    return this->_solve(operation, right_hand_side, initial, inverse_diagonal,
                        N);
  }

  /**
   * @brief Solves a system represented by a matrix-vector callback.
   *
   * The callback receives distinct input and output vectors and must
   * overwrite all output components. Partial-subspace calls mask the inactive
   * components. No explicit matrix or preconditioning vector is required.
   *
   * @tparam Operator A callable accepting input and output vector references.
   * @param operation The matrix-vector callback, borrowed for this solve.
   * @param right_hand_side The right-hand side b.
   * @param initial The initial solution estimate.
   * @param active_size The number of active leading components, clamped to N.
   * @return The retained solution after the configured cycles.
   */
  template <typename Operator>
  inline Vector<T, N>
  solve_operator(const Operator &operation, const Vector<T, N> &right_hand_side,
                 const Vector<T, N> &initial, std::size_t active_size = N) {
    GmresOperation::IdentityPreconditioner<T> identity;
    return this->_solve(operation, right_hand_side, initial, identity,
                        active_size > N ? N : active_size);
  }

  /**
   * @brief Returns the true residual norm of the last retained solution.
   *
   * @return The residual norm over the active subspace.
   */
  inline Real get_residual_norm() const { return this->residual_norm_; }
  /**
   * @brief Returns the number of Arnoldi updates performed by the last solve.
   *
   * @return The actual update count across all executed cycles.
   */
  inline std::size_t get_iterations() const { return this->iterations_; }
  /**
   * @brief Reports whether the retained true residual meets the tolerance.
   *
   * @return True if the last solve met the configured residual criterion.
   */
  inline bool is_converged() const { return this->converged_; }
  /**
   * @brief Reports an unusable triangular correction or
   * unsuccessful Arnoldi breakdown.
   *
   * @return True if breakdown prevented further cycles in the last solve.
   */
  inline bool has_breakdown() const { return this->breakdown_; }
  /**
   * @brief Reports whether a nonpositive division threshold
   * required a local fallback.
   *
   * @return True if division_min was nonpositive for the last solve.
   */
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

  /**
   * @brief Accesses the packed upper-triangular least-squares factor.
   *
   * @param row The row index, with row <= column.
   * @param column The column index, less than K.
   * @return A mutable reference to the stored triangular entry.
   */
  inline T &_r(std::size_t row, std::size_t column) {
    return this->triangular_[column * (column + 1) / 2 + row];
  }

  /**
   * @brief Runs bounded right-preconditioned GMRES and
   * updates the solve status.
   *
   * Uses modified Gram-Schmidt, Givens rotations, and true-residual checks.
   * Completed cycle and Arnoldi slots skip numerical updates. The caller
   * supplies an active size at most N and an indexable inverse diagonal
   * covering N entries. A failed triangular correction leaves the last
   * retained solution intact.
   *
   * @tparam Operator A callable providing the matrix-vector product.
   * @tparam Preconditioner An indexable provider of inverse-diagonal entries.
   * @param operation The borrowed matrix-vector callback.
   * @param right_hand_side The right-hand side b.
   * @param initial The starting solution estimate.
   * @param inverse_diagonal The fixed right inverse
   * diagonal, or the identity.
   * @param active_size The number of leading components to include.
   * @return The retained solution after convergence, breakdown,
   * or budget exhaustion.
   */
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
        // Below the reciprocal threshold, retain direct division.
        const bool initial_reciprocal_safe =
            initial_norm >=
            static_cast<Real>(NORMALIZATION_RECIPROCAL_MIN_GMRES);
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
                  next_norm >=
                  static_cast<Real>(NORMALIZATION_RECIPROCAL_MIN_GMRES);
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

/**
 * @brief Builds an optional right inverse-diagonal preconditioner.
 *
 * Diagonal magnitudes below division_min or the reciprocal-use threshold
 * produce a unit entry, leaving that component unpreconditioned. A
 * nonpositive division_min is replaced locally by the numerical default.
 *
 * @tparam T The real or complex matrix element type.
 * @tparam N The square matrix size.
 * @param matrix The coefficient matrix.
 * @param division_min The positive denominator-protection threshold.
 * @return The inverse-diagonal vector, with unit entries for small diagonals.
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
    const bool use_identity =
        magnitude < floor ||
        magnitude < static_cast<Real>(PRECONDITIONER_RECIPROCAL_MIN_GMRES);
    inverse[i] = use_identity
                     ? static_cast<T>(1)
                     : Scalar::divide(static_cast<T>(1), matrix(i, i), floor);
  }
  return inverse;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_GMRES_HPP_
