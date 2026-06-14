/**
 * @file base_matrix_diagonal.hpp
 * @brief Diagonal Matrix Operations for the Base::Matrix Namespace
 *
 * This header defines the DiagMatrix class template and a comprehensive set of
 * operations for diagonal matrices within the Base::Matrix namespace. The code
 * provides efficient, type-safe, and optionally loop-unrolled implementations
 * for diagonal matrix arithmetic, including addition, subtraction,
 * multiplication, division, inversion, and conversion between real and complex
 * representations. It also supports interactions with dense matrices and
 * vectors, as well as utility functions such as trace computation and dense
 * matrix extraction.
 *
 * @note
 * tparam M is the number of rows in the matrix.
 * tparam N is the number of columns in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef BASE_MATRIX_DIAGONAL_HPP_
#define BASE_MATRIX_DIAGONAL_HPP_

#include "base_matrix_macros.hpp"

#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace Base {
namespace Matrix {

/**
 * @brief DiagMatrix is a fixed-size diagonal matrix class template.
 *
 * This class represents a diagonal matrix of size MxM, storing only the
 * diagonal elements. The storage can be either std::vector<T> or std::array<T,
 * M> depending on the
 * BASE_MATRIX_USE_STD_VECTOR_ macro.
 *
 * @tparam T Type of the matrix elements.
 * @tparam M Size of the matrix (number of columns and rows).
 *
 * @note Only the diagonal elements are stored and manipulated.
 */
template <typename T, std::size_t M> class DiagMatrix {
public:
  /* Constant */
  static constexpr std::size_t ROWS = M;
  static constexpr std::size_t COLS = M;

public:
#ifdef BASE_MATRIX_USE_STD_VECTOR_

  DiagMatrix() : data(M, static_cast<T>(0)) {}

  DiagMatrix(const std::vector<T> &input) : data(input) {}

  DiagMatrix(const std::initializer_list<T> &input) : data(input) {}

  DiagMatrix(T input[M]) : data(M, static_cast<T>(0)) {

    for (std::size_t i = 0; i < M; i++) {
      this->data[i] = input[i];
    }
  }

#else // BASE_MATRIX_USE_STD_VECTOR_

  DiagMatrix() : data{} {}

  DiagMatrix(const std::initializer_list<T> &input) : data{} {

    // This may cause runtime error if the size of values is larger than M.
    std::copy(input.begin(), input.end(), this->data.begin());
  }

  DiagMatrix(const std::array<T, M> &input) : data(input) {}

  DiagMatrix(const std::vector<T> &input) : data{} {

    // This may cause runtime error if the size of values is larger than M.
    std::copy(input.begin(), input.end(), this->data.begin());
  }

  DiagMatrix(T input[M]) : data{} {

    for (std::size_t i = 0; i < M; i++) {
      this->data[i] = input[i];
    }
  }

#endif // BASE_MATRIX_USE_STD_VECTOR_

  /* Copy Constructor */
  DiagMatrix(const DiagMatrix<T, M> &other) : data(other.data) {}

  DiagMatrix<T, M> &operator=(const DiagMatrix<T, M> &other) {
    if (this != &other) {
      this->data = other.data;
    }
    return *this;
  }

  /* Move Constructor */
  DiagMatrix(DiagMatrix<T, M> &&other) noexcept : data(std::move(other.data)) {}

  DiagMatrix<T, M> &operator=(DiagMatrix<T, M> &&other) noexcept {
    if (this != &other) {
      this->data = std::move(other.data);
    }
    return *this;
  }

  /* Function */

  /**
   * @brief Creates and returns an identity diagonal matrix.
   *
   * This static method constructs a diagonal matrix of size M x M,
   * where all diagonal elements are set to 1 (of type T), and all
   * off-diagonal elements are zero. The resulting matrix is an identity
   * matrix in the context of diagonal matrices.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The size (number of columns and rows) of the square matrix.
   * @return DiagMatrix<T, M> An identity diagonal matrix of type T and size M.
   */
  static inline DiagMatrix<T, M> identity() {
    DiagMatrix<T, M> identity(std::vector<T>(M, static_cast<T>(1)));

    return identity;
  }

  /**
   * @brief Creates and returns a diagonal matrix filled with a specified value.
   *
   * This static method constructs a diagonal matrix of size M x M,
   * where all diagonal elements are set to the specified value, and all
   * off-diagonal elements are zero.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The size (number of columns and rows) of the square matrix.
   * @param value The value to fill the diagonal elements.
   * @return DiagMatrix<T, M> A diagonal matrix of type T and size M filled with
   * the specified value.
   */
  static inline DiagMatrix<T, M> full(const T &value) {
    DiagMatrix<T, M> full(std::vector<T>(M, value));

    return full;
  }

  /**
   * @brief Creates and returns a diagonal matrix filled with zeros.
   *
   * This static method constructs a diagonal matrix of size M x M,
   * where all diagonal elements are set to zero, and all off-diagonal
   * elements are also zero.
   *
   * @tparam T The type of the matrix elements.
   * @tparam M The size (number of columns and rows) of the square matrix.
   * @return DiagMatrix<T, M> A diagonal matrix of type T and size M filled with
   * zeros.
   */
  T &operator[](std::size_t index) { return this->data[index]; }

  /**
   * @brief Accesses the diagonal element at the specified index.
   *
   * This method provides read-only access to the diagonal element at the
   * specified index. If the index is out of bounds, it returns the last
   * element in the diagonal.
   *
   * @param index The index of the diagonal element to access.
   * @return const T& A constant reference to the diagonal element at the
   * specified index.
   */
  const T &operator[](std::size_t index) const { return this->data[index]; }

  /**
   * @brief Accesses the diagonal element at the specified index.
   *
   * This method provides read-write access to the diagonal element at the
   * specified index. If the index is out of bounds, it returns the last
   * element in the diagonal.
   *
   * @param index The index of the diagonal element to access.
   * @return T& A reference to the diagonal element at the specified index.
   */
  constexpr std::size_t cols() const { return M; }

  /**
   * @brief Returns the number of rows in the diagonal matrix.
   *
   * Since this is a square matrix, the number of rows is equal to the
   * number of columns.
   *
   * @return std::size_t The number of rows in the diagonal matrix.
   */
  constexpr std::size_t rows() const { return M; }

  /**
   * @brief Returns a vector representing the specified row of the diagonal
   * matrix.
   *
   * For a diagonal matrix, only the diagonal element at the given row index is
   * non-zero. If the provided row index is out of bounds (greater than or equal
   * to M), it is clamped to the last valid index.
   *
   * @param row The index of the column to retrieve.
   * @return Vector<T, M> A vector with only the diagonal element at the
   * specified row set, all other elements are zero.
   */
  inline Vector<T, M> get_row(std::size_t row) const {
    if (row >= M) {
      row = M - 1;
    }

    Vector<T, M> result;
    result[row] = this->data[row];

    return result;
  }

  /**
   * @brief Returns a vector representing the specified column of the diagonal
   * matrix.
   *
   * For a diagonal matrix, only the diagonal element at the given column index
   * is non-zero. If the provided column index is out of bounds (greater than or
   * equal to M), it is clamped to the last valid index.
   *
   * @param col The index of the row to retrieve.
   * @return Vector<T, M> A vector with only the diagonal element at the
   * specified column set, all other elements are zero.
   */
  inline DiagMatrix<T, M> inv(T division_min) const {
    DiagMatrix<T, M> result;

    for (std::size_t i = 0; i < M; i++) {
      result[i] = static_cast<T>(1) /
                  Base::Utility::avoid_zero_divide(this->data[i], division_min);
    }

    return result;
  }

/* Variable */
#ifdef BASE_MATRIX_USE_STD_VECTOR_
  std::vector<T> data;
#else  // BASE_MATRIX_USE_STD_VECTOR_
  std::array<T, M> data;
#endif // BASE_MATRIX_USE_STD_VECTOR_
};

/**
 * @brief Converts a diagonal matrix to a column vector.
 *
 * This function takes a diagonal matrix as input and returns a column vector
 * containing the diagonal elements of the matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The size of the diagonal matrix (number of columns and rows).
 * @param matrix The diagonal matrix to be converted.
 * @return Matrix<T, M, 1> A column vector containing the diagonal elements of
 * the input matrix.
 */
template <typename T, std::size_t M>
inline Matrix<T, M, 1> output_diagonal_vector(const DiagMatrix<T, M> &matrix) {

  return Matrix<T, M, 1>(matrix.data);
}

/* Matrix Addition */
namespace DiagMatrixAddDiagMatrix {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                      DiagMatrix<T, M> &result) {
    Core<T, M, Start, Mid>::compute(A, B, result);
    Core<T, M, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, const DiagMatrix<T, M> &,
                      DiagMatrix<T, M> &) {}
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                      DiagMatrix<T, M> &result) {
    result[Start] = A[Start] + B[Start];
  }
};

/**
 * @brief Computes the element-wise addition of two diagonal matrices.
 *
 * This function initiates the recursive computation of the sum of two diagonal
 * matrices A and B, storing the result in the provided result matrix.
 *
 * @param A The first diagonal matrix.
 * @param B The second diagonal matrix.
 * @param result The resulting diagonal matrix after addition.
 */
template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                    DiagMatrix<T, M> &result) {
  Core<T, M, 0, M>::compute(A, B, result);
}

} // namespace DiagMatrixAddDiagMatrix

/**
 * @brief Adds two diagonal matrices element-wise.
 *
 * This function computes the sum of two diagonal matrices A and B, returning a
 * new diagonal matrix that contains the element-wise sums of the corresponding
 * diagonal elements.
 *
 * @param A The first diagonal matrix.
 * @param B The second diagonal matrix.
 * @return DiagMatrix<T, M> The resulting diagonal matrix after addition.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator+(const DiagMatrix<T, M> &A,
                                  const DiagMatrix<T, M> &B) {
  DiagMatrix<T, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] + B[j];
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixAddDiagMatrix::compute<T, M>(A, B, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

namespace DiagMatrixAddMatrix {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
    Core<T, M, Start, Mid>::compute(A, result);
    Core<T, M, Mid, End>::compute(A, result);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, Matrix<T, M, M> &) {}
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
    result.template set<Start, Start>(result.template get<Start, Start>() +
                                      A[Start]);
  }
};

/**
 * @brief Computes the element-wise addition of a diagonal matrix and a dense
 * matrix.
 *
 * This function initiates the recursive computation of the sum of a diagonal
 * matrix A and a dense matrix B, storing the result in the provided result
 * matrix.
 *
 * @param A The diagonal matrix.
 * @param result The resulting dense matrix after addition.
 */
template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
  Core<T, M, 0, M>::compute(A, result);
}

} // namespace DiagMatrixAddMatrix

/**
 * @brief Adds a diagonal matrix to a dense matrix element-wise.
 *
 * This function computes the sum of a diagonal matrix A and a dense matrix B,
 * returning a new dense matrix that contains the element-wise sums of the
 * corresponding diagonal elements.
 *
 * @param A The diagonal matrix.
 * @param B The dense matrix.
 * @return Matrix<T, M, M> The resulting dense matrix after addition.
 */
template <typename T, std::size_t M>
inline Matrix<T, M, M> operator+(const DiagMatrix<T, M> &A,
                                 const Matrix<T, M, M> &B) {
  Matrix<T, M, M> result = B;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += A[i];
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixAddMatrix::compute<T, M>(A, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/**
 * @brief Adds a dense matrix to a diagonal matrix element-wise.
 *
 * This function computes the sum of a dense matrix A and a diagonal matrix B,
 * returning a new dense matrix that contains the element-wise sums of the
 * corresponding diagonal elements.
 *
 * @param A The dense matrix.
 * @param B The diagonal matrix.
 * @return Matrix<T, M, M> The resulting dense matrix after addition.
 */
template <typename T, std::size_t M>
inline Matrix<T, M, M> operator+(const Matrix<T, M, M> &A,
                                 const DiagMatrix<T, M> &B) {
  Matrix<T, M, M> result = A;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += B[i];
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixAddMatrix::compute<T, M>(B, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Minus */
namespace DiagMatrixMinus {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Loop;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Loop<T, M, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> &result) {
    Loop<T, M, Start, Mid>::compute(A, result);
    Loop<T, M, Mid, End>::compute(A, result);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Loop<T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, DiagMatrix<T, M> &) {}
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Loop<T, M, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> &result) {
    result[Start] = -A[Start];
  }
};

/**
 * @brief Computes the element-wise negation of a diagonal matrix.
 *
 * This function initiates the recursive computation of the negation of a
 * diagonal matrix A, storing the result in the provided result matrix.
 *
 * @param A The diagonal matrix to negate.
 * @param result The resulting diagonal matrix after negation.
 */
template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> &result) {
  Loop<T, M, 0, M>::compute(A, result);
}

} // namespace DiagMatrixMinus

/**
 * @brief Negates a diagonal matrix element-wise.
 *
 * This function computes the negation of a diagonal matrix A, returning a new
 * diagonal matrix that contains the negated values of the corresponding
 * diagonal elements.
 *
 * @param A The diagonal matrix to negate.
 * @return DiagMatrix<T, M> The resulting diagonal matrix after negation.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator-(const DiagMatrix<T, M> &A) {
  DiagMatrix<T, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = -A[i];
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixMinus::compute<T, M>(A, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Matrix Subtraction */
namespace DiagMatrixSubDiagMatrix {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                      DiagMatrix<T, M> &result) {
    Core<T, M, Start, Mid>::compute(A, B, result);
    Core<T, M, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, const DiagMatrix<T, M> &,
                      DiagMatrix<T, M> &) {}
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                      DiagMatrix<T, M> &result) {
    result[Start] = A[Start] - B[Start];
  }
};

/**
 * @brief Computes the element-wise subtraction of two diagonal matrices.
 *
 * This function initiates the recursive computation of the difference between
 * two diagonal matrices A and B, storing the result in the provided result
 * matrix.
 *
 * @param A The first diagonal matrix.
 * @param B The second diagonal matrix.
 * @param result The resulting diagonal matrix after subtraction.
 */
template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                    DiagMatrix<T, M> &result) {
  Core<T, M, 0, M>::compute(A, B, result);
}

} // namespace DiagMatrixSubDiagMatrix

/**
 * @brief Subtracts two diagonal matrices element-wise.
 *
 * This function computes the difference between two diagonal matrices A and B,
 * returning a new diagonal matrix that contains the element-wise differences
 * of the corresponding diagonal elements.
 *
 * @param A The first diagonal matrix.
 * @param B The second diagonal matrix.
 * @return DiagMatrix<T, M> The resulting diagonal matrix after subtraction.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator-(const DiagMatrix<T, M> &A,
                                  const DiagMatrix<T, M> &B) {
  DiagMatrix<T, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] - B[j];
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixSubDiagMatrix::compute<T, M>(A, B, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/**
 * @brief Subtracts a diagonal matrix from a dense matrix element-wise.
 *
 * This function computes the difference between a diagonal matrix A and a
 * dense matrix B, returning a new dense matrix that contains the element-wise
 * differences of the corresponding diagonal elements.
 *
 * @param A The diagonal matrix.
 * @param B The dense matrix.
 * @return Matrix<T, M, M> The resulting dense matrix after subtraction.
 */
template <typename T, std::size_t M>
inline Matrix<T, M, M> operator-(const DiagMatrix<T, M> &A,
                                 const Matrix<T, M, M> &B) {
  Matrix<T, M, M> result = -B;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += A[i];
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixAddMatrix::compute<T, M>(A, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

namespace MatrixSubDiagMatrix {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
    Core<T, M, Start, Mid>::compute(A, result);
    Core<T, M, Mid, End>::compute(A, result);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, Matrix<T, M, M> &) {}
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
    result.template set<Start, Start>(result.template get<Start, Start>() -
                                      A[Start]);
  }
};

/**
 * @brief Computes the element-wise subtraction of a diagonal matrix from a
 * dense matrix.
 *
 * This function initiates the recursive computation of the difference between
 * a dense matrix A and a diagonal matrix B, storing the result in the provided
 * result matrix.
 *
 * @param A The diagonal matrix.
 * @param result The resulting dense matrix after subtraction.
 */
template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
  Core<T, M, 0, M>::compute(A, result);
}

} // namespace MatrixSubDiagMatrix

/**
 * @brief Subtracts a dense matrix from a diagonal matrix element-wise.
 *
 * This function computes the difference between a diagonal matrix A and a
 * dense matrix B, returning a new dense matrix that contains the element-wise
 * differences of the corresponding diagonal elements.
 *
 * @param A The diagonal matrix.
 * @param B The dense matrix.
 * @return Matrix<T, M, M> The resulting dense matrix after subtraction.
 */
template <typename T, std::size_t M>
inline Matrix<T, M, M> operator-(const Matrix<T, M, M> &A,
                                 const DiagMatrix<T, M> &B) {
  Matrix<T, M, M> result = A;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) -= B[i];
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixSubDiagMatrix::compute<T, M>(B, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Diag Matrix multiply Scalar */
namespace DiagMatrixMultiplyScalar {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &mat, const T scalar,
                      DiagMatrix<T, M> &result) {
    Core<T, M, Start, Mid>::compute(mat, scalar, result);
    Core<T, M, Mid, End>::compute(mat, scalar, result);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, const T, DiagMatrix<T, M> &) {}
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &mat, const T scalar,
                      DiagMatrix<T, M> &result) {
    result[Start] = mat[Start] * scalar;
  }
};

/**
 * @brief Computes the element-wise multiplication of a diagonal matrix by a
 * scalar.
 *
 * This function initiates the recursive computation of the product of a
 * diagonal matrix and a scalar, storing the result in the provided result
 * matrix.
 *
 * @param mat The diagonal matrix to multiply.
 * @param scalar The scalar value to multiply with.
 * @param result The resulting diagonal matrix after multiplication.
 */
template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &mat, const T &scalar,
                    DiagMatrix<T, M> &result) {
  Core<T, M, 0, M>::compute(mat, scalar, result);
}

} // namespace DiagMatrixMultiplyScalar

/**
 * @brief Multiplies a diagonal matrix by a scalar.
 *
 * This function computes the product of a diagonal matrix A and a scalar value,
 * returning a new diagonal matrix that contains the products of the
 * corresponding diagonal elements with the scalar.
 *
 * @param A The diagonal matrix to multiply.
 * @param scalar The scalar value to multiply with.
 * @return DiagMatrix<T, M> The resulting diagonal matrix after multiplication.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator*(const DiagMatrix<T, M> &A, const T &scalar) {
  DiagMatrix<T, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * scalar;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixMultiplyScalar::compute<T, M>(A, scalar, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/**
 * @brief Multiplies a scalar by a diagonal matrix.
 *
 * This function computes the product of a scalar value and a diagonal matrix A,
 * returning a new diagonal matrix that contains the products of the
 * corresponding diagonal elements with the scalar.
 *
 * @param scalar The scalar value to multiply with.
 * @param A The diagonal matrix to multiply.
 * @return DiagMatrix<T, M> The resulting diagonal matrix after multiplication.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator*(const T &scalar, const DiagMatrix<T, M> &A) {
  DiagMatrix<T, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * scalar;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixMultiplyScalar::compute<T, M>(A, scalar, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Diag Matrix multiply Vector */
namespace DiagMatrixMultiplyVector {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &A, const Vector<T, M> &vec,
                      Vector<T, M> &result) {
    Core<T, M, Start, Mid>::compute(A, vec, result);
    Core<T, M, Mid, End>::compute(A, vec, result);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, const Vector<T, M> &,
                      Vector<T, M> &) {}
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &A, const Vector<T, M> &vec,
                      Vector<T, M> &result) {
    result[Start] = A[Start] * vec[Start];
  }
};

/**
 * @brief Computes the multiplication of a diagonal matrix with a vector.
 *
 * This function initiates the recursive computation of the product of a
 * diagonal matrix A and a vector vec, storing the result in the provided
 * result vector.
 *
 * @param A The diagonal matrix.
 * @param vec The vector to multiply with.
 * @param result The resulting vector after multiplication.
 */
template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, const Vector<T, M> &vec,
                    Vector<T, M> &result) {
  Core<T, M, 0, M>::compute(A, vec, result);
}

} // namespace DiagMatrixMultiplyVector

/**
 * @brief Multiplies a diagonal matrix by a vector.
 *
 * This function computes the product of a diagonal matrix A and a vector vec,
 * returning a new vector that contains the products of the corresponding
 * diagonal elements with the vector elements.
 *
 * @param A The diagonal matrix to multiply.
 * @param vec The vector to multiply with.
 * @return Vector<T, M> The resulting vector after multiplication.
 */
template <typename T, std::size_t M>
inline Vector<T, M> operator*(const DiagMatrix<T, M> &A,
                              const Vector<T, M> &vec) {
  Vector<T, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * vec[i];
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixMultiplyVector::compute<T, M>(A, vec, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Matrix Multiplication */
namespace DiagMatrixMultiplyDiagMatrix {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                      DiagMatrix<T, M> &result) {
    Core<T, M, Start, Mid>::compute(A, B, result);
    Core<T, M, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, const DiagMatrix<T, M> &,
                      DiagMatrix<T, M> &) {}
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                      DiagMatrix<T, M> &result) {
    result[Start] = A[Start] * B[Start];
  }
};

/**
 * @brief Computes the element-wise multiplication of two diagonal matrices.
 *
 * This function initiates the recursive computation of the product of two
 * diagonal matrices A and B, storing the result in the provided result matrix.
 *
 * @param A The first diagonal matrix.
 * @param B The second diagonal matrix.
 * @param result The resulting diagonal matrix after multiplication.
 */
template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                    DiagMatrix<T, M> &result) {
  Core<T, M, 0, M>::compute(A, B, result);
}

} // namespace DiagMatrixMultiplyDiagMatrix

/**
 * @brief Multiplies two diagonal matrices element-wise.
 *
 * This function computes the product of two diagonal matrices A and B,
 * returning a new diagonal matrix that contains the products of the
 * corresponding diagonal elements.
 *
 * @param A The first diagonal matrix.
 * @param B The second diagonal matrix.
 * @return DiagMatrix<T, M> The resulting diagonal matrix after multiplication.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> operator*(const DiagMatrix<T, M> &A,
                                  const DiagMatrix<T, M> &B) {
  DiagMatrix<T, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] * B[j];
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixMultiplyDiagMatrix::compute<T, M>(A, B, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Diag Matrix Multiply Matrix */
namespace DiagMatrixMultiplyMatrix {

template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End>
struct Core<T, M, N, J, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Core<T, M, N, J, Start, Mid>::compute(A, B, result);
    Core<T, M, N, J, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End>
struct Core<T, M, N, J, Start, End,
            typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, const Matrix<T, M, N> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End>
struct Core<T, M, N, J, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    result.template set<J, Start>(A[J] * B.template get<J, Start>());
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Row<T, M, N, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Row<T, M, N, Start, Mid>::compute(A, B, result);
    Row<T, M, N, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Row<T, M, N, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, const Matrix<T, M, N> &,
                      Matrix<T, M, N> &) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Row<T, M, N, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Core<T, M, N, Start, 0, N>::compute(A, B, result);
  }
};

/**
 * @brief Computes the multiplication of a diagonal matrix with a dense matrix.
 *
 * This function initiates the recursive computation of the product of a
 * diagonal matrix A and a dense matrix B, storing the result in the provided
 * result matrix.
 *
 * @param A The diagonal matrix.
 * @param B The dense matrix.
 * @param result The resulting dense matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                    Matrix<T, M, N> &result) {
  Row<T, M, N, 0, M>::compute(A, B, result);
}

} // namespace DiagMatrixMultiplyMatrix

/**
 * @brief Multiplies a diagonal matrix by a dense matrix.
 *
 * This function computes the product of a diagonal matrix A and a dense matrix
 * B, returning a new dense matrix that contains the products of the
 * corresponding diagonal elements with the dense matrix elements.
 *
 * @param A The diagonal matrix to multiply.
 * @param B The dense matrix to multiply with.
 * @return Matrix<T, M, N> The resulting dense matrix after multiplication.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> operator*(const DiagMatrix<T, M> &A,
                                 const Matrix<T, M, N> &B) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = A[j] * B(j, k);
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixMultiplyMatrix::compute<T, M, N>(A, B, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

namespace MatrixMultiplyDiagMatrix {

template <typename T, std::size_t L, std::size_t M, std::size_t I,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Core;

template <typename T, std::size_t L, std::size_t M, std::size_t I,
          std::size_t Start, std::size_t End>
struct Core<T, L, M, I, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {
    Core<T, L, M, I, Start, Mid>::compute(A, B, result);
    Core<T, L, M, I, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t L, std::size_t M, std::size_t I,
          std::size_t Start, std::size_t End>
struct Core<T, L, M, I, Start, End,
            typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, L, M> &, const DiagMatrix<T, M> &,
                      Matrix<T, L, M> &) {}
};

template <typename T, std::size_t L, std::size_t M, std::size_t I,
          std::size_t Start, std::size_t End>
struct Core<T, L, M, I, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {
    result.template set<I, Start>(A.template get<I, Start>() * B[Start]);
  }
};

template <typename T, std::size_t L, std::size_t M, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t L, std::size_t M, std::size_t Start,
          std::size_t End>
struct Row<T, L, M, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {
    Row<T, L, M, Start, Mid>::compute(A, B, result);
    Row<T, L, M, Mid, End>::compute(A, B, result);
  }
};

template <typename T, std::size_t L, std::size_t M, std::size_t Start,
          std::size_t End>
struct Row<T, L, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const Matrix<T, L, M> &, const DiagMatrix<T, M> &,
                      Matrix<T, L, M> &) {}
};

template <typename T, std::size_t L, std::size_t M, std::size_t Start,
          std::size_t End>
struct Row<T, L, M, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {
    Core<T, L, M, Start, 0, M>::compute(A, B, result);
  }
};

/**
 * @brief Computes the multiplication of a dense matrix with a diagonal matrix.
 *
 * This function initiates the recursive computation of the product of a dense
 * matrix A and a diagonal matrix B, storing the result in the provided result
 * matrix.
 *
 * @param A The dense matrix.
 * @param B The diagonal matrix.
 * @param result The resulting dense matrix after multiplication.
 */
template <typename T, std::size_t L, std::size_t M>
inline void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                    Matrix<T, L, M> &result) {
  Row<T, L, M, 0, L>::compute(A, B, result);
}

} // namespace MatrixMultiplyDiagMatrix

/**
 * @brief Multiplies a dense matrix by a diagonal matrix.
 *
 * This function computes the product of a dense matrix A and a diagonal matrix
 * B, returning a new dense matrix that contains the products of the
 * corresponding diagonal elements with the dense matrix elements.
 *
 * @param A The dense matrix to multiply.
 * @param B The diagonal matrix to multiply with.
 * @return Matrix<T, L, M> The resulting dense matrix after multiplication.
 */
template <typename T, std::size_t L, std::size_t M>
inline Matrix<T, L, M> operator*(const Matrix<T, L, M> &A,
                                 const DiagMatrix<T, M> &B) {
  Matrix<T, L, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t j = 0; j < L; ++j) {
    for (std::size_t k = 0; k < M; ++k) {
      result(j, k) = A(j, k) * B[k];
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  MatrixMultiplyDiagMatrix::compute<T, L, M>(A, B, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Trace */
namespace DiagMatrixTrace {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static T compute(const DiagMatrix<T, M> &A) {
    return Core<T, M, Start, Mid>::compute(A) +
           Core<T, M, Mid, End>::compute(A);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static T compute(const DiagMatrix<T, M> &) { return static_cast<T>(0); }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static T compute(const DiagMatrix<T, M> &A) { return A[Start]; }
};

/**
 * @brief Computes the trace of a diagonal matrix.
 *
 * This function initiates the recursive computation of the trace of a diagonal
 * matrix A by summing its diagonal elements.
 *
 * @param A The diagonal matrix.
 * @return T The computed trace of the diagonal matrix.
 */
template <typename T, std::size_t M>
inline T compute(const DiagMatrix<T, M> &A) {
  return Core<T, M, 0, M>::compute(A);
}

} // namespace DiagMatrixTrace

/**
 * @brief Computes the trace of a diagonal matrix.
 *
 * This function calculates the trace of a diagonal matrix A by summing its
 * diagonal elements, returning the computed trace value.
 *
 * @param A The diagonal matrix.
 * @return T The computed trace of the diagonal matrix.
 */
template <typename T, std::size_t M>
inline T output_trace(const DiagMatrix<T, M> &A) {
  T trace = static_cast<T>(0);

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; i++) {
    trace += A[i];
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  trace = DiagMatrixTrace::compute<T, M>(A);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return trace;
}

/* Create dense */
namespace DiagMatrixToDense {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void assign(Matrix<T, M, M> &result, const DiagMatrix<T, M> &A) {
    Core<T, M, Start, Mid>::assign(result, A);
    Core<T, M, Mid, End>::assign(result, A);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void assign(Matrix<T, M, M> &, const DiagMatrix<T, M> &) {}
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void assign(Matrix<T, M, M> &result, const DiagMatrix<T, M> &A) {
    result.template set<Start, Start>(A[Start]);
  }
};

/**
 * @brief Computes the dense matrix representation of a diagonal matrix.
 *
 * This function initiates the recursive assignment of diagonal elements from a
 * diagonal matrix A to a dense matrix result, effectively creating a dense
 * representation of the diagonal matrix.
 *
 * @param A The diagonal matrix to convert to dense format.
 * @param result The resulting dense matrix after conversion.
 */
template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
  Core<T, M, 0, M>::assign(result, A);
}

} // namespace DiagMatrixToDense

/**
 * @brief Converts a diagonal matrix to a dense matrix.
 *
 * This function creates a dense matrix representation of a diagonal matrix A,
 * where the diagonal elements of A are assigned to the corresponding diagonal
 * positions in the resulting dense matrix.
 *
 * @param A The diagonal matrix to convert.
 * @return Matrix<T, M, M> The resulting dense matrix after conversion.
 */
template <typename T, std::size_t M>
inline Matrix<T, M, M> output_dense_matrix(const DiagMatrix<T, M> &A) {
  Matrix<T, M, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; i++) {
    result(i, i) = A[i];
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixToDense::compute<T, M>(A, result);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/* Diag Matrix divide Diag Matrix */
namespace DiagMatrixDivideDiagMatrix {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Core;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                      DiagMatrix<T, M> &result, const T division_min) {
    Core<T, M, Start, Mid>::compute(A, B, result, division_min);
    Core<T, M, Mid, End>::compute(A, B, result, division_min);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, const DiagMatrix<T, M> &,
                      DiagMatrix<T, M> &, const T) {}
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Core<T, M, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                      DiagMatrix<T, M> &result, const T division_min) {
    result[Start] =
        A[Start] / Base::Utility::avoid_zero_divide(B[Start], division_min);
  }
};

/**
 * @brief Computes the element-wise division of two diagonal matrices.
 *
 * This function initiates the recursive computation of the division of two
 * diagonal matrices A and B, storing the result in the provided result matrix.
 *
 * @param A The first diagonal matrix.
 * @param B The second diagonal matrix.
 * @param result The resulting diagonal matrix after division.
 * @param division_min The minimum value to avoid division by zero.
 */
template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                    DiagMatrix<T, M> &result, const T division_min) {
  Core<T, M, 0, M>::compute(A, B, result, division_min);
}

} // namespace DiagMatrixDivideDiagMatrix

/**
 * @brief Divides two diagonal matrices element-wise.
 *
 * This function computes the element-wise division of two diagonal matrices A
 * and B, returning a new diagonal matrix that contains the results of the
 * division of the corresponding diagonal elements.
 *
 * @param A The first diagonal matrix.
 * @param B The second diagonal matrix.
 * @param division_min The minimum value to avoid division by zero.
 * @return DiagMatrix<T, M> The resulting diagonal matrix after division.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> diag_divide_diag(const DiagMatrix<T, M> &A,
                                         const DiagMatrix<T, M> &B,
                                         const T division_min) {
  DiagMatrix<T, M> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] / Base::Utility::avoid_zero_divide(B[j], division_min);
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixDivideDiagMatrix::compute<T, M>(A, B, result, division_min);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/**
 * @brief Divides two diagonal matrices element-wise with partitioning.
 *
 * This function computes the element-wise division of two diagonal matrices A
 * and B, returning a new diagonal matrix that contains the results of the
 * division of the corresponding diagonal elements, limited to a specified
 * matrix size.
 *
 * @param A The first diagonal matrix.
 * @param B The second diagonal matrix.
 * @param division_min The minimum value to avoid division by zero.
 * @param matrix_size The size of the matrices to consider for the operation.
 * @return DiagMatrix<T, M> The resulting diagonal matrix after division.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M>
diag_divide_diag_partition(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> &B,
                           const T &division_min,
                           const std::size_t &matrix_size) {
  DiagMatrix<T, M> result;

  for (std::size_t j = 0; j < matrix_size; ++j) {
    result[j] = A[j] / Base::Utility::avoid_zero_divide(B[j], division_min);
  }

  return result;
}

/* Diag Matrix Inverse multiply Matrix */
namespace DiagMatrixInverseMultiplyMatrix {

template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End, typename Enable = void>
struct Row;

template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, J, Start, End,
           typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    Row<T, M, N, J, Start, Mid>::compute(A, B, result, division_min);
    Row<T, M, N, J, Mid, End>::compute(A, B, result, division_min);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, J, Start, End,
           typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, const Matrix<T, M, N> &,
                      Matrix<T, M, N> &, const T) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t Start, std::size_t End>
struct Row<T, M, N, J, Start, End,
           typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    result(J, Start) =
        B(J, Start) / Base::Utility::avoid_zero_divide(A[J], division_min);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End, typename Enable = void>
struct Column;

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    Column<T, M, N, Start, Mid>::compute(A, B, result, division_min);
    Column<T, M, N, Mid, End>::compute(A, B, result, division_min);
  }
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, const Matrix<T, M, N> &,
                      Matrix<T, M, N> &, const T) {}
};

template <typename T, std::size_t M, std::size_t N, std::size_t Start,
          std::size_t End>
struct Column<T, M, N, Start, End,
              typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    Row<T, M, N, Start, 0, N>::compute(A, B, result, division_min);
  }
};

/**
 * @brief Computes the element-wise division of a dense matrix by a diagonal
 * matrix.
 *
 * This function initiates the recursive computation of the division of each
 * element in a dense matrix B by the corresponding diagonal element in a
 * diagonal matrix A, storing the result in the provided result matrix.
 *
 * @param A The diagonal matrix.
 * @param B The dense matrix to divide.
 * @param result The resulting dense matrix after division.
 * @param division_min The minimum value to avoid division by zero.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                    Matrix<T, M, N> &result, T division_min) {
  Column<T, M, N, 0, M>::compute(A, B, result, division_min);
}

} // namespace DiagMatrixInverseMultiplyMatrix

/**
 * @brief Divides each element of a dense matrix by the corresponding diagonal
 * element of a diagonal matrix.
 *
 * This function computes the element-wise division of each element in a dense
 * matrix B by the corresponding diagonal element in a diagonal matrix A,
 * returning a new dense matrix that contains the results of the division.
 *
 * @param A The diagonal matrix.
 * @param B The dense matrix to divide.
 * @param division_min The minimum value to avoid division by zero.
 * @return Matrix<T, M, N> The resulting dense matrix after division.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> diag_inv_multiply_dense(const DiagMatrix<T, M> &A,
                                               const Matrix<T, M, N> &B,
                                               const T division_min) {
  Matrix<T, M, N> result;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) =
          B(j, k) / Base::Utility::avoid_zero_divide(A[j], division_min);
    }
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixInverseMultiplyMatrix::compute<T, M, N>(A, B, result, division_min);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return result;
}

/**
 * @brief Divides each element of a dense matrix by the corresponding diagonal
 * element of a diagonal matrix, with partitioning.
 *
 * This function computes the element-wise division of each element in a dense
 * matrix B by the corresponding diagonal element in a diagonal matrix A,
 * returning a new dense matrix that contains the results of the division,
 * limited to a specified matrix size.
 *
 * @param A The diagonal matrix.
 * @param B The dense matrix to divide.
 * @param division_min The minimum value to avoid division by zero.
 * @param matrix_size The size of the matrices to consider for the operation.
 * @return Matrix<T, M, N> The resulting dense matrix after division.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> diag_inv_multiply_dense_partition(
    const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B, const T &division_min,
    const std::size_t &matrix_size) {
  Matrix<T, M, N> result;

  for (std::size_t j = 0; j < matrix_size; ++j) {
    for (std::size_t k = 0; k < matrix_size; ++k) {
      result(j, k) =
          B(j, k) / Base::Utility::avoid_zero_divide(A[j], division_min);
    }
  }

  return result;
}

/* Convert Real Matrix to Complex */
namespace DiagMatrixRealToComplex {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct DiagMatrixRealToComplexLoop;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct DiagMatrixRealToComplexLoop<
    T, M, Start, End, typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<T, M> &From_matrix,
                      DiagMatrix<Complex<T>, M> &To_matrix) {
    DiagMatrixRealToComplexLoop<T, M, Start, Mid>::compute(From_matrix,
                                                           To_matrix);
    DiagMatrixRealToComplexLoop<T, M, Mid, End>::compute(From_matrix,
                                                         To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct DiagMatrixRealToComplexLoop<
    T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<T, M> &, DiagMatrix<Complex<T>, M> &) {}
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct DiagMatrixRealToComplexLoop<
    T, M, Start, End, typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<T, M> &From_matrix,
                      DiagMatrix<Complex<T>, M> &To_matrix) {
    To_matrix[Start].real = From_matrix[Start];
  }
};

/**
 * @brief Converts a real diagonal matrix to a complex diagonal matrix.
 *
 * This function initiates the recursive conversion of a real diagonal matrix
 * to a complex diagonal matrix, where each element is converted to a complex
 * number with the real part preserved and the imaginary part set to zero.
 *
 * @param From_matrix The real diagonal matrix to convert.
 * @param To_matrix The resulting complex diagonal matrix after conversion.
 */
template <typename T, std::size_t M>
inline void compute(const DiagMatrix<T, M> &From_matrix,
                    DiagMatrix<Complex<T>, M> &To_matrix) {
  DiagMatrixRealToComplexLoop<T, M, 0, M>::compute(From_matrix, To_matrix);
}

} // namespace DiagMatrixRealToComplex

/**
 * @brief Converts a real diagonal matrix to a complex diagonal matrix.
 *
 * This function creates a complex diagonal matrix from a real diagonal matrix
 * A, where each element of A is converted to a complex number with the real
 * part preserved and the imaginary part set to zero.
 *
 * @param From_matrix The real diagonal matrix to convert.
 * @return DiagMatrix<Complex<T>, M> The resulting complex diagonal matrix after
 * conversion.
 */
template <typename T, std::size_t M>
inline DiagMatrix<Complex<T>, M>
convert_matrix_real_to_complex(const DiagMatrix<T, M> &From_matrix) {

  DiagMatrix<Complex<T>, M> To_matrix;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    To_matrix[i].real = From_matrix[i];
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  DiagMatrixRealToComplex::compute<T, M>(From_matrix, To_matrix);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return To_matrix;
}

/* Get Real Matrix from Complex */
namespace GetRealDiagMatrixFromComplex {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Loop;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Loop<T, M, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                      DiagMatrix<T, M> &To_matrix) {
    Loop<T, M, Start, Mid>::compute(From_matrix, To_matrix);
    Loop<T, M, Mid, End>::compute(From_matrix, To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Loop<T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<Complex<T>, M> &, DiagMatrix<T, M> &) {}
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Loop<T, M, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                      DiagMatrix<T, M> &To_matrix) {
    To_matrix[Start] = From_matrix[Start].real;
  }
};

/**
 * @brief Extracts the real part of a complex diagonal matrix and stores it in a
 * real diagonal matrix.
 *
 * This function initiates the recursive extraction of the real part of each
 * element in a complex diagonal matrix and stores it in a corresponding
 * position in a real diagonal matrix.
 *
 * @param From_matrix The complex diagonal matrix to extract from.
 * @param To_matrix The resulting real diagonal matrix after extraction.
 */
template <typename T, std::size_t M>
inline void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                    DiagMatrix<T, M> &To_matrix) {
  Loop<T, M, 0, M>::compute(From_matrix, To_matrix);
}

} // namespace GetRealDiagMatrixFromComplex

/**
 * @brief Extracts the real part of a complex diagonal matrix and stores it in a
 * real diagonal matrix.
 *
 * This function creates a real diagonal matrix from a complex diagonal matrix
 * A, where each element of A is converted to its real part.
 *
 * @param From_matrix The complex diagonal matrix to extract from.
 * @return DiagMatrix<T, M> The resulting real diagonal matrix after extraction.
 */
template <typename T, std::size_t M>
inline auto get_real_matrix_from_complex_matrix(
    const DiagMatrix<Complex<T>, M> &From_matrix) -> DiagMatrix<T, M> {

  DiagMatrix<T, M> To_matrix;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    To_matrix[i] = From_matrix[i].real;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  GetRealDiagMatrixFromComplex::compute<T, M>(From_matrix, To_matrix);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return To_matrix;
}

/* Get Imag Matrix from Complex */
namespace GetImagDiagMatrixFromComplex {

template <typename T, std::size_t M, std::size_t Start, std::size_t End,
          typename Enable = void>
struct Loop;

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Loop<T, M, Start, End,
            typename std::enable_if<(End - Start > 1)>::type> {
  static constexpr std::size_t Mid = Start + (End - Start) / 2;
  static void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                      DiagMatrix<T, M> &To_matrix) {
    Loop<T, M, Start, Mid>::compute(From_matrix, To_matrix);
    Loop<T, M, Mid, End>::compute(From_matrix, To_matrix);
  }
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Loop<T, M, Start, End, typename std::enable_if<(End == Start)>::type> {
  static void compute(const DiagMatrix<Complex<T>, M> &, DiagMatrix<T, M> &) {}
};

template <typename T, std::size_t M, std::size_t Start, std::size_t End>
struct Loop<T, M, Start, End,
            typename std::enable_if<(End - Start == 1)>::type> {
  static void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                      DiagMatrix<T, M> &To_matrix) {
    To_matrix[Start] = From_matrix[Start].imag;
  }
};

/**
 * @brief Extracts the imaginary part of a complex diagonal matrix and stores it
 * in a real diagonal matrix.
 *
 * This function initiates the recursive extraction of the imaginary part of
 * each element in a complex diagonal matrix and stores it in a corresponding
 * position in a real diagonal matrix.
 *
 * @param From_matrix The complex diagonal matrix to extract from.
 * @param To_matrix The resulting real diagonal matrix after extraction.
 */
template <typename T, std::size_t M>
inline void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                    DiagMatrix<T, M> &To_matrix) {
  Loop<T, M, 0, M>::compute(From_matrix, To_matrix);
}

} // namespace GetImagDiagMatrixFromComplex

/**
 * @brief Extracts the imaginary part of a complex diagonal matrix and stores it
 * in a real diagonal matrix.
 *
 * This function creates a real diagonal matrix from a complex diagonal matrix
 * A, where each element of A is converted to its imaginary part.
 *
 * @param From_matrix The complex diagonal matrix to extract from.
 * @return DiagMatrix<T, M> The resulting real diagonal matrix after extraction.
 */
template <typename T, std::size_t M>
inline auto get_imag_matrix_from_complex_matrix(
    const DiagMatrix<Complex<T>, M> &From_matrix) -> DiagMatrix<T, M> {

  DiagMatrix<T, M> To_matrix;

#ifdef BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  for (std::size_t i = 0; i < M; ++i) {
    To_matrix[i] = From_matrix[i].imag;
  }

#else // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  GetImagDiagMatrixFromComplex::compute<T, M>(From_matrix, To_matrix);

#endif // BASE_MATRIX_USE_FOR_LOOP_OPERATION_

  return To_matrix;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_DIAGONAL_HPP_
