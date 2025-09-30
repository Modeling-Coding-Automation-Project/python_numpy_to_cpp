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
 * tparam M is the number of columns in the matrix.
 * tparam N is the number of rows in the matrix.
 * Somehow Programming custom is vice versa,
 * but in this project, we use the mathematical custom.
 */
#ifndef __BASE_MATRIX_DIAGONAL_HPP__
#define __BASE_MATRIX_DIAGONAL_HPP__

#include "base_matrix_macros.hpp"

#include "base_matrix_matrix.hpp"
#include "base_matrix_vector.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
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
 * __BASE_MATRIX_USE_STD_VECTOR__ macro.
 *
 * @tparam T Type of the matrix elements.
 * @tparam M Size of the matrix (number of rows and columns).
 *
 * @note Only the diagonal elements are stored and manipulated.
 */
template <typename T, std::size_t M> class DiagMatrix {
public:
  /* Constant */
  static constexpr std::size_t COLS = M;
  static constexpr std::size_t ROWS = M;

public:
#ifdef __BASE_MATRIX_USE_STD_VECTOR__

  DiagMatrix() : data(M, static_cast<T>(0)) {}

  DiagMatrix(const std::vector<T> &input) : data(input) {}

  DiagMatrix(const std::initializer_list<T> &input) : data(input) {}

  DiagMatrix(T input[M]) : data(M, static_cast<T>(0)) {

    for (std::size_t i = 0; i < M; i++) {
      this->data[i] = input[i];
    }
  }

#else // __BASE_MATRIX_USE_STD_VECTOR__

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

#endif // __BASE_MATRIX_USE_STD_VECTOR__

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
   * @tparam M The size (number of rows and columns) of the square matrix.
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
   * @tparam M The size (number of rows and columns) of the square matrix.
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
   * @tparam M The size (number of rows and columns) of the square matrix.
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
  constexpr std::size_t rows() const { return M; }

  /**
   * @brief Returns the number of columns in the diagonal matrix.
   *
   * Since this is a square matrix, the number of columns is equal to the
   * number of rows.
   *
   * @return std::size_t The number of columns in the diagonal matrix.
   */
  constexpr std::size_t cols() const { return M; }

  /**
   * @brief Returns a vector representing the specified row of the diagonal
   * matrix.
   *
   * For a diagonal matrix, only the diagonal element at the given row index is
   * non-zero. If the provided row index is out of bounds (greater than or equal
   * to M), it is clamped to the last valid index.
   *
   * @param row The index of the row to retrieve.
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
   * @param col The index of the column to retrieve.
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
#ifdef __BASE_MATRIX_USE_STD_VECTOR__
  std::vector<T> data;
#else  // __BASE_MATRIX_USE_STD_VECTOR__
  std::array<T, M> data;
#endif // __BASE_MATRIX_USE_STD_VECTOR__
};

/**
 * @brief Converts a diagonal matrix to a column vector.
 *
 * This function takes a diagonal matrix as input and returns a column vector
 * containing the diagonal elements of the matrix.
 *
 * @tparam T The type of the matrix elements.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
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

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  /**
   * @brief Computes the element-wise addition of two diagonal matrices.
   *
   * This function recursively computes the sum of two diagonal matrices A and
   * B, storing the result in the provided result matrix.
   *
   * @param A The first diagonal matrix.
   * @param B The second diagonal matrix.
   * @param result The resulting diagonal matrix after addition.
   */
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[M_idx] = A[M_idx] + B[M_idx];
    Core<T, M, M_idx - 1>::compute(A, B, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  /**
   * @brief Computes the element-wise addition of two diagonal matrices.
   *
   * This function serves as the base case for the recursive addition of two
   * diagonal matrices A and B, storing the result in the provided result
   * matrix.
   *
   * @param A The first diagonal matrix.
   * @param B The second diagonal matrix.
   * @param result The resulting diagonal matrix after addition.
   */
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[0] = A[0] + B[0];
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
  Core<T, M, M - 1>::compute(A, B, result);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] + B[j];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixAddDiagMatrix::compute<T, M>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

namespace DiagMatrixAddMatrix {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  /**
   * @brief Computes the element-wise addition of a diagonal matrix and a dense
   * matrix.
   *
   * This function recursively computes the sum of a diagonal matrix A and a
   * dense matrix B, storing the result in the provided result matrix.
   *
   * @param A The diagonal matrix.
   * @param B The dense matrix.
   * @param result The resulting dense matrix after addition.
   */
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {

    result.template set<M_idx, M_idx>(result.template get<M_idx, M_idx>() +
                                      A[M_idx]);
    Core<T, M, M_idx - 1>::compute(A, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  /**
   * @brief Computes the element-wise addition of a diagonal matrix and a dense
   * matrix.
   *
   * This function serves as the base case for the recursive addition of a
   * diagonal matrix A and a dense matrix B, storing the result in the provided
   * result matrix.
   *
   * @param A The diagonal matrix.
   * @param result The resulting dense matrix after addition.
   */
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {

    result.template set<0, 0>(result.template get<0, 0>() + A[0]);
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
  Core<T, M, M - 1>::compute(A, result);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixAddMatrix::compute<T, M>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += B[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixAddMatrix::compute<T, M>(B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Minus */
namespace DiagMatrixMinus {

// when I_idx < M
template <typename T, std::size_t M, std::size_t I_idx> struct Loop {
  /**
   * @brief Computes the element-wise negation of a diagonal matrix.
   *
   * This function recursively computes the negation of a diagonal matrix A,
   * storing the result in the provided result matrix.
   *
   * @param A The diagonal matrix to negate.
   * @param result The resulting diagonal matrix after negation.
   */
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> &result) {
    result[I_idx] = -A[I_idx];
    Loop<T, M, I_idx - 1>::compute(A, result);
  }
};

// row recursion termination
template <typename T, std::size_t M> struct Loop<T, M, 0> {
  /**
   * @brief Computes the element-wise negation of a diagonal matrix.
   *
   * This function serves as the base case for the recursive negation of a
   * diagonal matrix A, storing the result in the provided result matrix.
   *
   * @param A The diagonal matrix to negate.
   * @param result The resulting diagonal matrix after negation.
   */
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> &result) {
    result[0] = -A[0];
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
  Loop<T, M, M - 1>::compute(A, result);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = -A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixMinus::compute<T, M>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix Subtraction */
namespace DiagMatrixSubDiagMatrix {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  /**
   * @brief Computes the element-wise subtraction of two diagonal matrices.
   *
   * This function recursively computes the difference between two diagonal
   * matrices A and B, storing the result in the provided result matrix.
   *
   * @param A The first diagonal matrix.
   * @param B The second diagonal matrix.
   * @param result The resulting diagonal matrix after subtraction.
   */
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[M_idx] = A[M_idx] - B[M_idx];
    Core<T, M, M_idx - 1>::compute(A, B, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  /**
   * @brief Computes the element-wise subtraction of two diagonal matrices.
   *
   * This function serves as the base case for the recursive subtraction of two
   * diagonal matrices A and B, storing the result in the provided result
   * matrix.
   *
   * @param A The first diagonal matrix.
   * @param B The second diagonal matrix.
   * @param result The resulting diagonal matrix after subtraction.
   */
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[0] = A[0] - B[0];
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
  Core<T, M, M - 1>::compute(A, B, result);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] - B[j];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixSubDiagMatrix::compute<T, M>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

namespace DiagMatrixSubMatrix {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  /**
   * @brief Computes the element-wise subtraction of a diagonal matrix from a
   * dense matrix.
   *
   * This function recursively computes the difference between a dense matrix A
   * and a diagonal matrix B, storing the result in the provided result matrix.
   *
   * @param A The dense matrix.
   * @param B The diagonal matrix.
   * @param result The resulting dense matrix after subtraction.
   */
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {

    result.template set<M_idx, M_idx>(result.template get<M_idx, M_idx>() +
                                      A[M_idx]);
    Core<T, M, M_idx - 1>::compute(A, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  /**
   * @brief Computes the element-wise subtraction of a diagonal matrix from a
   * dense matrix.
   *
   * This function serves as the base case for the recursive subtraction of a
   * diagonal matrix A from a dense matrix, storing the result in the provided
   * result matrix.
   *
   * @param A The diagonal matrix.
   * @param result The resulting dense matrix after subtraction.
   */
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {

    result.template set<0, 0>(result.template get<0, 0>() + A[0]);
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
inline void compute(const Matrix<T, M, M> &A, Matrix<T, M, M> &result) {
  Core<T, M, M - 1>::compute(A, result);
}

} // namespace DiagMatrixSubMatrix

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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) += A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixAddMatrix::compute<T, M>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

namespace MatrixSubDiagMatrix {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  /**
   * @brief Computes the element-wise subtraction of a diagonal matrix from a
   * dense matrix.
   *
   * This function recursively computes the difference between a dense matrix A
   * and a diagonal matrix B, storing the result in the provided result matrix.
   *
   * @param A The diagonal matrix.
   * @param result The resulting dense matrix after subtraction.
   */
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {

    result.template set<M_idx, M_idx>(result.template get<M_idx, M_idx>() -
                                      A[M_idx]);
    Core<T, M, M_idx - 1>::compute(A, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  /**
   * @brief Computes the element-wise subtraction of a diagonal matrix from a
   * dense matrix.
   *
   * This function serves as the base case for the recursive subtraction of a
   * diagonal matrix A from a dense matrix, storing the result in the provided
   * result matrix.
   *
   * @param A The diagonal matrix.
   * @param result The resulting dense matrix after subtraction.
   */
  static void compute(const DiagMatrix<T, M> &A, Matrix<T, M, M> &result) {
    result.template set<0, 0>(result.template get<0, 0>() - A[0]);
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
  Core<T, M, M - 1>::compute(A, result);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result(i, i) -= B[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixSubDiagMatrix::compute<T, M>(B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Diag Matrix multiply Scalar */
namespace DiagMatrixMultiplyScalar {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  /**
   * @brief Computes the element-wise multiplication of a diagonal matrix by a
   * scalar.
   *
   * This function recursively computes the product of a diagonal matrix and a
   * scalar, storing the result in the provided result matrix.
   *
   * @param mat The diagonal matrix to multiply.
   * @param scalar The scalar value to multiply with.
   * @param result The resulting diagonal matrix after multiplication.
   */
  static void compute(const DiagMatrix<T, M> &mat, const T scalar,
                      DiagMatrix<T, M> &result) {
    result[M_idx] = mat[M_idx] * scalar;
    Core<T, M, M_idx - 1>::compute(mat, scalar, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  /**
   * @brief Computes the element-wise multiplication of a diagonal matrix by a
   * scalar.
   *
   * This function serves as the base case for the recursive multiplication of a
   * diagonal matrix and a scalar, storing the result in the provided result
   * matrix.
   *
   * @param mat The diagonal matrix to multiply.
   * @param scalar The scalar value to multiply with.
   * @param result The resulting diagonal matrix after multiplication.
   */
  static void compute(const DiagMatrix<T, M> &mat, const T scalar,
                      DiagMatrix<T, M> &result) {
    result[0] = mat[0] * scalar;
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
  Core<T, M, M - 1>::compute(mat, scalar, result);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * scalar;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixMultiplyScalar::compute<T, M>(A, scalar, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * scalar;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixMultiplyScalar::compute<T, M>(A, scalar, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Diag Matrix multiply Vector */
namespace DiagMatrixMultiplyVector {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  /**
   * @brief Computes the multiplication of a diagonal matrix with a vector.
   *
   * This function recursively computes the product of a diagonal matrix A and a
   * vector vec, storing the result in the provided result vector.
   *
   * @param A The diagonal matrix.
   * @param vec The vector to multiply with.
   * @param result The resulting vector after multiplication.
   */
  static void compute(const DiagMatrix<T, M> &A, Vector<T, M> vec,
                      Vector<T, M> &result) {
    result[M_idx] = A[M_idx] * vec[M_idx];
    Core<T, M, M_idx - 1>::compute(A, vec, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  /**
   * @brief Computes the multiplication of a diagonal matrix with a vector.
   *
   * This function serves as the base case for the recursive multiplication of a
   * diagonal matrix A and a vector vec, storing the result in the provided
   * result vector.
   *
   * @param A The diagonal matrix.
   * @param vec The vector to multiply with.
   * @param result The resulting vector after multiplication.
   */
  static void compute(const DiagMatrix<T, M> &A, Vector<T, M> vec,
                      Vector<T, M> &result) {
    result[0] = A[0] * vec[0];
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
  Core<T, M, M - 1>::compute(A, vec, result);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    result[i] = A[i] * vec[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixMultiplyVector::compute<T, M>(A, vec, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Matrix Multiplication */
namespace DiagMatrixMultiplyDiagMatrix {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  /**
   * @brief Computes the element-wise multiplication of two diagonal matrices.
   *
   * This function recursively computes the product of two diagonal matrices A
   * and B, storing the result in the provided result matrix.
   *
   * @param A The first diagonal matrix.
   * @param B The second diagonal matrix.
   * @param result The resulting diagonal matrix after multiplication.
   */
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[M_idx] = A[M_idx] * B[M_idx];
    Core<T, M, M_idx - 1>::compute(A, B, result);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  /**
   * @brief Computes the element-wise multiplication of two diagonal matrices.
   *
   * This function serves as the base case for the recursive multiplication of
   * two diagonal matrices A and B, storing the result in the provided result
   * matrix.
   *
   * @param A The first diagonal matrix.
   * @param B The second diagonal matrix.
   * @param result The resulting diagonal matrix after multiplication.
   */
  static void compute(const DiagMatrix<T, M> &A, DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result) {
    result[0] = A[0] * B[0];
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
  Core<T, M, M - 1>::compute(A, B, result);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] * B[j];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixMultiplyDiagMatrix::compute<T, M>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Diag Matrix Multiply Matrix */
namespace DiagMatrixMultiplyMatrix {

// Core multiplication for DiagMatrix and Matrix
template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t K>
struct Core {
  /**
   * @brief Computes the multiplication of a diagonal matrix with a dense
   * matrix.
   *
   * This function recursively computes the product of a diagonal matrix A and a
   * dense matrix B, storing the result in the provided result matrix.
   *
   * @param A The diagonal matrix.
   * @param B The dense matrix.
   * @param result The resulting dense matrix after multiplication.
   */
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<J, K>(A[J] * B.template get<J, K>());
    Core<T, M, N, J, K - 1>::compute(A, B, result);
  }
};

// Specialization for K == 0
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct Core<T, M, N, J, 0> {
  /**
   * @brief Computes the multiplication of a diagonal matrix with a dense
   * matrix.
   *
   * This function serves as the base case for the recursive multiplication of a
   * diagonal matrix A and a dense matrix B, storing the result in the provided
   * result matrix.
   *
   * @param A The diagonal matrix.
   * @param B The dense matrix.
   * @param result The resulting dense matrix after multiplication.
   */
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {

    result.template set<J, 0>(A[J] * B.template get<J, 0>());
  }
};

// Column-wise multiplication
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct Column {
  /**
   * @brief Computes the multiplication of a diagonal matrix with a dense
   * matrix, column by column.
   *
   * This function recursively computes the product of a diagonal matrix A and a
   * dense matrix B, storing the result in the provided result matrix.
   *
   * @param A The diagonal matrix.
   * @param B The dense matrix.
   * @param result The resulting dense matrix after multiplication.
   */
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Core<T, M, N, J, N - 1>::compute(A, B, result);
    Column<T, M, N, J - 1>::compute(A, B, result);
  }
};

// Specialization for J == 0
template <typename T, std::size_t M, std::size_t N> struct Column<T, M, N, 0> {
  /**
   * @brief Computes the multiplication of a diagonal matrix with a dense
   * matrix, starting from the first column.
   *
   * This function serves as the base case for the recursive multiplication of a
   * diagonal matrix A and a dense matrix B, storing the result in the provided
   * result matrix.
   *
   * @param A The diagonal matrix.
   * @param B The dense matrix.
   * @param result The resulting dense matrix after multiplication.
   */
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result) {
    Core<T, M, N, 0, N - 1>::compute(A, B, result);
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
  Column<T, M, N, M - 1>::compute(A, B, result);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) = A[j] * B(j, k);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixMultiplyMatrix::compute<T, M, N>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

namespace MatrixMultiplyDiagMatrix {

// Core multiplication for each element
template <typename T, std::size_t L, std::size_t M, std::size_t I,
          std::size_t J>
struct Core {
  /**
   * @brief Computes the multiplication of a dense matrix with a diagonal
   * matrix.
   *
   * This function recursively computes the product of a dense matrix A and a
   * diagonal matrix B, storing the result in the provided result matrix.
   *
   * @param A The dense matrix.
   * @param B The diagonal matrix.
   * @param result The resulting dense matrix after multiplication.
   */
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {

    result.template set<I, J>(A.template get<I, J>() * B[J]);
    Core<T, L, M, I, J - 1>::compute(A, B, result);
  }
};

// Specialization for J = 0
template <typename T, std::size_t L, std::size_t M, std::size_t I>
struct Core<T, L, M, I, 0> {
  /**
   * @brief Computes the multiplication of a dense matrix with a diagonal
   * matrix.
   *
   * This function serves as the base case for the recursive multiplication of a
   * dense matrix A and a diagonal matrix B, storing the result in the provided
   * result matrix.
   *
   * @param A The dense matrix.
   * @param B The diagonal matrix.
   * @param result The resulting dense matrix after multiplication.
   */
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {

    result.template set<I, 0>(A.template get<I, 0>() * B[0]);
  }
};

// Column-wise multiplication
template <typename T, std::size_t L, std::size_t M, std::size_t I>
struct Column {
  /**
   * @brief Computes the multiplication of a dense matrix with a diagonal
   * matrix, column by column.
   *
   * This function recursively computes the product of a dense matrix A and a
   * diagonal matrix B, storing the result in the provided result matrix.
   *
   * @param A The dense matrix.
   * @param B The diagonal matrix.
   * @param result The resulting dense matrix after multiplication.
   */
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {
    Core<T, L, M, I, M - 1>::compute(A, B, result);
    Column<T, L, M, I - 1>::compute(A, B, result);
  }
};

// Specialization for I = 0
template <typename T, std::size_t L, std::size_t M> struct Column<T, L, M, 0> {
  /**
   * @brief Computes the multiplication of a dense matrix with a diagonal
   * matrix, starting from the first column.
   *
   * This function serves as the base case for the recursive multiplication of a
   * dense matrix A and a diagonal matrix B, storing the result in the provided
   * result matrix.
   *
   * @param A The dense matrix.
   * @param B The diagonal matrix.
   * @param result The resulting dense matrix after multiplication.
   */
  static void compute(const Matrix<T, L, M> &A, const DiagMatrix<T, M> &B,
                      Matrix<T, L, M> &result) {
    Core<T, L, M, 0, M - 1>::compute(A, B, result);
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
  Column<T, L, M, L - 1>::compute(A, B, result);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < L; ++j) {
    for (std::size_t k = 0; k < M; ++k) {
      result(j, k) = A(j, k) * B[k];
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  MatrixMultiplyDiagMatrix::compute<T, L, M>(A, B, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Trace */
namespace DiagMatrixTrace {

// Base case: when index reaches 0
template <typename T, std::size_t M, std::size_t Index> struct Core {
  /**
   * @brief Computes the trace of a diagonal matrix.
   *
   * This function recursively computes the trace of a diagonal matrix A by
   * summing its diagonal elements.
   *
   * @param A The diagonal matrix.
   * @return T The computed trace of the diagonal matrix.
   */
  static T compute(const DiagMatrix<T, M> &A) {
    return A[Index] + Core<T, M, Index - 1>::compute(A);
  }
};

// Specialization for the base case when Index is 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  /**
   * @brief Computes the trace of a diagonal matrix when the index is 0.
   *
   * This function serves as the base case for the recursive computation of the
   * trace of a diagonal matrix A, returning its first diagonal element.
   *
   * @param A The diagonal matrix.
   * @return T The first diagonal element of the diagonal matrix.
   */
  static T compute(const DiagMatrix<T, M> &A) { return A[0]; }
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
  return Core<T, M, M - 1>::compute(A);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; i++) {
    trace += A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  trace = DiagMatrixTrace::compute<T, M>(A);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return trace;
}

/* Create dense */
namespace DiagMatrixToDense {

// Diagonal element assignment core
template <typename T, std::size_t M, std::size_t I> struct Core {
  /**
   * @brief Assigns the diagonal elements of a diagonal matrix to a dense
   * matrix.
   *
   * This function recursively assigns the diagonal elements of a diagonal
   * matrix A to the corresponding positions in a dense matrix result.
   *
   * @param result The resulting dense matrix.
   * @param A The diagonal matrix from which to assign elements.
   */
  static void assign(Matrix<T, M, M> &result, const DiagMatrix<T, M> &A) {

    result.template set<I, I>(A[I]);
    Core<T, M, I - 1>::assign(result, A);
  }
};

// Base case for recursion termination
template <typename T, std::size_t M> struct Core<T, M, 0> {
  /**
   * @brief Assigns the first diagonal element of a diagonal matrix to a dense
   * matrix.
   *
   * This function serves as the base case for the recursive assignment of
   * diagonal elements from a diagonal matrix A to a dense matrix result.
   *
   * @param result The resulting dense matrix.
   * @param A The diagonal matrix from which to assign the first element.
   */
  static void assign(Matrix<T, M, M> &result, const DiagMatrix<T, M> &A) {
    result.template set<0, 0>(A[0]);
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
  Core<T, M, M - 1>::assign(result, A);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; i++) {
    result(i, i) = A[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixToDense::compute<T, M>(A, result);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return result;
}

/* Diag Matrix divide Diag Matrix */
namespace DiagMatrixDivideDiagMatrix {

// M_idx < M
template <typename T, std::size_t M, std::size_t M_idx> struct Core {
  /**
   * @brief Computes the element-wise division of two diagonal matrices.
   *
   * This function recursively computes the division of two diagonal matrices A
   * and B, storing the result in the provided result matrix.
   *
   * @param A The first diagonal matrix.
   * @param B The second diagonal matrix.
   * @param result The resulting diagonal matrix after division.
   * @param division_min The minimum value to avoid division by zero.
   */
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result, const T division_min) {
    result[M_idx] =
        A[M_idx] / Base::Utility::avoid_zero_divide(B[M_idx], division_min);
    Core<T, M, M_idx - 1>::compute(A, B, result, division_min);
  }
};

// Termination condition: M_idx == 0
template <typename T, std::size_t M> struct Core<T, M, 0> {
  /**
   * @brief Computes the element-wise division of two diagonal matrices when the
   * index is 0.
   *
   * This function serves as the base case for the recursive division of two
   * diagonal matrices A and B, storing the result in the provided result
   * matrix.
   *
   * @param A The first diagonal matrix.
   * @param B The second diagonal matrix.
   * @param result The resulting diagonal matrix after division.
   * @param division_min The minimum value to avoid division by zero.
   */
  static void compute(const DiagMatrix<T, M> &A, const DiagMatrix<T, M> B,
                      DiagMatrix<T, M> &result, const T division_min) {
    result[0] = A[0] / Base::Utility::avoid_zero_divide(B[0], division_min);
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
  Core<T, M, M - 1>::compute(A, B, result, division_min);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    result[j] = A[j] / Base::Utility::avoid_zero_divide(B[j], division_min);
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixDivideDiagMatrix::compute<T, M>(A, B, result, division_min);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

// core multiplication for each element
template <typename T, std::size_t M, std::size_t N, std::size_t J,
          std::size_t K>
struct Column {
  /**
   * @brief Computes the element-wise division of a dense matrix by a diagonal
   * matrix.
   *
   * This function recursively computes the division of each element in a dense
   * matrix B by the corresponding diagonal element in a diagonal matrix A,
   * storing the result in the provided result matrix.
   *
   * @param A The diagonal matrix.
   * @param B The dense matrix to divide.
   * @param result The resulting dense matrix after division.
   * @param division_min The minimum value to avoid division by zero.
   */
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    result(J, K) =
        B(J, K) / Base::Utility::avoid_zero_divide(A[J], division_min);
    Column<T, M, N, J, K - 1>::compute(A, B, result, division_min);
  }
};

// if K == 0
template <typename T, std::size_t M, std::size_t N, std::size_t J>
struct Column<T, M, N, J, 0> {
  /**
   * @brief Computes the element-wise division of the first column of a dense
   * matrix by a diagonal matrix.
   *
   * This function serves as the base case for the recursive division of each
   * element in the first column of a dense matrix B by the corresponding
   * diagonal element in a diagonal matrix A, storing the result in the
   * provided result matrix.
   *
   * @param A The diagonal matrix.
   * @param B The dense matrix to divide.
   * @param result The resulting dense matrix after division.
   * @param division_min The minimum value to avoid division by zero.
   */
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    result(J, 0) =
        B(J, 0) / Base::Utility::avoid_zero_divide(A[J], division_min);
  }
};

// Column-wise multiplication
template <typename T, std::size_t M, std::size_t N, std::size_t J> struct Row {
  /**
   * @brief Computes the element-wise division of a dense matrix by a diagonal
   * matrix, row by row.
   *
   * This function recursively computes the division of each element in a dense
   * matrix B by the corresponding diagonal element in a diagonal matrix A,
   * storing the result in the provided result matrix.
   *
   * @param A The diagonal matrix.
   * @param B The dense matrix to divide.
   * @param result The resulting dense matrix after division.
   * @param division_min The minimum value to avoid division by zero.
   */
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    Column<T, M, N, J, N - 1>::compute(A, B, result, division_min);
    Row<T, M, N, J - 1>::compute(A, B, result, division_min);
  }
};

// if J == 0
template <typename T, std::size_t M, std::size_t N> struct Row<T, M, N, 0> {
  /**
   * @brief Computes the element-wise division of the first row of a dense
   * matrix by a diagonal matrix.
   *
   * This function serves as the base case for the recursive division of each
   * element in the first row of a dense matrix B by the corresponding diagonal
   * element in a diagonal matrix A, storing the result in the provided result
   * matrix.
   *
   * @param A The diagonal matrix.
   * @param B The dense matrix to divide.
   * @param result The resulting dense matrix after division.
   * @param division_min The minimum value to avoid division by zero.
   */
  static void compute(const DiagMatrix<T, M> &A, const Matrix<T, M, N> &B,
                      Matrix<T, M, N> &result, const T division_min) {
    Column<T, M, N, 0, N - 1>::compute(A, B, result, division_min);
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
  Row<T, M, N, M - 1>::compute(A, B, result, division_min);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t j = 0; j < M; ++j) {
    for (std::size_t k = 0; k < N; ++k) {
      result(j, k) =
          B(j, k) / Base::Utility::avoid_zero_divide(A[j], division_min);
    }
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixInverseMultiplyMatrix::compute<T, M, N>(A, B, result, division_min);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

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

/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t I>
struct DiagMatrixRealToComplexLoop {
  /**
   * @brief Converts a real diagonal matrix to a complex diagonal matrix.
   *
   * This function recursively converts each element of a real diagonal matrix
   * to a complex diagonal matrix, where the real part is preserved and the
   * imaginary part is set to zero.
   *
   * @param From_matrix The real diagonal matrix to convert.
   * @param To_matrix The resulting complex diagonal matrix after conversion.
   */
  static void compute(const DiagMatrix<T, M> &From_matrix,
                      DiagMatrix<Complex<T>, M> &To_matrix) {
    To_matrix[I].real = From_matrix[I];
    DiagMatrixRealToComplexLoop<T, M, I - 1>::compute(From_matrix, To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M>
struct DiagMatrixRealToComplexLoop<T, M, 0> {
  /**
   * @brief Converts the first element of a real diagonal matrix to a complex
   * diagonal matrix.
   *
   * This function serves as the base case for the recursive conversion of a
   * real diagonal matrix to a complex diagonal matrix, where the first element
   * is converted.
   *
   * @param From_matrix The real diagonal matrix to convert.
   * @param To_matrix The resulting complex diagonal matrix after conversion.
   */
  static void compute(const DiagMatrix<T, M> &From_matrix,
                      DiagMatrix<Complex<T>, M> &To_matrix) {
    To_matrix[0].real = From_matrix[0];
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
  DiagMatrixRealToComplexLoop<T, M, M - 1>::compute(From_matrix, To_matrix);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    To_matrix[i].real = From_matrix[i];
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  DiagMatrixRealToComplex::compute<T, M>(From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Get Real Matrix from Complex */
namespace GetRealDiagMatrixFromComplex {

/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t I> struct Loop {
  /**
   * @brief Extracts the real part of a complex diagonal matrix and stores it in
   * a real diagonal matrix.
   *
   * This function recursively extracts the real part of each element in a
   * complex diagonal matrix and stores it in a corresponding position in a real
   * diagonal matrix.
   *
   * @param From_matrix The complex diagonal matrix to extract from.
   * @param To_matrix The resulting real diagonal matrix after extraction.
   */
  static void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                      DiagMatrix<T, M> &To_matrix) {
    To_matrix[I] = From_matrix[I].real;
    Loop<T, M, I - 1>::compute(From_matrix, To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M> struct Loop<T, M, 0> {
  /**
   * @brief Extracts the real part of the first element of a complex diagonal
   * matrix and stores it in a real diagonal matrix.
   *
   * This function serves as the base case for the recursive extraction of the
   * real part of a complex diagonal matrix, storing the first element in a real
   * diagonal matrix.
   *
   * @param From_matrix The complex diagonal matrix to extract from.
   * @param To_matrix The resulting real diagonal matrix after extraction.
   */
  static void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                      DiagMatrix<T, M> &To_matrix) {
    To_matrix[0] = From_matrix[0].real;
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
inline void compute(const DiagMatrix<Complex<T>, 3> &From_matrix,
                    DiagMatrix<T, 3> &To_matrix) {
  Loop<T, M, M - 1>::compute(From_matrix, To_matrix);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    To_matrix[i] = From_matrix[i].real;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  GetRealDiagMatrixFromComplex::compute<T, M>(From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

/* Get Imag Matrix from Complex */
namespace GetImagDiagMatrixFromComplex {

/* Helper struct for unrolling the loop */
template <typename T, std::size_t M, std::size_t I> struct Loop {
  /**
   * @brief Extracts the imaginary part of a complex diagonal matrix and stores
   * it in a real diagonal matrix.
   *
   * This function recursively extracts the imaginary part of each element in a
   * complex diagonal matrix and stores it in a corresponding position in a real
   * diagonal matrix.
   *
   * @param From_matrix The complex diagonal matrix to extract from.
   * @param To_matrix The resulting real diagonal matrix after extraction.
   */
  static void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                      DiagMatrix<T, M> &To_matrix) {
    To_matrix[I] = From_matrix[I].imag;
    Loop<T, M, I - 1>::compute(From_matrix, To_matrix);
  }
};

/* Specialization to end the recursion */
template <typename T, std::size_t M> struct Loop<T, M, 0> {
  /**
   * @brief Extracts the imaginary part of the first element of a complex
   * diagonal matrix and stores it in a real diagonal matrix.
   *
   * This function serves as the base case for the recursive extraction of the
   * imaginary part of a complex diagonal matrix, storing the first element in a
   * real diagonal matrix.
   *
   * @param From_matrix The complex diagonal matrix to extract from.
   * @param To_matrix The resulting real diagonal matrix after extraction.
   */
  static void compute(const DiagMatrix<Complex<T>, M> &From_matrix,
                      DiagMatrix<T, M> &To_matrix) {
    To_matrix[0] = From_matrix[0].imag;
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
  Loop<T, M, M - 1>::compute(From_matrix, To_matrix);
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

#ifdef __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  for (std::size_t i = 0; i < M; ++i) {
    To_matrix[i] = From_matrix[i].imag;
  }

#else // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  GetImagDiagMatrixFromComplex::compute<T, M>(From_matrix, To_matrix);

#endif // __BASE_MATRIX_USE_FOR_LOOP_OPERATION__

  return To_matrix;
}

} // namespace Matrix
} // namespace Base

#endif // __BASE_MATRIX_DIAGONAL_HPP__
