#ifndef BASE_MATRIX_MATH_HPP_
#define BASE_MATRIX_MATH_HPP_

#include "base_matrix_compiled_sparse.hpp"
#include "base_matrix_diagonal.hpp"
#include "base_matrix_matrix.hpp"

#include "python_math.hpp"

namespace Base {
namespace Matrix {

/* Constants */

constexpr double PI = PythonMath::PI;
constexpr double TWO_PI = PythonMath::TWO_PI;
constexpr double HALF_PI = PythonMath::HALF_PI;
constexpr double TAU = PythonMath::TAU;

constexpr double E = PythonMath::E;

constexpr double LN_2 = PythonMath::LN_2;
constexpr double LN_10 = PythonMath::LN_10;

/* abs */

/**
 * @brief Computes the absolute value of a given input.
 *
 * This function is a template that computes the absolute value of its argument
 * by delegating to PythonMath::abs. It works for any type T for which
 * PythonMath::abs is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose absolute value is to be computed.
 * @return The absolute value of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
abs(const T &x) {
  return PythonMath::abs(x);
}

/**
 * @brief Computes the element-wise absolute value of a std::array.
 *
 * This function takes a constant reference to a std::array of type T and size
 * N, and returns a new std::array where each element is the absolute value of
 * the corresponding element in the input array. The absolute value is computed
 * using PythonMath::abs.
 *
 * @tparam T The type of the elements in the array.
 * @tparam N The size of the array.
 * @param array The input array whose elements' absolute values are to be
 * computed.
 * @return std::array<T, N> A new array containing the absolute values of the
 * input array's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> abs(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::abs(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise absolute value of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the absolute
 * value of the corresponding diagonal element in the input matrix. The absolute
 * value is computed using PythonMath::abs.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' absolute values
 * are to be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the absolute values of
 * the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> abs(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::abs(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise absolute value of a CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the absolute value of the corresponding non-zero element in the
 * input matrix. The absolute value is computed using PythonMath::abs.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements'
 * absolute values are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the absolute values of the input matrix's
 * non-zero elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
abs(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::abs(matrix.values);

  return result;
}

/* fmod */

/**
 * @brief Computes the floating-point modulus of two values.
 *
 * This function is a template that computes the floating-point modulus of its
 * arguments by delegating to PythonMath::fmod. It works for any type T for
 * which PythonMath::fmod is defined.
 *
 * @tparam T The type of the input values.
 * @param x The dividend.
 * @param y The divisor.
 * @return The floating-point modulus of x and y.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
fmod(const T &x, const T &y) {
  return PythonMath::fmod(x, y);
}

namespace BaseMatrixFmodAction {

template <typename T, std::size_t M, std::size_t N, std::size_t Index>
struct FmodCoreColMajor {

  /**
   * @brief Recursively computes the element-wise floating-point modulus of a
   * 2D std::array (matrix) in column-major order.
   *
   * @param result The 2D array to store the results.
   * @param matrix The input 2D array whose elements will be processed.
   * @param y The divisor to use for the modulus operation.
   */
  static void compute(std::array<std::array<T, M>, N> &result,
                      const std::array<std::array<T, M>, N> &matrix,
                      const T &y) {
    result[Index] = PythonMath::fmod(matrix[Index], y);
    FmodCoreColMajor<T, M, N, Index - 1>::compute(result, matrix, y);
  }

  /**
   * @brief Recursively computes the element-wise floating-point modulus of a
   * 2D std::vector (matrix) in column-major order.
   *
   * @param result The 2D vector to store the results.
   * @param matrix The input 2D vector whose elements will be processed.
   * @param y The divisor to use for the modulus operation.
   */
  static void compute(std::vector<std::vector<T>> &result,
                      const std::vector<std::vector<T>> &matrix, const T &y) {
    result[Index] = PythonMath::fmod(matrix[Index], y);
    FmodCoreColMajor<T, M, N, Index - 1>::compute(result, matrix, y);
  }
};

template <typename T, std::size_t M, std::size_t N>
struct FmodCoreColMajor<T, M, N, 0> {

  /**
   * @brief Base case for the recursive computation of floating-point modulus
   * in a 2D std::array (matrix) in column-major order.
   * @param result The 2D array to store the results.
   * @param matrix The input 2D array whose elements will be processed.
   * @param y The divisor to use for the modulus operation.
   */
  static void compute(std::array<std::array<T, M>, N> &result,
                      const std::array<std::array<T, M>, N> &matrix,
                      const T &y) {
    result[0] = PythonMath::fmod(matrix[0], y);
  }

  /**
   * @brief Base case for the recursive computation of floating-point modulus
   * in a 2D std::vector (matrix) in column-major order.
   * @param result The 2D vector to store the results.
   * @param matrix The input 2D vector whose elements will be processed.
   * @param y The divisor to use for the modulus operation.
   */
  static void compute(std::vector<std::vector<T>> &result,
                      const std::vector<std::vector<T>> &matrix, const T &y) {
    result[0] = PythonMath::fmod(matrix[0], y);
  }
};

/**
 * @brief Computes the element-wise floating-point modulus of a 2D std::array
 * (matrix) in column-major order.
 *
 * This function takes a 2D std::array (matrix) and computes the floating-point
 * modulus of each element with respect to a given divisor y. The results are
 * stored in the provided result array.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param result The 2D array to store the results.
 * @param matrix The input 2D array whose elements will be processed.
 * @param y The divisor to use for the modulus operation.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(std::array<std::array<T, M>, N> &result,
                    const std::array<std::array<T, M>, N> &matrix, const T &y) {
  FmodCoreColMajor<T, M, N, N - 1>::compute(result, matrix, y);
}

/**
 * @brief Computes the element-wise floating-point modulus of a 2D std::vector
 * (matrix) in column-major order.
 *
 * This function takes a 2D std::vector (matrix) and computes the floating-point
 * modulus of each element with respect to a given divisor y. The results are
 * stored in the provided result vector.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param result The 2D vector to store the results.
 * @param matrix The input 2D vector whose elements will be processed.
 * @param y The divisor to use for the modulus operation.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(std::vector<std::vector<T>> &result,
                    const std::vector<std::vector<T>> &matrix, const T &y) {
  FmodCoreColMajor<T, M, N, N - 1>::compute(result, matrix, y);
}

} // namespace BaseMatrixFmodAction

/**
 * @brief Computes the element-wise floating-point modulus of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and a divisor y, and returns a new Matrix where each element is the
 * floating-point modulus of the corresponding element in the input matrix with
 * respect to y.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' floating-point modulus is to
 * be computed.
 * @param y The divisor to use for the modulus operation.
 * @return Matrix<T, M, N> A new Matrix containing the floating-point modulus of
 * the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> fmod(const Matrix<T, M, N> &matrix, const T &y) {
  Matrix<T, M, N> result;

  BaseMatrixFmodAction::compute<T, M, N>(result.data, matrix.data, y);

  return result;
}

/**
 * @brief Computes the element-wise floating-point modulus of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and a divisor y, and returns a new DiagMatrix where each diagonal element
 * is the floating-point modulus of the corresponding diagonal element in the
 * input matrix with respect to y.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' floating-point
 * modulus is to be computed.
 * @param y The divisor to use for the modulus operation.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the floating-point
 * modulus of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> fmod(const DiagMatrix<T, M> &matrix, const T &y) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::fmod(matrix.data, y);

  return result;
}

/**
 * @brief Computes the element-wise floating-point modulus of a
 * CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and a divisor y, and returns a new CompiledSparseMatrix where
 * each non-zero element is the floating-point modulus of the corresponding
 * non-zero element in the input matrix with respect to y.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements'
 * floating-point modulus is to be computed.
 * @param y The divisor to use for the modulus operation.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the floating-point modulus of the input
 * matrix's non-zero elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
fmod(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix,
     const T &y) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::fmod(matrix.values, y);

  return result;
}

/* sqrt */

/**
 * @brief Computes the square root of a given input.
 *
 * This function is a template that computes the square root of its argument
 * by delegating to PythonMath::sqrt. It works for any type T for which
 * PythonMath::sqrt is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose square root is to be computed.
 * @return The square root of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
sqrt(const T &x) {
  return PythonMath::sqrt(x);
}

/**
 * @brief Computes the element-wise square root of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the square root of the
 * corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' square roots are to be
 * computed.
 * @return Matrix<T, M, N> A new Matrix containing the square roots of the input
 * matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> sqrt(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::sqrt(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise square root of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the square
 * root of the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' square roots are
 * to be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the square roots of the
 * input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> sqrt(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::sqrt(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise square root of a CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the square root of the corresponding non-zero element in the input
 * matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements' square
 * roots are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the square roots of the input matrix's
 * non-zero elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
sqrt(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::sqrt(matrix.values);

  return result;
}

/* exp */

/**
 * @brief Computes the exponential of a given input.
 *
 * This function is a template that computes the exponential of its argument
 * by delegating to PythonMath::exp. It works for any type T for which
 * PythonMath::exp is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose exponential is to be computed.
 * @return The exponential of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
exp(const T &x) {
  return PythonMath::exp(x);
}

/**
 * @brief Computes the element-wise exponential of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the exponential of the
 * corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' exponentials are to be
 * computed.
 * @return Matrix<T, M, N> A new Matrix containing the exponentials of the input
 * matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> exp(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::exp(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise exponential of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the
 * exponential of the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' exponentials are
 * to be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the exponentials of the
 * input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> exp(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::exp(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise exponential of a CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the exponential of the corresponding non-zero element in the input
 * matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements'
 * exponentials are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the exponentials of the input matrix's
 * non-zero elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
exp(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::exp(matrix.values);

  return result;
}

/* exp2 */

/**
 * @brief Computes the base-2 exponential of a given input.
 *
 * This function is a template that computes the base-2 exponential of its
 * argument by delegating to PythonMath::exp2. It works for any type T for
 * which PythonMath::exp2 is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose base-2 exponential is to be computed.
 * @return The base-2 exponential of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
exp2(const T &x) {
  return PythonMath::exp2(x);
}

/**
 * @brief Computes the element-wise base-2 exponential of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the base-2 exponential of
 * the corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' base-2 exponentials are to be
 * computed.
 * @return Matrix<T, M, N> A new Matrix containing the base-2 exponentials of
 * the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> exp2(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::exp2(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise base-2 exponential of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the base-2
 * exponential of the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' base-2
 * exponentials are to be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the base-2 exponentials
 * of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> exp2(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::exp2(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise base-2 exponential of a
 * CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the base-2 exponential of the corresponding non-zero element in
 * the input matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements' base-2
 * exponentials are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the base-2 exponentials of the input matrix's
 * non-zero elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
exp2(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::exp2(matrix.values);

  return result;
}

/* log */

/**
 * @brief Computes the natural logarithm of a given input.
 *
 * This function is a template that computes the natural logarithm of its
 * argument by delegating to PythonMath::log. It works for any type T for which
 * PythonMath::log is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose natural logarithm is to be computed.
 * @return The natural logarithm of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
log(const T &x) {
  return PythonMath::log(x);
}

/**
 * @brief Computes the element-wise natural logarithm of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the natural logarithm of
 * the corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' natural logarithms are to be
 * computed.
 * @return Matrix<T, M, N> A new Matrix containing the natural logarithms of
 * the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> log(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::log(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise natural logarithm of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the natural
 * logarithm of the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' natural
 * logarithms are to be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the natural logarithms
 * of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> log(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::log(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise natural logarithm of a
 * CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the natural logarithm of the corresponding non-zero element in the
 * input matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements' natural
 * logarithms are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the natural logarithms of the input matrix's
 * non-zero elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
log(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::log(matrix.values);

  return result;
}

/* log2 */

/**
 * @brief Computes the base-2 logarithm of a given input.
 *
 * This function is a template that computes the base-2 logarithm of its
 * argument by delegating to PythonMath::log2. It works for any type T for which
 * PythonMath::log2 is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose base-2 logarithm is to be computed.
 * @return The base-2 logarithm of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
log2(const T &x) {
  return PythonMath::log2(x);
}

/**
 * @brief Computes the element-wise base-2 logarithm of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the base-2 logarithm of
 * the corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' base-2 logarithms are to be
 * computed.
 * @return Matrix<T, M, N> A new Matrix containing the base-2 logarithms of
 * the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> log2(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::log2(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise base-2 logarithm of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the base-2
 * logarithm of the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' base-2
 * logarithms are to be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the base-2 logarithms
 * of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> log2(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::log2(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise base-2 logarithm of a
 * CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the base-2 logarithm of the corresponding non-zero element in the
 * input matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements' base-2
 * logarithms are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the base-2 logarithms of the input matrix's
 * non-zero elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
log2(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::log2(matrix.values);

  return result;
}

/* log10 */

/**
 * @brief Computes the base-10 logarithm of a given input.
 *
 * This function is a template that computes the base-10 logarithm of its
 * argument by delegating to PythonMath::log10. It works for any type T for
 * which PythonMath::log10 is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose base-10 logarithm is to be computed.
 * @return The base-10 logarithm of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
log10(const T &x) {
  return PythonMath::log10(x);
}

/**
 * @brief Computes the element-wise base-10 logarithm of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the base-10 logarithm of
 * the corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' base-10 logarithms are to be
 * computed.
 * @return Matrix<T, M, N> A new Matrix containing the base-10 logarithms of
 * the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> log10(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::log10(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise base-10 logarithm of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the base-10
 * logarithm of the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' base-10
 * logarithms are to be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the base-10 logarithms
 * of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> log10(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::log10(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise base-10 logarithm of a
 * CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the base-10 logarithm of the corresponding non-zero element in the
 * input matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements' base-10
 * logarithms are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the base-10 logarithms of the input matrix's
 * non-zero elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
log10(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::log10(matrix.values);

  return result;
}

/* pow */

/**
 * @brief Computes the power of a given input raised to a specified exponent.
 *
 * This function is a template that computes the power of its first argument
 * raised to the second argument by delegating to PythonMath::pow. It works for
 * any type T for which PythonMath::pow is defined.
 *
 * @tparam T The type of the input values.
 * @param x The base value.
 * @param y The exponent value.
 * @return The result of raising x to the power of y.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
pow(const T &x, const T &y) {
  return PythonMath::pow(x, y);
}

namespace BaseMatrixPowAction {

template <typename T, std::size_t M, std::size_t N, std::size_t Index>
struct PowCoreColMajor {

  /**
   * @brief Recursively computes the element-wise power of a 2D std::array
   * (matrix) in column-major order.
   *
   * @param result The 2D array to store the results.
   * @param matrix The input 2D array whose elements will be processed.
   * @param y The exponent to use for the power operation.
   */
  static void compute(std::array<std::array<T, M>, N> &result,
                      const std::array<std::array<T, M>, N> &matrix,
                      const T &y) {
    result[Index] = PythonMath::pow(matrix[Index], y);
    PowCoreColMajor<T, M, N, Index - 1>::compute(result, matrix, y);
  }

  /**
   * @brief Recursively computes the element-wise power of a 2D std::vector
   * (matrix) in column-major order.
   *
   * @param result The 2D vector to store the results.
   * @param matrix The input 2D vector whose elements will be processed.
   * @param y The exponent to use for the power operation.
   */
  static void compute(std::vector<std::vector<T>> &result,
                      const std::vector<std::vector<T>> &matrix, const T &y) {
    result[Index] = PythonMath::pow(matrix[Index], y);
    PowCoreColMajor<T, M, N, Index - 1>::compute(result, matrix, y);
  }
};

template <typename T, std::size_t M, std::size_t N>
struct PowCoreColMajor<T, M, N, 0> {

  /**
   * @brief Base case for the recursive computation of power in a 2D
   * std::array (matrix) in column-major order.
   * @param result The 2D array to store the results.
   * @param matrix The input 2D array whose elements will be processed.
   * @param y The exponent to use for the power operation.
   */
  static void compute(std::array<std::array<T, M>, N> &result,
                      const std::array<std::array<T, M>, N> &matrix,
                      const T &y) {
    result[0] = PythonMath::pow(matrix[0], y);
  }

  /**
   * @brief Base case for the recursive computation of power in a 2D
   * std::vector (matrix) in column-major order.
   * @param result The 2D vector to store the results.
   * @param matrix The input 2D vector whose elements will be processed.
   * @param y The exponent to use for the power operation.
   */
  static void compute(std::vector<std::vector<T>> &result,
                      const std::vector<std::vector<T>> &matrix, const T &y) {
    result[0] = PythonMath::pow(matrix[0], y);
  }
};

/**
 * @brief Computes the element-wise power of a 2D std::array (matrix) in
 * column-major order.
 *
 * This function takes a 2D std::array (matrix) and computes the power of each
 * element with respect to a given exponent y. The results are stored in the
 * provided result array.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param result The 2D array to store the results.
 * @param matrix The input 2D array whose elements will be processed.
 * @param y The exponent to use for the power operation.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(std::array<std::array<T, M>, N> &result,
                    const std::array<std::array<T, M>, N> &matrix, const T &y) {
  PowCoreColMajor<T, M, N, N - 1>::compute(result, matrix, y);
}

/**
 * @brief Computes the element-wise power of a 2D std::vector (matrix) in
 * column-major order.
 *
 * This function takes a 2D std::vector (matrix) and computes the power of each
 * element with respect to a given exponent y. The results are stored in the
 * provided result vector.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param result The 2D vector to store the results.
 * @param matrix The input 2D vector whose elements will be processed.
 * @param y The exponent to use for the power operation.
 */
template <typename T, std::size_t M, std::size_t N>
inline void compute(std::vector<std::vector<T>> &result,
                    const std::vector<std::vector<T>> &matrix, const T &y) {
  PowCoreColMajor<T, M, N, N - 1>::compute(result, matrix, y);
}

} // namespace BaseMatrixPowAction

/**
 * @brief Computes the element-wise power of a Matrix raised to a specified
 * exponent.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is raised to the power of y.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements will be raised to the power of
 * y.
 * @param y The exponent to which each element will be raised.
 * @return Matrix<T, M, N> A new Matrix containing the results of raising each
 * element of the input matrix to the power of y.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> pow(const Matrix<T, M, N> &matrix, const T &y) {
  Matrix<T, M, N> result;

  BaseMatrixPowAction::compute<T, M, N>(result.data, matrix.data, y);

  return result;
}

/**
 * @brief Computes the element-wise power of a DiagMatrix raised to a specified
 * exponent.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is raised to the
 * power of y.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements will be raised to
 * the power of y.
 * @param y The exponent to which each diagonal element will be raised.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the results of raising
 * each diagonal element of the input matrix to the power of y.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> pow(const DiagMatrix<T, M> &matrix, const T &y) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::pow(matrix.data, y);

  return result;
}

/**
 * @brief Computes the element-wise power of a CompiledSparseMatrix raised to a
 * specified exponent.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is raised to the power of y.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements will be
 * raised to the power of y.
 * @param y The exponent to which each non-zero element will be raised.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the results of raising each non-zero element
 * of the input matrix to the power of y.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
pow(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix,
    const T &y) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::pow(matrix.values, y);

  return result;
}

/* sin */

/**
 * @brief Computes the sine of a given input.
 *
 * This function is a template that computes the sine of its argument by
 * delegating to PythonMath::sin. It works for any type T for which
 * PythonMath::sin is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose sine is to be computed.
 * @return The sine of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
sin(const T &x) {
  return PythonMath::sin(x);
}

/**
 * @brief Computes the element-wise sine of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the sine of the
 * corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' sines are to be computed.
 * @return Matrix<T, M, N> A new Matrix containing the sines of the input
 * matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> sin(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::sin(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise sine of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the sine of
 * the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' sines are to be
 * computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the sines of the input
 * matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> sin(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::sin(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise sine of a CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the sine of the corresponding non-zero element in the input
 * matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements' sines
 * are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the sines of the input matrix's non-zero
 * elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
sin(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::sin(matrix.values);

  return result;
}

/* cos */

/**
 * @brief Computes the cosine of a given input.
 *
 * This function is a template that computes the cosine of its argument by
 * delegating to PythonMath::cos. It works for any type T for which
 * PythonMath::cos is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose cosine is to be computed.
 * @return The cosine of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
cos(const T &x) {
  return PythonMath::cos(x);
}

/**
 * @brief Computes the element-wise cosine of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the cosine of the
 * corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' cosines are to be computed.
 * @return Matrix<T, M, N> A new Matrix containing the cosines of the input
 * matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> cos(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::cos(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise cosine of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the cosine of
 * the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' cosines are to be
 * computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the cosines of the input
 * matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> cos(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::cos(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise cosine of a CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the cosine of the corresponding non-zero element in the input
 * matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements' cosines
 * are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the cosines of the input matrix's non-zero
 * elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
cos(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::cos(matrix.values);

  return result;
}

/* tan */

/**
 * @brief Computes the tangent of a given input.
 *
 * This function is a template that computes the tangent of its argument by
 * delegating to PythonMath::tan. It works for any type T for which
 * PythonMath::tan is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose tangent is to be computed.
 * @return The tangent of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
tan(const T &x) {
  return PythonMath::tan(x);
}

/**
 * @brief Computes the element-wise tangent of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the tangent of the
 * corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' tangents are to be computed.
 * @return Matrix<T, M, N> A new Matrix containing the tangents of the input
 * matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> tan(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::tan(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise tangent of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the tangent
 * of the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' tangents are to
 * be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the tangents of the
 * input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> tan(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::tan(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise tangent of a CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the tangent of the corresponding non-zero element in the input
 * matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements'
 * tangents are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the tangents of the input matrix's non-zero
 * elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
tan(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::tan(matrix.values);

  return result;
}

/* asin */

/**
 * @brief Computes the arcsine (inverse sine) of a given input.
 *
 * This function is a template that computes the arcsine of its argument by
 * delegating to PythonMath::asin. It works for any type T for which
 * PythonMath::asin is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose arcsine is to be computed.
 * @return The arcsine of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
asin(const T &x) {
  return PythonMath::asin(x);
}

/**
 * @brief Computes the element-wise arcsine of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the arcsine of the
 * corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' arcsines are to be computed.
 * @return Matrix<T, M, N> A new Matrix containing the arcsines of the input
 * matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> asin(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::asin(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise arcsine of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the arcsine
 * of the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' arcsines are to
 * be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the arcsines of the
 * input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> asin(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::asin(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise arcsine of a CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the arcsine of the corresponding non-zero element in the input
 * matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements'
 * arcsines are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the arcsines of the input matrix's non-zero
 * elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
asin(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::asin(matrix.values);

  return result;
}

/* acos */

/**
 * @brief Computes the arccosine (inverse cosine) of a given input.
 *
 * This function is a template that computes the arccosine of its argument by
 * delegating to PythonMath::acos. It works for any type T for which
 * PythonMath::acos is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose arccosine is to be computed.
 * @return The arccosine of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
acos(const T &x) {
  return PythonMath::acos(x);
}

/**
 * @brief Computes the element-wise arccosine of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the arccosine of the
 * corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' arccosines are to be computed.
 * @return Matrix<T, M, N> A new Matrix containing the arccosines of the input
 * matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> acos(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::acos(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise arccosine of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the arccosine
 * of the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' arccosines are
 * to be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the arccosines of the
 * input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> acos(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::acos(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise arccosine of a CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the arccosine of the corresponding non-zero element in the input
 * matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements'
 * arccosines are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the arccosines of the input matrix's non-zero
 * elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
acos(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::acos(matrix.values);

  return result;
}

/* atan */

/**
 * @brief Computes the arctangent (inverse tangent) of a given input.
 *
 * This function is a template that computes the arctangent of its argument by
 * delegating to PythonMath::atan. It works for any type T for which
 * PythonMath::atan is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose arctangent is to be computed.
 * @return The arctangent of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
atan(const T &x) {
  return PythonMath::atan(x);
}

/**
 * @brief Computes the element-wise arctangent of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the arctangent of the
 * corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' arctangents are to be
 * computed.
 * @return Matrix<T, M, N> A new Matrix containing the arctangents of the input
 * matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> atan(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::atan(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise arctangent of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the
 * arctangent of the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' arctangents are
 * to be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the arctangents of the
 * input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> atan(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::atan(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise arctangent of a CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the arctangent of the corresponding non-zero element in the input
 * matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements'
 * arctangents are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the arctangents of the input matrix's
 * non-zero elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
atan(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::atan(matrix.values);

  return result;
}

/* atan2 */

/**
 * @brief Computes the arctangent of the quotient of its arguments.
 *
 * This function is a template that computes the arctangent of the quotient of
 * its arguments y and x by delegating to PythonMath::atan2. It works for any
 * type T for which PythonMath::atan2 is defined.
 *
 * @tparam T The type of the input values.
 * @param y The numerator value.
 * @param x The denominator value.
 * @return The arctangent of y/x, taking into account the signs of both
 * arguments to determine the correct quadrant.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
atan2(const T &y, const T &x) {
  return PythonMath::atan2(y, x);
}

/* sinh */

/**
 * @brief Computes the hyperbolic sine of a given input.
 *
 * This function is a template that computes the hyperbolic sine of its
 * argument by delegating to PythonMath::sinh. It works for any type T for
 * which PythonMath::sinh is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose hyperbolic sine is to be computed.
 * @return The hyperbolic sine of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
sinh(const T &x) {
  return PythonMath::sinh(x);
}

/**
 * @brief Computes the element-wise hyperbolic sine of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the hyperbolic sine of the
 * corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' hyperbolic sines are to be
 * computed.
 * @return Matrix<T, M, N> A new Matrix containing the hyperbolic sines of the
 * input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> sinh(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::sinh(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise hyperbolic sine of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the
 * hyperbolic sine of the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' hyperbolic sines
 * are to be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the hyperbolic sines of
 * the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> sinh(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::sinh(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise hyperbolic sine of a CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the hyperbolic sine of the corresponding non-zero element in the
 * input matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements'
 * hyperbolic sines are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the hyperbolic sines of the input matrix's
 * non-zero elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
sinh(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::sinh(matrix.values);

  return result;
}

/* cosh */

/**
 * @brief Computes the hyperbolic cosine of a given input.
 *
 * This function is a template that computes the hyperbolic cosine of its
 * argument by delegating to PythonMath::cosh. It works for any type T for
 * which PythonMath::cosh is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose hyperbolic cosine is to be computed.
 * @return The hyperbolic cosine of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
cosh(const T &x) {
  return PythonMath::cosh(x);
}

/**
 * @brief Computes the element-wise hyperbolic cosine of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the hyperbolic cosine of
 * the corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' hyperbolic cosines are to be
 * computed.
 * @return Matrix<T, M, N> A new Matrix containing the hyperbolic cosines of the
 * input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> cosh(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::cosh(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise hyperbolic cosine of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the
 * hyperbolic cosine of the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' hyperbolic
 * cosines are to be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the hyperbolic cosines
 * of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> cosh(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::cosh(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise hyperbolic cosine of a CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the hyperbolic cosine of the corresponding non-zero element in the
 * input matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements'
 * hyperbolic cosines are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the hyperbolic cosines of the input matrix's
 * non-zero elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
cosh(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::cosh(matrix.values);

  return result;
}

/* tanh */

/**
 * @brief Computes the hyperbolic tangent of a given input.
 *
 * This function is a template that computes the hyperbolic tangent of its
 * argument by delegating to PythonMath::tanh. It works for any type T for
 * which PythonMath::tanh is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose hyperbolic tangent is to be computed.
 * @return The hyperbolic tangent of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
tanh(const T &x) {
  return PythonMath::tanh(x);
}

/**
 * @brief Computes the element-wise hyperbolic tangent of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the hyperbolic tangent of
 * the corresponding element in the input matrix.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' hyperbolic tangents are to be
 * computed.
 * @return Matrix<T, M, N> A new Matrix containing the hyperbolic tangents of
 * the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> tanh(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::tanh(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise hyperbolic tangent of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the
 * hyperbolic tangent of the corresponding diagonal element in the input matrix.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' hyperbolic
 * tangents are to be computed.
 * @return DiagMatrix<T, M> A new DiagMatrix containing the hyperbolic tangents
 * of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline DiagMatrix<T, M> tanh(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::tanh(matrix.data);

  return result;
}

/**
 * @brief Computes the element-wise hyperbolic tangent of a
 * CompiledSparseMatrix.
 *
 * This function takes a constant reference to a CompiledSparseMatrix of type T
 * and size M x N, and returns a new CompiledSparseMatrix where each non-zero
 * element is the hyperbolic tangent of the corresponding non-zero element in
 * the input matrix.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam CSRIndices The type representing the indices in the compressed
 * sparse row format.
 * @tparam CSRPointers The type representing the pointers in the compressed
 * sparse row format.
 * @param matrix The input CompiledSparseMatrix whose non-zero elements'
 * hyperbolic tangents are to be computed.
 * @return CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> A new
 * CompiledSparseMatrix containing the hyperbolic tangents of the input matrix's
 * non-zero elements.
 */
template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
tanh(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::tanh(matrix.values);

  return result;
}

} // namespace Matrix
} // namespace Base

#endif // BASE_MATRIX_MATH_HPP_
