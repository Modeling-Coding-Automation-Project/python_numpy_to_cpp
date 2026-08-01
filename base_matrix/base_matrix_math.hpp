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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
abs(const T &x) {
  return PythonMath::abs(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> abs(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::abs(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> abs(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::abs(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
abs(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::abs(matrix.values);

  return result;
}

/* fmod */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
fmod(const T &x, const T &y) {
  return PythonMath::fmod(x, y);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> fmod(const Matrix<T, M, N> &matrix, const T &y) {
  Matrix<T, M, N> result;

  result.data = PythonMath::fmod(matrix.data, y);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> fmod(const DiagMatrix<T, M> &matrix, const T &y) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::fmod(matrix.data, y);

  return result;
}

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
sqrt(const T &x) {
  return PythonMath::sqrt(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> sqrt(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::sqrt(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> sqrt(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::sqrt(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
sqrt(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::sqrt(matrix.values);

  return result;
}

/* exp */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
exp(const T &x) {
  return PythonMath::exp(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> exp(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::exp(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> exp(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::exp(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
exp(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::exp(matrix.values);

  return result;
}

/* exp2 */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
exp2(const T &x) {
  return PythonMath::exp2(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> exp2(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::exp2(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> exp2(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::exp2(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
exp2(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::exp2(matrix.values);

  return result;
}

/* log */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
log(const T &x) {
  return PythonMath::log(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> log(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::log(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> log(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::log(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
log(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::log(matrix.values);

  return result;
}

/* log2 */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
log2(const T &x) {
  return PythonMath::log2(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> log2(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::log2(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> log2(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::log2(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
log2(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::log2(matrix.values);

  return result;
}

/* log10 */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
log10(const T &x) {
  return PythonMath::log10(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> log10(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::log10(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> log10(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::log10(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
log10(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::log10(matrix.values);

  return result;
}

/* pow */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
pow(const T &x, const T &y) {
  return PythonMath::pow(x, y);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> pow(const Matrix<T, M, N> &matrix, const T &y) {
  Matrix<T, M, N> result;

  result.data = PythonMath::pow(matrix.data, y);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> pow(const DiagMatrix<T, M> &matrix, const T &y) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::pow(matrix.data, y);

  return result;
}

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
sin(const T &x) {
  return PythonMath::sin(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> sin(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::sin(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> sin(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::sin(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
sin(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::sin(matrix.values);

  return result;
}

/* cos */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
cos(const T &x) {
  return PythonMath::cos(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> cos(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::cos(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> cos(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::cos(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
cos(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::cos(matrix.values);

  return result;
}

/* tan */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
tan(const T &x) {
  return PythonMath::tan(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> tan(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::tan(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> tan(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::tan(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
tan(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::tan(matrix.values);

  return result;
}

/* asin */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
asin(const T &x) {
  return PythonMath::asin(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> asin(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::asin(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> asin(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::asin(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
asin(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::asin(matrix.values);

  return result;
}

/* acos */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
acos(const T &x) {
  return PythonMath::acos(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> acos(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::acos(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> acos(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::acos(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
acos(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::acos(matrix.values);

  return result;
}

/* atan */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
atan(const T &x) {
  return PythonMath::atan(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> atan(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::atan(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> atan(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::atan(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
atan(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::atan(matrix.values);

  return result;
}

/* atan2 */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
atan2(const T &y, const T &x) {
  return PythonMath::atan2(y, x);
}

/* sinh */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
sinh(const T &x) {
  return PythonMath::sinh(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> sinh(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::sinh(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> sinh(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::sinh(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
sinh(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::sinh(matrix.values);

  return result;
}

/* cosh */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
cosh(const T &x) {
  return PythonMath::cosh(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> cosh(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::cosh(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> cosh(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::cosh(matrix.data);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename CSRIndices,
          typename CSRPointers>
inline CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers>
cosh(const CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> &matrix) {
  CompiledSparseMatrix<T, M, N, CSRIndices, CSRPointers> result;

  result.values = PythonMath::cosh(matrix.values);

  return result;
}

/* tanh */

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
tanh(const T &x) {
  return PythonMath::tanh(x);
}

template <typename T, std::size_t M, std::size_t N>
inline Matrix<T, M, N> tanh(const Matrix<T, M, N> &matrix) {
  Matrix<T, M, N> result;

  result.data = PythonMath::tanh(matrix.data);

  return result;
}

template <typename T, std::size_t M>
inline DiagMatrix<T, M> tanh(const DiagMatrix<T, M> &matrix) {
  DiagMatrix<T, M> result;

  result.data = PythonMath::tanh(matrix.data);

  return result;
}

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
