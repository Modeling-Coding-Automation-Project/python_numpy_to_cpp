#ifndef PYTHON_NUMPY_MATH_HPP_
#define PYTHON_NUMPY_MATH_HPP_

#include "base_matrix.hpp"

#include "python_numpy_augmented_matrix.hpp"
#include "python_numpy_base.hpp"

namespace PythonNumpy {

/* Constants */

constexpr double PI = Base::Matrix::PI;
constexpr double TWO_PI = Base::Matrix::TWO_PI;
constexpr double HALF_PI = Base::Matrix::HALF_PI;
constexpr double TAU = Base::Matrix::TAU;

constexpr double E = Base::Matrix::E;

constexpr double LN_2 = Base::Matrix::LN_2;
constexpr double LN_10 = Base::Matrix::LN_10;

/* abs */

/**
 * @brief Computes the absolute value of a given input.
 *
 * This function is a template that computes the absolute value of its argument
 * by delegating to Base::Matrix::abs. It works for any type T for which
 * Base::Matrix::abs is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose absolute value is to be computed.
 * @return The absolute value of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
abs(const T &x) {
  return Base::Matrix::abs(x);
}

/**
 * @brief Computes the element-wise absolute value of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the absolute value of the
 * corresponding element in the input matrix. The absolute value is computed
 * using Base::Matrix::abs.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' absolute values are to be
 * computed.
 * @return Matrix<T, M, N> A new Matrix containing the absolute values of the
 * input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto abs(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::abs(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise absolute value of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the absolute
 * value of the corresponding diagonal element in the input matrix. The
 * absolute value is computed using Base::Matrix::abs.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' absolute values
 * are to be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the absolute values
 * of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto abs(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::abs(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise absolute value of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the absolute
 * value of the corresponding element in the input matrix. The absolute value is
 * computed using Base::Matrix::abs.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' absolute values are to
 * be computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the absolute values of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto abs(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::abs(matrix.matrix);

  return result;
}

namespace AbsAugmentedMatrixAction {

/**
 * @brief Helper struct to apply abs to each element of a std::tuple
 * using template metaprogramming.
 *
 * This struct recursively applies Base::Matrix::abs to each element of
 * the input tuple. Uses index-based recursion for C++11 compatibility.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @tparam Index The current index in the tuple (decremented recursively).
 */
template <typename Tuple_Type, std::size_t Index> struct ApplyAbsTupleCore {
  /**
   * @brief Recursively applies abs to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where abs results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::abs(std::get<Index>(input).matrix);
    ApplyAbsTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

/**
 * @brief Base case specialization for index 0.
 *
 * @tparam Tuple_Type The type of the tuple.
 */
template <typename Tuple_Type> struct ApplyAbsTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where abs results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::abs(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply abs to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where abs results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyAbsTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output);
}

} // namespace AbsAugmentedMatrixAction

/**
 * @brief Computes the element-wise absolute value of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the absolute value of the
 * corresponding element in the input augmented matrix. The absolute value is
 * computed using Base::Matrix::abs for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' absolute
 * values are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the absolute values of the input augmented
 * matrix's elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto
abs(const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  AbsAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* fmod */

/**
 * @brief Computes the floating-point modulus of a given input with respect to
 * a divisor.
 *
 * This function is a template that computes the floating-point modulus of its
 * argument by delegating to Base::Matrix::fmod. It works for any type T for
 * which Base::Matrix::fmod is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose floating-point modulus is to be computed.
 * @param y The divisor for the modulus operation.
 * @return The floating-point modulus of x with respect to y.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
fmod(const T &x, const T &y) {
  return Base::Matrix::fmod(x, y);
}

/**
 * @brief Computes the element-wise floating-point modulus of a Matrix with
 * respect to a divisor.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and a divisor y, and returns a new Matrix where each element is the
 * floating-point modulus of the corresponding element in the input matrix with
 * respect to y. The modulus is computed using Base::Matrix::fmod.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' floating-point modulus is to
 * be computed.
 * @param y The divisor for the modulus operation.
 * @return Matrix<T, M, N> A new Matrix containing the floating-point modulus of
 * the input matrix's elements with respect to y.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto fmod(const Matrix<DefDense, T, M, N> &matrix, const T &y)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::fmod(matrix.matrix, y);

  return result;
}

/**
 * @brief Computes the element-wise floating-point modulus of a DiagMatrix with
 * respect to a divisor.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and a divisor y, and returns a new DiagMatrix where each diagonal element
 * is the floating-point modulus of the corresponding diagonal element in the
 * input matrix with respect to y. The modulus is computed using
 * Base::Matrix::fmod.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' floating-point
 * modulus is to be computed.
 * @param y The divisor for the modulus operation.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the floating-point
 * modulus of the input matrix's diagonal elements with respect to y.
 */
template <typename T, std::size_t M>
inline auto fmod(const Matrix<DefDiag, T, M> &matrix, const T &y)
    -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::fmod(matrix.matrix, y);

  return result;
}

/**
 * @brief Computes the element-wise floating-point modulus of a SparseMatrix
 * with respect to a divisor.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and a divisor y, and returns a new SparseMatrix where each element is
 * the floating-point modulus of the corresponding element in the input matrix
 * with respect to y. The modulus is computed using Base::Matrix::fmod.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' floating-point modulus
 * is to be computed.
 * @param y The divisor for the modulus operation.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the floating-point modulus of the input matrix's elements with
 * respect to y.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto fmod(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                 const T &y) -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::fmod(matrix.matrix, y);

  return result;
}

namespace FmodAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyFmodTupleCore {
  /**
   * @brief Recursively applies fmod to tuple elements.
   *
   * @tparam T The type of the divisor for the modulus operation.
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where fmod results are stored.
   * @param y The divisor for the modulus operation.
   */
  template <typename T>
  static void compute(const Tuple_Type &input, Tuple_Type &output, const T &y) {
    std::get<Index>(output).matrix =
        Base::Matrix::fmod(std::get<Index>(input).matrix, y);
    ApplyFmodTupleCore<Tuple_Type, Index - 1>::compute(input, output, y);
  }
};

template <typename Tuple_Type> struct ApplyFmodTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @tparam T The type of the divisor for the modulus operation.
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where fmod results are stored.
   * @param y The divisor for the modulus operation.
   */
  template <typename T>
  static void compute(const Tuple_Type &input, Tuple_Type &output, const T &y) {
    std::get<0>(output).matrix =
        Base::Matrix::fmod(std::get<0>(input).matrix, y);
  }
};

template <typename Tuple_Type, typename T>
inline void compute(const Tuple_Type &input, Tuple_Type &output, const T &y) {
  ApplyFmodTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output, y);
}

} // namespace FmodAugmentedMatrixAction

/**
 * @brief Computes the element-wise floating-point modulus of an AugmentedMatrix
 * with respect to a divisor.
 *
 * This function takes a constant reference to an AugmentedMatrix and a divisor
 * y, and returns a new AugmentedMatrix where each element is the
 * floating-point modulus of the corresponding element in the input augmented
 * matrix with respect to y. The modulus is computed using Base::Matrix::fmod
 * for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam T The type of the divisor for the modulus operation.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' floating-
 * point modulus is to be computed.
 * @param y The divisor for the modulus operation.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the floating-point modulus of the input augmented
 * matrix's elements with respect to y.
 */
template <typename Tuple_Type, typename T, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto fmod(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix,
    const T &y) -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  FmodAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix, y);

  return result;
}

/* sqrt */

/**
 * @brief Computes the square root of a given input.
 *
 * This function is a template that computes the square root of its argument
 * by delegating to Base::Matrix::sqrt. It works for any type T for which
 * Base::Matrix::sqrt is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose square root is to be computed.
 * @return The square root of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
sqrt(const T &x) {
  return Base::Matrix::sqrt(x);
}

/**
 * @brief Computes the element-wise square root of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the square root of the
 * corresponding element in the input matrix. The square root is computed using
 * Base::Matrix::sqrt.
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
inline auto sqrt(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::sqrt(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise square root of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the square
 * root of the corresponding diagonal element in the input matrix. The square
 * root is computed using Base::Matrix::sqrt.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' square roots are
 * to be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the square roots of
 * the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto sqrt(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::sqrt(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise square root of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the square root
 * of the corresponding element in the input matrix. The square root is computed
 * using Base::Matrix::sqrt.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' square roots are to be
 * computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the square roots of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto sqrt(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::sqrt(matrix.matrix);

  return result;
}

namespace SqrtAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplySqrtTupleCore {
  /**
   * @brief Recursively applies sqrt to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where sqrt results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::sqrt(std::get<Index>(input).matrix);
    ApplySqrtTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplySqrtTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where sqrt results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::sqrt(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply sqrt to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where sqrt results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplySqrtTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace SqrtAugmentedMatrixAction

/**
 * @brief Computes the element-wise square root of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the square root of the
 * corresponding element in the input augmented matrix. The square root is
 * computed using Base::Matrix::sqrt for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' square
 * roots are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the square roots of the input augmented matrix's
 * elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto sqrt(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  SqrtAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* exp */

/**
 * @brief Computes the exponential of a given input.
 *
 * This function is a template that computes the exponential of its argument
 * by delegating to Base::Matrix::exp. It works for any type T for which
 * Base::Matrix::exp is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose exponential is to be computed.
 * @return The exponential of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
exp(const T &x) {
  return Base::Matrix::exp(x);
}

/**
 * @brief Computes the element-wise exponential of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the exponential of the
 * corresponding element in the input matrix. The exponential is computed using
 * Base::Matrix::exp.
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
inline auto exp(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::exp(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise exponential of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the
 * exponential of the corresponding diagonal element in the input matrix. The
 * exponential is computed using Base::Matrix::exp.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' exponentials are
 * to be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the exponentials of
 * the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto exp(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::exp(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise exponential of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the exponential
 * of the corresponding element in the input matrix. The exponential is computed
 * using Base::Matrix::exp.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' exponentials are to be
 * computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the exponentials of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto exp(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::exp(matrix.matrix);

  return result;
}

namespace ExpAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyExpTupleCore {
  /**
   * @brief Recursively applies exp to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where exp results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::exp(std::get<Index>(input).matrix);
    ApplyExpTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyExpTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where exp results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::exp(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply exp to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where exp results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyExpTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output);
}

} // namespace ExpAugmentedMatrixAction

/* @brief Computes the element-wise exponential of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the exponential of the
 * corresponding element in the input augmented matrix. The exponential is
 * computed using Base::Matrix::exp for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements'
 * exponentials are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the exponentials of the input augmented matrix's
 * elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto
exp(const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  ExpAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* exp2 */

/**
 * @brief Computes the base-2 exponential of a given input.
 *
 * This function is a template that computes the base-2 exponential of its
 * argument by delegating to Base::Matrix::exp2. It works for any type T for
 * which Base::Matrix::exp2 is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose base-2 exponential is to be computed.
 * @return The base-2 exponential of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
exp2(const T &x) {
  return Base::Matrix::exp2(x);
}

/**
 * @brief Computes the element-wise base-2 exponential of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the base-2 exponential of
 * the corresponding element in the input matrix. The base-2 exponential is
 * computed using Base::Matrix::exp2.
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
inline auto exp2(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::exp2(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise base-2 exponential of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the base-2
 * exponential of the corresponding diagonal element in the input matrix. The
 * base-2 exponential is computed using Base::Matrix::exp2.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' base-2
 * exponentials are to be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the base-2
 * exponentials of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto exp2(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::exp2(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise base-2 exponential of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the base-2
 * exponential of the corresponding element in the input matrix. The base-2
 * exponential is computed using Base::Matrix::exp2.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' base-2 exponentials are
 * to be computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the base-2 exponentials of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto exp2(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::exp2(matrix.matrix);

  return result;
}

namespace Exp2AugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyExp2TupleCore {
  /**
   * @brief Recursively applies exp2 to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where exp2 results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::exp2(std::get<Index>(input).matrix);
    ApplyExp2TupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyExp2TupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where exp2 results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::exp2(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply exp2 to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where exp2 results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyExp2TupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace Exp2AugmentedMatrixAction

/**
 * @brief Computes the element-wise base-2 exponential of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the base-2 exponential of the
 * corresponding element in the input augmented matrix. The base-2 exponential
 * is computed using Base::Matrix::exp2 for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' base-2
 * exponentials are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the base-2 exponentials of the input augmented
 * matrix's elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto exp2(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  Exp2AugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* log */

/**
 * @brief Computes the natural logarithm of a given input.
 *
 * This function is a template that computes the natural logarithm of its
 * argument by delegating to Base::Matrix::log. It works for any type T for
 * which Base::Matrix::log is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose natural logarithm is to be computed.
 * @return The natural logarithm of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
log(const T &x) {
  return Base::Matrix::log(x);
}

/**
 * @brief Computes the element-wise natural logarithm of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the natural logarithm of
 * the corresponding element in the input matrix. The natural logarithm is
 * computed using Base::Matrix::log.
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
inline auto log(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::log(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise natural logarithm of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the natural
 * logarithm of the corresponding diagonal element in the input matrix. The
 * natural logarithm is computed using Base::Matrix::log.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' natural
 * logarithms are to be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the natural
 * logarithms of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto log(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::log(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise natural logarithm of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the natural
 * logarithm of the corresponding element in the input matrix. The natural
 * logarithm is computed using Base::Matrix::log.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' natural logarithms are
 * to be computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the natural logarithms of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto log(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::log(matrix.matrix);

  return result;
}

namespace LogAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyLogTupleCore {
  /**
   * @brief Recursively applies log to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where log results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::log(std::get<Index>(input).matrix);
    ApplyLogTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyLogTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where log results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::log(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply log to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where log results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyLogTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output);
}

} // namespace LogAugmentedMatrixAction

/**
 * @brief Computes the element-wise natural logarithm of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the natural logarithm of the
 * corresponding element in the input augmented matrix. The natural logarithm
 * is computed using Base::Matrix::log for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' natural
 * logarithms are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the natural logarithms of the input augmented
 * matrix's elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto
log(const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  LogAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* log2 */

/**
 * @brief Computes the base-2 logarithm of a given input.
 *
 * This function is a template that computes the base-2 logarithm of its
 * argument by delegating to Base::Matrix::log2. It works for any type T for
 * which Base::Matrix::log2 is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose base-2 logarithm is to be computed.
 * @return The base-2 logarithm of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
log2(const T &x) {
  return Base::Matrix::log2(x);
}

/**
 * @brief Computes the element-wise base-2 logarithm of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the base-2 logarithm of
 * the corresponding element in the input matrix. The base-2 logarithm is
 * computed using Base::Matrix::log2.
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
inline auto log2(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::log2(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise base-2 logarithm of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the base-2
 * logarithm of the corresponding diagonal element in the input matrix. The
 * base-2 logarithm is computed using Base::Matrix::log2.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' base-2
 * logarithms are to be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the base-2
 * logarithms of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto log2(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::log2(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise base-2 logarithm of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the base-2
 * logarithm of the corresponding element in the input matrix. The base-2
 * logarithm is computed using Base::Matrix::log2.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' base-2 logarithms are
 * to be computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the base-2 logarithms of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto log2(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::log2(matrix.matrix);

  return result;
}

namespace Log2AugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyLog2TupleCore {
  /**
   * @brief Recursively applies log2 to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where log2 results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::log2(std::get<Index>(input).matrix);
    ApplyLog2TupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyLog2TupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where log2 results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::log2(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply log2 to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where log2 results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyLog2TupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace Log2AugmentedMatrixAction

/**
 * @brief Computes the element-wise base-2 logarithm of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the base-2 logarithm of the
 * corresponding element in the input augmented matrix. The base-2 logarithm is
 * computed using Base::Matrix::log2 for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' base-2
 * logarithms are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the base-2 logarithms of the input augmented
 * matrix's elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto log2(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  Log2AugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* log10 */

/**
 * @brief Computes the base-10 logarithm of a given input.
 *
 * This function is a template that computes the base-10 logarithm of its
 * argument by delegating to Base::Matrix::log10. It works for any type T for
 * which Base::Matrix::log10 is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose base-10 logarithm is to be computed.
 * @return The base-10 logarithm of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
log10(const T &x) {
  return Base::Matrix::log10(x);
}

/**
 * @brief Computes the element-wise base-10 logarithm of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the base-10 logarithm of
 * the corresponding element in the input matrix. The base-10 logarithm is
 * computed using Base::Matrix::log10.
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
inline auto log10(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::log10(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise base-10 logarithm of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the base-10
 * logarithm of the corresponding diagonal element in the input matrix. The
 * base-10 logarithm is computed using Base::Matrix::log10.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' base-10
 * logarithms are to be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the base-10
 * logarithms of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto log10(const Matrix<DefDiag, T, M> &matrix)
    -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::log10(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise base-10 logarithm of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the base-10
 * logarithm of the corresponding element in the input matrix. The base-10
 * logarithm is computed using Base::Matrix::log10.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' base-10 logarithms are
 * to be computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the base-10 logarithms of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto log10(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::log10(matrix.matrix);

  return result;
}

namespace Log10AugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyLog10TupleCore {
  /**
   * @brief Recursively applies log10 to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where log10 results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::log10(std::get<Index>(input).matrix);
    ApplyLog10TupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyLog10TupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where log10 results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::log10(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply log10 to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where log10 results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyLog10TupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                      1>::compute(input, output);
}

} // namespace Log10AugmentedMatrixAction

/**
 * @brief Computes the element-wise base-10 logarithm of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the base-10 logarithm of the
 * corresponding element in the input augmented matrix. The base-10 logarithm
 * is computed using Base::Matrix::log10 for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' base-10
 * logarithms are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the base-10 logarithms of the input augmented
 * matrix's elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto log10(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  Log10AugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* pow */

/**
 * @brief Computes the power of a given input raised to a specified exponent.
 *
 * This function is a template that computes the power of its argument by
 * delegating to Base::Matrix::pow. It works for any type T for which
 * Base::Matrix::pow is defined.
 *
 * @tparam T The type of the input value.
 * @param x The base value.
 * @param y The exponent value.
 * @return The result of raising x to the power of y.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
pow(const T &x, const T &y) {
  return Base::Matrix::pow(x, y);
}

/**
 * @brief Computes the element-wise power of a Matrix raised to a specified
 * exponent.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is raised to the power of y.
 * The power operation is computed using Base::Matrix::pow.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements are to be raised to the power
 * of y.
 * @param y The exponent value.
 * @return Matrix<T, M, N> A new Matrix containing the elements of the input
 * matrix raised to the power of y.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto pow(const Matrix<DefDense, T, M, N> &matrix, const T &y)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::pow(matrix.matrix, y);

  return result;
}

/**
 * @brief Computes the element-wise power of a DiagMatrix raised to a specified
 * exponent.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is raised to the
 * power of y. The power operation is computed using Base::Matrix::pow.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements are to be raised
 * to the power of y.
 * @param y The exponent value.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the diagonal
 * elements of the input matrix raised to the power of y.
 */
template <typename T, std::size_t M>
inline auto pow(const Matrix<DefDiag, T, M> &matrix, const T &y)
    -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::pow(matrix.matrix, y);

  return result;
}

/**
 * @brief Computes the element-wise power of a SparseMatrix raised to a
 * specified exponent.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is raised to the
 * power of y. The power operation is computed using Base::Matrix::pow.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements are to be raised to the
 * power of y.
 * @param y The exponent value.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the elements of the input matrix raised to the power of y.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto pow(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                const T &y) -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::pow(matrix.matrix, y);

  return result;
}

namespace PowAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyPowTupleCore {
  /**
   * @brief Recursively applies pow to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where pow results are stored.
   * @param y The exponent value to which each element is raised.
   */
  template <typename T>
  static void compute(const Tuple_Type &input, Tuple_Type &output, const T &y) {
    std::get<Index>(output).matrix =
        Base::Matrix::pow(std::get<Index>(input).matrix, y);
    ApplyPowTupleCore<Tuple_Type, Index - 1>::compute(input, output, y);
  }
};

template <typename Tuple_Type> struct ApplyPowTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where pow results are stored.
   * @param y The exponent value to which the element is raised.
   */
  template <typename T>
  static void compute(const Tuple_Type &input, Tuple_Type &output, const T &y) {
    std::get<0>(output).matrix =
        Base::Matrix::pow(std::get<0>(input).matrix, y);
  }
};

/**
 * @brief Wrapper function to apply pow to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @tparam T The type of the exponent value.
 * @param input The input tuple.
 * @param output The output tuple where pow results are stored.
 * @param y The exponent value to which each element is raised.
 */
template <typename Tuple_Type, typename T>
inline void compute(const Tuple_Type &input, Tuple_Type &output, const T &y) {
  ApplyPowTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output, y);
}

} // namespace PowAugmentedMatrixAction

/**
 * @brief Computes the element-wise power of an AugmentedMatrix raised to a
 * specified exponent.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is raised to the power of y. The
 * power operation is computed using Base::Matrix::pow for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam T The type of the exponent value.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements are to be
 * raised to the power of y.
 * @param y The exponent value.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the elements of the input augmented matrix raised
 * to the power of y.
 */
template <typename Tuple_Type, typename T, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto
pow(const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix,
    const T &y) -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  PowAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix, y);

  return result;
}

/* sin */

/**
 * @brief Computes the sine of a given input.
 *
 * This function is a template that computes the sine of its argument by
 * delegating to Base::Matrix::sin. It works for any type T for which
 * Base::Matrix::sin is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose sine is to be computed.
 * @return The sine of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
sin(const T &x) {
  return Base::Matrix::sin(x);
}

/**
 * @brief Computes the element-wise sine of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the sine of the
 * corresponding element in the input matrix. The sine operation is computed
 * using Base::Matrix::sin.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' sines are to be computed.
 * @return Matrix<T, M, N> A new Matrix containing the sines of the input
 * matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto sin(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::sin(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise sine of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the sine of
 * the corresponding diagonal element in the input matrix. The sine operation
 * is computed using Base::Matrix::sin.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' sines are to be
 * computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the sines of the
 * input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto sin(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::sin(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise sine of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the sine of the
 * corresponding element in the input matrix. The sine operation is computed
 * using Base::Matrix::sin.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' sines are to be
 * computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the sines of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto sin(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::sin(matrix.matrix);

  return result;
}

namespace SinAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplySinTupleCore {
  /**
   * @brief Recursively applies sin to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where sin results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::sin(std::get<Index>(input).matrix);
    ApplySinTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplySinTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where sin results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::sin(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply sin to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where sin results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplySinTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output);
}

} // namespace SinAugmentedMatrixAction

/**
 * @brief Computes the element-wise sine of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the sine of the corresponding
 * element in the input augmented matrix. The sine operation is computed using
 * Base::Matrix::sin for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' sines are
 * to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the sines of the input augmented matrix's
 * elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto
sin(const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  SinAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* cos */

/**
 * @brief Computes the cosine of a given input.
 *
 * This function is a template that computes the cosine of its argument by
 * delegating to Base::Matrix::cos. It works for any type T for which
 * Base::Matrix::cos is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose cosine is to be computed.
 * @return The cosine of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
cos(const T &x) {
  return Base::Matrix::cos(x);
}

/**
 * @brief Computes the element-wise cosine of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the cosine of the
 * corresponding element in the input matrix. The cosine operation is computed
 * using Base::Matrix::cos.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' cosines are to be computed.
 * @return Matrix<T, M, N> A new Matrix containing the cosines of the input
 * matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto cos(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::cos(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise cosine of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the cosine of
 * the corresponding diagonal element in the input matrix. The cosine operation
 * is computed using Base::Matrix::cos.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' cosines are to be
 * computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the cosines of the
 * input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto cos(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::cos(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise cosine of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the cosine of the
 * corresponding element in the input matrix. The cosine operation is computed
 * using Base::Matrix::cos.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' cosines are to be
 * computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the cosines of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto cos(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::cos(matrix.matrix);

  return result;
}

namespace CosAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyCosTupleCore {
  /**
   * @brief Recursively applies cos to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where cos results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::cos(std::get<Index>(input).matrix);
    ApplyCosTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyCosTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where cos results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::cos(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply cos to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where cos results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyCosTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output);
}

} // namespace CosAugmentedMatrixAction

/**
 * @brief Computes the element-wise cosine of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the cosine of the corresponding
 * element in the input augmented matrix. The cosine operation is computed using
 * Base::Matrix::cos for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' cosines are
 * to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the cosines of the input augmented matrix's
 * elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto
cos(const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  CosAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* tan */

/**
 * @brief Computes the tangent of a given input.
 *
 * This function is a template that computes the tangent of its argument by
 * delegating to Base::Matrix::tan. It works for any type T for which
 * Base::Matrix::tan is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose tangent is to be computed.
 * @return The tangent of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
tan(const T &x) {
  return Base::Matrix::tan(x);
}

/**
 * @brief Computes the element-wise tangent of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the tangent of the
 * corresponding element in the input matrix. The tangent operation is computed
 * using Base::Matrix::tan.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' tangents are to be computed.
 * @return Matrix<T, M, N> A new Matrix containing the tangents of the input
 * matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto tan(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::tan(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise tangent of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the tangent of
 * the corresponding diagonal element in the input matrix. The tangent operation
 * is computed using Base::Matrix::tan.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' tangents are to
 * be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the tangents of the
 * input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto tan(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::tan(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise tangent of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the tangent of
 * the corresponding element in the input matrix. The tangent operation is
 * computed using Base::Matrix::tan.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' tangents are to be
 * computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the tangents of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto tan(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::tan(matrix.matrix);

  return result;
}

namespace TanAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyTanTupleCore {
  /**
   * @brief Recursively applies tan to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where tan results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::tan(std::get<Index>(input).matrix);
    ApplyTanTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyTanTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where tan results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::tan(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply tan to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where tan results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyTanTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output);
}

} // namespace TanAugmentedMatrixAction

/**
 * @brief Computes the element-wise tangent of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the tangent of the corresponding
 * element in the input augmented matrix. The tangent operation is computed
 * using Base::Matrix::tan for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' tangents
 * are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the tangents of the input augmented matrix's
 * elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto
tan(const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  TanAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* asin */

/**
 * @brief Computes the arcsine of a given input.
 *
 * This function is a template that computes the arcsine of its argument by
 * delegating to Base::Matrix::asin. It works for any type T for which
 * Base::Matrix::asin is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose arcsine is to be computed.
 * @return The arcsine of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
asin(const T &x) {
  return Base::Matrix::asin(x);
}

/**
 * @brief Computes the element-wise arcsine of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the arcsine of the
 * corresponding element in the input matrix. The arcsine operation is computed
 * using Base::Matrix::asin.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' arcsines are to be computed.
 * @return Matrix<T, M, N> A new Matrix containing the arcsines of the input
 * matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto asin(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::asin(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise arcsine of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the arcsine of
 * the corresponding diagonal element in the input matrix. The arcsine operation
 * is computed using Base::Matrix::asin.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' arcsines are to
 * be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the arcsines of the
 * input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto asin(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::asin(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise arcsine of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the arcsine of
 * the corresponding element in the input matrix. The arcsine operation is
 * computed using Base::Matrix::asin.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' arcsines are to be
 * computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the arcsines of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto asin(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::asin(matrix.matrix);

  return result;
}

namespace AsinAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyAsinTupleCore {
  /**
   * @brief Recursively applies asin to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where asin results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::asin(std::get<Index>(input).matrix);
    ApplyAsinTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyAsinTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where asin results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::asin(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply asin to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where asin results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyAsinTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace AsinAugmentedMatrixAction

/**
 * @brief Computes the element-wise arcsine of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the arcsine of the corresponding
 * element in the input augmented matrix. The arcsine operation is computed
 * using Base::Matrix::asin for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' arcsines
 * are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the arcsines of the input augmented matrix's
 * elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto asin(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  AsinAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* acos */

/**
 * @brief Computes the arccosine of a given input.
 *
 * This function is a template that computes the arccosine of its argument by
 * delegating to Base::Matrix::acos. It works for any type T for which
 * Base::Matrix::acos is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose arccosine is to be computed.
 * @return The arccosine of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
acos(const T &x) {
  return Base::Matrix::acos(x);
}

/**
 * @brief Computes the element-wise arccosine of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the arccosine of the
 * corresponding element in the input matrix. The arccosine operation is
 * computed using Base::Matrix::acos.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam M The number of rows in the matrix.
 * @tparam N The number of columns in the matrix.
 * @param matrix The input Matrix whose elements' arccosines are to be computed.
 * @return Matrix<T, M, N> A new Matrix containing the arccosines of the input
 * matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto acos(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::acos(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise arccosine of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the arccosine
 * of the corresponding diagonal element in the input matrix. The arccosine
 * operation is computed using Base::Matrix::acos.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' arccosines are
 * to be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the arccosines of
 * the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto acos(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::acos(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise arccosine of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the arccosine of
 * the corresponding element in the input matrix. The arccosine operation is
 * computed using Base::Matrix::acos.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' arccosines are to be
 * computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the arccosines of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto acos(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::acos(matrix.matrix);

  return result;
}

namespace AcosAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyAcosTupleCore {
  /**
   * @brief Recursively applies acos to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where acos results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::acos(std::get<Index>(input).matrix);
    ApplyAcosTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyAcosTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where acos results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::acos(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply acos to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where acos results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyAcosTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace AcosAugmentedMatrixAction

/**
 * @brief Computes the element-wise arccosine of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the arccosine of the corresponding
 * element in the input augmented matrix. The arccosine operation is computed
 * using Base::Matrix::acos for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' arccosines
 * are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the arccosines of the input augmented matrix's
 * elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto acos(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  AcosAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* atan */

/**
 * @brief Computes the arctangent of a given input.
 *
 * This function is a template that computes the arctangent of its argument by
 * delegating to Base::Matrix::atan. It works for any type T for which
 * Base::Matrix::atan is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose arctangent is to be computed.
 * @return The arctangent of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
atan(const T &x) {
  return Base::Matrix::atan(x);
}

/**
 * @brief Computes the element-wise arctangent of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the arctangent of the
 * corresponding element in the input matrix. The arctangent operation is
 * computed using Base::Matrix::atan.
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
inline auto atan(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::atan(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise arctangent of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the
 * arctangent of the corresponding diagonal element in the input matrix. The
 * arctangent operation is computed using Base::Matrix::atan.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' arctangents are
 * to be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the arctangents of
 * the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto atan(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::atan(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise arctangent of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the arctangent of
 * the corresponding element in the input matrix. The arctangent operation is
 * computed using Base::Matrix::atan.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' arctangents are to be
 * computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the arctangents of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto atan(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::atan(matrix.matrix);

  return result;
}

namespace AtanAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyAtanTupleCore {
  /**
   * @brief Recursively applies atan to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where atan results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::atan(std::get<Index>(input).matrix);
    ApplyAtanTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyAtanTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where atan results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::atan(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply atan to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where atan results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyAtanTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace AtanAugmentedMatrixAction

/**
 * @brief Computes the element-wise arctangent of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the arctangent of the corresponding
 * element in the input augmented matrix. The arctangent operation is computed
 * using Base::Matrix::atan for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' arctangents
 * are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the arctangents of the input augmented matrix's
 * elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto atan(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  AtanAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* atan2 */

/**
 * @brief Computes the element-wise arctangent of the quotient of two values.
 *
 * This function computes the arctangent of the quotient of two values, `x` and
 * `y`, and returns the angle in radians. It is a wrapper around the
 * Base::Matrix::atan2 function, which handles both scalar and matrix inputs.
 *
 * @tparam T The type of the input values (must be arithmetic).
 * @param x The numerator value.
 * @param y The denominator value.
 * @return The angle in radians as a value of type T.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
atan2(const T &x, const T &y) {
  return Base::Matrix::atan2(x, y);
}

/**
 * @brief Computes the element-wise arctangent of the quotient of two matrices.
 *
 * This function computes the arctangent of the quotient of two matrices,
 * `matrix_y` and `matrix_x`, and returns a matrix containing the angles in
 * radians. It is a wrapper around the Base::Matrix::atan2 function, which
 * handles both dense and sparse matrix inputs.
 *
 * @tparam T The type of the elements in the matrices (must be arithmetic).
 * @tparam M The number of rows in the matrices.
 * @tparam N The number of columns in the matrices.
 * @param matrix_y The numerator matrix.
 * @param matrix_x The denominator matrix.
 * @return A matrix containing the angles in radians, with the same dimensions
 *         as the input matrices.
 */
template <typename T, std::size_t M, std::size_t N>
inline auto atan2(const Matrix<DefDense, T, M, N> &matrix_y,
                  const Matrix<DefDense, T, M, N> &matrix_x)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::atan2(matrix_y.matrix, matrix_x.matrix);

  return result;
}

/**
 * @brief Computes the element-wise arctangent of the quotient of two diagonal
 * matrices.
 *
 * This function computes the arctangent of the quotient of two diagonal
 * matrices, `matrix_y` and `matrix_x`, and returns a diagonal matrix containing
 * the angles in radians. It is a wrapper around the Base::Matrix::atan2
 * function, which handles both dense and sparse matrix inputs.
 *
 * @tparam T The type of the elements in the matrices (must be arithmetic).
 * @tparam M The number of rows (and columns) in the diagonal matrices.
 * @param matrix_y The numerator diagonal matrix.
 * @param matrix_x The denominator diagonal matrix.
 * @return A diagonal matrix containing the angles in radians, with the same
 *         dimensions as the input matrices.
 */
template <typename T, std::size_t M>
inline auto atan2(const Matrix<DefDiag, T, M> &matrix_y,
                  const Matrix<DefDiag, T, M> &matrix_x)
    -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::atan2(matrix_y.matrix, matrix_x.matrix);

  return result;
}

/**
 * @brief Computes the element-wise arctangent of the quotient of two sparse
 * matrices.
 *
 * This function computes the arctangent of the quotient of two sparse
 * matrices, `matrix_y` and `matrix_x`, and returns a sparse matrix containing
 * the angles in radians. It is a wrapper around the Base::Matrix::atan2
 * function, which handles both dense and sparse matrix inputs.
 *
 * @tparam T The type of the elements in the matrices (must be arithmetic).
 * @tparam M The number of rows in the matrices.
 * @tparam N The number of columns in the matrices.
 * @tparam SparseAvailable A type trait indicating whether sparse storage is
 * available for the matrices.
 * @param matrix_y The numerator sparse matrix.
 * @param matrix_x The denominator sparse matrix.
 * @return A sparse matrix containing the angles in radians, with the same
 *         dimensions as the input matrices.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto atan2(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix_y,
                  const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix_x)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::atan2(matrix_y.matrix, matrix_x.matrix);

  return result;
}

namespace Atan2AugmentedMatrixAction {

/**
 * @brief Helper struct to apply atan2 to each element of two std::tuples
 * using template metaprogramming.
 *
 * This struct recursively applies Base::Matrix::atan2 to each element pair of
 * the input tuples. Uses index-based recursion for C++11 compatibility.
 *
 * @tparam Tuple_Type The type of the tuples.
 * @tparam Index The current index in the tuples (decremented recursively).
 */
template <typename Tuple_Type, std::size_t Index> struct ApplyAtan2TupleCore {
  /**
   * @brief Recursively applies atan2 to tuple elements.
   *
   * @param input_y The input numerator tuple (const reference).
   * @param output The output tuple (reference) where atan2 results are stored.
   * @param input_x The input denominator tuple (const reference).
   */
  static void compute(const Tuple_Type &input_y, Tuple_Type &output,
                      const Tuple_Type &input_x) {
    std::get<Index>(output).matrix = Base::Matrix::atan2(
        std::get<Index>(input_y).matrix, std::get<Index>(input_x).matrix);
    ApplyAtan2TupleCore<Tuple_Type, Index - 1>::compute(input_y, output,
                                                        input_x);
  }
};

/**
 * @brief Base case specialization for index 0.
 *
 * @tparam Tuple_Type The type of the tuples.
 */
template <typename Tuple_Type> struct ApplyAtan2TupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element pair.
   *
   * @param input_y The input numerator tuple (const reference).
   * @param output The output tuple (reference) where atan2 results are stored.
   * @param input_x The input denominator tuple (const reference).
   */
  static void compute(const Tuple_Type &input_y, Tuple_Type &output,
                      const Tuple_Type &input_x) {
    std::get<0>(output).matrix = Base::Matrix::atan2(
        std::get<0>(input_y).matrix, std::get<0>(input_x).matrix);
  }
};

/**
 * @brief Wrapper function to apply atan2 to all element pairs of two tuples.
 *
 * @tparam Tuple_Type The type of the tuples.
 * @param input_y The input numerator tuple.
 * @param output The output tuple where atan2 results are stored.
 * @param input_x The input denominator tuple.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input_y, Tuple_Type &output,
                    const Tuple_Type &input_x) {
  ApplyAtan2TupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                      1>::compute(input_y, output, input_x);
}

} // namespace Atan2AugmentedMatrixAction

/**
 * @brief Computes the element-wise arctangent of the quotient of two
 * AugmentedMatrix objects.
 *
 * This function computes the arctangent of the quotient of two
 * AugmentedMatrix objects, `augmented_matrix_y` and `augmented_matrix_x`, and
 * returns an AugmentedMatrix containing the angles in radians. It is a wrapper
 * around the Base::Matrix::atan2 function, which handles both dense and sparse
 * matrix inputs.
 *
 * @tparam Tuple_Type The type of the elements in the AugmentedMatrix (must be
 * arithmetic).
 * @tparam Row_Blocks The number of row blocks in the AugmentedMatrix.
 * @tparam Col_Blocks The number of column blocks in the AugmentedMatrix.
 * @param augmented_matrix_y The numerator AugmentedMatrix.
 * @param augmented_matrix_x The denominator AugmentedMatrix.
 * @return An AugmentedMatrix containing the angles in radians, with the same
 *         dimensions as the input AugmentedMatrix objects.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto atan2(const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks>
                      &augmented_matrix_y,
                  const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks>
                      &augmented_matrix_x)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  Atan2AugmentedMatrixAction::compute(augmented_matrix_y.matrix, result.matrix,
                                      augmented_matrix_x.matrix);

  return result;
}

/* sinh */

/**
 * @brief Computes the hyperbolic sine of a given input.
 *
 * This function is a template that computes the hyperbolic sine of its argument
 * by delegating to Base::Matrix::sinh. It works for any type T for which
 * Base::Matrix::sinh is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose hyperbolic sine is to be computed.
 * @return The hyperbolic sine of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
sinh(const T &x) {
  return Base::Matrix::sinh(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto sinh(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::sinh(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise hyperbolic sine of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the hyperbolic
 * sine of the corresponding diagonal element in the input matrix. The
 * hyperbolic sine operation is computed using Base::Matrix::sinh.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' hyperbolic sines
 * are to be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the hyperbolic
 * sines of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto sinh(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::sinh(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise hyperbolic sine of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the hyperbolic
 * sine of the corresponding element in the input matrix. The hyperbolic sine
 * operation is computed using Base::Matrix::sinh.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' hyperbolic sines are to
 * be computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the hyperbolic sines of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto sinh(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::sinh(matrix.matrix);

  return result;
}

namespace SinhAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplySinhTupleCore {
  /**
   * @brief Recursively applies sinh to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where sinh results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::sinh(std::get<Index>(input).matrix);
    ApplySinhTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplySinhTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where sinh results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::sinh(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply sinh to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where sinh results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplySinhTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace SinhAugmentedMatrixAction

/**
 * @brief Computes the element-wise hyperbolic sine of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the hyperbolic sine of the
 * corresponding element in the input augmented matrix. The hyperbolic sine
 * operation is computed using Base::Matrix::sinh for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' hyperbolic
 * sines are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the hyperbolic sines of the input augmented
 * matrix's elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto sinh(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  SinhAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* cosh */

/**
 * @brief Computes the hyperbolic cosine of a given input.
 *
 * This function is a template that computes the hyperbolic cosine of its
 * argument by delegating to Base::Matrix::cosh. It works for any type T for
 * which Base::Matrix::cosh is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose hyperbolic cosine is to be computed.
 * @return The hyperbolic cosine of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
cosh(const T &x) {
  return Base::Matrix::cosh(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto cosh(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::cosh(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise hyperbolic cosine of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the hyperbolic
 * cosine of the corresponding diagonal element in the input matrix. The
 * hyperbolic cosine operation is computed using Base::Matrix::cosh.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' hyperbolic
 * cosines are to be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the hyperbolic
 * cosines of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto cosh(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::cosh(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise hyperbolic cosine of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the hyperbolic
 * cosine of the corresponding element in the input matrix. The hyperbolic
 * cosine operation is computed using Base::Matrix::cosh.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' hyperbolic cosines are
 * to be computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the hyperbolic cosines of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto cosh(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::cosh(matrix.matrix);

  return result;
}

namespace CoshAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyCoshTupleCore {
  /**
   * @brief Recursively applies cosh to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where cosh results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::cosh(std::get<Index>(input).matrix);
    ApplyCoshTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyCoshTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where cosh results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::cosh(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply cosh to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where cosh results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyCoshTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace CoshAugmentedMatrixAction

/**
 * @brief Computes the element-wise hyperbolic cosine of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the hyperbolic cosine of the
 * corresponding element in the input augmented matrix. The hyperbolic cosine
 * operation is computed using Base::Matrix::cosh for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' hyperbolic
 * cosines are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the hyperbolic cosines of the input augmented
 * matrix's elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto cosh(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  CoshAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

/* tanh */

/**
 * @brief Computes the hyperbolic tangent of a given input.
 *
 * This function is a template that computes the hyperbolic tangent of its
 * argument by delegating to Base::Matrix::tanh. It works for any type T for
 * which Base::Matrix::tanh is defined.
 *
 * @tparam T The type of the input value.
 * @param x The value whose hyperbolic tangent is to be computed.
 * @return The hyperbolic tangent of x.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
tanh(const T &x) {
  return Base::Matrix::tanh(x);
}

/**
 * @brief Computes the element-wise hyperbolic tangent of a Matrix.
 *
 * This function takes a constant reference to a Matrix of type T and size M x
 * N, and returns a new Matrix where each element is the hyperbolic tangent of
 * the corresponding element in the input matrix. The hyperbolic tangent
 * operation is computed using Base::Matrix::tanh.
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
inline auto tanh(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::tanh(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise hyperbolic tangent of a DiagMatrix.
 *
 * This function takes a constant reference to a DiagMatrix of type T and size
 * M, and returns a new DiagMatrix where each diagonal element is the
 * hyperbolic tangent of the corresponding diagonal element in the input matrix.
 * The hyperbolic tangent operation is computed using Base::Matrix::tanh.
 *
 * @tparam T The type of the elements in the diagonal matrix.
 * @tparam M The size of the diagonal matrix (number of rows and columns).
 * @param matrix The input DiagMatrix whose diagonal elements' hyperbolic
 * tangents are to be computed.
 * @return Matrix<DefDiag, T, M> A new DiagMatrix containing the hyperbolic
 * tangents of the input matrix's diagonal elements.
 */
template <typename T, std::size_t M>
inline auto tanh(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::tanh(matrix.matrix);

  return result;
}

/**
 * @brief Computes the element-wise hyperbolic tangent of a SparseMatrix.
 *
 * This function takes a constant reference to a SparseMatrix of type T and size
 * M x N, and returns a new SparseMatrix where each element is the hyperbolic
 * tangent of the corresponding element in the input matrix. The hyperbolic
 * tangent operation is computed using Base::Matrix::tanh.
 *
 * @tparam T The type of the elements in the sparse matrix.
 * @tparam M The number of rows in the sparse matrix.
 * @tparam N The number of columns in the sparse matrix.
 * @tparam SparseAvailable A type trait indicating whether sparse operations
 * are available for this matrix type.
 * @param matrix The input SparseMatrix whose elements' hyperbolic tangents are
 * to be computed.
 * @return Matrix<DefSparse, T, M, N, SparseAvailable> A new SparseMatrix
 * containing the hyperbolic tangents of the input matrix's elements.
 */
template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto tanh(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::tanh(matrix.matrix);

  return result;
}

namespace TanhAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyTanhTupleCore {
  /**
   * @brief Recursively applies tanh to tuple elements.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where tanh results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::tanh(std::get<Index>(input).matrix);
    ApplyTanhTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyTanhTupleCore<Tuple_Type, 0> {
  /**
   * @brief Base case: processes the first (index 0) element.
   *
   * @param input The input tuple (const reference).
   * @param output The output tuple (reference) where tanh results are stored.
   */
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::tanh(std::get<0>(input).matrix);
  }
};

/**
 * @brief Wrapper function to apply tanh to all elements of a tuple.
 *
 * @tparam Tuple_Type The type of the tuple.
 * @param input The input tuple.
 * @param output The output tuple where tanh results are stored.
 */
template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyTanhTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace TanhAugmentedMatrixAction

/**
 * @brief Computes the element-wise hyperbolic tangent of an AugmentedMatrix.
 *
 * This function takes a constant reference to an AugmentedMatrix and returns a
 * new AugmentedMatrix where each element is the hyperbolic tangent of the
 * corresponding element in the input augmented matrix. The hyperbolic tangent
 * operation is computed using Base::Matrix::tanh for each nested matrix.
 *
 * @tparam Tuple_Type The type of the tuple containing the nested matrices.
 * @tparam Row_Blocks The number of row blocks in the augmented matrix (default
 * is 0).
 * @tparam Col_Blocks The number of column blocks in the augmented matrix
 * (default is 0).
 * @param augmented_matrix The input AugmentedMatrix whose elements' hyperbolic
 * tangents are to be computed.
 * @return AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> A new
 * AugmentedMatrix containing the hyperbolic tangents of the input augmented
 * matrix's elements.
 */
template <typename Tuple_Type, std::size_t Row_Blocks = 0,
          std::size_t Col_Blocks = 0>
inline auto tanh(
    const AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> &augmented_matrix)
    -> AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> {
  AugmentedMatrix<Tuple_Type, Row_Blocks, Col_Blocks> result;

  TanhAugmentedMatrixAction::compute(augmented_matrix.matrix, result.matrix);

  return result;
}

} // namespace PythonNumpy

#endif // PYTHON_NUMPY_MATH_HPP_
