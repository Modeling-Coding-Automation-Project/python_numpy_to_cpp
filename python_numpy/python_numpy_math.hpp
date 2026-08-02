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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
abs(const T &x) {
  return Base::Matrix::abs(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto abs(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::abs(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto abs(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::abs(matrix.matrix);

  return result;
}

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
fmod(const T &x, const T &y) {
  return Base::Matrix::fmod(x, y);
}

template <typename T, std::size_t M, std::size_t N>
inline auto fmod(const Matrix<DefDense, T, M, N> &matrix, const T &y)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::fmod(matrix.matrix, y);

  return result;
}

template <typename T, std::size_t M>
inline auto fmod(const Matrix<DefDiag, T, M> &matrix, const T &y)
    -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::fmod(matrix.matrix, y);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto fmod(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                 const T &y) -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::fmod(matrix.matrix, y);

  return result;
}

namespace FmodAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyFmodTupleCore {
  template <typename T>
  static void compute(const Tuple_Type &input, Tuple_Type &output, const T &y) {
    std::get<Index>(output).matrix =
        Base::Matrix::fmod(std::get<Index>(input).matrix, y);
    ApplyFmodTupleCore<Tuple_Type, Index - 1>::compute(input, output, y);
  }
};

template <typename Tuple_Type> struct ApplyFmodTupleCore<Tuple_Type, 0> {
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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
sqrt(const T &x) {
  return Base::Matrix::sqrt(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto sqrt(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::sqrt(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto sqrt(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::sqrt(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto sqrt(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::sqrt(matrix.matrix);

  return result;
}

namespace SqrtAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplySqrtTupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::sqrt(std::get<Index>(input).matrix);
    ApplySqrtTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplySqrtTupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::sqrt(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplySqrtTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace SqrtAugmentedMatrixAction

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
exp(const T &x) {
  return Base::Matrix::exp(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto exp(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::exp(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto exp(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::exp(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto exp(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::exp(matrix.matrix);

  return result;
}

namespace ExpAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyExpTupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::exp(std::get<Index>(input).matrix);
    ApplyExpTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyExpTupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::exp(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyExpTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output);
}

} // namespace ExpAugmentedMatrixAction

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
exp2(const T &x) {
  return Base::Matrix::exp2(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto exp2(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::exp2(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto exp2(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::exp2(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto exp2(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::exp2(matrix.matrix);

  return result;
}

namespace Exp2AugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyExp2TupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::exp2(std::get<Index>(input).matrix);
    ApplyExp2TupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyExp2TupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::exp2(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyExp2TupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace Exp2AugmentedMatrixAction

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
log(const T &x) {
  return Base::Matrix::log(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto log(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::log(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto log(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::log(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto log(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::log(matrix.matrix);

  return result;
}

namespace LogAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyLogTupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::log(std::get<Index>(input).matrix);
    ApplyLogTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyLogTupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::log(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyLogTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output);
}

} // namespace LogAugmentedMatrixAction

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
log2(const T &x) {
  return Base::Matrix::log2(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto log2(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::log2(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto log2(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::log2(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto log2(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::log2(matrix.matrix);

  return result;
}

namespace Log2AugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyLog2TupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::log2(std::get<Index>(input).matrix);
    ApplyLog2TupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyLog2TupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::log2(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyLog2TupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace Log2AugmentedMatrixAction

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
log10(const T &x) {
  return Base::Matrix::log10(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto log10(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::log10(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto log10(const Matrix<DefDiag, T, M> &matrix)
    -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::log10(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto log10(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::log10(matrix.matrix);

  return result;
}

namespace Log10AugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyLog10TupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::log10(std::get<Index>(input).matrix);
    ApplyLog10TupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyLog10TupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::log10(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyLog10TupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                      1>::compute(input, output);
}

} // namespace Log10AugmentedMatrixAction

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
pow(const T &x, const T &y) {
  return Base::Matrix::pow(x, y);
}

template <typename T, std::size_t M, std::size_t N>
inline auto pow(const Matrix<DefDense, T, M, N> &matrix, const T &y)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::pow(matrix.matrix, y);

  return result;
}

template <typename T, std::size_t M>
inline auto pow(const Matrix<DefDiag, T, M> &matrix, const T &y)
    -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::pow(matrix.matrix, y);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto pow(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix,
                const T &y) -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::pow(matrix.matrix, y);

  return result;
}

namespace PowAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyPowTupleCore {
  template <typename T>
  static void compute(const Tuple_Type &input, Tuple_Type &output, const T &y) {
    std::get<Index>(output).matrix =
        Base::Matrix::pow(std::get<Index>(input).matrix, y);
    ApplyPowTupleCore<Tuple_Type, Index - 1>::compute(input, output, y);
  }
};

template <typename Tuple_Type> struct ApplyPowTupleCore<Tuple_Type, 0> {
  template <typename T>
  static void compute(const Tuple_Type &input, Tuple_Type &output, const T &y) {
    std::get<0>(output).matrix =
        Base::Matrix::pow(std::get<0>(input).matrix, y);
  }
};

template <typename Tuple_Type, typename T>
inline void compute(const Tuple_Type &input, Tuple_Type &output, const T &y) {
  ApplyPowTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output, y);
}

} // namespace PowAugmentedMatrixAction

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
sin(const T &x) {
  return Base::Matrix::sin(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto sin(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::sin(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto sin(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::sin(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto sin(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::sin(matrix.matrix);

  return result;
}

namespace SinAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplySinTupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::sin(std::get<Index>(input).matrix);
    ApplySinTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplySinTupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::sin(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplySinTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output);
}

} // namespace SinAugmentedMatrixAction

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
cos(const T &x) {
  return Base::Matrix::cos(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto cos(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::cos(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto cos(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::cos(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto cos(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::cos(matrix.matrix);

  return result;
}

namespace CosAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyCosTupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::cos(std::get<Index>(input).matrix);
    ApplyCosTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyCosTupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::cos(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyCosTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output);
}

} // namespace CosAugmentedMatrixAction

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
tan(const T &x) {
  return Base::Matrix::tan(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto tan(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::tan(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto tan(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::tan(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto tan(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::tan(matrix.matrix);

  return result;
}

namespace TanAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyTanTupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::tan(std::get<Index>(input).matrix);
    ApplyTanTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyTanTupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::tan(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyTanTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                    1>::compute(input, output);
}

} // namespace TanAugmentedMatrixAction

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
asin(const T &x) {
  return Base::Matrix::asin(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto asin(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::asin(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto asin(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::asin(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto asin(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::asin(matrix.matrix);

  return result;
}

namespace AsinAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyAsinTupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::asin(std::get<Index>(input).matrix);
    ApplyAsinTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyAsinTupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::asin(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyAsinTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace AsinAugmentedMatrixAction

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
acos(const T &x) {
  return Base::Matrix::acos(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto acos(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::acos(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto acos(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::acos(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto acos(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::acos(matrix.matrix);

  return result;
}

namespace AcosAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyAcosTupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::acos(std::get<Index>(input).matrix);
    ApplyAcosTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyAcosTupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::acos(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyAcosTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace AcosAugmentedMatrixAction

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
atan(const T &x) {
  return Base::Matrix::atan(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto atan(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::atan(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto atan(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::atan(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto atan(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::atan(matrix.matrix);

  return result;
}

namespace AtanAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyAtanTupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::atan(std::get<Index>(input).matrix);
    ApplyAtanTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyAtanTupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::atan(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyAtanTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace AtanAugmentedMatrixAction

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

template <typename T, std::size_t M>
inline auto sinh(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::sinh(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto sinh(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::sinh(matrix.matrix);

  return result;
}

namespace SinhAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplySinhTupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::sinh(std::get<Index>(input).matrix);
    ApplySinhTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplySinhTupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::sinh(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplySinhTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace SinhAugmentedMatrixAction

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

template <typename T, std::size_t M>
inline auto cosh(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::cosh(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto cosh(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::cosh(matrix.matrix);

  return result;
}

namespace CoshAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyCoshTupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::cosh(std::get<Index>(input).matrix);
    ApplyCoshTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyCoshTupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::cosh(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyCoshTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace CoshAugmentedMatrixAction

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

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
tanh(const T &x) {
  return Base::Matrix::tanh(x);
}

template <typename T, std::size_t M, std::size_t N>
inline auto tanh(const Matrix<DefDense, T, M, N> &matrix)
    -> Matrix<DefDense, T, M, N> {
  Matrix<DefDense, T, M, N> result;

  result.matrix = Base::Matrix::tanh(matrix.matrix);

  return result;
}

template <typename T, std::size_t M>
inline auto tanh(const Matrix<DefDiag, T, M> &matrix) -> Matrix<DefDiag, T, M> {
  Matrix<DefDiag, T, M> result;

  result.matrix = Base::Matrix::tanh(matrix.matrix);

  return result;
}

template <typename T, std::size_t M, std::size_t N, typename SparseAvailable>
inline auto tanh(const Matrix<DefSparse, T, M, N, SparseAvailable> &matrix)
    -> Matrix<DefSparse, T, M, N, SparseAvailable> {
  Matrix<DefSparse, T, M, N, SparseAvailable> result;

  result.matrix = Base::Matrix::tanh(matrix.matrix);

  return result;
}

namespace TanhAugmentedMatrixAction {

template <typename Tuple_Type, std::size_t Index> struct ApplyTanhTupleCore {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<Index>(output).matrix =
        Base::Matrix::tanh(std::get<Index>(input).matrix);
    ApplyTanhTupleCore<Tuple_Type, Index - 1>::compute(input, output);
  }
};

template <typename Tuple_Type> struct ApplyTanhTupleCore<Tuple_Type, 0> {
  static void compute(const Tuple_Type &input, Tuple_Type &output) {
    std::get<0>(output).matrix = Base::Matrix::tanh(std::get<0>(input).matrix);
  }
};

template <typename Tuple_Type>
inline void compute(const Tuple_Type &input, Tuple_Type &output) {
  ApplyTanhTupleCore<Tuple_Type, std::tuple_size<Tuple_Type>::value -
                                     1>::compute(input, output);
}

} // namespace TanhAugmentedMatrixAction

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
